/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/MultiHeadIVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>

namespace faiss {
namespace gpu {

MultiHeadIVFFlat::MultiHeadIVFFlat(
        GpuResources* resources,
        int numHeads,
        int dim,
        std::vector<idx_t> nlists,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : MultiHeadIVFBase(resources,
                          numHeads,
                          dim,
                          nlists,
                          metric,
                          metricArg,
                          useResidual,
                          interleavedLayout,
                          indicesOptions,
                          space) {
    
    // Initialize scalar quantizers for each head
    scalarQs_.resize(numHeads);
    for (int h = 0; h < numHeads; ++h) {
        if (scalarQsPerHead[h]) {
            scalarQs_[h] = std::make_unique<GpuScalarQuantizer>(resources, *scalarQsPerHead[h]);
        } else {
            scalarQs_[h] = nullptr;
        }
    }
}

MultiHeadIVFFlat::~MultiHeadIVFFlat() {}

size_t MultiHeadIVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    // For multi-head, we use the maximum encoding size across all heads
    size_t maxSize = 0;
    for (int h = 0; h < numHeads_; ++h) {
        size_t headSize;
        if (interleavedLayout_) {
            // bits per scalar code
            idx_t bits = scalarQs_[h] ? scalarQs_[h]->bits : 32 /* float */;
            
            int warpSize = getWarpSizeCurrentDevice();
            
            // bytes to encode a block of warpSize vectors (single dimension)
            idx_t bytesPerDimBlock = bits * warpSize / 8;
            
            // bytes to fully encode warpSize vectors
            idx_t bytesPerBlock = bytesPerDimBlock * dim_;
            
            // number of blocks of warpSize vectors we have
            idx_t numBlocks = utils::divUp(numVecs, warpSize);
            
            // total size to encode numVecs
            headSize = bytesPerBlock * numBlocks;
        } else {
            size_t sizePerVector =
                    (scalarQs_[h] ? scalarQs_[h]->code_size : sizeof(float) * dim_);
            
            headSize = (size_t)numVecs * sizePerVector;
        }
        maxSize = std::max(maxSize, headSize);
    }
    return maxSize;
}

size_t MultiHeadIVFFlat::getCpuVectorsEncodingSize_(idx_t numVecs) const {
    // For multi-head, we use the maximum encoding size across all heads
    size_t maxSize = 0;
    for (int h = 0; h < numHeads_; ++h) {
        size_t sizePerVector =
                (scalarQs_[h] ? scalarQs_[h]->code_size : sizeof(float) * dim_);
        
        size_t headSize = (size_t)numVecs * sizePerVector;
        maxSize = std::max(maxSize, headSize);
    }
    return maxSize;
}

std::vector<uint8_t> MultiHeadIVFFlat::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    // For multi-head, we need to handle each head's portion separately
    // This is a simplified implementation that assumes all heads use the same encoding
    // In practice, you might need to handle different scalar quantizers per head
    
    // Use the first head's scalar quantizer as representative
    int bitsPerCode = scalarQs_[0] ? scalarQs_[0]->bits : 32;

    auto up = unpackNonInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

std::vector<uint8_t> MultiHeadIVFFlat::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    // For multi-head, we need to handle each head's portion separately
    // This is a simplified implementation that assumes all heads use the same encoding
    
    // Use the first head's scalar quantizer as representative
    int bitsPerCode = scalarQs_[0] ? scalarQs_[0]->bits : 32;

    auto up = unpackInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packNonInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

void MultiHeadIVFFlat::appendVectors_(
        Tensor<float, 2, true>* vecs,
        Tensor<float, 2, true>* ivfCentroidResiduals,
        Tensor<idx_t, 1, true>* userIndices,
        Tensor<idx_t, 1, true>* uniqueLists,
        Tensor<idx_t, 1, true>* vectorsByUniqueList,
        Tensor<idx_t, 1, true>* uniqueListVectorStart,
        Tensor<idx_t, 1, true>* uniqueListStartOffset,
        Tensor<idx_t, 1, true>* assignedListIds,
        Tensor<idx_t, 1, true>* listOffsets,
        cudaStream_t stream) {
    
    // For multi-head IVF, the append operation is more complex because
    // we need to handle multiple heads with potentially different configurations
    
    // Append indices to the IVF lists (this is shared across heads)
    // runIVFIndicesAppend(
    //         *assignedListIds,
    //         *listOffsets,
    //         *userIndices,
    //         indicesOptions_,
    //         deviceListIndexPointers_,
    //         stream);

    // Append the encoded vectors to the IVF lists
    // Note: In multi-head setup, the list IDs in assignedListIds are global
    if (interleavedLayout_) {
        // For interleaved layout, we need to determine which head each vector belongs to
        // and use the appropriate scalar quantizer
        
        // This is a simplified implementation - in practice, you'd need to
        // partition the vectors by head and process each head separately
        // runIVFFlatInterleavedAppend(
        //         *assignedListIds,
        //         *listOffsets,
        //         *uniqueLists,
        //         *vectorsByUniqueList,
        //         *uniqueListVectorStart,
        //         *uniqueListStartOffset,
        //         useResidual_ ? *ivfCentroidResiduals : *vecs,
        //         scalarQs_[0].get(), // Simplified: use first head's quantizer
        //         deviceListDataPointers_,
        //         resources_,
        //         stream);
    } else {
        // runIVFFlatAppend(
        //         *assignedListIds,
        //         *listOffsets,
        //         useResidual_ ? *ivfCentroidResiduals : *vecs,
        //         scalarQs_[0].get(), // Simplified: use first head's quantizer
        //         deviceListDataPointers_,
        //         stream);
    }
}

void MultiHeadIVFFlat::search(
        std::vector<Index*>& coarseQuantizers,
        Tensor<float, 2, true>* queries,
        const std::vector<int>& nprobe,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices) {
    
    FAISS_ASSERT(coarseQuantizers.size() == numHeads_);
    FAISS_ASSERT(nprobe.size() == numHeads_);
    FAISS_ASSERT(k.size() == numHeads_);
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // Validate input parameters for each head
    idx_t totalQueries = 0;
    idx_t maxNprobe = 0;
    for (int h = 0; h < numHeads_; ++h) {
        FAISS_ASSERT(nprobe[h] <= GPU_MAX_SELECTION_K);
        FAISS_ASSERT(k[h] <= GPU_MAX_SELECTION_K);
        FAISS_ASSERT(queries[h].getSize(1) == dim_);
        FAISS_ASSERT(outDistances[h].getSize(0) == queries[h].getSize(0));
        FAISS_ASSERT(outIndices[h].getSize(0) == queries[h].getSize(0));
        
        totalQueries += queries[h].getSize(0);
        maxNprobe = std::max(maxNprobe, (idx_t)std::min(idx_t(nprobe[h]), nlists_[h]));
    }
    
    // Allocate multi-head data structures
    // For coarse distances: one tensor per head
    DeviceTensor<float, 2, true> coarseDistances[numHeads_];
    DeviceTensor<idx_t, 2, true> coarseIndices[numHeads_];
    DeviceTensor<float, 3, true> residualBase[numHeads_];
    
    // coarseDistancesPerHead.reserve(numHeads_);
    // coarseIndicesPerHead.reserve(numHeads_);
    // residualBasePerHead.reserve(numHeads_);
    
    // Allocate memory for each head
    for (int h = 0; h < numHeads_; ++h) {
        int adjustedNprobe = int(std::min(idx_t(nprobe[h]), nlists_[h]));
        
        coarseDistances[h] = DeviceTensor<float, 2, true>(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {queries[h].getSize(0), adjustedNprobe});
        coarseIndices[h] = DeviceTensor<idx_t, 2, true>(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {queries[h].getSize(0), adjustedNprobe});
        residualBase[h] = DeviceTensor<float, 3, true>(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {queries[h].getSize(0), adjustedNprobe, dim_});
    }
    
    searchCoarseQuantizer_(
        coarseQuantizers, 
        nprobe, 
        queries, 
        coarseDistances, 
        coarseIndices, 
        nullptr, 
        useResidual_ ? residualBase : nullptr);
    
    searchImpl_(
        queries, 
        coarseDistances, 
        coarseIndices, 
        residualBase, 
        k, 
        outDistances, 
        outIndices, 
        false);
    
}

void MultiHeadIVFFlat::searchPreassigned(
        std::vector<Index*>& coarseQuantizers,
        Tensor<float, 2, true>* vecs,
        Tensor<float, 2, true>* ivfDistances,
        Tensor<idx_t, 2, true>* ivfAssignments,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices,
        bool storePairs) {
    
    FAISS_ASSERT(coarseQuantizers.size() == numHeads_);
    FAISS_ASSERT(k.size() == numHeads_);
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    DeviceTensor<float, 3, true> ivfCentroids[numHeads_];
    GpuIndex* gpuQuantizer[numHeads_];
    
    // Process each head separately
    for (int h = 0; h < numHeads_; ++h) {
        FAISS_ASSERT(ivfDistances[h].getSize(0) == vecs[h].getSize(0));
        FAISS_ASSERT(ivfAssignments[h].getSize(0) == vecs[h].getSize(0));
        FAISS_ASSERT(outDistances[h].getSize(0) == vecs[h].getSize(0));
        FAISS_ASSERT(outIndices[h].getSize(0) == vecs[h].getSize(0));
        FAISS_ASSERT(vecs[h].getSize(1) == dim_);

        auto nprobe = ivfAssignments[h].getSize(1);

        ivfCentroids[h] = DeviceTensor<float, 3, true>(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecs[h].getSize(0), nprobe, dim_});
    
        gpuQuantizer[h] = tryCastGpuIndex(coarseQuantizers[h]);
        if (gpuQuantizer[h]) {
            // We can pass device pointers directly
            gpuQuantizer[h]->reconstruct_batch(
                    vecs[h].getSize(0) * nprobe,
                    ivfAssignments[h].data(),
                    ivfCentroids[h].data());
        } else {
            // CPU coarse quantizer
            auto cpuIVFCentroids =
                    std::vector<float>(vecs[h].getSize(0) * nprobe * dim_);
    
            // We need to copy `ivfAssignments` to the CPU, in order to pass to a
            // CPU index
            auto cpuIVFAssignments = ivfAssignments[h].copyToVector(stream);
    
            coarseQuantizers[h]->reconstruct_batch(
                    vecs[h].getSize(0) * nprobe,
                    cpuIVFAssignments.data(),
                    cpuIVFCentroids.data());
    
            ivfCentroids[h].copyFrom(cpuIVFCentroids, stream);
        }
    }

    searchImpl_(
        vecs, 
        ivfDistances, 
        ivfAssignments, 
        ivfCentroids, 
        k, 
        outDistances, 
        outIndices, 
        storePairs);

}

void MultiHeadIVFFlat::reconstruct_n(idx_t headId, idx_t i0, idx_t ni, float* out) {
    FAISS_ASSERT(headId < numHeads_);
    
    if (ni == 0) {
        return;
    }
    
    int warpSize = getWarpSizeCurrentDevice();
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // Iterate through lists for this head
    for (idx_t local_list = 0; local_list < nlists_[headId]; ++local_list) {
        size_t list_size = deviceListData_[headId][local_list]->numVecs;
        
        auto idlist = getListIndices(headId, local_list);
        
        for (idx_t offset = 0; offset < list_size; ++offset) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }
            
            // Handle interleaved vs non-interleaved layout
            if (interleavedLayout_) {
                auto vectorChunk = offset / warpSize;
                auto vectorWithinChunk = offset % warpSize;
                
                auto listDataPtr = (float*)deviceListData_[headId][local_list]->data.data();
                listDataPtr += vectorChunk * warpSize * dim_ + vectorWithinChunk;
                
                for (int d = 0; d < dim_; ++d) {
                    fromDevice<float>(
                            listDataPtr + warpSize * d,
                            out + (id - i0) * dim_ + d,
                            1,
                            stream);
                }
            } else {
                // Non-interleaved: vectors are stored consecutively
                auto listDataPtr = (float*)deviceListData_[headId][local_list]->data.data();
                listDataPtr += offset * dim_;
                
                fromDevice<float>(
                        listDataPtr,
                        out + (id - i0) * dim_,
                        dim_,
                        stream);
            }
        }
    }
}

void MultiHeadIVFFlat::searchImpl_(
        Tensor<float, 2, true>* queries,
        Tensor<float, 2, true>* coarseDistances,
        Tensor<idx_t, 2, true>* coarseIndices,
        Tensor<float, 3, true>* ivfCentroids,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices,
        bool storePairs) {
    
    FAISS_ASSERT(storePairs == false);
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    if (interleavedLayout_) {
        runMultiHeadIVFInterleavedScan(
            numHeads_,
            queries, 
            coarseIndices, 
            deviceListDataPointers_.data(), 
            deviceListIndexPointers_.data(), 
            indicesOptions_, 
            deviceListLengths_.data(), 
            k[0], 
            metric_, 
            useResidual_, 
            ivfCentroids, 
            scalarQs_[0].get(), 
            outDistances, 
            outIndices, 
            resources_);
    } else {
        
    }
}

} // namespace gpu
} // namespace faiss