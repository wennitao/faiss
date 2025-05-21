#include <faiss/gpu/impl/MultiHeadIVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh> // For runIVFFlatScan
#include <faiss/gpu/impl/IVFInterleaved.cuh> // For runIVFInterleavedScan
#include <faiss/gpu/impl/IVFAppend.cuh> // For runIVFIndicesAppend, runIVFFlatAppend, runIVFFlatInterleavedAppend
#include <faiss/gpu/impl/InterleavedCodes.h> // For pack/unpack
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/GpuIndex.h> // For tryCastGpuIndex
#include <faiss/gpu/utils/HostTensor.cuh> // For toHost
#include <faiss/gpu/utils/MathOps.cuh> // For runRemapCoarseIndicesToGlobal


namespace faiss {
namespace gpu {

MultiHeadIVFFlat::MultiHeadIVFFlat(
    GpuResources* res,
    int numHeads,
    int dim,
    idx_t nlistPerHead,
    faiss::MetricType metric,
    float metricArg,
    bool useResidual,
    const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead,
    bool interleavedLayout,
    IndicesOptions indicesOptions,
    MemorySpace space)
    : MultiHeadIVFBase(
          res,
          numHeads,
          dim,
          nlistPerHead,
          metric,
          metricArg,
          useResidual,
          interleavedLayout, // Pass to base
          indicesOptions,
          space) {
    scalarQs_.resize(numHeads);
    if (!scalarQsPerHead.empty()) {
        FAISS_THROW_IF_NOT(scalarQsPerHead.size() == (size_t)numHeads);
        for (int h = 0; h < numHeads; ++h) {
            if (scalarQsPerHead[h]) {
                scalarQs_[h] = std::make_unique<GpuScalarQuantizer>(res, *scalarQsPerHead[h]);
            }
        }
    }
}

MultiHeadIVFFlat::~MultiHeadIVFFlat() {}

size_t MultiHeadIVFFlat::getGpuVectorsEncodingSize_(idx_t headId, idx_t numVecs) const {
    FAISS_ASSERT(headId >= 0 && headId < numHeads_);
    const auto& currentSQ = scalarQs_[headId];

    if (interleavedLayout_) {
        idx_t bits = currentSQ ? currentSQ->bits : 32 /* float */;
        int warpSize = getWarpSizeCurrentDevice();
        idx_t bytesPerDimBlock = bits * warpSize / 8;
        idx_t bytesPerBlock = bytesPerDimBlock * dim_;
        idx_t numBlocks = utils::divUp(numVecs, warpSize);
        return bytesPerBlock * numBlocks;
    } else {
        size_t sizePerVector = (currentSQ ? currentSQ->code_size : sizeof(float) * dim_);
        return (size_t)numVecs * sizePerVector;
    }
}

size_t MultiHeadIVFFlat::getCpuVectorsEncodingSize_(idx_t headId, idx_t numVecs) const {
    FAISS_ASSERT(headId >= 0 && headId < numHeads_);
    const auto& currentSQ = scalarQs_[headId];
    size_t sizePerVector = (currentSQ ? currentSQ->code_size : sizeof(float) * dim_);
    return (size_t)numVecs * sizePerVector;
}

std::vector<uint8_t> MultiHeadIVFFlat::translateCodesToGpu_(
    idx_t headId,
    std::vector<uint8_t> codes,
    idx_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }
    FAISS_ASSERT(headId >= 0 && headId < numHeads_);
    const auto& currentSQ = scalarQs_[headId];
    int bitsPerCode = currentSQ ? currentSQ->bits : 32;

    auto up = unpackNonInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

std::vector<uint8_t> MultiHeadIVFFlat::translateCodesFromGpu_(
    idx_t headId,
    std::vector<uint8_t> codes,
    idx_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }
    FAISS_ASSERT(headId >= 0 && headId < numHeads_);
    const auto& currentSQ = scalarQs_[headId];
    int bitsPerCode = currentSQ ? currentSQ->bits : 32;

    auto up = unpackInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packNonInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

void MultiHeadIVFFlat::appendVectors_(
    idx_t headId,
    Tensor<float, 2, true>& vecs, // vecs or residuals for this head
    Tensor<float, 2, true>& ivfCentroidResiduals, // residuals for this head
    Tensor<idx_t, 1, true>& userIndices, // user-provided IDs for this head
    Tensor<idx_t, 1, true>& uniqueLists, // unique list IDs *within this head*
    Tensor<idx_t, 1, true>& vectorsByUniqueList,
    Tensor<idx_t, 1, true>& uniqueListVectorStart,
    Tensor<idx_t, 1, true>& uniqueListStartOffset,
    Tensor<idx_t, 1, true>& assignedListIds, // assigned list ID *within this head* for each input vector
    Tensor<idx_t, 1, true>& listOffsets, // offset *within the assigned list* for each input vector
    cudaStream_t stream) {

    FAISS_ASSERT(headId >= 0 && headId < numHeads_);
    const auto& currentSQ = scalarQs_[headId];

    // Map assignedListIds (local to head) and uniqueLists (local to head) to global list IDs
    // for runIVFIndicesAppend and the runIVFFlatXXXAppend kernels.
    // These kernels expect global list IDs to index into deviceListXXXPointers_.

    DeviceTensor<idx_t, 1, true> globalAssignedListIds(
        resources_, makeTempAlloc(AllocType::Other, stream), {assignedListIds.getSize(0)});
    runRemapCoarseIndicesToGlobal(assignedListIds, headId, numLists_, globalAssignedListIds, stream);
    
    DeviceTensor<idx_t, 1, true> globalUniqueLists(
        resources_, makeTempAlloc(AllocType::Other, stream), {uniqueLists.getSize(0)});
    runRemapCoarseIndicesToGlobal(uniqueLists, headId, numLists_, globalUniqueLists, stream);

    runIVFIndicesAppend(
        globalAssignedListIds, // Use global list IDs
        listOffsets,
        userIndices,
        indicesOptions_,
        deviceListIndexPointers_, // Global pointers from base
        stream);

    Tensor<float, 2, true>& dataToEncode = useResidual_ ? ivfCentroidResiduals : vecs;

    if (interleavedLayout_) {
        runIVFFlatInterleavedAppend(
            globalAssignedListIds, // Use global list IDs
            listOffsets,
            globalUniqueLists,     // Use global list IDs
            vectorsByUniqueList,
            uniqueListVectorStart,
            uniqueListStartOffset,
            dataToEncode,
            currentSQ.get(),
            deviceListDataPointers_, // Global pointers from base
            resources_,
            stream);
    } else {
        runIVFFlatAppend(
            globalAssignedListIds, // Use global list IDs
            listOffsets,
            dataToEncode,
            currentSQ.get(),
            deviceListDataPointers_, // Global pointers from base
            stream);
    }
}

void MultiHeadIVFFlat::search(
    std::vector<Index*>& coarseQuantizers,
    Tensor<float, 2, true>* queries,
    const std::vector<int>& nprobe,
    const std::vector<int>& k,
    Tensor<float, 2, true>* outDistances,
    Tensor<idx_t, 2, true>* outIndices) {
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    // FAISS_THROW_IF_NOT(queries.size() == (size_t)numHeads_);
    // ... (add other size assertions for inputs)

    std::vector<DeviceTensor<float, 2, true>> coarseDistances_temp(numHeads_);
    std::vector<DeviceTensor<idx_t, 2, true>> coarseIndices_temp(numHeads_); // Will store LOCAL list IDs
    std::vector<DeviceTensor<float, 3, true>> residualBase_temp;

    std::vector<Tensor<float, 2, true>*> coarseDistancesPerHead_ptr(numHeads_);
    std::vector<Tensor<idx_t, 2, true>*> coarseIndicesPerHead_ptr(numHeads_); // LOCAL list IDs
    std::vector<Tensor<float, 3, true>*> residualBasePerHead_ptr_vec;
    std::vector<Tensor<float, 3, true>*>* residualBasePerHead_ptr_to_vec = nullptr;

    if (useResidual_) {
        residualBase_temp.resize(numHeads_);
        residualBasePerHead_ptr_vec.resize(numHeads_);
        residualBasePerHead_ptr_to_vec = &residualBasePerHead_ptr_vec;
    }

    for (int h = 0; h < numHeads_; ++h) {
        FAISS_THROW_IF_NOT(queriesPerHead[h] != nullptr && coarseQuantizers[h] != nullptr);
        idx_t numQueries_h = queriesPerHead[h]->getSize(0);
        int nprobe_h = std::min((idx_t)nprobePerHead[h], numLists_); // nprobe per head vs numLists per head
        FAISS_THROW_IF_NOT(nprobe_h <= GPU_MAX_SELECTION_K);


        if (numQueries_h > 0) {
            coarseDistances_temp[h] = DeviceTensor<float, 2, true>(
                resources_, makeTempAlloc(AllocType::Other, stream), {numQueries_h, (size_t)nprobe_h});
            coarseDistancesPerHead_ptr[h] = &coarseDistances_temp[h];

            coarseIndices_temp[h] = DeviceTensor<idx_t, 2, true>( // Stores LOCAL list IDs
                resources_, makeTempAlloc(AllocType::Other, stream), {numQueries_h, (size_t)nprobe_h});
            coarseIndicesPerHead_ptr[h] = &coarseIndices_temp[h];

            if (useResidual_) {
                residualBase_temp[h] = DeviceTensor<float, 3, true>(
                    resources_, makeTempAlloc(AllocType::Other, stream), {numQueries_h, (size_t)nprobe_h, (size_t)dim_});
                residualBasePerHead_ptr_vec[h] = &residualBase_temp[h];
            }
        } else {
             coarseDistancesPerHead_ptr[h] = nullptr; // Or dummy
             coarseIndicesPerHead_ptr[h] = nullptr;
             if(useResidual_) residualBasePerHead_ptr_vec[h] = nullptr;
        }
    }

    searchCoarseQuantizer_( // This is from MultiHeadIVFBase
        coarseQuantizers,
        nprobePerHead, // Pass the original nprobe vector
        queriesPerHead,
        coarseDistancesPerHead_ptr,
        coarseIndicesPerHead_ptr, // Receives LOCAL list IDs
        nullptr, // No separate residuals needed here, residualBase is for centroids
        useResidual_ ? residualBasePerHead_ptr_to_vec : nullptr);

    searchImpl_(
        queriesPerHead,
        coarseDistancesPerHead_ptr,
        coarseIndicesPerHead_ptr, // Pass LOCAL list IDs
        useResidual_ ? residualBasePerHead_ptr_vec : std::vector<Tensor<float, 3, true>*>(numHeads_, nullptr),
        kPerHead,
        outDistancesPerHead,
        outIndicesPerHead,
        false);
}

void MultiHeadIVFFlat::searchPreassigned(
    std::vector<Index*>& coarseQuantizers,
    const std::vector<Tensor<float, 2, true>*>& vecsPerHead,
    const std::vector<Tensor<float, 2, true>*>& ivfDistancesPerHead, // Distances to coarse centroids
    const std::vector<Tensor<idx_t, 2, true>*>& ivfAssignmentsPerHead, // LOCAL list IDs
    const std::vector<int>& kPerHead,
    std::vector<Tensor<float, 2, true>*>& outDistancesPerHead,
    std::vector<Tensor<idx_t, 2, true>*>& outIndicesPerHead,
    bool storePairs) {

    auto stream = resources_->getDefaultStreamCurrentDevice();
    FAISS_THROW_IF_NOT(vecsPerHead.size() == (size_t)numHeads_);
    // ... (add other size assertions)

    std::vector<DeviceTensor<float, 3, true>> ivfCentroids_temp(numHeads_);
    std::vector<Tensor<float, 3, true>*> ivfCentroidsPerHead_ptr(numHeads_);

    for (int h = 0; h < numHeads_; ++h) {
        FAISS_THROW_IF_NOT(vecsPerHead[h] && ivfAssignmentsPerHead[h] && coarseQuantizers[h]);
        idx_t numVecs_h = vecsPerHead[h]->getSize(0);
        idx_t nprobe_h = ivfAssignmentsPerHead[h]->getSize(1);

        if (numVecs_h > 0 && nprobe_h > 0) {
             ivfCentroids_temp[h] = DeviceTensor<float, 3, true>(
                resources_, makeTempAlloc(AllocType::Other, stream), {numVecs_h, nprobe_h, (size_t)dim_});
            ivfCentroidsPerHead_ptr[h] = &ivfCentroids_temp[h];

            auto gpuQuantizer = tryCastGpuIndex(coarseQuantizers[h]);
            if (gpuQuantizer) {
                gpuQuantizer->reconstruct_batch(
                    numVecs_h * nprobe_h,
                    ivfAssignmentsPerHead[h]->data(), // LOCAL list IDs
                    ivfCentroids_temp[h].data());
            } else {
                auto cpuIVFAssignments = ivfAssignmentsPerHead[h]->copyToVector(stream);
                std::vector<float> cpuIVFCentroids(numVecs_h * nprobe_h * dim_);
                coarseQuantizers[h]->reconstruct_batch(
                    numVecs_h * nprobe_h,
                    cpuIVFAssignments.data(),
                    cpuIVFCentroids.data());
                ivfCentroids_temp[h].copyFrom(cpuIVFCentroids, stream);
            }
        } else {
            ivfCentroidsPerHead_ptr[h] = nullptr;
        }
    }

    searchImpl_(
        vecsPerHead,
        ivfDistancesPerHead,
        ivfAssignmentsPerHead, // Pass LOCAL list IDs
        ivfCentroidsPerHead_ptr,
        kPerHead,
        outDistancesPerHead,
        outIndicesPerHead,
        storePairs);
}

void MultiHeadIVFFlat::reconstruct_n(idx_t headId, idx_t i0, idx_t ni, float* out) {
    // This is a simplified adaptation. A full GPU-based reconstruction would be more complex,
    // especially with SQs and interleaved layout. This version is CPU-centric after fetching list data.
    if (ni == 0) return;
    FAISS_ASSERT(headId >= 0 && headId < numHeads_);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const auto& currentSQ = scalarQs_[headId];

    for (idx_t list_no_in_head = 0; list_no_in_head < numLists_; ++list_no_in_head) {
        size_t list_size = getListLength(headId, list_no_in_head);
        if (list_size == 0) continue;

        std::vector<idx_t> idlist_cpu = getListIndices(headId, list_no_in_head); // User IDs
        std::vector<uint8_t> list_data_gpu_format = getListVectorData(headId, list_no_in_head, true);
        
        // Translate from GPU format (potentially interleaved) to CPU format (non-interleaved)
        std::vector<uint8_t> list_data_cpu_format = translateCodesFromGpu_(headId, std::move(list_data_gpu_format), list_size);

        for (idx_t offset_in_list = 0; offset_in_list < list_size; ++offset_in_list) {
            idx_t user_id = idlist_cpu[offset_in_list];
            if (!(user_id >= i0 && user_id < i0 + ni)) {
                continue;
            }

            float* target_output_vector = out + (user_id - i0) * dim_;
            
            if (currentSQ) {
                // Dequantize
                const uint8_t* encoded_vector_ptr = list_data_cpu_format.data() + offset_in_list * currentSQ->code_size;
                currentSQ->decode(encoded_vector_ptr, target_output_vector, 1);
            } else {
                // Float data
                const float* float_vector_ptr = reinterpret_cast<const float*>(list_data_cpu_format.data()) + offset_in_list * dim_;
                memcpy(target_output_vector, float_vector_ptr, sizeof(float) * dim_);
            }
        }
    }
}


void MultiHeadIVFFlat::searchImpl_(
    const std::vector<Tensor<float, 2, true>*>& queriesPerHead,
    const std::vector<Tensor<float, 2, true>*>& coarseDistancesPerHead,
    const std::vector<Tensor<idx_t, 2, true>*>& coarseIndicesPerHead, // LOCAL list IDs
    const std::vector<Tensor<float, 3, true>*>& ivfCentroidsPerHead, // Residual base
    const std::vector<int>& kPerHead,
    std::vector<Tensor<float, 2, true>*>& outDistancesPerHead,
    std::vector<Tensor<idx_t, 2, true>*>& outIndicesPerHead,
    bool storePairs) {

    FAISS_ASSERT(storePairs == false); // Not supported in this adaptation
    auto stream = resources_->getDefaultStreamCurrentDevice();

    std::vector<DeviceTensor<idx_t, 2, true>> globalCoarseIndices_temp(numHeads_);

    for (int h = 0; h < numHeads_; ++h) {
        if (!queriesPerHead[h] || queriesPerHead[h]->getSize(0) == 0 || !coarseIndicesPerHead[h]) {
            continue; // Skip empty queries for this head
        }
        
        // Remap local coarse indices to global coarse indices for scan kernels
        globalCoarseIndices_temp[h] = DeviceTensor<idx_t, 2, true>(
            resources_, makeTempAlloc(AllocType::Other, stream),
            {coarseIndicesPerHead[h]->getSize(0), coarseIndicesPerHead[h]->getSize(1)});
        
        runRemapCoarseIndicesToGlobal(
            *coarseIndicesPerHead[h], // Input: local list IDs for head h
            h,
            numLists_, // numLists per head
            globalCoarseIndices_temp[h], // Output: global list IDs
            stream
        );

        const auto& currentSQ = scalarQs_[h];
        Tensor<float, 3, true>* currentIvfCentroids = (ivfCentroidsPerHead.empty() || ivfCentroidsPerHead[h] == nullptr) ? 
                                                       nullptr : ivfCentroidsPerHead[h];
        // Create a dummy if null to pass to kernels that expect a non-null (but possibly empty) tensor
        DeviceTensor<float, 3, true> dummyCentroids; 
        if (!currentIvfCentroids && useResidual_) { // Kernels might still expect a valid tensor object
             dummyCentroids = DeviceTensor<float, 3, true>(resources_, makeTempAlloc(AllocType::Other, stream), {0,0,0});
             currentIvfCentroids = &dummyCentroids;
        }


        if (interleavedLayout_) {
            runIVFInterleavedScan(
                *queriesPerHead[h],
                globalCoarseIndices_temp[h], // Use global list IDs
                deviceListDataPointers_,     // From base, global
                deviceListIndexPointers_,    // From base, global
                indicesOptions_,
                deviceListLengths_,          // From base, global
                kPerHead[h],
                metric_,
                useResidual_,
                currentIvfCentroids, // Pass the specific head's ivfCentroids or dummy
                currentSQ.get(),
                *outDistancesPerHead[h],
                *outIndicesPerHead[h],
                resources_);
        } else {
            runIVFFlatScan(
                *queriesPerHead[h],
                globalCoarseIndices_temp[h], // Use global list IDs
                deviceListDataPointers_,     // From base, global
                deviceListIndexPointers_,    // From base, global
                indicesOptions_,
                deviceListLengths_,          // From base, global
                maxListLength_, // From base
                kPerHead[h],
                metric_,
                useResidual_,
                currentIvfCentroids, // Pass the specific head's ivfCentroids or dummy
                currentSQ.get(),
                *outDistancesPerHead[h],
                *outIndicesPerHead[h],
                resources_);
        }

        if (indicesOptions_ == INDICES_CPU) {
            HostTensor<idx_t, 2, true> hostOutIndices(*outIndicesPerHead[h], stream);
            ivfOffsetToUserIndex( // This function needs adaptation for multi-head's listOffsetToUserIndex_
                hostOutIndices.data(),
                h, // Pass headId
                numLists_, // Pass numLists per head
                hostOutIndices.getSize(0),
                hostOutIndices.getSize(1),
                listOffsetToUserIndex_); // This is std::vector<std::vector<std::vector<idx_t>>>
            outIndicesPerHead[h]->copyFrom(hostOutIndices, stream);
        }
    }
}

} // namespace gpu
} // namespace faiss