/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/ThrustUtils.cuh>

namespace faiss {
namespace gpu {

IVFBaseContiguous::IVFBaseContiguous(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        bool interleavedLayout,
        bool useResidual,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFBase(resources,
                 dim,
                 nlist,
                 metric,
                 metricArg,
                 interleavedLayout,
                 useResidual,
                 indicesOptions,
                 space),
          contiguousData_(
                  resources,
                  AllocInfo(
                          AllocType::IVFLists,
                          getCurrentDevice(),
                          space,
                          resources->getDefaultStreamCurrentDevice())),
          contiguousIndices_(
                  resources,
                  AllocInfo(
                          AllocType::IVFLists,
                          getCurrentDevice(),
                          space,
                          resources->getDefaultStreamCurrentDevice())),
          totalDataSize_(0),
          totalIndicesSize_(0),
          growthFactor_(1.2) {
    // Initialize offset arrays
    listDataOffsets_.resize(nlist, 0);
    listIndicesOffsets_.resize(nlist, 0);
    
    // Call reset to initialize the necessary structures
    reset();
}

IVFBaseContiguous::~IVFBaseContiguous() {
}

void IVFBaseContiguous::reset() {
    // Call parent class reset to initialize the base structures
    IVFBase::reset();
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // Reset our contiguous memory blocks
    contiguousData_.clear();
    contiguousIndices_.clear();
    
    // Reset offsets and sizes
    std::fill(listDataOffsets_.begin(), listDataOffsets_.end(), 0);
    std::fill(listIndicesOffsets_.begin(), listIndicesOffsets_.end(), 0);
    
    totalDataSize_ = 0;
    totalIndicesSize_ = 0;
    
    // Update device pointers
    updateDeviceListInfo_(stream);
}

void IVFBaseContiguous::reserveMemory(idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // Calculate average vectors per list
    auto vecsPerList = numVecs / numLists_;
    if (vecsPerList < 1) {
        return;
    }
    
    // Calculate the total size needed for all lists
    auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);
    auto totalDataBytes = bytesPerDataList * numLists_;
    
    // Reserve with some extra space to minimize future reallocations
    auto reserveDataSize = totalDataBytes * growthFactor_;
    contiguousData_.reserve(reserveDataSize, stream);
    
    // Reserve for indices if needed
    if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
        size_t bytesPerIndex = (indicesOptions_ == INDICES_32_BIT) ? 
                               sizeof(int) : sizeof(idx_t);
        auto totalIndicesBytes = vecsPerList * bytesPerIndex * numLists_;
        auto reserveIndicesSize = totalIndicesBytes * growthFactor_;
        contiguousIndices_.reserve(reserveIndicesSize, stream);
    }
    
    // Update device info
    updateDeviceListInfo_(stream);
}

size_t IVFBaseContiguous::reclaimMemory() {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    size_t totalReclaimed = 0;
    
    // Reclaim memory for data
    if (contiguousData_.capacity() > totalDataSize_) {
        // Leave some slack to avoid frequent reallocation
        size_t newCapacity = totalDataSize_ * growthFactor_;
        
        // Allocate new buffer
        DeviceVector<uint8_t> newData(
                resources_,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space_,
                        stream));
        
        // Only reserve and copy if we have data
        if (totalDataSize_ > 0) {
            newData.reserve(newCapacity, stream);
            
            // Copy existing data
            CUDA_VERIFY(cudaMemcpyAsync(
                    newData.data(),
                    contiguousData_.data(),
                    totalDataSize_,
                    cudaMemcpyDeviceToDevice,
                    stream));
        }
        
        // Calculate reclaimed bytes
        totalReclaimed += contiguousData_.capacity() - newCapacity;
        
        // Swap with new data
        contiguousData_ = std::move(newData);
    }
    
    // Do the same for indices if needed
    if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
        if (contiguousIndices_.capacity() > totalIndicesSize_) {
            // Leave some slack
            size_t newCapacity = totalIndicesSize_ * growthFactor_;
            
            // Allocate new buffer
            DeviceVector<uint8_t> newIndices(
                    resources_,
                    AllocInfo(
                            AllocType::IVFLists,
                            getCurrentDevice(),
                            space_,
                            stream));
            
            // Only reserve and copy if we have data
            if (totalIndicesSize_ > 0) {
                newIndices.reserve(newCapacity, stream);
                
                // Copy existing indices
                CUDA_VERIFY(cudaMemcpyAsync(
                        newIndices.data(),
                        contiguousIndices_.data(),
                        totalIndicesSize_,
                        cudaMemcpyDeviceToDevice,
                        stream));
            }
            
            // Calculate reclaimed bytes
            totalReclaimed += contiguousIndices_.capacity() - newCapacity;
            
            // Swap with new indices
            contiguousIndices_ = std::move(newIndices);
        }
    }
    
    // Update device list pointers
    updateDeviceListInfo_(stream);
    
    return totalReclaimed;
}

void IVFBaseContiguous::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    // Call the optimized version with automatic size calculation
    addEncodedVectorsToList_(listId, codes, indices, numVecs, 0);
}

void IVFBaseContiguous::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs,
        size_t preReserveSize) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // This list must already exist
    FAISS_ASSERT(listId < numLists_);
    
    // The list must be empty (for now - could be extended to support appending)
    auto& listCodes = deviceListData_[listId];
    FAISS_ASSERT(listCodes->numVecs == 0);
    
    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }
    
    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);
    
    // Calculate how much total size we'll need after adding this data
    size_t newTotalDataSize = totalDataSize_ + gpuListSizeInBytes;
    
    // Check if we need to resize our contiguous buffer
    if (newTotalDataSize > contiguousData_.capacity()) {
        // Use preReserveSize if provided, otherwise calculate our own
        size_t reserveSize = preReserveSize > 0 ? 
                             totalDataSize_ + preReserveSize : 
                             newTotalDataSize * growthFactor_;
        
        // Create a new buffer with larger capacity
        DeviceVector<uint8_t> newContiguousData(
                resources_,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space_,
                        stream));
        
        newContiguousData.reserve(reserveSize, stream);
        
        // Copy existing data if any
        if (totalDataSize_ > 0) {
            CUDA_VERIFY(cudaMemcpyAsync(
                    newContiguousData.data(),
                    contiguousData_.data(),
                    totalDataSize_,
                    cudaMemcpyDeviceToDevice,
                    stream));
        }
        
        // Swap with the new buffer
        contiguousData_ = std::move(newContiguousData);
    }
    
    // Set the offset for this list
    listDataOffsets_[listId] = totalDataSize_;
    
    // Translate the codes as needed to our preferred form
    std::vector<uint8_t> codesV(cpuListSizeInBytes);
    std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
    auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);
    
    // Copy the codes to our contiguous buffer at the appropriate offset
    CUDA_VERIFY(cudaMemcpyAsync(
            contiguousData_.data() + totalDataSize_,
            translatedCodes.data(),
            gpuListSizeInBytes,
            cudaMemcpyHostToDevice,
            stream));
    
    // Update the total size
    totalDataSize_ = newTotalDataSize;
    
    // Update the list's metadata
    listCodes->numVecs = numVecs;
    
    // Handle the indices similarly
    if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
        size_t indexSize = indicesOptions_ == INDICES_32_BIT ? 
                          sizeof(int) : sizeof(idx_t);
        size_t indicesSizeInBytes = numVecs * indexSize;
        size_t newTotalIndicesSize = totalIndicesSize_ + indicesSizeInBytes;
        
        // Check if we need to resize indices buffer
        if (newTotalIndicesSize > contiguousIndices_.capacity()) {
            // Calculate reserve size
            size_t reserveSize = newTotalIndicesSize * growthFactor_;
            
            // Create a new buffer with larger capacity
            DeviceVector<uint8_t> newContiguousIndices(
                    resources_,
                    AllocInfo(
                            AllocType::IVFLists,
                            getCurrentDevice(),
                            space_,
                            stream));
            
            newContiguousIndices.reserve(reserveSize, stream);
            
            // Copy existing indices if any
            if (totalIndicesSize_ > 0) {
                CUDA_VERIFY(cudaMemcpyAsync(
                        newContiguousIndices.data(),
                        contiguousIndices_.data(),
                        totalIndicesSize_,
                        cudaMemcpyDeviceToDevice,
                        stream));
            }
            
            // Swap with the new buffer
            contiguousIndices_ = std::move(newContiguousIndices);
        }
        
        // Set the offset for this list's indices
        listIndicesOffsets_[listId] = totalIndicesSize_;
        
        auto& listIndices = deviceListIndices_[listId];
        
        // Copy indices based on type
        if (indicesOptions_ == INDICES_32_BIT) {
            // Convert indices to 32-bit
            std::vector<int> indices32(numVecs);
            for (idx_t i = 0; i < numVecs; ++i) {
                auto ind = indices[i];
                FAISS_ASSERT(ind <= (idx_t)std::numeric_limits<int>::max());
                indices32[i] = (int)ind;
            }
            
            // Copy to contiguous buffer
            CUDA_VERIFY(cudaMemcpyAsync(
                    contiguousIndices_.data() + totalIndicesSize_,
                    indices32.data(),
                    indicesSizeInBytes,
                    cudaMemcpyHostToDevice,
                    stream));
        } else {
            // Copy 64-bit indices directly
            CUDA_VERIFY(cudaMemcpyAsync(
                    contiguousIndices_.data() + totalIndicesSize_,
                    indices,
                    indicesSizeInBytes,
                    cudaMemcpyHostToDevice,
                    stream));
        }
        
        // Update the total indices size
        totalIndicesSize_ = newTotalIndicesSize;
        
        // Update the list's metadata
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_CPU) {
        // Store indices on CPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
        auto& userIndices = listOffsetToUserIndex_[listId];
        userIndices.insert(userIndices.begin(), indices, indices + numVecs);
    } else {
        // Indices not stored
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
    }
    
    // Update device pointers
    updateDeviceListInfo_(std::vector<idx_t>{listId}, stream);
    
    // We update this as well, since the multi-pass algorithm uses it
    maxListLength_ = std::max(maxListLength_, numVecs);
}

void IVFBaseContiguous::updateDeviceListInfo_(cudaStream_t stream) {
    std::vector<idx_t> listIds(numLists_);
    for (idx_t i = 0; i < numLists_; ++i) {
        listIds[i] = i;
    }
    
    updateDeviceListInfo_(listIds, stream);
}

void IVFBaseContiguous::updateDeviceListInfo_(
        const std::vector<idx_t>& listIds,
        cudaStream_t stream) {
    idx_t listSize = listIds.size();
    HostTensor<idx_t, 1, true> hostListsToUpdate({listSize});
    HostTensor<idx_t, 1, true> hostNewListLength({listSize});
    HostTensor<void*, 1, true> hostNewDataPointers({listSize});
    HostTensor<void*, 1, true> hostNewIndexPointers({listSize});
    
    for (idx_t i = 0; i < listSize; ++i) {
        auto listId = listIds[i];
        auto& data = deviceListData_[listId];
        auto& indices = deviceListIndices_[listId];
        
        hostListsToUpdate[i] = listId;
        hostNewListLength[i] = data->numVecs;
        
        // If we have data for this list, set the pointer to the offset in our
        // contiguous buffer
        if (data->numVecs > 0) {
            hostNewDataPointers[i] = contiguousData_.data() + listDataOffsets_[listId];
        } else {
            hostNewDataPointers[i] = nullptr;
        }
        
        // Same for indices
        if (indices->numVecs > 0 && 
            ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT))) {
            hostNewIndexPointers[i] = contiguousIndices_.data() + listIndicesOffsets_[listId];
        } else {
            hostNewIndexPointers[i] = nullptr;
        }
    }
    
    // Copy the update sets to the GPU
    DeviceTensor<idx_t, 1, true> listsToUpdate(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostListsToUpdate);
    DeviceTensor<idx_t, 1, true> newListLength(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewListLength);
    DeviceTensor<void*, 1, true> newDataPointers(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewDataPointers);
    DeviceTensor<void*, 1, true> newIndexPointers(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewIndexPointers);
    
    // Update all pointers to the lists on the device
    runUpdateListPointers(
            listsToUpdate,
            newListLength,
            newDataPointers,
            newIndexPointers,
            deviceListLengths_,
            deviceListDataPointers_,
            deviceListIndexPointers_,
            stream);
}

void IVFBaseContiguous::addEncodedVectorsToListWithReservation(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs,
        size_t preReserveSize) {
    // Use our optimized internal implementation
    addEncodedVectorsToList_(listId, codes, indices, numVecs, preReserveSize);
}

void IVFBaseContiguous::copyInvertedListsFrom(const InvertedLists* ivf) {
    // First calculate total size needed
    size_t totalDataSize = 0;
    size_t totalIndicesSize = 0;
    
    idx_t nlist = ivf ? ivf->nlist : 0;
    for (idx_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        if (listSize > 0) {
            totalDataSize += getGpuVectorsEncodingSize_(listSize);
            
            if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
                size_t indexSize = indicesOptions_ == INDICES_32_BIT ? 
                                  sizeof(int) : sizeof(idx_t);
                totalIndicesSize += listSize * indexSize;
            }
        }
    }
    
    // Reserve space for all lists at once
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    // Add some extra space to avoid immediate reallocation
    size_t reserveDataSize = totalDataSize * growthFactor_;
    size_t reserveIndicesSize = totalIndicesSize * growthFactor_;
    
    // Reserve space
    if (reserveDataSize > 0) {
        contiguousData_.reserve(reserveDataSize, stream);
    }
    
    if (reserveIndicesSize > 0 && 
        ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT))) {
        contiguousIndices_.reserve(reserveIndicesSize, stream);
    }
    
    // Now add each list
    size_t dataOffset = 0;
    size_t indicesOffset = 0;
    
    for (idx_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        if (listSize > 0) {
            // Store the offset for this list
            listDataOffsets_[i] = dataOffset;
            listIndicesOffsets_[i] = indicesOffset;
            
            // Add the vectors with our pre-allocated space
            addEncodedVectorsToList_(
                    i, ivf->get_codes(i), ivf->get_ids(i), listSize, 0);
            
            // Update offsets for next list
            dataOffset += getGpuVectorsEncodingSize_(listSize);
            
            if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
                size_t indexSize = indicesOptions_ == INDICES_32_BIT ? 
                                  sizeof(int) : sizeof(idx_t);
                indicesOffset += listSize * indexSize;
            }
        }
    }
}

} // namespace gpu
} // namespace faiss
