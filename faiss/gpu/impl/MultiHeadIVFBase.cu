/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/MultiHeadIVFBase.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/ThrustUtils.cuh>
#include <limits>
#include <unordered_map>

namespace faiss {
namespace gpu {

MultiHeadIVFBase::DeviceIVFList::DeviceIVFList(GpuResources* res, const AllocInfo& info)
        : data(res, info), numVecs(0) {}

MultiHeadIVFBase::MultiHeadIVFBase(
        GpuResources* resources,
        int numHeads,
        int dim,
        std::vector<idx_t>& nlists,
        faiss::MetricType metric,
        float metricArg,
        bool interleavedLayout,
        bool useResidual,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : resources_(resources),
          metric_(metric),
          metricArg_(metricArg),
          numHeads_(numHeads),
          dim_(dim),
          nlists_(nlists),
          useResidual_(useResidual),
          interleavedLayout_(interleavedLayout),
          indicesOptions_(indicesOptions),
          space_(space),
          maxListLength_(0) {

    // std::cerr << "interleavedLayout: " << interleavedLayout_ << std::endl;
    
    FAISS_THROW_IF_NOT(numHeads > 0);
    FAISS_THROW_IF_NOT(nlists.size() == numHeads);
    
    // Initialize device vectors for each head
    // deviceListDataPointers_.resize(numHeads);
    // deviceListIndexPointers_.resize(numHeads);
    // deviceListLengths_.resize(numHeads);
    // deviceListData_.resize(numHeads, std::vector<std::unique_ptr<DeviceIVFList>>());
    // deviceListIndices_.resize(numHeads, std::vector<std::unique_ptr<DeviceIVFList>>());
    // translatedCodes_.resize(numHeads, std::vector<uint8_t*>());
    // listOffsetToUserIndex_.resize(numHeads, std::vector<std::vector<idx_t>>());
    
    for (int h = 0; h < numHeads; ++h) {
        deviceListDataPointers_.emplace_back(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice()));
        
        deviceListIndexPointers_.emplace_back(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice()));
        
        deviceListLengths_.emplace_back(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice()));
    }

    // Initialize centroids array for each head
    ivfCentroids_ = new DeviceTensor<float, 2, true>[numHeads_];
    
    reset();
}

MultiHeadIVFBase::~MultiHeadIVFBase() {
    // if (isTranslatedCodesStored_) {
    //     // std::cerr << "Freeing translated codes" << std::endl;
    //     for (int h = 0; h < numHeads_; ++h) {
    //         for (auto& ptr : translatedCodes_[h]) {
    //             cudaFreeHost(ptr);
    //         }
    //     }
    // }

    deviceListData_.clear();
    deviceListIndices_.clear();
    // translatedCodes_.clear();
    listOffsetToUserIndex_.clear();
    
    delete[] ivfCentroids_;
}

void MultiHeadIVFBase::reserveMemory(idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    for (int h = 0; h < numHeads_; ++h) {
        auto vecsPerList = numVecs / deviceListData_[h].size();
        if (vecsPerList < 1) {
            continue;
        }

        auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

        for (auto& list : deviceListData_[h]) {
            list->data.reserve(bytesPerDataList, stream);
        }

        if ((indicesOptions_ == INDICES_32_BIT) ||
            (indicesOptions_ == INDICES_64_BIT)) {
            // Reserve for index lists as well
            size_t bytesPerIndexList = vecsPerList *
                    (indicesOptions_ == INDICES_32_BIT ? sizeof(int)
                                                       : sizeof(idx_t));

            for (auto& list : deviceListIndices_[h]) {
                list->data.reserve(bytesPerIndexList, stream);
            }
        }

        // Update device info for all lists, since the base pointers may have changed
        updateDeviceListInfo_(h, stream);
    }
}

void MultiHeadIVFBase::reset() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // if (isTranslatedCodesStored_) {
    //     for (int h = 0; h < numHeads_; ++h) {
    //         for (auto& ptr : translatedCodes_[h]) {
    //             cudaFreeHost(ptr);
    //         }
    //     }
    // }

    deviceListData_.clear();
    deviceListIndices_.clear();
    // translatedCodes_.clear();
    listOffsetToUserIndex_.clear();

    // Clear all head data
    for (int h = 0; h < numHeads_; ++h) {
        deviceListData_.emplace_back (std::vector<std::unique_ptr<DeviceIVFList>>());
        deviceListIndices_.emplace_back (std::vector<std::unique_ptr<DeviceIVFList>>());
        // translatedCodes_.emplace_back (std::vector<uint8_t*>());
        listOffsetToUserIndex_.emplace_back (std::vector<std::vector<idx_t>>());

        auto info = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, stream);

        // Initialize lists for this head
        for (idx_t i = 0; i < nlists_[h]; ++i) {
            deviceListData_[h].emplace_back(std::unique_ptr<DeviceIVFList>(
                    new DeviceIVFList(resources_, info)));

            deviceListIndices_[h].emplace_back(std::unique_ptr<DeviceIVFList>(
                    new DeviceIVFList(resources_, info)));

            listOffsetToUserIndex_[h].emplace_back(std::vector<idx_t>());
            // translatedCodes_[h].emplace_back(nullptr);
        }

        deviceListDataPointers_[h].resize(nlists_[h], stream);
        deviceListDataPointers_[h].setAll(nullptr, stream);

        deviceListIndexPointers_[h].resize(nlists_[h], stream);
        deviceListIndexPointers_[h].setAll(nullptr, stream);

        deviceListLengths_[h].resize(nlists_[h], stream);
        deviceListLengths_[h].setAll(0, stream);
    }

    maxListLength_ = 0;
    // isTranslatedCodesStored_ = false;
}

void MultiHeadIVFBase::initTranslatedCodes (
    std::vector<std::vector<uint8_t*>>& translatedCodes) {
    translatedCodes.resize(numHeads_);
    for (int h = 0; h < numHeads_; ++h) {
        translatedCodes[h].resize(nlists_[h], nullptr);
    }
}

idx_t MultiHeadIVFBase::getDim() const {
    return dim_;
}

size_t MultiHeadIVFBase::reclaimMemory() {
    // Reclaim all unused memory exactly
    return reclaimMemory_(true);
}

size_t MultiHeadIVFBase::reclaimMemory_(bool exact) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    size_t totalReclaimed = 0;

    for (int h = 0; h < numHeads_; ++h) {
        for (idx_t i = 0; i < deviceListData_[h].size(); ++i) {
            auto& data = deviceListData_[h][i]->data;
            totalReclaimed += data.reclaim(exact, stream);
            deviceListDataPointers_[h].setAt(i, (void*)data.data(), stream);
        }

        for (idx_t i = 0; i < deviceListIndices_[h].size(); ++i) {
            auto& indices = deviceListIndices_[h][i]->data;
            totalReclaimed += indices.reclaim(exact, stream);
            deviceListIndexPointers_[h].setAt(i, (void*)indices.data(), stream);
        }

        // Update device info for all lists, since the base pointers may have changed
        updateDeviceListInfo_(h, stream);
    }

    return totalReclaimed;
}

void MultiHeadIVFBase::updateDeviceListInfo_(int headId, cudaStream_t stream) {
    std::vector<idx_t> listIds(deviceListData_[headId].size());
    for (idx_t i = 0; i < deviceListData_[headId].size(); ++i) {
        listIds[i] = i;
    }

    updateDeviceListInfo_(headId, listIds, stream);
}

void MultiHeadIVFBase::updateDeviceListInfo_(
        int headId, 
        const std::vector<idx_t>& listIds,
        cudaStream_t stream) {
    idx_t listSize = listIds.size();
    HostTensor<idx_t, 1, true> hostListsToUpdate({listSize});
    HostTensor<idx_t, 1, true> hostNewListLength({listSize});
    HostTensor<void*, 1, true> hostNewDataPointers({listSize});
    HostTensor<void*, 1, true> hostNewIndexPointers({listSize});

    for (idx_t i = 0; i < listSize; ++i) {
        auto listId = listIds[i];
        auto& data = deviceListData_[headId][listId];
        auto& indices = deviceListIndices_[headId][listId];

        hostListsToUpdate[i] = listId;
        hostNewListLength[i] = data->numVecs;
        hostNewDataPointers[i] = data->data.data();
        hostNewIndexPointers[i] = indices->data.data();
    }

    // Copy the above update sets to the GPU
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

    // Update all pointers to the lists on the device that may have changed
    runUpdateListPointers(
            listsToUpdate,
            newListLength,
            newDataPointers,
            newIndexPointers,
            deviceListLengths_[headId],
            deviceListDataPointers_[headId],
            deviceListIndexPointers_[headId],
            stream);
}

idx_t MultiHeadIVFBase::getNumHeads() const {
    return numHeads_;
}

idx_t MultiHeadIVFBase::getNumLists(idx_t headId) const {
    FAISS_THROW_IF_NOT_FMT(
            headId < numHeads_,
            "Head %ld is out of bounds (%d heads total)",
            headId,
            numHeads_);
    return nlists_[headId];
}

idx_t MultiHeadIVFBase::getListLength(idx_t headId, idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(
            headId < numHeads_,
            "Head %ld is out of bounds (%d heads total)",
            headId,
            numHeads_);
    FAISS_THROW_IF_NOT_FMT(
            listId < nlists_[headId],
            "IVF list %ld is out of bounds (%ld lists total) for head %ld",
            listId,
            nlists_[headId],
            headId);
    FAISS_ASSERT(listId < deviceListLengths_[headId].size());
    FAISS_ASSERT(listId < deviceListData_[headId].size());

    return deviceListData_[headId][listId]->numVecs;
}

std::vector<idx_t> MultiHeadIVFBase::getListIndices(idx_t headId, idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(
            headId < numHeads_,
            "Head %ld is out of bounds (%d heads total)",
            headId,
            numHeads_);
    FAISS_THROW_IF_NOT_FMT(
            listId < nlists_[headId],
            "IVF list %ld is out of bounds (%ld lists total) for head %ld",
            listId,
            nlists_[headId],
            headId);
    FAISS_ASSERT(listId < deviceListData_[headId].size());
    FAISS_ASSERT(listId < deviceListLengths_[headId].size());

    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (indicesOptions_ == INDICES_32_BIT) {
        // The data is stored as int32 on the GPU
        FAISS_ASSERT(listId < deviceListIndices_[headId].size());

        auto intInd = deviceListIndices_[headId][listId]->data.copyToHost<int>(stream);

        std::vector<idx_t> out(intInd.size());
        for (size_t i = 0; i < intInd.size(); ++i) {
            out[i] = (idx_t)intInd[i];
        }

        return out;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        // The data is stored as int64 on the GPU
        FAISS_ASSERT(listId < deviceListIndices_[headId].size());

        return deviceListIndices_[headId][listId]->data.copyToHost<idx_t>(stream);
    } else if (indicesOptions_ == INDICES_CPU) {
        // The data is not stored on the GPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_[headId].size());

        auto& userIds = listOffsetToUserIndex_[headId][listId];

        // We should have the same number of indices on the CPU as we do vectors
        // encoded on the GPU
        FAISS_ASSERT(userIds.size() == deviceListData_[headId][listId]->numVecs);

        // this will return a copy
        return userIds;
    } else {
        // unhandled indices type (includes INDICES_IVF)
        FAISS_ASSERT(false);
        return std::vector<idx_t>();
    }
}

std::vector<uint8_t> MultiHeadIVFBase::getListVectorData(idx_t headId, idx_t listId, bool gpuFormat) const {
    FAISS_THROW_IF_NOT_FMT(
            headId < numHeads_,
            "Head %ld is out of bounds (%d heads total)",
            headId,
            numHeads_);
    FAISS_THROW_IF_NOT_FMT(
            listId < nlists_[headId],
            "IVF list %ld is out of bounds (%ld lists total) for head %ld",
            listId,
            nlists_[headId],
            headId);
    FAISS_ASSERT(listId < deviceListData_[headId].size());
    FAISS_ASSERT(listId < deviceListLengths_[headId].size());

    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto& list = deviceListData_[headId][listId];
    auto gpuCodes = list->data.copyToHost<uint8_t>(stream);

    if (gpuFormat) {
        return gpuCodes;
    } else {
        // The GPU layout may be different than the CPU layout (e.g., vectors
        // rather than dimensions interleaved), translate back if necessary
        return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
    }
}

void MultiHeadIVFBase::copyInvertedListsFrom(const std::vector<InvertedLists*>& ivfs) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        idx_t nlist = ivf->nlist;
        for (idx_t i = 0; i < nlist && i < nlists_[h]; ++i) {
            addEncodedVectorsToList_(
                    h, i, ivf->get_codes(i), ivf->get_ids(i), nullptr, ivf->list_size(i));
        }
    }
}

void MultiHeadIVFBase::storeTranslatedCodes(const std::vector<InvertedLists*>& ivfs, std::vector<std::vector<uint8_t*>>& translatedCodes_) {
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        idx_t nlist = ivf->nlist;
        for (idx_t i = 0; i < nlist; ++i) {
            auto codes = ivf->get_codes(i);
            auto numVecs = ivf->list_size(i);
            
            // The GPU might have a different layout of the memory
            auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
            auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

            // Translate the codes as needed to our preferred form
            std::vector<uint8_t> codesV(cpuListSizeInBytes);
            std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
            std::vector<uint8_t> translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

            cudaError_t err = cudaMallocHost(
                    (void**)&translatedCodes_[h][i],
                    gpuListSizeInBytes, 
                    cudaHostAllocPortable);
            if (err != cudaSuccess) {
                std::cout << "cudaMallocHost " << cudaGetErrorString(err) << std::endl;
            }
            
            // copy translated codes to pinned memory
            err = cudaMemcpy(
                    translatedCodes_[h][i],
                    translatedCodes.data(),
                    gpuListSizeInBytes,
                    cudaMemcpyHostToHost);
            if (err != cudaSuccess) {
                std::cout << "cudaMemcpy " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
    // isTranslatedCodesStored_ = true;
}

std::vector<size_t> MultiHeadIVFBase::getInvertedListsDataMemory(const std::vector<InvertedLists*>& ivfs) const {
    std::vector<size_t> reserveSizes(numHeads_, 0);
    
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        idx_t nlist = ivf->nlist;
        for (idx_t i = 0; i < nlist; ++i) {
            reserveSizes[h] += getGpuVectorsEncodingSize_(ivf->list_size(i));
        }
    }
    return reserveSizes;
}

std::vector<size_t> MultiHeadIVFBase::getInvertedListsIndexMemory(const std::vector<InvertedLists*>& ivfs) const {
    std::vector<size_t> reserveSizes(numHeads_, 0);
    
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        idx_t nlist = ivf->nlist;
        for (idx_t i = 0; i < nlist; ++i) {
            reserveSizes[h] += ivf->list_size(i) * sizeof(idx_t);
        }
    }
    return reserveSizes;
}

void MultiHeadIVFBase::reserveInvertedListsDataMemory(const std::vector<InvertedLists*>& ivfs) {
    auto reserveSizes = getInvertedListsDataMemory(ivfs);
    auto allocInfo = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, resources_->getDefaultStreamCurrentDevice());
    
    size_t totalSize = 0;
    for (size_t size : reserveSizes) {
        totalSize += size;
    }
    
    if (totalSize > 0) {
        ivfListDataReservation_ = resources_->allocMemoryHandle(AllocRequest(allocInfo, totalSize));
    }
}

void MultiHeadIVFBase::reserveInvertedListsIndexMemory(const std::vector<InvertedLists*>& ivfs) {
    auto reserveSizes = getInvertedListsIndexMemory(ivfs);
    auto allocInfo = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, resources_->getDefaultStreamCurrentDevice());
    
    size_t totalSize = 0;
    for (size_t size : reserveSizes) {
        totalSize += size;
    }
    
    if (totalSize > 0) {
        ivfListIndexReservation_ = resources_->allocMemoryHandle(AllocRequest(allocInfo, totalSize));
    }
}

// TODO: change copyInvertedLists to copy all ivf data at once (call cudaMemcpy once for all lists)
void MultiHeadIVFBase::copyInvertedListsFromNoRealloc(
        const std::vector<InvertedLists*>& ivfs, 
        std::vector<std::vector<uint8_t*>>& translatedCodes_,
        GpuMemoryReservation* ivfListDataReservation, 
        GpuMemoryReservation* ivfListIndexReservation) {
    
    // if (!isTranslatedCodesStored_) {
    //     storeTranslatedCodes(ivfs);
    // }

    size_t offsetData = 0;
    size_t offsetIndex = 0;
    
    #pragma omp parallel for
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        idx_t nlist = ivf->nlist;
        for (idx_t i = 0; i < nlist; ++i) {
            size_t curDataSize = getGpuVectorsEncodingSize_(ivf->list_size(i));
            size_t curIndexSize = ivf->list_size(i) * sizeof(idx_t);

            auto& listCodes = deviceListData_[h][i];
            listCodes->numVecs = 0;
            // std::cerr << "listCodes head " << h << " list " << i << " numVecs: " << listCodes->numVecs << std::endl;
            listCodes->data.assignReservedMemoryPointer(
                    (uint8_t*)ivfListDataReservation->get() + offsetData, curDataSize);
            offsetData += curDataSize;

            auto& listIndices = deviceListIndices_[h][i];
            listIndices->numVecs = 0;
            listIndices->data.assignReservedMemoryPointer(
                    (uint8_t*)ivfListIndexReservation->get() + offsetIndex, curIndexSize);
            offsetIndex += curIndexSize;

            addEncodedVectorsToList_(
                    h, i, ivf->get_codes(i), ivf->get_ids(i), translatedCodes_[h][i], ivf->list_size(i));
        }
    }
}

void MultiHeadIVFBase::copyInvertedListsFromNoRealloc(
        const std::vector<InvertedLists*>& ivfs, 
        std::vector<std::vector<idx_t>>& nlistIds, 
        std::vector<std::vector<uint8_t*>>& translatedCodes_,
        GpuMemoryReservation* ivfListDataReservation, 
        GpuMemoryReservation* ivfListIndexReservation) {
    
    size_t offsetData = 0;
    size_t offsetIndex = 0;

    #pragma omp parallel for
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        for (int idx = 0; idx < nlistIds[h].size(); ++idx) {
            idx_t i = nlistIds[h][idx];
            size_t curDataSize = getGpuVectorsEncodingSize_(ivf->list_size(i));
            size_t curIndexSize = ivf->list_size(i) * sizeof(idx_t);

            auto& listCodes = deviceListData_[h][i];
            listCodes->numVecs = 0;
            listCodes->data.assignReservedMemoryPointer(
                    (uint8_t*)ivfListDataReservation->get() + offsetData, curDataSize);
            offsetData += curDataSize;

            auto& listIndices = deviceListIndices_[h][i];
            listIndices->numVecs = 0;
            listIndices->data.assignReservedMemoryPointer(
                    (uint8_t*)ivfListIndexReservation->get() + offsetIndex, curIndexSize);
            offsetIndex += curIndexSize;

            addEncodedVectorsToList_(
                    h, i, ivf->get_codes(i), ivf->get_ids(i), translatedCodes_[h][i], ivf->list_size(i));
        }
    }
}

void MultiHeadIVFBase::copyInvertedListsTo(std::vector<InvertedLists*>& ivfs) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    
    for (int h = 0; h < numHeads_; ++h) {
        InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        
        for (idx_t i = 0; i < nlists_[h]; ++i) {
            auto listIndices = getListIndices(h, i);
            auto listData = getListVectorData(h, i, false);

            ivf->add_entries(
                    i, listIndices.size(), listIndices.data(), listData.data());
        }
    }
}

void MultiHeadIVFBase::reconstruct_n(idx_t headId, idx_t i0, idx_t n, float* out) {
    FAISS_THROW_MSG("not implemented");
}

void MultiHeadIVFBase::addEncodedVectorsToList_(
        idx_t headId,
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        uint8_t* translatedCodes,
        idx_t numVecs) {
    // auto stream = resources_->getDefaultStreamCurrentDevice();
    auto stream = resources_->getAsyncCopyStreamCurrentDevice();

    // This list must already exist
    FAISS_ASSERT(headId < numHeads_);
    FAISS_ASSERT(listId < deviceListData_[headId].size());

    // This list must currently be empty
    auto& listCodes = deviceListData_[headId][listId];
    // std::cerr << listCodes->data.size() << std::endl;
    FAISS_ASSERT(listCodes->data.size() == 0);
    // std::cerr << "listCodes head " << headId << " list " << listId << " numVecs: " << listCodes->numVecs << std::endl;
    FAISS_ASSERT(listCodes->numVecs == 0);

    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

    if (translatedCodes) {
        listCodes->data.append(
            translatedCodes,
            gpuListSizeInBytes,
            stream,
            true /* exact reserved size */);
    } else {
        // Translate the codes as needed to our preferred form
        std::vector<uint8_t> codesV(cpuListSizeInBytes);
        std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
        auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

        listCodes->data.append(
                translatedCodes.data(),
                gpuListSizeInBytes,
                stream,
                true /* exact reserved size */);
    }
    
    listCodes->numVecs = numVecs;

    // Handle the indices as well
    addIndicesFromCpu_(headId, listId, indices, numVecs);

    deviceListDataPointers_[headId].setAt(
            listId, (void*)listCodes->data.data(), stream);
    deviceListLengths_[headId].setAt(listId, numVecs, stream);

    // We update this as well, since the multi-pass algorithm uses it
    maxListLength_ = std::max(maxListLength_, numVecs);
}

void MultiHeadIVFBase::addIndicesFromCpu_(
        idx_t headId,
        idx_t listId,
        const idx_t* indices,
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // This list must currently be empty
    auto& listIndices = deviceListIndices_[headId][listId];
    FAISS_ASSERT(listIndices->data.size() == 0);
    FAISS_ASSERT(listIndices->numVecs == 0);

    if (indicesOptions_ == INDICES_32_BIT) {
        // Make sure that all indices are in bounds
        int* indices32;
        cudaError_t err = cudaMallocHost(
                (void**)&indices32,
                numVecs * sizeof(int), 
                cudaHostAllocPortable);
        for (idx_t i = 0; i < numVecs; ++i) {
            auto ind = indices[i];
            FAISS_ASSERT(ind <= (idx_t)std::numeric_limits<int>::max());
            indices32[i] = (int)ind;
        }

        static_assert(sizeof(int) == 4, "");

        listIndices->data.append(
                (uint8_t*)indices32,
                numVecs * sizeof(int),
                stream,
                true /* exact reserved size */);

        // We have added the given indices to the raw data vector; update the
        // count as well
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        listIndices->data.append(
                (uint8_t*)indices,
                numVecs * sizeof(idx_t),
                stream,
                true /* exact reserved size */);

        // We have added the given indices to the raw data vector; update the
        // count as well
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_[headId].size());

        auto& userIndices = listOffsetToUserIndex_[headId][listId];
        userIndices.insert(userIndices.begin(), indices, indices + numVecs);
    } else {
        // indices are not stored
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
    }

    deviceListIndexPointers_[headId].setAt(
            listId, (void*)listIndices->data.data(), stream);
}

void MultiHeadIVFBase::updateQuantizer(std::vector<Index*>& quantizers) {
    FAISS_THROW_IF_NOT(quantizers.size() == numHeads_);
    
    auto stream = resources_->getDefaultStreamCurrentDevice();

    for (int h = 0; h < numHeads_; ++h) {
        Index* quantizer = quantizers[h];
        FAISS_THROW_IF_NOT(quantizer->is_trained);

        // Must match our basic IVF parameters
        FAISS_THROW_IF_NOT(quantizer->d == getDim());
        FAISS_THROW_IF_NOT(quantizer->ntotal == nlists_[h]);

        // If the index instance is a GpuIndexFlat, then we can use direct access to
        // the centroids within.
        auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
        if (gpuQ) {
            auto gpuData = gpuQ->getGpuData();

            if (gpuData->getUseFloat16()) {
                // The FlatIndex keeps its data in float16; we need to reconstruct
                // as float32 and store locally
                DeviceTensor<float, 2, true> centroids(
                        resources_,
                        makeSpaceAlloc(AllocType::FlatData, space_, stream),
                        {nlists_[h], getDim()});

                gpuData->reconstruct(0, gpuData->getSize(), centroids);

                ivfCentroids_[h] = std::move(centroids);
            } else {
                // The FlatIndex keeps its data in float32, so we can merely
                // reference it
                auto ref32 = gpuData->getVectorsFloat32Ref();

                // Create a DeviceTensor that merely references, doesn't own the
                // data
                auto refOnly = DeviceTensor<float, 2, true>(
                        ref32.data(), {ref32.getSize(0), ref32.getSize(1)});

                ivfCentroids_[h] = std::move(refOnly);

                // For debugging, we can print the centroids to verify they are
                // std::cerr << "Update quantizer for head " << h << std::endl;
                // auto ivfCentroids_vector = ivfCentroids_[h].copyToVector(stream);
                // std::cerr << "ivfCentroids_: " << std::endl ;
                // for (size_t i = 0; i < nlists_[h]; ++i) {
                //     std::cerr << "List " << i << ": " << std::endl ;
                //     for (size_t j = 0; j < dim_; ++j) {
                //         std::cerr << ivfCentroids_vector[i * dim_ + j] << " ";
                //     }
                //     std::cerr << std::endl;
                // }
            }
        } else {
            // Otherwise, we need to reconstruct all vectors from the index and copy
            // them to the GPU, in order to have access as needed for residual
            // computation
            auto vecs = std::vector<float>(nlists_[h] * getDim());
            quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

            // Copy to a new DeviceTensor; this will own the data
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {quantizer->ntotal, quantizer->d});
            centroids.copyFrom(vecs, stream);

            ivfCentroids_[h] = std::move(centroids);
        }
    }
}

void MultiHeadIVFBase::searchCoarseQuantizer_(
        std::vector<Index*>& coarseQuantizers,
        const std::vector<int>& nprobe,
        // guaranteed resident on device
        Tensor<float, 2, true>* vecs,
        // Output tensors per head
        Tensor<float, 2, true>* distances,
        Tensor<idx_t, 2, true>* indices,
        Tensor<float, 3, true>* residuals,
        Tensor<float, 3, true>* centroids) {
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    
    FAISS_THROW_IF_NOT(coarseQuantizers.size() == numHeads_);
    FAISS_THROW_IF_NOT(nprobe.size() == numHeads_);

    bool vecsOnDevice = getDeviceForAddress(vecs->data()) == 0;
    bool distancesOnDevice = getDeviceForAddress(distances->data()) == 0;
    bool indicesOnDevice = getDeviceForAddress(indices->data()) == 0;
    FAISS_ASSERT(vecsOnDevice);

    // auto deviceVecs = (DeviceTensor<float, 2, true>*)vecs;

    // Process each head separately
    for (int h = 0; h < numHeads_; ++h) {
        // std::cerr << "Head " << h << " searchCoarseQuantizer_" << std::endl;
        // auto vecs_vector = vecs[h].copyToVector(stream);
        // for (size_t i = 0; i < vecs[h].getSize(0); ++i) {
        //     std::cerr << "vecs[" << h << "][" << i << "]: ";
        //     for (size_t j = 0; j < vecs[h].getSize(1); ++j) {
        //         std::cerr << vecs_vector[i * vecs[h].getSize(1) + j] << " ";
        //     }
        //     std::cerr << std::endl;
        // }

        Index* coarseQuantizer = coarseQuantizers[h];
        
        // The provided IVF quantizer may be CPU or GPU resident.
        auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
        if (gpuQuantizer) {
            // std::cerr << (vecs + h)->getSize(0) << " vectors, "
            //           << nprobe[h] << " probes" << std::endl;
            auto deviceDistances = (DeviceTensor<float, 2, true>*)distances;
            auto deviceIndices = (DeviceTensor<idx_t, 2, true>*)indices;
            // We can pass device pointers directly
            gpuQuantizer->search(
                    (vecs + h)->getSize(0),
                    (vecs + h)->data(),
                    nprobe[h],
                    (deviceDistances + h)->data(),
                    (deviceIndices + h)->data());

            if (residuals) {
                auto deviceResiduals = (DeviceTensor<float, 3, true>*)residuals;
                gpuQuantizer->compute_residual_n(
                    vecs[h].getSize(0) * nprobe[h],
                    vecs[h].data(),
                        deviceResiduals[h].data(),
                        deviceIndices[h].data());
            }

            if (centroids) {
                auto deviceCentroids = (DeviceTensor<float, 3, true>*)centroids;
                gpuQuantizer->reconstruct_batch(
                    vecs[h].getSize(0) * nprobe[h],
                        deviceIndices[h].data(),
                        deviceCentroids[h].data());
            }
        } else {
            // temporary host storage for querying a CPU index
            auto cpuVecs = toHost<float, 2>(
                    vecs[h].data(), stream, {vecs[h].getSize(0), vecs[h].getSize(1)});
            auto cpuDistances = std::vector<float>(vecs[h].getSize(0) * nprobe[h]);
            auto cpuIndices = std::vector<idx_t>(vecs[h].getSize(0) * nprobe[h]);

            coarseQuantizer->search(
                    vecs[h].getSize(0),
                    cpuVecs.data(),
                    nprobe[h],
                    cpuDistances.data(),
                    cpuIndices.data());
            
            auto deviceDistances = (DeviceTensor<float, 2, true>*)distances;
            auto deviceIndices = (DeviceTensor<idx_t, 2, true>*)indices;
            deviceDistances[h].copyFrom(cpuDistances, stream);

            // Did we also want to return IVF cell residuals for the query vectors?
            if (residuals) {
                auto cpuResiduals =
                        std::vector<float>(vecs[h].getSize(0) * nprobe[h] * dim_);

                coarseQuantizer->compute_residual_n(
                        vecs[h].getSize(0) * nprobe[h],
                        cpuVecs.data(),
                        cpuResiduals.data(),
                        cpuIndices.data());

                auto deviceResiduals = (DeviceTensor<float, 3, true>*)residuals;
                deviceResiduals[h].copyFrom(cpuResiduals, stream);
            }

            // Did we also want to return the IVF cell centroids themselves?
            if (centroids) {
                auto cpuCentroids =
                        std::vector<float>(vecs[h].getSize(0) * nprobe[h] * dim_);

                coarseQuantizer->reconstruct_batch(
                        vecs[h].getSize(0) * nprobe[h],
                        cpuIndices.data(),
                        cpuCentroids.data());

                auto deviceCentroids = (DeviceTensor<float, 3, true>*)centroids;
                deviceCentroids[h].copyFrom(cpuCentroids, stream);
            }

            deviceIndices[h].copyFrom(cpuIndices, stream);
        }
    }
}

idx_t MultiHeadIVFBase::addVectors(
        std::vector<Index*>& coarseQuantizers,
        Tensor<float, 2, true>* vecs,
        Tensor<idx_t, 1, true>* indices) {
    
    FAISS_THROW_IF_NOT(coarseQuantizers.size() == numHeads_);
    
    auto stream = resources_->getDefaultStreamCurrentDevice();
    idx_t totalAdded = 0;

    // Process each head separately
    for (int h = 0; h < numHeads_; ++h) {
        FAISS_ASSERT(vecs[h].getSize(0) == indices[h].getSize(0));
        FAISS_ASSERT(vecs[h].getSize(1) == dim_);

        // Determine which IVF lists we need to append to
        DeviceTensor<float, 2, true> unusedIVFDistances(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecs[h].getSize(0), 1});

        // We do need the closest IVF cell IDs though
        DeviceTensor<idx_t, 2, true> ivfIndices(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecs[h].getSize(0), 1});

        // Calculate residuals for these vectors, if needed
        DeviceTensor<float, 3, true> residuals(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecs[h].getSize(0), 1, dim_});

        // Create single-head arrays for searchCoarseQuantizer_
        std::vector<Index*> singleQuantizer = {coarseQuantizers[h]};
        std::vector<int> singleNprobe = {1};
        Tensor<float, 2, true> singleVecs[1] = {vecs[h]};
        Tensor<float, 2, true> singleDistances[1] = {unusedIVFDistances};
        Tensor<idx_t, 2, true> singleIndices[1] = {ivfIndices};
        Tensor<float, 3, true> singleResiduals[1] = {residuals};

        searchCoarseQuantizer_(
                singleQuantizer,
                singleNprobe,
                singleVecs,
                singleDistances,
                singleIndices,
                useResidual_ ? singleResiduals : nullptr,
                nullptr);

        // Copy the lists that we wish to append to back to the CPU
        auto ivfIndicesHost = ivfIndices.copyToVector(stream);

        // Now we add the encoded vectors to the individual lists
        // First, make sure that there is space available for adding the new
        // encoded vectors and indices

        // list id -> vectors being added
        std::unordered_map<idx_t, std::vector<idx_t>> listToVectorIds;

        // vector id -> which list it is being appended to
        std::vector<idx_t> vectorIdToList(vecs[h].getSize(0));

        // vector id -> offset in list
        std::vector<idx_t> listOffsetHost(ivfIndicesHost.size());

        // Number of valid vectors that we actually add; we return this
        idx_t numAdded = 0;

        for (idx_t i = 0; i < ivfIndicesHost.size(); ++i) {
            auto listId = ivfIndicesHost[i];

            // Add vector could be invalid (contains NaNs etc)
            if (listId < 0) {
                listOffsetHost[i] = -1;
                vectorIdToList[i] = -1;
                continue;
            }

            FAISS_ASSERT(listId < nlists_[h]);
            ++numAdded;
            vectorIdToList[i] = listId;

            auto offset = deviceListData_[h][listId]->numVecs;

            auto it = listToVectorIds.find(listId);
            if (it != listToVectorIds.end()) {
                offset += it->second.size();
                it->second.push_back(i);
            } else {
                listToVectorIds[listId] = std::vector<idx_t>{i};
            }

            listOffsetHost[i] = offset;
        }

        // If we didn't add anything (all invalid vectors that didn't map to IVF
        // clusters), no need to continue
        if (numAdded == 0) {
            continue;
        }

        totalAdded += numAdded;

        // unique lists being added to
        std::vector<idx_t> uniqueLists;

        for (auto& vecsInList : listToVectorIds) {
            uniqueLists.push_back(vecsInList.first);
        }

        std::sort(uniqueLists.begin(), uniqueLists.end());

        // In the same order as uniqueLists, list the vectors being added to that
        // list contiguously
        std::vector<idx_t> vectorsByUniqueList;

        // For each of the unique lists, the start offset in vectorsByUniqueList
        std::vector<idx_t> uniqueListVectorStart;

        // For each of the unique lists, where we start appending in that list by
        // the vector offset
        std::vector<idx_t> uniqueListStartOffset;

        // For each of the unique lists, find the vectors which should be appended
        // to that list
        for (auto ul : uniqueLists) {
            uniqueListVectorStart.push_back(vectorsByUniqueList.size());

            FAISS_ASSERT(listToVectorIds.count(ul) != 0);

            // The vectors we are adding to this list
            auto& vecsInList = listToVectorIds[ul];
            vectorsByUniqueList.insert(
                    vectorsByUniqueList.end(), vecsInList.begin(), vecsInList.end());

            // How many vectors we previously had (which is where we start appending
            // on the device)
            uniqueListStartOffset.push_back(deviceListData_[h][ul]->numVecs);
        }

        // We terminate uniqueListVectorStart with the overall number of vectors
        // being added
        uniqueListVectorStart.push_back(vectorsByUniqueList.size());

        // Resize all of the lists that we are appending to
        for (auto& counts : listToVectorIds) {
            auto listId = counts.first;
            idx_t numVecsToAdd = counts.second.size();

            auto& codes = deviceListData_[h][listId];
            auto oldNumVecs = codes->numVecs;
            auto newNumVecs = codes->numVecs + numVecsToAdd;

            auto newSizeBytes = getGpuVectorsEncodingSize_(newNumVecs);
            codes->data.resize(newSizeBytes, stream);
            codes->numVecs = newNumVecs;

            auto& indicesData = deviceListIndices_[h][listId];
            if ((indicesOptions_ == INDICES_32_BIT) ||
                (indicesOptions_ == INDICES_64_BIT)) {
                size_t indexSize = (indicesOptions_ == INDICES_32_BIT)
                        ? sizeof(int)
                        : sizeof(idx_t);

                indicesData->data.resize(
                        indicesData->data.size() + numVecsToAdd * indexSize,
                        stream);
                FAISS_ASSERT(indicesData->numVecs == oldNumVecs);
                indicesData->numVecs = newNumVecs;

            } else if (indicesOptions_ == INDICES_CPU) {
                // indices are stored on the CPU side
                FAISS_ASSERT(listId < listOffsetToUserIndex_[h].size());

                auto& userIndices = listOffsetToUserIndex_[h][listId];
                userIndices.resize(newNumVecs);
            } else {
                // indices are not stored on the GPU or CPU side
                FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
            }

            // This is used by the multi-pass query to decide how much scratch
            // space to allocate for intermediate results
            maxListLength_ = std::max(maxListLength_, newNumVecs);
        }

        // Update all pointers and sizes on the device for lists that we
        // appended to
        updateDeviceListInfo_(h, uniqueLists, stream);

        // If we're maintaining the indices on the CPU side, update our
        // map. We already resized our map above.
        if (indicesOptions_ == INDICES_CPU) {
            // We need to maintain the indices on the CPU side
            HostTensor<idx_t, 1, true> hostIndices(indices[h], stream);

            for (idx_t i = 0; i < hostIndices.getSize(0); ++i) {
                idx_t listId = ivfIndicesHost[i];

                // Add vector could be invalid (contains NaNs etc)
                if (listId < 0) {
                    continue;
                }

                auto offset = listOffsetHost[i];
                FAISS_ASSERT(offset >= 0);

                FAISS_ASSERT(listId < listOffsetToUserIndex_[h].size());
                auto& userIndices = listOffsetToUserIndex_[h][listId];

                FAISS_ASSERT(offset < userIndices.size());
                userIndices[offset] = hostIndices[i];
            }
        }

        // Copy the offsets to the GPU
        auto ivfIndices1dDevice = ivfIndices.downcastOuter<1>();
        auto residuals2dDevice = residuals.downcastOuter<2>();
        auto listOffsetDevice =
                toDeviceTemporary(resources_, listOffsetHost, stream);
        auto uniqueListsDevice = toDeviceTemporary(resources_, uniqueLists, stream);
        auto vectorsByUniqueListDevice =
                toDeviceTemporary(resources_, vectorsByUniqueList, stream);
        auto uniqueListVectorStartDevice =
                toDeviceTemporary(resources_, uniqueListVectorStart, stream);
        auto uniqueListStartOffsetDevice =
                toDeviceTemporary(resources_, uniqueListStartOffset, stream);

        // Actually encode and append the vectors
        appendVectors_(
                &vecs[h],
                &residuals2dDevice,
                &indices[h],
                &uniqueListsDevice,
                &vectorsByUniqueListDevice,
                &uniqueListVectorStartDevice,
                &uniqueListStartOffsetDevice,
                &ivfIndices1dDevice,
                &listOffsetDevice,
                stream);
    }

    return totalAdded;
}

} // namespace gpu
} // namespace faiss
