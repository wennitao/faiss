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
#include <faiss/invlists/InvertedLists.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/impl/MultiHeadIVFBase.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
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
        bool useResidual,
        bool interleavedLayout,
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
        ivfCentroids_(nullptr), // Initialize to nullptr
        // deviceListDataPointers_(
        //         resources,
        //         AllocInfo(
        //                 AllocType::IVFLists,
        //                 getCurrentDevice(),
        //                 space,
        //                 resources->getDefaultStreamCurrentDevice())),
        // deviceListIndexPointers_(
        //         resources,
        //         AllocInfo(
        //                 AllocType::IVFLists,
        //                 getCurrentDevice(),
        //                 space,
        //                 resources->getDefaultStreamCurrentDevice())),
        // deviceListLengths_(
        //         resources,
        //         AllocInfo(
        //                 AllocType::IVFLists,
        //                 getCurrentDevice(),
        //                 space,
        //                 resources->getDefaultStreamCurrentDevice())),
        maxListLength_(0) {
    FAISS_THROW_IF_NOT(numHeads_ > 0);
    if (numHeads_ > 0) {
        // deviceListDataPointers_.reserve (numHeads_);
        for (int h = 0; h < numHeads_; ++h) {
            deviceListDataPointers_.emplace_back(
                resources_, 
                AllocInfo(AllocType::IVFLists, getCurrentDevice(), 
                space_, 
                resources->getDefaultStreamCurrentDevice()));
        }

        // deviceListIndexPointers_.reserve (numHeads_);
        for (int h = 0; h < numHeads_; ++h) {
            deviceListIndexPointers_.emplace_back(
                resources_, 
                AllocInfo(AllocType::IVFLists, getCurrentDevice(), 
                space_, 
                resources->getDefaultStreamCurrentDevice()));
        }

        // deviceListLengths_.reserve (numHeads_);
        for (int h = 0; h < numHeads_; ++h) {
            deviceListLengths_.emplace_back(
                resources_, 
                AllocInfo(AllocType::IVFLists, getCurrentDevice(), 
                space_, 
                resources->getDefaultStreamCurrentDevice()));
        }
    }
    reset();
}

MultiHeadIVFBase::~MultiHeadIVFBase() {
    // Delete the array of DeviceTensor objects
    if (ivfCentroids_) {
        delete[] ivfCentroids_;
        ivfCentroids_ = nullptr;
    }
    // DeviceVector members will clean up their own GPU memory.
    // std::vector<std::vector<std::unique_ptr<DeviceIVFList>>> will also clean up.
}

void MultiHeadIVFBase::reserveMemory(idx_t totalNumVecs) { // totalNumVecs across all heads
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Approximate vecs per list across all heads and lists
    idx_t totalLists = 0 ;
    for (int h = 0; h < numHeads_; ++h) {
        totalLists += nlists_[h];
    }
    if (totalLists == 0) return; // Avoid division by zero

    auto vecsPerList = totalNumVecs / totalLists;
    if (vecsPerList < 1) {
       return ;
    }

    auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

    for (int h = 0; h < numHeads_; ++h) {
        for (idx_t l = 0; l < nlists_[h]; ++l) {
            if (h < deviceListData_.size() && l < deviceListData_[h].size() && deviceListData_[h][l]) {
                 deviceListData_[h][l]->data.reserve(bytesPerDataList, stream);
            }
        }
    }

    if ((indicesOptions_ == INDICES_32_BIT) ||
        (indicesOptions_ == INDICES_64_BIT)) {
        size_t bytesPerIndexList = vecsPerList *
                (indicesOptions_ == INDICES_32_BIT ? sizeof(int)
                                                : sizeof(idx_t));

        for (int h = 0; h < numHeads_; ++h) {
            for (idx_t l = 0; l < nlists_[h]; ++l) {
                 if (h < deviceListIndices_.size() && l < deviceListIndices_[h].size() && deviceListIndices_[h][l]) {
                    deviceListIndices_[h][l]->data.reserve(bytesPerIndexList, stream);
                }
            }
        }
    }

    // Update device info for all lists, since the base pointers may
    // have changed. This function needs to be aware of the multi-head structure
    // to correctly update the flat global device vectors.
    for (int h = 0; h < numHeads_; ++h) {
        updateDeviceListInfo_(h, stream);
    }
}

void MultiHeadIVFBase::reset() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (isTranslatedCodesStored_) {
        for (int h = 0; h < numHeads_; ++h) {
            if (h < translatedCodes_.size()) {
                for (idx_t l = 0; l < nlists_[h]; ++l) {
                    if (l < translatedCodes_[h].size() && translatedCodes_[h][l]) {
                        cudaFreeHost(translatedCodes_[h][l]);
                        translatedCodes_[h][l] = nullptr; // Good practice
                    }
                }
            }
        }
    }

    deviceListData_.clear();
    deviceListIndices_.clear();
    translatedCodes_.clear();
    listOffsetToUserIndex_.clear();

    // Clear global device vectors
    for (int h = 0; h < numHeads_; ++ h) {
        deviceListDataPointers_[h].clear();
        deviceListIndexPointers_[h].clear();
        deviceListLengths_[h].clear();
    }


    auto info =
            AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, stream);

    deviceListData_.resize(numHeads_);
    deviceListIndices_.resize(numHeads_);
    translatedCodes_.resize(numHeads_);
    listOffsetToUserIndex_.resize(numHeads_);

    for (int h = 0; h < numHeads_; ++h) {
        deviceListData_[h].resize(nlists_[h]);
        deviceListIndices_[h].resize(nlists_[h]);
        translatedCodes_[h].resize(nlists_[h], nullptr); // Initialize with nullptr
        listOffsetToUserIndex_[h].resize(nlists_[h]);

        for (idx_t l = 0; l < nlists_[h]; ++l) {
            deviceListData_[h][l] = std::make_unique<DeviceIVFList>(resources_, info);
            deviceListIndices_[h][l] = std::make_unique<DeviceIVFList>(resources_, info);
            // listOffsetToUserIndex_[h][l] is an empty std::vector<idx_t> by default
            // translatedCodes_[h][l] is already nullptr
        }
    }

    if (nlists_.size() > 0) {
        for (int h = 0; h < numHeads_; ++ h) {
            deviceListDataPointers_[h].resize(nlists_[h], stream);
            deviceListDataPointers_[h].setAll(nullptr, stream);

            deviceListIndexPointers_[h].resize(nlists_[h], stream);
            deviceListIndexPointers_[h].setAll(nullptr, stream);

            deviceListLengths_[h].resize(nlists_[h], stream);
            deviceListLengths_[h].setAll(0, stream);
        }
    }

    maxListLength_ = 0;
    isTranslatedCodesStored_ = false;
}

idx_t MultiHeadIVFBase::getDim() const {
    return dim_;
}

idx_t MultiHeadIVFBase::getNumHeads() const {
    return numHeads_;
}

// Returns the number of lists *per head*
idx_t MultiHeadIVFBase::getNumLists(idx_t headId) const {
    FAISS_THROW_IF_NOT_FMT(headId < numHeads_, "Head ID %ld out of bounds (%d heads total)", headId, numHeads_);
    return nlists_[headId]; // numLists_ stores lists per head
}


size_t MultiHeadIVFBase::reclaimMemory() {
    // Reclaim all unused memory exactly
    return reclaimMemory_(true);
}

size_t MultiHeadIVFBase::reclaimMemory_(bool exact) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    size_t totalReclaimed = 0;

    for (int h = 0; h < numHeads_; ++h) {
        for (idx_t l = 0; l < nlists_[h]; ++l) {
            // idx_t globalListIdx = h * numLists_ + l;

            if (h < deviceListData_.size() && l < deviceListData_[h].size() && deviceListData_[h][l]) {
                auto& dataList = deviceListData_[h][l];
                totalReclaimed += dataList->data.reclaim(exact, stream);
                
                deviceListDataPointers_[h].setAt(l, (void*)dataList->data.data(), stream);
            }

            if (h < deviceListIndices_.size() && l < deviceListIndices_[h].size() && deviceListIndices_[h][l]) {
                auto& indexList = deviceListIndices_[h][l];
                totalReclaimed += indexList->data.reclaim(exact, stream);
                
                deviceListIndexPointers_[h].setAt(l, (void*)indexList->data.data(), stream);
            }
        }

        updateDeviceListInfo_(h, stream);
    }

    return totalReclaimed;
}

void MultiHeadIVFBase::updateDeviceListInfo_(int headId, cudaStream_t stream) {
    std::vector<idx_t> listIds(deviceListData_[headId].size());
    for (idx_t l = 0; l < deviceListData_[headId].size(); ++l) {
        listIds[l] = l;
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

    // Update all pointers to the lists on the device that may have
    // changed
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

// Example for getListLength:
idx_t MultiHeadIVFBase::getListLength(idx_t headId, idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(headId < numHeads_, "Head ID %ld out of bounds (%d heads total)", headId, numHeads_);
    FAISS_THROW_IF_NOT_FMT(listId < nlists_[headId], "List ID %ld out of bounds (%ld lists per head)", listId, nlists_[headId]);
    
    FAISS_ASSERT(headId < deviceListData_.size());
    FAISS_ASSERT(listId < deviceListData_[headId].size());
    FAISS_ASSERT(deviceListData_[headId][listId]);

    return deviceListData_[headId][listId]->numVecs;
}

std::vector<idx_t> MultiHeadIVFBase::getListIndices(idx_t headId, idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(headId < numHeads_, "Head ID %ld out of bounds (%d heads total)", headId, numHeads_);
    FAISS_THROW_IF_NOT_FMT(listId < nlists_[headId], "List ID %ld out of bounds (%ld lists per head)", listId, nlists_[headId]);
    FAISS_ASSERT(headId < deviceListIndices_.size() && listId < deviceListIndices_[headId].size() && deviceListIndices_[headId][listId]);
    FAISS_ASSERT(headId < deviceListData_.size() && listId < deviceListData_[headId].size() && deviceListData_[headId][listId]);


    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (indicesOptions_ == INDICES_32_BIT) {
        auto intInd = deviceListIndices_[headId][listId]->data.copyToHost<int>(stream);
        std::vector<idx_t> out(intInd.size());
        for (size_t i = 0; i < intInd.size(); ++i) {
            out[i] = (idx_t)intInd[i];
        }
        return out;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        return deviceListIndices_[headId][listId]->data.copyToHost<idx_t>(stream);
    } else if (indicesOptions_ == INDICES_CPU) {
        FAISS_ASSERT(headId < listOffsetToUserIndex_.size() && listId < listOffsetToUserIndex_[headId].size());
        auto& userIds = listOffsetToUserIndex_[headId][listId];
        FAISS_ASSERT(userIds.size() == deviceListData_[headId][listId]->numVecs);
        return userIds; // returns a copy
    } else {
        FAISS_ASSERT(false); // unhandled indices type (includes INDICES_IVF)
        return std::vector<idx_t>();
    }
}

std::vector<uint8_t> MultiHeadIVFBase::getListVectorData(idx_t headId, idx_t listId, bool gpuFormat) const {
    FAISS_THROW_IF_NOT_FMT(headId < numHeads_, "Head ID %ld out of bounds (%d heads total)", headId, numHeads_);
    FAISS_THROW_IF_NOT_FMT(listId < nlists_[headId], "List ID %ld out of bounds (%ld lists per head)", listId, nlists_[headId]);
    FAISS_ASSERT(headId < deviceListData_.size() && listId < deviceListData_[headId].size() && deviceListData_[headId][listId]);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto& list = deviceListData_[headId][listId];
    auto gpuCodes = list->data.copyToHost<uint8_t>(stream);

    if (gpuFormat) {
        return gpuCodes;
    } else {
        return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
    }
}

void MultiHeadIVFBase::copyInvertedListsFrom(const std::vector<InvertedLists*>& ivfs) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        for (idx_t l = 0; l < ivf->nlist; ++l) {
            addEncodedVectorsToList_(
                    h, l, ivf->get_codes(l), ivf->get_ids(l), ivf->list_size(l));
        }
    }
}

void MultiHeadIVFBase::storeTranslatedCodes(const std::vector<InvertedLists*>& ivfs) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    auto stream = resources_->getDefaultStreamCurrentDevice();

    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        FAISS_ASSERT(h < translatedCodes_.size());

        for (idx_t l = 0; l < ivf->nlist; ++l) {
            FAISS_ASSERT(l < translatedCodes_[h].size());
            if (translatedCodes_[h][l]) { // Free existing if any
                cudaFreeHost(translatedCodes_[h][l]);
                translatedCodes_[h][l] = nullptr;
            }

            auto codes = ivf->get_codes(l);
            auto numVecs = ivf->list_size(l);
            if (numVecs == 0) continue;

            auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
            auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

            std::vector<uint8_t> codesV(cpuListSizeInBytes);
            std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
            std::vector<uint8_t> translatedGpuCodes = translateCodesToGpu_(std::move(codesV), numVecs);

            FAISS_ASSERT(translatedGpuCodes.size() == gpuListSizeInBytes);

            cudaError_t err = cudaMallocHost(
                    (void**)&translatedCodes_[h][l],
                    gpuListSizeInBytes,
                    cudaHostAllocPortable);
            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "cudaMallocHost failed: %s", cudaGetErrorString(err));

            err = cudaMemcpy(
                    translatedCodes_[h][l],
                    translatedGpuCodes.data(),
                    gpuListSizeInBytes,
                    cudaMemcpyHostToHost); // Host to Host for pinned memory setup
            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "cudaMemcpy (HostToHost for pinned) failed: %s", cudaGetErrorString(err));
        }
    }
    isTranslatedCodesStored_ = true;
}


std::vector<size_t> MultiHeadIVFBase::getInvertedListsDataMemory(const std::vector<InvertedLists*>& ivfs) const {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    std::vector<size_t> headMemory(numHeads_, 0);
    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        size_t reserveSize = 0;
        for (idx_t l = 0; l < ivf->nlist; ++l) {
            reserveSize += getGpuVectorsEncodingSize_(ivf->list_size(l));
        }
        headMemory[h] = reserveSize;
    }
    return headMemory;
}

std::vector<size_t> MultiHeadIVFBase::getInvertedListsIndexMemory(const std::vector<InvertedLists*>& ivfs) const {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    std::vector<size_t> headMemory(numHeads_, 0);
    size_t indexEntrySize = (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(idx_t));

    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        size_t reserveSize = 0;
        if (indicesOptions_ == INDICES_32_BIT || indicesOptions_ == INDICES_64_BIT) {
            for (idx_t l = 0; l < ivf->nlist; ++l) {
                reserveSize += ivf->list_size(l) * indexEntrySize;
            }
        }
        headMemory[h] = reserveSize;
    }
    return headMemory;
}

void MultiHeadIVFBase::reserveInvertedListsDataMemory(const std::vector<InvertedLists*>& ivfs) {
    // This function might need to sum up memory for a single global reservation
    // or handle per-head reservations if ivfListDataReservation_ becomes a vector.
    // Assuming a single global reservation for now.
    std::vector<size_t> headMem = getInvertedListsDataMemory(ivfs);
    size_t totalReserveSize = 0;
    for(size_t mem : headMem) {
        totalReserveSize += mem;
    }

    if (totalReserveSize == 0) return;
    auto allocInfo = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, resources_->getDefaultStreamCurrentDevice());
    ivfListDataReservation_ = resources_->allocMemoryHandle(AllocRequest(allocInfo, totalReserveSize));
}

void MultiHeadIVFBase::reserveInvertedListsIndexMemory(const std::vector<InvertedLists*>& ivfs) {
    std::vector<size_t> headMem = getInvertedListsIndexMemory(ivfs);
    size_t totalReserveSize = 0;
    for(size_t mem : headMem) {
        totalReserveSize += mem;
    }

    if (totalReserveSize == 0) return;
    auto allocInfo = AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, resources_->getDefaultStreamCurrentDevice());
    ivfListIndexReservation_ = resources_->allocMemoryHandle(AllocRequest(allocInfo, totalReserveSize));
}

void MultiHeadIVFBase::copyInvertedListsFromNoRealloc(const std::vector<InvertedLists*>& ivfs, GpuMemoryReservation* extDataReservation, GpuMemoryReservation* extIndexReservation) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    GpuMemoryReservation* dataReservation = extDataReservation ? extDataReservation : &ivfListDataReservation_;
    GpuMemoryReservation* indexReservation = extIndexReservation ? extIndexReservation : &ivfListIndexReservation_;

    if (!isTranslatedCodesStored_) {
        storeTranslatedCodes(ivfs); // This uses the member translatedCodes_
    }

    size_t currentDataOffset = 0;
    size_t currentIndexOffset = 0;

    for (int h = 0; h < numHeads_; ++h) {
        const InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        FAISS_ASSERT(h < deviceListData_.size() && h < deviceListIndices_.size());

        for (idx_t l = 0; l < ivf->nlist; ++l) {
            FAISS_ASSERT(l < deviceListData_[h].size() && l < deviceListIndices_[h].size());
            idx_t numVecsInList = ivf->list_size(l);

            size_t curDataSizeBytes = getGpuVectorsEncodingSize_(numVecsInList);
            if (numVecsInList > 0 && curDataSizeBytes > 0) {
                 FAISS_ASSERT(dataReservation && dataReservation->get());
                 FAISS_ASSERT(currentDataOffset + curDataSizeBytes <= dataReservation->size);
                 deviceListData_[h][l]->data.assignReservedMemoryPointer(
                    (uint8_t*)dataReservation->get() + currentDataOffset, curDataSizeBytes);
                 currentDataOffset += curDataSizeBytes;
            }


            if (indicesOptions_ == INDICES_32_BIT || indicesOptions_ == INDICES_64_BIT) {
                size_t indexEntrySize = (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(idx_t));
                size_t curIndexSizeBytes = numVecsInList * indexEntrySize;

                if (numVecsInList > 0 && curIndexSizeBytes > 0) {
                    FAISS_ASSERT(indexReservation && indexReservation->get());
                    FAISS_ASSERT(currentIndexOffset + curIndexSizeBytes <= indexReservation->size);
                    deviceListIndices_[h][l]->data.assignReservedMemoryPointer(
                        (uint8_t*)indexReservation->get() + currentIndexOffset, curIndexSizeBytes);
                    currentIndexOffset += curIndexSizeBytes;
                }
            }
            // Call the original addEncodedVectorsToList_ which now uses the assigned pointers
            // and translatedCodes_ if isTranslatedCodesStored_ is true.
            addEncodedVectorsToList_(h, l, ivf->get_codes(l), ivf->get_ids(l), numVecsInList);
        }
    }
}

void MultiHeadIVFBase::copyInvertedListsTo(std::vector<InvertedLists*>& ivfs) {
    FAISS_THROW_IF_NOT(ivfs.size() == numHeads_);
    for (int h = 0; h < numHeads_; ++h) {
        InvertedLists* ivf = ivfs[h];
        if (!ivf) continue;
        // FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist for output. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        // It's safer to resize/clear the target InvertedLists or assume it's empty and properly sized.
        // For now, we assume it's ready to receive `nlists_[h]` lists.
        // If ivf->nlist is 0, it might mean it's a new InvertedLists object.
        if (ivf->nlist == 0 && nlists_[h] > 0) {
            // Potentially resize or initialize the target InvertedLists if it's empty.
            // This depends on InvertedLists API, e.g., ivf->resize(nlists_[h], code_size_of_ivf)
            // For now, we'll proceed assuming it can handle add_entries up to nlists_[h]
        } else {
            FAISS_THROW_IF_NOT_FMT(ivf->nlist == nlists_[h], "Head %d: Mismatch in nlist for output. Expected %ld, got %ld", h, nlists_[h], ivf->nlist);
        }


        for (idx_t l = 0; l < nlists_[h]; ++l) {
            auto listIndices = getListIndices(h, l); // This is already adapted
            auto listData = getListVectorData(h, l, false); // This is already adapted

            // Ensure the target InvertedLists has this list, or can add it.
            // The InvertedLists::add_entries might overwrite or append.
            // If overwriting, ensure list 'l' exists.
            // If list_size(l) was 0, it should be fine.
            ivf->add_entries(
                l, (size_t)listIndices.size(), listIndices.data(), listData.data());
        }
    }
}


void MultiHeadIVFBase::updateQuantizer(std::vector<Index*>& quantizers) {
    FAISS_THROW_IF_NOT(quantizers.size() == numHeads_);
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Assuming ivfCentroids_ is sized [numHeads_, numLists_, dim_]
    // If ivfCentroids_ is not yet allocated or sized, do it here.
    // if (ivfCentroids_.data() == nullptr ||
    //     ivfCentroids_.getSize(0) != numHeads_ ||
    //     ivfCentroids_.getSize(1) != numLists_ ||
    //     ivfCentroids_.getSize(2) != dim_) {
    //     ivfCentroids_ = DeviceTensor<float, 3, true>(
    //             resources_,
    //             makeSpaceAlloc(AllocType::FlatData, space_, stream),
    //             { (size_t)numHeads_, (size_t)numLists_, (size_t)dim_});
    // }


    for (int h = 0; h < numHeads_; ++h) {
        Index* quantizer = quantizers[h];
        FAISS_THROW_IF_NOT(quantizer && quantizer->is_trained);
        FAISS_THROW_IF_NOT(quantizer->d == getDim());
        FAISS_THROW_IF_NOT(quantizer->ntotal == nlists_[h]); // numLists_ is per head

        // Get a slice for the current head's centroids
        // This creates a view, not a copy.
        auto headCentroidsSlice = ivfCentroids_ + h;
        // headCentroidsSlice is now {numLists_, dim_}

        auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
        if (gpuQ) {
            auto gpuData = gpuQ->getGpuData();
            if (gpuData->getUseFloat16()) {
                // gpuData is float16, headCentroidsSlice is float32
                // Need to reconstruct into the slice
                DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(h), getDim()});
                gpuData->reconstruct(0, gpuData->getSize(), centroids);

                *headCentroidsSlice = std::move (centroids);
            } else {
                // gpuData is float32, headCentroidsSlice is float32
                // We need to copy data from gpuQ's storage to our headCentroidsSlice
                auto ref32 = gpuData->getVectorsFloat32Ref();
                auto refOnly = DeviceTensor<float, 2, true>(
                    ref32.data(), {ref32.getSize(0), ref32.getSize(1)});

                *headCentroidsSlice = std::move (refOnly);
            }
        } else {
            // CPU quantizer
            std::vector<float> vecs(nlists_[h] * dim_);
            quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

            DeviceTensor<float, 2, true> centroids(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {quantizer->ntotal, quantizer->d});
            centroids.copyFrom(vecs, stream);

            *headCentroidsSlice = std::move (centroids);
        }
    }
}


void MultiHeadIVFBase::reconstruct_n(idx_t headId, idx_t i0, idx_t n, float* out) {
    // This function is complex and depends on how data is stored (e.g. PQ codes vs Flat).
    // IVFBase.cu also throws for this.
    // A full implementation would require knowing the derived type (Flat, PQ)
    // and accessing the correct list data for the given headId, then decoding.
    FAISS_THROW_MSG("MultiHeadIVFBase::reconstruct_n not implemented");
}

// Placeholder for addEncodedVectorsToList_ (needs careful implementation)
void MultiHeadIVFBase::addEncodedVectorsToList_(
        idx_t headId,
        idx_t listId,
        const void* codes, // from host
        const idx_t* indices, // from host
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    FAISS_ASSERT(headId < deviceListData_.size() && listId < deviceListData_[headId].size());
    auto& listCodes = deviceListData_[headId][listId];
    FAISS_ASSERT(listCodes->data.size() == 0); // Assuming adding to an empty (pre-reserved) list segment
    FAISS_ASSERT(listCodes->numVecs == 0);

    if (numVecs == 0) {
        return;
    }

    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

    if (isTranslatedCodesStored_) {
        FAISS_ASSERT(headId < translatedCodes_.size() && listId < translatedCodes_[headId].size() && translatedCodes_[headId][listId]);
        listCodes->data.append(
            translatedCodes_[headId][listId],
            gpuListSizeInBytes,
            stream,
            true /* exact reserved size */);
    } else {
        auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);
        std::vector<uint8_t> codesV(cpuListSizeInBytes);
        std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
        auto translatedGpuCodes = translateCodesToGpu_(std::move(codesV), numVecs);

        listCodes->data.append(
                translatedGpuCodes.data(),
                gpuListSizeInBytes,
                stream,
                true /* exact reserved size */);
    }
    
    listCodes->numVecs = numVecs;

    addIndicesFromCpu_(headId, listId, indices, numVecs);

    deviceListDataPointers_[headId].setAt(listId, (void*)listCodes->data.data(), stream);
    deviceListLengths_[headId].setAt(listId, numVecs, stream);

    maxListLength_ = std::max(maxListLength_, numVecs);
}

// Placeholder for addIndicesFromCpu_
void MultiHeadIVFBase::addIndicesFromCpu_(
            idx_t headId, idx_t listId, const idx_t* indices, idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    idx_t globalListId = headId * nlists_[headId] + listId;

    FAISS_ASSERT(headId < deviceListIndices_.size() && listId < deviceListIndices_[headId].size());
    auto& listIndices = deviceListIndices_[headId][listId];
    FAISS_ASSERT(listIndices->data.size() == 0);
    FAISS_ASSERT(listIndices->numVecs == 0);

    if (numVecs == 0) return;

    if (indicesOptions_ == INDICES_32_BIT) {
        std::vector<int> indices32(numVecs);
        for (idx_t i = 0; i < numVecs; ++i) {
            FAISS_ASSERT(indices[i] <= (idx_t)std::numeric_limits<int>::max());
            indices32[i] = (int)indices[i];
        }
        listIndices->data.append(
                (uint8_t*)indices32.data(), numVecs * sizeof(int), stream, true);
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        listIndices->data.append(
                (uint8_t*)indices, numVecs * sizeof(idx_t), stream, true);
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_CPU) {
        FAISS_ASSERT(headId < listOffsetToUserIndex_.size() && listId < listOffsetToUserIndex_[headId].size());
        auto& userIndices = listOffsetToUserIndex_[headId][listId];
        userIndices.insert(userIndices.begin(), indices, indices + numVecs);
        // numVecs for listIndices DeviceIVFList remains 0 as data is on CPU
    } else {
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
        // numVecs for listIndices DeviceIVFList remains 0
    }

    deviceListIndexPointers_[headId].setAt(listId, (void*)listIndices->data.data(), stream);
}


// Other function implementations would follow, adapting to the multi-head structure.
// For example, copyInvertedListsFrom would iterate through the input std::vector<InvertedLists*>
// and call addEncodedVectorsToList_ for each head and list.
// searchCoarseQuantizer_ would iterate through heads, slice the input/output tensors,
// and call the underlying coarse quantizer's search/compute_residual methods for each head.

void MultiHeadIVFBase::searchCoarseQuantizer_(
        std::vector<Index*>& coarseQuantizers,
        std::vector<int>& nprobe,
        Tensor<float, 2, true>* vecs,
        Tensor<float, 2, true>* distances,
        Tensor<idx_t, 2, true>* indices,
        Tensor<float, 3, true>* residuals,
        Tensor<float, 3, true>* centroids) { 
    auto stream = resources_->getDefaultStreamCurrentDevice();

    for (int h = 0; h < numHeads_; ++h) {
        Index* coarseQuantizer = coarseQuantizers[h];
        int currentNprobe = nprobe[h];
        Tensor<float, 2, true>* currentVecs = vecs + h;
        Tensor<float, 2, true>* currentDistances = distances + h;
        Tensor<idx_t, 2, true>* currentIndices = indices + h;
        Tensor<float, 3, true>* currentResiduals = residuals ? residuals + h : nullptr;
        Tensor<float, 3, true>* currentCentroids = centroids ? centroids + h : nullptr;

        FAISS_THROW_IF_NOT(coarseQuantizer);
        FAISS_THROW_IF_NOT(currentVecs && currentDistances && currentIndices);
        // Add size assertions for currentVecs, currentDistances, currentIndices based on currentNprobe

        // Tensor<float, 3, true>* currentResiduals = nullptr;
        // if (residualsPerHeadVec && (*residualsPerHeadVec)[h]) {
        //     currentResiduals = (*residualsPerHeadVec)[h];
        //     // Add size assertions for currentResiduals
        // }

        // Tensor<float, 3, true>* currentCentroids = nullptr;
        // if (centroidsPerHeadVec && (*centroidsPerHeadVec)[h]) {
        //     currentCentroids = (*centroidsPerHeadVec)[h];
        //     // Add size assertions for currentCentroids
        // }

        auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
        if (gpuQuantizer) {
            gpuQuantizer->search(
                    currentVecs->getSize(0),
                    currentVecs->data(),
                    currentNprobe,
                    currentDistances->data(),
                    currentIndices->data());

            if (currentResiduals) {
                gpuQuantizer->compute_residual_n(
                        currentVecs->getSize(0) * currentNprobe,
                        currentVecs->data(),
                        currentResiduals->data(),
                        currentIndices->data());
            }

            if (currentCentroids) {
                 gpuQuantizer->reconstruct_batch(
                    currentVecs->getSize(0) * currentNprobe,
                    currentIndices->data(),
                    currentCentroids->data());
            }
        } else { // CPU quantizer
            auto cpuVecs = toHost<float, 2>(
                    currentVecs->data(), stream, {currentVecs->getSize(0), currentVecs->getSize(1)});
            auto cpuDistances = std::vector<float>(currentVecs->getSize(0) * currentNprobe);
            auto cpuIndices = std::vector<idx_t>(currentVecs->getSize(0) * currentNprobe);

            coarseQuantizer->search(
                    currentVecs->getSize(0),
                    cpuVecs.data(),
                    currentNprobe,
                    cpuDistances.data(),
                    cpuIndices.data());

            currentDistances->copyFrom(cpuDistances, stream);
            currentIndices->copyFrom(cpuIndices, stream);

            if (currentResiduals) {
                auto cpuResiduals = std::vector<float>(currentVecs->getSize(0) * currentNprobe * dim_);
                coarseQuantizer->compute_residual_n(
                        currentVecs->getSize(0) * currentNprobe,
                        cpuVecs.data(),
                        cpuResiduals.data(),
                        cpuIndices.data());
                currentResiduals->copyFrom(cpuResiduals, stream);
            }

            if (currentCentroids) {
                auto cpuCentroids = std::vector<float>(currentVecs->getSize(0) * currentNprobe * dim_);
                 coarseQuantizer->reconstruct_batch(
                    currentVecs->getSize(0) * currentNprobe,
                    cpuIndices.data(),
                    cpuCentroids.data());
                currentCentroids->copyFrom(cpuCentroids, stream);
            }
        }
    }
}


idx_t MultiHeadIVFBase::addVectors(
        std::vector<Index*>& coarseQuantizers,
        Tensor<float, 2, true>* vecs,
        Tensor<idx_t, 1, true>* indices) {
    // FAISS_THROW_IF_NOT(coarseQuantizers.size() == (size_t)numHeads_);
    // FAISS_THROW_IF_NOT(vecsPerHead.size() == (size_t)numHeads_);
    // FAISS_THROW_IF_NOT(userIndicesPerHead.size() == (size_t)numHeads_);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    idx_t totalAdded = 0;

    std::vector<int> nprobes (numHeads_, 1); 

    DeviceTensor<float, 2, true> unusedIVFDistances[numHeads_];
    DeviceTensor<idx_t, 2, true> ivfIndices[numHeads_];
    DeviceTensor<float, 3, true> residuals[numHeads_];

    for (int h = 0; h < numHeads_; ++h) {
        unusedIVFDistances[h] = DeviceTensor<float, 2, true>(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {(vecs + h)->getSize(0), 1});
        ivfIndices[h] = DeviceTensor<idx_t, 2, true>(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {(vecs + h)->getSize(0), 1});
        residuals[h] = DeviceTensor<float, 3, true>(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {(vecs + h)->getSize(0), 1, dim_});
    }

    searchCoarseQuantizer_(
        coarseQuantizers,
        nprobes, // nprobe
        vecs,
        unusedIVFDistances,
        ivfIndices,
        useResidual_ ? residuals : nullptr,
        nullptr);

    for (int h = 0; h < numHeads_; ++h) {
        if ((vecs + h) -> getSize(0) == 0) continue;

        // Host-side processing for list assignments for the current head
        auto ivfIndicesHost = (ivfIndices + h) -> copyToVector(stream); // (numVecsForHead, 1)

        std::unordered_map<idx_t, std::vector<idx_t>> listToVectorIds_h; // listId in head -> original vector indices for this head
        std::vector<idx_t> vectorIdToList_h((vecs + h) -> getSize(0));
        std::vector<idx_t> listOffsetHost_h(ivfIndicesHost.size());
        idx_t numAdded_h = 0;

        for (idx_t i = 0; i < ivfIndicesHost.size(); ++i) { // ivfIndicesHost.size() == numVecsForHead
            idx_t listIdInHead = ivfIndicesHost[i]; // This is listId within the current head h

            if (listIdInHead < 0) { // Invalid vector
                listOffsetHost_h[i] = -1;
                vectorIdToList_h[i] = -1;
                continue;
            }
            FAISS_ASSERT(listIdInHead < nlists_[h]); // numLists_ is lists per head
            ++numAdded_h;
            vectorIdToList_h[i] = listIdInHead;

            idx_t offset = deviceListData_[h][listIdInHead]->numVecs;
            auto it = listToVectorIds_h.find(listIdInHead);
            if (it != listToVectorIds_h.end()) {
                offset += it->second.size();
                it->second.push_back(i);
            } else {
                listToVectorIds_h[listIdInHead] = std::vector<idx_t>{i};
            }
            listOffsetHost_h[i] = offset;
        }

        if (numAdded_h == 0) {
            continue;
        }
        totalAdded += numAdded_h;

        std::vector<idx_t> uniqueLists_h; // list IDs *within this head*
        for (auto& entry : listToVectorIds_h) {
            uniqueLists_h.push_back(entry.first);
        }
        std::sort(uniqueLists_h.begin(), uniqueLists_h.end());

        std::vector<idx_t> vectorsByUniqueList_h;
        std::vector<idx_t> uniqueListVectorStart_h;
        std::vector<idx_t> uniqueListStartOffset_h; // Start offset in the actual IVF list *for this head*

        for (auto ul_h : uniqueLists_h) {
            uniqueListVectorStart_h.push_back(vectorsByUniqueList_h.size());
            auto& vecs_h_for_ul = listToVectorIds_h[ul_h];
            vectorsByUniqueList_h.insert(vectorsByUniqueList_h.end(), vecs_h_for_ul.begin(), vecs_h_for_ul.end());
            uniqueListStartOffset_h.push_back(deviceListData_[h][ul_h]->numVecs);
        }
        uniqueListVectorStart_h.push_back(vectorsByUniqueList_h.size());

        // Resize device list data structures for head h
        {
            for (auto ul_h : uniqueLists_h) {
                idx_t numVecsToAdd = listToVectorIds_h[ul_h].size();
                auto& codes = deviceListData_[h][ul_h];
                idx_t oldNumVecs = codes->numVecs;
                idx_t newNumVecs = oldNumVecs + numVecsToAdd;

                codes->data.resize(getGpuVectorsEncodingSize_(newNumVecs), stream);
                codes->numVecs = newNumVecs;

                if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
                    auto& indicesList = deviceListIndices_[h][ul_h];
                    size_t indexEntrySize = (indicesOptions_ == INDICES_32_BIT) ? sizeof(int) : sizeof(idx_t);
                    indicesList->data.resize(indicesList->data.size() + numVecsToAdd * indexEntrySize, stream);
                    FAISS_ASSERT(indicesList->numVecs == oldNumVecs); // Assuming numVecs was also for indices
                    indicesList->numVecs = newNumVecs;
                } else if (indicesOptions_ == INDICES_CPU) {
                    listOffsetToUserIndex_[h][ul_h].resize(newNumVecs);
                }
                maxListLength_ = std::max(maxListLength_, newNumVecs);
            }

            updateDeviceListInfo_(
                h, uniqueLists_h, stream); // Update device list info for all heads
        }

        if (indicesOptions_ == INDICES_CPU) {
            HostTensor<idx_t, 1, true> currentHeadUserIndices(*(indices + h), stream); // Copy to host
            for (idx_t i = 0; i < currentHeadUserIndices.getSize(0); ++i) {
                idx_t listIdInHead = vectorIdToList_h[i];
                if (listIdInHead < 0) continue;
                idx_t offsetInList = listOffsetHost_h[i];
                FAISS_ASSERT(offsetInList >= 0 && offsetInList < listOffsetToUserIndex_[h][listIdInHead].size());
                listOffsetToUserIndex_[h][listIdInHead][offsetInList] = currentHeadUserIndices[i];
            }
        }

        // Prepare Tensors for appendVectors_ (all are 1D for a single head's processing)
        // assignedListIds_h: for each vector in vecsPerHead[h], which listIdInHead it's assigned to.
        auto assignedListIds_h_dev =
            ivfIndices[h].downcastOuter<1>(); // This is (numVecsForHead, 1), so downcast is (numVecsForHead)

        auto listOffset_h_dev =
            toDeviceTemporary(resources_, listOffsetHost_h, stream);
        auto uniqueLists_h_dev =
            toDeviceTemporary(resources_, uniqueLists_h, stream);
        auto vectorsByUniqueList_h_dev =
            toDeviceTemporary(resources_, vectorsByUniqueList_h, stream);
        auto uniqueListVectorStart_h_dev =
            toDeviceTemporary(resources_, uniqueListVectorStart_h, stream);
        auto uniqueListStartOffset_h_dev =
            toDeviceTemporary(resources_, uniqueListStartOffset_h, stream);

        // TODO: appendVectors_
        // Call pure virtual appendVectors_
        // Note: vecsPerHead[h] and userIndicesPerHead[h] are T*, so dereference them.
        // appendVectors_(
        //     h,                              // headId
        //     *vecsPerHead[h],                // vecs for this head
        //     residuals2D_h,                  // residuals for this head
        //     *userIndicesPerHead[h],         // user indices for this head
        //     uniqueLists_h_dev,
        //     vectorsByUniqueList_h_dev,
        //     uniqueListVectorStart_h_dev,
        //     uniqueListStartOffset_h_dev,
        //     assignedListIds_h_dev,          // listIds for each vector in vecsPerHead[h]
        //     listOffset_h_dev,               // listOffset for each vector in vecsPerHead[h]
        //     stream);
        // }
    }

    return totalAdded;
}

} // namespace gpu
} // namespace faiss
