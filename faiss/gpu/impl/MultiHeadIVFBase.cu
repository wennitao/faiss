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
        idx_t nlistPerHead, // Renamed for clarity, numLists_ member stores this
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
        numLists_(nlistPerHead), // numLists_ now means lists per head
        useResidual_(useResidual),
        interleavedLayout_(interleavedLayout),
        indicesOptions_(indicesOptions),
        space_(space),
        // Global device vectors are sized for total lists across all heads
        deviceListDataPointers_(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice())),
        deviceListIndexPointers_(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice())),
        deviceListLengths_(
                resources,
                AllocInfo(
                        AllocType::IVFLists,
                        getCurrentDevice(),
                        space,
                        resources->getDefaultStreamCurrentDevice())),
        maxListLength_(0) {
    FAISS_THROW_IF_NOT(numHeads_ > 0);
    FAISS_THROW_IF_NOT(numLists_ > 0); // numLists_ is nlistPerHead
    reset();
}

MultiHeadIVFBase::~MultiHeadIVFBase() {}

void MultiHeadIVFBase::reserveMemory(idx_t totalNumVecs) { // totalNumVecs across all heads
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Approximate vecs per list across all heads and lists
    idx_t totalLists = numHeads_ * numLists_;
    if (totalLists == 0) return; // Avoid division by zero

    auto vecsPerList = totalNumVecs / totalLists;
    if (vecsPerList < 1) {
        // Not enough vectors to even put one in each list,
        // or no lists defined.
        // Depending on desired behavior, could return or log.
        // For now, if vecsPerList is 0, reserving 0 bytes is fine.
        if (totalNumVecs > 0 && vecsPerList == 0) {
            // If there are some vecs but not enough for 1 per list,
            // we might still want to reserve a minimal amount for some lists.
            // For simplicity, let's assume if vecsPerList is 0, we reserve for 0.
            // A more sophisticated strategy could be to reserve for 'totalNumVecs'
            // in the first list of the first head, for example.
            return;
        }
    }

    auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

    for (int h = 0; h < numHeads_; ++h) {
        for (idx_t l = 0; l < numLists_; ++l) {
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
            for (idx_t l = 0; l < numLists_; ++l) {
                 if (h < deviceListIndices_.size() && l < deviceListIndices_[h].size() && deviceListIndices_[h][l]) {
                    deviceListIndices_[h][l]->data.reserve(bytesPerIndexList, stream);
                }
            }
        }
    }

    // Update device info for all lists, since the base pointers may
    // have changed. This function needs to be aware of the multi-head structure
    // to correctly update the flat global device vectors.
    updateDeviceListInfo_(stream);
}

void MultiHeadIVFBase::reset() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (isTranslatedCodesStored_) {
        for (int h = 0; h < numHeads_; ++h) {
            if (h < translatedCodes_.size()) {
                for (idx_t l = 0; l < numLists_; ++l) {
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
    deviceListDataPointers_.clear();
    deviceListIndexPointers_.clear();
    deviceListLengths_.clear();


    auto info =
            AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, stream);

    deviceListData_.resize(numHeads_);
    deviceListIndices_.resize(numHeads_);
    translatedCodes_.resize(numHeads_);
    listOffsetToUserIndex_.resize(numHeads_);

    for (int h = 0; h < numHeads_; ++h) {
        deviceListData_[h].resize(numLists_);
        deviceListIndices_[h].resize(numLists_);
        translatedCodes_[h].resize(numLists_, nullptr); // Initialize with nullptr
        listOffsetToUserIndex_[h].resize(numLists_);

        for (idx_t l = 0; l < numLists_; ++l) {
            deviceListData_[h][l] = std::make_unique<DeviceIVFList>(resources_, info);
            deviceListIndices_[h][l] = std::make_unique<DeviceIVFList>(resources_, info);
            // listOffsetToUserIndex_[h][l] is an empty std::vector<idx_t> by default
            // translatedCodes_[h][l] is already nullptr
        }
    }

    idx_t totalLists = numHeads_ * numLists_;
    if (totalLists > 0) {
        deviceListDataPointers_.resize(totalLists, stream);
        deviceListDataPointers_.setAll(nullptr, stream);

        deviceListIndexPointers_.resize(totalLists, stream);
        deviceListIndexPointers_.setAll(nullptr, stream);

        deviceListLengths_.resize(totalLists, stream);
        deviceListLengths_.setAll(0, stream);
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
    return numLists_; // numLists_ stores lists per head
}


size_t MultiHeadIVFBase::reclaimMemory() {
    // Reclaim all unused memory exactly
    return reclaimMemory_(true);
}

size_t MultiHeadIVFBase::reclaimMemory_(bool exact) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    size_t totalReclaimed = 0;

    for (int h = 0; h < numHeads_; ++h) {
        for (idx_t l = 0; l < numLists_; ++l) {
            idx_t globalListIdx = h * numLists_ + l;

            if (h < deviceListData_.size() && l < deviceListData_[h].size() && deviceListData_[h][l]) {
                auto& dataList = deviceListData_[h][l];
                totalReclaimed += dataList->data.reclaim(exact, stream);
                if (globalListIdx < deviceListDataPointers_.size()) {
                    deviceListDataPointers_.setAt(globalListIdx, (void*)dataList->data.data(), stream);
                }
            }

            if (h < deviceListIndices_.size() && l < deviceListIndices_[h].size() && deviceListIndices_[h][l]) {
                 if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
                    auto& indexList = deviceListIndices_[h][l];
                    totalReclaimed += indexList->data.reclaim(exact, stream);
                    if (globalListIdx < deviceListIndexPointers_.size()) {
                        deviceListIndexPointers_.setAt(globalListIdx, (void*)indexList->data.data(), stream);
                    }
                }
            }
        }
    }

    updateDeviceListInfo_(stream);
    return totalReclaimed;
}

void MultiHeadIVFBase::updateDeviceListInfo_(cudaStream_t stream) {
    idx_t totalLists = numHeads_ * numLists_;
    if (totalLists == 0) return;

    std::vector<idx_t> globalListIds(totalLists);
    for (idx_t i = 0; i < totalLists; ++i) {
        globalListIds[i] = i;
    }

    updateDeviceListInfo_(globalListIds, stream);
}

void MultiHeadIVFBase::updateDeviceListInfo_(
        const std::vector<idx_t>& globalListIdsToUpdate, // These are global list IDs
        cudaStream_t stream) {
    idx_t numListsToUpdate = globalListIdsToUpdate.size();
    if (numListsToUpdate == 0) return;

    HostTensor<idx_t, 1, true> hostListsToUpdate({numListsToUpdate});
    HostTensor<idx_t, 1, true> hostNewListLength({numListsToUpdate});
    HostTensor<void*, 1, true> hostNewDataPointers({numListsToUpdate});
    HostTensor<void*, 1, true> hostNewIndexPointers({numListsToUpdate});

    for (idx_t i = 0; i < numListsToUpdate; ++i) {
        auto globalListId = globalListIdsToUpdate[i];
        idx_t headId = globalListId / numLists_;
        idx_t listIdInHead = globalListId % numLists_;

        FAISS_ASSERT(headId < numHeads_);
        FAISS_ASSERT(listIdInHead < numLists_);
        FAISS_ASSERT(headId < deviceListData_.size() && listIdInHead < deviceListData_[headId].size());
        FAISS_ASSERT(headId < deviceListIndices_.size() && listIdInHead < deviceListIndices_[headId].size());


        auto& dataList = deviceListData_[headId][listIdInHead];
        auto& indexList = deviceListIndices_[headId][listIdInHead];

        hostListsToUpdate[i] = globalListId; // Use the global list ID for the update kernel
        hostNewListLength[i] = dataList->numVecs;
        hostNewDataPointers[i] = dataList->data.data();

        if ((indicesOptions_ == INDICES_32_BIT) || (indicesOptions_ == INDICES_64_BIT)) {
            hostNewIndexPointers[i] = indexList->data.data();
        } else {
            hostNewIndexPointers[i] = nullptr; // Or handle as appropriate for other options
        }
    }

    DeviceTensor<idx_t, 1, true> listsToUpdateDev(
            resources_, makeTempAlloc(AllocType::Other, stream), hostListsToUpdate);
    DeviceTensor<idx_t, 1, true> newListLengthDev(
            resources_, makeTempAlloc(AllocType::Other, stream), hostNewListLength);
    DeviceTensor<void*, 1, true> newDataPointersDev(
            resources_, makeTempAlloc(AllocType::Other, stream), hostNewDataPointers);
    DeviceTensor<void*, 1, true> newIndexPointersDev(
            resources_, makeTempAlloc(AllocType::Other, stream), hostNewIndexPointers);

    runUpdateListPointers(
            listsToUpdateDev,
            newListLengthDev,
            newDataPointersDev,
            newIndexPointersDev,
            deviceListLengths_,         // Global
            deviceListDataPointers_,    // Global
            deviceListIndexPointers_,   // Global
            stream);
}

// ... (Implement other functions like getListLength, getListIndices, getListVectorData,
//      copyInvertedListsFrom, storeTranslatedCodes, addEncodedVectorsToList_, addIndicesFromCpu_,
//      updateQuantizer, searchCoarseQuantizer_ etc. adapting them for the multi-head structure,
//      using headId and listIdInHead for per-head structures, and globalListId for global DeviceVectors)
//
// Example for getListLength:
idx_t MultiHeadIVFBase::getListLength(idx_t headId, idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(headId < numHeads_, "Head ID %ld out of bounds (%d heads total)", headId, numHeads_);
    FAISS_THROW_IF_NOT_FMT(listId < numLists_, "List ID %ld out of bounds (%ld lists per head)", listId, numLists_);
    
    FAISS_ASSERT(headId < deviceListData_.size());
    FAISS_ASSERT(listId < deviceListData_[headId].size());
    FAISS_ASSERT(deviceListData_[headId][listId]);

    return deviceListData_[headId][listId]->numVecs;
}


// Placeholder for addEncodedVectorsToList_ (needs careful implementation)
void MultiHeadIVFBase::addEncodedVectorsToList_(
        idx_t headId,
        idx_t listId,
        const void* codes, // from host
        const idx_t* indices, // from host
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    idx_t globalListId = headId * numLists_ + listId;

    FAISS_ASSERT(headId < deviceListData_.size() && listId < deviceListData_[headId].size());
    auto& listCodes = deviceListData_[headId][listId];
    FAISS_ASSERT(listCodes->data.size() == 0); // Assuming adding to an empty (pre-reserved) list segment
    FAISS_ASSERT(listCodes->numVecs == 0);

    if (numVecs == 0) {
        return;
    }

    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);

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

    if (globalListId < deviceListDataPointers_.size()) {
        deviceListDataPointers_.setAt(globalListId, (void*)listCodes->data.data(), stream);
    }
    if (globalListId < deviceListLengths_.size()) {
        deviceListLengths_.setAt(globalListId, numVecs, stream);
    }

    maxListLength_ = std::max(maxListLength_, numVecs);
}

// Placeholder for addIndicesFromCpu_
void MultiHeadIVFBase::addIndicesFromCpu_(
            idx_t headId, idx_t listId, const idx_t* indices, idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    idx_t globalListId = headId * numLists_ + listId;

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

    if ((indicesOptions_ == INDICES_32_BIT || indicesOptions_ == INDICES_64_BIT) &&
        globalListId < deviceListIndexPointers_.size()) {
        deviceListIndexPointers_.setAt(globalListId, (void*)listIndices->data.data(), stream);
    }
}


// Other function implementations would follow, adapting to the multi-head structure.
// For example, copyInvertedListsFrom would iterate through the input std::vector<InvertedLists*>
// and call addEncodedVectorsToList_ for each head and list.
// searchCoarseQuantizer_ would iterate through heads, slice the input/output tensors,
// and call the underlying coarse quantizer's search/compute_residual methods for each head.

} // namespace gpu
} // namespace faiss
