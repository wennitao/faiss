/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <memory>
#include <vector>

namespace faiss {
struct InvertedLists;
}

namespace faiss {
namespace gpu {

class GpuResources;
class FlatIndex;

/// Uses a single contiguous memory allocation for all IVF lists
/// This avoids fragmentation and frequent re-allocation of device memory
class IVFBaseContiguous : public IVFBase {
   public:
    IVFBaseContiguous(
            GpuResources* resources,
            int dim,
            idx_t nlist,
            faiss::MetricType metric,
            float metricArg,
            bool interleavedLayout,
            bool useResidual,
            IndicesOptions indicesOptions,
            MemorySpace space);

    ~IVFBaseContiguous() override;

    /// Reserve GPU memory for all inverted lists in a single allocation
    void reserveMemory(idx_t numVecs) override;

    /// Clear out all inverted lists but keep allocated memory
    void reset() override;

    /// After adding vectors, reclaim device memory to exactly the amount needed
    size_t reclaimMemory() override;

    /// Copy all inverted lists from a CPU representation efficiently
    void copyInvertedListsFrom(const InvertedLists* ivf) override;

   protected:
    /// Adds encoded vectors to a specific list
    void addEncodedVectorsToList_(
            idx_t listId,
            // resident on the host
            const void* codes,
            // resident on the host
            const idx_t* indices,
            idx_t numVecs) override;

    /// Update memory pointers for all lists on device
    void updateDeviceListInfo_(cudaStream_t stream) override;

    /// Update pointers for specific lists on device
    void updateDeviceListInfo_(
            const std::vector<idx_t>& listIds,
            cudaStream_t stream) override;

   private:
    /// Single contiguous memory allocation for all list data
    DeviceVector<uint8_t> contiguousData_;
    
    /// Single contiguous memory allocation for all list indices
    DeviceVector<uint8_t> contiguousIndices_;
    
    /// Store offsets into contiguous memory for each list's data
    std::vector<size_t> listDataOffsets_;
    
    /// Store offsets into contiguous memory for each list's indices
    std::vector<size_t> listIndicesOffsets_;
    
    /// Total size in bytes of all list data
    size_t totalDataSize_;
    
    /// Total size in bytes of all list indices
    size_t totalIndicesSize_;
    
    /// Allocation growth factor to minimize reallocations
    float growthFactor_;
};

} // namespace gpu
} // namespace faiss
