#pragma once

#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/impl/MultiHeadIVFBase.cuh>
#include <vector>
#include <memory>

namespace faiss {
namespace gpu {

class MultiHeadIVFFlat : public MultiHeadIVFBase {
public:
    MultiHeadIVFFlat(
        GpuResources* resources,
        int numHeads,
        int dim,
        idx_t nlistPerHead,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead, // One SQ per head, can be nullptr
        bool interleavedLayout, // This is now a member of MultiHeadIVFBase
        IndicesOptions indicesOptions,
        MemorySpace space);

    ~MultiHeadIVFFlat() override;

    void search(
        std::vector<Index*>& coarseQuantizers,
        const std::vector<Tensor<float, 2, true>*>& queriesPerHead,
        const std::vector<int>& nprobePerHead,
        const std::vector<int>& kPerHead,
        std::vector<Tensor<float, 2, true>*>& outDistancesPerHead,
        std::vector<Tensor<idx_t, 2, true>*>& outIndicesPerHead) override;

    void searchPreassigned(
        std::vector<Index*>& coarseQuantizers,
        const std::vector<Tensor<float, 2, true>*>& vecsPerHead,
        const std::vector<Tensor<float, 2, true>*>& ivfDistancesPerHead,
        const std::vector<Tensor<idx_t, 2, true>*>& ivfAssignmentsPerHead,
        const std::vector<int>& kPerHead,
        std::vector<Tensor<float, 2, true>*>& outDistancesPerHead,
        std::vector<Tensor<idx_t, 2, true>*>& outIndicesPerHead,
        bool storePairs) override;

    void reconstruct_n(idx_t headId, idx_t i0, idx_t n, float* out) override;

protected:
    size_t getGpuVectorsEncodingSize_(idx_t headId, idx_t numVecs) const override;
    size_t getCpuVectorsEncodingSize_(idx_t headId, idx_t numVecs) const override;

    std::vector<uint8_t> translateCodesToGpu_(
        idx_t headId,
        std::vector<uint8_t> codes,
        idx_t numVecs) const override;

    std::vector<uint8_t> translateCodesFromGpu_(
        idx_t headId,
        std::vector<uint8_t> codes,
        idx_t numVecs) const override;

    void appendVectors_(
        idx_t headId,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfCentroidResiduals,
        Tensor<idx_t, 1, true>& userIndices,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        Tensor<idx_t, 1, true>& assignedListIds, // list IDs within headId
        Tensor<idx_t, 1, true>& listOffsets, // offsets within listId
        cudaStream_t stream) override;

    virtual void searchImpl_(
        const std::vector<Tensor<float, 2, true>*>& queriesPerHead,
        const std::vector<Tensor<float, 2, true>*>& coarseDistancesPerHead,
        // coarseIndicesPerHead contains LOCAL list IDs (0 to numLists_ - 1) for each head
        const std::vector<Tensor<idx_t, 2, true>*>& coarseIndicesPerHead,
        const std::vector<Tensor<float, 3, true>*>& ivfCentroidsPerHead, // Residual base if useResidual_
        const std::vector<int>& kPerHead,
        std::vector<Tensor<float, 2, true>*>& outDistancesPerHead,
        std::vector<Tensor<idx_t, 2, true>*>& outIndicesPerHead,
        bool storePairs);

protected:
    std::vector<std::unique_ptr<GpuScalarQuantizer>> scalarQs_;
};

} // namespace gpu
} // namespace faiss