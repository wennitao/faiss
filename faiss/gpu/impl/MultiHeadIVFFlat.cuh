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
        std::vector<idx_t> nlists,
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
        Tensor<float, 2, true>* queries,
        const std::vector<int>& nprobe,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices) override;

    void searchPreassigned(
        std::vector<Index*>& coarseQuantizers,
        Tensor<float, 2, true>* vecs,
        Tensor<float, 2, true>* ivfDistances,
        Tensor<idx_t, 2, true>* ivfAssignments,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices,
        bool storePairs) override;

    void reconstruct_n(idx_t headId, idx_t i0, idx_t n, float* out) override;

protected:
    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
    size_t getCpuVectorsEncodingSize_(idx_t numVecs) const override;

    std::vector<uint8_t> translateCodesToGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const override;

    std::vector<uint8_t> translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const override;

    void appendVectors_(
        Tensor<float, 2, true>* vecs,
        Tensor<float, 2, true>* ivfCentroidResiduals,
        Tensor<idx_t, 1, true>* userIndices,
        Tensor<idx_t, 1, true>* uniqueLists,
        Tensor<idx_t, 1, true>* vectorsByUniqueList,
        Tensor<idx_t, 1, true>* uniqueListVectorStart,
        Tensor<idx_t, 1, true>* uniqueListStartOffset,
        Tensor<idx_t, 1, true>* assignedListIds, // list IDs within headId
        Tensor<idx_t, 1, true>* listOffsets, // offsets within listId
        cudaStream_t stream) override;

    virtual void searchImpl_(
        Tensor<float, 2, true>* queries,
        Tensor<float, 2, true>* coarseDistances,
        Tensor<idx_t, 2, true>* coarseIndices,
        Tensor<float, 3, true>* ivfCentroids,
        const std::vector<int>& k,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices,
        bool storePairs);

protected:
    std::vector<std::unique_ptr<GpuScalarQuantizer>> scalarQs_;
};

} // namespace gpu
} // namespace faiss