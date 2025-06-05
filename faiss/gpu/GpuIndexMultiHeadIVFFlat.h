#pragma once

#include <faiss/gpu/GpuIndexMultiHeadIVF.h>
#include <faiss/impl/ScalarQuantizer.h> // For faiss::ScalarQuantizer
#include <memory>
#include <vector>

namespace faiss {
struct IndexIVFFlat; // Forward declaration
}

namespace faiss {
namespace gpu {

class MultiHeadIVFFlat; // Forward declaration

struct GpuIndexMultiHeadIVFFlatConfig : public GpuIndexMultiHeadIVFConfig {
    bool interleavedLayout = true;
    // If you need per-head scalar quantizers managed by this config,
    // you could add std::vector<faiss::ScalarQuantizer> scalarQuantizers;
    // and manage their lifetime. For now, we'll assume SQs are passed directly.
};

class GpuIndexMultiHeadIVFFlat : public GpuIndexMultiHeadIVF {
   public:
    GpuIndexMultiHeadIVFFlat(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFFlat* indices, // Source CPU index
            int num_heads_target, // Number of heads for the new GPU index
            GpuIndexMultiHeadIVFFlatConfig config = GpuIndexMultiHeadIVFFlatConfig());

    GpuIndexMultiHeadIVFFlat(
            GpuResourcesProvider* provider,
            int dims,
            int num_heads,
            std::vector<idx_t> nlists,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexMultiHeadIVFFlatConfig config = GpuIndexMultiHeadIVFFlatConfig());

    GpuIndexMultiHeadIVFFlat(
            GpuResourcesProvider* provider,
            std::vector<Index*> coarseQuantizers, // One per head
            int dims,
            int num_heads, 
            // num_heads is implicit from coarseQuantizers.size()
            std::vector<idx_t> nlists,
            faiss::MetricType metric = faiss::METRIC_L2,
            // Optional: pass SQs, one per head, can contain nullptrs
            const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead = {},
            GpuIndexMultiHeadIVFFlatConfig config = GpuIndexMultiHeadIVFFlatConfig());

    ~GpuIndexMultiHeadIVFFlat() override;

    void reserveMemory(size_t numVecsTotal); // Total vectors across all lists/heads

    void copyFromIndexOnly (const faiss::IndexIVFFlat* indices);
    void translateCodesToGpu (const faiss::IndexIVFFlat* indices);
    void copyInvertedLists(
            const faiss::IndexIVFFlat* indices,
            GpuMemoryReservation* ivfListDataReservation,
            GpuMemoryReservation* ivfListIndexReservation);

    void copyFrom(const faiss::IndexIVFFlat* indices);

    // copyTo for a single IndexIVFFlat.
    // Coarse quantizer and lists from head 0 are copied.
    // deprecated
    void copyTo(faiss::IndexIVFFlat* index) const;

    size_t reclaimMemory();

    void reset() override;

    // updateQuantizer is inherited from GpuIndexMultiHeadIVF
    // and calls multiHeadBaseIndex_->updateQuantizer(quantizers_)

    void updateQuantizer() override;

    void train(idx_t n, const float* x) override;

    // Reconstructs from head 0 for GpuIndex compatibility
    void reconstruct_n(idx_t i0, idx_t n, float* out) const override;
    // Multi-head specific reconstruction
    virtual void reconstruct_n_for_head(int headId, idx_t i0, idx_t n, float* out) const;


   protected:
    void setIndex_(
            GpuResources* resources,
            int dims,
            int num_heads,
            std::vector<idx_t> nlists,
            faiss::MetricType metric,
            float metricArg,
            bool useResidual, // Typically false for IVFFlat
            const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

   protected:
    const GpuIndexMultiHeadIVFFlatConfig ivfFlatConfig_;
    size_t reserveMemoryVecs_;

    // Instance that we own; contains the multi-head inverted lists
    std::shared_ptr<MultiHeadIVFFlat> index_;
    // GpuIndexMultiHeadIVF::multiHeadBaseIndex_ will also point to index_
};

} // namespace gpu
} // namespace faiss