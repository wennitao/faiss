#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexMultiHeadIVF.h>
#include <faiss/gpu/GpuIndexMultiHeadIVFFlat.h> // Changed include
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/MultiHeadIVFBase.cuh>
#include <faiss/gpu/impl/MultiHeadIVFFlat.cuh> // Changed include
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>


namespace faiss {
namespace gpu {

GpuIndexMultiHeadIVFFlat::GpuIndexMultiHeadIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* indices,
        int num_heads_target,
        GpuIndexMultiHeadIVFFlatConfig config)
        : GpuIndexMultiHeadIVF(
                  provider,
                  indices->d,
                  num_heads_target, // num_heads for the GpuIndexMultiHeadIVF base
                  indices->metric_type,
                  indices->metric_arg,
                  std::vector<idx_t>(num_heads_target, indices->nlist), // nlist_per_head
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // GpuIndexMultiHeadIVF constructor will create num_heads_target default quantizers
    // if own_coarse_quantizer_ is true.
    // We then copy the single source quantizer to all of them.
    // And copy invlists to head 0.
    copyFrom(indices);
}

GpuIndexMultiHeadIVFFlat::GpuIndexMultiHeadIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        int num_heads,
        std::vector<idx_t> nlists,
        faiss::MetricType metric,
        GpuIndexMultiHeadIVFFlatConfig config)
        : GpuIndexMultiHeadIVF(provider, dims, num_heads, metric, 0, nlists, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // Base class GpuIndexMultiHeadIVF handles creation of empty quantizers.
    // MultiHeadIVFFlat index (this->index_) is not constructed until training.
}

GpuIndexMultiHeadIVFFlat::GpuIndexMultiHeadIVFFlat(
        GpuResourcesProvider* provider,
        std::vector<Index*> coarseQuantizers,
        int dims,
        int num_heads,
        std::vector<idx_t> nlists,
        faiss::MetricType metric,
        const std::vector<faiss::ScalarQuantizer*>& scalarQsPerHead,
        GpuIndexMultiHeadIVFFlatConfig config)
        : GpuIndexMultiHeadIVF(
                  provider,
                  std::move(coarseQuantizers), // Pass vector of CQs to base
                  dims,
                  metric,
                  0,
                  nlists,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // Base class GpuIndexMultiHeadIVF takes ownership if coarseQuantizers were not pre-owned.
    // If all coarse quantizers are trained, we can initialize the MultiHeadIVFFlat index.
    bool all_cqs_trained = true;
    if (quantizers_.empty()) all_cqs_trained = false;
    for (int h = 0; h < num_heads_; ++h) {
        if (!quantizers_[h] || !quantizers_[h]->is_trained || quantizers_[h]->ntotal != nlists[h]) {
            all_cqs_trained = false;
            break;
        }
    }

    if (all_cqs_trained) {
        this->is_trained = true; // Mark overall index as trained if CQs are ready
        setIndex_(
                resources_.get(),
                this->d,
                this->num_heads_,
                this->nlists_,
                this->metric_type,
                this->metric_arg,
                false, // useResidual for IVFFlat is typically false
                scalarQsPerHead,
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        // GpuIndexMultiHeadIVF::multiHeadBaseIndex_ needs to point to this->index_
        this->multiHeadBaseIndex_ = std::static_pointer_cast<MultiHeadIVFBase, MultiHeadIVFFlat>(index_);
        updateQuantizer(); // Sync CQs with the new MultiHeadIVFFlat
    } else {
        this->is_trained = false;
    }
}

GpuIndexMultiHeadIVFFlat::~GpuIndexMultiHeadIVFFlat() {}

void GpuIndexMultiHeadIVFFlat::setIndex_(
        GpuResources* res,
        int dims,
        int num_h, // num_heads
        std::vector<idx_t> nlists, // nlist_per_head
        faiss::MetricType mt,
        float ma, // metricArg
        bool useResidual,
        const std::vector<faiss::ScalarQuantizer*>& scalarQsPh, // scalarQsPerHead
        bool interleavedLayout,
        IndicesOptions indicesOpt,
        MemorySpace mspace) {

    index_.reset(new MultiHeadIVFFlat(
            res,
            num_h,
            dims,
            nlists,
            mt,
            ma,
            useResidual, // Should be false for pure IVFFlat
            scalarQsPh,  // Pass the SQs
            interleavedLayout,
            indicesOpt,
            mspace));
    // GpuIndexMultiHeadIVF::multiHeadBaseIndex_ should point to this.
    multiHeadBaseIndex_ = std::static_pointer_cast<MultiHeadIVFBase, MultiHeadIVFFlat>(index_);
}


void GpuIndexMultiHeadIVFFlat::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);
    reserveMemoryVecs_ = numVecs;
    if (index_) {
        // MultiHeadIVFFlat would need a reserveMemory method
        // For now, this is a placeholder. The reservation logic might be complex
        // for multi-head (e.g. total, or per head/list).
        // index_->reserveMemory(numVecs);
        fprintf(stderr, "GpuIndexMultiHeadIVFFlat::reserveMemory: Underlying MultiHeadIVFFlat reserveMemory not yet implemented.\n");
    }
}

void GpuIndexMultiHeadIVFFlat::copyFromIndexOnly (const faiss::IndexIVFFlat* indices) {
    DeviceScope scope(config_.device);

    GpuIndexMultiHeadIVF::copyFrom (indices) ;

    index_.reset();

    for (int h = 0; h < num_heads_; ++h) {
        if (!(indices + h)->is_trained) {
            FAISS_ASSERT(!is_trained);
            return;
        }
    }

    FAISS_ASSERT(is_trained);

    std::vector<idx_t> nlists(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        nlists[h] = (indices + h)->nlist;
    }

    setIndex_(
        resources_.get(),
        d,
        num_heads_,
        nlists,
        indices->metric_type,
        indices->metric_arg,
        false,   // no residual
        std::vector<faiss::ScalarQuantizer*>(num_heads_, nullptr), // no scalar quantizer
        ivfFlatConfig_.interleavedLayout,
        ivfFlatConfig_.indicesOptions,
        config_.memorySpace);
    
    multiHeadBaseIndex_ = std::static_pointer_cast<MultiHeadIVFBase, MultiHeadIVFFlat>(index_);
    
    updateQuantizer();
}

void GpuIndexMultiHeadIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    DeviceScope scope(config_.device);

    // This copies GpuIndex data (d, metric, etc.)
    // and handles coarse quantizer:
    // - If own_coarse_quantizers_ is true, it creates num_heads_ new CQs.
    // - It then clones index->quantizer into quantizers_[0...num_heads-1].
    // - And copy invlists to head 0.
    GpuIndexMultiHeadIVF::copyFrom(index); // This sets this->is_trained based on index->is_trained

    // Clear out our old data store if any
    index_.reset();
    // GpuIndexMultiHeadIVF::baseIndex_ is also reset by GpuIndexMultiHeadIVF::copyFrom

    if (!this->is_trained) { // if index->is_trained was false
        FAISS_ASSERT(!is_trained);
        return;
    }

    // At this point, GpuIndexMultiHeadIVF::is_trained is true,
    // and all quantizers_ (0 to num_heads-1) are clones of index->quantizer.
    FAISS_ASSERT(is_trained);

    // Create the MultiHeadIVFFlat storage
    // ScalarQuantizers: For IVFFlat, there's no top-level SQ in IndexIVFFlat.
    // If MultiHeadIVFFlat supports per-head SQs, they'd be passed here.
    // For a simple copyFrom IndexIVFFlat, assume no SQs.
    std::vector<faiss::ScalarQuantizer*> emptySQs(num_heads_, nullptr);
    setIndex_(
            resources_.get(),
            d,
            num_heads_,
            nlists_, // Copied from index->nlist by GpuIndexMultiHeadIVF::copyFrom
            index->metric_type,
            index->metric_arg,
            false, // useResidual for IVFFlat
            emptySQs,
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);

    updateQuantizer(); // Ensure MultiHeadIVFFlat gets the CQs
}

void GpuIndexMultiHeadIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(
            ivfFlatConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU IndexIVFFlat as GPU index doesn't retain "
            "original vector IDs (INDICES_IVF set). Use INDICES_32_BIT or INDICES_64_BIT.");

    // This copies coarse quantizer from quantizers_[0] to index->quantizer
    // and sets nlist, nprobe (from head 0), d, metric, ntotal.
    GpuIndexMultiHeadIVF::copyTo(index);

    index->code_size = this->d * sizeof(float); // For IVFFlat

    // If the target index already has invlists, and they are not ArrayInvertedLists,
    // or if their properties don't match, replace them.
    bool replace_invlists = true;
    if (index->invlists) {
        auto* ail = dynamic_cast<ArrayInvertedLists*>(index->invlists);
        if (ail && ail->nlist == index->nlist && ail->code_size == index->code_size) {
            ail->reset(); // Clear existing lists
            replace_invlists = false;
        }
    }

    if (replace_invlists) {
        if (index->own_invlists) delete index->invlists;
        index->invlists = new ArrayInvertedLists(index->nlist, index->code_size);
        index->own_invlists = true;
    }


    if (index_) {
        // Copy IVF lists from head 0 of our MultiHeadIVFFlat to the CPU index.
        // MultiHeadIVFFlat needs a method like copyInvertedListsTo(headId, dest_invlists)
        // index_->copyInvertedListsTo(0, index->invlists); // Conceptual
        fprintf(stderr, "GpuIndexMultiHeadIVFFlat::copyTo: Inverted list copy from MultiHeadIVFFlat (head 0) to IndexIVFFlat needs specific implementation in MultiHeadIVFFlat.\n");
    }
    // After copying lists, ntotal in 'index' should match this->ntotal (set by GpuIndexMultiHeadIVF::copyTo)
    // If direct map is needed for CPU index:
    if (index->direct_map.type != DirectMap::NoMap) {
         index->make_direct_map(true);
    }
}


size_t GpuIndexMultiHeadIVFFlat::reclaimMemory() {
    DeviceScope scope(config_.device);
    if (index_) {
        // return index_->reclaimMemory(); // MultiHeadIVFFlat needs this method
        fprintf(stderr, "GpuIndexMultiHeadIVFFlat::reclaimMemory: Underlying MultiHeadIVFFlat reclaimMemory not yet implemented.\n");
        return 0;
    }
    return 0;
}

void GpuIndexMultiHeadIVFFlat::reset() {
    DeviceScope scope(config_.device);
    // GpuIndexMultiHeadIVF::reset(); // Resets ntotal in base, but not CQs.
    if (index_) {
        index_->reset(); // Resets lists in MultiHeadIVFFlat
    }
    // ntotal is already set to 0 by GpuIndexMultiHeadIVF::reset()
}

void GpuIndexMultiHeadIVFFlat::updateQuantizer() {
    DeviceScope scope(config_.device);
    // MultiHeadIVFFlat::updateQuantizer should update the CQs in the index_
    // This is a conceptual call, actual implementation depends on MultiHeadIVFFlat
    if (index_) {
        index_->updateQuantizer(quantizers_); // Update CQs in MultiHeadIVFFlat
    }
}

void GpuIndexMultiHeadIVFFlat::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // Train all coarse quantizers (if not already trained)
    // GpuIndexMultiHeadIVF::trainQuantizers_ is protected, GpuIndex::train calls it.
    GpuIndex::train(n, x); // This will call trainQuantizers_ and set this->is_trained = true

    FAISS_ASSERT(this->is_trained); // Should be true after GpuIndex::train

    // If MultiHeadIVFFlat storage (index_) isn't created yet, or if CQs changed
    if (!index_) { // quantizers_updated_ is a hypothetical flag
                                        // More robust: check if CQs in index_ match current ones.
                                        // For simplicity, recreate/update if CQs were trained.
        std::vector<faiss::ScalarQuantizer*> emptySQs(num_heads_, nullptr); // Assuming no SQs for basic IVFFlat
        setIndex_(
                resources_.get(),
                this->d,
                this->num_heads_,
                this->nlists_,
                this->metric_type,
                this->metric_arg,
                false, // useResidual for IVFFlat
                emptySQs,
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
    }
    // Ensure the MultiHeadIVFFlat instance has the latest trained quantizers
    updateQuantizer(); // This calls multiHeadBaseIndex_->updateQuantizer(quantizers_)

    if (reserveMemoryVecs_ > 0 && index_) {
        // index_->reserveMemory(reserveMemoryVecs_); // Placeholder
    }
    // this->is_trained is already true from GpuIndex::train
}

void GpuIndexMultiHeadIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* out) const {
    // Delegate to head 0 for GpuIndex compatibility
    reconstruct_n_for_head(0, i0, ni, out);
}

void GpuIndexMultiHeadIVFFlat::reconstruct_n_for_head(int headId, idx_t i0, idx_t ni, float* out) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(index_);
    FAISS_THROW_IF_NOT(headId >= 0 && headId < num_heads_);

    if (ni == 0) return;

    // ntotal in GpuIndex is the sum across all heads (or should be).
    // Reconstruction is tricky if IDs are not unique across heads or if i0, ni refer to global IDs.
    // Assuming i0, ni refer to IDs that can be found within headId's lists.
    // MultiHeadIVFFlat::reconstruct_n needs to handle this.
    index_->reconstruct_n(headId, i0, ni, out);
}

void GpuIndexMultiHeadIVFFlat::translateCodesToGpu(const faiss::IndexIVFFlat* indices) {
    std::vector<InvertedLists*> ivfLists(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        if (indices && indices[h].invlists) {
            ivfLists[h] = indices[h].invlists;
        } else {
            ivfLists[h] = nullptr; // No lists for this head
        }
    }
    index_ -> storeTranslatedCodes(ivfLists);
}

void GpuIndexMultiHeadIVFFlat::copyInvertedLists(
        const faiss::IndexIVFFlat* indices,
        GpuMemoryReservation* ivfListDataReservation,
        GpuMemoryReservation* ivfListIndexReservation) {
    std::vector<InvertedLists*> ivfLists(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        if (indices && indices[h].invlists) {
            ivfLists[h] = indices[h].invlists;
        } else {
            ivfLists[h] = nullptr; // No lists for this head
        }
    }
    index_ -> copyInvertedListsFromNoRealloc(ivfLists, 
                                             ivfListDataReservation,
                                             ivfListIndexReservation);
}


} // namespace gpu
} // namespace faiss