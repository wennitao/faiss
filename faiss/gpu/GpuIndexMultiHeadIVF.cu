#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexMultiHeadIVF.h> // Changed include
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/MultiHeadIVFBase.cuh> // Changed include
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>


namespace faiss {
namespace gpu {

GpuIndexMultiHeadIVF::GpuIndexMultiHeadIVF(
        GpuResourcesProvider* provider,
        int dims,
        int num_heads,
        faiss::MetricType metric,
        float metricArg,
        std::vector<idx_t> nlists, 
        GpuIndexMultiHeadIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          // Initialize IndexIVFInterface with head 0's quantizer (once available) and nlist_per_head
          IndexIVFInterface(nullptr, nlists[0]),
          ivfConfig_(config),
          num_heads_(num_heads),
          nlists_(nlists),
          own_coarse_quantizers_(true) {
    FAISS_THROW_IF_NOT_MSG(num_heads_ > 0, "num_heads must be > 0");
    quantizers_.resize(num_heads_);
    nprobes_.resize(num_heads_, 1); // Default nprobe to 1 for each head
    this->nprobe = 1; // Sync IndexIVFInterface::nprobe
    ntotals_.resize(num_heads_, 0); // Initialize ntotals for each head

    if (!(metric_type == faiss::METRIC_L2 ||
          metric_type == faiss::METRIC_INNER_PRODUCT)) {
        FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
    }
    init_();
    // this->quantizer = quantizers_[0]; // Set IndexIVFInterface quantizer
}

GpuIndexMultiHeadIVF::GpuIndexMultiHeadIVF(
        GpuResourcesProvider* provider,
        std::vector<Index*> coarseQuantizers,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        std::vector<idx_t> nlists,
        GpuIndexMultiHeadIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          IndexIVFInterface(coarseQuantizers.empty() ? nullptr : coarseQuantizers[0], nlists[0]),
          ivfConfig_(config),
          num_heads_(coarseQuantizers.size()),
          nlists_(nlists),
          quantizers_(std::move(coarseQuantizers)),
          own_coarse_quantizers_(false) {
    FAISS_THROW_IF_NOT_MSG(num_heads_ > 0, "Must provide at least one coarse quantizer.");
    FAISS_THROW_IF_NOT_MSG(
            !quantizers_.empty() && quantizers_[0] != nullptr,
            "expecting valid coarse quantizer objects; none or null provided for head 0");
    nprobes_.resize(num_heads_, 1); // Default nprobe to 1 for each head
    this->nprobe = 1; // Sync IndexIVFInterface::nprobe
    ntotals_.resize(num_heads_, 0);

    if (!(metric_type == faiss::METRIC_L2 ||
          metric_type == faiss::METRIC_INNER_PRODUCT)) {
        FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
    }
    init_();
    // this->quantizer already set by IndexIVFInterface constructor
}

void GpuIndexMultiHeadIVF::init_() {
    FAISS_THROW_IF_NOT_MSG(nlists_[0] > 0, "nlist_per_head must be > 0");

    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
    cp.niter = 10;
    cp.verbose = verbose;

    is_trained = true; // Tentatively true, verifyIVFSettings_ will check
    for (int h = 0; h < num_heads_; ++h) {
        if (quantizers_[h]) {
            if (!(quantizers_[h]->is_trained && quantizers_[h]->ntotal == nlists_[h])) {
                is_trained = false;
            }
        } else {
            if (own_coarse_quantizers_) {
                GpuIndexFlatConfig flat_config = ivfConfig_.flatConfig;
                flat_config.device = config_.device;
                flat_config.use_cuvs = config_.use_cuvs;

                if (metric_type == faiss::METRIC_L2) {
                    quantizers_[h] = new GpuIndexFlatL2(resources_, d, flat_config);
                } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
                    quantizers_[h] = new GpuIndexFlatIP(resources_, d, flat_config);
                } else {
                    FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
                }
            } else {
                 FAISS_THROW_FMT("Coarse quantizer for head %d is null and own_coarse_quantizers_ is false.", h);
            }
            is_trained = false; // Needs training if any quantizer was just created
            quantizers_[h]->is_trained = false; 
        }
    }

    verifyIVFSettings_();
}

GpuIndexMultiHeadIVF::~GpuIndexMultiHeadIVF() {
    if (own_coarse_quantizers_) {
        for (Index* q : quantizers_) {
            delete q;
        }
    }
}

void GpuIndexMultiHeadIVF::verifyIVFSettings_() const {
    for (int h = 0; h < num_heads_; ++h) {
        FAISS_THROW_IF_NOT_FMT(quantizers_[h] != nullptr, "Coarse quantizer for head %d is null.", h);
        FAISS_THROW_IF_NOT_FMT(d == quantizers_[h]->d, "Dimension mismatch for head %d's quantizer.", h);

        if (quantizers_[h]->is_trained) { // If individual CQ is trained
             FAISS_THROW_IF_NOT_FMT(
                quantizers_[h]->ntotal == nlists_[h],
                "Head %d: IVF nlist_per_head count (%ld) does not match trained coarse quantizer size (%ld)",
                h, nlists_[h], quantizers_[h]->ntotal);
        }
        // If this->is_trained is true, all quantizers must be trained and match nlist_per_head.
        // If this->is_trained is false, individual quantizers might or might not be trained.
        // ntotal == 0 check is also important for overall trained status.

        auto gpuQuantizer = tryCastGpuIndex(quantizers_[h]);
        if (gpuQuantizer && gpuQuantizer->getDevice() != getDevice()) {
            FAISS_THROW_FMT(
                    "GpuIndexMultiHeadIVF: head %d coarse quantizer resident on different GPU (%d) "
                    "than the GpuIndexMultiHeadIVF (%d)",
                    h, gpuQuantizer->getDevice(), getDevice());
        }
    }
    if (is_trained) { // Overall trained status
        for (int h = 0; h < num_heads_; ++h) {
            FAISS_THROW_IF_NOT_FMT(quantizers_[h]->is_trained, "If index is trained, coarse quantizer for head %d must be trained.", h);
            FAISS_THROW_IF_NOT_FMT(quantizers_[h]->ntotal == nlists_[h], "If index is trained, coarse quantizer for head %d ntotal must match nlist_per_head.", h);
        }
    } else {
        FAISS_THROW_IF_NOT_FMT(ntotal == 0, "If index is not trained, ntotal should be 0. Found %ld", ntotal);
    }
}

void GpuIndexMultiHeadIVF::copyFrom(const faiss::IndexIVF* indices) {
    DeviceScope scope(config_.device);
    GpuIndex::copyFrom(indices); // Copies d, metric, ntotal, is_trained, verbose

    std::cerr << num_heads_ << std::endl;
    for (int h = 0; h < num_heads_; h ++) {
        std::cerr << (indices + h) -> ntotal << std::endl;
    }

    for (int h = 0; h < num_heads_; ++h) {
        ntotals_[h] = (indices + h)->ntotal; // Assuming all heads share the same ntotal
        nlists_[h] = (indices + h)->nlist; // Assuming all heads share the same nlist
        validateNProbe((indices + h)->nprobe); // Validate nprobe for each head
        nprobes_[h] = (indices + h)->nprobe; // Assuming all heads share the same nprobe
    }

    if (own_fields) {
        quantizers_.resize(num_heads_);
    }
    
    for (int h = 0; h < num_heads_; ++h) {
        FAISS_THROW_IF_NOT((indices + h)->quantizer);

        if (!isGpuIndex((indices + h)->quantizer)) {
            // The coarse quantizer used in the IndexIVF is non-GPU.
            // If it is something that we support on the GPU, we wish to copy it
            // over to the GPU, on the same device that we are on.
            GpuResourcesProviderFromInstance pfi(getResources());
    
            // Attempt to clone the index to GPU. If it fails because the coarse
            // quantizer is not implemented on GPU and the flag to allow CPU
            // fallback is set, retry it with CPU cloner and re-throw errors.
            try {
                GpuClonerOptions options;
                auto cloner = ToGpuCloner(&pfi, getDevice(), options);
                quantizers_[h] = cloner.clone_Index((indices + h)->quantizer);
            } catch (const std::exception& e) {
                if (strstr(e.what(), "not implemented on GPU")) {
                    if (ivfConfig_.allowCpuCoarseQuantizer) {
                        Cloner cpuCloner;
                        quantizer = cpuCloner.clone_Index((indices + h)->quantizer);
                    } else {
                        FAISS_THROW_MSG(
                                "This index type is not implemented on "
                                "GPU and allowCpuCoarseQuantizer is set to false. "
                                "Please set the flag to true to allow the CPU "
                                "fallback in cloning.");
                    }
                } else {
                    throw;
                }
            }
            own_fields = true;
        } else {
            // Otherwise, this is a GPU coarse quantizer index instance found in a
            // CPU instance. It is unclear what we should do here, but for now we'll
            // flag this as an error (we're expecting a pure CPU index)
            FAISS_THROW_MSG(
                    "GpuIndexIVF::copyFrom: copying a CPU IVF index to GPU "
                    "that already contains a GPU coarse (level 1) quantizer "
                    "is not currently supported");
        }
    
        // Validate equality
        FAISS_ASSERT(ntotals_[h] == (indices + h)->ntotal);
        FAISS_ASSERT(nlists_[h] == (indices + h)->nlist);
        FAISS_ASSERT(quantizers_[h]->is_trained == (indices + h)->quantizer->is_trained);
        FAISS_ASSERT(quantizers_[h]->ntotal == (indices + h)->quantizer->ntotal);
    }

    verifyIVFSettings_(); // Validate settings after copying
}

void GpuIndexMultiHeadIVF::copyTo(faiss::IndexIVF* index) const {
    DeviceScope scope(config_.device);
    GpuIndex::copyTo(index);

    index->nlist = nlists_[0];
    index->nprobe = nprobes_[0]; // Copy nprobe of head 0

    FAISS_ASSERT(!quantizers_.empty() && quantizers_[0]);
    if (index->own_fields) {
        delete index->quantizer;
    }
    index->quantizer = index_gpu_to_cpu(quantizers_[0]); // Copy head 0's quantizer
    FAISS_THROW_IF_NOT(index->quantizer);

    FAISS_ASSERT(
            index->quantizer->is_trained == quantizers_[0]->is_trained);
    FAISS_ASSERT(index->quantizer->ntotal == quantizers_[0]->ntotal);

    index->own_fields = true;
    index->quantizer_trains_alone = 0; // Default
    index->cp = this->cp; // Copy clustering params

    // Copying lists from multiHeadBaseIndex_ to index->invlists
    if (multiHeadBaseIndex_ && index->invlists) {
        // This would typically involve clearing index->invlists and then
        // populating it from multiHeadBaseIndex_ (e.g., from head 0).
        // multiHeadBaseIndex_->copyInvertedListsTo(0, *index->invlists); // Hypothetical
        // For now, this is complex and depends on MultiHeadIVFBase.
        // The derived GpuIndex<Product/Flat>IVF's copyTo handles this.
    } else if (multiHeadBaseIndex_ && !index->invlists) {
        // Create and populate invlists
    }
    index->make_direct_map(false); // Default
}


void GpuIndexMultiHeadIVF::updateQuantizer() {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(multiHeadBaseIndex_);
    multiHeadBaseIndex_->updateQuantizer(quantizers_);
}

int GpuIndexMultiHeadIVF::getNumHeads() const {
    return num_heads_;
}

idx_t GpuIndexMultiHeadIVF::getNumListsPerHead() const {
    return nlists_[0];
}

const std::vector<Index*>& GpuIndexMultiHeadIVF::getCoarseQuantizers() const {
    return quantizers_;
}

Index* GpuIndexMultiHeadIVF::getCoarseQuantizer(int headId) const {
    FAISS_THROW_IF_NOT(headId >= 0 && headId < num_heads_);
    return quantizers_[headId];
}


idx_t GpuIndexMultiHeadIVF::getListLength(int headId, idx_t listIdInHead) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(multiHeadBaseIndex_);
    return multiHeadBaseIndex_->getListLength(headId, listIdInHead);
}

std::vector<uint8_t> GpuIndexMultiHeadIVF::getListVectorData(
        int headId, idx_t listIdInHead, bool gpuFormat) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(multiHeadBaseIndex_);
    return multiHeadBaseIndex_->getListVectorData(headId, listIdInHead, gpuFormat);
}

std::vector<idx_t> GpuIndexMultiHeadIVF::getListIndices(int headId, idx_t listIdInHead) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(multiHeadBaseIndex_);
    return multiHeadBaseIndex_->getListIndices(headId, listIdInHead);
}

// --- IndexIVFInterface methods implementation ---
idx_t GpuIndexMultiHeadIVF::getNumLists() const {
    return nlists_[0]; // As per IndexIVFInterface, this is 'nlist'
}

idx_t GpuIndexMultiHeadIVF::getListLength(idx_t listId) const {
    // Delegates to head 0 for IndexIVFInterface compatibility
    return getListLength(0, listId);
}

std::vector<uint8_t> GpuIndexMultiHeadIVF::getListVectorData(idx_t listId, bool gpuFormat) const {
    // Delegates to head 0
    return getListVectorData(0, listId, gpuFormat);
}

std::vector<idx_t> GpuIndexMultiHeadIVF::getListIndices(idx_t listId) const {
    // Delegates to head 0
    return getListIndices(0, listId);
}

void GpuIndexMultiHeadIVF::setNProbe(int headId, idx_t nprobe_val) {
    validateNProbe(nprobe_val);
    if (headId == -1) { // Set for all heads
        for (int h = 0; h < num_heads_; ++h) {
            nprobes_[h] = nprobe_val;
        }
        this->nprobe = nprobe_val; // Sync IndexIVFInterface member
    } else {
        FAISS_THROW_IF_NOT(headId >= 0 && headId < num_heads_);
        nprobes_[headId] = nprobe_val;
        if (headId == 0) {
            this->nprobe = nprobe_val; // Sync IndexIVFInterface member
        }
    }
}

idx_t GpuIndexMultiHeadIVF::getNProbe(int headId) const {
    FAISS_THROW_IF_NOT(headId >= 0 && headId < num_heads_);
    return nprobes_[headId];
}


std::vector<int> GpuIndexMultiHeadIVF::getCurrentNProbePerHead_(const SearchParameters* params) const {
    std::vector<int> current_nprobes(num_heads_);
    idx_t base_nprobe_val = this->nprobe; // from IndexIVFInterface, typically nprobe_per_head_[0]

    if (params) {
        auto ivfParams = dynamic_cast<const SearchParametersIVF*>(params);
        if (ivfParams) {
            base_nprobe_val = ivfParams->nprobe;
            FAISS_THROW_IF_NOT_FMT(
                    ivfParams->max_codes == 0,
                    "GPU MultiHeadIVF index does not currently support "
                    "SearchParametersIVF::max_codes (passed %ld, must be 0)",
                    ivfParams->max_codes);
        } else {
            FAISS_THROW_MSG(
                    "GPU MultiHeadIVF index: passed unhandled SearchParameters "
                    "class to search function; only SearchParametersIVF "
                    "implemented at present");
        }
    }

    for(int h=0; h<num_heads_; ++h) {
        // If SearchParametersIVF is given, its nprobe overrides individual settings.
        // Otherwise, use the per-head setting.
        idx_t use_nprobe_h = params ? base_nprobe_val : nprobes_[h];
        validateNProbe(use_nprobe_h);
        current_nprobes[h] = static_cast<int>(use_nprobe_h);
    }
    return current_nprobes;
}


void GpuIndexMultiHeadIVF::addImpl_(idx_t n, const float* x, const idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(multiHeadBaseIndex_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    // Create std::vector<Tensor*> for vecsPerHead and userIndicesPerHead
    // For add, all heads receive the same input data 'x' and 'xids'.
    // The MultiHeadIVFBase::addVectors will then assign them to lists within each head
    // based on that head's coarse quantizer.

    std::vector<Tensor<float, 2, true>> vecsPerHead(num_heads_);
    std::vector<Tensor<idx_t, 1, true>> indicesPerHead(num_heads_);

    // Create temporary DeviceTensors that all point to the same input data.
    // This is okay if MultiHeadIVFBase::addVectors understands this setup or
    // if we create distinct copies (less efficient).
    // For now, assume MultiHeadIVFBase::addVectors can handle shared input if appropriate,
    // or that it internally processes head by head with the same input.
    // A safer approach for addVectors might be to loop here and call a per-head add.
    // However, the current MultiHeadIVFBase::addVectors takes vectors of tensors.

    DeviceTensor<float, 2, true> dataTensor(const_cast<float*>(x), {n, this->d});
    DeviceTensor<idx_t, 1, true> labelsTensor(const_cast<idx_t*>(xids), {n});

    for(int h=0; h<num_heads_; ++h) {
        // These will be non-owning views if dataTensor/labelsTensor are stack-allocated
        // or if they are members. If they are temporary, ensure lifetime.
        // For simplicity, let's assume they are valid for the call.
        // A robust way: create owning DeviceTensors for each head if they need to differ,
        // or ensure the single input is correctly handled.
        // Here, we pass the same tensor pointer for all heads.
        vecsPerHead[h] = dataTensor;
        indicesPerHead[h] = labelsTensor;
    }

    idx_t numAdded = multiHeadBaseIndex_->addVectors(quantizers_, vecsPerHead.data(), indicesPerHead.data());
    // ntotal is based on attempted adds, not necessarily successful ones if addVectors filters.
    // If addVectors returns actual added, use that. GpuIndexIVF uses n.
    this->ntotal += n; // Consistent with GpuIndexIVF
}


void GpuIndexMultiHeadIVF::searchImpl_(
        idx_t n, // number of queries per head
        const float* x, // nhead * n
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Device was already set in GpuIndex::search
    std::vector<int> current_nprobes_per_head = getCurrentNProbePerHead_(params);

    FAISS_ASSERT(is_trained && multiHeadBaseIndex_);
    FAISS_ASSERT(n > 0);

    Tensor<float, 2, true> queries[num_heads_];
    Tensor<float, 2, true> outDistances[num_heads_];
    Tensor<idx_t, 2, true> outLabels[num_heads_];

    for (int h = 0; h < num_heads_; ++h) {
        // Create output tensors for each head
        queries[h] = Tensor<float, 2, true>(const_cast<float*>(x) + h * n * d, {n, d});
        outDistances[h] = Tensor<float, 2, true>(distances + h * n * k, {n, k});
        outLabels[h] = Tensor<idx_t, 2, true>(labels + h * n * k, {n, k});
    }

    std::vector<int> ks(num_heads_, k);

    multiHeadBaseIndex_ -> search (const_cast<std::vector<Index*>&>(quantizers_), queries, current_nprobes_per_head, ks,
                                outDistances, outLabels);
}


void GpuIndexMultiHeadIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign, // global list ids
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const SearchParametersIVF* params, // Note: This is SearchParametersIVF in .h
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(stats == nullptr, "IVF stats not supported for GpuIndexMultiHeadIVF");
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    FAISS_THROW_IF_NOT_MSG(
            !store_pairs,
            "GpuIndexMultiHeadIVF::search_preassigned does not "
            "currently support store_pairs");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "GpuIndexMultiHeadIVF not trained");
    FAISS_ASSERT(multiHeadBaseIndex_);

    validateKSelect(k);

    if (n == 0 || k == 0) {
        return;
    }

    std::vector<int> current_nprobes_per_head = getCurrentNProbePerHead_(params);
    // For search_preassigned, nprobe is implicitly defined by the shape of `assign` and `centroid_dis`.
    // Let's assume `assign` is (n, total_probes_for_query) where total_probes_for_query
    // might be different from nprobe_per_head_[h] * num_heads.
    // The `MultiHeadIVFBase::searchPreassigned` needs to handle this.
    // The `assign` tensor contains global list IDs.

    // Ensure that all data/output buffers are resident on our desired device
    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, const_cast<float*>(x), stream, {n, d});

    // The `assign` and `centroid_dis` are (n, num_probes_per_query_total).
    // These are global. MultiHeadIVFBase::searchPreassigned needs to handle this.
    // The number of probes here is determined by the input 'assign' tensor's second dimension.
    // Let's assume it's passed as a single tensor and MultiHeadIVFBase::searchPreassigned
    // can use it directly or it's processed per head if needed.
    // The current MultiHeadIVFBase::searchPreassigned takes vectors of tensors.
    // This implies inputs should be split per head if they are head-specific.
    // However, `assign` contains global list IDs, so it's not inherently per-head.

    // Simplification: Assume `assign` and `centroid_dis` are passed as single tensors,
    // and `multiHeadBaseIndex_->searchPreassigned` can handle them, possibly by
    // filtering relevant lists for each head.
    // This part of the API mapping is complex.

    // For now, let's assume the MultiHeadIVFBase::searchPreassigned expects
    // per-head assignments if it takes std::vector<Tensor*>.
    // If `assign` is global, it needs to be processed.
    // This API is hard to map directly if MultiHeadIVFBase expects per-head assignments.

    // Let's assume `multiHeadBaseIndex_->searchPreassigned` is adapted to take global assignments
    // or we only support this for a single effective head in this top-level API.
    // Given the MultiHeadIVFBase.cuh signature for searchPreassigned:
    // (..., const std::vector<Tensor<idx_t, 2, true>*>& ivfAssignmentsPerHead, ...)
    // This means `assign` (global) needs to be split/filtered into per-head local assignments.
    // This is non-trivial.

    // --- TEMPORARY SIMPLIFICATION: This function is hard to map directly ---
    // A full implementation would require logic to distribute global assignments to per-head
    // local assignments, or MultiHeadIVFBase::searchPreassigned would need to take global assignments.
    // For now, this will likely throw or be incorrect without further adaptation.
    FAISS_THROW_MSG("GpuIndexMultiHeadIVF::search_preassigned with global assignments to per-head structure is complex and not fully implemented in this sketch.");


    // Placeholder for what would be needed if it were to proceed (highly conceptual):
    // 1. Convert global `assign` list IDs to per-head local list IDs and filter.
    // 2. Prepare per-head Tensors for `ivfAssignmentsPerHead` and `ivfDistancesPerHead`.
    // 3. Call `multiHeadBaseIndex_->searchPreassigned`.
    // 4. Merge results from `outDistancesPerHead` and `outLabelsPerHead`.
}


void GpuIndexMultiHeadIVF::range_search_preassigned(
        idx_t nx, const float* x, float radius,
        const idx_t* keys, const float* coarse_dis,
        RangeSearchResult* result, bool store_pairs,
        const IVFSearchParameters* params, IndexIVFStats* stats) const {
    FAISS_THROW_MSG("range_search_preassigned not implemented for GpuIndexMultiHeadIVF");
}

bool GpuIndexMultiHeadIVF::addImplRequiresIDs_() const {
    return true; // IVF indices generally store IDs
}

void GpuIndexMultiHeadIVF::trainQuantizers_(idx_t n, const float* x) {
    DeviceScope scope(config_.device);
    if (n == 0) return;

    bool needs_any_training = false;
    for (int h = 0; h < num_heads_; ++h) {
        if (!quantizers_[h]->is_trained || quantizers_[h]->ntotal != nlists_[h]) {
            needs_any_training = true;
            break;
        }
    }
    if (!needs_any_training) {
        if (verbose) printf("MultiHeadIVF quantizers do not need training.\n");
        return;
    }

    if (verbose) {
        printf("Training MultiHeadIVF quantizers on %ld vectors in %dD for %d heads, nlist_per_head=%ld\n",
               n, d, num_heads_, nlists_[0]);
    }

    for (int h = 0; h < num_heads_; ++h) {
        if (quantizers_[h]->is_trained && quantizers_[h]->ntotal == nlists_[h]) {
            if (verbose) printf("Head %d quantizer already trained.\n", h);
            continue;
        }

        if (verbose) printf("Training quantizer for head %d...\n", h);
        quantizers_[h]->reset();

        // Each head's quantizer is trained on the same data `x`.
        // Alternatively, one might partition `x` or use different data per head.
        Clustering clus(this->d, nlists_[h], this->cp);
        clus.verbose = verbose; // Use GpuIndexMultiHeadIVF's verbose flag
        clus.train(n, x, *quantizers_[h]);

        quantizers_[h]->is_trained = true;
        FAISS_ASSERT(quantizers_[h]->ntotal == nlists_[h]);
    }
    // After training all quantizers, the overall index can be considered trained
    // if ntotal is also > 0 (i.e., data has been added).
    // GpuIndex::train handles setting this->is_trained based on ntotal.
    // Here, we ensure individual quantizers are marked.
    // The overall `this->is_trained` will be true if all quantizers are trained
    // and ntotal > 0 (or if ntotal == 0, it's "trained" in terms of CQ but empty).
    // GpuIndex::train will set this->is_trained = true after this function.
}


} // namespace gpu
} // namespace faiss