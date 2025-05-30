/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexIVF.h> // for SearchParametersIVF
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

class MultiHeadIVFBase; // Forward declaration

// Configuration struct remains the same as in the prompt
struct GpuIndexMultiHeadIVFConfig : public GpuIndexConfig {
    IndicesOptions indicesOptions = INDICES_64_BIT;
    GpuIndexFlatConfig flatConfig;
    bool allowCpuCoarseQuantizer = false;
};

class GpuIndexMultiHeadIVF : public GpuIndex, public IndexIVFInterface {
   public:
    GpuIndexMultiHeadIVF(
            GpuResourcesProvider* provider,
            int dims,
            int num_heads, // Added num_heads
            faiss::MetricType metric,
            float metricArg,
            std::vector<idx_t> nlists, 
            GpuIndexMultiHeadIVFConfig config = GpuIndexMultiHeadIVFConfig());

    GpuIndexMultiHeadIVF(
            GpuResourcesProvider* provider,
            std::vector<Index*> coarseQuantizers, // Takes a vector of quantizers
            int dims,
            // num_heads is implicit from coarseQuantizers.size()
            faiss::MetricType metric,
            float metricArg,
            std::vector<idx_t> nlists, 
            GpuIndexMultiHeadIVFConfig config = GpuIndexMultiHeadIVFConfig());

    ~GpuIndexMultiHeadIVF() override;

   private:
    void init_();

   public:
    // copyFrom/To for single IndexIVF - behavior needs careful definition (e.g., replicates or uses head 0)
    virtual void copyFrom(const faiss::IndexIVF* index);
    virtual void copyTo(faiss::IndexIVF* index) const;

    // Updates the MultiHeadIVFBase with the current state of all coarse quantizers
    virtual void updateCoarseQuantizers();

    // --- Methods for multi-head access ---
    int getNumHeads() const;
    idx_t getNumListsPerHead() const; // nlist for each head (assuming uniform)
    const std::vector<Index*>& getCoarseQuantizers() const;
    Index* getCoarseQuantizer(int headId) const;

    virtual idx_t getListLength(int headId, idx_t listIdInHead) const;
    virtual std::vector<uint8_t> getListVectorData(
            int headId,
            idx_t listIdInHead,
            bool gpuFormat = false) const;
    virtual std::vector<idx_t> getListIndices(int headId, idx_t listIdInHead) const;

    // --- IndexIVFInterface methods (mostly operate on head 0 or require adaptation) ---
    // quantizer, nlist, nprobe are inherited from IndexIVFInterface
    // GpuIndexMultiHeadIVF will set them based on head 0 or overall config.

    // Returns nlist_per_head_
    virtual idx_t getNumLists() const;

    // Returns getListLength(0, listId)
    virtual idx_t getListLength(idx_t listId) const;

    // Returns getListVectorData(0, listId, gpuFormat)
    virtual std::vector<uint8_t> getListVectorData(
            idx_t listId,
            bool gpuFormat = false) const;

    // Returns getListIndices(0, listId)
    virtual std::vector<idx_t> getListIndices(idx_t listId) const;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign, // Global list ids expected
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const SearchParametersIVF* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    /// Set nprobe for a specific head, or all if headId is -1
    virtual void setNProbe(int headId, idx_t nprobe);
    /// Get nprobe for a specific head
    virtual idx_t getNProbe(int headId) const;


   protected:
    std::vector<int> getCurrentNProbePerHead_(const SearchParameters* params) const;
    void verifyIVFSettings_() const; // Verifies all coarse quantizers
    bool addImplRequiresIDs_() const override;
    virtual void trainQuantizers_(idx_t n, const float* x); // Trains all coarse quantizers

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   protected:
    const GpuIndexMultiHeadIVFConfig ivfConfig_;
    int num_heads_;
    std::vector<idx_t> nlists_; // nlist for each head

    std::vector<Index*> quantizers_; // One coarse quantizer per head
    bool own_coarse_quantizers_; // True if we allocated them

    std::vector<idx_t> nprobes_; // nprobe for each head

    // For a trained/initialized index, this is a reference to the base class
    std::shared_ptr<MultiHeadIVFBase> multiHeadBaseIndex_;
};

} // namespace gpu
} // namespace faiss
