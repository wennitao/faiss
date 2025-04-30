/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>

namespace faiss {
namespace gpu {

template <>
void IVFINT_RUN<
        SUB_CODEC_TYPE,
        SUB_METRIC_TYPE,
        SUB_THREADS,
        SUB_NUM_WARP_Q,
        SUB_NUM_THREAD_Q>(
        SUB_CODEC_TYPE& codec,
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        const int k,
        SUB_METRIC_TYPE metric,
        const bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    const auto nq = queries.getSize(0);
    const auto dim = queries.getSize(1);
    const auto nprobe = listIds.getSize(1);

    const auto stream = res->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 3, true> distanceTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), listIds.getSize(1), k});
    DeviceTensor<idx_t, 3, true> indicesTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), listIds.getSize(1), k});

    // TEST ONLY!!!
//     const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y));
    const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y), 8);

    ivfInterleavedScan<
            SUB_CODEC_TYPE,
            SUB_METRIC_TYPE,
            SUB_THREADS,
            SUB_NUM_WARP_Q,
            SUB_NUM_THREAD_Q>
            <<<grid, SUB_THREADS, codec.getSmemSize(dim), stream>>>(
                    queries,
                    residualBase,
                    listIds,
                    listData.data(),
                    listLengths.data(),
                    codec,
                    metric,
                    k,
                    distanceTemp,
                    indicesTemp,
                    useResidual);

    runIVFInterleavedScan2(
            distanceTemp,
            indicesTemp,
            listIds,
            k,
            listIndices,
            indicesOptions,
            SUB_METRIC_TYPE::kDirection,
            outDistances,
            outIndices,
            stream);
}

template <>
void multiHeadIVFINT_RUN<
        SUB_CODEC_TYPE,
        SUB_METRIC_TYPE,
        SUB_THREADS,
        SUB_NUM_WARP_Q,
        SUB_NUM_THREAD_Q>(
        SUB_CODEC_TYPE& codec,
        Tensor<float, 3, true>& queries,
        Tensor<idx_t, 3, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        const int k,
        SUB_METRIC_TYPE metric,
        const bool useResidual,
        Tensor<float, 4, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 3, true>& outDistances,
        Tensor<idx_t, 3, true>& outIndices,
        GpuResources* res) {
    const auto nhead = queries.getSize (0);
    const auto nq = queries.getSize(1);
    const auto dim = queries.getSize(2);
    const auto nprobe = listIds.getSize(2);

    const auto stream = res->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 4, true> distanceTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), queries.getSize(1), listIds.getSize(2), k});
    DeviceTensor<idx_t, 4, true> indicesTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), queries.getSize(1), listIds.getSize(1), k});

    const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y), nhead);

    multiHeadIvfInterleavedScan<
            SUB_CODEC_TYPE,
            SUB_METRIC_TYPE,
            SUB_THREADS,
            SUB_NUM_WARP_Q,
            SUB_NUM_THREAD_Q>
            <<<grid, SUB_THREADS, codec.getSmemSize(dim), stream>>>(
                    queries,
                    residualBase,
                    listIds,
                    listData.data(),
                    listLengths.data(),
                    codec,
                    metric,
                    k,
                    distanceTemp,
                    indicesTemp,
                    useResidual);

    runMultiHeadIVFInterleavedScan2(
            distanceTemp,
            indicesTemp,
            listIds,
            k,
            listIndices,
            indicesOptions,
            SUB_METRIC_TYPE::kDirection,
            outDistances,
            outIndices,
            stream);
}

} // namespace gpu
} // namespace faiss
