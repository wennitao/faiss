/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/MetricType.h>
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

    const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y));
    // TEST ONLY!!!
    // const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y), 8);

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
        const int nhead, 
        Tensor<float, 2, true>* queries,
        Tensor<idx_t, 2, true>* listIds,
        DeviceVector<void*>* listData,
        DeviceVector<void*>* listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>* listLengths,
        const int k,
        SUB_METRIC_TYPE metric,
        const bool useResidual,
        Tensor<float, 3, true>* residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>* outDistances,
        Tensor<idx_t, 2, true>* outIndices,
        GpuResources* res) {
    const auto nq = queries -> getSize(0);
    const auto dim = queries -> getSize(1);
    const auto nprobe = listIds -> getSize(1);

    const auto stream = res->getDefaultStreamCurrentDevice();

    // DeviceTensor<float, 3, true> distanceTemp(
    //         res,
    //         makeTempAlloc(AllocType::Other, stream),
    //         {queries -> getSize(0), listIds -> getSize(1), k});
    // DeviceTensor<idx_t, 3, true> indicesTemp(
    //         res,
    //         makeTempAlloc(AllocType::Other, stream),
    //         {queries -> getSize(0), listIds -> getSize(1), k});

    DeviceTensor<float, 3, true> distanceTemp[nhead] ;
    DeviceTensor<idx_t, 3, true> indicesTemp[nhead] ;

    for (int h = 0; h < nhead; h ++) {
        distanceTemp[h] = DeviceTensor<float, 3, true> (
                res, 
                makeTempAlloc(AllocType::Other, stream),
                {queries -> getSize(0), listIds -> getSize(1), k});
    }

    for (int h = 0; h < nhead; h ++) {
        indicesTemp[h] = DeviceTensor<idx_t, 3, true> (
                res, 
                makeTempAlloc(AllocType::Other, stream),
                {queries -> getSize(0), listIds -> getSize(1), k});
    }

    Tensor<float, 2, true>* devQueries ;
    Tensor<float, 3, true>* devResidualBase ;
    DeviceTensor<idx_t, 2, true>* devListIds ;

    DeviceTensor<float, 3, true>* devDistanceTemp ;
    DeviceTensor<idx_t, 3, true>* devIndicesTemp ;

    DeviceTensor<float, 2, true>* devOutDistances ;
    DeviceTensor<idx_t, 2, true>* devOutIndices ;

    cudaMalloc((void**)&devQueries, nhead * sizeof(Tensor<float, 2, true>));
    cudaMalloc((void**)&devResidualBase, nhead * sizeof(Tensor<float, 3, true>));
    cudaMalloc((void**)&devListIds, nhead * sizeof(DeviceTensor<idx_t, 2, true>));

    cudaMalloc((void**)&devDistanceTemp, nhead * sizeof(DeviceTensor<float, 3, true>));
    cudaMalloc((void**)&devIndicesTemp, nhead * sizeof(DeviceTensor<idx_t, 3, true>));

    cudaMalloc((void**)&devOutDistances, nhead * sizeof(DeviceTensor<float, 2, true>));
    cudaMalloc((void**)&devOutIndices, nhead * sizeof(DeviceTensor<idx_t, 2, true>));

    cudaMemcpy(devQueries, queries, nhead * sizeof(Tensor<float, 2, true>), cudaMemcpyHostToDevice);
    cudaMemcpy(devResidualBase, residualBase, nhead * sizeof(Tensor<float, 3, true>), cudaMemcpyHostToDevice);
    cudaMemcpy(devListIds, listIds, nhead * sizeof(DeviceTensor<idx_t, 2, true>), cudaMemcpyHostToDevice);

    cudaMemcpy(devDistanceTemp, distanceTemp, nhead * sizeof(DeviceTensor<float, 3, true>), cudaMemcpyHostToDevice);
    cudaMemcpy(devIndicesTemp, indicesTemp, nhead * sizeof(DeviceTensor<idx_t, 3, true>), cudaMemcpyHostToDevice);

    cudaMemcpy(devOutDistances, outDistances, nhead * sizeof(DeviceTensor<float, 2, true>), cudaMemcpyHostToDevice);
    cudaMemcpy(devOutIndices, outIndices, nhead * sizeof(DeviceTensor<idx_t, 2, true>), cudaMemcpyHostToDevice);

    void*** listData_ptr = new void**[nhead];
    idx_t** listLengths_ptr = new idx_t*[nhead];
    for (int h = 0; h < nhead; h ++) {
        listData_ptr[h] = listData[h].data();
        listLengths_ptr[h] = listLengths[h].data();
    }

    void*** devListData; 
    idx_t** devListLengths;

    cudaMalloc((void**)&devListData, nhead * sizeof(void**));
    cudaMalloc((void**)&devListLengths, nhead * sizeof(idx_t*));
    cudaMemcpy(devListData, listData_ptr, nhead * sizeof(void**), cudaMemcpyHostToDevice);
    cudaMemcpy(devListLengths, listLengths_ptr, nhead * sizeof(idx_t*), cudaMemcpyHostToDevice);

    const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y), nhead);
    // const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y));

    multiHeadIvfInterleavedScan<
            SUB_CODEC_TYPE,
            SUB_METRIC_TYPE,
            SUB_THREADS,
            SUB_NUM_WARP_Q,
            SUB_NUM_THREAD_Q>
            <<<grid, SUB_THREADS, codec.getSmemSize(dim), stream>>>(
                    devQueries,
                    devResidualBase,
                    devListIds,
                    devListData,
                    devListLengths,
                    codec,
                    metric,
                    k,
                    devDistanceTemp,
                    devIndicesTemp,
                    useResidual);

    std::cerr << "debug: multiHeadIvfInterleavedScan kernel launched" << std::endl;
    for (int h = 0; h < nhead; h ++) {
        std::cerr << "head " << h << std::endl ;
        auto distanceTemp_vector = distanceTemp[h].copyToVector(stream);
        auto indicesTemp_vector = indicesTemp[h].copyToVector(stream);
        for (int i = 0; i < distanceTemp_vector.size(); i++) {
            std::cerr << distanceTemp_vector[i] << " ";
        }
        std::cerr << std::endl;
        for (int i = 0; i < indicesTemp_vector.size(); i++) {
            std::cerr << indicesTemp_vector[i] << " ";
        }
        std::cerr << std::endl;
    }

    runMultiHeadIVFInterleavedScan2(
            nhead, 
            nq, 
            devDistanceTemp,
            devIndicesTemp,
            devListIds,
            k,
            listIndices,
            indicesOptions,
            SUB_METRIC_TYPE::kDirection,
            devOutDistances,
            devOutIndices,
            stream);

    cudaMemcpy(outDistances, devOutDistances, nhead * sizeof(DeviceTensor<float, 2, true>), cudaMemcpyDeviceToHost);
    cudaMemcpy(outIndices, devOutIndices, nhead * sizeof(DeviceTensor<idx_t, 2, true>), cudaMemcpyDeviceToHost);
    
    cudaFree(devQueries);
    cudaFree(devResidualBase);
    cudaFree(devListIds);
    cudaFree(devDistanceTemp);
    cudaFree(devIndicesTemp);
    cudaFree(devOutDistances);
    cudaFree(devOutIndices);
    cudaFree(devListData);
    cudaFree(devListLengths);

    delete[] listData_ptr;
    delete[] listLengths_ptr;
}

} // namespace gpu
} // namespace faiss
