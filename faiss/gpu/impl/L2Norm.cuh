/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runL2Norm(
        Tensor<float, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        cudaStream_t stream);

void runL2Norm(
        Tensor<half, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        cudaStream_t stream);

void runL2Norm(
        Tensor<__nv_bfloat16, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
