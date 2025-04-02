// Modified by Hung-Yueh Chiang
// The code is adpated from https://github.com/Bruce-Lee-LY/cuda_hgemv

#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert> // For assert

#include "marlin_utils.h" // str()

namespace QuambaGEMV {

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void W8A8O8GEMVKernel(
    int8_t *__restrict__ C,
    const int8_t *__restrict__ A, const float *__restrict__ s_a,
    const int8_t *__restrict__ B, const float *__restrict__ s_b,
    size_t N, size_t K) {

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (warp_col >= N) {
        return;
    }

    const size_t K_iters = div_ceil(K, WARP_SIZE);
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    int tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * WARP_SIZE + lane_id + warp_col * K;
        size_t B_idx = i * WARP_SIZE + lane_id;
        tmp +=  A[A_idx] * B[B_idx];
    }

    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    float scale = s_a[warp_col] * s_b[0];
    if (lane_id == 0) {
        float output = roundf(scale * static_cast<float>(tmp));
        C[warp_col] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, output)));
    }
}


void W8A8O8GEMV(void *C, const void *A, const void* s_a,
    const void *B, const void *s_b, int64_t N, int64_t K,
    int dev = 0, cudaStream_t stream = 0) {

    const int8_t* A_ptr = (const int8_t*)A;
    const float* s_a_ptr = (const float*)s_a;
    const int8_t* B_ptr = (const int8_t*)B;
    const float* s_b_ptr = (const float*)s_b;
    int8_t* C_ptr = (int8_t*)C;

    // Set the device
    cudaSetDevice(dev);
    // Get device attributes
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, 
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    // Ensure the device supports shared memory requirements
    assert(max_shared_mem > 0);
    // Configure grid and block dimensions
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK));
    // Launch kernel with specified stream
    W8A8O8GEMVKernel<<<grid, block, 0, stream>>>(
       C_ptr,  A_ptr, s_a_ptr, B_ptr, s_b_ptr, N, K);
}

} // end namespace QuambaGEMV


torch::Tensor w8a8o8_gemv(
    torch::Tensor & c,              // output vector with shape (1, n)
    torch::Tensor const& a,         // weight matrix with shape (n, k)
    torch::Tensor const& b,         // input vector with shape (1, k)
    torch::Tensor const& scale_a,   // weight scale with shape (n,)
    torch::Tensor const& scale_b    // input scale with shape (1,)
) {
  // Checks for conformality
  // a: w, b: x, c: y => y = wx
  int64_t N = a.size(0);
  int64_t K = a.size(1);
  TORCH_CHECK(scale_a.numel() == N,
              "Shape mismatch: scale_b.numel() = " + str(scale_a.numel()) +
                  ", size_n = " + str(N));
  TORCH_CHECK(scale_b.numel() == 1,
              "Shape mismatch: scale_a.numel() = " + str(scale_b.numel()));
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == b.size(0) && a.size(1) == b.size(1) &&
              a.size(0) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(1) == 1);                      // Row-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(0) % 16 == 0);  // 16 Byte Alignment

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  // GEMV
  int dev = a.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  QuambaGEMV::W8A8O8GEMV(
    c.data_ptr(),
    a.data_ptr(), scale_a.data_ptr(),
    b.data_ptr(), scale_b.data_ptr(),
    N, K, dev, stream
  );

  return c;
}


