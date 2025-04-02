#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>


// Helper function for error checking
#define CHECK_CUDA(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using Half2VecT = Vec<half2, 2>;


// CUDA kernel for embedding lookup
__global__ void w8a16_embedding_lookup_cuda(
    const int4* __restrict__ embedding_ptr,  // Embedding matrix
    const half* __restrict__ scales_ptr,     // scaling factors
    const int64_t* __restrict__ tokens_ptr,      // Input tokens
    int4* __restrict__ output_ptr,           // Output vectors
    int vocab_size, int embedding_dim,
    int batch_size, int seqlen) {

    int batch_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int num_threads = blockDim.x;

    int embedding_token_stride = embedding_dim / 16;
    int output_batch_stride = seqlen * embedding_dim / 8;
    int output_token_stride = embedding_dim / 8;

    if (token_idx < seqlen) {
        int embedding_idx = tokens_ptr[batch_idx * seqlen + token_idx];
        half scale = scales_ptr[embedding_idx];
        const int4* embedding_token_ptr = embedding_ptr + embedding_idx * embedding_token_stride;
        int4* output_token_ptr = output_ptr + batch_idx * output_batch_stride + token_idx * output_token_stride;
        for (int d = threadIdx.x; d < embedding_dim / 16; d+=num_threads) {
            int4 x_packed = *(embedding_token_ptr + d); // 4*4 8-bit int
            unsigned int* x = reinterpret_cast<unsigned int*>(&x_packed);
            int4* output = output_token_ptr + 2*d;      // 16 fp16 = 2 int4
            __half x_fp16[2][8];
            #pragma unroll
            for (int i=0; i<4; i++) {
                x_fp16[i/2][i%2*4 + 0] = scale * __int2half_rn(static_cast<int>((int8_t)((x[i] >>  0) & 0xFF)));
                x_fp16[i/2][i%2*4 + 1] = scale * __int2half_rn(static_cast<int>((int8_t)((x[i] >>  8) & 0xFF)));
                x_fp16[i/2][i%2*4 + 2] = scale * __int2half_rn(static_cast<int>((int8_t)((x[i] >> 16) & 0xFF)));
                x_fp16[i/2][i%2*4 + 3] = scale * __int2half_rn(static_cast<int>((int8_t)((x[i] >> 24) & 0xFF)));                
            }
            output[0] = *reinterpret_cast<int4*>(x_fp16[0]); // 8 half = 1 int4
            output[1] = *reinterpret_cast<int4*>(x_fp16[1]); // 8 half = 1 int4
        }
    }
}


torch::Tensor w8a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale) {

    TORCH_CHECK(weight.dim() == 2);
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "weight's dtype must be int8");
    int vocab_size = weight.size(0);
    int embed_dim = weight.size(1);

    TORCH_CHECK(scale.dim() == 1);
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(scale.dtype() == torch::kFloat16, "scale's dtype must be float16");
    TORCH_CHECK(scale.numel() == vocab_size);

    TORCH_CHECK(input.dim() == 1 || input.dim() == 2);
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input's dtype must be int64");

    torch::Tensor out;
    int batch_size, seqlen;
    auto options_out = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());
    if (input.dim() == 1) {
        batch_size = 1;
        seqlen = input.size(0);
        out = torch::empty({seqlen, embed_dim}, options_out);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
        seqlen = input.size(1);
        out = torch::empty({batch_size, seqlen, embed_dim}, options_out);
    } else {
        TORCH_CHECK(false, "input must be a 2D or 3D tensor");
    }

    // get grid and block
    int threads_per_block = 128; // 4 warps per block
    int blocks_per_grid = (seqlen + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch_size, seqlen);
    // get stream
    int dev = input.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
    // Launch kernel
    w8a16_embedding_lookup_cuda<<<grid, threads_per_block, 0, stream>>>(
        (const int4*) weight.data_ptr(), (const half*) scale.data_ptr(),
        (const int64_t*) input.data_ptr(), (int4*) out.data_ptr(),
        vocab_size, embed_dim, batch_size, seqlen);
    // cudaDeviceSynchronize(); // comment this for cuda graph

    return out;
}


// Efficiently dequantize an int32 value into 4 fp16 values.
__device__ inline Half2VecT dequant(int q, half scale) {
  static constexpr uint32_t LO = 0x000f000f;
  static constexpr uint32_t HI = 0x00f000f0;
  static constexpr uint32_t EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  uint32_t t0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  uint32_t t1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  static constexpr uint32_t SUB = 0x64086408;
  static constexpr uint32_t MUL = 0x2c002c00;
  static constexpr uint32_t ADD = 0xd480d480;
  *reinterpret_cast<half2*>(&t0) = __hsub2(
      *reinterpret_cast<half2*>(&t0), *reinterpret_cast<const half2*>(&SUB));
  *reinterpret_cast<half2*>(&t1) = __hfma2(
      *reinterpret_cast<half2*>(&t1), *reinterpret_cast<const half2*>(&MUL),
      *reinterpret_cast<const half2*>(&ADD));

  uint16_t s = *reinterpret_cast<uint16_t*>(&scale);
  uint32_t double_s;
  // pack half s to half2
  asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(double_s) : "h"(s), "h"(s));
  // dequant and convert 4 half to 4 uint8 (be placed at the low 8 bits of 4
  // half, respectively)
  Half2VecT half2vec;
  half2vec[0] = __hmul2(
      *reinterpret_cast<half2*>(&t0), *reinterpret_cast<half2*>(&double_s));
  half2vec[1] = __hmul2(
      *reinterpret_cast<half2*>(&t1), *reinterpret_cast<half2*>(&double_s));
  return half2vec;
}


// CUDA kernel for embedding lookup
__global__ void w4a16_embedding_lookup_cuda(
    const int4* __restrict__ embedding_ptr, // Embedding matrix
    const half* __restrict__ scales_ptr,    // scaling factors
    const int64_t* __restrict__ tokens_ptr, // Input tokens
    int4* __restrict__ output_ptr,          // Output vectors
    int vocab_size, int embedding_dim,
    int batch_size, int seqlen) {

    int batch_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int num_threads = blockDim.x;

    int embedding_token_stride = embedding_dim / 32;
    int output_batch_stride = seqlen * embedding_dim / 8;
    int output_token_stride = embedding_dim / 8;

    if (token_idx < seqlen) {
        int embedding_idx = tokens_ptr[batch_idx * seqlen + token_idx];
        half scale = scales_ptr[embedding_idx];
        const int4* embedding_token_ptr = embedding_ptr + embedding_idx * embedding_token_stride;
        int4* output_token_ptr = output_ptr + batch_idx * output_batch_stride + token_idx * output_token_stride;
        for (int d = threadIdx.x; d < embedding_dim / 32; d+=num_threads) {
            int4 x_packed = *(embedding_token_ptr + d); // 4*8 4-bit int
            int4* output = output_token_ptr + 4*d;      // 32 fp16 = 4 int4
            Half2VecT x_fp16[4][2]; // 4*2*2 half2 = (4*2*2)*2 half = 32 half
            unsigned int* x = reinterpret_cast<unsigned int*>(&x_packed);
            #pragma unroll
            for (int i=0; i<4; i++) {
              x_fp16[i][0] = dequant(x[i], scale);       // 2 half2 = 4 half
              x_fp16[i][1] = dequant(x[i] >> 8, scale);  // 2 half2 = 4 half
              output[i] = *reinterpret_cast<int4*>(x_fp16[i]); // 8 half = 1 int4
            }
        }
    }
}


torch::Tensor w4a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale) {

    static constexpr int pack_factor = 32 / 4;
    TORCH_CHECK(weight.dim() == 2);
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kInt32, "weight's dtype must be int32");
    int vocab_size = weight.size(0);
    int embed_dim = weight.size(1) * pack_factor;

    TORCH_CHECK(scale.dim() == 1);
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(scale.dtype() == torch::kFloat16, "scale's dtype must be float16");
    TORCH_CHECK(scale.numel() == vocab_size);

    TORCH_CHECK(input.dim() == 1 || input.dim() == 2);
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input's dtype must be int64");

    torch::Tensor out;
    int batch_size, seqlen;
    auto options_out = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());
    if (input.dim() == 1) {
        batch_size = 1;
        seqlen = input.size(0);
        out = torch::empty({seqlen, embed_dim}, options_out);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
        seqlen = input.size(1);
        out = torch::empty({batch_size, seqlen, embed_dim}, options_out);
    } else {
        TORCH_CHECK(false, "input must be a 2D or 3D tensor");
    }

    // get grid and block
    int threads_per_block = 128; // 4 warps per block
    int blocks_per_grid = (seqlen + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch_size, seqlen);
    // get stream
    int dev = input.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
    // Launch kernel
    w4a16_embedding_lookup_cuda<<<grid, threads_per_block, 0, stream>>>(
        (const int4*) weight.data_ptr(), (const half*) scale.data_ptr(),
        (const int64_t*) input.data_ptr(), (int4*) out.data_ptr(),
        vocab_size, embed_dim, batch_size, seqlen);
    // CHECK_CUDA(cudaDeviceSynchronize()); // comment this for cuda graph

    return out;
}
