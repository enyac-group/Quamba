/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "common/static_switch.h"

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    // static_assert(kNBytes == 2 || kNBytes == 4);
    static_assert(kNBytes == 1);
    // static constexpr int kNElts = kNBytes == 4 ? 4 : 8; // 8 (bits) * 4 (bytes) * 4 (NElts) = 128 bits; 8 (bits) * 2 (bytes) * 8 (NElts) = 128 bits 
    // static constexpr int kNElts = 16; // 8 (bits) * 1 (bytes) * 16 (NElts) = 128 bits
    static constexpr int kNElts = 8; // 8 (bits) * 1 (bytes) * 16 (NElts) = 128 bits
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : std::max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void quant_causal_conv1d_fwd_kernel(QuantConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Scaling factors
    float scale_x = params.scale_x;
    float scale_w = params.scale_w;
    float scale_b = params.scale_b;
    float scale_out = params.scale_out;
    float scale_wx = scale_w * scale_x;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);
    bias_val *= scale_b; // dequant

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0) {
        input_t zeros[kNElts] = {0};
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(zeros)[0];
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            Ktraits::BlockLoadVecT(smem_load_vec).Load(
                reinterpret_cast<vec_t*>(x),
                *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]),
                (params.seqlen - chunk * kChunkSize) / kNElts
            );
        } else {
            __syncthreads();
            Ktraits::BlockLoadT(smem_load).Load(
                x,
                *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]),
                params.seqlen - chunk * kChunkSize
            );
        }
        x += kChunkSize;
        __syncthreads();
        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }
        __syncthreads();
        reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx == kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }

        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }

        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            // out_vals[i] = bias_val;
            float out_tmp = 0;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                // out_vals[i] += scale_w * scale_x * weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)]; // dequant
                out_tmp += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)]; // dequant
            }
            out_vals[i] = scale_wx*out_tmp + bias_val;
        }

        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            int tmp = int(roundf(out_vals[i] / scale_out));
            out_vals_store[i] =  tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<input_t>(tmp);
        }
        if constexpr(kIsVecLoad) {
            Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t (&)[1]>(out_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, params.seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void quant_causal_conv1d_fwd_launch(QuantConvParamsBase &params, cudaStream_t stream) {
    // static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;  // not sure should be changed for int8
    static constexpr int kNElts = 8;
    BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
        using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);
        auto kernel = &quant_causal_conv1d_fwd_kernel<Ktraits>;
        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void quant_causal_conv1d_fwd_cuda(QuantConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        if (params.seqlen <= 256) {
            quant_causal_conv1d_fwd_launch<32, 2, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            quant_causal_conv1d_fwd_launch<64, 2, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            quant_causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 2048) {
            quant_causal_conv1d_fwd_launch<256, 2, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 4096){
            quant_causal_conv1d_fwd_launch<512, 2, input_t, weight_t>(params, stream);
        } else {
            quant_causal_conv1d_fwd_launch<1024, 2, input_t, weight_t>(params, stream);
        }
    } else if (params.width == 3) {
        if (params.seqlen <= 256) {
            quant_causal_conv1d_fwd_launch<32, 3, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            quant_causal_conv1d_fwd_launch<64, 3, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            quant_causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 2048) {
            quant_causal_conv1d_fwd_launch<256, 3, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 4096){
            quant_causal_conv1d_fwd_launch<512, 3, input_t, weight_t>(params, stream);
        } else {
            quant_causal_conv1d_fwd_launch<1024, 3, input_t, weight_t>(params, stream);
        }
    } else if (params.width == 4) {
        if (params.seqlen <= 256) {
            quant_causal_conv1d_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            quant_causal_conv1d_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            quant_causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 2048) {
            quant_causal_conv1d_fwd_launch<256, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 4096){
            quant_causal_conv1d_fwd_launch<512, 4, input_t, weight_t>(params, stream);
        } else {
            quant_causal_conv1d_fwd_launch<1024, 4, input_t, weight_t>(params, stream);
        }
    }
}
