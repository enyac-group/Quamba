/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "common/static_switch.h"

template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void quant_causal_conv1d_update_kernel(QuantConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    // Scaling factors
    float scale_x = params.scale_x;
    float scale_w = params.scale_w;
    float scale_b = params.scale_b;
    float scale_out = params.scale_out;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr) + batch_id * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr || channel_id >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);
    bias_val *= scale_b; // dequant

    float weight_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }
    }

    float x_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = float(conv_state[(i + 1) * params.conv_state_l_stride]); }
        x_vals[kWidth - 1] = float(x[0]);
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { conv_state[i * params.conv_state_l_stride] = input_t(x_vals[i]); }
    }

    float out_val = bias_val;
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { out_val += scale_w * scale_x * weight_vals[i] * x_vals[i]; /*dequant*/ }
    if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
    if (channel_id < params.dim) {
        out_val = roundf(out_val / scale_out);
        out[0] =  out_val > 127 ? 127 : out_val < -128 ? -128 : static_cast<input_t>(out_val);
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void quant_causal_conv1d_update_launch(QuantConvParamsBase &params, cudaStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = &quant_causal_conv1d_update_kernel<Ktraits>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void quant_causal_conv1d_update_cuda(QuantConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        quant_causal_conv1d_update_launch<64, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        quant_causal_conv1d_update_launch<64, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        quant_causal_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
    }
}
