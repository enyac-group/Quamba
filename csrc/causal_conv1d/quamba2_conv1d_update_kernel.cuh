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
struct quamba2_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void quamba2_conv1d_update_kernel(Quamba2ConvParams params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    // input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
    //     + channel_id * params.x_c_stride;
    // xBC
    input_t *xBC = reinterpret_cast<input_t *>(params.xBC_ptr) + batch_id * params.xBC_batch_stride
        + channel_id * params.xBC_c_stride;
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr) + batch_id * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    float bias_val = params.bias_ptr == nullptr || channel_id >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    float weight_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }
    }

    float xBC_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { xBC_vals[i] = float(conv_state[(i + 1) * params.conv_state_l_stride]); }
        xBC_vals[kWidth - 1] = float(xBC[0]);
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { conv_state[i * params.conv_state_l_stride] = input_t(xBC_vals[i]); }
    }

    // output settings
    int x_dim = params.x_dim;
    int x_headdim = params.x_headdim;
    int d_state = params.d_state;
    int n_groups = params.n_groups;

    input_t *out = nullptr;
    // Scaling factors
    float scale_in = 0;
    float scale_bias = 0;
    float scale_out = 0;
    if (channel_id < x_dim) {
        // int8 output pointer
        out = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
            + channel_id * params.x_c_stride;
        // x_in
        scale_bias = params.scale_bx;
        scale_in = params.scale_wx * params.scale_x;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.x_scales_ptr);
        } else {
            // get scale_out for x
            const int group_size = x_dim / n_groups;
            const int group_idx = channel_id / group_size;
            const int head_idx = (channel_id % group_size) / x_headdim;
            const int dim_idx = channel_id % x_headdim;
            float *x_scales = reinterpret_cast<float *>(params.x_scales_ptr) + group_idx*params.x_nhead_group*params.x_ndim_group;   // [n_ssd_groups, n_head_groups, n_dim_groups]
            int *x_head_group_range = reinterpret_cast<int *>(params.x_head_group_range_ptr) + group_idx*params.x_nhead_group;   // [n_ssd_groups, n_head_groups]
            int *x_dim_group_range = reinterpret_cast<int *>(params.x_dim_group_range_ptr) + group_idx*params.x_nhead_group*params.x_ndim_group;    // [n_ssd_groups, n_head_groups, n_dim_groups]
            // get head group, x_head_group_range = [hg1, hg1+hg2, hg1+hg2+hg3, ..., num_head]
            int h_start = 0;
            for (int hg_idx = 0; hg_idx < params.x_nhead_group; hg_idx++) {
                if (h_start <= head_idx && head_idx < x_head_group_range[hg_idx]) {
                    // get dim group, x_dim_group_range = [dg1, dg1+dg2, dg1+dg2+dg3, ..., headdim]
                    int ch_start = 0;
                    for (int dg_idx = 0; dg_idx < params.x_ndim_group; dg_idx++) {
                        if (ch_start <= dim_idx && dim_idx < x_dim_group_range[hg_idx * params.x_ndim_group + dg_idx]) {
                            scale_out = x_scales[hg_idx * params.x_ndim_group + dg_idx]; // <--- get scale_out for x and break
                            break;
                        }
                        ch_start = x_dim_group_range[hg_idx * params.x_ndim_group + dg_idx];
                    }
                    break;
                }
                h_start = x_head_group_range[hg_idx];
            }
        }
    } else if (x_dim <= channel_id && channel_id < x_dim + d_state*n_groups) {
        // int8 output pointer
        out = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride
            + (channel_id - x_dim) * params.B_c_stride;
        // B_in
        scale_bias = params.scale_bB;
        scale_in = params.scale_wB * params.scale_B;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.B_scales_ptr);
        } else {
            int group_idx = (channel_id - x_dim) / d_state;
            scale_out = *(reinterpret_cast<float *>(params.B_scales_ptr) + group_idx);
        }
    } else if (x_dim + d_state*n_groups <= channel_id && channel_id < x_dim + 2*d_state*n_groups) {
        // int8 output pointer
        out = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride
            + (channel_id - x_dim - d_state*n_groups) * params.C_c_stride;
        // C_in
        scale_bias = params.scale_bC;
        scale_in = params.scale_wC * params.scale_C;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.C_scales_ptr);
        } else {
            int group_idx = (channel_id - x_dim - d_state*n_groups) / d_state;
            scale_out = *(reinterpret_cast<float *>(params.C_scales_ptr) + group_idx);
        }
    } else {
        assert(false); // should not reach here
    }

    float out_val = scale_bias*bias_val; // dequant
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { out_val += scale_in * weight_vals[i] * xBC_vals[i]; /*dequant*/ }
    if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }     
    if (channel_id < params.dim) {
        out_val = roundf(out_val / scale_out);
        out[0] =  out_val > 127 ? 127 : out_val < -128 ? -128 : static_cast<input_t>(out_val);
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void quamba2_conv1d_update_launch(Quamba2ConvParams &params, cudaStream_t stream) {
    using Ktraits = quamba2_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = &quamba2_conv1d_update_kernel<Ktraits>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void quamba2_conv1d_update_cuda(Quamba2ConvParams &params, cudaStream_t stream) {
    if (params.width == 2) {
        throw std::logic_error("No implementation for conv kernel width == 2");
    } else if (params.width == 3) {
        throw std::logic_error("No implementation for conv kernel width == 2");
    } else if (params.width == 4) {
        quamba2_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
    }
}
