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

/********************************************************/
/*          quamba2_conv1d_channellast_fwd_kernel       */
/********************************************************/


template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kIsVecLoad_, typename input_t_, typename weight_t_, int x_headdim_>
struct quamba2_conv1d_channellast_fwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int x_headdim = x_headdim_;
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 32 == 0);
    static constexpr int kNWarps = kNThreads / 32;
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    // static_assert(kNBytes == 2 || kNBytes == 4);
    static_assert(kNBytes == 1);
    // static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // static constexpr int kNElts = 8;
    static constexpr int kNElts = 16; // 8 (bits) * 1 (bytes) * 16 (NElts) = 128 bits
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // Always 8 for now
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
    static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;  // Always 4 for now
    static_assert(kNColsPerWarp * kNThreadsPerRow == 32);
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
};



template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void quamba2_conv1d_channellast_fwd_kernel(Quamba2ConvParams params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    // constexpr int kNWarp = Ktraits::kNWarps;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    constexpr int kHeadInChunk = kChunkSizeC / Ktraits::x_headdim;

    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // output settings
    int x_dim = params.x_dim;
    int x_headdim = params.x_headdim;
    int d_state = params.d_state;
    int n_groups = params.n_groups;
    // int n_head = x_dim / x_headdim;

    // Shared memory.
    __shared__ input_t xBC_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts]; // int8

    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    const int th_load_dim = chunk_c_id * kChunkSizeC + c_idx * kNElts;
    // xBC
    input_t *xBC = reinterpret_cast<input_t *>(params.xBC_ptr) + batch_id * params.xBC_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.xBC_l_stride + th_load_dim;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t xBC_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen && th_load_dim < params.dim) {
            reinterpret_cast<vec_t *>(xBC_vals_load)[0] = *reinterpret_cast<vec_t *>(xBC + l * kLPerLoad * params.xBC_l_stride);
        }
        reinterpret_cast<vec_t *>(xBC_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(xBC_vals_load)[0];

        // // Just for debugging
        // // https://forums.developer.nvidia.com/t/vector-load-int4-veca1-reinterpret-cast-int4-a-2-0-valid/112136/4
        // // https://stackoverflow.com/questions/71360564/cuda-misaligned-address-for-a-reused-shared-block-memory
        // // https://forums.developer.nvidia.com/t/compiling-with-debug-flag-gives-errors-while-normal-compilation-work-well/27604/7
        // input_t xBC_vals_load[kNElts] = {0};
        // if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen && th_load_dim < params.dim) {
        //     #pragma unroll
        //     for (int i = 0; i < kNElts; ++i) {
        //         xBC_vals_load[i] = xBC[l * kLPerLoad * params.xBC_l_stride + i];
        //     }
        // }
        // #pragma unroll
        // for (int i = 0; i < kNElts; ++i) {
        //     xBC_smem[kWidth - 1 + l * kLPerLoad + l_idx][c_idx * kNElts + i] = xBC_vals_load[i];
        // }
    }
    // Load the elements from the previous chunk that are needed for convolution.
    if (l_idx < kWidth - 1) {
        input_t xBC_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && th_load_dim < params.dim) {
            reinterpret_cast<vec_t *>(xBC_vals_load)[0] = *reinterpret_cast<vec_t *>(xBC - (kWidth - 1) * params.xBC_l_stride);
        }
        reinterpret_cast<vec_t *>(xBC_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(xBC_vals_load)[0];

        // // Just for debugging
        // // https://forums.developer.nvidia.com/t/vector-load-int4-veca1-reinterpret-cast-int4-a-2-0-valid/112136/4
        // // https://stackoverflow.com/questions/71360564/cuda-misaligned-address-for-a-reused-shared-block-memory
        // // https://forums.developer.nvidia.com/t/compiling-with-debug-flag-gives-errors-while-normal-compilation-work-well/27604/7
        // input_t xBC_vals_load[kNElts] = {0};
        // if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
        //     && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
        //     && th_load_dim < params.dim) {
        //     #pragma unroll
        //     for (int i = 0; i < kNElts; ++i) {
        //         xBC_vals_load[i] = *(xBC - (kWidth - 1) * params.xBC_l_stride + i);
        //     }
        // }
        // #pragma unroll
        // for (int i = 0; i < kNElts; ++i) {
        //     xBC_smem[l_idx][c_idx * kNElts + i] = xBC_vals_load[i];
        // }
    }

    __syncthreads();

    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    // kChunkSizeL, kLPerThread, kNThreadsPerRow should be powers of 2 for simplicity
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 32);

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;

    float scale_in = 0;
    float scale_bias = 0;
    float scale_out = 0;

    if (chunk_c_id * kChunkSizeC + row_idx < x_dim) {
        // x_in
        scale_bias = params.scale_bx;
        scale_in = params.scale_wx * params.scale_x;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.x_scales_ptr);
        } else {
            const int group_size = x_dim / n_groups;
            const int group_idx = (chunk_c_id * kChunkSizeC + row_idx) / group_size;
            const int head_idx = ((chunk_c_id * kChunkSizeC + row_idx) % group_size) / x_headdim;
            const int dim_idx = (chunk_c_id * kChunkSizeC + row_idx) % x_headdim;
            float *x_scales = reinterpret_cast<float *>(params.x_scales_ptr) + group_idx*params.x_nhead_group*params.x_ndim_group;  // [n_ssd_groups, n_head_groups, n_dim_groups]
            int *x_head_group_range = reinterpret_cast<int *>(params.x_head_group_range_ptr) + group_idx*params.x_nhead_group;  // [n_ssd_groups, n_head_groups]
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
    } else if (x_dim <= chunk_c_id * kChunkSizeC + row_idx && chunk_c_id * kChunkSizeC + row_idx < x_dim + d_state*n_groups) {
        // B_in
        scale_bias = params.scale_bB;
        scale_in = params.scale_wB * params.scale_B;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.B_scales_ptr);
        } else {
            int group_idx = (chunk_c_id * kChunkSizeC + row_idx - x_dim) / d_state;
            scale_out = *(reinterpret_cast<float *>(params.B_scales_ptr) + group_idx);
        }
    } else if (x_dim + d_state*n_groups <= chunk_c_id * kChunkSizeC + row_idx && chunk_c_id * kChunkSizeC + row_idx < x_dim + 2*d_state*n_groups) {
        // C_in
        scale_bias = params.scale_bC;
        scale_in = params.scale_wC * params.scale_C;
        if (params.x_head_group_range_ptr == nullptr && params.x_dim_group_range_ptr == nullptr) {
            scale_out = *reinterpret_cast<float *>(params.C_scales_ptr);
        } else {
            int group_idx = (chunk_c_id * kChunkSizeC + row_idx - x_dim - d_state*n_groups) / d_state;
            scale_out = *(reinterpret_cast<float *>(params.C_scales_ptr) + group_idx);
        }
    }

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = float(weight[row_idx * params.weight_c_stride + w * params.weight_width_stride]);
        }
    }
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = static_cast<float>(xBC_smem[col_idx * kLPerThread + i][row_idx]);
    }

    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        float tmp = 0;
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            tmp += weight_vals[w] * x_vals[i + w];
        }
        out_vals[i] = scale_in * tmp + scale_bias * bias_val;
        if (params.silu_activation) {out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i])); }
    }

    __syncthreads();

    int out_dim = chunk_c_id * kChunkSizeC + row_idx;
    if (out_dim < x_dim) {
        // write int8 x to shared memory
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            int tmp = int(roundf(out_vals[i] / scale_out));
            xBC_smem[col_idx * kLPerThread + i][row_idx] =  tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<input_t>(tmp);
        }
    } else if (x_dim <= out_dim && out_dim < x_dim + d_state*n_groups) {
        // write int8 B to shared memory
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            int tmp = int(roundf(out_vals[i] / scale_out));
            xBC_smem[col_idx * kLPerThread + i][row_idx] =  tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<input_t>(tmp);
        }
    } else if (x_dim + d_state*n_groups <= out_dim && out_dim < x_dim + 2*d_state*n_groups) {
        // write int8 C to shared memory 
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            int tmp = int(roundf(out_vals[i] / scale_out));
            xBC_smem[col_idx * kLPerThread + i][row_idx] =  tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<input_t>(tmp);
        }
    }

    __syncthreads();
    
    input_t *out = nullptr;
    int out_l_stride = 0;
    if (th_load_dim < x_dim) {
        // int8 output pointer
        out = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
            + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride
            + th_load_dim;
        out_l_stride = params.x_l_stride;
        #pragma unroll
        for (int l = 0; l < Ktraits::kNLoads; ++l) {
            input_t out_vals_store[kNElts];
            reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(xBC_smem[l * kLPerLoad + l_idx])[c_idx];
            // #pragma unroll
            // for (int i = 0; i < kNElts; ++i) {
            //     out_vals_store[i] = xBC_smem[l * kLPerLoad + l_idx][c_idx * kNElts + i];
            // }
            if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                *reinterpret_cast<vec_t *>(out + l * kLPerLoad * out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
                // #pragma unroll
                // for (int i = 0; i < kNElts; ++i) {
                //     out[l * kLPerLoad * out_l_stride + i] = out_vals_store[i];
                // }
            }
        }
    } else if (x_dim <= th_load_dim && th_load_dim < x_dim + d_state*n_groups) {
        // B_out
        out = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride
            + (chunk_l_id * kChunkSizeL + l_idx) * params.B_l_stride
            + th_load_dim - x_dim;
        out_l_stride = params.B_l_stride;
        #pragma unroll
        for (int l = 0; l < Ktraits::kNLoads; ++l) {
            input_t out_vals_store[kNElts];
            reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(xBC_smem[l * kLPerLoad + l_idx])[c_idx];
            // #pragma unroll
            // for (int i = 0; i < kNElts; ++i) {
            //     out_vals_store[i] = xBC_smem[l * kLPerLoad + l_idx][c_idx * kNElts + i];
            // }
            if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                *reinterpret_cast<vec_t *>(out + l * kLPerLoad * out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
                // #pragma unroll
                // for (int i = 0; i < kNElts; ++i) {
                //     out[l * kLPerLoad * out_l_stride + i] = out_vals_store[i];
                // }
            }
        }
    } else if (x_dim + d_state*n_groups <= th_load_dim && th_load_dim < x_dim + 2*d_state*n_groups) {
        // C_out
        out = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride
            + (chunk_l_id * kChunkSizeL + l_idx) * params.C_l_stride
            + th_load_dim - x_dim - d_state*n_groups;
        out_l_stride = params.C_l_stride;
        #pragma unroll
        for (int l = 0; l < Ktraits::kNLoads; ++l) {
            input_t out_vals_store[kNElts];
            reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(xBC_smem[l * kLPerLoad + l_idx])[c_idx];
            // #pragma unroll
            // for (int i = 0; i < kNElts; ++i) {
            //     out_vals_store[i] = xBC_smem[l * kLPerLoad + l_idx][c_idx * kNElts + i];
            // }
            if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                *reinterpret_cast<vec_t *>(out + l * kLPerLoad * out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
                // #pragma unroll
                // for (int i = 0; i < kNElts; ++i) {
                //     out[l * kLPerLoad * out_l_stride + i] = out_vals_store[i];
                // }
            }         
        }
    }
}


template<int kNThreads, int kWidth, typename input_t, typename weight_t, int x_headdim>
void quamba2_conv1d_channellast_fwd_launch(Quamba2ConvParams &params, cudaStream_t stream) {
    using Ktraits = quamba2_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64, true, input_t, weight_t, x_headdim>;
    // constexpr int kSmemSize = Ktraits::kSmemSize;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
    dim3 block(Ktraits::kNThreads);
    auto kernel = &quamba2_conv1d_channellast_fwd_kernel<Ktraits>;
    // if (kSmemSize >= 48 * 1024) {
    //     C10_CUDA_CHECK(cudaFuncSetAttribute(
    //         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    //     }
    // kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void quamba2_conv1d_channellast_fwd_cuda(Quamba2ConvParams &params, cudaStream_t stream) {
    if (params.width == 2) {
        throw std::logic_error("No implementation for conv kernel width == 2");
    } else if (params.width == 3) {
        throw std::logic_error("No implementation for conv kernel width == 3");
    } else if (params.width == 4) {
        quamba2_conv1d_channellast_fwd_launch<128, 4, input_t, weight_t, 64>(params, stream);
    }
}

