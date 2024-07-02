/*
The code is modfied from
https://github.com/state-spaces/mamba
*/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "quant_sscan.h"
#include "quant_sscan_common.h"
#include "common/static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Quant_SScan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1);
    static constexpr int kNElts = std::min(8, kNItems); // not sure should be changed for int8
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsInt8 = std::is_same_v<weight_t, int8_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using out_vec_t = typename BytesToType<2 * kNElts>::Type; // output float16
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    // output is fp16
    using BlockStoreT = cub::BlockStore<at::Half, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<out_vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void quant_sscan_fwd_kernel(QuantSSMParams params) {
    // constexpr bool kIsInt8 = Ktraits::kIsInt8; // no used by now
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows; // Only kNRows == 1 is tested for now
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    // load quantization scaling factors
    if (params.D_ptr != nullptr) {
        #pragma unroll
        const float scale_D = *reinterpret_cast<float *>(params.scale_D_ptr); // scale_D: (1)
        for (int r = 0; r < kNRows; ++r) {
            weight_t D_load = reinterpret_cast<weight_t *>(params.D_ptr)[dim_id * kNRows + r];
            D_val[r] = scale_D * static_cast<float>(D_load); //  TODO (HY) need to dequant D_val
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        const float scale_delta_bias = *reinterpret_cast<float *>(params.scale_delta_bias_ptr); // scale_delta_bias: (1)
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            weight_t delta_bias_load = reinterpret_cast<weight_t *>(params.delta_bias_ptr)[dim_id * kNRows + r];
            delta_bias[r] = scale_delta_bias * static_cast<float>(delta_bias_load); //  TODO (HY) need to dequant delta_bias
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    const float scale_delta = *reinterpret_cast<float *>(params.scale_delta_ptr); // scale_delta: (1)
    const float scale_u = *reinterpret_cast<float *>(params.scale_u_ptr); // scale_u: (1)
    const float scale_A = *reinterpret_cast<float *>(params.scale_A_ptr); // scale_A: (1)
    const float scale_B = *reinterpret_cast<float *>(params.scale_B_ptr); // scale_B: (1)
    const float scale_C = *reinterpret_cast<float *>(params.scale_C_ptr); // scale_C: (1)
    const float scale_out = *reinterpret_cast<float *>(params.scale_out_ptr); // scale_out: (1)
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals_load[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = scale_u * static_cast<float>(u_vals_load[r][i]); //  TODO (HY) need to dequant u_val
                delta_vals[r][i] = scale_delta * static_cast<float>(delta_vals_load[r][i]) + delta_bias[r]; //  TODO (HY) need to dequant delta_vals_load
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            float A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                // A = -torch.exp(self.A_log.float())  # (d_inner, d_state) # let's do this in the cuda kernel, so we can load int Log_A
                A_val[r] = -expf(scale_A * static_cast<float>(A[state_idx * params.A_dstate_stride + r * params.A_d_stride])); //  TODO (HY) need to dequant A_val? Not very sure for A
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                // constexpr float kLog2e = M_LOG2E;
                // A_val[r] *= kLog2e;
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            float BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = scale_C * static_cast<float>(C[state_idx * params.C_dstate_stride + r * params.C_d_stride]);
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = scale_B * static_cast<float>(B[state_idx * params.C_dstate_stride + r * params.C_d_stride]);
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = scale_B * static_cast<float>(B[state_idx * params.B_dstate_stride + r * params.B_d_stride])
                        * scale_C * static_cast<float>(C[state_idx * params.C_dstate_stride + r * params.C_d_stride]);
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    //  TODO (HY) need to dequant delta_u_vals, B_vals
                    thread_data[i] = make_float2(expf(delta_vals[r][i] * A_val[r]),
                                                     !kIsVariableB ? delta_u_vals[r][i] : 
                                                        scale_B * static_cast<float>(B_vals[i]) * delta_u_vals[r][i]);
                    if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);
                Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<float>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    //  TODO (HY) need to dequant C_vals
                    const float C_val = !kIsVariableC ? BC_val[r] :
                                                (!kIsVariableB ? BC_val[r] * scale_C * static_cast<float>(C_vals[i]) : 
                                                    scale_C * static_cast<float>(C_vals[i]));
                    out_vals[r][i] += thread_data[i].y * C_val;
                } // endfor i kNItems
            } // endfor r kNRows
        } // endfor state_idx params.dstate

        at::Half *out = reinterpret_cast<at::Half *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();

        at::Half out_q_vals[kNRows][kNItems];
        if constexpr (kHasZ) {
            const float scale_z = *reinterpret_cast<float *>(params.scale_z_ptr); // scale_z: (1)
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals_load[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
                //  TODO (HY) need to dequant z_vals_load and quant out_vals
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = scale_z * static_cast<float>(z_vals_load[i]);
                    out_q_vals[r][i] =  __float2half(out_vals[r][i] * z_val / (1 + expf(-z_val)));
                }
                __syncthreads();
                store_output<Ktraits>(out + r * params.out_d_stride, out_q_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        } else {
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if constexpr (!kDirectIO) {
                    if (r > 0) { __syncthreads(); }
                }
                //  TODO (HY) need to quant out_vals
                for (int i = 0; i < kNItems; ++i) {
                    out_q_vals[r][i] =  __float2half(out_vals[r][i]);
                }
                store_output<Ktraits>(out + r * params.out_d_stride, out_q_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize;
        Cvar += kChunkSize;
    } // endfor chunk params.n_chunks
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void quant_sscan_fwd_launch(QuantSSMParams &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                    using Ktraits = Quant_SScan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    // constexpr int kSmemSize = Ktraits::kSmemSize;
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    // printf("smem_size = %d\n", kSmemSize);
                    dim3 grid(params.batch, params.dim / kNRows);
                    auto kernel = &quant_sscan_fwd_kernel<Ktraits>;
                    if (kSmemSize >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    }
                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename input_t, typename weight_t>
void quant_sscan_fwd_cuda(QuantSSMParams &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        quant_sscan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 256) {
        quant_sscan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 512) {
        quant_sscan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        quant_sscan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        quant_sscan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
}
