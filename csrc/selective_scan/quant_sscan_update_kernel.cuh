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

template<int kNThreads_, int kNItems_, bool kHasZ_, typename input_t_, typename weight_t_>
struct Quant_SScan_update_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    // static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3; // Not sure what is this
    static constexpr int kNItems = kNItems_;
    // static constexpr int kNBytes = sizeof(input_t);
    // static_assert(kNBytes == 1);
    // static constexpr int kNElts = std::min(8, kNItems); // not sure should be changed for int8
    // static_assert(kNItems % kNElts == 0);
    // static constexpr int kNLoads = kNItems / kNElts;
    // static constexpr bool kHasZ = kHasZ_;

    // static constexpr bool kDirectIO = kNLoads == 1;

    // using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
    //     !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    // using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
    //     !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    // using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
    //     !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
    //                                              sizeof(typename BlockLoadVecT::TempStorage),
    //                                              sizeof(typename BlockStoreT::TempStorage),
    //                                              sizeof(typename BlockStoreVecT::TempStorage)});
    // static constexpr int kSmemSize = kSmemIOSize;
    static constexpr int kSmemSize = sizeof(typename BlockLoadT::TempStorage);
};

template<typename Ktraits>
__global__
// __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void quant_sscan_update_kernel(QuantSSMParams params) {
    // constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    // constexpr int kNItems = Ktraits::kNItems;
    // constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;


    // Shared memory.
    // extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    // auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    // auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    // auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    // auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);

    const int batch_id = blockIdx.x;
    const int thd_id = threadIdx.x;

    const float scale_delta = *reinterpret_cast<float *>(params.scale_delta_ptr); // scale_delta: (1)
    const float scale_u = *reinterpret_cast<float *>(params.scale_u_ptr); // scale_u: (1)
    const float scale_A = *reinterpret_cast<float *>(params.scale_A_ptr); // scale_A: (1)
    const float scale_B = *reinterpret_cast<float *>(params.scale_B_ptr); // scale_B: (1)
    const float scale_C = *reinterpret_cast<float *>(params.scale_C_ptr); // scale_C: (1)
    const float scale_D = params.D_ptr == nullptr? 0 : *reinterpret_cast<float *>(params.scale_D_ptr); // scale_D: (1)
    const float scale_z = params.z_ptr == nullptr? 0 : *reinterpret_cast<float *>(params.scale_z_ptr); // scale_z: (1)
    const float scale_delta_bias = params.delta_bias_ptr == nullptr? 0 : *reinterpret_cast<float *>(params.scale_delta_bias_ptr); // scale_delta_bias: (1)
    // const float scale_out = *reinterpret_cast<float *>(params.scale_out_ptr); // scale_out: (1)

    // vectorization parameters
    using vecType = float4;
    assert(sizeof(vecType) % sizeof(weight_t) == 0);
    int PackSize = sizeof(vecType) / sizeof(weight_t);
    // printf("%d, %d, %d\n", params.dstate, PackSize, params.dstate % PackSize);
    assert(params.dstate % PackSize == 0);
    int nRead = params.dstate / PackSize;

    /* Load B */
    weight_t* B = reinterpret_cast<weight_t *>(params.B_ptr) + batch_id * params.B_batch_stride; // params.u_d_stride == 1
    weight_t B_load[MAX_DSTATE] = {0};
    for (int  i=0; i < nRead; i++) {
        // vectorization
        vecType tmp = *(reinterpret_cast<vecType*>(B) + i);
        #pragma unroll
        for (int j=0; j<PackSize; j++){
            B_load[i*PackSize + j] = reinterpret_cast<weight_t*>(&tmp)[j];
        }
    } 

    /* Load C */
    weight_t* C = reinterpret_cast<weight_t *>(params.C_ptr) + batch_id * params.C_batch_stride; // params.u_d_stride == 1
    weight_t C_load[MAX_DSTATE] = {0}; // not sure how to use BlockLoad to load params.dstate (dynamic) elements
    for (int  i=0; i < nRead; i++) {
        // vectorization
        vecType tmp = *(reinterpret_cast<vecType*>(C) + i);
        #pragma unroll
        for (int j=0; j<PackSize; j++){
            C_load[i*PackSize + j] = reinterpret_cast<weight_t*>(&tmp)[j];
        }
    }    

    /* input pointers and registers*/
    float u_val, delta_bias_val, delta_val, z_val;
    float x_load[MAX_DSTATE] = {0};   // ssm_state is float type
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride;
    input_t *z = params.z_ptr == nullptr? nullptr : reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride;
    at::Half *out = reinterpret_cast<at::Half *>(params.out_ptr) + batch_id * params.out_batch_stride;
    input_t* delta_bias = params.delta_bias_ptr == nullptr? nullptr : reinterpret_cast<input_t *>(params.delta_bias_ptr);
    /* weight pointers and registers*/
    float D_val;
    weight_t A_load[MAX_DSTATE] = {0};
    weight_t *D = params.D_ptr == nullptr? nullptr : reinterpret_cast<weight_t *>(params.D_ptr);

    int dim_id = thd_id;
    int nChunk = (params.dim + kNThreads) / kNThreads;
    for (int i=0; i<nChunk; i++) {
        float out_val = 0;
        if (dim_id < params.dim) {
            /* Load x !!! FLOAT TYPE !!! */
            float* x = reinterpret_cast<float *>(params.x_ptr) + batch_id * params.x_batch_stride + dim_id * params.x_d_stride;
            for (int  i=0; i < params.dstate / 4; i++) {
                // vectorization
                vecType tmp = *(reinterpret_cast<vecType*>(x) + i);
                #pragma unroll
                for (int j=0; j<4; j++){
                    x_load[i*4 + j] = reinterpret_cast<float*>(&tmp)[j];
                }
            }
            
            /* Load A */
            weight_t* A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
            for (int  i=0; i < nRead; i++) {
                // vectorization
                vecType tmp = *(reinterpret_cast<vecType*>(A) + i);
                #pragma unroll
                for (int j=0; j<PackSize; j++){
                    A_load[i*PackSize + j] = reinterpret_cast<weight_t*>(&tmp)[j];
                }
            }

            /* load u, delta, D*/
            u_val = scale_u * static_cast<float>(u[dim_id]);
            delta_bias_val = params.delta_bias_ptr == nullptr? 0.0 : scale_delta_bias * static_cast<float>(delta_bias[dim_id]);
            delta_val = scale_delta * static_cast<float>(delta[dim_id]) + delta_bias_val;
            if (params.delta_softplus) {
                delta_val = delta_val <= 20.f ? log1pf(expf(delta_val)) : delta_val;
            }
            if (params.D_ptr != nullptr) {
                D_val = scale_D * static_cast<float>(D[dim_id]); // dequant
                out_val += static_cast<float>(D_val * u_val);
            }

            /* ssm update */
            for (int ds=0; ds<params.dstate; ds++) {
                float dA = expf(delta_val * -expf(scale_A * static_cast<float>(A_load[ds])));
                float dB = delta_val * scale_B * static_cast<float>(B_load[ds]);
                x_load[ds] = dA * x_load[ds] + dB * u_val;
                out_val += x_load[ds] * scale_C * static_cast<float>(C_load[ds]);
            }

            /* load z */
            if (params.z_ptr != nullptr) {
                z_val = scale_z * static_cast<float>(z[dim_id]);
                out_val = out_val * z_val / (1 + expf(-z_val));
            }

            /* write y FP16*/
            out[dim_id] =  __float2half(out_val);
            
            /* write x (ssm_state) !!! FLOAT TYPE !!! */
            for (int  i=0; i < params.dstate / 4; i++) {
                // vectorization
                vecType tmp = *(reinterpret_cast<vecType*>(x_load) + i);
                #pragma unroll
                for (int j=0; j<4; j++){
                    x[i*4 + j] = reinterpret_cast<float*>(&tmp)[j];
                }
            }
        }
        dim_id += kNThreads;
    } // endfor i nChunk
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void quant_sscan_update_launch(QuantSSMParams &params, cudaStream_t stream) {
    BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
        using Ktraits = Quant_SScan_update_kernel_traits<kNThreads, kNItems, kHasZ, input_t, weight_t>;
        // constexpr int kSmemSize = Ktraits::kSmemSize;
        constexpr int kSmemSize = Ktraits::kSmemSize + MAX_DSTATE * sizeof(float); // FIXME: sscan_state use half or int?
        // // printf("smem_size = %d\n", kSmemSize);
        auto kernel = &quant_sscan_update_kernel<Ktraits>;
        // if (kSmemSize >= 48 * 1024) {
        //     C10_CUDA_CHECK(cudaFuncSetAttribute(
        //         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        // }
        dim3 gridsize(params.batch);
        dim3 blocksize(Ktraits::kNThreads);
        // kernel<<<gridsize, blocksize, kSmemSize, stream>>>(params);
        kernel<<<gridsize, blocksize, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void quant_sscan_update_cuda(QuantSSMParams &params, cudaStream_t stream) {
    // constexpr int kNItems = 32;
    // const int kNThreads = (params.dim + kNItems) / kNItems;
    // quant_sscan_update_launch<kNThreads, kNItems, input_t, weight_t>(params, stream);
    // quant_sscan_update_launch<32, 32, input_t, weight_t>(params, stream);

    // TODO: remove kNItems
    if (params.dim <= 1024) {
        quant_sscan_update_launch<32, 32, input_t, weight_t>(params, stream);
    } else if (params.dim <= 2048) {
        quant_sscan_update_launch<128, 32, input_t, weight_t>(params, stream);
    } else {
        quant_sscan_update_launch<256, 64, input_t, weight_t>(params, stream);
    }
}
