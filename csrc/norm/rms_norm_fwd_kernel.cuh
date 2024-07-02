/*
The code is from: https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu#L1021
*/

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/type_shim.h"

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename U> __device__
void cuRMSOnlineSum(
  const U curr,
  U& sigma2)
{
  sigma2 = sigma2 + curr * curr;
}

template<typename U> __device__
void cuChanRMSOnlineSum(
  const U sigma2B,
  U& sigma2)
{
  sigma2 = sigma2 + sigma2B;
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1*n2;
    int l = 4*thrx;
    for (;  l+3 < n2;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      } else {
       cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        U muB = WARP_SHFL(mu, srcLaneB);
        U countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2*wrt_y+1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            U muB = ubuf[2*threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B,sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/U(n2), 0);
    }
  }
}


template<> __device__
void cuWelfordMuSigma2(
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu= float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const at::Half* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }

      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float2 curr = __half22float2(*((__half2*)(lvals+l+k)));
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x, mu, sigma2, count);
          cuWelfordOnlineSum(curr.y, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      if (!rms_only) {
        float muB = WARP_SHFL(mu, srcLaneB);
        float countB = WARP_SHFL(count, srcLaneB);
        cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y+1] = sigma2;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            float muB = ubuf[2*threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = WARP_SHFL(mu, 0);
      }
      sigma2 = WARP_SHFL(sigma2/float(n2), 0);
    }
  }
}

template<typename U> U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template<> float rsqrt(float v) {
  return rsqrtf(v);
}
template<> double rsqrt(double v) {
  return rsqrt(v);
}


namespace {

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};

} // end namespace


// T: input type, U: Compute type, V: Output type
template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  const T* __restrict__ input_vals,
  const int n1,
  const int n2,
  const U epsilon,
  const U scale_out,
  const T* __restrict__ gamma,
  const T* __restrict__ beta,
  T* __restrict__ residual_out,
  const T* __restrict__ residual_in,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T *vals, *lvals; // input pointer

    // add residual
    if (residual_in != NULL) {
      T* res_out = residual_out + i1*n2;
      const T* res_in = residual_in + i1*n2;
      const T* in_vals = input_vals + i1*n2;
      #pragma unroll
      for (int i = thrx; i < n2; i+=numx) {
        res_out[i] = res_in[i] + in_vals[i]; // add residual to vals
      }
      __syncthreads();
      vals = residual_out;
      lvals = residual_out + i1*n2;
    } else {
      vals = input_vals;
      lvals = input_vals + i1*n2;
    }

    // compute mu and sigma
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, rms_only);

    // normalized, affine and quantization
    V* out_vals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx; i < n2; i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          U tmp = static_cast<U>(gamma[i]) * c_invvar * (curr - mu) + static_cast<U>(beta[i]);
          tmp = roundf(tmp / scale_out);
          out_vals[i] = tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<V>(tmp);
        } else {
          U tmp = static_cast<U>(gamma[i]) * c_invvar * curr;
          tmp = roundf(tmp / scale_out);
          out_vals[i] = tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<V>(tmp);
        }

      }
    } else {
      for (int i = thrx; i < n2; i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          U tmp = c_invvar * (curr - mu);
          tmp = roundf(tmp / scale_out);
          out_vals[i] = tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<V>(tmp);
        } else {
          U tmp = c_invvar * curr;
          tmp = roundf(tmp / scale_out);
          out_vals[i] = tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<V>(tmp);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
    }
    __syncthreads();
  }
}

// T: input type, U: Compute type, V: Output type
template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNorm(
  V* __restrict__ output_vals,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const U scale_out,
  const T* __restrict__ gamma,
  T* __restrict__ residual_out,
  const T* __restrict__ residual_in)
{
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, vals, n1, n2, epsilon, scale_out, gamma, NULL, residual_out, residual_in, true);
}


// T: input type, U: Compute type, V: Output type
template<typename T, typename U, typename V=T>
void HostApplyRMSNorm(
    V* output,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    double scale_out,
    const T* gamma,
    T* residual_out,
    const T* residual_in)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32, 4, 1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U) + (threads.y/2)*sizeof(U) :
            0;
    cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, input, n1, n2, U(epsilon), U(scale_out), gamma, residual_out, residual_in);
}


void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntList normalized_shape,
    at::Tensor* gamma,
    at::Tensor* residual_out,
    at::Tensor* residual_in,
    double epsilon,
    double scale_out)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_cuda_kernel", [&] {
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->data_ptr<scalar_t_out>(),
          input->data_ptr<scalar_t_in>(),
          n1, n2,
          epsilon,
          scale_out,
          gamma != NULL ? gamma->data_ptr<scalar_t_in>() : NULL,
          residual_out != NULL ? residual_out->data_ptr<scalar_t_in>() : NULL,
          residual_in != NULL ? residual_in->data_ptr<scalar_t_in>() : NULL
          );
      });
}
