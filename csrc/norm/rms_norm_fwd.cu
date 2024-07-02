#include "rms_norm_fwd_kernel.cuh"

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
    double scale_out // per-tensor scaling factor
);