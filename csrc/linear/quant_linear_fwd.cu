#include "quant_linear_fwd_kernel.cuh"

// GEMM
void cutlass_scaled_mm_dq_sm80(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales);
// GEMV
void cutlass_scaled_mv_dq_sm80(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               float const& alpha,
                               float const& beta);