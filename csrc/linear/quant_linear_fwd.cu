#include "quant_linear_fwd_kernel.cuh"

void cutlass_scaled_mm_dq(
      torch::Tensor& c, torch::Tensor const& a,
      torch::Tensor const& b, torch::Tensor const& a_scales,
      torch::Tensor const& b_scales
);


void cutlass_scaled_mv_dq(
      torch::Tensor& c, torch::Tensor const& a,
      torch::Tensor const& b, float const& alpha,
      float const& beta
);

#include "quant_linear_fwd_gemv_kernel.cuh"
torch::Tensor w8a8o8_gemv(
    torch::Tensor& c, 
    torch::Tensor const& a,         // input vector with shape (1, k)
    torch::Tensor const& b,         // weight matrix with shape (n, k)
    torch::Tensor const& scale_a,   // input scale with shape (1,)
    torch::Tensor const& scale_b    // weight scale with shape (n,)
);

// W4A16O16
#include "marlin_gemm_kernel.cuh"
torch::Tensor w4a16o16_gemm(
      const torch::Tensor& A,
      const torch::Tensor& B,
      const torch::Tensor& s,
            torch::Tensor& workspace,
      int64_t size_m,
      int64_t size_n,
      int64_t size_k,
      bool output_transpose,
      int thread_k,
      int thread_n,
      int sms,
      int max_par
);

// W4A8O16
#include "marlin_qqq_gemm_kernel.cuh"
torch::Tensor w4a8o16_gemm(
      torch::Tensor const& a,
      torch::Tensor const& b_q_weight,
      torch::Tensor const& s_tok,
      torch::Tensor const& s_ch,
      torch::Tensor const& s_group,
      torch::Tensor& workspace, int64_t size_m,
      int64_t size_n, int64_t size_k,
      bool output_transpose
);

// W4A8O8
#include "marlin_qqq_gemm_kernel_o8.cuh"
torch::Tensor w4a8o8_gemm(
      torch::Tensor const& a,
      torch::Tensor const& b_q_weight,
      torch::Tensor const& s_tok,
      torch::Tensor const& s_ch,
      torch::Tensor const& s_group,
      torch::Tensor& workspace, int64_t size_m,
      int64_t size_n, int64_t size_k,
      bool output_transpose
);