#include <torch/extension.h>


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

torch::Tensor w8a8o8_gemv(
    torch::Tensor& c, 
    torch::Tensor const& a,         // input vector with shape (1, k)
    torch::Tensor const& b,         // weight matrix with shape (n, k)
    torch::Tensor const& scale_a,   // input scale with shape (1,)
    torch::Tensor const& scale_b    // weight scale with shape (n,)
);

// W4A16O16, adapted from Marlin
torch::Tensor  w4a16o16_gemm(
  const torch::Tensor& A,
  const torch::Tensor& B,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int64_t size_m,
  int64_t size_n,
  int64_t size_k,
  bool output_transpose,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
);

// W4A8O16, adapted from Marlin_QQQ
torch::Tensor w4a8o16_gemm(
  torch::Tensor const& a,
  torch::Tensor const& b_q_weight,
  torch::Tensor const& s_tok,
  torch::Tensor const& s_ch,
  torch::Tensor const& s_group,
  torch::Tensor& workspace, int64_t size_m,
  int64_t size_n, int64_t size_k, bool output_transpose
);

// W4A8O8, adapted from Marlin_QQQ
torch::Tensor w4a8o8_gemm(
  torch::Tensor const& a,
  torch::Tensor const& b_q_weight,
  torch::Tensor const& s_tok,
  torch::Tensor const& s_ch,
  torch::Tensor const& s_group,
  torch::Tensor& workspace, int64_t size_m,
  int64_t size_n, int64_t size_k, bool output_transpose
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_scaled_mm_dq", &cutlass_scaled_mm_dq,
          "CUTLASS w8a8 GEMM, supporting symmetric per-tensor or "
          "per-row/column quantization.");
  m.def("cutlass_scaled_mv_dq", &cutlass_scaled_mv_dq,
          "CUTLASS w8a8 GEMV, supporting only symmetric per-tensor quantization.");
  m.def("w8a8o8_gemv", &w8a8o8_gemv,
          "w8a8o8 GEMV, supporting only symmetric per-tensor quantization.");
 // adapted from Marlin
  m.def("w4a16o16_gemm", &w4a16o16_gemm);
 // adapted from Marlin_QQQ
  m.def("w4a8o16_gemm", &w4a8o16_gemm);
  m.def("w4a8o8_gemm", &w4a8o8_gemm);
}
