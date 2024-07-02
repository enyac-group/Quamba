#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

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

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void cutlass_scaled_mm_dq(torch::Tensor& c, torch::Tensor const& a,
                          torch::Tensor const& b, torch::Tensor const& a_scales,
                          torch::Tensor const& b_scales) {

  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  // Ampere
  cutlass_scaled_mm_dq_sm80(c, a, b, a_scales, b_scales);
}



void cutlass_scaled_mv_dq(torch::Tensor& c, torch::Tensor const& a,
                          torch::Tensor const& b, float const& alpha,
                          float const& beta) {

  // Checks for conformality
  // a: w, b: x, c: y => y = wx
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == b.size(0) && a.size(1) == b.size(1) &&
              a.size(0) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(1) == 1);                      // Row-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(0) % 16 == 0);  // 16 Byte Alignment

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  // Ampere
  cutlass_scaled_mv_dq_sm80(c, a, b, alpha, beta);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("linear_int8", &linear_int8_forward, "Linear W8A8B8O8 forward (CUDA)");
  m.def("cutlass_scaled_mm_dq", &cutlass_scaled_mm_dq,
          "CUTLASS w8a8 GEMM, supporting symmetric per-tensor or "
          "per-row/column quantization.");
  m.def("cutlass_scaled_mv_dq", &cutlass_scaled_mv_dq,
          "CUTLASS w8a8 GEMV, supporting only symmetric per-tensor quantization.");
}
