/*
The code is from: https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp
*/


#include <torch/extension.h>

#include <vector>

#include "check_args.h"

// CUDA forward declarations
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
    double scale_out
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> rms_norm_affine(
    at::Tensor input,
    at::IntList normalized_shape,
    at::Tensor gamma,
    c10::optional<at::Tensor> residual_in,
    double epsilon,
    double scale_out
) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1, n2;
  check_args(input, normalized_shape, gamma, n1, n2);
  at::Tensor output = at::empty_like(input, input.options().dtype(at::ScalarType::Char));

  const bool has_residual = residual_in.has_value();
  if (has_residual) {
    CHECK_INPUT(residual_in.value());
    TORCH_CHECK(residual_in.value().scalar_type() == input.scalar_type());
    // Not sure if we should recreate a space for residual or just reuse the old one
    at::Tensor residual_out = at::empty_like(residual_in.value(), residual_in.value().options());
    cuda_rms_norm(&output, &input, n1, n2,
        normalized_shape, &gamma, 
        &residual_out, &residual_in.value(),
        epsilon, scale_out);
    return {output, residual_out};
  } else {
    cuda_rms_norm(&output, &input, n1, n2,
        normalized_shape, &gamma, 
        nullptr, nullptr,
        epsilon, scale_out);
    return {output};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &rms_norm_affine, "RMS Norm (CUDA)");
}
