/*
The code is from: https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp
*/
#pragma once

#include <torch/extension.h>

#include <vector>

namespace {
void compute_n1_n2(
    at::Tensor input,
    at::IntList normalized_shape,
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input.sizes()[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input.sizes()[i];
    }
}

void check_args(
    at::Tensor input,
    at::IntList normalized_shape,
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input, normalized_shape, n1, n2);
}

void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}


void check_args(
    at::Tensor input,
    at::IntList normalized_shape,
    at::Tensor gamma,
    int& n1,
    int& n2
    )
{
    check_args(input, normalized_shape, n1, n2);
    check_args(normalized_shape, gamma);
}

}