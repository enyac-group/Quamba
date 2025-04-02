#include "quant_embedding_kernel.cuh"

torch::Tensor  w8a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale);

torch::Tensor w4a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale);