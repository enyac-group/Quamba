#include <torch/extension.h>

torch::Tensor  w8a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale);

torch::Tensor  w4a16_embedding_lookup(torch::Tensor const& input,
    torch::Tensor const& weight, torch::Tensor const& scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_embedding_lookup", &w8a16_embedding_lookup,
          "W8A16 embedding lookup, supporting only symmetric per-token quantization.");
  m.def("w4a16_embedding_lookup", &w4a16_embedding_lookup,
          "W4A16 embedding lookup, supporting only symmetric per-token quantization.");
}
