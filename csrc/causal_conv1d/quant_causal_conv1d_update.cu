#include "quant_causal_conv1d_update_kernel.cuh"

template void quant_causal_conv1d_update_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);