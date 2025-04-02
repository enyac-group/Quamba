#include "quant_causal_conv1d_fwd_kernel.cuh"

template void quant_causal_conv1d_fwd_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);

template void quant_causal_conv1d_channellast_fwd_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);