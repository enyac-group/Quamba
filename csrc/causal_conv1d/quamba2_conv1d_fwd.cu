#include "quamba2_conv1d_fwd_kernel.cuh"

template void quamba2_conv1d_channellast_fwd_cuda<int8_t, int8_t>(
    Quamba2ConvParams &params, cudaStream_t stream);