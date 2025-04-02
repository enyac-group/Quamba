#include "quamba2_conv1d_update_kernel.cuh"

template void quamba2_conv1d_update_cuda<int8_t, int8_t>(
    Quamba2ConvParams &params, cudaStream_t stream);