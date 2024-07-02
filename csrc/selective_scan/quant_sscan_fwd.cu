/*
The code is modfied from
https://github.com/state-spaces/mamba
*/

// Split into multiple files to compile in paralell
#include "quant_sscan_fwd_kernel.cuh"

// quant_sscan_fwd_cuda<input_t, weight_t>(params, stream)
// int8_t -> symmetric quant; uint8_t -> asymmetric quant 
template void quant_sscan_fwd_cuda<int8_t, int8_t>(QuantSSMParams &params, cudaStream_t stream);