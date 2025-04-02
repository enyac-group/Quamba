#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "causal_conv1d.h"      // include Quamba2ConvParams
#include "common/type_shim.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// CUDA forward declarations
template<typename input_t, typename weight_t>
void quamba2_conv1d_channellast_fwd_cuda(Quamba2ConvParams &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void quamba2_conv1d_update_cuda(Quamba2ConvParams &params, cudaStream_t stream);

void set_quamba2_conv1d_fwd_params(
                        Quamba2ConvParams &params,
                         // ----- input -----
                         const at::Tensor xBC,
                         const size_t batch,        // batch size
                         const size_t dim,          // total dimension: x_dim + 2* d_state
                         const size_t seqlen,       // sequence length
                         const float scale_x,       // input x tensor scaling factor, x splitted from xBC
                         const float scale_B,       // input B tensor scaling factor, B splitted from xBC
                         const float scale_C,       // input C tensor scaling factor, C splitted from xBC
                         // ----- output settings -----
                         const int x_dim,           // x dim
                         const int x_headdim,
                         const int d_state,         // B, C: d_state*n_groups
                         const int n_groups,
                         // ----- output -----
                         const at::Tensor x,        // output x
                         const int x_nhead_group,
                         const int x_ndim_group,
                         void* x_head_group_range_ptr,     // x_head_group range
                         void* x_dim_group_range_ptr,      // x_dim_group range
                         const at::Tensor x_scales,         // x scales: nhead_group*ndim_group 
                         const at::Tensor B,        // output B
                         const at::Tensor scale_B_out,   // input B tensor scaling factor
                         const at::Tensor C,        // output C
                         const at::Tensor scale_C_out,   // input C tensor scaling factor
                         // ----- weights -----
                         const at::Tensor weight,   // conv weight
                         const size_t width,        // conv width = 4
                         const float scale_wx,
                         const float scale_wB,
                         const float scale_wC,
                         void* bias_ptr,            // conv bias
                         const float scale_bx,
                         const float scale_bB,
                         const float scale_bC,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // ----- input -----
    params.xBC_ptr = xBC.data_ptr();    // input xBC
    params.xBC_batch_stride = xBC.stride(0);
    params.xBC_c_stride = xBC.stride(1);
    params.xBC_l_stride = xBC.stride(-1);
    params.batch = batch;               // batch
    params.dim = dim;                   // xBC dimension
    params.seqlen = seqlen;             // sequence length
    params.scale_x = scale_x;           // input x scaling factor (splitted from xBC)
    params.scale_B = scale_B;           // input B scaling factor (splitted from xBC)
    params.scale_C = scale_C;           // input C scaling factor (splitted from xBC)

    // ----- output tensor settings -----
    params.x_dim = x_dim;
    params.x_headdim = x_headdim;
    params.d_state = d_state;
    params.n_groups = n_groups;

    // ----- output -----
    // output x
    params.x_ptr = x.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(-1);
    // params.scale_x_out = scale_x_out;
    params.x_head_group_range_ptr = x_head_group_range_ptr;  // [hg1, hg1+hg2, hg1+hg2+hg3, ..., num_head]
    params.x_nhead_group = x_nhead_group;    // [n_ssd_groups, n_head_groups]
    params.x_dim_group_range_ptr = x_dim_group_range_ptr;    // [dg1, dg1+dg2, dg1+dg2+dg3, ..., headdim]
    params.x_ndim_group = x_ndim_group;      // [n_ssd_groups, n_dim_groups]
    params.x_scales_ptr = x_scales.data_ptr();


    // output B
    params.B_ptr = B.data_ptr();
    params.B_scales_ptr = scale_B_out.data_ptr();
    params.B_batch_stride = B.stride(0);
    params.B_c_stride = B.stride(1);
    params.B_l_stride = B.stride(-1);
    // output C
    params.C_ptr = C.data_ptr(); 
    params.C_scales_ptr = scale_C_out.data_ptr();
    params.C_batch_stride = C.stride(0);
    params.C_c_stride = C.stride(1);
    params.C_l_stride = C.stride(-1);

    // ----- weights -----
    params.weight_ptr = weight.data_ptr();      // conv weight
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.bias_ptr = bias_ptr;
    params.width = width;                       // conv width
    // weight scales: xBC
    params.scale_wx = scale_wx;
    params.scale_wB = scale_wB;
    params.scale_wC = scale_wC;
    // bias scales: xBC
    params.scale_bx = scale_bx;
    params.scale_bB = scale_bB;
    params.scale_bC = scale_bC;
    params.silu_activation = silu_activation;
}


std::vector<at::Tensor> quamba2_conv1d_fwd(
    // inputs
    const at::Tensor &xBC, const float &scale_x, const float &scale_B, const float &scale_C,
    // output settings
    const int x_dim, const int x_headdim, const int d_state, const int n_groups,
    // outputs
    const c10::optional<at::Tensor> &x_head_group_range_,
    const c10::optional<at::Tensor> &x_dim_group_range_,
    const at::Tensor &x_scales, const at::Tensor &scale_B_out, const at::Tensor &scale_C_out,
    // wegihts
    const at::Tensor &weight,
    const float &scale_wx, const float &scale_wB, const float &scale_wC,
    const c10::optional<at::Tensor> &bias_,
    const float &scale_bx, const float &scale_bB, const float &scale_bC,
    const c10::optional<at::Tensor> &seq_idx_,
    const c10::optional<at::Tensor> &initial_states_,
    c10::optional<at::Tensor> &final_states_out_,
    bool silu_activation
) {
    auto input_type = xBC.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(weight_type == at::ScalarType::Char);

    TORCH_CHECK(xBC.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = xBC.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(xBC, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(xBC.stride(2) == 1 || xBC.stride(1) == 1);
    const bool is_channel_last = xBC.stride(1) == 1 && xBC.stride(2) > 1;

    if (is_channel_last) {
        TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        TORCH_CHECK(xBC.stride(2) % 8 == 0 and xBC.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
    }
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    // check output settings
    TORCH_CHECK(x_dim <= dim, "x_dim must less or equal to input dim");
    TORCH_CHECK(x_dim % x_headdim == 0, "x_dim must be devided by x_headdim");
    TORCH_CHECK(x_dim % 128 == 0, "only supports x_dim divisible by 128 (kChunkSizeC) for now");
    TORCH_CHECK(d_state*n_groups % 128 == 0, "only supports d_state*n_groups divisible by 128 (kChunkSizeC) for now");
    TORCH_CHECK(x_dim + 2*d_state*n_groups == dim, "x_dim + 2*d_state*n_groups must be equl to total dim");
    int x_nhead_group = 0;
    int x_ndim_group = 0;
    int x_nscales = x_scales.numel();
    if (x_head_group_range_.has_value() && x_dim_group_range_.has_value()) {
        auto x_head_group_range = x_head_group_range_.value();
        auto x_dim_group_range = x_dim_group_range_.value();
        TORCH_CHECK(x_head_group_range.sizes()[0] == n_groups, "x_head_group_range.shape[0] must be equl to total number of groups");
        TORCH_CHECK(x_dim_group_range.sizes()[0] == n_groups, "x_dim_group_range.shape[0] must be equl to total number of groups");
        x_nhead_group = x_head_group_range.sizes()[1];
        x_ndim_group = x_dim_group_range.sizes()[2];
        TORCH_CHECK(x_dim_group_range.sizes()[1] == x_nhead_group, "x_dim_group_range.shape[1] must be equl to total number of head groups");
        TORCH_CHECK(x_nscales == n_groups*x_nhead_group*x_ndim_group, "number of x scales must be equl to total number of groups");
        TORCH_CHECK(scale_B_out.numel() == n_groups, "number of B scales must be equl to total number of SSD groups");
        TORCH_CHECK(scale_C_out.numel() == n_groups, "number of B scales must be equl to total number of SSD groups");
    } else {
        TORCH_CHECK(x_nscales == 1, "number of x scales must be equl to 1");
        TORCH_CHECK(scale_B_out.numel() == 1, "number of B scales must be equl to 1");
        TORCH_CHECK(scale_C_out.numel() == 1, "number of C scales must be equl to 1");
    }

    // output tensors
    TORCH_CHECK(is_channel_last, "only support channel_last");
    at::Tensor x = torch::empty({batch_size, seqlen, x_dim}, xBC.options().dtype(at::ScalarType::Char)).transpose(1, 2); // int8
    at::Tensor B = torch::empty({batch_size, seqlen, d_state*n_groups}, xBC.options().dtype(at::ScalarType::Char)).transpose(1, 2); // int8
    at::Tensor C = torch::empty({batch_size, seqlen, d_state*n_groups}, xBC.options().dtype(at::ScalarType::Char)).transpose(1, 2); // int8


    Quamba2ConvParams params;
    set_quamba2_conv1d_fwd_params(params,
        // ----- input -----
        xBC, batch_size, dim, seqlen, scale_x, scale_B, scale_C,
        // ----- output settings -----
        x_dim, x_headdim, d_state, n_groups,
        // ----- output -----
        x, x_nhead_group, x_ndim_group,
        x_head_group_range_.has_value() ? x_head_group_range_.value().data_ptr() : nullptr,
        x_dim_group_range_.has_value() ? x_dim_group_range_.value().data_ptr() : nullptr,
        x_scales,
        B, scale_B_out, C, scale_C_out,
        // ----- weight -----
        weight, width, scale_wx, scale_wB, scale_wC,
        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
        scale_bx, scale_bB, scale_bC, silu_activation
    );


    if (seq_idx_.has_value()) {
        throw std::logic_error("No implementation for seq_idx");
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        throw std::logic_error("No implementation for initial_states");
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        throw std::logic_error("No implementation for final_states_out");
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)xBC.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(xBC.scalar_type(), "quamba2_conv1d_fwd", [&] {
        DISPATCH_WTYPE_INTEGRAL(weight.scalar_type(), "quamba2_conv1d_fwd", [&] {
            if (!is_channel_last) {
                 throw std::logic_error("No implementation for seqlen_last");
            } else {
                quamba2_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });

    return {x, B, C};
}


std::vector<at::Tensor> quamba2_conv1d_update(
    // inputs
    const at::Tensor &xBC, const at::Tensor &conv_state,
    const float &scale_x, const float &scale_B, const float &scale_C,
    // output settings
    const int x_dim, const int x_headdim, const int d_state, const int n_groups,
    // outputs
    const c10::optional<at::Tensor> &x_head_group_range_,
    const c10::optional<at::Tensor> &x_dim_group_range_,
    const at::Tensor &x_scales, const at::Tensor &scale_B_out, const at::Tensor &scale_C_out,
    // wegihts
    const at::Tensor &weight,
    const float &scale_wx, const float &scale_wB, const float &scale_wC,
    const c10::optional<at::Tensor> &bias_,
    const float &scale_bx, const float &scale_bB, const float &scale_bC,
    const c10::optional<at::Tensor> &seq_idx_,
    const c10::optional<at::Tensor> &initial_states_,
    c10::optional<at::Tensor> &final_states_out_,
    bool silu_activation
) {

    auto input_type = xBC.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(weight_type == at::ScalarType::Char);
    TORCH_CHECK(conv_state.scalar_type() == input_type);

    TORCH_CHECK(xBC.is_cuda());
    TORCH_CHECK(conv_state.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = xBC.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int width = weight.size(-1);

    CHECK_SHAPE(xBC, batch_size, dim); // seqlen = 1
    CHECK_SHAPE(conv_state, batch_size, dim, width);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(width == 4, "causal_conv1d only supports width = 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    // check output settings
    TORCH_CHECK(x_dim <= dim, "x_dim must less or equal to input dim");
    TORCH_CHECK(x_dim % x_headdim == 0, "x_dim must be devided by x_headdim");
    TORCH_CHECK(x_dim % 128 == 0, "only supports x_dim divisible by 128 (kChunkSizeC) for now");
    TORCH_CHECK(d_state*n_groups % 128 == 0, "only supports d_state*n_groups divisible by 128 (kChunkSizeC) for now");
    TORCH_CHECK(x_dim + 2*d_state*n_groups == dim, "x_dim + 2*d_state*n_groups must be equl to total dim");

    int x_nhead_group = 0;
    int x_ndim_group = 0;
    int x_nscales = x_scales.numel();
    if (x_head_group_range_.has_value() && x_dim_group_range_.has_value()) {
        auto x_head_group_range = x_head_group_range_.value();
        auto x_dim_group_range = x_dim_group_range_.value();
        TORCH_CHECK(x_head_group_range.sizes()[0] == n_groups, "x_head_group_range.shape[0] must be equl to total number of groups");
        TORCH_CHECK(x_dim_group_range.sizes()[0] == n_groups, "x_dim_group_range.shape[0] must be equl to total number of groups");
        x_nhead_group = x_head_group_range.sizes()[1];
        x_ndim_group = x_dim_group_range.sizes()[2];
        TORCH_CHECK(x_dim_group_range.sizes()[1] == x_nhead_group, "x_dim_group_range.shape[1] must be equl to total number of head groups");
        TORCH_CHECK(x_nscales == n_groups*x_nhead_group*x_ndim_group, "number of x scales must be equl to total number of groups");
        TORCH_CHECK(scale_B_out.numel() == n_groups, "number of B scales must be equl to total number of SSD groups");
        TORCH_CHECK(scale_C_out.numel() == n_groups, "number of B scales must be equl to total number of SSD groups");
    } else {
        TORCH_CHECK(x_nscales == 1, "number of x scales must be equl to 1");
        TORCH_CHECK(scale_B_out.numel() == 1, "number of B scales must be equl to 1");
        TORCH_CHECK(scale_C_out.numel() == 1, "number of B scales must be equl to 1");
    }



    // output tensors
    at::Tensor x = torch::empty({batch_size, x_dim}, xBC.options().dtype(at::ScalarType::Char)); // int8
    at::Tensor B = torch::empty({batch_size, d_state*n_groups}, xBC.options().dtype(at::ScalarType::Char)); // int8
    at::Tensor C = torch::empty({batch_size, d_state*n_groups}, xBC.options().dtype(at::ScalarType::Char)); // int8

    Quamba2ConvParams params;
    set_quamba2_conv1d_fwd_params(params,
        // ----- input -----
        xBC, batch_size, dim, /*seqlen=*/1, scale_x, scale_B, scale_C,
        // ----- output settings -----
        x_dim, x_headdim, d_state, n_groups,
        // ----- output -----
        x, x_nhead_group, x_ndim_group,
        x_head_group_range_.has_value() ? x_head_group_range_.value().data_ptr() : nullptr,
        x_dim_group_range_.has_value() ? x_dim_group_range_.value().data_ptr() : nullptr,
        x_scales,
        B, scale_B_out, C, scale_C_out,
        // ----- weight -----
        weight, width, scale_wx, scale_wB, scale_wC,
        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
        scale_bx, scale_bB, scale_bC, silu_activation
    );
    params.conv_state_ptr = conv_state.data_ptr();
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(x.scalar_type(), "quamba2_conv1d_update_cuda", [&] {
        DISPATCH_WTYPE_INTEGRAL(weight.scalar_type(), "quamba2_conv1d_update_cuda", [&] {
            quamba2_conv1d_update_cuda<input_t, weight_t>(params, stream);
        });
    });
    return {x, B, C};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &quamba2_conv1d_fwd, "Quamba2 causal convolution forward");
    m.def("update", &quamba2_conv1d_update, "Quamba2 causal convolution update");
    // m.def("update", &quant_causal_conv1d_mixed_out_update, "Quantized causal conv1d mixed precision output update");
}