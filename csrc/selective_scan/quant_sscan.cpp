/*
The code is modfied from
https://github.com/state-spaces/mamba
*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "quant_sscan.h"
#include "common/type_shim.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// CUDA forward declarations
template<typename input_t, typename weight_t>
void quant_sscan_fwd_cuda(QuantSSMParams &params, cudaStream_t stream);

// CUDA forward declarations
template<typename input_t, typename weight_t>
void quant_sscan_update_cuda(QuantSSMParams &params, cudaStream_t stream);
// void quant_sscan_update_cuda(int8_t *u);

// this is exactly the same with set_ssm_params_fwd, should combine
void set_quant_ssm_params_fwd(QuantSSMParams &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor scale_u,
                        const at::Tensor delta,
                        const at::Tensor scale_delta,
                        const at::Tensor A,
                        const at::Tensor scale_A,
                        const at::Tensor B,
                        const at::Tensor scale_B,
                        const at::Tensor C,
                        const at::Tensor scale_C,
                        const at::Tensor out,
                        const at::Tensor scale_out,
                        const at::Tensor z,
                        const at::Tensor scale_z,
                        void* D_ptr,
                        void* scale_D_ptr,
                        void* delta_bias_ptr,
                        void* scale_delta_bias_ptr,
                        void* x_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.scale_u_ptr = scale_u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.scale_delta_ptr = scale_delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.scale_A_ptr = scale_A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.scale_B_ptr = scale_B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.scale_C_ptr = scale_C.data_ptr();
    params.D_ptr = D_ptr;
    params.scale_D_ptr = scale_D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.scale_delta_bias_ptr = scale_delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.scale_out_ptr = scale_out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.scale_z_ptr = has_z ? scale_z.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        // params.out_z_batch_stride = out_z.stride(0);
        // params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}


std::vector<at::Tensor>
quant_sscan_fwd(const at::Tensor &u, const at::Tensor &scale_u,
                const at::Tensor &delta, const at::Tensor &scale_delta,
                const at::Tensor &A, const at::Tensor &scale_A, 
                const at::Tensor &B, const at::Tensor &scale_B,
                const at::Tensor &C, const at::Tensor &scale_C, const at::Tensor &scale_out,
                const c10::optional<at::Tensor> &D_, const c10::optional<at::Tensor> &scale_D_,
                const c10::optional<at::Tensor> &z_, const c10::optional<at::Tensor> &scale_z_,
                const c10::optional<at::Tensor> &delta_bias_, const c10::optional<at::Tensor> &scale_delta_bias_,
                bool delta_softplus) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(scale_A.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(weight_type == at::ScalarType::Char);
    TORCH_CHECK(scale_u.scalar_type() == at::ScalarType::Float);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(scale_delta.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(scale_B.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    TORCH_CHECK(scale_C.scalar_type() == at::ScalarType::Float);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "quant_sscan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(scale_u, 1);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(scale_delta, 1);
    CHECK_SHAPE(A, dim, dstate);
    CHECK_SHAPE(scale_A, 1);
    if (!is_variable_B) {
        // remove?
        throw std::logic_error("Must be input-dependent B and C");
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    CHECK_SHAPE(scale_B, 1);

    if (!is_variable_C) {
        // remove?
        throw std::logic_error("Must be input-dependent B and C");
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    CHECK_SHAPE(scale_C, 1);

    TORCH_CHECK(scale_out.scalar_type() == at::ScalarType::Float);
    CHECK_SHAPE(scale_out, 1);

    if (D_.has_value()) {
        // auto D is just for checking
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == weight_type);
        auto scale_D = scale_D_.value();
        TORCH_CHECK(scale_D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
        CHECK_SHAPE(scale_D, 1);
    }

    if (delta_bias_.has_value()) {
        // auto delta_bias is just for checking
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == weight_type);
        auto scale_delta_bias = scale_delta_bias_.value();
        TORCH_CHECK(scale_delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
        CHECK_SHAPE(scale_delta_bias, 1);
    }

    // at::Tensor z, out_z;
    at::Tensor z, scale_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        scale_z = scale_z_.value();
        TORCH_CHECK(scale_z.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        CHECK_SHAPE(scale_z, 1);
        // out_z = torch::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    at::Tensor out = torch::empty({batch_size, dim, seqlen}, u.options().dtype(at::ScalarType::Half));
    // {batch_size, dim, n_chunks, dstate} should be enough if not using complex
    at::Tensor x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(at::ScalarType::Float));  // use floating x for computing

    QuantSSMParams params;
    set_quant_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, scale_u, delta, scale_delta, A, scale_A, B, scale_B, C, scale_C, out, scale_out, z, scale_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       scale_D_.has_value() ? scale_D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       scale_delta_bias_.has_value() ? scale_delta_bias_.value().data_ptr() : nullptr,
                       x.data_ptr(), has_z, delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(u.scalar_type(), "quant_sscan_fwd", [&] {
        DISPATCH_WTYPE_INTEGRAL(A.scalar_type(), "quant_sscan_fwd", [&] {
            quant_sscan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    // std::vector<at::Tensor> result = {out, x};
    // // if (has_z) { result.push_back(out_z); }
    // B D L -> B L D for the following out projection
    auto out_T = out.transpose(1, 2).contiguous();
    std::vector<at::Tensor> result = {out_T, x};
    return result;
}


// this is exactly the same with set_ssm_params_fwd, should combine
void set_quant_ssm_params_update(QuantSSMParams &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t dstate,
                        // device pointers
                        const at::Tensor x,
                        const at::Tensor u,
                        const at::Tensor scale_u,
                        const at::Tensor delta,
                        const at::Tensor scale_delta,
                        const at::Tensor A,
                        const at::Tensor scale_A,
                        const at::Tensor B,
                        const at::Tensor scale_B,
                        const at::Tensor C,
                        const at::Tensor scale_C,
                        const at::Tensor out,
                        const at::Tensor z,
                        const at::Tensor scale_z,
                        void* D_ptr,
                        void* scale_D_ptr,
                        void* delta_bias_ptr,
                        void* scale_delta_bias_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.dstate = dstate;

    params.delta_softplus = delta_softplus;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.u_ptr = u.data_ptr();
    params.scale_u_ptr = scale_u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.scale_delta_ptr = scale_delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.scale_A_ptr = scale_A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.scale_B_ptr = scale_B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.scale_C_ptr = scale_C.data_ptr();
    params.D_ptr = D_ptr;
    params.scale_D_ptr = scale_D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.scale_delta_bias_ptr = scale_delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.scale_z_ptr = has_z ? scale_z.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    params.B_batch_stride = B.stride(0);
    params.B_dstate_stride = B.stride(1);
    params.C_batch_stride = C.stride(0);
    params.C_dstate_stride = C.stride(1);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.x_batch_stride = x.stride(0);
    params.x_d_stride = x.stride(1);
    params.x_dstate_stride = x.stride(2);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        // params.out_z_batch_stride = out_z.stride(0);
        // params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}


/*
ssm_state: torch.Size([1, 1536, 16])
u: torch.Size([1, 1536])
delta: torch.Size([1, 1536])
A:  torch.Size([1536, 16])
B:  torch.Size([1, 16])
C:  torch.Size([1, 16])
D:  torch.Size([1536])
z:  torch.Size([1, 1536])
delta_bias:  torch.Size([1536])
*/

at::Tensor
quant_sscan_update(
                const at::Tensor &ssm_state,
                const at::Tensor &u, const at::Tensor &scale_u,
                const at::Tensor &delta, const at::Tensor &scale_delta,
                const at::Tensor &A, const at::Tensor &scale_A, 
                const at::Tensor &B, const at::Tensor &scale_B,
                const at::Tensor &C, const at::Tensor &scale_C,
                const c10::optional<at::Tensor> &D_, const c10::optional<at::Tensor> &scale_D_,
                const c10::optional<at::Tensor> &z_, const c10::optional<at::Tensor> &scale_z_,
                const c10::optional<at::Tensor> &delta_bias_, const c10::optional<at::Tensor> &scale_delta_bias_,
                bool delta_softplus) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(scale_A.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(weight_type == at::ScalarType::Char);
    TORCH_CHECK(scale_u.scalar_type() == at::ScalarType::Float);

    // FIXME: this should be half or char?
    TORCH_CHECK(ssm_state.scalar_type() == at::ScalarType::Float);
    // TORCH_CHECK(ssm_state.scalar_type() == at::ScalarType::Half);
    // TORCH_CHECK(
    //     ssm_state.scalar_type() == at::ScalarType::Half ||
    //     ssm_state.scalar_type() == at::ScalarType::Float ||
    //     ssm_state.scalar_type() == at::ScalarType::BFloat16
    // );


    const bool is_variable_B = B.dim() >= 2; // [bsize, dstate]
    const bool is_variable_C = C.dim() >= 2; // [bsize, dstate]

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(scale_delta.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(scale_B.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    TORCH_CHECK(scale_C.scalar_type() == at::ScalarType::Float);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes(); // u: [bsize, dim]
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int dstate = A.size(1); // A: [dim, dstate]

    TORCH_CHECK(dstate <= 256, "quant_sscan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim); // u: [bsize, dim]
    CHECK_SHAPE(scale_u, 1);
    CHECK_SHAPE(ssm_state, batch_size, dim, dstate); // ssm_state: [bsize, dim, dstate]
    CHECK_SHAPE(delta, batch_size, dim); // delta: [bsize, dim]
    CHECK_SHAPE(scale_delta, 1);
    CHECK_SHAPE(A, dim, dstate); // A: [dim, dstate]
    CHECK_SHAPE(scale_A, 1);
    if (!is_variable_B) {
        // remove?
        throw std::logic_error("Must be input-dependent B and C");
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, dstate);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    CHECK_SHAPE(scale_B, 1);

    if (!is_variable_C) {
        // remove?
        throw std::logic_error("Must be input-dependent B and C");
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, dstate);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    CHECK_SHAPE(scale_C, 1);

    if (D_.has_value()) {
        // auto D is just for checking
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == weight_type);
        auto scale_D = scale_D_.value();
        TORCH_CHECK(scale_D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim); // D: [ndim]
        CHECK_SHAPE(scale_D, 1);
    }

    if (delta_bias_.has_value()) {
        // auto delta_bias is just for checking
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == weight_type);
        auto scale_delta_bias = scale_delta_bias_.value();
        TORCH_CHECK(scale_delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim); // D: [ndim]
        CHECK_SHAPE(scale_delta_bias, 1);
    }

    // at::Tensor z, out_z;
    at::Tensor z, scale_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        scale_z = scale_z_.value();
        TORCH_CHECK(scale_z.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim);
        CHECK_SHAPE(scale_z, 1);
        // out_z = torch::empty_like(z);
    }

    // const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::empty({batch_size, dim}, u.options().dtype(at::ScalarType::Half));

    QuantSSMParams params;
    set_quant_ssm_params_update(params, batch_size, dim, dstate,
                       ssm_state, u, scale_u, delta, scale_delta, A, scale_A, B, scale_B, C, scale_C, out, z, scale_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       D_.has_value() ? scale_D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? scale_delta_bias_.value().data_ptr() : nullptr,
                       has_z, delta_softplus);
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_INTEGRAL(u.scalar_type(), "quant_sscan_update", [&] {
        DISPATCH_WTYPE_INTEGRAL(A.scalar_type(), "quant_sscan_update", [&] {
            quant_sscan_update_cuda<input_t, weight_t>(params, stream);
        });
    });

    // auto stream = at::cuda::getCurrentCUDAStream().stream();
    // quant_sscan_update_cuda(u.data_ptr<int8_t>());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &quant_sscan_fwd, "Quantized selective scan forward");
    m.def("update", &quant_sscan_update, "Quantized selective scan update");
}
