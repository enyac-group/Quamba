/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

// ////////////////////////////////////////////////////////////////////////////////////////////////////

// struct SSMScanParamsBase {
//     using index_t = uint32_t;

//     int batch, seqlen, n_chunks;
//     index_t a_batch_stride;
//     index_t b_batch_stride;
//     index_t out_batch_stride;

//     // Common data pointers.
//     void *__restrict__ a_ptr;
//     void *__restrict__ b_ptr;
//     void *__restrict__ out_ptr;
//     void *__restrict__ x_ptr;
// };

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;
};

struct QuantSSMParams: public SSMParamsBase {

    index_t x_batch_stride;
    index_t x_d_stride;
    index_t x_dstate_stride;
    // Scaling factor pointers.
    void *__restrict__ scale_A_ptr;
    void *__restrict__ scale_B_ptr;
    void *__restrict__ scale_C_ptr;
    void *__restrict__ scale_D_ptr;
    void *__restrict__ scale_u_ptr;
    void *__restrict__ scale_z_ptr;
    void *__restrict__ scale_ssm_state_ptr;
    void *__restrict__ scale_delta_ptr;
    void *__restrict__ scale_delta_bias_ptr;
};
