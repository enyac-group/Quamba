/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;

    void *__restrict__ seq_idx_ptr;

    // No __restrict__ since initial_states could be the same as final_states.
    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;

    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;
};

struct QuantConvParamsBase: public ConvParamsBase {
    float scale_x;
    float scale_w;
    float scale_b;
    float scale_out;
};

struct Quamba2ConvParams: public QuantConvParamsBase {
    // ----- inputs -----
    // define a pointer for xBC to avoid confusion
    void *__restrict__ xBC_ptr;     
    index_t xBC_batch_stride;
    index_t xBC_c_stride;
    index_t xBC_l_stride;
    // reuse scale_x the scaling factor of the input x
    float scale_B;
    float scale_C;

    // ----- output settings -----
    // x out
    int x_dim;
    int x_headdim;
    int d_state;
    int n_groups;
    void *__restrict__ x_head_group_range_ptr ; // [0, hg1, hg1+hg1, hg1+hg2+hg3, ..., num_head]
    int x_nhead_group;
    void *__restrict__ x_dim_group_range_ptr;  // [0, dg1, dg1+dg1, dg1+dg2+dg3, ..., headdim]
    int x_ndim_group;
    void *__restrict__ x_scales_ptr;
    // B out
    void *__restrict__ B_ptr;
    index_t B_batch_stride;
    index_t B_c_stride;
    index_t B_l_stride;
    void *__restrict__ B_scales_ptr;
    // float scale_B_out;
    // C out
    void *__restrict__ C_ptr;
    index_t C_batch_stride;
    index_t C_c_stride;
    index_t C_l_stride;
    void *__restrict__ C_scales_ptr;
    // float scale_C_out;

    // ----- wegihts -----
    float scale_wx;
    float scale_wB;
    float scale_wC;
    float scale_bx;
    float scale_bB;
    float scale_bC;
};