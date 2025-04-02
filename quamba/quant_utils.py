import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange, repeat

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

@torch.no_grad()
def quantize_tensor_per_tensor_absmax(
        w: torch.tensor, n_bits, clip_ratio=1.0,
    ):
    q_min, q_max = _get_quant_range(n_bits, sym=True)
    w_max = w.abs().amax().clamp(min=1e-5)
    if clip_ratio < 1.0:
        w_max = w_max * clip_ratio
    scales = w_max / q_max
    # real quant
    w = torch.clamp(torch.round(w / scales), q_min, q_max)
    # return w, scales.float().item()
    return w, scales.float()


@torch.no_grad()
def quantize_tensor_head_channel_grouping(w, w_head_group_range, w_dim_group_range, n_bits, scales=None, fake_quant=False, clip_ratio=1.0):

    # decoding mode
    if len(w.shape) == 3: # [batch, nheads, headdim]
        w = w.unsqueeze(1)  # [batch, nheads, headdim] -> [batch, 1, nheads, headdim]
    
    assert len(w.shape) == 4, "Only support 4D tensor with shape [batch, seqlen, nheads, headdim]"
    batch, seqlen, nheads, headdim = w.shape

    assert len(w_head_group_range.shape) == 2, "w_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(w_dim_group_range.shape) == 3, "w_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"

    saved_type = w.dtype
    saved_device = w.device
    saved_shape = w.shape
    ngroups = w_head_group_range.shape[0]
    n_head_groups = w_head_group_range.shape[1]
    n_dim_groups = w_dim_group_range.shape[2]
    assert nheads % ngroups == 0
    assert ngroups == w_dim_group_range.shape[0]
    assert w_head_group_range.dtype == torch.int32 # [n_ssd_groups, n_head_groups]
    assert w_dim_group_range.dtype == torch.int32  # [n_ssd_groups, n_head_groups, n_dim_groups]

    if scales is not None: # [ngroups, n_head_groups, n_dim_groups]
        assert scales.shape == (ngroups, n_head_groups, n_dim_groups)

    w = w.reshape(batch, seqlen, ngroups, nheads//ngroups, headdim)
    
    # quantize x by head group and channel group
    q_min, q_max = _get_quant_range(n_bits, sym=True)
    qw = []
    w_scales = []
    for ssd_g in range(ngroups):
        qw_h = []
        h_start = 0
        for hg_idx in range(n_head_groups):
            qw_hg = []
            h_end = w_head_group_range[ssd_g][hg_idx].item()
            w_hg = w[:, :, ssd_g, h_start:h_end, :] # [b, l, gh, d]
            d_start = 0
            for dg_idx in range(n_dim_groups):
                d_end = w_dim_group_range[ssd_g][hg_idx][dg_idx].item()
                w_hg_dg = w_hg[..., d_start:d_end]  # [b, l, gh, gd]
                if scales is not None:
                    s_hg_dg = scales[ssd_g, hg_idx, dg_idx]
                else:
                    w_max = w_hg_dg.abs().max().clamp(min=1e-5)
                    if clip_ratio < 1.0:
                        w_max = w_max * clip_ratio
                    s_hg_dg = (w_max / q_max).to(torch.float32)
                    w_scales.append(s_hg_dg.reshape(1)) # for torch.cat
                qw_hg_dg = (w_hg_dg / s_hg_dg).round().clamp(q_min, q_max) # quant
                if fake_quant:
                    qw_hg_dg = qw_hg_dg * s_hg_dg # scale back to float
                qw_hg.append(qw_hg_dg)
                d_start = d_end
            qw_hg = torch.cat(qw_hg, dim=-1) # [b, l, gh, d]
            qw_h.append(qw_hg)
            h_start = h_end
        qw_h = torch.cat(qw_h, dim=2) # [b, l, h, d]
        qw.append(qw_h.unsqueeze(2)) # [b, l, 1, h, d]
    qw = torch.cat(qw, dim=2).to(saved_type).to(saved_device) # [b, l, g, h, d], we don't convert it to integer type here
    qw = qw.reshape(saved_shape)
    if scales is None:
        w_scales = torch.cat(w_scales, dim=0).to(torch.float32).to(saved_device)
        w_scales = w_scales.reshape(ngroups, n_head_groups, n_dim_groups)
        scales = w_scales
    return qw, scales


@torch.no_grad()
def dequantize_tensor_head_channel_grouping(qw, w_head_group_range, w_dim_group_range, scales):

    # decoding mode
    if len(qw.shape) == 3: # [batch, nheads, headdim]
        qw = qw.unsqueeze(1)  # [batch, nheads, headdim] -> [batch, 1, nheads, headdim]
    
    assert len(qw.shape) == 4, "Only support 4D tensor with shape [batch, seqlen, nheads, headdim]"
    batch, seqlen, nheads, headdim = qw.shape

    assert len(w_head_group_range.shape) == 2, "w_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(w_dim_group_range.shape) == 3, "w_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"

    saved_type = qw.dtype
    saved_device = qw.device
    saved_shape = qw.shape
    ngroups = w_head_group_range.shape[0]
    n_head_groups = w_head_group_range.shape[1]
    n_dim_groups = w_dim_group_range.shape[2]
    assert nheads % ngroups == 0
    assert ngroups == w_dim_group_range.shape[0]
    assert w_head_group_range.dtype == torch.int32 # [n_ssd_groups, n_head_groups]
    assert w_dim_group_range.dtype == torch.int32  # [n_ssd_groups, n_head_groups, n_dim_groups]

    qw = qw.reshape(batch, seqlen, ngroups, nheads//ngroups, headdim)
    
    # dequantize x by head group and channel group
    w = []
    for ssd_g in range(ngroups):
        w_h = []
        h_start = 0
        for hg_idx in range(n_head_groups):
            w_hg = []
            h_end = w_head_group_range[ssd_g][hg_idx].item()
            qw_hg = qw[:, :, ssd_g, h_start:h_end, :] # [b, l, gh, d]
            d_start = 0
            for dg_idx in range(n_dim_groups):
                d_end = w_dim_group_range[ssd_g][hg_idx][dg_idx].item()
                qw_hg_dg = qw_hg[..., d_start:d_end]  # [b, l, gh, gd]
                s_hg_dg = scales[ssd_g, hg_idx, dg_idx]
                w_hg_dg = qw_hg_dg * s_hg_dg # dequant
                w_hg.append(w_hg_dg)
                d_start = d_end
            w_hg = torch.cat(w_hg, dim=-1) # [b, l, gh, d]
            w_h.append(w_hg)
            h_start = h_end
        w_h = torch.cat(w_h, dim=2) # [b, l, h, d]
        w.append(w_h.unsqueeze(2)) # [b, l, 1, h, d]
    w = torch.cat(w, dim=2).to(saved_device) # [b, l, g, h, d]
    w = w.reshape(saved_shape)
    return w


@torch.no_grad()
def quantize_states_head_channel_grouping(w, w_head_group_range, w_dim_group_range, n_bits, scales=None, fake_quant=False, clip_ratio=1.0):

    assert len(w.shape) == 4, "Only support 4D tensor with shape [batch, nheads, headdim, dstate]"
    batch, nheads, headdim, dstate = w.shape

    assert len(w_head_group_range.shape) == 2, "w_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(w_dim_group_range.shape) == 3, "w_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"

    saved_type = w.dtype
    saved_device = w.device
    saved_shape = w.shape
    ngroups = w_head_group_range.shape[0]
    n_head_groups = w_head_group_range.shape[1]
    n_dim_groups = w_dim_group_range.shape[2]
    assert nheads % ngroups == 0
    assert ngroups == w_dim_group_range.shape[0]
    assert w_head_group_range.dtype == torch.int32 # [n_ssd_groups, n_head_groups]
    assert w_dim_group_range.dtype == torch.int32  # [n_ssd_groups, n_head_groups, n_dim_groups]

    if scales is not None: # [n_head_groups, n_dim_groups]
        assert scales.shape == (ngroups, n_head_groups, n_dim_groups, dstate)

    w = w.reshape(batch, ngroups, nheads//ngroups, headdim, dstate)

    # quantize x by head group and channel group
    q_min, q_max = _get_quant_range(n_bits, sym=True)
    qw = []
    w_scales = []
    for ds in range(dstate):
        wd = w[:, :, :, :, ds] # [b, ng, nh, hd, ds] -> [b, ng, nh, hd]
        qwd = []
        qwd_scales = []
        for ssd_g in range(ngroups):
            qwd_h = []
            h_start = 0
            for hg_idx in range(n_head_groups):
                qwd_hg = []
                h_end = w_head_group_range[ssd_g][hg_idx].item()
                wd_hg = wd[:, ssd_g, h_start:h_end, :] # [b, ng, nh, hd] -> [b, gh, hd]
                d_start = 0
                for dg_idx in range(n_dim_groups):
                    d_end = w_dim_group_range[ssd_g][hg_idx][dg_idx].item()
                    wd_hg_dg = wd_hg[..., d_start:d_end]  # [b, gh, hd] -> [b, gh, gd]
                    if scales is not None:
                        sd_hg_dg = scales[ssd_g, hg_idx, dg_idx, ds]
                    else:
                        w_max = wd_hg_dg.abs().max()
                        if clip_ratio < 1.0:
                            w_max = w_max * clip_ratio
                        sd_hg_dg = (w_max / q_max).to(torch.float32).clamp(min=1e-6)
                    qwd_scales.append(sd_hg_dg.reshape(1)) # for torch.cat
                    qwd_hg_dg = (wd_hg_dg.to(torch.float64) / sd_hg_dg).round().clamp(q_min, q_max) # quant
                    if fake_quant:
                        qwd_hg_dg = qwd_hg_dg * sd_hg_dg # scale back to float
                    qwd_hg.append(qwd_hg_dg) # [b, gh, gd]
                    d_start = d_end
                qwd_hg = torch.cat(qwd_hg, dim=-1) # [[b, gh, gd], [b, gh, gd], ...] -> [b, gh, hd]
                qwd_h.append(qwd_hg) # [[b, gh, hd], [b, gh, hd], ...]
                h_start = h_end
            qwd_h = torch.cat(qwd_h, dim=1) # [[b, gh, hd], [b, gh, hd], ...] -> [b, nh, hd]
            qwd.append(qwd_h.unsqueeze(1)) # [b, 1, nh, hd]
        qwd_scales = torch.cat(qwd_scales, dim=0).to(torch.float32).reshape(ngroups, n_head_groups, n_dim_groups)
        w_scales.append(qwd_scales.unsqueeze(-1)) # [ng, n_head_groups, n_dim_groups, 1]
        qwd = torch.cat(qwd, dim=1) # [[b, 1, nh, hd], [b, 1, nh, hd], ...] -> [b, ng, nh, hd]
        qw.append(qwd.unsqueeze(-1)) # [[b, ng, nh, hd 1], [b, ng, nh, hd 1], ...]
    qw = torch.cat(qw, dim=-1).to(saved_type).to(saved_device) # [[b, ng, nh, hd, 1], [b, ng, nh, hd, 1], ...] -> [b, ng, nh, hd, ds], we don't convert it to integer type here
    qw = rearrange(qw, "b g h d n -> b (g h) d n")
    assert qw.shape == saved_shape
    if scales is None:
        w_scales = torch.cat(w_scales, dim=-1).to(torch.float32).to(saved_device)
        scales = w_scales
    return qw.contiguous(), scales


@torch.no_grad()
def dequantize_states_head_channel_grouping(qw, w_head_group_range, w_dim_group_range, scales):

    assert len(qw.shape) == 4, "Only support 4D tensor with shape [batch, nheads, headdim, dstate]"
    batch, nheads, headdim, dstate = qw.shape

    assert len(w_head_group_range.shape) == 2, "w_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(w_dim_group_range.shape) == 3, "w_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    
    if qw.dtype != torch.float32:
        qw = qw.to(torch.float32)
    
    saved_device = qw.device
    saved_shape = qw.shape
    ngroups = w_head_group_range.shape[0]
    n_head_groups = w_head_group_range.shape[1]
    n_dim_groups = w_dim_group_range.shape[2]
    assert nheads % ngroups == 0
    assert ngroups == w_dim_group_range.shape[0]
    assert w_head_group_range.dtype == torch.int32 # [n_ssd_groups, n_head_groups]
    assert w_dim_group_range.dtype == torch.int32  # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert scales.shape == (ngroups, n_head_groups, n_dim_groups, dstate)

    qw = qw.reshape(batch, ngroups, nheads//ngroups, headdim, dstate)
    
    w = []
    for ds in range(dstate):
        wd = []
        qwd = qw[:, :, :, :, ds] # [b, ng, nh, hd, ds] -> [b, ng, nh, hd]
        scale_wd = scales[..., ds] # [ng, nh, hd, ds] -> [ng, nh, hd]
        for ssd_g in range(ngroups):
            wd_h = []
            h_start = 0
            for hg_idx in range(n_head_groups):
                wd_hg = []
                h_end = w_head_group_range[ssd_g][hg_idx].item()
                qwd_hg = qwd[:, ssd_g, h_start:h_end, :] # [b, ng, nh, hd] -> [b, gh, hd]
                d_start = 0
                for dg_idx in range(n_dim_groups):
                    d_end = w_dim_group_range[ssd_g][hg_idx][dg_idx].item()
                    qwd_hg_dg = qwd_hg[..., d_start:d_end]  # [b, gh, hd] -> [b, gh, gd]
                    wd_hg_dg = qwd_hg_dg * scale_wd[ssd_g, hg_idx, dg_idx] # scale back to float
                    wd_hg.append(wd_hg_dg) # [b, gh, gd]
                    d_start = d_end
                wd_hg = torch.cat(wd_hg, dim=-1) # [[b, gh, gd], [b, gh, gd], ...] -> [b, gh, hd]
                wd_h.append(wd_hg) # [[b, gh, hd], [b, gh, hd], ...]
                h_start = h_end
            wd_h = torch.cat(wd_h, dim=1) # [[b, gh, hd], [b, gh, hd], ...] -> [b, nh, hd]
            wd.append(wd_h.unsqueeze(1)) # [b, 1, nh, hd]
        wd = torch.cat(wd, dim=1) # [[b, 1, nh, hd], [b, 1, nh, hd], ...] -> [b, ng, nh, hd]
        w.append(wd.unsqueeze(-1)) # [[b, ng, nh, hd 1], [b, ng, nh, hd 1], ...]
    w = torch.cat(w, dim=-1).to(saved_device) # [[b, ng, nh, hd, 1], [b, ng, nh, hd, 1], ...] -> [b, ng, nh, hd, ds], we don't convert it to integer type here
    w = rearrange(w, "b g h d n -> b (g h) d n")
    assert w.shape == saved_shape
    return w.contiguous()
