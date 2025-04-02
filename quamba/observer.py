import numpy as np

import torch
import torch.nn as nn
from einops import rearrange

from .quant_utils import _get_quant_range

def _get_uniform_quantization_params(w_max, n_bits, clip_ratio):
    _, q_max = _get_quant_range(n_bits=n_bits, sym=True)
    if clip_ratio < 1.0:
        w_max = w_max * clip_ratio
    scales = w_max / q_max
    return scales

def _get_minmax_quantization_params(w_max, w_min, n_bits, clip_ratio, sym):
    q_min, q_max = _get_quant_range(n_bits=n_bits, sym=sym)
    if sym:
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        assert w_min is not None, "w_min should not be None for asymmetric quantization."
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        
    return scales.to(torch.float32).clamp(min=1e-6), base.to(torch.float32)


class PerTensorMinmaxObserver:
    def __init__(self, n_bits, clip_ratio, sym):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.w_max = None
        self.w_min = None
        self.sym = sym
        self.has_statistic = False

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        if self.sym:
            comming_max = w.abs().amax().clamp(min=1e-5)
        else:
            comming_max = w.amax()
            comming_min = w.amin()

        if self.w_max is None:
            self.w_max = comming_max
        else:
            self.w_max = torch.max(comming_max, self.w_max)
        
        if not self.sym:
            if self.w_min is None:
                self.w_min = comming_min
            else:
                self.w_min = torch.min(comming_min, self.w_min)
        
        
    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio,
            sym=self.sym
        )
        

class PerTensorPercentileObserver:
    def __init__(self, n_bits, clip_ratio, sym,
                 percentile_sigma=0.01, percentile_alpha=0.99999):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.sym = sym
        self.w_max = None
        self.w_min = None
        self.has_statistic = False
        self.percentile_sigma = percentile_sigma
        self.percentile_alpha = percentile_alpha

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        w = w.clone().to(torch.float32) # quantile() input must be float
        if self.sym:
            cur_max = torch.quantile(w.abs().reshape(-1), self.percentile_alpha)
        else:
            cur_max = torch.quantile(w.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(w.reshape(-1),
                                        1.0 - self.percentile_alpha)

        if self.w_max is None:
            self.w_max = cur_max
        else:
            self.w_max = self.w_max + self.percentile_sigma * (cur_max - self.w_max)

        if not self.sym:
            if self.w_min is None:
                self.w_min = cur_min
            else:
                self.w_min = self.w_min + self.percentile_sigma * (cur_min - self.w_min)

    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            sym=self.sym,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio
        )
        

class PerSSDGroupObserver:
    def __init__(self, n_bits, clip_ratio, sym, dstate):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.sym = sym
        self.dstate = dstate
        self.w_max = None
        self.w_min = None
        self.has_statistic = False

    def update(self, w):
        self.has_statistic = True

        # the activation should have shape [bsize, seqlen, dim],
        assert w.dim() == 3, "PerSSDGroupObserver only support 3d tensor, please handle the shape outside."

        # and the coming_max have shape [bsize, nh, 1]
        w = rearrange(w.clone(), "b l (g d) -> b l g d", d=self.dstate)
        wt = w.transpose(1, 2) # [b, l, g, d] -> [b, g, l, d]
        b, g, l, d = wt.shape
        
        wt = wt.transpose(0, 1) # [b, g, l, d] -> [g, b, l, d]
        wt_reshape = wt.reshape(g, -1) # [g, b, l, d] -> [g, b*l*d]
        if self.sym:
            coming_max = wt_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        else:
            coming_max = wt_reshape.amax(dim=-1, keepdim=True)
            coming_min = wt_reshape.amin(dim=-1, keepdim=True)

        if self.w_max is None:
            self.w_max = coming_max
        else:
            self.w_max = torch.max(coming_max, self.w_max)
        
        if not self.sym:
            if self.w_min is None:
                self.w_min = coming_min
            else:
                self.w_min = torch.min(coming_min, self.w_min)

    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        scales, base = _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            sym=self.sym,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio
        )
        return scales.flatten(), base.flatten()


class CrossHeadMinmaxObserver:

    def __init__(self, n_bits, clip_ratio, sym, ngroups, headdim, head_groups, channel_group):

        assert len(head_groups) == ngroups, f"Number of head groups must be equal to ngroups, " \
            f"but got {len(head_groups)} vs. {ngroups}"
        assert len(channel_group) == ngroups, f"Number of head groups must be equal to ngroups, " \
            f"but got {len(channel_group)} vs. {ngroups}"
        
        for i in range(ngroups):
            for j in range(len(head_groups[i])):
                assert sum(channel_group[i][j]) == headdim, f"The sum of channel_group must be equal to headdim, " \
                    f"but got {sum(channel_group[i][j])} vs. {headdim}"
        
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio

        self.sym = sym
        self.num_ssd_group = ngroups
        self.headdim = headdim
        self.has_statistic = False
        # Number of channel to be kept
        self.head_groups = head_groups # head_group is a list
        self.num_head_group = len(head_groups[0])
        self.num_channel_group = len(channel_group[0][0])
        self.channel_group = channel_group
        self.w_max = np.full((self.num_ssd_group, self.num_head_group, self.num_channel_group), None).tolist() # [ssd_group x head_groups x dim_groups]
        self.w_min = np.full((self.num_ssd_group, self.num_head_group, self.num_channel_group), None).tolist() # [ssd_group x head_groups x dim_groups]

    def update(self, w):
        self.has_statistic = True

        # the activation should have shape [bsize, seqlen, dim],
        assert w.dim() == 3, "CrossHeadMinmaxObserver only support 3d tensor, please handle the shape outside."

        # and the coming_max have shape [bsize, nh, 1]
        w = rearrange(w.clone().to(torch.float32), "b l (g h p) -> b l g h p", g=self.num_ssd_group, p=self.headdim)
        # wt = w.transpose(1, 2) # [b, l, nh, hd] -> [b, nh, l, hd]
        # b, nh, l, hd = wt.shape

        for g in range(self.num_ssd_group):
            w_ssd_group = w[:, :, g, :, :] # [b, l, nh, hd]
            h_start = 0
            for h_gidx, h_gsize in enumerate(self.head_groups[g]):
                ch_start = 0
                for ch_gidx, ch_gsize in enumerate(self.channel_group[g][h_gidx]):
                    w_group = w_ssd_group[:, :, h_start:h_start + h_gsize, ch_start:ch_start + ch_gsize] # [b, l, gh, gd]
                    if self.sym:
                        coming_max = w_group.abs().amax().clamp(min=1e-5)
                        self.w_max[g][h_gidx][ch_gidx] = coming_max if self.w_max[g][h_gidx][ch_gidx] is None else torch.max(coming_max, self.w_max[g][h_gidx][ch_gidx])
                    else:
                        coming_max = w_group.amax()
                        coming_min = w_group.amin()  
                        self.w_max[g][h_gidx][ch_gidx] = coming_max if self.w_max[g][h_gidx][ch_gidx] is None else torch.max(coming_max, self.w_max[g][h_gidx][ch_gidx])
                        self.w_min[g][h_gidx][ch_gidx] = coming_min if self.w_min[g][h_gidx][ch_gidx] is None else torch.min(coming_min, self.w_min[g][h_gidx][ch_gidx])
                    ch_start = ch_start + ch_gsize
                h_start = h_start + h_gsize
        
        
    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        ssd_group_scales = []
        ssd_group_base = []
        for g in range(self.num_ssd_group):
            head_group_scales = []
            head_group_base = []
            for h_gidx, h_gsize in enumerate(self.head_groups[g]):
                ch_gsize = self.channel_group[g][h_gidx]
                h_gscales = []
                h_gbase = []
                for w_max, w_min in zip(self.w_max[g][h_gidx], self.w_min[g][h_gidx]):
                    scales, base = _get_minmax_quantization_params(
                        w_max=w_max,
                        w_min=w_min,
                        n_bits=self.n_bits,
                        clip_ratio=self.clip_ratio,
                        sym=self.sym
                    )
                    h_gscales.append(scales)
                    h_gbase.append(base)
                h_gscales = torch.stack(h_gscales, dim=0)
                h_gbase = torch.stack(h_gbase, dim=0)
                head_group_scales.append((h_gsize, ch_gsize, h_gscales))
                head_group_base.append((h_gsize, ch_gsize, h_gbase))
            ssd_group_scales.append(head_group_scales)
            ssd_group_base.append(head_group_base)
        # HY: in order to unify the api (scale, base) = act_scales.get(name), 
        #     I chose to return a tuple with scales and base, where
        #     scales: list = [
        #       # ssd_group 0
        #       [
        #         (hg1_size, [ch1_gsize, ch2_gsize, ...], [ch1_scale, ch2_scale, ...]),
        #         (hg2_size, [ch1_gsize, ch2_gsize, ...], [ch1_scale, ch2_scale, ...]),
        #       ],
        #       # ssd_group 1
        #       [
        #         ...
        #       ],
        #     ]
        #     bases: list = [
        #       # ssd_group 0
        #       [
        #         (hg1_size, [ch1_gsize, ch2_gsize, ...], [ch1_base, ch2_base, ...]),
        #         (hg2_size, [ch1_gsize, ch2_gsize, ...], [ch1_base, ch2_base, ...]),
        #       ],
        #       # ssd_group 1
        #       [
        #         ...
        #       ],
        #     ]
        #     hg: head group, ch_g: channel group
        return (ssd_group_scales, ssd_group_base)



class CachedStatesCrossHeadMinmaxObserver:

    def __init__(self, n_bits, clip_ratio, sym, ngroups, headdim, dstate, head_groups, channel_group):

        assert len(head_groups) == ngroups, f"Number of head groups must be equal to ngroups, " \
            f"but got {len(head_groups)} vs. {ngroups}"
        assert len(channel_group) == ngroups, f"Number of head groups must be equal to ngroups, " \
            f"but got {len(channel_group)} vs. {ngroups}"
        
        for i in range(ngroups):
            for j in range(len(head_groups[i])):
                assert sum(channel_group[i][j]) == headdim, f"The sum of channel_group must be equal to headdim, " \
                    f"but got {sum(channel_group[i][j])} vs. {headdim}"
        
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio

        self.sym = sym
        self.num_ssd_group = ngroups
        self.headdim = headdim
        self.dstate = dstate
        self.has_statistic = False
        # Number of channel to be kept
        self.head_groups = head_groups # head_group is a list
        self.num_head_group = len(head_groups[0])
        self.num_channel_group = len(channel_group[0][0])
        self.channel_group = channel_group
        self.w_max = np.full((self.num_ssd_group, self.num_head_group, self.num_channel_group), None).tolist() # [ssd_group x head_groups x dim_groups x dstate]
        self.w_min = np.full((self.num_ssd_group, self.num_head_group, self.num_channel_group), None).tolist() # [ssd_group x head_groups x dim_groups x dstate]

    def update(self, w):
        self.has_statistic = True

        # the activation should have shape [bsize, nhead, headdim, dstate],
        assert w.dim() == 4, "CachedStatesCrossHeadMinmaxObserver only support 4d tensor, please handle the shape outside."
        w = rearrange(w.clone().to(torch.float32), "b (g h) d p -> b g h d p", g=self.num_ssd_group)

        for g in range(self.num_ssd_group):
            w_ssd_group = w[:, g, :, :, :] # [b, ng, nh, hd, ds] -> [b, nh, hd, ds]
            h_start = 0
            for h_gidx, h_gsize in enumerate(self.head_groups[g]):
                ch_start = 0
                for ch_gidx, ch_gsize in enumerate(self.channel_group[g][h_gidx]):
                    w_group = w_ssd_group[:, h_start:h_start + h_gsize, ch_start:ch_start + ch_gsize, :] # [b, nh, hd, ds] -> [b, gh, gd, ds]
                    if self.sym:
                        coming_max = w_group.abs().amax(dim=[0, 1, 2]).clamp(min=1e-5) # [ds]
                        if self.w_max[g][h_gidx][ch_gidx] is None:
                            self.w_max[g][h_gidx][ch_gidx] = coming_max # [ds]
                        else:
                            self.w_max[g][h_gidx][ch_gidx] = torch.max(coming_max, self.w_max[g][h_gidx][ch_gidx])
                    else:
                        coming_max = w_group.amax(dim=[0, 1, 2]) # [ds]
                        if self.w_max[g][h_gidx][ch_gidx] is None:
                            self.w_max[g][h_gidx][ch_gidx] = coming_max # [ds]
                        else:
                            self.w_max[g][h_gidx][ch_gidx] = torch.max(coming_max, self.w_max[g][h_gidx][ch_gidx])
                        coming_min = w_group.amin(dim=[0, 1, 2]) # [ds]
                        if self.w_min[g][h_gidx][ch_gidx] is None:
                            self.w_min[g][h_gidx][ch_gidx] = coming_min # [ds]
                        else:
                            self.w_min[g][h_gidx][ch_gidx] = torch.min(coming_min, self.w_min[g][h_gidx][ch_gidx])
                    ch_start = ch_start + ch_gsize
                h_start = h_start + h_gsize
        
    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        ssd_group_scales = []
        ssd_group_base = []
        for g in range(self.num_ssd_group):
            head_group_scales = []
            head_group_base = []
            for h_gidx, h_gsize in enumerate(self.head_groups[g]):
                ch_gsize = self.channel_group[g][h_gidx]
                h_gscales = []
                h_gbase = []
                for ch_gidx, ch_gsize in enumerate(self.channel_group[g][h_gidx]):
                    w_max, w_min = self.w_max[g][h_gidx][ch_gidx], self.w_min[g][h_gidx][ch_gidx]
                    scales, base = _get_minmax_quantization_params(
                        w_max=w_max,
                        w_min=w_min,
                        n_bits=self.n_bits,
                        clip_ratio=self.clip_ratio,
                        sym=self.sym
                    )
                    h_gscales.append(scales)
                    h_gbase.append(base)
                h_gscales = torch.stack(h_gscales, dim=0)
                h_gbase = torch.stack(h_gbase, dim=0)
                head_group_scales.append((h_gsize, ch_gsize, h_gscales))
                head_group_base.append((h_gsize, ch_gsize, h_gbase))
            ssd_group_scales.append(head_group_scales)
            ssd_group_base.append(head_group_base)
        # HY: in order to unify the api (scale, base) = act_scales.get(name), 
        #     I chose to return a tuple with scales and base, where
        #     scales: list = [
        #       # ssd_group 0
        #       [
        #         (hg1_size, [ch1_gsize, ch2_gsize, ...], [ch1_scale, ch2_scale, ...]),
        #         (hg2_size, [ch1_gsize, ch2_gsize, ...], [ch1_scale, ch2_scale, ...]),
        #       ],
        #       # ssd_group 1
        #       [
        #         ...
        #       ],
        #     ]
        #     bases: list = [
        #       # ssd_group 0
        #       [
        #         (hg1_size, [ch1_gsize, ch2_gsize, ...], [ch1_base, ch2_base, ...]),
        #         (hg2_size, [ch1_gsize, ch2_gsize, ...], [ch1_base, ch2_base, ...]),
        #       ],
        #       # ssd_group 1
        #       [
        #         ...
        #       ],
        #     ]
        #     hg: head group, ch_g: channel group
        return (ssd_group_scales, ssd_group_base)
