import torch
import torch.nn as nn
from .quantUtils import _get_quant_range

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
        
    return scales, base


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


def build_observer(observer_type, n_bits, clip_ratio,sym,
        percentile_sigma=0.01, percentile_alpha=0.99999
    ):
    if observer_type == "PerTensorMinmaxObserver":
        return PerTensorMinmaxObserver(n_bits, clip_ratio, sym)
    elif observer_type == "PerTensorPercentileObserver":
        return PerTensorPercentileObserver(
            n_bits, clip_ratio, sym, 
            percentile_sigma=percentile_sigma, percentile_alpha=percentile_alpha
        )
    # elif observer_type == "PerTensorOmseObserver":
    #     return PerTensorOmseObserver(*args, **kwargs)
    else:
        raise ValueError("Invalid observer type")