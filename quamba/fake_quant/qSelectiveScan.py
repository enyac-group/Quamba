"""
The code is modfied from
https://github.com/state-spaces/mamba
"""

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

try:
    import quant_sscan_cuda
except ImportError:
    print("[WARNING] Failed to import quant_sscan_cuda, please run `pip install -e .`")
    quant_sscan_cuda = None

from quamba.fake_quant.quantUtils import dynamic_per_tensor_absmax_quantization
from quamba.fake_quant.qActLayer import QAct
from quamba.fake_quant.rotation_utils import HadamardTransform
from quamba.fake_quant.smooth_quant_utils import SmoothModule

#Fake W8A8B8O8 QSScan
class QSScan(nn.Module):
    def __init__(self, A_log, D, dt_bias=None, delta_softplus = True):
        super().__init__()
        self.register_buffer('A_log', A_log)
        if D is not None:
            self.register_buffer('D', D)
        else:
            self.D = None
        if dt_bias is not None:
            self.register_buffer('dt_bias', dt_bias)
        else:
            self.dt_bias = None
        self.delta_softplus = delta_softplus
        self.is_quant_mode = False
        self.forward = self.fake_qssm_forward
        self.weight_quantizer = None
        self.ssm_out_quant = QAct(tensor_name="ssm_out_quant")
        self.ssm_out_had = HadamardTransform()
        self.ssm_out_smooth = SmoothModule(weight_to_smooth="out_proj", tensor_name="ssm_out_smooth")

    @property
    def quant_A_log(self):
        return self.weight_quantizer(w=self.A_log.clone())

    @property
    def quant_D(self):
        return self.weight_quantizer(w=self.D.clone())
    
    @torch.no_grad()
    def fake_qssm_forward(self, u, dt, B, C,
                z=None, return_last_state=False):

        if not self.is_quant_mode:
            A_log = self.A_log
            D = self.D
            dt_bias = self.dt_bias
        else:
            # quant A_log and dequant
            A_log = self.weight_quantizer(w=self.A_log.clone())

            # quant D and dequant
            D = None
            if self.D is not None:
                D = self.weight_quantizer(w=self.D.clone())
            
            # quant dt_bias and dequant
            dt_bias = None
            if self.dt_bias is not None:
                dt_bias = self.weight_quantizer(w=self.dt_bias.clone())

        A = -torch.exp(A_log)
        # use cuda sscan. selective_scan_ref is only for debugging and demonstration
        y = selective_scan_fn(
        #y = selective_scan_ref(
            u,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=return_last_state,
        )
        
        if return_last_state:
            y, last_state = y
        else:
            last_state = None
        
        y = rearrange(y, "b d l -> b l d")
        #NOTE(brian1009): Do hadamard transformer here
        y = self.ssm_out_smooth(y)
        y = self.ssm_out_had(y)
        y = self.ssm_out_quant(y) #NOTE(brian1009): Assuming y is in FP32         
        #NOTE(brian1009): Uncommment this, if you have not fused the second Hadamard transform into the following layer
        #y = self.ssm_out_had(y, transpose=False)
        return (y, last_state) if return_last_state else y

    def to(self, *args, **kwargs):
        super(QSScan, self).to(*args, **kwargs)
        self.A_log = self.A_log.to(*args, **kwargs)
        if self.D is not None:
            self.D = self.D.to(*args, **kwargs)
        if self.dt_bias is not None:
            self.dt_bias = self.dt_bias.to(*args, **kwargs)
        return self
    
    def configure(self,
            n_bits,
            clip_ratio=1.0,
        ):
        self.weight_quantizer = partial(
            dynamic_per_tensor_absmax_quantization,
            n_bits = n_bits,
            clip_ratio = clip_ratio,
            sym=True
        )
        self.is_configured = True

    def __repr__(self):
        return f"QSScan()"