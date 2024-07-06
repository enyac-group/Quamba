import torch
import torch.nn as nn
from functools import partial
from quamba.fake_quant.quantUtils import (
    bached_dynamic_per_tensor_absmax_quantization, 
    dynamic_per_token_absmax_quantization,
    uniform_affine_fake_quantization,
)


class QAct(nn.Module):
    def __init__(
        self,
        tensor_name
    ):
        super().__init__()
        self.act_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
        self.is_sym = None
        self.is_static_quant = None
        self.tensor_name = tensor_name
        self.register_buffer("a_scales", None)
        
    @torch.no_grad()
    def forward(self, x):
        if not self.is_quant_mode:  
            # For calibration purpose
            return x
        else:
            # For quantized mode
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            # then, re-quant them to the target scale             
            return self.act_quantizer(x)
    
    def configure(self, 
            n_bits,
            sym,
            o_scales = None,
            o_base = None,
            clip_ratio = 1.0,
            static_quant=True,
            quantization_type = "per_tensor"
        ):
        self.is_configured = True
        #NOTE(brian1009): Do no quantization when bit width is larger than 16
        if n_bits >= 16:
            self.act_quantizer = lambda x: x
            return
            
        self.is_static_quant = static_quant
        self.is_sys = sym
        if self.is_static_quant:
            assert (o_scales is not None or o_base is None), "Static quantization requires scales/base to be provided"
            self.act_quantizer = partial(
                uniform_affine_fake_quantization,
                n_bits=n_bits,
                sym=sym,
                scales=o_scales,
                base=o_base
            )
        else:
            if quantization_type == "per_tensor":
                self.act_quantizer = partial(
                    bached_dynamic_per_tensor_absmax_quantization,
                    n_bits=n_bits,
                    sym=sym,
                    clip_ratio=clip_ratio
                )
            elif quantization_type == "per_token":
                self.act_quantizer = partial(
                    dynamic_per_token_absmax_quantization,
                    n_bits=n_bits,
                    sym=sym,
                    clip_ratio=clip_ratio,
                )
            else:
                raise ValueError(f"Invalid activation quantization type: {quantization_type}")
        

    def __repr__(self):
        return f"QAct()"


    
    
    