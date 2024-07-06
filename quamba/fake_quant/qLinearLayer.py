import torch
import torch.nn as nn
from functools import partial
from quamba.fake_quant.quantUtils import dynamic_per_tensor_absmax_quantization

#W8A8BFP32OFP32Linear
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
    ):
        super().__init__()

        if  originalLayer.weight.dtype == torch.int8:
            raise NotImplementedError("Are you quantizing as Bits&Bytes in addition to another quantization setting? Dtype = int8 is not implemented for QLinearLayer. Try adding model layer to --skip_modules (ex: self_attn, mamba, moe) to ensure Bits&Bytes is not quantizing first.")

            
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
    
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
        
        self.weight_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
        
    @torch.no_grad()
    def forward(self, x):
        if not self.is_quant_mode:
            # Calibration mode
            return torch.functional.F.linear(x, self.weight, self.bias)
        else: 
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            # Using fake quantized weights for inference
            w_fq = self.fake_quant_weight
            out = torch.functional.F.linear(x, w_fq)
            if self.bias is not None:
                out = out + self.bias
            return out
            
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
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

    @property
    def fake_quant_weight(self):
        w_fake_quant = self.weight_quantizer(w=self.weight.clone())
        return w_fake_quant
        
    def __repr__(self):
        return f"QLinearLayer(in_features={self.in_features}, out_features={self.out_features})"
    