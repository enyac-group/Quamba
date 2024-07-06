import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from quamba.fake_quant.quantUtils import dynamic_per_tensor_absmax_quantization


class QConv1D(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Conv1d,
    ):
        super().__init__()
        self.register_buffer('weight', originalLayer.weight)
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None


        # Copy convolution-specific parameters
        self.in_channels = originalLayer.in_channels
        self.out_channels = originalLayer.out_channels
        self.kernel_size = originalLayer.kernel_size
        self.stride = originalLayer.stride
        self.padding = originalLayer.padding
        self.dilation = originalLayer.dilation
        self.groups = originalLayer.groups
        
        # Quantization Related 
        self.weight_quantizer = None
        self.is_configured = False
        self.is_quant_mode = False
    
    @property
    def fake_quant_weight(self):
        if not self.is_quant_mode:
            return self.weight
        w_fake_quant = self.weight_quantizer(w=self.weight.clone())
        return w_fake_quant

    @torch.no_grad()
    def forward(self, x, i_scales = torch.tensor(1.0)):
        if not self.is_quant_mode:
            #calibration mode
            return F.conv1d(
                x, 
                self.weight, 
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        else: 
            assert self.is_configured, "Please run the configure() first, before running in quant mode"
            #quantized mode
            w_fq = self.fake_quant_weight
            out = F.conv1d(
                            x,
                            w_fq,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups
                        )
            if self.bias is not None:
                out = out + self.bias.reshape(-1, 1)
            return out

    def to(self, *args, **kwargs):
        super(QConv1D, self).to(*args, **kwargs)
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
            
    def __repr__(self):
        return f"Conv1d({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"

    