"""
This file is a modified version of the original file from the GPTQ repo.
https://github.com/IST-DASLab/gptq
"""
import math
import time
import gc

import torch
import torch.nn as nn
import transformers

from quamba.qLinearLayer import HadLinear
from quamba.datatype_utils import fake_quantize_with_type, get_datatypes, get_type_scales, get_quant_value_from_dtype_lists

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# class Quantizer_GPTQ(nn.Module):
#     def __init__(self, shape=1):
#         super(Quantizer_GPTQ, self).__init__()

#     def configure(self, bits, sym=True, clip_ratio=1.0, data_type='int'):
#         self.bits = bits
#         self.datatype_list = get_datatypes(data_type, bits)
#         quant_value_list = get_quant_value_from_dtype_lists(self.datatype_list)
#         quant_value_tensor_list = [torch.tensor(quant_value) for quant_value in quant_value_list]
#         self.quant_values_tensor = torch.stack(quant_value_tensor_list)  # Shape: (num_qsets, num_qvalues)
#         #logging.info(f"Set Quantization values in GPTQ_quantizer: {self.quant_values_tensor}")
#         # HY: Asymmetric quantization has not been tested yet
#         self.sym = sym
#         self.clip_ratio = clip_ratio

#     @torch.no_grad()
#     def find_params(self, x, dtype):
#         dev = x.device
#         self.quant_values_tensor = self.quant_values_tensor.to(dev)

#         # Reshape x to 2D tensor (num_rows, num_elements)
#         original_shape = x.shape
#         x_flat = x.reshape(-1, x.shape[-1])  # Shape: (num_rows, num_elements)
#         num_rows = x_flat.shape[0]

#         # Compute scales and zero points for each quantization set
#         scale, zero = get_type_scales(x_flat, self.quant_values_tensor, self.sym)  # Shape: (3, num_rows, 1)

#         # Quantize and dequantize x with each quantization set
#         x_dq = fake_quantize_with_type(x_flat, scale, zero, self.quant_values_tensor, dtype=dtype)  # Shape: (3, num_rows, num_elements)

#         # Compute MSE between original and quantized x with each quantization set
#         mse = ((x_dq - x_flat.unsqueeze(0)) ** 2).mean(dim=2)  # Shape: (3, num_rows)
#         # Select the quantization set with minimal MSE for each row
#         _, best_qset_indices = mse.min(dim=0)  # Shape: (num_rows)
        
#         # Gather the best scales, zero points, and quantization values for each row
#         best_scale = scale[best_qset_indices, torch.arange(num_rows)]  # Shape: (num_rows, 1)
#         best_zero = zero[best_qset_indices, torch.arange(num_rows)]    # Shape: (num_rows, 1)

#         # Store the parameters
#         scale = best_scale.view(*original_shape[:-1], 1)  # Reshape to match x
#         zero = best_zero.view(*original_shape[:-1], 1)
#         torch.cuda.empty_cache()
#         return scale, zero, best_qset_indices

@torch.no_grad()
def get_per_channel_scale(w, num_bits=4):
    # HY: use the simple quant in qqq_quantize_weights
    max_q_val = 2**num_bits - 1 # 15
    # Compute scale for each output channel
    s = torch.max(torch.abs(w), 1, keepdim=True)[0] # w: [Dout, Din] 
    s *= 2 / max_q_val  # 2 => symmetric, 2 / 15 # s: [Dout, 1] 
    return s

@torch.no_grad()
def quant(w, s, num_bits=4):
    # HY: use the simple quant in qqq_quantize_weights
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2
    # w: [Dout, Din], s: [Dout, 1]
    q_w = torch.round(w / s).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
    q_w += half_q_val # [0, 16]
    q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
    # Compute ref (dequantized)
    w_ref = (q_w - half_q_val) * s
    return w_ref.half()

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        # HY: To save memory, we use float16 to compute distances for non-uniform quantization
        #     see datatype_utils.py for def fake_quantize_with_type 
        self.gptq_dtype = torch.float16
        W = layer.weight.data.clone()

        if not isinstance(self.layer, (nn.Linear, HadLinear)):
            raise NotImplementedError("Only HadLinear and nn.Linear is supported for now")
        
        self.out_dim = W.shape[0] # out dim, row
        self.in_dim = W.shape[1] # in dim, columns
        self.H = torch.zeros((self.in_dim, self.in_dim), device=self.dev) 
        self.nsamples = 0 
        del W

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] 
        if isinstance(self.layer, (nn.Linear, HadLinear)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 
        else:
            raise NotImplementedError("Only HadLinear and nn.Linear is supported for now")
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def fasterquant(self, group_size=128, percdamp=.01, w_bits=4, dtype=torch.float32):
        bits = w_bits # 4-bit quantization
        W = self.layer.weight.data.clone().to(dtype)
        device = W.device
        
        # preprocess H
        H = self.H.clone().to(torch.float32)
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.in_dim, device=self.dev)
        H[diag, diag] += damp 
        # cholesky must be torch.float32
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True).to(dtype)
        Hinv = H
        
        # init Losses and Q
        assert group_size <= self.in_dim
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        n_groups = math.ceil(self.in_dim / group_size)
        group_scale = torch.zeros(n_groups, self.out_dim, dtype=torch.float32, device=device)   # QQQ requires [n_groups, out_dim]
        for i1 in range(0, self.in_dim, group_size):
            gidx = i1 // group_size
            i2 = min(i1 + group_size, self.in_dim)
            count = i2 - i1
            # get weight group
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            # Get per-group scale and zero
            per_group_scale = get_per_channel_scale(W1, num_bits=bits)
            group_scale[gidx] = per_group_scale.squeeze()
            for i in range(count):
                w = W1[:, i].clone() # [Dout]
                d = Hinv1[i, i]
                q = quant(w.unsqueeze(1), per_group_scale, num_bits=bits).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        torch.cuda.synchronize()

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype) # fake quantized weight
        self.layer.weight.data = Q.contiguous() # [Dout, Din]
        self.layer.apply_gptq = True
        self.layer.bits = bits
        self.layer.group_size = group_size
        self.layer.group_scale = group_scale    # QQQ requires [n_groups, out_dim]

        del Losses
        del H
        del W
        torch.cuda.empty_cache()

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        gc.collect()