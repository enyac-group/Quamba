import random
import numpy as np

import torch
import torch.nn as nn

from quamba import HadW8A8BF16OF16Linear, W8A8BF16OF16Linear, HadLinear, Hadamard, QHadamard # editable installation
# from q_mamba import HadW8A8BF16OF16Linear, W8A8BF16OF16Linear, HadLinear, Hadamard, QHadamard

random.seed(0)
torch.manual_seed(0)
np.random.seed(seed=0)

bsize = 1
seqlen = 1
dim_in = 768*2
dim_out = 768
# dim_in = 2560*2
# dim_out = 2560
had_scale = 1.0/torch.tensor(dim_in).sqrt() # hadamard transform scaling factor

# the original linear layer
x = torch.rand((bsize, seqlen, dim_in)).to(torch.float16).cuda()
x[:, :, 0] *=1000 # assuming ch0 is the outlier channel
x[:, :, 10] *=1000 # assuming ch10 is the outlier channel
linear = torch.nn.Linear(dim_in, dim_out, bias=False, dtype=torch.float16).cuda()
y = linear(x)

linearH = HadLinear(linear)
had = Hadamard(dim_in)
x_H = had(x)
y_ = linearH(x_H)
assert torch.allclose(y, y_, rtol=1e-2, atol=1e-2)

x_scale = x.abs().max() / 127
q_linear = W8A8BF16OF16Linear(linear, x_scale).cuda()
# forward
qx = (x/x_scale).round().to(torch.int8) # quant
qy = q_linear(qx) # fp16
# assert torch.allclose(y, qy) # this will not pass...
r2 = (y - qy).pow(2).mean() / qy.pow(2).mean()
print("============================ >>>>>>>>>>> W8A8BF16OF16Linear diff: ", r2)

# initialize HadW8A8BF16OF16Linear and QHadamard layer
x_H_scale = x_H.abs().max() / 127
q_had = QHadamard(dim_in, x_H_scale)
q_linear = HadW8A8BF16OF16Linear(linear, x_H_scale).cuda()
# forward
# fused hadamard (hadamard + quantization)
qx = q_had(x)
qy = q_linear(qx) # fp16
# assert torch.allclose(y, qy) # this will not pass...
r2 = (y - qy).pow(2).mean() / qy.pow(2).mean()
print("============================ >>>>>>>>>>> HadW8A8BF16OF16Linear diff: ", r2)


# import time
# time.sleep(20)

q_had = QHadamard(dim_in, x_H_scale)
qx_ = q_had(x)
print(qx_.shape, qx_.dtype)

had = Hadamard(dim_in)
hx = had(x)
qx = (hx / x_H_scale).round().to(torch.int8) # quant

print(qx_)
print(qx)
r2 = (qx.float() - qx_.float()).pow(2).mean() / qx.float().pow(2).mean()
print("============================ >>>>>>>>>>> diff: ", r2)

