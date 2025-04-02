import numpy as np

import torch

MARLIN_QQQ_TILE = 16
MARLIN_QQQ_MIN_THREAD_N = 64
MARLIN_QQQ_MIN_THREAD_K = 128
MARLIN_QQQ_MAX_PARALLEL = 16
GPTQ_MARLIN_TILE = 16

SUPPORTED_NUM_BITS = [4, 8]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")

def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


@torch.no_grad()
def permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_n = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


@torch.no_grad()
def get_w4a8_permute_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    # Permute
    q_w = permute_weights(q_w, size_k, size_n, perm) # [k, n] -> [-1, 1024]

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=np.uint32)
    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << num_bits * i
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device) # [-1, 1024] -> [-1, 1024/pack_factor]

    return q_packed


# NOTE(HandH1998): QQQ employs different perms for per-group and per-channel weight quantization. # noqa: E501
@torch.no_grad()
def get_w4a8_weight_perm(num_bits: int, quant_type: str):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    assert quant_type in ["per-channel",
                          "per-group"], "not supported quantization type"
    if num_bits == 4:
        if quant_type == "per-channel":
            interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise Exception("num_bits must be 4, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


@torch.no_grad()
def get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


@torch.no_grad()
def w4a8_permute_scales(s_group, s_channel, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()

    return s_group, s_channel


# QQQ employs different quant schemes for per-group and
# per-channel quantization.
@torch.no_grad()
def get_w4a8_quantize_weights(w: torch.Tensor, num_bits: int, group_size: int = -1):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"
    assert num_bits in SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if group_size < size_k:
        # Reshape to [groupsize, -1]
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        s_group = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_group *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / s_group).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
        q_w += half_q_val # [0, 16]
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val).half() * s_group

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

        # Compute int8 quantization scale for each channel
        s_channel = torch.max(torch.abs(w_ref), 0, keepdim=True)[0]
        s_channel /= 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(1, -1).to(dtype=torch.float)

        # Fuse scales
        s_group = (s_group.reshape(-1, size_n).contiguous() /
                   s_channel).to(dtype=torch.half)
    else:
        assert group_size == size_k
        max_q_val = 2**(num_bits - 1) - 1

        # Compute scale for each channel
        s_channel = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_channel /= max_q_val

        # Quantize
        q_w = torch.round(w / s_channel).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w.half() * s_channel

        s_group = torch.tensor([], dtype=torch.half)
        # div 2 ** (8 - self.bits)) to offset right shift in unpacking
        s_channel /= (2**(8 - num_bits))
        s_channel = s_channel.reshape(-1, size_n).contiguous().to(torch.float)

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s_group.to(device=orig_device),
        s_channel.to(device=orig_device),
    )


@torch.no_grad()
def w4a8_quantize(
    w: torch.Tensor, num_bits: int,
    group_size: int = -1, s_group: torch.Tensor = None,
    s_channel: torch.Tensor = None, out_scales: torch.Tensor = None,
    pad_out: int = 0
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    quant_type = "per-channel" if group_size == size_k else "per-group"

    # Quantize
    if s_group is None or s_channel is None:
        w_ref, q_w, s_group, s_channel = get_w4a8_quantize_weights(w, num_bits, group_size)
    else: # s_group and s_channel are not None
        # w is already (fake) quantized
        w_ref = w.clone()
        # represent w in integer
        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2
        w = w.reshape((-1, group_size, size_n))
        q_w = ((w / s_group.unsqueeze(1)) + half_q_val).round().int()
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        q_w = q_w.reshape((size_k, size_n)).contiguous()
        s_channel = s_channel.to(dtype=torch.float) # per-group scale must be float32
        # Fuse scales
        s_group = (s_group.reshape(-1, size_n).contiguous() /
                    s_channel).to(dtype=torch.half)
        s_group = torch.nan_to_num(s_group, nan=0.0) # s_channel may be 0

    # fuse output scales to s_channel
    if out_scales is not None:
        s_channel = s_channel / out_scales.to(s_channel.device).to(dtype=torch.float)

    if pad_out !=0:    
        w_ref = torch.nn.functional.pad(w_ref, (0, pad_out, 0, 0), "constant", 0) # w_ref: [Din, Dout]
        q_w = torch.nn.functional.pad(q_w, (0, pad_out, 0, 0), "constant", 0) # q_w: [Din, Dout]
        s_group = torch.nn.functional.pad(s_group, (0, pad_out), "constant", 0) # s_group: [n_group, Dout]
        s_channel = torch.nn.functional.pad(s_channel, (0, pad_out), "constant", 1e-6) # s_channel: [1, Dout]
        size_k, size_n = w_ref.shape # new size after padding

    # weight permutation
    weight_perm = get_w4a8_weight_perm(num_bits, quant_type)
    marlin_qqq_q_w = get_w4a8_permute_weights(q_w, size_k, size_n, num_bits,
                                        weight_perm, group_size)
    marlin_qqq_s_group, marlin_qqq_s_channel = w4a8_permute_scales(
        s_group, s_channel, size_k, size_n, group_size)

    # Create result
    res_list = [
        w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel
    ]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


@torch.no_grad()
def get_w4a16_permute_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    # Permute
    q_w = permute_weights(q_w, size_k, size_n, perm) # [k, n] -> [-1, 1024]

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device) # [-1, 1024] -> [-1, 1024/pack_factor]

    return q_packed


@torch.no_grad()
def get_w4a16_quantize_weights(w: torch.Tensor, num_bits: int, group_size: int = -1):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"
    assert num_bits in SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if group_size < size_k:
        # Reshape to [groupsize, -1]
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        scale = torch.max(torch.abs(w), 0, keepdim=True)[0]
        scale *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / scale).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
        q_w += half_q_val # [0, 16]
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val).half() * scale

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)
        scale = scale.reshape(-1, size_n).contiguous()

    else:
        assert group_size == size_k
        max_q_val = 2**(num_bits - 1) - 1

        # Compute scale for each channel
        scale = torch.max(torch.abs(w), 0, keepdim=True)[0]
        scale /= max_q_val

        # Quantize
        q_w = torch.round(w / scale).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w.half() * scale

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        scale.to(device=orig_device),
    )


@torch.no_grad()
def get_w4a16_weight_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


@torch.no_grad()
def w4a16_permute_scales(scale, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        scale = scale.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        assert group_size == size_k
        scale = scale.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
    scale = scale.reshape((-1, size_n)).contiguous()

    return scale


@torch.no_grad()
def w4a16_quantize(
    w: torch.Tensor, num_bits: int,
    group_size: int = -1, scale: torch.Tensor = None,
    pad_out: int = 0
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    quant_type = "per-channel" if group_size == size_k else "per-group"

    # Quantize
    if scale is None:
        w_ref, q_w, scale = get_w4a16_quantize_weights(w, num_bits, group_size) # per-group

    else: # scale
        # w is already (fake) quantized
        w_ref = w.clone()
        # represent w in integer
        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2
        w = w.reshape((-1, group_size, size_n))
        q_w = ((w / scale.unsqueeze(1)) + half_q_val).round().int()
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        q_w = q_w.reshape((size_k, size_n)).contiguous()

    if pad_out != 0:    
        w_ref = torch.nn.functional.pad(w_ref, (0, pad_out, 0, 0), "constant", 0) # w_ref: [Din, Dout]
        q_w = torch.nn.functional.pad(q_w, (0, pad_out, 0, 0), "constant", 0) # q_w: [Din, Dout]
        scale = torch.nn.functional.pad(scale, (0, pad_out), "constant", 0) # scale: [n_group, Dout]
        size_k, size_n = w_ref.shape # new size after padding

    # weight permutation
    weight_perm = get_w4a16_weight_perms()
    marlin_qqq_q_w = get_w4a16_permute_weights(q_w, size_k, size_n, num_bits,
                                        weight_perm, group_size)
    marlin_scale = w4a16_permute_scales(
        scale, size_k, size_n, group_size)

    # Create result
    res_list = [
        w_ref, marlin_qqq_q_w, marlin_scale
    ]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list
