import math
import fast_hadamard_transform_cuda
import quant_hadamard

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def get_had_fn(dim):
    had_scale = 1.0 / math.sqrt(dim) # hadamard transform scaling factor
    if dim % 12 == 0:
        assert (is_pow2(dim // 12))
        N = 12
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_12N
    elif dim % 40 == 0:
        assert (is_pow2(dim // 40))
        N = 40
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_40N
    else:
        assert (is_pow2(dim))
        N = 2
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform
    return transform_fn, N, had_scale

def had_transform(w):
    saved_shape = w.shape
    dim = saved_shape[-1]
    transform_fn, N, had_scale = get_had_fn(dim)
    w_H = transform_fn(w.reshape(-1, dim), had_scale)
    if w_H.shape != saved_shape:
        w_H = w_H.reshape(saved_shape)
    return w_H

def get_qhad_fn(dim):
    had_scale = 1.0 / math.sqrt(dim) # hadamard transform scaling factor
    if dim % 12 == 0:
        assert (is_pow2(dim // 12))
        N = 12
        transform_fn = quant_hadamard.fast_hadamard_transform_12N
    elif dim % 40 == 0:
        assert (is_pow2(dim // 40))
        N = 40
        transform_fn = quant_hadamard.fast_hadamard_transform_40N
    else:
        assert (is_pow2(dim))
        N = 2
        transform_fn = quant_hadamard.fast_hadamard_transform
    return transform_fn, N, had_scale

