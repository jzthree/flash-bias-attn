"""
flash-bias-attn: Fast exact attention with per-head relative position bias.

Modifies Flash Attention's CUDA kernel to add a Toeplitz bias table lookup
in the inner loop. Exact computation, ~1.5x overhead vs no-bias flash attention.

Usage:
    from flash_bias_attn import flash_attn_bias

    output = flash_attn_bias(q, k, v, bias_table, window_size, softmax_scale=None)

    q, k, v     : (batch, seqlen, nheads, headdim)  float16/bfloat16
    bias_table  : (nheads, window_size)              float32 or same as q
    window_size : int
"""

import math
import torch

try:
    import flash_attn_bias_cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


def flash_attn_bias(q, k, v, bias_table, window_size, softmax_scale=None):
    """
    Flash attention with per-head relative position bias (windowed).

    Adds bias_table[j - i + window_size//2] to attention scores for
    positions within the window. Uses a modified Flash Attention CUDA kernel
    for exact computation with minimal overhead.

    Args:
        q: (batch, seqlen_q, nheads, headdim) float16 or bfloat16
        k: (batch, seqlen_k, nheads_k, headdim) float16 or bfloat16
        v: (batch, seqlen_k, nheads_k, headdim) float16 or bfloat16
        bias_table: (nheads, window_size) — per-head relative position bias
        window_size: int — symmetric window size
        softmax_scale: float, default 1/sqrt(headdim)

    Returns:
        output: (batch, seqlen_q, nheads, headdim) same dtype as q
    """
    if not _HAS_CUDA:
        raise RuntimeError(
            "flash_attn_bias_cuda not found. Build with: "
            "CC=gcc CXX=g++ python setup.py build_ext --inplace"
        )

    B, L_q, H, D = q.shape
    _, L_k, _, _ = k.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(D)
    half_w = window_size // 2

    # Ensure contiguous
    q, k, v = [x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v)]

    # bias_table to float32 contiguous
    if bias_table.dtype != torch.float32:
        bias_table = bias_table.float()
    bias_table = bias_table.contiguous()

    out, lse = flash_attn_bias_cuda.flash_attn_bias_fwd(
        q, k, v, bias_table, softmax_scale,
        half_w,                          # window_size_left
        half_w - 1 + (window_size % 2),  # window_size_right
    )
    return out


__all__ = ["flash_attn_bias"]
__version__ = "0.1.0"
