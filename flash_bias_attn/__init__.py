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
import importlib.util
from pathlib import Path
import torch
from .concat import flash_attn_bias_concat

try:
    import flash_attn_bias_cuda
    _HAS_CUDA = True
except ImportError:
    try:
        _repo_root = Path(__file__).resolve().parent.parent
        _candidates = sorted(_repo_root.glob("flash_attn_bias_cuda*.so"))
        if _candidates:
            _spec = importlib.util.spec_from_file_location("flash_attn_bias_cuda", _candidates[0])
            flash_attn_bias_cuda = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(flash_attn_bias_cuda)
            _HAS_CUDA = True
        else:
            _HAS_CUDA = False
    except Exception:
        _HAS_CUDA = False


class FlashAttnBiasFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias_table, window_size, softmax_scale):
        B, L_q, H, D = q.shape
        softmax_scale = softmax_scale or 1.0 / math.sqrt(D)
        half_w = window_size // 2
        ws_left = half_w
        ws_right = half_w - 1 + (window_size % 2)

        q, k, v = [x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v)]
        bt = bias_table.contiguous()

        out, lse = flash_attn_bias_cuda.flash_attn_bias_fwd(
            q, k, v, bt, softmax_scale, ws_left, ws_right,
        )

        ctx.save_for_backward(q, k, v, out, lse, bt)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.ws_left = ws_left
        ctx.ws_right = ws_right
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, bt = ctx.saved_tensors
        dout = dout.contiguous()

        dq, dk, dv, dbias = flash_attn_bias_cuda.flash_attn_bias_bwd(
            dout, q, k, v, out, lse, bt,
            ctx.softmax_scale, ctx.ws_left, ctx.ws_right, ctx.needs_input_grad[3],
        )
        return dq, dk, dv, dbias, None, None


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
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    return FlashAttnBiasFunc.apply(q, k, v, bias_table, window_size, softmax_scale)


__all__ = ["flash_attn_bias"]
__version__ = "0.1.0"
