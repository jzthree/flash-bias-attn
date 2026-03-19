import math
import warnings

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False


_CONCAT_DEPRECATION_MESSAGE = (
    "flash_attn_bias_concat is deprecated and retained only as an experimental "
    "baseline. For trainable relative-position bias it showed large "
    "approximation error at practical ranks; prefer flash_attn_bias for exact "
    "behavior."
)


def _pad_last_dim(x, multiple):
    pad = (-x.shape[-1]) % multiple
    if pad == 0:
        return x
    return F.pad(x, (0, pad))


def flash_attn_bias_concat(
    q,
    k,
    v,
    bias_table,
    window_size,
    softmax_scale=None,
    rank=None,
    pad_to_multiple=8,
):
    """
    Deprecated experimental approximation for windowed relative-position bias.

    This concatenates Fourier position features onto Q and K, then dispatches
    to flash-attn's CUDA kernel unchanged.

    Args:
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        v: (batch, seqlen, nheads, headdim)
        bias_table: (nheads, window_size)
        window_size: int
        softmax_scale: float, defaults to 1 / sqrt(headdim)
        rank: total number of added concat dimensions. Rounded down to an
            even number of cosine/sine pairs. `None` keeps all Fourier modes,
            which is exact for small windows but can exceed flash-attn's
            supported head dimension on large windows.
        pad_to_multiple: zero-pad the augmented head dimension to this multiple
            before calling flash-attn.
    """
    warnings.warn(_CONCAT_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    if not _HAS_FLASH_ATTN:
        raise RuntimeError(
            "flash_attn not found. Install flash-attn to use flash_attn_bias_concat."
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all have shape (batch, seqlen, nheads, headdim)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    if bias_table.ndim != 2:
        raise ValueError("bias_table must have shape (nheads, window_size)")
    if q.shape[2] != bias_table.shape[0]:
        raise ValueError("bias_table.shape[0] must match nheads")
    if bias_table.shape[1] != window_size:
        raise ValueError("bias_table.shape[1] must equal window_size")
    if q.device != k.device or q.device != v.device or q.device != bias_table.device:
        raise ValueError("q, k, v, and bias_table must be on the same device")

    bsz, seqlen, nheads, headdim = q.shape
    dtype = q.dtype
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
    half_w = window_size // 2
    ws_right = half_w - 1 + (window_size % 2)

    bt_float = bias_table.float()
    bhat = torch.fft.rfft(bt_float, dim=-1)
    n_freq = bhat.shape[-1]

    if rank is None:
        top_idx = torch.arange(n_freq, device=q.device)
    else:
        n_keep = min(n_freq, max(1, rank // 2))
        with torch.no_grad():
            avg_mag = bhat.abs().mean(dim=0)
            top_idx = avg_mag.topk(n_keep).indices.sort().values

    n_keep = top_idx.shape[0]
    aug_dims = 2 * n_keep
    padded_dim = headdim + aug_dims
    if pad_to_multiple > 1:
        padded_dim += (-padded_dim) % pad_to_multiple
    if padded_dim > 256:
        raise ValueError(
            f"Augmented head dim {padded_dim} is too large for typical flash-attn builds. "
            f"Reduce rank or window_size."
        )

    pos = torch.arange(seqlen, device=q.device, dtype=torch.float32)
    freqs = top_idx.to(torch.float32)
    angles_q = 2 * math.pi * pos[:, None] * freqs[None, :] / window_size
    angles_k = 2 * math.pi * (pos[:, None] + half_w) * freqs[None, :] / window_size

    cos_q = angles_q.cos()
    sin_q = angles_q.sin()
    cos_k = angles_k.cos()
    sin_k = angles_k.sin()

    q_pos = torch.stack([cos_q, sin_q], dim=-1).reshape(seqlen, aug_dims)

    bhat_selected = bhat.index_select(-1, top_idx)
    freq_scale = torch.full((n_keep,), 2.0 / window_size, device=q.device, dtype=torch.float32)
    freq_scale = torch.where(top_idx == 0, torch.full_like(freq_scale, 1.0 / window_size), freq_scale)
    if window_size % 2 == 0:
        freq_scale = torch.where(
            top_idx == (window_size // 2),
            torch.full_like(freq_scale, 1.0 / window_size),
            freq_scale,
        )

    a_k = bhat_selected.real * freq_scale[None, :]
    b_k = bhat_selected.imag * freq_scale[None, :]

    k_even = a_k[:, None, :] * cos_k[None, :, :] - b_k[:, None, :] * sin_k[None, :, :]
    k_odd = a_k[:, None, :] * sin_k[None, :, :] + b_k[:, None, :] * cos_k[None, :, :]
    k_pos = torch.stack([k_even, k_odd], dim=-1).reshape(nheads, seqlen, aug_dims)

    q_bias = q_pos[None, :, None, :].expand(bsz, seqlen, nheads, aug_dims).to(dtype)
    k_bias = k_pos.permute(1, 0, 2).unsqueeze(0).expand(bsz, seqlen, nheads, aug_dims).to(dtype)

    q_aug = torch.cat([q * softmax_scale, q_bias], dim=-1)
    k_aug = torch.cat([k, k_bias], dim=-1)
    v_aug = torch.cat([v, q.new_zeros(bsz, seqlen, nheads, aug_dims)], dim=-1)

    q_aug = _pad_last_dim(q_aug, pad_to_multiple)
    k_aug = _pad_last_dim(k_aug, pad_to_multiple)
    v_aug = _pad_last_dim(v_aug, pad_to_multiple)

    out_aug = flash_attn_func(
        q_aug,
        k_aug,
        v_aug,
        softmax_scale=1.0,
        window_size=(half_w, ws_right),
    )
    return out_aug[..., :headdim]
