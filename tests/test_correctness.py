"""Correctness tests for flash-bias-attn."""
import math
import torch
import torch.nn.functional as F
import pytest


def ref_windowed_attn_bias(q, k, v, bias_table, window_size, softmax_scale=None):
    """Naive PyTorch reference: materializes full N*N attention matrix."""
    B, L, H, D = q.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(D)
    half_w = window_size // 2
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
    i_idx = torch.arange(L, device=q.device)
    rel = i_idx[None, :] - i_idx[:, None]
    in_win = (rel >= -half_w) & (rel < (window_size - half_w))
    bias_idx = (rel + half_w).clamp(0, window_size - 1)
    bias = bias_table[:, bias_idx]
    bias = torch.where(in_win.unsqueeze(0), bias, torch.zeros_like(bias))
    scores += bias.unsqueeze(0)
    scores = torch.where(in_win[None, None, :, :], scores, torch.full_like(scores, float("-inf")))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v_t).permute(0, 2, 1, 3)


@pytest.mark.parametrize("B,L,H,D,W", [
    (2, 64, 2, 16, 32),
    (2, 256, 4, 16, 64),
    (1, 512, 4, 16, 128),
    (2, 1000, 4, 16, 64),
])
def test_forward_correctness(B, L, H, D, W):
    from flash_bias_attn import flash_attn_bias
    device = "cuda"
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    bt = torch.randn(H, W, dtype=torch.bfloat16, device=device)

    out = flash_attn_bias(q, k, v, bt, W, scale)
    out_ref = ref_windowed_attn_bias(q.float(), k.float(), v.float(), bt.float(), W, scale)

    diff = (out.float() - out_ref).abs().max().item()
    assert diff < 1e-2, f"max_diff={diff:.4e}"


def test_zero_bias_matches_flash_attn():
    """Zero bias should produce identical results to unmodified flash-attn."""
    from flash_bias_attn import flash_attn_bias
    from flash_attn import flash_attn_func
    device = "cuda"
    B, L, H, D, W = 2, 256, 4, 16, 64
    scale = 1.0 / math.sqrt(D)
    half_w = W // 2
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    bt_zero = torch.zeros(H, W, dtype=torch.float32, device=device)

    out_bias = flash_attn_bias(q, k, v, bt_zero, W, scale)
    out_orig = flash_attn_func(q, k, v, softmax_scale=scale, window_size=(half_w, half_w))

    diff = (out_bias.float() - out_orig.float()).abs().max().item()
    assert diff == 0.0, f"Zero-bias should be identical, got diff={diff}"


def test_const_bias_invariant():
    """Constant bias is softmax-invariant: same output as zero bias."""
    from flash_bias_attn import flash_attn_bias
    device = "cuda"
    B, L, H, D, W = 2, 64, 2, 16, 32
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)

    bt_zero = torch.zeros(H, W, dtype=torch.float32, device=device)
    bt_const = torch.ones(H, W, dtype=torch.float32, device=device) * 5.0

    out_zero = flash_attn_bias(q, k, v, bt_zero, W, scale)
    out_const = flash_attn_bias(q, k, v, bt_const, W, scale)

    diff = (out_zero.float() - out_const.float()).abs().max().item()
    assert diff < 1e-3, f"Const bias should be invariant, got diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
