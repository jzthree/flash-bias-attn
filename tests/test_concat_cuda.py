import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_bias_attn import flash_attn_bias_concat


def dense_windowed_bias_attention(q, k, v, bias_table, window_size, softmax_scale=None):
    bsz, seqlen, nheads, headdim = q.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
    half_w = window_size // 2
    ws_right = half_w - 1 + (window_size % 2)

    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()

    scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale
    pos = torch.arange(seqlen, device=q.device)
    rel = pos[None, :] - pos[:, None]
    valid = (rel >= -half_w) & (rel <= ws_right)
    scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))
    bias_idx = (rel + half_w).clamp(0, window_size - 1)
    scores = scores + bias_table[:, bias_idx][None, :, :, :]
    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, vh)
    return out.permute(0, 2, 1, 3).to(q.dtype)


def check_small_exact():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen, nheads, headdim = 2, 64, 3, 16
    window_size = 31

    q = torch.randn(bsz, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bias_table = torch.randn(nheads, window_size, device=device, dtype=torch.float32, requires_grad=True)

    out_ref = dense_windowed_bias_attention(q, k, v, bias_table, window_size)
    out_cat = flash_attn_bias_concat(q, k, v, bias_table, window_size, rank=None)

    max_err = (out_ref - out_cat).abs().max().item()
    print(f"exact-window check max_err={max_err:.4e}")


def bench_large(rank=32):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen, nheads, headdim = 600, 4600, 8, 16
    window_size = 257

    q = torch.randn(bsz, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bias_table = torch.randn(nheads, window_size, device=device, dtype=torch.float32, requires_grad=True)

    def run():
        out = flash_attn_bias_concat(q, k, v, bias_table, window_size, rank=rank)
        loss = out.square().mean()
        loss.backward()

    for _ in range(2):
        for t in (q, k, v, bias_table):
            if t.grad is not None:
                t.grad.zero_()
        run()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(5):
        for t in (q, k, v, bias_table):
            if t.grad is not None:
                t.grad.zero_()
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    print(f"concat rank={rank} avg_ms={avg_ms:.1f} samples={[round(x, 1) for x in times]}")


if __name__ == "__main__":
    check_small_exact()
    bench_large()
