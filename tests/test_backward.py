import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flash_bias_attn
from flash_attn import flash_attn_func


def reference_attention(q, k, v, bias_table, half_w):
    b, lq, h, d = q.shape
    lk = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    q_idx = torch.arange(lq, device=q.device)[:, None]
    k_idx = torch.arange(lk, device=q.device)[None, :]
    rel = k_idx - q_idx
    mask = (rel >= -half_w) & (rel <= half_w)
    scores = scores.masked_fill(~mask, float('-inf'))
    bias_idx = (rel + half_w).clamp_(0, 2 * half_w)
    scores = scores + bias_table[:, bias_idx].unsqueeze(0)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)
    return out.permute(0, 2, 1, 3).to(q.dtype)


def bench_ms(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def run_small_correctness():
    print('=== FORWARD + BACKWARD ===')
    cases = [(2, 64, 32), (2, 256, 64), (1, 512, 128)]
    for b, l, w in cases:
        h = 8
        d = 16
        table_size = 2 * w + 1
        q = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(h, table_size, device='cuda', dtype=torch.float32, requires_grad=True)

        out = flash_bias_attn.flash_attn_bias(q, k, v, bias, table_size)
        loss = out.float().square().mean()
        loss.backward()
        dbias_fast = bias.grad.detach().float().clone()

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        out_ref = reference_attention(q_ref, k_ref, v_ref, bias_ref, w)
        loss_ref = out_ref.float().square().mean()
        loss_ref.backward()
        dbias_ref = bias_ref.grad.detach().float()

        ratio = dbias_fast.norm() / dbias_ref.norm().clamp_min(1e-12)
        status = 'PASS' if torch.allclose(dbias_fast, dbias_ref, rtol=2e-2, atol=2e-2) else 'FAIL'
        print(f'  B={b} L={l:3d} W={w:3d}: dBT_ratio={ratio.item():.4f}  {status}')


def run_benchmarks():
    print('\n=== BENCHMARK (B=600, L=4600, H=8, D=16, W=128) ===')
    b, l, h, d, w = 600, 4600, 8, 16, 128
    table_size = 2 * w + 1

    q0 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k0 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v0 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    def no_bias_step():
        out = flash_attn_func(q0, k0, v0, dropout_p=0.0, causal=False, window_size=(w, w))
        out.sum().backward()
        q0.grad = None
        k0.grad = None
        v0.grad = None

    no_bias_ms = bench_ms(no_bias_step)

    q1 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k1 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v1 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    bias_frozen = torch.randn(h, table_size, device='cuda', dtype=torch.float32, requires_grad=False)

    def frozen_bias_step():
        out = flash_bias_attn.flash_attn_bias(q1, k1, v1, bias_frozen, table_size)
        out.sum().backward()
        q1.grad = None
        k1.grad = None
        v1.grad = None

    frozen_ms = bench_ms(frozen_bias_step)

    q2 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k2 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v2 = torch.randn(b, l, h, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    bias_train = torch.randn(h, table_size, device='cuda', dtype=torch.float32, requires_grad=True)

    def trainable_bias_step():
        out = flash_bias_attn.flash_attn_bias(q2, k2, v2, bias_train, table_size)
        out.sum().backward()
        q2.grad = None
        k2.grad = None
        v2.grad = None
        bias_train.grad = None

    trainable_ms = bench_ms(trainable_bias_step)

    print(f'  flash-attn windowed (no bias):              {no_bias_ms:.0f} ms')
    print(f'  flash-attn + bias (fwd+bwd, frozen bias):  {frozen_ms:.0f} ms  ({frozen_ms / no_bias_ms:.2f}x)')
    print(f'  flash-attn + bias (fwd+bwd, trainable):    {trainable_ms:.0f} ms  ({trainable_ms / no_bias_ms:.2f}x)')


if __name__ == '__main__':
    torch.manual_seed(0)
    run_small_correctness()
    run_benchmarks()
