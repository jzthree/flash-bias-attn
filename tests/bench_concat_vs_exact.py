import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flash_bias_attn import flash_attn_bias, flash_attn_bias_concat


def bench_ms(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def rel_l2(a, b, eps=1e-12):
    denom = b.float().norm().item()
    if denom < eps:
        return float((a.float() - b.float()).norm().item())
    return float((a.float() - b.float()).norm().item() / denom)


def compare_rank(rank, B=2, L=512, H=8, D=16, half_w=128):
    window_size = 2 * half_w + 1
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    bt = torch.randn(H, window_size, device=device, dtype=torch.float32, requires_grad=True)
    dout = torch.randn(B, L, H, D, device=device, dtype=dtype)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    bt_ref = bt.detach().clone().requires_grad_(True)

    out_exact = flash_attn_bias(q_ref, k_ref, v_ref, bt_ref, window_size)
    out_exact.backward(dout)

    out_concat = flash_attn_bias_concat(q, k, v, bt, window_size, rank=rank)
    out_concat.backward(dout)

    return {
        "out_rel_l2": rel_l2(out_concat, out_exact),
        "dq_rel_l2": rel_l2(q.grad, q_ref.grad),
        "dk_rel_l2": rel_l2(k.grad, k_ref.grad),
        "dv_rel_l2": rel_l2(v.grad, v_ref.grad),
        "dbias_rel_l2": rel_l2(bt.grad, bt_ref.grad),
    }


def bench_rank(rank, B=64, L=4600, H=8, D=16, half_w=128):
    window_size = 2 * half_w + 1
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    bt = torch.randn(H, window_size, device=device, dtype=torch.float32, requires_grad=True)

    def run_exact():
        q.grad = None
        k.grad = None
        v.grad = None
        bt.grad = None
        out = flash_attn_bias(q, k, v, bt, window_size)
        out.sum().backward()

    def run_concat():
        q.grad = None
        k.grad = None
        v.grad = None
        bt.grad = None
        out = flash_attn_bias_concat(q, k, v, bt, window_size, rank=rank)
        out.sum().backward()

    exact_ms = bench_ms(run_exact, warmup=1, iters=5)
    concat_ms = bench_ms(run_concat, warmup=1, iters=5)
    return exact_ms, concat_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ranks = [16, 32, 64, 96, 128, 160, 192, 224, 240]
    print("shape_error:  B=2 L=512 H=8 D=16 half_w=128")
    print("shape_bench:  B=64 L=4600 H=8 D=16 half_w=128")
    print("")

    for rank in ranks:
        aug_dim = 16 + rank
        errors = compare_rank(rank)
        exact_ms, concat_ms = bench_rank(rank)
        speedup = exact_ms / concat_ms
        print(
            f"rank={rank:>3} aug_dim={aug_dim:>3} "
            f"bench_exact_ms={exact_ms:>7.1f} bench_concat_ms={concat_ms:>7.1f} speedup={speedup:>5.2f}x "
            f"out_rel_l2={errors['out_rel_l2']:.3e} dq_rel_l2={errors['dq_rel_l2']:.3e} "
            f"dk_rel_l2={errors['dk_rel_l2']:.3e} dv_rel_l2={errors['dv_rel_l2']:.3e} "
            f"dbias_rel_l2={errors['dbias_rel_l2']:.3e}"
        )


if __name__ == "__main__":
    main()
