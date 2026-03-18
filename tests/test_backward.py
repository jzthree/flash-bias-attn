"""Test forward + backward correctness and benchmark."""
import math, time, torch, sys
sys.path.insert(0, "/work/07390/jzthree/vista/Code/flash_attn_bias")

def ref(q, k, v, bt, W, scale):
    B, L, H, D = q.shape
    half_w = W // 2
    q_t = q.permute(0,2,1,3); k_t = k.permute(0,2,1,3); v_t = v.permute(0,2,1,3)
    scores = torch.matmul(q_t, k_t.transpose(-2,-1)) * scale
    i_idx = torch.arange(L, device=q.device)
    rel = i_idx[None,:] - i_idx[:,None]
    in_win = (rel >= -half_w) & (rel < (W - half_w))
    bias_idx = (rel + half_w).clamp(0, W-1)
    bias = bt[:, bias_idx]
    bias = torch.where(in_win.unsqueeze(0), bias, torch.zeros_like(bias))
    scores += bias.unsqueeze(0)
    scores = torch.where(in_win[None,None,:,:], scores, torch.full_like(scores, float("-inf")))
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v_t).permute(0,2,1,3)

device = "cuda"
torch.manual_seed(42)

from flash_bias_attn import flash_attn_bias

# === Forward + Backward correctness ===
print("=== FORWARD + BACKWARD ===")
for B, L, H, D, W in [(2, 64, 2, 16, 32), (2, 256, 4, 16, 64), (1, 512, 4, 16, 128)]:
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
    bt = torch.randn(H,W, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Triton forward+backward
    out = flash_attn_bias(q, k, v, bt, W, scale)
    out.float().sum().backward()
    dq_t, dk_t, dv_t, dbt_t = q.grad.float(), k.grad.float(), v.grad.float(), bt.grad.float()

    # Reference
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    btf = bt.detach().float().requires_grad_(True)
    out_ref = ref(qf, kf, vf, btf, W, scale)
    out_ref.sum().backward()

    fwd_diff = (out.float() - out_ref).abs().max().item()
    dq_diff = (dq_t - qf.grad).abs().max().item()
    dk_diff = (dk_t - kf.grad).abs().max().item()
    dv_diff = (dv_t - vf.grad).abs().max().item()
    dbt_diff = (dbt_t - btf.grad).abs().max().item()

    # Compute empirical scale ratio for dBT
    mask = btf.grad.abs() > 0.1
    if mask.any():
        ratio = (dbt_t[mask] / btf.grad[mask]).median().item()
    else:
        ratio = float('nan')
    print(f"  B={B} L={L:>3d} W={W:>3d}: fwd={fwd_diff:.3e} dQ={dq_diff:.3e} dK={dk_diff:.3e} dV={dv_diff:.3e} dBT={dbt_diff:.3e} dBT_ratio={ratio:.4f}  "
          f"{'PASS' if max(fwd_diff, dq_diff, dk_diff, dv_diff) < 5e-2 and dbt_diff < 1e-1 else 'FAIL'}")

# === Benchmark fwd+bwd ===
print("\n=== BENCHMARK (B=600, L=4600, H=8, D=16, W=128) ===")
B, L, H, D, W = 600, 4600, 8, 16, 128
scale = 1.0 / math.sqrt(D)

def bench(fn, *args, n_warmup=3, n_steps=5):
    for _ in range(n_warmup):
        out = fn(*args)
        out.float().sum().backward()
        torch.cuda.synchronize()
        for a in args:
            if hasattr(a, 'grad') and a.grad is not None:
                a.grad = None
    times = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        out.float().sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        for a in args:
            if hasattr(a, 'grad') and a.grad is not None:
                a.grad = None
    return sum(times) / len(times)

from flash_attn import flash_attn_func

# Baseline: flash-attn windowed no bias
q1 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
k1 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
v1 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
half_w = W // 2
t_base = bench(lambda q,k,v: flash_attn_func(q,k,v, softmax_scale=scale, window_size=(half_w,half_w)), q1, k1, v1)
print(f"  flash-attn windowed (no bias): {t_base:.0f} ms")

# Ours: flash-attn + bias table
q2 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
k2 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
v2 = torch.randn(B,L,H,D, dtype=torch.bfloat16, device=device, requires_grad=True)
bt2 = torch.randn(H,W, dtype=torch.bfloat16, device=device, requires_grad=True)
t_bias = bench(lambda q,k,v,bt: flash_attn_bias(q,k,v,bt,W,scale), q2, k2, v2, bt2)
print(f"  flash-attn + bias (fwd+bwd):   {t_bias:.0f} ms  ({t_bias/t_base:.2f}x)")
