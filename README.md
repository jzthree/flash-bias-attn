# flash-bias-attn

**Fast exact attention with per-head relative position bias.**

Adds a Toeplitz bias table lookup to Flash Attention's sliding-window CUDA kernel. Exact forward and backward are implemented for local attention, with materially lower training-step cost than generic score-mod approaches.

```python
from flash_bias_attn import flash_attn_bias

# q, k, v: (batch, seqlen, nheads, headdim) float16/bfloat16
# bias_table: (nheads, window_size) — learned per-head relative position bias
# window_size is the full bias-table width; use 257 for a local window of +/-128
output = flash_attn_bias(q, k, v, bias_table, window_size=257)
```

## What it does

For each attention head, adds a learned bias based on relative position:

```
score[i, j] = (q[i] · k[j]) / √d + bias_table[j - i + window_size//2]
```

Only positions within the sliding window contribute. The bias table is a 1D array indexed by relative position (Toeplitz structure).

## Performance

Benchmarked on NVIDIA GH200, bfloat16, `B=600, L=4600, H=8, D=16`, with local window `+/-128` (`window_size=257`):

**Methods that support sliding-window relative-position bias**

| Method | Exact bias semantics | Trainable bias table | Step (ms) |
|--------|----------------------|----------------------|-----------|
| **flash-bias-attn (this)** | **Yes** | **Yes** | **262** |
| Custom Triton kernel | Yes | Yes | 749 |
| FlexAttention + `score_mod` | Yes | Yes | 9,386 |

These are the relevant comparisons: all three support local attention with learned relative-position bias. The no-bias flash-attn baseline is useful for understanding incremental overhead, but it is not an alternative implementation of the same feature set.

**Exact-path phase breakdown**

Forward only:

| Configuration | Forward (ms) | vs no-bias |
|--------------|-------------|------------|
| Flash Attention (no bias, same local window) | 18.4 | 1.0x |
| flash-bias-attn exact | 64.4 | 3.5x |

Backward only:

| Configuration | Backward (ms) | vs no-bias |
|--------------|--------------|------------|
| Flash Attention (no bias, same local window) | 43.6 | 1.0x |
| flash-bias-attn exact (frozen bias table) | 97.3 | 2.2x |
| flash-bias-attn exact (trainable bias table) | 193.5 | 4.4x |

The trainable-bias case includes `dbias` accumulation. When the bias table is frozen, that path is skipped and the backward pass is materially cheaper. Phase timings are measured separately, so they will not sum exactly to the full-step table above.

**Why this is a good result**

- The `262 ms` number is for the hard case: **exact, trainable, sliding-window attention with per-head relative-position bias**. It includes the backward pass for the bias table itself, not just `dQ/dK/dV`.
- The gap from frozen to trainable bias is only about **`110 ms`** (`262 - 152`). That isolates the remaining cost to `dbias` accumulation; the rest of the flash-attn backward path stays close to the frozen-bias case.
- The frozen-bias result, `152 ms`, shows the cost of exact bias-aware **local attention** without bias-table gradients. That is still only **2.6x** over no-bias sliding-window flash attention while preserving exact Toeplitz bias semantics.
- This repo is solving a problem that stock flash-attn does not handle directly: **exact backward for trainable relative-position bias tables inside sliding-window attention**. Low-rank concat-style methods can be faster, but in our experiments they showed large approximation error at practical ranks and are not a drop-in replacement.
- The relevant sales pitch is not “we are close to no-bias flash attention.” It is “among implementations that actually support exact trainable relative-position bias inside fused local attention, this keeps the training step in the low hundreds of milliseconds instead of `749 ms` Triton or multi-second score-mod paths.”

## How it works

We modify Flash Attention 2's CUDA kernel (~30 lines changed) to add a bias table lookup in the inner attention loop. The modification is minimal:

1. **`flash.h`** — Add `bias_table_ptr` and `bias_table_window_size` to params struct
2. **`mask.h`** — Add table lookup where ALiBi bias is applied:
   ```cuda
   if (bias_table != nullptr) {
       int rel = col_idx - (row_idx + max_seqlen_k - max_seqlen_q);
       int idx = rel + half_w;
       if (idx >= 0 && idx < bias_table_size)
           tensor(...) += bias_table[idx];
   }
   ```
3. **`flash_fwd_kernel.h`** — Pass per-head bias table pointer to Mask struct

The bias table values are pre-divided by `softmax_scale` to match Flash Attention's internal score representation.

### Why not other approaches?

- **FlexAttention + score_mod**: The `score_mod` callback prevents kernel fusion, causing 122x slowdown
- **FlashBias concatenation trick**: Fast at very low rank, but for trainable relative-position bias it showed large approximation error at practical ranks. The concat path is now deprecated and kept only as an experimental baseline.
- **Custom Triton kernel**: Correct but 12x slower than Flash Attention's optimized CUDA (the "Triton tax")
- **This approach**: Directly modifies the CUDA kernel — exact and fast

## Installation

### Prerequisites

- PyTorch >= 2.0 with CUDA
- [CUTLASS](https://github.com/NVIDIA/cutlass) headers
- CUDA toolkit matching your PyTorch installation
- `flash-attn` >= 2.0

### Build from source

```bash
git clone https://github.com/jzthree/flash-bias-attn.git
cd flash-bias-attn

# Clone CUTLASS if not already available
git clone --depth 1 https://github.com/NVIDIA/cutlass.git

# Build
CC=gcc CXX=g++ CUTLASS_PATH=./cutlass pip install .

# Or build in-place for development
CC=gcc CXX=g++ CUTLASS_PATH=./cutlass python setup.py build_ext --inplace
```

### Current limitations

- **Backward is implemented, but trainable bias is still materially slower than no-bias flash attention** — the remaining cost is dominated by `dbias` accumulation
- **Head dim ≤ 32** — only `hdim32` kernel instantiation included (add more `.cu` files from flash-attn for larger head dims)
- **bfloat16 only** — add float16 `.cu` instantiation for fp16 support
- **Concat path is deprecated** — retained only as an experimental baseline, not recommended for trainable exact bias
- Based on Flash Attention 2.8.3 (sm80 CUDA core path)

## API

```python
flash_attn_bias(
    q,              # (batch, seqlen_q, nheads, headdim)
    k,              # (batch, seqlen_k, nheads_k, headdim)
    v,              # (batch, seqlen_k, nheads_k, headdim)
    bias_table,     # (nheads, window_size) — relative position bias
    window_size,    # int — full bias-table width
    softmax_scale=None,  # float, default 1/sqrt(headdim)
) -> output  # (batch, seqlen_q, nheads, headdim)
```

The bias for position pair `(i, j)` is `bias_table[head, j - i + window_size // 2]`, applied only inside the local window implied by `window_size`.

Example: for a local window of `+/-128`, use `window_size=257`.

## Use cases

- **Genomic sequence models**: Per-head position bias captures biological distance effects (splice site proximity, nucleosome positioning)
- **Vision transformers**: Swin-style relative position bias with sliding window attention
- **Long-context models**: Efficient local attention with learned position encoding

## Citation

Based on:
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) by Tri Dao
- [FlashBias](https://github.com/thuml/FlashBias) (NeurIPS 2025) for the concatenation trick comparison

## License

BSD-3-Clause (same as Flash Attention)
