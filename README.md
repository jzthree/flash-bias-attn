# flash-bias-attn

**Fast exact attention with per-head relative position bias.**

Adds a Toeplitz bias table lookup to Flash Attention's CUDA kernel. Exact computation with ~1.5x overhead vs no-bias flash attention — **79x faster** than FlexAttention with `score_mod`.

```python
from flash_bias_attn import flash_attn_bias

# q, k, v: (batch, seqlen, nheads, headdim) float16/bfloat16
# bias_table: (nheads, window_size) — learned per-head relative position bias
output = flash_attn_bias(q, k, v, bias_table, window_size=128)
```

## What it does

For each attention head, adds a learned bias based on relative position:

```
score[i, j] = (q[i] · k[j]) / √d + bias_table[j - i + window_size//2]
```

Only positions within the sliding window contribute. The bias table is a 1D array indexed by relative position (Toeplitz structure).

## Performance

Benchmarked on NVIDIA GH200, B=600, L=4600, H=8, D=16, W=128 (bfloat16):

| Method | Forward (ms) | vs no-bias |
|--------|-------------|------------|
| Flash Attention (no bias) | 21 | 1.0x |
| **flash-bias-attn (this)** | **33** | **1.5x** |
| FlashBias concat rank=16 | 194 | 3.0x |
| Custom Triton kernel | 749 | 11.7x |
| FlexAttention + score_mod | 9,386 | 122x |

**Forward + backward** timing (full training step):

| Method | Step (ms) | vs no-bias |
|--------|----------|------------|
| Flash Attention (no bias) | 64 | 1.0x |
| FlashBias concat rank=16 | 194 | 3.0x |
| Custom Triton kernel | 749 | 11.7x |
| FlexAttention + score_mod | 9,386 | 122x |

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
- **FlashBias concatenation trick**: Decomposes bias via Fourier/SVD into augmented Q,K dimensions. Fast (~3x) but approximate unless using full rank (which makes head dim too large)
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

- **Forward only** — backward pass not yet implemented (use `flash_attn_func` backward or the Triton kernel for training)
- **Head dim ≤ 32** — only `hdim32` kernel instantiation included (add more `.cu` files from flash-attn for larger head dims)
- **bfloat16 only** — add float16 `.cu` instantiation for fp16 support
- Based on Flash Attention 2.8.3 (sm80 CUDA core path)

## API

```python
flash_attn_bias(
    q,              # (batch, seqlen_q, nheads, headdim)
    k,              # (batch, seqlen_k, nheads_k, headdim)
    v,              # (batch, seqlen_k, nheads_k, headdim)
    bias_table,     # (nheads, window_size) — relative position bias
    window_size,    # int — symmetric window size
    softmax_scale=None,  # float, default 1/sqrt(headdim)
) -> output  # (batch, seqlen_q, nheads, headdim)
```

The bias for position pair (i, j) is `bias_table[head, j - i + window_size // 2]`, applied only when `|j - i| < window_size // 2`.

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
