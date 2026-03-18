// Minimal Python binding for flash attention with bias table.
// Forward only (for now). Based on flash_attn 2.8 API.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
}

// Set up params struct from PyTorch tensors
void set_params_fwd(FLASH_NAMESPACE::Flash_fwd_params &params,
                    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                    at::Tensor &out, at::Tensor &softmax_lse,
                    float softmax_scale,
                    int window_size_left, int window_size_right,
                    const at::Tensor &bias_table) {
    memset(&params, 0, sizeof(params));

    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_size = q.size(3);
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_size;

    // Round up for internal buffers
    params.seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
    params.d_rounded = ((head_size + 31) / 32) * 32;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // Bias table
    if (bias_table.defined() && bias_table.numel() > 0) {
        // bias_table should be (nheads, window_size) float32
        params.bias_table_ptr = bias_table.data_ptr<float>();
        params.bias_table_window_size = bias_table.size(1);
    } else {
        params.bias_table_ptr = nullptr;
        params.bias_table_window_size = 0;
    }

    // No alibi, dropout, softcap, etc.
    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;
    params.p_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softcap = 0.0f;
    params.p_dropout = 1.0f;  // 1.0 = no dropout
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.rotary_dim = 0;
    params.num_splits = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

std::vector<at::Tensor> flash_attn_bias_fwd(
    at::Tensor &q, at::Tensor &k, at::Tensor &v,
    at::Tensor &bias_table,
    float softmax_scale,
    int window_size_left, int window_size_right
) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.dtype() == at::kBFloat16, "Only bf16 supported in this minimal build");

    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_size = q.size(3);
    const int seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;

    auto out = torch::empty_like(q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q_rounded},
                                     q.options().dtype(at::kFloat));

    // Convert bias_table to float32 and pre-divide by softmax_scale.
    // The kernel adds bias to un-scaled QK^T scores, so bias values
    // must be in the same "un-scaled" space (like alibi_slope is divided).
    at::Tensor bias_table_f32;
    if (bias_table.defined() && bias_table.numel() > 0) {
        bias_table_f32 = (bias_table.to(at::kFloat) / softmax_scale).contiguous();
    } else {
        bias_table_f32 = at::Tensor();
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    set_params_fwd(params, q, k, v, out, softmax_lse,
                   softmax_scale, window_size_left, window_size_right,
                   bias_table_f32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Only hdim32, bf16, non-causal is compiled
    TORCH_CHECK(head_size <= 32, "Only head_size <= 32 supported (compiled for hdim32)");

    // Is_local = true when window_size >= 0
    const bool is_local = (window_size_left >= 0) || (window_size_right >= 0);

    // Is_local=true for windowed attention ensures row indices are computed in mask.h,
    // which our bias_table lookup needs. No template changes required.
    FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 32, /*Is_causal=*/false>(params, stream);

    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_bias_fwd", &flash_attn_bias_fwd,
          "Flash attention forward with bias table");
}
