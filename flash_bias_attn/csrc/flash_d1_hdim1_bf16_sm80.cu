#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kKChunk = 64;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(const float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ int64_t idx3(
    const int b, const int row, const int head,
    const int64_t batch_stride, const int64_t row_stride, const int64_t head_stride
) {
    return static_cast<int64_t>(b) * batch_stride
         + static_cast<int64_t>(row) * row_stride
         + static_cast<int64_t>(head) * head_stride;
}

__global__ void flash_d1_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ bias_t,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_row_stride, int64_t v_head_stride,
    int64_t out_batch_stride, int64_t out_row_stride, int64_t out_head_stride,
    int64_t lse_batch_stride, int64_t lse_head_stride,
    int batch_size, int seqlen, int num_heads,
    int window_size_left, int window_size_right,
    float softmax_scale
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch = blockIdx.x;
    const int head = blockIdx.y * 32 + lane;
    if (batch >= batch_size || head >= num_heads) { return; }

    for (int qrow = warp; qrow < seqlen; qrow += kWarpsPerBlock) {
        const float qv = bf16_to_float(q[idx3(batch, qrow, head, q_batch_stride, q_row_stride, q_head_stride)]);
        const int k_start = max(0, qrow - window_size_left);
        const int k_end = min(seqlen - 1, qrow + window_size_right);

        float m = -INFINITY;
        float s = 0.0f;
        float acc = 0.0f;

        for (int krow = k_start; krow <= k_end; ++krow) {
            const float kval = bf16_to_float(k[idx3(batch, krow, head, k_batch_stride, k_row_stride, k_head_stride)]);
            const float vval = bf16_to_float(v[idx3(batch, krow, head, v_batch_stride, v_row_stride, v_head_stride)]);
            const int rel = krow - qrow + window_size_left;
            const float score = qv * kval * softmax_scale + bias_t[rel * num_heads + head];
            const float m_new = fmaxf(m, score);
            const float alpha = isfinite(m) ? __expf(m - m_new) : 0.0f;
            const float p = __expf(score - m_new);
            acc = acc * alpha + p * vval;
            s = s * alpha + p;
            m = m_new;
        }

        const float outv = s > 0.0f ? acc / s : 0.0f;
        out[idx3(batch, qrow, head, out_batch_stride, out_row_stride, out_head_stride)] = float_to_bf16(outv);
        lse[static_cast<int64_t>(batch) * lse_batch_stride + static_cast<int64_t>(head) * lse_head_stride + qrow] = m + logf(s);
    }
}

__global__ void flash_d1_bwd_dq_dbias_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ out,
    const float* __restrict__ lse,
    const float* __restrict__ bias_t,
    __nv_bfloat16* __restrict__ dq,
    __nv_bfloat16* __restrict__ dk,
    __nv_bfloat16* __restrict__ dv,
    float* __restrict__ dbias_t,
    int64_t do_batch_stride, int64_t do_row_stride, int64_t do_head_stride,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_row_stride, int64_t v_head_stride,
    int64_t out_batch_stride, int64_t out_row_stride, int64_t out_head_stride,
    int64_t dq_batch_stride, int64_t dq_row_stride, int64_t dq_head_stride,
    int64_t dk_batch_stride, int64_t dk_row_stride, int64_t dk_head_stride,
    int64_t dv_batch_stride, int64_t dv_row_stride, int64_t dv_head_stride,
    int64_t lse_batch_stride, int64_t lse_head_stride,
    int batch_size, int seqlen, int num_heads,
    int window_size_left, int window_size_right,
    float softmax_scale
) {
    extern __shared__ float shared_mem[];
    float* dbias_shared = shared_mem;
    float* dk_shared = dbias_shared + (window_size_left + window_size_right + 1) * 32;
    float* dv_shared = dk_shared + kKChunk * 32;
    __nv_bfloat16* k_shared = reinterpret_cast<__nv_bfloat16*>(dv_shared + kKChunk * 32);
    __nv_bfloat16* v_shared = k_shared + kKChunk * 32;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch = blockIdx.x;
    const int head_group = blockIdx.y;
    const int head = head_group * 32 + lane;
    const int window_size = window_size_left + window_size_right + 1;
    const bool compute_dbias = dbias_t != nullptr;

    if (compute_dbias) {
        for (int idx = threadIdx.x; idx < window_size * 32; idx += blockDim.x) {
            dbias_shared[idx] = 0.0f;
        }
        __syncthreads();
    }

    if (batch >= batch_size || head >= num_heads) { return; }

    for (int qblock = 0; qblock < seqlen; qblock += kWarpsPerBlock) {
        const int qrow = qblock + warp;
        const bool q_valid = qrow < seqlen;
        const float qv = (q_valid && head < num_heads)
            ? bf16_to_float(q[idx3(batch, qrow, head, q_batch_stride, q_row_stride, q_head_stride)])
            : 0.0f;
        const float outv = (q_valid && head < num_heads)
            ? bf16_to_float(out[idx3(batch, qrow, head, out_batch_stride, out_row_stride, out_head_stride)])
            : 0.0f;
        const float dov = (q_valid && head < num_heads)
            ? bf16_to_float(dout[idx3(batch, qrow, head, do_batch_stride, do_row_stride, do_head_stride)])
            : 0.0f;
        const float lsev = (q_valid && head < num_heads)
            ? lse[static_cast<int64_t>(batch) * lse_batch_stride + static_cast<int64_t>(head) * lse_head_stride + qrow]
            : 0.0f;
        const int k_start = q_valid ? max(0, qrow - window_size_left) : 0;
        const int k_end = q_valid ? min(seqlen - 1, qrow + window_size_right) : -1;
        float dq_acc = 0.0f;

        const int chunk_start_min = max(0, qblock - window_size_left);
        const int chunk_end_max = min(seqlen - 1, qblock + kWarpsPerBlock - 1 + window_size_right);

        for (int chunk_start = chunk_start_min; chunk_start <= chunk_end_max; chunk_start += kKChunk) {
            const int chunk_len = min(kKChunk, chunk_end_max - chunk_start + 1);
            for (int idx = threadIdx.x; idx < chunk_len * 32; idx += blockDim.x) {
                const int local_k = idx / 32;
                const int lane_idx = idx % 32;
                const int global_head = head_group * 32 + lane_idx;
                const int krow = chunk_start + local_k;
                if (global_head < num_heads) {
                    k_shared[idx] = k[idx3(batch, krow, global_head, k_batch_stride, k_row_stride, k_head_stride)];
                    v_shared[idx] = v[idx3(batch, krow, global_head, v_batch_stride, v_row_stride, v_head_stride)];
                } else {
                    k_shared[idx] = float_to_bf16(0.0f);
                    v_shared[idx] = float_to_bf16(0.0f);
                }
                dk_shared[idx] = 0.0f;
                dv_shared[idx] = 0.0f;
            }
            __syncthreads();

            if (q_valid && head < num_heads) {
                const int q_chunk_start = max(k_start, chunk_start);
                const int q_chunk_end = min(k_end, chunk_start + chunk_len - 1);
                for (int krow = q_chunk_start; krow <= q_chunk_end; ++krow) {
                    const int rel = krow - qrow + window_size_left;
                    const int local_k = krow - chunk_start;
                    const float kval = bf16_to_float(k_shared[local_k * 32 + lane]);
                    const float vval = bf16_to_float(v_shared[local_k * 32 + lane]);
                    const float score = qv * kval * softmax_scale + bias_t[rel * num_heads + head];
                    const float p = __expf(score - lsev);
                    const float ds = p * dov * (vval - outv);
                    dq_acc += ds * kval * softmax_scale;
                    atomicAdd(&dk_shared[local_k * 32 + lane], ds * qv * softmax_scale);
                    atomicAdd(&dv_shared[local_k * 32 + lane], p * dov);
                    if (compute_dbias) {
                        atomicAdd(&dbias_shared[rel * 32 + lane], ds);
                    }
                }
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < chunk_len * 32; idx += blockDim.x) {
                const int local_k = idx / 32;
                const int lane_idx = idx % 32;
                const int global_head = head_group * 32 + lane_idx;
                const int krow = chunk_start + local_k;
                if (global_head < num_heads) {
                    const int64_t dk_idx = idx3(batch, krow, global_head, dk_batch_stride, dk_row_stride, dk_head_stride);
                    const int64_t dv_idx = idx3(batch, krow, global_head, dv_batch_stride, dv_row_stride, dv_head_stride);
                    dk[dk_idx] = float_to_bf16(bf16_to_float(dk[dk_idx]) + dk_shared[idx]);
                    dv[dv_idx] = float_to_bf16(bf16_to_float(dv[dv_idx]) + dv_shared[idx]);
                }
            }
            __syncthreads();
        }

        if (q_valid && head < num_heads) {
            dq[idx3(batch, qrow, head, dq_batch_stride, dq_row_stride, dq_head_stride)] = float_to_bf16(dq_acc);
        }
    }

    if (!compute_dbias) { return; }
    __syncthreads();

    for (int idx = threadIdx.x; idx < window_size * 32; idx += blockDim.x) {
        const int rel = idx / 32;
        const int lane_idx = idx % 32;
        const int global_head = head_group * 32 + lane_idx;
        if (global_head < num_heads) {
            atomicAdd(&dbias_t[rel * num_heads + global_head], dbias_shared[idx]);
        }
    }
}

__global__ void flash_d1_bwd_dk_dv_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ out,
    const float* __restrict__ lse,
    const float* __restrict__ bias_t,
    __nv_bfloat16* __restrict__ dk,
    __nv_bfloat16* __restrict__ dv,
    int64_t do_batch_stride, int64_t do_row_stride, int64_t do_head_stride,
    int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
    int64_t v_batch_stride, int64_t v_row_stride, int64_t v_head_stride,
    int64_t out_batch_stride, int64_t out_row_stride, int64_t out_head_stride,
    int64_t dk_batch_stride, int64_t dk_row_stride, int64_t dk_head_stride,
    int64_t dv_batch_stride, int64_t dv_row_stride, int64_t dv_head_stride,
    int64_t lse_batch_stride, int64_t lse_head_stride,
    int batch_size, int seqlen, int num_heads,
    int window_size_left, int window_size_right,
    float softmax_scale
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch = blockIdx.x;
    const int head = blockIdx.y * 32 + lane;
    if (batch >= batch_size || head >= num_heads) { return; }

    for (int krow = warp; krow < seqlen; krow += kWarpsPerBlock) {
        const float kval = bf16_to_float(k[idx3(batch, krow, head, k_batch_stride, k_row_stride, k_head_stride)]);
        const float vval = bf16_to_float(v[idx3(batch, krow, head, v_batch_stride, v_row_stride, v_head_stride)]);
        const int q_start = max(0, krow - window_size_right);
        const int q_end = min(seqlen - 1, krow + window_size_left);

        float dk_acc = 0.0f;
        float dv_acc = 0.0f;

        for (int qrow = q_start; qrow <= q_end; ++qrow) {
            const float qv = bf16_to_float(q[idx3(batch, qrow, head, q_batch_stride, q_row_stride, q_head_stride)]);
            const float outv = bf16_to_float(out[idx3(batch, qrow, head, out_batch_stride, out_row_stride, out_head_stride)]);
            const float dov = bf16_to_float(dout[idx3(batch, qrow, head, do_batch_stride, do_row_stride, do_head_stride)]);
            const float lsev = lse[static_cast<int64_t>(batch) * lse_batch_stride + static_cast<int64_t>(head) * lse_head_stride + qrow];
            const int rel = krow - qrow + window_size_left;
            const float score = qv * kval * softmax_scale + bias_t[rel * num_heads + head];
            const float p = __expf(score - lsev);
            const float ds = p * dov * (vval - outv);
            dv_acc += p * dov;
            dk_acc += ds * qv * softmax_scale;
        }

        dk[idx3(batch, krow, head, dk_batch_stride, dk_row_stride, dk_head_stride)] = float_to_bf16(dk_acc);
        dv[idx3(batch, krow, head, dv_batch_stride, dv_row_stride, dv_head_stride)] = float_to_bf16(dv_acc);
    }
}

}  // namespace

void flash_attn_bias_d1_fwd_cuda(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &bias_table_t,
    at::Tensor &out,
    at::Tensor &softmax_lse,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    cudaStream_t stream
) {
    const int batch_size = q.size(0);
    const int seqlen = q.size(1);
    const int num_heads = q.size(2);

    dim3 grid(batch_size, (num_heads + 31) / 32);
    flash_d1_fwd_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        bias_table_t.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        softmax_lse.data_ptr<float>(),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1),
        batch_size, seqlen, num_heads,
        window_size_left, window_size_right,
        softmax_scale
    );
}

void flash_attn_bias_d1_bwd_cuda(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    const at::Tensor &bias_table_t,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dbias_table_t,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    bool compute_dbias,
    cudaStream_t stream
) {
    const int batch_size = q.size(0);
    const int seqlen = q.size(1);
    const int num_heads = q.size(2);
    const int window_size = window_size_left + window_size_right + 1;

    dim3 grid(batch_size, (num_heads + 31) / 32);
    const size_t shared_mem =
        static_cast<size_t>(window_size) * 32 * sizeof(float) +
        static_cast<size_t>(2 * kKChunk) * 32 * sizeof(float) +
        static_cast<size_t>(2 * kKChunk) * 32 * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(
        flash_d1_bwd_dq_dbias_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_mem)
    );

    flash_d1_bwd_dq_dbias_kernel<<<grid, kThreads, shared_mem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        softmax_lse.data_ptr<float>(),
        bias_table_t.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
        compute_dbias ? dbias_table_t.data_ptr<float>() : nullptr,
        dout.stride(0), dout.stride(1), dout.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1),
        batch_size, seqlen, num_heads,
        window_size_left, window_size_right,
        softmax_scale
    );
}
