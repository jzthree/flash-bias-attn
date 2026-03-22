#pragma nv_maxrregcount 255
// Generalized small-D kernel for D=2,4,8
// Based on the D1 kernel design: warp-per-query, K chunks in shared memory.
// Template parameter kHeadDim is the head dimension.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 24;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kKChunk = 64;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(const float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ int64_t idx3(int b, int row, int head,
    int64_t batch_stride, int64_t row_stride, int64_t head_stride) {
    return static_cast<int64_t>(b) * batch_stride
         + static_cast<int64_t>(row) * row_stride
         + static_cast<int64_t>(head) * head_stride;
}

// ===== FORWARD =====
template <int kHeadDim>
__global__ void flash_dsmall_fwd_kernel(
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
        // Load Q vector
        float qv[kHeadDim];
        const int64_t q_base = idx3(batch, qrow, head, q_batch_stride, q_row_stride, q_head_stride);
        #pragma unroll
        for (int d = 0; d < kHeadDim; d++) qv[d] = bf16_to_float(q[q_base + d]);

        const int k_start = max(0, qrow - window_size_left);
        const int k_end = min(seqlen - 1, qrow + window_size_right);

        float m = -INFINITY;
        float s = 0.0f;
        float acc[kHeadDim];
        #pragma unroll
        for (int d = 0; d < kHeadDim; d++) acc[d] = 0.0f;

        for (int krow = k_start; krow <= k_end; ++krow) {
            const int64_t k_base = idx3(batch, krow, head, k_batch_stride, k_row_stride, k_head_stride);
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < kHeadDim; d++)
                dot += qv[d] * bf16_to_float(k[k_base + d]);

            const int rel = krow - qrow + window_size_left;
            const float score = dot * softmax_scale + bias_t[rel * num_heads + head];
            const float m_new = fmaxf(m, score);
            const float alpha = isfinite(m) ? __expf(m - m_new) : 0.0f;
            const float p = __expf(score - m_new);

            const int64_t v_base = idx3(batch, krow, head, v_batch_stride, v_row_stride, v_head_stride);
            #pragma unroll
            for (int d = 0; d < kHeadDim; d++)
                acc[d] = acc[d] * alpha + p * bf16_to_float(v[v_base + d]);
            s = s * alpha + p;
            m = m_new;
        }

        const int64_t out_base = idx3(batch, qrow, head, out_batch_stride, out_row_stride, out_head_stride);
        #pragma unroll
        for (int d = 0; d < kHeadDim; d++)
            out[out_base + d] = float_to_bf16(s > 0.0f ? acc[d] / s : 0.0f);
        lse[static_cast<int64_t>(batch) * lse_batch_stride + static_cast<int64_t>(head) * lse_head_stride + qrow] = m + logf(s);
    }
}

// ===== BACKWARD (dQ + dBias, with shared-mem K/V chunks) =====
template <int kHeadDim>
__global__ void flash_dsmall_bwd_dq_dbias_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ out_tensor,
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
    const int window_size = window_size_left + window_size_right + 1;
    float* dbias_shared = shared_mem;
    // dk/dv shared: kKChunk * 32 * kHeadDim floats each
    float* dk_shared = dbias_shared + window_size * 32;
    float* dv_shared = dk_shared + kKChunk * 32 * kHeadDim;
    // k/v shared: kKChunk * 32 * kHeadDim bf16 each
    __nv_bfloat16* k_shared = reinterpret_cast<__nv_bfloat16*>(dv_shared + kKChunk * 32 * kHeadDim);
    __nv_bfloat16* v_shared = k_shared + kKChunk * 32 * kHeadDim;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int batch = blockIdx.x;
    const int head_group = blockIdx.y;
    const int head = head_group * 32 + lane;
    const bool compute_dbias = dbias_t != nullptr;

    if (compute_dbias) {
        for (int idx = threadIdx.x; idx < window_size * 32; idx += blockDim.x)
            dbias_shared[idx] = 0.0f;
        __syncthreads();
    }

    if (batch >= batch_size || head >= num_heads) { return; }

    for (int qblock = 0; qblock < seqlen; qblock += kWarpsPerBlock) {
        const int qrow = qblock + warp;
        const bool q_valid = qrow < seqlen && head < num_heads;

        // Load Q, out, dout vectors
        float qv[kHeadDim], outv[kHeadDim], dov[kHeadDim];
        if (q_valid) {
            const int64_t q_base = idx3(batch, qrow, head, q_batch_stride, q_row_stride, q_head_stride);
            const int64_t o_base = idx3(batch, qrow, head, out_batch_stride, out_row_stride, out_head_stride);
            const int64_t do_base = idx3(batch, qrow, head, do_batch_stride, do_row_stride, do_head_stride);
            #pragma unroll
            for (int d = 0; d < kHeadDim; d++) {
                qv[d] = bf16_to_float(q[q_base + d]);
                outv[d] = bf16_to_float(out_tensor[o_base + d]);
                dov[d] = bf16_to_float(dout[do_base + d]);
            }
        }
        const float lsev = q_valid
            ? lse[static_cast<int64_t>(batch) * lse_batch_stride + static_cast<int64_t>(head) * lse_head_stride + qrow]
            : 0.0f;
        const int k_start = q_valid ? max(0, qrow - window_size_left) : 0;
        const int k_end = q_valid ? min(seqlen - 1, qrow + window_size_right) : -1;
        float dq_acc[kHeadDim];
        #pragma unroll
        for (int d = 0; d < kHeadDim; d++) dq_acc[d] = 0.0f;

        const int chunk_start_min = max(0, qblock - window_size_left);
        const int chunk_end_max = min(seqlen - 1, qblock + kWarpsPerBlock - 1 + window_size_right);

        for (int chunk_start = chunk_start_min; chunk_start <= chunk_end_max; chunk_start += kKChunk) {
            const int chunk_len = min(kKChunk, chunk_end_max - chunk_start + 1);
            // Load K, V chunk into shared memory
            for (int idx = threadIdx.x; idx < chunk_len * 32 * kHeadDim; idx += blockDim.x) {
                const int local_k = idx / (32 * kHeadDim);
                const int rem = idx % (32 * kHeadDim);
                const int lane_idx = rem / kHeadDim;
                const int d = rem % kHeadDim;
                const int global_head = head_group * 32 + lane_idx;
                const int krow = chunk_start + local_k;
                if (global_head < num_heads) {
                    k_shared[idx] = k[idx3(batch, krow, global_head, k_batch_stride, k_row_stride, k_head_stride) + d];
                    v_shared[idx] = v[idx3(batch, krow, global_head, v_batch_stride, v_row_stride, v_head_stride) + d];
                } else {
                    k_shared[idx] = float_to_bf16(0.0f);
                    v_shared[idx] = float_to_bf16(0.0f);
                }
            }
            // Zero dk/dv shared
            for (int idx = threadIdx.x; idx < chunk_len * 32 * kHeadDim; idx += blockDim.x) {
                dk_shared[idx] = 0.0f;
                dv_shared[idx] = 0.0f;
            }
            __syncthreads();

            if (q_valid) {
                const int q_chunk_start = max(k_start, chunk_start);
                const int q_chunk_end = min(k_end, chunk_start + chunk_len - 1);
                for (int krow = q_chunk_start; krow <= q_chunk_end; ++krow) {
                    const int rel = krow - qrow + window_size_left;
                    const int local_k = krow - chunk_start;
                    const int sh_base = (local_k * 32 + lane) * kHeadDim;

                    // Dot product Q·K
                    float dot = 0.0f;
                    float kval[kHeadDim], vval[kHeadDim];
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++) {
                        kval[d] = bf16_to_float(k_shared[sh_base + d]);
                        dot += qv[d] * kval[d];
                    }
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++)
                        vval[d] = bf16_to_float(v_shared[sh_base + d]);

                    const float score = dot * softmax_scale + bias_t[rel * num_heads + head];
                    const float p = __expf(score - lsev);

                    // ds = p * sum_d(dov[d] * (vval[d] - outv[d]))
                    float dv_dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++)
                        dv_dot += dov[d] * (vval[d] - outv[d]);
                    const float ds = p * dv_dot;

                    // dQ += ds * K * scale
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++)
                        dq_acc[d] += ds * kval[d] * softmax_scale;

                    // dK += ds * Q * scale (shared mem accumulate)
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++)
                        atomicAdd(&dk_shared[sh_base + d], ds * qv[d] * softmax_scale);

                    // dV += p * dO (shared mem accumulate)
                    #pragma unroll
                    for (int d = 0; d < kHeadDim; d++)
                        atomicAdd(&dv_shared[sh_base + d], p * dov[d]);

                    if (compute_dbias)
                        atomicAdd(&dbias_shared[rel * 32 + lane], ds);
                }
            }
            __syncthreads();

            // Write dK, dV from shared to global
            for (int idx = threadIdx.x; idx < chunk_len * 32 * kHeadDim; idx += blockDim.x) {
                const int local_k = idx / (32 * kHeadDim);
                const int rem = idx % (32 * kHeadDim);
                const int lane_idx = rem / kHeadDim;
                const int d = rem % kHeadDim;
                const int global_head = head_group * 32 + lane_idx;
                const int krow = chunk_start + local_k;
                if (global_head < num_heads) {
                    const int64_t dk_idx = idx3(batch, krow, global_head, dk_batch_stride, dk_row_stride, dk_head_stride) + d;
                    const int64_t dv_idx = idx3(batch, krow, global_head, dv_batch_stride, dv_row_stride, dv_head_stride) + d;
                    dk[dk_idx] = float_to_bf16(bf16_to_float(dk[dk_idx]) + dk_shared[idx]);
                    dv[dv_idx] = float_to_bf16(bf16_to_float(dv[dv_idx]) + dv_shared[idx]);
                }
            }
            __syncthreads();
        }

        if (q_valid) {
            const int64_t dq_base = idx3(batch, qrow, head, dq_batch_stride, dq_row_stride, dq_head_stride);
            #pragma unroll
            for (int d = 0; d < kHeadDim; d++)
                dq[dq_base + d] = float_to_bf16(dq_acc[d]);
        }
    }

    if (!compute_dbias) { return; }
    __syncthreads();
    for (int idx = threadIdx.x; idx < window_size * 32; idx += blockDim.x) {
        const int rel = idx / 32;
        const int lane_idx = idx % 32;
        const int global_head = head_group * 32 + lane_idx;
        if (global_head < num_heads)
            atomicAdd(&dbias_t[rel * num_heads + global_head], dbias_shared[idx]);
    }
}

} // namespace

// Explicit instantiations
template __global__ void flash_dsmall_fwd_kernel<2>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);
template __global__ void flash_dsmall_fwd_kernel<4>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);
template __global__ void flash_dsmall_fwd_kernel<8>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);

template __global__ void flash_dsmall_bwd_dq_dbias_kernel<2>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);
template __global__ void flash_dsmall_bwd_dq_dbias_kernel<4>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);
template __global__ void flash_dsmall_bwd_dq_dbias_kernel<8>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, float);

// ===== C++ wrappers called from flash_api_bias.cpp =====

template <int kHeadDim>
void flash_dsmall_fwd_impl(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &bias_table_t,
    at::Tensor &out, at::Tensor &softmax_lse,
    float softmax_scale, int window_size_left, int window_size_right,
    cudaStream_t stream
) {
    const int batch_size = q.size(0);
    const int seqlen = q.size(1);
    const int num_heads = q.size(2);
    const int heads_per_group = 32;
    dim3 grid(batch_size, (num_heads + heads_per_group - 1) / heads_per_group);
    dim3 block(kThreads);
    flash_dsmall_fwd_kernel<kHeadDim><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        bias_table_t.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        softmax_lse.data_ptr<float>(),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1),
        batch_size, seqlen, num_heads,
        window_size_left, window_size_right, softmax_scale
    );
}

template <int kHeadDim>
void flash_dsmall_bwd_impl(
    const at::Tensor &dout,
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &out, const at::Tensor &softmax_lse,
    const at::Tensor &bias_table_t,
    at::Tensor &dq, at::Tensor &dk, at::Tensor &dv, at::Tensor &dbias_table_t,
    float softmax_scale, int window_size_left, int window_size_right,
    bool compute_dbias, cudaStream_t stream
) {
    const int batch_size = q.size(0);
    const int seqlen = q.size(1);
    const int num_heads = q.size(2);
    const int window_size = window_size_left + window_size_right + 1;
    const int heads_per_group = 32;
    dim3 grid(batch_size, (num_heads + heads_per_group - 1) / heads_per_group);
    dim3 block(kThreads);
    size_t smem = static_cast<size_t>(window_size) * 32 * sizeof(float)
                + static_cast<size_t>(2 * kKChunk) * 32 * kHeadDim * sizeof(float)
                + static_cast<size_t>(2 * kKChunk) * 32 * kHeadDim * sizeof(__nv_bfloat16);

    flash_dsmall_bwd_dq_dbias_kernel<kHeadDim><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
        softmax_lse.data_ptr<float>(),
        bias_table_t.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr()),
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
        window_size_left, window_size_right, softmax_scale
    );
}

// Public dispatch functions
void flash_attn_bias_dsmall_fwd_cuda(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &bias_table_t,
    at::Tensor &out, at::Tensor &softmax_lse,
    float softmax_scale, int window_size_left, int window_size_right,
    cudaStream_t stream
) {
    const int D = q.size(3);
    if (D == 2) flash_dsmall_fwd_impl<2>(q, k, v, bias_table_t, out, softmax_lse, softmax_scale, window_size_left, window_size_right, stream);
    else if (D == 4) flash_dsmall_fwd_impl<4>(q, k, v, bias_table_t, out, softmax_lse, softmax_scale, window_size_left, window_size_right, stream);
    else if (D == 8) flash_dsmall_fwd_impl<8>(q, k, v, bias_table_t, out, softmax_lse, softmax_scale, window_size_left, window_size_right, stream);
    else TORCH_CHECK(false, "flash_dsmall: unsupported head_dim=", D);
}

void flash_attn_bias_dsmall_bwd_cuda(
    const at::Tensor &dout,
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &out, const at::Tensor &softmax_lse,
    const at::Tensor &bias_table_t,
    at::Tensor &dq, at::Tensor &dk, at::Tensor &dv, at::Tensor &dbias_table_t,
    float softmax_scale, int window_size_left, int window_size_right,
    bool compute_dbias, cudaStream_t stream
) {
    const int D = q.size(3);
    if (D == 2) flash_dsmall_bwd_impl<2>(dout, q, k, v, out, softmax_lse, bias_table_t, dq, dk, dv, dbias_table_t, softmax_scale, window_size_left, window_size_right, compute_dbias, stream);
    else if (D == 4) flash_dsmall_bwd_impl<4>(dout, q, k, v, out, softmax_lse, bias_table_t, dq, dk, dv, dbias_table_t, softmax_scale, window_size_left, window_size_right, compute_dbias, stream);
    else if (D == 8) flash_dsmall_bwd_impl<8>(dout, q, k, v, out, softmax_lse, bias_table_t, dq, dk, dv, dbias_table_t, softmax_scale, window_size_left, window_size_right, compute_dbias, stream);
    else TORCH_CHECK(false, "flash_dsmall: unsupported head_dim=", D);
}
