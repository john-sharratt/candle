// The device half of sampling rows constrained to an allow-list.
//
// A constrained row samples among a handful of allowed tokens. Its device work
// is a gather of those logits and either their argmax or their softmax at the
// row's temperature; the host then draws from the probabilities (top-k, top-p,
// multinomial) exactly as for any probability vector. One block per row, every
// row of a dispatch in one launch, and the results written into a buffer the
// caller already holds — so a constrained row allocates nothing.
//
// Inputs, per row r:
//   logits[r]    device address of the row's first logit
//   allow_off[r] first entry of the row's allow-list in `allow`
//   allow_len[r] its length
//   temp[r]      temperature; <= 0 asks for the argmax
//   out_off[r]   first output word of the row
// `allow` holds token ids; BANNED marks an allowed position the row may not
// produce, which reads as -inf and so draws with probability zero.
//
// Output, per row: with temperature <= 0, one word — the local index of the
// first maximum; otherwise `allow_len` F32 probabilities.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define ALLOW_BLOCK 256
#define BANNED 0xFFFFFFFFu

// dtype codes: 0 = F32, 1 = F16, 2 = BF16.
__device__ __forceinline__ float load_logit(uint64_t row, uint32_t token, int dtype) {
    switch (dtype) {
        case 1: return __half2float(reinterpret_cast<const __half*>(row)[token]);
        case 2: return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(row)[token]);
        default: return reinterpret_cast<const float*>(row)[token];
    }
}

__device__ __forceinline__ float allowed_logit(uint64_t row, uint32_t token, int dtype) {
    return token == BANNED ? -INFINITY : load_logit(row, token, dtype);
}

extern "C" __global__ void allow_list_rows_kernel(
    const uint64_t* __restrict__ logits,
    const uint32_t* __restrict__ allow,
    const uint32_t* __restrict__ allow_off,
    const uint32_t* __restrict__ allow_len,
    const float* __restrict__ temp,
    const uint32_t* __restrict__ out_off,
    int dtype,
    uint32_t* __restrict__ out
) {
    __shared__ float s_val[ALLOW_BLOCK];
    __shared__ uint32_t s_idx[ALLOW_BLOCK];

    const uint32_t r = blockIdx.x;
    const uint64_t row = logits[r];
    const uint32_t* tokens = allow + allow_off[r];
    const uint32_t n = allow_len[r];
    const float t = temp[r];
    uint32_t* dst = out + out_off[r];

    // Maximum, with the lowest index among equal values — what an argmax over
    // the gathered row returns.
    float best = -INFINITY;
    uint32_t best_i = 0;
    bool any = false;
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        float v = allowed_logit(row, tokens[j], dtype);
        if (!any || v > best) {
            best = v;
            best_i = j;
            any = true;
        }
    }
    s_val[threadIdx.x] = any ? best : -INFINITY;
    s_idx[threadIdx.x] = any ? best_i : UINT32_MAX;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = s_val[threadIdx.x + s];
            uint32_t oi = s_idx[threadIdx.x + s];
            if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const float row_max = s_val[0];
    const uint32_t row_argmax = s_idx[0];
    __syncthreads();

    if (t <= 0.0f) {
        if (threadIdx.x == 0) {
            dst[0] = row_argmax == UINT32_MAX ? 0u : row_argmax;
        }
        return;
    }

    // Softmax of logit / t: the maximum of the scaled row is the scaled
    // maximum, since t > 0.
    const float m = row_max / t;
    float sum = 0.0f;
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        sum += expf(allowed_logit(row, tokens[j], dtype) / t - m);
    }
    s_val[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_val[threadIdx.x] += s_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float total = s_val[0];
    float* probs = reinterpret_cast<float*>(dst);
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        probs[j] = expf(allowed_logit(row, tokens[j], dtype) / t - m) / total;
    }
}

// Launch one block per row over `rows` rows on `stream`.
extern "C" void run_allow_list_rows(
    const uint64_t* logits,
    const uint32_t* allow,
    const uint32_t* allow_off,
    const uint32_t* allow_len,
    const float* temp,
    const uint32_t* out_off,
    uint32_t rows,
    int dtype,
    uint32_t* out,
    void* stream
) {
    if (rows == 0) {
        return;
    }
    allow_list_rows_kernel<<<rows, ALLOW_BLOCK, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, allow, allow_off, allow_len, temp, out_off, dtype, out);
}
