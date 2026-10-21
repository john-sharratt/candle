// =============================================================================
// fletcher32 — the KV-chunk "golden" integrity checksum
// =============================================================================
//
// Computes a Fletcher-32 checksum over each record of a host-built plan, one
// CUDA block per record. It shares the (ptr, len) plan model of
// simple/kv_migrate.cu: the caller resolves every chunk's device address, so
// the kernel needs no arena-table indexing and reads the quantized KV bytes in
// place on the GPU. Only the tiny u32-per-record output leaves the device — the
// KV data never makes a round trip to host to be checksummed.
//
// This golden is taken over the freshly-quantized arena bytes, before any
// device→host copy, so a later CPU recompute (candle-core `fletcher32`) over
// the warm/cold copy detects corruption introduced by the DtoH copy or on disk
// — corruption the old host-computed CRC (taken AFTER the copy) could not see.
//
// Math — the closed form of Fletcher-32, which parallelises as two independent
// sum-reductions (no ordered combine needed):
//
//   sum1 = ( Σ_m w[m] )              mod 65535
//   sum2 = ( Σ_m (N - m) · w[m] )    mod 65535        (= Σ_k prefixsum_k)
//   checksum = (sum2 << 16) | sum1
//
// where w[m] is the m-th little-endian 16-bit word of the record's byte span,
// N the total word count, and an odd trailing byte forms a final word with a
// zero high byte. Because (a mod M) ≡ a (mod M), reducing only the final totals
// is bit-identical to a reduce-every-word reference. The u64 accumulators hold
// the un-reduced sums without overflow for any record up to ~33 MB
// (N < sqrt(2^64 / 65535) ≈ 1.67e7 words).
//
//   Grid:  (n_records, 1, 1)
//   Block: (FLETCHER_BLOCK, 1, 1)   — a multiple of the warp size
//
// An empty record (len == 0) yields checksum 0, matching the CPU reference.

#include <cstdint>
#include <cuda_runtime.h>

#define FLETCHER_BLOCK 256
#define FLETCHER_WARPS (FLETCHER_BLOCK / 32)
#define FLETCHER_MOD 65535u

// One word w at index m into both sums.
__device__ __forceinline__ void fletcher_word(
    uint64_t& acc1, uint64_t& acc2, uint64_t w, int64_t m, int64_t N
) {
    acc1 += w;
    acc2 += (uint64_t)(N - m) * w;
}

// Sum `v` across the warp; lane 0 holds the total.
__device__ __forceinline__ uint64_t fletcher_warp_sum(uint64_t v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__global__ void fletcher32_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ byte_lens,
    uint32_t* __restrict__ out,
    int n_records
) {
    const int r = blockIdx.x;
    if (r >= n_records) return;

    const int64_t len = byte_lens[r];
    const int64_t nfull = len >> 1;                 // full 16-bit words
    const int odd = (int)(len & 1);                 // trailing half-word?
    const int64_t N = nfull + odd;                  // total words incl. tail
    const uint8_t* __restrict__ base = (const uint8_t*)src_ptrs[r];
    const int t = threadIdx.x;
    const int stride = blockDim.x;

    uint64_t acc1 = 0;
    uint64_t acc2 = 0;

    if ((((uintptr_t)base) & 1u) == 0) {
        // 2-byte aligned. The body is read 16 bytes — eight words — per load
        // from the first 16-byte boundary; the words before it and after the
        // last whole vector are read one at a time.
        const uint16_t* __restrict__ w16 = (const uint16_t*)base;
        int64_t head = (int64_t)(((16u - ((uintptr_t)base & 15u)) & 15u) >> 1);
        if (head > nfull) head = nfull;
        for (int64_t m = t; m < head; m += stride) {
            fletcher_word(acc1, acc2, w16[m], m, N);
        }
        const uint4* __restrict__ vec = (const uint4*)(w16 + head);
        const int64_t nvec = (nfull - head) >> 3;
        for (int64_t k = t; k < nvec; k += stride) {
            const uint4 q = vec[k];
            const int64_t m0 = head + (k << 3);
            const uint32_t parts[4] = {q.x, q.y, q.z, q.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                fletcher_word(acc1, acc2, parts[i] & 0xffffu, m0 + 2 * i, N);
                fletcher_word(acc1, acc2, parts[i] >> 16, m0 + 2 * i + 1, N);
            }
        }
        for (int64_t m = head + (nvec << 3) + t; m < nfull; m += stride) {
            fletcher_word(acc1, acc2, w16[m], m, N);
        }
    } else {
        // Unaligned: assemble each little-endian word from bytes.
        for (int64_t m = t; m < nfull; m += stride) {
            uint64_t w = (uint64_t)base[2 * m] | ((uint64_t)base[2 * m + 1] << 8);
            fletcher_word(acc1, acc2, w, m, N);
        }
    }

    // Odd trailing byte: the final word at index nfull, weight (N - nfull) = 1.
    if (odd && t == 0) {
        uint64_t w = base[len - 1];
        acc1 += w;
        acc2 += w;
    }

    // Reduce within each warp by shuffle, then across the block's warps.
    __shared__ uint64_t s1[FLETCHER_WARPS];
    __shared__ uint64_t s2[FLETCHER_WARPS];
    acc1 = fletcher_warp_sum(acc1);
    acc2 = fletcher_warp_sum(acc2);
    const int lane = t & 31;
    const int warp = t >> 5;
    if (lane == 0) {
        s1[warp] = acc1;
        s2[warp] = acc2;
    }
    __syncthreads();
    if (t == 0) {
        uint64_t total1 = 0;
        uint64_t total2 = 0;
        #pragma unroll
        for (int i = 0; i < FLETCHER_WARPS; ++i) {
            total1 += s1[i];
            total2 += s2[i];
        }
        uint32_t sum1 = (uint32_t)(total1 % FLETCHER_MOD);
        uint32_t sum2 = (uint32_t)(total2 % FLETCHER_MOD);
        out[r] = (sum2 << 16) | sum1;
    }
}

extern "C" void run_fletcher32(
    const int64_t* src_ptrs,
    const int64_t* byte_lens,
    uint32_t* out,
    int n_records,
    cudaStream_t stream
) {
    if (n_records <= 0) return;
    fletcher32_kernel<<<n_records, FLETCHER_BLOCK, 0, stream>>>(
        src_ptrs, byte_lens, out, n_records
    );
}
