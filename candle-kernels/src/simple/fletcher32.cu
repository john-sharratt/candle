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
//   Block: (FLETCHER_BLOCK, 1, 1)   — a power of two
//
// An empty record (len == 0) yields checksum 0, matching the CPU reference.

#include <cstdint>
#include <cuda_runtime.h>

#define FLETCHER_BLOCK 256
#define FLETCHER_MOD 65535u

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
        // Fast path: 2-byte aligned → coalesced 16-bit loads across the warp.
        const uint16_t* __restrict__ w16 = (const uint16_t*)base;
        for (int64_t m = t; m < nfull; m += stride) {
            uint64_t w = w16[m];
            acc1 += w;
            acc2 += (uint64_t)(N - m) * w;
        }
    } else {
        // Unaligned: assemble each little-endian word from bytes.
        for (int64_t m = t; m < nfull; m += stride) {
            uint64_t w = (uint64_t)base[2 * m] | ((uint64_t)base[2 * m + 1] << 8);
            acc1 += w;
            acc2 += (uint64_t)(N - m) * w;
        }
    }

    // Odd trailing byte: the final word at index nfull, weight (N - nfull) = 1.
    if (odd && t == 0) {
        uint64_t w = base[len - 1];
        acc1 += w;
        acc2 += w;
    }

    // Block-reduce the two sums.
    __shared__ uint64_t s1[FLETCHER_BLOCK];
    __shared__ uint64_t s2[FLETCHER_BLOCK];
    s1[t] = acc1;
    s2[t] = acc2;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (t < off) {
            s1[t] += s1[t + off];
            s2[t] += s2[t + off];
        }
        __syncthreads();
    }

    if (t == 0) {
        uint32_t sum1 = (uint32_t)(s1[0] % FLETCHER_MOD);
        uint32_t sum2 = (uint32_t)(s2[0] % FLETCHER_MOD);
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
