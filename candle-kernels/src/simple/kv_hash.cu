// =============================================================================
// kv_hash — a content hash over the K/V a set of slots actually holds
// =============================================================================
//
// The integrity primitive behind `kv_cache::chunked::kv_integrity`. One block per
// band; each block hashes `lens[b]` bytes at `ptrs[b]` and folds the result into
// its slot's accumulator. The plan is built host-side, so the kernel indexes no
// arena table and resolves no gid — it is handed addresses, exactly like
// `kv_migrate_copy`.
//
//   Grid:  (n_bands, 1, 1)
//   Block: (256, 1, 1)
//
// # The combine is commutative on purpose
//
// Blocks retire in whatever order the scheduler picks, so a fold that depended on
// order would return a different hash for identical bytes and the check would
// report corruption on every call. `atomicAdd` over `uint64_t` is commutative and
// associative, so the accumulator is a function of the set of bands and nothing
// else.
//
// Addition alone is not enough, though: two bands holding identical bytes would
// contribute identical terms, and a plan that swapped them — the exact fault this
// is meant to catch, a band pointing at another band's slot — would sum the same.
// So each band's hash is multiplied by an odd, caller-supplied seed before the
// add. Odd because an odd multiplier is invertible modulo 2^64, which keeps the
// mixing from collapsing a term to zero.
//
// # Why a reduction per block rather than one thread per band
//
// A band is a few hundred bytes to a few KB. One thread per band would read it
// serially at one word per instruction; a block strides it 256 words at a time and
// reduces in shared memory, which is the difference between a bandwidth-bound scan
// and a latency-bound one. The hash is therefore over 8-byte words, with the tail
// bytes folded in separately so a length that is not a multiple of eight still
// contributes every byte.

#include <cstdint>
#include <cuda_runtime.h>

// FNV-1a's 64-bit prime, applied to whole words. Not a cryptographic hash and does
// not need to be: this answers "are these the same bytes as before", where the
// adversary is a stray memcpy rather than someone searching for a collision.
__device__ __forceinline__ uint64_t mix64(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 0x100000001b3ULL;
    return h;
}

__global__ void kv_hash_kernel(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ lens,
    const int32_t* __restrict__ slot_of,
    const uint64_t* __restrict__ seeds,
    int n_bands,
    uint64_t* __restrict__ out
) {
    const int b = blockIdx.x;
    if (b >= n_bands) return;

    const int64_t len = lens[b];
    if (len <= 0) return;
    const char* base = (const char*)ptrs[b];
    const int t = threadIdx.x;
    const int stride = blockDim.x;

    // Whole 8-byte words first. Unaligned bases are read byte-wise below rather
    // than with a misaligned 64-bit load, which is undefined on this hardware.
    const bool aligned = (((uintptr_t)base) & 7) == 0;
    const int64_t n_words = aligned ? (len >> 3) : 0;

    uint64_t acc = 0xcbf29ce484222325ULL; // FNV-1a offset basis
    if (aligned) {
        const uint64_t* w = (const uint64_t*)base;
        for (int64_t i = t; i < n_words; i += stride) {
            // The index is folded in as well as the value, so a permutation of a
            // band's own words is not a collision either.
            acc = mix64(acc, w[i] ^ (uint64_t)i);
        }
    }
    // The tail: every byte past the last whole word, and the whole band when the
    // base is unaligned.
    for (int64_t i = n_words * 8 + t; i < len; i += stride) {
        acc = mix64(acc, (uint64_t)(unsigned char)base[i] ^ (uint64_t)i);
    }

    // Block reduction. Commutative within the block for the same reason it is
    // commutative across blocks — the lanes' strides interleave arbitrarily.
    __shared__ uint64_t partial[256];
    partial[t] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) {
            partial[t] += partial[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        const int slot = slot_of[b];
        // `seeds[b]` is odd by construction on the host, so the multiply is
        // invertible and cannot fold a band's contribution to zero.
        atomicAdd(out + slot, partial[0] * seeds[b]);
    }
}

extern "C" void run_kv_hash(
    const int64_t*  ptrs,
    const int64_t*  lens,
    const int32_t*  slot_of,
    const uint64_t* seeds,
    int             n_bands,
    uint64_t*       out,
    cudaStream_t    stream
) {
    if (n_bands <= 0) return;
    const int threads = 256;
    kv_hash_kernel<<<n_bands, threads, 0, stream>>>(
        ptrs, lens, slot_of, seeds, n_bands, out
    );
}
