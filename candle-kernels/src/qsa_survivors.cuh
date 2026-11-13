// =============================================================================
// qsa_survivors — a block's running top-`keep` over streamed u64 keys
// =============================================================================
//
// The ranking the QSA selection (`simple/qsa_topk.cu`) and the shadow scan
// (`simple/qsa_shadow.cu`) share. A block streams its candidates through a
// shared buffer against a running threshold:
//
//   - The buffer's first `count` slots hold every key still in the running,
//     in no order. A chunk appends only keys above the threshold.
//   - When the buffer could not absorb another chunk, a TRIM selects the top
//     `keep` of it and compacts them to the front, and the smallest of them
//     becomes the threshold: a later key that does not beat it cannot be in
//     the top `keep`, because `keep` keys above it are already known.
//
// The trim is an MSB radix select over the keys, held in registers: a
// histogram of the next byte of every key still matching the chosen prefix
// names the byte the `keep`-th key has, and the walk stops at the first byte
// whose bin holds exactly the keys still needed. Nothing is sorted.
//
// A key is `bits(score) << 32 | (0xFFFFFFFF − b)`: scores are sums of ReLUs
// and never negative, so the IEEE-754 bit pattern orders as the value, and the
// key is a total order that is exactly `(score desc, block asc)` compared as
// an unsigned integer. No real key is zero (`b < 0xFFFFFFFF`), so zero pads a
// key list harmlessly.
//
// `keep ≤ CAP − THREADS` is what bounds the append: a chunk adds at most
// THREADS keys and is checked immediately after.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace qsa_survivors {

constexpr int THREADS = 256;
constexpr int CAP = 1024;          // shared candidate buffer, u64 keys
// The append bound: a chunk adds at most THREADS keys into the free space
// below the survivors, and the trim fires as soon as the free space could not
// absorb another chunk.
constexpr int MAX_KEEP = CAP - THREADS;

__device__ __forceinline__ uint32_t key_block(unsigned long long key) {
    return 0xFFFFFFFFu - (uint32_t)(key & 0xFFFFFFFFull);
}

__device__ __forceinline__ unsigned long long score_key(float score, int b) {
    return ((unsigned long long)__float_as_uint(score) << 32)
        | (unsigned long long)(0xFFFFFFFFu - (uint32_t)b);
}

/// The score a key was made from.
__device__ __forceinline__ float key_score(unsigned long long key) {
    return __uint_as_float((uint32_t)(key >> 32));
}

constexpr int PER_THREAD = CAP / THREADS;
constexpr int BINS = 256;
constexpr int WARP = 32;

// The survivor buffer and the radix select's state, in shared memory.
struct Survivors {
    unsigned long long* buf;  // CAP keys; [0, count) live, in no order
    int* count;
    unsigned long long* thr;  // a key must exceed this to enter
    int* hist;                // BINS counters
    int* pick;                // the select's answer: digit, keys above it, its bin
};

__device__ __forceinline__ void survivors_reset(const Survivors& s, int tid) {
    if (tid == 0) {
        *s.count = 0;
        *s.thr = 0ull;
    }
    __syncthreads();
}

// Reduce the buffer's `n` keys to its top `keep`, compacted to the front, and
// set the threshold to the smallest of them. `n ≥ keep`. Every thread calls
// this, so its barriers are uniform.
__device__ void survivors_select(const Survivors& s, int n, int keep, int tid) {
    unsigned long long k[PER_THREAD];
    bool live[PER_THREAD];
#pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) {
        const int i = tid + j * THREADS;
        live[j] = i < n;
        k[j] = live[j] ? s.buf[i] : 0ull;
    }

    // The top `keep` are the keys whose masked bits are at least `prefix`.
    unsigned long long prefix = 0ull, mask = 0ull;
    int need = keep;
    if (n > keep) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            s.hist[tid] = 0;
            __syncthreads();
#pragma unroll
            for (int j = 0; j < PER_THREAD; ++j) {
                if (live[j] && (k[j] & mask) == prefix) {
                    atomicAdd(&s.hist[(int)((k[j] >> shift) & 0xFFull)], 1);
                }
            }
            __syncthreads();
            if (tid < WARP) {
                // Lane l holds bins [8l, 8l + 8); `above` counts the keys in
                // every higher lane's bins.
                int bins[BINS / WARP];
                int t = 0;
#pragma unroll
                for (int b = 0; b < BINS / WARP; ++b) {
                    bins[b] = s.hist[tid * (BINS / WARP) + b];
                    t += bins[b];
                }
                int incl = t;
#pragma unroll
                for (int off = 1; off < WARP; off <<= 1) {
                    const int v = __shfl_down_sync(0xFFFFFFFFu, incl, off);
                    if (tid + off < WARP) incl += v;
                }
                int above = incl - t;
                if (above < need && need <= incl) {
                    for (int b = BINS / WARP - 1; b >= 0; --b) {
                        if (above + bins[b] >= need) {
                            s.pick[0] = tid * (BINS / WARP) + b;
                            s.pick[1] = above;
                            s.pick[2] = bins[b];
                            break;
                        }
                        above += bins[b];
                    }
                }
            }
            __syncthreads();
            prefix |= (unsigned long long)s.pick[0] << shift;
            mask |= 0xFFull << shift;
            need -= s.pick[1];
            const bool whole_bin = s.pick[2] == need;
            __syncthreads();
            if (whole_bin) break;
        }
    }

    if (tid == 0) {
        *s.count = 0;
        *s.thr = ~0ull;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) {
        if (live[j] && (k[j] & mask) >= prefix) {
            s.buf[atomicAdd(s.count, 1)] = k[j];
            atomicMin(s.thr, k[j]);
        }
    }
    __syncthreads();
}

// Offer one chunk's keys (each thread's `key`, `0` for none) to the buffer,
// then trim it if another chunk might not fit. Every thread calls this the
// same number of times, so the barriers inside are uniform.
//
// `n` is each thread's own copy of the buffer's count, advanced by the
// barrier's own tally of the chunk. The trim decision has to be the same in
// every thread, and a count read back from shared memory is not: a thread
// that reads it late can see the next chunk's appends from a thread that ran
// ahead, decide differently, and leave the block split across two barriers.
__device__ __forceinline__ void survivors_offer(
    const Survivors& s,
    unsigned long long key,
    int keep,
    int& n,
    int tid
) {
    const bool append = key > *s.thr;
    if (append) {
        const int slot = atomicAdd(s.count, 1);
        s.buf[slot] = key;
    }
    n += __syncthreads_count(append);
    if (n > CAP - THREADS) {
        survivors_select(s, n, keep, tid);
        n = keep;
    }
}

// Settle the buffer to exactly its top `keep` once the pool is streamed.
__device__ __forceinline__ void survivors_finish(const Survivors& s, int keep, int n, int tid) {
    survivors_select(s, n, keep, tid);
}

} // namespace qsa_survivors
