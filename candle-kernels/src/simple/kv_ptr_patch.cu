// =============================================================================
// KV band-pointer patch: rewrite resolved device addresses after a compaction
// =============================================================================
// A KV compaction relocates chunk slots to lower physical addresses. Nothing on
// the device holds a *gid* — every attention and store kernel reaches its bytes
// through `TokenSlice.kvheads_ptr` → a `KvHead[n_kv_head]` record holding eight
// raw band addresses per head (`k_ptr[4]`, `v_ptr[4]`), resolved on the host when
// the record was built. So what a compaction has to fix on the device is a set of
// 8-byte pointer words, and this kernel is the one launch that fixes them.
//
// WHY THIS IS A SCATTER AND NOT A SEARCH
//
// The obvious shape is a translation table of `old_addr → new_addr` that a kernel
// binary-searches for every pointer word in every live record. That is the right
// shape when the mapping is all you have. It is not the situation here: the host
// rewrites each chunk's `HeadGids` during the same pass, so at the moment it
// writes a new gid it knows the record it belongs to and the exact slot within
// it — `record_addr + offset(head, palette, is_value)`. Emitting the word address
// alongside the new value costs nothing there and turns the device side from
// `O(records · 8 · n_head · log table)` into `O(words changed)`, with no table to
// stage and no shared memory to size.
//
// It also removes the per-chunk host→device record upload the alternative keeps:
// `serialize_kv_heads` builds a whole fresh record on the host and ships it, so a
// pass moving a few thousand slots is a few thousand small `memcpy_htod`s. This
// moves two arrays and writes the words in place.
//
// WHY IT IS SAFE TO WRITE IN PLACE
//
// A record is refcount-shared by every holder of its chunk, and the KV cache's
// rule is that records are never rewritten — a relocation mints a fresh one. That
// rule exists because the prior relocation pass moved a chunk *for one holder*,
// leaving the others on the old chunk, so a shared record could not describe both.
// A perfect compaction moves each slot once, globally, and rewrites every holder,
// so there is exactly one correct address afterwards and patching the shared
// record is precisely right.
//
// Two further conditions make it a write and not a race, and both are the
// caller's to hold:
//   * The pass runs inside the arena window, so no forward is in flight and no
//     kernel is reading a record while this writes it.
//   * A compaction's sources and destinations are DISJOINT — destinations are
//     gaps, sources are live slots above the packed frontier — so no word this
//     writes is another pair's `old` value, and the patch cannot chain.
//
// COALESCING
//
// The caller sorts the pairs by word address. That matters more than it looks:
// one head's eight band pointers are 64 contiguous bytes, so a sorted run of
// pairs from one record lands in one or two sectors instead of eight scattered
// ones. The reads of `addrs`/`vals` are perfectly coalesced either way.
//
// Registers stay in single digits — two loads, one store, a grid-stride index —
// so occupancy is bounded by blocks, not by the kernel, and the grid is sized to
// fill the device rather than to the pair count.

#include <cuda_runtime.h>

// One 8-byte word written per thread.
//
// `__ldg` on both inputs: they are read once, never written, and marking them so
// keeps them out of the L1 write-allocate path that the scattered stores want.
extern "C" __global__ void kv_ptr_patch_kernel(
    const unsigned long long* __restrict__ addrs,
    const unsigned long long* __restrict__ vals,
    int n_words)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += stride) {
        unsigned long long* dst = (unsigned long long*)__ldg(&addrs[i]);
        *dst = __ldg(&vals[i]);
    }
}

// Verification pass: count words that do NOT already hold `vals[i]`.
//
// Separate from the patch so the patch keeps its two-loads-one-store shape. Used
// by the compaction's own self-check and by the benchmark to prove the patch
// landed, rather than trusting that it did — a wrong band pointer does not fault,
// it reads whatever now occupies the slot, so "it ran" is not evidence.
extern "C" __global__ void kv_ptr_verify_kernel(
    const unsigned long long* __restrict__ addrs,
    const unsigned long long* __restrict__ vals,
    int n_words,
    unsigned int* __restrict__ mismatches)
{
    const int stride = gridDim.x * blockDim.x;
    unsigned int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += stride) {
        const unsigned long long* dst = (const unsigned long long*)__ldg(&addrs[i]);
        if (*dst != __ldg(&vals[i])) {
            local += 1;
        }
    }
    // One atomic per thread that found anything, not one per word.
    if (local != 0) {
        atomicAdd(mismatches, local);
    }
}

// Blocks sized to fill the device rather than to the word count: the grid-stride
// loop covers any `n_words`, and a pass that moved forty slots should not launch
// a grid of one block on a card with 100+ SMs.
static inline int kv_ptr_patch_blocks(int n_words, int threads)
{
    int by_work = (n_words + threads - 1) / threads;
    if (by_work < 1) by_work = 1;
    // 8 waves of 256 threads per SM is enough to saturate a scatter; more only
    // adds tail imbalance.
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (sms < 1) sms = 1;
    const int by_device = sms * 8;
    return by_work < by_device ? by_work : by_device;
}

extern "C" void run_kv_ptr_patch(
    const unsigned long long* addrs,
    const unsigned long long* vals,
    int n_words,
    void* stream)
{
    if (n_words <= 0) return;
    const int threads = 256;
    kv_ptr_patch_kernel<<<kv_ptr_patch_blocks(n_words, threads), threads, 0,
                          (cudaStream_t)stream>>>(addrs, vals, n_words);
}

extern "C" void run_kv_ptr_verify(
    const unsigned long long* addrs,
    const unsigned long long* vals,
    int n_words,
    unsigned int* mismatches,
    void* stream)
{
    if (n_words <= 0) return;
    const int threads = 256;
    kv_ptr_verify_kernel<<<kv_ptr_patch_blocks(n_words, threads), threads, 0,
                           (cudaStream_t)stream>>>(addrs, vals, n_words, mismatches);
}
