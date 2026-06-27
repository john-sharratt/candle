// Batched Penalty and Sampling Kernel for efficient token generation
// This kernel processes all sequences in a batch with a single kernel launch,
// eliminating the per-sequence kernel launch overhead.
// 
// Key features:
// - Single kernel launch for N sequences (was N launches)
// - O(1) bitset lookup for recent tokens
// - O(n) radix select for top-k (avoids full vocab sort)
// - Truly branchless penalty application
// - Temperature <= 0 triggers argmax (greedy) path
// - Supports FP32, FP16, BF16, and FP8 E4M3 input logits
// - float4 vectorized loads for FP32 (4x memory bandwidth)

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>  // For std::is_same_v
#include "philox.cuh"
#include "../fast_exp.cuh"

namespace batched_sampling {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int MAX_TOP_K = 256;         // Maximum k for top-k sampling

// Tiled processing constants
// Each tile loads TILE_SIZE floats into shared memory, processes them once,
// then reuses for max-finding, softmax-sum, and top-k collection
constexpr int TILE_SIZE = 1024;        // Floats per tile (4KB shared memory)
constexpr int TILE_FLOATS_PER_THREAD = TILE_SIZE / THREADS_PER_BLOCK;  // 1 float/thread (1024 threads)

// Radix select threshold: use O(n) radix select when top_k is below this fraction
// of vocab_size. For small top_k values, radix select is more efficient than
// collecting all candidates and sorting. Value of 4 means: use radix when k < vocab/4
constexpr int RADIX_SELECT_DIVISOR = 4;

// Bitset is allocated dynamically via shared memory - supports any vocab size
// Limited only by GPU shared memory (typically 48-100KB per SM)

// ============================================================================
// Input Data Type Enum (for runtime dispatch)
// ============================================================================

enum class LogitDType : int {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP8_E4M3 = 3
};

// ============================================================================
// Atomic Helpers
// ============================================================================

// Atomic max for floats using CAS loop (for non-negative floats)
// Uses the fact that for non-negative floats, integer comparison gives same order
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    if (val <= 0.0f) return;  // DRY penalties are always positive
    
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        // For positive floats, bit pattern comparison == float comparison
        if (__int_as_float(assumed) >= val) return;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

// ============================================================================
// Type Conversion Helpers
// ============================================================================

// Load and convert any supported type to float
template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int idx);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* ptr, int idx) {
    return __ldg(&ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half* ptr, int idx) {
    return __half2float(__ldg(&ptr[idx]));
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(__ldg(&ptr[idx]));
}

// FP8 E4M3 - software conversion works on all architectures (SM80+)
template <>
__device__ __forceinline__ float load_as_float<__nv_fp8_e4m3>(const __nv_fp8_e4m3* ptr, int idx) {
    return float(ptr[idx]);
}

// ============================================================================
// Penalty Configuration (per-batch or global)
// ============================================================================

struct PenaltyParams {
    // Repeat penalty (applied to tokens seen in recent context)
    // 1.0 = disabled, >1.0 penalizes repeats
    float repeat_penalty;
    
    // Frequency penalty (scales with token occurrence count)
    // 0.0 = disabled
    float frequency_penalty;
    
    // Presence penalty (binary: was token seen at all?)
    // 0.0 = disabled
    float presence_penalty;
    
    // DRY (Don't Repeat Yourself) penalty - penalizes sequence repetitions
    // Detects when the model is about to continue a repeated n-gram sequence
    // 0.0 = disabled, typical values 0.5-1.5
    float dry_multiplier;
    // Base value for penalty calculation: penalty = multiplier * base^(match_length - allowed_length)
    // Typical value: 1.75
    float dry_base;
    // Minimum match length before penalty applies (allows short repeats)
    // Typical value: 2-4
    int32_t dry_allowed_length;
    // How far back to look for sequence matches (0 = use full recent_tokens)
    int32_t dry_range;
    
    // Cross-turn penalty: flat additive penalty for tokens seen in PRIOR turns.
    // Lighter than presence_penalty (which covers current-turn tokens).
    // 0.0 = disabled. Typical: 0.01 - 0.3.
    float cross_turn_penalty;
    // Per-token counts from prior turns. Layout: [batch_size, vocab_size] or null.
    // Value > 0 means token appeared in a previous conversation turn.
    const int32_t* cross_turn_counts;

    // EOS boost (dynamic termination encouragement)
    // 0.0 = disabled, additive logit boost
    float eos_boost;
    int32_t eos_token_id;
    // Dynamic EOS: scale eos_boost with a linear ramp from ramp_start to ramp_len.
    // effective = eos_boost * clamp((current_len - ramp_start) / (ramp_len - ramp_start), 0, 1) * max_multiplier.
    // 0 = disabled (use flat eos_boost).
    int32_t eos_ramp_start;
    int32_t eos_ramp_len;
    float eos_boost_max_multiplier;
    int32_t current_len;   // Current generated sequence length for this batch item.

    // Segment-close boost — same ramp formula as EOS,
    // but targets the segment-close token and uses segment_len.
    // 0.0 = disabled.
    float segment_close_boost;
    int32_t segment_close_token_id;
    int32_t segment_close_ramp_start;
    int32_t segment_close_ramp_len;
    float segment_close_max_multiplier;
    int32_t segment_len;  // Tokens generated since the segment opened for this batch item.
    
    // Token counts for frequency/presence penalty
    // Layout: [batch_size, vocab_size] - each sequence has its own counts
    // Access: token_counts[batch_idx * vocab_size + token_id]
    // Set to null to disable frequency/presence penalties
    const int32_t* token_counts;
    int32_t vocab_size;  // Needed to index into per-sequence token_counts
    
    // Banned tokens (set to -inf)
    // Layout: [batch_size, max_banned_per_seq] or [num_banned_tokens] if shared
    // For per-sequence bans, use banned_tokens_per_seq > 0
    const int32_t* banned_tokens;
    int32_t num_banned_tokens;      // Total banned tokens (if shared)
    int32_t banned_tokens_per_seq;  // Tokens per sequence (0 = shared list)
    
    // Stencil (constrained vocabulary - only allowed tokens)
    // When provided, ONLY stencil tokens are considered for sampling.
    // Layout: [stencil_size] - array of allowed token IDs
    // Optimized for small stencils (~100 tokens) - iterates directly over stencil
    // instead of full vocab for massive performance gains.
    // Set to null to disable (use full vocabulary)
    const int32_t* stencil;
    int32_t stencil_size;           // Number of allowed tokens (0 = disabled)

    // Token suppression (the in-segment ceiling lever).
    // While this sequence is inside a segment, `suppress_penalty` is
    // subtracted from the logit of every token in `suppress_tokens`. Outside a
    // segment `suppress_penalty` is set to 0 by the caller, so tokens outside
    // the segment are untouched. The token list is shared across the batch.
    // Layout: [suppress_count] - array of suppress token IDs.
    const int32_t* suppress_tokens;
    int32_t suppress_count;         // Number of suppress tokens (0 = disabled)
    float suppress_penalty;         // Per-sequence penalty (0.0 = disabled)
};

// ============================================================================
// Shared Memory Layout (fixed portion)
// ============================================================================

// Radix select constants
constexpr int RADIX_BITS = 8;                    // Bits per pass
constexpr int RADIX_SIZE = 1 << RADIX_BITS;     // 256 buckets
constexpr int RADIX_MASK = RADIX_SIZE - 1;

// Maximum DRY cache entries - defined early for use in conditional structs
constexpr int DRY_CACHE_MAX_ENTRIES = 512;

// ============================================================================
// Conditional Shared Memory - Template-Based Size Optimization
// ============================================================================
// Fine-grained template parameters allow the compiler to eliminate unused
// shared memory, improving occupancy when features are disabled.
//
// Template parameters:
// - MAX_K: Maximum top-k value (affects topk arrays and radix histogram)
// - THREADS: Block size (affects reduction arrays)
// - USE_TILED: When true, allocates tile_buffer (4KB) for penalty path
// - USE_DRY: When true, allocates DryPenaltyCache (4KB)
// - USE_RADIX: When true, allocates radix_histogram (1KB) and radix state
//
// Shared memory breakdown:
// - Core (always):    reduction (8KB) + topk (2KB) + scalars (~32B) = ~10KB
// - USE_TILED:        tile_buffer (4KB)
// - USE_DRY:          dry_cache (4KB)  
// - USE_RADIX:        radix_histogram (1KB) + scalars (8B)
//
// Examples:
// - Full penalties + top-p: 10 + 4 + 4 + 1 = 19KB
// - Standard penalties only: 10 + 4 = 14KB
// - No penalties, just sampling: 10KB
// - Argmax only: ~10KB

// Conditional array helper - evaluates to array of size N or size 0
template <bool ENABLED, typename T, int N>
struct ConditionalArray {
    T data[N];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
    // Implicit conversion to pointer for compatibility with existing functions
    __device__ __forceinline__ operator T*() { return data; }
    __device__ __forceinline__ operator const T*() const { return data; }
};

template <typename T, int N>
struct ConditionalArray<false, T, N> {
    // Member dummy avoids illegal static local in device code
    // Note: This does take sizeof(T) space, but operator returns nullptr
    // so actual array storage is avoided. Alternative would be thread_local
    // but that's not supported in device code either.
    T _dummy;
    
    __device__ __forceinline__ T& operator[](int) { 
        // Should never be called when disabled, returns dummy to avoid UB
        return _dummy;
    }
    __device__ __forceinline__ const T& operator[](int) const {
        return _dummy;
    }
    // Implicit conversion to nullptr for disabled arrays
    __device__ __forceinline__ operator T*() { return nullptr; }
    __device__ __forceinline__ operator const T*() const { return nullptr; }
};

// Forward declaration of DryPenaltyCache (defined later, ~line 370)
struct DryPenaltyCache;

// Conditional DryPenaltyCache wrapper
// When ENABLED=true, wraps a real DryPenaltyCache
// When ENABLED=false, takes no space and all ops are no-ops
template <bool ENABLED>
struct ConditionalDryCache {
    // Actual storage - same layout as DryPenaltyCache for compatibility
    int32_t token_ids[DRY_CACHE_MAX_ENTRIES];
    float penalties[DRY_CACHE_MAX_ENTRIES];
    int num_entries;
    
    // Provide pointer access for functions expecting DryPenaltyCache*
    __device__ __forceinline__ DryPenaltyCache* as_ptr() {
        return reinterpret_cast<DryPenaltyCache*>(this);
    }
    __device__ __forceinline__ const DryPenaltyCache* as_ptr() const {
        return reinterpret_cast<const DryPenaltyCache*>(this);
    }
};

template <>
struct ConditionalDryCache<false> {
    // Empty - no storage, takes 0 bytes
    __device__ __forceinline__ DryPenaltyCache* as_ptr() { return nullptr; }
    __device__ __forceinline__ const DryPenaltyCache* as_ptr() const { return nullptr; }
};

template <int MAX_K, int THREADS, bool USE_TILED = true, bool USE_DRY = true, bool USE_RADIX = true>
struct SamplingSharedMem {
    // ========== CORE (always allocated) ==========
    // Parallel reduction scratch
    float reduction_max[THREADS];           // 1024 * 4 = 4 KB
    int   reduction_idx[THREADS];           // 1024 * 4 = 4 KB
    
    // Top-K candidates  
    float topk_vals[MAX_K];                 // 256 * 4 = 1 KB
    int   topk_idxs[MAX_K];                 // 256 * 4 = 1 KB
    
    // Global scalars (always needed)
    float global_max;
    float global_sum;
    int best_idx;
    int topk_count;
    
    // ========== CONDITIONAL: Radix select (USE_RADIX) ==========
    // Only needed when top_k > small threshold (typically 32)
    ConditionalArray<USE_RADIX, int, RADIX_SIZE> radix_histogram;  // 256 * 4 = 1 KB
    
    // Radix select state (compile-time conditional)
    // When USE_RADIX=false, these are optimized away
    float radix_threshold;                  // 4B - kept for simplicity
    int radix_count;                        // 4B - kept for simplicity
    
    // ========== CONDITIONAL: Tile buffer (USE_TILED) ==========
    // Stores penalized logits for reuse across max-find, softmax-sum, top-k
    ConditionalArray<USE_TILED, float, TILE_SIZE> tile_buffer;    // 1024 * 4 = 4 KB
    
    // ========== CONDITIONAL: DRY penalty cache (USE_DRY) ==========
    // Precomputed penalties for O(1) lookup
    ConditionalDryCache<USE_DRY> dry_cache;                       // ~4 KB
    
    // Bitset allocated via extern __shared__ for dynamic sizing
};

// Type alias for backward compatibility (full features)
template <int MAX_K, int THREADS>
using SamplingSharedMemFull = SamplingSharedMem<MAX_K, THREADS, true, true, true>;

// Optimized variants
template <int MAX_K, int THREADS>
using SamplingSharedMemNoPenalties = SamplingSharedMem<MAX_K, THREADS, false, false, true>;

template <int MAX_K, int THREADS>
using SamplingSharedMemStandardOnly = SamplingSharedMem<MAX_K, THREADS, true, false, true>;

template <int MAX_K, int THREADS>
using SamplingSharedMemArgmax = SamplingSharedMem<MAX_K, THREADS, false, false, false>;

// ============================================================================
// Device Helper Functions
// ============================================================================

// Build recent token bitset collaboratively
template <int THREADS>
__device__ void build_recent_bitset(
    uint32_t* __restrict__ bitset,
    const int32_t* __restrict__ recent_tokens,
    int recent_len,
    int bitset_words
) {
    const int tid = threadIdx.x;
    
    // Clear bitset
    for (int i = tid; i < bitset_words; i += THREADS) {
        bitset[i] = 0;
    }
    __syncthreads();
    
    // Guard against null pointer or empty list
    if (recent_tokens == nullptr || recent_len <= 0) {
        return;  // Bitset already cleared, nothing more to do
    }
    
    // Set bits for recent tokens
    for (int i = tid; i < recent_len; i += THREADS) {
        int token = recent_tokens[i];
        if (token >= 0 && token < bitset_words * 32) {
            atomicOr(&bitset[token >> 5], 1u << (token & 31));
        }
    }
    __syncthreads();
}

// O(1) lookup for recent token
__device__ __forceinline__ bool is_in_recent(
    const uint32_t* __restrict__ bitset,
    int token_id
) {
    return (bitset[token_id >> 5] >> (token_id & 31)) & 1;
}

// ============================================================================
// DRY (Don't Repeat Yourself) Penalty - OPTIMIZED O(n² + V) Algorithm
// ============================================================================
// DRY penalty detects when the model is about to continue a repeated n-gram.
// Given recent tokens [t0, t1, ..., tn] and candidate token t:
// 1. Look at the suffix [t_{n-k}, ..., tn] for various lengths k
// 2. Find if this suffix appeared earlier in the context
// 3. If the token after that earlier occurrence is t, penalize based on match length
//
// Example: Context "the cat sat on the cat" + candidate "sat"
//   Suffix "the cat" appeared before, and was followed by "sat"
//   So "sat" gets penalized with match_length=2
//
// OPTIMIZATION: Instead of computing O(n²) per vocab token (total O(V×n²)),
// we precompute penalties for all positions ONCE (O(n²)), then look up in O(1).
// Total complexity: O(n² + V) instead of O(V × n²)

// Note: DRY_CACHE_MAX_ENTRIES is defined earlier (line ~168) = 512

// Structure to hold precomputed DRY penalties
// Stored in shared memory for fast lookup
// Uses DRY_CACHE_MAX_ENTRIES defined at top of file
struct DryPenaltyCache {
    int32_t token_ids[DRY_CACHE_MAX_ENTRIES];   // Token IDs that have penalties
    float penalties[DRY_CACHE_MAX_ENTRIES];      // Corresponding penalties
    int num_entries;                              // Number of valid entries
};

// Precompute DRY penalties for a single sequence
// Called ONCE per batch item, O(n²) complexity
// Results stored in shared memory cache for O(1) lookup
template <int THREADS>
__device__ void precompute_dry_penalties(
    const int32_t* __restrict__ recent_tokens,  // This sequence's recent tokens
    int recent_len,                              // Actual length of recent tokens
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range,                               // How far back to look (0 = full range)
    DryPenaltyCache* cache                       // Output: shared memory cache
) {
    const int tid = threadIdx.x;
    
    // Thread 0 initializes the cache
    if (tid == 0) {
        cache->num_entries = 0;
    }
    __syncthreads();
    
    // Early exit if DRY disabled or not enough context
    if (dry_multiplier == 0.0f || recent_len < 2) {
        return;
    }
    
    // Determine search range
    int search_end = recent_len - 1;  // Last position in suffix
    int search_start = (dry_range > 0 && dry_range < recent_len) ? (recent_len - dry_range) : 0;
    int num_positions = search_end - search_start;
    
    // Temporary storage for this thread's findings
    // We'll use a small local buffer then merge
    constexpr int LOCAL_BUFFER_SIZE = 16;
    int32_t local_tokens[LOCAL_BUFFER_SIZE];
    float local_penalties[LOCAL_BUFFER_SIZE];
    int local_count = 0;
    
    // Each thread processes a subset of positions
    for (int idx = tid; idx < num_positions; idx += THREADS) {
        int match_pos = search_start + idx;
        
        // Find longest suffix match ending at match_pos
        int match_len = 0;
        for (int offset = 1; offset <= min(match_pos + 1, recent_len); offset++) {
            int suffix_idx = recent_len - offset;      // Index in suffix (from end)
            int match_idx = match_pos - offset + 1;    // Corresponding position earlier
            
            if (match_idx < search_start) break;       // Don't go before search range
            if (recent_tokens[suffix_idx] != recent_tokens[match_idx]) break;
            
            match_len = offset;
        }
        
        // If match exceeds allowed length, record penalty for next token
        if (match_len > dry_allowed_length && match_pos + 1 < recent_len) {
            int32_t next_token = recent_tokens[match_pos + 1];
            int excess = match_len - dry_allowed_length;
            float penalty = dry_multiplier * powf(dry_base, (float)excess);
            
            // Store locally first
            if (local_count < LOCAL_BUFFER_SIZE) {
                local_tokens[local_count] = next_token;
                local_penalties[local_count] = penalty;
                local_count++;
            }
        }
    }
    
    __syncthreads();
    
    // Merge local buffers into shared cache using atomic operations
    // This handles deduplication by taking max penalty per token
    for (int i = 0; i < local_count; i++) {
        int32_t token = local_tokens[i];
        float penalty = local_penalties[i];
        
        // Check if token already exists in cache
        bool found = false;
        for (int j = 0; j < cache->num_entries && !found; j++) {
            if (cache->token_ids[j] == token) {
                // Use atomic max to avoid race condition when multiple threads
                // try to update the same token's penalty simultaneously
                atomicMaxFloat(&cache->penalties[j], penalty);
                found = true;
            }
        }
        
        // Add new entry if not found
        if (!found) {
            int slot = atomicAdd(&cache->num_entries, 1);
            if (slot < DRY_CACHE_MAX_ENTRIES) {
                cache->token_ids[slot] = token;
                cache->penalties[slot] = penalty;
            }
        }
    }
    
    __syncthreads();
    
    // Final deduplication pass - merge duplicate entries (taking max)
    // Use O(n log n) approach: insertion sort by token_id, then linear merge
    // Only thread 0 does this to avoid races
    if (tid == 0 && cache->num_entries > 1) {
        int n = cache->num_entries;
        
        // Clamp to valid range
        if (n > DRY_CACHE_MAX_ENTRIES) {
            n = DRY_CACHE_MAX_ENTRIES;
        }
        
        // Step 1: Insertion sort by token_id - O(n²) but fast for small n (~100)
        // and much better cache behavior than the nested loop dedup
        for (int i = 1; i < n; i++) {
            int32_t key_tok = cache->token_ids[i];
            float key_pen = cache->penalties[i];
            int j = i - 1;
            
            while (j >= 0 && cache->token_ids[j] > key_tok) {
                cache->token_ids[j + 1] = cache->token_ids[j];
                cache->penalties[j + 1] = cache->penalties[j];
                j--;
            }
            cache->token_ids[j + 1] = key_tok;
            cache->penalties[j + 1] = key_pen;
        }
        
        // Step 2: Linear merge of adjacent duplicates - O(n)
        int write_idx = 0;
        for (int i = 1; i < n; i++) {
            if (cache->token_ids[i] == cache->token_ids[write_idx]) {
                // Merge: take max penalty
                cache->penalties[write_idx] = fmaxf(cache->penalties[write_idx], cache->penalties[i]);
            } else {
                // New unique token
                write_idx++;
                if (write_idx != i) {
                    cache->token_ids[write_idx] = cache->token_ids[i];
                    cache->penalties[write_idx] = cache->penalties[i];
                }
            }
        }
        cache->num_entries = write_idx + 1;
    }
    
    __syncthreads();
}

// Branchless lookup in precomputed DRY cache - avoids warp divergence
// All threads execute the same number of iterations regardless of match position.
// Cache is in shared memory so extra reads are cheap (typically < 100 entries).
__device__ __forceinline__ float lookup_dry_penalty(
    int32_t token_id,
    const DryPenaltyCache* cache
) {
    float penalty = 0.0f;
    for (int i = 0; i < cache->num_entries; i++) {
        // Branchless select: compiles to selp instruction (no warp divergence)
        int match = (cache->token_ids[i] == token_id);
        penalty = match ? cache->penalties[i] : penalty;
    }
    return penalty;
}

// Legacy function for backward compatibility (not used in optimized path)
// Compute DRY penalty for a single token - O(n²) per call, DO NOT USE in hot path
__device__ __forceinline__ float compute_dry_penalty(
    int32_t candidate_token,
    const int32_t* __restrict__ recent_tokens,  // This sequence's recent tokens
    int recent_len,                              // Actual length of recent tokens
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range                               // How far back to look (0 = full range)
) {
    // Early exit if DRY disabled or not enough context
    if (dry_multiplier == 0.0f || recent_len < 2) {
        return 0.0f;
    }
    
    // Determine search range
    int search_end = recent_len - 1;  // Last position in suffix
    int search_start = (dry_range > 0 && dry_range < recent_len) ? (recent_len - dry_range) : 0;
    
    // Find the longest match
    int max_match_length = 0;
    
    // For each potential match position (excluding the current suffix)
    for (int match_pos = search_start; match_pos < search_end; match_pos++) {
        // Let's find how long a suffix match we can get ending at match_pos
        int match_len = 0;
        for (int offset = 1; offset <= min(match_pos + 1, recent_len); offset++) {
            int suffix_idx = recent_len - offset;      // Index in suffix (from end)
            int match_idx = match_pos - offset + 1;    // Corresponding position earlier
            
            if (match_idx < search_start) break;       // Don't go before search range
            if (recent_tokens[suffix_idx] != recent_tokens[match_idx]) break;
            
            match_len = offset;
        }
        
        // If we found a match of length >= 1, check if the next token matches candidate
        if (match_len >= 1 && match_pos + 1 < recent_len) {
            if (recent_tokens[match_pos + 1] == candidate_token) {
                max_match_length = max(max_match_length, match_len);
            }
        }
    }
    
    // Apply penalty if match exceeds allowed length
    if (max_match_length > dry_allowed_length) {
        int excess = max_match_length - dry_allowed_length;
        return dry_multiplier * powf(dry_base, (float)excess);
    }
    
    return 0.0f;
}

// Truly branchless penalty application (per-sequence aware)
__device__ __forceinline__ float apply_penalties_branchless(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset
) {
    // Protected tokens: EOS and the segment-close token must never be penalised.
    // Penalties would suppress the model's ability to stop generating.
    // Save the logit so we can restore it for protected tokens after penalties.
    float saved_logit = logit;
    int eos_diff = token_id - params.eos_token_id;
    int eos_abs  = (eos_diff ^ (eos_diff >> 31)) - (eos_diff >> 31);
    int is_eos   = 1 - min(eos_abs, 1);
    int segment_close_diff = token_id - params.segment_close_token_id;
    int segment_close_abs  = (segment_close_diff ^ (segment_close_diff >> 31)) - (segment_close_diff >> 31);
    int is_segment_close   = 1 - min(segment_close_abs, 1);
    int is_protected = is_eos | is_segment_close;

    // 1. Repeat Penalty (truly branchless, per-sequence via recent_bitset)
    // Defensive check: token_id should always be in [0, vocab_size) but guard anyway
    uint32_t in_recent = 0;
    if (token_id >= 0) {
        in_recent = (recent_bitset[token_id >> 5] >> (token_id & 31)) & 1;
    }
    
    // Sign-based penalty: positive logits divided, negative multiplied
    uint32_t logit_bits = __float_as_uint(logit);
    uint32_t sign_bit = logit_bits >> 31;  // 0 for positive, 1 for negative
    
    float inv_penalty = __fdividef(1.0f, params.repeat_penalty);
    float sign_f = __int2float_rn(sign_bit);
    float repeat_factor = sign_f * params.repeat_penalty + (1.0f - sign_f) * inv_penalty;
    
    float in_recent_f = __int2float_rn(in_recent);
    float final_factor = in_recent_f * repeat_factor + (1.0f - in_recent_f) * 1.0f;
    logit *= final_factor;
    
    // 2. Frequency Penalty (per-sequence token counts)
    // NOTE: token_counts MUST be [batch_size, vocab_size] layout (replicated if needed)
    // The caller is responsible for ensuring this - see Rust wrapper which replicates
    // a shared [vocab_size] array to per-sequence layout.
    int32_t count = 0;
    if (params.token_counts != nullptr && params.vocab_size > 0 && token_id < params.vocab_size) {
        // Per-sequence indexing: token_counts[batch_idx * vocab_size + token_id]
        count = __ldg(&params.token_counts[batch_idx * params.vocab_size + token_id]);
    }
    logit -= __int2float_rn(count) * params.frequency_penalty;
    
    // 3. Presence Penalty (per-sequence, based on same token_counts)
    int present = min(count, 1);
    logit -= __int2float_rn(present) * params.presence_penalty;

    // 3b. Cross-turn Penalty (tokens seen in prior turns, lighter penalty)
    if (params.cross_turn_counts != nullptr && params.cross_turn_penalty != 0.0f
        && params.vocab_size > 0 && token_id < params.vocab_size) {
        int32_t cross_count = __ldg(&params.cross_turn_counts[batch_idx * params.vocab_size + token_id]);
        int cross_present = min(cross_count, 1);
        logit -= __int2float_rn(cross_present) * params.cross_turn_penalty;
    }

    // Restore logit for protected tokens (EOS, segment-close) — undo all penalties above
    float pf = __int2float_rn(is_protected);
    logit = fmaf(pf, saved_logit - logit, logit);
    
    // 4. EOS Boost (truly branchless, with optional linear ramp)
    //    Reuses is_eos computed above for protection.
    //    Ramp: effective = eos_boost * clamp((current_len - ramp_start) / (ramp_len - ramp_start), 0, 1) * max_mult
    //    When ramp disabled (ramp_len <= 0 or max_mult <= 0): effective = eos_boost (flat)
    {
        float eos_f = __int2float_rn(is_eos);
        int ramp_span = max(params.eos_ramp_len - params.eos_ramp_start, 1);
        float t = fminf(__fdividef(float(max(params.current_len - params.eos_ramp_start, 0)),
                                   float(ramp_span)), 1.0f);
        int use_ramp = (params.eos_ramp_len > 0) & (int)(params.eos_boost_max_multiplier > 0.0f);
        float use_ramp_f = __int2float_rn(use_ramp);
        // When ramp active: boost * t * max_mult.  When inactive: boost * 1.0
        float multiplier = fmaf(use_ramp_f, t * params.eos_boost_max_multiplier - 1.0f, 1.0f);
        logit = fmaf(eos_f, params.eos_boost * multiplier, logit);
    }

    // 4b. Per-segment close boost (truly branchless, same ramp formula). Ramps on
    //     segment_len — which resets each segment — so it fires in EVERY segment,
    //     early ones included.  Targets BOTH the segment-close token AND EOS: inside
    //     a steered segment the model's EOS is intercepted into a segment close, so
    //     EOS is a second per-segment close lever and must get the same per-segment
    //     ramp, not just the total-length EOS boost above (which is dormant until
    //     late in the turn).  Outside a segment (segment_len == 0) this contributes
    //     nothing, so EOS there is governed solely by the total-length boost/failsafe.
    {
        float segment_close_f = __int2float_rn(is_segment_close | is_eos);
        int ramp_span = max(params.segment_close_ramp_len - params.segment_close_ramp_start, 1);
        float t = fminf(__fdividef(float(max(params.segment_len - params.segment_close_ramp_start, 0)),
                                   float(ramp_span)), 1.0f);
        int use_ramp = (params.segment_close_ramp_len > 0) & (int)(params.segment_close_max_multiplier > 0.0f);
        float use_ramp_f = __int2float_rn(use_ramp);
        float multiplier = fmaf(use_ramp_f, t * params.segment_close_max_multiplier - 1.0f, 1.0f);
        logit = fmaf(segment_close_f, params.segment_close_boost * multiplier, logit);
    }
    
    // 5. Banned Tokens (per-sequence or shared)
    float banned_mult = 1.0f;
    if (params.banned_tokens != nullptr) {  // Guard against null pointer
        if (params.banned_tokens_per_seq > 0) {
            // Per-sequence banned tokens
            const int32_t* my_banned = params.banned_tokens + batch_idx * params.banned_tokens_per_seq;
            for (int i = 0; i < params.banned_tokens_per_seq; i++) {
                int banned_token = my_banned[i];
                // Use -1 as sentinel for "no more banned tokens"
                int is_valid = (banned_token >= 0);
                int is_match = is_valid & (banned_token == token_id);
                banned_mult *= (1.0f - __int2float_rn(is_match));
            }
        } else if (params.num_banned_tokens > 0) {
            // Shared banned tokens (legacy behavior)
            for (int i = 0; i < params.num_banned_tokens; i++) {
                int is_match = (params.banned_tokens[i] == token_id);
                banned_mult *= (1.0f - __int2float_rn(is_match));
            }
        }
    }
    // If banned, set to -inf (branchless via selp)
    // Note: Cannot use `banned_mult * logit - (1-banned_mult) * INFINITY` because
    // when banned_mult==1.0 (no ban), 0.0f * INFINITY = NaN in IEEE 754.
    logit = (banned_mult < 1.0f) ? -INFINITY : logit;

    // 6. Token suppression (in-segment ceiling lever).
    // `suppress_penalty` is non-zero only while this sequence is inside a segment
    // (the caller zeroes it otherwise), so tokens outside the segment are never
    // touched.  Subtract it once from every token in the shared suppress list.
    if (params.suppress_penalty != 0.0f && params.suppress_tokens != nullptr) {
        float suppress_mult = 0.0f;
        for (int i = 0; i < params.suppress_count; i++) {
            int t = params.suppress_tokens[i];
            int is_match = (t >= 0) & (t == token_id);
            suppress_mult += __int2float_rn(is_match);
        }
        logit -= suppress_mult * params.suppress_penalty;
    }

    return logit;
}

// Extended penalty application with OPTIMIZED DRY using precomputed cache
// Use this version when you have precomputed the DRY penalties - O(1) lookup
__device__ __forceinline__ float apply_penalties_with_dry_cached(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache  // Precomputed DRY penalties (shared memory)
) {
    // Apply standard penalties first
    logit = apply_penalties_branchless(logit, token_id, batch_idx, params, recent_bitset);
    
    // Apply DRY penalty from precomputed cache - O(1) average lookup
    if (dry_cache != nullptr && dry_cache->num_entries > 0) {
        float dry_penalty = lookup_dry_penalty(token_id, dry_cache);
        logit -= dry_penalty;
    }
    
    return logit;
}

// Legacy: Extended penalty application including DRY penalty (SLOW - O(n²) per token)
// DEPRECATED: Use precompute_dry_penalties + apply_penalties_with_dry_cached instead
__device__ __forceinline__ float apply_penalties_with_dry(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset,
    const int32_t* __restrict__ recent_tokens,  // [batch_size * max_recent_len]
    const int32_t* __restrict__ recent_lens,    // [batch_size]
    int max_recent_len,
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range
) {
    // Apply standard penalties first
    logit = apply_penalties_branchless(logit, token_id, batch_idx, params, recent_bitset);
    
    // Apply DRY penalty if enabled and we have recent tokens
    if (dry_multiplier != 0.0f && recent_tokens != nullptr && recent_lens != nullptr) {
        const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
        int my_recent_len = recent_lens[batch_idx];
        
        float dry_penalty = compute_dry_penalty(
            token_id, my_recent, my_recent_len,
            dry_multiplier, dry_base, dry_allowed_length, dry_range
        );
        logit -= dry_penalty;
    }
    
    return logit;
}

// ============================================================================
// Vectorized Pass Helpers (FP32 with float4 loads)
// ============================================================================

// Vectorized max-finding pass with penalties (FP32 only)
// Falls back to scalar loads when pointer is not 16-byte aligned
// (e.g., when vocab_size is not a multiple of 4 and batch_idx > 0).
template <int THREADS>
__device__ void find_max_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float& local_max,
    int& local_best
) {
    const int tid = threadIdx.x;
    local_max = -INFINITY;
    local_best = 0;
    
    const bool aligned = ((uintptr_t)logits % 16) == 0;
    
    if (aligned) {
        // Number of complete float4 groups
        const int vec_size = vocab_size / 4;
        const int remainder_start = vec_size * 4;
        
        // Vectorized pass: process 4 elements at a time
        const float4* logits_vec = reinterpret_cast<const float4*>(logits);
        
        #pragma unroll 2
        for (int i = tid; i < vec_size; i += THREADS) {
            float4 v = __ldg(&logits_vec[i]);
            int base_idx = i * 4;
            
            // Unroll 4 elements (fixed count)
            float logits_arr[4] = {v.x, v.y, v.z, v.w};
            
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = logits_arr[j];
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                int is_better = (int)(logit > local_max);
                local_max = fmaxf(local_max, logit);
                local_best = is_better * token_idx + (1 - is_better) * local_best;
            }
        }
        
        // Handle remainder (typically 0-3 elements, no unroll needed)
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_best = is_better * i + (1 - is_better) * local_best;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_best = is_better * i + (1 - is_better) * local_best;
        }
    }
}

// Vectorized softmax sum pass with penalties (FP32 only)
template <int THREADS>
__device__ float compute_softmax_sum_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float global_max,
    float inv_temp
) {
    const int tid = threadIdx.x;
    float local_sum = 0.f;
    
    const bool aligned = ((uintptr_t)logits % 16) == 0;
    
    if (aligned) {
        // Number of complete float4 groups
        const int vec_size = vocab_size / 4;
        const int remainder_start = vec_size * 4;
        
        // Vectorized pass
        const float4* logits_vec = reinterpret_cast<const float4*>(logits);
        
        #pragma unroll 2
        for (int i = tid; i < vec_size; i += THREADS) {
            float4 v = __ldg(&logits_vec[i]);
            int base_idx = i * 4;
            
            float logits_arr[4] = {v.x, v.y, v.z, v.w};
            
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = logits_arr[j];
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
                local_sum += prob;
            }
        }
        
        // Handle remainder
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    }
    
    return local_sum;
}

// ============================================================================
// Block Reduction Functions
// ============================================================================

// Block-level max reduction with index tracking
template <int THREADS>
__device__ void block_reduce_max_with_idx(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    float& out_max,
    int& out_idx
) {
    const int tid = threadIdx.x;
    
    // log2(THREADS) iterations of branchless max reduction
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(vals[tid + s] > vals[tid]);
            vals[tid] = fmaxf(vals[tid], vals[tid + s]);
            idxs[tid] = other_better * idxs[tid + s] + (1 - other_better) * idxs[tid];
        }
        __syncthreads();
    }
    
    out_max = vals[0];
    out_idx = idxs[0];
}

// Block-level sum reduction
template <int THREADS>
__device__ float block_reduce_sum(float val, float* __restrict__ smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    
    // log2(THREADS) iterations of sum reduction
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    return smem[0];
}

// ============================================================================
// Tiled Vocabulary Processing
// ============================================================================
// Process vocabulary in tiles to maximize shared memory reuse.
// Each tile: load logits → apply penalties → store in smem → reuse for:
//   1. Max-finding pass
//   2. Softmax sum pass
//   3. Top-k collection pass
// This reduces global memory reads from 3x to 1x.

// Load a tile of logits, apply penalties (including DRY), store to shared memory
template <typename T, int THREADS>
__device__ void load_tile_with_penalties(
    const T* __restrict__ logits,
    int tile_start,
    int tile_size,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties (nullptr if disabled)
    float* __restrict__ tile_buffer
) {
    const int tid = threadIdx.x;
    
    // Each thread loads multiple elements
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        float logit;
        
        if (global_idx < vocab_size) {
            logit = load_as_float(logits, global_idx);
            
            if (penalties != nullptr) {
                // Apply standard penalties + DRY from precomputed cache
                logit = apply_penalties_with_dry_cached(logit, global_idx, batch_idx, *penalties, recent_bitset, dry_cache);
            }
        } else {
            logit = -INFINITY;  // Padding for out-of-bounds
        }
        
        tile_buffer[i] = logit;
    }
    __syncthreads();
}

// Process tile for max-finding (returns local max and index)
template <int THREADS>
__device__ void process_tile_for_max(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float& local_max,
    int& local_idx
) {
    const int tid = threadIdx.x;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_better * global_idx + (1 - is_better) * local_idx;
        }
    }
}

// Process tile for softmax sum
template <int THREADS>
__device__ float process_tile_for_softmax(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float global_max,
    float inv_temp
) {
    const int tid = threadIdx.x;
    float local_sum = 0.f;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    }
    
    return local_sum;
}

// Process tile for top-k collection
template <int THREADS>
__device__ void process_tile_for_topk(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float threshold,
    float global_max,
    float inv_temp,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    const int tid = threadIdx.x;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            
            if (prob >= threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_probs[idx] = prob;
                    out_idxs[idx] = global_idx;
                }
            }
        }
    }
}

// Complete tiled sampling pass: max + softmax + topk in one vocab scan
template <typename T, int THREADS, int MAX_K>
__device__ int tiled_sampling_pass(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    int top_k,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties
    float* __restrict__ tile_buffer,
    float* __restrict__ reduction_scratch,
    int* __restrict__ reduction_idx_scratch,
    float& out_global_max,
    float& out_global_sum,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_idxs,
    int* __restrict__ topk_count,
    float inv_temp
) {
    const int tid = threadIdx.x;
    const int num_tiles = (vocab_size + TILE_SIZE - 1) / TILE_SIZE;
    
    // =========================================================================
    // SINGLE-PASS with Online Max Tracking (Approximate Max Algorithm)
    // =========================================================================
    // Instead of two passes (find max, then softmax), we use an online algorithm:
    // 1. Track running max as we process tiles
    // 2. Compute softmax contributions using current max estimate
    // 3. When max increases, apply correction factor to running sum
    //
    // Key insight: exp(x - new_max) = exp(x - old_max) * exp(old_max - new_max)
    // So if max increases by delta, multiply accumulated sum by exp(-delta)
    // =========================================================================
    
    // Thread-local state for online softmax
    float local_max = -INFINITY;
    float local_sum = 0.f;
    int local_max_idx = 0;
    
    // Reset top-k counter
    if (tid == 0) {
        *topk_count = 0;
    }
    __syncthreads();
    
    // Process all tiles in a single pass
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_size = min(TILE_SIZE, vocab_size - tile_start);
        
        // Load tile with penalties applied (including DRY from cache)
        load_tile_with_penalties<T, THREADS>(
            logits, tile_start, tile_size, vocab_size,
            batch_idx, penalties, recent_bitset, dry_cache, tile_buffer
        );
        
        // Process this tile: update max, accumulate softmax with correction
        // TILE_SIZE/THREADS = 1 iteration per thread
        #pragma unroll 1
        for (int i = tid; i < tile_size; i += THREADS) {
            int global_idx = tile_start + i;
            if (global_idx < vocab_size) {
                float logit = tile_buffer[i];
                
                // Online max update with correction
                if (logit > local_max) {
                    // New max found - correct accumulated sum
                    // sum_new = sum_old * exp(old_max - new_max)
                    float correction = fast_exp::exp<float, fast_exp::Softmax>((local_max - logit) * inv_temp);
                    local_sum *= correction;
                    local_max_idx = global_idx;
                    local_max = logit;
                }
                
                // Accumulate softmax contribution using current max
                float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - local_max) * inv_temp);
                local_sum += prob;
            }
        }
        __syncthreads();
    }
    
    // =========================================================================
    // Block-wide reduction: combine local max/sum with online correction
    // =========================================================================
    reduction_scratch[tid] = local_max;
    reduction_idx_scratch[tid] = local_max_idx;
    __syncthreads();
    
    // Find block-wide max first - branchless (log2(THREADS) iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(reduction_scratch[tid + s] > reduction_scratch[tid]);
            reduction_scratch[tid] = fmaxf(reduction_scratch[tid], reduction_scratch[tid + s]);
            reduction_idx_scratch[tid] = other_better * reduction_idx_scratch[tid + s] +
                                         (1 - other_better) * reduction_idx_scratch[tid];
        }
        __syncthreads();
    }
    
    float global_max = reduction_scratch[0];
    out_global_max = global_max;
    __syncthreads();
    
    // Now correct each thread's local_sum to use the global_max
    // local_sum was computed with local_max, need to adjust
    float correction = fast_exp::exp<float, fast_exp::Softmax>((local_max - global_max) * inv_temp);
    float corrected_sum = local_sum * correction;
    
    // Reduce corrected sums
    out_global_sum = block_reduce_sum<THREADS>(corrected_sum, reduction_scratch);
    __syncthreads();
    
    // =========================================================================
    // Second mini-pass: collect top-k candidates (tiles already in L2 cache)
    // =========================================================================
    // We need the global_max for proper probability comparison
    // But tiles are now in L2 cache, so this is much faster than global memory
    //
    // NOTE: This tiled collection path is only used when top_k <= 0 (no top-k filtering).
    // When top_k > 0, we use the radix select path which provides unbiased O(n) selection.
    // The threshold-based collection here would be biased toward lower token IDs due to
    // the atomicAdd race, but that's acceptable when we're collecting all tokens anyway.
    
    float prob_threshold = 0.0f;  // Collect first MAX_K tokens (approximate when top_k=0)
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_size = min(TILE_SIZE, vocab_size - tile_start);
        
        // Reload from L2 cache (penalties recomputed but logits cached)
        load_tile_with_penalties<T, THREADS>(
            logits, tile_start, tile_size, vocab_size,
            batch_idx, penalties, recent_bitset, dry_cache, tile_buffer
        );
        
        // Collect top-k candidates
        process_tile_for_topk<THREADS>(
            tile_buffer, tile_start, tile_size, vocab_size,
            prob_threshold, global_max, inv_temp,
            topk_probs, topk_idxs, top_k, topk_count
        );
        __syncthreads();  // Ensure all atomic updates visible before next tile
    }
    __syncthreads();  // Final sync before reading topk_count
    
    return min(*topk_count, top_k);
}

// ============================================================================
// Argmax Functions
// ============================================================================

// Vectorized argmax for greedy decoding (templated for input type)
template <typename T, int THREADS>
__device__ int branchless_argmax_typed(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs
) {
    const int tid = threadIdx.x;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
    // Process logits with penalties (including DRY from cache)
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        
        if (penalties != nullptr) {
            logit = apply_penalties_with_dry_cached(logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
        }
        
        int is_greater = (int)(logit > local_max);
        local_max = fmaxf(local_max, logit);
        local_idx = is_greater * i + (1 - is_greater) * local_idx;
    }
    
    // Store for reduction
    smem_vals[tid] = local_max;
    smem_idxs[tid] = local_idx;
    __syncthreads();
    
    // Tree reduction (log2(THREADS) = 8 iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(smem_vals[tid + s] > smem_vals[tid]);
            smem_vals[tid] = fmaxf(smem_vals[tid], smem_vals[tid + s]);
            smem_idxs[tid] = other_better * smem_idxs[tid + s] + 
                             (1 - other_better) * smem_idxs[tid];
        }
        __syncthreads();
    }
    
    return smem_idxs[0];
}

// Vectorized FP32 argmax with float4 loads (4x memory bandwidth)
// Falls back to scalar loads when pointer is not 16-byte aligned.
template <int THREADS>
__device__ int branchless_argmax_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs
) {
    const int tid = threadIdx.x;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
    const bool aligned = ((uintptr_t)logits % 16) == 0;
    
    if (aligned) {
        // Number of complete float4 groups
        const int vec_size = vocab_size / 4;
        const int remainder_start = vec_size * 4;
        
        // Vectorized pass: process 4 elements at a time
        const float4* logits_vec = reinterpret_cast<const float4*>(logits);
        
        #pragma unroll 2
        for (int i = tid; i < vec_size; i += THREADS) {
            float4 v = __ldg(&logits_vec[i]);
            int base_idx = i * 4;
            
            // Process each element of float4
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                int is_greater = (int)(logit > local_max);
                local_max = fmaxf(local_max, logit);
                local_idx = is_greater * token_idx + (1 - is_greater) * local_idx;
            }
        }
        
        // Handle remainder (vocab_size not divisible by 4)
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_greater = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_greater * i + (1 - is_greater) * local_idx;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_greater = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_greater * i + (1 - is_greater) * local_idx;
        }
    }
    
    // Store for reduction
    smem_vals[tid] = local_max;
    smem_idxs[tid] = local_idx;
    __syncthreads();
    
    // Tree reduction (log2(THREADS) = 8 iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(smem_vals[tid + s] > smem_vals[tid]);
            smem_vals[tid] = fmaxf(smem_vals[tid], smem_vals[tid + s]);
            smem_idxs[tid] = other_better * smem_idxs[tid + s] + 
                             (1 - other_better) * smem_idxs[tid];
        }
        __syncthreads();
    }
    
    return smem_idxs[0];
}

// FP32 specialization - uses vectorized version
template <int THREADS>
__device__ int branchless_argmax(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs
) {
    return branchless_argmax_vectorized<THREADS>(
        logits, vocab_size, batch_idx, penalties, recent_bitset, smem_vals, smem_idxs
    );
}

// ============================================================================
// Warp Shuffle-Based Sorting for Small K (k ≤ 32)
// ============================================================================
// Uses warp shuffles instead of shared memory for much faster sorting.
// Each thread in a warp holds one (val, idx) pair and exchanges via shuffles.
// No __syncthreads() needed - warp-synchronous execution.

constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFF;

// Warp-level compare-and-swap using shuffles
// For descending sort: keep larger value in lower lane when dir=true
__device__ __forceinline__ void warp_compare_swap(
    float& val, int& idx, int lane_mask, bool dir
) {
    float other_val = __shfl_xor_sync(FULL_WARP_MASK, val, lane_mask);
    int other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, lane_mask);
    
    int lane_id = threadIdx.x & 31;
    bool is_lower_lane = (lane_id & lane_mask) == 0;
    
    // For descending: lower lane keeps max, higher lane keeps min
    bool swap = dir ? (is_lower_lane ? (val < other_val) : (val > other_val))
                    : (is_lower_lane ? (val > other_val) : (val < other_val));
    
    if (swap) {
        val = other_val;
        idx = other_idx;
    }
}

// Warp-level bitonic sort for exactly 32 elements (one per lane)
// Sorts in descending order (largest first)
__device__ __forceinline__ void warp_bitonic_sort_32(float& val, int& idx) {
    // Bitonic sort network for 32 elements
    // 5 stages: k = 2, 4, 8, 16, 32
    
    // Stage 1: k=2 (pairs)
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 2: k=4
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 3: k=8
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 4: k=16
    warp_compare_swap(val, idx, 8, true);
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 5: k=32
    warp_compare_swap(val, idx, 16, true);
    warp_compare_swap(val, idx, 8, true);
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
}

// Warp-level bitonic sort for n ≤ 32 elements
// Elements beyond n are set to -INFINITY to sort to the end
__device__ __forceinline__ void warp_bitonic_sort_n(
    float& val, int& idx, int n
) {
    int lane_id = threadIdx.x & 31;
    
    // Invalid lanes get -INFINITY (will sort to end in descending order)
    if (lane_id >= n) {
        val = -INFINITY;
        idx = -1;
    }
    
    // Full 32-element sort (invalid elements naturally go to end)
    warp_bitonic_sort_32(val, idx);
}

// Sort top-k candidates using warp shuffles (for k ≤ 32)
// Loads from shared memory, sorts in registers, writes back
__device__ __forceinline__ void warp_sort_topk(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    int n  // Must be ≤ 32
) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    
    // Only first warp does the sort
    if (warp_id == 0) {
        // Load into registers
        float my_val = (lane_id < n) ? vals[lane_id] : -INFINITY;
        int my_idx = (lane_id < n) ? idxs[lane_id] : -1;
        
        // Sort using warp shuffles
        warp_bitonic_sort_32(my_val, my_idx);
        
        // Write back sorted results
        if (lane_id < n) {
            vals[lane_id] = my_val;
            idxs[lane_id] = my_idx;
        }
    }
}

// Merge two sorted sequences of length k using warp shuffles
// Both sequences must be in registers across lanes 0..k-1 and k..2k-1
__device__ __forceinline__ void warp_bitonic_merge(
    float& val, int& idx, int k
) {
    // Bitonic merge: compare lanes i and i+k, then sort each half
    int lane_id = threadIdx.x & 31;
    
    if (k <= 16) {
        // Exchange between halves
        warp_compare_swap(val, idx, k, true);
        
        // Sort within each half
        for (int j = k / 2; j >= 1; j /= 2) {
            warp_compare_swap(val, idx, j, true);
        }
    }
}

// Parallel top-k selection using warp-level operations
// Each warp maintains its own sorted top-k candidates, then merges
template <int THREADS>
__device__ int warp_parallel_topk(
    float* __restrict__ smem_vals,      // [MAX_K] shared memory for values
    int* __restrict__ smem_idxs,        // [MAX_K] shared memory for indices
    int current_count,                   // Current number of candidates
    int target_k                         // Target k (must be ≤ 32)
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / 32;
    const int num_warps = THREADS / 32;
    
    int actual_k = min(current_count, target_k);
    
    if (actual_k <= 32 && warp_id == 0) {
        // Simple case: just sort with first warp
        float my_val = (lane_id < actual_k) ? smem_vals[lane_id] : -INFINITY;
        int my_idx = (lane_id < actual_k) ? smem_idxs[lane_id] : -1;
        
        warp_bitonic_sort_32(my_val, my_idx);
        
        if (lane_id < actual_k) {
            smem_vals[lane_id] = my_val;
            smem_idxs[lane_id] = my_idx;
        }
    }
    else if (actual_k > 32) {
        // Multi-warp case: each warp sorts a chunk, then merge
        // For now, fall through to regular bitonic sort
        return -1;  // Signal to use regular bitonic sort
    }
    
    return actual_k;
}

// ============================================================================
// Hybrid Sort: Warp shuffle for small k, shared memory for large k
// ============================================================================

// Helper: round up to next power of 2
__device__ __forceinline__ int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Bitonic sort for small arrays (used after radix select narrows candidates)
// Handles non-power-of-2 sizes by padding with -INFINITY sentinels
__device__ __forceinline__ void bitonic_sort_descending(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    int n
) {
    // Use warp shuffle sort for n ≤ 32 (much faster)
    if (n <= 32) {
        warp_sort_topk(vals, idxs, n);
        __syncthreads();
        return;
    }
    
    // For bitonic sort, we need to work with power-of-2 sizes
    // Pad with -INFINITY sentinels that will sort to the end (descending)
    int padded_n = next_power_of_2(n);
    
    // Pad with sentinels (only threads with indices in [n, padded_n) do this)
    for (int i = n + threadIdx.x; i < padded_n; i += blockDim.x) {
        vals[i] = -INFINITY;
        idxs[i] = -1;  // Invalid index
    }
    __syncthreads();
    
    // Standard bitonic sort on padded array
    for (int k = 2; k <= padded_n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = threadIdx.x; i < padded_n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // Descending order: swap if vals[i] < vals[ixj] in first half of bitonic sequence
                    bool ascending = ((i & k) == 0);
                    bool swap = ascending ? (vals[i] < vals[ixj]) : (vals[i] > vals[ixj]);
                    if (swap) {
                        float tmp_v = vals[i]; vals[i] = vals[ixj]; vals[ixj] = tmp_v;
                        int tmp_i = idxs[i]; idxs[i] = idxs[ixj]; idxs[ixj] = tmp_i;
                    }
                }
            }
            __syncthreads();
        }
    }
    // After sorting, vals[0..n-1] contains the sorted data, vals[n..padded_n-1] = -INFINITY
}

// ============================================================================
// Radix Select for O(n) Top-K Selection
// ============================================================================

// Convert float to sortable uint32 (handles negative numbers correctly)
__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t bits = __float_as_uint(f);
    // Flip sign bit for positive, flip all bits for negative
    uint32_t mask = -int32_t(bits >> 31) | 0x80000000;
    return bits ^ mask;
}

// Convert sortable uint32 back to float
__device__ __forceinline__ float sortable_to_float(uint32_t u) {
    uint32_t mask = ((u >> 31) - 1) | 0x80000000;
    return __uint_as_float(u ^ mask);
}

// ============================================================================
// Optimized Radix Select on RAW LOGITS (no exp, no global_max needed)
// ============================================================================
// Since softmax is monotonic, the top-k logits ARE the top-k probabilities.
// This eliminates exp() from all 4 radix passes AND removes the need for
// global max/sum passes entirely, reducing the top-k path from 8 to 5 passes.

// 8-bit radix select operating directly on (penalized) logit values.
// exp() is completely eliminated from the hot loop.
template <typename T, int THREADS>
__device__ float radix_select_logit_threshold(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    int k,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,
    int* __restrict__ histogram,
    float* __restrict__ smem_scratch
) {
    const int tid = threadIdx.x;
    
    uint32_t prefix = 0;
    int remaining_k = k;
    __shared__ int radix_broadcast[2];
    // Track the maximum sortable value seen across all tokens (computed during
    // pass 0 where matches=true for all tokens).  Used to clamp the threshold
    // so we never return a value below (max_logit - DEAD_ZONE), which would
    // cause collect_and_locally_normalize to include garbage (-inf / -100)
    // tokens when top_k exceeds the number of "live" tokens in the vocab.
    __shared__ uint32_t smem_max_sortable;
    if (tid == 0) smem_max_sortable = 0u;
    __syncthreads();

    for (int pass = 0; pass < 4; pass++) {
        int shift = 24 - pass * RADIX_BITS;
        int prefix_shift = shift + RADIX_BITS;
        
        for (int i = tid; i < RADIX_SIZE; i += THREADS) {
            histogram[i] = 0;
        }
        __syncthreads();
        
        // Key optimization: operate on raw logits, no exp()!
        // float_to_sortable handles negatives correctly.
        #pragma unroll 2
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            
            if (penalties != nullptr) {
                logit = apply_penalties_with_dry_cached(
                    logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
            }
            
            uint32_t sortable = float_to_sortable(logit);

            // On pass 0 every token matches; track the global maximum.
            if (pass == 0) {
                atomicMax(&smem_max_sortable, sortable);
            }
            
            bool matches = (prefix_shift >= 32) ||
                           ((sortable >> prefix_shift) == (prefix >> prefix_shift));
            
            if (matches) {
                int digit = (sortable >> shift) & RADIX_MASK;
                atomicAdd(&histogram[digit], 1);
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            int cumsum = 0;
            int selected_bucket = 0;
            
            for (int b = RADIX_SIZE - 1; b >= 0; b--) {
                int count = histogram[b];
                if (cumsum + count >= remaining_k) {
                    selected_bucket = b;
                    remaining_k -= cumsum;
                    break;
                }
                cumsum += count;
            }
            
            prefix |= (uint32_t(selected_bucket) << shift);
            radix_broadcast[0] = remaining_k;
            radix_broadcast[1] = selected_bucket;
        }
        __syncthreads();
        
        remaining_k = radix_broadcast[0];
        int selected = radix_broadcast[1];
        prefix = (prefix & ~(uint32_t(RADIX_MASK) << shift)) |
                 (uint32_t(selected) << shift);
        __syncthreads();
    }
    
    float threshold = sortable_to_float(prefix);

    // SAFETY: If top_k exceeds the number of "live" tokens (e.g. most tokens
    // are masked to -100 or -inf), the radix select will push the threshold
    // into garbage territory, causing collect_and_locally_normalize to admit
    // those masked tokens.  Clamp the threshold so it is never more than
    // DEAD_ZONE logit units below the maximum penalised logit.  The value
    // 50.0 is generous; real model logit gaps are rarely > 30.
    constexpr float DEAD_ZONE = 50.0f;
    float max_logit = sortable_to_float(smem_max_sortable);
    threshold = fmaxf(threshold, max_logit - DEAD_ZONE);

    return threshold;  // Returns a logit threshold, not a probability
}

// ============================================================================
// Fused Collect + Local Softmax (single pass, probabilities computed in smem)
// ============================================================================
// Collects top-k candidates by raw logit threshold, then computes softmax
// ONLY over the collected candidates. No global softmax sum needed.
// This is the standard approach used by vLLM and HuggingFace:
// sampling only needs relative probabilities among the top-k set.

template <typename T, int THREADS>
__device__ int collect_and_locally_normalize(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float logit_threshold,     // Raw logit threshold from radix select
    float inv_temp,            // 1.0 / temperature
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,
    float* __restrict__ out_vals,     // Output: probabilities (after local softmax)
    int* __restrict__ out_idxs,       // Output: token indices
    int max_k,
    int* __restrict__ count_ptr,
    float* __restrict__ smem_scratch  // [THREADS] for reduction
) {
    const int tid = threadIdx.x;

    // ---- Two-phase collection ----
    //
    // Phase 1: Collect tokens STRICTLY ABOVE the threshold.
    // These are guaranteed top-(k-1) candidates regardless of ties.
    //
    // Phase 2: Fill remaining slots with tokens EQUAL TO the threshold
    // (the boundary value). Ties at the boundary are resolved by
    // lowest token-index first (deterministic across runs).
    //
    // Why two phases instead of single >= pass:
    //   A single >= pass uses atomicAdd to race-fill max_k slots. When
    //   many tokens tie at the boundary (e.g., all vocab at same logit),
    //   the top token may lose the race and be excluded.  Two phases
    //   guarantee the strictly-better tokens fill first.

    if (tid == 0) *count_ptr = 0;
    __syncthreads();

    // Phase 1: strictly above threshold
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        if (penalties != nullptr) {
            logit = apply_penalties_with_dry_cached(
                logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
        }

        if (logit > logit_threshold) {
            int idx = atomicAdd(count_ptr, 1);
            if (idx < max_k) {
                out_vals[idx] = logit;
                out_idxs[idx] = i;
            }
        }
    }
    __syncthreads();

    // Phase 2: exactly at threshold — fill remaining slots (if any)
    int phase1_count = min(*count_ptr, max_k);
    if (phase1_count < max_k) {
        if (tid == 0) *count_ptr = phase1_count; // reset to phase-1 fill level
        __syncthreads();

        #pragma unroll 2
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            if (penalties != nullptr) {
                logit = apply_penalties_with_dry_cached(
                    logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
            }

            if (logit == logit_threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_vals[idx] = logit;
                    out_idxs[idx] = i;
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    int num_collected = min(*count_ptr, max_k);

    // ---- Find max among collected candidates ----
    float collected_max_local = -INFINITY;
    for (int i = tid; i < num_collected; i += THREADS) {
        collected_max_local = fmaxf(collected_max_local, out_vals[i]);
    }
    smem_scratch[tid] = collected_max_local;
    __syncthreads();

    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem_scratch[tid] = fmaxf(smem_scratch[tid], smem_scratch[tid + s]);
        __syncthreads();
    }
    float collected_max = smem_scratch[0];
    __syncthreads();

    // ---- Convert collected logits -> normalised probabilities in-place ----
    float local_sum = 0.f;
    for (int i = tid; i < num_collected; i += THREADS) {
        float prob = fast_exp::exp<float, fast_exp::Softmax>(
            (out_vals[i] - collected_max) * inv_temp);
        out_vals[i] = prob;
        local_sum += prob;
    }
    __syncthreads();

    float total_sum = block_reduce_sum<THREADS>(local_sum, smem_scratch);
    __syncthreads();

    if (total_sum > 0.f) {
        float inv_total = 1.f / total_sum;
        for (int i = tid; i < num_collected; i += THREADS) {
            out_vals[i] *= inv_total;
        }
        __syncthreads();
    }

    return num_collected;
}

// ============================================================================
// Legacy Top-K Collection (threshold-based)
// ============================================================================

// Collect top-k tokens above threshold (templated for input type)
template <typename T, int THREADS>
__device__ int collect_topk_tokens_typed(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float prob_threshold,
    float global_max,
    float inv_temp,
    float inv_sum,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    const int tid = threadIdx.x;
    
    // Initialize count
    if (tid == 0) {
        *count_ptr = 0;
    }
    __syncthreads();
    
    // Phase 1: Collect elements STRICTLY above threshold.
    // These are guaranteed top-k members and must not be displaced by
    // threshold-equal tokens racing via atomicAdd.
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        
        if (penalties != nullptr) {
            logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
        }
        
        float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp) * inv_sum;
        
        if (prob > prob_threshold) {
            int idx = atomicAdd(count_ptr, 1);
            if (idx < max_k) {
                out_probs[idx] = prob;
                out_idxs[idx] = i;
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Fill remaining slots from elements at the threshold boundary.
    // These are interchangeable (same probability), so order doesn't matter.
    if (*count_ptr < max_k) {
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp) * inv_sum;
            
            if (prob == prob_threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_probs[idx] = prob;
                    out_idxs[idx] = i;
                }
            }
        }
        __syncthreads();
    }
    
    return min(*count_ptr, max_k);
}

// FP32 specialization for backward compatibility
template <int THREADS>
__device__ int collect_topk_tokens(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float prob_threshold,
    float global_max,
    float inv_temp,
    float inv_sum,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    return collect_topk_tokens_typed<float, THREADS>(
        logits, vocab_size, batch_idx, prob_threshold, global_max, inv_temp, inv_sum,
        penalties, recent_bitset, out_probs, out_idxs, max_k, count_ptr
    );
}

// Sample from top-k candidates using Philox RNG
__device__ __forceinline__ int sample_from_topk(
    const float* __restrict__ probs,
    const int* __restrict__ indices,
    int num_candidates,
    uint64_t seed,
    uint64_t offset
) {
    // Safety: must have at least 1 candidate
    if (num_candidates <= 0) {
        return 0;  // Return token 0 as fallback (should never happen)
    }
    
    // Fast path: single candidate
    if (num_candidates == 1) {
        return indices[0];
    }
    
    // Generate uniform random in [0, 1)
    uint4 rand = flash::philox(seed, 0, offset);
    float u = (rand.x & 0x00FFFFFF) / 16777216.f;
    
    // Renormalize probabilities
    float total = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        total += probs[i];
    }
    
    // Guard against zero total (all -inf logits)
    if (total <= 0.f) {
        return indices[0];  // Return first candidate
    }
    
    float inv_total = 1.f / total;
    
    // Sample via CDF
    float cumsum = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        cumsum += probs[i] * inv_total;
        if (u < cumsum) return indices[i];
    }
    return indices[num_candidates - 1];
}

// ============================================================================
// Stencil Sampling Kernel (Optimized for small constrained vocabularies)
// ============================================================================

// When a stencil (list of allowed token IDs) is provided and small (~100 tokens),
// this kernel is MUCH faster than iterating over the full vocabulary.
// Iterates directly over stencil indices: O(stencil_size) vs O(vocab_size).
// For stencil_size=100 vs vocab_size=128K, this is ~1000x less work!

// Shared memory for stencil kernel
template <int MAX_STENCIL, int THREADS>
struct StencilSharedMem {
    // Stencil token IDs loaded into shared memory for fast access
    int32_t stencil_tokens[MAX_STENCIL];
    // Corresponding logit values (penalized)
    float stencil_logits[MAX_STENCIL];
    // Reduction scratch
    float reduction_max[THREADS];
    int reduction_idx[THREADS];
    // Global scalars
    float global_max;
    float global_sum;
};

// Maximum stencil size supported (fits comfortably in shared memory)
constexpr int MAX_STENCIL_SIZE = 1024;

// Stencil sampling kernel - optimized for small constrained vocabularies
// Iterates ONLY over allowed tokens, not full vocab
template <typename T, int MAX_STENCIL, int THREADS, bool USE_PENALTIES>
__global__ void stencil_sampling_kernel(
    // Input logits
    const T* __restrict__ logits,           // [batch_size, vocab_size]
    int vocab_size,
    
    // Stencil (allowed tokens) - shared across batch for now
    const int32_t* __restrict__ stencil,    // [stencil_size]
    int stencil_size,
    
    // Penalty scalars (applied to stencil tokens only)
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float eos_boost,
    int32_t eos_token_id,
    int32_t segment_close_token_id,
    
    // Penalty GPU pointers
    const int32_t* __restrict__ token_counts,     // [batch_size, vocab_size] or null
    const int32_t* __restrict__ banned_tokens,    // [num_banned] or null (rarely used with stencil)
    int32_t num_banned_tokens,
    
    // Recent tokens for repeat penalty
    const int32_t* __restrict__ recent_tokens,    // [batch_size * max_recent_len] or null
    const int32_t* __restrict__ recent_lens,      // [batch_size] or null
    int32_t max_recent_len,
    
    // Sampling configuration
    float temperature,
    int top_k,
    float top_p,
    
    // Output
    uint32_t* __restrict__ output_tokens,    // [batch_size]
    
    // RNG
    uint64_t seed,
    uint64_t* __restrict__ rng_offsets       // [batch_size]
) {
    __shared__ StencilSharedMem<MAX_STENCIL, THREADS> smem;
    
    // Dynamic shared memory for recent token bitset (if penalties used)
    extern __shared__ uint32_t recent_bitset[];
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Safety checks
    if (output_tokens == nullptr || logits == nullptr || stencil == nullptr || stencil_size <= 0) {
        if (tid == 0 && output_tokens != nullptr) {
            output_tokens[batch_idx] = 0;
        }
        return;
    }
    
    // Clamp stencil size to supported maximum
    const int actual_stencil_size = min(stencil_size, MAX_STENCIL);
    
    const T* my_logits = logits + batch_idx * vocab_size;
    const int bitset_words = (vocab_size + 31) / 32;
    
    // =========================================================
    // STEP 1: Load stencil into shared memory (collaborative)
    // =========================================================
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        smem.stencil_tokens[i] = stencil[i];
    }
    __syncthreads();
    
    // =========================================================
    // STEP 2: Build recent token bitset (if penalties enabled)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (recent_tokens != nullptr && recent_lens != nullptr) {
            // Clear bitset
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            __syncthreads();
            
            // Get this sequence's recent tokens
            const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
            int my_recent_len = recent_lens[batch_idx];
            
            // Set bits for recent tokens
            for (int i = tid; i < my_recent_len; i += THREADS) {
                int32_t tok = my_recent[i];
                if (tok >= 0 && tok < vocab_size) {
                    atomicOr(&recent_bitset[tok >> 5], 1u << (tok & 31));
                }
            }
            __syncthreads();
        } else {
            // Clear bitset if no recent tokens
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            __syncthreads();
        }
    }
    
    // =========================================================
    // STEP 3: Load logits for stencil tokens, apply penalties
    // =========================================================
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        int32_t token_id = smem.stencil_tokens[i];
        float logit = load_as_float(my_logits, token_id);
        
        if constexpr (USE_PENALTIES) {
            // Protected tokens: EOS and the segment-close token must never be penalised
            float saved_logit = logit;
            int eos_d = token_id - eos_token_id;
            int eos_a = (eos_d ^ (eos_d >> 31)) - (eos_d >> 31);
            int is_eos = 1 - min(eos_a, 1);
            int segment_close_d = token_id - segment_close_token_id;
            int segment_close_a = (segment_close_d ^ (segment_close_d >> 31)) - (segment_close_d >> 31);
            int is_segment_close = 1 - min(segment_close_a, 1);
            int is_protected = is_eos | is_segment_close;

            // Apply repeat penalty (branchless - same pattern as apply_penalties_branchless)
            if (recent_tokens != nullptr) {
                uint32_t word = recent_bitset[token_id >> 5];
                uint32_t in_recent = (word >> (token_id & 31)) & 1u;
                float in_recent_f = __int2float_rn(in_recent);
                
                // Sign-based penalty: positive logits divided, negative multiplied
                uint32_t logit_bits = __float_as_uint(logit);
                uint32_t sign_bit = logit_bits >> 31;
                float sign_f = __int2float_rn(sign_bit);
                float inv_penalty = __fdividef(1.0f, repeat_penalty);
                float repeat_factor = sign_f * repeat_penalty + (1.0f - sign_f) * inv_penalty;
                float final_factor = in_recent_f * repeat_factor + (1.0f - in_recent_f) * 1.0f;
                logit *= final_factor;
            }
            
            // Apply frequency/presence penalties (branchless)
            if (token_counts != nullptr) {
                int32_t count = __ldg(&token_counts[batch_idx * vocab_size + token_id]);
                logit -= frequency_penalty * __int2float_rn(count);
                int present = min(count, 1);
                logit -= __int2float_rn(present) * presence_penalty;
            }

            // Restore logit for protected tokens (EOS, segment-close) — undo all penalties
            float pf = __int2float_rn(is_protected);
            logit = fmaf(pf, saved_logit - logit, logit);
            
            // Apply EOS boost (branchless)
            {
                logit += float(is_eos) * eos_boost;
            }
        }
        
        smem.stencil_logits[i] = logit;
    }
    __syncthreads();
    
    // =========================================================
    // STEP 4: Check for banned tokens (set to -inf)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (banned_tokens != nullptr && num_banned_tokens > 0) {
            // For each banned token, check if it's in stencil and mark as -inf
            for (int b = tid; b < num_banned_tokens; b += THREADS) {
                int32_t banned_id = banned_tokens[b];
                // Linear search in stencil (fast for small stencils)
                for (int i = 0; i < actual_stencil_size; i++) {
                    if (smem.stencil_tokens[i] == banned_id) {
                        smem.stencil_logits[i] = -INFINITY;
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // =========================================================
    // STEP 5: Find max logit (for argmax or softmax normalization)
    // =========================================================
    float local_max = -INFINITY;
    int local_best_idx = 0;
    
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        float logit = smem.stencil_logits[i];
        int is_better = (int)(logit > local_max);
        local_max = fmaxf(local_max, logit);
        local_best_idx = is_better * i + (1 - is_better) * local_best_idx;
    }
    
    smem.reduction_max[tid] = local_max;
    smem.reduction_idx[tid] = local_best_idx;
    __syncthreads();
    
    // Block reduction for max (branchless)
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other_better = (int)(smem.reduction_max[tid + stride] > smem.reduction_max[tid]);
            smem.reduction_max[tid] = fmaxf(smem.reduction_max[tid], smem.reduction_max[tid + stride]);
            smem.reduction_idx[tid] = other_better * smem.reduction_idx[tid + stride] +
                                      (1 - other_better) * smem.reduction_idx[tid];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        smem.global_max = smem.reduction_max[0];
    }
    __syncthreads();
    
    float global_max = smem.global_max;
    int best_stencil_idx = smem.reduction_idx[0];
    
    // =========================================================
    // STEP 6: Argmax path (temperature <= 0)
    // =========================================================
    if (!(temperature > 0.f)) {
        if (tid == 0) {
            output_tokens[batch_idx] = smem.stencil_tokens[best_stencil_idx];
        }
        return;
    }
    
    // =========================================================
    // STEP 7: Compute softmax probabilities
    // =========================================================
    const float inv_temp = 1.0f / temperature;
    float local_sum = 0.f;
    
    // Compute exp((logit - max) / temp) for each stencil token
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        float prob = fast_exp::exp<float, fast_exp::Softmax>((smem.stencil_logits[i] - global_max) * inv_temp);
        smem.stencil_logits[i] = prob;  // Reuse array for probabilities
        local_sum += prob;
    }
    __syncthreads();
    
    // Block reduction for sum
    smem.reduction_max[tid] = local_sum;  // Reuse for sum reduction
    __syncthreads();
    
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem.reduction_max[tid] += smem.reduction_max[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        smem.global_sum = smem.reduction_max[0];
    }
    __syncthreads();
    
    float global_sum = smem.global_sum;
    
    // Guard against zero sum
    if (global_sum <= 0.f) {
        if (tid == 0) {
            output_tokens[batch_idx] = smem.stencil_tokens[best_stencil_idx];
        }
        return;
    }
    
    // =========================================================
    // STEP 8: Apply top-k and top-p filtering (optional)
    // =========================================================
    int sample_size = actual_stencil_size;
    const float inv_sum = 1.0f / global_sum;
    
    // For small stencils, top-k/top-p typically don't apply much
    // but we support them for correctness
    if (top_k > 0 && top_k < actual_stencil_size) {
        // Use insertion sort for better cache behavior than bubble sort
        // O(n²) worst case but O(n) for nearly-sorted data, and better constants
        if (tid == 0) {
            // Insertion sort descending by probability
            for (int i = 1; i < actual_stencil_size; i++) {
                float key_prob = smem.stencil_logits[i];
                int32_t key_tok = smem.stencil_tokens[i];
                int j = i - 1;
                
                // Move elements smaller than key_prob to the right
                while (j >= 0 && smem.stencil_logits[j] < key_prob) {
                    smem.stencil_logits[j + 1] = smem.stencil_logits[j];
                    smem.stencil_tokens[j + 1] = smem.stencil_tokens[j];
                    j--;
                }
                smem.stencil_logits[j + 1] = key_prob;
                smem.stencil_tokens[j + 1] = key_tok;
            }
            sample_size = min(top_k, actual_stencil_size);
        }
        __syncthreads();
    }
    
    // Apply top-p (nucleus) filtering
    if (top_p < 1.0f && top_p > 0.f) {
        if (tid == 0) {
            float cumsum = 0.f;
            for (int i = 0; i < sample_size; i++) {
                cumsum += smem.stencil_logits[i] * inv_sum;
                if (cumsum >= top_p) {
                    sample_size = i + 1;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    // =========================================================
    // STEP 9: Sample from filtered distribution
    // =========================================================
    if (tid == 0) {
        uint64_t offset = rng_offsets ? rng_offsets[batch_idx] : 0;
        if (rng_offsets) {
            rng_offsets[batch_idx] = offset + 1;
        }
        
        // Generate uniform random in [0, 1)
        uint4 rand = flash::philox(seed, 0, offset);
        float u = (rand.x & 0x00FFFFFF) / 16777216.f;
        
        // Renormalize over sample_size tokens
        float total = 0.f;
        for (int i = 0; i < sample_size; i++) {
            total += smem.stencil_logits[i];
        }
        
        if (total <= 0.f) {
            output_tokens[batch_idx] = smem.stencil_tokens[0];
            return;
        }
        
        float cumsum = 0.f;
        for (int i = 0; i < sample_size; i++) {
            cumsum += smem.stencil_logits[i] / total;
            if (u < cumsum) {
                output_tokens[batch_idx] = smem.stencil_tokens[i];
                return;
            }
        }
        output_tokens[batch_idx] = smem.stencil_tokens[sample_size - 1];
    }
}

// Stencil dispatch helper
template <typename T>
inline void dispatch_stencil_sampling(
    const T* logits,
    int batch_size,
    int vocab_size,
    // Stencil
    const int32_t* stencil,
    int stencil_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float eos_boost,
    int32_t eos_token_id,
    int32_t segment_close_token_id,
    // Penalty GPU pointers
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    // Recent tokens
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Sampling params
    float temperature,
    int top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets,
    // Dispatch flag
    bool use_penalties,
    // CUDA stream (0 = default stream)
    cudaStream_t stream = 0
) {
    
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    // Dynamic shared memory for bitset (only if penalties used)
    const size_t bitset_bytes = use_penalties ? ((vocab_size + 31) / 32) * sizeof(uint32_t) : 0;
    
    if (use_penalties) {
        stencil_sampling_kernel<T, MAX_STENCIL_SIZE, THREADS_PER_BLOCK, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                stencil, stencil_size,
                repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
                token_counts, banned_tokens, num_banned_tokens,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else {
        stencil_sampling_kernel<T, MAX_STENCIL_SIZE, THREADS_PER_BLOCK, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                stencil, stencil_size,
                repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
                token_counts, banned_tokens, num_banned_tokens,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    }
}

// ============================================================================
// Main Batched Sampling Kernel (Templated for Input Type and Mode)
// ============================================================================

// Template parameters for compile-time optimization:
// - T: Input logit type (float, half, __nv_bfloat16, __nv_fp8_*)
// - MAX_K: Maximum top-k value supported
// - THREADS: Threads per block
// - USE_PENALTIES: When false, all standard penalty code is eliminated (saves tile_buffer 4KB)
// - USE_DRY: When false, DRY penalty cache is eliminated (saves 4KB)
// - USE_TOP_P: When false, nucleus filtering loop is eliminated
//
// Shared memory savings:
// - USE_PENALTIES=false: eliminates tile_buffer (4KB)
// - USE_DRY=false: eliminates dry_cache (4KB)
// - Both false: save 8KB → better occupancy for no-penalty/argmax paths

// Main batched sampling kernel - takes all parameters directly (no cudaMalloc needed)
// All GPU pointers are already on device (from Tensors), scalars are passed by value
template <typename T, int MAX_K, int THREADS, bool USE_PENALTIES, bool USE_DRY, bool USE_TOP_P>
__global__ void __launch_bounds__(THREADS, USE_PENALTIES ? 1 : 2)
batched_penalty_sampling_kernel(
    // Input logits from LM head matmul
    const T* __restrict__ logits,           // [batch_size, vocab_size]
    int vocab_size,
    
    // Penalty scalars (passed by value, CUDA copies to constant memory)
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    // Cross-turn penalty
    float cross_turn_penalty,
    const int32_t* __restrict__ cross_turn_counts,  // [batch_size, vocab_size] or null
    // Per-sequence current lengths (for dynamic EOS ramp)
    const int32_t* __restrict__ current_lens,       // [batch_size] or null
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* __restrict__ segment_lens,      // [batch_size] or null
    // Added to `temperature` for sequences inside a segment (segment_lens[seq] > 0).
    float segment_temp_boost,
    // Token suppression: subtract `suppress_penalties[seq]` from each
    // `suppress_tokens` logit while sequence `seq` is inside a segment.
    const int32_t* __restrict__ suppress_tokens,    // [suppress_count] (shared) or null
    int32_t suppress_count,
    const float* __restrict__ suppress_penalties,   // [batch_size] or null

    // Penalty GPU pointers (already on device from Tensors)
    const int32_t* __restrict__ token_counts,     // [batch_size, vocab_size] or null
    const int32_t* __restrict__ banned_tokens,    // [batch_size, banned_per_seq] or [num_banned] or null
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    
    // Recent tokens for repeat penalty (per-sequence, already on device)
    const int32_t* __restrict__ recent_tokens,    // [batch_size * max_recent_len] or null
    const int32_t* __restrict__ recent_lens,      // [batch_size] or null
    int32_t max_recent_len,
    
    // Sampling configuration
    float temperature,
    int top_k,
    float top_p,
    
    // Output
    uint32_t* __restrict__ output_tokens,    // [batch_size]
    
    // RNG
    uint64_t seed,
    uint64_t* __restrict__ rng_offsets       // [batch_size], updated in-place
) {
    // Shared memory with conditional allocation based on feature flags
    // USE_PENALTIES controls tile_buffer (4KB), USE_DRY controls dry_cache (4KB)
    __shared__ SamplingSharedMem<MAX_K, THREADS, USE_PENALTIES, USE_DRY, true> smem;
    
    // Dynamic shared memory for bitset (allocated based on vocab_size at launch)
    extern __shared__ uint32_t recent_bitset[];
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Safety: validate required pointers
    // Note: rng_offsets can be null for argmax (temperature <= 0) since RNG isn't used
    if (output_tokens == nullptr || logits == nullptr) {
        if (tid == 0 && output_tokens != nullptr) {
            output_tokens[batch_idx] = 0;  // Fallback to token 0
        }
        return;
    }
    
    const T* my_logits = logits + batch_idx * vocab_size;
    
    // Calculate bitset words needed for this vocab size
    const int bitset_words = (vocab_size + 31) / 32;
    
    // Build PenaltyParams struct on registers from scalar parameters
    // This is free - compiler optimizes it to direct register usage
    PenaltyParams penalty_params_local;
    penalty_params_local.repeat_penalty = repeat_penalty;
    penalty_params_local.frequency_penalty = frequency_penalty;
    penalty_params_local.presence_penalty = presence_penalty;
    penalty_params_local.dry_multiplier = dry_multiplier;
    penalty_params_local.dry_base = dry_base;
    penalty_params_local.dry_allowed_length = dry_allowed_length;
    penalty_params_local.dry_range = dry_range;
    penalty_params_local.eos_boost = eos_boost;
    penalty_params_local.eos_token_id = eos_token_id;
    penalty_params_local.eos_ramp_start = eos_ramp_start;
    penalty_params_local.eos_ramp_len = eos_ramp_len;
    penalty_params_local.eos_boost_max_multiplier = eos_boost_max_multiplier;
    penalty_params_local.current_len = (current_lens != nullptr) ? current_lens[batch_idx] : 0;
    penalty_params_local.segment_close_boost = segment_close_boost;
    penalty_params_local.segment_close_token_id = segment_close_token_id;
    penalty_params_local.segment_close_ramp_start = segment_close_ramp_start;
    penalty_params_local.segment_close_ramp_len = segment_close_ramp_len;
    penalty_params_local.segment_close_max_multiplier = segment_close_max_multiplier;
    penalty_params_local.segment_len = (segment_lens != nullptr) ? segment_lens[batch_idx] : 0;
    penalty_params_local.cross_turn_penalty = cross_turn_penalty;
    penalty_params_local.cross_turn_counts = cross_turn_counts;
    penalty_params_local.token_counts = token_counts;
    penalty_params_local.vocab_size = vocab_size;
    penalty_params_local.banned_tokens = banned_tokens;
    penalty_params_local.num_banned_tokens = num_banned_tokens;
    penalty_params_local.banned_tokens_per_seq = banned_tokens_per_seq;
    penalty_params_local.suppress_tokens = suppress_tokens;
    penalty_params_local.suppress_count = suppress_count;
    // Per-sequence suppression strength; gated to segments just below.
    penalty_params_local.suppress_penalty = 0.0f;

    // =========================================================
    // In-segment steering (per-sequence, since each block = one sequence).
    // While this sequence is inside a segment (segment_lens[seq] > 0):
    //   - sample a touch hotter (temperature += segment_temp_boost)
    //   - enable the DRY n-gram penalty (gated off entirely outside the segment
    //     so verbatim code/number copying is never corrupted).
    // Outside a segment both revert to the batch-wide values.
    // =========================================================
    const bool in_segment = (segment_lens != nullptr) && (segment_lens[batch_idx] > 0);
    if (in_segment) {
        temperature += segment_temp_boost;
        // Activate token suppression for this sequence: subtract the per-sequence
        // penalty from every suppress-token logit (applied in
        // apply_penalties_branchless). Outside a segment this stays 0.
        if (suppress_penalties != nullptr) {
            penalty_params_local.suppress_penalty = suppress_penalties[batch_idx];
        }
    } else {
        // Disable DRY for the answer: zero the multiplier the precompute and
        // cached-lookup paths key off.  Shared memory for the cache is still
        // allocated (USE_DRY is a batch-wide template param), but no penalties
        // are produced for this sequence.
        dry_multiplier = 0.0f;
        penalty_params_local.dry_multiplier = 0.0f;
    }

    // =========================================================
    // STEP 1: Build recent token bitset (if penalties enabled)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (recent_tokens != nullptr && recent_lens != nullptr) {
            // Get this sequence's recent tokens (oldest-first; newest at the end).
            const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
            int my_recent_len = recent_lens[batch_idx];

            // In-segment scoping: while inside a segment, repeat/DRY must only
            // see the tokens generated inside the current segment — never the
            // prompt or prior turns that precede the segment-open token. The
            // segment spans the last `segment_len` tokens, so restrict the recent
            // window to that suffix. `recent_tokens` is newest-at-the-end, so the
            // last `effective_len` entries are exactly the in-segment tokens.
            // Outside a segment, keep the full window (existing behavior).
            const int32_t* eff_recent = my_recent;
            int eff_recent_len = my_recent_len;
            int eff_dry_range = dry_range;
            if (in_segment) {
                int effective_len = min(segment_lens[batch_idx], my_recent_len);
                int offset = my_recent_len - effective_len;
                eff_recent = my_recent + offset;
                eff_recent_len = effective_len;
                // Cap the DRY look-back to the in-block window.
                eff_dry_range = (dry_range > 0) ? min(dry_range, effective_len) : effective_len;
            }

            build_recent_bitset<THREADS>(
                recent_bitset,
                eff_recent,
                eff_recent_len,
                bitset_words
            );

            // Precompute DRY penalties (O(n²) once, then O(1) lookup per token)
            // Only when USE_DRY template param is enabled
            if constexpr (USE_DRY) {
                if (dry_multiplier != 0.0f && eff_recent_len >= 2) {
                    precompute_dry_penalties<THREADS>(
                        eff_recent, eff_recent_len,
                        dry_multiplier, dry_base, dry_allowed_length, eff_dry_range,
                        smem.dry_cache.as_ptr()
                    );
                } else {
                    // No DRY - clear cache
                    if (tid == 0) {
                        smem.dry_cache.num_entries = 0;
                    }
                    __syncthreads();
                }
            }
        } else {
            // Clear bitset
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            // Clear DRY cache (only if enabled)
            if constexpr (USE_DRY) {
                if (tid == 0) {
                    smem.dry_cache.num_entries = 0;
                }
            }
            __syncthreads();
        }
    } else {
        // No penalties - just clear bitset (minimal cost, ensures valid reads)
        for (int i = tid; i < bitset_words; i += THREADS) {
            recent_bitset[i] = 0;
        }
        __syncthreads();
    }
    
    // Pointer to DRY cache (use cached version for O(1) lookup)
    // When USE_DRY=false, dry_cache.as_ptr() returns nullptr
    const DryPenaltyCache* dry_cache_ptr = USE_DRY ? smem.dry_cache.as_ptr() : nullptr;
    
    // Pointer to use - null when penalties disabled for cleaner downstream code
    const PenaltyParams* effective_penalties = USE_PENALTIES ? &penalty_params_local : nullptr;

    // =========================================================
    // STEP 2: Early exit for argmax (temperature <= 0)
    // =========================================================
    if (!(temperature > 0.f)) {  // Handles NaN, <=0, -inf
        int best = branchless_argmax_typed<T, THREADS>(
            my_logits, vocab_size, batch_idx,
            effective_penalties, recent_bitset, dry_cache_ptr,
            smem.reduction_max, smem.reduction_idx
        );
        
        if (tid == 0) {
            output_tokens[batch_idx] = best;
        }
        return;
    }
    
    const float inv_temp = 1.0f / temperature;
    const int k = top_k > 0 ? min(top_k, MAX_K) : MAX_K;
    
    float global_max, global_sum;
    int num_topk;
    
    // =========================================================
    // Processing strategy based on penalties and top-k
    // =========================================================
    if constexpr (USE_PENALTIES) {
        // Always use radix select when top_k is specified to avoid bias
        // toward low token IDs that threshold-based collection causes
        bool use_radix = (top_k > 0);
        
        if (!use_radix) {
            num_topk = tiled_sampling_pass<T, THREADS, MAX_K>(
                my_logits, vocab_size, batch_idx, k,
                effective_penalties, recent_bitset, dry_cache_ptr,
                smem.tile_buffer,
                smem.reduction_max, smem.reduction_idx,
                global_max, global_sum,
                smem.topk_vals, smem.topk_idxs, &smem.topk_count,
                inv_temp
            );
            
            if (tid == 0) {
                smem.global_max = global_max;
                smem.global_sum = global_sum;
                smem.best_idx = smem.reduction_idx[0];
            }
            __syncthreads();
        } else {
            // =========================================================
            // OPTIMIZED TOP-K + PENALTY PATH: 5 passes instead of 8
            // Radix select on RAW LOGITS (no exp), then local softmax
            // over only the ~k collected candidates in shared memory.
            // =========================================================
            
            // Step 1: Radix select on raw (penalized) logits — 4 vocab passes, NO exp()
            float logit_threshold = radix_select_logit_threshold<T, THREADS>(
                my_logits, vocab_size, batch_idx, k,
                effective_penalties, recent_bitset, dry_cache_ptr,
                smem.radix_histogram, smem.reduction_max
            );
            
            // Broadcast threshold to ensure all threads agree
            if (tid == 0) smem.global_max = logit_threshold;
            __syncthreads();
            logit_threshold = smem.global_max;
            
            // Step 2: Collect + local softmax — 1 vocab pass, probs computed in smem
            num_topk = collect_and_locally_normalize<T, THREADS>(
                my_logits, vocab_size, batch_idx, logit_threshold, inv_temp,
                effective_penalties, recent_bitset, dry_cache_ptr,
                smem.topk_vals, smem.topk_idxs, k, &smem.topk_count,
                smem.reduction_max
            );
            
            // Probabilities are already locally normalized (sum to 1.0)
            if (tid == 0) {
                smem.global_sum = 1.0f;
                smem.best_idx = (num_topk > 0) ? smem.topk_idxs[0] : 0;
            }
            __syncthreads();
        }
    } else {
        // =========================================================
        // NO-PENALTY PATH
        // =========================================================
        
        if (top_k > 0) {
            // =========================================================
            // OPTIMIZED TOP-K PATH: 5 passes instead of 8, no exp()
            // Radix select on raw logits, then local softmax over ~k items
            // =========================================================
            
            // Step 1: Radix select on raw logits — 4 vocab passes, NO exp()
            float logit_threshold = radix_select_logit_threshold<T, THREADS>(
                my_logits, vocab_size, batch_idx, k,
                nullptr, recent_bitset, nullptr,  // No penalties, no DRY
                smem.radix_histogram, smem.reduction_max
            );
            
            // Broadcast threshold
            if (tid == 0) smem.global_max = logit_threshold;
            __syncthreads();
            logit_threshold = smem.global_max;
            
            // Step 2: Collect + local softmax — 1 vocab pass
            num_topk = collect_and_locally_normalize<T, THREADS>(
                my_logits, vocab_size, batch_idx, logit_threshold, inv_temp,
                nullptr, recent_bitset, nullptr,
                smem.topk_vals, smem.topk_idxs, k, &smem.topk_count,
                smem.reduction_max
            );
            
            // Probabilities are already locally normalized (sum to 1.0)
            if (tid == 0) {
                smem.global_sum = 1.0f;
                smem.best_idx = (num_topk > 0) ? smem.topk_idxs[0] : 0;
            }
            __syncthreads();
            
        } else {
            // =========================================================
            // NO TOP-K: need global max + sum for full softmax
            // (2-pass approach, then threshold collect)
            // =========================================================
            
            float local_max = -INFINITY;
            int local_best = 0;
            
            if constexpr (std::is_same_v<T, float>) {
                find_max_vectorized<THREADS>(
                    my_logits, vocab_size, batch_idx,
                    nullptr, nullptr,
                    local_max, local_best
                );
            } else {
                for (int i = tid; i < vocab_size; i += THREADS) {
                    float logit = load_as_float(my_logits, i);
                    int is_better = (int)(logit > local_max);
                    local_max = fmaxf(local_max, logit);
                    local_best = is_better * i + (1 - is_better) * local_best;
                }
            }
            
            smem.reduction_max[tid] = local_max;
            smem.reduction_idx[tid] = local_best;
            __syncthreads();
            
            block_reduce_max_with_idx<THREADS>(
                smem.reduction_max, smem.reduction_idx,
                smem.global_max, smem.best_idx
            );
            __syncthreads();
            
            global_max = smem.global_max;
            
            float local_sum = 0.f;
            if constexpr (std::is_same_v<T, float>) {
                local_sum = compute_softmax_sum_vectorized<THREADS>(
                    my_logits, vocab_size, batch_idx,
                    nullptr, nullptr,
                    global_max, inv_temp
                );
            } else {
                for (int i = tid; i < vocab_size; i += THREADS) {
                    float logit = load_as_float(my_logits, i);
                    float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
                    local_sum += prob;
                }
            }
            
            global_sum = block_reduce_sum<THREADS>(local_sum, smem.reduction_max);
            if (tid == 0) {
                smem.global_max = global_max;
                smem.global_sum = global_sum;
            }
            __syncthreads();
            
            // Collect all tokens above threshold
            float prob_threshold = 0.0f;
            float inv_sum = 1.0f / global_sum;
            
            num_topk = collect_topk_tokens_typed<T, THREADS>(
                my_logits, vocab_size, batch_idx, prob_threshold,
                global_max, inv_temp, inv_sum,
                nullptr, recent_bitset,
                smem.topk_vals, smem.topk_idxs, k,
                &smem.topk_count
            );
        }
    }
    
    const float inv_sum = 1.0f / smem.global_sum;
    
    if (num_topk <= 0) {
        if (tid == 0) {
            smem.topk_vals[0] = 1.0f;
            smem.topk_idxs[0] = smem.best_idx;
        }
        num_topk = 1;
    }
    __syncthreads();
    
    bitonic_sort_descending(smem.topk_vals, smem.topk_idxs, num_topk);
    __syncthreads();
    
    int nucleus_size = num_topk;
    
    if constexpr (USE_TOP_P) {
        // NOTE on top-p normalization semantics:
        // When top-k is active, collect_and_locally_normalize has already
        // normalized probabilities over ONLY the top-k set (local normalization).
        // inv_sum was set to 1.0f so topk_vals are already proper probabilities
        // that sum to 1.0 within the top-k set.  The cumsum here accumulates
        // those local probabilities directly — this matches vLLM / HuggingFace
        // behavior where top-p is applied AFTER top-k filtering.
        //
        // When top-k is NOT active (no-top-k tiled path), probabilities come
        // from global softmax with inv_sum = 1/global_sum, so the cumsum
        // operates over globally normalized probabilities.
        if (top_p < 1.0f && tid == 0) {
            float cumsum = 0.f;
            for (int i = 0; i < num_topk; i++) {
                cumsum += smem.topk_vals[i] * inv_sum;
                if (cumsum >= top_p) {
                    nucleus_size = i + 1;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // rng_offsets should never be null when temperature > 0, but check defensively
        uint64_t offset = rng_offsets ? rng_offsets[batch_idx] : 0;
        if (rng_offsets) {
            rng_offsets[batch_idx] = offset + 1;
        }
        
        output_tokens[batch_idx] = sample_from_topk(
            smem.topk_vals, smem.topk_idxs, 
            nucleus_size > 0 ? nucleus_size : 1,
            seed, offset
        );
    }
}

// ============================================================================
// Kernel Launch Wrappers with Compile-Time Dispatch
// ============================================================================

// Internal dispatch helper - selects kernel variant based on runtime flags
template <typename T>
inline void dispatch_batched_sampling(
    const T* logits,
    int batch_size,
    int vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Sampling params
    float temperature,
    int top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets,
    // Dispatch flags
    bool use_penalties,
    bool use_top_p,
    // CUDA stream (0 = default stream)
    cudaStream_t stream = 0
) {
    
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    // Dynamic shared memory for bitset: ceil(vocab_size / 32) words * 4 bytes
    const size_t bitset_bytes = ((vocab_size + 31) / 32) * sizeof(uint32_t);
    
    // 4-way dispatch based on penalties and top_p
    // USE_DRY is set to true when penalties are enabled AND dry_multiplier != 0
    // This allows eliminating 4KB shared memory when DRY is not used
    const bool use_dry = use_penalties && (dry_multiplier != 0.0f);
    
    if (use_penalties && use_dry && use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else if (use_penalties && use_dry && !use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else if (use_penalties && !use_dry && use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else if (use_penalties && !use_dry && !use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else if (!use_penalties && use_top_p) {
        // No penalties implies no DRY either
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    } else {
        // No penalties, no DRY, no top_p - fastest path
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
    }
}

// Templated launch wrapper with auto-detection of modes and stencil support
template <typename T>
inline void launch_batched_sampling_typed(
    const T* logits,
    int batch_size,
    int vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary) - pass nullptr or stencil_size=0 to disable
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets,
    // CUDA stream (0 = default stream)
    cudaStream_t stream = 0
) {
    // Auto-detect whether penalties are needed
    bool use_penalties = (repeat_penalty != 1.0f) ||
                         (frequency_penalty != 0.0f) ||
                         (presence_penalty != 0.0f) ||
                         (cross_turn_penalty != 0.0f) ||
                         (dry_multiplier != 0.0f) ||
                         (eos_boost != 0.0f) ||
                         (segment_close_boost != 0.0f) ||
                         (num_banned_tokens > 0) ||
                         (banned_tokens_per_seq > 0) ||
                         // Token suppression needs the penalty path.
                         (suppress_tokens != nullptr && suppress_count > 0 &&
                          suppress_penalties != nullptr);
    bool use_top_p = (top_p < 1.0f);
    
    // Use optimized stencil kernel when stencil is provided and reasonably small
    // For stencils up to MAX_STENCIL_SIZE (1024), direct iteration is faster
    // Note: Stencil kernel doesn't support DRY penalty yet - fall through if DRY enabled
    if (stencil != nullptr && stencil_size > 0 && stencil_size <= MAX_STENCIL_SIZE && dry_multiplier == 0.0f) {
        dispatch_stencil_sampling<T>(
            logits, batch_size, vocab_size,
            stencil, stencil_size,
            repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
            token_counts, banned_tokens, num_banned_tokens,
            recent_tokens, recent_lens, max_recent_len,
            temperature, top_k, top_p,
            output_tokens, seed, rng_offsets,
            use_penalties,
            stream
        );
    } else {
        // Fall back to full vocabulary kernel
        dispatch_batched_sampling<T>(
            logits, batch_size, vocab_size,
            repeat_penalty, frequency_penalty, presence_penalty,
            dry_multiplier, dry_base, dry_allowed_length, dry_range,
            eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
            cross_turn_penalty, cross_turn_counts, current_lens,
            segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
            segment_temp_boost,
            suppress_tokens, suppress_count, suppress_penalties,
            token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
            recent_tokens, recent_lens, max_recent_len,
            temperature, top_k, top_p,
            output_tokens, seed, rng_offsets,
            use_penalties, use_top_p,
            stream
        );
    }
}

} // namespace batched_sampling
