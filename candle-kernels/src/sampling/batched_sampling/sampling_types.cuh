// Sampling constants, per-sequence dials, penalty parameters, and the
// shared-memory layouts.
//
// Defines the thread and tile constants, the radix-select threshold,
// `LogitDType`, the non-finite logit test, `SeqDials` and `PenaltyParams`.
// Also defines the `Conditional*` wrappers and the `SamplingSharedMem`
// variants, which size shared memory to the features a launch enables.

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
// Non-finite logits
// ============================================================================
//
// A row whose logits hold a NaN or a +inf came from a broken forward: no sample
// drawn from it means anything, and an argmax over it quietly returns some
// token, because NaN never wins a `>`. Each kernel tests every raw logit it
// already loads — before any penalty, since a ban is −inf on purpose — and
// writes the row's verdict to `nonfinite_rows[row]` (1 or 0) for the host to
// fail the row on. A bit test, not a float compare, so no fast-math setting can
// fold it away; −inf passes, since a masked logit is one. The verdict joins the
// block on a barrier the kernel already takes (`__syncthreads_or`), so the test
// costs a few integer ops per logit and no extra pass or barrier.
__device__ __forceinline__ bool logit_not_finite(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x7F800000u) == 0x7F800000u && b != 0xFF800000u;
}

// ============================================================================
// Per-sequence sampling dials
// ============================================================================
//
// The scalar dials — temperature, top-k/top-p, the penalties, the EOS and
// segment-close ramps — were passed once per launch and applied to every row,
// so a wave that mixes configs (a deliberating turn beside an impulsive one, a
// narrator beside a reflection) sampled every row at whichever config sorted
// first. That let one row's EOS ramp cut another row's turn off after a single
// token. This struct carries the dials PER ROW instead: when the kernel is
// handed a `[batch_size]` array of these, each block reads its own from
// `seq_dials[blockIdx.x]` and the scalar arguments are ignored. A null pointer
// keeps the legacy shared-scalar behaviour.
//
// **Layout is load-bearing.** The Rust host uploads a `#[repr(C)]` twin of this
// struct (`batched_sampler::SeqDials`) as raw bytes; the field order and types
// here must match it exactly. Every field is 4 bytes (f32/i32) so the struct is
// naturally packed with no padding — keep it that way.
//
// `eos_token_id`, `vocab_size` (the row stride) and `live_vocab` (the tokens a
// row may produce) are NOT here: they are a property of the model/tokenizer,
// identical across the wave, so they stay scalar.
//
// `segment_close_token_id` IS here, per-row: a wave mixes rows that close on
// different tokens (a deliberating row closes its `</think>` span; a narrator
// row has no segment, token id -1), and a scalar taken from row 0 would gate
// the whole wave on one row's token — the exact cross-row bleed this struct
// exists to remove. -1 keeps a row's segment-close path off.
//
// `typical_draft` is the proposal a speculative verify row tests (-1 on a row
// that tests none), with `typical_eps`/`typical_delta` the Medusa thresholds it
// is accepted on when the sample did not land on it.
//
// `logits_row` is the row of the logits block this entry samples. A dispatch
// whose other rows are resolved elsewhere (a grammar allow-list) hands the
// kernel the whole block and names each kernel row's source row here, instead
// of gathering the kernel rows into a compacted copy first.
//
// `allowed_bits_lo`/`_hi` are the device address of the row's allowed-token
// bitmap, one bit per token of the vocabulary, or zero on a row with none. The
// bitmaps stand on the wave's span beside the logits. A grammar step that allows
// most of the vocabulary samples here, through its row's bitmap, instead of on
// an allow-list naming hundreds of thousands of tokens.
struct SeqDials {
    float temperature;
    int32_t top_k;
    float top_p;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    float dry_multiplier;
    float dry_base;
    int32_t dry_allowed_length;
    int32_t dry_range;
    float eos_boost;
    int32_t eos_ramp_start;
    int32_t eos_ramp_len;
    float eos_boost_max_multiplier;
    float cross_turn_penalty;
    float segment_close_boost;
    int32_t segment_close_token_id;
    int32_t segment_close_ramp_start;
    int32_t segment_close_ramp_len;
    float segment_close_max_multiplier;
    float segment_temp_boost;
    int32_t typical_draft;
    float typical_eps;
    float typical_delta;
    int32_t logits_row;
    uint32_t allowed_bits_lo;
    uint32_t allowed_bits_hi;
};

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

    // This row's allowed-token bitmap, or null: a clear bit is a -inf logit.
    const uint32_t* allowed_bits;

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
