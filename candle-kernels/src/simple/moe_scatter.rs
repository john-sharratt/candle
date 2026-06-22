//! FFI bindings for fused MoE gather and weighted-scatter-add CUDA kernels.
//!
//! These are batched operations that process ALL experts in a single kernel
//! launch, replacing multiple Tensor ops (index_select, broadcast_mul,
//! index_add) with one fused kernel.

use std::ffi::c_void;

/// Data type enum matching the dispatcher switch on the CUDA side.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum MoeScatterDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

extern "C" {
    /// Fused gather: out[i, j] = xs[token_ids[i], j]
    ///
    /// - `dtype`: MoeScatterDType value
    /// - `out`: device pointer to output [total_rows, hidden_dim]
    /// - `xs`: device pointer to input [num_tokens, hidden_dim]
    /// - `token_ids`: device pointer to u32 index array [total_rows]
    /// - `total_rows`: number of rows to gather
    /// - `hidden_dim`: column count
    pub fn run_moe_gather(
        dtype: i32,
        out: *mut c_void,
        xs: *const c_void,
        token_ids: *const u32,
        total_rows: usize,
        hidden_dim: usize,
    );

    /// Fused router: softmax + top-k select + (optional) renormalize, one thread per token.
    ///
    /// Replaces the `softmax → sort → narrow(top-k) → renorm → flatten` op chain with one
    /// launch. Top-k of softmax == top-k of logits (monotonic) and renorm cancels the global
    /// softmax denominator, so the kernel makes a single pass over the experts.
    ///
    /// - `dtype`: MoeScatterDType value (logits dtype: 0=f32, 1=f16, 2=bf16)
    /// - `logits`: device pointer to `[num_tokens, n_experts]`
    /// - `out_idx`: device pointer to u32 `[num_tokens, k]` top-k expert indices (descending)
    /// - `out_weights`: device pointer to f32 `[num_tokens, k]` routing weights (descending)
    /// - `norm_topk`: 1 = renormalized top-k softmax, 0 = plain full-softmax weights
    pub fn run_moe_route(
        dtype: i32,
        logits: *const c_void,
        out_idx: *mut u32,
        out_weights: *mut f32,
        num_tokens: i32,
        n_experts: i32,
        k: i32,
        norm_topk: i32,
    );

    /// Deterministic scatter: sequential per-token reduce, no atomicAdd.
    /// perm[i] maps token-major index i to the expert-major row in down_out,
    /// so no CPU-side reorder of down_out is needed before calling.
    /// Variable k is supported via per-token prefix-sum offsets in token_starts.
    ///
    /// ACCUMULATES into ys (+=). Initialize ys to zero before the first call.
    ///
    /// - `dtype`: MoeScatterDType value
    /// - `ys`: device pointer to output [num_tokens, hidden_dim] (accumulated into)
    /// - `down_out`: device pointer to expert outputs in expert-major order [total_batch, hidden_dim]
    /// - `perm`: device pointer to u32 gather-permutation [total_batch]; perm[i] = expert-major row
    /// - `weights_flat`: device pointer to f32 routing weights [total_weight_entries]
    /// - `reordered_weight_ids`: device pointer to u32 weight indices in token-major order [total_batch]
    /// - `token_starts`: device pointer to i32 prefix-sum array [num_tokens + 1]
    ///   where token_starts[t] is the start index for token t in the token-major arrays
    /// - `num_tokens`: number of output tokens
    /// - `hidden`: hidden dimension
    pub fn run_deterministic_scatter(
        dtype: i32,
        ys: *mut c_void,
        down_out: *const c_void,
        perm: *const u32,
        weights_flat: *const f32,
        reordered_weight_ids: *const u32,
        token_starts: *const i32,
        num_tokens: i32,
        hidden: i32,
    );
}
