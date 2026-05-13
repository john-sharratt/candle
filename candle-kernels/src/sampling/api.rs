//! FFI bindings for batched sampling kernels
//!
//! Provides a unified dispatcher for batched sampling with support for
//! multiple data types and various penalty/constraint options.

use core::ffi::c_void;

/// Data type enum for sampling dispatcher
#[repr(i32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    FP8E4M3 = 3,
}

extern "C" {
    /// Unified batched sampling dispatcher.
    ///
    /// Supports all penalty types (repeat, frequency, presence, DRY),
    /// EOS boosting, banned tokens, and stencil-constrained sampling.
    ///
    /// # Parameters
    /// - `logits`: Input logits tensor (type determined by `dtype`)
    /// - `batch_size`: Number of sequences in the batch
    /// - `vocab_size`: Vocabulary size
    /// - `dtype`: Data type (0=f32, 1=f16, 2=bf16, 3=fp8_e4m3)
    /// - `temperature`: Sampling temperature (0 = argmax/greedy)
    /// - `top_k`: Top-k sampling (0 = disabled)
    /// - `top_p`: Top-p/nucleus sampling (1.0 = disabled)
    /// - `repeat_penalty`: Repeat penalty multiplier (1.0 = disabled)
    /// - `frequency_penalty`: Frequency penalty (0.0 = disabled)
    /// - `presence_penalty`: Presence penalty (0.0 = disabled)
    /// - `dry_multiplier`: DRY penalty multiplier (0.0 = disabled)
    /// - `dry_base`: DRY penalty base
    /// - `dry_allowed_length`: DRY allowed length
    /// - `dry_range`: DRY range
    /// - `eos_boost`: EOS token boost (0.0 = disabled)
    /// - `eos_token_id`: EOS token ID
    /// - `eos_ramp_start`: Token count where EOS ramp begins (typically 80% of ramp_len)
    /// - `eos_ramp_len`: Length of ramp-up for dynamic EOS boost (0 = static boost)
    /// - `eos_boost_max_multiplier`: Multiplier applied at ramp peak (0 = static boost)
    /// - `cross_turn_penalty`: Additive penalty for tokens seen in previous turns (0.0 = disabled)
    /// - `cross_turn_counts`: Per-token prior-turn counts for cross-turn penalty (nullable)
    /// - `current_lens`: Current generated length per sequence for EOS ramp (nullable)
    /// - `eot_boost`: EOT token boost for end-of-thinking (0.0 = disabled)
    /// - `eot_token_id`: Token ID of `</think>` (-1 = disabled)
    /// - `eot_ramp_start`: Thinking-token count where EOT ramp begins
    /// - `eot_ramp_len`: Thinking-token count where EOT ramp reaches full
    /// - `eot_boost_max_multiplier`: Multiplier at EOT ramp peak
    /// - `thinking_lens`: Per-sequence thinking token counts (nullable)
    /// - `token_counts`: Per-token counts for frequency penalty (nullable)
    /// - `banned_tokens`: Banned token IDs (nullable)
    /// - `num_banned_tokens`: Total number of banned tokens
    /// - `banned_tokens_per_seq`: Banned tokens per sequence
    /// - `recent_tokens`: Recent token history for DRY (nullable)
    /// - `recent_lens`: Length of recent tokens per sequence (nullable)
    /// - `max_recent_len`: Maximum recent token history length
    /// - `stencil`: Allowed token IDs for constrained sampling (nullable)
    /// - `stencil_size`: Number of allowed tokens (0 = disabled)
    /// - `output_tokens`: Output sampled token IDs
    /// - `seed`: RNG seed
    /// - `rng_offsets`: Per-sequence RNG offsets (nullable for argmax)
    pub fn run_batched_sampling(
        logits: *const c_void,
        batch_size: i32,
        vocab_size: i32,
        dtype: i32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: i32,
        dry_range: i32,
        eos_boost: f32,
        eos_token_id: i32,
        eos_ramp_start: i32,
        eos_ramp_len: i32,
        eos_boost_max_multiplier: f32,
        cross_turn_penalty: f32,
        cross_turn_counts: *const i32,
        current_lens: *const i32,
        eot_boost: f32,
        eot_token_id: i32,
        eot_ramp_start: i32,
        eot_ramp_len: i32,
        eot_boost_max_multiplier: f32,
        thinking_lens: *const i32,
        token_counts: *const i32,
        banned_tokens: *const i32,
        num_banned_tokens: i32,
        banned_tokens_per_seq: i32,
        recent_tokens: *const i32,
        recent_lens: *const i32,
        max_recent_len: i32,
        stencil: *const i32,
        stencil_size: i32,
        output_tokens: *mut u32,
        seed: u64,
        rng_offsets: *mut u64,
    );
}
