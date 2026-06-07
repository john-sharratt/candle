//! Batched sampling wrapper around the CUDA kernel.
//!
//! Provides a high-level Rust API for the batched sampling kernel with:
//! - GPU buffer management
//! - Automatic dtype dispatching
//!
//! State (token counts, recent history) is owned by the caller (DecodeState).

use crate::config::SamplingConfig;
use crate::token_buffer::TokenBuffer;
use candle::cuda_backend::CudaStorageSlice;
use candle::{DType, Device, IndexOp, Tensor};
use candle_kernels::sampling::{run_batched_sampling, DType as KernelDType};
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// Per-sequence sampling state.
///
/// Tracks token counts and recent history for penalty calculations.
/// This struct persists across turns (owned by the Scheduler) so that
/// DRY penalty can see a rolling window of recent tokens spanning
/// turn boundaries.  Per-turn state (token_counts, current_len) is
/// reset via `end_turn()` at the start of each new turn.
#[derive(Debug, Clone)]
pub struct SequenceSamplingState {
    /// Token occurrence counts (for frequency/presence penalty).
    /// Indexed by token ID.
    pub token_counts: Vec<i32>,

    /// Prior-turn token counts (for cross-turn penalty).
    /// Incremented when `end_turn()` is called; cleared when the conversation is reset.
    pub cross_turn_counts: Vec<i32>,

    /// Recent token history (for repeat/DRY penalty).
    /// Stored oldest-first; the scheduler copies the tail window to the GPU buffer.
    pub recent_tokens: Vec<i32>,

    /// Number of tokens generated so far this turn (for dynamic EOS ramp).
    pub current_len: i32,

    /// Whether this sequence is currently inside a `<think>` block.
    pub in_thinking: bool,

    /// Tokens generated since entering thinking mode (for EOT ramp).
    /// Reset when the sequence exits thinking (`</think>` is emitted).
    pub thinking_len: i32,

    /// Current RNG offset (for deterministic sampling across calls).
    pub rng_offset: u64,
}

impl SequenceSamplingState {
    /// Create new state for a sequence.
    pub fn new(vocab_size: usize, max_recent_len: usize) -> Self {
        Self {
            token_counts: vec![0; vocab_size],
            cross_turn_counts: vec![0; vocab_size],
            recent_tokens: Vec::with_capacity(max_recent_len),
            current_len: 0,
            in_thinking: false,
            thinking_len: 0,
            rng_offset: 0,
        }
    }

    /// Record a generated token.
    pub fn record_token(&mut self, token: u32, max_recent_len: usize) {
        let token_idx = token as usize;
        if token_idx < self.token_counts.len() {
            self.token_counts[token_idx] += 1;
        }

        self.recent_tokens.push(token as i32);
        self.current_len += 1;

        // Track thinking length if inside a <think> block
        if self.in_thinking {
            self.thinking_len += 1;
        }

        // Maintain fixed-size sliding window
        if self.recent_tokens.len() > max_recent_len {
            self.recent_tokens.remove(0);
        }
    }

    /// Record multiple tokens (e.g., after prefill).
    pub fn record_tokens(&mut self, tokens: &[u32], max_recent_len: usize) {
        for &token in tokens {
            self.record_token(token, max_recent_len);
        }
    }

    /// Record prompt/context tokens for repeat-penalty context only.
    ///
    /// Populates `recent_tokens` (used by repeat penalty) but NOT
    /// `token_counts` (used by frequency/presence penalty).  This matches
    /// the standard behaviour where frequency and presence penalties
    /// apply only to *generated* tokens, not the prompt.
    pub fn record_context_tokens(&mut self, tokens: &[u32], max_recent_len: usize) {
        for &token in tokens {
            self.recent_tokens.push(token as i32);
            if self.recent_tokens.len() > max_recent_len {
                self.recent_tokens.remove(0);
            }
        }
    }

    /// Clear all state (for conversation reset).
    pub fn clear(&mut self) {
        self.token_counts.fill(0);
        self.cross_turn_counts.fill(0);
        self.recent_tokens.clear();
        self.current_len = 0;
        self.in_thinking = false;
        self.thinking_len = 0;
        self.rng_offset = 0;
    }

    /// Snapshot current turn counts into cross-turn counts, then reset for the next turn.
    /// Call this at the end of each assistant turn.
    ///
    /// NOTE: `recent_tokens` is intentionally NOT cleared here.
    /// It acts as a rolling window across the entire conversation so that
    /// DRY penalty can detect repeated n-gram sequences spanning turn
    /// boundaries.  The sliding-window cap (`max_recent_len`) keeps it
    /// bounded.  Repeat penalty also uses `recent_tokens` and benefits
    /// from the cross-turn window.
    pub fn end_turn(&mut self) {
        // Accumulate into cross-turn counts
        for (cross, &cur) in self
            .cross_turn_counts
            .iter_mut()
            .zip(self.token_counts.iter())
        {
            *cross = cross.saturating_add(cur);
        }
        // Reset per-turn state (frequency/presence penalties are per-turn)
        self.token_counts.fill(0);
        self.current_len = 0;
    }

    /// Advance RNG offset (called after each sampling).
    pub fn advance_rng(&mut self) {
        self.rng_offset = self.rng_offset.wrapping_add(1);
    }

    /// Enter thinking mode (called when `<think>` is emitted).
    pub fn enter_thinking(&mut self) {
        self.in_thinking = true;
        self.thinking_len = 0;
    }

    /// Exit thinking mode (called when `</think>` is emitted).
    pub fn exit_thinking(&mut self) {
        self.in_thinking = false;
        self.thinking_len = 0;
    }

    /// Update thinking state after a sampled token.
    ///
    /// Call this after each token is sampled to automatically detect
    /// `<think>` / `</think>` transitions.
    pub fn update_thinking_state(&mut self, token: u32, think_start_id: i32, eot_id: i32) {
        if think_start_id < 0 || eot_id < 0 {
            return; // Thinking not configured
        }
        if token as i32 == think_start_id {
            self.enter_thinking();
        } else if token as i32 == eot_id && self.in_thinking {
            self.exit_thinking();
        }
    }
}

/// Stateless batched sampler that invokes the CUDA kernel.
///
/// This sampler does not own per-sequence state. Instead, callers pass
/// `SequenceSamplingState` references which are updated in place.
pub struct BatchedSampler {
    /// Device the sampler operates on.
    #[allow(dead_code)]
    device: Device,

    /// Vocabulary size.
    vocab_size: usize,

    /// Maximum recent token history length.
    max_recent_len: usize,

    /// EOS token ID.
    eos_tokens: TokenBuffer,

    /// Optional path to write penalty state during decoding.
    penalty_log_path: Option<std::path::PathBuf>,
}

impl BatchedSampler {
    /// Create a new batched sampler.
    pub fn new(
        device: Device,
        vocab_size: usize,
        max_recent_len: usize,
        eos_tokens: TokenBuffer,
        penalty_log_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            device,
            vocab_size,
            max_recent_len,
            eos_tokens,
            penalty_log_path,
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the maximum recent token history length.
    pub fn max_recent_len(&self) -> usize {
        self.max_recent_len
    }

    /// Sample tokens for a batch of sequences.
    ///
    /// # Arguments
    /// - `logits`: Batched logits tensor, shape `[batch_size, vocab_size]` or similar.
    /// - `states`: Mutable references to per-sequence sampling states.
    /// - `configs`: Per-sequence sampling configs.
    ///
    /// # Returns
    /// Sampled token IDs for each sequence. States are updated in place.
    pub fn sample_batch(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        let batch_size = states.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        if matches!(self.device, Device::Cuda(_)) {
            return self.sample_batch_cuda(logits, states, configs);
        }

        self.sample_batch_cpu(logits, states, configs)
    }

    /// CPU fallback implementation using candle's built-in sampling.
    fn sample_batch_cpu(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        use candle_transformers::generation::{LogitsProcessor, Sampling};

        let batch_size = states.len();
        let mut results = Vec::with_capacity(batch_size);

        for (i, (state, &config)) in states.iter_mut().zip(configs.iter()).enumerate() {
            // Extract logits for this sequence
            let seq_logits = if logits.dims().len() == 2 {
                logits.i(i)?
            } else {
                logits.clone()
            };

            // Convert SamplingConfig to Sampling enum for CPU path
            let sampling = if config.temperature <= 0.0 {
                Sampling::ArgMax
            } else if config.top_k > 0 && config.top_p < 1.0 {
                Sampling::TopKThenTopP {
                    k: config.top_k as usize,
                    p: config.top_p as f64,
                    temperature: config.temperature as f64,
                }
            } else if config.top_k > 0 {
                Sampling::TopK {
                    k: config.top_k as usize,
                    temperature: config.temperature as f64,
                }
            } else if config.top_p < 1.0 {
                Sampling::TopP {
                    p: config.top_p as f64,
                    temperature: config.temperature as f64,
                }
            } else {
                Sampling::All {
                    temperature: config.temperature as f64,
                }
            };

            let seed = config.seed.wrapping_add(state.rng_offset);
            let mut processor = LogitsProcessor::from_sampling(seed, sampling);
            let mut token = processor.sample(&seq_logits)?;

            // EOT overrides: force </think> when thinking budget is exhausted.
            if config.eot_token_id >= 0 && state.in_thinking {
                let should_force = (config.graceful_eot_after > 0
                    && state.thinking_len >= config.graceful_eot_after
                    && state
                        .recent_tokens
                        .last()
                        .map(|&t| config.sentence_end_token_ids.contains(&t))
                        .unwrap_or(false))
                    || (config.force_eot_after > 0 && state.thinking_len >= config.force_eot_after);

                if should_force {
                    token = config.eot_token_id as u32;
                }
            }

            // EOS failsafe overrides (post-sampler)
            let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);
            if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
                // Hard stop: unconditionally force EOS regardless of sentence position.
                token = eos_token_id;
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && !config.sentence_end_token_ids.is_empty()
            {
                // Graceful stop: emit EOS only when the last token was a sentence-ending
                // token (`.`, `!`, `?`, `\n`).  This lets the current sentence complete
                // before termination, preventing mid-sentence truncation.
                // `forced_eos_after` acts as the hard backstop if no boundary is seen.
                if state
                    .recent_tokens
                    .last()
                    .map(|&t| config.sentence_end_token_ids.contains(&t))
                    .unwrap_or(false)
                {
                    token = eos_token_id;
                }
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && config.sentence_end_token_ids.is_empty()
            {
                // No sentence-end tokens resolved (e.g. model loaded without tokenizer
                // resolution): fall back to hard stop at the graceful threshold.
                token = eos_token_id;
            }

            // Record the token and advance RNG
            state.record_token(token, self.max_recent_len);
            state.advance_rng();

            results.push(token);
        }

        Ok(results)
    }

    /// CUDA kernel implementation.
    fn sample_batch_cuda(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        let batch_size = states.len();

        // Determine dtype from logits
        let dtype = logits.dtype();
        let dtype_enum = match dtype {
            DType::F32 => KernelDType::F32 as i32,
            DType::F16 => KernelDType::F16 as i32,
            DType::BF16 => KernelDType::BF16 as i32,
            _ => {
                return Err(candle::Error::Msg(format!(
                    "unsupported dtype for sampling: {:?}",
                    dtype
                )))
            }
        };

        // Flatten logits to [batch_size, vocab_size]
        let logits_flat = match logits.dims().len() {
            1 => logits.unsqueeze(0)?,
            2 => logits.clone(),
            3 => {
                // [batch, seq_len, vocab] -> take last position
                let seq_len = logits.dim(1)?;
                logits.i((.., seq_len - 1, ..))?
            }
            n => return Err(candle::Error::Msg(format!("unexpected logits rank: {}", n))),
        };

        let logits_vocab_size = logits_flat.dim(1)? as i32;

        // Validate that our penalty buffer vocab_size matches the logits.
        // A mismatch means token_counts is undersized and the kernel would
        // read out-of-bounds GPU memory for high token IDs (e.g. EOS tokens).
        if (self.vocab_size as i32) != logits_vocab_size {
            return Err(candle::Error::Msg(format!(
                "vocab_size mismatch: sampler has {} but logits have {} — \
                 set EngineConfig::vocab_size to match the model/tokenizer",
                self.vocab_size, logits_vocab_size
            )));
        }
        let vocab_size = logits_vocab_size;

        // For simplicity in Phase 1, we'll use the first config for all sequences
        // (batched configs will be supported in Phase 2)
        let config = configs[0];

        // Get DRY params
        let (dry_multiplier, dry_base, dry_allowed_length, dry_range) =
            if let Some(ref dry) = config.dry {
                (dry.multiplier, dry.base, dry.allowed_length, dry.range)
            } else {
                (0.0, 1.75, 2, 0)
            };

        // Build penalty buffers from states
        let (token_counts, cross_turn_counts, recent_tokens, recent_lens, current_lens) = self
            .build_penalty_buffers_from_states(
                states,
                config.presence_penalty,
                config.repeat_last_n,
                dry_range,
            )?;

        // Get EOS token
        let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);

        // Build banned tokens buffer
        let banned_tokens = &config.banned_tokens;
        let num_banned = banned_tokens.len() as i32;

        // Build stencil buffer
        let stencil = &config.stencil;
        let stencil_size = stencil.len() as i32;

        // Allocate output buffer
        let mut output_tokens = vec![0u32; batch_size];

        // Build RNG offsets from states
        let mut rng_offsets: Vec<u64> = states.iter().map(|s| s.rng_offset).collect();

        // Compute EOS ramp params
        let (eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier) = if config.dynamic_eos_boost {
            (
                config.eos_ramp_start,
                config.eos_ramp_len,
                config.eos_boost_max_multiplier,
            )
        } else {
            (0, 0, 0.0)
        };

        // Compute EOT (end-of-thinking) params
        // Only active when eot_boost > 0, eot_token_id >= 0, and at least one sequence is in thinking mode
        let thinking_lens: Vec<i32> = states
            .iter()
            .map(|s| if s.in_thinking { s.thinking_len } else { 0 })
            .collect();
        let eot_active = config.eot_boost != 0.0 && config.eot_token_id >= 0;
        let (eot_boost, eot_token_id, eot_ramp_start, eot_ramp_len, eot_boost_max_multiplier) =
            if eot_active {
                (
                    config.eot_boost,
                    config.eot_token_id,
                    config.eot_ramp_start,
                    config.eot_ramp_len,
                    config.eot_boost_max_multiplier,
                )
            } else {
                (0.0, -1, 0, 0, 0.0)
            };

        // Invoke the CUDA kernel
        self.invoke_cuda_kernel(
            &logits_flat,
            batch_size as i32,
            vocab_size,
            dtype_enum,
            config.temperature,
            config.top_k,
            config.top_p,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_range,
            config.eos_boost,
            eos_token_id as i32,
            eos_ramp_start,
            eos_ramp_len,
            eos_boost_max_multiplier,
            config.cross_turn_penalty,
            &cross_turn_counts,
            &current_lens,
            eot_boost,
            eot_token_id,
            eot_ramp_start,
            eot_ramp_len,
            eot_boost_max_multiplier,
            &thinking_lens,
            &token_counts,
            banned_tokens,
            num_banned,
            &recent_tokens,
            &recent_lens,
            stencil,
            stencil_size,
            &mut output_tokens,
            config.seed,
            &mut rng_offsets,
        )?;

        // Update states with sampled tokens and new RNG offsets.
        // Apply post-sampler EOS failsafe overrides: if the sequence has exceeded
        // the configured length limits, replace the sampled token with EOS.
        for (i, state) in states.iter_mut().enumerate() {
            let mut token = output_tokens[i];

            // EOT overrides: force </think> when thinking budget is exhausted.
            if config.eot_token_id >= 0 && state.in_thinking {
                let should_force = (config.graceful_eot_after > 0
                    && state.thinking_len >= config.graceful_eot_after
                    && state
                        .recent_tokens
                        .last()
                        .map(|&t| config.sentence_end_token_ids.contains(&t))
                        .unwrap_or(false))
                    || (config.force_eot_after > 0 && state.thinking_len >= config.force_eot_after);

                if should_force {
                    token = config.eot_token_id as u32;
                }
            }

            // EOS failsafe overrides (post-sampler)
            if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
                // Hard stop: unconditionally force EOS regardless of sentence position.
                token = eos_token_id;
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && !config.sentence_end_token_ids.is_empty()
            {
                // Graceful stop: emit EOS only at the next sentence boundary so the
                // current sentence completes before generation stops.  Prevents the
                // mid-sentence truncation that occurred when EOS was unconditionally
                // forced here.  `forced_eos_after` is the hard backstop.
                if state
                    .recent_tokens
                    .last()
                    .map(|&t| config.sentence_end_token_ids.contains(&t))
                    .unwrap_or(false)
                {
                    token = eos_token_id;
                }
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && config.sentence_end_token_ids.is_empty()
            {
                // No sentence-end tokens resolved: fall back to hard stop.
                token = eos_token_id;
            }
            output_tokens[i] = token;

            state.record_token(token, self.max_recent_len);
            state.rng_offset = rng_offsets[i];
            // Auto-detect <think> / </think> transitions for EOT boost
            state.update_thinking_state(token, config.think_start_token_id, config.eot_token_id);
        }

        Ok(output_tokens)
    }

    /// Build penalty buffers from states.
    fn build_penalty_buffers_from_states(
        &self,
        states: &[&mut SequenceSamplingState],
        presence_penalty: f32,
        repeat_last_n: i32,
        dry_range: i32,
    ) -> candle::Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let batch_size = states.len();

        // Flatten token counts: [batch_size * vocab_size]
        let mut token_counts = Vec::with_capacity(batch_size * self.vocab_size);
        for state in states.iter() {
            token_counts.extend_from_slice(&state.token_counts);
        }

        // Flatten cross-turn counts: [batch_size * vocab_size]
        let mut cross_turn_counts = Vec::with_capacity(batch_size * self.vocab_size);
        for state in states.iter() {
            cross_turn_counts.extend_from_slice(&state.cross_turn_counts);
        }

        // Log penalty state if a log path is configured
        if let Some(ref log_path) = self.penalty_log_path {
            if let Err(e) = self.write_penalty_log(log_path, states, presence_penalty) {
                tracing::warn!("Failed to write penalty log: {}", e);
            }
        }

        // Flatten recent tokens: [batch_size * max_recent_len].
        //
        // The window must be large enough for BOTH penalties that use this buffer:
        //   • Repeat penalty needs `repeat_last_n` tokens.
        //   • DRY penalty needs `dry_range` tokens (kernel line:
        //       search_start = recent_len - dry_range when dry_range < recent_len).
        // Using only `repeat_last_n` here would silently cap DRY to that smaller
        // window even when dry_range >> repeat_last_n, causing cross-turn phrases
        // to slide out of view before the model reaches the position where it
        // repeats them.  Take the max of both requirements so each penalty sees
        // the context depth it was configured for.
        let mut recent_tokens = Vec::with_capacity(batch_size * self.max_recent_len);
        let mut recent_lens = Vec::with_capacity(batch_size);

        for state in states.iter() {
            let total = state.recent_tokens.len();
            let repeat_win = if repeat_last_n > 0 {
                repeat_last_n as usize
            } else {
                self.max_recent_len
            };
            let dry_win = if dry_range > 0 {
                (dry_range as usize).min(self.max_recent_len)
            } else {
                0
            };
            let window = repeat_win.max(dry_win).min(total);
            // Copy the newest `window` tokens (tail of the oldest-first buffer)
            let start = total - window;
            recent_lens.push(window as i32);
            recent_tokens.extend_from_slice(&state.recent_tokens[start..]);
            recent_tokens.extend(std::iter::repeat_n(0, self.max_recent_len - window));
        }

        // Current generated lengths (for dynamic EOS ramp)
        let current_lens: Vec<i32> = states.iter().map(|s| s.current_len).collect();

        Ok((
            token_counts,
            cross_turn_counts,
            recent_tokens,
            recent_lens,
            current_lens,
        ))
    }

    fn write_penalty_log(
        &self,
        log_path: &std::path::Path,
        states: &[&mut SequenceSamplingState],
        presence_penalty: f32,
    ) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(log_path)?;

        // Write the presence penalty value at the top
        writeln!(file, "=== PRESENCE PENALTY: {} ===", presence_penalty)?;
        writeln!(file)?;

        for (batch_idx, state) in states.iter().enumerate() {
            writeln!(file, "=== Batch {} ===", batch_idx)?;
            writeln!(file, "Vocab size: {}", state.token_counts.len())?;
            writeln!(file)?;

            // Write nonzero token counts
            let nonzero_tokens: Vec<_> = state
                .token_counts
                .iter()
                .enumerate()
                .filter(|(_, &count)| count > 0)
                .collect();

            if nonzero_tokens.is_empty() {
                writeln!(file, "No penalties applied yet")?;
            } else {
                writeln!(file, "Tokens with nonzero counts (will be penalized):")?;
                for (token_id, &count) in nonzero_tokens {
                    writeln!(file, "  Token {}: count={}", token_id, count)?;
                }
            }

            writeln!(file)?;
            writeln!(
                file,
                "Recent token history ({} tokens):",
                state.recent_tokens.len()
            )?;
            for (i, &token_id) in state.recent_tokens.iter().enumerate() {
                write!(file, "  {} ", token_id)?;
                if (i + 1) % 20 == 0 {
                    writeln!(file)?;
                }
            }
            if !state.recent_tokens.is_empty() {
                writeln!(file)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }

    /// Invoke the CUDA kernel with the given parameters.
    #[allow(clippy::too_many_arguments)]
    fn invoke_cuda_kernel(
        &self,
        logits: &Tensor,
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
        cross_turn_counts: &[i32],
        current_lens: &[i32],
        eot_boost: f32,
        eot_token_id: i32,
        eot_ramp_start: i32,
        eot_ramp_len: i32,
        eot_boost_max_multiplier: f32,
        thinking_lens: &[i32],
        token_counts: &[i32],
        banned_tokens: &[i32],
        num_banned: i32,
        recent_tokens: &[i32],
        recent_lens: &[i32],
        stencil: &[i32],
        stencil_size: i32,
        output_tokens: &mut [u32],
        seed: u64,
        rng_offsets: &mut [u64],
    ) -> candle::Result<()> {
        // Get the CUDA device and stream
        let cuda_device = match &self.device {
            Device::Cuda(dev) => dev,
            _ => return Err(candle::Error::Msg("expected CUDA device".into())),
        };

        let stream = cuda_device.cuda_stream();

        // Get logits storage and layout
        let (logits_storage, logits_layout) = logits.storage_and_layout();
        let cuda_storage = match &*logits_storage {
            candle::Storage::Cuda(cs) => cs,
            _ => return Err(candle::Error::Msg("logits must be on CUDA".into())),
        };

        // Upload buffers to GPU
        let token_counts_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(token_counts)
            .map_err(|e| candle::Error::Msg(format!("failed to upload token_counts: {}", e)))?;

        let cross_turn_gpu: cudarc::driver::CudaSlice<i32> = if cross_turn_penalty != 0.0 {
            stream.memcpy_stod(cross_turn_counts).map_err(|e| {
                candle::Error::Msg(format!("failed to upload cross_turn_counts: {}", e))
            })?
        } else {
            stream.memcpy_stod(&[-1i32]).map_err(|e| {
                candle::Error::Msg(format!("failed to upload cross_turn_counts: {}", e))
            })?
        };

        let current_lens_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(current_lens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload current_lens: {}", e)))?;

        let thinking_lens_gpu: cudarc::driver::CudaSlice<i32> =
            stream.memcpy_stod(thinking_lens).map_err(|e| {
                candle::Error::Msg(format!("failed to upload thinking_lens: {}", e))
            })?;

        let banned_gpu: cudarc::driver::CudaSlice<i32> = if banned_tokens.is_empty() {
            stream
                .memcpy_stod(&[-1i32])
                .map_err(|e| candle::Error::Msg(format!("failed to upload banned: {}", e)))?
        } else {
            stream
                .memcpy_stod(banned_tokens)
                .map_err(|e| candle::Error::Msg(format!("failed to upload banned: {}", e)))?
        };

        let recent_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(recent_tokens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload recent_tokens: {}", e)))?;

        let recent_lens_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(recent_lens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload recent_lens: {}", e)))?;

        let stencil_gpu: cudarc::driver::CudaSlice<i32> = if stencil.is_empty() {
            stream
                .memcpy_stod(&[-1i32])
                .map_err(|e| candle::Error::Msg(format!("failed to upload stencil: {}", e)))?
        } else {
            stream
                .memcpy_stod(stencil)
                .map_err(|e| candle::Error::Msg(format!("failed to upload stencil: {}", e)))?
        };

        let mut output_gpu: cudarc::driver::CudaSlice<u32> = stream
            .memcpy_stod(output_tokens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload output: {}", e)))?;

        let mut rng_gpu: cudarc::driver::CudaSlice<u64> = stream
            .memcpy_stod(rng_offsets)
            .map_err(|e| candle::Error::Msg(format!("failed to upload rng_offsets: {}", e)))?;

        // Get device pointers and call kernel in a scoped block
        // so guards are dropped before download
        {
            let (tc_ptr, _g1) = token_counts_gpu.device_ptr(&stream);
            let (cross_ptr, _g2) = cross_turn_gpu.device_ptr(&stream);
            let (cur_lens_ptr, _g3) = current_lens_gpu.device_ptr(&stream);
            let (think_lens_ptr, _g3b) = thinking_lens_gpu.device_ptr(&stream);
            let (ban_ptr, _g4) = banned_gpu.device_ptr(&stream);
            let (recent_ptr, _g5) = recent_gpu.device_ptr(&stream);
            let (recent_lens_ptr, _g6) = recent_lens_gpu.device_ptr(&stream);
            let (stencil_ptr, _g7) = stencil_gpu.device_ptr(&stream);
            let (output_ptr, _g8) = output_gpu.device_ptr_mut(&stream);
            let (rng_ptr, _g9) = rng_gpu.device_ptr_mut(&stream);

            // Helper closure to call kernel with logits pointer
            let call_kernel = |logits_ptr: *const std::ffi::c_void| unsafe {
                run_batched_sampling(
                    logits_ptr,
                    batch_size,
                    vocab_size,
                    dtype,
                    temperature,
                    top_k,
                    top_p,
                    repeat_penalty,
                    frequency_penalty,
                    presence_penalty,
                    dry_multiplier,
                    dry_base,
                    dry_allowed_length,
                    dry_range,
                    eos_boost,
                    eos_token_id,
                    eos_ramp_start,
                    eos_ramp_len,
                    eos_boost_max_multiplier,
                    cross_turn_penalty,
                    if cross_turn_penalty != 0.0 {
                        cross_ptr as *const i32
                    } else {
                        std::ptr::null()
                    },
                    cur_lens_ptr as *const i32,
                    eot_boost,
                    eot_token_id,
                    eot_ramp_start,
                    eot_ramp_len,
                    eot_boost_max_multiplier,
                    think_lens_ptr as *const i32,
                    tc_ptr as *const i32,
                    ban_ptr as *const i32,
                    num_banned,
                    0, // banned_tokens_per_seq (global banned list)
                    recent_ptr as *const i32,
                    recent_lens_ptr as *const i32,
                    self.max_recent_len as i32,
                    if stencil_size > 0 {
                        stencil_ptr as *const i32
                    } else {
                        std::ptr::null()
                    },
                    stencil_size,
                    output_ptr as *mut u32,
                    seed,
                    rng_ptr as *mut u64,
                );
            };

            // Match on dtype to get the properly-typed slice and its pointer
            let start_offset = logits_layout.start_offset();
            match &cuda_storage.slice {
                CudaStorageSlice::F32(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr =
                        (ptr + (start_offset as u64 * 4)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::F16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr =
                        (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::BF16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr =
                        (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                _ => {
                    return Err(candle::Error::Msg(format!(
                        "unsupported dtype for sampling: {:?}",
                        logits.dtype()
                    )));
                }
            }
        } // Guards dropped here

        // Synchronize and download results
        stream
            .synchronize()
            .map_err(|e| candle::Error::Msg(format!("CUDA sync failed: {}", e)))?;

        let output_vec = stream
            .memcpy_dtov(&output_gpu)
            .map_err(|e| candle::Error::Msg(format!("failed to download output: {}", e)))?;

        let rng_vec = stream
            .memcpy_dtov(&rng_gpu)
            .map_err(|e| candle::Error::Msg(format!("failed to download rng: {}", e)))?;

        output_tokens.copy_from_slice(&output_vec);
        rng_offsets.copy_from_slice(&rng_vec);

        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SamplingConfig;

    const VOCAB_SIZE: usize = 100;
    const MAX_RECENT: usize = 32;
    const EOS_TOKEN: u32 = 2;

    fn make_sampler() -> BatchedSampler {
        BatchedSampler::new(
            candle::Device::Cpu,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        )
    }

    fn make_state() -> SequenceSamplingState {
        SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT)
    }

    // ── EOS failsafe override tests ────────────────────────────────────

    #[test]
    fn test_forced_eos_after_overrides_token() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(0, 10);
        let mut state = make_state();

        // Simulate 10 tokens already generated
        for _ in 0..10 {
            state.record_token(42, MAX_RECENT);
        }
        assert_eq!(state.current_len, 10);

        // Build logits that strongly favor token 42
        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        // Should have been overridden to EOS despite logits favoring 42
        assert_eq!(
            tokens[0], EOS_TOKEN,
            "forced_eos_after should override to EOS"
        );
    }

    #[test]
    fn test_graceful_eos_after_overrides_token() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(5, 0);
        let mut state = make_state();

        // Generate 5 tokens
        for _ in 0..5 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(
            tokens[0], EOS_TOKEN,
            "graceful_eos_after should override to EOS"
        );
    }

    #[test]
    fn test_eos_failsafe_disabled_when_zero() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax(); // both 0 = disabled
        let mut state = make_state();

        for _ in 0..1000 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(tokens[0], 42, "failsafe disabled: should sample normally");
    }

    #[test]
    fn test_eos_failsafe_below_threshold_no_override() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(100, 200);
        let mut state = make_state();

        // Only 50 tokens — under both thresholds
        for _ in 0..50 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(tokens[0], 42, "below threshold: should sample normally");
    }

    #[test]
    fn test_eos_failsafe_records_eos_token_in_state() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(5, 10);
        let mut state = make_state();

        for _ in 0..5 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let _ = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        // State should have recorded the EOS token (not the original 42)
        assert_eq!(state.current_len, 6); // 5 + 1 for the overridden token
        assert_eq!(*state.recent_tokens.last().unwrap(), EOS_TOKEN as i32);
    }

    // ── update_thinking_state tests ────────────────────────────────────

    #[test]
    fn test_think_start_enters_thinking() {
        let mut state = make_state();
        let think_start = 10i32;
        let eot = 11i32;

        assert!(!state.in_thinking);
        assert_eq!(state.thinking_len, 0);

        state.update_thinking_state(think_start as u32, think_start, eot);
        assert!(
            state.in_thinking,
            "should enter thinking on think_start token"
        );
        assert_eq!(state.thinking_len, 0, "thinking_len reset on enter");
    }

    #[test]
    fn test_eot_exits_thinking() {
        let mut state = make_state();
        let think_start = 10i32;
        let eot = 11i32;

        // Enter thinking
        state.update_thinking_state(think_start as u32, think_start, eot);
        assert!(state.in_thinking);

        // Generate some tokens while thinking
        state.record_token(42, MAX_RECENT);
        state.record_token(43, MAX_RECENT);
        assert_eq!(state.thinking_len, 2);

        // Exit thinking
        state.update_thinking_state(eot as u32, think_start, eot);
        assert!(!state.in_thinking, "should exit thinking on eot token");
        assert_eq!(state.thinking_len, 0, "thinking_len reset on exit");
    }

    #[test]
    fn test_eot_without_thinking_is_noop() {
        let mut state = make_state();
        let think_start = 10i32;
        let eot = 11i32;

        // EOT when not in thinking should be a no-op
        state.update_thinking_state(eot as u32, think_start, eot);
        assert!(!state.in_thinking, "should remain not-thinking");
    }

    #[test]
    fn test_thinking_disabled_when_ids_negative() {
        let mut state = make_state();

        // -1 means not configured
        state.update_thinking_state(10, -1, 11);
        assert!(
            !state.in_thinking,
            "should not enter thinking when think_start_id < 0"
        );

        state.update_thinking_state(10, 10, -1);
        assert!(
            !state.in_thinking,
            "should not enter thinking when eot_id < 0"
        );
    }

    #[test]
    fn test_thinking_len_tracks_tokens() {
        let mut state = make_state();
        let think_start = 10i32;
        let eot = 11i32;

        state.update_thinking_state(think_start as u32, think_start, eot);

        for i in 0..5 {
            state.record_token(40 + i, MAX_RECENT);
        }
        assert_eq!(state.thinking_len, 5);

        // Re-entering thinking resets the counter
        state.update_thinking_state(think_start as u32, think_start, eot);
        assert_eq!(state.thinking_len, 0);
    }
}
