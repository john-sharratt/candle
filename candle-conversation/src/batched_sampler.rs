//! Batched sampling wrapper around the CUDA kernel.
//!
//! Provides a high-level Rust API for the batched sampling kernel with:
//! - GPU buffer management
//! - Automatic dtype dispatching
//!
//! State (token counts, recent history) is owned by the caller (DecodeState).

use crate::config::SamplingConfig;
use crate::stencil::ban;
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

    /// Number of tokens generated so far this turn (for the dynamic EOS ramp).
    pub current_len: i32,

    /// Whether this sequence is currently inside a marked segment (a caller-defined
    /// span between two token ids).
    pub in_segment: bool,

    /// Tokens generated since the current segment opened (for the segment-close
    /// ramp).  Reset to 0 when the segment opens or closes.
    pub segment_len: i32,

    /// Tokens generated in the current DRY span — the structural span bounded by
    /// `<think>` / `</think>` / `<tool_call>` / `</tool_call>`.  Reset at each of
    /// those boundaries and at turn start; drives the kernel's `dry_lens` (DRY's
    /// own look-back window, independent of the think segment).
    pub dry_span_len: i32,

    /// True while this sequence is inside a tool call (stencil active).  DRY is
    /// suppressed (`dry_lens` forced to 0) because the tool-call grammar is
    /// already steered by the stencil.
    pub dry_suppressed: bool,

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
            in_segment: false,
            segment_len: 0,
            dry_span_len: 0,
            dry_suppressed: false,
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

        // Advance the segment length while inside a segment.
        if self.in_segment {
            self.segment_len += 1;
        }

        // Advance the DRY span. It is reset at every structural boundary
        // (`<think>`/`</think>`/`<tool_call>`/`</tool_call>`) and at turn start,
        // so it always counts exactly the current span's generated tokens.
        self.dry_span_len += 1;

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
        self.in_segment = false;
        self.segment_len = 0;
        self.dry_span_len = 0;
        self.dry_suppressed = false;
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
        // A new turn starts a fresh DRY span; any tool-call suppression from the
        // prior turn is cleared.
        self.dry_span_len = 0;
        self.dry_suppressed = false;
    }

    /// Advance RNG offset (called after each sampling).
    pub fn advance_rng(&mut self) {
        self.rng_offset = self.rng_offset.wrapping_add(1);
    }

    /// Open a segment (the caller signals this when the segment-open token is
    /// sampled), restarting the per-segment length.  The `<think>` boundary also
    /// starts a fresh DRY span.
    pub fn enter_segment(&mut self) {
        self.in_segment = true;
        self.segment_len = 0;
        self.dry_span_len = 0;
    }

    /// Close the segment (the caller signals this when the segment-close token is
    /// sampled).  The `</think>` boundary starts a fresh DRY span for the prose.
    pub fn exit_segment(&mut self) {
        self.in_segment = false;
        self.segment_len = 0;
        self.dry_span_len = 0;
    }

    /// Enter a tool call (the caller signals this when the `<tool_call>` trigger
    /// fires and the stencil starts driving).  DRY is suppressed for the duration
    /// because the grammar is already steered; the span is reset so prose after
    /// the tool call does not see the tool-call tokens.
    pub fn enter_tool_call(&mut self) {
        self.dry_suppressed = true;
        self.dry_span_len = 0;
    }

    /// Exit a tool call (the caller signals this when the stencil driver
    /// completes, `</tool_call>`).  DRY resumes over a fresh prose span.
    pub fn exit_tool_call(&mut self) {
        self.dry_suppressed = false;
        self.dry_span_len = 0;
    }

    /// Advance the segment state for a sampled token: open it on `segment_open_id`,
    /// close it on `segment_close_id`.  The sampler is told *which* token ids
    /// delimit the segment — it has no notion of what the segment means.
    pub fn update_segment_state(
        &mut self,
        token: u32,
        segment_open_id: i32,
        segment_close_id: i32,
    ) {
        if segment_open_id < 0 || segment_close_id < 0 {
            return; // Segment tracking not configured.
        }
        if token as i32 == segment_open_id {
            self.enter_segment();
        } else if token as i32 == segment_close_id && self.in_segment {
            self.exit_segment();
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
        let logits2d = self.flatten_to_2d(logits)?;
        let mut results = vec![0u32; batch_size];

        // Split rows by stencil constraint.  Constrained rows take cheap CPU paths
        // — a forced token (allow-list of one) needs no logits, a small allow-list
        // is a tiny gather + sample — so only UNCONSTRAINED rows go to the
        // full-vocab device kernel.  When nothing is constrained (the common wave)
        // every row is a kernel row.
        let mut kernel_idx: Vec<usize> = Vec::new();
        let mut kernel_states: Vec<&mut SequenceSamplingState> = Vec::new();
        let mut kernel_configs: Vec<&SamplingConfig> = Vec::new();
        for (i, slot) in states.iter_mut().enumerate() {
            let state: &mut SequenceSamplingState = slot;
            let config = configs[i];
            match config.stencil.as_slice() {
                // Forced: the single allowed token, decided without logits.
                [forced] => {
                    let token = *forced as u32;
                    state.record_token(token, self.max_recent_len);
                    state.advance_rng();
                    results[i] = token;
                }
                // Small allow-list: a tiny gather + sample, CPU-side.
                [_, _, ..] => results[i] = self.sample_allow_list(&logits2d, i, config, state)?,
                // Unconstrained: defer to the device kernel below.
                [] => {
                    kernel_idx.push(i);
                    kernel_states.push(state);
                    kernel_configs.push(config);
                }
            }
        }

        if !kernel_idx.is_empty() {
            // Gather just the kernel rows — unless they ARE the whole batch, in
            // which case skip the copy and run the kernel over every row.
            let kernel_logits = if kernel_idx.len() == batch_size {
                logits2d.clone()
            } else {
                let idx = Tensor::from_vec(
                    kernel_idx.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                    kernel_idx.len(),
                    &self.device,
                )?;
                logits2d.index_select(&idx, 0)?
            };
            let tokens =
                self.sample_full_vocab(&kernel_logits, &mut kernel_states, &kernel_configs)?;
            for (k, &i) in kernel_idx.iter().enumerate() {
                results[i] = tokens[k];
            }
        }

        Ok(results)
    }

    /// Flatten logits of rank 1/2/3 to `[batch, vocab]` (taking the last
    /// position for rank-3).
    fn flatten_to_2d(&self, logits: &Tensor) -> candle::Result<Tensor> {
        match logits.dims().len() {
            1 => logits.unsqueeze(0),
            2 => Ok(logits.clone()),
            3 => {
                let seq_len = logits.dim(1)?;
                logits.i((.., seq_len - 1, ..))
            }
            n => Err(candle::Error::Msg(format!("unexpected logits rank: {n}"))),
        }
    }

    /// Dispatch the unconstrained (full-vocab) rows to the device sampler.
    fn sample_full_vocab(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        if matches!(self.device, Device::Cuda(_)) {
            self.sample_batch_cuda(logits, states, configs)
        } else {
            self.sample_batch_cpu(logits, states, configs)
        }
    }

    /// Sample one row constrained to its stencil allow-list: gather just the
    /// allowed logits (a handful) and sample among them with the row's strategy.
    /// `O(allow-list)`, never the full vocab, and CPU-side regardless of device.
    fn sample_allow_list(
        &self,
        logits2d: &Tensor,
        row: usize,
        config: &SamplingConfig,
        state: &mut SequenceSamplingState,
    ) -> candle::Result<u32> {
        use candle_transformers::generation::LogitsProcessor;
        let allow: Vec<u32> = config.stencil.iter().map(|&t| t as u32).collect();
        let idx = Tensor::from_vec(allow.clone(), allow.len(), logits2d.device())?;
        // Gather the allowed logits (small download), then sample over just them.
        let gathered = logits2d.i(row)?.index_select(&idx, 0)?;
        let gathered = apply_banned_local(&gathered, &allow, config)?;
        let seed = config.seed.wrapping_add(state.rng_offset);
        let mut processor = LogitsProcessor::from_sampling(seed, config_to_sampling(config));
        let local = processor.sample(&gathered)? as usize;
        let token = allow[local];
        state.record_token(token, self.max_recent_len);
        state.advance_rng();
        Ok(token)
    }

    /// CPU fallback implementation using candle's built-in sampling.  Receives
    /// only unconstrained (full-vocab) rows; stencil rows are resolved by
    /// `sample_batch` before this is called.
    fn sample_batch_cpu(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        use candle_transformers::generation::LogitsProcessor;

        let batch_size = states.len();
        let mut results = Vec::with_capacity(batch_size);

        for (i, (state, &config)) in states.iter_mut().zip(configs.iter()).enumerate() {
            // Extract logits for this sequence
            let seq_logits = if logits.dims().len() == 2 {
                logits.i(i)?
            } else {
                logits.clone()
            };

            // Apply this row's banned tokens (a small deny-list, e.g. a few EOS
            // ids) by setting just those values to `-inf`.  Cheap on CPU — only the
            // banned values change (apply_banned copies the row to host F32 to do
            // it); a no-op when the list is empty.
            let seq_logits = apply_banned(&seq_logits, config)?;

            // Token suppression: while inside a segment, subtract the per-turn
            // penalty from each suppress-token logit. Mirrors the kernel's
            // in-segment gate, so tokens outside the segment are never touched.
            let seq_logits = if state.in_segment
                && config.segment_suppress_penalty != 0.0
                && !config.segment_suppress_tokens.is_empty()
            {
                apply_suppression(&seq_logits, config)?
            } else {
                seq_logits
            };

            // In-segment steering: while this sequence is inside a segment,
            // sample a touch hotter (temperature + segment_temp_boost).
            // Mirrors the kernel's per-seq gate so tokens outside the segment
            // stay at the base temperature.  DRY is GPU-only — the CPU
            // LogitsProcessor has no DRY path, so there is nothing to gate here for it.
            let sampling = if state.in_segment && config.segment_temp_boost != 0.0 {
                let mut boosted = config.clone();
                boosted.temperature += config.segment_temp_boost;
                config_to_sampling(&boosted)
            } else {
                config_to_sampling(config)
            };
            let seed = config.seed.wrapping_add(state.rng_offset);
            let mut processor = LogitsProcessor::from_sampling(seed, sampling);
            let mut token = processor.sample(&seq_logits)?;

            // Segment-close overrides: force the close token when the segment budget is exhausted.
            if config.segment_close_token_id >= 0 && state.in_segment {
                let should_force = (config.graceful_segment_close_after > 0
                    && state.segment_len >= config.graceful_segment_close_after
                    && state
                        .recent_tokens
                        .last()
                        .map(|&t| config.sentence_end_token_ids.contains(&t))
                        .unwrap_or(false))
                    || (config.force_segment_close_after > 0
                        && state.segment_len >= config.force_segment_close_after);

                if should_force {
                    token = config.segment_close_token_id as u32;
                }
            }

            // EOS failsafe overrides (post-sampler)
            let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);
            if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
                // Hard stop: unconditionally force EOS regardless of sentence position.
                token = eos_token_id;
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row = i,
                    current_len = state.current_len,
                    forced_eos_after = config.forced_eos_after,
                    "hard EOS forced (length cap)",
                );
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
                    tracing::debug!(
                        target: "candle_conversation::eos",
                        row = i,
                        current_len = state.current_len,
                        graceful_eos_after = config.graceful_eos_after,
                        "soft EOS forced (sentence boundary)",
                    );
                }
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && config.sentence_end_token_ids.is_empty()
            {
                // No sentence-end tokens resolved (e.g. model loaded without tokenizer
                // resolution): fall back to hard stop at the graceful threshold.
                token = eos_token_id;
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row = i,
                    current_len = state.current_len,
                    graceful_eos_after = config.graceful_eos_after,
                    "hard EOS forced (no sentence-end tokens)",
                );
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

        // This path only ever receives unconstrained (full-vocab) rows —
        // stencil-constrained rows are resolved by `sample_batch` before the
        // kernel and never reach here.  Scalar params still come from the first
        // config for the whole sub-batch (the kernel's shared-config behavior).
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

        // No stencil here — constrained rows were resolved before the kernel.
        let stencil: &[i32] = &[];
        let stencil_size = 0i32;

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

        // Compute segment-close params
        // Only active when segment_close_boost > 0, segment_close_token_id >= 0, and at least one sequence is inside a segment
        let segment_lens: Vec<i32> = states
            .iter()
            .map(|s| if s.in_segment { s.segment_len } else { 0 })
            .collect();
        // DRY span lengths (the kernel's `dry_lens`): the current structural
        // span's generated-token count, or 0 while suppressed inside a tool call.
        // This gates and scopes DRY independently of the think segment.
        let dry_lens: Vec<i32> = states
            .iter()
            .map(|s| if s.dry_suppressed { 0 } else { s.dry_span_len })
            .collect();
        // Token suppression (the in-segment ceiling lever).
        // The token list is shared across the batch (config[0]); the penalty is
        // per-sequence (large = HARD ban, moderate = SOFT, 0.0 = off). Activate
        // only when the list is non-empty AND at least one sequence has a nonzero
        // penalty — otherwise pass null/0 so the kernel skips it entirely.
        let suppress_penalties: Vec<f32> =
            configs.iter().map(|c| c.segment_suppress_penalty).collect();
        let suppress_tokens: Vec<i32> = config.segment_suppress_tokens.clone();
        let suppress_active =
            !suppress_tokens.is_empty() && suppress_penalties.iter().any(|&p| p != 0.0);

        let segment_close_active =
            config.segment_close_boost != 0.0 && config.segment_close_token_id >= 0;
        let (
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
        ) = if segment_close_active {
            (
                config.segment_close_boost,
                config.segment_close_token_id,
                config.segment_close_ramp_start,
                config.segment_close_ramp_len,
                config.segment_close_max_multiplier,
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
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
            &segment_lens,
            &dry_lens,
            config.segment_temp_boost,
            &suppress_tokens,
            &suppress_penalties,
            suppress_active,
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

            // One-shot: the dynamic EOS boost ramp begins as `current_len` reaches
            // `eos_ramp_start` (it increments by one, so this fires exactly once per
            // turn).  After this point EOS pressure builds toward the graceful/hard
            // caps below.
            if config.eos_boost != 0.0 && state.current_len == config.eos_ramp_start {
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row = i,
                    current_len = state.current_len,
                    eos_ramp_start = config.eos_ramp_start,
                    eos_ramp_len = config.eos_ramp_len,
                    graceful_eos_after = config.graceful_eos_after,
                    forced_eos_after = config.forced_eos_after,
                    "EOS boost ramp entered",
                );
            }

            // Segment-close overrides: force the close token when the segment budget is exhausted.
            if config.segment_close_token_id >= 0 && state.in_segment {
                let should_force = (config.graceful_segment_close_after > 0
                    && state.segment_len >= config.graceful_segment_close_after
                    && state
                        .recent_tokens
                        .last()
                        .map(|&t| config.sentence_end_token_ids.contains(&t))
                        .unwrap_or(false))
                    || (config.force_segment_close_after > 0
                        && state.segment_len >= config.force_segment_close_after);

                if should_force {
                    token = config.segment_close_token_id as u32;
                }
            }

            // EOS failsafe overrides (post-sampler)
            if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
                // Hard stop: unconditionally force EOS regardless of sentence position.
                token = eos_token_id;
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row = i,
                    current_len = state.current_len,
                    forced_eos_after = config.forced_eos_after,
                    "hard EOS forced (length cap)",
                );
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
                    tracing::debug!(
                        target: "candle_conversation::eos",
                        row = i,
                        current_len = state.current_len,
                        graceful_eos_after = config.graceful_eos_after,
                        "soft EOS forced (sentence boundary)",
                    );
                }
            } else if config.graceful_eos_after > 0
                && state.current_len >= config.graceful_eos_after
                && config.sentence_end_token_ids.is_empty()
            {
                // No sentence-end tokens resolved: fall back to hard stop.
                token = eos_token_id;
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row = i,
                    current_len = state.current_len,
                    graceful_eos_after = config.graceful_eos_after,
                    "hard EOS forced (no sentence-end tokens)",
                );
            }
            output_tokens[i] = token;

            state.record_token(token, self.max_recent_len);
            state.rng_offset = rng_offsets[i];
            // Detect segment open/close transitions for the segment-close boost.
            state.update_segment_state(
                token,
                config.segment_open_token_id,
                config.segment_close_token_id,
            );
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
        segment_close_boost: f32,
        segment_close_token_id: i32,
        segment_close_ramp_start: i32,
        segment_close_ramp_len: i32,
        segment_close_max_multiplier: f32,
        segment_lens: &[i32],
        dry_lens: &[i32],
        segment_temp_boost: f32,
        suppress_tokens: &[i32],
        suppress_penalties: &[f32],
        suppress_active: bool,
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

        let segment_lens_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(segment_lens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload segment_lens: {}", e)))?;

        let dry_lens_gpu: cudarc::driver::CudaSlice<i32> = stream
            .memcpy_stod(dry_lens)
            .map_err(|e| candle::Error::Msg(format!("failed to upload dry_lens: {}", e)))?;

        let banned_gpu: cudarc::driver::CudaSlice<i32> = if banned_tokens.is_empty() {
            stream
                .memcpy_stod(&[-1i32])
                .map_err(|e| candle::Error::Msg(format!("failed to upload banned: {}", e)))?
        } else {
            stream
                .memcpy_stod(banned_tokens)
                .map_err(|e| candle::Error::Msg(format!("failed to upload banned: {}", e)))?
        };

        // Token suppression buffers. Always upload non-empty slices
        // (cudarc rejects zero-length copies); the kernel pointers are nulled out
        // below when suppression is inactive so these uploads are never read.
        let suppress_tokens_gpu: cudarc::driver::CudaSlice<i32> = if suppress_tokens.is_empty() {
            stream.memcpy_stod(&[-1i32]).map_err(|e| {
                candle::Error::Msg(format!("failed to upload suppress_tokens: {}", e))
            })?
        } else {
            stream.memcpy_stod(suppress_tokens).map_err(|e| {
                candle::Error::Msg(format!("failed to upload suppress_tokens: {}", e))
            })?
        };

        let suppress_penalties_gpu: cudarc::driver::CudaSlice<f32> =
            stream.memcpy_stod(suppress_penalties).map_err(|e| {
                candle::Error::Msg(format!("failed to upload suppress_penalties: {}", e))
            })?;

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
            let (segment_lens_ptr, _g3b) = segment_lens_gpu.device_ptr(&stream);
            let (dry_lens_ptr, _g3b2) = dry_lens_gpu.device_ptr(&stream);
            let (suppress_tok_ptr, _g3c) = suppress_tokens_gpu.device_ptr(&stream);
            let (suppress_pen_ptr, _g3d) = suppress_penalties_gpu.device_ptr(&stream);
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
                    segment_close_boost,
                    segment_close_token_id,
                    segment_close_ramp_start,
                    segment_close_ramp_len,
                    segment_close_max_multiplier,
                    segment_lens_ptr as *const i32,
                    dry_lens_ptr as *const i32,
                    segment_temp_boost,
                    if suppress_active {
                        suppress_tok_ptr as *const i32
                    } else {
                        std::ptr::null()
                    },
                    if suppress_active {
                        suppress_tokens.len() as i32
                    } else {
                        0
                    },
                    if suppress_active {
                        suppress_pen_ptr as *const f32
                    } else {
                        std::ptr::null()
                    },
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
                    let logits_ptr = (ptr + (start_offset as u64 * 4)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::F16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
                    call_kernel(logits_ptr);
                }
                CudaStorageSlice::BF16(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    let logits_ptr = (ptr + (start_offset as u64 * 2)) as *const std::ffi::c_void;
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
/// Set this row's banned (deny-list) logits to `-inf`.  Modifies only the few
/// banned *values* (on an F32 host copy of the row), and is a no-op when the list
/// is empty, so unconstrained rows keep their exact logits.  Preserves the input
/// dtype.
fn apply_banned(logits: &Tensor, config: &SamplingConfig) -> candle::Result<Tensor> {
    if config.banned_tokens.is_empty() {
        return Ok(logits.clone());
    }
    let dtype = logits.dtype();
    let dims = logits.dims().to_vec();
    let mut v: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for &b in &config.banned_tokens {
        ban(&mut v, b as u32);
    }
    Tensor::from_vec(v, dims, logits.device())?.to_dtype(dtype)
}

/// Subtract the suppression penalty from each `segment_suppress_tokens` logit.
/// The CPU mirror of the kernel's in-segment ceiling lever; the caller has
/// already confirmed the sequence is inside a segment and the penalty is
/// nonzero, so this is applied unconditionally here.
fn apply_suppression(logits: &Tensor, config: &SamplingConfig) -> candle::Result<Tensor> {
    let dtype = logits.dtype();
    let dims = logits.dims().to_vec();
    let vocab = logits.elem_count();
    let mut v: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for &t in &config.segment_suppress_tokens {
        if t >= 0 && (t as usize) < vocab {
            v[t as usize] -= config.segment_suppress_penalty;
        }
    }
    Tensor::from_vec(v, dims, logits.device())?.to_dtype(dtype)
}

/// Apply banned tokens within a gathered allow-list logit vector: any banned
/// token that also appears in `allow` is set to `-inf` at its local position.
/// A no-op when no banned token intersects the allow-list.
fn apply_banned_local(
    gathered: &Tensor,
    allow: &[u32],
    config: &SamplingConfig,
) -> candle::Result<Tensor> {
    if config.banned_tokens.is_empty() {
        return Ok(gathered.clone());
    }
    let banned: std::collections::HashSet<u32> =
        config.banned_tokens.iter().map(|&b| b as u32).collect();
    if !allow.iter().any(|t| banned.contains(t)) {
        return Ok(gathered.clone());
    }
    let dtype = gathered.dtype();
    let dims = gathered.dims().to_vec();
    let mut v: Vec<f32> = gathered.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for (local, t) in allow.iter().enumerate() {
        if banned.contains(t) {
            v[local] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, dims, gathered.device())?.to_dtype(dtype)
}

/// Map a [`SamplingConfig`] to candle's `Sampling` strategy.
fn config_to_sampling(config: &SamplingConfig) -> candle_transformers::generation::Sampling {
    use candle_transformers::generation::Sampling;
    if config.temperature <= 0.0 {
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
    }
}

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

    // ── Per-row stencil / banned-token masking ─────────────────────────

    /// Build a `[batch, VOCAB]` logits tensor from per-row spikes.
    fn logits_from_rows(rows: &[&[(usize, f32)]]) -> Tensor {
        let mut data = vec![0.0f32; rows.len() * VOCAB_SIZE];
        for (r, spikes) in rows.iter().enumerate() {
            for &(tok, val) in *spikes {
                data[r * VOCAB_SIZE + tok] = val;
            }
        }
        Tensor::from_vec(data, (rows.len(), VOCAB_SIZE), &Device::Cpu).expect("logits")
    }

    #[test]
    fn single_token_stencil_forces_that_token() {
        // Token 90 dominates, but the stencil allows only 33 — it must win.
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_stencil(vec![33]);
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(90, 100.0), (33, -10.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 33, "single-token stencil forces its token");
    }

    #[test]
    fn stencil_picks_best_within_allow_list() {
        // Global best (50) is outside the stencil; best *inside* {10,20} is 20.
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(50, 100.0), (20, 5.0), (10, 1.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 20);
    }

    #[test]
    fn banned_token_excluded() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.banned_tokens = vec![50];
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(50, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 60, "banned best token → next best");
    }

    #[test]
    fn stencil_and_banned_combine() {
        // Stencil {10,20,30}; 30 is best but banned → 20 wins.
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax().with_stencil(vec![10, 20, 30]);
        config.banned_tokens = vec![30];
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(30, 100.0), (20, 50.0), (10, 10.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 20);
    }

    #[test]
    fn empty_stencil_is_unconstrained() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax(); // no stencil, no bans
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(77, 100.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 77);
    }

    #[test]
    fn per_row_stencils_are_independent() {
        // Two stenciled rows with disjoint allow-lists; the shared global best
        // (50) is outside both and must be masked in each.
        let sampler = make_sampler();
        let c0 = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let c1 = SamplingConfig::argmax().with_stencil(vec![30, 40]);
        let mut s0 = make_state();
        let mut s1 = make_state();
        let logits = logits_from_rows(&[
            &[(50, 100.0), (20, 5.0), (10, 1.0)],
            &[(50, 100.0), (30, 7.0), (40, 2.0)],
        ]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut s0, &mut s1], &[&c0, &c1])
            .expect("sample");
        assert_eq!(tokens, vec![20, 30]);
    }

    #[test]
    fn mixed_batch_stencils_only_its_own_row() {
        // THE per-row property the global-config kernel lacks: row 0 is forced
        // to token 10 while row 1 (free) keeps its global best 90 — the stencil
        // must not leak across rows.
        let sampler = make_sampler();
        let stenciled = SamplingConfig::argmax().with_stencil(vec![10]);
        let free = SamplingConfig::argmax();
        let mut s0 = make_state();
        let mut s1 = make_state();
        let logits = logits_from_rows(&[&[(90, 100.0), (10, 1.0)], &[(90, 100.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut s0, &mut s1], &[&stenciled, &free])
            .expect("sample");
        assert_eq!(tokens, vec![10, 90], "stencil applies to row 0 only");
    }

    #[test]
    fn allow_list_gather_stays_in_set_under_temperature() {
        // Stochastic sampling (temperature > 0) over the gathered allow-list must
        // never escape it, across many seeds.
        let sampler = make_sampler();
        let allowed = [13usize, 41, 88];
        let logits = logits_from_rows(&[&[(13, 2.0), (41, 1.0), (88, 1.5), (90, 100.0)]]);
        for seed in 0..200u64 {
            let mut config = SamplingConfig::argmax().with_stencil(vec![13, 41, 88]);
            config.temperature = 1.0;
            config.top_p = 0.99;
            config.seed = seed;
            let mut state = make_state();
            let tokens = sampler
                .sample_batch(&logits, &mut [&mut state], &[&config])
                .expect("sample");
            assert!(
                allowed.contains(&(tokens[0] as usize)),
                "sampled {} escaped the allow-list at seed {seed}",
                tokens[0]
            );
        }
    }

    // ── update_segment_state tests ────────────────────────────────────

    #[test]
    fn test_open_token_enters_segment() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        assert!(!state.in_segment);
        assert_eq!(state.segment_len, 0);

        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert!(
            state.in_segment,
            "should enter the segment on the open token"
        );
        assert_eq!(state.segment_len, 0, "segment_len reset on enter");
    }

    #[test]
    fn test_close_token_exits_segment() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        // Open the segment
        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert!(state.in_segment);

        // Generate some tokens inside the segment
        state.record_token(42, MAX_RECENT);
        state.record_token(43, MAX_RECENT);
        assert_eq!(state.segment_len, 2);

        // Close the segment
        state.update_segment_state(seg_close as u32, seg_open, seg_close);
        assert!(
            !state.in_segment,
            "should exit the segment on the close token"
        );
        assert_eq!(state.segment_len, 0, "segment_len reset on exit");
    }

    #[test]
    fn test_close_token_without_open_segment_is_noop() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        // The close token outside a segment should be a no-op
        state.update_segment_state(seg_close as u32, seg_open, seg_close);
        assert!(!state.in_segment, "should remain outside a segment");
    }

    #[test]
    fn test_segment_tracking_disabled_when_ids_negative() {
        let mut state = make_state();

        // -1 means not configured
        state.update_segment_state(10, -1, 11);
        assert!(
            !state.in_segment,
            "should not enter a segment when segment_open_id < 0"
        );

        state.update_segment_state(10, 10, -1);
        assert!(
            !state.in_segment,
            "should not enter a segment when segment_close_id < 0"
        );
    }

    #[test]
    fn test_segment_len_tracks_tokens() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        state.update_segment_state(seg_open as u32, seg_open, seg_close);

        for i in 0..5 {
            state.record_token(40 + i, MAX_RECENT);
        }
        assert_eq!(state.segment_len, 5);

        // Re-opening the segment resets the counter
        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert_eq!(state.segment_len, 0);
    }
}
