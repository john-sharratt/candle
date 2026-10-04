//! Batched sampling wrapper around the CUDA kernel.
//!
//! Provides a high-level Rust API for the batched sampling kernel with:
//! - GPU buffer management
//! - Automatic dtype dispatching
//!
//! State (token counts, recent history) is owned by the caller (DecodeState).

use crate::banned_rows::banned_buffer;
use crate::config::SamplingConfig;
use crate::line_ends::ends_a_line;
use crate::penalty_counts::{count_table_ptr, PenaltyTables, SparseCounts};
use crate::sampler_args::{split_rng_and_outputs, ArgPack};
use crate::scheduler::profile;
use crate::stencil::ban;
use crate::token_buffer::TokenBuffer;
use candle::cuda_backend::CudaStorageSlice;
use candle::{DType, Device, IndexOp, Tensor};
use candle_kernels::sampling::{run_batched_sampling, DType as KernelDType};
use candle_transformers::generation::{LogitsProcessor, PendingSample};
use candle_transformers::models::speculative_choice::TypicalAcceptance;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::sync::Mutex;

/// A stencil-constrained row's sample with its device work enqueued
/// ([`BatchedSampler::issue_allow_list`]) and not yet read back: the allowed
/// tokens its result indexes, and the row's processor, which holds the RNG the
/// host half draws from.
struct PendingAllowList {
    allow: Vec<u32>,
    processor: LogitsProcessor,
    pending: PendingSample,
}

/// Per-sequence sampling state.
///
/// Tracks token counts and recent history for penalty calculations.
/// Consecutive token-0 emissions that mark a decode as degenerate rather than
/// merely repetitive. Comfortably above anything language produces — token 0 is
/// `!` in the Qwen vocab and no real text repeats it eight times — while short
/// enough that a broken forward is caught in a few steps instead of running to
/// the length cap.
pub const DEGENERATE_TOKEN_RUN: u32 = 8;

/// Per-sequence sampling dials, uploaded as a `[batch_size]` array and read by
/// the kernel at `seq_dials[row]` so every row samples on its own config.
///
/// **The C twin is `batched_sampling::SeqDials` in `batched_sampling.cuh`.** The
/// kernel reads these bytes back as that struct, so the field order and types
/// here must match it exactly — every field is 4 bytes (`f32`/`i32`), packed
/// with no padding, and `#[repr(C)]` keeps the layout. The EOS id, the row
/// stride and the live vocabulary are not here (they are the same across the
/// wave and stay scalar).
///
/// `typical_draft` is the proposal a
/// speculative verify row tests (-1 on any other row), accepted on
/// `typical_eps`/`typical_delta` when the sample did not land on it — see
/// `candle_transformers::models::speculative_choice::TypicalAcceptance`.
///
/// Before this existed the kernel took these dials as scalars from the first
/// row's config and applied them to the whole launch, so a wave that mixed
/// configs — a deliberating turn beside an impulsive one, a narrator beside a
/// reflection — sampled every row at whichever config sorted first, and one
/// row's EOS ramp could cut another's turn off after a single token.
#[repr(C)]
#[derive(Clone, Copy)]
struct SeqDials {
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
    eos_ramp_start: i32,
    eos_ramp_len: i32,
    eos_boost_max_multiplier: f32,
    cross_turn_penalty: f32,
    segment_close_boost: f32,
    segment_close_token_id: i32,
    segment_close_ramp_start: i32,
    segment_close_ramp_len: i32,
    segment_close_max_multiplier: f32,
    segment_temp_boost: f32,
    typical_draft: i32,
    typical_eps: f32,
    typical_delta: f32,
}

// The kernel reads this struct as a flat run of 4-byte words (all fields are
// f32/i32), one per row of the batch, and casts back to the identically-laid-out
// CUDA `SeqDials`. If the size or field count drifts from the CUDA side the
// kernel reads a row at the wrong stride, so pin it: 24 fields × 4 bytes.
const _: () = assert!(std::mem::size_of::<SeqDials>() == 96);

impl SeqDials {
    /// Read one row's dials from its config, resolving the same Option/gate logic
    /// the scalar path applies (DRY defaults, the dynamic-EOS gate, the
    /// segment-close-active gate) so a per-row wave behaves identically to a
    /// uniform one row-for-row. The row tests no draft until
    /// [`Self::with_draft`] gives it one.
    fn from_config(c: &SamplingConfig) -> Self {
        let (dry_multiplier, dry_base, dry_allowed_length, dry_range) = match &c.dry {
            Some(d) => (d.multiplier, d.base, d.allowed_length, d.range),
            None => (0.0, 1.75, 2, 0),
        };
        let (eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier) = if c.dynamic_eos_boost {
            (c.eos_ramp_start, c.eos_ramp_len, c.eos_boost_max_multiplier)
        } else {
            (0, 0, 0.0)
        };
        let (
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
        ) = if c.segment_close_boost != 0.0 && c.segment_close_token_id >= 0 {
            (
                c.segment_close_boost,
                c.segment_close_token_id,
                c.segment_close_ramp_start,
                c.segment_close_ramp_len,
                c.segment_close_max_multiplier,
            )
        } else {
            // Disabled for this row: -1 token id keeps the kernel's per-row
            // segment-close path off even when a co-batched row has it on.
            (0.0, -1, 0, 0, 0.0)
        };
        SeqDials {
            temperature: c.temperature,
            top_k: c.top_k,
            top_p: c.top_p,
            repeat_penalty: c.repeat_penalty,
            frequency_penalty: c.frequency_penalty,
            presence_penalty: c.presence_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_range,
            eos_boost: c.eos_boost,
            eos_ramp_start,
            eos_ramp_len,
            eos_boost_max_multiplier,
            cross_turn_penalty: c.cross_turn_penalty,
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
            segment_temp_boost: c.segment_temp_boost,
            typical_draft: -1,
            typical_eps: 0.0,
            typical_delta: 0.0,
        }
    }

    /// This row is a speculative verify row testing `draft` on `typical`.
    /// A `None` draft leaves the row a plain one.
    fn with_draft(mut self, draft: Option<u32>, typical: TypicalAcceptance) -> Self {
        if let Some(d) = draft {
            self.typical_draft = d as i32;
            self.typical_eps = typical.epsilon;
            self.typical_delta = typical.delta;
        }
        self
    }
}

/// The kernel's candidate cap (`batched_sampling::MAX_TOP_K`): with top-k off
/// it still samples the best this many.
const KERNEL_MAX_TOP_K: usize = 256;
/// The kernel's candidate floor (`radix_select_logit_threshold`'s
/// `DEAD_ZONE`): no candidate sits more than this far below the row's best.
const KERNEL_DEAD_ZONE: f32 = 50.0;

/// The typical-acceptance rule over one row of logits: true when `draft`'s
/// probability exceeds `min(ε, δ·e^(−H))`. The CPU twin of the kernel's
/// `typical_accepts`, measured over the distribution the kernel samples — the
/// top-k candidates (at most [`KERNEL_MAX_TOP_K`], none more than
/// [`KERNEL_DEAD_ZONE`] below the best), softmaxed at `temperature`, cut to
/// the nucleus at `top_p`, renormalised. A draft outside it has probability
/// zero.
fn typical_accepts_row(
    logits: &[f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    draft: u32,
    typical: TypicalAcceptance,
) -> bool {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return false;
    }
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, l)| l >= max - KERNEL_DEAD_ZONE)
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = if top_k > 0 {
        (top_k as usize).min(KERNEL_MAX_TOP_K)
    } else {
        KERNEL_MAX_TOP_K
    };
    candidates.truncate(k);

    let inv_t = 1.0 / temperature;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&(_, l)| ((l - max) * inv_t).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut nucleus = candidates.len();
    if top_p < 1.0 {
        let mut cumsum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cumsum += w / total;
            if cumsum >= top_p {
                nucleus = i + 1;
                break;
            }
        }
    }
    let kept: f32 = weights[..nucleus].iter().sum();
    let mut entropy = 0.0f32;
    let mut p_draft = 0.0f32;
    for (&(id, _), w) in candidates[..nucleus].iter().zip(&weights) {
        let p = w / kept;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        if id == draft as usize {
            p_draft = p;
        }
    }
    p_draft > typical.epsilon.min(typical.delta * (-entropy).exp())
}

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

    /// The turns still inside the cross-turn window, oldest first, each as the
    /// `(token, count)` pairs it added — so a turn can be taken back out of
    /// [`Self::cross_turn_counts`] when it leaves. Empty when the window is
    /// unbounded.
    pub cross_turn_history: Vec<Vec<(u32, i32)>>,

    /// The tokens whose [`Self::token_counts`] entry is nonzero, in the order
    /// they were first sampled this turn — the index a dispatch stamps the
    /// device count table from, so it never walks the vocabulary.
    counted: Vec<u32>,

    /// The tokens whose [`Self::cross_turn_counts`] entry is nonzero.
    cross_counted: Vec<u32>,

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

    /// True while this sequence is inside ANY stencil-steered span (think block
    /// OR tool call).  DRY is suppressed (`dry_lens` forced to 0) because the
    /// grammar is already steered.
    pub dry_suppressed: bool,

    /// True while this sequence is inside a TOOL CALL specifically (not the think
    /// block).  The remaining repetition penalties (repeat/frequency/presence)
    /// are suppressed for these rows: a tool call's arguments legitimately
    /// reproduce prompt content verbatim — the query's numbers, file paths,
    /// identifiers — which those penalties would otherwise demote, corrupting the
    /// value.  Kept distinct from `dry_suppressed` so reasoning (the think block)
    /// retains full repetition control.
    pub in_tool_call: bool,

    /// True while the tool-call stencil is steering this sequence — it is
    /// writing a call — whatever `in_tool_call`'s penalty policy says. Synced
    /// from the stencil each decode step.
    ///
    /// **The turn's length budget does not apply while it is set**: the EOS
    /// boost ramp sees a length of 0 and the graceful and forced EOS failsafes
    /// stand down. Those budgets size a prose answer, and inside a call an EOS
    /// is not an ending — the stencil intercepts it and closes the value where
    /// it stands. A `write` whose content outran the answer budget came out as
    /// a file cut mid-sentence. The call is bounded by its own grammar instead
    /// (each value's `forced_after`), and the turn by `max_tokens`.
    pub writing_call: bool,

    /// Next index into [`crate::SamplingConfig::segment_close_script`] while the
    /// hard-cap closer script is playing; `None` when no script is in flight.
    /// The script overrides sampling until every phrase token has played, then
    /// the sampler emits the segment-close token itself and clears this.
    pub close_script_pos: Option<usize>,

    /// True while the active steering span retires into further decoding rather
    /// than into the close — a forced close here is dropped by the stencil and
    /// more content follows, so the hard-cap closer script must NOT play (it is
    /// a terminal closing statement). Synced from the stencil each decode step;
    /// false for unsteered blocks and terminal spans.
    pub close_would_continue: bool,

    /// Consecutive emissions of token id 0 this turn. Degenerate logits — all
    /// equal, or non-finite — make argmax return index 0, which every vocab this
    /// engine runs maps to a printable character (`!` in Qwen), so the failure
    /// looks like output rather than an error. A run of them is the signature of
    /// a broken forward, not of language; [`DEGENERATE_TOKEN_RUN`] bounds it.
    pub degenerate_run: u32,

    /// Current RNG offset (for deterministic sampling across calls).
    pub rng_offset: u64,
}

impl SequenceSamplingState {
    /// Create new state for a sequence.
    pub fn new(vocab_size: usize, max_recent_len: usize) -> Self {
        Self {
            token_counts: vec![0; vocab_size],
            cross_turn_counts: vec![0; vocab_size],
            cross_turn_history: Vec::new(),
            counted: Vec::new(),
            cross_counted: Vec::new(),
            recent_tokens: Vec::with_capacity(max_recent_len),
            current_len: 0,
            in_segment: false,
            segment_len: 0,
            dry_span_len: 0,
            dry_suppressed: false,
            in_tool_call: false,
            writing_call: false,
            close_script_pos: None,
            close_would_continue: false,
            degenerate_run: 0,
            rng_offset: 0,
        }
    }

    /// Record a generated token.
    pub fn record_token(&mut self, token: u32, max_recent_len: usize) {
        let token_idx = token as usize;
        if token_idx < self.token_counts.len() {
            if self.token_counts[token_idx] == 0 {
                self.counted.push(token);
            }
            self.token_counts[token_idx] += 1;
        }

        self.recent_tokens.push(token as i32);
        self.current_len += 1;
        // Token 0 is what argmax yields from an all-equal or non-finite logit
        // row, so a run of it means the forward produced nothing usable.
        if token == 0 {
            self.degenerate_run += 1;
        } else {
            self.degenerate_run = 0;
        }

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
        self.cross_turn_history.clear();
        self.counted.clear();
        self.cross_counted.clear();
        self.recent_tokens.clear();
        self.current_len = 0;
        self.in_segment = false;
        self.segment_len = 0;
        self.dry_span_len = 0;
        self.dry_suppressed = false;
        self.in_tool_call = false;
        self.writing_call = false;
        self.close_script_pos = None;
        self.close_would_continue = false;
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
    pub fn end_turn(&mut self, cross_turn_window: usize) {
        // Fold this turn into the cross-turn counts, and — when the window is
        // finite — take back out the turn that falls off the far end of it.
        // Each turn is kept as its own `(token, count)` pairs rather than a
        // dense copy, so the history costs what the turns said, not a
        // vocabulary per turn.
        //
        // A turn that sampled nothing is not recorded: this also runs at a
        // conversation's first decode, with nothing said yet, and letting that
        // take a slot would make a window of one forget the only turn it had.
        let mut turn: Vec<(u32, i32)> = self
            .counted
            .iter()
            .map(|&t| (t, self.token_counts[t as usize]))
            .collect();
        turn.sort_unstable_by_key(|&(t, _)| t);
        for &(t, c) in &turn {
            let cross = &mut self.cross_turn_counts[t as usize];
            if *cross == 0 {
                self.cross_counted.push(t);
            }
            *cross = cross.saturating_add(c);
        }
        if cross_turn_window > 0 && !turn.is_empty() {
            self.cross_turn_history.push(turn);
            while self.cross_turn_history.len() > cross_turn_window {
                for (t, c) in self.cross_turn_history.remove(0) {
                    let cross = &mut self.cross_turn_counts[t as usize];
                    *cross = cross.saturating_sub(c).max(0);
                }
            }
            let cross = &self.cross_turn_counts;
            self.cross_counted.retain(|&t| cross[t as usize] > 0);
        }
        // Reset per-turn state (frequency/presence penalties are per-turn)
        self.token_counts.fill(0);
        self.counted.clear();
        self.current_len = 0;
        // A new turn starts a fresh DRY span; any tool-call suppression, open
        // segment, or in-flight closer script from the prior turn is cleared.
        self.dry_span_len = 0;
        self.dry_suppressed = false;
        self.writing_call = false;
        self.in_segment = false;
        self.segment_len = 0;
        self.close_script_pos = None;
        self.close_would_continue = false;
        self.degenerate_run = 0;
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
        // A fresh segment cannot inherit a closer script from a previous one.
        self.close_script_pos = None;
    }

    /// Close the segment (the caller signals this when the segment-close token is
    /// sampled).  The `</think>` boundary starts a fresh DRY span for the prose.
    pub fn exit_segment(&mut self) {
        self.in_segment = false;
        self.segment_len = 0;
        self.dry_span_len = 0;
        // The segment is closed; any in-flight closer script is finished or moot.
        self.close_script_pos = None;
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

    /// True when the most recent token ends a sentence (`.`, `!`, `?`) or a
    /// line. The graceful EOS waits for this, so an answer is not cut
    /// mid-sentence.
    fn at_sentence_end(&self, config: &SamplingConfig) -> bool {
        self.recent_tokens.last().is_some_and(|&t| {
            config.sentence_end_token_ids.contains(&t) || ends_a_line(&config.line_end_token_ids, t)
        })
    }

    /// True when the most recent token ends a line. The segment closes wait
    /// for this — see [`SamplingConfig::line_end_token_ids`] for why a line
    /// rather than a sentence.
    fn at_line_end(&self, config: &SamplingConfig) -> bool {
        self.recent_tokens
            .last()
            .is_some_and(|&t| ends_a_line(&config.line_end_token_ids, t))
    }
}

/// Segment-close override for one sampled token, applied after sampling on
/// both the CPU and GPU paths.
///
/// Three tiers:
/// - a closer script in flight overrides everything: it plays the configured
///   phrase to its end, then emits the segment-close token itself (the close
///   is appended by this function, not stored in the script, so a played
///   script can never fail to close the segment);
/// - the GRACEFUL cap closes with the bare token at the end of a line — no
///   rescue needed;
/// - the HARD cap starts the configured closer script (a canned
///   self-interruption that turns the mid-sentence amputation into sensible
///   prose and primes the answer with an explicit commitment). It falls back
///   to the bare close token when no script is configured, when the sentence
///   happens to already be complete, or when the steering span would drop the
///   close and carry on decoding — a terminal closing statement does not belong
///   in the middle of a span that continues.
///
/// When this returns `Some`, the token is authoritative for the step: the EOS
/// failsafes must not replace it (they fire on a later step, once the segment
/// is closed and `in_segment` is false).
/// Summarise one `[batch, vocab]` logits row for the degenerate-decode report.
///
/// Names what the row actually is — how many entries are non-finite, its
/// min/max, and whether every entry is identical — so the fault is diagnosable
/// from the log alone. An all-equal row means the forward wrote nothing
/// (argmax lands on index 0, which is `!` in the Qwen vocabularies); a
/// non-finite row means the arithmetic blew up. Those have different causes and
/// the guard cannot distinguish them without looking.
fn describe_logit_row(logits2d: &Tensor, row: usize) -> candle::Result<String> {
    let values = logits2d.i(row)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = values.len();
    let non_finite = values.iter().filter(|v| !v.is_finite()).count();
    let finite_min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let first = values.first().copied().unwrap_or(0.0);
    let all_equal = values.iter().all(|v| *v == first);
    // The argmax names what the row actually wanted. A sampled token that is
    // not this one, at a logit far below it, is the sampler disagreeing with
    // the distribution rather than the forward producing a bad row — and those
    // two faults are indistinguishable from the emitted text alone.
    let argmax = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(format!(
        "logits row: vocab={n} non_finite={non_finite} finite_min={finite_min:.6} \
         finite_max={finite_max:.6} first={first:.6} argmax={argmax} all_equal={all_equal}"
    ))
}

fn segment_close_override(
    config: &SamplingConfig,
    state: &mut SequenceSamplingState,
) -> Option<u32> {
    if config.segment_close_token_id < 0 || !state.in_segment {
        return None;
    }
    if let Some(pos) = state.close_script_pos {
        return Some(if pos < config.segment_close_script.len() {
            state.close_script_pos = Some(pos + 1);
            config.segment_close_script[pos]
        } else {
            state.close_script_pos = None;
            config.segment_close_token_id as u32
        });
    }
    let at_line_end = state.at_line_end(config);
    if config.graceful_segment_close_after > 0
        && state.segment_len >= config.graceful_segment_close_after
        && at_line_end
    {
        return Some(config.segment_close_token_id as u32);
    }
    if config.force_segment_close_after > 0 && state.segment_len >= config.force_segment_close_after
    {
        return Some(
            if config.segment_close_script.is_empty() || at_line_end || state.close_would_continue {
                config.segment_close_token_id as u32
            } else {
                state.close_script_pos = Some(1);
                config.segment_close_script[0]
            },
        );
    }
    None
}

/// A tool call opened inside an open segment closes the segment first: the
/// sampled `<tool_call>` is committed as `</think>`.
///
/// The template closes the reasoning block before the answer, and a call is an
/// answer. Written inside the block it is reasoning to every reader downstream,
/// so the client never receives it. Measured on a Cline turn: the block opened,
/// the call was written inside it, and the reply ended with the block still
/// open. Committing the close where the model chose to act ends the block
/// there; the call follows on the next step, outside it, where the tool-call
/// grammar takes it.
fn close_before_call(
    config: &SamplingConfig,
    state: &SequenceSamplingState,
    sampled: u32,
) -> Option<u32> {
    (config.tool_call_open_token_id >= 0
        && config.segment_close_token_id >= 0
        && state.in_segment
        && sampled == config.tool_call_open_token_id as u32)
        .then_some(config.segment_close_token_id as u32)
}

/// Stateless batched sampler that invokes the CUDA kernel.
///
/// This sampler does not own per-sequence state. Instead, callers pass
/// `SequenceSamplingState` references which are updated in place.
pub struct BatchedSampler {
    /// Device the sampler operates on.
    #[allow(dead_code)]
    device: Device,

    /// Vocabulary size — the logits row width, which a checkpoint may pad.
    vocab_size: usize,

    /// Tokens a row may produce: the tokenizer's last id + 1. The padded tail
    /// of each row past it carries no probability.
    live_vocab: usize,

    /// Maximum recent token history length.
    max_recent_len: usize,

    /// EOS token ID.
    eos_tokens: TokenBuffer,

    /// Optional path to write penalty state during decoding.
    penalty_log_path: Option<std::path::PathBuf>,

    /// The kernel's count tables, kept on the device between dispatches.
    penalty_tables: Mutex<PenaltyTables>,
}

impl BatchedSampler {
    /// Create a new batched sampler. `live_vocab` is clamped to `vocab_size`.
    pub fn new(
        device: Device,
        vocab_size: usize,
        live_vocab: usize,
        max_recent_len: usize,
        eos_tokens: TokenBuffer,
        penalty_log_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            device,
            vocab_size,
            live_vocab: live_vocab.min(vocab_size),
            max_recent_len,
            eos_tokens,
            penalty_log_path,
            penalty_tables: Mutex::new(PenaltyTables::new(vocab_size)),
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
        self.sample_rows(logits, states, configs, None)
    }

    /// [`Self::sample_batch`] for speculative verify rows: `drafts[i]` is the
    /// proposal row `i` tests (`None` on a bonus row), committed instead of the
    /// row's sample when `typical` accepts it. A row under a stencil samples
    /// its allow-list exactly as a plain row does and tests nothing.
    pub fn sample_verify_rows(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
        drafts: &[Option<u32>],
        typical: TypicalAcceptance,
    ) -> candle::Result<Vec<u32>> {
        if drafts.len() != states.len() {
            candle::bail!(
                "sample_verify_rows: {} drafts for {} rows",
                drafts.len(),
                states.len()
            );
        }
        self.sample_rows(logits, states, configs, Some((drafts, typical)))
    }

    fn sample_rows(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
        verify: Option<(&[Option<u32>], TypicalAcceptance)>,
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
        // Small allow-lists: every such row's device work is enqueued in this
        // pass and read back after the kernel rows below, so the rows share one
        // pipeline drain instead of each draining it for its own handful of
        // logits.
        let mut allow_rows: Vec<(usize, PendingAllowList, &mut SequenceSamplingState)> = Vec::new();
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
                // Small allow-list: a tiny gather + sample over just the allowed
                // logits.
                [_, _, ..] => {
                    let pending = self.issue_allow_list(&logits2d, i, config, state)?;
                    allow_rows.push((i, pending, state));
                }
                // Unconstrained: defer to the device kernel below. Collected in
                // this same pass — a second walk filtering on `kernel_idx` would
                // re-scan it per row, on a path that runs once per decode step.
                [] => {
                    kernel_idx.push(i);
                    kernel_states.push(state);
                    kernel_configs.push(config);
                }
            }
        }

        // **Each row samples on its OWN dials.** Temperature, top-k/top-p, the
        // repetition and DRY penalties, the EOS ramp and the segment-close ramp
        // are packed per row into the `SeqDials` array (`SeqDials::from_config`
        // below) and the kernel reads `seq_dials[row]`, so a wave that mixes
        // dials — a `ThinkMode::Off` ingest summary at
        // `SamplingConfig::compression()` beside a dialogue turn, or a narrator
        // beside a deliberating reflection — samples each row correctly instead
        // of collapsing the whole launch onto row 0's config. This is the
        // per-sequence-array shape `banned_tokens_per_seq` already uses, not
        // host-side regrouping (splitting into one launch per distinct config
        // would be ~one launch per sequence, since `zend` randomises `seed` per
        // turn — the batching the engine exists to do, thrown away).
        //
        // The remaining scalars — `eos_token_id`, `vocab_size`, the shared
        // banned/suppress token *lists* — are genuinely model-wide and stay
        // scalar. And anything resolved per row on the HOST still reads
        // `configs[i]` directly (the segment-close budget and EOS failsafes in
        // the post-kernel loop), never row 0.
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
            let kernel_verify = verify.map(|(drafts, typical)| {
                let rows: Vec<Option<u32>> = kernel_idx.iter().map(|&i| drafts[i]).collect();
                (rows, typical)
            });
            let tokens = self.sample_full_vocab(
                &kernel_logits,
                &mut kernel_states,
                &kernel_configs,
                kernel_verify.as_ref().map(|(r, t)| (r.as_slice(), *t)),
            )?;
            for (k, &i) in kernel_idx.iter().enumerate() {
                results[i] = tokens[k];
            }
        }
        for (i, pending, state) in allow_rows {
            results[i] = self.finish_allow_list(pending, state)?;
        }

        // A row that has just crossed the degenerate bar gets its logits
        // described, once, in the log. `resolve_final_token` can only say the
        // row was unusable — it never sees the logits — so without this the
        // operator is left with the guard's own guess ("all-equal or
        // non-finite") and no way to tell a dead forward from a NaN one. This
        // reads one row off the device and runs only on the step the bar is
        // crossed, so it costs nothing until something is already wrong.
        for (i, state) in states.iter().enumerate() {
            if state.degenerate_run == DEGENERATE_TOKEN_RUN {
                let stencil = &configs[i].stencil;
                match describe_logit_row(&logits2d, i) {
                    Ok(desc) => tracing::error!(
                        target: "candle_conversation::eos",
                        row = i,
                        stencil_len = stencil.len(),
                        stencil_head = ?stencil.iter().take(4).collect::<Vec<_>>(),
                        "degenerate decode: {desc}"
                    ),
                    Err(e) => tracing::error!(
                        target: "candle_conversation::eos",
                        row = i,
                        "degenerate decode: logits row unreadable ({e})"
                    ),
                }
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
        verify: Option<(&[Option<u32>], TypicalAcceptance)>,
    ) -> candle::Result<Vec<u32>> {
        if matches!(self.device, Device::Cuda(_)) {
            self.sample_batch_cuda(logits, states, configs, verify)
        } else {
            self.sample_batch_cpu(logits, states, configs, verify)
        }
    }

    /// The device half of a row constrained to its stencil allow-list: gather
    /// just the allowed logits (a handful) and enqueue the row's strategy over
    /// them. `O(allow-list)`, never the full vocab. Nothing is read back here
    /// and the row's state is not advanced — [`Self::finish_allow_list`] does
    /// both.
    fn issue_allow_list(
        &self,
        logits2d: &Tensor,
        row: usize,
        config: &SamplingConfig,
        state: &SequenceSamplingState,
    ) -> candle::Result<PendingAllowList> {
        let allow: Vec<u32> = config.stencil.iter().map(|&t| t as u32).collect();
        let idx = Tensor::from_vec(allow.clone(), allow.len(), logits2d.device())?;
        let gathered = logits2d.i(row)?.index_select(&idx, 0)?;
        let gathered = apply_banned_local(&gathered, &allow, config)?;
        let seed = config.seed.wrapping_add(state.rng_offset);
        let processor = LogitsProcessor::from_sampling(seed, config_to_sampling(config));
        let pending = processor.sample_issue(&gathered)?;
        Ok(PendingAllowList {
            allow,
            processor,
            pending,
        })
    }

    /// The host half: read the row's result back, map it to its allowed token
    /// and advance the row.
    fn finish_allow_list(
        &self,
        row: PendingAllowList,
        state: &mut SequenceSamplingState,
    ) -> candle::Result<u32> {
        let PendingAllowList {
            allow,
            mut processor,
            pending,
        } = row;
        let local = processor.sample_finish(pending)? as usize;
        let token = allow[local];
        state.record_token(token, self.max_recent_len);
        state.advance_rng();
        Ok(token)
    }

    /// Resolve a raw sampled token into the token the sequence actually commits.
    ///
    /// Applies, in priority order: the segment-close override (authoritative for
    /// the step), the degenerate-decode abort, then the EOS length failsafes.
    /// `row` names the batch row for the logs. `state` is taken by `&mut` for
    /// `segment_close_override`, which flips `in_segment` as it closes a
    /// segment; nothing here records the committed token, which stays with the
    /// caller that owns the advance.
    ///
    /// Both sampling paths resolve through here because the two copies of this
    /// logic drifted once already: the degenerate-decode abort was written into
    /// the CPU fallback alone, and `sample_full_vocab` sends every unconstrained
    /// row to the kernel whenever the device is CUDA — which is every
    /// deployment that matters. A forward producing unusable logits therefore
    /// ran to the length cap instead of stopping at
    /// [`DEGENERATE_TOKEN_RUN`], writing hundreds of `!` into the conversation
    /// and into the substrate, where the turn's signatures then polluted
    /// retrieval. One authority means a guard added here holds on whichever
    /// path the device selects.
    fn resolve_final_token(
        &self,
        row: usize,
        sampled: u32,
        state: &mut SequenceSamplingState,
        config: &SamplingConfig,
    ) -> u32 {
        let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);

        // One-shot: the dynamic EOS boost ramp begins as `current_len` reaches
        // `eos_ramp_start` (it increments by one, so this fires exactly once per
        // turn).  After this point EOS pressure builds toward the graceful/hard
        // caps below.
        if config.eos_boost != 0.0 && state.current_len == config.eos_ramp_start {
            tracing::debug!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                eos_ramp_start = config.eos_ramp_start,
                eos_ramp_len = config.eos_ramp_len,
                graceful_eos_after = config.graceful_eos_after,
                forced_eos_after = config.forced_eos_after,
                "EOS boost ramp entered",
            );
        }

        // Segment-close override: force the close token when the segment budget
        // is exhausted.  Authoritative for the step — the EOS failsafes must not
        // clobber the close token or a closer-script token (they fire on a later
        // step, once the segment is closed).
        if let Some(t) = segment_close_override(config, state) {
            return t;
        }

        // A tool call opened inside the reasoning block closes the block first.
        if let Some(t) = close_before_call(config, state, sampled) {
            return t;
        }

        if state.degenerate_run >= DEGENERATE_TOKEN_RUN {
            // Degenerate decode: the forward is producing token 0 repeatedly,
            // which is what argmax returns from an all-equal or non-finite
            // logit row. Left alone this runs to the length cap and lands
            // hundreds of `!` in the conversation AND in the substrate, where
            // the turn's signatures then pollute retrieval. Stop at the first
            // sign of it and say so loudly — this is a fault, not an answer.
            tracing::error!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                run = state.degenerate_run,
                "degenerate decode: token 0 emitted {} times consecutively — \
                 forcing EOS. The forward pass produced unusable logits \
                 (all-equal or non-finite); the turn is truncated here.",
                state.degenerate_run,
            );
            return eos_token_id;
        }

        // A call being written is bounded by its grammar, not by the answer's
        // length budget — see `SequenceSamplingState::writing_call`.
        if state.writing_call {
            return sampled;
        }

        if config.forced_eos_after > 0 && state.current_len >= config.forced_eos_after {
            // Hard stop: unconditionally force EOS regardless of sentence position.
            tracing::debug!(
                target: "candle_conversation::eos",
                row,
                current_len = state.current_len,
                forced_eos_after = config.forced_eos_after,
                "hard EOS forced (length cap)",
            );
            return eos_token_id;
        }

        if config.graceful_eos_after > 0 && state.current_len >= config.graceful_eos_after {
            if config.sentence_end_token_ids.is_empty() && config.line_end_token_ids.is_empty() {
                // No sentence-end tokens resolved (e.g. model loaded without
                // tokenizer resolution): fall back to hard stop at the graceful
                // threshold.
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row,
                    current_len = state.current_len,
                    graceful_eos_after = config.graceful_eos_after,
                    "hard EOS forced (no sentence-end tokens)",
                );
                return eos_token_id;
            }
            // Graceful stop: emit EOS only when the last token ended a sentence
            // (`.`, `!`, `?`) or a line.  This lets the current
            // sentence complete before termination, preventing mid-sentence
            // truncation.  `forced_eos_after` is the hard backstop if no boundary
            // is ever seen.
            if state.at_sentence_end(config) {
                tracing::debug!(
                    target: "candle_conversation::eos",
                    row,
                    current_len = state.current_len,
                    graceful_eos_after = config.graceful_eos_after,
                    "soft EOS forced (sentence boundary)",
                );
                return eos_token_id;
            }
        }

        sampled
    }

    /// CPU fallback implementation using candle's built-in sampling.  Receives
    /// only unconstrained (full-vocab) rows; stencil rows are resolved by
    /// `sample_batch` before this is called.
    fn sample_batch_cpu(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
        verify: Option<(&[Option<u32>], TypicalAcceptance)>,
    ) -> candle::Result<Vec<u32>> {
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

            // Structural: ban the think-close token while outside a think block —
            // there is nothing for it to close there, and a stray one derails the
            // turn (see `think_close_ban_active`).
            let seq_logits = if think_close_ban_active(config, state) {
                let dtype = seq_logits.dtype();
                let dims = seq_logits.dims().to_vec();
                let mut v: Vec<f32> = seq_logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                ban(&mut v, config.segment_close_token_id as u32);
                Tensor::from_vec(v, dims, seq_logits.device())?.to_dtype(dtype)?
            } else {
                seq_logits
            };

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
            // The checkpoint's padded tail is not a token.
            let seq_logits = seq_logits.narrow(0, 0, self.live_vocab)?;

            // In-segment steering: while this sequence is inside a segment,
            // sample a touch hotter (temperature + segment_temp_boost).
            // Mirrors the kernel's per-seq gate so tokens outside the segment
            // stay at the base temperature.  DRY is GPU-only — the CPU
            // LogitsProcessor has no DRY path, so there is nothing to gate here for it.
            let temperature = if state.in_segment {
                config.temperature + config.segment_temp_boost
            } else {
                config.temperature
            };
            let sampling = if state.in_segment && config.segment_temp_boost != 0.0 {
                let mut boosted = config.clone();
                boosted.temperature += config.segment_temp_boost;
                config_to_sampling(&boosted)
            } else {
                config_to_sampling(config)
            };
            let seed = config.seed.wrapping_add(state.rng_offset);
            let mut processor = LogitsProcessor::from_sampling(seed, sampling);
            let mut sampled = processor.sample(&seq_logits)?;
            // A verify row's draft, accepted on the typical-acceptance rule over
            // the same top-k/nucleus distribution the kernel measures it on.
            if let Some((drafts, typical)) = verify {
                if let Some(draft) = drafts[i].filter(|_| temperature > 0.0) {
                    let row: Vec<f32> = seq_logits.to_dtype(DType::F32)?.to_vec1()?;
                    if typical_accepts_row(
                        &row,
                        temperature,
                        config.top_k,
                        config.top_p,
                        draft,
                        typical,
                    ) {
                        sampled = draft;
                    }
                }
            }

            // Segment close, degenerate-decode abort and the EOS failsafes all
            // resolve in `resolve_final_token`, shared with the CUDA path.
            let token = self.resolve_final_token(i, sampled, state, config);

            // Record the token and advance RNG
            state.record_token(token, self.max_recent_len);
            state.update_segment_state(
                token,
                config.segment_open_token_id,
                config.segment_close_token_id,
            );
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
        verify: Option<(&[Option<u32>], TypicalAcceptance)>,
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
        // kernel and never reach here. `config` (the first row's) supplies the
        // scalar FFI arguments below, but those are only the null-fallback
        // defaults: the kernel reads its real per-row dials from the `seq_dials`
        // array built further down, so no row inherits row 0's dials.
        let config = configs[0];

        // Get DRY params
        let (dry_multiplier, dry_base, dry_allowed_length, dry_range) =
            if let Some(ref dry) = config.dry {
                (dry.multiplier, dry.base, dry.allowed_length, dry.range)
            } else {
                (0.0, 1.75, 2, 0)
            };

        // Build penalty buffers from states
        let build_span = profile::span("sample:build");
        // The kernel prices each row on its own dials, so the cross-turn table
        // is needed when ANY row carries the penalty, not only row 0.
        let cross_turn = configs.iter().any(|c| c.cross_turn_penalty != 0.0);
        let (token_counts, cross_turn_counts, recent_tokens, recent_lens, current_lens) = self
            .build_penalty_buffers_from_states(
                states,
                config.presence_penalty,
                config.repeat_last_n,
                dry_range,
                cross_turn,
            )?;

        // Get EOS token
        let eos_token_id = self.eos_tokens.iter().copied().next().unwrap_or(0);

        // Build banned tokens buffer — each row's OWN deny-list, plus the
        // structural think-close ban for rows outside a block
        // (`think_close_ban_active` — a `</think>` outside a think block is
        // never valid output). Unlike the scalar dials above, a ban is
        // row-specific: the answer that closes a stuck tool loop bans
        // `<tool_call>` for itself alone. See `banned_rows`.
        let rows: Vec<(&[i32], Option<i32>)> = states
            .iter()
            .zip(configs.iter())
            .map(|(s, c)| {
                let close = think_close_ban_active(c, s).then_some(c.segment_close_token_id);
                (c.banned_tokens.as_slice(), close)
            })
            .collect();
        let (banned_tokens, num_banned, banned_per_seq) = banned_buffer(&rows);
        let banned_tokens = &banned_tokens;

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

        // **Per-sequence dials — every row samples on its own config.** Built
        // from each row's config (not `configs[0]`), so the kernel's scalar
        // arguments below are only the null-fallback defaults; the kernel reads
        // its dials from this array instead. This is what stops one row's EOS
        // ramp (or temperature, or penalties) bleeding into another in a wave
        // that mixes configs.
        // A verify row also carries the draft it tests (see `SeqDials`).
        let seq_dials: Vec<SeqDials> = configs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dials = SeqDials::from_config(c);
                match verify {
                    Some((drafts, typical)) => dials.with_draft(drafts[i], typical),
                    None => dials,
                }
            })
            .collect();
        build_span.end();

        let stamp_span = profile::span("sample:stamp");
        let mut tables = self
            .penalty_tables
            .lock()
            .map_err(|_| candle::Error::Msg("sampler count tables poisoned".into()))?;
        let token_table = tables
            .tokens
            .stamp(&self.device, batch_size, &token_counts)?;
        let cross_table = if cross_turn {
            match tables
                .cross
                .stamp(&self.device, batch_size, &cross_turn_counts)
            {
                Ok(stamped) => Some(stamped),
                Err(e) => {
                    // The token table is already stamped; leave it zero.
                    tables.tokens.clear(token_table)?;
                    return Err(e);
                }
            }
        } else {
            None
        };
        stamp_span.end();

        // Invoke the CUDA kernel
        let launch_span = profile::span("sample:launch_readback");
        let launched = self.invoke_cuda_kernel(
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
            cross_table.as_ref().map(|t| t.table()),
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
            token_table.table(),
            banned_tokens,
            num_banned,
            banned_per_seq,
            &recent_tokens,
            &recent_lens,
            stencil,
            stencil_size,
            &mut output_tokens,
            config.seed,
            &mut rng_offsets,
            &seq_dials,
        );
        launch_span.end();
        // Both cleared whether or not the launch succeeded, and the second
        // whether or not the first did: a table left stamped would price the
        // next dispatch's rows with this one's counts.
        let clear_span = profile::span("sample:clear");
        let tokens_cleared = tables.tokens.clear(token_table);
        let cross_cleared = match cross_table {
            Some(cross) => tables.cross.clear(cross),
            None => Ok(()),
        };
        clear_span.end();
        drop(tables);
        launched?;
        tokens_cleared?;
        cross_cleared?;

        // Update states with sampled tokens and new RNG offsets.
        // Apply post-sampler EOS failsafe overrides: if the sequence has exceeded
        // the configured length limits, replace the sampled token with EOS.
        //
        // **Each row resolves against ITS OWN config, not the shared `config`.**
        // The `configs[0]` collapse above is the *kernel's* constraint — one set
        // of scalar params per launch — and it does not extend to this host-side
        // loop, which visits every row individually. Reading `config` here made a
        // wave's row 0 govern every other row's segment-close budget, EOS
        // failsafes, and think-token ids, so a sequence's own limits applied only
        // when it happened to sort first.
        //
        // That is not hypothetical: it is why a `ThinkMode::Off` ingest summary
        // (`force_segment_close_after == 1`, a forced empty `<think></think>`)
        // closed its block only when it led the wave. Measured over one repo_map
        // pass — 22 summaries opened a block, 7 closed, and all 7 closed at
        // exactly token 2, the forced close firing. The other 15 shared a wave
        // with a dialogue-budget row (a `force_segment_close_after` in the thousands),
        // inherited its budget, and burned the whole 200-token summary allowance
        // on reasoning that was then stored as the summary. The CPU path
        // (`sample_batch_cpu`) always zipped configs per row, so CPU tests could
        // not see it.
        for (i, state) in states.iter_mut().enumerate() {
            let row_config = configs[i];
            // Segment close, degenerate-decode abort and the EOS failsafes all
            // resolve in `resolve_final_token`, shared with the CPU path.
            let token = self.resolve_final_token(i, output_tokens[i], state, row_config);

            output_tokens[i] = token;

            state.record_token(token, self.max_recent_len);
            state.rng_offset = rng_offsets[i];
            // Detect segment open/close transitions for the segment-close boost.
            state.update_segment_state(
                token,
                row_config.segment_open_token_id,
                row_config.segment_close_token_id,
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
        cross_turn: bool,
    ) -> candle::Result<(SparseCounts, SparseCounts, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let batch_size = states.len();

        // Inside a TOOL CALL, all repetition penalties are suppressed, not just
        // DRY.  Tool-call arguments legitimately reproduce content verbatim from
        // the prompt or an earlier span — the query's numbers, file paths,
        // identifiers — so frequency/presence/repeat penalties (which see the
        // `<think>`/prior-span tokens via `token_counts` and `recent_tokens`)
        // would demote exactly those tokens, corrupting the value.  This mirrors
        // the DRY gate but is scoped to tool calls only (`in_tool_call`), so the
        // think block keeps full repetition control.  Presenting empty penalty
        // state for these rows is the per-row equivalent of turning them off:
        // the row stamps nothing, so its table row reads zero.
        let token_counts = SparseCounts::gather(
            self.vocab_size,
            states
                .iter()
                .map(|s| (!s.in_tool_call).then_some((&s.counted[..], &s.token_counts[..]))),
        );
        // The cross-turn table is read only when the penalty is on, so it is
        // only gathered then.
        let cross_turn_counts = if cross_turn {
            SparseCounts::gather(
                self.vocab_size,
                states.iter().map(|s| {
                    (!s.in_tool_call).then_some((&s.cross_counted[..], &s.cross_turn_counts[..]))
                }),
            )
        } else {
            SparseCounts::default()
        };

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
            // Inside a tool call, present a zero-length repeat window so the repeat
            // penalty sees no history (DRY is already gated via `dry_lens`).  Tool
            // arguments must be free to reproduce the query's numbers/paths/names
            // verbatim. The buffer is still padded to keep the batch stride fixed.
            recent_lens.push(if state.in_tool_call { 0 } else { window as i32 });
            recent_tokens.extend_from_slice(&state.recent_tokens[start..]);
            recent_tokens.extend(std::iter::repeat_n(0, self.max_recent_len - window));
        }

        // Current generated lengths (for dynamic EOS ramp). A sequence writing
        // a tool call reports 0, which holds its ramp at zero boost — see
        // `SequenceSamplingState::writing_call`.
        let current_lens: Vec<i32> = states
            .iter()
            .map(|s| if s.writing_call { 0 } else { s.current_len })
            .collect();

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
        cross_turn_counts: Option<&Tensor>,
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
        token_counts: &Tensor,
        banned_tokens: &[i32],
        num_banned: i32,
        banned_per_seq: i32,
        recent_tokens: &[i32],
        recent_lens: &[i32],
        stencil: &[i32],
        stencil_size: i32,
        output_tokens: &mut [u32],
        seed: u64,
        rng_offsets: &mut [u64],
        seq_dials: &[SeqDials],
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

        // The count tables are already on the device; hold their storage for
        // the launch.
        let (tc_storage, tc_layout) = token_counts.storage_and_layout();
        let cross_storage = cross_turn_counts.map(Tensor::storage_and_layout);

        // Every small per-dispatch array in one upload. Per-sequence dials go
        // in as raw 4-byte words — `SeqDials` is packed `f32`/`i32` fields (its
        // C twin `batched_sampling::SeqDials` has the identical layout), so the
        // byte image is what the kernel reads back. The two arrays the kernel
        // writes, RNG offsets then outputs, go last so one copy reads both.
        const SEQ_DIALS_WORDS: usize = std::mem::size_of::<SeqDials>() / 4;
        // SAFETY: `SeqDials` is `#[repr(C)]` with only 4-byte `f32`/`i32`
        // fields and no padding, so a contiguous slice of them is a valid
        // `[i32]` of `len * SEQ_DIALS_WORDS` words.
        let seq_dials_words: &[i32] = unsafe {
            std::slice::from_raw_parts(
                seq_dials.as_ptr() as *const i32,
                seq_dials.len() * SEQ_DIALS_WORDS,
            )
        };
        let rows = output_tokens.len();
        let mut pack = ArgPack::new();
        let cur_lens_at = pack.push_i32(current_lens);
        let segment_lens_at = pack.push_i32(segment_lens);
        let dry_lens_at = pack.push_i32(dry_lens);
        let banned_at = pack.push_i32(banned_tokens);
        let suppress_tok_at = pack.push_i32(suppress_tokens);
        let suppress_pen_at = pack.push_f32(suppress_penalties);
        let recent_at = pack.push_i32(recent_tokens);
        let recent_lens_at = pack.push_i32(recent_lens);
        let stencil_at = pack.push_i32(stencil);
        let seq_dials_at = pack.push_i32(seq_dials_words);
        let rng_at = pack.push_u64(rng_offsets);
        let output_at = pack.reserve(rows);
        let mut packed: cudarc::driver::CudaSlice<u32> = stream
            .memcpy_stod(pack.words())
            .map_err(|e| candle::Error::Msg(format!("failed to upload sampling args: {e}")))?;

        // Get device pointers and call kernel in a scoped block
        // so guards are dropped before download
        {
            let (tc_ptr, _g1) = count_table_ptr(&tc_storage, tc_layout, &stream)?;
            let cross = match &cross_storage {
                Some((storage, layout)) => Some(count_table_ptr(storage, layout, &stream)?),
                None => None,
            };
            let (base, _g2) = packed.device_ptr_mut(&stream);
            let at = |word: usize| base + (word * 4) as u64;
            let cur_lens_ptr = at(cur_lens_at);
            let segment_lens_ptr = at(segment_lens_at);
            let dry_lens_ptr = at(dry_lens_at);
            let suppress_tok_ptr = at(suppress_tok_at);
            let suppress_pen_ptr = at(suppress_pen_at);
            let ban_ptr = at(banned_at);
            let recent_ptr = at(recent_at);
            let recent_lens_ptr = at(recent_lens_at);
            let stencil_ptr = at(stencil_at);
            let output_ptr = at(output_at);
            let rng_ptr = at(rng_at);
            let seq_dials_ptr = at(seq_dials_at);

            // Helper closure to call kernel with logits pointer
            let call_kernel = |logits_ptr: *const std::ffi::c_void| unsafe {
                run_batched_sampling(
                    logits_ptr,
                    batch_size,
                    vocab_size,
                    self.live_vocab as i32,
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
                    match &cross {
                        Some((cross_ptr, _)) => *cross_ptr as *const i32,
                        None => std::ptr::null(),
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
                    banned_per_seq,
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
                    // Null on an empty (guard) upload, else the per-row dials.
                    if seq_dials.is_empty() {
                        std::ptr::null()
                    } else {
                        seq_dials_ptr as *const std::ffi::c_void
                    },
                    stream.cu_stream() as *mut std::ffi::c_void,
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

        // One stream-ordered copy reads the RNG offsets and the outputs back.
        // It is queued behind the kernel, and a device-to-host copy into
        // pageable memory returns only once it has completed, so no separate
        // synchronise is needed.
        let tail = stream
            .memcpy_dtov(&packed.slice(rng_at..output_at + rows))
            .map_err(|e| candle::Error::Msg(format!("failed to download sampling results: {e}")))?;
        let (rng, outputs) = split_rng_and_outputs(&tail, rows);
        output_tokens.copy_from_slice(&outputs);
        rng_offsets.copy_from_slice(&rng);

        Ok(())
    }
}

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

/// Whether the structural think-close ban applies to this row right now.
///
/// **A `</think>` outside a think block is never valid output.** It is a
/// structural token: inside a block the steering owns it (a sampled close is
/// intercepted and replaced by the stencil's canonical prefill), and outside a
/// block there is nothing for it to close. The window this guards is real and
/// was hit twice on the same turn shape: the stencil finishes its walk at the
/// prefilled close, steering ends, and the very next free-decoded tokens have
/// no owner for the close id — the model emitted a second `</think>`, and with
/// the segment state already cleared every think-scoped control (DRY,
/// suppression) was off, so nothing resisted it. The turn's answer then
/// derailed off the malformed transcript.
///
/// Gated on the sampler's own `in_segment` — deliberately not a new flag. This
/// state has desynced from its twins twice before, and the ban is shaped to
/// fail safe against both directions: wrongly *outside* (ban active in a think
/// block) cannot strand the block, because closing it is the stencil's job and
/// the hard-cap closer script forces the token rather than sampling it;
/// wrongly *inside* (ban off after a close) is exactly today's behaviour.
///
/// # The cost of the ban, and why it is still the right trade
///
/// The "cannot strand" argument above holds only for a block opened with the
/// open *token*. A model that spells `<think>` out as plain text never arms
/// `in_segment`, so on an unsteered turn — no stencil, and the hard cap keyed
/// off a flag that is clear — nothing can end it. Measured on a thinking-off
/// ingest: rather than stay stuck, the model closed with a spelling that is not
/// the banned id, and 2 of 22 summaries ended in a bare `</thinking>`.
///
/// Lifting the ban for suppressed turns was tried and is **wrong**: any
/// `</think>` sets `think_close_at` and cuts an index page at a reasoning
/// boundary (`scheduler::mod`), so a stray close outside a block would carve a
/// spurious page into the turn's K/V. A mis-cut index page is the worse of the
/// two, so the ban stays.
///
/// What the ban leaves behind is only partly cleaned up, and knowingly so.
/// `think_strip::strip_trailing_orphan_close` removes a leaked closer when it is
/// the LAST thing in a turn that opened no block — the shape actually measured.
/// A block the model spelled out in plain text mid-answer is NOT recovered: the
/// text is paired with its K/V span by offset, and editing the middle of it
/// would shift that pairing for every consumer that maps text onto tokens. The
/// leaked run is stored verbatim in that case.
///
fn think_close_ban_active(config: &SamplingConfig, state: &SequenceSamplingState) -> bool {
    config.segment_close_token_id >= 0 && !state.in_segment
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
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use std::collections::HashSet;
    use std::sync::Arc;

    use super::*;
    use crate::config::SamplingConfig;

    const VOCAB_SIZE: usize = 100;
    const MAX_RECENT: usize = 32;
    const EOS_TOKEN: u32 = 2;

    /// The per-row dials the kernel reads must carry each config's OWN
    /// segment-close token, not row 0's. A wave that co-batches a deliberating
    /// row (closes `</think>` = 90) with a narrator row (no segment) must give
    /// the deliberating row token 90 and the narrator row -1 — the exact bleed
    /// `SeqDials` exists to remove, and the one a scalar `configs[0]` reintroduced.
    #[test]
    fn seq_dials_carry_each_rows_own_segment_close_token() {
        let mut deliberating = SamplingConfig::argmax();
        deliberating.temperature = 0.7;
        deliberating.segment_close_boost = 3.0;
        deliberating.segment_close_token_id = 90;
        deliberating.segment_close_ramp_start = 2;
        deliberating.segment_close_ramp_len = 16;
        deliberating.segment_close_max_multiplier = 4.0;

        // A narrator: no segment close at all.
        let mut narrator = SamplingConfig::argmax();
        narrator.temperature = 0.9;
        narrator.segment_close_boost = 0.0;
        narrator.segment_close_token_id = -1;

        let d = SeqDials::from_config(&deliberating);
        let n = SeqDials::from_config(&narrator);

        assert_eq!(
            d.segment_close_token_id, 90,
            "deliberating row keeps its close token"
        );
        assert_eq!(d.segment_close_boost, 3.0);
        assert_eq!(d.segment_close_ramp_len, 16);
        assert_eq!(d.temperature, 0.7);

        assert_eq!(
            n.segment_close_token_id, -1,
            "narrator row's close path stays off"
        );
        assert_eq!(n.segment_close_boost, 0.0);
        assert_eq!(n.temperature, 0.9);
    }

    /// A config with the boost dialled up but NO token id is not "half on": the
    /// whole segment-close path is gated off (token id -1), so a co-batched row
    /// with it genuinely on is unaffected.
    #[test]
    fn seq_dials_gate_segment_close_off_when_token_missing() {
        let mut c = SamplingConfig::argmax();
        c.segment_close_boost = 5.0;
        c.segment_close_token_id = -1; // boost set, but no token to boost
        let d = SeqDials::from_config(&c);
        assert_eq!(d.segment_close_token_id, -1);
        assert_eq!(
            d.segment_close_boost, 0.0,
            "boost neutralised without a token"
        );
    }

    /// A run of token 0 is counted, and any other token clears it — the guard
    /// must fire on a *consecutive* run, not on token 0 being frequent.
    #[test]
    fn degenerate_run_counts_consecutive_zero_tokens() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        assert_eq!(st.degenerate_run, 0);
        for expected in 1..=DEGENERATE_TOKEN_RUN {
            st.record_token(0, MAX_RECENT);
            assert_eq!(st.degenerate_run, expected);
        }
        // A real token breaks the run.
        st.record_token(7, MAX_RECENT);
        assert_eq!(st.degenerate_run, 0);
        // Interleaved zeros never accumulate to the bar.
        for _ in 0..50 {
            st.record_token(0, MAX_RECENT);
            st.record_token(7, MAX_RECENT);
        }
        assert_eq!(st.degenerate_run, 0);
    }

    /// The run is per-turn: a fresh turn must not inherit a previous turn's tail.
    #[test]
    fn degenerate_run_resets_at_turn_end() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for _ in 0..DEGENERATE_TOKEN_RUN {
            st.record_token(0, MAX_RECENT);
        }
        assert_eq!(st.degenerate_run, DEGENERATE_TOKEN_RUN);
        st.end_turn(0);
        assert_eq!(st.degenerate_run, 0);
    }

    /// **The cross-turn penalty sees a window of turns, not the whole day.**
    ///
    /// It is flat — any count above zero costs the same — so without a bound it
    /// ends up on a character's entire working vocabulary, and a uniform shift
    /// tells no act from any other. With a window, a turn that leaves takes its
    /// tokens with it.
    #[test]
    fn the_cross_turn_window_forgets_turns_that_fall_out_of_it() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 4, 5] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(2);
        }
        assert_eq!(
            st.cross_turn_counts[3], 0,
            "two turns on, the first is forgotten"
        );
        assert_eq!(st.cross_turn_counts[4], 1);
        assert_eq!(st.cross_turn_counts[5], 1);
    }

    /// A token used again is still penalised until its *last* use leaves.
    #[test]
    fn a_token_used_again_stays_counted_until_its_last_use_leaves() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 3, 4] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(2);
        }
        assert_eq!(
            st.cross_turn_counts[3], 1,
            "its second use is still in the window"
        );
        st.record_token(5, MAX_RECENT);
        st.end_turn(2);
        assert_eq!(st.cross_turn_counts[3], 0, "and now it is not");
    }

    /// `0` keeps every turn, which is what every caller that sets no window —
    /// all of them with the penalty off — has always had.
    #[test]
    fn a_zero_window_remembers_every_turn() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 4, 5, 6] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(0);
        }
        assert!([3usize, 4, 5, 6]
            .iter()
            .all(|&t| st.cross_turn_counts[t] == 1));
        assert!(
            st.cross_turn_history.is_empty(),
            "nothing to forget, nothing kept"
        );
    }

    /// `end_turn` also runs at a conversation's first decode with nothing
    /// sampled yet. That must not use up a slot.
    #[test]
    fn a_turn_that_sampled_nothing_takes_no_slot() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        st.record_token(3, MAX_RECENT);
        st.end_turn(1);
        st.end_turn(1);
        assert_eq!(st.cross_turn_counts[3], 1);
    }

    /// The bar has to be low enough to stop a broken forward promptly, and high
    /// enough that ordinary text can never reach it.
    #[test]
    fn degenerate_run_bar_is_small_but_out_of_language_range() {
        assert!(
            DEGENERATE_TOKEN_RUN >= 4,
            "must tolerate a brief coincidence"
        );
        assert!(
            DEGENERATE_TOKEN_RUN <= 16,
            "must fire long before the length cap: the observed failure ran 1219 tokens",
        );
    }

    /// The degenerate-decode abort must live on the resolver BOTH sampling
    /// paths run through, not in one path's copy of the overrides.
    ///
    /// It was written into `sample_batch_cpu` alone, and `sample_full_vocab`
    /// sends every unconstrained row to the CUDA kernel whenever the device is
    /// CUDA — so on the only configuration production runs, the guard was dead:
    /// a forward emitting token 0 forever ran to the length cap instead of
    /// stopping at `DEGENERATE_TOKEN_RUN`, writing hundreds of `!` into the
    /// conversation and into the substrate, where the turn's signatures then
    /// polluted retrieval. Asserting on `resolve_final_token` is what keeps the
    /// guard device-independent: there is no second copy to be missing from.
    #[test]
    fn degenerate_run_forces_eos_on_the_resolver_both_paths_share() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax();
        let mut state = make_state();

        // One short of the bar: an ordinary sampled token is committed as-is.
        for _ in 0..DEGENERATE_TOKEN_RUN - 1 {
            state.record_token(0, MAX_RECENT);
        }
        assert_eq!(
            sampler.resolve_final_token(0, 7, &mut state, &config),
            7,
            "below the bar the sampler's own token stands"
        );

        // The next consecutive zero crosses it, and the turn is cut short.
        state.record_token(0, MAX_RECENT);
        assert_eq!(
            sampler.resolve_final_token(0, 7, &mut state, &config),
            EOS_TOKEN,
            "a degenerate run must force EOS whichever path sampled the row"
        );

        // A real token clears the run, and decoding resumes normally — the
        // guard fires on a consecutive run, never on token 0 being frequent.
        state.record_token(42, MAX_RECENT);
        assert_eq!(sampler.resolve_final_token(0, 7, &mut state, &config), 7);
    }

    /// **A call being written is not cut by the answer's length budget.** Past
    /// both EOS thresholds a prose turn is ended; the same length inside a tool
    /// call keeps the model's own token, and the EOS ramp sees a length of 0.
    /// The call's own grammar and the turn's `max_tokens` bound it instead.
    #[test]
    fn writing_a_call_stands_the_length_budget_down() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.graceful_eos_after = 10;
        config.forced_eos_after = 20;
        let mut state = make_state();
        for _ in 0..25 {
            state.record_token(42, MAX_RECENT);
        }
        assert_eq!(
            sampler.resolve_final_token(0, 7, &mut state, &config),
            EOS_TOKEN,
            "a prose turn past its budget is ended"
        );

        state.writing_call = true;
        assert_eq!(
            sampler.resolve_final_token(0, 7, &mut state, &config),
            7,
            "a call past the same budget keeps its token"
        );
        let (_, _, _, _, current_lens) = sampler
            .build_penalty_buffers_from_states(&[&mut state], 0.0, 16, 0, false)
            .unwrap();
        assert_eq!(current_lens, vec![0], "the EOS ramp sees no length");

        // The degenerate-decode guard is a fault check, not a budget: it still
        // fires inside a call.
        for _ in 0..DEGENERATE_TOKEN_RUN {
            state.record_token(0, MAX_RECENT);
        }
        assert_eq!(
            sampler.resolve_final_token(0, 7, &mut state, &config),
            EOS_TOKEN
        );
    }

    fn make_sampler() -> BatchedSampler {
        BatchedSampler::new(
            candle::Device::Cpu,
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        )
    }

    fn make_state() -> SequenceSamplingState {
        SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT)
    }

    // ── Hard-cap closer script (segment_close_override tiers) ──────────

    /// Config with segment tracking on: close=90, graceful after 4 at a line
    /// end (token 7), a sentence end that is not a line end (token 8, `.`),
    /// hard cap at 8, closer phrase "A B C" (the sampler appends the close
    /// token 90 itself).
    fn closer_config() -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.segment_close_token_id = 90;
        c.segment_open_token_id = 89;
        c.graceful_segment_close_after = 4;
        c.force_segment_close_after = 8;
        c.sentence_end_token_ids = vec![8];
        c.line_end_token_ids = Arc::from([7]);
        c.segment_close_script = vec![100, 101, 102];
        c
    }

    /// A state `n` tokens into an open segment, last token `last`.
    fn in_segment_state(n: i32, last: u32) -> SequenceSamplingState {
        let mut s = make_state();
        s.record_token(last, MAX_RECENT);
        s.in_segment = true;
        s.segment_len = n;
        s
    }

    #[test]
    fn hard_cap_plays_the_closer_script_to_the_close_token() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42); // past force, mid-sentence
        let mut played = Vec::new();
        for _ in 0..4 {
            played.push(segment_close_override(&config, &mut state).expect("override"));
        }
        assert_eq!(
            played,
            vec![100, 101, 102, 90],
            "phrase then the sampler-appended close, in order"
        );
        assert_eq!(
            state.close_script_pos, None,
            "script state cleared at the end"
        );
    }

    #[test]
    fn graceful_close_at_line_end_skips_the_script() {
        let config = closer_config();
        // Past graceful (not force), last token IS a line end.
        let mut state = in_segment_state(5, 7);
        assert_eq!(
            segment_close_override(&config, &mut state),
            Some(90),
            "soft cut closes bare — a completed line needs no rescue"
        );
        assert_eq!(state.close_script_pos, None);
    }

    /// **A `.` is not where a thought ends.** Past the graceful cap, a period
    /// that is not a line end — the one in `169.254` — leaves the block open;
    /// the close waits for the line to end.
    #[test]
    fn graceful_close_does_not_fire_at_a_period_mid_line() {
        let config = closer_config();
        let mut state = in_segment_state(5, 8);
        assert_eq!(segment_close_override(&config, &mut state), None);
    }

    /// The graceful EOS still accepts a sentence end: an answer that is one
    /// paragraph must not wait for a newline it will never write.
    #[test]
    fn the_answer_ends_at_a_sentence_or_a_line() {
        let config = closer_config();
        for last in [7, 8] {
            let mut state = make_state();
            state.record_token(last, MAX_RECENT);
            assert!(state.at_sentence_end(&config), "token {last}");
        }
        let mut mid = make_state();
        mid.record_token(42, MAX_RECENT);
        assert!(!mid.at_sentence_end(&config));
    }

    #[test]
    fn continuation_span_gets_the_bare_close_not_the_script() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42);
        // A span that retires into more content: the steering drops the close
        // and decoding continues, so no terminal closing statement.
        state.close_would_continue = true;
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None, "no script started");
    }

    #[test]
    fn hard_cap_at_a_completed_sentence_closes_bare() {
        let config = closer_config();
        // Past force, but the last token IS a sentence end — the amputation
        // rescue is for dangling fragments only.
        let mut state = in_segment_state(8, 7);
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None, "no script started");
    }

    #[test]
    fn segment_close_override_is_inert_outside_a_segment_or_unconfigured() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42);
        state.in_segment = false;
        assert_eq!(segment_close_override(&config, &mut state), None);

        let mut unconfigured = closer_config();
        unconfigured.segment_close_token_id = -1;
        let mut state = in_segment_state(8, 42);
        assert_eq!(segment_close_override(&unconfigured, &mut state), None);
    }

    #[test]
    fn below_both_caps_no_override() {
        let config = closer_config();
        let mut state = in_segment_state(3, 7);
        assert_eq!(segment_close_override(&config, &mut state), None);
    }

    /// A call opened inside a think block closes the block first — the opener
    /// is committed as the close — and nowhere else does the opener change.
    #[test]
    fn a_call_opened_inside_a_segment_closes_it_first() {
        let mut config = closer_config();
        config.tool_call_open_token_id = 70;
        let state = in_segment_state(3, 42);
        assert_eq!(close_before_call(&config, &state, 70), Some(90));
        assert_eq!(
            close_before_call(&config, &state, 71),
            None,
            "any other token stands"
        );
        let mut outside = in_segment_state(3, 42);
        outside.in_segment = false;
        assert_eq!(
            close_before_call(&config, &outside, 70),
            None,
            "a call outside the block stands"
        );
        config.tool_call_open_token_id = -1;
        assert_eq!(
            close_before_call(&config, &state, 70),
            None,
            "an unresolved opener changes nothing"
        );
    }

    /// Through the one authority both sampling paths resolve with: under the
    /// segment budget, a sampled opener inside the block commits as the close.
    #[test]
    fn the_committed_token_for_a_call_inside_a_segment_is_the_close() {
        let sampler = make_sampler();
        let mut config = closer_config();
        config.tool_call_open_token_id = 70;
        let mut state = in_segment_state(3, 42);
        assert_eq!(sampler.resolve_final_token(0, 70, &mut state, &config), 90);
    }

    #[test]
    fn hard_cap_without_script_falls_back_to_bare_close() {
        let mut config = closer_config();
        config.segment_close_script = Vec::new();
        let mut state = in_segment_state(8, 42);
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
    }

    #[test]
    fn script_in_flight_overrides_graceful_and_force_conditions() {
        let config = closer_config();
        let mut state = in_segment_state(9, 7); // sentence end AND past force
        state.close_script_pos = Some(2);
        // Mid-script: the next scripted token wins over every other tier.
        assert_eq!(segment_close_override(&config, &mut state), Some(102));
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None);
    }

    #[test]
    fn segment_close_wins_over_eos_failsafes_for_the_step() {
        let sampler = make_sampler();
        let mut config = closer_config();
        config.forced_eos_after = 5; // far exceeded — EOS wants to fire every step
        let mut state = in_segment_state(8, 42);
        for _ in 0..7 {
            state.record_token(42, MAX_RECENT);
        }
        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        // The closer script plays to completion; the EOS failsafe never
        // clobbers a scripted step or the close itself.
        let mut played = Vec::new();
        for _ in 0..4 {
            played.push(
                sampler
                    .sample_batch(&logits, &mut [&mut state], &[&config])
                    .expect("sample")[0],
            );
        }
        assert_eq!(played, vec![100, 101, 102, 90]);
        assert!(
            !state.in_segment,
            "the sampled close token exits the segment on the CPU path"
        );

        // With the segment closed, the deferred EOS failsafe fires next step.
        let next = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample")[0];
        assert_eq!(next, EOS_TOKEN);
    }

    #[test]
    fn segment_boundaries_and_turn_end_cancel_a_stranded_script() {
        let mut state = make_state();
        state.enter_segment();
        state.close_script_pos = Some(1);
        state.exit_segment();
        assert_eq!(state.close_script_pos, None, "exit cancels the script");

        state.enter_segment();
        state.close_script_pos = Some(2);
        state.enter_segment();
        assert_eq!(
            state.close_script_pos, None,
            "a fresh segment cannot inherit a script"
        );

        state.close_script_pos = Some(1);
        state.close_would_continue = true;
        state.end_turn(0);
        assert!(!state.in_segment, "turn end closes a dangling segment");
        assert_eq!(state.segment_len, 0);
        assert_eq!(state.close_script_pos, None);
        assert!(!state.close_would_continue);
    }

    // ── Tool-call penalty suppression ──────────────────────────────────

    #[test]
    fn tool_call_row_penalty_state_is_zeroed_and_think_row_is_not() {
        let sampler = make_sampler();
        let mut in_call = make_state();
        let mut thinking = make_state();
        // Both rows generated the same tokens (e.g. digits reasoned in <think>).
        for _ in 0..5 {
            in_call.record_token(42, MAX_RECENT);
            thinking.record_token(42, MAX_RECENT);
        }
        in_call.in_tool_call = true;
        // A prior turn, so the cross-turn table has something to suppress.
        in_call.end_turn(0);
        thinking.end_turn(0);
        for _ in 0..5 {
            in_call.record_token(42, MAX_RECENT);
            thinking.record_token(42, MAX_RECENT);
        }

        let (token_counts, cross_turn_counts, _recent, recent_lens, _cur) = sampler
            .build_penalty_buffers_from_states(&[&mut in_call, &mut thinking], 0.0, 16, 0, true)
            .expect("buffers");

        // Row 0 (tool call) stamps nothing, so its table rows read zero — the
        // model is free to reproduce the query's tokens verbatim in the
        // arguments. Row 1 (think block) keeps full repetition control.
        let row1_42 = VOCAB_SIZE as u32 + 42;
        assert_eq!(
            token_counts,
            SparseCounts {
                offsets: vec![row1_42],
                values: vec![5],
            }
        );
        assert_eq!(
            cross_turn_counts,
            SparseCounts {
                offsets: vec![row1_42],
                values: vec![5],
            }
        );
        assert_eq!(recent_lens[0], 0);
        assert_eq!(recent_lens[1], 10);
    }

    /// With the cross-turn penalty off, the cross table is never read, so
    /// nothing is gathered for it.
    #[test]
    fn the_cross_turn_table_is_gathered_only_when_its_penalty_is_on() {
        let sampler = make_sampler();
        let mut state = make_state();
        state.record_token(7, MAX_RECENT);
        state.end_turn(0);
        let (_, cross, _, _, _) = sampler
            .build_penalty_buffers_from_states(&[&mut state], 0.0, 16, 0, false)
            .expect("buffers");
        assert_eq!(cross, SparseCounts::default());
    }

    /// The sparse indexes name exactly the nonzero dense entries, through
    /// recording, a turn end and a windowed cross-turn eviction — they are what
    /// the device table is stamped from, so a token missing from them is a
    /// penalty silently not applied.
    #[test]
    fn the_sparse_indexes_track_the_nonzero_counts_across_turns() {
        fn nonzero(counts: &[i32]) -> Vec<u32> {
            (0..counts.len() as u32)
                .filter(|&t| counts[t as usize] > 0)
                .collect()
        }
        fn sorted(v: &[u32]) -> Vec<u32> {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        }
        let mut st = make_state();
        for t in [9, 3, 9, 4] {
            st.record_token(t, MAX_RECENT);
        }
        assert_eq!(st.counted, vec![9, 3, 4]);
        assert_eq!(sorted(&st.counted), nonzero(&st.token_counts));

        // Window of one: turn A enters the cross counts.
        st.end_turn(1);
        assert!(st.counted.is_empty());
        assert_eq!(nonzero(&st.token_counts), Vec::<u32>::new());
        assert_eq!(sorted(&st.cross_counted), vec![3, 4, 9]);
        assert_eq!(sorted(&st.cross_counted), nonzero(&st.cross_turn_counts));

        // Turn B evicts turn A: only B's tokens remain.
        for t in [4, 5] {
            st.record_token(t, MAX_RECENT);
        }
        st.end_turn(1);
        assert_eq!(sorted(&st.cross_counted), vec![4, 5]);
        assert_eq!(sorted(&st.cross_counted), nonzero(&st.cross_turn_counts));
        assert_eq!(st.cross_turn_counts[4], 1);

        st.clear();
        assert!(st.counted.is_empty() && st.cross_counted.is_empty());
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

    /// Constrained rows are issued in the pass over the batch and finished
    /// after the unconstrained rows' kernel: each must land in its OWN row, and
    /// advance its own state, with an unconstrained row between them. Every row
    /// peaks at 50; the two allow-lists exclude it and pick their own best. On
    /// CUDA when a card is present — the path the deferral is for — and on the
    /// CPU path always.
    #[test]
    fn deferred_allow_list_rows_land_in_their_own_rows() {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        let first = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let free = SamplingConfig::argmax();
        let last = SamplingConfig::argmax().with_stencil(vec![30, 40]);
        for device in devices {
            let sampler = BatchedSampler::new(
                device.clone(),
                VOCAB_SIZE,
                VOCAB_SIZE,
                MAX_RECENT,
                vec![EOS_TOKEN].into(),
                None,
            );
            let (mut s0, mut s1, mut s2) = (make_state(), make_state(), make_state());
            let logits = logits_from_rows(&[
                &[(50, 100.0), (20, 5.0), (10, 1.0)],
                &[(50, 100.0), (20, 5.0)],
                &[(50, 100.0), (30, 7.0), (40, 9.0)],
            ])
            .to_device(&device)
            .expect("logits");
            let tokens = sampler
                .sample_batch(
                    &logits,
                    &mut [&mut s0, &mut s1, &mut s2],
                    &[&first, &free, &last],
                )
                .expect("sample");
            assert_eq!(tokens, vec![20, 50, 40], "{device:?}");
            assert_eq!((s0.rng_offset, s2.rng_offset), (1, 1), "{device:?}");
        }
    }

    /// The typical-acceptance rule on raw probabilities. `[0.5, 0.3, 0.2]` has
    /// entropy 1.0297, so `δ·e^(−H)` = 0.1071 and the bar is ε = 0.09: the 0.2
    /// draft clears it. `[0.9, 0.05, 0.05]` has entropy 0.3944, the bar is again
    /// 0.09, and a 0.05 draft does not.
    #[test]
    fn typical_rule_accepts_on_mass_against_the_entropy_bar() {
        let ln = |ps: &[f32]| ps.iter().map(|p| p.ln()).collect::<Vec<f32>>();
        let m = TypicalAcceptance::MEDUSA;
        let accepts =
            |ps: &[f32], t: f32, draft: u32| typical_accepts_row(&ln(ps), t, 0, 1.0, draft, m);
        assert!(accepts(&[0.5, 0.3, 0.2], 1.0, 2));
        assert!(!accepts(&[0.9, 0.05, 0.05], 1.0, 2));
        // Temperature sharpens the row: at 0.5, [0.5, 0.3, 0.2] becomes
        // [0.658, 0.237, 0.105] (entropy 0.8533, bar 0.09), and 0.105 still
        // clears it; [0.9, 0.05, 0.05] becomes [0.994, 0.003, 0.003].
        assert!(accepts(&[0.5, 0.3, 0.2], 0.5, 2));
        assert!(!accepts(&[0.9, 0.05, 0.05], 0.5, 1));
        // A draft past the row is not a token.
        assert!(!accepts(&[0.5, 0.5], 1.0, 7));
    }

    /// **The rule is measured over the distribution the kernel samples, not
    /// the whole row.** `[0.5, 0.3, 0.2]` accepts its 0.2 draft over the whole
    /// row, but a nucleus at 0.75 keeps `[0.5, 0.3]` (cumulative 0.8) and top-k
    /// 2 keeps the same two: the draft is outside either, has probability zero,
    /// and is not accepted — exactly as the kernel decides it.
    #[test]
    fn typical_rule_is_measured_over_the_truncated_distribution() {
        let row: Vec<f32> = [0.5f32, 0.3, 0.2].iter().map(|p| p.ln()).collect();
        let m = TypicalAcceptance::MEDUSA;
        assert!(typical_accepts_row(&row, 1.0, 0, 1.0, 2, m));
        assert!(!typical_accepts_row(&row, 1.0, 0, 0.75, 2, m));
        assert!(!typical_accepts_row(&row, 1.0, 2, 1.0, 2, m));
        // Renormalised over the nucleus `[0.625, 0.375]` (entropy 0.6616, bar
        // 0.09), the 0.375 draft clears it.
        assert!(typical_accepts_row(&row, 1.0, 0, 0.75, 1, m));
    }

    /// One logits row with `p(5) = 0.8` and `p(7) = 0.2`; every other token
    /// sits at logit 0, ~2e-9 each.
    fn verify_row_logits() -> Tensor {
        logits_from_rows(&[&[(5, 20.0), (7, 20.0 + 0.25f32.ln())]])
    }

    fn sampled_config(seed: u64) -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.temperature = 1.0;
        c.top_k = 0;
        c.top_p = 1.0;
        c.seed = seed;
        c
    }

    fn devices() -> Vec<Device> {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        devices
    }

    /// **A draft the distribution gives enough mass is committed whatever the
    /// sample.** `p(7) = 0.2` clears the 0.09 bar, so every seed commits 7 —
    /// including the ~80% whose sample lands on 5. The sample is still drawn,
    /// so the row's RNG advances exactly as a plain row's.
    #[test]
    fn a_draft_over_the_bar_is_committed_whatever_the_sample() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            for seed in 0..16 {
                let mut state = make_state();
                let tokens = sampler
                    .sample_verify_rows(
                        &logits,
                        &mut [&mut state],
                        &[&sampled_config(seed)],
                        &[Some(7)],
                        TypicalAcceptance::MEDUSA,
                    )
                    .expect("sample");
                assert_eq!(tokens, vec![7], "{device:?} seed {seed}");
                assert_eq!(state.rng_offset, 1, "{device:?} seed {seed}");
            }
        }
    }

    /// **A draft under the bar commits the sample.** Token 9 sits at logit 0,
    /// ~2e-9 of the row, so the row commits what plain sampling draws: 5 or 7.
    /// Across 32 seeds both turn up, which a correction pinned to the argmax
    /// could never produce.
    #[test]
    fn a_draft_under_the_bar_commits_the_sample() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            let mut seen = HashSet::new();
            for seed in 0..32 {
                let mut state = make_state();
                let tokens = sampler
                    .sample_verify_rows(
                        &logits,
                        &mut [&mut state],
                        &[&sampled_config(seed)],
                        &[Some(9)],
                        TypicalAcceptance::MEDUSA,
                    )
                    .expect("sample");
                assert!(
                    tokens[0] == 5 || tokens[0] == 7,
                    "{device:?} seed {seed}: {tokens:?}"
                );
                seen.insert(tokens[0]);
            }
            assert_eq!(seen.len(), 2, "{device:?}: corrections {seen:?}");
        }
    }

    /// **Greedy verification is unchanged.** At temperature zero the row is a
    /// point mass on its argmax, so a 0.2 draft is not accepted and the row
    /// commits 5.
    #[test]
    fn at_temperature_zero_only_the_argmax_is_committed() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            let mut state = make_state();
            let tokens = sampler
                .sample_verify_rows(
                    &logits,
                    &mut [&mut state],
                    &[&SamplingConfig::argmax()],
                    &[Some(7)],
                    TypicalAcceptance::MEDUSA,
                )
                .expect("sample");
            assert_eq!(tokens, vec![5], "{device:?}");
        }
    }

    /// **The padded tail of a row is never a token.** The sampler's live
    /// vocabulary ends at 50; token 60, in the padding, carries the row's
    /// largest logit by far. Neither the argmax nor a sample may produce it,
    /// and a draft naming it is not accepted.
    #[test]
    fn the_padded_tail_of_a_row_is_never_produced() {
        for device in devices() {
            let sampler = make_sampler_on(&device, 50);
            let logits = logits_from_rows(&[&[(5, 20.0), (7, 20.0 + 0.25f32.ln()), (60, 90.0)]])
                .to_device(&device)
                .expect("logits");
            let mut state = make_state();
            let greedy = sampler
                .sample_batch(&logits, &mut [&mut state], &[&SamplingConfig::argmax()])
                .expect("sample");
            assert_eq!(greedy, vec![5], "{device:?}");
            for seed in 0..16 {
                let mut state = make_state();
                let tokens = sampler
                    .sample_verify_rows(
                        &logits,
                        &mut [&mut state],
                        &[&sampled_config(seed)],
                        &[Some(60)],
                        TypicalAcceptance::MEDUSA,
                    )
                    .expect("sample");
                assert!(
                    tokens[0] == 5 || tokens[0] == 7,
                    "{device:?} seed {seed}: {tokens:?}"
                );
            }
        }
    }

    fn make_sampler_on(device: &Device, live_vocab: usize) -> BatchedSampler {
        BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            live_vocab,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        )
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

    /// **A row's ban is its own inside a shared launch.** One row bans nothing;
    /// the other — the answer closing a stuck tool loop — bans 50. Both logit
    /// rows peak at 50: the free row takes it, the banning row its next best.
    /// Both orderings, because the kernel read the ban from `configs[0]`: with
    /// the banning row second its ban vanished, with it first the free row lost
    /// its token. On CUDA when a card is present — the path that had the defect
    /// — and on the CPU path always.
    #[test]
    fn a_row_s_ban_stays_in_its_row_in_a_shared_launch() {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        let free = SamplingConfig::argmax();
        let mut closing = SamplingConfig::argmax();
        closing.banned_tokens = vec![50];
        for device in devices {
            let sampler = BatchedSampler::new(
                device.clone(),
                VOCAB_SIZE,
                VOCAB_SIZE,
                MAX_RECENT,
                vec![EOS_TOKEN].into(),
                None,
            );
            for (configs, expected) in [
                ([&free, &closing], vec![50, 60]),
                ([&closing, &free], vec![60, 50]),
            ] {
                let (mut a, mut b) = (make_state(), make_state());
                let logits =
                    logits_from_rows(&[&[(50, 100.0), (60, 50.0)], &[(50, 100.0), (60, 50.0)]])
                        .to_device(&device)
                        .expect("logits");
                let tokens = sampler
                    .sample_batch(&logits, &mut [&mut a, &mut b], &configs)
                    .expect("sample");
                assert_eq!(tokens, expected, "{device:?}");
            }
        }
    }

    // ── Structural think-close ban ─────────────────────────────────────

    /// **A `</think>` outside a think block is never sampleable.** The stencil
    /// owns the close inside a block; outside one there is nothing to close,
    /// and a stray close after the stencil's walk has finished is exactly what
    /// produced the doubled `</think>` that derailed a live turn — twice, on
    /// the same turn shape.
    #[test]
    fn a_think_close_outside_a_think_block_is_banned() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.segment_open_token_id = 89;
        config.segment_close_token_id = 90;
        let mut state = make_state();
        assert!(!state.in_segment, "a turn starts outside any think block");
        assert!(think_close_ban_active(&config, &state));

        let logits = logits_from_rows(&[&[(90, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(
            tokens[0], 60,
            "the close token must be unreachable outside a block"
        );
    }

    // ── Per-row limits inside one shared launch ───────────────────────

    /// **A row's own segment-close budget applies inside a shared launch.**
    ///
    /// The host-side post-kernel loop resolves the segment-close budget (and the
    /// EOS failsafes) per row from `configs[i]`, never row 0. Both rows here go
    /// through ONE launch: row 0 is greedy over a decisive logit row, row 1 is
    /// mid-block with a hard cap of 1 and must still be forced closed on its own
    /// config, not row 0's.
    #[test]
    fn a_row_obeys_its_own_close_budget_in_a_shared_launch() {
        let sampler = make_sampler();
        let greedy = SamplingConfig::argmax();
        let mut forced = SamplingConfig::argmax();
        forced.segment_open_token_id = 89;
        forced.segment_close_token_id = 90;
        forced.force_segment_close_after = 1;

        let mut plain = make_state();
        let mut in_block = make_state();
        in_block.in_segment = true;
        in_block.segment_len = 5;

        let logits = logits_from_rows(&[&[(42, 100.0), (7, 1.0)], &[(42, 100.0), (7, 1.0)]]);
        let tokens = sampler
            .sample_batch(
                &logits,
                &mut [&mut plain, &mut in_block],
                &[&greedy, &forced],
            )
            .expect("sample");

        assert_eq!(tokens[0], 42, "the greedy row takes its own argmax");
        assert_eq!(
            tokens[1], 90,
            "the forced-close row closes on ITS config, not row 0's"
        );
    }

    /// Inside a block the ban is off: the model must stay free to emit the
    /// close (the steering intercepts it — `TokenClosedDrop` — or the hard-cap
    /// closer forces it; the sampler's job is only to not fight either).
    #[test]
    fn a_think_close_inside_a_think_block_is_free() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.segment_open_token_id = 89;
        config.segment_close_token_id = 90;
        let mut state = make_state();
        state.enter_segment();
        assert!(!think_close_ban_active(&config, &state));

        let logits = logits_from_rows(&[&[(90, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 90, "inside a block the close samples normally");
    }

    /// Without segment tracking configured (close id < 0) the ban never
    /// activates — reference models with no think protocol are untouched.
    #[test]
    fn no_think_protocol_means_no_ban() {
        let config = SamplingConfig::argmax();
        let state = make_state();
        assert!(!think_close_ban_active(&config, &state));
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

    // ── Per-row config in a mixed wave ────────────────────────────────────

    /// A row inside a think block, `n` tokens deep, with `force` as its hard
    /// segment-close cap. Close id 90, open id 80, no closer script and no
    /// sentence-end ids — so the ONLY thing that can emit 90 is the force cap.
    fn wave_row(force: i32) -> (SamplingConfig, SequenceSamplingState) {
        let mut config = SamplingConfig::top_k(1, 1.0);
        config.segment_close_token_id = 90;
        config.segment_open_token_id = 80;
        config.force_segment_close_after = force;
        config.graceful_segment_close_after = 0;
        let mut state = make_state();
        state.in_segment = true;
        state.segment_len = 5;
        (config, state)
    }

    /// **Every row of a wave resolves against its OWN config.**
    ///
    /// The kernel takes one set of scalar params per launch, and the CUDA path
    /// reads them from `configs[0]`. That collapse must NOT reach the host-side
    /// per-row loop that applies the segment-close and EOS overrides: a wave
    /// mixes sequences with genuinely different budgets — a 200-token
    /// `ThinkMode::Off` ingest summary beside a dialogue turn — and each must
    /// get its own.
    ///
    /// Both orderings are asserted because either one alone passes under the
    /// wrong fix: reading `configs[0]` satisfies the first, reading the last
    /// row's config satisfies the second.
    ///
    /// Runs on CUDA, which is the only path that had the defect —
    /// `sample_batch_cpu` always zipped configs per row, so a CPU-only test
    /// could never have caught it.
    #[test]
    fn each_row_of_a_wave_honours_its_own_segment_budget() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return; // No CUDA device on this box; the CPU path is covered above.
        };
        let sampler = BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        );

        // Logits that make token 42 the argmax and the close token 90 the
        // least likely, so a row that emits 90 can only have been forced.
        let mut row = vec![0.0f32; VOCAB_SIZE];
        row[42] = 10.0;
        row[90] = -10.0;

        for (label, forces) in [("summary first", [1, 1536]), ("dialogue first", [1536, 1])] {
            let (cfg_a, mut st_a) = wave_row(forces[0]);
            let (cfg_b, mut st_b) = wave_row(forces[1]);
            let configs = [&cfg_a, &cfg_b];
            let mut states = [&mut st_a, &mut st_b];

            let logits = Tensor::from_vec(
                row.iter().chain(row.iter()).copied().collect::<Vec<f32>>(),
                (2, VOCAB_SIZE),
                &device,
            )
            .expect("logits");

            let tokens = sampler
                .sample_batch(&logits, &mut states, &configs)
                .expect("sample_batch");

            for (i, &force) in forces.iter().enumerate() {
                if force == 1 {
                    assert_eq!(
                        tokens[i], 90,
                        "{label}: row {i} is 5 tokens into its block with a hard cap of 1 — \
                         its close must be forced, whatever the other row's budget is"
                    );
                } else {
                    assert_ne!(
                        tokens[i], 90,
                        "{label}: row {i} has a cap of 1536 and is only 5 tokens in — \
                         it must NOT inherit the other row's forced close"
                    );
                }
            }
        }
    }

    /// The kernel reads its counts from the device-resident tables: each row
    /// is priced by its own history, a tool-call row by none, and the next
    /// dispatch starts from a clean table rather than the last one's counts.
    ///
    /// Token 42 leads token 43 by one logit, so a penalty of more than one on
    /// 42 flips the greedy pick — the pick says whether the count reached the
    /// kernel.
    #[test]
    fn the_kernel_prices_each_row_from_its_own_resident_counts() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return; // No CUDA device on this box.
        };
        let sampler = BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        );
        let mut row = vec![0.0f32; VOCAB_SIZE];
        row[42] = 10.0;
        row[43] = 9.0;
        let logits = |rows: usize| {
            Tensor::from_vec(row.repeat(rows), (rows, VOCAB_SIZE), &device).expect("logits")
        };

        // Frequency: 42 said three times costs it 3 × 2.0.
        let freq = SamplingConfig {
            frequency_penalty: 2.0,
            ..SamplingConfig::argmax()
        };
        let mut repeated = make_state();
        let mut fresh = make_state();
        let mut in_call = make_state();
        for _ in 0..3 {
            repeated.record_token(42, MAX_RECENT);
            in_call.record_token(42, MAX_RECENT);
        }
        in_call.in_tool_call = true;
        let tokens = sampler
            .sample_batch(
                &logits(3),
                &mut [&mut repeated, &mut fresh, &mut in_call],
                &[&freq, &freq, &freq],
            )
            .expect("sample_batch");
        assert_eq!(tokens, vec![43, 42, 42]);

        // The next dispatch puts a fresh row where the repeated one was: it must
        // read zero there, not the counts the last dispatch stamped.
        let mut after = make_state();
        let tokens = sampler
            .sample_batch(&logits(1), &mut [&mut after], &[&freq])
            .expect("sample_batch");
        assert_eq!(tokens, vec![42]);

        // Cross-turn: 42 said in a prior turn costs it 5.0 this turn.
        let cross = SamplingConfig {
            cross_turn_penalty: 5.0,
            ..SamplingConfig::argmax()
        };
        let mut said_before = make_state();
        said_before.record_token(42, MAX_RECENT);
        said_before.end_turn(0);
        let mut never = make_state();
        let tokens = sampler
            .sample_batch(
                &logits(2),
                &mut [&mut said_before, &mut never],
                &[&cross, &cross],
            )
            .expect("sample_batch");
        assert_eq!(tokens, vec![43, 42]);
    }
}
