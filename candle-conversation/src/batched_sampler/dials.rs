//! Per-sequence sampling dials, the kernel's top-k and dead-zone constants, the typical-acceptance rule, and the kernel row table the launch reads.

use super::*;

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
pub(super) struct SeqDials {
    pub(super) temperature: f32,
    pub(super) top_k: i32,
    pub(super) top_p: f32,
    pub(super) repeat_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: i32,
    pub(super) dry_range: i32,
    pub(super) eos_boost: f32,
    pub(super) eos_ramp_start: i32,
    pub(super) eos_ramp_len: i32,
    pub(super) eos_boost_max_multiplier: f32,
    pub(super) cross_turn_penalty: f32,
    pub(super) segment_close_boost: f32,
    pub(super) segment_close_token_id: i32,
    pub(super) segment_close_ramp_start: i32,
    pub(super) segment_close_ramp_len: i32,
    pub(super) segment_close_max_multiplier: f32,
    pub(super) segment_temp_boost: f32,
    pub(super) typical_draft: i32,
    pub(super) typical_eps: f32,
    pub(super) typical_delta: f32,
    /// The row of the logits block this entry samples. A dispatch whose other
    /// rows are resolved on the host (a grammar allow-list) hands the kernel
    /// the whole block and names each kernel row's source here, rather than
    /// gathering the kernel rows into a compacted copy first — a copy that, on
    /// a wave's head span, would claim a block of logits the wave never priced.
    pub(super) logits_row: i32,
    /// The device address of the row's allowed-token bitmap, low word first, or
    /// zero for none (`SamplingConfig::allowed_bits`).
    pub(super) allowed_bits_lo: u32,
    pub(super) allowed_bits_hi: u32,
}

// The kernel reads this struct as a flat run of 4-byte words (all fields are
// f32/i32), one per row of the batch, and casts back to the identically-laid-out
// CUDA `SeqDials`. If the size or field count drifts from the CUDA side the
// kernel reads a row at the wrong stride, so pin it: 27 fields × 4 bytes.
const _: () = assert!(std::mem::size_of::<SeqDials>() == 108);

impl SeqDials {
    /// Read one row's dials from its config, resolving the same Option/gate logic
    /// the scalar path applies (DRY defaults, the dynamic-EOS gate, the
    /// segment-close-active gate) so a per-row wave behaves identically to a
    /// uniform one row-for-row. The row samples `logits_row` of the dispatch's
    /// logits block, and tests no draft until [`Self::with_draft`] gives it one.
    pub(super) fn from_config(c: &SamplingConfig, logits_row: usize) -> Self {
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
            logits_row: logits_row as i32,
            allowed_bits_lo: 0,
            allowed_bits_hi: 0,
        }
    }

    /// This row is a speculative verify row testing `draft` on `typical`.
    /// A `None` draft leaves the row a plain one.
    pub(super) fn with_draft(mut self, draft: Option<u32>, typical: TypicalAcceptance) -> Self {
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
pub(super) const KERNEL_MAX_TOP_K: usize = 256;

/// The kernel's candidate floor (`radix_select_logit_threshold`'s
/// `DEAD_ZONE`): no candidate sits more than this far below the row's best.
pub(super) const KERNEL_DEAD_ZONE: f32 = 50.0;

/// The typical-acceptance rule over one row of logits: true when `draft`'s
/// probability exceeds `min(ε, δ·e^(−H))`. The CPU twin of the kernel's
/// `typical_accepts`, measured over the distribution the kernel samples — the
/// top-k candidates (at most [`KERNEL_MAX_TOP_K`], none more than
/// [`KERNEL_DEAD_ZONE`] below the best), softmaxed at `temperature`, cut to
/// the nucleus at `top_p`, renormalised. A draft outside it has probability
/// zero.
pub(super) fn typical_accepts_row(
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

/// The full-vocabulary rows of one dispatch, each read off its sequence's
/// state as it stood when the row was scored ([`BatchedSampler::score_rows`]),
/// in the arrays the kernel takes.
pub(super) struct KernelRows<'a> {
    pub(super) vocab: usize,
    pub(super) max_recent_len: usize,
    /// Whether the cross-turn table is gathered: some row carries the penalty.
    pub(super) cross_turn: bool,
    /// Each row's index in the dispatch's output and the state it was read
    /// from.
    pub(super) scored: Vec<(usize, RowKey)>,
    pub(super) configs: Vec<&'a SamplingConfig>,
    pub(super) counts: SparseCounts,
    pub(super) cross: SparseCounts,
    pub(super) recent_tokens: Vec<i32>,
    pub(super) recent_lens: Vec<i32>,
    pub(super) current_lens: Vec<i32>,
    pub(super) segment_lens: Vec<i32>,
    pub(super) dry_lens: Vec<i32>,
    pub(super) banned: Vec<(&'a [i32], Option<i32>)>,
    pub(super) rng_offsets: Vec<u64>,
    pub(super) suppress_penalties: Vec<f32>,
    pub(super) seq_dials: Vec<SeqDials>,
    /// The bitmaps of the rows that carry one, a vocabulary's width each.
    pub(super) allowed_bits: Vec<u32>,
    /// Those rows, in the order of their bitmaps.
    pub(super) masked: Vec<usize>,
}

impl<'a> KernelRows<'a> {
    pub(super) fn new(vocab: usize, max_recent_len: usize, cross_turn: bool) -> Self {
        Self {
            vocab,
            max_recent_len,
            cross_turn,
            scored: Vec::new(),
            configs: Vec::new(),
            counts: SparseCounts::default(),
            cross: SparseCounts::default(),
            recent_tokens: Vec::new(),
            recent_lens: Vec::new(),
            current_lens: Vec::new(),
            segment_lens: Vec::new(),
            dry_lens: Vec::new(),
            banned: Vec::new(),
            rng_offsets: Vec::new(),
            suppress_penalties: Vec::new(),
            seq_dials: Vec::new(),
            allowed_bits: Vec::new(),
            masked: Vec::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.scored.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.scored.is_empty()
    }

    /// Read `row`'s inputs off `state` as it stands. `out` is the row's index
    /// in the dispatch's output and `key` the state's.
    pub(super) fn push(
        &mut self,
        out: usize,
        state: &SequenceSamplingState,
        config: &'a SamplingConfig,
        row: &RowSpec<'_>,
        typical: Option<TypicalAcceptance>,
        key: RowKey,
    ) {
        let r = self.len();
        // Inside a TOOL CALL, all repetition penalties are suppressed, not just
        // DRY.  Tool-call arguments legitimately reproduce content verbatim from
        // the prompt or an earlier span — the query's numbers, file paths,
        // identifiers — so frequency/presence/repeat penalties (which see the
        // `<think>`/prior-span tokens via `token_counts` and `recent_tokens`)
        // would demote exactly those tokens, corrupting the value.  This mirrors
        // the DRY gate but is scoped to tool calls only (`in_tool_call`), so the
        // think block keeps full repetition control.  Presenting empty penalty
        // state for these rows is the per-row equivalent of turning them off:
        // the row stamps nothing, so its table row reads zero. A step the
        // grammar marks unpenalized (`SamplingConfig::unpenalized_step`) is
        // lifted the same way, for that step alone.
        let lifted = state.in_tool_call || config.unpenalized_step;
        if !lifted {
            self.counts
                .push_row(self.vocab, r, &state.counted, &state.token_counts);
            if self.cross_turn {
                self.cross.push_row(
                    self.vocab,
                    r,
                    &state.cross_counted,
                    &state.cross_turn_counts,
                );
            }
        }

        // The recent-token window, `max_recent_len` wide per row.
        //
        // The window must be large enough for BOTH penalties that use this buffer:
        //   • Repeat penalty needs `repeat_last_n` tokens.
        //   • DRY penalty needs `dry_range` tokens (kernel line:
        //       search_start = recent_len - dry_range when dry_range < recent_len).
        // Using only `repeat_last_n` here would silently cap DRY to that smaller
        // window even when dry_range >> repeat_last_n, causing cross-turn phrases
        // to slide out of view before the model reaches the position where it
        // repeats them.  Take the max of both requirements so each penalty sees
        // the context depth it was configured for. Both are the dispatch's first
        // row's, as the kernel's scalar defaults are.
        let first = self.configs.first().copied().unwrap_or(config);
        let dry_range = first.dry.as_ref().map_or(0, |d| d.range);
        let repeat_win = if first.repeat_last_n > 0 {
            first.repeat_last_n as usize
        } else {
            self.max_recent_len
        };
        let dry_win = if dry_range > 0 {
            (dry_range as usize).min(self.max_recent_len)
        } else {
            0
        };
        let total = state.recent_tokens.len();
        let window = repeat_win.max(dry_win).min(total);
        // Inside a tool call, present a zero-length repeat window so the repeat
        // penalty sees no history (DRY is already gated via `dry_lens`).  Tool
        // arguments must be free to reproduce the query's numbers/paths/names
        // verbatim. The buffer is still padded to keep the batch stride fixed.
        self.recent_lens
            .push(if lifted { 0 } else { window as i32 });
        self.recent_tokens
            .extend_from_slice(&state.recent_tokens[total - window..]);
        self.recent_tokens
            .extend(std::iter::repeat_n(0, self.max_recent_len - window));

        // Current generated length (for dynamic EOS ramp). A sequence writing a
        // tool call reports 0, which holds its ramp at zero boost — see
        // `SequenceSamplingState::writing_call`.
        self.current_lens.push(if state.writing_call {
            0
        } else {
            state.current_len
        });
        self.segment_lens.push(if state.in_segment {
            state.segment_len
        } else {
            0
        });
        // DRY's window (the kernel's `dry_lens`): the free-text value being
        // written, and 0 — DRY off — anywhere else or on a lifted row.
        // Live only where the config has DRY at all: a config that cleared it
        // (zend's coding row) keeps the repeat penalty it tuned for no DRY.
        let dry_live = state.in_free_text && !lifted && config.has_dry();
        self.dry_lens
            .push(if dry_live { state.dry_span_len } else { 0 });
        // The row's OWN deny-list, plus the structural think-close ban for a row
        // outside a block (`think_close_ban_active` — a `</think>` outside a
        // think block is never valid output). A ban is row-specific: the answer
        // that closes a stuck tool loop bans `<tool_call>` for itself alone. See
        // `banned_rows`.
        let close = think_close_ban_active(config, state).then_some(config.segment_close_token_id);
        self.banned.push((config.banned_tokens.as_slice(), close));
        self.rng_offsets.push(state.rng_offset);
        self.suppress_penalties
            .push(config.segment_suppress_penalty);
        // **Per-sequence dials — every row samples on its own config.** The
        // kernel's scalar arguments are only the null-fallback defaults; it reads
        // its dials from this array, which is what stops one row's EOS ramp (or
        // temperature, or penalties) bleeding into another in a wave that mixes
        // configs. A verify row also carries the draft it tests.
        // The repeat penalty flips with DRY: where DRY is live it is the
        // config's own value for that ([`SamplingConfig::repeat_penalty_for`]).
        let dials = SeqDials {
            repeat_penalty: config.repeat_penalty_for(dry_live),
            ..SeqDials::from_config(config, row.logits_row)
        };
        if let Some(bits) = &config.allowed_bits {
            self.masked.push(r);
            let width = self.vocab.div_ceil(32);
            let end = self.allowed_bits.len() + width;
            self.allowed_bits
                .extend_from_slice(&bits[..bits.len().min(width)]);
            self.allowed_bits.resize(end, 0);
        }
        self.seq_dials.push(match typical {
            Some(typical) => dials.with_draft(row.draft, typical),
            None => dials,
        });
        self.configs.push(config);
        self.scored.push((out, key));
    }
}
