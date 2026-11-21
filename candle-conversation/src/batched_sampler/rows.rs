//! The row descriptors of one dispatch, and the table of full-vocabulary rows that the kernel reads in place.
//!
//! `RowSpec` describes one row of a dispatch, `RowKind` says how the row's token is committed, and `Scored` holds a row's pick before it is committed.
//! `KernelRows` gathers the full-vocabulary rows of one dispatch, each read off its sequence's state as it stands, for the kernel's inputs.

use super::*;

/// One row of a dispatch ([`BatchedSampler::score_rows`]).
#[derive(Debug, Clone, Copy)]
pub struct RowSpec<'a> {
    /// The sequence it samples: an index into the dispatch's states.
    pub seq: usize,
    /// The row of the logits block it reads.
    pub logits_row: usize,
    /// The proposal a speculative verify row tests.
    pub draft: Option<u32>,
    /// The tokens the sequence commits before this row within the step —
    /// empty for its first row. A sequence's rows each extend the previous
    /// one's by a token.
    pub ahead: &'a [u32],
}

/// How a row's token is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowKind {
    /// Sampled over the vocabulary, then resolved — segment close, the EOS
    /// failsafes — and tracked for segment edges.
    Free,
    /// Forced or drawn from a grammar's allow-list: recorded as it stands.
    Constrained,
}

/// A row's pick, scored and not yet committed
/// ([`BatchedSampler::commit_row`]).
#[derive(Debug, Clone, Copy)]
pub struct Scored {
    pub(super) raw: u32,
    pub(super) kind: RowKind,
    pub(super) rng_after: u64,
    pub(super) key: RowKey,
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
