//! The per-sequence sampling state: the degenerate-run bar, the rolling token history, turn and segment tracking, and the penalty windows.

use super::*;

/// Per-sequence sampling state.
///
/// Tracks token counts and recent history for penalty calculations.
/// Consecutive token-0 emissions that mark a decode as degenerate rather than
/// merely repetitive. Comfortably above anything language produces — token 0 is
/// `!` in the Qwen vocab and no real text repeats it eight times — while short
/// enough that a broken forward is caught in a few steps instead of running to
/// the length cap.
pub const DEGENERATE_TOKEN_RUN: u32 = 8;

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
    pub(super) counted: Vec<u32>,

    /// The tokens whose [`Self::cross_turn_counts`] entry is nonzero.
    pub(super) cross_counted: Vec<u32>,

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

    /// Tokens written so far in the free-text value being decoded — DRY's
    /// look-back window, the kernel's `dry_lens`. Zero outside a value.
    pub dry_span_len: i32,

    /// True while a grammar is decoding a free-text value, outside the think
    /// block — the only place DRY applies ([`crate::SamplingConfig::dry`]).
    /// Synced from the stencil each decode step ([`Self::set_free_text`]).
    pub in_free_text: bool,

    /// True while this sequence is inside a TOOL CALL specifically (not the think
    /// block).  The remaining repetition penalties (repeat/frequency/presence)
    /// are suppressed for these rows: a tool call's arguments legitimately
    /// reproduce prompt content verbatim — the query's numbers, file paths,
    /// identifiers — which those penalties would otherwise demote, corrupting the
    /// value.  Reasoning (the think block) retains full repetition control.
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

    /// The turn's run of token 0 reached [`DEGENERATE_TOKEN_RUN`] — sticky
    /// until the turn ends, whatever is committed after it. The run itself
    /// resets on the next other token, and the guard's own forced EOS is one:
    /// a speculative step that commits the run's last token and that EOS
    /// together leaves the run at 0 by the time the step is checked, and a
    /// turn of `!` was sealed as an answer. The decode loop reads this.
    pub degenerate: bool,

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
            in_free_text: false,
            in_tool_call: false,
            writing_call: false,
            close_script_pos: None,
            close_would_continue: false,
            degenerate_run: 0,
            degenerate: false,
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
            if self.degenerate_run >= DEGENERATE_TOKEN_RUN {
                self.degenerate = true;
            }
        } else {
            self.degenerate_run = 0;
        }

        // Advance the segment length while inside a segment.
        if self.in_segment {
            self.segment_len += 1;
        }

        // Advance the DRY window while a free-text value is being written, so it
        // counts exactly that value's tokens.
        if self.in_free_text {
            self.dry_span_len += 1;
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
        self.cross_turn_history.clear();
        self.counted.clear();
        self.cross_counted.clear();
        self.recent_tokens.clear();
        self.current_len = 0;
        self.in_segment = false;
        self.segment_len = 0;
        self.dry_span_len = 0;
        self.in_free_text = false;
        self.in_tool_call = false;
        self.writing_call = false;
        self.close_script_pos = None;
        self.close_would_continue = false;
        self.degenerate_run = 0;
        self.degenerate = false;
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
        // A new turn is in no value yet; any open segment or in-flight closer
        // script from the prior turn is cleared.
        self.dry_span_len = 0;
        self.in_free_text = false;
        self.writing_call = false;
        self.in_segment = false;
        self.segment_len = 0;
        self.close_script_pos = None;
        self.close_would_continue = false;
        self.degenerate_run = 0;
        self.degenerate = false;
    }

    /// Advance RNG offset (called after each sampling).
    pub fn advance_rng(&mut self) {
        self.rng_offset = self.rng_offset.wrapping_add(1);
    }

    /// Open a segment (the caller signals this when the segment-open token is
    /// sampled), restarting the per-segment length.
    pub fn enter_segment(&mut self) {
        self.in_segment = true;
        self.segment_len = 0;
        // A fresh segment cannot inherit a closer script from a previous one.
        self.close_script_pos = None;
    }

    /// Close the segment (the caller signals this when the segment-close token is
    /// sampled).
    pub fn exit_segment(&mut self) {
        self.in_segment = false;
        self.segment_len = 0;
        // The segment is closed; any in-flight closer script is finished or moot.
        self.close_script_pos = None;
    }

    /// Whether a grammar is decoding a free-text value now, as the stencil
    /// stands after the token just committed — which [`Self::record_token`]
    /// has already counted.
    ///
    /// **Each value is its own window.** Entering one starts DRY's look-back
    /// at the one token that opened it; leaving it empties the window. A
    /// value never sees the structure around it, the reasoning before it, or
    /// the value before it.
    pub fn set_free_text(&mut self, live: bool) {
        match (self.in_free_text, live) {
            (false, true) => self.dry_span_len = 1,
            (_, false) => self.dry_span_len = 0,
            (true, true) => {}
        }
        self.in_free_text = live;
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

    /// Everything a hypothetical advance through a block's drafts changes,
    /// saved so [`Self::rewind`] can undo it exactly.
    pub(super) fn mark(&self) -> SamplingMark {
        SamplingMark {
            counted_len: self.counted.len(),
            recent_tokens: self.recent_tokens.clone(),
            current_len: self.current_len,
            in_segment: self.in_segment,
            segment_len: self.segment_len,
            dry_span_len: self.dry_span_len,
            close_script_pos: self.close_script_pos,
            degenerate_run: self.degenerate_run,
            degenerate: self.degenerate,
            rng_offset: self.rng_offset,
        }
    }

    /// Undo every advance since `mark`, given the tokens those advances
    /// recorded.
    ///
    /// The dense counts are not saved, only walked back: each recorded token
    /// had its count raised by one, and a token first seen since the mark was
    /// appended to `counted` past the length the mark kept — so lowering the
    /// counts and truncating the list restores both exactly.
    pub(super) fn rewind(&mut self, mark: SamplingMark, recorded: &[u32]) {
        for &t in recorded {
            if let Some(c) = self.token_counts.get_mut(t as usize) {
                *c -= 1;
            }
        }
        self.counted.truncate(mark.counted_len);
        self.recent_tokens = mark.recent_tokens;
        self.current_len = mark.current_len;
        self.in_segment = mark.in_segment;
        self.segment_len = mark.segment_len;
        self.dry_span_len = mark.dry_span_len;
        self.close_script_pos = mark.close_script_pos;
        self.degenerate_run = mark.degenerate_run;
        self.degenerate = mark.degenerate;
        self.rng_offset = mark.rng_offset;
    }

    /// The fields a row's dispatch inputs and its commit read, as they stand.
    /// Two states with the same key score and commit a row identically.
    pub(super) fn row_key(&self) -> RowKey {
        RowKey {
            counted: self.counted.len(),
            current_len: self.current_len,
            in_segment: self.in_segment,
            segment_len: self.segment_len,
            dry_span_len: self.dry_span_len,
            in_free_text: self.in_free_text,
            in_tool_call: self.in_tool_call,
            writing_call: self.writing_call,
            close_script_pos: self.close_script_pos,
            degenerate_run: self.degenerate_run,
            rng_offset: self.rng_offset,
            last: self.recent_tokens.last().copied(),
        }
    }

    /// True when the most recent token ends a sentence (`.`, `!`, `?`) or a
    /// line. The graceful EOS waits for this, so an answer is not cut
    /// mid-sentence.
    pub(super) fn at_sentence_end(&self, config: &SamplingConfig) -> bool {
        self.recent_tokens.last().is_some_and(|&t| {
            config.sentence_end_token_ids.contains(&t) || ends_a_line(&config.line_end_token_ids, t)
        })
    }

    /// True when the most recent token ends a line. The segment closes wait
    /// for this — see [`SamplingConfig::line_end_token_ids`] for why a line
    /// rather than a sentence.
    pub(super) fn at_line_end(&self, config: &SamplingConfig) -> bool {
        self.recent_tokens
            .last()
            .is_some_and(|&t| ends_a_line(&config.line_end_token_ids, t))
    }
}
