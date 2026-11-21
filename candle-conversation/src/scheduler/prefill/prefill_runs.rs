//! Forwards a prefill's tokens for one sequence, cut at each reasoning or turn
//! boundary so every boundary lands on a page edge.
//!
//! A span's index rows are pooled by the forward that carries them. A pooled
//! block that crossed a boundary cannot be split afterwards, and the reasoning
//! would not occupy whole pages, so the span is cut after each break token that
//! has something following it and the pieces are forwarded in order. Spans from
//! several sequences that finish together share waves, one wave per split round,
//! and a failure fails only the turn it belongs to. Whether an assistant lead
//! leaves a think block open is decided by its most recent marker, and only the
//! assistant lead is scanned, so quoted text in the user half cannot arm it.

use super::*;

impl Scheduler {
    /// Forward `tokens` on `sequence_id`, splitting the pass at the turn's
    /// reasoning boundary if it falls inside them.
    ///
    /// **A forward must not carry tokens from both sides of `</think>`.** A
    /// span's index rows are pooled by the forward that carries it, so a block
    /// pooled across the boundary cannot be un-pooled at seal time and the
    /// reasoning would not occupy whole pages. Plain decode never straddles —
    /// one token per sequence per wave — but a static run does: a think-steer
    /// tree suppresses the model's own `</think>` and injects the closing tag as
    /// a run, which can carry tokens after it.
    ///
    /// Splitting here rather than at the two call sites is what makes the
    /// invariant structural: this is the one function that forwards an arbitrary
    /// token span for a single sequence, so a future third caller inherits it.
    pub(in super::super) fn run_prefill(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<(), ConversationError> {
        self.run_prefill_pieces(sequence_id, tokens, Continuation::Ignore)
            .map(|_| ())
    }

    /// [`Self::run_prefill`], then the continuation's first token sampled under
    /// `config` — **from the logits where the head wrote them**, while the
    /// forward still holds its span. Copying the row off the span to sample it
    /// afterwards was a vocabulary-wide device allocation per prefill.
    pub(in super::super) fn run_prefill_sampled(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
        config: &SamplingConfig,
        state: &mut SequenceSamplingState,
    ) -> Result<u32, ConversationError> {
        match self.run_prefill_pieces(sequence_id, tokens, Continuation::Sample(config, state))? {
            Some(Continued::Token(t)) => Ok(t),
            _ => Err(ConversationError::Channel(
                "prefill sampled no token".into(),
            )),
        }
    }

    pub(super) fn run_prefill_pieces(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
        mut continuation: Continuation<'_>,
    ) -> Result<Option<Continued>, ConversationError> {
        // **Every break token in the span, not just the first.** A prefilled
        // assistant head carries `<think>` and `</think>` in one pass, and a
        // multi-turn prefill carries a turn closer as well — splitting once
        // would leave the later markers pooled across their own boundaries,
        // which is not correctable afterwards.
        let pieces = break_pieces(tokens, self.prefill_breaks());
        let mut continued = None;
        for (i, piece) in pieces.iter().enumerate() {
            let last = i + 1 == pieces.len();
            // Only the last piece's row is the span's continuation.
            let here = if last {
                std::mem::replace(&mut continuation, Continuation::Ignore)
            } else {
                Continuation::Ignore
            };
            continued = self.run_prefill_span(sequence_id, piece, here)?;
            if !last {
                self.close_page_at_break(sequence_id, piece);
            }
        }
        Ok(continued)
    }

    /// Forward several sequences' spans together: the same pieces, page closes
    /// and order per sequence as [`Self::run_prefill`] gives each alone, with one
    /// wave per split round carrying every sequence that has a piece in it.
    ///
    /// **This is what lets turns that finish together close together.** Every
    /// finished turn forwards its dialect's closing marker before it seals, and on
    /// a dialect whose closer is a page-break token that is two forwards per turn
    /// — one for the marker, one for what follows — of a single row each. Run one
    /// turn at a time they cost ~33 ms per turn on the loop thread, measured on
    /// the Qwen3.6-35B-A3B with eight turns finishing in two groups; batched, a
    /// group pays two waves.
    ///
    /// A span longer than one forward's budget goes through [`Self::run_prefill`]
    /// on its own, which knows how to chunk it.
    ///
    /// Returns the turns whose span failed, each with its error. **A failure is
    /// the failing turn's alone**, as it is when each turn forwards by itself: a
    /// wave that fails stops advancing the members it carried, and every other
    /// turn's span still completes. Sequences are grouped by adapter, because a
    /// wave carries one adapter or none (see `form_wave_group`) — mixing them is
    /// refused by the forward for the whole wave. A sequence named twice keeps
    /// its spans in order: the repeat forwards after the batch, on its own,
    /// since one wave cannot carry a sequence twice.
    pub(in super::super) fn run_prefill_batch(
        &mut self,
        spans: &[(SequenceId, &[u32])],
    ) -> Vec<(SequenceId, ConversationError)> {
        let pass = self.prefill_pass_budget();
        let mut failed: Vec<(SequenceId, ConversationError)> = Vec::new();
        let mut groups: Vec<(Option<String>, Vec<(SequenceId, Vec<&[u32]>)>)> = Vec::new();
        let mut alone: Vec<(SequenceId, &[u32])> = Vec::new();
        let mut seen: Vec<SequenceId> = Vec::with_capacity(spans.len());
        for &(sequence_id, tokens) in spans {
            let pieces = break_pieces(tokens, self.prefill_breaks());
            let repeat = seen.contains(&sequence_id);
            seen.push(sequence_id);
            if repeat || pieces.iter().any(|p| p.len() > pass) {
                alone.push((sequence_id, tokens));
                continue;
            }
            let adapter = self
                .session
                .sequence_adapter(sequence_id.0)
                .map(str::to_owned);
            match groups.iter_mut().find(|(a, _)| *a == adapter) {
                Some((_, members)) => members.push((sequence_id, pieces)),
                None => groups.push((adapter, vec![(sequence_id, pieces)])),
            }
        }
        for (_, members) in groups {
            self.run_prefill_rounds(&members, pass, &mut failed);
        }
        for (sequence_id, tokens) in alone {
            // A turn whose earlier span failed does not forward the next one
            // over the gap.
            if failed.iter().any(|(s, _)| *s == sequence_id) {
                continue;
            }
            if let Err(e) = self.run_prefill(sequence_id, tokens) {
                failed.push((sequence_id, e));
            }
        }
        failed
    }

    /// One adapter group of [`Self::run_prefill_batch`]: a wave per split round
    /// and per budget's worth of rows, a failed wave's members left out of every
    /// round after it and recorded in `failed`.
    pub(super) fn run_prefill_rounds(
        &mut self,
        members: &[(SequenceId, Vec<&[u32]>)],
        pass: usize,
        failed: &mut Vec<(SequenceId, ConversationError)>,
    ) {
        let rounds = members.iter().map(|(_, p)| p.len()).max().unwrap_or(0);
        let nl = self.model.num_layers().max(1);
        let mut dead: Vec<SequenceId> = Vec::new();
        for round in 0..rounds {
            let live: Vec<(SequenceId, &[u32], bool)> = members
                .iter()
                .filter(|(seq, _)| !dead.contains(seq))
                .filter_map(|(seq, pieces)| {
                    pieces
                        .get(round)
                        .map(|piece| (*seq, *piece, round + 1 < pieces.len()))
                })
                .collect();
            // One wave per budget's worth of rows: a round carries one piece per
            // sequence, each within the budget, so only their sum can exceed it.
            let lens: Vec<usize> = live.iter().map(|(_, p, _)| p.len()).collect();
            let mut start = 0;
            while start < live.len() {
                let take =
                    super::super::admission::admit_within(lens[start..].iter().copied(), pass);
                let group = &live[start..start + take];
                start += take;
                if let Err(e) = self.forward_tail_wave(group, nl) {
                    let reason = e.to_string();
                    for &(sequence_id, _, _) in group {
                        dead.push(sequence_id);
                        failed.push((
                            sequence_id,
                            ConversationError::Other(format!(
                                "the closing-tail wave this turn rode failed: {reason}"
                            )),
                        ));
                    }
                }
            }
        }
    }

    /// Forward one wave of closing-tail pieces and advance each member past its
    /// piece, closing its page where the piece ends at a break.
    pub(super) fn forward_tail_wave(
        &mut self,
        group: &[(SequenceId, &[u32], bool)],
        nl: usize,
    ) -> Result<(), ConversationError> {
        let seqs: Vec<usize> = group.iter().map(|(s, _, _)| s.0).collect();
        let inputs = group
            .iter()
            .map(|(_, piece, _)| Tensor::new(*piece, &Device::Cpu).and_then(|t| t.unsqueeze(0)))
            .collect::<candle::Result<Vec<_>>>()
            .map_err(ConversationError::Model)?;
        let step = self
            .model
            .forward_wave(
                &mut self.session,
                &[],
                &[],
                &seqs,
                &inputs,
                &[],
                &[],
                0,
                nl,
                None,
            )
            .map_err(ConversationError::Model)?;
        drop(step);
        for &(sequence_id, piece, closes) in group {
            self.session
                .advance_sequence(sequence_id.0, piece.len())
                .map_err(ConversationError::Model)?;
            super::super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, piece);
            if closes {
                self.close_page_at_break(sequence_id, piece);
            }
        }
        Ok(())
    }

    /// The tokens a forwarded span is cut after so each boundary lands on a page
    /// edge — or none, for a model with no pages to land on.
    ///
    /// **The cut exists for the positional index alone.** A span's index rows are
    /// pooled by the forward that carries them, and only a model that keeps a
    /// per-position index ([`ManagedBatchedModel::carries_positional_state`]) has
    /// rows to pool; on every other model the page close at the cut is a no-op and
    /// the cut is a whole extra forward for nothing. Measured on the Qwen3.6-35B-A3B,
    /// which keeps none: every finished turn's `<|im_end|>\n` tail took two forwards,
    /// ~20 ms each, on the loop thread between the decode steps of the turns still
    /// running.
    pub(in super::super) fn prefill_breaks(&self) -> &[u32] {
        if self.model.carries_positional_state() {
            &self.page_break_tokens
        } else {
            &[]
        }
    }

    /// Close `sequence_id`'s index page after `head`, a piece that ended on a
    /// break token — so the region it bounds occupies whole pages.
    pub(in super::super) fn close_page_at_break(&mut self, sequence_id: SequenceId, head: &[u32]) {
        match self.model.close_positional_page(sequence_id.0) {
            // The head's own token ids when it is short. A surplus page is
            // identified by what is IN it, and the pages that do not belong
            // to any turn are consistently 5 and 7 tokens wide — small
            // enough to name outright rather than infer from their width.
            // Debug: every prefilled turn closes one.
            Ok(closed) => tracing::debug!(
                target: "candle_conversation::scheduler::unit_boundary",
                seq_id = sequence_id.0,
                site = "prefill-break-token",
                closed,
                at = head.len(),
                head = ?(head.len() <= 16).then_some(head),
                break_token = ?head.last(),
                "index: closed a page mid-prefill at a break token"
            ),
            Err(e) => tracing::warn!(
                seq_id = sequence_id.0,
                "closing the index page at a prefilled break token failed ({e}); the \
                 region it bounds will not occupy whole pages and cannot be windowed \
                 out of a later projection"
            ),
        }
    }

    /// Forward `tokens` for one sequence, in chunks of the pass budget so the
    /// activation buffers stay bounded, and — when `sample` is given — sample
    /// the last chunk's scored row while its forward still holds the span.
    ///
    /// The ids are handed over **on the host**: the model lays every row's id
    /// on its forward span in one upload, so the scheduler allocates nothing on
    /// the device for them.
    pub(super) fn run_prefill_span(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
        mut continuation: Continuation<'_>,
    ) -> Result<Option<Continued>, ConversationError> {
        let pass = self.prefill_pass_budget().max(1);
        let nl = self.model.num_layers().max(1);
        let n_chunks = tokens.len().div_ceil(pass).max(1);
        let mut continued = None;
        for k in 0..n_chunks {
            let chunk = &tokens[(k * pass).min(tokens.len())..((k + 1) * pass).min(tokens.len())];
            let input = Tensor::new(chunk, &Device::Cpu)
                .and_then(|t| t.unsqueeze(0))
                .map_err(ConversationError::Model)?;
            let step = self
                .model
                .forward_wave(
                    &mut self.session,
                    &[],
                    &[],
                    &[sequence_id.0],
                    &[input],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )
                .map_err(ConversationError::Model)?;
            if k + 1 == n_chunks {
                let first_row = || {
                    step.logits_on_span().into_iter().next().ok_or_else(|| {
                        ConversationError::Channel("no logits returned from prefill".into())
                    })
                };
                match std::mem::replace(&mut continuation, Continuation::Ignore) {
                    Continuation::Ignore => {}
                    Continuation::Sample(config, state) => {
                        let row = first_row()?;
                        continued =
                            Some(Continued::Token(self.sample_single(&row, config, state)?));
                    }
                }
            }
            drop(step);
            self.session
                .advance_sequence(sequence_id.0, chunk.len())
                .map_err(ConversationError::Model)?;
            // Mirror these tokens into the slot's diagnostic log so the
            // turn-complete dump can reconstruct the exact context the kernel
            // saw (compiled out without the `context-dump` feature).
            super::super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, chunk);
        }
        // Nothing to bring up to date here: the prefill's own commit
        // (`KvCache::commit_written_tokens`, the one place every write outside
        // the decode kernel goes through) marked the cached decode slot buffer,
        // and the next sync that reads it re-serialises its writer region.
        Ok(continued)
    }
}

/// What a forwarded span's last row is wanted for.
pub(super) enum Continuation<'a> {
    /// Nothing: the span only writes its K/V.
    Ignore,
    /// The continuation's first token, sampled in place.
    Sample(&'a SamplingConfig, &'a mut SequenceSamplingState),
}

/// What [`Continuation`] produced.
pub(super) enum Continued {
    Token(u32),
}

/// `tokens` cut one past every break token that has something after it, so each
/// reasoning or turn boundary lands on a page edge — the pieces a span is
/// forwarded in, in order.
///
/// A break token at the very end makes no cut: nothing follows it in this pass,
/// so the next forward is already the edge. Always at least one piece, and an
/// empty span is one empty piece — a forward of nothing, exactly as an unsplit
/// span of nothing was.
///
/// **Reads the token stream, not the decode state.** A turn's `DecodeState` does
/// not exist while it prefills — it is built from the prefill's own logits — so
/// the stream is the only place a `<think>…</think>` block baked into the prompt
/// can be seen in time to split the pass.
pub(super) fn break_pieces<'a>(tokens: &'a [u32], breaks: &[u32]) -> Vec<&'a [u32]> {
    let mut pieces = Vec::new();
    let mut rest = tokens;
    while let Some(at) = rest
        .iter()
        .position(|t| breaks.contains(t))
        .map(|i| i + 1)
        .filter(|&at| at < rest.len())
    {
        let (head, tail) = rest.split_at(at);
        pieces.push(head);
        rest = tail;
    }
    pieces.push(rest);
    pieces
}

/// Whether an assistant prefill leaves a think block **open** at the point
/// decode takes over.
///
/// **Containing `<think>` is not the question.** Qwen3.5 suppresses reasoning by
/// prefilling an already-closed block, `<think>\n\n</think>\n\n`, so the open
/// marker is in every suppressed turn's prefill by construction — and the
/// sampler's segment state only ever sees *sampled* tokens, so the prefilled
/// close never reaches it. Asked "is `<think>` among the last few tokens", the
/// check answered yes on exactly the turns where thinking had been turned off,
/// and the sampler then decoded the whole answer believing it was inside a
/// think block: the ban on `</think>` outside a block lifted, and any
/// `force_segment_close_after` fired a closer into the prose. Measured on an
/// unstencilled reflection turn: `The belt is</think>`, nine tokens, the answer
/// ended by a forced close of a block that had closed before it began.
///
/// The most recent marker decides, which is the rule that holds whatever the
/// prefill is: a closed block, a closed block followed by a tool-call opener, a
/// bare `<think>` for a turn that is meant to reason, or an earlier turn's
/// markers further back in the buffer. The scan stops at the first marker from
/// the end, so it costs a handful of comparisons on every real prefill.
/// **The caller decides what is scanned, and it hands over the ASSISTANT lead
/// only.** The markers are ordinary vocabulary ids and the tokenizer emits them
/// for literal text, so a `<tool_response>` carrying source that merely mentions
/// `<think>` puts the open id in the USER half of the grid — this repo's own
/// `dialect.rs` does exactly that, and the code-reading ingest feeds it back.
/// Scanning a whole prefill would arm `in_segment` off that quoted text before
/// the turn had decoded anything; under `ThinkMode::Off`, where the hard cap is
/// one token, the second decoded token then gets rewritten to `</think>`. The
/// slice at `assistant_content_start` is what prevents it, and
/// `only_the_assistant_lead_is_scanned` holds the boundary.
pub(super) fn prefill_leaves_think_open<'a>(
    tokens: impl DoubleEndedIterator<Item = &'a u32>,
    open: u32,
    close: Option<u32>,
) -> bool {
    tokens
        .rev()
        .find(|&&t| t == open || Some(t) == close)
        .is_some_and(|&t| t == open)
}
