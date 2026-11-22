//! The per-turn decode state the scheduler carries from a turn's first generated
//! token to its seal.
//!
//! A turn's generated run has three states: committed, forwarded, and waiting. A
//! count of forwarded tokens says whether anything is waiting, so a token the
//! sampler dropped cannot be sent a second time. The reasoning span is recorded
//! as the decode goes, from the first close marker only, because the seal needs
//! one position to bound the block. A turn that faults is ended with its error and
//! discarded rather than sealed, since non-finite activations sealed into the
//! conversation would corrupt every later turn that attends over them. A page cut
//! is recorded at each break token and taken only once any speculative rewind has
//! settled, because a cut inside that window would count the same rows twice.

use super::*;

/// A finished turn between its disposition and its seal — what
/// [`Scheduler::cleanup_finished`] carries across the batched closing-tail forward.
pub(super) struct FinishedTurn {
    pub(super) state: DecodeState,
    /// The reply as the caller receives it: the written half, then the decoded
    /// one, split where the model wrote tags ([`TurnResponse::answer`]).
    pub(super) answer: TurnText,
    pub(super) decode_ms: f64,
    pub(super) total_ms: f64,
    pub(super) tokens_generated: usize,
    pub(super) tokens_per_second: f64,
    /// Read before the view was finalized, which drops the view slot's stats.
    pub(super) sequence_stats: SequenceStats,
    pub(super) context_tokens: usize,
    /// The slot the turn seals from — the parent, once the view is finalized.
    pub(super) seal_slot: SequenceId,
    pub(super) seal_block_from: usize,
}

/// Per-sequence state while actively generating tokens.
pub(super) struct DecodeState {
    /// Channel back to the caller.
    pub(super) event_tx: Sender<TurnEvent>,
    /// Tokens generated so far.
    pub(super) generated_tokens: TokenBuffer,
    /// Maximum tokens to generate.
    pub(super) max_tokens: usize,
    /// Sampling configuration for this turn.
    pub(super) sampling_config: SamplingConfig,
    /// What the scheduler does in `cleanup_finished` — see [`SealAction`].
    /// The `(layer, group, timeline)` to write to is looked up from
    /// [`Scheduler::slot_targets`] at seal time, not carried here.
    pub(super) seal_action: SealAction,
    /// Index into [`Self::generated_tokens`] of this turn's `</think>`, when one
    /// has been decoded.
    ///
    /// **The turn's reasoning span, recorded as it happens.** The span the seal
    /// needs is `[assistant_content_start, here + 1)` — the assistant body up to
    /// and including the close marker — so one position settles it, and any
    /// preamble the model emits before `<think>` is inside the reasoning region
    /// where it belongs rather than needing a second marker to bound it.
    ///
    /// Only the FIRST close is recorded. A `</think>` in the answer body (the
    /// sampler bans one outside a think block, but a think-steer tree can drop
    /// the model's own and play its own close) must not move a boundary that is
    /// already fixed.
    pub(super) think_close_at: Option<u32>,
    /// How many of [`Self::generated_tokens`] have actually been **forwarded**.
    ///
    /// **The invariant this replaces was implicit, and one path broke it.** The
    /// scheduler took each decode step's input from `generated_tokens.last()`,
    /// on the understanding that the last entry is always the one token
    /// committed-but-not-yet-forwarded. A suppressed `</think>` breaks it: the
    /// token is sampled and *dropped* rather than committed, so `last()` still
    /// names the token this step already forwarded — and
    /// `play_stencil_runs`, re-driven by the same drop, forwards it a
    /// second time.
    ///
    /// The K/V absorbs that silently, because it is position-addressed: the
    /// sequence offset never advanced, so the re-forward overwrites the same
    /// slot and the length does not change. The index does not — `append_wave`
    /// appends a row unconditionally. Measured on a think-steered turn: 47
    /// tokens forwarded, 59 tokens' worth of index rows, `turn_token_count` and
    /// the token grid agreeing at the lower number, and the surplus surfacing
    /// many layers later as a turn whose pages do not cover it.
    ///
    /// Counting forwards explicitly makes the question answerable instead of
    /// assumed: what is pending is `generated_tokens[forwarded_generated..]`,
    /// which is empty exactly when there is nothing to forward.
    ///
    /// **It answers "is anything waiting", and nothing else.** It is maintained
    /// by the paths that forward, so a path that forwards without touching it
    /// reads low — which is survivable for the pending question (the caller
    /// takes the last entry, still the right token) and is NOT survivable for
    /// sizing the sealed grid. Sealing from it produced a grid of 287 against a
    /// K/V of 289 on a measured turn, breaking the 1:1 contract `token_ids`
    /// exists to keep; the grid's authority is the K/V, and the seal drops
    /// exactly the trailing unforwarded token instead.
    pub(super) forwarded_generated: usize,
    /// How deep this turn drafts: one past its own running acceptance, clipped
    /// at the model's ladder for the wave's width. A turn writing free prose
    /// keeps ~2 drafted tokens a step and a turn quoting a file keeps nearly all
    /// of them, so a depth set once per wave pays for proposals the first
    /// throws away.
    pub(super) draft_depth: DraftDepth,
    /// A page cut this turn has earned but not yet taken.
    ///
    /// **A cut may not land between a speculative wave's snapshot and its
    /// rewind.** `IndexCache::snapshot` captures the live tail only — not the
    /// pages — so a cut in that window moves the tail's rows into a page, and
    /// the rewind then restores the tail from the snapshot while the page still
    /// holds the same rows. They are counted twice, and the index ends exactly
    /// one drafted block past the K/V for the rest of the sequence. Measured at
    /// a constant +5 against a 5-token block, on a full accept as much as a
    /// partial one.
    ///
    /// The rollback runs *after* the commit loop for its own reasons (a stash
    /// span is good for one step), so the commit cannot move; the cut does. It
    /// is recorded here and taken by [`Scheduler::flush_page_cut`] once the
    /// rewind has settled — which is also the only point at which the tail is
    /// the accepted prefix rather than the drafted block.
    pub(super) pending_page_cut: bool,
    /// The break token the pending cut is following — the predecessor whose
    /// identity armed it, or `None` for the turn's first token.
    ///
    /// Diagnostic only. "A page closed here" is not enough to debug a
    /// misaligned reasoning span: the question is always *which* break the cut
    /// was answering, and by flush time the run has moved on.
    pub(super) pending_page_cut_after: Option<u32>,
    /// Whether this sequence has finished (EOS or max_tokens).
    pub(super) finished: bool,
    /// The turn ended on a fault ([`DecodeState::fail`]): its caller has the
    /// error, and what it decoded is discarded rather than sealed.
    ///
    /// A fault is a forward or a sampler that could not go on — non-finite
    /// logits, a refused tier past its retries, a failed reprojection — and
    /// what the view holds past that point is not the reply it was meant to
    /// be. Sealed, it became the conversation's history: K/V computed from
    /// non-finite activations, and a recurrent state that had advanced over
    /// them, moved onto the parent. Every later turn of that conversation
    /// attended over it and failed the same way, and so did every other
    /// conversation that recalled it — measured: both npcs failing every turn
    /// for the rest of the process after one row went non-finite.
    pub(super) faulted: bool,
    /// Why it finished, once it has. [`FinishReason::Length`] only when the
    /// response budget ran out, set where the budget is checked; every other
    /// ending — end-of-sequence, a completed stencil, an abort — is a `Stop`.
    pub(super) finish: FinishReason,
    /// Decode start time (for stats).
    pub(super) decode_start: Instant,
    /// Microseconds this turn actually spent INSIDE decode forwards.
    ///
    /// Not the same as `decode_start.elapsed()`, which is turn wall-clock and
    /// therefore includes every wave the turn merely *survived* — prefill
    /// slices, other sequences' decode slices, relief passes, persistence
    /// stalls. Dividing tokens by that reports a latency figure under a
    /// throughput label: a 200-token summary alive for 60 s of wall-clock read
    /// as "3 tok/s" no matter how fast the model actually stepped.
    ///
    /// Each co-batched forward adds its FULL duration to every participating
    /// sequence (not a 1/N share): the question this answers is "while this turn
    /// was being decoded, how fast did it go", and a forward costs nearly the
    /// same whether it carries one sequence or eight. Summing to more than
    /// wall-clock across sequences is expected and correct.
    pub(super) decode_busy_us: u64,
    /// Prefill duration (for stats).
    pub(super) prefill_ms: f64,
    /// The sequence's KV depth when the turn's prefill finished — its projected
    /// context plus this turn's prompt. Recorded for the health diagnostic dump
    /// so that step depth can be interpreted relative to the context window size.
    pub(super) prefill_token_count: usize,
    /// Turn start time (for total_ms stat).
    pub(super) turn_start: Instant,
    /// Per-sequence decode health tracking state.
    pub(super) health: DecodeHealthState,
    /// Continuous re-projection policy.  When `Some`, the scheduler fires a
    /// view swap at each boundary its rule fires.  Carried across mid-decode
    /// swaps unchanged.
    pub(super) reprojection: Option<ReprojectionPolicy>,
    /// Where the reprojection boundary rule stands: the content counted since
    /// the last boundary, and whether decode is inside a `<tool_call>` body,
    /// where every trigger is suppressed so the generic call body can't
    /// re-orient the committed tool (`provenance::boundaries`). Migrates with
    /// the decode state across view swaps.
    pub(super) boundary: BoundaryState,
    /// Generated-token index of the previous projection point, so a reprojection
    /// event's span is `[last_projection_end, current_gen)`. Lives on the decode
    /// state (which migrates correctly across reprojection view swaps), unlike the
    /// separate view-keyed anchor map.
    pub(super) last_projection_end: u32,
    /// Trailing structural tokens forwarded into the slot after
    /// decode finishes, before the seal.  Today's seal path leaves
    /// this empty (the `assistant_end` boundary is a live
    /// `Generated` segment at projection time, not part of the
    /// persisted turn).  Kept on `DecodeState` for the no-decode
    /// `insert_turn` path that still needs to close its own brackets.
    pub(super) post_decode_tokens: TokenBuffer,
    /// Online belief accumulated across this turn's reprojections. Each
    /// reprojection seeds `project()` from it and writes the result back, so the
    /// RelLeak decay/reinforcement carries across the turn (§80). Migrates with
    /// the decode state through view swaps.
    pub(super) belief: PriorBelief,
    /// Full prefill token sequence pinned into the slot at this
    /// turn's submit:
    /// `[user_msg][user_end][assistant_start][lead]` — the lead is whatever the
    /// dialect's thinking suppression puts at the head of the assistant turn
    /// (`Dialect::thinking_suppression`, `turn_submitter::assistant_lead`) and
    /// the turn's assistant prefill. A thinking turn's lead holds no think block:
    /// it opens its own `<think>`, which lands in `generated_tokens`. A suppressed
    /// turn on a family with a `/no_think` switch carries the switch in the user
    /// opener and decodes its own empty block; on a family without one (Qwen3.5,
    /// Qwen3.8) the lead IS the closed block, so it is baked here, as the
    /// template renders `enable_thinking=false`.
    /// Concatenated with `generated_tokens` and `post_decode_tokens`
    /// at seal time to form `TurnContent::token_ids` — the
    /// cross-process replay sequence persisted to the redo log.
    pub(super) prefill_tokens: TokenBuffer,
    /// The raw user message string — passed through from
    /// `submit_turn` verbatim (no role-marker envelope, no
    /// `/no_think` prefix).  Stored directly on the substrate at
    /// seal time as `TurnPart::user_text` so the sidebar reload
    /// path never has to re-tokenise or scan for boundary markers.
    pub(super) user_text: String,
    /// Gather-scope tags for this turn, from the submit request. Persisted onto
    /// the turn's `TurnDecl` at seal so a projection policy's `tags:` filter can
    /// scope its provenance gallery. Empty for live/untagged turns.
    pub(super) tags: Vec<String>,
    /// Content boundaries that frame the user-message body and the
    /// assistant-response body inside the sealed grid.  Set from the submit
    /// path (CPU-tokenised against the prefill prefix strings) and used at seal
    /// time to build the turn's [`TurnLayout`] so the compressor can window
    /// content-only halves on demand.
    pub(super) user_content_start: u32,
    pub(super) user_content_end: u32,
    pub(super) assistant_content_start: u32,
    /// Whether this turn was submitted with thinking SUPPRESSED (the composer
    /// `/no_think` dial active).  Recorded into the turn's [`TurnLayout`] at
    /// seal so the projection re-injects the `/no_think` soft-switch into this
    /// turn's user opener when it re-renders the turn as history.
    pub(super) no_think: bool,
    /// The prefilled assistant half (repo_map / code_reading ingest), stored
    /// verbatim as `TurnPart::assistant_text` at seal time. Empty for decode
    /// turns, where the seal uses the model's decoded text instead.
    pub(super) prefill_assistant_text: String,
    /// Whether this turn has written a tool call — its open tag sampled, or a
    /// tool-call stencil walked to the turn's end. Its caller acts on the
    /// `Done` (it runs the tool and submits the result), so the finished turn's
    /// cleanup is not deferred behind the turns still decoding.
    pub(super) wrote_tool_call: bool,
    /// Whether this turn's schema asked for repetition penalties to be lifted
    /// while a tool call runs. Off unless the schema names it — see
    /// [`crate::projection::Schema::free_tool_calls_from_penalties`].
    pub(super) free_tool_calls_from_penalties: bool,
    /// Tool-call stencils available this turn, keyed by trigger token.  Empty
    /// registry = free decode.  Carried from the turn's `SubmitTurn`.
    pub(super) triggers: Arc<TriggerRegistry>,
    /// The constrained-decoding walk currently in progress (a `<tool_call>` is
    /// mid-emission).  `None` whenever the model is free-decoding — a genuine
    /// present/absent, not a feature flag.
    pub(super) stencil: Option<StencilDriver>,
    /// The decode action (`Branch`/`Free`/`Done`) the stencil yielded for the
    /// upcoming decode step.  Set by `play_stencil_runs` once any preceding
    /// static runs are injected; consumed by `batch_decode_step` (apply the mask,
    /// then `accept` the sampled token and clear).  `None` ⇒ the driver needs
    /// advancing again.
    pub(super) pending_mask: Option<StepMask>,
    /// The ids this turn commits in place of what it samples, when it replays
    /// a recorded turn ([`crate::TurnOptions::recorded_turn`]); the end of turn
    /// follows the last. Everything else about the step is a live decode's —
    /// the page cuts, the reasoning boundary, the reprojections — which is what
    /// makes the sealed turn the recorded one. `None` = a sampled decode.
    pub(super) recorded_reply: Option<Replay>,
}

impl DecodeState {
    /// Whether this turn drafts at its next step: not a replay, whose next token
    /// is its recording's, and not held to a plain row by its grammar
    /// ([`block_guard::BlockGuard::drafts`]). The one rule the step and a tier
    /// price over it both read.
    pub(super) fn drafts(&self) -> bool {
        self.recorded_reply.is_none()
            && block_guard::BlockGuard::drafts(self.stencil.as_ref(), self.pending_mask.as_ref())
    }

    /// End the turn on a fault: the caller gets `error`, and `cleanup_finished`
    /// discards what the turn decoded instead of sealing it ([`Self::faulted`]).
    /// Once is enough — a later fault on the same turn adds nothing the caller
    /// can act on.
    pub(super) fn fail(&mut self, error: ConversationError) {
        if self.faulted {
            return;
        }
        let _ = self.event_tx.send(TurnEvent::Error(error));
        self.faulted = true;
        self.finished = true;
    }

    /// Append one token to the turn's generated run, recording the reasoning
    /// boundary as it goes past.
    ///
    /// **Every token enters `generated_tokens` through here**, because they
    /// arrive by three different routes and the boundary has to be seen on all
    /// of them: the sampler's own commit, an assistant prefill prefix, and a
    /// stencil's static run — which is where a think-steered `</think>` actually
    /// lands, the tree having suppressed the model's own. Recording at the
    /// sampler alone would leave every steered turn with no span at all.
    ///
    /// `think_close` is [`Scheduler::think_close`], passed rather than read
    /// because the caller already holds a mutable borrow of this state.
    ///
    /// **Private, because the answer to "does this open a page?" is only usable
    /// together with the model.** The two ways to call it say when the token is
    /// forwarded relative to the call, which is the whole of what decides
    /// whether a cut is possible: [`commit_generated`] for a token about to be
    /// forwarded, [`Self::push_forwarded`] for one already carried.
    pub(super) fn push_generated(
        &mut self,
        token: u32,
        think_close: Option<u32>,
        breaks: &[u32],
    ) -> bool {
        let closes_page = self.opens_index_page(breaks);
        // Before the push: the index of the token being appended. Only the
        // first close counts — see `think_close_at`. This still tracks
        // `</think>` alone, because it locates the reasoning SPAN for the
        // layout, which is a different question from where a page ends.
        if self.think_close_at.is_none() && Some(token) == think_close {
            self.think_close_at = Some(self.generated_tokens.len() as u32);
        }
        self.generated_tokens.push(token);
        closes_page
    }

    /// Record a token whose forward has **already happened**.
    ///
    /// A cut is not merely unnecessary here, it would be wrong: the token's
    /// index row is appended by the forward that carried it, so closing now
    /// would leave that row on the page the new region was supposed to start
    /// after. The static-run path forwards its whole run and then records it,
    /// and [`Scheduler::run_prefill`] has already split that forward at the
    /// reasoning boundary — the only moment the cut could land.
    pub(super) fn push_forwarded(&mut self, token: u32, think_close: Option<u32>, breaks: &[u32]) {
        let _ = self.push_generated(token, think_close, breaks);
        // Already carried, so it is forwarded the moment it is recorded.
        self.forwarded_generated = self.generated_tokens.len();
    }

    /// Record a token that is committed but **not yet forwarded** — it rides the
    /// next decode step, which cuts at its own commit.
    ///
    /// The third of the three states a recorded token can be in, and the one
    /// that was previously assumed rather than tracked: exactly the trailing
    /// entry, always. A suppressed token leaves none in this state, which is why
    /// the assumption had to become a count.
    pub(super) fn push_pending(&mut self, token: u32, think_close: Option<u32>, breaks: &[u32]) {
        let _ = self.push_generated(token, think_close, breaks);
    }

    /// Record a token the caller is about to forward, and note the page cut if
    /// it opens a new region.
    ///
    /// **The cut is recorded, not taken.** It must not land between a
    /// speculative wave's snapshot and its rewind — see
    /// [`Self::pending_page_cut`] — so the caller takes it with
    /// [`Scheduler::flush_page_cut`] once the K/V is final. The third of the
    /// three ways a token enters the run, and the only one that can earn a cut:
    /// the other two are already forwarded ([`Self::push_forwarded`]) or
    /// deliberately waiting ([`Self::push_pending`]).
    pub(super) fn push_committed(&mut self, token: u32, think_close: Option<u32>, breaks: &[u32]) {
        let prev = self.generated_tokens.last().copied();
        if self.push_generated(token, think_close, breaks) {
            self.pending_page_cut = true;
            self.pending_page_cut_after = prev;
        }
    }

    /// The committed tokens still waiting to be forwarded.
    ///
    /// Normally the single token the last commit produced; empty after a
    /// suppressed token, because nothing was committed and the previous entry
    /// went out with the step that sampled it.
    pub(super) fn pending_forward(&self) -> &[u32] {
        &self.generated_tokens[self.forwarded_generated.min(self.generated_tokens.len())..]
    }

    /// Record that everything committed so far has now been forwarded.
    pub(super) fn mark_forwarded(&mut self) {
        self.forwarded_generated = self.generated_tokens.len();
    }

    /// Whether the token about to be pushed begins a new region of the turn's
    /// index, so the pages either side of it can be handed over — or dropped —
    /// independently.
    ///
    /// **The timing is off by one from where this is asked**, and that is the
    /// whole of the subtlety. A token committed at step *t* is *forwarded* at
    /// step *t+1*, and a token's index row is appended by the forward that
    /// carries it. The cut is deferred within step *t* — past the speculative
    /// rewind, see [`Self::pending_page_cut`] — but still lands before step
    /// *t+1*'s append, so `token` ends up at the head of the new page.
    ///
    /// Two regions open, and neither test mentions `<think>`: the reasoning
    /// region starts where the assistant body does, not at the open marker, so
    /// any preamble the model emits before it is inside the reasoning where it
    /// belongs.
    ///
    /// | This token | Opens |
    /// |---|---|
    /// | the turn's first | the reasoning page, at the prefill/decode boundary |
    /// | the one after any break token | the next region's page |
    ///
    /// Keyed off the **previous token's identity**, read straight from the run.
    /// The earlier form asked whether `think_close_at + 1` had been reached,
    /// which made the cut depend on the span having been located — so a turn
    /// whose `</think>` was prefilled or played by a steering tree, rather than
    /// sampled, recorded no boundary and cut nothing.
    pub(super) fn opens_index_page(&self, breaks: &[u32]) -> bool {
        match self.generated_tokens.last() {
            None => true,
            Some(prev) => breaks.contains(prev),
        }
    }
}

/// Close `slot`'s index at a unit boundary — the shared body of
/// [`Scheduler::begin_unit`], reachable from the projection assembler, which
/// holds the model but not the scheduler.
///
/// `held` is the slot's token length, for the log only: it says whether the
/// closed width accounts for everything standing on the slot or only part of it.
pub(in super::super) fn close_unit_boundary(
    model: &(dyn ManagedBatchedModel + Send),
    slot: SequenceId,
    unit: &'static str,
    held: usize,
) {
    match model.close_positional_page(slot.0) {
        Ok(0) => {}
        Ok(closed) => tracing::info!(
            target: "candle_conversation::scheduler::unit_boundary",
            seq_id = slot.0,
            unit,
            closed,
            held,
            "index: closed the prefix into a page at this unit's boundary"
        ),
        Err(e) => tracing::warn!(
            seq_id = slot.0,
            unit,
            "closing the index page at a unit boundary failed ({e}); this unit's rows will \
             be sealed together with the prefix ahead of it"
        ),
    }
}

/// Take a page cut [`commit_generated`] recorded, if any.
///
/// Call once the sequence's K/V is final for the step: after a speculative
/// rollback, or immediately when none is possible. Closing earlier double-counts
/// the flushed rows, because the rewind's snapshot does not cover pages.
pub(super) fn flush_page_cut(
    model: &(dyn ManagedBatchedModel + Send),
    slot: SequenceId,
    state: &mut DecodeState,
) {
    if !take(&mut state.pending_page_cut) {
        return;
    }
    let after = state.pending_page_cut_after.take();
    match model.close_positional_page(slot.0) {
        // Every reasoning block closes a page: an event, read when a page's
        // bounds are in question.
        Ok(closed) => note("index: closed a page at a reasoning boundary", || {
            format!(
                "seq_id={} closed={closed} generated={} after={after:?} think_close_at={:?}",
                slot.0,
                state.generated_tokens.len(),
                state.think_close_at,
            )
        }),
        Err(e) => tracing::warn!(
            seq_id = slot.0,
            "closing the index page at a reasoning boundary failed ({e}); this turn's \
             reasoning will not occupy whole pages and cannot be windowed out of a later \
             projection"
        ),
    }
}
