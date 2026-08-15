//! The async summariser thread (`docs/immutable_summary_forest.md`).
//!
//! Spawned alongside the persistence thread at engine start.  Mirrors its
//! trigger/tick/shutdown idiom.  Builds the per-timeline **append-only
//! immutable forest** (a ternary Merkle Mountain Range): drains pending Normal
//! turns into `SummaryOfTurns` leaves, runs the ternary carry to seal
//! `SummaryOfSummaries` internals, and persists each node's `TreeMetadata`.
//! Nodes are never rewritten, so there is no `dirty` bit and no regeneration.
//!
//! ```text
//!   run_pass(timeline):
//!     1. drain pending (high priority): each Normal → §6 probe → new SoT leaf
//!        → carry: while the last MERGE_FANOUT peaks share a level, §6 probe a
//!        SoS over them.
//!     2. reconcile (low priority, only when nothing pending): build at most
//!        one missing internal node toward the canonical shape — backfills a
//!        forest loaded from disk or migrated from the superseded AVL.
//! ```
//!
//! Failures:
//! - `ProbeError::Soft` → re-enqueue the failed children, log, keep
//!   running.  Next pass tries again.
//! - `ProbeError::Hard` → log, return from `run_loop`.  Engine
//!   shutdown proceeds normally.

use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam::channel::{self, Receiver, Sender};

use crate::persistence::record::TreeMetadataPayload;
use crate::projection::{Conversation, TimelineId, TurnIndex};
use crate::scheduler::SchedulerRequest;
use crate::substrate::TreeNodeMeta;
use crate::summary_tree::exchange;
use crate::summary_tree::probe::{ProbeError, ProbeRequest, ProbeResponse, ProbeRunner};
use crate::summary_tree::tree::carry_run;

/// How many of a timeline's newest Normal turns stay **verbatim** — no
/// exchange whose last member sits inside this tail is absorbed into a
/// summary leaf, however settled it is.
///
/// Compression exists to bound old context, and it trades fidelity for room —
/// a trade that is all loss on the conversation's working tail, where the
/// model needs its own recent exchanges word-for-word (to quote them, list
/// them, or follow up on them) and where a few turns of raw dialogue cost
/// almost nothing. `exchange::is_settled` alone protects only the frontier
/// exchange, so the moment a new exchange opened, the previous one became
/// absorbable: measured, a four-question conversation had its first turn
/// compressed sixty-four seconds after it sealed — while the user was typing
/// question three — and by question five ("what did I ask you?") the
/// projection held digests where the dialogue had been, and the model
/// answered that it had no access to the conversation history.
///
/// Eight Normal turns is three to four exchanges — a tool round-trip spans
/// several turns. An **archived** timeline is exempt: it is terminal, nothing
/// will ever need its tail verbatim, and the peak cover wants it sealed.
const RAW_TAIL_TURNS: usize = 8;
use crate::summary_tree::TurnKind;

/// How often the summariser wakes up on its own when no triggers
/// arrive.  Short cadence (250 ms) so backpressure clears quickly
/// once the foreground tempo eases.
pub const SUMMARISER_TICK: Duration = Duration::from_millis(250);

/// Clone-able fire-and-forget trigger for the summariser thread.
/// Held by the scheduler so every assistant-turn seal can wake the
/// summariser without needing the [`SummariserThread`] handle.
#[derive(Clone)]
pub struct SummariserTrigger {
    tx: Sender<()>,
}

impl SummariserTrigger {
    /// Wake the summariser thread now.  No-op when the trigger queue
    /// is full (one wake is as good as several).
    pub fn fire(&self) {
        let _ = self.tx.try_send(());
    }

    /// Test-only no-op trigger.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        Self { tx }
    }
}

/// Handle to a running summariser thread.  Triggers wake the loop
/// early; [`Self::shutdown`] (or `Drop`) signals the loop, drains a
/// final pass, and joins the thread.
pub struct SummariserThread {
    handle: Mutex<Option<JoinHandle<()>>>,
    trigger_tx: Sender<()>,
    shutdown_tx: Sender<()>,
}

impl SummariserThread {
    /// Spawn the thread.  `runner` is the probe execution backend —
    /// either [`ChannelProbeRunner`] (production, scheduler-backed)
    /// or [`MockProbeRunner`] (tests).
    /// `max_concurrent` is how many probes a single pass submits at once
    /// (their decodes batch in the scheduler's wave loop). Chosen by total
    /// VRAM at engine init — see `summary_probe_concurrency`.
    pub fn spawn(
        conversation: Conversation,
        runner: Arc<dyn ProbeRunner>,
        max_concurrent: usize,
    ) -> Self {
        let (trigger_tx, trigger_rx) = channel::bounded::<()>(1);
        let (shutdown_tx, shutdown_rx) = channel::bounded::<()>(1);

        let max_concurrent = max_concurrent.max(1);
        let handle = std::thread::Builder::new()
            .name("substrate-summariser".into())
            .spawn(move || {
                run_loop(
                    conversation,
                    runner,
                    max_concurrent,
                    trigger_rx,
                    shutdown_rx,
                )
            })
            .expect("failed to spawn substrate-summariser thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
            shutdown_tx,
        }
    }

    /// A disabled handle: spawns no thread. `trigger`/`trigger_handle().fire()`
    /// and `shutdown` are no-ops (the receivers are dropped, so sends fail
    /// silently and there is no thread to join). Used when the engine is
    /// configured with `disable_summariser`.
    pub fn disabled() -> Self {
        let (trigger_tx, _) = channel::bounded::<()>(1);
        let (shutdown_tx, _) = channel::bounded::<()>(1);
        Self {
            handle: Mutex::new(None),
            trigger_tx,
            shutdown_tx,
        }
    }

    /// Wake the thread now.  No-op if a trigger is already pending or
    /// the thread is mid-pass.
    pub fn trigger(&self) {
        let _ = self.trigger_tx.try_send(());
    }

    pub fn trigger_handle(&self) -> SummariserTrigger {
        SummariserTrigger {
            tx: self.trigger_tx.clone(),
        }
    }

    /// Stop the thread: signal shutdown, wait for the loop to drain,
    /// then join.  Idempotent.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.try_send(());
        if let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Drop for SummariserThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_loop(
    conversation: Conversation,
    runner: Arc<dyn ProbeRunner>,
    max_concurrent: usize,
    trigger_rx: Receiver<()>,
    shutdown_rx: Receiver<()>,
) {
    loop {
        let mut shutting_down = false;
        crossbeam::channel::select! {
            recv(trigger_rx) -> _ => {}
            recv(shutdown_rx) -> _ => { shutting_down = true; }
            default(SUMMARISER_TICK) => {}
        }

        match run_pass(&conversation, runner.as_ref(), max_concurrent) {
            Ok(()) => {}
            // Only a hard error (GPU fault, scheduler shutdown) stops the
            // thread. A soft error means one timeline's pass couldn't
            // complete this round; it's already logged per-timeline in
            // run_pass — keep the loop alive so every other conversation
            // (and the next tick) still gets summarised.
            Err(ProbeError::Hard(msg)) => {
                tracing::warn!(
                    target: "candle_conversation::summariser",
                    "summariser stopped (hard error): {msg}"
                );
                return;
            }
            Err(ProbeError::Soft(msg) | ProbeError::Permanent(msg)) => {
                // Permanent is handled per-turn inside `run_pass` and doesn't
                // reach here; kept exhaustive so a future propagation path can't
                // silently stop the thread.
                tracing::warn!(
                    target: "candle_conversation::summariser",
                    "summariser pass soft-failed, continuing: {msg}"
                );
            }
        }

        if shutting_down {
            return;
        }
    }
}

/// One full summariser pass: drain pending turns into the summary tree,
/// then clear at most one dirty bit per timeline.
///
/// Returns `Err(ProbeError::Hard(...))` to abort the run loop.
/// Soft probe failures are logged and the affected turns are re-queued
/// onto the pending queue for the next pass.
pub fn run_pass(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    max_concurrent: usize,
) -> Result<(), ProbeError> {
    // Engine-internal (reserved) conversations — the titler especially —
    // accumulate substrate turns as they work but have no user-facing
    // projection/summary to compress against, so any probe soft-fails. Exclude
    // them from the sweep outright; otherwise every pass re-enqueues a doomed
    // compression and floods the log. Append-only ingest layers (repo_map,
    // code_reading) are excluded the same way: their turns never enqueue
    // (`push_pending_summary` refuses them), and skipping the sweep here also
    // keeps reconcile from probing forests these timelines are never meant to
    // grow — plus ignores any queue entries persisted before the exclusion.
    let timeline_ids: Vec<TimelineId> = {
        let guard = conversation.read();
        let all: Vec<TimelineId> = guard.all_timeline_ids().collect();
        all.into_iter()
            .filter(|t| {
                !guard.timeline_target(*t).is_some_and(|(layer, _)| {
                    layer.is_reserved() || guard.is_append_only_layer(layer)
                })
            })
            .collect()
    };
    for timeline in &timeline_ids {
        let pending = conversation.read().pending_summary_len(*timeline);
        if pending > 0 {
            tracing::trace!(
                target: "candle_conversation::summariser",
                timeline = %timeline,
                pending,
                "run_pass: timeline has pending normal turns to absorb",
            );
        }
    }
    for timeline in timeline_ids {
        // Per-timeline isolation: a soft failure on one timeline (e.g. a
        // corrupt summary forest from an older run failing to reconcile)
        // must not abort the whole pass — it would starve every other
        // timeline, including freshly-started conversations. Log it and move
        // on; only a hard error propagates and stops the thread.
        if let Err(e) =
            absorb_pending_turns(conversation, runner, timeline, max_concurrent, RAW_TAIL_TURNS)
        {
            match e {
                ProbeError::Hard(_) => return Err(e),
                ProbeError::Soft(msg) | ProbeError::Permanent(msg) => {
                    tracing::warn!(
                        target: "candle_conversation::summariser",
                        timeline = %timeline,
                        "absorb soft-failed, skipping timeline this pass: {msg}"
                    );
                    continue;
                }
            }
        }
        // Low-priority reconcile: only when nothing is pending, build at most one
        // missing internal node toward the canonical ternary shape. New turns
        // (high priority) keep the live frontier whole; this backfills a forest
        // loaded from disk (or migrated from the old AVL) without blocking them.
        if conversation.read().pending_summary_len(timeline) == 0 {
            if let Err(e) = reconcile_pass(conversation, runner, timeline) {
                match e {
                    ProbeError::Hard(_) => return Err(e),
                    ProbeError::Permanent(msg) => {
                        // Structural, won't self-heal (e.g. the target layer has no
                        // summary-of-summaries projection). Disarm reconcile for this
                        // timeline so we stop re-submitting a doomed probe every pass;
                        // it only re-arms on a fresh load via `mark_for_reconcile`.
                        conversation.write().set_needs_reconcile(timeline, false);
                        tracing::warn!(
                            target: "candle_conversation::summariser",
                            timeline = %timeline,
                            "reconcile permanently unavailable; disarming this timeline: {msg}"
                        );
                        continue;
                    }
                    ProbeError::Soft(msg) => {
                        tracing::warn!(
                            target: "candle_conversation::summariser",
                            timeline = %timeline,
                            "reconcile soft-failed, skipping timeline this pass: {msg}"
                        );
                        continue;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Drain every pending Normal turn for `timeline` into the summary
/// forest.  For each pending Normal:
///
/// 1. Run a probe over `[normal_idx]` to seal a fresh
///    `SummaryOfTurns` leaf turn.
/// 2. Write the leaf's `TreeNodeMeta` (kind, children = [normal_idx],
///    height = 1).
/// 3. Append the leaf as the rightmost peak, then carry-merge: while the
///    last `MERGE_FANOUT` peaks share a level, allocate one
///    `SummaryOfSummaries` over them via additional probes.
/// 4. Persist every change as `TreeMetadata` redo-log records.  Nodes are
///    immutable once written — no root pointer, no rotations, no dirty bit
///    (`docs/immutable_summary_forest.md`).
/// `raw_tail` is how many of the timeline's newest Normal turns stay verbatim
/// — production passes [`RAW_TAIL_TURNS`]; tests that exercise settledness and
/// batching pass `0` so their small fixtures stay legible.
fn absorb_pending_turns(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
    max_concurrent: usize,
    raw_tail: usize,
) -> Result<(), ProbeError> {
    loop {
        // Collect up to `max_concurrent` eligible pending Normal turns, so
        // their SoT probes can be submitted together and their decodes batch
        // in the scheduler's wave loop instead of running one at a time.
        // The exchange view: a leaf covers one complete exchange, not one turn.
        // A tool round-trip spans several Normal turns, and summarising the call
        // turn alone leaves the compressor with no answer among its children — so
        // it invents one. Rebuilt each pass: both inputs grow as turns land.
        let (normals, couplings, finalize) = {
            let read = conversation.read();
            let normals = read.normal_turns_chrono(timeline);
            let couplings = exchange::over_normals(&read.couplings_of(timeline), &normals);
            // Once archived, the conversation is terminal — the frontier exchange
            // will never gain a successor, so seal it now rather than leave a hole
            // in the peak cover.
            let finalize = read.is_archived(timeline);
            (normals, couplings, finalize)
        };
        let mut batch: Vec<Vec<TurnIndex>> = Vec::with_capacity(max_concurrent);
        let mut deferred: Vec<TurnIndex> = Vec::new();
        while batch.len() < max_concurrent {
            let normal_idx = match conversation.write().pop_pending_summary(timeline) {
                Some(idx) => idx,
                None => break,
            };
            // Skip if a previous pass already absorbed this index (e.g. via
            // restart-reload's re-enqueue + a partially-applied tree).
            let already_in_tree = conversation
                .read()
                .tree_meta_of(timeline, normal_idx)
                .map(|m| m.kind != TurnKind::Normal)
                .unwrap_or(false);
            if already_in_tree {
                tracing::trace!(
                    target: "candle_conversation::summariser",
                    timeline = %timeline,
                    normal = %normal_idx,
                    "absorb: turn already in tree, skipping",
                );
                continue;
            }
            let Some(pos) = normals.iter().position(|n| *n == normal_idx) else {
                // A pending index that is not in the timeline's Normal set. It was
                // enqueued as pending, so this drop loses a turn silently — log it
                // rather than swallow it. (Expected only if the turn was already
                // promoted out of Normal between pop and this read; anything else
                // is a real bug.)
                tracing::warn!(
                    target: "candle_conversation::summariser",
                    timeline = %timeline,
                    normal = %normal_idx,
                    n_normals = normals.len(),
                    "absorb: pending turn is not in the Normal set — dropping (should be rare)",
                );
                continue;
            };
            let group = exchange::exchange_of(&couplings, normals.len(), pos);
            // Only the head drives its exchange. The other members are pending
            // too, and pop later in index order; skipping them here is what stops
            // one exchange being summarised once per member.
            if group.start != pos {
                continue;
            }
            if !exchange::is_settled(&group, normals.len(), finalize) {
                // This is the frontier exchange — the live tail of the
                // conversation. A tool round-trip here may still be waiting on its
                // response, and its coupling may not be in the log yet, so sealing
                // now risks freezing a leaf over half an exchange. Leaf children
                // are frozen at creation, so we wait until a later turn exists
                // (which guarantees the exchange is complete and fully coupled —
                // see `exchange::is_settled`). Nothing is lost: the frontier is
                // anchored verbatim by recency until then.
                tracing::trace!(
                    target: "candle_conversation::summariser",
                    timeline = %timeline,
                    normal = %normal_idx,
                    "absorb: frontier exchange, deferring until a later turn settles it",
                );
                deferred.push(normal_idx);
                continue;
            }
            // Settled is necessary, not sufficient: the exchange must also have
            // left the conversation's verbatim tail (see [`RAW_TAIL_TURNS`]).
            // An archived timeline is terminal and compresses to the end.
            if !finalize && normals.len() - group.end < raw_tail {
                tracing::trace!(
                    target: "candle_conversation::summariser",
                    timeline = %timeline,
                    normal = %normal_idx,
                    "absorb: exchange still inside the verbatim tail, deferring",
                );
                deferred.push(normal_idx);
                continue;
            }
            batch.push(normals[group].to_vec());
        }
        // Re-enqueue deferred heads AFTER draining, so this pass can't pop the
        // same open exchange again and spin.
        if !deferred.is_empty() {
            let mut write = conversation.write();
            for idx in deferred {
                write.push_pending_summary(timeline, idx);
            }
        }
        if batch.is_empty() {
            return Ok(());
        }
        tracing::trace!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            batch = batch.len(),
            "absorb: probing SoT leaves for pending normal turns",
        );

        let requests: Vec<ProbeRequest> = batch
            .iter()
            .map(|group| ProbeRequest {
                timeline,
                kind: TurnKind::SummaryOfTurns,
                children: group.clone(),
                height: 1,
            })
            .collect();
        let results = runner.run_batch(requests);

        // Append the sealed leaves into the forest.  The probe decodes ran
        // concurrently; the append + carry-merge (which may emit serial
        // SoS-allocation probes) is applied one leaf at a time, in batch order — ascending
        // chronological order among the turns that succeeded.
        //
        // A soft-failed probe re-enqueues only that turn, to the *back* of the
        // pending queue so a persistently-failing turn can't head-of-line block
        // newer ones.  We still insert the later successes in this pass rather
        // than deferring them: their summary turns are already sealed in the
        // substrate, so deferring would leave them with default `Normal` tree
        // meta — orphans that pollute projections and get re-summarised.  The
        // re-enqueued turn is retried on a later pass and inserted then, which
        // can place its leaf after a younger sibling; that bounded local disorder
        // is the accepted cost of liveness + no orphans.
        let mut soft_failed = false;
        for (group, result) in batch.iter().zip(results) {
            // The exchange's head — the turn the pending queue handed us, and the
            // one re-enqueued if its probe soft-fails.
            let normal_idx = group[0];
            match result {
                Ok(sealed) => {
                    seal_leaf(conversation, timeline, sealed, group.clone())?;
                    carry_merge(conversation, runner, timeline)?;
                }
                Err(ProbeError::Soft(msg)) => {
                    tracing::warn!(
                        target: "candle_conversation::summariser",
                        timeline = %timeline,
                        normal = %normal_idx,
                        "soft probe error: {msg}; re-enqueueing"
                    );
                    conversation
                        .write()
                        .push_pending_summary(timeline, normal_idx);
                    soft_failed = true;
                }
                Err(ProbeError::Permanent(msg)) => {
                    // Retrying is futile (an empty compression half under argmax
                    // reproduces identically), so DO NOT re-enqueue — that would
                    // loop forever. Log an error and leave the turn a raw `Normal`
                    // node; it stays uncompressed but the conversation is correct.
                    tracing::error!(
                        target: "candle_conversation::summariser",
                        timeline = %timeline,
                        normal = %normal_idx,
                        "permanent probe error: {msg}; giving up — turn left un-summarised"
                    );
                }
                Err(e @ ProbeError::Hard(_)) => return Err(e),
            }
        }
        // Don't burn the queue in a tight loop on persistent soft failures —
        // the next tick / trigger retries the re-enqueued turns.
        if soft_failed {
            return Ok(());
        }
    }
}

/// Promote a freshly-sealed summary turn into a `SummaryOfTurns` leaf and
/// persist its `TreeMetadata`. Leaves are immutable — written exactly once.
fn seal_leaf(
    conversation: &Conversation,
    timeline: TimelineId,
    sealed: ProbeResponse,
    normal_children: Vec<TurnIndex>,
) -> Result<(), ProbeError> {
    let leaf_idx = sealed.sealed_turn;
    conversation.write().set_tree_meta(
        timeline,
        leaf_idx,
        TreeNodeMeta {
            kind: TurnKind::SummaryOfTurns,
            children: normal_children,
            tree_height: 1,
        },
    );
    persist_tree_meta(conversation, timeline, leaf_idx);
    Ok(())
}

/// Run the ternary carry off the substrate's current peaks: while the last
/// `MERGE_FANOUT` peaks share a level, seal a `SummaryOfSummaries` over them.
/// Peaks are recomputed each iteration so a cascading carry (e.g. at a power
/// of three) resolves fully. Existing nodes are never mutated.
fn carry_merge(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
) -> Result<(), ProbeError> {
    loop {
        let peaks = conversation.read().peaks_of(timeline);
        let levels: Vec<u8> = peaks.iter().map(|(_, l)| *l).collect();
        let Some(start) = carry_run(&levels) else {
            break;
        };
        let children: Vec<TurnIndex> = peaks[start..].iter().map(|(idx, _)| *idx).collect();
        let level = levels[start] + 1;
        build_sos(conversation, runner, timeline, children, level)?;
    }
    Ok(())
}

/// Seal one `SummaryOfSummaries` over `children` (a `MERGE_FANOUT`-run of
/// same-level peaks) via a §6 probe, write its immutable `TreeNodeMeta`, and
/// persist it. Returns the new node's turn index.
fn build_sos(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
    children: Vec<TurnIndex>,
    level: u8,
) -> Result<TurnIndex, ProbeError> {
    let probe = ProbeRequest {
        timeline,
        kind: TurnKind::SummaryOfSummaries,
        children: children.clone(),
        height: level,
    };
    let idx = runner.run(probe)?.sealed_turn;
    conversation.write().set_tree_meta(
        timeline,
        idx,
        TreeNodeMeta {
            kind: TurnKind::SummaryOfSummaries,
            children,
            tree_height: level,
        },
    );
    persist_tree_meta(conversation, timeline, idx);
    Ok(idx)
}

/// Build at most one missing internal node toward the canonical ternary shape,
/// clearing the reconcile hint once the forest is whole. One node per pass keeps
/// reconciliation low-priority. See `docs/immutable_summary_forest.md`.
fn reconcile_pass(
    conversation: &Conversation,
    runner: &dyn ProbeRunner,
    timeline: TimelineId,
) -> Result<(), ProbeError> {
    if !conversation.read().needs_reconcile(timeline) {
        return Ok(());
    }
    let Some(children) = conversation.read().reconcile_next(timeline) else {
        conversation.write().set_needs_reconcile(timeline, false);
        return Ok(());
    };
    let level = conversation
        .read()
        .tree_meta_of(timeline, children[0])
        .map(|m| m.tree_height)
        .unwrap_or(1)
        + 1;
    build_sos(conversation, runner, timeline, children, level)?;
    Ok(())
}

/// Persist one node's `TreeMetadata` redo-log record from its current
/// (immutable) substrate meta.
fn persist_tree_meta(conversation: &Conversation, timeline: TimelineId, idx: TurnIndex) {
    let meta = match conversation.read().tree_meta_of(timeline, idx).cloned() {
        Some(m) => m,
        None => return,
    };
    let payload = TreeMetadataPayload {
        timeline_id: timeline.raw(),
        turn_index: idx.0,
        kind: match meta.kind {
            TurnKind::Normal => 0,
            TurnKind::SummaryOfTurns => 1,
            TurnKind::SummaryOfSummaries => 2,
        },
        tree_height: meta.tree_height,
        children: meta.children.iter().map(|c| c.0).collect(),
    };
    if let Err(e) = conversation.write_tree_metadata(payload) {
        tracing::warn!(
            target: "candle_conversation::summariser",
            "write_tree_metadata failed for {idx}: {e}"
        );
    }
}

// ── ChannelProbeRunner ───────────────────────────────────────────────

/// Production-quality `ProbeRunner` backed by a `SchedulerRequest`
/// channel.  Sends [`SubmitSummaryProbe`](SchedulerRequest::SubmitSummaryProbe)
/// to the scheduler and blocks for the response.
///
/// Construction is `pub(crate)` because `SchedulerRequest` is a
/// crate-internal type — the engine wires this up at startup; callers
/// don't construct it directly.
pub struct ChannelProbeRunner {
    request_tx: Sender<SchedulerRequest>,
}

impl ChannelProbeRunner {
    pub(crate) fn new(request_tx: Sender<SchedulerRequest>) -> Self {
        Self { request_tx }
    }
}

impl ProbeRunner for ChannelProbeRunner {
    fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        let scheduler_request = SchedulerRequest::SubmitSummaryProbe {
            timeline: request.timeline,
            kind: request.kind,
            children: request.children.clone(),
            height: request.height,
            response_tx,
        };
        if let Err(e) = self.request_tx.send(scheduler_request) {
            return Err(ProbeError::Hard(format!("scheduler channel closed: {e}")));
        }
        match response_rx.recv() {
            Ok(Ok(turn_idx)) => Ok(ProbeResponse {
                sealed_turn: turn_idx,
            }),
            Ok(Err(pe)) => Err(pe),
            Err(e) => Err(ProbeError::Hard(format!("scheduler response channel: {e}"))),
        }
    }

    fn run_batch(&self, requests: Vec<ProbeRequest>) -> Vec<Result<ProbeResponse, ProbeError>> {
        // Submit every probe first (non-blocking sends) so the scheduler
        // registers all their decodes before the next decode quantum — they
        // then batch into a single forward instead of running one at a time.
        let mut receivers: Vec<Result<Receiver<Result<TurnIndex, ProbeError>>, ProbeError>> =
            Vec::with_capacity(requests.len());
        for request in requests {
            let (response_tx, response_rx) = crossbeam::channel::bounded(1);
            let scheduler_request = SchedulerRequest::SubmitSummaryProbe {
                timeline: request.timeline,
                kind: request.kind,
                children: request.children.clone(),
                height: request.height,
                response_tx,
            };
            match self.request_tx.send(scheduler_request) {
                Ok(()) => receivers.push(Ok(response_rx)),
                Err(e) => receivers.push(Err(ProbeError::Hard(format!(
                    "scheduler channel closed: {e}"
                )))),
            }
        }
        // Then collect, in submission order. Each `recv` blocks only until that
        // probe's batched decode completes.
        receivers
            .into_iter()
            .map(|r| match r {
                Ok(rx) => match rx.recv() {
                    Ok(Ok(turn_idx)) => Ok(ProbeResponse {
                        sealed_turn: turn_idx,
                    }),
                    Ok(Err(pe)) => Err(pe),
                    Err(e) => Err(ProbeError::Hard(format!("scheduler response channel: {e}"))),
                },
                Err(e) => Err(e),
            })
            .collect()
    }
}

// ── MockProbeRunner ──────────────────────────────────────────────────

/// Test-only `ProbeRunner` that appends a placeholder summary turn
/// directly to the substrate without invoking the model.  Returns the
/// new TurnIndex via [`crate::substrate::Substrate::append_with_blocks`].
///
/// Doesn't write content; the test asserts on tree structure, not on
/// the summary text.
pub struct MockProbeRunner {
    conversation: Conversation,
    summary_tokens: usize,
}

impl MockProbeRunner {
    pub fn new(conversation: Conversation) -> Self {
        Self {
            conversation,
            summary_tokens: 20,
        }
    }

    pub fn with_summary_tokens(mut self, n: usize) -> Self {
        self.summary_tokens = n;
        self
    }
}

impl ProbeRunner for MockProbeRunner {
    fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
        // Append a placeholder turn into the substrate at the request's
        // timeline.  Block range is 0..0 (no KV chunks); residence
        // stays cold — fine for unit tests that only exercise tree
        // bookkeeping.
        //
        // `append_with_blocks` pushes the new turn onto the pending
        // queue (its default classification is `Normal` until
        // `set_tree_meta` overwrites it).  That's fine: the summariser
        // loop will pop it back later, then short-circuit on the
        // `already_in_tree` check (the seal_leaf path has by then set
        // its kind to `SummaryOfTurns` / `SummaryOfSummaries`).  Don't
        // pop here — the queue is FIFO and popping would steal the
        // wrong entry.
        let idx = self.conversation.write().append_with_blocks(
            request.timeline,
            self.summary_tokens,
            0,
            0,
        );
        Ok(ProbeResponse { sealed_turn: idx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{Conversation, TimelineId, TurnIndex};
    use crate::summary_tree::tree::MERGE_FANOUT;
    use TurnKind;

    fn ephemeral_workspace() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn fresh_conversation(_workspace: &std::path::Path) -> (Conversation, TimelineId) {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let conversation = Conversation::ephemeral();
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        conversation
            .write()
            .register_timeline(timeline, layer, group);
        (conversation, timeline)
    }

    #[test]
    fn mock_runner_produces_a_fresh_turn_index() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let normal = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let _ = conv.write().pop_pending_summary(timeline); // drain
        let runner = MockProbeRunner::new(conv.clone());
        let resp = runner
            .run(ProbeRequest {
                timeline,
                kind: TurnKind::SummaryOfTurns,
                children: vec![normal],
                height: 1,
            })
            .unwrap();
        assert_eq!(resp.sealed_turn.0, 1);
    }

    /// Test runner that soft-fails the `SummaryOfTurns` probe whose first
    /// child is `fail_on`, delegating every other probe to a real
    /// [`MockProbeRunner`].  Used to exercise a mid-batch soft failure.
    struct FailOnChildRunner {
        inner: MockProbeRunner,
        fail_on: TurnIndex,
    }

    impl ProbeRunner for FailOnChildRunner {
        fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
            if request.kind == TurnKind::SummaryOfTurns
                && request.children.first() == Some(&self.fail_on)
            {
                return Err(ProbeError::Soft("injected mid-batch failure".into()));
            }
            self.inner.run(request)
        }
    }

    /// A soft failure on the *middle* turn of a batch must not orphan the
    /// later successes.  Their summary turns are already sealed, so they have
    /// to be AVL-inserted (kind = SoT) this pass; deferring them would leave
    /// them with default `Normal` meta — stray turns that pollute projections
    /// and get re-summarised.  Only the failed turn is re-enqueued.
    #[test]
    fn absorb_mid_batch_soft_failure_does_not_orphan_successes() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let n1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        let n2 = conv.write().append_with_blocks(timeline, 10, 2, 3);
        // A trailing turn so n2 is settled (not the deferred frontier exchange).
        let _sentinel = conv.write().append_with_blocks(timeline, 10, 3, 4);
        assert_eq!(conv.pending_summary_len(timeline), 4);

        let runner = FailOnChildRunner {
            inner: MockProbeRunner::new(conv.clone()),
            fail_on: n1,
        };
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");

        // Every SoT leaf's children, across the whole tree.
        let mut summarised: Vec<TurnIndex> = Vec::new();
        for i in 0..16u32 {
            if let Some(meta) = conv.read().tree_meta_of(timeline, TurnIndex(i)) {
                if meta.kind == TurnKind::SummaryOfTurns {
                    summarised.extend(meta.children.iter().copied());
                }
            }
        }
        assert!(
            summarised.contains(&n0),
            "n0 succeeded → must have a SoT leaf"
        );
        assert!(
            summarised.contains(&n2),
            "n2 succeeded → must have a SoT leaf"
        );
        assert!(
            !summarised.contains(&n1),
            "n1 failed → must not appear in any leaf"
        );

        // The failed turn is back on the pending queue for a later pass.
        let mut requeued_n1 = false;
        while let Some(idx) = conv.write().pop_pending_summary(timeline) {
            if idx == n1 {
                requeued_n1 = true;
            }
        }
        assert!(requeued_n1, "failed turn n1 must be re-enqueued");
    }

    /// A timeline with `summarize = false` — the gate set for utility/reference
    /// layers (repo_map, code_reading) — must not enqueue its sealed turns onto
    /// the pending-summary queue, so the summariser never touches them. A
    /// timeline left at the default (`true`) does enqueue.
    #[test]
    fn summarize_gate_off_skips_pending_enqueue() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        conv.write().set_timeline_summarize(timeline, false);
        let _n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        assert_eq!(
            conv.pending_summary_len(timeline),
            0,
            "summarize=false must not enqueue pending summaries"
        );

        let (conv2, timeline2) = fresh_conversation(tmp.path());
        let _m0 = conv2.write().append_with_blocks(timeline2, 10, 0, 1);
        assert_eq!(
            conv2.pending_summary_len(timeline2),
            1,
            "default summarize=true must enqueue"
        );
    }

    #[test]
    fn absorb_pending_creates_leaf_with_normal_child() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let _normal0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        // A trailing turn settles #0's exchange (making it sealable); the trailing
        // turn itself stays the deferred frontier.
        let _normal1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        assert_eq!(conv.pending_summary_len(timeline), 2);
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        // #0 sealed; the frontier #1 stays pending.
        assert_eq!(conv.pending_summary_len(timeline), 1);
        // The SoT leaf over #0 is recorded - its index follows the two Normals.
        let leaf_meta = conv
            .read()
            .tree_meta_of(timeline, TurnIndex(2))
            .cloned()
            .expect("leaf meta exists");
        assert_eq!(leaf_meta.kind, TurnKind::SummaryOfTurns);
        assert_eq!(leaf_meta.children, vec![TurnIndex(0)]);
        assert_eq!(leaf_meta.tree_height, 1);
        // The lone sealed leaf is the single peak.
        assert_eq!(conv.read().peaks_of(timeline), vec![(TurnIndex(2), 1)]);
    }

    /// Every `SummaryOfTurns` leaf's children, in turn order.
    fn leaf_children(conv: &Conversation, timeline: TimelineId) -> Vec<Vec<TurnIndex>> {
        let read = conv.read();
        (0..read.turn_count(timeline))
            .filter_map(|i| {
                read.tree_meta_of(timeline, TurnIndex(i))
                    .filter(|m| m.kind == TurnKind::SummaryOfTurns)
                    .map(|m| m.children.clone())
            })
            .collect()
    }

    /// The fabrication fix. Two turns coupled into one tool round-trip must seal
    /// ONE leaf covering BOTH — not two half-leaves. A leaf over the call turn
    /// alone has no answer among its children, which is what made the compressor
    /// invent "15:30 UTC" for a tool that returned 12:21.
    #[test]
    fn a_coupled_round_trip_seals_one_leaf_over_both_turns() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let call = conv.write().append_with_blocks(timeline, 10, 0, 1);
        conv.write().couple_turn(timeline, call.0);
        let response = conv.write().append_with_blocks(timeline, 10, 1, 2);
        // A trailing turn settles the [call, response] exchange (otherwise it is
        // the deferred frontier).
        let _sentinel = conv.write().append_with_blocks(timeline, 10, 2, 3);
        assert_eq!(conv.pending_summary_len(timeline), 3);

        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        // Only the frontier sentinel remains; the round-trip is drained.
        assert_eq!(
            conv.pending_summary_len(timeline),
            1,
            "only the frontier remains"
        );

        // Exactly ONE leaf, covering the whole exchange in order.
        let leaves = leaf_children(&conv, timeline);
        assert_eq!(
            leaves.len(),
            1,
            "one exchange must seal one leaf, got {leaves:?}"
        );
        assert_eq!(
            leaves[0],
            vec![call, response],
            "the leaf must cover the call AND its tool response"
        );
    }

    /// The live frontier: a coupling is durable but its tool response has not
    /// landed. Sealing now would freeze a leaf that has to invent its answer, so
    /// the exchange must defer — and stay queued for a later pass.
    #[test]
    fn an_open_exchange_defers_instead_of_sealing_half() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let call = conv.write().append_with_blocks(timeline, 10, 0, 1);
        conv.write().couple_turn(timeline, call.0);
        // The response turn does not exist yet — the tools are still running.
        assert_eq!(conv.pending_summary_len(timeline), 1);

        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");

        assert!(
            leaf_children(&conv, timeline).is_empty(),
            "must not seal a leaf over half an exchange"
        );
        assert_eq!(
            conv.pending_summary_len(timeline),
            1,
            "the deferred head must stay queued for the next pass"
        );

        // The response lands, but [call, response] is now the frontier, so it
        // still defers - the fabrication guard holds until a later turn settles it.
        let response = conv.write().append_with_blocks(timeline, 10, 1, 2);
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        assert!(
            leaf_children(&conv, timeline).is_empty(),
            "the round-trip is still the frontier - not yet sealed"
        );
        // A later turn settles it -> seals as ONE leaf over both halves.
        let _sentinel = conv.write().append_with_blocks(timeline, 10, 2, 3);
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        assert_eq!(
            leaf_children(&conv, timeline),
            vec![vec![call, response]],
            "seals once a later turn settles the exchange"
        );
    }

    /// Uncoupled turns are untouched — no coupling means every turn is its own
    /// exchange, so the forest is shaped exactly as it was before exchanges.
    #[test]
    fn uncoupled_turns_still_seal_one_leaf_each() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let n1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        // A trailing turn so both n0 and n1 are settled (non-frontier).
        let _sentinel = conv.write().append_with_blocks(timeline, 10, 2, 3);

        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        let leaves = leaf_children(&conv, timeline);
        assert_eq!(leaves.len(), 2, "two settled exchanges, two leaves");
        assert!(leaves.contains(&vec![n0]));
        assert!(leaves.contains(&vec![n1]));
    }

    /// A live conversation defers its last exchange (frontier); once archived,
    /// a summariser pass seals it too, so an archived conversation has no hole in
    /// its peak cover.
    #[test]
    /// **A live conversation's recent turns are never compressed.** The
    /// summariser absorbs an exchange only once `RAW_TAIL_TURNS` newer turns
    /// exist — the working tail stays verbatim, because a model that needs to
    /// quote or follow up on its own recent exchanges cannot do it from a
    /// digest. Measured before this gate: a four-question conversation had its
    /// first turn compressed 64 s after sealing, and by "what did I ask you?"
    /// the model answered that it had no access to the conversation history.
    /// Archiving is the exemption — a terminal timeline compresses to the end.
    #[test]
    fn the_recent_tail_stays_verbatim_until_the_conversation_outgrows_it() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let runner = MockProbeRunner::new(conv.clone());

        // A short conversation: every turn is inside the tail. Nothing absorbs,
        // everything stays pending for a later pass.
        let n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        for i in 1..4u32 {
            conv.write()
                .append_with_blocks(timeline, 10, i as u64, i as u64 + 1);
        }
        absorb_pending_turns(&conv, &runner, timeline, 4, RAW_TAIL_TURNS).expect("absorb ok");
        assert!(
            leaf_children(&conv, timeline).is_empty(),
            "four turns, tail of eight: the whole conversation is verbatim"
        );
        assert_eq!(conv.pending_summary_len(timeline), 4, "all still pending");

        // The conversation grows past the tail: the oldest turns become
        // absorbable, the newest eight stay raw.
        for i in 4..12u32 {
            conv.write()
                .append_with_blocks(timeline, 10, i as u64, i as u64 + 1);
        }
        absorb_pending_turns(&conv, &runner, timeline, 16, RAW_TAIL_TURNS).expect("absorb ok");
        let absorbed: Vec<TurnIndex> = leaf_children(&conv, timeline).concat();
        assert!(
            absorbed.contains(&n0),
            "the oldest turn has left the tail and compresses"
        );
        let newest_raw = TurnIndex(11);
        assert!(
            !absorbed.contains(&newest_raw),
            "the newest turn is deep inside the verbatim tail"
        );

        // Archived → terminal: the tail exemption ends and everything seals.
        conv.write().set_archived(timeline, true);
        absorb_pending_turns(&conv, &runner, timeline, 16, RAW_TAIL_TURNS).expect("absorb ok");
        let absorbed: Vec<TurnIndex> = leaf_children(&conv, timeline).concat();
        assert!(
            absorbed.contains(&newest_raw),
            "archiving compresses the conversation to its end"
        );
        assert_eq!(conv.pending_summary_len(timeline), 0);
    }

    #[test]
    fn archiving_seals_the_deferred_frontier_exchange() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        let n0 = conv.write().append_with_blocks(timeline, 10, 0, 1);
        let n1 = conv.write().append_with_blocks(timeline, 10, 1, 2);
        let runner = MockProbeRunner::new(conv.clone());

        // Live: n0 settles (n1 is its successor); n1 is the frontier and defers.
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        assert_eq!(leaf_children(&conv, timeline), vec![vec![n0]]);
        assert_eq!(
            conv.pending_summary_len(timeline),
            1,
            "frontier n1 deferred"
        );

        // Archive → a finalising pass seals the frontier n1 too.
        conv.write().set_archived(timeline, true);
        absorb_pending_turns(&conv, &runner, timeline, 4, 0).expect("absorb ok");
        let leaves = leaf_children(&conv, timeline);
        assert_eq!(leaves.len(), 2, "archived: frontier sealed");
        assert!(
            leaves.contains(&vec![n1]),
            "the deferred frontier n1 is now sealed"
        );
        assert_eq!(
            conv.pending_summary_len(timeline),
            0,
            "nothing left pending"
        );
    }

    #[test]
    fn absorb_a_full_run_carries_into_one_sos() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        // MERGE_FANOUT turns to carry, plus a trailing turn so the last of them is
        // settled rather than the deferred frontier.
        for i in 0..MERGE_FANOUT as u64 + 1 {
            conv.write().append_with_blocks(timeline, 10, i, i + 1);
        }
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, MERGE_FANOUT * 2, 0).expect("absorb ok");
        // MERGE_FANOUT Normal turns → that many SoT leaves → one SoS peak over them.
        let peaks = conv.read().peaks_of(timeline);
        assert_eq!(peaks.len(), 1, "the SoS should be the sole peak");
        let (root, level) = peaks[0];
        assert_eq!(level, 2);
        let root_meta = conv
            .read()
            .tree_meta_of(timeline, root)
            .cloned()
            .expect("root meta");
        assert_eq!(root_meta.kind, TurnKind::SummaryOfSummaries);
        assert_eq!(root_meta.children.len(), MERGE_FANOUT);
        assert_eq!(root_meta.tree_height, 2);
        // Forest is whole — nothing to reconcile.
        assert_eq!(conv.read().reconcile_next(timeline), None);
    }

    #[test]
    fn many_pending_build_canonical_forest() {
        let tmp = ephemeral_workspace();
        let (conv, timeline) = fresh_conversation(tmp.path());
        // F + 1 turns → base-F "11" → one full SoS (level 2) + one leftover leaf
        // (level 1). Peak levels {1, 2}.
        // F + 1 exchanges must SEAL to give base-F 11 (levels {1,2}); append one
        // extra so the (F+1)-th is settled rather than the deferred frontier.
        let sealed = MERGE_FANOUT as u64 + 1;
        for i in 0..sealed + 1 {
            conv.write().append_with_blocks(timeline, 10, i, i + 1);
        }
        let runner = MockProbeRunner::new(conv.clone());
        absorb_pending_turns(&conv, &runner, timeline, (sealed as usize + 1) * 2, 0)
            .expect("absorb ok");
        let mut levels: Vec<u8> = conv
            .read()
            .peaks_of(timeline)
            .into_iter()
            .map(|(_, l)| l)
            .collect();
        levels.sort_unstable();
        assert_eq!(levels, vec![1, 2]);
        // Every SoS is canonical (exactly MERGE_FANOUT children) and the forest
        // is whole.
        let tree = conv.read().build_summary_tree_in_memory(timeline);
        for id in tree.all_ids() {
            let node = tree.get(id).unwrap();
            if node.kind == TurnKind::SummaryOfSummaries {
                assert_eq!(node.children.len(), MERGE_FANOUT);
            }
        }
        assert_eq!(conv.read().reconcile_next(timeline), None);
    }
}
