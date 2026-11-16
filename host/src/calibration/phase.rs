//! The calibration phase: retire what is stale, resume what is sealed, write
//! the rest.

use std::collections::HashSet;
use std::time::Instant;

use candle::vram;
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::projection::TimelineId;
use candle_conversation::{
    ConversationEngine, ConversationError, ProjectionEvent, Sequence, SequenceConfig, TurnEvent,
    TurnHandle, TurnOptions, TurnResponse,
};
use futures::executor::block_on;
use futures::future::select_all;

use super::case::{cases, Case, InFlight};
use super::exemplar::{Calibrated, MARKER_KEY, PROJECTION_MARKER};
use super::projection::{build, Frame, TOOL_SCOPE_TAG, TOOL_SELECTOR};
use super::timing::Timing;
use super::window::{want, BATCH};

/// Max tokens for a calibration turn's **decode fallback**. Calibration
/// prefills an authored trajectory verbatim, so this cap does nothing there; it
/// applies only to a trajectory with a multi-turn lead-in, which is decoded
/// live. Generous so a verbose reasoner (or a long call payload) finishes
/// through `</tool_call>` rather than being cut mid-think; the decode still
/// stops early on the natural end of turn, so short cases pay nothing for the
/// headroom.
const MAX_TOKENS: usize = 2048;

/// What one run of the phase is given.
pub struct Phase<'a> {
    pub engine: &'a ConversationEngine,
    /// The daemon's conversation settings; the phase runs its cases under them
    /// with the K/V kept lossless.
    pub config: &'a SequenceConfig,
    /// The system prompt a case's tool is shown inside.
    pub frame: &'a Frame,
    /// Told how many exemplars are done, of how many.
    pub progress: &'a dyn Fn(u64, u64),
    /// Whether the load has been asked to stop. Read between waves: nothing
    /// more is submitted, the cases in flight are dropped un-archived, and the
    /// next load regenerates each one.
    pub stopped: &'a dyn Fn() -> bool,
}

/// What a run did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Every exemplar the catalog asks for.
    pub exemplars: usize,
    /// Exemplars a previous load had already sealed.
    pub resumed_exemplars: usize,
    /// Cases submitted this run — a question group is one, carrying many
    /// exemplars.
    pub submissions: usize,
    /// Submissions that sealed and were archived.
    pub sealed: usize,
    /// Whether a stop ended the run before its corpus was written.
    pub stopped: bool,
}

/// Mark a completed calibration `timeline` for distillation, idempotently. The
/// authored definitions are the source of truth, so once the wide-Q signature
/// is captured the K/V and token content is dead weight compaction can shed.
fn mark_distill(
    engine: &ConversationEngine,
    timeline: TimelineId,
    tool: &str,
    timing: &mut Timing,
) {
    timing.time(
        |t| &mut t.distill,
        || {
            // Whether or not any chunk reached disk: an exemplar is transient,
            // so none did, and without the marker a peer holds its turns as
            // still arriving, waiting for K/V that is never written.
            if !engine.is_timeline_distilled(timeline) {
                // Calibration exemplars stay retrievable by signature — provenance-only.
                if let Err(e) = engine.distill_timeline(timeline, DistillMode::ProvenanceOnly) {
                    tracing::warn!(tool = tool, "calibration distill mark failed: {e}");
                }
            }
        },
    );
}

/// Seal a completed calibration turn the way a live decode is sealed: finish
/// the turn (which runs the scan `projection_event` reads) and persist its
/// projection events, so the substrate carries the durable turn→tool-section
/// provenance link — the `tools` collection with the pinned tool marked
/// selected. Best-effort: a failure only drops the supplementary provenance
/// record, never the case itself. The turn's tokens, K/V signatures and wide-Q
/// window are sealed by the scheduler independently of this call.
fn seal_turn(
    conv: &mut Sequence,
    handle: TurnHandle,
    resp: &TurnResponse,
    tool: &str,
    mut events: Vec<ProjectionEvent>,
    timing: &mut Timing,
) {
    if let Err(e) = timing.time(|t| &mut t.seal, || conv.finish_turn(handle, resp)) {
        tracing::warn!(tool = tool, "calibration finish_turn failed: {e}");
        return;
    }
    // `events` holds the mid-decode reprojection events streamed during the turn;
    // append the final projection (composition at seal, like the serve path), then
    // persist the whole per-decode trajectory — the real projection sequence, each
    // event's `tools` collection carrying the pinned tool as its selected section.
    if let Some(event) = conv.projection_event(&resp.stats) {
        events.push(event);
    }
    if events.is_empty() {
        tracing::warn!(tool = tool, "calibration: no projection events to persist");
        return;
    }
    if let Err(e) = timing.time(
        |t| &mut t.persist_events,
        || conv.persist_projection_events(&events),
    ) {
        tracing::warn!(
            tool = tool,
            "calibration persist projection events failed: {e}"
        );
    }
}

/// Seal, archive and mark for distillation a case whose turn completed whole.
/// The archive is the atomic done-marker the next load's resume reads.
fn commit(
    engine: &ConversationEngine,
    conv: &mut Sequence,
    handle: TurnHandle,
    resp: &TurnResponse,
    tool: &str,
    events: Vec<ProjectionEvent>,
    timing: &mut Timing,
) {
    seal_turn(conv, handle, resp, tool, events, timing);
    if let Err(e) = timing.time(
        |t| &mut t.archive,
        || engine.set_conversation_archived(conv.timeline_id(), true),
    ) {
        tracing::warn!(tool = tool, "calibration archive failed: {e}");
    }
    // Only the wide-Q signature is needed henceforth.
    mark_distill(engine, conv.timeline_id(), tool, timing);
}

/// What the retire sweep keeps count of across passes.
struct Tally {
    /// Exemplars done, resumed ones included.
    done: usize,
    /// Every exemplar the catalog asks for.
    total: usize,
    /// Submissions sealed and archived — counted as they retire, never read
    /// off the reclaim list, which holds failed ones too.
    sealed: usize,
    /// Timelines of retired cases, sealed or not, awaiting hot→warm demotion.
    /// Their sealed K/V is never attended again — only the persisted wide-Q
    /// signature is — so dropping the hot copy as they retire keeps the phase's
    /// VRAM footprint flat instead of letting the corpus's lossless K/V pile up
    /// hot and starve whatever prefills next.
    reclaim: Vec<TimelineId>,
}

/// Non-blocking sweep: archive and retire every case finished since the last
/// sweep. Archives only complete trajectories; an incomplete or failed one
/// stays un-archived to retry next load. Returns whether anything was retired.
fn retire_completed(
    phase: &Phase<'_>,
    inflight: &mut Vec<InFlight<'_>>,
    tally: &mut Tally,
    timing: &mut Timing,
) -> bool {
    let mut retired = false;
    let mut i = 0;
    while i < inflight.len() {
        let mut collected: Vec<ProjectionEvent> = Vec::new();
        // A terminal the idle wait already absorbed retires this case without
        // touching the channel again.
        let terminal = match inflight[i].finished.take() {
            Some(t) => Some(t),
            None => loop {
                match inflight[i].handle.try_recv() {
                    Some(TurnEvent::Done(resp)) => break Some(Ok(resp)),
                    Some(TurnEvent::Error(e)) => break Some(Err(e)),
                    Some(TurnEvent::Projection(ev)) => collected.push(ev),
                    Some(_) => {}
                    None => break None,
                }
            },
        };
        inflight[i].events.append(&mut collected);
        let Some(result) = terminal else {
            i += 1;
            continue;
        };
        let InFlight {
            mut conv,
            handle,
            tool,
            events,
            finished: _,
            is_prefill,
            exemplars,
        } = inflight.remove(i);
        match result {
            Ok(resp) if is_prefill || resp.text.contains("</tool_call>") => {
                commit(phase.engine, &mut conv, handle, &resp, tool, events, timing);
                tally.sealed += 1;
            }
            Ok(_) => tracing::warn!(
                tool = tool,
                "calibration trajectory incomplete — left un-archived for retry"
            ),
            Err(e) => tracing::warn!(tool = tool, "calibration case failed: {e}"),
        }
        // Retired either way, and its sealed K/V is never attended again. A
        // turn group that lost one case still sealed the rest, so its timeline
        // is demoted hot→warm like any other.
        tally.reclaim.push(conv.timeline_id());
        tally.done += exemplars;
        (phase.progress)(tally.done as u64, tally.total as u64);
        retired = true;
    }
    retired
}

/// Park until whichever in-flight case speaks next — completion order, no poll
/// cadence. The one event the wait consumes is absorbed into its case (a
/// reprojection into `events`, a terminal into `finished` for the next sweep),
/// so nothing the channel says is lost. Driven by `block_on` because this is
/// the loader thread: the futures wake on the engine's own channels, so no
/// timer or I/O driver is needed.
fn wait_for_one(inflight: &mut [InFlight<'_>]) {
    let (event, which) = {
        let waits: Vec<_> = inflight
            .iter()
            .map(|c| Box::pin(c.handle.next_event_async()))
            .collect();
        let (event, which, rest) = block_on(select_all(waits));
        drop(rest);
        (event, which)
    };
    let case = &mut inflight[which];
    match event {
        Some(TurnEvent::Done(resp)) => case.finished = Some(Ok(resp)),
        Some(TurnEvent::Error(e)) => case.finished = Some(Err(e)),
        None => {
            case.finished = Some(Err(ConversationError::Channel(
                "the turn's event channel closed before it finished".into(),
            )))
        }
        Some(TurnEvent::Projection(ev)) => case.events.push(ev),
        Some(_) => {}
    }
}

/// Submit one case's turn on its conversation. Returns the turn's handle and
/// whether it was prefilled (complete by construction) rather than decoded.
///
/// A question group goes in as one turn group, sealed one turn each; its
/// questions share this submission's pinned selection, which is correct
/// precisely because a group is one tool's. A question exemplar's assistant
/// half is empty — the routing happens on the question.
///
/// A trajectory is prefilled verbatim in one batched forward instead of being
/// decoded token by token — the wide-Q is captured identically at seal. Only a
/// single-turn example (exactly one assistant header) qualifies: split into the
/// body (after the header — the think→call, projection markers kept) and the
/// user prompt (before it, minus the trailing `user_end`). The prefill strips
/// the markers and records each one's token offset so the staged prefill wave
/// fires a projection there, reproducing the decode's projection sequence. A
/// multi-turn lead-in is decoded live rather than prefilling a wrong-grid body.
fn submit(
    conv: &mut Sequence,
    case: &Case<'_>,
    opts: TurnOptions,
    assistant_start: &str,
    user_end: &str,
) -> (Result<TurnHandle, ConversationError>, bool) {
    match case {
        Case::Questions { questions, .. } => {
            let group: Vec<(String, String, Vec<String>)> = questions
                .iter()
                .map(|q| (q.clone(), String::new(), opts.tags.clone()))
                .collect();
            (
                conv.submit_prefilled_turn_group(&group, opts.selection.clone())
                    .map(|(handle, _)| handle),
                true,
            )
        }
        Case::Trajectory { text, .. } => {
            if text.matches(assistant_start).count() == 1 {
                let (before, body) = text.split_once(assistant_start).expect("count == 1");
                let prompt = before.trim_end();
                let prompt = prompt
                    .strip_suffix(user_end.trim())
                    .unwrap_or(prompt)
                    .trim_end();
                (
                    conv.submit_prefilled_turn(
                        prompt,
                        body,
                        PROJECTION_MARKER,
                        opts.selection.clone(),
                        opts.tags.clone(),
                    ),
                    true,
                )
            } else {
                (conv.submit_turn_with_options(text.trim(), opts), false)
            }
        }
    }
}

/// Run the warm-up case to its end, alone, so the shared tool sections upload
/// once before the concurrent window opens; then seal it if it completed.
fn run_alone(
    phase: &Phase<'_>,
    mut conv: Sequence,
    handle: TurnHandle,
    tool: &str,
    is_prefill: bool,
    timing: &mut Timing,
) -> (TimelineId, bool) {
    // Drain the event stream (blocking) so the warm-up case collects its
    // mid-decode reprojection events too.
    let mut events: Vec<ProjectionEvent> = Vec::new();
    let mut done: Option<TurnResponse> = None;
    for ev in handle.stream() {
        match ev {
            TurnEvent::Projection(e) => events.push(e),
            TurnEvent::Done(resp) => {
                done = Some(resp);
                break;
            }
            TurnEvent::Error(e) => {
                tracing::warn!(tool = tool, "calibration case failed: {e}");
                break;
            }
            _ => {}
        }
    }
    let sealed = match done {
        Some(resp) if is_prefill || resp.text.contains("</tool_call>") => {
            commit(phase.engine, &mut conv, handle, &resp, tool, events, timing);
            true
        }
        Some(_) => {
            tracing::warn!(
                tool = tool,
                "calibration trajectory incomplete — left un-archived for retry"
            );
            false
        }
        None => false,
    };
    (conv.timeline_id(), sealed)
}

/// Prefill each tool's authored trajectories and questions into the hidden
/// calibration layer, capturing each think→call trajectory (and its
/// per-reprojection wide-Q windows) and each question.
///
/// All cases share ONE projection: the whole catalog as a name-keyed `tools`
/// collection. Each case pins exactly its tool by name, so the model sees a
/// single available tool. The tool sections are sealed once (the first case);
/// every later case reuses them. One conversation per case; a failing case is
/// logged and skipped so it can never break a daemon's load.
///
/// A case is *done* only once its conversation is ARCHIVED — the atomic
/// commit. Each conversation is tagged with its case's marker at creation, so
/// a half-finished one is still findable: present but unarchived is tombstoned
/// and regenerated.
pub fn run<T: Calibrated>(phase: &Phase<'_>, defs: &[T]) -> anyhow::Result<Report> {
    let engine = phase.engine;
    let start = Instant::now();
    let mut timing = Timing::default();
    let (builder, layer, group) = build(phase.frame, defs)?;
    // Pin calibration turns to native R16/F16 (no hot→warm quantize) so every
    // token's Q survives to the seal-time wide `sign(Q)` capture — the
    // full-resolution provenance exemplars this phase exists to produce.
    let config = {
        let mut c = phase.config.clone();
        c.kv_lossless = true;
        c
    };
    let assistant_start = phase.config.dialect.assistant_start;
    let user_end = phase.config.dialect.user_end;
    let all = cases(defs, assistant_start, user_end);
    let total: usize = all.iter().map(Case::exemplars).sum();
    let asks: usize = all
        .iter()
        .filter(|c| matches!(c, Case::Questions { .. }))
        .map(Case::exemplars)
        .sum();
    // Submissions, not exemplars: a tool's question set is ONE prefill carved
    // into one turn per question, so the number of forwards the phase costs is
    // trajectories + tools, not the corpus size.
    tracing::info!(
        exemplars = total,
        question_only = asks,
        submissions = all.len(),
        "calibrating sections: per-tool example prefills (named tool selection)"
    );

    // Retire exemplars that no longer match any current case. A tool whose
    // description, parameters, or example changed keeps its old conversation
    // ARCHIVED under the *previous* marker, where the by-marker lookup below
    // can never find it — so without this sweep it stays tagged with the tool
    // and keeps answering the belief scan with signatures captured against
    // wording that no longer exists, alongside the freshly calibrated ones.
    // Tombstoning drops it from the gather, leaving exactly the current corpus.
    let current: HashSet<&str> = all.iter().map(|c| c.marker()).collect();
    let mut retired = 0usize;
    for (timeline, marker) in engine.conversations_with_metadata_key(MARKER_KEY) {
        if current.contains(marker.as_str()) {
            continue;
        }
        match engine.tombstone_timeline(timeline) {
            Ok(()) => retired += 1,
            Err(e) => {
                tracing::warn!(%marker, "retiring stale calibration exemplar failed: {e}")
            }
        }
    }
    if retired > 0 {
        tracing::info!(
            retired,
            "calibration: retired stale exemplars whose tool definition or example changed",
        );
    }

    // Resume filter: skip cases already archived, tombstone any half-finished
    // prior, and collect the cases that still need running. Conversations are
    // NOT created here — creation is a synchronous scheduler round-trip, so
    // pre-creating one per case would allocate hundreds of empty slots at
    // once. Each case's conversation is created as the window refills, bounding
    // the live slot count to the window.
    let mut tally = Tally {
        done: 0,
        total,
        sealed: 0,
        reclaim: Vec::new(),
    };
    let mut to_run: Vec<&Case<'_>> = Vec::new();
    for case in all.iter() {
        let (name, marker) = (case.tool(), case.marker());
        // Distill-inclusive: a finished exemplar's designed end state is
        // archived + distilled, and the live-only lookup cannot see it.
        // Ordinary tombstones stay excluded, so a genuinely retired
        // conversation is still ignored here.
        let prior = engine.find_conversations_by_metadata_including_distilled(MARKER_KEY, marker);
        // Done iff a prior conversation for this case is archived.
        if prior.iter().any(|t| engine.is_conversation_archived(*t)) {
            for t in prior
                .iter()
                .filter(|t| engine.is_conversation_archived(**t))
            {
                mark_distill(engine, *t, name, &mut timing);
            }
            tally.done += case.exemplars();
            (phase.progress)(tally.done as u64, total as u64);
            continue;
        }
        // Any non-archived prior conversation is a half-finished run —
        // tombstone it before regenerating, so exactly one (complete)
        // conversation per case survives.
        for t in &prior {
            if let Err(e) = engine.tombstone_timeline(*t) {
                tracing::warn!(tool = name, "tombstone partial calibration failed: {e}");
            }
        }
        to_run.push(case);
    }
    // Say up front how much of the corpus is being regenerated. `exemplars`
    // and `submissions` are deliberately both reported: a question group is
    // one submission carrying many exemplars, so a single figure would
    // misrepresent either the work or the corpus.
    tracing::info!(
        exemplars = total,
        resumed = tally.done,
        submissions = to_run.len(),
        exemplars_to_run = to_run.iter().map(|c| c.exemplars()).sum::<usize>(),
        "calibration resume filter"
    );

    let resumed_exemplars = tally.done;
    let submissions = to_run.len();
    let mut to_run_iter = to_run.into_iter();
    let mut inflight: Vec<InFlight<'_>> = Vec::new();
    let mut warmed = false;
    // Index into the reclaim list up to which the hot copies are already
    // dropped, so each incremental sweep only demotes the fresh tail.
    let mut reclaimed_up_to = 0usize;
    let mut stopped = false;
    loop {
        // A stop ends the phase between waves: nothing more is submitted, and
        // the cases in flight are dropped un-archived — the scheduler ends a
        // turn whose handle is gone, and the next load's resume filter
        // tombstones and regenerates each one.
        if (phase.stopped)() {
            inflight.clear();
            stopped = true;
            break;
        }
        // Refill the window in ONE pipelined round-trip, and in blocks rather
        // than per retired case: slot allocation is served by the scheduler
        // between waves, so ANY create call — for one case or sixteen — blocks
        // the caller until the current wave ends. `inflight.is_empty()` is the
        // tail case: fewer cases remain than the threshold, so nothing is
        // running to wait behind and the block is issued at whatever is left.
        let free = BATCH.saturating_sub(inflight.len());
        let batch: Vec<&Case<'_>> = (0..want(free, warmed, inflight.is_empty()))
            .map_while(|_| to_run_iter.next())
            .collect();
        let created_any = !batch.is_empty();
        let convs = if created_any {
            timing.time(
                |t| &mut t.create,
                || {
                    engine.new_conversations_with_projection_batch(
                        batch.len(),
                        &phase.frame.open,
                        &builder,
                        layer,
                        group,
                        &config,
                    )
                },
            )
        } else {
            Vec::new()
        };
        for (case, conv_res) in batch.into_iter().zip(convs) {
            let (name, marker) = (case.tool(), case.marker());
            let mut conv = match conv_res {
                Ok(conv) => conv,
                Err(e) => {
                    tracing::warn!(tool = name, "calibration create failed: {e}");
                    // The whole case's exemplars, as a failed submit forfeits —
                    // a group is one creation carrying many.
                    tally.done += case.exemplars();
                    (phase.progress)(tally.done as u64, total as u64);
                    continue;
                }
            };
            // **This exemplar's KV never goes to disk.** The seal captures its
            // wide-Q signature — which persists on its own record — and the
            // case's designed end state is provenance only: KV dropped,
            // signature kept. Cold-persisting it would write bytes the next
            // load's distillation immediately deletes. Marked before the turn
            // is submitted so the residence is flagged as it completes rather
            // than after it has been queued for a cold write.
            engine
                .conversation()
                .mark_timeline_transient(conv.timeline_id());
            // Tag at creation so a half-finished case is findable next load.
            if let Err(e) = timing.time(
                |t| &mut t.tag,
                || engine.set_conversation_metadata(conv.timeline_id(), MARKER_KEY, marker),
            ) {
                tracing::warn!(tool = name, "calibration tag failed: {e}");
            }
            let mut opts = TurnOptions {
                max_tokens: Some(MAX_TOKENS),
                // Gather-scope tags: the scope tag puts this turn in the tools
                // collection's belief gallery; the tool's name labels it
                // directly, so the seed corpus needs no cold-start bootstrap.
                tags: vec![TOOL_SCOPE_TAG.to_string(), name.to_string()],
                ..Default::default()
            };
            // Pin exactly this tool: the collection emits only the member whose
            // name matches the selector value.
            opts.selection.select(TOOL_SELECTOR, name);
            let (submitted, is_prefill) = timing.time(
                |t| &mut t.submit,
                || submit(&mut conv, case, opts, assistant_start, user_end),
            );
            match submitted {
                Ok(handle) if !warmed => {
                    warmed = true;
                    let (timeline, sealed) =
                        run_alone(phase, conv, handle, name, is_prefill, &mut timing);
                    tally.sealed += usize::from(sealed);
                    // Queued for hot→warm demotion whatever its outcome, so the
                    // warm-up case's K/V is reclaimed like every other case's.
                    tally.reclaim.push(timeline);
                    tally.done += case.exemplars();
                    (phase.progress)(tally.done as u64, total as u64);
                }
                Ok(handle) => inflight.push(InFlight {
                    conv,
                    handle,
                    tool: name,
                    events: Vec::new(),
                    finished: None,
                    is_prefill,
                    exemplars: case.exemplars(),
                }),
                Err(e) => {
                    tracing::warn!(tool = name, "calibration submit failed: {e}");
                    // A failed group forfeits all of its exemplars, not one —
                    // otherwise the count never reaches its total.
                    tally.done += case.exemplars();
                    (phase.progress)(tally.done as u64, total as u64);
                }
            }
        }
        // Terminate only when nothing is in flight AND nothing remained to
        // submit this pass. After the warm-up case (which leaves `inflight`
        // empty but `created_any` true) this continues to the concurrent
        // window instead of ending early.
        if inflight.is_empty() && !created_any {
            break;
        }
        if !retire_completed(phase, &mut inflight, &mut tally, &mut timing) && !inflight.is_empty()
        {
            timing.time(|t| &mut t.idle, || wait_for_one(&mut inflight));
        }
        // Incremental hot→warm demotion: once a full window's worth of cases
        // has retired since the last sweep, drop the hot K/V of just the fresh
        // tail. No flush — the persistence thread has already migrated the
        // older cases to warm; any straggler is caught by the flushing boundary
        // sweep below. The host-RAM trough is sampled on every pass, so a
        // from-empty calibration records it as it happens.
        vram::sample_available_low_water();
        if tally.reclaim.len() - reclaimed_up_to >= BATCH {
            if let Err(e) = timing.time(
                |t| &mut t.demote,
                || engine.demote_timelines_hot(&tally.reclaim[reclaimed_up_to..], false),
            ) {
                tracing::warn!("calibration hot→warm demote failed: {e}");
            }
            reclaimed_up_to = tally.reclaim.len();
            // Logged once per window, not only at the phase's end: a raw
            // allocator abort kills the process with no unwind, so a line only
            // at completion is never written when the phase itself is what
            // crashes.
            tracing::info!(
                done = tally.done,
                total,
                free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
                "calibrating sections: window checkpoint"
            );
        }
    }
    let report = Report {
        exemplars: total,
        resumed_exemplars,
        submissions,
        sealed: tally.sealed,
        stopped,
    };
    // Stopped mid-phase: no boundary sweep — what sealed is durable, and the
    // rest is the next load's.
    if stopped {
        return Ok(report);
    }
    // Boundary sweep: flush the pending hot→warm migration so the final
    // window's just-sealed cases are warm-backed, then demote every calibration
    // timeline, so whatever prefills next does not meet a card still full of
    // the corpus's hot K/V.
    if let Err(e) = timing.time(
        |t| &mut t.demote,
        || engine.demote_timelines_hot(&tally.reclaim, true),
    ) {
        tracing::warn!("calibration boundary hot→warm demote failed: {e}");
    }
    timing.log(start.elapsed());
    (phase.progress)(total as u64, total as u64);
    // `failed` is the rest of the submissions — every one submitted either
    // sealed or did not, each failure logged with its cause as it retired.
    tracing::info!(
        exemplars = total,
        resumed_exemplars,
        submissions,
        sealed = tally.sealed,
        failed = submissions.saturating_sub(tally.sealed),
        free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
        "calibrating sections complete"
    );
    Ok(report)
}
