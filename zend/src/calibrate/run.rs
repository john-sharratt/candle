//! Calibrating a layer's units: find each ready unit without a calibration,
//! ask it for its questions, and prefill them as its calibration.
//!
//! A unit is asked in a **draft** — a conversation forked from its layer's
//! base with the unit as its lineage, so the unit's whole conversation is the
//! draft's opening and the model writes its questions with what it read in
//! front of it. The draft is tombstoned once its answer is read: only the
//! questions are kept.
//!
//! Each question is then asked again on its own — a conversation of the
//! hidden calibration layer, the system prompt every conversation opens with,
//! nothing projected — with thinking on, and what the model reasons, up to the
//! layer's `think` budget, is kept ([`super::thinking`]). Every question of a
//! unit is asked at once, so they decode in one wave, and more units think
//! than draft at once: only the draft carries the unit's reading.
//!
//! The questions are calibrated the way a tool's are (`ToolDef::questions`):
//! every one in ONE prefill, a turn group in a conversation of the hidden
//! calibration layer, each masked to itself and sealed one turn apiece, its
//! answer its thinking. A question prefilled alone, under the system prompt
//! every conversation opens with, and the thinking it starts, are what a live
//! turn looks like before anything is projected for it — the region a live
//! probe resembles, where the unit's own windows are its content. The
//! answer is assistant content, not a reasoning segment, so the seal keeps its
//! signatures. The calibration carries
//! [`Substrate::CALIBRATES_KEY`], so its questions vote for the unit wherever
//! the unit is scored, and is then archived — the commit — and distilled
//! provenance-only: its K/V is dropped and its signatures stay.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_conversation::chain_health::chain_break;
use candle_conversation::projection::{
    DecodePriority, GroupId, OptionalState, Reserved, SelectionState, TimelineId,
    NO_THINK_SELECTOR, TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::{ThinkMode, TriggerRegistry};
use candle_conversation::substrate::Substrate;
use candle_conversation::{
    ConversationEngine, FinishOrder, Sequence, SequenceConfig, TurnOptions, MAX_DECODE_WIDTH,
    MAX_PREFILL_WIDTH,
};

use super::ask::Ask;
use super::permits::{Permit, Permits};
use super::plan::{plan, Prior, Unit};
use super::questions::questions;
use super::retire_replies::retire_reply_records;
use super::seal::seal_calibration;
use super::sections::{compose, split};
use super::spec::CalibrateSpec;
use super::thinking::{exemplar, thinking_of};
use crate::branch_ingest::keys::{key_version, CONTENT_KEY, KEY_VERSION};
use crate::code_read::PATH_KEY;
use crate::ingest_backlog::{self, Work};
use crate::ingest_lease::{self, Share};
use crate::layers::keys::{FILE_KEY, SUBJECT_KEY};
use crate::layers::run::{answer_of, LAYER_PARALLELISM};
use crate::refresh_ctx::RefreshContext;
use crate::types::ToolMode;

/// The calibration marker a calibration carries ([`CalibrateSpec::marker`]).
pub const MARKER_KEY: &str = "calibration";

/// The layer a draft asks for — set on every draft, so a draft a crash left
/// is found and retired.
pub const DRAFT_KEY: &str = "calibration_draft";

/// The tag every calibration turn carries: it keeps the questions out of
/// every gather that reads untagged turns as dialogue.
pub const TAG: &str = "calibration";

/// Decode budget for a draft's answer: room for the questions, which are
/// asked for without reasoning — up to [`super::spec::MAX_QUESTIONS`] of them,
/// a developer's question naming what it asks about running to ~40 tokens.
const DRAFT_MAX_TOKENS: usize = 4096;

/// On a unit, the marker its calibration last failed under and how many times.
const FAILED_KEY: &str = "calibration_failed";

/// Passes whose draft may write no question before the unit is left
/// uncalibrated. Four units answered every draft the same way and were asked
/// again every pass on both members (2026-10-31). A new reading is a new
/// marker, and is tried afresh.
const MAX_FAILURES: u32 = 3;

/// How a calibration that failed for want of a question says so.
const NO_QUESTION: &str = "the draft wrote no question";

/// How a calibration whose draft stopped short says so.
const SHORT_DRAFT: &str = "the draft left a part with no question";

/// The required parts of a draft that kept no question — `(name, min, kept)`
/// for each part asked — in the order asked.
///
/// **A draft that stops early leaves its later parts empty.** Sealed as it
/// stood it counted as calibrated: `zend/src/held_values.rs` kept 1 question
/// and `npc-map/src/lift.rs` 16, their `uses` and `purpose` parts empty, and
/// the sweep never came back to either (BlackWorld, 2026-11-15, 11 units in
/// 283). Judged instead by how many it kept against its count, a file with
/// fewer names than its budget — told to write fewer rather than ask about a
/// name twice — would be sent back for doing as it was told. An optional part
/// (`min` 0) may rightly write none.
fn empty_parts<'a>(parts: &[(&'a str, usize, usize)]) -> Vec<&'a str> {
    parts
        .iter()
        .filter(|&&(_, min, kept)| min > 0 && kept == 0)
        .map(|&(name, _, _)| name)
        .collect()
}

/// Whether `error` is the unit's own failure — its draft wrote nothing, or
/// stopped short — which counts against it, rather than a decode lost to a
/// full card, which says nothing about the unit.
fn units_own_failure(error: &str) -> bool {
    error.starts_with(NO_QUESTION) || error.starts_with(SHORT_DRAFT)
}

/// How many times `marker` has failed, as [`FAILED_KEY`] holds it in `recorded`.
fn failures_of(recorded: Option<&str>, marker: &str) -> u32 {
    recorded
        .and_then(|r| r.rsplit_once(' '))
        .filter(|(m, _)| *m == marker)
        .and_then(|(_, n)| n.parse().ok())
        .unwrap_or(0)
}

/// Units calibrated at once. A unit's draft forks from its whole reading, so
/// drafts are held to the width the layer reads at (`LAYER_PARALLELISM`); its
/// thinking and prefill carry only the system prompt — shared — and a
/// question, so three times as many units are in flight, and one thinking
/// does not hold a draft's place. The scheduler's waves pace the decode.
const CALIBRATE_PARALLELISM: usize = 3 * LAYER_PARALLELISM;

/// The places every unit's questions share while they think ([`thoughts`]):
/// enough to keep the card full, and few enough that no thinking turn stands
/// queued for admission long enough to be given up.
///
/// **Past both of the scheduler's widths, because a thinking turn spends most
/// of its life decoding.** Its prefill is the shared system prompt and one
/// question; its decode is the thinking budget. Gated at the prefill width
/// alone (24), the places were held by turns decoding while the prefill sat
/// half empty — 11.8 prefill and 13.3 decode rows in flight on BlackWorld
/// (2026-11-15), and a unit's thinking took up to 337 s. At 128, the
/// scheduler's prefill and decode ([`MAX_PREFILL_WIDTH`] + [`MAX_DECODE_WIDTH`]
/// = 88) are both kept full with a short queue behind them, far below the
/// ~550 that, asked all at once, stood long enough to trip the stall limit.
const THINKING_WIDTH: usize = 128;
const _: () = assert!(THINKING_WIDTH >= MAX_PREFILL_WIDTH + MAX_DECODE_WIDTH);
static THINKING: Permits = Permits::new(THINKING_WIDTH);

/// What one layer's pass did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Units already calibrated.
    pub done: usize,
    /// Units calibrated by this pass.
    pub sealed: usize,
    /// Units that failed, calibrated again by a later pass.
    pub failed: usize,
    /// Units whose draft a shutdown or an ingest cancel stopped — no failure
    /// of theirs, and calibrated again by a later pass. Counted apart, or every
    /// restart reads as one in-flight unit per worker failing.
    pub cancelled: usize,
    /// Units another member was calibrating, or had calibrated by the time
    /// this one held its lease ([`ingest_lease::share_calibration`]). Their
    /// calibrations arrive with replication.
    pub elsewhere: usize,
    /// Units neither calibrated nor ready, calibrated by a later pass once
    /// they are.
    pub waiting: usize,
    /// Calibrations retired, and drafts a crash left.
    pub retired: usize,
    /// Reply records this pass ended ([`retire_reply_records`]).
    pub replies: usize,
}

/// One layer to calibrate: its spec, the group its units are in, and the base
/// their drafts fork from.
pub struct Layer<'a> {
    pub spec: &'a CalibrateSpec,
    pub group: GroupId,
    pub base: &'a Sequence,
}

/// Where calibrations are prefilled: the system prompt every conversation
/// opens with, dialect-formatted, the sequence config they run under, and the
/// steering a question's thinking decodes under — a live dialogue turn's at
/// the default effort, so its reasoning block opens and closes as one does.
pub struct Frame<'a> {
    pub system_prompt: &'a str,
    pub config: &'a SequenceConfig,
    pub thinking: Arc<TriggerRegistry>,
}

/// Calibrate every unit of `layer` that is ready and not calibrated — those
/// its ingest did not calibrate as it committed them ([`calibrate_committed`])
/// — and retire every calibration of a unit no longer there.
pub fn calibrate_layer(
    engine: &ConversationEngine,
    layer: &Layer<'_>,
    frame: &Frame<'_>,
) -> Report {
    let spec = layer.spec;
    let mut report = Report::default();
    let todo = {
        let e = engine;
        for (tl, of) in e.conversations_with_metadata_key(DRAFT_KEY) {
            if of == spec.layer && e.tombstone_timeline(tl).is_ok() {
                report.retired += 1;
            }
        }
        let units = units(e, layer);
        let priors: Vec<(TimelineId, String, bool)> = e
            .conversations_with_metadata_key(MARKER_KEY)
            .into_iter()
            .filter(|(_, m)| spec.owns(m))
            .map(|(tl, m)| (tl, m, e.is_conversation_archived(tl)))
            .collect();
        let priors: Vec<Prior<'_>> = priors
            .iter()
            .map(|(tl, m, archived)| Prior {
                timeline: *tl,
                marker: m,
                unit: CalibrateSpec::unit_of(m).unwrap_or(m),
                archived: *archived,
            })
            .collect();
        let todo = plan(units, &priors);
        for &tl in &todo.retire {
            if e.tombstone_timeline(tl).is_ok() {
                report.retired += 1;
            }
        }
        todo
    };
    report.done = todo.done;
    report.waiting = todo.waiting;
    tracing::info!(
        target: "zend::calibrate",
        layer = %spec.layer,
        calibrated = todo.done,
        to_calibrate = todo.run.len(),
        waiting = report.waiting,
        retired = report.retired,
        "calibrating the layer's units",
    );
    // **The sweep is backlog the bar shows.** A pass reads every file before
    // it calibrates any (`docs/recurrent_reply_calibration.md` §6), so this
    // sweep is where a re-read tree spends most of its decode; each unit is
    // counted until it is sealed, failed or found to be another member's, and
    // shown as `<path> (calibrated)`.
    ingest_backlog::add_pending(Work::Calibrating, todo.run.len() as u64);
    let finished = AtomicUsize::new(0);
    let cursor = AtomicUsize::new(0);
    let sealed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let cancelled = AtomicUsize::new(0);
    let elsewhere = AtomicUsize::new(0);
    let drafts = Permits::new(LAYER_PARALLELISM);
    let done = |unit: &Unit| {
        finished.fetch_add(1, Ordering::Relaxed);
        ingest_backlog::item_done(Work::Calibrating, &format!("{} (calibrated)", unit.subject));
    };
    std::thread::scope(|s| {
        for _ in 0..CALIBRATE_PARALLELISM.min(todo.run.len()).max(1) {
            // Only a shutdown stops it.
            s.spawn(|| loop {
                if crate::ingest_cancel::cancelled() {
                    return;
                }
                let Some(unit) = todo.run.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    return;
                };
                // **A unit is leased only once it can start.** Its draft's
                // permit first, then its lease: a member holds no more units
                // than it is working on, and the units it has not reached stay
                // unleased for whichever member reaches them first. Leased
                // first, every thread of a pass held a unit while most waited
                // for a draft's place — WhiteWorld (2026-11-11) held 48 at one
                // unit every 2.5 minutes while BlackWorld, idle, found all 48
                // `elsewhere`.
                let t_permit = Instant::now();
                let permit = drafts.acquire();
                let permit_wait_ms = t_permit.elapsed().as_millis() as u64;
                if crate::ingest_cancel::cancelled() {
                    return;
                }
                // **One member calibrates a unit.** Every member's pass plans
                // the same units from the same replicated substrate; the one
                // the vote names asks for them, and a calibration handed back
                // is looked for again before it is asked for twice.
                let _lease = match ingest_lease::share_calibration(&unit.marker) {
                    Share::Theirs => {
                        elsewhere.fetch_add(1, Ordering::Relaxed);
                        done(unit);
                        continue;
                    }
                    Share::Mine(lease) => {
                        if lease.recheck() && calibrated(engine, &unit.marker) {
                            elsewhere.fetch_add(1, Ordering::Relaxed);
                            done(unit);
                            continue;
                        }
                        lease
                    }
                };
                let outcome = calibrate_one(
                    engine,
                    spec,
                    layer.base,
                    frame,
                    Some((permit, permit_wait_ms)),
                    unit,
                );
                done(unit);
                match outcome {
                    Ok(()) => {
                        sealed.fetch_add(1, Ordering::Relaxed);
                    }
                    // A draft a shutdown stopped is no failure of the unit's.
                    Err(_) if crate::ingest_cancel::cancelled() => {
                        cancelled.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        // Only the unit's own failure counts against it: a
                        // decode lost to a full card says nothing about the
                        // unit, and is simply tried again.
                        if units_own_failure(&format!("{e:#}")) {
                            let tries = failures_of(unit.prior_failures.as_deref(), &unit.marker)
                                .saturating_add(1);
                            if let Err(err) = engine.set_conversation_metadata(
                                unit.timeline,
                                FAILED_KEY,
                                &format!("{} {tries}", unit.marker),
                            ) {
                                tracing::warn!(
                                    target: "zend::calibrate",
                                    subject = %unit.subject,
                                    "a failed calibration could not be counted: {err}",
                                );
                            }
                        }
                        tracing::warn!(
                            target: "zend::calibrate",
                            layer = %spec.layer,
                            subject = %unit.subject,
                            "calibration failed: {e:#}",
                        );
                    }
                }
            });
        }
    });
    // A shutdown stops the sweep between units: what it never started is
    // handed back, so the bar still reaches zero.
    ingest_backlog::drop_pending(
        Work::Calibrating,
        todo.run.len().saturating_sub(finished.into_inner()) as u64,
    );
    report.sealed = sealed.into_inner();
    report.failed = failed.into_inner();
    report.cancelled = cancelled.into_inner();
    report.elsewhere = elsewhere.into_inner();
    report.replies = retire_reply_records(engine);
    tracing::info!(
        target: "zend::calibrate",
        layer = %spec.layer,
        calibrated = report.done,
        sealed = report.sealed,
        failed = report.failed,
        cancelled = report.cancelled,
        elsewhere = report.elsewhere,
        waiting = report.waiting,
        retired = report.retired,
        replies_ended = report.replies,
        "the layer's calibration pass ended",
    );
    report
}

/// Whether a calibration carrying `marker` stands — one another member sealed
/// and replication has brought here.
fn calibrated(engine: &ConversationEngine, marker: &str) -> bool {
    engine
        .conversations_with_metadata_key(MARKER_KEY)
        .into_iter()
        .any(|(_, m)| m == marker)
}

/// Every unit of `layer`: each conversation of its group committed under a
/// content key and naming its subject — ready when its chain finished, every
/// turn holds its content, and the reading it stands on is of the current
/// [`KEY_VERSION`].
///
/// **A unit standing on an older reading waits.** Its draft forks from that
/// reading, which the current rendering no longer projects: asked anyway, the
/// draft saw no file and answered that none had been read, or called a tool,
/// or wrote questions about nothing. Such a unit is read again under the
/// current version by the pass a version change starts, and is calibrated
/// then; its calibration stands until it is.
fn units(engine: &ConversationEngine, layer: &Layer<'_>) -> Vec<Unit> {
    let conv = engine.conversation();
    let substrate = conv.read();
    units_of(&substrate, layer.group, layer.spec)
}

/// [`units`] over `substrate`.
fn units_of(substrate: &Substrate, group: GroupId, spec: &CalibrateSpec) -> Vec<Unit> {
    let mut units: Vec<Unit> = substrate
        .active_timelines_for_group(group)
        .filter(|&tl| substrate.calibrates(tl).is_none())
        .filter_map(|tl| {
            let meta = substrate.custom_of(tl)?;
            let key = meta.get(CONTENT_KEY)?;
            let subject = meta.get(SUBJECT_KEY).or_else(|| meta.get(PATH_KEY))?;
            // A file's reading is keyed by its own content key; a unit built
            // on one records the reading's under `FILE_KEY`.
            let reading = meta.get(FILE_KEY).unwrap_or(key);
            let link = meta.contains_key(Substrate::FORKED_FROM_KEY);
            let marker = spec.marker(key, subject, link);
            let prior_failures = meta.get(FAILED_KEY).cloned();
            Some(Unit {
                timeline: tl,
                subject: subject.clone(),
                key: key.clone(),
                ready: key_version(reading) == KEY_VERSION
                    && chain_break(substrate, tl).is_none()
                    && substrate.holds_whole(tl)
                    && failures_of(prior_failures.as_deref(), &marker) < MAX_FAILURES,
                marker,
                prior_failures,
                link,
            })
        })
        .collect();
    units.sort_by_key(|u| u.timeline);
    units
}

/// What names every calibration of `unit`: its kind, the unit it calibrates
/// ([`Substrate::CALIBRATES_KEY`]), and the unit's subject.
pub(crate) fn calibration_meta(unit: TimelineId, subject: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("kind".to_string(), TAG.to_string()),
        (
            Substrate::CALIBRATES_KEY.to_string(),
            unit.raw().to_string(),
        ),
        (SUBJECT_KEY.to_string(), subject.to_string()),
    ])
}

/// The selection a draft is asked under: no reasoning, and no tool block.
pub(super) fn draft_selection() -> SelectionState {
    let mut selection = SelectionState::new();
    selection.select("thinking_effort", "off");
    selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Absent);
    selection
}

/// Calibrate a unit the moment its conversation commits, from the ingest
/// worker that read it: its reading is still resident then — the next step
/// evicts it to cold — so the draft forked onto it projects the reading from
/// the card, where a later pass lifts it back from disk first. Its reading
/// is of the current key version by construction.
pub fn calibrate_committed(
    ctx: &RefreshContext<'_>,
    spec: &CalibrateSpec,
    base: &Sequence,
    timeline: TimelineId,
    subject: &str,
    key: &str,
) -> anyhow::Result<()> {
    let frame = Frame {
        system_prompt: ctx.formatted_prompt,
        config: &ctx.config,
        thinking: ctx.triggers_for(ThinkMode::Balanced, ToolMode::None),
    };
    let link = ctx
        .engine
        .conversation()
        .read()
        .custom_of(timeline)
        .is_some_and(|meta| meta.contains_key(Substrate::FORKED_FROM_KEY));
    let unit = Unit {
        timeline,
        subject: subject.to_string(),
        key: key.to_string(),
        marker: spec.marker(key, subject, link),
        ready: true,
        prior_failures: None,
        link,
    };
    calibrate_one(ctx.engine, spec, base, &frame, None, &unit)
}

/// Ask `unit` for its questions — holding `draft_permit`, when given with how
/// long it was waited for, while it does, and handing it back once the draft
/// is written — and calibrate them.
fn calibrate_one(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    frame: &Frame<'_>,
    draft_permit: Option<(Permit<'_>, u64)>,
    unit: &Unit,
) -> anyhow::Result<()> {
    let permit_wait_ms = draft_permit.as_ref().map_or(0, |(_, waited)| *waited);
    let tokens = engine
        .conversation()
        .read()
        .timeline_token_total(unit.timeline);
    let (drafted, draft_ms) = {
        let _draft = draft_permit;
        let t_draft = Instant::now();
        let drafted = draft(engine, spec, base, unit, tokens)?;
        (drafted, t_draft.elapsed().as_millis() as u64)
    };
    let wanted = drafted.wanted;
    let found = drafted.questions;
    if found.is_empty() {
        let last = &drafted.last_answer;
        let opening: String = last.chars().take(160).collect();
        anyhow::bail!("{NO_QUESTION} ({} chars: {opening:?})", last.len());
    }
    // **A short draft is tried again, but not forever**: on the unit's last
    // try what it wrote is sealed, which is more than leaving it with none.
    let last_try =
        failures_of(unit.prior_failures.as_deref(), &unit.marker).saturating_add(1) >= MAX_FAILURES;
    if !drafted.empty.is_empty() && !last_try {
        anyhow::bail!("{SHORT_DRAFT} ({})", drafted.empty.join(", "));
    }
    let t_think = Instant::now();
    let thought = thoughts(engine, frame, spec, &found)?;
    let think_ms = t_think.elapsed().as_millis() as u64;
    // The exemplars' prefill and seal, the calibration's archive and distill.
    let t_exemplars = Instant::now();
    let mut meta = calibration_meta(unit.timeline, &unit.subject);
    meta.insert(MARKER_KEY.to_string(), unit.marker.clone());
    let cases: Vec<(String, String)> = found
        .iter()
        .zip(&thought)
        .map(|(q, t)| (q.clone(), exemplar(t)))
        .collect();
    seal_calibration(engine, frame, &meta, &cases)?;
    tracing::debug!(
        target: "zend::calibrate",
        layer = %spec.layer,
        subject = %unit.subject,
        tokens,
        wanted,
        questions = found.len(),
        permit_wait_ms,
        draft_ms,
        think_ms,
        exemplars_ms = t_exemplars.elapsed().as_millis() as u64,
        "unit calibrated",
    );
    Ok(())
}

/// The selection a question's thinking is decoded under: a live dialogue
/// turn's, every dial at its default.
fn thinking_selection() -> SelectionState {
    SelectionState::new()
}

/// Each of `questions`' thinking, in order: each question asked alone in a
/// transient conversation under the frame's system prompt, decoded to the
/// spec's `think` budget. Each conversation is tombstoned either way. A
/// question the model did not think about fails the unit, which is calibrated
/// again by a later pass.
///
/// **No more in the queue than the scheduler runs.** Every unit's questions
/// share one gate ([`THINKING`]), as wide as the scheduler runs turns: a unit
/// submits a question only while it holds a place, and gives the place back
/// as the question's turn ends, when it submits its next. Asked every question
/// at once, a dozen units stood ~550 turns in the queue against 24 run at a
/// time on BlackWorld (2026-11-15), and a unit none of whose turns moved for a
/// minute gave its whole calibration up ([`FinishOrder`]) — two units in
/// three, each redone from its draft by a later pass. A unit holding no place
/// waits for one; a unit holding some takes more only when one is free
/// ([`Permits::next_place`]), so it never waits on places it keeps from
/// another.
///
/// **A question gives up its conversation the moment its turn ends**, read in
/// the order the turns finish. A transient conversation's recurrent store is
/// never parked — it writes no record to park into — so a question waited on
/// in submit order held its store until every question before it was done: on
/// BlackWorld (2026-10-29) a hundred finished questions at a time, ~15 GiB of
/// state beside a dozen running ones.
fn thoughts(
    engine: &ConversationEngine,
    frame: &Frame<'_>,
    spec: &CalibrateSpec,
    questions: &[String],
) -> anyhow::Result<Vec<String>> {
    let n = questions.len();
    let mut asked: Vec<Option<Sequence>> = Vec::with_capacity(n);
    let mut timelines: Vec<TimelineId> = Vec::with_capacity(n);
    let mut places: Vec<Option<Permit<'static>>> = Vec::with_capacity(n);
    let mut out: Vec<Option<String>> = vec![None; n];
    let mut finished = FinishOrder::new(Vec::new());
    let thought = (|| -> anyhow::Result<()> {
        loop {
            while asked.len() < n {
                let Some(place) = THINKING.next_place(finished.pending()) else {
                    break;
                };
                let i = asked.len();
                // **Background, as the ingest is.** A sweep thinks dozens of
                // questions at once; at `Normal` they rode every forward beside
                // a live turn, where `Low` rows step aside for an `Urgent`
                // decode (`scheduler::band`, `urgent_hold`).
                let conv = engine
                    .new_reserved_conversation(
                        frame.system_prompt,
                        Reserved::Calibration,
                        DecodePriority::Low,
                        frame.config.clone(),
                    )
                    .map_err(|err| anyhow::anyhow!("thinking conversation: {err}"))?;
                let timeline = conv.timeline_id();
                asked.push(Some(conv));
                timelines.push(timeline);
                places.push(Some(place));
                engine.conversation().mark_timeline_transient(timeline);
                engine.set_timeline_summarize(timeline, false);
                engine
                    .set_conversation_metadata(timeline, DRAFT_KEY, &spec.layer)
                    .map_err(|err| anyhow::anyhow!("naming the thinking conversation: {err}"))?;
                let options = TurnOptions {
                    max_tokens: Some(spec.think),
                    selection: thinking_selection(),
                    triggers: Arc::clone(&frame.thinking),
                    tags: vec![TAG.to_string()],
                    ..Default::default()
                };
                let handle = asked[i]
                    .as_mut()
                    .expect("the question's conversation was just opened")
                    .submit_turn_with_options(questions[i].clone(), options)
                    .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
                finished.push(i, handle);
            }
            let Some((i, handle, resp)) = finished
                .next_cancellable(crate::ingest_cancel::token())
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?
            else {
                return Ok(());
            };
            places[i] = None;
            let q = &questions[i];
            let mut conv = asked[i].take().expect("a turn finishes once");
            let thought = resp
                .map_err(|e| anyhow::anyhow!("decode: {e}"))
                .and_then(|resp| {
                    conv.finish_turn(handle, &resp)
                        .map_err(|e| anyhow::anyhow!("finish: {e}"))?;
                    thinking_of(&resp.answer).ok_or_else(|| {
                        let opening: String = resp.text.chars().take(80).collect();
                        anyhow::anyhow!(
                            "the model did not think about {q:?} (finish {:?}, it wrote {opening:?})",
                            resp.stats.finish
                        )
                    })
                });
            let _ = engine.tombstone_timeline(timelines[i]);
            drop(conv);
            out[i] = Some(thought?);
        }
    })();
    // The places go back before the unfinished conversations are retired.
    drop(places);
    drop(finished);
    for (conv, timeline) in asked.iter().zip(&timelines) {
        if conv.is_some() {
            let _ = engine.tombstone_timeline(*timeline);
        }
    }
    thought.map(|()| {
        out.into_iter()
            .map(|t| t.expect("every turn finished"))
            .collect()
    })
}

/// The path a unit's questions name it by: its file's path within its
/// repository, as the prompts' `<<path>>` is, or the whole subject for a unit
/// that is not a file.
fn unit_path(subject: &str) -> &str {
    match crate::repo_path::split(subject).1 {
        "" => subject,
        path => path,
    }
}

/// What a unit's draft wrote: its questions from every prompt, in the order
/// asked, how many the prompts asked for, and the last answer, which a draft
/// that wrote no question is reported by.
struct Drafted {
    questions: Vec<String>,
    wanted: usize,
    /// The required parts that kept no question ([`empty_parts`]); for an
    /// answer with no headings, `all` when it kept under the parts' minimums
    /// together.
    empty: Vec<String>,
    last_answer: String,
}

/// The unit's answer to its layer's calibration prompts, asked together in
/// one turn of a draft whose lineage is the unit — each a part under its own
/// heading ([`super::sections`]), each for as many questions as its rule gives
/// a unit of `tokens`; the draft is tombstoned either way.
///
/// **Every part in one turn, with the reading in front of it.** The parts see
/// each other, so a later one can be told to ask nothing an earlier one asked;
/// a question two parts repeat word for word is kept once. An answer that
/// carries no headings is read as one list, against the parts' counts
/// together.
fn draft(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    unit: &Unit,
    tokens: usize,
) -> anyhow::Result<Drafted> {
    on_a_draft(
        engine,
        &spec.layer,
        base,
        unit.timeline,
        &unit.subject,
        |conv| {
            // A draft has no tools, and one that opens a call anyway writes no
            // question: a file too long for its projection was asked for again
            // with `file_read`, every pass.
            let mut sampling = conv.default_sampling();
            if sampling.tool_call_open_token_id >= 0 {
                sampling
                    .banned_tokens
                    .push(sampling.tool_call_open_token_id);
            }
            // A part a unit is asked for none of is left out of the turn.
            let (asks, wanted): (Vec<&Ask>, Vec<usize>) = spec
                .asks_for(&unit.subject)
                .iter()
                .map(|a| (a, a.questions.for_tokens(tokens)))
                .filter(|&(_, n)| n > 0)
                .unzip();
            let parts: Vec<(&str, String)> = asks
                .iter()
                .zip(&wanted)
                .map(|(a, &n)| (a.name.as_str(), a.prompt_for(&unit.subject, n)))
                .collect();
            let options = TurnOptions {
                max_tokens: Some(DRAFT_MAX_TOKENS),
                sampling: Some(sampling),
                selection: draft_selection(),
                tags: vec![TAG.to_string()],
                ..Default::default()
            };
            let handle = conv
                .submit_turn_with_options(
                    compose(&parts, unit.link.then(|| unit_path(&unit.subject))),
                    options,
                )
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            let resp = handle
                .wait_cancellable(crate::ingest_cancel::token())
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            let answer = answer_of(&resp.answer, 0, &resp.stats.finish);
            conv.finish_turn(handle, &resp)
                .map_err(|e| anyhow::anyhow!("finish: {e}"))?;
            let answer = answer?;
            let total: usize = wanted.iter().sum();
            let names: Vec<&str> = asks.iter().map(|a| a.name.as_str()).collect();
            let mut seen: HashSet<String> = HashSet::new();
            let mut kept: Vec<String> = Vec::new();
            let empty: Vec<String> = match split(&answer, &names) {
                Some(bodies) => {
                    let mut parts: Vec<(&str, usize, usize)> = Vec::with_capacity(asks.len());
                    for ((body, &n), ask) in bodies.iter().zip(&wanted).zip(&asks) {
                        let before = kept.len();
                        kept.extend(
                            questions(body, n)
                                .into_iter()
                                .filter(|q| seen.insert(q.to_lowercase())),
                        );
                        let got = kept.len() - before;
                        // An optional part is told to stop short rather than
                        // pad, so writing few is its answer, not a failure.
                        if ask.questions.min > 0 {
                            short_of(unit, &ask.name, n, got, body);
                        }
                        parts.push((ask.name.as_str(), ask.questions.min, got));
                    }
                    empty_parts(&parts)
                        .into_iter()
                        .map(str::to_string)
                        .collect()
                }
                None => {
                    kept = questions(&answer, total)
                        .into_iter()
                        .filter(|q| seen.insert(q.to_lowercase()))
                        .collect();
                    short_of(unit, "all", total, kept.len(), &answer);
                    let least: usize = asks.iter().map(|a| a.questions.min).sum();
                    if kept.len() < least {
                        vec!["all".to_string()]
                    } else {
                        Vec::new()
                    }
                }
            };
            Ok(Drafted {
                questions: kept,
                wanted: total,
                empty,
                last_answer: answer,
            })
        },
    )
}

/// **A part that writes far short of its count is said so, with what it
/// wrote.** The draft is gone once read, and a unit one of whose parts kept
/// nothing reads like one with a small budget.
fn short_of(unit: &Unit, part: &str, wanted: usize, got: usize, wrote: &str) {
    if got * 2 >= wanted {
        return;
    }
    let opening: String = wrote.chars().take(240).collect();
    tracing::warn!(
        target: "zend::calibrate",
        subject = %unit.subject,
        part,
        wanted,
        got,
        chars = wrote.len(),
        "a calibration part wrote under half its questions: {opening:?}",
    );
}

/// Run `f` on a draft: a fork of `base` whose lineage is the unit at `unit` —
/// so the unit's reading is the draft's opening and its recurrent memory the
/// one the reading left — named as `layer`'s draft so a crash's is retired.
/// The draft is tombstoned whatever `f` returns.
pub(super) fn on_a_draft<T>(
    engine: &ConversationEngine,
    layer: &str,
    base: &Sequence,
    unit: TimelineId,
    subject: &str,
    f: impl FnOnce(&mut Sequence) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let mut conv = base
        .fork()
        .map_err(|e| anyhow::anyhow!("draft conversation: {e}"))?;
    let timeline = conv.timeline_id();
    let named = {
        let e = engine;
        e.conversation().mark_timeline_transient(timeline);
        e.set_timeline_summarize(timeline, false);
        e.set_conversation_metadata(timeline, DRAFT_KEY, layer)
            .and_then(|()| e.set_forked_from(timeline, unit))
            .map_err(|err| anyhow::anyhow!("draft lineage: {err}"))
    };
    let out = named.and_then(|()| {
        if let Err(err) = conv.seed_recurrent_from_lineage() {
            tracing::warn!(
                target: "zend::calibrate",
                subject,
                "seeding the draft's recurrent memory from its unit failed: {err:#}",
            );
        }
        f(&mut conv)
    });
    let _ = engine.tombstone_timeline(timeline);
    out
}

#[cfg(test)]
mod tests {
    use super::super::ask::RawAsk;
    use super::super::count::QuestionCount;
    use super::super::profile::{Profile, RawProfile};
    use super::*;
    use candle_conversation::projection::{Conversation, LayerId};
    use candle_conversation::substrate::TurnPartWrite;
    use candle_conversation::turn_layout::{KvSpan, TurnLayout, TurnSegment};
    use candle_conversation::Role;

    /// Failures count against the marker they happened under: a new reading
    /// starts again.
    #[test]
    fn a_failure_counts_only_against_its_own_marker() {
        assert_eq!(failures_of(None, "m:1"), 0);
        assert_eq!(failures_of(Some("m:1 2"), "m:1"), 2);
        assert_eq!(failures_of(Some("m:1 2"), "m:2"), 0);
        assert_eq!(failures_of(Some("garbled"), "m:1"), 0);
    }

    /// **A draft that left a required part empty stopped short** — `lift.rs`
    /// with 16 definitions and no uses or purpose; a part that wrote fewer than
    /// asked, and an optional part that wrote none, did not.
    #[test]
    fn a_required_part_with_no_question_is_a_draft_stopped_short() {
        assert_eq!(
            empty_parts(&[
                ("definitions", 4, 16),
                ("uses", 2, 0),
                ("messages", 0, 0),
                ("purpose", 2, 0)
            ]),
            ["uses", "purpose"]
        );
        assert!(
            empty_parts(&[("definitions", 4, 6), ("uses", 2, 3), ("messages", 0, 0)]).is_empty()
        );
        assert!(empty_parts(&[]).is_empty());
    }

    /// **Only the unit's own failures count against it**: a draft that wrote
    /// nothing or stopped short, not a decode lost to a full card.
    #[test]
    fn only_the_units_own_failures_count_against_it() {
        assert!(units_own_failure(
            "the draft wrote no question (12 chars: \"I'll read it\")"
        ));
        assert!(units_own_failure(
            "the draft left a part with no question (uses, purpose)"
        ));
        assert!(!units_own_failure(
            "decode: the turn made no progress for 60 s: it was not admitted, or was lost"
        ));
    }

    /// A file unit is named by its path within its repository; a unit that is
    /// not a file, by its whole subject.
    #[test]
    fn a_unit_is_named_by_its_path_within_its_repository() {
        assert_eq!(unit_path("yenbot/AGENTS.md"), "AGENTS.md");
        assert_eq!(unit_path("candle/zend/src/lib.rs"), "zend/src/lib.rs");
        assert_eq!(unit_path("journals"), "journals");
    }

    /// **Every keyed unit is found; only a whole, finished one is ready.** A
    /// unit with a turn still to arrive, and one cut off mid-thought, are
    /// found but not ready; a calibration and an unkeyed conversation of the
    /// group are not units.
    #[test]
    fn a_unit_is_ready_only_whole_and_finished() {
        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).unwrap();
        let group = GroupId::from_raw(1).unwrap();
        let spec = CalibrateSpec {
            layer: "code_reading".into(),
            think: 64,
            profiles: vec![Profile::of(RawProfile {
                name: "default".into(),
                files: Vec::new(),
                prompts: vec![RawAsk {
                    name: "definitions".into(),
                    questions: QuestionCount {
                        per_tokens: 250,
                        min: 4,
                        max: 4,
                    },
                    prompt: "ask <<n>>".into(),
                }],
            })
            .unwrap()],
            guidance: None,
        };
        let said = |tl: TimelineId, text: &str| {
            conv.record_turn(
                tl,
                Role::User,
                TurnPartWrite {
                    layout: TurnLayout::new(vec![TurnSegment::Assistant {
                        text: Some(text.into()),
                        kv: KvSpan::new(0, 4),
                    }]),
                    token_count: 4,
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .unwrap();
        };
        let answered = |tl: TimelineId| said(tl, "It reads the file.");
        let reading = |path: &str, version: u32| format!("v{version}:{path}@1");
        let keyed = |tl: TimelineId, path: &str, key: &str| {
            conv.set_conversation_metadata(tl, CONTENT_KEY, key)
                .unwrap();
            conv.set_conversation_metadata(tl, PATH_KEY, path).unwrap();
        };
        let [whole, gappy, cal, unkeyed, cut, old, built, built_old] =
            [11u64, 12, 13, 14, 15, 16, 17, 18].map(|raw| TimelineId::from_raw(raw).unwrap());
        for tl in [whole, gappy, cal, unkeyed, cut, old, built, built_old] {
            conv.register_timeline(tl, layer, group);
        }
        let current = KEY_VERSION;
        answered(whole);
        keyed(whole, "candle/a.rs", &reading("candle/a.rs", current));
        conv.update(|s| s.restore_gap_turn(gappy));
        answered(gappy);
        keyed(gappy, "candle/b.rs", &reading("candle/b.rs", current));
        answered(cal);
        keyed(cal, "candle/c.rs", &reading("candle/c.rs", current));
        conv.set_conversation_metadata(cal, Substrate::CALIBRATES_KEY, &whole.raw().to_string())
            .unwrap();
        answered(unkeyed);
        said(cut, "<think>It reads the fi");
        keyed(cut, "candle/d.rs", &reading("candle/d.rs", current));
        // Read under the version before: waits to be read again.
        answered(old);
        keyed(old, "candle/e.rs", &reading("candle/e.rs", current - 1));
        // Units built on a reading, keyed apart, judged by the reading's key.
        for (tl, version) in [(built, current), (built_old, current - 1)] {
            answered(tl);
            keyed(tl, "candle/f.rs", &format!("u1:review:{}", tl.raw()));
            conv.set_conversation_metadata(tl, FILE_KEY, &reading("candle/f.rs", version))
                .unwrap();
        }

        let units = units_of(&conv.read(), group, &spec);
        let unit = |timeline: TimelineId, subject: &str, key: String, ready: bool| Unit {
            timeline,
            subject: subject.into(),
            marker: spec.marker(&key, subject, false),
            key,
            ready,
            prior_failures: None,
            link: false,
        };
        assert_eq!(
            units,
            vec![
                unit(whole, "candle/a.rs", reading("candle/a.rs", current), true),
                unit(gappy, "candle/b.rs", reading("candle/b.rs", current), false),
                unit(cut, "candle/d.rs", reading("candle/d.rs", current), false),
                unit(
                    old,
                    "candle/e.rs",
                    reading("candle/e.rs", current - 1),
                    false
                ),
                unit(built, "candle/f.rs", "u1:review:17".into(), true),
                unit(built_old, "candle/f.rs", "u1:review:18".into(), false),
            ]
        );
    }
    /// A question's thinking is decoded as a live dialogue turn is: no dial
    /// is set, so thinking stays on at the dialogue's default effort.
    #[test]
    fn a_question_is_thought_about_as_a_live_turn_is() {
        let s = thinking_selection();
        assert_eq!(s.get("thinking_effort"), None);
        assert_eq!(s.optional(NO_THINK_SELECTOR), None);
    }

    /// A draft is asked without reasoning and without a tool block.
    #[test]
    fn a_draft_neither_reasons_nor_sees_tools() {
        let s = draft_selection();
        assert_eq!(s.get("thinking_effort"), Some("off"));
        assert_eq!(s.optional(NO_THINK_SELECTOR), Some(OptionalState::Present));
        assert_eq!(
            s.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Absent)
        );
    }
}
