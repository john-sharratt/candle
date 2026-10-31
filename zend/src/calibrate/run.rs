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

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_conversation::chain_health::chain_break;
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::projection::{
    DecodePriority, GroupId, OptionalState, Reserved, SelectionState, TimelineId,
    NO_THINK_SELECTOR, TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::{ThinkMode, TriggerRegistry};
use candle_conversation::substrate::Substrate;
use candle_conversation::{ConversationEngine, FinishOrder, Sequence, SequenceConfig, TurnOptions};

use super::permits::Permits;
use super::plan::{plan, Prior, Unit};
use super::questions::questions;
use super::spec::CalibrateSpec;
use super::thinking::{exemplar, thinking_of};
use crate::branch_ingest::keys::{key_version, CONTENT_KEY, KEY_VERSION};
use crate::code_read::PATH_KEY;
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
/// asked for without reasoning.
const DRAFT_MAX_TOKENS: usize = 2048;

/// On a unit, the marker its calibration last failed under and how many times.
const FAILED_KEY: &str = "calibration_failed";

/// Passes whose draft may write no question before the unit is left
/// uncalibrated. Four units answered every draft the same way and were asked
/// again every pass on both members (2026-10-31). A new reading is a new
/// marker, and is tried afresh.
const MAX_FAILURES: u32 = 3;

/// How a calibration that failed for want of a question says so.
const NO_QUESTION: &str = "the draft wrote no question";

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

/// What one layer's pass did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Units already calibrated.
    pub done: usize,
    /// Units calibrated by this pass.
    pub sealed: usize,
    /// Units that failed, calibrated again by a later pass.
    pub failed: usize,
    /// Units another member was calibrating, or had calibrated by the time
    /// this one held its lease ([`ingest_lease::share_calibration`]). Their
    /// calibrations arrive with replication.
    pub elsewhere: usize,
    /// Units neither calibrated nor ready, calibrated by a later pass once
    /// they are.
    pub waiting: usize,
    /// Calibrations retired, and drafts a crash left.
    pub retired: usize,
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
    let cursor = AtomicUsize::new(0);
    let sealed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let elsewhere = AtomicUsize::new(0);
    let drafts = Permits::new(LAYER_PARALLELISM);
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
                // **One member calibrates a unit.** Every member's pass plans
                // the same units from the same replicated substrate; the one
                // the vote names asks for them, and a calibration handed back
                // is looked for again before it is asked for twice.
                let _lease = match ingest_lease::share_calibration(&unit.marker) {
                    Share::Theirs => {
                        elsewhere.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    Share::Mine(lease) => {
                        if lease.recheck() && calibrated(engine, &unit.marker) {
                            elsewhere.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        lease
                    }
                };
                match calibrate_one(engine, spec, layer.base, frame, Some(&drafts), unit) {
                    Ok(()) => {
                        sealed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        if !crate::ingest_cancel::cancelled() {
                            // Only the unit's own failure counts against it: a
                            // decode lost to a restart or a full card says
                            // nothing about the unit, and is simply tried again.
                            if format!("{e:#}").starts_with(NO_QUESTION) {
                                let tries =
                                    failures_of(unit.prior_failures.as_deref(), &unit.marker)
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
                }
            });
        }
    });
    report.sealed = sealed.into_inner();
    report.failed = failed.into_inner();
    report.elsewhere = elsewhere.into_inner();
    tracing::info!(
        target: "zend::calibrate",
        layer = %spec.layer,
        calibrated = report.done,
        sealed = report.sealed,
        failed = report.failed,
        elsewhere = report.elsewhere,
        waiting = report.waiting,
        retired = report.retired,
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
            let marker = spec.marker(key);
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
            })
        })
        .collect();
    units.sort_by_key(|u| u.timeline);
    units
}

/// The config a calibration is prefilled under: the frame's, lossless until
/// the seal, as the tools' calibrations are — the signature is captured from
/// every token's Q at the seal, and a quantized K/V would capture it from a
/// degraded one.
fn calibration_config(frame: &SequenceConfig) -> SequenceConfig {
    SequenceConfig {
        kv_lossless: true,
        ..frame.clone()
    }
}

/// The selection a draft is asked under: no reasoning, and no tool block.
fn draft_selection() -> SelectionState {
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
    let unit = Unit {
        timeline,
        subject: subject.to_string(),
        key: key.to_string(),
        marker: spec.marker(key),
        ready: true,
        prior_failures: None,
    };
    calibrate_one(ctx.engine, spec, base, &frame, None, &unit)
}

/// Ask `unit` for its questions — holding one of `drafts`, when given, while
/// it does — and calibrate them.
fn calibrate_one(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    frame: &Frame<'_>,
    drafts: Option<&Permits>,
    unit: &Unit,
) -> anyhow::Result<()> {
    let t_wait = Instant::now();
    let (asked, permit_wait_ms, draft_ms) = {
        let _draft = drafts.map(Permits::acquire);
        let waited = t_wait.elapsed();
        let t_draft = Instant::now();
        let asked = draft(engine, spec, base, unit)?;
        (
            asked,
            waited.as_millis() as u64,
            t_draft.elapsed().as_millis() as u64,
        )
    };
    let found = questions(&asked, spec.questions);
    if found.is_empty() {
        let opening: String = asked.chars().take(160).collect();
        anyhow::bail!("{NO_QUESTION} ({} chars: {opening:?})", asked.len());
    }
    let t_think = Instant::now();
    let thought = thoughts(engine, frame, spec, &found)?;
    let think_ms = t_think.elapsed().as_millis() as u64;
    // The exemplars' prefill and seal, the calibration's archive and distill.
    let t_exemplars = Instant::now();
    let mut conv = engine
        .new_reserved_conversation(
            frame.system_prompt,
            Reserved::Calibration,
            DecodePriority::Normal,
            calibration_config(frame.config),
        )
        .map_err(|e| anyhow::anyhow!("calibration conversation: {e}"))?;
    let timeline = conv.timeline_id();
    // Its K/V is dropped at the distill; written cold, it would be bytes
    // the next compaction deletes.
    engine.conversation().mark_timeline_transient(timeline);
    let mut meta = BTreeMap::new();
    meta.insert("kind".to_string(), TAG.to_string());
    meta.insert(MARKER_KEY.to_string(), unit.marker.clone());
    meta.insert(
        Substrate::CALIBRATES_KEY.to_string(),
        unit.timeline.raw().to_string(),
    );
    meta.insert(SUBJECT_KEY.to_string(), unit.subject.clone());
    let sealed = conv
        .set_metadata_many(&meta)
        .map_err(|e| anyhow::anyhow!("naming the calibration: {e}"))
        .and_then(|()| {
            let cases: Vec<(String, String, Vec<String>)> = found
                .iter()
                .zip(&thought)
                .map(|(q, t)| (q.clone(), exemplar(t), vec![TAG.to_string()]))
                .collect();
            let (handle, _) = conv
                .submit_prefilled_turn_group(&cases, SelectionState::new())
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            let resp = handle
                .wait_cancellable(crate::ingest_cancel::token())
                .map_err(|e| anyhow::anyhow!("prefill: {e}"))?;
            conv.finish_turn(handle, &resp)
                .map_err(|e| anyhow::anyhow!("seal: {e}"))?;
            Ok(())
        });
    let e = engine;
    if let Err(err) = sealed {
        let _ = e.tombstone_timeline(timeline);
        return Err(err);
    }
    e.set_conversation_archived(timeline, true)
        .map_err(|err| anyhow::anyhow!("archiving the calibration: {err}"))?;
    if e.timeline_has_kv(timeline) && !e.is_timeline_distilled(timeline) {
        e.distill_timeline(timeline, DistillMode::ProvenanceOnly)
            .map_err(|err| anyhow::anyhow!("distilling the calibration: {err}"))?;
    }
    tracing::debug!(
        target: "zend::calibrate",
        layer = %spec.layer,
        subject = %unit.subject,
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

/// Each of `questions`' thinking, in order: every question asked at once, each
/// alone in a transient conversation under the frame's system prompt, decoded
/// to the spec's `think` budget. Each conversation is tombstoned either way.
/// A question the model did not think about fails the unit, which is
/// calibrated again by a later pass.
///
/// **A question gives up its conversation the moment its turn ends**, read in
/// the order the turns finish ([`FinishOrder`]). A transient conversation's
/// recurrent store is never parked — it writes no record to park into — so a
/// question waited on in submit order held its store until every question
/// before it was done: on BlackWorld (2026-10-29) a hundred finished questions
/// at a time, ~15 GiB of state beside a dozen running ones.
fn thoughts(
    engine: &ConversationEngine,
    frame: &Frame<'_>,
    spec: &CalibrateSpec,
    questions: &[String],
) -> anyhow::Result<Vec<String>> {
    let mut asked: Vec<Option<Sequence>> = Vec::with_capacity(questions.len());
    let mut timelines: Vec<TimelineId> = Vec::with_capacity(questions.len());
    let opened: anyhow::Result<()> = questions.iter().try_for_each(|_| {
        let e = engine;
        let conv = e
            .new_reserved_conversation(
                frame.system_prompt,
                Reserved::Calibration,
                DecodePriority::Normal,
                frame.config.clone(),
            )
            .map_err(|err| anyhow::anyhow!("thinking conversation: {err}"))?;
        let timeline = conv.timeline_id();
        asked.push(Some(conv));
        timelines.push(timeline);
        e.conversation().mark_timeline_transient(timeline);
        e.set_timeline_summarize(timeline, false);
        e.set_conversation_metadata(timeline, DRAFT_KEY, &spec.layer)
            .map_err(|err| anyhow::anyhow!("naming the thinking conversation: {err}"))
    });
    let thought = opened.and_then(|()| {
        let mut handles = Vec::with_capacity(questions.len());
        for (conv, q) in asked.iter_mut().zip(questions) {
            let options = TurnOptions {
                max_tokens: Some(spec.think),
                selection: thinking_selection(),
                triggers: Arc::clone(&frame.thinking),
                tags: vec![TAG.to_string()],
                ..Default::default()
            };
            let handle = conv
                .as_mut()
                .expect("every question's conversation is open")
                .submit_turn_with_options(q.clone(), options)
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            handles.push(handle);
        }
        let mut finished = FinishOrder::new(handles);
        let mut out: Vec<Option<String>> = vec![None; questions.len()];
        while let Some((i, handle, resp)) = finished
            .next_cancellable(crate::ingest_cancel::token())
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?
        {
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
        Ok(out
            .into_iter()
            .map(|t| t.expect("every turn finished"))
            .collect())
    });
    let e = engine;
    for (conv, timeline) in asked.iter().zip(&timelines) {
        if conv.is_some() {
            let _ = e.tombstone_timeline(*timeline);
        }
    }
    thought
}

/// The unit's answer to its layer's calibration prompt, asked in a draft
/// whose lineage is the unit; the draft is tombstoned either way.
fn draft(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    unit: &Unit,
) -> anyhow::Result<String> {
    let mut conv = base
        .fork()
        .map_err(|e| anyhow::anyhow!("draft conversation: {e}"))?;
    let timeline = conv.timeline_id();
    let named = {
        let e = engine;
        e.conversation().mark_timeline_transient(timeline);
        e.set_timeline_summarize(timeline, false);
        e.set_conversation_metadata(timeline, DRAFT_KEY, &spec.layer)
            .and_then(|()| e.set_forked_from(timeline, unit.timeline))
            .map_err(|err| anyhow::anyhow!("draft lineage: {err}"))
    };
    let answer = named.and_then(|()| {
        if let Err(err) = conv.seed_recurrent_from_lineage() {
            tracing::warn!(
                target: "zend::calibrate",
                subject = %unit.subject,
                "seeding the draft's recurrent memory from its unit failed: {err:#}",
            );
        }
        // A draft has no tools, and one that opens a call anyway writes no
        // question: a file too long for its projection was asked for again
        // with `file_read`, every pass.
        let mut sampling = conv.default_sampling();
        if sampling.tool_call_open_token_id >= 0 {
            sampling
                .banned_tokens
                .push(sampling.tool_call_open_token_id);
        }
        let options = TurnOptions {
            max_tokens: Some(DRAFT_MAX_TOKENS),
            sampling: Some(sampling),
            selection: draft_selection(),
            tags: vec![TAG.to_string()],
            ..Default::default()
        };
        let handle = conv
            .submit_turn_with_options(spec.prompt_for(&unit.subject), options)
            .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
        let resp = handle
            .wait_cancellable(crate::ingest_cancel::token())
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let answer = answer_of(&resp.answer, 0, &resp.stats.finish);
        conv.finish_turn(handle, &resp)
            .map_err(|e| anyhow::anyhow!("finish: {e}"))?;
        answer
    });
    let _ = engine.tombstone_timeline(timeline);
    answer
}

#[cfg(test)]
mod tests {
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
            questions: 4,
            think: 64,
            prompt: "ask".into(),
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
            marker: spec.marker(&key),
            key,
            ready,
            prior_failures: None,
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
