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
use std::sync::{Arc, Mutex};

use candle_conversation::chain_health::chain_break;
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::projection::{
    DecodePriority, GroupId, OptionalState, Reserved, SelectionState, TimelineId,
    NO_THINK_SELECTOR, TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::substrate::Substrate;
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig, TurnOptions};

use super::permits::Permits;
use super::plan::{plan, Prior, Unit};
use super::questions::questions;
use super::spec::CalibrateSpec;
use super::thinking::{exemplar, thinking_of};
use crate::branch_ingest::keys::CONTENT_KEY;
use crate::code_read::PATH_KEY;
use crate::layers::keys::SUBJECT_KEY;
use crate::layers::run::{answer_of, LAYER_PARALLELISM};

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
    pub base: &'a Mutex<Sequence>,
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

/// Calibrate every unit of `layer` that is ready and not calibrated, and
/// retire every calibration of a unit no longer there.
pub fn calibrate_layer(
    engine: &Mutex<ConversationEngine>,
    layer: &Layer<'_>,
    frame: &Frame<'_>,
) -> Report {
    let spec = layer.spec;
    let mut report = Report::default();
    let todo = {
        let e = engine.lock().unwrap();
        for (tl, of) in e.conversations_with_metadata_key(DRAFT_KEY) {
            if of == spec.layer && e.tombstone_timeline(tl).is_ok() {
                report.retired += 1;
            }
        }
        let units = units(&e, layer);
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
    let drafts = Permits::new(LAYER_PARALLELISM);
    std::thread::scope(|s| {
        for _ in 0..CALIBRATE_PARALLELISM.min(todo.run.len()).max(1) {
            // Only a shutdown stops it: a pass asked for meanwhile waits for
            // the layer to finish, since this runs ahead of every pass.
            s.spawn(|| loop {
                if crate::ingest_cancel::cancelled() {
                    return;
                }
                let Some(unit) = todo.run.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    return;
                };
                match calibrate_one(engine, layer, frame, &drafts, unit) {
                    Ok(()) => {
                        sealed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        if !crate::ingest_cancel::cancelled() {
                            tracing::warn!(
                                target: "zend::calibrate",
                                layer = %spec.layer,
                                subject = %unit.subject,
                                "calibration failed (calibrated again next pass): {e:#}",
                            );
                        }
                    }
                }
            });
        }
    });
    report.sealed = sealed.into_inner();
    report.failed = failed.into_inner();
    tracing::info!(
        target: "zend::calibrate",
        layer = %spec.layer,
        calibrated = report.done,
        sealed = report.sealed,
        failed = report.failed,
        waiting = report.waiting,
        retired = report.retired,
        "the layer's calibration pass ended",
    );
    report
}

/// Every unit of `layer`: each conversation of its group committed under a
/// content key and naming its subject — ready when its chain finished and
/// every turn holds its content.
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
            Some(Unit {
                timeline: tl,
                subject: subject.clone(),
                key: key.clone(),
                marker: spec.marker(key),
                ready: chain_break(substrate, tl).is_none() && substrate.holds_whole(tl),
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

/// Ask `unit` for its questions — holding one of `drafts` while it does — and
/// calibrate them.
fn calibrate_one(
    engine: &Mutex<ConversationEngine>,
    layer: &Layer<'_>,
    frame: &Frame<'_>,
    drafts: &Permits,
    unit: &Unit,
) -> anyhow::Result<()> {
    let spec = layer.spec;
    let asked = {
        let _draft = drafts.acquire();
        draft(engine, layer, unit)?
    };
    let found = questions(&asked, spec.questions);
    if found.is_empty() {
        anyhow::bail!("the draft wrote no question ({} chars)", asked.len());
    }
    let thought = thoughts(engine, frame, spec, &found)?;
    let mut conv = engine
        .lock()
        .unwrap()
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
    engine
        .lock()
        .unwrap()
        .conversation()
        .mark_timeline_transient(timeline);
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
    let e = engine.lock().unwrap();
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
fn thoughts(
    engine: &Mutex<ConversationEngine>,
    frame: &Frame<'_>,
    spec: &CalibrateSpec,
    questions: &[String],
) -> anyhow::Result<Vec<String>> {
    let mut asked: Vec<(Sequence, TimelineId)> = Vec::with_capacity(questions.len());
    let opened: anyhow::Result<()> = questions.iter().try_for_each(|_| {
        let e = engine.lock().unwrap();
        let conv = e
            .new_reserved_conversation(
                frame.system_prompt,
                Reserved::Calibration,
                DecodePriority::Normal,
                frame.config.clone(),
            )
            .map_err(|err| anyhow::anyhow!("thinking conversation: {err}"))?;
        let timeline = conv.timeline_id();
        asked.push((conv, timeline));
        e.conversation().mark_timeline_transient(timeline);
        e.set_timeline_summarize(timeline, false);
        e.set_conversation_metadata(timeline, DRAFT_KEY, &spec.layer)
            .map_err(|err| anyhow::anyhow!("naming the thinking conversation: {err}"))
    });
    let thought = opened.and_then(|()| {
        let mut handles = Vec::with_capacity(questions.len());
        for ((conv, _), q) in asked.iter_mut().zip(questions) {
            let options = TurnOptions {
                max_tokens: Some(spec.think),
                selection: thinking_selection(),
                triggers: Arc::clone(&frame.thinking),
                tags: vec![TAG.to_string()],
                ..Default::default()
            };
            let handle = conv
                .submit_turn_with_options(q.clone(), options)
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            handles.push(handle);
        }
        let mut out = Vec::with_capacity(questions.len());
        for (((conv, _), q), handle) in asked.iter_mut().zip(questions).zip(handles) {
            let resp = handle
                .wait_cancellable(crate::ingest_cancel::token())
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            conv.finish_turn(handle, &resp)
                .map_err(|e| anyhow::anyhow!("finish: {e}"))?;
            let thought = thinking_of(&resp.answer).ok_or_else(|| {
                let opening: String = resp.text.chars().take(80).collect();
                anyhow::anyhow!(
                    "the model did not think about {q:?} (finish {:?}, it wrote {opening:?})",
                    resp.stats.finish
                )
            })?;
            out.push(thought);
        }
        Ok(out)
    });
    let e = engine.lock().unwrap();
    for (_, timeline) in &asked {
        let _ = e.tombstone_timeline(*timeline);
    }
    thought
}

/// The unit's answer to its layer's calibration prompt, asked in a draft
/// whose lineage is the unit; the draft is tombstoned either way.
fn draft(
    engine: &Mutex<ConversationEngine>,
    layer: &Layer<'_>,
    unit: &Unit,
) -> anyhow::Result<String> {
    let spec = layer.spec;
    let mut conv = layer
        .base
        .lock()
        .unwrap()
        .fork()
        .map_err(|e| anyhow::anyhow!("draft conversation: {e}"))?;
    let timeline = conv.timeline_id();
    let named = {
        let e = engine.lock().unwrap();
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
        let options = TurnOptions {
            max_tokens: Some(DRAFT_MAX_TOKENS),
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
    let _ = engine.lock().unwrap().tombstone_timeline(timeline);
    answer
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::{Conversation, LayerId};
    use candle_conversation::substrate::TurnPartWrite;
    use candle_conversation::turn_layout::{KvSpan, TurnLayout, TurnSegment};
    use candle_conversation::Role;

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
        let keyed = |tl: TimelineId, path: &str| {
            conv.set_conversation_metadata(tl, CONTENT_KEY, &format!("key:{path}"))
                .unwrap();
            conv.set_conversation_metadata(tl, PATH_KEY, path).unwrap();
        };
        let [whole, gappy, cal, unkeyed, cut] =
            [11u64, 12, 13, 14, 15].map(|raw| TimelineId::from_raw(raw).unwrap());
        for tl in [whole, gappy, cal, unkeyed, cut] {
            conv.register_timeline(tl, layer, group);
        }
        answered(whole);
        keyed(whole, "candle/a.rs");
        conv.write().restore_gap_turn(gappy);
        answered(gappy);
        keyed(gappy, "candle/b.rs");
        answered(cal);
        keyed(cal, "candle/c.rs");
        conv.set_conversation_metadata(cal, Substrate::CALIBRATES_KEY, &whole.raw().to_string())
            .unwrap();
        answered(unkeyed);
        said(cut, "<think>It reads the fi");
        keyed(cut, "candle/d.rs");

        let units = units_of(&conv.read(), group, &spec);
        assert_eq!(
            units,
            vec![
                Unit {
                    timeline: whole,
                    subject: "candle/a.rs".into(),
                    key: "key:candle/a.rs".into(),
                    marker: spec.marker("key:candle/a.rs"),
                    ready: true,
                },
                Unit {
                    timeline: gappy,
                    subject: "candle/b.rs".into(),
                    key: "key:candle/b.rs".into(),
                    marker: spec.marker("key:candle/b.rs"),
                    ready: false,
                },
                Unit {
                    timeline: cut,
                    subject: "candle/d.rs".into(),
                    key: "key:candle/d.rs".into(),
                    marker: spec.marker("key:candle/d.rs"),
                    ready: false,
                },
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
