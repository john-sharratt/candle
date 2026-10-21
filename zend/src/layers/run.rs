//! Reading a layer's units: what the layer holds, and the worker pool that
//! runs each queued unit as a hidden conversation of its own.
//!
//! A unit's conversation is a conversation like any other, run the way
//! `code_reading`'s files are (`crate::code_read`): forked from its layer's
//! base, its lineage the unit it builds on — so that unit's whole conversation,
//! and the units under it, are this one's opening — asked its layer's
//! question, answered under its layer's thinking dial, tagged, and let go.
//! What it says IS the unit.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::chain_health::chain_break;
use candle_conversation::projection::{
    GroupId, OptionalState, SelectionState, TimelineId, NO_THINK_SELECTOR,
};
use candle_conversation::stencil::ThinkMode;
use candle_conversation::{ConversationEngine, Sequence, TurnOptions, TurnText};

use super::edges::edges;
use super::keys::{
    deps_value, parse_deps, AFTER_KEY, BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY, DEPS_KEY, FILE_KEY,
    LAYER_KEY, QUANT_KEY, SUBJECT_KEY,
};
use super::plan::{dep_version, Done, LiveFiles};
use super::spec::LayerSpec;
use crate::branch_ingest::keys::branches_value;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::Failures;
use crate::ingest_snapshots::{finished, Unit};
use crate::refresh_ctx::RefreshContext;

/// Units read concurrently. A unit's conversation inherits its file's whole
/// reading as its lineage — the same context `code_reading`'s own workers
/// carry — so it is held to `code_reading`'s width, whose ceiling is the
/// card's expert zone rather than throughput (`crate::code_read::
/// CODE_READ_PARALLELISM`).
pub const LAYER_PARALLELISM: usize = crate::code_read::CODE_READ_PARALLELISM;

/// Failures one layer's pass tolerates before it stops.
pub const MAX_FAILURES: usize = 16;

/// One unit to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitJob {
    /// The content key it commits under.
    pub key: String,
    /// What it is of.
    pub subject: String,
    /// The key of the unit it builds on.
    pub after: String,
    /// The conversation it continues: the unit it builds on, or the priming
    /// chain's end for a day read from the dialogue itself.
    pub parent: Option<TimelineId>,
    /// Its file's `code_reading` key; `None` for a day.
    pub file_key: Option<String>,
    pub branches: Vec<String>,
    pub at: Option<String>,
    /// What it is asked.
    pub prompt: String,
    /// Each scoped group it may retrieve from, and what it may retrieve there.
    pub scopes: Vec<(GroupId, Arc<HashSet<TimelineId>>)>,
}

/// Every unit `layer` holds that finished, and the conversations of it that
/// did not — begun and never committed, or committed by a chain that stopped
/// part-way — which no pass will finish.
pub fn committed(engine: &Mutex<ConversationEngine>, layer: &str) -> (Vec<Done>, Vec<TimelineId>) {
    let e = engine.lock().unwrap();
    let timelines = e.find_conversations_by_metadata(LAYER_KEY, layer);
    let metas: Vec<(TimelineId, BTreeMap<String, String>)> = timelines
        .into_iter()
        .filter_map(|tl| Some((tl, e.conversation_metadata(tl)?)))
        .collect();
    let conv = e.conversation();
    let substrate = conv.read();
    let mut done = Vec::new();
    let mut broken = Vec::new();
    for (timeline, meta) in metas {
        let (Some(key), Some(subject), Some(after)) = (
            meta.get(CONTENT_KEY),
            meta.get(SUBJECT_KEY),
            meta.get(AFTER_KEY),
        ) else {
            broken.push(timeline);
            continue;
        };
        if chain_break(&substrate, timeline).is_some() {
            broken.push(timeline);
            continue;
        }
        done.push(Done {
            timeline,
            key: key.clone(),
            subject: subject.clone(),
            after: after.clone(),
            deps: meta
                .get(DEPS_KEY)
                .map(|v| parse_deps(v))
                .unwrap_or_default(),
            stamp: substrate
                .first_turn_stamp(timeline.raw())
                .unwrap_or_default(),
            quant: meta.get(QUANT_KEY).and_then(|q| q.parse().ok()),
        });
    }
    (done, broken)
}

/// Every conversation of `layer` — `--wipe-layer`'s for a layer of units.
pub fn all_of(engine: &ConversationEngine, layer: &str) -> Vec<TimelineId> {
    engine.find_conversations_by_metadata(LAYER_KEY, layer)
}

/// Retire every conversation of `layer` begun and never committed — what a
/// crash leaves. Called once per boot, before any pass runs, as
/// `code_read::retire_crashed_partials` is: a pass can overlap a pool, and an
/// in-flight unit is indistinguishable from a crashed one by metadata alone.
pub fn retire_crashed_partials(engine: &Mutex<ConversationEngine>, layer: &str) {
    let e = engine.lock().unwrap();
    let mut retired = 0usize;
    for tl in e.find_conversations_by_metadata(LAYER_KEY, layer) {
        let committed = e
            .conversation_metadata(tl)
            .is_some_and(|m| m.contains_key(CONTENT_KEY));
        if !committed && e.tombstone_timeline(tl).is_ok() {
            retired += 1;
        }
    }
    if retired > 0 {
        tracing::info!(
            target: "zend::layers",
            layer,
            retired,
            "retired the layer's units a crash left uncommitted",
        );
    }
}

/// What the dependency answer of a unit resolves against: every file version
/// a branch holds, and their paths.
pub struct EdgeFiles<'a> {
    pub live: &'a LiveFiles,
    pub known: &'a [&'a str],
}

/// Read `jobs` for `spec`'s layer, forking each off `base`. Answers how many
/// failed; each failure is retried by a later pass.
pub fn ingest_units(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    jobs: &[UnitJob],
    base: &Mutex<Sequence>,
    files: Option<&EdgeFiles<'_>>,
) -> anyhow::Result<usize> {
    let total = jobs.len();
    tracing::info!(
        target: "zend::layers",
        layer = %spec.layer,
        units = total,
        workers = LAYER_PARALLELISM,
        thinking = %spec.effort,
        "reading the layer's units",
    );
    crate::ingest_backlog::add_pending(total as u64);
    let backlog_done = AtomicUsize::new(0);
    let cursor = AtomicUsize::new(0);
    let failures = Failures::new();
    std::thread::scope(|s| {
        for _ in 0..LAYER_PARALLELISM.min(total).max(1) {
            s.spawn(|| loop {
                if failures.aborted()
                    || crate::ingest_cancel::cancelled()
                    || crate::ingest_worker::preempted()
                {
                    return;
                }
                let idx = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(idx) else {
                    return;
                };
                if let Err(e) = read_one(ctx, spec, job, base, files) {
                    if !crate::ingest_cancel::cancelled() {
                        let n = failures.record(&job.subject, format!("{e:#}"));
                        tracing::warn!(
                            target: "zend::layers",
                            layer = %spec.layer,
                            subject = %job.subject,
                            "unit failed (read again next pass): {e:#}",
                        );
                        if n > MAX_FAILURES {
                            failures.set_abort();
                        }
                    }
                }
                backlog_done.fetch_add(1, Ordering::Relaxed);
                crate::ingest_backlog::item_done(&job.subject);
            });
        }
    });
    crate::ingest_backlog::drop_pending(
        (total as u64).saturating_sub(backlog_done.load(Ordering::Relaxed) as u64),
    );
    let report = failures.into_report(total);
    let n_failed = report.n_failed;
    if report.is_incomplete() {
        tracing::error!(
            target: "zend::layers",
            layer = %spec.layer,
            units = total,
            n_failed,
            aborted = report.aborted,
            first_failure = report.failures.first().map(|f| f.error.as_str()).unwrap_or("-"),
            "the layer's units INCOMPLETE — each failed one is read again next pass",
        );
    } else {
        let read = backlog_done.load(Ordering::Relaxed);
        if read < total {
            tracing::info!(
                target: "zend::layers",
                layer = %spec.layer,
                read,
                units = total,
                "stopped for a pass: the rest of the layer's units are read next pass",
            );
        } else {
            tracing::info!(target: "zend::layers", layer = %spec.layer, units = total, "the layer's units read");
        }
    }
    crate::ingest_report::publish(&spec.layer, report);
    Ok(n_failed)
}

/// Read one unit: lease it, mint its conversation, ask it, commit it.
fn read_one(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    job: &UnitJob,
    base: &Mutex<Sequence>,
    files: Option<&EdgeFiles<'_>>,
) -> anyhow::Result<()> {
    let hold = match ingest_lease::share(&job.key) {
        Share::Theirs => return Ok(()),
        Share::Mine(hold) => hold,
    };
    if hold.recheck() && is_committed(ctx.engine, &job.key) {
        return Ok(());
    }
    let mut conv = base
        .lock()
        .unwrap()
        .fork()
        .map_err(|e| anyhow::anyhow!("{} unit conversation: {e}", spec.layer))?;
    let timeline = conv.timeline_id();
    {
        let e = ctx.engine.lock().unwrap();
        if let Some(parent) = job.parent.or(ctx.chain_end) {
            e.set_forked_from(timeline, parent)
                .map_err(|err| anyhow::anyhow!("{} unit lineage: {err}", spec.layer))?;
        }
        e.set_timeline_summarize(timeline, false);
        for (group, allowed) in &job.scopes {
            e.set_retrieval_scope(timeline, *group, Arc::clone(allowed));
        }
    }
    if let Err(err) = conv.seed_recurrent_from_lineage() {
        tracing::warn!(
            target: "zend::layers",
            layer = %spec.layer,
            subject = %job.subject,
            "seeding recurrent memory from the unit's lineage failed: {err:#}",
        );
    }
    // Named before the decode, so a crash's partial is found and retired at
    // the next boot; the content key is withheld until it has answered.
    let mut early = BTreeMap::new();
    early.insert("kind".to_string(), spec.layer.clone());
    early.insert(LAYER_KEY.to_string(), spec.layer.clone());
    early.insert(SUBJECT_KEY.to_string(), job.subject.clone());
    let answered = conv
        .set_metadata_many(&early)
        .map_err(|e| anyhow::anyhow!("naming the unit: {e}"))
        .and_then(|()| ask(ctx, spec, job, &mut conv));
    let answer = match answered {
        Ok(answer) => answer,
        Err(e) => {
            let _ = ctx.engine.lock().unwrap().tombstone_timeline(timeline);
            return Err(e);
        }
    };
    // Every dependency is taken as ONE branch of the unit's holds it — its
    // first. Taken over all its branches, each dependency came from whichever
    // branch held it first, and where the branches disagree a unit could hold
    // one dependency as `main` has it and another as `topic` has it: current on
    // no branch at all, stale the moment it committed and read again every
    // pass. As one branch sees them, the unit stands for that branch at least,
    // and the branches that see them differently are split off to units of
    // their own (`plan::plan`) — so the splitting ends.
    let seen_from: &[String] = job
        .branches
        .first()
        .map(std::slice::from_ref)
        .unwrap_or(&[]);
    let deps: Vec<(String, String)> = match files {
        Some(f) if spec.edges => edges(&answer, &job.subject, f.known)
            .into_iter()
            .filter_map(|path| dep_version(f.live, path, seen_from).map(|k| (path.to_string(), k)))
            .collect(),
        _ => Vec::new(),
    };
    // A unit that names its dependencies is keyed by them too: read again
    // because one moved on, it is a new unit, and every unit built on it is
    // stale with it — the change is followed up the layers.
    let key = if spec.edges {
        super::keys::unit_key(
            &spec.layer,
            &format!("{}\n{}", job.after, deps_value(&deps)),
        )
    } else {
        job.key.clone()
    };
    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), spec.layer.clone());
    tags.insert(LAYER_KEY.to_string(), spec.layer.clone());
    tags.insert(SUBJECT_KEY.to_string(), job.subject.clone());
    tags.insert(AFTER_KEY.to_string(), job.after.clone());
    tags.insert(CONTENT_KEY.to_string(), key);
    if spec.edges {
        tags.insert(DEPS_KEY.to_string(), deps_value(&deps));
    }
    if let Some(file_key) = &job.file_key {
        tags.insert(FILE_KEY.to_string(), file_key.clone());
    }
    if !job.branches.is_empty() {
        tags.insert(BRANCHES_KEY.to_string(), branches_value(&job.branches));
    }
    if let Some(at) = &job.at {
        tags.insert(COMMIT_KEY.to_string(), at.clone());
    }
    if let Some(bits) = crate::branch_ingest::keys::model_bits() {
        tags.insert(QUANT_KEY.to_string(), bits.to_string());
    }
    if let Err(err) = conv.set_metadata_many(&tags) {
        let _ = ctx.engine.lock().unwrap().tombstone_timeline(timeline);
        anyhow::bail!("the content key was not written: {err:#}");
    }
    ctx.retrieval.mark_stale();
    let e = ctx.engine.lock().unwrap();
    finished(&e, timeline, Unit::Layer(&job.subject));
    e.evict_ingest_timeline(timeline);
    tracing::debug!(
        target: "zend::layers",
        layer = %spec.layer,
        subject = %job.subject,
        deps = deps.len(),
        answer_chars = answer.len(),
        "unit committed",
    );
    Ok(())
}

/// Whether a unit is committed under `key`.
fn is_committed(engine: &Mutex<ConversationEngine>, key: &str) -> bool {
    !engine
        .lock()
        .unwrap()
        .find_conversations_by_metadata(CONTENT_KEY, key)
        .is_empty()
}

/// The selection a unit's turn runs under: no tools — every unit answers from
/// what its lineage and the layers below it hold — and the layer's thinking.
fn unit_selection(spec: &LayerSpec) -> SelectionState {
    let mut selection = SelectionState::new();
    selection.select("thinking_effort", &spec.effort);
    if spec.think_mode() == ThinkMode::Off {
        selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    }
    selection
}

/// Ask the unit its question, and answer with what it said once its thinking
/// is set aside.
fn ask(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    job: &UnitJob,
    conv: &mut Sequence,
) -> anyhow::Result<String> {
    if crate::ingest_cancel::cancelled() {
        anyhow::bail!("shutdown cancelled the unit");
    }
    // A unit has no tools, so it may not open a call: one that finds its file
    // missing from what it read — a reading below that never read it — says
    // so in prose rather than committing a call nothing will run, as
    // `code_read`'s closing round forces its answer.
    let mut sampling = conv.default_sampling();
    if sampling.tool_call_open_token_id >= 0 {
        sampling
            .banned_tokens
            .push(sampling.tool_call_open_token_id);
    }
    let options = TurnOptions {
        sampling: Some(sampling),
        selection: unit_selection(spec),
        triggers: ctx.triggers_for(spec.think_mode()),
        tags: vec![spec.layer.clone(), job.subject.clone()],
        ..Default::default()
    };
    let handle = conv
        .submit_turn_with_options(TurnText::from(job.prompt.clone()), options)
        .map_err(|e| anyhow::anyhow!("submit_turn: {e}"))?;
    let resp = handle
        .wait_cancellable(crate::ingest_cancel::token())
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    conv.finish_turn(handle, &resp)
        .map_err(|e| anyhow::anyhow!("finish_turn: {e}"))?;
    // What the unit's projection drew in — kept on its turn as a dialogue
    // turn's is, and named in the log: the layers compound only if their
    // provenance brings the right units of OTHER files in, which the unit's
    // own lineage never shows.
    {
        let event = conv.recompute_projection_event(&resp.stats);
        let candidates = conv.last_turn_candidate_scores();
        log_provenance(
            ctx.engine,
            spec,
            job,
            conv.timeline_id(),
            &event,
            &candidates,
        );
        if let Err(e) = conv.persist_projection_events(std::slice::from_ref(&event)) {
            tracing::debug!(target: "zend::layers", "the unit's projection was not kept: {e:#}");
        }
    }
    let answer = after_thinking(&resp.text).trim().to_string();
    if answer.is_empty() {
        anyhow::bail!("the unit answered nothing");
    }
    Ok(answer)
}

/// Name, in the log, every turn `event` selected from outside the unit's own
/// lineage — the units of other files (or days) its provenance brought in —
/// by layer and subject, with the tokens they cost.
fn log_provenance(
    engine: &Mutex<ConversationEngine>,
    spec: &LayerSpec,
    job: &UnitJob,
    timeline: TimelineId,
    event: &candle_conversation::ProjectionEvent,
    candidates: &[(TimelineId, f32)],
) {
    let e = engine.lock().unwrap();
    let lineage: HashSet<u64> = e
        .conversation()
        .read()
        .inherited_chain(timeline)
        .into_iter()
        .map(|tl| tl.raw())
        .collect();
    let mut drawn: BTreeMap<String, u32> = BTreeMap::new();
    let mut tokens = 0u32;
    for t in &event.selection.turns {
        let Some(raw) = t.timeline else {
            continue;
        };
        if lineage.contains(&raw) || raw == timeline.raw() {
            continue;
        }
        let subject = TimelineId::from_raw(raw)
            .and_then(|tl| e.conversation_metadata(tl))
            .and_then(|m| {
                m.get(SUBJECT_KEY)
                    .or_else(|| m.get(crate::code_read::PATH_KEY))
                    .cloned()
            })
            .unwrap_or_else(|| format!("timeline {raw}"));
        *drawn.entry(format!("{}:{subject}", t.layer)).or_default() += 1;
        tokens += t.tokens;
    }
    let names: Vec<&str> = drawn.keys().map(String::as_str).collect();
    // The best-scored conversations the scan offered from outside the lineage,
    // selected or not, and how many each scoped group was allowed to offer.
    let name_of = |tl: TimelineId| {
        let m = e.conversation_metadata(tl).unwrap_or_default();
        let layer = m
            .get(LAYER_KEY)
            .or_else(|| m.get("kind"))
            .cloned()
            .unwrap_or_default();
        let subject = m
            .get(SUBJECT_KEY)
            .or_else(|| m.get(crate::code_read::PATH_KEY))
            .cloned()
            .unwrap_or_else(|| format!("timeline {}", tl.raw()));
        format!("{layer}:{subject}")
    };
    let scored: Vec<String> = candidates
        .iter()
        .filter(|(tl, _)| !lineage.contains(&tl.raw()) && *tl != timeline)
        .take(6)
        .map(|(tl, s)| format!("{}={s:.2}", name_of(*tl)))
        .collect();
    // The best score each layer below reached, selected or not: a layer that
    // never shows in `scored` is one whose units all scored under the others.
    let mut best: BTreeMap<String, f32> = BTreeMap::new();
    for (tl, s) in candidates {
        if lineage.contains(&tl.raw()) || *tl == timeline {
            continue;
        }
        let layer = e
            .conversation_metadata(*tl)
            .and_then(|m| m.get(LAYER_KEY).cloned())
            .unwrap_or_default();
        let b = best.entry(layer).or_insert(f32::MIN);
        *b = b.max(*s);
    }
    let best: Vec<String> = best.iter().map(|(l, s)| format!("{l}={s:.2}")).collect();
    let offered: Vec<String> = job
        .scopes
        .iter()
        .map(|(g, allowed)| format!("{}:{}", g.raw(), allowed.len()))
        .collect();
    tracing::info!(
        target: "zend::layers::provenance",
        layer = %spec.layer,
        subject = %job.subject,
        drawn = drawn.len(),
        tokens,
        from = %names.join(", "),
        scored = %scored.join(", "),
        best = %best.join(" "),
        offered = %offered.join(" "),
        buckets = %event
            .buckets
            .iter()
            .map(|b| format!("{}:{}", b.label, b.tokens))
            .collect::<Vec<_>>()
            .join(" "),
        "what the unit's projection brought in beside its own lineage",
    );
}

/// `text` past its thinking block, when it has one.
fn after_thinking(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_is_what_follows_the_thinking() {
        assert_eq!(
            after_thinking("<think>hmm</think>\nIt reads."),
            "\nIt reads."
        );
        assert_eq!(after_thinking("It reads."), "It reads.");
    }

    fn spec(effort: &str) -> LayerSpec {
        LayerSpec {
            layer: "critical_analysis".into(),
            unit: super::super::spec::UnitKind::File,
            after: Some("architectural_analysis".into()),
            prompt: String::new(),
            edges: false,
            effort: effort.into(),
        }
    }

    /// A unit runs at its layer's thinking level, with no tools; only an `off`
    /// layer carries the no-think switch.
    #[test]
    fn a_unit_thinks_at_its_layers_level_with_no_tools() {
        let s = unit_selection(&spec("deep"));
        assert_eq!(s.get("thinking_effort"), Some("deep"));
        assert_ne!(s.optional(NO_THINK_SELECTOR), Some(OptionalState::Present));
        assert_eq!(
            s.optional(candle_conversation::projection::TOOLS_ENABLED_SELECTOR),
            None
        );
        let off = unit_selection(&spec("off"));
        assert_eq!(
            off.optional(NO_THINK_SELECTOR),
            Some(OptionalState::Present)
        );
    }
}
