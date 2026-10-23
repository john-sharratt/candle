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
    GroupId, OptionalState, SelectionState, TimelineId, FORCE_TOOL_SELECTOR, FORCE_TOOL_SEPARATOR,
    NO_THINK_SELECTOR, TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::ThinkMode;
use candle_conversation::{ConversationEngine, FinishReason, Sequence, TurnOptions, TurnText};

use super::edges::edges;
use super::keys::{
    deps_value, parse_deps, spec_digest, AFTER_KEY, BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY,
    DEPS_KEY, FILE_KEY, JOB_KEY, LAYER_KEY, QUANT_KEY, SPEC_KEY, SUBJECT_KEY,
};
use super::plan::{dep_version, Done, LiveFiles};
use super::spec::LayerSpec;
use crate::branch_ingest::keys::branches_value;
use crate::ingest_lease::{self, Share};
use crate::ingest_report::{Failures, IngestReport};
use crate::ingest_snapshots::{finished, Unit};
use crate::refresh_ctx::RefreshContext;
use crate::tool_round::{self, Step};
use crate::tools::{format_tool_responses, ToolResult};
use zend_tools::ToolContext;
use zend_vfs::Oid;

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
            spec: meta.get(SPEC_KEY).cloned(),
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

/// Read `jobs` for `spec`'s layer, forking each off `base`. Answers the
/// round's report, which the caller publishes merged with its other rounds;
/// each failure is retried by a later pass.
pub fn ingest_units(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    jobs: &[UnitJob],
    base: &Mutex<Sequence>,
    files: Option<&EdgeFiles<'_>>,
) -> anyhow::Result<IngestReport> {
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
    Ok(report)
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
    // Only a file that branch holds is an edge: a name found on some other
    // branch alone would key the unit by a version its own branch never had,
    // and every edit there would rebuild it for nothing.
    let deps: Vec<(String, String)> = match files {
        Some(f) if spec.edges => {
            let on_branch = |path: &str| {
                seen_from.is_empty()
                    || f.live.get(path).is_some_and(|versions| {
                        versions
                            .iter()
                            .any(|(_, on)| on.iter().any(|b| seen_from.contains(b)))
                    })
            };
            edges(&answer, &job.subject, f.known)
                .into_iter()
                .filter(|path| on_branch(path))
                .filter_map(|path| {
                    dep_version(f.live, path, seen_from).map(|k| (path.to_string(), k))
                })
                .collect()
        }
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
    tags.insert(SPEC_KEY.to_string(), spec_digest(spec));
    tags.insert(JOB_KEY.to_string(), job.key.clone());
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

/// Whether a unit leased under `key` is committed — under that key, or, for
/// an edges layer, under the key of what it depended on, which records the
/// lease's as [`JOB_KEY`].
fn is_committed(engine: &Mutex<ConversationEngine>, key: &str) -> bool {
    let e = engine.lock().unwrap();
    [CONTENT_KEY, JOB_KEY].iter().any(|field| {
        e.find_conversations_by_metadata(field, key)
            .iter()
            .any(|tl| {
                e.conversation_metadata(*tl)
                    .is_some_and(|m| m.contains_key(CONTENT_KEY))
            })
    })
}

/// Tool rounds a unit may run before it is made to answer on what it has read
/// — a backstop for a model that keeps reading, as `code_read`'s
/// `MAX_FILE_READ_ROUNDS` is; an ordinary unit answers well before it.
pub const MAX_TOOL_ROUNDS: usize = 16;

/// The selection a unit's turns run under: the layer's thinking, and its
/// read-only tools when it has any — the catalog pinned to exactly those, so
/// nothing else is ever offered. A unit with none has the tool block left out.
fn unit_selection(spec: &LayerSpec) -> SelectionState {
    let mut selection = SelectionState::new();
    selection.select("thinking_effort", &spec.effort);
    if spec.think_mode() == ThinkMode::Off {
        selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    }
    if !spec.tools.is_empty() {
        selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Present);
        selection.select(
            FORCE_TOOL_SELECTOR,
            spec.tools.join(&FORCE_TOOL_SEPARATOR.to_string()),
        );
    } else {
        // Left to the schema's default the tool block is projected, and a
        // unit shown a catalog reaches for it.
        selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Absent);
    }
    selection
}

/// The tools a unit's calls run against: file stores reading the commit its
/// file was found on, so what it reads is the version it analyses.
fn unit_tools(ctx: &RefreshContext<'_>, job: &UnitJob) -> Arc<ToolContext> {
    let at = job.at.as_deref().and_then(|at| Oid::parse(at).ok());
    let repo = crate::repo_path::split(&job.subject).0;
    match at.and_then(|at| ctx.tool_ctx.files.fresh_at(repo, &at)) {
        Some(files) => Arc::new(ctx.tool_ctx.with_files(Arc::new(files), "")),
        None => Arc::clone(&ctx.tool_ctx),
    }
}

/// `steps` with every call of a tool the unit was not given refused, not run.
fn only_given(steps: Vec<Step>, tools: &[String]) -> Vec<Step> {
    steps
        .into_iter()
        .map(|step| match step {
            Step::Run(call) if !tools.contains(&call.name) => Step::Refuse(ToolResult {
                response: serde_json::json!({
                    "error": "not_available",
                    "detail": format!(
                        "`{}` is not available here; this conversation may call only {}",
                        call.name,
                        tools.join(", ")
                    ),
                }),
                call,
            }),
            other => other,
        })
        .collect()
}

/// Ask the unit its question — reading with its tools for as many rounds as it
/// needs — and answer with what it said once it stopped calling them, its
/// thinking set aside.
fn ask(
    ctx: &RefreshContext<'_>,
    spec: &LayerSpec,
    job: &UnitJob,
    conv: &mut Sequence,
) -> anyhow::Result<String> {
    let selection = unit_selection(spec);
    let tools = (!spec.tools.is_empty()).then(|| unit_tools(ctx, job));
    let rounds = if tools.is_some() { MAX_TOOL_ROUNDS } else { 0 };
    let mut message = TurnText::from(job.prompt.clone());
    for round in 0..=rounds {
        if crate::ingest_cancel::cancelled() {
            anyhow::bail!("shutdown cancelled the unit");
        }
        // The last round — the only one, for a unit with no tools — may not
        // open a call: it answers in prose on what it has, rather than
        // committing a call nothing will run.
        let closing = round == rounds;
        let mut sampling = conv.default_sampling();
        if closing && sampling.tool_call_open_token_id >= 0 {
            sampling
                .banned_tokens
                .push(sampling.tool_call_open_token_id);
        }
        let options = TurnOptions {
            sampling: Some(sampling),
            selection: selection.clone(),
            triggers: ctx.triggers_for(spec.think_mode()),
            tags: vec![spec.layer.clone(), job.subject.clone()],
            ..Default::default()
        };
        let handle = conv
            .submit_turn_with_options(message, options)
            .map_err(|e| anyhow::anyhow!("submit_turn: {e}"))?;
        let resp = handle
            .wait_cancellable(crate::ingest_cancel::token())
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let steps = if closing {
            Vec::new()
        } else {
            tool_round::plan(&resp.answer)
        };
        let call_idx = resp.seal.as_ref().and_then(|s| s.turn_index);
        conv.finish_turn(handle, &resp)
            .map_err(|e| anyhow::anyhow!("finish_turn: {e}"))?;
        // What the question drew in — kept on its turn as a dialogue turn's
        // is, and named in the log: the layers compound only if their
        // provenance brings the right units of OTHER files in, which the
        // unit's own lineage never shows.
        if round == 0 {
            let (event, candidates) = conv.unit_projection(&resp.stats);
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
        if closing && !tool_round::plan(&resp.answer).is_empty() {
            // The opener could not be banned (the model has no single token
            // for it): a call nothing will run is no answer.
            anyhow::bail!("the unit's closing round called a tool (round {round})");
        }
        let Some(tools) = tools.as_ref().filter(|_| !steps.is_empty()) else {
            return answer_of(&resp.answer, round, &resp.stats.finish);
        };
        let results = tool_round::run(tools, only_given(steps, &spec.tools));
        let response = format_tool_responses(&results);
        if response.is_blank() {
            anyhow::bail!("the unit's tool calls answered nothing (round {round})");
        }
        // The round-trip is certain now: couple the call turn to the
        // response that answers it, by its own sealed index.
        if let Some(idx) = call_idx {
            conv.couple_turn(idx)
                .map_err(|e| anyhow::anyhow!("couple_turn: {e}"))?;
        }
        message = response;
    }
    unreachable!("the closing round always answers")
}

/// The unit's answer: what it said, its thinking set aside. A turn that spent
/// its budget, or stopped inside its thinking, has not answered — what it
/// wrote is reasoning, or an answer cut short, and it is read again.
fn answer_of(written: &TurnText, round: usize, finish: &FinishReason) -> anyhow::Result<String> {
    let chars: usize = written.pieces().iter().map(|p| p.text.len()).sum();
    let (unclosed, answer) = split_thinking(written);
    if *finish == FinishReason::Length || unclosed {
        anyhow::bail!(
            "the unit stopped before it answered (round {round}, finish {finish:?}, {chars} chars written, thinking {})",
            if unclosed { "never closed" } else { "closed" },
        );
    }
    let answer = answer.trim().to_string();
    if answer.is_empty() {
        anyhow::bail!(
            "the unit answered nothing (round {round}, finish {finish:?}, {chars} chars written)"
        );
    }
    Ok(answer)
}

/// Whether `written` stopped inside its thinking, and what it said past it.
///
/// **Only the model's own tags bound the thinking.** `<think>` and `</think>`
/// are control tokens; the same characters in ordinary text are a quotation —
/// an analysis of a file that handles reasoning blocks quotes them — and are
/// part of the answer. Read as plain text, a quoted `<think>` made a finished
/// answer look unclosed and failed the unit, and a quoted `</think>` cut every
/// word before it out of the answer.
fn split_thinking(written: &TurnText) -> (bool, String) {
    let mut open = false;
    let mut after = String::new();
    for piece in written.pieces() {
        if piece.literal {
            if !open {
                after.push_str(&piece.text);
            }
            continue;
        }
        // The answer is what follows the model's last `</think>`: an open tag
        // starts a block, a close ends it and starts the answer afresh.
        let mut rest = piece.text.as_str();
        loop {
            match (rest.find(THINK_OPEN), rest.find(THINK_CLOSE)) {
                (Some(o), c) if c.is_none_or(|c| o < c) => {
                    open = true;
                    rest = &rest[o + THINK_OPEN.len()..];
                }
                (_, Some(c)) => {
                    open = false;
                    after.clear();
                    rest = &rest[c + THINK_CLOSE.len()..];
                }
                (None, None) => {
                    if !open {
                        after.push_str(rest);
                    }
                    break;
                }
            }
        }
    }
    (open, after)
}

/// The reasoning block's tags, as the model writes them.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

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

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn as the model writes it: its thinking tags are markup, and
    /// everything it says in words is literal.
    fn turn(thought: &str, said: &str) -> TurnText {
        TurnText::markup("<think>")
            .then_literal(thought)
            .then_markup("</think>")
            .then_literal(said)
    }

    #[test]
    fn the_answer_is_what_follows_the_thinking() {
        assert_eq!(
            split_thinking(&turn("hmm", "\nIt reads.")),
            (false, "\nIt reads.".to_string())
        );
        assert_eq!(
            split_thinking(&TurnText::literal("It reads.")),
            (false, "It reads.".to_string())
        );
    }

    /// Reasoning that never closed, or an answer cut off by the budget, is
    /// no answer: the unit is read again rather than committed.
    #[test]
    fn an_unclosed_thought_or_a_spent_budget_is_no_answer() {
        let stop = FinishReason::Stop;
        assert_eq!(
            answer_of(&turn("a", "It reads."), 0, &stop).unwrap(),
            "It reads."
        );
        let still = TurnText::markup("<think>").then_literal("still thinking");
        assert!(answer_of(&still, 0, &stop).is_err());
        let again = turn("a", "x").then_markup("<think>").then_literal("again");
        assert!(answer_of(&again, 0, &stop).is_err());
        assert!(answer_of(&turn("a", "It rea"), 0, &FinishReason::Length).is_err());
        assert!(answer_of(&turn("a", "  "), 0, &stop).is_err());
        assert_eq!(
            answer_of(&TurnText::literal("No thinking."), 0, &stop).unwrap(),
            "No thinking."
        );
    }

    /// The tags an answer quotes are words, not thinking: a file that handles
    /// reasoning blocks is described with `<think>` and `</think>` in it, and
    /// that answer is whole and closed — not failed as unclosed, nor cut at
    /// the quoted close.
    #[test]
    fn quoted_think_tags_are_part_of_the_answer() {
        let said = "The parser opens on `<think>` and closes on `</think>`; \
                    a reply ending in `<think>` stays open.";
        assert_eq!(
            split_thinking(&turn("read the file", said)),
            (false, said.to_string())
        );
        let quoted_in_thought = turn("it handles `</think>` tags", "It parses tags.");
        assert_eq!(
            split_thinking(&quoted_in_thought),
            (false, "It parses tags.".to_string())
        );
    }

    fn spec(effort: &str) -> LayerSpec {
        LayerSpec {
            layer: "critical_analysis".into(),
            unit: super::super::spec::UnitKind::File,
            after: Some("architectural_analysis".into()),
            prompt: String::new(),
            edges: false,
            effort: effort.into(),
            tools: Vec::new(),
        }
    }

    /// A unit runs at its layer's thinking level; one given no tools has the
    /// tool block left out, and only an `off` layer carries the no-think switch.
    #[test]
    fn a_unit_thinks_at_its_layers_level_with_no_tools() {
        let s = unit_selection(&spec("deep"));
        assert_eq!(s.get("thinking_effort"), Some("deep"));
        assert_ne!(s.optional(NO_THINK_SELECTOR), Some(OptionalState::Present));
        assert_eq!(
            s.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Absent)
        );
        let off = unit_selection(&spec("off"));
        assert_eq!(
            off.optional(NO_THINK_SELECTOR),
            Some(OptionalState::Present)
        );
    }

    /// A unit given tools sees exactly those, pinned in the catalog.
    #[test]
    fn a_unit_given_tools_sees_exactly_them() {
        let mut s = spec("deep");
        s.tools = vec!["file_read".into(), "file_grep".into()];
        let sel = unit_selection(&s);
        assert_eq!(
            sel.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Present)
        );
        assert_eq!(sel.get(FORCE_TOOL_SELECTOR), Some("file_read,file_grep"));
    }

    /// A call of a tool the unit was not given is refused, never run; a call
    /// of one it was given runs.
    #[test]
    fn a_call_outside_the_units_tools_is_refused() {
        let call = |name: &str| crate::tools::ToolCall {
            name: name.into(),
            arguments: serde_json::json!({}),
        };
        let steps = vec![Step::Run(call("file_read")), Step::Run(call("file_edit"))];
        let out = only_given(steps, &["file_read".to_string()]);
        assert!(matches!(&out[0], Step::Run(c) if c.name == "file_read"));
        match &out[1] {
            Step::Refuse(r) => assert_eq!(r.response["error"], "not_available"),
            other => panic!("file_edit was not refused: {other:?}"),
        }
    }
}
