//! Calibrating the knowledge base's files (`docs/provenance_system_collection.md`
//! §7 step 3, §7.1, §12 D11–D13): find each sealed file without a
//! calibration of its current version, ask for its questions, and seal them
//! as its calibrations.
//!
//! A file is asked in a **draft** — a fork of the dialogue's base, projected
//! as a dialogue's first turn is, the seed absent and the knowledge base
//! pinned to the file alone, whole
//! ([`FilesPin::Only`]) — so the model writes its questions with the file in
//! front of it, in the notation a dialogue reads files in. The draft is
//! tombstoned once its answer is read: only the questions are kept.
//!
//! Each question is then asked again, **alternating between two knowledge
//! bases** in draft order ([`KbFrame`]): the first, third, fifth… behind the
//! empty one (`files_open` with nothing under it), the second, fourth… behind
//! the target (the file alone). Each is a fork of the same base, the seed
//! absent, thinking on; what the model reasons, up to the layer's `think`
//! budget, is kept ([`super::thinking`]). A file's questions are asked at
//! once, so they decode in one wave.
//!
//! Each frame's questions are then prefilled behind the frame they were asked
//! in, in ONE turn group of a fork pinned the same way, each masked to itself
//! and sealed one turn apiece, its answer its thinking: the frame a live query
//! has before it has found the file, and once the file is in front of it. The
//! calibration carries [`Substrate::CALIBRATES_KEY`] — the file's scan id
//! (`FileKey::scan_id`) — so its questions vote for the file wherever it is
//! scored, and [`CALIBRATION_FRAME_KEY`]; it is then archived — the commit —
//! and distilled provenance-only: its K/V is dropped and its signatures stay.
//! Neither frame holds the seed or another file, so nothing bleeds in and a
//! seed change recalibrates nothing.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_conversation::files::names::{chunk_name, heading_name};
use candle_conversation::files::{files_owner, FileKey, FileRecord, FileUnit, FilesPin};
use candle_conversation::projection::{
    Conversation, DecodePriority, OptionalState, SelectionState, TimelineId, NO_THINK_SELECTOR,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::{TriggerRegistry, TurnGrammar};
use candle_conversation::substrate::Substrate;
use candle_conversation::{
    ConversationEngine, FinishOrder, Sequence, SequenceConfig, TurnOptions, MAX_DECODE_WIDTH,
    MAX_PREFILL_WIDTH,
};

use super::ask::Ask;
use super::draft_sight::members_held;
use super::failures;
use super::kb_frame::{KbFrame, CALIBRATION_FRAME_KEY};
use super::owed;
use super::permits::{Permit, Permits};
use super::plan::{plan, Prior, Unit};
use super::questions::questions;
use super::retire_replies::retire_reply_records;
use super::seal::seal_framed;
use super::sections::{compose, grammar, split};
use super::spec::CalibrateSpec;
use super::thinking::{exemplar, thinking_of};
use crate::files::records;
use crate::ingest_backlog::{self, Work};
use crate::ingest_holdoff;
use crate::ingest_lease::{self, Share};
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
/// asked for without reasoning — up to [`super::spec::MAX_QUESTIONS`] of them,
/// a developer's question naming what it asks about running to ~40 tokens.
const DRAFT_MAX_TOKENS: usize = 4096;

/// Passes whose draft may write no question before the file is left
/// uncalibrated. Four units answered every draft the same way and were asked
/// again every pass on both members (2026-10-31). A new version is a new
/// marker, and is tried afresh.
const MAX_FAILURES: u32 = 3;

/// How a calibration that failed for want of a question says so.
const NO_QUESTION: &str = "the draft wrote no question";

/// Why a draft that holds none of its file is not asked: its questions would
/// be the file's name, guessed at (`draft_sight`).
const BLIND_DRAFT: &str = "the draft holds none of its file";

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

/// Whether `error` is the file's own failure — its draft wrote nothing, or
/// stopped short — which counts against it, rather than a decode lost to a
/// full card, which says nothing about the file.
fn units_own_failure(error: &str) -> bool {
    error.starts_with(NO_QUESTION) || error.starts_with(SHORT_DRAFT)
}

/// How many times `marker` has failed, as its failure record holds it in
/// `recorded` ([`failures`]).
fn failures_of(recorded: Option<&str>, marker: &str) -> u32 {
    recorded
        .and_then(|r| r.rsplit_once(' '))
        .filter(|(m, _)| *m == marker)
        .and_then(|(_, n)| n.parse().ok())
        .unwrap_or(0)
}

/// Files calibrated at once. A draft holds its whole file, so drafts are held
/// to the width the layers read at (`LAYER_PARALLELISM`); a file's thinking
/// and seal carry at most its file and a question, so three times as many
/// files are in flight, and one thinking does not hold a draft's place. The
/// scheduler's waves pace the decode.
const CALIBRATE_PARALLELISM: usize = 3 * LAYER_PARALLELISM;

/// The places every file's questions share while they think ([`thoughts`]):
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

/// The files layer to calibrate: its spec, and the base its drafts, thinking
/// and calibrations fork from — the dialogue's, so each projects as a live
/// query's first turn does.
pub struct Layer<'a> {
    pub spec: &'a CalibrateSpec,
    pub base: &'a Sequence,
}

/// Where a calibration runs: the system prompt every conversation opens with,
/// dialect-formatted — what a use-exemplar is prefilled under
/// (`crate::guidance`) — the sequence config, and the steering a question's
/// thinking decodes under: a live dialogue turn's at the default effort, so
/// its reasoning block opens and closes as one does.
pub struct Frame<'a> {
    pub system_prompt: &'a str,
    pub config: &'a SequenceConfig,
    pub thinking: Arc<TriggerRegistry>,
}

/// Calibrate every file of the knowledge base that is sealed here and not
/// calibrated behind both frames, and retire every calibration of a file no
/// longer there.
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
        let units = units_of(&e.conversation(), spec);
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
    owed::global().planned(&spec.layer, todo.run.len(), todo.waiting);
    let files = by_file(&todo.run);
    tracing::info!(
        target: "zend::calibrate",
        layer = %spec.layer,
        calibrated = todo.done,
        to_calibrate = todo.run.len(),
        files = files.len(),
        waiting = report.waiting,
        retired = report.retired,
        "calibrating the knowledge base's files",
    );
    // **The sweep is backlog the bar shows**, each file counted until it is
    // sealed, failed or found to be another member's, as `<path> (calibrated)`.
    ingest_backlog::add_pending(Work::Calibrating, files.len() as u64);
    let finished = AtomicUsize::new(0);
    let cursor = AtomicUsize::new(0);
    let sealed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let cancelled = AtomicUsize::new(0);
    let elsewhere = AtomicUsize::new(0);
    let drafts = Permits::new(LAYER_PARALLELISM);
    let done = |subject: &str| {
        finished.fetch_add(1, Ordering::Relaxed);
        ingest_backlog::item_done(Work::Calibrating, &format!("{subject} (calibrated)"));
    };
    let conversation = engine.conversation();
    std::thread::scope(|s| {
        for _ in 0..CALIBRATE_PARALLELISM.min(files.len()).max(1) {
            // Only a shutdown stops it.
            s.spawn(|| loop {
                if crate::ingest_cancel::cancelled() {
                    return;
                }
                let Some(units) = files.get(cursor.fetch_add(1, Ordering::Relaxed)) else {
                    return;
                };
                let subject = units[0].subject.as_str();
                // **A file is leased only once it can start.** Its draft's
                // permit first, then its lease: a member holds no more files
                // than it is working on, and the files it has not reached stay
                // unleased for whichever member reaches them first.
                let t_permit = Instant::now();
                let permit = drafts.acquire();
                let permit_wait_ms = t_permit.elapsed().as_millis() as u64;
                if crate::ingest_cancel::cancelled() {
                    return;
                }
                // **One member calibrates a file.** Every member's pass plans
                // the same files from the same replicated substrate; the one
                // the vote names asks for them, and a calibration handed back
                // is looked for again before it is asked for twice.
                let _lease = match ingest_lease::share_calibration(&units[0].content_key) {
                    Share::Theirs => {
                        elsewhere.fetch_add(units.len(), Ordering::Relaxed);
                        for _ in units.iter() {
                            owed::global().committed(&spec.layer);
                        }
                        done(subject);
                        continue;
                    }
                    Share::Mine(lease) => lease,
                };
                let owed_here: Vec<Unit> = units
                    .iter()
                    .filter(|u| !calibrated(engine, &u.marker))
                    .cloned()
                    .collect();
                if owed_here.is_empty() {
                    elsewhere.fetch_add(units.len(), Ordering::Relaxed);
                    for _ in units.iter() {
                        owed::global().committed(&spec.layer);
                    }
                    done(subject);
                    continue;
                }
                let outcome = calibrate_file(
                    engine,
                    spec,
                    layer.base,
                    frame,
                    Some((permit, permit_wait_ms)),
                    &owed_here,
                );
                done(subject);
                match outcome {
                    Ok(()) => {
                        sealed.fetch_add(owed_here.len(), Ordering::Relaxed);
                        for _ in &owed_here {
                            owed::global().committed(&spec.layer);
                        }
                    }
                    // A draft a shutdown stopped is no failure of the file's.
                    Err(_) if crate::ingest_cancel::cancelled() => {
                        cancelled.fetch_add(owed_here.len(), Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(owed_here.len(), Ordering::Relaxed);
                        // Only the file's own failure counts against it: a
                        // decode lost to a full card says nothing about the
                        // file, and is simply tried again. The other way
                        // round for the member: a file's own failure says
                        // nothing about the card, so only the card's holds
                        // this member off the next file.
                        let own = units_own_failure(&format!("{e:#}"));
                        if own {
                            for unit in &owed_here {
                                let tries =
                                    failures_of(unit.prior_failures.as_deref(), &unit.marker)
                                        .saturating_add(1);
                                if let Err(err) =
                                    failures::record(&conversation, &unit.key, &unit.marker, tries)
                                {
                                    tracing::warn!(
                                        target: "zend::calibrate",
                                        subject,
                                        "a failed calibration could not be counted: {err:#}",
                                    );
                                }
                            }
                        }
                        tracing::warn!(
                            target: "zend::calibrate",
                            layer = %spec.layer,
                            subject,
                            "calibration failed: {e:#}",
                        );
                        if !own {
                            ingest_holdoff::cool_down(subject, &format!("{e:#}"));
                        }
                    }
                }
            });
        }
    });
    // A shutdown stops the sweep between files: what it never started is
    // handed back, so the bar still reaches zero.
    ingest_backlog::drop_pending(
        Work::Calibrating,
        files.len().saturating_sub(finished.into_inner()) as u64,
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

/// `units` grouped by file, in the order their files first appear: what one
/// draft calibrates.
fn by_file(units: &[Unit]) -> Vec<Vec<Unit>> {
    let mut out: Vec<Vec<Unit>> = Vec::new();
    for u in units {
        match out.iter_mut().find(|g| g[0].content_key == u.content_key) {
            Some(group) => group.push(u.clone()),
            None => out.push(vec![u.clone()]),
        }
    }
    out
}

/// Whether a calibration carrying `marker` stands — one another member sealed
/// and replication has brought here.
fn calibrated(engine: &ConversationEngine, marker: &str) -> bool {
    engine
        .conversations_with_metadata_key(MARKER_KEY)
        .into_iter()
        .any(|(_, m)| m == marker)
}

/// Every file of the knowledge base, behind each frame ([`KbFrame::ALL`]):
/// ready when its members are sealed in this process — what its draft and its
/// target frame place — and its own failures have not run out.
fn units_of(conversation: &Conversation, spec: &CalibrateSpec) -> Vec<Unit> {
    let mut units = Vec::new();
    for record in records::all(conversation) {
        if record.unit != FileUnit::File {
            continue;
        }
        let placeable = records::placeable(conversation, &record);
        for frame in KbFrame::ALL {
            let key = format!("{}:{}", frame.as_str(), record.content_key);
            let marker = spec.marker(&key, &record.path);
            let prior_failures = failures::recorded(conversation, &key);
            units.push(Unit {
                timeline: record.key().scan_id(),
                subject: record.path.clone(),
                content_key: record.content_key.clone(),
                frame,
                ready: placeable && failures_of(prior_failures.as_deref(), &marker) < MAX_FAILURES,
                key,
                marker,
                prior_failures,
            });
        }
    }
    units
}

/// What names every calibration of the file `unit` (its scan id): its kind,
/// the file it calibrates ([`Substrate::CALIBRATES_KEY`]), and its subject.
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

/// The selection a draft is asked under: no reasoning, no tool block, and
/// the seed's absent branch (§12 D11).
pub(super) fn draft_selection() -> SelectionState {
    let mut selection = SelectionState::new();
    selection.select("thinking_effort", "off");
    selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Absent);
    selection.set_seeds(false);
    selection
}

/// The selection a question's thinking, and its calibration's prefill, run
/// under: a live dialogue turn's, every dial at its default, on the seed's
/// absent branch.
pub(super) fn frame_selection() -> SelectionState {
    let mut selection = SelectionState::new();
    selection.set_seeds(false);
    selection
}

/// The file `units` — one file's units, a frame apiece — are of, and its
/// record.
fn record_of(engine: &ConversationEngine, units: &[Unit]) -> anyhow::Result<FileRecord> {
    let first = &units[0];
    records::of_path(&engine.conversation(), &first.subject)
        .into_iter()
        .find(|r| r.content_key == first.content_key)
        .ok_or_else(|| anyhow::anyhow!("{} has no record any more", first.subject))
}

/// The raw ids of `record`'s members in this process: what a draft that sees
/// its file holds.
fn member_ids(conversation: &Conversation, record: &FileRecord) -> HashSet<u32> {
    let mut names = vec![heading_name(&record.path, &record.blob)];
    names.extend(
        record
            .chunks
            .iter()
            .map(|c| chunk_name(&record.path, &record.blob, c.lines)),
    );
    names
        .iter()
        .filter_map(|n| conversation.owned_section_named(files_owner(), n))
        .map(|id| id.raw())
        .collect()
}

/// Ask the file `units` are of for its questions — holding `draft_permit`,
/// when given with how long it was waited for, while it does, and handing it
/// back once the draft is written — and calibrate each unit's frame with its
/// share of them.
fn calibrate_file(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    frame: &Frame<'_>,
    draft_permit: Option<(Permit<'_>, u64)>,
    units: &[Unit],
) -> anyhow::Result<()> {
    let permit_wait_ms = draft_permit.as_ref().map_or(0, |(_, waited)| *waited);
    let first = &units[0];
    let record = record_of(engine, units)?;
    let file = record.key();
    let tokens = record.tokens() as usize;
    let members = member_ids(&engine.conversation(), &record);
    let (drafted, draft_ms) = {
        let _draft = draft_permit;
        let t_draft = Instant::now();
        let drafted = draft(engine, spec, base, file, &first.subject, &members, tokens)?;
        (drafted, t_draft.elapsed().as_millis() as u64)
    };
    let wanted = drafted.wanted;
    let found = drafted.questions;
    if found.is_empty() {
        let last = &drafted.last_answer;
        let opening: String = last.chars().take(160).collect();
        anyhow::bail!("{NO_QUESTION} ({} chars: {opening:?})", last.len());
    }
    // **A short draft is tried again, but not forever**: on the file's last
    // try what it wrote is sealed, which is more than leaving it with none.
    let last_try = failures_of(first.prior_failures.as_deref(), &first.marker).saturating_add(1)
        >= MAX_FAILURES;
    if !drafted.empty.is_empty() && !last_try {
        anyhow::bail!("{SHORT_DRAFT} ({})", drafted.empty.join(", "));
    }
    // Each owed frame's questions, in draft order: its alternate share, or —
    // a draft too short to give the frame one — every question.
    let asked: Vec<(KbFrame, Vec<usize>)> = units
        .iter()
        .map(|u| {
            let share = u.frame.questions_of(found.len());
            let share = if share.is_empty() {
                (0..found.len()).collect()
            } else {
                share
            };
            (u.frame, share)
        })
        .collect();
    let to_think: Vec<(String, KbFrame)> = asked
        .iter()
        .flat_map(|(kb, idx)| idx.iter().map(|&i| (found[i].clone(), *kb)))
        .collect();
    let t_think = Instant::now();
    let thought = thoughts(engine, base, frame, spec, file, &to_think)?;
    let think_ms = t_think.elapsed().as_millis() as u64;
    let t_exemplars = Instant::now();
    let mut at = 0usize;
    for ((kb, idx), unit) in asked.iter().zip(units) {
        let cases: Vec<(String, String)> = idx
            .iter()
            .enumerate()
            .map(|(j, &i)| (found[i].clone(), exemplar(&thought[at + j])))
            .collect();
        at += idx.len();
        let mut meta = calibration_meta(unit.timeline, &unit.subject);
        meta.insert(MARKER_KEY.to_string(), unit.marker.clone());
        meta.insert(CALIBRATION_FRAME_KEY.to_string(), kb.as_str().to_string());
        seal_framed(engine, base, &meta, &cases, kb.pin(file), frame_selection())?;
    }
    tracing::debug!(
        target: "zend::calibrate",
        layer = %spec.layer,
        subject = %first.subject,
        tokens,
        wanted,
        questions = found.len(),
        frames = units.len(),
        permit_wait_ms,
        draft_ms,
        think_ms,
        exemplars_ms = t_exemplars.elapsed().as_millis() as u64,
        "file calibrated",
    );
    Ok(())
}

/// Each of `asked`'s thinking, in order: each question asked alone in a
/// transient fork of `base` behind its frame's knowledge base, decoded to the
/// spec's `think` budget. Each conversation is tombstoned either way. A
/// question the model did not think about fails the file, which is calibrated
/// again by a later pass.
///
/// **No more in the queue than the scheduler runs.** Every file's questions
/// share one gate ([`THINKING`]), as wide as the scheduler runs turns: a file
/// submits a question only while it holds a place, and gives the place back
/// as the question's turn ends, when it submits its next. Asked every question
/// at once, a dozen units stood ~550 turns in the queue against 24 run at a
/// time on BlackWorld (2026-11-15), and a unit none of whose turns moved for a
/// minute gave its whole calibration up ([`FinishOrder`]). A file holding no
/// place waits for one; a file holding some takes more only when one is free
/// ([`Permits::next_place`]), so it never waits on places it keeps from
/// another.
///
/// **A question gives up its conversation the moment its turn ends**, read in
/// the order the turns finish, so a finished question's recurrent store is not
/// held while the questions before it run.
fn thoughts(
    engine: &ConversationEngine,
    base: &Sequence,
    frame: &Frame<'_>,
    spec: &CalibrateSpec,
    file: FileKey,
    asked_for: &[(String, KbFrame)],
) -> anyhow::Result<Vec<String>> {
    let n = asked_for.len();
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
                let (question, kb) = &asked_for[i];
                let conv = framed_fork(engine, base, &spec.layer, kb.pin(file))?;
                let timeline = conv.timeline_id();
                asked.push(Some(conv));
                timelines.push(timeline);
                places.push(Some(place));
                let options = TurnOptions {
                    max_tokens: Some(spec.think),
                    selection: frame_selection(),
                    triggers: Arc::clone(&frame.thinking),
                    tags: vec![TAG.to_string()],
                    ..Default::default()
                };
                let handle = asked[i]
                    .as_mut()
                    .expect("the question's conversation was just opened")
                    .submit_turn_with_options(question.clone(), options)
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
            let q = &asked_for[i].0;
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

/// A transient fork of `base` whose knowledge base is `pin`, named as
/// `layer`'s draft so a crash's is retired, and decoding at the ingest's
/// background band. The pin is set before its first turn, which places it.
fn framed_fork(
    engine: &ConversationEngine,
    base: &Sequence,
    layer: &str,
    pin: FilesPin,
) -> anyhow::Result<Sequence> {
    let conv = base
        .fork_at(DecodePriority::Low)
        .map_err(|e| anyhow::anyhow!("calibration conversation: {e}"))?;
    let timeline = conv.timeline_id();
    engine.conversation().mark_timeline_transient(timeline);
    engine.set_timeline_summarize(timeline, false);
    engine.files_pin(timeline, pin);
    if let Err(err) = engine.set_conversation_metadata(timeline, DRAFT_KEY, layer) {
        let _ = engine.tombstone_timeline(timeline);
        anyhow::bail!("naming the calibration conversation: {err}");
    }
    Ok(conv)
}

/// What a file's draft wrote: its questions from every prompt, in the order
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

/// The file's answer to its layer's calibration prompts, asked together in
/// one turn of a draft whose knowledge base is the file alone, whole — each a
/// part under its own heading ([`super::sections`]), each for as many
/// questions as its rule gives a file of `tokens`; the draft is tombstoned
/// either way.
///
/// **Every part in one turn, with the file in front of it.** The parts see
/// each other, so a later one can be told to ask nothing an earlier one asked;
/// a question two parts repeat word for word is kept once. An answer that
/// carries no headings is read as one list, against the parts' counts
/// together.
fn draft(
    engine: &ConversationEngine,
    spec: &CalibrateSpec,
    base: &Sequence,
    file: FileKey,
    subject: &str,
    members: &HashSet<u32>,
    tokens: usize,
) -> anyhow::Result<Drafted> {
    let mut conv = framed_fork(engine, base, &spec.layer, FilesPin::Only(file))?;
    let timeline = conv.timeline_id();
    let out = (|| {
        // A draft has no tools, and one that opens a call anyway writes no
        // question: a file too long for its projection was asked for again
        // with `file_read`, every pass.
        let mut sampling = conv.default_sampling();
        if sampling.tool_call_open_token_id >= 0 {
            sampling
                .banned_tokens
                .push(sampling.tool_call_open_token_id);
        }
        // A part a file is asked for none of is left out of the turn.
        let (asks, wanted): (Vec<&Ask>, Vec<usize>) = spec
            .asks_for(subject)
            .iter()
            .map(|a| (a, a.questions.for_tokens(tokens)))
            .filter(|&(_, n)| n > 0)
            .unzip();
        let parts: Vec<(&str, String)> = asks
            .iter()
            .zip(&wanted)
            .map(|(a, &n)| (a.name.as_str(), a.prompt_for(subject, n)))
            .collect();
        // Every part's heading is the grammar's, so no part is skipped and
        // the answer cannot end before the last ([`grammar`]).
        let headings: Vec<(&str, usize)> = asks
            .iter()
            .zip(&wanted)
            .map(|(a, &n)| (a.name.as_str(), n))
            .collect();
        let grammar_spec =
            grammar(&headings).map_err(|e| anyhow::anyhow!("the draft's grammar: {e}"))?;
        let tree = engine
            .compile_stencil(&grammar_spec)
            .map_err(|e| anyhow::anyhow!("compiling the draft's grammar: {e}"))?;
        let options = TurnOptions {
            max_tokens: Some(DRAFT_MAX_TOKENS),
            sampling: Some(sampling),
            turn_grammar: Some(TurnGrammar::of(Arc::new(tree))),
            selection: draft_selection(),
            tags: vec![TAG.to_string()],
            ..Default::default()
        };
        // What of the file the draft is about to be asked in front of.
        let dump = conv.prompt_dump();
        let held = dump.as_ref().map(|d| members_held(d, members));
        tracing::info!(
            target: "zend::calibrate",
            subject,
            file = tokens,
            held = ?held,
            prompt = ?dump.as_ref().map(|d| d.token_count),
            "calibration draft projected",
        );
        if held.unwrap_or(0) == 0 {
            anyhow::bail!("{BLIND_DRAFT} ({tokens} tokens)");
        }
        let handle = conv
            .submit_turn_with_options(compose(&parts), options)
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
                        short_of(subject, &ask.name, n, got, body);
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
                short_of(subject, "all", total, kept.len(), &answer);
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
    })();
    let _ = engine.tombstone_timeline(timeline);
    out
}

/// **A part that writes far short of its count is said so, with what it
/// wrote.** The draft is gone once read, and a file one of whose parts kept
/// nothing reads like one with a small budget.
fn short_of(subject: &str, part: &str, wanted: usize, got: usize, wrote: &str) {
    if got * 2 >= wanted {
        return;
    }
    let opening: String = wrote.chars().take(240).collect();
    tracing::warn!(
        target: "zend::calibrate",
        subject,
        part,
        wanted,
        got,
        chars = wrote.len(),
        "a calibration part wrote under half its questions: {opening:?}",
    );
}

#[cfg(test)]
mod tests {
    use super::super::ask::RawAsk;
    use super::super::count::QuestionCount;
    use super::super::profile::{Profile, RawProfile};
    use super::*;
    use candle_conversation::files::{ChunkRecord, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;
    use candle_conversation::projection::SEED_SELECTOR;

    /// Failures count against the marker they happened under: a new version
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

    /// **Only the file's own failures count against it**: a draft that wrote
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

    fn spec() -> CalibrateSpec {
        CalibrateSpec {
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
        }
    }

    fn record(path: &str, unit: FileUnit) -> FileRecord {
        let member = MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 1, hi: 0 },
                section_hash: ContentHash { lo: 2, hi: 0 },
            },
            tokens: 10,
        };
        FileRecord {
            content_key: format!("v7:{path}@1"),
            path: path.into(),
            blob: "1".into(),
            unit,
            lines: 3,
            lang: "rs".into(),
            branches: vec!["main".into()],
            heading: member,
            chunks: vec![ChunkRecord {
                lines: LineSpan { first: 1, last: 3 },
                member,
            }],
        }
    }

    /// **A file is two units, one a frame, named by the file's scan id; only
    /// a file sealed here is ready, and a folder is no unit.** Each frame's
    /// marker is its own, and a file whose failures ran out under a frame's
    /// marker waits behind that frame alone.
    #[test]
    fn a_file_is_a_unit_per_frame_ready_once_sealed_here() {
        let conv = Conversation::new();
        let spec = spec();
        let sealed = record("candle/a.rs", FileUnit::File);
        let arrived = record("candle/b.rs", FileUnit::File);
        let folder = record("candle/src/", FileUnit::Folder);
        for r in [&sealed, &arrived, &folder] {
            records::put(&conv, r).unwrap();
        }
        let id = conv.allocate_owned_section().unwrap();
        conv.register_owned_section(files_owner(), &heading_name(&sealed.path, &sealed.blob), id);
        let worn = format!("target:{}", sealed.content_key);
        failures::record(
            &conv,
            &worn,
            &spec.marker(&worn, &sealed.path),
            MAX_FAILURES,
        )
        .unwrap();

        let units = units_of(&conv, &spec);
        let got: Vec<(&str, KbFrame, bool)> = units
            .iter()
            .map(|u| (u.subject.as_str(), u.frame, u.ready))
            .collect();
        assert_eq!(
            got,
            vec![
                ("candle/a.rs", KbFrame::Empty, true),
                ("candle/a.rs", KbFrame::Target, false),
                ("candle/b.rs", KbFrame::Empty, false),
                ("candle/b.rs", KbFrame::Target, false),
            ]
        );
        assert_eq!(units[0].timeline, sealed.key().scan_id());
        assert_eq!(units[0].key, format!("empty:{}", sealed.content_key));
        assert_ne!(units[0].marker, units[1].marker);
        assert_eq!(
            CalibrateSpec::unit_of(&units[1].marker),
            Some(units[1].key.as_str())
        );
        assert_eq!(by_file(&units).len(), 2);
    }

    /// **A calibration names its file's scan id, and the frame it was asked
    /// in.**
    #[test]
    fn a_calibration_names_its_file_key_and_frame() {
        let file = FileKey(0x473e_088d_5653_2900);
        let meta = calibration_meta(file.scan_id(), "candle/a.rs");
        assert_eq!(
            meta.get(Substrate::CALIBRATES_KEY).map(String::as_str),
            Some("5133550028380645632")
        );
        assert_eq!(
            meta.get(SUBJECT_KEY).map(String::as_str),
            Some("candle/a.rs")
        );
        assert_eq!(CALIBRATION_FRAME_KEY, "calibration_frame");
    }

    /// A question's thinking is decoded as a live dialogue turn is — no dial
    /// but the seed set, so thinking stays on at the dialogue's default effort
    /// — and on the seed's absent branch.
    #[test]
    fn a_question_is_thought_about_as_a_live_turn_is_without_the_seed() {
        let s = frame_selection();
        assert_eq!(s.get("thinking_effort"), None);
        assert_eq!(s.optional(NO_THINK_SELECTOR), None);
        assert_eq!(s.optional(SEED_SELECTOR), Some(OptionalState::Absent));
    }

    /// A draft is asked without reasoning, without a tool block and without
    /// the seed.
    #[test]
    fn a_draft_neither_reasons_nor_sees_tools_nor_the_seed() {
        let s = draft_selection();
        assert_eq!(s.get("thinking_effort"), Some("off"));
        assert_eq!(s.optional(NO_THINK_SELECTOR), Some(OptionalState::Present));
        assert_eq!(
            s.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Absent)
        );
        assert_eq!(s.optional(SEED_SELECTOR), Some(OptionalState::Absent));
    }
}
