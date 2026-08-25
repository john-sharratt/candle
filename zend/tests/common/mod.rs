//! Shared harness for the behavioural test catalogue
//! (`docs/recurrent_state_behavioural_tests.md`).
//!
//! Every test in that catalogue is an invariant over what a conversation
//! **does**, so the harness's job is to make a scenario cheap to express and to
//! make two runs comparable. Two things it deliberately owns:
//!
//! - **One definition of a deterministic run.** Equivalence oracles compare two
//!   runs, and any difference in sampling makes the comparison meaningless
//!   rather than merely noisy. There is one place that fixes seed, sampler and
//!   budget, and every test uses it.
//! - **One definition of "open the engine".** A restart test needs a *second*
//!   engine over the *same* workspace, which is the whole point of the
//!   distinction between [`Workspace::open`] and a fresh one.
//!
//! ## Why `thinking` is not configurable here
//!
//! `ModelBuilder::thinking(false)` is inert in this configuration and it took a
//! confusing test output to notice: it writes `/no_think` into the *system*
//! prompt, which Qwen3 honours only from the user turn, and its other lever
//! (the model's non-thinking sampling params) is skipped whenever sampling is
//! set explicitly — which a deterministic run always does. So the harness does
//! not offer it. Tests that care about reasoning assert on memory, and a
//! `<think>` block in the output is the model's choice, not a defect.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::persistence::content_hash::ContentHash;
use candle_conversation::persistence::record::SnapshotPayload;
use candle_conversation::projection::{self, TimelineId};
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence, SequenceConfig};

const PROJECTION_YAML: &str = include_str!("../../src/prompts/projection.yaml");

/// The hybrid under test — 40 layers at 3:1, so 30 recurrent and 10 attention.
pub const HYBRID: Model = Model::Qwen36_35B_A3B_Q4;

/// Recurrent layers on this stack; a complete memory record carries all of them.
pub const RECURRENT_LAYERS: usize = 30;

/// Response budget.
///
/// Generous on purpose, and it took a false failure to learn why. This model
/// opens a reasoning block before answering, so a budget that only covers the
/// reasoning produces a reply with **no answer in it** — and a recall assertion
/// then fails for a conversation that remembered perfectly well. The first
/// version of this used 24 and reported a resuming fork as having forgotten,
/// when the truth was that it had not finished thinking.
const MAX_RESPONSE_TOKENS: usize = 512;

/// A workspace directory that engines can be opened over repeatedly.
///
/// Holds the `TempDir` so the directory outlives every engine opened on it —
/// which is what makes a restart expressible: drop the engine, keep the disk.
pub struct Workspace {
    dir: tempfile::TempDir,
}

/// Route the engine's WARNs to stderr, once per process.
///
/// The refusal paths this suite exercises are DESIGNED to be quiet failures
/// with a distinguishable WARN as their only trace (catalogue D8) — without a
/// subscriber every one of them is dropped on the floor, and a scenario that
/// "resumed empty" gives no clue which refusal fired. `RUST_LOG` still
/// overrides for deeper digs.
fn init_tracing() {
    static TRACING: std::sync::Once = std::sync::Once::new();
    TRACING.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            )
            .with_writer(std::io::stderr)
            .try_init();
    });
}

impl Workspace {
    pub fn new() -> Self {
        init_tracing();
        Self {
            dir: tempfile::tempdir().expect("tempdir"),
        }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Open an engine over this workspace and start a conversation on it.
    ///
    /// Called a second time after the first engine is dropped, this **is** the
    /// daemon restart: nothing survives but the bytes on disk.
    pub fn open(&self, device: &Device) -> (ConversationEngine, Sequence) {
        let session = self.session(device);
        let conv = session.start();
        (session.into_engine(), conv)
    }

    /// Open an engine and keep what is needed to start **more** conversations on
    /// it — the shape a daemon actually has, and what isolation tests need.
    pub fn session(&self, device: &Device) -> Session {
        Session::open_with_yaml(
            self.path().to_path_buf(),
            device,
            PROJECTION_YAML.to_string(),
        )
    }

    /// [`Self::session`] with an **edited system prompt** — the daemon
    /// restarted after a `projection.yaml` edit. A prompt edit is a schema
    /// edit landing across a restart: sections are content-addressed in the
    /// substrate, so the edited persona seals fresh and the prompt branch key
    /// changes with it. (An in-process schema swap is NOT this path —
    /// `set_projection` requires the new builder to share the current one's
    /// section ids, and two independent builders in one engine alias their
    /// numeric ids.)
    pub fn session_with_edited_prompt(&self, device: &Device) -> Session {
        Session::open_with_yaml(self.path().to_path_buf(), device, edited_projection_yaml())
    }
}

/// The projection schema with the persona section's content edited — the
/// E-family's prompt edit. A single-line phrase, because a multi-line pattern
/// against an `include_str!` file is a silent no-op under CRLF; the assert
/// makes a failed edit loud either way.
fn edited_projection_yaml() -> String {
    let edited = PROJECTION_YAML.replace(
        "You are Zen, an AI coding assistant",
        "You are Zen, an AI coding assistant who signs replies with 'aurora'",
    );
    assert_ne!(
        edited, PROJECTION_YAML,
        "the persona phrase moved — this edit no longer edits anything"
    );
    edited
}

/// Announce a scenario on stderr, so interleaved engine logs still say which
/// invariant the surrounding output belongs to.
pub fn scenario(id: &str, what: &str) {
    eprintln!("\n── {id} ── {what}");
}

/// The store behind [`shared_session`]: one engine per test process, loaded
/// on first touch, droppable by tests that need the VRAM for their own.
fn shared_store() -> &'static std::sync::Mutex<Option<Session>> {
    static SHARED: std::sync::OnceLock<std::sync::Mutex<Option<Session>>> =
        std::sync::OnceLock::new();
    SHARED.get_or_init(|| std::sync::Mutex::new(None))
}

/// Exclusive access to the process-wide shared [`Session`], derefing to it.
pub struct SharedSession(std::sync::MutexGuard<'static, Option<Session>>);

impl std::ops::Deref for SharedSession {
    type Target = Session;
    fn deref(&self) -> &Session {
        self.0
            .as_ref()
            .expect("the shared session is present while a guard exists")
    }
}

/// The shared engine for single-engine scenarios: loaded ONCE per test
/// process, on first touch, and reused by every `#[test]` that asks for it.
///
/// This is what lets each scenario be its own `#[test]` — separate pass/fail
/// reporting, runnable alone by name at the cost of one load, a panic in one
/// never aborting the others — while the 22 GB checkpoint still loads once
/// for the whole binary. The guard serialises scenarios on the engine even if
/// the harness runs test threads in parallel; deterministic runs additionally
/// want `--test-threads=1` so scenario ORDER is stable too.
///
/// Tests that open their OWN engines (the restart family — their subject is
/// the drop and reopen) call [`release_shared_session`] first, so peak VRAM
/// stays one engine even on the 16 GB machine. libtest's sorted order runs
/// the single-engine scenarios (`a…`–`h…`) before `restart…`, so the release
/// costs nothing on a full run and a later re-touch simply reloads.
pub fn shared_session() -> SharedSession {
    let mut guard = shared_store()
        .lock()
        // A panicked scenario must not cascade into every later one as a
        // PoisonError — the engine itself is still fine.
        .unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let device = Device::new_cuda(0).expect("cuda");
        // Leak the workspace guard: the tempdir must outlive the stored
        // engine, and process teardown reclaims both.
        let ws: &'static Workspace = Box::leak(Box::new(Workspace::new()));
        *guard = Some(ws.session(&device));
    }
    SharedSession(guard)
}

/// Exclusive occupancy of the engine slot for tests that open their OWN
/// engines (the restart family): drops the shared engine if loaded, and holds
/// the same lock every [`shared_session`] caller queues on — so no scenario
/// can reload the shared engine beside this test's, and everything in the
/// binary serialises through one mutex regardless of test threading.
pub struct ExclusiveEngineSlot(#[allow(dead_code)] std::sync::MutexGuard<'static, Option<Session>>);

pub fn exclusive_engine_slot() -> ExclusiveEngineSlot {
    let mut guard = shared_store().lock().unwrap_or_else(|e| e.into_inner());
    guard.take(); // return the shared engine's VRAM before loading our own
    ExclusiveEngineSlot(guard)
}

/// An open engine plus everything needed to start further conversations on it.
pub struct Session {
    engine: ConversationEngine,
    workspace: PathBuf,
    prompt: String,
    config: SequenceConfig,
    /// The projection schema this session's conversations are built from —
    /// `PROJECTION_YAML`, or an edited variant for the prompt-edit scenarios.
    yaml: String,
}

impl Session {
    fn open_with_yaml(workspace: PathBuf, device: &Device, yaml: String) -> Self {
        let (engine, prompt, config) = build_engine(workspace.clone(), device);
        await_substrate_reload(&engine);
        Self {
            engine,
            workspace,
            prompt,
            config,
            yaml,
        }
    }

    pub fn engine(&self) -> &ConversationEngine {
        &self.engine
    }

    pub fn into_engine(self) -> ConversationEngine {
        self.engine
    }

    /// Start another conversation on this engine. Each needs its own projection
    /// builder (the constructor consumes one), so this rebuilds it — the same
    /// thing the daemon does per conversation.
    pub fn start(&self) -> Sequence {
        let (builder, layer, group) =
            projection_for_yaml(&self.yaml, &self.workspace, self.engine.tokenizer());
        self.engine
            .new_conversation_with_projection(
                &self.prompt,
                builder,
                layer,
                group,
                self.config.clone(),
            )
            .expect("new conv")
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Block until the engine has finished rebuilding the substrate from the redo
/// log.
///
/// **An ordering requirement, not a convenience.** Recovering a workspace is
/// asynchronous, and a `fork_resuming` issued before it completes gets a
/// timeline the substrate does not yet know has turns — so the conversation
/// comes up with its memory restored and its *history* empty, and answers from
/// the system prompt alone. That is what the first run of this suite did, and
/// the model said so in its own reasoning: *"the current prompt provided to me
/// is just…"* followed by the system prompt.
///
/// Every embedder that resumes a workspace has to obey this, so the harness
/// obeys it in the one place engines are opened rather than leaving each test
/// to remember.
fn await_substrate_reload(engine: &ConversationEngine) {
    let status = engine.substrate_reload_status();
    for _ in 0..600 {
        let (_done, _total, finished) = status.snapshot();
        if finished {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("the substrate did not finish reloading within 60s");
}

/// A projection builder over an explicit schema, with its templates tokenised
/// — one per conversation (the constructor consumes it). Parameterised on the
/// YAML because the E-family's prompt edit is a SCHEMA edit: the branch
/// checkpoint is keyed over the projected sections (the schema is the single
/// source of truth for the system prompt), not the ChatML prompt string.
fn projection_for_yaml(
    yaml: &str,
    workspace: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> (
    projection::Builder,
    candle_conversation::projection::LayerId,
    candle_conversation::projection::GroupId,
) {
    let dialect = HYBRID.spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut b = projection::Builder::from_yaml_with_vars_and_dialect(
        yaml,
        &[("workspace", workspace_str.as_str())],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let layer = b.id_for_layer("dialogue").expect("dialogue layer");
    let group = b
        .id_for_group("primary_conversation")
        .expect("primary group");
    b.tokenize_templates::<anyhow::Error, _>(|s| {
        let encoded = tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
        Ok(encoded.get_ids().to_vec())
    })
    .expect("tokenize templates");
    (b, layer, group)
}

/// The one deterministic run configuration. Argmax and a fixed seed, so two runs
/// of the same tokens are comparable token-for-token rather than
/// distribution-for-distribution.
fn build_engine(
    workspace: PathBuf,
    device: &Device,
) -> (ConversationEngine, String, SequenceConfig) {
    let mut builder = HYBRID
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(MAX_RESPONSE_TOKENS);
    let config = builder.conversation_config();
    let prompt = builder.format_system_prompt();
    let engine = builder.engine(device).expect("engine load");
    (engine, prompt, config)
}

/// A fresh workspace with an engine already open on it — the common case.
pub fn fresh(device: &Device) -> (Workspace, ConversationEngine, Sequence) {
    let ws = Workspace::new();
    let (engine, conv) = ws.open(device);
    (ws, engine, conv)
}

/// A fresh workspace with a [`Session`] — for tests that need more than one
/// conversation on one engine.
pub fn fresh_session(device: &Device) -> (Workspace, Session) {
    let ws = Workspace::new();
    let s = ws.session(device);
    (ws, s)
}

// ── Driving a conversation ───────────────────────────────────────────────────

/// One complete turn, the shape the daemon uses, so the seal (and the memory
/// record it writes) runs exactly as it does in production.
///
/// That shape includes persisting the turn's final projection event after the
/// seal (`zend/src/session.rs` does exactly this): a resume seeds the
/// conversation's carried selection belief from those events, so a turn
/// without them resumes with a colder selection prior than the conversation
/// actually evolved — which is invisible until an A1-style twin comparison.
pub fn say(conv: &mut Sequence, text: &str) -> String {
    let resp = conv.send_turn(text).expect("send_turn");
    if let Some(event) = conv.projection_event(&resp.stats) {
        conv.persist_projection_events(&[event])
            .expect("persist projection events");
    }
    resp.text
}

/// [`say`], also returning a full rendering of the turn's **opening context**:
/// the exact prefill string the model received, and the opening projection's
/// materialized layout — every boundary-glue island's decoded text and every
/// turn's identity, exactly as the assembler injected them. This is the
/// context at the granularity where two conversations can actually differ;
/// everything coarser (block counts, item-level selection) has already
/// compared equal across a divergence.
pub fn say_opening(conv: &mut Sequence, text: &str) -> (String, String) {
    use candle_conversation::projection::{MaterializedPiece, SystemItem};
    use candle_conversation::TurnEvent;
    use std::fmt::Write;

    let handle = conv.submit_turn(text).expect("submit_turn");
    let mut opening = None;
    let mut prefill = None;
    let mut response = None;
    for ev in handle.stream() {
        match ev {
            TurnEvent::Projection(ev) => {
                if opening.is_none() {
                    opening = Some(ev);
                }
            }
            TurnEvent::Prefill(s) => prefill = Some(s),
            TurnEvent::Done(r) => {
                response = Some(r);
                break;
            }
            TurnEvent::Error(e) => panic!("turn error: {e}"),
            _ => {}
        }
    }
    let resp = response.expect("the stream ended without Done");
    conv.finish_turn(handle, &resp).expect("finish_turn");
    if let Some(event) = conv.projection_event(&resp.stats) {
        conv.persist_projection_events(&[event])
            .expect("persist projection events");
    }

    let mut s = String::new();
    if let Some(ev) = &opening {
        for item in &ev.selection.system {
            match item {
                SystemItem::Glue { name, tokens, .. } => {
                    let _ = write!(s, "glue:{name}({tokens}) ");
                }
                SystemItem::Section { name, tokens } => {
                    let _ = write!(s, "sec:{name}({tokens}) ");
                }
                SystemItem::Collection { name, sections } => {
                    let _ = write!(s, "coll:{name}[");
                    for sec in sections.iter().filter(|sec| sec.selected) {
                        let _ = write!(s, "{}({}) ", sec.name, sec.tokens);
                    }
                    let _ = write!(s, "] ");
                }
            }
        }
        for piece in &ev.materialized {
            match piece {
                MaterializedPiece::Glue { text } => {
                    let _ = write!(s, "GLUE{text:?} ");
                }
                MaterializedPiece::Turn { turn } => {
                    // Tag non-Normal kinds: a summary node standing in for a
                    // span of turns is exactly what the A7 oracle needs to see.
                    let _ = write!(s, "TURN:{}#{}({})", turn.group, turn.index, turn.tokens);
                    if turn.kind != candle_conversation::summary_tree::TurnKind::Normal {
                        let _ = write!(s, "[{:?}]", turn.kind);
                    }
                    s.push(' ');
                }
            }
        }
    } else {
        s.push_str("NO-OPENING-EVENT ");
    }
    if let Some(p) = prefill {
        let _ = write!(s, "PREFILL{p:?}");
    }
    (resp.text, s)
}

/// A run of filler turns, for tests that need depth rather than content.
pub fn say_n(conv: &mut Sequence, n: usize, tag: &str) {
    for i in 0..n {
        say(conv, &format!("{tag} {i}: reply with one short sentence."));
    }
}

// ── Observing memory ─────────────────────────────────────────────────────────

/// A conversation's **live** memory digest, or `None` when it holds none yet.
///
/// `None` is a real and distinct state, not an error: a conversation that has
/// not run a wave has no memory to read — a freshly forked one, for instance —
/// and that is different from holding memory that happens to be zeros. Tests
/// that mean "has not started" assert `None`; tests that mean "has forgotten"
/// assert [`memory_is_empty`].
pub fn memory_of(conv: &Sequence) -> Option<ContentHash> {
    conv.memory_digest().expect("read memory")
}

/// A conversation's live memory digest, requiring that it has some.
pub fn memory(conv: &Sequence) -> ContentHash {
    memory_of(conv).expect("this conversation has run at least one wave")
}

/// The same digest a live conversation reports, computed from layer rows.
///
/// Lets a **sealed record** be compared against **live memory** on equal terms —
/// the one comparison that says whether a record describes the K/V it was
/// written beside. It must hash exactly what `Sequence::memory_digest` hashes,
/// in the same order, or the comparison is vacuous.
pub fn digest_of_layers<'a>(
    layers: impl Iterator<Item = (u32, &'a [u8], &'a [u8])>,
) -> ContentHash {
    let mut h = candle_conversation::persistence::content_hash::ContentHasher::new();
    for (index, state, tail) in layers {
        h.update(&index.to_le_bytes());
        h.update(state);
        h.update(tail);
    }
    h.finish()
}

/// True when the conversation's memory is entirely zeros — the shape of one
/// that has forgotten everything, and which reads perfectly while doing so.
pub fn memory_is_empty(conv: &Sequence) -> bool {
    conv.memory_is_empty().expect("read memory")
}

/// The timeline's persisted memory record, read back through the path a resume
/// uses.
///
/// Polls for a record **for `turn_index`**, not merely for any record — and the
/// difference is not pedantry. The seal enqueues onto the off-thread writer, so
/// a read issued right after turn N routinely finds turn N-1's record still
/// standing. Waiting only for existence therefore compares the wrong turn, and
/// it does so silently: the first version of this helper made a correct seal
/// look like a memory-vs-K/V divergence and sent a whole investigation after a
/// bug that was not there.
pub fn sealed_memory_at(
    engine: &ConversationEngine,
    timeline: TimelineId,
    turn_index: u32,
) -> SnapshotPayload {
    let conv = engine.conversation();
    let mut seen = None;
    let mut last_err = None;
    // 120s: this is an ORDERING wait, not a latency assertion. Every turn's
    // seal enqueues a full ~63 MiB snapshot onto the off-thread writer
    // (single-tail supersede reclaims all but the last at compaction, but the
    // WRITE traffic is per-turn), so a deep multi-trial build can put whole
    // gigabytes ahead of this record — and it stays "indexed but unreadable"
    // until the group commit flushes its bytes.
    for _ in 0..2400 {
        match conv.read_recurrent_snapshot(timeline) {
            Ok(Some(p)) if p.turn_index == turn_index => return p,
            Ok(Some(p)) => {
                seen = Some(p.turn_index);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
            // Indexed-but-unreadable is transient while the off-thread writer
            // is mid-flight: the index entry can land a poll tick before the
            // payload bytes flush. Keep polling; a record that STAYS unreadable
            // for the whole window is genuine corruption and still panics below.
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
    // Report the whole picture: which turn's record was newest, and whether
    // any poll hit a torn read. (An early transient error must not shadow the
    // real diagnosis — "the turn's record never became readable".)
    panic!(
        "no readable memory record for turn {turn_index} within 120s (newest seen: \
         {seen:?}, last read error: {last_err:?}) — this conversation cannot be \
         resumed to that point, and nothing about the run would have said so"
    );
}

/// The timeline's newest persisted memory record, whatever turn it is for.
///
/// For tests that only need *a* durable record — that one exists, that it is
/// complete, that it is non-zero. Anything comparing a record against a
/// particular moment wants [`sealed_memory_at`].
pub fn sealed_memory(engine: &ConversationEngine, timeline: TimelineId) -> SnapshotPayload {
    let conv = engine.conversation();
    let mut last_err = None;
    // 120s ordering wait — see [`sealed_memory_at`].
    for _ in 0..2400 {
        match conv.read_recurrent_snapshot(timeline) {
            Ok(Some(p)) => return p,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
            // Transient while the off-thread writer is mid-flight — see
            // [`sealed_memory_at`].
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
    panic!(
        "no readable memory record within 120s (last read error: {last_err:?}) — \
         this conversation cannot be resumed, and nothing about the run would \
         have said so"
    );
}

/// Whether a sealed record carries memory that is not all zeros.
pub fn sealed_is_non_zero(p: &SnapshotPayload) -> bool {
    p.layers
        .iter()
        .any(|l| l.state.iter().any(|&b| b != 0) || l.conv_tail.iter().any(|&b| b != 0))
}

/// Layer indices whose memory differs between two sealed records.
pub fn sealed_diff(a: &SnapshotPayload, b: &SnapshotPayload) -> Vec<u32> {
    a.layers
        .iter()
        .zip(b.layers.iter())
        .filter(|(x, y)| x.state != y.state || x.conv_tail != y.conv_tail)
        .map(|(x, _)| x.layer_index)
        .collect()
}

// ── The amnesia control ──────────────────────────────────────────────────────

/// Produce the **amnesia control**: a conversation with the full history in its
/// K/V and nothing in its recurrent memory.
///
/// This is §1's failure condition, reached through a production path rather
/// than a test switch. A memory record is written for `timeline` under a
/// schedule hash the loaded model does not have; the resume then refuses it
/// (correctly, and loudly), the K/V splices in full, and the conversation runs
/// with its recurrent layers at zero.
///
/// It is what makes every quality claim a *paired* comparison. "The model
/// answered well" is not evidence; "the model answered well, and better than
/// the same model without its memory" is.
pub fn poison_memory_record(engine: &ConversationEngine, timeline: TimelineId) {
    let conv = engine.conversation();
    let mut payload = conv
        .read_recurrent_snapshot(timeline)
        .expect("read")
        .expect("the conversation has sealed memory to poison");
    // Any hash the model does not report. `import` validates it before touching
    // a tensor, so the refusal is total and the store is untouched.
    payload.schedule_hash ^= 0xFFFF_FFFF_FFFF_FFFF;
    conv.enqueue_recurrent_snapshot(timeline, payload.encode());
    // The write is fire-and-forget; give the writer a moment to land it before
    // a resume reads.
    std::thread::sleep(std::time::Duration::from_millis(300));
}

// ── Text helpers ─────────────────────────────────────────────────────────────

/// The answer half of a reply — everything after the reasoning block, or the
/// whole reply when there is none.
fn answer_of(reply: &str) -> &str {
    reply
        .rsplit_once("</think>")
        .map(|(_, a)| a)
        .unwrap_or(reply)
}

/// Whether a reply contains `needle`, case-insensitively, in its **answer** —
/// not in its reasoning. A model musing "I need to find the codeword" is not
/// recall.
pub fn recalls(reply: &str, needle: &str) -> bool {
    answer_of(reply)
        .to_lowercase()
        .contains(&needle.to_lowercase())
}

/// Ask `question` until the model produces an answer rather than only
/// reasoning, and report whether that answer recalls `needle`.
///
/// **Retrying is not papering over flakiness — it removes a confound.** This
/// model opens a reasoning block before answering, and a reply that ends inside
/// one contains no answer at all: matching it as "did not recall" fails a
/// conversation that remembered perfectly, which is the same class of false
/// signal these tests exist to eliminate, pointed the other way. A reply that
/// *does* reach an answer is a real observation, so the loop is bounded and only
/// discards non-observations.
///
/// Returns `(recalled, last_reply)` so a caller can print what it actually got.
pub fn probe_recall(conv: &mut Sequence, question: &str, needle: &str) -> (bool, String) {
    let mut last = String::new();
    for attempt in 0..3 {
        let reply = say(conv, question);
        let answered = !answer_of(&reply).trim().is_empty();
        last = reply;
        if answered {
            return (recalls(&last, needle), last);
        }
        eprintln!("[probe] attempt {attempt} produced no answer, only reasoning; retrying");
    }
    (recalls(&last, needle), last)
}
