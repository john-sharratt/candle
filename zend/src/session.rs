use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::SystemTime;

use futures::{Stream, StreamExt};
use notify::RecommendedWatcher;

use candle_conversation::models::{Dialect, Model};
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::persistence::{content_hash, SUBSTRATE_DIR};
use candle_conversation::projection::{
    self, Builder, GroupSchema, Reserved, SectionId, SelectionRule, SystemItem, SystemPromptItem,
    SystemPromptSchema, TimelineId, TurnIndex,
};
use candle_conversation::stencil::{ThinkMode, ToolSpec, TriggerRegistry};
use candle_conversation::substrate::Substrate;
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::{
    ConversationEngine, GlueMarkers, ProjectionEvent, Sequence, ThinkSteering, TokenDecoder,
    TurnEvent, TurnHandle, TurnResponse,
};
use serde_json::Value;

use crate::api::substrate::{
    ConvView, Counts, GroupView, LayerConversations, LayerView, ProjectTile, ProjectView,
    SectionView, SegmentView, Storage, SubstrateOverview, SystemPromptView, TimelineDetail,
    ToolView, ToolsView, TurnView,
};
use crate::code_read::CodeReadState;
use crate::config::DaemonConfig;
use crate::conv_file_store::ConvFileStore;
use crate::loading::{LoadProgress, LoadStep, LoadingSnapshot};
use crate::log_broadcast::LogBus;
use crate::projection_event::ProjectionEventOut;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{ClusterState, RepoMap};
use crate::tools::{
    extract_tool_calls, format_tool_responses, install_tool_catalog, run_tool_calls, ToolHost,
};
use crate::types::{ChatMessage, Role, ToolMode};

const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("prompts/projection.yaml");

/// Bound on the titler queue. A backlog beyond this many un-started titles
/// means the worker can't keep up; further jobs are dropped (labels are
/// best-effort), which is preferable to unbounded growth under a burst.
const TITLER_QUEUE_DEPTH: usize = 256;

/// Work item for the titler worker thread.
enum TitleJob {
    /// Generate a sidebar title for `timeline` from `message` and write it.
    Title {
        timeline: TimelineId,
        message: String,
    },
    /// Stop draining and exit (sent during shutdown to wake an idle worker).
    Shutdown,
}

// ── Repo-map handle ──────────────────────────────────────────────────────────

/// The `repo_map` layer's owning conversation paired with the
/// per-cluster hash record.  Refresh is at this granularity — the
/// whole conversation resets and re-inserts when any cluster's
/// file-name hash changes, so the [`Sequence`] and the [`ClusterState`]
/// must stay in lock-step.  Wrapping them in one struct keeps that
/// invariant under one mutex.
pub(crate) struct RepoMapConv {
    pub(crate) sequence: Sequence,
    pub(crate) state: ClusterState,
}

/// The `code_reading` layer's owning conversation paired with the
/// per-file content-hash record.  Same lock-step contract as
/// [`RepoMapConv`].
pub(crate) struct CodeReadConv {
    /// Merged per-file content-hash record. There are no live sequences:
    /// `code_reading` is now **one conversation per file**, each freed
    /// after its turns seal into the substrate. Retrieval queries the
    /// substrate's `active_timelines_for_group`; refresh finds and
    /// tombstones a changed file's conversation(s) by a `path` metadata
    /// scan rather than from a held sequence list.
    pub(crate) state: CodeReadState,
}

// The number of tools surfaced into the system prompt is governed by
// the `selection: { kind: top_k, k: N }` rule on the `tools` collection
// in `prompts/projection.yaml`.  Tune K there.

// ── Stream item ───────────────────────────────────────────────────────────────

/// A tool-execution lifecycle notice for the GUI.  Tool calls run *between*
/// streamed turns (the model emits the call, the daemon executes it, then a
/// `<tool_response>` turn continues), a gap during which the client otherwise
/// shows a static "no result" with no sign of activity.  Emitted with
/// `phase = "running"` just before the batch executes and `phase = "done"` once
/// results are back, so the GUI can show a spinner on the in-flight tool cards.
#[derive(Clone, serde::Serialize)]
pub struct ToolStatusOut {
    /// `"running"` while the batch executes, `"done"` once results are back.
    pub phase: &'static str,
    /// The tool names in this batch, in call order.
    pub tools: Vec<String>,
    /// Each tool's JSON response, in the same order as `tools` — populated only
    /// on the `"done"` notice (empty/omitted while running).  Lets the GUI
    /// resolve the in-flight cards immediately, before the post-stream hydrate.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<Value>,
}

/// Items yielded by [`ZendSession::submit`].
pub enum StreamItem {
    Status(String),
    Token(String),
    /// A projection event emitted once per decode (at seal): the materialized
    /// context's composition + decode throughput, driving the GUI timeline dots
    /// and the projection popover (docs/zend_ui_redesign.md §2.3).
    Projection(ProjectionEventOut),
    /// A tool-execution lifecycle notice (running / done) for the in-flight
    /// tool cards.  Display-only: never part of the collected completion body.
    Tool(ToolStatusOut),
}

/// Process-global monotonic id for projection events, so dot ids stay unique
/// across conversations and requests.
static PROJ_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// ── Inference state ───────────────────────────────────────────────────────────

struct ConvState {
    conv: Sequence,
    /// The tools mode last applied to this conversation's projection (set each
    /// turn in `run_inference_stream`). The projection panel uses it to display
    /// the tool summary that was actually injected — the restricted (safe-subset)
    /// summary in Restricted mode, the full one in Comprehensive, none in None.
    tool_mode: ToolMode,
}

/// Map a turn's `thinking_effort` dial to the steering [`ThinkMode`].  Mirrors
/// `dial_selection` in `api/chat.rs`: effort 0 → `off` (the `/no_think` glue
/// yields the empty block, so no tree steers it); an unset dial defaults to the
/// projection's `balanced` (free flow).
fn think_mode_from_selection(selection: &candle_conversation::SelectionState) -> ThinkMode {
    match selection.get("thinking_effort") {
        Some("off") => ThinkMode::Off,
        Some("quick") => ThinkMode::Quick,
        Some("deep") => ThinkMode::Deep,
        Some("exhaustive") => ThinkMode::Exhaustive,
        // Explicit `balanced`, or no dial set (projection default) → free flow.
        _ => ThinkMode::Balanced,
    }
}

/// Maps the composer's `response_length` dial (terse/concise/standard/detailed/
/// comprehensive = 0..4, the verbosity toggle) to the answer's token budget — the
/// room reserved after the think block, which the total-length EOS ramp covers
/// (`ThinkMode::eos_budget`).  These are generous hard caps (the typical answer is
/// shorter); the EOS boost/failsafe makes them the turn-ender backstop.
fn response_budget_from_selection(selection: &candle_conversation::SelectionState) -> i32 {
    match selection.get("response_length") {
        Some("terse") => 256,
        Some("concise") => 512,
        Some("detailed") => 2048,
        Some("comprehensive") => 3584,
        // Explicit `standard`, or no dial set (projection default).
        _ => 1024,
    }
}

/// Render the answer's held tool-call object for the client stream.  The GUI
/// renders tool cards from `<tool_call>…</tool_call>` tags, but Qwen3 frequently
/// emits the JSON object bare (no wrapper).  When `held` is a recognized tool call
/// (via the registry-gated [`extract_tool_calls`]), wrap it so the card renders;
/// otherwise return it verbatim (a bare object that isn't a call — e.g. a JSON
/// answer — streams as plain text).  `None` when there's nothing to emit.
fn render_held_tail(held: &str) -> Option<String> {
    let held = held.trim();
    if held.is_empty() {
        return None;
    }
    if extract_tool_calls(held).is_empty() {
        Some(held.to_string())
    } else {
        Some(format!("<tool_call>\n{held}\n</tool_call>"))
    }
}

struct InferenceState {
    decoder: TokenDecoder,
    /// Owns the scheduler `JoinHandle` (conversations hold cloned
    /// `scheduler_tx` senders) and the substrate persistence handle —
    /// locked for the per-turn group-commit and for shutdown checkpointing.
    engine: Mutex<ConversationEngine>,
    /// Per-conversation state, keyed by the client-supplied conv_id string.
    conversations: Mutex<HashMap<String, Arc<Mutex<ConvState>>>>,
    /// System-prompt already prefilled; all new conversations fork from this.
    base_conv: Mutex<Sequence>,
    /// Queue feeding the dedicated titler worker thread. The request path
    /// enqueues a [`TitleJob`] (non-blocking, dropped if the worker is backed
    /// up) instead of spawning a thread per submit, so title generation runs
    /// in the background — concurrently with main decode, never serialised
    /// against the request path — and drains cleanly on shutdown.
    titler_tx: SyncSender<TitleJob>,
    /// Handle to the titler worker thread, joined during [`ZendSession::shutdown`]
    /// so any in-flight title turn unwinds before the process exits.
    titler_worker: Mutex<Option<JoinHandle<()>>>,
    /// The titler's timeline id — excluded from `list_conversations` so
    /// it doesn't show up in the user-facing sidebar.
    titler_timeline: TimelineId,
    /// Set once shutdown begins. The titler worker checks it between jobs and
    /// stops draining; the request path stops enqueuing new title jobs.
    shutting_down: AtomicBool,
    /// `repo_map` layer's owning conversation.  Holds one
    /// prefilled turn pair per directory cluster; [`ClusterState`]
    /// is the per-cluster file-name hash record the refresh path
    /// consults to decide whether to rebuild atomically.
    repo_map_conv: Mutex<RepoMapConv>,
    /// Workspace root captured at startup — the refresh path
    /// re-walks from here on every filesystem event.
    workspace: PathBuf,
    /// `code_reading` layer's owning conversations — one per file,
    /// each a per-part tool-call exchange closed by a whole-file
    /// summary; [`CodeReadState`] is the per-file content hash record
    /// the refresh path consults.
    code_read_conv: Mutex<CodeReadConv>,
    /// Projection builder + sequence config kept alive for the
    /// atomic refresh paths.  Minting a fresh repo_map or
    /// code_reading timeline after a file change reuses the same
    /// schema clone and the same dialect config the initial
    /// ingestion ran under.
    refresh_builder: Builder,
    refresh_config: candle_conversation::SequenceConfig,
    /// Per-tools-mode projection builders, built once at startup (Restricted
    /// drops the high-risk tools; None drops the whole catalog). Each turn hands
    /// out the matching one as a cheap `Arc` clone via [`ModeBuilders::get`].
    mode_builders: ModeBuilders,
    /// Tokenizer shared with the engine — used by `head_tail_truncate`
    /// to bound the titler's prefill at first 50 + last 50 tokens.
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// Shared tool execution context (notes / credentials / sessions /
    /// VFS stores).  Cloned per-conversation if scoping is later
    /// needed; for now it's workspace-wide.
    tool_host: ToolHost,
    /// Constrained-decoding stencil for the whole tool catalog, keyed by the
    /// `<tool_call>` trigger token.  Passed on every user turn so a tool call
    /// the model emits is forced to the catalog's exact JSON shape.  Compiled
    /// once at startup from `zend_tools::registry`.
    tool_stencil: Arc<TriggerRegistry>,
    /// Per-effort-dial thinking-block steering trees, compiled once at startup.
    /// `None` when the tokenizer has no single `<think>`/`</think>` token.  Each
    /// turn replaces the `<think>` trigger with the dial's tree (see
    /// `think_mode_from_selection`) atop the tool-call base registry.
    think_steering: Option<Arc<ThinkSteering>>,
    /// Tokenized closer phrase for the think block's HARD length cap: played
    /// (followed by `</think>`) instead of a bare mid-sentence close, so the
    /// amputated reasoning ends as intentional prose with an explicit
    /// commitment. Tokenized once at startup; empty when the tokenizer failed
    /// to encode it. See `SamplingConfig::segment_close_script`.
    think_closer_phrase: Vec<u32>,
}

// The titler uses this plain system prompt (not the projection schema), so the
// dialogue `no_think` *section* never applies. `/no_think` is baked here for the
// system side; the user side comes from the live `no_think_current` glue the
// scheduler emits because generate_one_title sets `NO_THINK_SELECTOR` — one
// glue mechanism shared with the dialogue, so Qwen3 suppresses reasoning instead
// of burning the token budget on a `<think>` block.
const TITLER_SYSTEM_PROMPT: &str = "/no_think\nYou write short conversation titles. \
Given the user's first message, reply with a 3-6 word title that captures its topic. \
No quotes, no period, no preamble — just the title.";

const TITLER_HEAD_TOKENS: usize = 50;
const TITLER_TAIL_TOKENS: usize = 50;
// Room for the title plus the empty `<think></think>` Qwen3 still decodes under
// /no_think (no longer prefilled for free) — the block is stripped afterwards.
const TITLER_MAX_TOKENS: usize = 24;
/// Max tokens for a calibration turn's **decode fallback**. Calibration normally
/// prefills the recorded `.md` trajectory verbatim (`max_decode = 0`), so this
/// cap does nothing; it only applies to a case whose trajectory body can't be
/// extracted and is decoded live instead. Generous so a verbose reasoner (or a
/// long tool-call payload) finishes through `</tool_call>` rather than being cut
/// mid-think; the decode still stops early on the natural `<|im_end|>`, so short
/// cases pay nothing for the headroom. At 1280 the longest thinks (verbose
/// TLS/HKDF spec musing, ~30+ reasoning lines) still truncated the `<tool_call>`;
/// 2048 gives them room to finish.
const CALIBRATION_MAX_TOKENS: usize = 2048;

/// Closer phrase the sampler plays (followed by `</think>`) when the think
/// block's HARD per-span cap fires mid-sentence — the em-dash lead-in reads as
/// a deliberate self-interruption after any dangling fragment, and the
/// commitment ("know what to do") primes the answer that follows. Tokenized
/// once at startup into `InferenceState::think_closer_phrase`.
const THINK_CLOSER_PHRASE: &str = " — actually, I've reasoned enough and know what to do.";

/// Seal a completed calibration turn the same way a normal decode does: finish the
/// turn (which runs the BDP scan `projection_event` reads) and persist its
/// projection event, so the substrate carries the durable turn→tool-section
/// provenance link — the `tools` collection with the pinned tool marked selected.
/// Best-effort: a failure only drops the supplementary provenance record, never the
/// calibration case itself. The turn's tokens / KV signatures / wide-Q window are
/// already sealed by the scheduler independently of this call.
/// Mark a completed calibration `timeline` for distillation, idempotently. Only
/// a timeline that still has KV to reclaim and isn't already marked is touched —
/// so re-running calibration (or resuming an already-distilled corpus) neither
/// re-marks nor re-triggers compaction. The `.md` trajectories are the source of
/// truth, so once the wide-Q sig is captured the KV/token content is dead weight
/// compaction can shed.
/// The non-template content sections that precede the `tools` collection in the
/// dialogue layer. This is the prefix the tool sections are sealed against (the
/// summariser excludes collection members + templates from the prefix chain), so
/// sealing the tool-summary section with the *same* prefix puts its KV exactly
/// where "just before the tools" is.
fn pre_tools_section_ids(builder: &Builder) -> Vec<SectionId> {
    let Some(layer) = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == "dialogue")
    else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) if !s.is_template => ids.push(s.id),
            // Section-tree variants are sealed (not live templates), and
            // `pre_collection_prelude` bakes their default content into the
            // prefix — so the tool summary must seal against the same default
            // variant ids to land at the right position.
            SystemPromptItem::SectionTree(t) => ids.extend(t.default_present_ids.iter().copied()),
            // The dialogue layer's only collection is `tools`; stop at it.
            SystemPromptItem::Collection(_) => break,
            SystemPromptItem::Section(_) => {}
        }
    }
    ids
}

fn mark_calibration_distill(engine: &ConversationEngine, timeline: TimelineId, tool: &str) {
    if engine.timeline_has_kv(timeline) && !engine.is_timeline_distilled(timeline) {
        // Calibration exemplars stay retrievable by signature — provenance-only.
        if let Err(e) = engine.distill_timeline(timeline, DistillMode::ProvenanceOnly) {
            tracing::warn!(tool = tool, "calibration distill mark failed: {e}");
        }
    }
}

fn seal_calibration_turn(
    conv: &mut Sequence,
    handle: TurnHandle,
    resp: &TurnResponse,
    tool: &str,
    mut events: Vec<ProjectionEvent>,
) {
    if let Err(e) = conv.finish_turn(handle, resp) {
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
    if let Err(e) = conv.persist_projection_events(&events) {
        tracing::warn!(
            tool = tool,
            "calibration persist projection events failed: {e}"
        );
    }
}

/// In-flight calibration cases held in the sliding window. Cases are
/// prefill-only (the exported trajectory is prefilled, not decoded — see
/// `submit_prefilled_turn`), so the whole window co-batches into one prefill
/// forward shape; a wider window just packs more sequences per forward, up to
/// the `max_prefill_pass_tokens` budget. 16 keeps the peak VRAM footprint of the
/// concurrently-resident lossless case K/V within a tight (16 GB) card while
/// still filling every forward — the window is now populated in one pipelined
/// batch (`new_conversations_with_projection_batch`), so it actually reaches
/// this width instead of trickling in one case per wave-latency.
const CALIBRATION_BATCH: usize = 16;
/// Conversation-metadata key tagging each calibration conversation with its
/// `"{tool}|{example}"` case **at creation**, so it is findable by case on a
/// later load — finished or half-finished. *Done* is signalled separately by
/// **archiving** the conversation once its trajectory completes (`</tool_call>`);
/// a tagged but un-archived conversation is a decode that was cut off mid-case,
/// so the next load tombstones and regenerates it (the archive is the atomic
/// commit). Changing an example's text changes its marker, so edited examples
/// regenerate automatically.
const CALIB_MARKER_KEY: &str = "calib";

impl InferenceState {
    fn load(
        mut proj_builder: Builder,
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        workspace: PathBuf,
        skip_code_read: bool,
        skip_repo_scan: bool,
        disable_summariser: bool,
        compact_substrate: bool,
        progress: Arc<LoadProgress>,
    ) -> anyhow::Result<Arc<Self>> {
        // Step 1: model. Engine ctor also reloads the substrate
        // internally, so the visible boundary between Model and
        // Substrate steps below is best-effort — the substrate has
        // actually already loaded by the time we announce its step.
        progress.set_step(LoadStep::Model);
        let device = candle::Device::cuda_if_available(0)
            .map_err(|e| anyhow::anyhow!("device init: {e}"))?;

        // VRAM advisory at startup.  The 4090 mobile baseline is
        // 16 GB; the daemon's model + expert cache + scheduler
        // buffers leave roughly 3-5 GB for parallel-ingest peak
        // working set.  When VRAM is constrained we shout a clear
        // recommendation in the log so a CUDA OOM during code-read
        // is traceable rather than an opaque exit-1.
        if let Ok((free, total)) = device.mem_get_info() {
            let free_gib = (free as f64) / (1024.0 * 1024.0 * 1024.0);
            let total_gib = (total as f64) / (1024.0 * 1024.0 * 1024.0);
            let n_workers = crate::code_read::CODE_READ_PARALLELISM;
            let recommended = if total_gib < 24.0 {
                "<= 8 workers recommended for 16 GB"
            } else if total_gib < 40.0 {
                "<= 16 workers recommended for 24-32 GB"
            } else {
                "32-64 workers usable on this VRAM budget"
            };
            tracing::info!(
                "VRAM {free_gib:.1}/{total_gib:.1} GiB free · code_read workers={n_workers} \
                 ({recommended}; override ZEND_CODE_READ_PARALLELISM=N)"
            );
        }

        // Resolve the dialogue layer / primary group up front.  The
        // tool-catalog injection adds sections to that layer.
        let dialogue_layer = proj_builder
            .id_for_layer("dialogue")
            .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
        let primary_group = proj_builder
            .id_for_group("primary_conversation")
            .ok_or_else(|| {
                anyhow::anyhow!("projection schema missing 'primary_conversation' group")
            })?;
        // (layer, group) are passed through to
        // `new_conversation_with_projection`, which mints a fresh
        // `TimelineId` internally.

        // Install the tool catalog into the schema before building the
        // base conversation.  Each tool gets a section in dialogue's
        // system_prompt; the projection layer's `tools` collection
        // governs `top_k` so only the K most relevant tools survive
        // into any single projection.
        //
        // Sections re-prefill on every daemon start in the current
        // configuration — section cold-load is plumbed end-to-end but
        // disabled on the runtime path (see Phase 2.5 in
        // `persistence/thread.rs` and the matching scheduler filter
        // notes).  Tool sections live hot for the daemon's lifetime
        // and the manifest never grows section chunk records.  Cost
        // is one prefill pass over the catalog per daemon start
        // (~90 tools × short JSON line); cheap on the 4090 mobile
        // baseline and easy to re-flip once cold-load is back.
        let tool_sections = install_tool_catalog(&mut proj_builder, dialogue_layer)
            .map_err(|e| anyhow::anyhow!("tool catalog install: {e}"))?;
        tracing::info!(
            n_tools = tool_sections.len(),
            "tool catalog installed (top_k governed by `tools` collection in projection.yaml)",
        );

        // The dialogue layer's `system_prompt.items` start with a static
        // prelude (mode/frame/grounding/tools_intro) →
        // then the `tools` collection (90+ tool sections, top_k=3) →
        // then `tools_outro`.  The pre-collection prelude is what we
        // pass as the engine's `system_prompt` so it gets ChatML-wrapped
        // for the dialect; everything after the first Collection is
        // expanded by `preemptive_prefill` itself.
        let before_text: String = pre_collection_prelude(&proj_builder);

        let mut builder = Model::Qwen3_30B_A3B_Q4
            .builder()
            .system_prompt(&before_text)
            .model_path(model_path)
            .tokenizer_path(tokenizer_path)
            .workspace_path(workspace.clone())
            // Dialogue turns compress at C5 (moderate adaptive quantization).
            // Paired with the removed uniform-K pin (see `ModelBuilder::engine`),
            // so K is adaptive too.
            .compression_level(5)
            // Thinking is ENABLED at the model level; per-turn suppression is
            // driven by the section-tree `no_think` selector (the composer
            // effort dial), not this static flag.
            .thinking(true)
            // `--disable-summariser`: bring the engine up without the AVL
            // summary-forest thread (e.g. for bulk corpus prefill).
            .disable_summariser(disable_summariser);
        let conv_config = builder.conversation_config();

        // Per-layer progress callback — the library reports
        // `(layers_loaded, total_layers)` after each transformer block
        // is mounted. We translate that into the LoadProgress fraction
        // without coupling the library to our state machine.
        let model_progress = Arc::clone(&progress);
        let model_hook = move |done: usize, total: usize| {
            model_progress.set_step_progress(done as u64, total as u64);
        };
        let engine = builder
            .engine_with_progress(&device, Some(&model_hook))
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;

        // Compile the whole tool catalog into one constrained-decoding stencil,
        // keyed by the `<tool_call>` trigger.  Passed on every user turn so any
        // tool call the model starts is forced to the catalog's exact shape.
        let tool_specs: Vec<ToolSpec> = zend_tools::registry::all_tools()
            .iter()
            .map(|t| ToolSpec::from_json_schema(t.name, &(t.schema)()))
            .collect();
        let tool_stencil = engine
            .compile_tool_stencil(&tool_specs)
            .map_err(|e| anyhow::anyhow!("tool stencil compile: {e}"))?;
        // The thinking-block steering trees (one per non-off effort dial),
        // compiled once and reused across turns alongside the tool-call base.
        let think_steering = engine
            .compile_think_steering()
            .map_err(|e| anyhow::anyhow!("think steering compile: {e}"))?;

        // The substrate reload (redo-log replay) runs on the scheduler thread
        // spawned by the engine ctor. As the substrate grows this is no longer
        // instantaneous, so surface real progress and block here until it
        // finishes before advancing to section prefill (which would otherwise
        // stall against the busy scheduler under a misleading step label).
        progress.set_step(LoadStep::Substrate);
        {
            let reload = engine.substrate_reload_status();
            loop {
                let (done, total, finished) = reload.snapshot();
                if total > 0 {
                    progress.set_step_progress(done as u64, total as u64);
                }
                if finished {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }

        // Reclaim is normally fully background: the segmented log's
        // persistence-thread maintenance pass drops / compacts / combines
        // segments incrementally (`docs/segmented_substrate_log.md` §6), so a
        // startup pays no whole-store rewrite. `--compact-substrate` forces the
        // eager path instead — a whole-store rewrite here, after the reload (so
        // the live set is known) and before serving. It always runs when the
        // flag is set (no reclaimable-marker gate): the operator asked for it.
        if compact_substrate {
            progress.set_step(LoadStep::Compacting);
            let cprog = Arc::clone(&progress);
            let cb = move |done: usize, total: usize| {
                cprog.set_step_progress(done as u64, total as u64);
            };
            match engine.compact_substrate(Some(&cb)) {
                Ok(()) => tracing::info!("substrate compaction complete"),
                Err(e) => tracing::warn!("substrate compaction failed: {e:#}"),
            }
            // Compaction rewrote the log — re-reconstruct the substrate so the
            // scheduler-side view (KV residence + offsets) matches the new log.
            let reload = engine.reload_substrate();
            loop {
                let (done, total, finished) = reload.snapshot();
                if total > 0 {
                    progress.set_step_progress(done as u64, total as u64);
                }
                if finished {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }

        let formatted_prompt = builder.format_system_prompt();
        let decoder = engine.token_decoder();

        // Tokenise every `kind: template` item in the projection schema
        // using the engine's tokenizer.  The projection engine emits a
        // `Generated` segment for each template at apply time carrying
        // these tokens; the assembler then runs (or caches) a live
        // prefill against them under the current runtime left context.
        let tokenizer = engine.tokenizer();
        proj_builder.tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })?;

        progress.set_step(LoadStep::Sections);
        // Per-section progress callback — bytes ingested out of total
        // section bytes (cheap proxy for tokens, smooth across uneven
        // sections without a pre-tokenise pass). The "Prefilling tool
        // sections" step is split in two: section prefill fills the first
        // half (0..50%), the tool-catalog summary decode fills the second
        // (50..100%) — both reported against a fixed 10_000-unit total so the
        // two phases compose into one continuous bar.
        let section_progress = Arc::clone(&progress);
        let section_hook = move |done: u64, total: u64| {
            let scaled = if total == 0 { 0 } else { done * 5_000 / total };
            section_progress.set_step_progress(scaled, 10_000);
        };
        // Two clones of the (tools-installed + templates-tokenised)
        // projection builder go to the repo_map and code_reading
        // conversations below.  Cloning is cheap (schemas are
        // `Arc`-backed) and avoids re-running the install +
        // tokenise passes.
        let proj_builder_repo_map = proj_builder.clone();
        let proj_builder_code_read = proj_builder.clone();
        // Third clone stays on `InferenceState` for the atomic
        // refresh paths to mint fresh repo_map / code_reading
        // timelines when the watcher fires.
        let proj_builder_refresh = proj_builder.clone();
        let mut base_conv = engine
            .new_conversation_with_projection_progress(
                &formatted_prompt,
                proj_builder,
                dialogue_layer,
                primary_group,
                conv_config.clone(),
                Some(&section_hook),
            )
            .map_err(|e| anyhow::anyhow!("base conv create: {e}"))?;

        // Section ingestion (prelude + parallel tool-section forks +
        // outro) happens eagerly inside
        // `new_conversation_with_projection`: the schema's declared
        // sections / collections are pinned into the workspace
        // substrate at construction time via `insert_section` /
        // `insert_section_collection`.  Each tool section's BDP sigs come
        // from that single prefill of its JSON definition; JSON
        // structural tokens are excluded from the reprojection probe
        // (`build_reproject_probe_filter_token_ids`) so they don't add
        // shared-structure noise to section scoring.
        tracing::info!(
            n_tool_sections = tool_sections.len(),
            "base conversation ready (prelude + tool catalog + outro pinned at init)",
        );

        // Tool-catalog summaries: one for "Comprehensive" tools mode (the full
        // catalog) and one for "Restricted" mode (the safe / non-high-risk
        // subset). Each is assembled **deterministically** from the catalog
        // metadata — every tool grouped under its category — so there is no model
        // call and nothing to cache: the text is rebuilt and its section prefilled
        // on every startup, exactly like the tool sections themselves.
        {
            let prelude = pre_tools_section_ids(&proj_builder_refresh);
            let safe_names = crate::tools::safe_tool_names();
            let safe_sections: Vec<crate::tool_summary::InstalledTool> = tool_sections
                .iter()
                .filter(|(name, _, _)| safe_names.contains(name))
                .cloned()
                .collect();
            let comp_text = crate::tool_summary::build_tool_summary(&tool_sections);
            let restr_text = crate::tool_summary::build_tool_summary(&safe_sections);
            // Assembly is instant — the step's second half completes immediately.
            progress.set_step_progress(10_000, 10_000);

            // Seal each mode's summary under its reserved section id, prefilled with
            // the same pre-tools prefix so its KV is position-correct for "just
            // before the tools". The Restricted projection points the tools
            // collection at `ToolSummaryRestricted`, Comprehensive at `ToolSummary`;
            // None emits neither.
            for (text, reserved, label) in [
                (comp_text, Reserved::ToolSummary, "comprehensive"),
                (restr_text, Reserved::ToolSummaryRestricted, "restricted"),
            ] {
                let sid = SectionId::reserved(reserved);
                match base_conv.insert_section_with_prefix(sid, &text, &prelude) {
                    Ok(()) => tracing::info!(
                        section = sid.raw(),
                        label,
                        "tool summary section sealed (prefilled before tools)",
                    ),
                    Err(e) => tracing::warn!(label, "tool summary section seal failed: {e}"),
                }
            }
        }

        // Dedicated titler conversation. Lives on the `Reserved::Titler`
        // id range (at the top of the u32 space) so its layer/group/section
        // can't collide with the user schema's YAML-allocated ids — even
        // though they share the same workspace substrate, the titler's
        // turns never enter a user conversation's projection.
        //
        // Its KV cache for the (tiny) system prompt is prefilled once here;
        // every title-generation reuses it.
        let titler_formatted = conv_config
            .dialect
            .format_system_prompt(TITLER_SYSTEM_PROMPT);
        let titler = engine
            .new_reserved_conversation(&titler_formatted, Reserved::Titler, conv_config.clone())
            .map_err(|e| anyhow::anyhow!("titler create: {e}"))?;
        let titler_timeline = titler.timeline_id();
        let tokenizer = Arc::new(engine.tokenizer().clone());
        tracing::info!(
            titler_timeline = ?titler_timeline,
            "titler conversation ready",
        );

        // ── Calibrating sections ──────────────────────────────────────────────
        // Free-decode each registered tool's authored examples into the hidden
        // `Reserved::Calibration` layer, capturing the full think→call trajectory
        // (and its per-reprojection wide-Q windows) for each.
        //
        // All cases share ONE projection: the whole catalog as a name-keyed `tools`
        // collection governed by `SelectionRule::Named`. Each run pins exactly its
        // tool by name (`TurnOptions::selection`), so the model sees a single
        // available tool — the condition under which a free decode reliably calls
        // the right one. The tool sections are sealed once (the first run); every
        // later run reuses them via the idempotent section ingest. One throwaway
        // conversation per (tool, example); a failing case is logged and skipped so
        // it can never break daemon load.
        progress.set_step(LoadStep::CalibratingSections);
        {
            let tools = zend_tools::registry::all_tools();
            let total: usize = tools
                .iter()
                .map(|t| t.examples.iter().filter(|e| !e.is_empty()).count())
                .sum();
            tracing::info!(
                cases = total,
                "calibrating sections: per-tool example decodes (named tool selection)"
            );
            let (calib_builder, calib_layer, calib_group) =
                crate::tools::build_calibration_projection(&conv_config.dialect)
                    .map_err(|e| anyhow::anyhow!("build calibration projection: {e}"))?;
            let calib_prelude = pre_collection_prelude(&calib_builder);
            // Pin calibration turns to native R16/F16 (no hot→warm quantize) so every
            // token's Q survives to the seal-time wide `sign(Q)` capture — the
            // full-resolution provenance exemplars this phase exists to produce.
            let calib_config = {
                let mut c = conv_config.clone();
                c.kv_lossless = true;
                c
            };
            // Flatten to (tool, example, index) cases in registry order. The
            // 1-based index is the position in the tool's `[_; 8]` examples array
            // (before filtering empties) — it keys the exported `{tool}_{NN}.md`
            // trajectory the prefill path loads.
            let cases: Vec<(&str, &str, usize)> = tools
                .iter()
                .flat_map(|t| {
                    t.examples
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| !e.is_empty())
                        .map(move |(i, e)| (t.name, *e, i + 1))
                })
                .collect();

            // A calibration case is "done" only once its conversation is
            // ARCHIVED — the atomic commit. Each conversation is tagged with its
            // case at creation, so a half-finished one (daemon cut off mid-decode)
            // is still findable: present-but-unarchived → tombstone + regenerate.
            // The archive flag is read per-timeline via `is_conversation_archived`
            // (NOT `known_conversations`, which omits internal conversations with
            // no `conv_id` — i.e. every calibration conversation, which is why an
            // earlier snapshot-based check never matched and resume never worked).

            // Resume filter: skip cases already archived, tombstone any
            // half-finished prior, and collect the cases that still need running.
            // Conversations are NOT created here — creation is a synchronous
            // scheduler round-trip, so pre-creating one per case would allocate
            // hundreds of (empty) slots at once. Instead each case's conversation
            // is created lazily as the window refills (below), bounding the live
            // slot count to `CALIBRATION_BATCH` — the concurrency the decode/prefill
            // window can actually keep busy.
            let mut done = 0usize;
            let mut to_run: Vec<(&str, &str, usize)> = Vec::new();
            for (name, example, idx) in cases.iter() {
                let marker = format!("{name}|{example}");
                let prior = engine.find_conversations_by_metadata(CALIB_MARKER_KEY, &marker);
                // Done iff a prior conversation for this case is archived.
                if prior.iter().any(|t| engine.is_conversation_archived(*t)) {
                    // The completed corpus only needs its wide-Q sigs, so mark the
                    // archived conversation for distillation — compaction reclaims
                    // its trajectory content (`.md` files are the source of truth).
                    // Idempotent: skip a timeline that's already distilled.
                    for t in prior
                        .iter()
                        .filter(|t| engine.is_conversation_archived(**t))
                    {
                        mark_calibration_distill(&engine, *t, name);
                    }
                    done += 1;
                    progress.set_step_progress(done as u64, total as u64);
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
                to_run.push((*name, *example, *idx));
            }

            // Non-blocking sweep: archive + retire every case finished since the
            // last sweep. Archives only complete trajectories (`</tool_call>`); an
            // incomplete/failed one stays un-archived to retry next load. Returns
            // whether anything was retired.
            // Each in-flight case carries the mid-decode reprojection events streamed
            // so far, accumulated across sweeps (they arrive interleaved with tokens).
            // The trailing `bool` is `is_prefill`: a prefilled trajectory is
            // complete by construction (it's the validated exported `.md`), and
            // its `Done` response carries no decoded text — so it must NOT be
            // gated on `resp.text` the way a live decode is.
            let mut inflight: Vec<(Sequence, TurnHandle, &str, Vec<ProjectionEvent>, bool)> =
                Vec::new();
            let retire_completed =
                |inflight: &mut Vec<(Sequence, TurnHandle, &str, Vec<ProjectionEvent>, bool)>,
                 done: &mut usize,
                 reclaim: &mut Vec<TimelineId>|
                 -> bool {
                    let mut retired = false;
                    let mut i = 0;
                    while i < inflight.len() {
                        let mut collected: Vec<ProjectionEvent> = Vec::new();
                        let terminal = loop {
                            match inflight[i].1.try_recv() {
                                Some(TurnEvent::Done(resp)) => break Some(Some(resp)),
                                Some(TurnEvent::Error(_)) => break Some(None),
                                Some(TurnEvent::Projection(ev)) => collected.push(ev),
                                Some(_) => {}
                                None => break None,
                            }
                        };
                        inflight[i].3.append(&mut collected);
                        match terminal {
                            Some(result) => {
                                let (mut conv, handle, name, events, is_prefill) =
                                    inflight.remove(i);
                                match result {
                                    Some(resp)
                                        if is_prefill || resp.text.contains("</tool_call>") =>
                                    {
                                        seal_calibration_turn(
                                            &mut conv, handle, &resp, name, events,
                                        );
                                        if let Err(e) = engine
                                            .set_conversation_archived(conv.timeline_id(), true)
                                        {
                                            tracing::warn!(
                                                tool = name,
                                                "calibration archive failed: {e}"
                                            );
                                        }
                                        // Only the wide-Q sig is needed henceforth —
                                        // mark for distillation so compaction sheds
                                        // the trajectory content.
                                        mark_calibration_distill(&engine, conv.timeline_id(), name);
                                        // Retired: its sealed K/V is never attended
                                        // again (only the persisted wide-Q sig is),
                                        // so queue its timeline for hot→warm demotion
                                        // to keep calibration's VRAM footprint flat.
                                        reclaim.push(conv.timeline_id());
                                    }
                                    Some(_) => tracing::warn!(
                                    tool = name,
                                    "calibration trajectory incomplete — left un-archived for retry"
                                ),
                                    None => tracing::warn!(tool = name, "calibration case failed"),
                                }
                                *done += 1;
                                progress.set_step_progress(*done as u64, total as u64);
                                retired = true;
                            }
                            None => i += 1,
                        }
                    }
                    retired
                };

            // The assistant header marker the exported trajectory is split on —
            // everything after it is the verbatim body we prefill.
            let assistant_start = conv_config.dialect.assistant_start;
            let mut to_run_iter = to_run.into_iter();
            let mut warmed = false;
            // Timelines of archived (retired) calibration cases, awaiting hot→warm
            // demotion. Their sealed K/V is never attended again — only the
            // persisted wide-Q sig is — so dropping the hot copy as they retire
            // keeps the phase's VRAM footprint flat instead of letting 700+ cases'
            // lossless K/V pile up hot and OOM the next phase's first prefill.
            let mut calib_timelines: Vec<TimelineId> = Vec::new();
            // Index into `calib_timelines` up to which we've already demoted, so
            // each incremental sweep only re-demotes the fresh tail.
            let mut reclaimed_up_to = 0usize;
            loop {
                // Refill the window in ONE pipelined round-trip:
                // `new_conversations_with_projection_batch` fires every slot
                // allocation before awaiting any, so the whole window's cases
                // prefill together in a wide forward instead of trickling in one
                // per wave-latency (which starved the batch to 2–4 wide). The
                // warm-up case is created alone first so the shared tool sections
                // pin once before the concurrent window opens.
                let want = if warmed {
                    CALIBRATION_BATCH.saturating_sub(inflight.len())
                } else {
                    1
                };
                let batch: Vec<(&str, &str, usize)> =
                    (0..want).map_while(|_| to_run_iter.next()).collect();
                let created_any = !batch.is_empty();
                let convs = if created_any {
                    engine.new_conversations_with_projection_batch(
                        batch.len(),
                        &calib_prelude,
                        &calib_builder,
                        calib_layer,
                        calib_group,
                        &calib_config,
                    )
                } else {
                    Vec::new()
                };
                for ((name, example, idx), conv_res) in batch.into_iter().zip(convs) {
                    let mut conv = match conv_res {
                        Ok(conv) => conv,
                        Err(e) => {
                            tracing::warn!(tool = name, "calibration create failed: {e}");
                            done += 1;
                            progress.set_step_progress(done as u64, total as u64);
                            continue;
                        }
                    };
                    // Tag at creation so a half-finished case is findable next load.
                    let marker = format!("{name}|{example}");
                    if let Err(e) = engine.set_conversation_metadata(
                        conv.timeline_id(),
                        CALIB_MARKER_KEY,
                        &marker,
                    ) {
                        tracing::warn!(tool = name, "calibration tag failed: {e}");
                    }
                    let mut opts = candle_conversation::TurnOptions {
                        max_tokens: Some(CALIBRATION_MAX_TOKENS),
                        // Gather-scope tags: `"tool"` puts this turn in the tools
                        // collection's belief gallery; the tool-name tag labels it
                        // directly, so the seed corpus needs no cold-start bootstrap.
                        tags: vec!["tool".to_string(), name.to_string()],
                        ..Default::default()
                    };
                    // Pin exactly this tool: `SelectionRule::Named` emits only the
                    // catalog member whose name matches the selector value.
                    opts.selection
                        .select(crate::tools::CALIB_TOOL_SELECTOR, name);
                    // Fast path: prefill the exported trajectory verbatim in one
                    // batched forward pass instead of decoding it token by token —
                    // the wide-Q is captured identically at seal. The body is
                    // everything after the assistant header (markers stripped). If
                    // the case has no exported trajectory, or a malformed one where
                    // the header isn't found, fall back to a live decode rather than
                    // prefilling a wrong-grid body.
                    // Keep the projection markers in the body: `submit_prefilled_turn`
                    // strips them for the prefilled text and records each one's token
                    // offset so the staged prefill wave fires a projection there,
                    // reproducing the decode's per-segment projection sequence.
                    let body =
                        zend_tools::calibration::trajectory_for(name, idx).and_then(|traj| {
                            traj.split_once(assistant_start).map(|(_, b)| b.to_string())
                        });
                    let (submit_result, is_prefill) = match body {
                        Some(body) => (
                            conv.submit_prefilled_turn(
                                example,
                                &body,
                                zend_tools::calibration::PROJECTION_MARKER,
                                opts.selection.clone(),
                                opts.tags.clone(),
                            ),
                            true,
                        ),
                        None => (conv.submit_turn_with_options(example, opts), false),
                    };
                    match submit_result {
                        Ok(handle) => {
                            if !warmed {
                                // Warm-up: the first case decodes alone so the shared
                                // tool sections upload once before concurrency.
                                warmed = true;
                                // Drain the event stream (blocking) so the warm-up
                                // case collects its mid-decode reprojection events too.
                                let mut events: Vec<ProjectionEvent> = Vec::new();
                                let mut done_resp: Option<TurnResponse> = None;
                                for ev in handle.stream() {
                                    match ev {
                                        TurnEvent::Projection(e) => events.push(e),
                                        TurnEvent::Done(resp) => {
                                            done_resp = Some(resp);
                                            break;
                                        }
                                        TurnEvent::Error(e) => {
                                            tracing::warn!(
                                                tool = name,
                                                "calibration case failed: {e}"
                                            );
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                                match done_resp {
                                    Some(resp) if is_prefill || resp.text.contains("</tool_call>") => {
                                        seal_calibration_turn(&mut conv, handle, &resp, name, events);
                                        if let Err(e) = engine
                                            .set_conversation_archived(conv.timeline_id(), true)
                                        {
                                            tracing::warn!(tool = name, "calibration archive failed: {e}");
                                        }
                                        // Sig captured — mark for distillation so
                                        // compaction sheds the trajectory content.
                                        mark_calibration_distill(&engine, conv.timeline_id(), name);
                                        // Queue for hot→warm demotion (see the retire
                                        // path) so the warm-up case's K/V is reclaimed
                                        // like every other case's.
                                        calib_timelines.push(conv.timeline_id());
                                    }
                                    Some(_) => tracing::warn!(
                                        tool = name,
                                        "calibration trajectory incomplete — left un-archived for retry"
                                    ),
                                    None => {}
                                }
                                done += 1;
                                progress.set_step_progress(done as u64, total as u64);
                            } else {
                                inflight.push((conv, handle, name, Vec::new(), is_prefill));
                            }
                        }
                        Err(e) => {
                            tracing::warn!(tool = name, "calibration submit failed: {e}");
                            done += 1;
                            progress.set_step_progress(done as u64, total as u64);
                        }
                    }
                }
                // Terminate only when nothing is in flight AND nothing remained
                // to submit this pass (`created_any` false ⇒ `to_run` is drained).
                // After the warm-up case (which leaves `inflight` empty but
                // `created_any` true) this correctly continues to the concurrent
                // window instead of breaking early.
                if inflight.is_empty() && !created_any {
                    break;
                }
                // Retire finished cases; if none finished this pass, yield briefly
                // (the unbounded channels buffer, so this never starves the wave).
                if !retire_completed(&mut inflight, &mut done, &mut calib_timelines) {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                // Incremental hot→warm demotion: once a full window's worth of
                // cases has retired since the last sweep, drop the hot K/V of just
                // the fresh tail (`[reclaimed_up_to..]`). No flush — the
                // persistence thread has already migrated the older cases to warm,
                // so they qualify; any not-yet-warm straggler is caught by the
                // flushing boundary sweep below (which re-demotes the whole list).
                // The call is fire-and-forget, so case submission never stalls.
                if calib_timelines.len() - reclaimed_up_to >= CALIBRATION_BATCH {
                    if let Err(e) =
                        engine.demote_timelines_hot(&calib_timelines[reclaimed_up_to..], false)
                    {
                        tracing::warn!("calibration hot→warm demote failed: {e}");
                    }
                    reclaimed_up_to = calib_timelines.len();
                }
            }
            // Boundary sweep: flush the pending hot→warm migration so the final
            // window's just-sealed cases are warm-backed, then demote every
            // calibration timeline. This reclaims the tail before the repo-scan
            // phase's first prefill, which would otherwise hit a card still full
            // of the calibration corpus's hot K/V.
            if let Err(e) = engine.demote_timelines_hot(&calib_timelines, true) {
                tracing::warn!("calibration boundary hot→warm demote failed: {e}");
            }
            progress.set_step_progress(total as u64, total as u64);
            tracing::info!(
                cases = total,
                demoted_timelines = calib_timelines.len(),
                "calibrating sections complete"
            );
        }

        // Walk the workspace, cluster the directory tree under a
        // token budget, and prefill one (user, assistant) turn pair
        // per cluster into the `repo_map` layer.  `ClusterState`
        // is the per-cluster file-name hash record `refresh_repo_map`
        // consults so a filesystem event triggers a rebuild only
        // when something actually changed.
        progress.set_step(LoadStep::RepoScan);
        let (repo_map_sequence, repo_map, repo_map_state) = crate::repo_scan::ingest_repo_map(
            &engine,
            proj_builder_repo_map,
            &workspace,
            conv_config.clone(),
            &progress,
            skip_repo_scan,
        )?;

        // Wrap the engine in its session Mutex now: code_read's per-file
        // pool takes `&Mutex<ConversationEngine>` so it can hold the lock
        // only for the quick create/tombstone ops and release it across the
        // decode-heavy summary generation (keeping refresh non-blocking).
        let engine = Mutex::new(engine);

        // For every file the walker surfaced, carve into scope-aware parts
        // and build a one-conversation-per-file `code_reading` ingest (one
        // prefill turn per part + a final whole-file summary).
        progress.set_step(LoadStep::CodeRead);
        let code_read_state = if skip_code_read {
            tracing::info!("--skip-code-read: bypassing per-file code-reading ingest");
            CodeReadState::default()
        } else {
            crate::code_read::ingest_code_reading(
                &engine,
                proj_builder_code_read,
                &workspace,
                &repo_map,
                conv_config.clone(),
                &progress,
            )?
        };

        // The titler runs on a single dedicated worker thread fed by this
        // queue. The worker owns the titler `Sequence` exclusively (no shared
        // mutex, so title generation never serialises against the request
        // path), and is joined on shutdown so its in-flight turn unwinds.
        let (titler_tx, titler_rx) = sync_channel(TITLER_QUEUE_DEPTH);
        // Build the three per-tools-mode projection builders once, up front, so
        // each turn only pays a cheap `Arc` clone instead of re-cloning the
        // ~93-section schema (see `ModeBuilders`).
        let mode_builders =
            ModeBuilders::build(&proj_builder_refresh, &crate::tools::safe_tool_names())
                .map_err(|e| anyhow::anyhow!("tools-mode projection builders: {e}"))?;
        let think_closer_phrase = tokenizer
            .encode(THINK_CLOSER_PHRASE, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        let state = Arc::new(Self {
            decoder,
            engine,
            conversations: Mutex::new(HashMap::new()),
            base_conv: Mutex::new(base_conv),
            titler_tx,
            titler_worker: Mutex::new(None),
            titler_timeline,
            shutting_down: AtomicBool::new(false),
            repo_map_conv: Mutex::new(RepoMapConv {
                sequence: repo_map_sequence,
                state: repo_map_state,
            }),
            code_read_conv: Mutex::new(CodeReadConv {
                state: code_read_state,
            }),
            refresh_builder: proj_builder_refresh,
            refresh_config: conv_config.clone(),
            mode_builders,
            workspace,
            think_closer_phrase,
            tokenizer,
            tool_host: ToolHost::new(),
            tool_stencil,
            think_steering,
        });
        let worker_state = Arc::clone(&state);
        let worker =
            std::thread::spawn(move || titler_worker_loop(worker_state, titler, titler_rx));
        *state.titler_worker.lock().unwrap() = Some(worker);
        Ok(state)
    }

    /// Build a [`RefreshContext`] bound to this state's engine,
    /// projection schema, and dialect config.  Cheap (the schema is
    /// `Arc`-backed) and avoids duplicating the construction in
    /// both refresh paths.
    fn refresh_ctx(&self) -> RefreshContext<'_> {
        RefreshContext {
            engine: &self.engine,
            proj_builder: self.refresh_builder.clone(),
            config: self.refresh_config.clone(),
        }
    }

    /// Atomic refresh of the `repo_map` conversation when any
    /// directory cluster's file-name hash changed.  Mints a fresh
    /// timeline, prefills the new clusters, tombstones the old
    /// timeline (so projection / compaction physically drop it),
    /// then swaps the new `Sequence` into `repo_map_conv`.  Returns
    /// `Ok(true)` when the conversation was replaced, `Ok(false)`
    /// when the hash check found nothing to do.
    ///
    /// Stale-better-than-missing: between the new timeline being
    /// registered (start of prefill) and the tombstone being
    /// written, both timelines are alive in the substrate.  The
    /// resolver's `active_timelines_for_group` iterator picks the
    /// older one (registered first), so dialogue retrieval against
    /// this layer keeps seeing the prior content the whole way
    /// through the refresh.  The swap-in is observed by the next
    /// query as a single atomic transition from old to new.
    ///
    /// When `map` is `Some`, the refresh reuses that workspace walk
    /// instead of doing its own — the watcher callback walks once
    /// per filesystem-event burst and shares the result with both
    /// refresh paths.
    pub(crate) fn refresh_repo_map(&self, map: Option<RepoMap>) -> anyhow::Result<bool> {
        // Throwaway progress sink (a refresh, not the model-load lifecycle) —
        // silent so a watcher burst doesn't log "load step started Loading model".
        let progress = LoadProgress::silent();
        let map = map.unwrap_or_else(|| crate::repo_scan::walk_workspace(&self.workspace));
        let (old_timeline, prior_state) = {
            let guard = self.repo_map_conv.lock().unwrap();
            (guard.sequence.timeline_id(), guard.state.clone())
        };
        let ctx = self.refresh_ctx();
        let outcome =
            crate::repo_scan::refresh_repo_map(&ctx, &map, &prior_state, old_timeline, &progress)?;
        match outcome {
            crate::repo_scan::RefreshOutcome::NoOp => Ok(false),
            crate::repo_scan::RefreshOutcome::Replaced { sequence, state } => {
                let mut guard = self.repo_map_conv.lock().unwrap();
                *guard = RepoMapConv { sequence, state };
                Ok(true)
            }
        }
    }

    /// Atomic refresh of the `code_reading` layer when any file's
    /// content hash changed.  Reconciles deleted files (tombstones
    /// their per-file conversations), then re-runs the parallel
    /// per-file ingestion — each changed file gets a fresh per-file
    /// conversation (per-part tool-call prefills closed by a decoded
    /// whole-file summary) while files whose hash is unchanged are
    /// skipped via the resume cache.  Stale-better-than-missing holds
    /// throughout: dialogue retrieval sees a file's old content until
    /// that file's new conversation tombstones the old one.
    ///
    /// When `map` is `Some`, the refresh reuses that workspace walk
    /// instead of doing its own.
    pub(crate) fn refresh_code_reading(&self, map: Option<RepoMap>) -> anyhow::Result<bool> {
        // Throwaway progress sink (a refresh, not the model-load lifecycle) —
        // silent so a watcher burst doesn't log "load step started Loading model".
        self.refresh_code_reading_with_progress(map, &Arc::new(LoadProgress::silent()))
    }

    /// Ingest **only** the given workspace-relative files into the
    /// `code_reading` layer — the upload's read_file phase. Bounded to those
    /// files (never a whole-workspace re-ingest, never a tombstone sweep), so
    /// it's safe under `--skip-code-read` and can't overload the model with the
    /// entire repo. Merges the ingested files' content hashes into the running
    /// `CodeReadState` so the resume cache and future watcher refreshes skip
    /// them. Reports per-scope progress into `progress`. Returns
    /// `(ingested, failed)`: whether anything was newly ingested, and whether
    /// at least one file's ingest tolerated-failed (e.g. the GPU ran out of KV
    /// VRAM) — so the upload can report a real failure instead of a silent
    /// no-op.
    pub(crate) fn ingest_uploaded_files(
        &self,
        rel_paths: &[String],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let ctx = self.refresh_ctx();
        let (state, n_failed) = crate::code_read::ingest_files(
            &self.engine,
            &ctx.proj_builder,
            &self.workspace,
            rel_paths,
            ctx.config,
            progress,
        )?;
        let failed = n_failed > 0;
        if state.file_hashes.is_empty() {
            return Ok((false, failed));
        }
        let mut guard = self.code_read_conv.lock().unwrap();
        for (path, hash) in state.file_hashes {
            guard.state.file_hashes.insert(path, hash);
        }
        Ok((true, failed))
    }

    /// As [`Self::refresh_code_reading`], but reports per-scope ingest
    /// progress into the caller's `progress` handle — the upload's read_file
    /// phase shares one and polls it to stream a progress bar to the GUI.
    pub(crate) fn refresh_code_reading_with_progress(
        &self,
        map: Option<RepoMap>,
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<bool> {
        let map = map.unwrap_or_else(|| crate::repo_scan::walk_workspace(&self.workspace));
        let prior_state = self.code_read_conv.lock().unwrap().state.clone();
        let ctx = self.refresh_ctx();
        let outcome = crate::code_read::refresh_code_reading(
            &ctx,
            &self.workspace,
            &map,
            &prior_state,
            progress,
        )?;
        match outcome {
            crate::code_read::RefreshOutcome::NoOp => Ok(false),
            crate::code_read::RefreshOutcome::Replaced { state } => {
                let mut guard = self.code_read_conv.lock().unwrap();
                *guard = CodeReadConv { state };
                Ok(true)
            }
        }
    }

    /// Tombstone the per-file `code_read` conversation of every uploaded file
    /// that has since been deleted from the `uploads/` folder.
    ///
    /// Uploads are endpoint-managed and deliberately excluded from the workspace
    /// walk (so `refresh_code_reading`'s `reconcile_deleted` never retires them —
    /// that would delete freshly-uploaded content on the next refresh). But a
    /// genuine deletion of an uploaded file — whether from the filesystem or via
    /// the GUI — must still retire its conversation; otherwise deleted uploads
    /// accumulate live turns forever, growing the redo log and the projection
    /// candidate set without bound.
    ///
    /// Cheap and self-limiting: a metadata scan plus one `Path::exists` probe per
    /// upload conversation, no workspace walk and no re-ingest. Tombstoning only
    /// *absent* files makes it a no-op for still-present uploads, so the watcher
    /// can fire it on any `uploads/` event (create / modify / delete) — an
    /// in-flight upload's own create events simply match nothing.
    ///
    /// Runs at startup (to retire uploads deleted while the daemon was down) and
    /// on every `uploads/` watcher burst.
    pub(crate) fn reconcile_uploaded_files(&self) {
        let engine = self.engine.lock().unwrap();
        let mut tombstoned = 0usize;
        for (tl, path) in engine.conversations_with_metadata_key("path") {
            if !crate::code_read::is_upload_path(&path) {
                continue;
            }
            // `path` is workspace-relative with `/` separators; `Path::join`
            // accepts them on win32, and `exists()` is case-insensitive there.
            if self.workspace.join(&path).exists() {
                continue;
            }
            match engine.tombstone_timeline(tl) {
                Ok(()) => {
                    tombstoned += 1;
                    tracing::info!(
                        target: "zend::session",
                        path = %path,
                        "reconcile_uploaded_files: tombstoned deleted upload's conversation",
                    );
                }
                Err(err) => tracing::warn!(
                    target: "zend::session",
                    path = %path,
                    "reconcile_uploaded_files: tombstone failed: {err:#}",
                ),
            }
        }
        if tombstoned > 0 {
            tracing::info!(
                target: "zend::session",
                count = tombstoned,
                "reconcile_uploaded_files: retired deleted uploads",
            );
        }
    }
}

/// Run a complete user request: stream tokens to the client as they
/// arrive, then on turn completion scan the response text for tool
/// calls.  If calls are found, dispatch them and loop with a
/// `<tool_response>` user turn; otherwise the turn is the final answer.
///
/// Every turn's tokens are streamed to the client — including tool-call
/// turns.  `<tool_call>` markup appears at the tail of those responses
/// so the user sees the natural-language prefix streamed live and the
/// tool markup appear at the end before the follow-up response begins.
/// Build the projection for one tools mode by cloning `base` and filtering the
/// `tools` collection MEMBERS: `Comprehensive` is the base unchanged (full
/// catalog); `Restricted` retains only the safe (non-high-risk) tool sections;
/// `None` retains none.  These builders control only WHICH members project — the
/// WHOLE tool block (markers, catalog, summary) is gated separately by the
/// `tools_enabled` optional_group, which chat.rs sets `absent` for `None`.  The
/// `tool_summary` overview is the comprehensive one for every mode (sealed once on
/// the base builder).  Called once per mode at startup by [`ModeBuilders::build`];
/// the results are cached and handed out as cheap `Arc` clones each turn, since
/// both projection and reprojection just read the swapped builder.
fn build_mode_builder(
    base: &Builder,
    safe_tool_names: &HashSet<String>,
    mode: ToolMode,
) -> anyhow::Result<Arc<Builder>> {
    let mut b = base.clone();
    let dialogue = b
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
    match mode {
        ToolMode::Comprehensive => {}
        ToolMode::Restricted => {
            // Drop the high-risk tools; the summary association below points this
            // mode at the restricted catalog listing.
            b.retain_collection_sections(dialogue, "tools", safe_tool_names)
                .map_err(|e| anyhow::anyhow!("restricted tools projection: {e}"))?;
        }
        ToolMode::None => {
            b.retain_collection_sections(dialogue, "tools", &HashSet::new())
                .map_err(|e| anyhow::anyhow!("none tools projection: {e}"))?;
        }
    }
    // Associate the sealed tool-catalog summary (built by `build_tool_summary`,
    // prefilled into its reserved section at startup) with the `tools` collection,
    // so projection emits the FULL tool-name listing just before the selected
    // subset. Without this the catalog K/V is sealed but orphaned — the model only
    // ever sees the 1-3 provenance-selected tools and concludes a tool it can't see
    // (e.g. `datetime` on a "what is the time?" turn) doesn't exist. The projection
    // emits it whenever the selection is a proper subset (§ `record` in
    // `emit_system_prompt_items`), which for a 93-tool catalog is every turn.
    // `None` mode has no tools block, so no summary.
    let summary = match mode {
        ToolMode::Comprehensive => Some(Reserved::ToolSummary),
        ToolMode::Restricted => Some(Reserved::ToolSummaryRestricted),
        ToolMode::None => None,
    };
    if let Some(reserved) = summary {
        let tools = b.id_for_collection_in(dialogue, "tools").ok_or_else(|| {
            anyhow::anyhow!("projection schema missing 'tools' collection in dialogue layer")
        })?;
        b.set_collection_summary_section(dialogue, tools, SectionId::reserved(reserved))
            .map_err(|e| anyhow::anyhow!("tool summary association ({mode:?}): {e}"))?;
    }
    Ok(Arc::new(b))
}

/// The three per-tools-mode projection builders, built once at startup so each
/// turn hands out a cheap `Arc` clone instead of re-cloning the ~93-section
/// schema. Restricted / None fall back to the comprehensive builder if their
/// section-retain fails, so the daemon still starts; Comprehensive is fatal (see
/// [`ModeBuilders::build`]).
struct ModeBuilders {
    none: Arc<Builder>,
    restricted: Arc<Builder>,
    comprehensive: Arc<Builder>,
}

impl ModeBuilders {
    fn build(base: &Builder, safe_tool_names: &HashSet<String>) -> anyhow::Result<Self> {
        // Comprehensive is the default mode and has no better fallback than itself:
        // a `base.clone()` here would silently re-orphan the tool-catalog summary
        // (the exact bug `build_mode_builder`'s association fixes), so a failure —
        // only reachable via a missing dialogue layer / `tools` collection, i.e. a
        // broken schema the daemon can't serve anyway — is fatal.
        let comprehensive = build_mode_builder(base, safe_tool_names, ToolMode::Comprehensive)?;
        let restricted = match build_mode_builder(base, safe_tool_names, ToolMode::Restricted) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("restricted tools projection build failed, using full catalog: {e}");
                Arc::clone(&comprehensive)
            }
        };
        let none = match build_mode_builder(base, safe_tool_names, ToolMode::None) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("none tools projection build failed, using full catalog: {e}");
                Arc::clone(&comprehensive)
            }
        };
        Ok(Self {
            none,
            restricted,
            comprehensive,
        })
    }

    /// The prebuilt projection for `mode` (a cheap `Arc` clone).
    fn get(&self, mode: ToolMode) -> Arc<Builder> {
        match mode {
            ToolMode::None => Arc::clone(&self.none),
            ToolMode::Restricted => Arc::clone(&self.restricted),
            ToolMode::Comprehensive => Arc::clone(&self.comprehensive),
        }
    }
}

/// Clone the live projection builder with a high-resolution capture override.
///
/// `spec` selects what to force:
/// - `"tools"` — force the whole `tools` collection to AllVisible (all sections);
/// - `"tools/datetime"` — emit ONLY the `datetime` section of `tools`.
///
/// The clone shares all section ids with the base, so a forked sequence's
/// already-sealed section KV is reused unchanged.
fn build_hires_projection(state: &InferenceState, spec: &str) -> anyhow::Result<Arc<Builder>> {
    let mut b = state.refresh_builder.clone();
    let dialogue = b
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
    if let Some((collection, section)) = spec.split_once('/') {
        b.set_collection_single_section(dialogue, collection, section)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    } else {
        b.set_collection_selection(dialogue, spec, SelectionRule::AlwaysVisible)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    }
    Ok(Arc::new(b))
}

fn run_inference_stream(
    state: Arc<InferenceState>,
    conv_id: String,
    user_message: String,
    max_tokens: Option<usize>,
    sampling: Option<candle_conversation::SamplingConfig>,
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    lossless_kv: bool,
    tools_mode: ToolMode,
    selection: candle_conversation::SelectionState,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);

    tokio::task::spawn_blocking(move || {
        let msg_preview: String = user_message.chars().take(60).collect();
        tracing::info!(
            conv_id = %conv_id,
            msg_len = user_message.len(),
            "inference start  \"{}{}\"",
            msg_preview,
            if user_message.len() > 60 { "…" } else { "" },
        );

        let timeline = timeline_for(&conv_id);
        let conv_arc: Arc<Mutex<ConvState>> = {
            let mut map = state.conversations.lock().unwrap();
            if let Some(existing) = map.get(&conv_id) {
                tracing::debug!(conv_id = %conv_id, "reusing existing conv");
                Arc::clone(existing)
            } else {
                tracing::info!(conv_id = %conv_id, "forking new conv from base");
                // Fork onto the timeline derived from `conv_id` — a stable
                // hash, so a daemon restart reconnects the client to the
                // turns the substrate reload recovered for this conversation
                // (§16.12). An unknown conv_id simply forks empty.
                let conv = match state.base_conv.lock().unwrap().fork_resuming(timeline) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!(conv_id = %conv_id, "fork failed: {e}");
                        let _ = tx.blocking_send(Err(anyhow::anyhow!("{e}")));
                        return;
                    }
                };
                let arc = Arc::new(Mutex::new(ConvState {
                    conv,
                    tool_mode: ToolMode::default(),
                }));
                map.insert(conv_id.clone(), Arc::clone(&arc));
                arc
            }
        };

        // Persist the conv_id ↔ timeline mapping *after* `fork_resuming`
        // has registered the timeline in the substrate — otherwise
        // `set_conv_id` no-ops in-RAM and the sidebar wouldn't see this
        // conversation until the next daemon restart. Idempotent on
        // repeat calls (no-op when the substrate already has it).
        if let Err(e) = state
            .engine
            .lock()
            .unwrap()
            .set_conversation_conv_id(timeline, &conv_id)
        {
            tracing::warn!(conv_id = %conv_id, "persist conv_id failed: {e}");
        }

        // Lossless capture: seal this conversation's turns WITHOUT KV
        // quantization — K/V persist in native R16/F16 so the provenance work
        // gets full-resolution keys. Set before the first turn's residence is
        // allocated so the hot→warm migration inherits it.
        if lossless_kv {
            state.engine.lock().unwrap().set_timeline_compression(
                timeline,
                Some(candle_conversation::substrate::ConvCompression {
                    lossless: true,
                    level: None,
                    disable_k_override: false,
                    force_k: None,
                    force_v: None,
                }),
            );
        }

        // Hand the titler off to its worker, now that the timeline is
        // registered and the conv_id is in-RAM (so the titler's
        // `set_conversation_label` write composes correctly with the conv_id
        // field of the same Label record). The decode runs in the background,
        // batched alongside the main decode. Enqueue is non-blocking: a full
        // queue or in-progress shutdown drops the job (labels are best-effort).
        let already_labelled = state
            .engine
            .lock()
            .unwrap()
            .conversation_label_of(timeline)
            .is_some();
        if !already_labelled
            && !user_message.is_empty()
            && !state.shutting_down.load(Ordering::Relaxed)
        {
            let job = TitleJob::Title {
                timeline,
                message: user_message.clone(),
            };
            if let Err(e) = state.titler_tx.try_send(job) {
                tracing::debug!("titler queue full or closed — skipping label: {e}");
            }
        }

        let mut cs = conv_arc.lock().unwrap();

        // Per-conversation projection swap. Both the prefill projection and the
        // reprojection read the swapped builder, so neither re-introduces a
        // filtered section mid-conversation.
        //
        // `force_hires` is the zend capture aid (force one collection to
        // AllVisible) and wins when present; otherwise apply the composer's
        // tools mode — Comprehensive restores the full catalog (so switching a
        // conversation back from Restricted/None works), Restricted drops the
        // high-risk tools, None drops the whole catalog.
        // `cs.tool_mode` records the mode actually applied, so the projection
        // panel shows the matching tool summary.
        if let Some(ref coll) = force_hires {
            match build_hires_projection(&state, coll) {
                Ok(b) => {
                    cs.conv.set_projection(b);
                    // The capture override forces the full catalog (AllVisible),
                    // so the panel should show the comprehensive summary.
                    cs.tool_mode = ToolMode::Comprehensive;
                    tracing::info!(conv_id = %conv_id, collection = %coll,
                        "force-high-resolution: collection forced to AllVisible");
                }
                Err(e) => {
                    tracing::warn!(conv_id = %conv_id, "force-high-resolution failed: {e}")
                }
            }
        } else {
            // Cheap `Arc` clone of the prebuilt mode projection (no per-turn
            // schema clone). Applied every turn so a mid-conversation dial change
            // takes effect, and both projection and reprojection use it.
            cs.conv.set_projection(state.mode_builders.get(tools_mode));
            cs.tool_mode = tools_mode;
            tracing::info!(conv_id = %conv_id, ?tools_mode, "tools mode applied");
        }

        let original_user_message = user_message.clone();
        let mut current_message = user_message;

        // The reflection-marker suppression ceiling is per-dial: derive the think
        // mode once, materialise the turn's sampling config (the conversation
        // default carries the resolved marker family), and bake in the penalty —
        // Quick/Balanced suppress the "Wait"/"Hmm"/… family in-block, Deep/
        // Exhaustive leave it 0 (reconsideration is wanted there).
        let think_mode = think_mode_from_selection(&selection);
        // An explicit caller-supplied config (e.g. a test's `argmax()`) is honoured
        // verbatim; only the conversation default gets the per-turn adjustments
        // below (thinking-vs-response sampling split + a fresh seed).
        let sampling_defaulted = sampling.is_none();
        let mut sampling = sampling.unwrap_or_else(|| cs.conv.default_sampling());
        sampling.segment_suppress_penalty = think_mode.suppress_penalty();
        if sampling_defaulted {
            // A `/no_think` turn is a direct response, not reasoning: strip the
            // thinking-temperature boost, which is meant for the `<think>` span
            // and has no business heating a plain answer.  DRY stays on: it is now
            // span-scoped (`dry_span_len` — it only ever sees the current prose
            // span, never the prompt or a prior span), so it breaks answer loops
            // without the old full-window DRY's failure of penalizing verbatim
            // reproduction of numbers/identifiers lifted from the prompt.  (It can
            // still nip content the model itself repeats WITHIN the current answer
            // — e.g. a long list's `- ` scaffolding — which is a penalty-tuning
            // question, not a scoping one.)  That span scoping is why disabling it
            // here is no longer necessary.
            if think_mode == ThinkMode::Off {
                sampling.segment_temp_boost = 0.0;
            }
            // Vary the RNG seed per turn from real entropy. The default base seed
            // is a fixed constant, which makes a whole conversation a deterministic
            // replay — the same context always samples the same tokens, so a turn
            // that lands in a bad attractor can never sample its way out. A fresh
            // per-turn seed restores genuine run-to-run variation. (The per-token
            // `rng_offset` still advances within the turn.)
            sampling.seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(sampling.seed);
        }
        // Per-dial thinking budget: the EOT close ramp's graceful/force thresholds
        // scale with the effort level (exhaustive thinks longest).  `segment_len`
        // restarts each steered span, so these are per-span — higher dials get more
        // room per span and, via more spans, far more total.
        let (graceful_eot, force_eot) = think_mode.eot_budget();
        sampling.graceful_segment_close_after = graceful_eot;
        sampling.force_segment_close_after = force_eot;
        // The per-span close boost ramps `</think>`+EOS over this dial's
        // [graceful, force] thinking-token window (resets each span), so it builds
        // pressure into the same point the force override hard-closes — and scales
        // with the dial instead of a fixed global ramp that misses the short dials.
        sampling.segment_close_ramp_start = graceful_eot;
        sampling.segment_close_ramp_len = force_eot;
        // The EOS (turn-ender) budget is the whole-turn backstop on total length,
        // derived from BOTH dials: the think budget (spans × per-span cap) fixes
        // where the answer starts, so the ramp begins as the think block ends and is
        // dormant during reasoning (the per-span EOT/EOS boost handles that); the
        // `response_length` dial sets the answer room above it.  So it can't truncate
        // the thinking budget, and it scales with both knobs.  (Keeps the preset's
        // eos_boost magnitude/mult; the boost ramps to the graceful threshold.)
        let response_tokens = response_budget_from_selection(&selection);
        let (eos_ramp_start, graceful_eos, forced_eos) = think_mode.eos_budget(response_tokens);
        sampling.eos_ramp_start = eos_ramp_start;
        sampling.eos_ramp_len = graceful_eos;
        sampling.graceful_eos_after = graceful_eos;
        sampling.forced_eos_after = forced_eos;
        // Hard-cap closer: when the per-span force budget amputates the think
        // block mid-sentence, the sampler plays this phrase and then closes
        // the block itself, so the reasoning ends as intentional prose with an
        // explicit commitment. All dials; the sampler skips it in continuation
        // spans (deep/exhaustive "But wait" retirement) where more reasoning
        // follows, and at completed sentences, which need no rescue.
        sampling.segment_close_script = state.think_closer_phrase.clone();

        // The tool loop runs until the model stops emitting tool calls (i.e.
        // produces a final answer) — there is no fixed iteration cap. A wedged
        // model that never stops calling tools is bounded only by the client
        // disconnecting (handled below) or the conversation being evicted, not
        // by cutting the workflow short at an arbitrary count.
        for iteration in 0.. {
            tracing::debug!(conv_id = %conv_id, iteration, "submitting turn");
            // Collect this turn's projection events (reprojections + decode-end)
            // so they survive a browser reload (served back on hydrate).
            let mut turn_events: Vec<ProjectionEventOut> = Vec::new();
            let options = candle_conversation::TurnOptions {
                max_tokens,
                sampling: Some(sampling.clone()),
                // Apply the caller's assistant prefill only on the first tool
                // iteration — re-prefilling it on every chained iteration would
                // prevent the model ever reaching a final answer.
                assistant_prefill: if iteration == 0 {
                    assistant_prefill.clone()
                } else {
                    None
                },
                selection: selection.clone(),
                // Force any `<tool_call>` the model emits to the catalog's exact
                // JSON shape (name ∈ catalog, required params in order, valid
                // JSON), and — atop that base — steer the `<think>` block per the
                // effort dial (replacing the `<think>` trigger atomically).  An
                // empty registry would just free-decode.
                triggers: match &state.think_steering {
                    Some(ts) => ts.registry_for(&state.tool_stencil, think_mode),
                    None => Arc::clone(&state.tool_stencil),
                },
                ..Default::default()
            };
            let handle = match cs.conv.submit_turn_with_options(&current_message, options) {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "submit_turn failed: {e}");
                    let _ = tx.blocking_send(Err(anyhow::anyhow!("{e}")));
                    return;
                }
            };

            // Stream tokens to the client as they arrive.  We track emitted_len
            // (byte offset into the full decoded string) so incremental deltas
            // are correct across BPE byte-fallback sequences.  Fragments that
            // still contain U+FFFD are held back until the sequence completes.
            let mut tokens: Vec<u32> = Vec::new();
            let mut emitted_len: usize = 0;
            // Byte offset where the answer's tool-call object begins: once set, the
            // object is held back (not streamed live) and re-emitted wrapped in
            // <tool_call> tags at the flush, so the GUI renders a card instead of
            // showing the bare JSON the model emits when it drops the wrapper.
            let mut hold_from: Option<usize> = None;
            let mut done_resp = None;
            let mut turn_error: Option<anyhow::Error> = None;
            let mut client_gone = false;

            for event in handle.stream() {
                match event {
                    TurnEvent::Token(id) => {
                        if turn_error.is_none() {
                            tokens.push(id);
                            let text = state.decoder.decode(&tokens);
                            // Once the post-</think> answer opens with `{`, it's a
                            // (usually un-tagged) tool-call object: stop streaming
                            // here and hold the rest, so the flush can wrap it.
                            if hold_from.is_none() {
                                let answer = text
                                    .rfind("</think>")
                                    .map(|i| i + "</think>".len())
                                    .unwrap_or(0);
                                let rest = text[answer..].trim_start();
                                if rest.starts_with('{') {
                                    hold_from = Some(text.len() - rest.len());
                                }
                            }
                            let emit_to = hold_from.unwrap_or(text.len());
                            if emit_to > emitted_len
                                && text.is_char_boundary(emitted_len)
                                && text.is_char_boundary(emit_to)
                            {
                                let new_part = &text[emitted_len..emit_to];
                                if !new_part.contains('\u{FFFD}') {
                                    if tx
                                        .blocking_send(Ok(StreamItem::Token(new_part.to_string())))
                                        .is_err()
                                    {
                                        // Client closed the connection.  Break
                                        // immediately so `handle` is dropped on
                                        // return, which closes event_rx and causes
                                        // the scheduler's next send to fail →
                                        // state.finished = true → decode stops.
                                        client_gone = true;
                                        break;
                                    }
                                    emitted_len = emit_to;
                                }
                            }
                        }
                    }
                    TurnEvent::Done(resp) => {
                        tracing::info!(
                            conv_id = %conv_id,
                            iteration,
                            tokens = resp.stats.tokens_generated,
                            tps    = resp.stats.tokens_per_second as u32,
                            prefill_ms = resp.stats.prefill_ms as u32,
                            "turn complete",
                        );
                        done_resp = Some(resp);
                    }
                    TurnEvent::Error(e) => {
                        let msg = format!("{e}");
                        tracing::error!(conv_id = %conv_id, iteration, "scheduler error: {msg}");
                        // Send as text so the client shows the message rather
                        // than dropping the connection.
                        let _ = tx.blocking_send(Ok(StreamItem::Token(format!("\n\n⚠ {msg}"))));
                        turn_error = Some(anyhow::anyhow!("{msg}"));
                        // Do not return — drain the iterator so the channel
                        // closes cleanly before we decide what to do with the
                        // conversation state.
                    }
                    TurnEvent::HealthWarning(msg) => {
                        tracing::warn!(conv_id = %conv_id, "decode health: {msg}");
                    }
                    // A mid-decode reprojection — stream it straight to the GUI
                    // timeline as it happens (a dot per reprojection).
                    TurnEvent::Projection(event) => {
                        let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let out = ProjectionEventOut::answer(seq, event);
                        turn_events.push(out.clone());
                        let _ = tx.blocking_send(Ok(StreamItem::Projection(out)));
                    }
                    _ => {}
                }
            }

            let resp = match done_resp {
                Some(r) => r,
                None => {
                    if client_gone {
                        // Normal disconnect — no error to report.
                        tracing::info!(
                            conv_id = %conv_id,
                            iteration,
                            "client disconnected mid-stream — cancelling decode",
                        );
                    } else if turn_error.is_none() {
                        // Scheduler closed the channel without Done — the
                        // conversation's turn_in_flight flag is stuck.
                        tracing::error!(
                            conv_id = %conv_id,
                            iteration,
                            "turn ended without Done or Error — evicting conversation",
                        );
                        let _ = tx.blocking_send(Ok(StreamItem::Token(
                            "\n\n⚠ Generation ended unexpectedly. Your next message will start fresh.".to_string()
                        )));
                    }
                    // Evict the conversation so the next request forks fresh
                    // rather than hitting the in-flight guard.  Dropping `handle`
                    // here (via return) closes event_rx and stops the scheduler.
                    drop(cs);
                    state.conversations.lock().unwrap().remove(&conv_id);
                    return;
                }
            };

            // Flush the tail.  If we held back the answer's tool-call object, emit
            // it wrapped in <tool_call> tags so the GUI renders a card (the model
            // dropped the wrapper); a held object that isn't a recognized call is
            // flushed verbatim by `render_held_tail`.  Otherwise flush the raw tail
            // the byte-fallback guard held back.
            if let Some(hf) = hold_from {
                if resp.text.is_char_boundary(hf) {
                    if let Some(out) = render_held_tail(&resp.text[hf..]) {
                        let _ = tx.blocking_send(Ok(StreamItem::Token(out)));
                    }
                }
            } else if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
                let tail = &resp.text[emitted_len..];
                if !tail.is_empty() {
                    let _ = tx.blocking_send(Ok(StreamItem::Token(tail.to_string())));
                }
            }

            let calls = extract_tool_calls(&resp.text);
            // Force-high-resolution is a capture mode: seal the first turn (the
            // tool invocation) into the substrate as the dataset baseline, but
            // do NOT execute the tools — capture-only, so `code_run` / network
            // tools have no real side effects.
            let is_final = calls.is_empty() || force_hires.is_some();

            // If tools will run, tell the GUI *now* — before sealing/persisting
            // this turn — so the in-flight tool cards show their spinner across
            // the whole execution window (seal + persist + dispatch), not just
            // the instant before dispatch (which an instant tool coalesces away,
            // making the card jump straight to a result).
            let tool_names: Vec<String> = if is_final {
                Vec::new()
            } else {
                calls.iter().map(|c| c.name.clone()).collect()
            };
            if !is_final {
                let _ = tx.blocking_send(Ok(StreamItem::Tool(ToolStatusOut {
                    phase: "running",
                    tools: tool_names.clone(),
                    results: Vec::new(),
                })));
            }

            if let Err(e) = cs.conv.finish_turn(handle, &resp) {
                tracing::warn!(conv_id = %conv_id, "finish_turn error: {e}");
            }

            // One projection event per decode: the scored reprojection now
            // reflects what provenance materialized for this turn (finish_turn
            // refreshed the BDP scores). Stream it so the GUI drops a timeline
            // dot the moment the turn seals — no separate fetch needed.
            if let Some(event) = cs.conv.projection_event(&resp.stats) {
                let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let out = ProjectionEventOut::answer(seq, event);
                turn_events.push(out.clone());
                let _ = tx.blocking_send(Ok(StreamItem::Projection(out)));
            }

            // Persist this turn's events to the substrate redo log so the
            // timeline survives a browser reload AND a daemon restart. Keyed to
            // the just-sealed turn; served back on hydrate via the substrate.
            if !turn_events.is_empty() {
                let events: Vec<candle_conversation::ProjectionEvent> =
                    turn_events.into_iter().map(|o| o.event).collect();
                if let Err(e) = cs.conv.persist_projection_events(&events) {
                    tracing::warn!(conv_id = %conv_id, "persist projection events: {e}");
                }
            }

            // Group-commit the substrate redo log: the just-sealed turn is
            // now durable on disk, so a crash or restart resumes it intact.
            if let Err(e) = state.engine.lock().unwrap().commit_persistence() {
                tracing::warn!(conv_id = %conv_id, "persistence commit error: {e}");
            }

            if is_final {
                break;
            }

            tracing::info!(
                conv_id = %conv_id,
                iteration,
                n_calls = calls.len(),
                "dispatching tool calls",
            );
            // The "running" notice was already sent above (before the seal).  Run
            // the tools — `run_tool_calls` blocks until every tool returns — then
            // the "done" notice clears the spinner and carries each result so the
            // cards resolve immediately, before the post-stream hydrate.
            let n_calls = calls.len();
            let results = run_tool_calls(&state.tool_host.ctx, calls);
            let _ = tx.blocking_send(Ok(StreamItem::Tool(ToolStatusOut {
                phase: "done",
                tools: tool_names,
                results: results.iter().map(|r| r.response.clone()).collect(),
            })));
            current_message = format_tool_responses(&results);
            // A tool round that produces no response text must NOT spawn a
            // follow-up turn. `format_tool_responses` wraps every real result in
            // `<tool_response>…</tool_response>`, so an empty `current_message`
            // means no tool actually ran (the model emitted a tool call that was
            // filtered/failed). Submitting it would seal a phantom turn whose grid
            // is nothing but boundary glue (no user content), which the model then
            // "answers" with a generic greeting — derailing the conversation. The
            // answer already streamed for this turn stands; end the loop here.
            if current_message.trim().is_empty() {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    n_calls,
                    "tool round yielded no response text; ending turn loop instead \
                     of submitting an empty follow-up turn",
                );
                break;
            }

            // The round-trip is now certain — the tools returned real output and
            // the follow-up turn is guaranteed to be submitted below — so couple
            // the just-sealed call turn to it. This is the only window where that
            // is knowable: at the call turn's seal the tools had not run, and by
            // the response turn's seal the call turn's record is already durable.
            // Couple the call turn by its OWN sealed index (`resp.seal.turn_index`),
            // never by "the last turn": the async summariser can append a summary
            // turn in this window, so the newest index may not be the call turn.
            if let Some(call_idx) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                if let Err(e) = cs.conv.couple_turn(call_idx) {
                    tracing::warn!(conv_id = %conv_id, "couple_turn: {e}");
                }
            } else {
                tracing::warn!(
                    conv_id = %conv_id,
                    "tool call turn sealed without an index — cannot couple its response"
                );
            }
        }

        // Title generation has already been fired in parallel from
        // `submit_with_sampling` — it doesn't depend on the main
        // convo's state, so we don't repeat it here.
        let _ = original_user_message;
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

/// The titler worker thread. Owns the titler [`Sequence`] exclusively and
/// drains title jobs one at a time off `rx`. Each title-gen overlaps the main
/// decode (the scheduler batches both sequences), but the worker never blocks
/// the request path and never piles up a thread per submit. Between jobs it
/// checks the shutdown flag, so once shutdown begins it stops draining queued
/// jobs immediately; on exit it abandons any in-flight turn so the sequence is
/// left clean.
fn titler_worker_loop(state: Arc<InferenceState>, mut titler: Sequence, rx: Receiver<TitleJob>) {
    while let Ok(job) = rx.recv() {
        if state.shutting_down.load(Ordering::Relaxed) {
            break;
        }
        match job {
            TitleJob::Shutdown => break,
            TitleJob::Title { timeline, message } => {
                generate_one_title(&state, &mut titler, timeline, &message);
            }
        }
    }
    // If shutdown interrupted a decode, the turn is still in flight — clear it
    // so the sequence (and its substrate timeline) unwinds cleanly.
    if titler.is_in_flight() {
        titler.abort_turn();
    }
}

/// Run the titler conversation against `user_message` and write the result as
/// the main conversation's sidebar label. Errors are logged and dropped — the
/// worst case is a missing sidebar label, never a failed response. Any turn
/// that doesn't reach `Done` (scheduler error, or shutdown mid-decode) is
/// aborted rather than left in flight, so the next title-gen can `reset`.
fn generate_one_title(
    state: &Arc<InferenceState>,
    titler: &mut Sequence,
    timeline: TimelineId,
    user_message: &str,
) {
    use candle_conversation::{TurnEvent, TurnOptions};

    let truncated = head_tail_truncate(
        user_message,
        &state.tokenizer,
        TITLER_HEAD_TOKENS,
        TITLER_TAIL_TOKENS,
    );
    // No `/no_think` baked into the user text: the selector below makes the
    // scheduler emit it as live glue (`no_think_current`) right after the user
    // opener — the same single mechanism the dialogue uses — so the only other
    // copy is in the system prompt. The titler must never reason, or it fills the
    // short budget with a `<think>` block and the stripped title comes back empty.

    // Clear the titler's in-memory turn tree so each title-gen starts from
    // just the system prompt (no accumulated history).
    if let Err(e) = titler.reset() {
        tracing::warn!("titler reset failed: {e}");
        return;
    }
    // The titler generates a short label — never reason. Suppress thinking
    // explicitly (the model thinks by default).
    let mut titler_sel = candle_conversation::SelectionState::new();
    titler_sel.set_optional(
        candle_conversation::NO_THINK_SELECTOR,
        candle_conversation::OptionalState::Present,
    );
    let opts = TurnOptions {
        max_tokens: Some(TITLER_MAX_TOKENS),
        selection: titler_sel,
        ..Default::default()
    };
    let handle = match titler.submit_turn_with_options(&truncated, opts) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("titler submit failed: {e}");
            return;
        }
    };
    let mut done = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Done(r) => {
                done = Some(r);
                break;
            }
            TurnEvent::Error(e) => {
                tracing::warn!("titler scheduler error: {e}");
                break;
            }
            _ => {}
        }
    }
    let Some(resp) = done else {
        // No response: scheduler error or shutdown mid-decode. Abandon the
        // turn so it doesn't wedge the sequence (the cause of the shutdown
        // "already has a turn in flight" reset loop).
        tracing::warn!("titler ended without a response");
        drop(handle);
        titler.abort_turn();
        return;
    };
    let title = clean_title(&resp.text);
    if let Err(e) = titler.finish_turn(handle, &resp) {
        tracing::warn!("titler finish_turn failed: {e}");
    }

    if title.is_empty() {
        tracing::warn!("titler produced an empty title — skipping label write");
        return;
    }
    // Don't touch the engine once shutdown has begun — it's being torn down.
    if state.shutting_down.load(Ordering::Relaxed) {
        return;
    }

    // Write the label via the engine's substrate-shared handle.
    let result = state
        .engine
        .lock()
        .unwrap()
        .set_conversation_label(timeline, &title);
    match result {
        Ok(()) => tracing::info!("conversation labelled: \"{title}\""),
        Err(e) => tracing::warn!("set_conversation_label failed: {e}"),
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct ZendSession {
    config: DaemonConfig,
    projection_builder: Builder,
    #[allow(dead_code)] // read by api/ws_logs.rs in the bin target
    pub(crate) log: Arc<LogBus>,
    /// Fires `true` once the daemon transitions to fully-ready (every
    /// load step complete). Used by submit-flow tasks waiting for the
    /// engine to come up.
    ready_tx: tokio::sync::watch::Sender<bool>,
    /// Current human-readable detail string for the active load step
    /// (e.g. `"Downloading shard 3/8"`). Surfaced as the loading
    /// section's `detail` field in `GET /v1/status`.
    pub(crate) status_tx: tokio::sync::watch::Sender<String>,
    /// Structured load-state machine — drives the frontend's loading
    /// overlay (current step + progress + completed list).
    load_progress: Arc<LoadProgress>,
    /// Wall-clock time (ms since Unix epoch) when this `ZendSession` was
    /// constructed — i.e. when the daemon process started. Surfaced via
    /// `GET /v1/status` so the frontend can detect daemon restarts and
    /// re-fetch the conversations list.
    started_at_ms: u64,
    /// Populated in the background after construction; None until model loads.
    inference: Arc<RwLock<Option<Arc<InferenceState>>>>,
    /// Workspace file-watcher.  Started after the model loads; held
    /// here so dropping the session also drops the watch.  None until
    /// the inference state is ready.
    watcher: Mutex<Option<RecommendedWatcher>>,
    /// Conversation-files store (uploads). Persistent under the workspace,
    /// independent of the inference engine — available before the model loads.
    file_store: ConvFileStore,
}

/// Snapshot returned by `GET /v1/status`. `loading` is `None` once the
/// daemon is fully ready (chat unlocked); `Some` while any startup step
/// is still in flight. `started_at_ms` changes only across daemon
/// restarts — the frontend uses it to detect a restart and re-fetch.
pub struct StatusSnapshot {
    pub loading: Option<LoadingSnapshot>,
    pub detail: String,
    pub started_at_ms: u64,
}

/// Measured throughput of an upload, filled in once its pipeline finishes and
/// persisted alongside the [`UploadInfo`] (optional — absent on older events or
/// when the model wasn't loaded to ingest). Batch-level: every file dropped
/// together shares one measurement. Drives the inline "in Xs (X t/s, X MB/s)"
/// stat and the upload-time line in the file viewer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadStats {
    /// Total bytes received across the batch.
    pub bytes: u64,
    /// Wall-clock to receive the batch over the wire (phase 1).
    pub upload_ms: u64,
    /// Tokens prefilled reading the batch into the substrate (phase 2).
    pub ingest_tokens: u64,
    /// Wall-clock of the read_file ingest (phase 2).
    pub ingest_ms: u64,
}

/// One uploaded file recorded against a conversation — the substrate
/// event that survives resume and drives the inline history tile. Stored
/// (JSON-serialized, as an array under the `uploads` custom-metadata key)
/// in the conversation's Label record, so it recovers with the conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UploadInfo {
    /// Conv-file-store id — the handle the GUI opens the file's content by.
    pub id: u64,
    pub name: String,
    /// Workspace-relative path of the file (e.g. `uploads/notes (1).txt`),
    /// with `/` separators — portable, not an absolute host path.
    pub path: String,
    pub ext: String,
    pub kind: String,
    /// Display size (e.g. `18.4 KB`).
    pub size: String,
    /// Display timestamp (e.g. `just now`).
    pub added: String,
    /// Number of turns present when the file was uploaded — positions the
    /// tile in the history stream (the tile renders after this many turns).
    #[serde(default)]
    pub turn_index: u32,
    /// Measured upload/ingest/summarize throughput, filled in once the
    /// pipeline finishes (`record_upload_stats`). `None` until then, or when
    /// the model wasn't loaded to run the ingest phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<UploadStats>,
}

/// Sidebar entry surfaced by `GET /v1/conversations`. Built directly from
/// the substrate — `RecordType::Label` records carry the `conv_id`
/// string and the human label; `RecordType::ConvState` carries the
/// `archived` lifecycle flag. Together they're the whole sidebar
/// contract.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConvEntry {
    /// Client-supplied `conv_id` string — stable across daemon restarts,
    /// used as the sidebar's `id` and echoed back in chat requests.
    pub id: String,
    /// Human-readable title. Empty during the brief window between
    /// first-submit and titler-completion.
    pub label: String,
    /// Number of recovered turns for this conversation, from the
    /// substrate. Advisory display field.
    pub turn_count: u32,
    /// Whether the user has archived (closed) this conversation. The
    /// sidebar hides archived entries by default; the "show archived"
    /// checkbox toggles them back in via `?include_archived=true`.
    pub archived: bool,
    /// Creation-order rank used by the sidebar to sort newest-first. This is a
    /// monotonic counter (`TimelineEntry::order`), NOT a millisecond clock — the
    /// `conv_id` is a random u64 with no time information, so the substrate
    /// stamps each conversation an increasing rank in redo-log (creation) order.
    /// The wire name is kept for the frontend's existing sort key.
    pub updated_ms: u64,
}

impl ZendSession {
    pub fn new(config: DaemonConfig, log: Arc<LogBus>) -> Self {
        let projection_builder = build_projection_builder(&config.workspace);
        tracing::info!(workspace = %config.workspace.display(), "session initialised");
        let (ready_tx, _) = tokio::sync::watch::channel(false);
        let (status_tx, _) = tokio::sync::watch::channel(String::new());
        let started_at_ms = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let file_store = ConvFileStore::open(&config.workspace);
        Self {
            inference: Arc::new(RwLock::new(None)),
            watcher: Mutex::new(None),
            config,
            projection_builder,
            log,
            ready_tx,
            status_tx,
            load_progress: Arc::new(LoadProgress::new()),
            started_at_ms,
            file_store,
        }
    }

    /// The conversation-files store (uploads). Available before the model loads.
    pub fn files(&self) -> &ConvFileStore {
        &self.file_store
    }

    /// The `uploads/` directory in the daemon's workspace — raw uploaded
    /// files are written here so the workspace watcher and `code_read`
    /// pick them up like any other source file. Created on demand.
    pub fn uploads_dir(&self) -> PathBuf {
        self.config.workspace.join("uploads")
    }

    /// **Phase 1 of the upload pipeline** — write `bytes` to
    /// `<workspace>/uploads/<safe, non-colliding name>` and return the actual
    /// path used. The name is sanitized to its final path component and a
    /// conservative charset (a malicious `name` can never escape the uploads
    /// directory — no `..`, no separators, no absolute paths), then de-duped
    /// against existing files so a second `notes.txt` becomes `notes (1).txt`
    /// rather than overwriting the first. The caller reads the final file name
    /// back off the returned path.
    pub fn write_upload_to_disk(&self, name: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
        let dir = self.uploads_dir();
        std::fs::create_dir_all(&dir)?;
        let safe = sanitize_upload_name(name);
        let path = dir.join(dedup_in_dir(&dir, &safe));
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    /// The path of a file under the workspace expressed relative to the
    /// workspace root, with `/` separators (matching `repo_scan`'s
    /// normalization). Falls back to the file name alone if `path` is
    /// somehow not under the workspace. This is the form persisted in the
    /// upload event — a portable, workspace-relative reference, not an
    /// absolute host path.
    /// Path of the daemon's rolling tracing log (`<workspace>/.substrate/zend.log`),
    /// the source the `/v1/telemetry` endpoint tail-parses for the live dashboard.
    /// The file name mirrors `log_file::LOG_NAME` (that module lives in the binary
    /// crate, so it can't be referenced from here).
    pub fn tracing_log_path(&self) -> PathBuf {
        self.config.workspace.join(SUBSTRATE_DIR).join("zend.log")
    }

    pub fn workspace_relative(&self, path: &std::path::Path) -> String {
        path.strip_prefix(&self.config.workspace)
            .ok()
            .and_then(|p| p.to_str())
            .map(|s| s.replace('\\', "/"))
            .unwrap_or_else(|| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("upload")
                    .to_string()
            })
    }

    /// **Phase 2 of the upload pipeline** — read the just-uploaded files into
    /// the substrate via the normal `code_read` ingest, scoped to **only those
    /// files** (`rel_paths`, workspace-relative). Bounded to the uploaded
    /// files' work — never a whole-workspace re-ingest — so it can't overload
    /// the model and is safe under `--skip-code-read`. Returns `Ok(true)` when
    /// something was ingested, `Ok(false)` for a no-op (non-code files, or the
    /// model not loaded). Per-scope progress streams into `progress`.
    /// Returns `(ingested, failed)` — see `ingest_uploaded_files`.
    pub fn read_file_phase(
        &self,
        rel_paths: &[String],
        progress: &Arc<crate::loading::LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return Ok((false, false));
        };
        state.ingest_uploaded_files(rel_paths, progress)
    }

    /// Kick the async summariser so the just-ingested turns start folding into
    /// their summary trees promptly instead of waiting for the next periodic
    /// tick. **Fire-and-forget**: summarisation is a fully background task,
    /// invisible to the upload — the pipeline never waits on it, shows no
    /// progress for it, and records nothing about it. A no-op when the model
    /// isn't loaded (the periodic tick picks the work up once it is).
    pub fn kick_summariser(&self) {
        if let Some(state) = self.inference.read().unwrap().as_ref() {
            state.engine.lock().unwrap().trigger_summariser();
        }
    }

    /// Snapshot for `GET /v1/status`. `loading=None` once every startup
    /// step is complete (chat unlocked); `Some` while any step is still
    /// in flight. `detail` is the active step's free-form sub-status
    /// (e.g. download progress). `started_at_ms` is fixed for the
    /// lifetime of this `ZendSession` and changes only across daemon
    /// restarts.
    pub fn status_snapshot(&self) -> StatusSnapshot {
        StatusSnapshot {
            loading: self.load_progress.snapshot(),
            detail: self.status_tx.subscribe().borrow().clone(),
            started_at_ms: self.started_at_ms,
        }
    }

    /// Sidebar entries for `GET /v1/conversations`. Built from the
    /// in-RAM substrate (the `RecordType::Label` records carry every
    /// conversation's `(conv_id, label)` pair), but **gated on the
    /// on-disk redo log still existing**.
    ///
    /// The gate exists because the dev workflow of wiping the
    /// `.substrate/` dir behind a running daemon used to leave ghost
    /// conversations in the sidebar: the daemon's in-RAM substrate
    /// kept the old `known_conversations` snapshot even though disk
    /// was empty. With the gate, deleting the log forces the next
    /// refresh to return an empty list — matching what the user
    /// observes on disk.
    ///
    /// Sorted newest-first by `conv_id` string descending (the frontend
    /// supplies `Date.now()` ids, so lexicographic-descending == newest).
    /// The titler's internal timeline is excluded. Archived
    /// conversations are filtered out unless `include_archived` is
    /// set — that's the "show archived" checkbox at the bottom of
    /// the sidebar.
    /// The segmented redo log's maintenance state — `(segment_count, last_op)`
    /// — for the `/v1/status` compaction indicator. `None` until the engine is
    /// loaded.
    pub fn substrate_maintenance(&self) -> Option<(usize, Option<(String, u64)>, bool)> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let engine = state.engine.lock().unwrap();
        Some(engine.substrate_maintenance_status())
    }

    /// Force one background-maintenance op now — the `POST /v1/debug/maintenance`
    /// test trigger. Seals the active segment (so a conversation archived this
    /// session, whose now-dead records sit in the active, becomes eligible) and
    /// compacts a dead-carrying sealed segment, waiving the age/ratio gates. Runs
    /// under phased locking (never holds the substrate write lock across the
    /// relocation I/O) and — crucially — is invoked on a cloned `Conversation`
    /// handle **without** holding the engine lock, so the user's in-flight chat
    /// (per-turn group-commits) isn't blocked. Returns `(ran, segments, last_op)`,
    /// or `None` if the model isn't loaded.
    pub fn force_maintenance(&self) -> Option<(bool, usize, Option<(String, u64)>)> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let conv = { state.engine.lock().unwrap().conversation() };
        let ran = match conv.force_compact_persistence() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("force maintenance failed: {e:#}");
                false
            }
        };
        let (segments, last_op, _running) = conv.maintenance_status();
        Some((ran, segments, last_op))
    }

    /// `GET /v1/substrate` — a structural snapshot of the live substrate for the
    /// viewer: on-disk size, live counts, and the projection hierarchy (layers →
    /// conversations). Reads the shared `Substrate` through a cloned
    /// `Conversation` handle (engine lock released immediately, only the
    /// substrate read guard held for the walk), never a rebuild from the log.
    /// `None` until the model is loaded.
    pub fn substrate_overview(&self) -> Option<SubstrateOverview> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;

        // Projection schema — the builder parsed once at session construction, no
        // lock held (and no per-request re-parse).
        let schema = self.projection_builder.schema();

        // On-disk footprint (fs stat, no lock).
        let (segments, total_bytes) = self.substrate_segment_files();

        let titler = state.titler_timeline;
        let target_layer = {
            let base = state.base_conv.lock().unwrap();
            base.target_layer_name()
        };

        // Clone the handle out (drop the engine lock), then read the substrate.
        let conv = { state.engine.lock().unwrap().conversation() };
        let dead_ratio = conv.dead_ratio();
        let read = conv.read();
        let s = &*read;

        let counts = Counts {
            timelines: s.timeline_count(),
            conversations: s.conversation_count(),
            sections: s.section_count(),
        };

        // Layer metadata only — each layer carries a conversation COUNT, never
        // the list. The list, the framing sections, and the tool catalog are
        // fetched per-expansion (substrate_layer / substrate_system_prompt /
        // substrate_tools) so init and the periodic refresh stay cheap no matter
        // how large the corpus grows.
        let layers: Vec<LayerView> = schema
            .layers
            .iter()
            .map(|l| {
                let (conv_count, tokens) = layer_totals(s, &l.groups, titler);
                LayerView {
                    name: l.name.clone(),
                    description: l.description.trim().to_string(),
                    window: l.window,
                    priority: l.budget.priority,
                    groups: l
                        .groups
                        .iter()
                        .map(|g| GroupView {
                            name: g.name.clone(),
                            selection: fmt_selection(&g.selection),
                        })
                        .collect(),
                    system_prompt: system_prompt_labels(&l.system_prompt),
                    conv_count,
                    tokens,
                }
            })
            .collect();

        Some(SubstrateOverview {
            storage: Storage {
                segments,
                total_bytes,
                live_chunks: s.live_chunk_count(),
                dead_ratio,
            },
            counts,
            target_layer,
            layers,
        })
    }

    /// `GET /v1/substrate/layer/{name}` — the conversations (timelines) that
    /// target one layer. Fetched when the layer card expands. `None` when the
    /// model isn't loaded or the layer name is unknown.
    pub fn substrate_layer(&self, name: &str) -> Option<LayerConversations> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let schema = self.projection_builder.schema();
        // Resolve the layer's groups before touching the substrate.
        let groups = schema
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.groups.clone())?;
        let titler = state.titler_timeline;
        let conv = { state.engine.lock().unwrap().conversation() };
        let read = conv.read();
        Some(LayerConversations {
            conversations: layer_conv_views(&read, &groups, titler),
        })
    }

    /// `GET /v1/substrate/system-prompt` — the target (dialogue) layer's framing:
    /// its ordered item labels, its always-emit sections with authored text +
    /// materialized token cost, and the live tool-catalog size (the catalog
    /// itself loads from `substrate_tools`). Fetched when the system-prompt card
    /// expands.
    pub fn substrate_system_prompt(&self) -> Option<SystemPromptView> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let schema = self.projection_builder.schema();
        let target_layer = {
            let base = state.base_conv.lock().unwrap();
            base.target_layer_name()
        };
        let layer = schema.layers.iter().find(|l| l.name == target_layer)?;

        let conv = { state.engine.lock().unwrap().conversation() };
        let read = conv.read();
        let s = &*read;

        // Always-emit framing sections (skips the tools collection and the
        // section tree, which appear as labels in `items`).
        let sections: Vec<SectionView> = layer
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Section(sec) => Some(SectionView {
                    name: sec.name.clone(),
                    tokens: s.section_tokens_of(sec.id).len(),
                    blocks: s.section_block_count(sec.id).unwrap_or(0),
                    content: sec.content.clone(),
                }),
                _ => None,
            })
            .collect();

        Some(SystemPromptView {
            target_layer,
            items: system_prompt_labels(&layer.system_prompt),
            sections,
            tool_count: zend_tools::registry::all_tools().len(),
        })
    }

    /// `GET /v1/substrate/tools` — the live tool catalog projected into the
    /// dialogue layer's `tools` collection. Sourced from the runtime registry
    /// (the schema YAML declares the collection empty — tools install at daemon
    /// start). Fetched when the tools collection expands. Needs only the model
    /// to be loaded.
    pub fn substrate_tools(&self) -> Option<ToolsView> {
        let _state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        Some(ToolsView {
            tools: zend_tools::registry::all_tools()
                .iter()
                .map(|t| ToolView {
                    name: t.name.to_string(),
                    description: t.description.to_string(),
                    high_risk: t.high_risk,
                })
                .collect(),
        })
    }

    /// `GET /v1/substrate/timeline/{tl}` — one conversation's full summary forest,
    /// every turn decoded. `tl` is the raw timeline id. `None` when the model
    /// isn't loaded or no such timeline exists.
    pub fn substrate_timeline(&self, tl_raw: u64) -> Option<TimelineDetail> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let tl = projection::TimelineId::from_raw(tl_raw)?;

        let schema = self.projection_builder.schema();

        let conv = { state.engine.lock().unwrap().conversation() };
        let read = conv.read();
        let s = &*read;

        // Reject an unknown timeline (nothing sealed and no registration).
        if s.turn_count(tl) == 0 && s.timeline_target(tl).is_none() {
            return None;
        }

        let (layer, group) = match s.timeline_target(tl) {
            Some((lid, gid)) => {
                let l = schema.layers.iter().find(|l| l.id == lid);
                (
                    l.map(|l| l.name.clone()).unwrap_or_default(),
                    l.and_then(|l| l.groups.iter().find(|g| g.id == gid))
                        .map(|g| g.name.clone())
                        .unwrap_or_default(),
                )
            }
            None => (String::new(), String::new()),
        };

        let peak_set: HashSet<u32> = s.peaks_of(tl).into_iter().map(|(idx, _)| idx.0).collect();
        let couplings = s.couplings_of(tl);
        let mut coupling_list: Vec<u32> = couplings.iter().copied().collect();
        coupling_list.sort_unstable();

        let mut indices: Vec<_> = s.turn_indices(tl).collect();
        indices.sort_by_key(|i| i.0);
        let turns: Vec<TurnView> = indices
            .into_iter()
            .map(|idx| {
                let (kind, height, children) = match s.tree_meta_of(tl, idx) {
                    Some(m) => (
                        match m.kind {
                            TurnKind::Normal => "normal",
                            TurnKind::SummaryOfTurns => "sot",
                            TurnKind::SummaryOfSummaries => "sos",
                        },
                        m.tree_height,
                        m.children.iter().map(|c| c.0).collect::<Vec<u32>>(),
                    ),
                    None => ("normal", 0, Vec::new()),
                };
                TurnView {
                    index: idx.0,
                    kind,
                    height,
                    tokens: s.turn_token_count_of(tl, idx),
                    no_think: s.turn_no_think(tl, idx),
                    children,
                    coupled: couplings.contains(&idx.0),
                    peak: peak_set.contains(&idx.0),
                    user: s.user_text_of(tl, idx),
                    assistant: s.assistant_text_of(tl, idx),
                    layout: s.turn_layout(tl, idx),
                }
            })
            .collect();

        Some(TimelineDetail {
            timeline: tl_raw.to_string(),
            conv_id: s.conv_id_of(tl).unwrap_or_default().to_string(),
            label: s.label_of(tl).unwrap_or_default().to_string(),
            archived: s.is_archived(tl),
            layer,
            group,
            total_tokens: s.total_token_count(tl),
            peaks: {
                let mut p: Vec<u32> = peak_set.into_iter().collect();
                p.sort_unstable();
                p
            },
            couplings: coupling_list,
            turns,
        })
    }

    /// `POST /v1/substrate/project` — project a typed query against the substrate.
    /// Prefills `text` under the base conversation's system prompt (no seal, no
    /// belief mutation), then resolves the selected turns + scored section members
    /// into score-sorted tiles. `None` until the model is loaded.
    pub fn substrate_project(&self, text: &str) -> Option<ProjectView> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        if text.trim().is_empty() {
            return Some(ProjectView {
                query_tokens: 0,
                tiles: Vec::new(),
            });
        }

        // Project the query against the substrate as a warm ephemeral turn: the
        // probe materializes the full system prompt, captures the query's wide-Q,
        // and scores it — writing NOTHING to the substrate. Snapshot the handles
        // under a brief `base_conv` lock, then run the GPU-bound probe UNLOCKED so
        // the fork source isn't held across a full turn round-trip.
        let ctx = { state.base_conv.lock().unwrap().probe_ctx() };
        let (event, query_tokens) = match ctx.probe(text) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("substrate_project probe failed: {e:#}");
                return Some(ProjectView {
                    query_tokens: 0,
                    tiles: Vec::new(),
                });
            }
        };

        // Resolve tile labels + bodies from the substrate (read lock only).
        let conv = { state.engine.lock().unwrap().conversation() };
        let read = conv.read();
        let s = &*read;

        let mut tiles: Vec<ProjectTile> = Vec::new();

        // Selected conversation turns / summary nodes.
        for t in &event.selection.turns {
            let Some(tl_raw) = t.timeline else {
                continue; // the live user message has no source timeline
            };
            let kind = match t.kind {
                TurnKind::Normal => "normal",
                TurnKind::SummaryOfTurns => "sot",
                TurnKind::SummaryOfSummaries => "sos",
            };
            let (label, body) = match projection::TimelineId::from_raw(tl_raw) {
                Some(tl) => {
                    let title = s
                        .label_of(tl)
                        .filter(|l| !l.is_empty())
                        .or_else(|| s.conv_id_of(tl).filter(|c| !c.is_empty()))
                        .map(|x| x.to_string())
                        .or_else(|| {
                            s.custom_of(tl)
                                .and_then(|m| m.get("path"))
                                .filter(|p| !p.is_empty())
                                .cloned()
                        })
                        .unwrap_or_else(|| format!("turn #{}", t.index));
                    let idx = TurnIndex(t.index);
                    let u = s.user_text_of(tl, idx);
                    let a = s.assistant_text_of(tl, idx);
                    let body = if u.trim().is_empty() {
                        a
                    } else if a.trim().is_empty() {
                        u
                    } else {
                        format!("{u}\n\n{a}")
                    };
                    (title, body)
                }
                None => (format!("turn #{}", t.index), String::new()),
            };
            tiles.push(ProjectTile {
                kind,
                score: t.score,
                selected: t.selected,
                layer: t.layer.clone(),
                group: t.group.clone(),
                label,
                tokens: t.tokens,
                timeline: Some(tl_raw.to_string()),
                index: Some(t.index),
                text: body,
            });
        }

        // Scored section members (the collection tools). Text comes from the live
        // registry (the schema declares the collection empty). Cut the noise floor:
        // keep only members the query selected or gave a positive belief score.
        let tool_desc: HashMap<&str, &str> = zend_tools::registry::all_tools()
            .iter()
            .map(|t| (t.name, t.description))
            .collect();
        for item in &event.selection.system {
            if let SystemItem::Collection { name, sections } = item {
                for sec in sections {
                    if !sec.selected && sec.score <= 0.0 {
                        continue;
                    }
                    tiles.push(ProjectTile {
                        kind: "section",
                        score: sec.score,
                        selected: sec.selected,
                        layer: String::new(),
                        group: name.clone(),
                        label: sec.name.clone(),
                        tokens: sec.tokens,
                        timeline: None,
                        index: None,
                        text: tool_desc
                            .get(sec.name.as_str())
                            .map(|d| d.to_string())
                            .unwrap_or_default(),
                    });
                }
            }
        }

        tiles.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Diagnostic: if `scored` is 0 while `query_tokens` > 0, the probe Q was
        // captured but didn't discriminate against the gallery (cold/partial-warm
        // context) — selection fell back to the default fill.
        let max_score = tiles.iter().map(|t| t.score).fold(0.0f32, f32::max);
        tracing::info!(
            query_tokens,
            tiles = tiles.len(),
            scored = tiles.iter().filter(|t| t.score.abs() > f32::EPSILON).count(),
            max_score,
            "substrate_project"
        );

        Some(ProjectView {
            query_tokens,
            tiles,
        })
    }

    /// The segmented redo log's on-disk segment files, ascending by id (the last
    /// is the active one), plus their total byte size. Pure `fs` stat — no lock,
    /// available whether or not the model has finished loading.
    fn substrate_segment_files(&self) -> (Vec<SegmentView>, u64) {
        let dir = self.config.workspace.join(SUBSTRATE_DIR);
        let mut segs: Vec<(u64, u64)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let name = e.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(rest) = name.strip_prefix("seg-") else {
                    continue;
                };
                let Some((num, ext)) = rest.rsplit_once('.') else {
                    continue;
                };
                if ext != "log" && ext != "active" {
                    continue;
                }
                let Ok(id) = num.parse::<u64>() else { continue };
                let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
                segs.push((id, bytes));
            }
        }
        segs.sort_by_key(|(id, _)| *id);
        let total: u64 = segs.iter().map(|(_, b)| *b).sum();
        let last = segs.len().saturating_sub(1);
        let views = segs
            .iter()
            .enumerate()
            .map(|(i, (id, bytes))| SegmentView {
                id: *id,
                active: i == last,
                bytes: *bytes,
            })
            .collect();
        (views, total)
    }

    pub fn list_conversations(&self, include_archived: bool) -> Vec<ConvEntry> {
        // On-disk gate: if the workspace's redo log is gone (no segment files
        // in `.substrate/`), return empty regardless of the in-RAM cache. The
        // daemon keeps running and any new turn re-mints the segment set.
        let sub_dir = self.config.workspace.join(SUBSTRATE_DIR);
        let has_segments = std::fs::read_dir(&sub_dir)
            .ok()
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("seg-"))
                })
            })
            .unwrap_or(false);
        if !has_segments {
            return Vec::new();
        }

        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return Vec::new();
        };
        let engine = state.engine.lock().unwrap();
        let titler_timeline = state.titler_timeline;
        let turn_counts: std::collections::HashMap<projection::TimelineId, u32> = {
            let base = state.base_conv.lock().unwrap();
            base.recovered_timelines().into_iter().collect()
        };
        let mut entries: Vec<ConvEntry> = engine
            .known_conversations()
            .into_iter()
            .filter(|(tl, _, _, _, _)| *tl != titler_timeline)
            .filter(|(_, _, _, archived, _)| include_archived || !*archived)
            .map(|(tl, conv_id, label, archived, order)| {
                // `order` is creation rank (see `TimelineEntry::order`) — the
                // conv_id itself is a random u64 and carries no time. The field
                // is named `updated_ms` for the wire, but it is a monotonic
                // rank, not a millisecond clock; the sidebar only ever sorts on
                // it, never displays it as a time.
                ConvEntry {
                    id: conv_id,
                    label,
                    turn_count: turn_counts.get(&tl).copied().unwrap_or(0),
                    archived,
                    updated_ms: order,
                }
            })
            .collect();
        // Newest-created first.
        entries.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms));
        entries
    }

    /// Archive a conversation — one-way. Sets the archived lifecycle flag and
    /// marks the timeline for `TextOnly` distillation, so the next compaction
    /// sheds its KV chunks + signatures + projections and leaves a read-only
    /// text record. There is no unarchive: distillation is lossy (the KV is
    /// gone), so an archived conversation can't be resumed. Returns `None` when
    /// the model isn't loaded yet.
    pub fn archive_conversation(&self, conv_id: &str) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let timeline = timeline_for(conv_id);
        let engine = state.engine.lock().unwrap();
        let result = engine
            .set_conversation_archived(timeline, true)
            .and_then(|()| engine.distill_timeline(timeline, DistillMode::TextOnly));
        Some(result)
    }

    /// Permanently tombstone a conversation — the delete path behind
    /// `DELETE /v1/conversations/{id}`. Writes a durable `Tombstone` record so the
    /// timeline is hidden from listings/resume immediately, and its records
    /// (turns, KV, provenance) are physically reclaimed at the next compaction
    /// (lazy — the tombstone is the commit, compaction does the delete). Also
    /// drops the conversation from the in-RAM map so it vanishes from the sidebar
    /// at once. Returns `None` when the model isn't loaded yet.
    pub fn tombstone_conversation(&self, conv_id: &str) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let timeline = timeline_for(conv_id);
        let result = state.engine.lock().unwrap().tombstone_timeline(timeline);
        if result.is_ok() {
            state.conversations.lock().unwrap().remove(conv_id);
        }
        Some(result)
    }

    /// Enable or disable AVL summarisation for `conv_id`'s timeline. Only takes
    /// effect once the timeline exists (i.e. after its first turn has been
    /// submitted). Returns `None` if the model isn't loaded. Exercised only by the
    /// CUDA-gated `duplication_replay` integration test (via the `zend` lib), to
    /// isolate whether the async summariser's concurrent activity influences a
    /// conversation's decode — so the `zend` *binary* never calls it and its copy
    /// of this module reads as dead; the lib copy the test links is public API.
    #[allow(dead_code)]
    pub fn set_conversation_summarize(&self, conv_id: &str, summarize: bool) -> Option<()> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let timeline = timeline_for(conv_id);
        let engine = state.engine.lock().unwrap();
        engine.set_timeline_summarize(timeline, summarize);
        Some(())
    }

    /// Decoded turn history for a single recovered conversation — backs
    /// `GET /v1/conversations/{id}`. Returns `None` when the model isn't
    /// loaded yet; an empty `Vec` when the conv_id has no recovered turns.
    pub fn conversation_history(&self, conv_id: &str) -> Option<Vec<(Role, String, bool)>> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = state.base_conv.lock().unwrap();
            // Conversation view: hide the summariser's ghost summary turns.
            base.recovered_history(timeline, false)
        };
        let decoded = raw
            .into_iter()
            .map(|(role, text, no_think)| {
                let role = match role {
                    candle_conversation::Role::User => Role::User,
                    candle_conversation::Role::Assistant => Role::Assistant,
                    candle_conversation::Role::System => Role::System,
                };
                (role, text, no_think)
            })
            .collect();
        Some(decoded)
    }

    /// Record a batch of just-uploaded files as an event in the substrate,
    /// so they recover with the conversation and can be replayed inline in
    /// its history. Each file is stamped with `turn_index` — the number of
    /// turns present at upload time — which positions it in the history
    /// stream. Persisted (last-writer-wins) in the conversation's Label
    /// record `custom` bag under the `uploads` key, a JSON array appended to
    /// on every upload. No-op when the model isn't loaded.
    pub fn record_uploads(&self, conv_id: &str, files: &[UploadInfo]) {
        if files.is_empty() {
            return;
        }
        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return;
        };
        let timeline = timeline_for(conv_id);
        let engine = state.engine.lock().unwrap();
        // Register the conv_id so an upload-only conversation (one started by
        // dropping a file on the home page, which may never submit a chat turn)
        // is a *listed* conversation: `known_conversations` — and thus the
        // sidebar + the client's server-authoritative sync — only surface
        // timelines that carry a conv_id. Normally the chat submit sets this;
        // without it here the conversation exists only client-side and gets
        // dropped on the next sync (vanishes from the list, and a send bounces
        // back to home). Idempotent for a conversation that already has one.
        if let Err(e) = engine.set_conversation_conv_id(timeline, conv_id) {
            tracing::warn!(conv_id = %conv_id, "record_uploads: set conv_id failed: {e}");
        }
        // Give it a provisional label from the file(s) if it has none yet, so it
        // shows a sensible name in the sidebar before any chat turn — the titler
        // refines it once the user actually talks. Never overwrite an existing
        // label (an upload into a conversation that already has turns).
        if engine
            .conversation_label_of(timeline)
            .unwrap_or_default()
            .is_empty()
        {
            let label = match files.first() {
                Some(f) if files.len() == 1 => f.name.clone(),
                _ => format!("{} files", files.len()),
            };
            if let Err(e) = engine.set_conversation_label(timeline, &label) {
                tracing::warn!(conv_id = %conv_id, "record_uploads: set label failed: {e}");
            }
        }
        // Position: the number of already-present turns. `record_uploads`
        // runs between turns (the user drops a file, then sends), so every
        // prior turn is sealed and visible in `recovered_history`.
        let turn_index = {
            let base = state.base_conv.lock().unwrap();
            base.recovered_history(timeline, false).len() as u32
        };
        // Read the current uploads array, append the new files, write it back.
        let mut arr: Vec<UploadInfo> = engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get("uploads").cloned())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        for f in files {
            let mut f = f.clone();
            f.turn_index = turn_index;
            arr.push(f);
        }
        if let Ok(json) = serde_json::to_string(&arr) {
            if let Err(e) = engine.set_conversation_metadata(timeline, "uploads", &json) {
                tracing::warn!(conv_id = %conv_id, "record_uploads write failed: {e}");
            }
        }
    }

    /// Attach measured throughput to an already-recorded upload batch. `ids`
    /// are the conv-file-store ids from the batch's [`UploadInfo`]s; every
    /// matching entry gets `stats` (batch-level, so they share one
    /// measurement). A no-op when the model isn't loaded. Called once the
    /// upload pipeline finishes, after `record_uploads` positioned the batch.
    pub fn record_upload_stats(&self, conv_id: &str, ids: &[u64], stats: &UploadStats) {
        if ids.is_empty() {
            return;
        }
        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return;
        };
        let timeline = timeline_for(conv_id);
        let engine = state.engine.lock().unwrap();
        let mut arr: Vec<UploadInfo> = engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get("uploads").cloned())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        for entry in &mut arr {
            if ids.contains(&entry.id) {
                entry.stats = Some(stats.clone());
            }
        }
        if let Ok(json) = serde_json::to_string(&arr) {
            if let Err(e) = engine.set_conversation_metadata(timeline, "uploads", &json) {
                tracing::warn!(conv_id = %conv_id, "record_upload_stats write failed: {e}");
            }
        }
    }

    /// The upload events recorded against a conversation, in upload order —
    /// backs both the files pane and the inline history tiles on resume.
    /// Empty when the model isn't loaded or nothing was uploaded.
    pub fn conversation_uploads(&self, conv_id: &str) -> Vec<UploadInfo> {
        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let engine = state.engine.lock().unwrap();
        engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get("uploads").cloned())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Projection-event buckets banked for a conversation this daemon session,
    /// one per assistant turn (in order). Backs the timeline-dot replay on a
    /// browser reload. Empty when the conversation isn't resident (e.g. it was
    /// recovered from disk but hasn't been chatted with since startup).
    pub fn conversation_projections(&self, conv_id: &str) -> Vec<Vec<ProjectionEventOut>> {
        let Some(state) = self.inference.read().unwrap().as_ref().map(Arc::clone) else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = state.base_conv.lock().unwrap();
            base.recovered_projection_events(timeline)
        };
        // Re-attach the display id/region/step the wire shape needs (these are
        // GUI concerns, not persisted); one bucket per assistant turn, in order.
        raw.into_iter()
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| {
                        let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ProjectionEventOut::answer(seq, event)
                    })
                    .collect()
            })
            .collect()
    }

    /// `(section name, authored content)` for every system-prompt section in the
    /// schema, plus the runtime tool summary matching this conversation's tools
    /// mode. Backs the projection panel's expandable section text — resolved on
    /// demand, never stored in the projection event. The schema is workspace-wide;
    /// `conv_id` selects which tool summary (restricted vs comprehensive) to serve.
    pub fn section_content(&self, conv_id: &str) -> Option<Vec<(String, String)>> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let mut out = {
            let base = state.base_conv.lock().unwrap();
            base.section_contents()
        };
        // The `tools` collection's summary section is generated at runtime (not a
        // schema section), so its text isn't in `section_contents`. Serve the
        // summary matching this conversation's tools mode — the restricted
        // (safe-subset) summary in Restricted, the full one in Comprehensive, and
        // none in None (no tools are projected) — under the key the projection
        // event uses (`<collection> summary`) so the panel expands the right list.
        // Copy the Arc out and release the conversations-map lock before locking
        // the per-conversation state: the inference loop holds a conversation's
        // lock for its entire decode, so locking it while still holding the map
        // mutex would stall every other conversation's turn submission.
        let cs = state.conversations.lock().unwrap().get(conv_id).cloned();
        let mode = cs
            .map(|cs| cs.lock().unwrap().tool_mode)
            .unwrap_or_default();
        let restricted = match mode {
            ToolMode::None => return Some(out),
            ToolMode::Restricted => true,
            ToolMode::Comprehensive => false,
        };
        // The tools-collection summary is assembled deterministically from the
        // catalog (the same text the startup seals); rebuild the mode-appropriate
        // one for the panel rather than reading a cache.
        let text = crate::tool_summary::tool_summary_for_mode(restricted);
        if !text.is_empty() {
            out.push(("tools summary".to_string(), text));
        }
        Some(out)
    }

    /// The target (dialogue) layer's name, for the panel's conversation prefix.
    pub fn target_layer_name(&self) -> Option<String> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let base = state.base_conv.lock().unwrap();
        Some(base.target_layer_name())
    }

    /// The `(user, assistant)` halves of a projected turn `index` in `timeline`,
    /// read from the substrate. Backs the projection panel's memory-tier bodies.
    /// `timeline` is the turn's resolved identity from `SelectedTurn::timeline` —
    /// never a group→timeline guess (the shared substrate holds many
    /// conversations under one group).
    pub fn resolve_turn_text(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<(String, String)> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let base = state.base_conv.lock().unwrap();
        base.resolve_turn_text(timeline, index)
    }

    /// The entire turn `index` in `timeline` as one continuous string (the full
    /// sealed token range, decoded verbatim). Backs the projection panel's turn
    /// cards — emit the whole turn rather than the split user/assistant halves.
    pub fn resolve_turn_full_text(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<String> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let base = state.base_conv.lock().unwrap();
        base.resolve_turn_full_text(timeline, index)
    }

    /// The turn's segment-vector `TurnLayout` — the complete K/V description
    /// (glue / user / thinking / assistant, real vs ethereal). Backs the panel's
    /// exact-segment rendering. `timeline` is the turn's resolved identity.
    pub fn turn_layout(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<candle_conversation::turn_layout::TurnLayout> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let base = state.base_conv.lock().unwrap();
        base.turn_layout(timeline, index)
    }

    /// The dialect's framing markers — the glue wrapped around the system prompt
    /// and each turn. The projection panel renders these between sections/turns.
    pub fn glue_markers(&self) -> Option<GlueMarkers> {
        let state = self.inference.read().unwrap().as_ref().map(Arc::clone)?;
        let base = state.base_conv.lock().unwrap();
        Some(base.glue_markers())
    }

    pub fn start_loading(self: &Arc<Self>) {
        let slot = Arc::clone(&self.inference);
        let proj_builder = self.projection_builder.clone();
        let ready_tx = self.ready_tx.clone();
        let status_tx = self.status_tx.clone();
        let load_progress = Arc::clone(&self.load_progress);
        let workspace = self.config.workspace.clone();
        let skip_code_read = self.config.skip_code_read;
        let skip_repo_scan = self.config.skip_repo_scan;
        let disable_summariser = self.config.disable_summariser;
        let compact_substrate = self.config.compact_substrate;
        // Handle to the ambient Tokio runtime (if any). The loader runs on a
        // plain OS thread and drops its temporary download runtime before the
        // model load, so the workspace watcher's `tokio::spawn` would otherwise
        // panic for lack of a runtime context — which killed the loader thread
        // before `mark_ready()`, wedging the loading screen. We re-enter this
        // handle after the download phase so the watcher can spawn. `None` in
        // test contexts with no ambient runtime (the watcher is then skipped).
        let rt_handle = tokio::runtime::Handle::try_current().ok();
        // Held by the spawned thread so the workspace watcher's
        // lifetime ends when the session is dropped — the watcher
        // is stored under `Arc<ZendSession>::watcher`.
        let session_for_watcher: Arc<Self> = Arc::clone(self);
        // OS thread, not `tokio::spawn` — `start_loading` may be
        // called from contexts without an ambient Tokio runtime
        // (integration tests, alternative binaries).  The bits that
        // need an async runtime (HF download) drive a local
        // current-thread runtime built inside this thread.
        std::thread::Builder::new()
            .name("zend-loader".into())
            .spawn(move || {
                load_progress.set_step(LoadStep::Model);
                status_tx.send("Checking for model…".into()).ok();

                // The download path is async (hf-hub, tokio::fs, reqwest);
                // build a tiny current-thread runtime so we can `block_on`
                // it without requiring the caller to be on Tokio.
                let download_runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!(
                            "local tokio runtime for download failed: {e:#}; exiting"
                        );
                        status_tx.send(format!("Runtime build failed: {e}")).ok();
                        std::process::exit(1);
                    }
                };
                let (model_path, tok_path) =
                    match download_runtime.block_on(crate::download::ensure_model(&status_tx)) {
                        Ok(p) => p,
                        Err(e) => {
                            // A missing model is fatal — the daemon cannot serve
                            // anything without it. Fail hard rather than limping
                            // into a fake-ready state that errors on first submit.
                            tracing::error!("model download failed: {e:#}; exiting");
                            status_tx.send(format!("Download failed: {e}")).ok();
                            std::process::exit(1);
                        }
                    };
                // Drop the runtime; the model load below is sync.
                drop(download_runtime);

                // Re-enter the main Tokio runtime for the rest of this thread so the
                // workspace watcher's `tokio::spawn` has a runtime context. Must come
                // *after* the download runtime is dropped (no nested `block_on`).
                // The model load is synchronous, so holding the enter guard is safe.
                let _rt_guard = rt_handle.as_ref().map(|h| h.enter());

                status_tx.send("Loading model…".into()).ok();
                tracing::info!("loading inference engine (Qwen3-30B-A3B) …");
                let load_progress_for_blocking = Arc::clone(&load_progress);
                // `InferenceState::load` is fully synchronous (CUDA model
                // load + substrate recovery + ingestion).  Call directly
                // on this thread — no `spawn_blocking` needed.
                match InferenceState::load(
                    proj_builder,
                    model_path,
                    tok_path,
                    workspace,
                    skip_code_read,
                    skip_repo_scan,
                    disable_summariser,
                    compact_substrate,
                    load_progress_for_blocking,
                ) {
                    Ok(state) => {
                        *slot.write().unwrap() = Some(Arc::clone(&state));
                        tracing::info!("inference engine ready");
                        status_tx.send(String::new()).ok();
                        // Substrate persistence runs in the engine's own
                        // thread (`PersistenceThread`) — 5 s tick + per-turn
                        // trigger from the scheduler — so no periodic-flush
                        // task is needed at the daemon level here.
                        //
                        // `LoadStep::RepoScan` and `LoadStep::CodeRead`
                        // are advanced inside `InferenceState::load`
                        // itself — the ingestion passes own those steps
                        // and report sub-step progress through the same
                        // `LoadProgress` handle.

                        // Arm the workspace watcher.  Filesystem events
                        // debounce into a single refresh call covering
                        // BOTH layers — name-relevant events (create /
                        // remove / rename) can move the repo-map
                        // cluster hashes, and content edits can move
                        // the code-reading file-content hashes.  Each
                        // refresh path short-circuits internally when
                        // its hash record is unchanged, so the work is
                        // bounded.
                        let inference_for_watcher = Arc::clone(&slot);
                        let on_refresh: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                            let Some(state) = inference_for_watcher
                                .read()
                                .unwrap()
                                .as_ref()
                                .map(Arc::clone)
                            else {
                                return;
                            };
                            // Walk the workspace once per event burst and
                            // share the result between both refresh paths.
                            // Walks are the dominant cost on workspaces
                            // with tens of thousands of files; doing it
                            // twice per burst is wasted work.
                            let map = crate::repo_scan::walk_workspace(&state.workspace);
                            if skip_repo_scan {
                                tracing::trace!(
                                    "--skip-repo-scan: watcher repo-map refresh suppressed"
                                );
                            } else {
                                match state.refresh_repo_map(Some(map.clone())) {
                                    Ok(true) => {
                                        tracing::info!("repo map refreshed after fs event burst")
                                    }
                                    Ok(false) => tracing::trace!(
                                        "fs event burst saw no cluster hash change — repo map skipped"
                                    ),
                                    Err(e) => tracing::warn!("repo map refresh failed: {e:#}"),
                                }
                            }
                            // Honour --skip-code-read for the watcher too: with code
                            // reading disabled the per-file content state is empty,
                            // so a refresh would treat every file as changed and
                            // re-ingest the whole repo (flooding the scheduler and
                            // starving interactive chat).
                            if skip_code_read {
                                tracing::trace!(
                                    "--skip-code-read: watcher code-reading refresh suppressed"
                                );
                            } else {
                                match state.refresh_code_reading(Some(map)) {
                                    Ok(true) => {
                                        tracing::info!(
                                            "code reading refreshed after fs event burst"
                                        )
                                    }
                                    Ok(false) => tracing::trace!(
                                    "fs event burst saw no file content change — code read skipped"
                                ),
                                    Err(e) => tracing::warn!("code read refresh failed: {e:#}"),
                                }
                            }
                        });
                        // Uploads are endpoint-managed, so upload churn never
                        // drives the source refresh above — but a deletion of an
                        // uploaded file still has to retire its substrate
                        // conversation. Fire the cheap tombstone-if-absent
                        // reconcile once at startup (uploads deleted while the
                        // daemon was down) and on every `uploads/` watcher burst.
                        state.reconcile_uploaded_files();
                        let inference_for_uploads = Arc::clone(&slot);
                        let on_uploads_changed: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                            let Some(state) = inference_for_uploads
                                .read()
                                .unwrap()
                                .as_ref()
                                .map(Arc::clone)
                            else {
                                return;
                            };
                            state.reconcile_uploaded_files();
                        });
                        match crate::watcher::spawn(
                            &state.workspace,
                            on_refresh,
                            on_uploads_changed,
                        ) {
                            Ok(w) => *session_for_watcher.watcher.lock().unwrap() = Some(w),
                            Err(e) => tracing::warn!("workspace watcher failed to start: {e:#}"),
                        }
                    }
                    Err(e) => {
                        tracing::error!("inference engine failed to load: {e:#}; exiting");
                        status_tx.send(format!("Load failed: {e}")).ok();
                        std::process::exit(1);
                    }
                }
                load_progress.mark_ready();
                ready_tx.send(true).ok();
            })
            .expect("spawn zend-loader thread");
    }

    /// Graceful shutdown: durably commit the substrate redo log, then
    /// stop the scheduler thread. Idempotent — safe to call when the model
    /// never finished loading (nothing to flush). Runs on the blocking pool
    /// since `commit_persistence` does synchronous `fsync` I/O.
    pub async fn shutdown(&self) {
        let state: Option<Arc<InferenceState>> =
            { self.inference.read().unwrap().as_ref().map(Arc::clone) };
        let Some(state) = state else {
            tracing::info!("shutdown: model never loaded — nothing to persist");
            return;
        };
        // Signal the titler worker to stop draining and wake it if it's idle.
        // This must happen before the scheduler stops so the worker skips any
        // queued jobs (rather than failing each one against a dead scheduler).
        state.shutting_down.store(true, Ordering::Relaxed);
        let _ = state.titler_tx.try_send(TitleJob::Shutdown);
        let _ = tokio::task::spawn_blocking(move || {
            {
                let engine = state.engine.lock().unwrap();
                match engine.commit_persistence() {
                    Ok(()) => tracing::info!("shutdown: substrate committed"),
                    Err(e) => tracing::error!("shutdown: commit failed: {e}"),
                }
                if let Err(e) = engine.shutdown() {
                    tracing::error!("shutdown: scheduler stop failed: {e}");
                }
            } // release the engine lock before joining the worker

            // Join the titler worker so any in-flight title turn unwinds
            // (via `abort_turn`) before the process exits.
            if let Some(worker) = state.titler_worker.lock().unwrap().take() {
                if worker.join().is_err() {
                    tracing::warn!("shutdown: titler worker panicked");
                }
            }
        })
        .await;
    }

    /// Submit the latest user message and return a stream of status + token items.
    pub async fn submit(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        selection: candle_conversation::SelectionState,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        self.submit_with_sampling(
            messages,
            max_tokens,
            conv_id,
            None,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            selection,
        )
        .await
    }

    /// Same as [`Self::submit`] but accepts an explicit
    /// [`candle_conversation::SamplingConfig`] override and an optional
    /// `force_hires` collection name (zend capture: force that collection to
    /// full resolution for this conversation — see [`ChatCompletionRequest`]).
    ///
    /// Used by tests that want deterministic generation (e.g.
    /// `SamplingConfig::argmax()` for greedy decoding) so the
    /// pass/fail signal isn't subject to top-k/top-p sampling noise.
    pub async fn submit_with_sampling(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        sampling: Option<candle_conversation::SamplingConfig>,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        selection: candle_conversation::SelectionState,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let inference = Arc::clone(&self.inference);
        let mut ready_rx = self.ready_tx.subscribe();
        let mut status_rx = self.status_tx.subscribe();

        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);

        tokio::spawn(async move {
            // Phase 1 — emit status updates while model is loading.
            if inference.read().unwrap().is_none() {
                let cur = status_rx.borrow_and_update().clone();
                if !cur.is_empty() && tx.send(Ok(StreamItem::Status(cur))).await.is_err() {
                    return;
                }

                loop {
                    if *ready_rx.borrow_and_update() {
                        break;
                    }
                    tokio::select! {
                        _ = ready_rx.changed() => { break; }
                        _ = status_rx.changed() => {
                            let msg = status_rx.borrow_and_update().clone();
                            if !msg.is_empty()
                                && tx.send(Ok(StreamItem::Status(msg))).await.is_err()
                            {
                                return;
                            }
                        }
                    }
                }

                if tx
                    .send(Ok(StreamItem::Status("Processing…".into())))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // Phase 2 — generate tokens. `run_inference_stream` handles
            // the fork (which registers the timeline in the substrate),
            // then persists the conv_id and spawns the titler in
            // parallel — all in the right order so the in-RAM substrate
            // sees both the timeline registration and the conv_id by
            // the time the next sidebar refresh runs.
            let state: Option<Arc<InferenceState>> =
                { inference.read().unwrap().as_ref().map(Arc::clone) };
            if let Some(state) = state {
                let mut ts = run_inference_stream(
                    state,
                    conv_id.clone(),
                    last_user.clone(),
                    max_tokens,
                    sampling,
                    force_hires,
                    assistant_prefill,
                    lossless_kv,
                    tools_mode,
                    selection,
                );
                while let Some(item) = ts.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            } else {
                tx.send(Ok(StreamItem::Status("Model unavailable.".into())))
                    .await
                    .ok();
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// The [`projection::TimelineId`] a client `conv_id` maps to — a stable hash
/// of the id. Deterministic across daemon restarts, so a reconnecting client
/// resolves to the same timeline the substrate reload recovered (§16.12).
fn timeline_for(conv_id: &str) -> projection::TimelineId {
    let h = content_hash::hash_bytes(conv_id.as_bytes());
    projection::TimelineId::from_raw(h.lo.max(1)).expect("timeline id is non-zero")
}

/// Reduce an uploaded file name to a safe basename that stays inside the
/// uploads directory: take the final path component (defeating `../` and
/// absolute paths), keep only `[A-Za-z0-9._-]`, collapse the rest to `_`,
/// and fall back to `upload` if nothing survives. A leading `.` is
/// prefixed with `_` so uploads are never hidden dotfiles.
fn sanitize_upload_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut out: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Windows strips trailing dots/spaces on create; drop them here so the
    // recorded name matches what actually lands on disk (and a `foo.exe.` can't
    // dodge the extension gate by re-materialising as `foo.exe`). Spaces have
    // already become `_`, so only trailing dots remain to trim.
    while out.ends_with('.') {
        out.pop();
    }
    // Reject the pure-dot names that name a directory, and hidden dotfiles.
    if out.is_empty() {
        return "upload".to_string();
    }
    if out.starts_with('.') {
        out.insert(0, '_');
    }
    out
}

/// Return `safe` if it doesn't already exist in `dir`, otherwise the first
/// free `stem-NNN.ext` variant (`NNN` = 001, 002, …, zero-padded to three
/// digits so directory listings sort correctly). `safe` is assumed already
/// sanitized. The scan is bounded; in the (pathological) event every
/// candidate up to the cap is taken, it falls back to the millisecond
/// timestamp so the write still succeeds rather than clobbering.
fn dedup_in_dir(dir: &std::path::Path, safe: &str) -> String {
    if !dir.join(safe).exists() {
        return safe.to_string();
    }
    // Split into stem + extension (the extension is the final `.` segment,
    // if any) so the counter lands before the extension: `notes-001.txt`.
    let (stem, ext) = match safe.rfind('.') {
        Some(i) if i > 0 => (&safe[..i], &safe[i..]), // ext includes the dot
        _ => (safe, ""),
    };
    for n in 1..=9999u32 {
        let cand = format!("{stem}-{n:03}{ext}");
        if !dir.join(&cand).exists() {
            return cand;
        }
    }
    let stamp = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{stem}-{stamp}{ext}")
}

/// Cap the titler's prefill at `head + tail` tokens. For short messages
/// (≤ head+tail) the text is returned verbatim; for long ones the head and
/// tail token windows are decoded back to text and joined with `" … "`.
/// This bounds title-generation latency regardless of how much the user
/// pasted into their first message.
fn head_tail_truncate(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    head: usize,
    tail: usize,
) -> String {
    let Ok(encoded) = tokenizer.encode(text, false) else {
        return text.to_string();
    };
    let ids = encoded.get_ids();
    if ids.len() <= head + tail {
        return text.to_string();
    }
    let head_text = tokenizer.decode(&ids[..head], false).unwrap_or_default();
    let tail_text = tokenizer
        .decode(&ids[ids.len() - tail..], false)
        .unwrap_or_default();
    format!("{head_text} … {tail_text}")
}

/// Clean a model-generated title: strip wrapping quotes, trailing
/// punctuation, and any leading "Title:" prefix the model might add.
fn clean_title(raw: &str) -> String {
    // The titler runs under `/no_think`, but Qwen3 still emits an empty
    // `<think></think>` (and may reason despite it) — strip any think block so it
    // never lands in the sidebar label.
    let stripped = candle_conversation::think_strip::strip_think_blocks(raw);
    let s = stripped.trim();
    let s = s.strip_prefix("Title:").unwrap_or(s).trim();
    let s = s
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(s);
    let s = s
        .strip_prefix('\'')
        .and_then(|r| r.strip_suffix('\''))
        .unwrap_or(s);
    let s = s.trim_end_matches(['.', '!', '?']);
    s.trim().to_string()
}

/// `(conversation count, total tokens)` across the non-empty, non-tombstoned,
/// non-titler timelines targeting a layer's groups — the cheap headline numbers
/// the overview shows without materializing the conversation list.
fn layer_totals(s: &Substrate, groups: &[GroupSchema], titler: TimelineId) -> (usize, usize) {
    groups
        .iter()
        .flat_map(|g| s.timelines_for_group(g.id))
        .filter(|tl| *tl != titler && !s.is_tombstoned(*tl) && s.turn_count(*tl) > 0)
        .fold((0, 0), |(n, tok), tl| (n + 1, tok + s.total_token_count(tl)))
}

/// The conversations targeting a layer's groups, largest first. Enumerates
/// EVERY registered timeline — not just user conversations — so internal
/// content layers (repo_map, code_reading, the analysis layers) surface their
/// timelines too; `known_conversations()` would return only the conv_id-bearing
/// dialogue conversations and leave a populated layer reading empty.
fn layer_conv_views(s: &Substrate, groups: &[GroupSchema], titler: TimelineId) -> Vec<ConvView> {
    let mut conversations: Vec<ConvView> = Vec::new();
    for g in groups {
        for tl in s.timelines_for_group(g.id) {
            if tl == titler || s.is_tombstoned(tl) || s.turn_count(tl) == 0 {
                continue;
            }
            // Title precedence: the sidebar label, then the conv_id, then the
            // conversation's `path` metadata — code_reading (and any file-scoped
            // ingest) records the file path there at creation, so those otherwise
            // conv_id-less timelines read as the file they cover instead of
            // "(untitled)".
            let label = s
                .label_of(tl)
                .filter(|l| !l.is_empty())
                .or_else(|| s.conv_id_of(tl).filter(|c| !c.is_empty()))
                .map(|s| s.to_string())
                .or_else(|| {
                    s.custom_of(tl)
                        .and_then(|m| m.get("path"))
                        .filter(|p| !p.is_empty())
                        .cloned()
                })
                .unwrap_or_default();
            let summary_nodes = s
                .turn_indices(tl)
                .filter(|idx| {
                    s.tree_meta_of(tl, *idx)
                        .is_some_and(|m| m.kind.is_summary())
                })
                .count();
            conversations.push(ConvView {
                timeline: tl.raw().to_string(),
                conv_id: s.conv_id_of(tl).unwrap_or_default().to_string(),
                label,
                archived: s.is_archived(tl),
                group: g.name.clone(),
                turns: s.turn_count(tl),
                tokens: s.total_token_count(tl),
                summary_nodes,
            });
        }
    }
    // Largest first — the most substantial timelines lead each layer.
    conversations.sort_by(|a, b| b.turns.cmp(&a.turns));
    conversations
}

/// One-line description of a group's selection rule, for the substrate viewer.
fn fmt_selection(rule: &SelectionRule) -> String {
    match rule {
        SelectionRule::AlwaysVisible => "all".to_string(),
        SelectionRule::TopK { k } => format!("top {k}"),
        SelectionRule::Single => "single".to_string(),
        SelectionRule::Named { selector } => format!("named({selector})"),
        SelectionRule::Sequence {
            recent,
            historical_top_k,
        } => format!("recent {recent} + top {historical_top_k}"),
    }
}

/// Declaration-order labels for a layer's system-prompt items — a bare section
/// by name, a collection by name + member count, a section tree by node count.
fn system_prompt_labels(sp: &SystemPromptSchema) -> Vec<String> {
    sp.items
        .iter()
        .map(|it| match it {
            SystemPromptItem::Section(s) => s.name.clone(),
            // Member count is NOT taken from `c.sections` — for the `tools`
            // collection those are installed at runtime and absent from a fresh
            // schema parse. The viewer lists the live members separately.
            SystemPromptItem::Collection(c) => format!("{} · collection", c.name),
            SystemPromptItem::SectionTree(t) => format!("section tree ({} nodes)", t.nodes.len()),
        })
        .collect()
}

fn build_projection_builder(workspace: &Path) -> Builder {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    // Qwen3-30B-A3B is a ChatML-family model; the dialect is required
    // so the schema's `kind: template` items (system_open / system_close,
    // tool_block_open/close, no_think_prefix) resolve to the right
    // structural-token strings at parse time.
    let dialect = Dialect::chat_ml();
    Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_SCHEMA_TEMPLATE,
        &[("workspace", name)],
        Some(&dialect),
    )
    .expect("projection.yaml failed to parse — check YAML syntax and {workspace} placeholder")
}

/// Concatenate the content of every top-level Section that appears
/// **before** the dialogue layer's first Collection, returning the
/// joined text.  Used as the engine's `system_prompt:` argument — it
/// gets ChatML-wrapped by the model builder, then handed to the
/// conversation as the pending-prefill prelude.  Everything from the
/// first Collection onward is expanded inside
/// [`Sequence::preemptive_prefill`] (collection sections via
/// fork-and-merge, post-collection sections via per-section prefill).
fn pre_collection_prelude(builder: &Builder) -> String {
    let layer = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == "dialogue")
        .expect("projection schema must declare a 'dialogue' layer");

    let mut out = String::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::SectionTree(t) => {
                // Default selection: each node's default option content, in order,
                // STOPPING at the embedded `tools` collection node (the prelude is
                // the text that precedes the tools; a collection node has no
                // options to render and everything below it is post-tools).
                for n in &t.nodes {
                    if n.collection.is_some() {
                        return out;
                    }
                    // Glue markers (`<tools>` etc.) are live-prefilled at projection,
                    // not part of the static system-prompt prelude — skip them here.
                    if n.glue.is_some() {
                        continue;
                    }
                    out.push_str(&n.options[n.chosen(&t.default_selection)].content);
                }
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize_upload_name;

    #[test]
    fn keeps_safe_basenames() {
        assert_eq!(sanitize_upload_name("notes.txt"), "notes.txt");
        assert_eq!(sanitize_upload_name("my-file_v2.rs"), "my-file_v2.rs");
    }

    #[test]
    fn strips_path_traversal_and_separators() {
        // Only the final component survives — no directory escape possible.
        assert_eq!(sanitize_upload_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_upload_name("/abs/path/x.md"), "x.md");
        assert_eq!(sanitize_upload_name("a\\b\\c.json"), "c.json");
        assert_eq!(sanitize_upload_name(".."), "upload");
        assert_eq!(sanitize_upload_name("."), "upload");
    }

    #[test]
    fn collapses_unsafe_chars_and_unhides_dotfiles() {
        assert_eq!(
            sanitize_upload_name("weird name!@#.txt"),
            "weird_name___.txt"
        );
        assert_eq!(sanitize_upload_name(".env"), "_.env"); // never a hidden dotfile
        assert_eq!(sanitize_upload_name(""), "upload");
    }

    #[test]
    fn dedup_appends_a_counter_before_the_extension() {
        use super::dedup_in_dir;
        let dir = std::env::temp_dir().join(format!(
            "zend_dedup_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // A free name is returned verbatim.
        assert_eq!(dedup_in_dir(&dir, "notes.txt"), "notes.txt");
        std::fs::write(dir.join("notes.txt"), b"a").unwrap();
        // Collision -> "notes-001.txt"; zero-padded, counter before the ext.
        assert_eq!(dedup_in_dir(&dir, "notes.txt"), "notes-001.txt");
        std::fs::write(dir.join("notes-001.txt"), b"b").unwrap();
        assert_eq!(dedup_in_dir(&dir, "notes.txt"), "notes-002.txt");
        // Extensionless names get the counter at the end.
        std::fs::write(dir.join("README"), b"c").unwrap();
        assert_eq!(dedup_in_dir(&dir, "README"), "README-001");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod projection_schema_tests {
    use super::{build_projection_builder, fmt_selection, system_prompt_labels};
    use candle_conversation::projection::SelectionRule;
    use std::path::Path;

    /// `fmt_selection` renders every selection rule as the viewer's one-liner.
    #[test]
    fn selection_rules_render_for_the_viewer() {
        assert_eq!(fmt_selection(&SelectionRule::AlwaysVisible), "all");
        assert_eq!(fmt_selection(&SelectionRule::TopK { k: 3 }), "top 3");
        assert_eq!(fmt_selection(&SelectionRule::Single), "single");
        assert_eq!(
            fmt_selection(&SelectionRule::Named {
                selector: "effort".into()
            }),
            "named(effort)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::Sequence {
                recent: 16,
                historical_top_k: 8,
            }),
            "recent 16 + top 8"
        );
    }

    /// The dialogue layer's system-prompt labels name its framing sections, its
    /// tools collection (with a member count), and its section tree — the
    /// structure the viewer lists under "Target layer".
    #[test]
    fn dialogue_system_prompt_labels_cover_sections_collection_and_tree() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let dialogue = builder
            .schema()
            .layers
            .iter()
            .find(|l| l.name == "dialogue")
            .expect("dialogue layer present");
        let labels = system_prompt_labels(&dialogue.system_prompt);

        assert!(
            labels.iter().any(|l| l == "history_stance"),
            "a bare framing section is named verbatim: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l == "tools · collection"),
            "the tools collection is labelled (members listed separately): {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l.starts_with("section tree (")),
            "the adaptive-thinking section tree is labelled: {labels:?}"
        );
    }

    /// The shipped `projection.yaml` parses, and the reconstructed repo map is
    /// capped and floored: `structure` is a `top_k(3)` group with a `"."`
    /// default so the workspace-root cluster always survives selection.
    #[test]
    fn projection_yaml_parses_and_repo_map_is_capped_with_default() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let structure = builder
            .id_for_group("structure")
            .expect("repo_map declares a 'structure' group");
        let group = builder.group(structure).expect("group schema present");

        match &group.selection {
            SelectionRule::TopK { k } => assert_eq!(*k, 3, "repo map capped at 3 clusters"),
            other => panic!("structure should be top_k(3), got {other:?}"),
        }
        assert_eq!(
            group.default.as_ref().map(|d| d.tag.as_str()),
            Some("."),
            "repo_map default floor is the workspace-root cluster",
        );
    }
}

#[cfg(test)]
mod held_tail_tests {
    use super::render_held_tail;

    #[test]
    fn wraps_a_bare_tool_call_so_the_gui_renders_a_card() {
        // The 584 shape: bare object, `parameters` key — the GUI needs the wrapper.
        let held = r#"{"name":"datetime","parameters":{"timezone":"Australia/Sydney"}}"#;
        let out = render_held_tail(held).expect("held tail");
        assert!(out.starts_with("<tool_call>"), "not wrapped: {out}");
        assert!(
            out.trim_end().ends_with("</tool_call>"),
            "not closed: {out}"
        );
        assert!(out.contains("Australia/Sydney"), "lost the args: {out}");
    }

    #[test]
    fn leaves_a_non_tool_json_object_verbatim() {
        // A bare object that isn't a registered call must stream as plain text.
        let held = r#"{"answer": 42, "note": "not a tool"}"#;
        let out = render_held_tail(held).expect("held tail");
        assert!(!out.contains("<tool_call>"), "wrongly wrapped: {out}");
        assert_eq!(out, held);
    }

    #[test]
    fn empty_tail_emits_nothing() {
        assert!(render_held_tail("   \n  ").is_none());
    }
}

#[cfg(test)]
mod title_tests {
    use super::clean_title;

    #[test]
    fn strips_empty_no_think_block() {
        // Qwen3 under /no_think still emits an empty think block before the label.
        assert_eq!(
            clean_title("<think>\n\n</think>\n\nUnbounded Context Engine"),
            "Unbounded Context Engine"
        );
    }

    #[test]
    fn strips_reasoned_think_block() {
        assert_eq!(
            clean_title("<think>The user is asking about X, so a good title is…</think>\nProjection Bar Rendering"),
            "Projection Bar Rendering"
        );
    }

    #[test]
    fn keeps_existing_cleanup() {
        // Think-strip composes with the prefix/quote/punctuation trimming.
        assert_eq!(
            clean_title("<think></think>Title: \"KV Cache Tiering.\""),
            "KV Cache Tiering"
        );
        assert_eq!(clean_title("Plain Title"), "Plain Title");
    }
}
