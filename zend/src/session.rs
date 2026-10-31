use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use agent::client::Agents;
use agent::jobs::Jobs;
use arc_swap::ArcSwapOption;
use cluster::member::{AppMember, ClusterView, Link};
use futures::executor::block_on;
use futures::future::select_all;
use futures::{Stream, StreamExt};
use host::cluster::store::ConversationStore;
use host::cluster::view::{
    admitted, catchup_detail, catchup_progress, connecting_detail, model_id,
};
use host::cluster::Join;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::Mutex as ConvLock;
use tokio::sync::{watch, Notify, OwnedMutexGuard};
use tokio::task::JoinHandle as TaskHandle;

use candle::{vram, Device};
use candle_conversation::models::{Dialect, Model, StrataTokens};
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::persistence::{
    content_hash, OpenProgressHook, OpenStage, SharedSubstrate, SUBSTRATE_DIR,
};
use candle_conversation::projection::{
    self, Builder, Conversation, DecodePriority, GroupId, LayerId, Reserved, SectionId,
    SectionLoads, SelectionRule, SystemItem, SystemPromptItem, TimelineId, TurnIndex,
};
use candle_conversation::provenance::gallery_arena::GroupSet;
use candle_conversation::provenance::FOLD_GROUPS;
use candle_conversation::stencil::{
    compile_action_loop, ThinkMode, ToolCallEnvelope, ToolSpec, TriggerRegistry,
};
use candle_conversation::substrate::pcoll::POrdMap;
use candle_conversation::substrate::Substrate;
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::working_set::marks::RELEASE_TAG;
use candle_conversation::working_set::WorkingSetConfig;
use candle_conversation::FinishReason;
use candle_conversation::LoadPhase;
use candle_conversation::RecoveredMessage;
use candle_conversation::Result as EngineResult;
use candle_conversation::Role as TurnRole;
use candle_conversation::TurnText;
use candle_conversation::{
    CancelToken, Carried, ConversationEngine, ConversationError, EngineSource, GlueMarkers,
    GroupScanner, Linking, OptionalState, ProjectionEvent, PromptDump, SamplingConfig,
    SelectionState, Sequence, ThinkSteering, TokenDecoder, TurnEvent, TurnHandle, TurnResponse,
    UnsealedAsker, Want,
};
use host::download::ensure_model_pack;
use host::process::fatal;
use host::substrate_views::{
    layer_conversations, overview, system_prompt, timeline, title, LayerConversations,
    SubstrateOverview, SystemPromptView, TimelineDetail, Unit,
};
use journal_core::entry::Entry;
use journal_core::fold;
use journal_core::folding::{self, Folded};
use journal_core::journal::Holds;
use journal_core::record::view as journal_view;
use journal_core::section::{JournalPrompt, Kept, JOURNAL};
use journal_core::span::Span;
use journal_core::transcript::{Filed, Hits, Miss, Page};
use serde_json::{json, Value};
use web::auth::Roles;
use zend_tools::state::Secrets;
use zend_tools::tools::journal::{JournalBook, JournalShelf};
use zend_tools::SubagentRunner;
use zend_vfs::{RepoFiles, Workspace};

use crate::api::chat::{
    apply_tools_dial, dial_selection, tool_round_selection, EFFORT_OPTIONS,
    RESPONSE_LENGTH_OPTIONS, TOOL_EXAMPLE_SELECTOR,
};
use crate::api::substrate::{
    ProjectTile, ProjectView, SelectedNodeView, SelectionView, SystemPromptRetired, ToolView,
    ToolsView,
};
use crate::branch_ingest::filter::IngestScope;
use crate::branch_ingest::keys::{BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY};
use crate::branch_ingest::{moves_branches, BranchIngest, LayerPass};
use crate::calibrate::{calibrate_layer, Frame as CalibrateFrame, Layer as CalibrateLayer};
use crate::code_read::{FILE_READ_TOOL, PATH_KEY};
use crate::coding_sampling;
use crate::config::{DaemonConfig, SelfCheck};
use crate::conv_branches::{self, BaseBranches};
use crate::conv_order;
use crate::conv_overlay::{self, Mirror};
use crate::conv_sweep::{CONV_IDLE_TTL, CONV_SWEEP_EVERY};
use crate::deletions::Deletions;
use crate::engine_asks;
use crate::fast_path::coverage::Coverage;
use crate::fast_path::{self, Screen};
use crate::ingest::{IngestConv, IngestLayer, IngestMode};
use crate::ingest_lease;
use crate::journal::desk::{Asking, Compile, Record, ZendDesk, FOLD_TOKENS, WRITE_TOKENS};
use crate::journal::record::{kept_in, metadata_of};
use crate::journal::registry::{Journals, Open};
use crate::journal::shelf::{Above, Drafted, Lazy, Notes, Shelf};
use crate::journal::state::{SessionJournal, FOLD_SWEEP_EVERY};
use crate::journal::style::style_for;
use crate::journal::text::JournalText;
use crate::journal::world::{Citable, JournalsAbove};
use crate::journal::{carry, draft, origin, pressure};
use crate::layers::keys::SUBJECT_KEY;
use crate::live_turn::{LiveTurn, LiveTurns};
use crate::loading::{LoadProgress, LoadStep, LoadingSnapshot};
use crate::mcp::ownership::owner_of;
use crate::mcp::state::McpState;
use crate::mcp::threads::{conv_names, project_journal};
use crate::mcp::{archiving, lineage, names, nudges, resuming, stops, threads};
use crate::passthrough::{self, Exchange, LiveConv, PassthroughCache, Transcript};
use crate::projection_event::{ProjectionEventOut, ProjectionSpanOut};
use crate::refresh_ctx::RefreshContext;
use crate::repeat_guard::{self, RepeatGuard, Verdict};
use crate::repo_ads::repo_ads;
use crate::repo_path::shown as shown_unit;
use crate::repo_scan::{utility_config, DIR_KEY};
use crate::resident::Resident;
use crate::resume;
use crate::retrieval_scope::RetrievalScope;
use crate::self_check::{self, Report as SelfCheckReport};
use crate::substrate_git::{self, Repositories};
use crate::think_budget;
use crate::think_progress::{ThinkProgress, ThinkUpdate};
use crate::tool_round;
use crate::tools::{
    extract_tool_calls, format_tool_responses, install_tool_catalog, run_tool_calls, Dispatch,
    ToolCall, ToolHost, ToolResult, CALIB_TOOL_SELECTOR,
};
use crate::turn_claims;
use crate::types::{ChatMessage, Role, ToolMode, Usage};
use crate::working_set::promises::{self, Release};
use crate::working_set::restore::restore_locks;
use crate::working_set::seeds::Seeding;
use crate::working_set::{dialogue_config, round_marks};
use crate::workspace::UPLOADS_REPO;

mod memory_writer;
mod replay;
pub use replay::{
    Demonstration, RecordedTurn, ReplayOutcome, ReplayReport, ReplaySampling, ReplaySpec,
    TurnComparison,
};

pub const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("prompts/projection.yaml");

/// Bound on the titler queue. A backlog beyond this many un-started titles
/// means the worker can't keep up; further jobs are dropped (labels are
/// best-effort), which is preferable to unbounded growth under a burst.
const TITLER_QUEUE_DEPTH: usize = 256;

/// How often a pass is asked for while a layer changes with time rather than
/// with its branches: a day layer reads a day that closed
/// (`crate::layers::days`), and a layer that expires reads again the units
/// whose time is up (`crate::layers::expiry`). No branch move announces
/// either, and a unit stands at most this long past its time.
const LAYER_CLOCK_EVERY: std::time::Duration = std::time::Duration::from_secs(30 * 60);

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

// The schema-driven ingest registry lives in [`crate::ingest`]; a raw layer's
// live state ([`IngestConv`]) is keyed by layer name in
// `InferenceState::ingest_convs`. The branch layers (`repo_map`,
// `code_reading`) keep no live state: each pass reads what they hold from the
// substrate ([`crate::branch_ingest`]).

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
    /// Each result's length in tokens as it will sit in the context, in the
    /// same order — encoded as the next turn will encode it, so it is the length
    /// the reloaded history shows for that result. `"done"` notice only.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<usize>,
}

/// Items yielded by [`ZendSession::submit`].
#[derive(Clone)]
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
    /// How far the prefill of the turn's input has got, in tokens. A tool
    /// round's results can run to tens of thousands of tokens, and this is the
    /// only sign of progress the GUI has before the answer starts. Display-only.
    Prefill {
        done: usize,
        total: usize,
    },
    /// The turn's reasoning so far, in generated tokens — sent while its
    /// `<think>` block is open, and once more (`done`) with the total when it
    /// closes. Counted as the turn records it (see `think_progress`), so the
    /// total is what a reloaded history shows. Display-only.
    Think {
        tokens: usize,
        done: bool,
    },
    /// A finished turn: its token counts, which become the reply's `usage`, and
    /// how it ended, which becomes its `finish_reason`. Sent once per turn; a
    /// reply the daemon answers over several turns sends one for each and ends
    /// the way its last turn did.
    TurnEnd {
        usage: Usage,
        finish: FinishReason,
    },
    /// The turn failed: what the engine reported, worded for the user. Its own
    /// item rather than text, because text lands wherever the reply has got to
    /// — inside an open `<think>` block, where the GUI folds it away — and an
    /// error is not part of the reply. A streamed reply sends it in-band; a
    /// collected one answers `500` with it, since a body-reading client has the
    /// status to go on.
    Error(String),
    /// The repeat guard or the round cap closed the turn's tool loop, so the
    /// turn that follows answers without tools — sent once, before it, naming
    /// which ([`CLOSED_BY_REPEATS`], [`CLOSED_BY_CAP`]). A task `/mcp` started
    /// reads it as stuck (`crate::mcp::workers`); the chat stream shows nothing
    /// for it.
    Closing(&'static str),
}

/// [`StreamItem::Closing`]'s reason when the repeat guard closed the loop.
pub const CLOSED_BY_REPEATS: &str = "repeats";
/// [`StreamItem::Closing`]'s reason when the round cap closed it.
pub const CLOSED_BY_CAP: &str = "round_cap";
/// [`StreamItem::Closing`]'s reason when Claude stopped the worker
/// (`repo_task_stop`, [`stops`]).
pub const CLOSED_BY_CLAUDE: &str = "stopped";

/// A dialogue conversation's recovered history, as `GET /v1/conversations/{id}`
/// shows it — the summariser's ghost summary turns hidden.
fn dialogue_history(state: &InferenceState, conv_id: &str) -> Vec<RecoveredMessage> {
    let timeline = timeline_for(conv_id);
    let base = &state.base_conv;
    base.recovered_history(timeline, false)
}

/// Forward a turn's items to the request that started it, recording each one
/// in `live` for a client that reconnects.
///
/// Drains `ts` to the end whether or not the request is still there: the turn
/// runs to completion either way, and a client that comes back mid-turn follows
/// it from `live`. An error item is recorded as the turn error the request
/// showed for it, so a follower sees the same failure.
async fn relay_live(
    ts: &mut Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>>,
    tx: &mpsc::Sender<anyhow::Result<StreamItem>>,
    live: &LiveTurn,
    conv_id: &str,
) {
    let mut listening = true;
    while let Some(item) = ts.next().await {
        live.push(match &item {
            Ok(i) => i.clone(),
            Err(e) => StreamItem::Error(e.to_string()),
        });
        if listening && tx.send(item).await.is_err() {
            listening = false;
            tracing::info!(
                conv_id,
                "client went away mid-stream — the turn runs to completion and a \
                 reconnecting client follows it live"
            );
        }
    }
}

/// Process-global monotonic id for projection events, so dot ids stay unique
/// across conversations and requests.
static PROJ_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// ── Inference state ───────────────────────────────────────────────────────────

struct ConvState {
    conv: Sequence,
    /// The identity this conversation speaks as (a sub-folder of `identities/`),
    /// or `None` for the default / no identity. Seeded from the substrate on
    /// fork, overridden when a request carries an explicit `identity`, and used
    /// each turn to scope the projection's identity collection.
    identity: Option<String>,
    /// This conversation's own file stores: what its `file_*` calls write,
    /// edit and delete, kept as deltas over each repository and seen by no other
    /// conversation. Every round of its tools runs against these
    /// ([`ToolHost::context_for`]); each round's changes are saved with the
    /// conversation and restored when this state is built again
    /// ([`conv_overlay`]).
    files: Arc<RepoFiles>,
    /// What the substrate holds of `files`, as events on the conversation's
    /// timeline — what each save writes the difference against. Unread until
    /// the conversation's events have been read back, and while unread
    /// nothing is saved.
    mirror: Arc<Mutex<Mirror>>,
    /// The journal sections this conversation holds, reconciled with its
    /// journal at the head of every round (`docs/zend_journal.md`).
    journal: Carried,
    /// The working-set promises a decode found ended — locks whose replies its
    /// projection left out — marked on the next turn this conversation submits,
    /// whichever request that is (`working_set::promises`).
    promise_marks: Vec<String>,
}

/// Metadata keys carrying the composer dials a conversation last ran under.
///
/// The values are the projection's own option ids (`balanced`, `standard`, …),
/// not the GUI's levels: the daemon speaks in ids, the ids are what a resumed
/// turn re-selects, and a stored id keeps its meaning if the GUI ever changes
/// its scale. [`ZendSession::conversation_dials`] maps them back to levels for
/// the client.
const DIAL_EFFORT_KEY: &str = "dial_effort";

/// The conversation metadata key a failed turn's error is kept under — one per
/// turn, by the turn's substrate index, so every failure stays beside its own
/// turn ([`ZendSession::conversation_turn_errors`]).
fn turn_error_key(turn: u32) -> String {
    format!("{TURN_ERROR_PREFIX}{turn}")
}

/// The prefix of [`turn_error_key`].
const TURN_ERROR_PREFIX: &str = "turn_error.";

/// Every failed turn's error in a conversation's metadata, by turn index. A key
/// whose suffix is not a turn index is not one of these.
fn turn_errors(meta: &BTreeMap<String, String>) -> BTreeMap<u32, String> {
    meta.iter()
        .filter_map(|(k, v)| {
            let turn = k.strip_prefix(TURN_ERROR_PREFIX)?.parse().ok()?;
            Some((turn, v.clone()))
        })
        .collect()
}
const DIAL_LENGTH_KEY: &str = "dial_response_length";
const DIAL_THINK_KEY: &str = "dial_think";
const DIAL_TOOLS_KEY: &str = "dial_tools";

/// The most tool rounds one turn runs before the daemon stops feeding it.
///
/// The bound matters because nothing else provides one — a client that closes
/// its tab no longer ends the turn (that is deliberate: the work finishes and
/// seals so a reload finds it done), so a model calling a tool every round
/// would otherwise decode, seal and dispatch forever, holding the
/// conversation's lock and the card with nobody waiting on the result.
///
/// **Reaching it closes the loop; it never just stops.** A long job reaches it
/// too: measured 2026-10-24, a turn revising a 1,400-line paper made ~60
/// edits over 31 rounds, every one landing, and the turn ended on its last
/// call's results with nothing said to the user. The round before the cap is
/// answered by a closing turn ([`closes_at_cap`]), as the repeat guard's is:
/// no tool call, a bounded answer saying what was done and what remains.
///
/// **128, not 32.** A worker's restructure of the journals — dozens of reads,
/// edits and deletes, a preview, a push — met the cap of 32 at its thirtieth
/// round on 2026-10-31 and sat stuck waiting to be told to go on. The bound is
/// a guard against a turn that never ends, not a budget for one that is
/// working; the repeat guard still closes a loop that only repeats itself.
const MAX_TOOL_ROUNDS: usize = 128;

/// Whether the results of round `iteration` are the last a turn may answer
/// under [`MAX_TOOL_ROUNDS`] — so the turn answering them must close the loop.
fn closes_at_cap(iteration: usize) -> bool {
    iteration + 2 >= MAX_TOOL_ROUNDS
}

/// The dials `selection` and `tools_mode` express, as the metadata bag stores
/// them. A selector the turn left unset is stored empty, so "ran without this
/// dial" stays distinguishable from "never recorded".
fn dials_metadata(
    selection: &candle_conversation::SelectionState,
    tools_mode: ToolMode,
) -> BTreeMap<String, String> {
    let think = match selection.optional(candle_conversation::NO_THINK_SELECTOR) {
        // The `/no_think` node present means the turn suppressed thinking.
        Some(candle_conversation::OptionalState::Present) => "off",
        Some(candle_conversation::OptionalState::Absent) => "on",
        None => "",
    };
    BTreeMap::from([
        (
            DIAL_EFFORT_KEY.to_string(),
            selection
                .get("thinking_effort")
                .unwrap_or_default()
                .to_string(),
        ),
        (
            DIAL_LENGTH_KEY.to_string(),
            selection
                .get("response_length")
                .unwrap_or_default()
                .to_string(),
        ),
        (DIAL_THINK_KEY.to_string(), think.to_string()),
        (DIAL_TOOLS_KEY.to_string(), tools_mode.id().to_string()),
    ])
}

/// The composer dials a conversation last ran under, as the levels the GUI's
/// dials are set from.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ConversationDials {
    pub effort: u8,
    pub verbosity: u8,
    pub think: bool,
    pub tools: u8,
}

/// Map a turn's `thinking_effort` dial to the steering [`ThinkMode`].  Mirrors
/// `dial_selection` in `api/chat.rs`: effort 0 → `off`, whose tree closes the
/// block on the token after `<think>`; an unset dial defaults to the
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

/// The hard cap on one assistant turn, in generated tokens — the scheduler's
/// `max_tokens` for every turn that does not name its own.
///
/// A prose answer never gets near it: the think and answer budgets
/// ([`think_budget::steer`]) end one long before. It exists for the turn those
/// budgets stand down for — one writing a tool call, whose string values may
/// each run to
/// [`MAX_STRING_VALUE_TOKENS`](candle_conversation::stencil::MAX_STRING_VALUE_TOKENS)
/// so a whole file can be written. So it has to hold the longest think block and
/// one full value with room to spare, which the engine's default of 16k does
/// not.
pub(crate) const MAX_TURN_TOKENS: usize = 65_536;

/// Maps the composer's `response_length` dial (terse/concise/standard/detailed/
/// comprehensive = 0..4, the verbosity toggle) to the answer's token budget — the
/// room reserved after the think block, which the total-length EOS ramp covers
/// (`ThinkMode::eos_budget`).  These are generous hard caps (the typical answer is
/// shorter); the EOS boost/failsafe makes them the turn-ender backstop.
fn response_budget_from_selection(selection: &candle_conversation::SelectionState) -> i32 {
    match selection.get("response_length") {
        Some("terse") => 512,
        Some("concise") => 1024,
        Some("detailed") => 4096,
        Some("comprehensive") => 7168,
        // Explicit `standard`, or no dial set (projection default).
        _ => 2048,
    }
}

/// The projection a chat turn runs under: the tools mode's prebuilt view — a
/// cheap `Arc` clone, no per-turn schema clone — scoped to the conversation's
/// identity's anchor and facets when it names one (`IdentityBuilders`).
fn turn_projection(
    state: &InferenceState,
    conv_id: &str,
    identity: Option<&str>,
    tools_mode: ToolMode,
) -> Arc<Builder> {
    let scope = state.identity_builders.resolve(identity);
    let (proj, applied) = match &scope {
        IdentityScope::Named(name) => (
            state
                .identity_builders
                .get(&state.mode_builders, name, tools_mode),
            Some(name.as_str()),
        ),
        IdentityScope::Empty => {
            tracing::warn!(conv_id = %conv_id,
                "conversation names no identity and mind.yaml sets no default — \
                 scoping identity collections to empty (no anchor emitted)");
            (
                state
                    .identity_builders
                    .get_empty(&state.mode_builders, tools_mode),
                None,
            )
        }
        IdentityScope::Unscoped => (state.mode_builders.get(tools_mode), None),
    };
    tracing::info!(conv_id = %conv_id, ?tools_mode, identity = ?applied, "projection applied");
    proj
}

/// A chat turn's sampling config. A caller's explicit config (e.g. a test's
/// `argmax()`) is honoured verbatim; the conversation default gets the per-turn
/// adjustments and `seed`. Either way the dials' thinking and answer budget is
/// steered in: the effort level sets the think block's close thresholds, and
/// `response_length` the answer's room above them (see `think_budget::steer`).
fn turn_sampling(
    conv: &Sequence,
    explicit: Option<SamplingConfig>,
    selection: &SelectionState,
    think_closer: &[u32],
    seed: u64,
) -> SamplingConfig {
    let think_mode = think_mode_from_selection(selection);
    let mut sampling = match explicit {
        Some(explicit) => explicit,
        None => {
            let mut sampling = conv.default_sampling();
            // A `/no_think` turn is a direct response, not reasoning: strip the
            // thinking-temperature boost, which is meant for the `<think>` span
            // and has no business heating a plain answer.
            if think_mode == ThinkMode::Off {
                sampling.segment_temp_boost = 0.0;
            }
            sampling.seed = seed;
            sampling
        }
    };
    think_budget::steer(
        &mut sampling,
        think_mode,
        response_budget_from_selection(selection),
        think_closer,
    );
    sampling
}

/// Calibration questions per file a layer's chunk exclusion is learned from:
/// a background of ~3,300 questions over the workspace's ~1,650 files, ~410
/// scans of 16 questions run between waves. The outliers it cuts sit orders
/// of magnitude above the rest, so a sample finds them; the cut it learns at
/// this sample has not been measured against the six-a-file one.
const EXCLUSION_QUESTIONS_PER_FILE: usize = 2;

/// The triggers a chat turn decodes under: any `<tool_call>` the model emits is
/// forced to the exact JSON shape of a tool `tools_mode` offers (name ∈ that
/// mode's catalog, required params in order, valid JSON), and atop that the `<think>` block is steered per the
/// effort dial, replacing the `<think>` trigger atomically.
fn turn_triggers(
    state: &InferenceState,
    think_mode: ThinkMode,
    tools_mode: ToolMode,
) -> Arc<TriggerRegistry> {
    let stencil = &state.tool_stencils[tools_mode.level() as usize];
    match &state.think_steering {
        Some(ts) => ts.registry_for(stencil, think_mode),
        None => Arc::clone(stencil),
    }
}

/// What each layer of units compounds on: every other layer of units below
/// it in the schema's stack, by name, with its group and whether that group is
/// scoped by branch (a layer of files) or not (a layer of days). A disabled
/// layer is out of service: it compounds on nothing and nothing on it.
type LayerSees = Vec<(String, Vec<(String, GroupId, bool)>)>;

fn layer_sees(builder: &Builder, layers: &[IngestLayer], disabled: &HashSet<String>) -> LayerSees {
    let schema = builder.schema();
    let rank = |name: &str| {
        schema
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.rank)
    };
    let derived: Vec<(&IngestLayer, i32, GroupId, bool)> = layers
        .iter()
        .filter(|il| il.mode == IngestMode::Derived && !disabled.contains(&il.name))
        .filter_map(|il| {
            let scoped = il
                .spec
                .as_ref()
                .is_some_and(|s| s.unit == crate::layers::UnitKind::File);
            Some((
                il,
                rank(&il.name)?,
                builder.id_for_group(&il.group)?,
                scoped,
            ))
        })
        .collect();
    derived
        .iter()
        .map(|(il, r, _, _)| {
            let below = derived
                .iter()
                .filter(|(other, other_rank, _, _)| other.name != il.name && other_rank < r)
                .map(|(other, _, group, scoped)| (other.name.clone(), *group, *scoped))
                .collect();
            (il.name.clone(), below)
        })
        .collect()
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
    /// `scheduler_tx` senders) and the substrate persistence handle. Every
    /// engine method takes `&self` and no reader waits on another, so it is
    /// shared bare.
    engine: ConversationEngine,
    /// The substrate, cloned out of the engine when it was built — what
    /// `/v1/status` reads maintenance from.
    substrate: Conversation,
    /// Per-conversation state, keyed by the client-supplied conv_id string, with
    /// when each was last used — what the idle sweep releases them by
    /// ([`InferenceState::sweep_idle_conversations`]).
    ///
    /// The map takes no lock (`crate::resident`). The per-conversation lock is
    /// async, and the inference task holds it across its decode awaits
    /// DELIBERATELY — that guard is the per-conversation in-flight
    /// exclusivity, and an async lock is what lets it live across an await
    /// without pinning a thread.
    conversations: Resident<ConvLock<ConvState>>,
    /// The tools mode last applied to each conversation's projection, keyed like
    /// `conversations` and set at the start of every turn. The projection panel
    /// reads it to show the tool summary that was actually injected — the
    /// restricted (safe-subset) summary in Restricted mode, the full one in
    /// Comprehensive, none in None. Kept outside the per-conversation lock
    /// because that lock is held for a whole turn, tool rounds included: a panel
    /// reading the mode through it waited for the turn to finish.
    tool_modes: Mutex<HashMap<String, ToolMode>>,
    /// System-prompt already prefilled; all new conversations fork from this.
    /// Every read and every fork takes it by `&self`.
    base_conv: Sequence,
    /// The layer `base_conv` targets, named once at boot — what the substrate
    /// views report as the target.
    target_layer: String,
    /// One prefilled template per `Folders`/`Files` ingest layer (`repo_map`,
    /// `code_reading`), keyed by layer name — `base_conv`'s exact counterpart
    /// for background ingestion. Built once at boot, alongside `base_conv`,
    /// and forked per unit (per directory, per file) by the ingest pools
    /// instead of each independently re-running the schema's "eager section
    /// ingestion" the way `base_conv` itself only ever does once. A live
    /// conversation and an ingest conversation used to construct themselves
    /// two different ways — one forking a shared, already-computed prefix,
    /// the other re-prefilling its own copy under concurrent load from up to
    /// `CODE_READ_PARALLELISM` workers at once — despite both claiming to
    /// "have no real difference from a normal one except hidden". Forking
    /// the same way dialogue does removes that difference for real. `Raw`
    /// layers are excluded: they only ever prefill records and never decode,
    /// so they have no reasoning to protect and stay on their existing
    /// from-scratch construction (`raw_read::ingest_raw`).
    ingest_bases: HashMap<String, Sequence>,
    /// The priming chain's final link (`BranchIngest::prime`, built once per
    /// boot before `ready`): `base_conv`'s parent, and the parent every ingest
    /// unit's conversation records before its reading starts. `None` when
    /// there is no chain — no git repository, or no ingest layer.
    chain_end: Mutex<Option<TimelineId>>,
    /// Queue feeding the dedicated titler task. The request path enqueues a
    /// [`TitleJob`] (non-blocking, dropped if the task is backed up) instead
    /// of spawning per submit, so title generation runs in the background —
    /// concurrently with main decode, never serialised against the request
    /// path — and drains cleanly on shutdown.
    titler_tx: mpsc::Sender<TitleJob>,
    /// Handle to the titler task, awaited during [`ZendSession::shutdown`]
    /// so any in-flight title turn unwinds before the process exits.
    titler_worker: Mutex<Option<TaskHandle<()>>>,
    /// The memory's writer (`memory_writer`), awaited during
    /// [`ZendSession::shutdown`] once `memory_stop` is requested, while the
    /// scheduler still answers.
    memory_writer: Mutex<Option<TaskHandle<()>>>,
    /// What the memory's writer stops on — between jobs, and mid-turn.
    memory_stop: CancelToken,
    /// The titler's timeline id — excluded from `list_conversations` so
    /// it doesn't show up in the user-facing sidebar.
    titler_timeline: TimelineId,
    /// Set once shutdown begins. The titler worker checks it between jobs and
    /// stops draining; the request path stops enqueuing new title jobs.
    shutting_down: AtomicBool,
    /// Live raw-layer ingest state, keyed by projection layer name. A raw
    /// layer absent here is not refreshed. A projection with no raw layers
    /// leaves this empty.
    ingest_convs: Mutex<HashMap<String, IngestConv>>,
    /// The schema's ingest layers in declaration order (identity + strategy +
    /// display label), resolved once at load.
    ingest_layers: Vec<IngestLayer>,
    /// The branch layers this boot ingests — `repo_map` and `code_reading`,
    /// less any `--disable-layer` names — with the scope each is walked under.
    branch_layers: Vec<(IngestLayer, IngestScope)>,
    /// For each layer of units, the scoped groups of the layers of units below
    /// it, by layer name — what its units retrieve on their own branches.
    layer_sees: HashMap<String, Vec<(String, GroupId)>>,
    /// The branch ingest's state between passes (`crate::branch_ingest`).
    branch_ingest: BranchIngest,
    /// Which ingested units each conversation may retrieve
    /// (`crate::retrieval_scope`), set on the engine before every turn.
    retrieval_scope: RetrievalScope,
    /// The ingest worker's wake, once it is running: a `git_*` tool round
    /// that can move a branch wakes it, since a push moves the tracking refs
    /// itself and no origin probe will see a difference afterwards.
    ingest_wake: OnceLock<Arc<Notify>>,
    /// Raised while the branch layers' chunk exclusions are being learned on
    /// their own thread ([`InferenceState::learn_chunk_exclusions_behind`]),
    /// so a pass ending while one learns does not start a second.
    learning_exclusions: AtomicBool,
    /// The dialogue layer's working set (`docs/zend_working_set.md`), its
    /// `release_on` names checked against the tool registry at load. `None`
    /// when the schema declares none: nothing is seeded, locked or released,
    /// and every tool call runs for real.
    working_set: Option<WorkingSetConfig>,
    /// Whether each `code_reading` chain read its whole file — asked before a
    /// file is served or seeded, answered once per chain.
    read_coverage: Coverage,
    /// The workspace captured at startup — its folder and repositories.
    workspace: Workspace,
    /// The branch a conversation starts on in each git repository — `main`,
    /// else `master`, else the checkout — read once at startup. A
    /// conversation is given these for every repository it has no branch in
    /// yet: every live one at startup, and each new one as it is created.
    /// See [`conv_branches`].
    base_branches: BaseBranches,
    /// Projection builder + sequence config kept alive for the
    /// atomic refresh paths.  Minting a fresh ingest-layer timeline
    /// after a file change reuses the same schema clone and the same
    /// dialect config the initial ingestion ran under.
    refresh_builder: Builder,
    refresh_config: candle_conversation::SequenceConfig,
    /// The dialect-formatted system-prompt prelude every real conversation
    /// primes on — the exact text `base_conv` was built from. Threaded into
    /// [`RefreshContext`] so `code_reading`'s hidden per-file conversations
    /// frame identically to a live dialogue turn instead of a bespoke
    /// ingest-only prompt.
    formatted_prompt: String,
    /// Per-tools-mode projection builders, built once at startup (Restricted
    /// drops the high-risk tools; None drops the whole catalog). Each turn hands
    /// out the matching one as a cheap `Arc` clone via [`ModeBuilders::get`].
    mode_builders: ModeBuilders,
    /// Lazy per-`(identity, tool_mode)` projection builders. Scopes a
    /// conversation's projection to its identity's anchor + facets over the
    /// tools-mode view. Empty `keeps` (no `identities/` folder) makes it a no-op.
    identity_builders: IdentityBuilders,
    /// Tokenizer shared with the engine — used by `head_tail_truncate`
    /// to bound the titler's prefill at first 50 + last 50 tokens.
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// Shared tool execution context (notes / credentials / sessions /
    /// VFS stores).  Cloned per-conversation if scoping is later
    /// needed; for now it's workspace-wide.
    tool_host: ToolHost,
    /// This load's repositories and their follower (`docs/substrate_git.md`
    /// §7); `None` when the workspace names none. Stopped by
    /// `ZendSession::shutdown` before the drain.
    repositories: Option<Repositories>,
    /// Constrained-decoding stencils, one per tools mode (indexed by
    /// [`ToolMode::level`]), each keyed by the `<tool_call>` trigger token and
    /// compiled once at startup from the tools that mode offers
    /// ([`crate::tools::catalog_for`]). A turn decodes under its mode's, so a
    /// tool call is forced to an offered tool's exact JSON shape.
    tool_stencils: [Arc<TriggerRegistry>; ToolMode::ALL.len()],
    /// The Restricted stencil with every call held to `file_read` — what a
    /// file's reading decodes under, so it can call no other tool. Built once
    /// at startup beside the stencils it narrows.
    read_stencil: Arc<TriggerRegistry>,
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
    /// Live OpenAI-passthrough conversations — see [`crate::passthrough`].
    passthrough: Arc<PassthroughCache>,
    /// Moved by every conversation delete, so a reply running on the deleted
    /// conversation ends mid-round rather than at its next one.
    deletions: Deletions,
    /// Whether the schema declares a journal (`journal::carry::install`). A
    /// workspace schema without one journals nothing.
    journaling: bool,
    /// Every dialogue's journal (`docs/zend_journal.md`).
    journals: Journals,
}

// The titler uses this plain system prompt (not the projection schema), so the
// dialogue `no_think` *section* never applies. Suppression is the `<think>`
// trigger bound to `ThinkMode::Off` in `generate_one_title`, which is structural
// and cannot be ignored — see there for why the soft switch is not enough.
//
// No `/no_think` here. It read as a system-side switch and was never one: the
// marker only does anything on a dialect that declares it
// (`Dialect::thinking_suppression`), this family declares the empty string, and
// on the families that do declare it, it is honoured from the *user* turn rather
// than the system prompt. It was prose to every model that ever saw it.
const TITLER_SYSTEM_PROMPT: &str = "You write short conversation titles. \
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
/// block's HARD token cap fires mid-sentence — the em-dash lead-in reads as
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
    let mut ids = Vec::new();
    for item in &builder.schema().system_prompt.items {
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

/// One unit of calibration work — one conversation, one submission, one resume
/// marker.
///
/// The two kinds differ in how many exemplars a submission produces, which is
/// the whole point of the split: a trajectory is one turn, a question group is
/// one prefill carved into as many turns as it has questions.
enum CalibCase<'a> {
    /// One authored trajectory from the tool's definition file: prompt +
    /// `<think>…</think>` + the call. Prefilled and sealed on its own.
    Trajectory {
        tool: &'a str,
        /// Borrowed from the definition, which outlives this list — the authored
        /// trajectories are the bulk of the corpus by bytes and copying them all
        /// would be pure waste.
        text: Cow<'a, str>,
        marker: String,
    },
    /// EVERY question exemplar of one tool, prefilled in a single forward and
    /// carved into one turn each.
    Questions {
        tool: &'a str,
        questions: Vec<String>,
        marker: String,
    },
}

impl CalibCase<'_> {
    fn tool(&self) -> &str {
        match self {
            CalibCase::Trajectory { tool, .. } | CalibCase::Questions { tool, .. } => tool,
        }
    }

    fn marker(&self) -> &str {
        match self {
            CalibCase::Trajectory { marker, .. } | CalibCase::Questions { marker, .. } => marker,
        }
    }

    /// How many provenance exemplars this submission produces — one per turn it
    /// seals. Progress is reported in exemplars rather than submissions so the
    /// figure keeps meaning the same thing it did before grouping.
    fn exemplars(&self) -> usize {
        match self {
            CalibCase::Trajectory { .. } => 1,
            CalibCase::Questions { questions, .. } => questions.len(),
        }
    }
}

/// A calibration submission in flight, awaiting its `Done`.
struct InFlightCase<'a> {
    conv: Sequence,
    handle: TurnHandle,
    tool: &'a str,
    /// Mid-decode reprojection events, accumulated across sweeps — they arrive
    /// interleaved with tokens, so a sweep that finds no terminal event still
    /// has to keep what it drained.
    events: Vec<ProjectionEvent>,
    /// A terminal event the idle wait absorbed: the wait parks on every
    /// in-flight handle and consumes exactly one event from whichever speaks
    /// first, so a `Done`/`Error` it takes has to be carried to the retire
    /// sweep rather than lost, and so does the error a failed turn carries.
    finished: Option<Result<TurnResponse, ConversationError>>,
    /// A prefilled trajectory is complete by construction (it is the validated
    /// exported `.md`) and its `Done` carries no decoded text, so it must NOT be
    /// gated on `resp.text` the way a live decode is.
    is_prefill: bool,
    /// How many exemplars this submission produces — one for a trajectory, one
    /// per question for a stuffed group. Progress is reported in exemplars, so
    /// crediting a 24-question group with a single step would stall the bar for
    /// a twenty-fourth of the corpus and then jump.
    exemplars: usize,
}

/// Wall-time attribution for the calibration phase.
///
/// The phase drives a batched wave loop from a single thread, so every bucket
/// below is time the GPU is *not* being fed: the window cannot refill and no
/// forward is in flight while the loading thread sits in one of these calls.
/// Measured against forward time the phase's prefill runs at ~6.9k tok/s — at
/// the batched-forward gate's rate — so its wall-clock rate is set entirely by
/// how much of the phase is spent here rather than by the model.
///
/// Each bucket names an engine call made once per calibration case. The
/// substrate ones (`tag`, `archive`, `distill`) each take the substrate write
/// lock and append a record under the persistence mutex, which is the same lock
/// the scheduler's wave loop needs to admit the next case.
#[derive(Default)]
struct CalibTiming {
    /// `new_conversations_with_projection_batch` — one pipelined round-trip per window.
    create: Duration,
    /// `set_conversation_metadata` — resume tag, per case.
    tag: Duration,
    /// `submit_prefilled_turn` / `submit_turn_with_options`, per case.
    submit: Duration,
    /// `finish_turn` — the seal, per case.
    seal: Duration,
    /// `persist_projection_events` — the trajectory's projection sequence, per case.
    persist_events: Duration,
    /// `set_conversation_archived` — the atomic done-marker, per case.
    archive: Duration,
    /// `distill_timeline` (+ its two state probes), per case.
    distill: Duration,
    /// `demote_timelines_hot` — incremental and boundary hot→warm sweeps.
    demote: Duration,
    /// The poll-loop yield taken when a pass retired nothing.
    idle: Duration,
}

impl CalibTiming {
    /// Run `f`, adding its wall time to the bucket `pick` selects.
    fn time<T>(&mut self, pick: fn(&mut Self) -> &mut Duration, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        *pick(self) += t.elapsed();
        out
    }

    fn log(&self, wall: Duration) {
        let ms = |d: &Duration| d.as_millis() as u64;
        let tracked = self.create
            + self.tag
            + self.submit
            + self.seal
            + self.persist_events
            + self.archive
            + self.distill
            + self.demote
            + self.idle;
        tracing::info!(
            wall_ms = ms(&wall),
            create_ms = ms(&self.create),
            tag_ms = ms(&self.tag),
            submit_ms = ms(&self.submit),
            seal_ms = ms(&self.seal),
            persist_events_ms = ms(&self.persist_events),
            archive_ms = ms(&self.archive),
            distill_ms = ms(&self.distill),
            demote_ms = ms(&self.demote),
            idle_ms = ms(&self.idle),
            tracked_ms = ms(&tracked),
            untracked_ms = ms(&wall.saturating_sub(tracked)),
            "calibration wall-time attribution"
        );
    }
}

fn mark_calibration_distill(
    engine: &ConversationEngine,
    timeline: TimelineId,
    tool: &str,
    timing: &mut CalibTiming,
) {
    timing.time(
        |t| &mut t.distill,
        || {
            if engine.timeline_has_kv(timeline) && !engine.is_timeline_distilled(timeline) {
                // Calibration exemplars stay retrievable by signature — provenance-only.
                if let Err(e) = engine.distill_timeline(timeline, DistillMode::ProvenanceOnly) {
                    tracing::warn!(tool = tool, "calibration distill mark failed: {e}");
                }
            }
        },
    );
}

fn seal_calibration_turn(
    conv: &mut Sequence,
    handle: TurnHandle,
    resp: &TurnResponse,
    tool: &str,
    mut events: Vec<ProjectionEvent>,
    timing: &mut CalibTiming,
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
/// How many of [`CALIBRATION_BATCH`]'s slots must be free before the window
/// refills. Creation is served between waves, so every refill costs one wave
/// boundary regardless of how many cases it creates — refilling at half the
/// window amortises that boundary over eight cases rather than paying it for
/// one. See the refill site for the measurement.
const CALIBRATION_REFILL: usize = CALIBRATION_BATCH / 2;
/// How many new calibration cases to submit this iteration. Split out of the
/// refill site so the decision is testable without a live engine — mirrors
/// `warm_sizing_from` being split out of `warm_slots_for` for the same reason.
///
/// The warm-up case (`!warmed`) always goes in alone, before anything else is
/// asked of the window: skipping it would leave the shared tool sections
/// unpinned and break every case after it.
///
/// Free host RAM is not an input. The phase runs the whole corpus; a host
/// that runs low is the OS's to page, not a reason to leave cases unrun.
fn calibration_window_want(free: usize, warmed: bool, inflight_empty: bool) -> usize {
    if !warmed {
        1
    } else if free >= CALIBRATION_REFILL || inflight_empty {
        free
    } else {
        0
    }
}

#[cfg(test)]
mod calibration_window_tests {
    use super::calibration_window_want;

    #[test]
    fn the_warm_up_case_goes_in_alone() {
        assert_eq!(calibration_window_want(16, false, true), 1);
        assert_eq!(calibration_window_want(16, false, false), 1);
    }

    #[test]
    fn the_ordinary_refill_cadence_holds() {
        // Half the window free: refills.
        assert_eq!(calibration_window_want(8, true, false), 8);
        // Fewer than half free, something still running: waits.
        assert_eq!(calibration_window_want(3, true, false), 0);
        // Nothing in flight regardless of the count: the tail case.
        assert_eq!(calibration_window_want(3, true, true), 3);
    }
}

/// Conversation-metadata key tagging each calibration conversation with its
/// `"{tool}|{example}"` case **at creation**, so it is findable by case on a
/// later load — finished or half-finished. *Done* is signalled separately by
/// **archiving** the conversation once its trajectory completes (`</tool_call>`);
/// a tagged but un-archived conversation is a decode that was cut off mid-case,
/// so the next load tombstones and regenerates it (the archive is the atomic
/// commit). Changing an example's text changes its marker, so edited examples
/// regenerate automatically.
const CALIB_MARKER_KEY: &str = "calib";

/// The last maintenance operation the redo log ran: `(op_name, unix_millis)`.
/// `None` when nothing has run since the log was opened.
pub type LastMaintenanceOp = Option<(String, u64)>;

/// Redo-log maintenance state for `/v1/status`:
/// `(segment_count, last_op, compaction_in_progress)`.
pub type MaintenanceStatus = (usize, LastMaintenanceOp, bool);

/// Whether a stop has ended the load at the boundary after `step` — and if
/// so, the engine drained here, on the loader thread that owns it, exactly as
/// the startup-ingest cancel drains it. The caller then returns `Ok(None)`: a
/// stopped load is a stop, not a failure, and what each step sealed so far is
/// durable while what it left half-built is found and redone next start (a
/// calibration case un-archived, an ingest unit uncommitted).
fn load_stopped(engine: &ConversationEngine, step: &str) -> bool {
    if !crate::ingest_cancel::cancelled() {
        return false;
    }
    tracing::warn!("shutdown after {step} — draining the engine on the loader thread");
    if let Err(e) = engine.shutdown() {
        tracing::error!("shutdown after {step}: drain failed: {e}");
    }
    true
}

impl InferenceState {
    /// Release every resident conversation nothing has used for
    /// [`CONV_IDLE_TTL`] (`conv_sweep`), returning how many went. One a turn
    /// holds is in use now, so its clock restarts: a long turn is not idle the
    /// moment it finishes.
    ///
    /// The released states are dropped after the map's lock is let go — each
    /// frees its scheduler slot as it drops.
    /// Drop what this member keeps in memory of the conversation `key` names —
    /// a dialogue's `conv_id`, a passthrough transcript's key — once it no
    /// longer holds the conversation (`turn_claims`). Another member may write
    /// its next turn, which this member's cached sequence would not hold: the
    /// next turn served here resumes from the substrate, which does. A turn
    /// still holding the state keeps its own reference to it.
    fn forget_conversation(&self, key: &str) {
        drop(self.conversations.remove(key));
        // Its next turn may be another member's, so its journal is read again
        // from the record when this member next holds it.
        self.journals.forget(key);
        self.passthrough.forget(key);
    }

    fn sweep_idle_conversations(&self, now: Instant) -> usize {
        let released = self.conversations.sweep(now, CONV_IDLE_TTL);
        // A released conversation's slot goes with it, and so does the asker a
        // draft would have put its questions through.
        for (conv_id, _) in &released {
            self.journals.closed(conv_id);
        }
        released.len()
    }

    // Every argument is a distinct piece of deployment config read from the CLI
    // / config file at startup, used once here and never carried together
    // afterwards, so there is no struct for them to already belong to.
    #[allow(clippy::too_many_arguments)]
    fn load(
        mut proj_builder: Builder,
        model: Model,
        engine_source: EngineSource,
        device: Device,
        pack: PathBuf,
        workspace: Workspace,
        secrets: Arc<Secrets>,
        disabled_layers: HashSet<String>,
        wiped_layers: HashSet<String>,
        ingest_dirs: HashMap<String, String>,
        max_depth: Option<usize>,
        host_runtime: Option<Handle>,
        agent: Option<Arc<Jobs>>,
        compact_substrate: bool,
        self_check: SelfCheck,
        read_only_substrate: bool,
        qsa_selection_budget: Option<usize>,
        qsa_strata: StrataTokens,
        summarize: bool,
        tombstone_retention: Duration,
        cluster: Option<Join>,
        status_tx: watch::Sender<String>,
        progress: Arc<LoadProgress>,
        loading_substrate: Arc<OnceLock<Conversation>>,
        parts: Arc<dyn SubagentRunner>,
    ) -> anyhow::Result<Option<Arc<Self>>> {
        // The workspace folder: where the substrate, the schema override and a
        // mind's section folders live. Repositories are reached through
        // `workspace` itself.
        let root = workspace.root().to_path_buf();

        // The pack is found; the steps from here move as the work does — the
        // substrate open's hook enters `Opening`, the first weight block
        // mounted enters `Model` (see the builder below). The engine ctor
        // also reloads the substrate internally, so the visible boundary
        // between Model and Substrate below is best-effort — the substrate
        // has already loaded by the time its step is announced.

        // The card's headroom at startup, beside the file-ingest pool width
        // that will spend it, so a CUDA OOM during code_read is traceable to
        // both figures rather than an opaque exit-1.
        if let Ok((free, total)) = device.mem_get_info() {
            let free_gib = (free as f64) / (1024.0 * 1024.0 * 1024.0);
            let total_gib = (total as f64) / (1024.0 * 1024.0 * 1024.0);
            let n_workers = crate::code_read::CODE_READ_PARALLELISM;
            tracing::info!(
                "VRAM {free_gib:.1}/{total_gib:.1} GiB free · code_read workers={n_workers}"
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
        let working_set = dialogue_config(proj_builder.schema(), dialogue_layer)?;
        // (layer, group) are passed through to
        // `new_conversation_with_projection`, which mints a fresh
        // `TimelineId` internally.

        // Install the tool catalog into the schema before building the
        // base conversation.  Each tool gets a section in dialogue's
        // system_prompt; the projection layer's `tools` collection
        // governs `top_k` so only the K most relevant tools survive
        // into any single projection.
        //
        // Sections are content-addressed in the redo log: a boot restores
        // every section whose stream is already persisted as a cold marker
        // (the next projection that needs it lifts it hot) and prefills only
        // the rest, whose seals the persistence thread then writes. An
        // unchanged catalog therefore costs no prefill and adds no records on
        // the next boot (`tools_integration::a_second_boot_restores_every_
        // prompt_section`).
        // Resolve the effective tool catalog before any consumer touches it: a
        // `<workspace>/tools/` folder (a mind/game's own tools) overrides the
        // bundled built-ins; absent it, the built-in coding-assistant catalog.
        // Every `repo` parameter is constrained to the workspace's repositories.
        crate::tool_def::init(&workspace);
        let tool_sections = install_tool_catalog(&mut proj_builder)
            .map_err(|e| anyhow::anyhow!("tool catalog install: {e}"))?;
        tracing::info!(
            n_tools = tool_sections.len(),
            "tool catalog installed (top_k governed by `tools` collection in projection.yaml)",
        );
        // The journal collection selects each conversation's own sections by
        // name; scoped before the base conversation is built, so no projection
        // ever reads it any other way.
        let journaling = carry::install(&mut proj_builder)
            .map_err(|e| anyhow::anyhow!("journal install: {e}"))?;
        tracing::info!(journaling, "journal collection installed");

        // Structure-derived section-collection population. Each empty section
        // collection (e.g. `response`, `mood`) is filled from its content folder
        // — `<collection>s/*.yaml` — as authored sections BEFORE base-conv build,
        // so each template's KV seals into the frame (baseline provenance from its
        // own prefill, exactly like a tool's JSON line). The folder and the fact
        // that it needs loading are DERIVED from the declared schema (see
        // `crate::ingest::section_sinks`); nothing is annotated in projection.yaml.
        // Each section's `examples` are parsed and validated here; selection
        // provenance currently comes from each template's own prefill (the
        // calibration phase below calibrates tool selection only). For the
        // coding-assistant schema there are no such collections, so this is a no-op.
        let identity = crate::response_section::Identity::load(&root);
        for sink in crate::ingest::section_sinks(proj_builder.schema()) {
            if disabled_layers.contains(&sink.collection) {
                tracing::info!(collection = %sink.collection, "--disable-layer: section collection suppressed");
                continue;
            }
            let sections =
                crate::response_section::load_sections(&root.join(&sink.folder), &identity);
            if sections.is_empty() {
                continue;
            }
            let installed = crate::response_section::install_sections(
                &mut proj_builder,
                &sink.collection,
                &sections,
            )
            .map_err(|e| anyhow::anyhow!("install '{}' sections: {e}", sink.collection))?;
            tracing::info!(
                collection = %sink.collection,
                folder = %sink.folder,
                n = installed.len(),
                "installed calibrated section collection",
            );
        }

        // Identity sections: the two-level `identities/<name>/*.yaml` tree, one
        // sub-folder per identity (its `anchor.yaml` + detail facets). Members are
        // namespaced `<name>::<stem>` and installed into the `identity_anchor`
        // (always-visible) and `identity` (top-k) collections so every identity's
        // KV seals once here; a conversation scopes to its own identity at
        // projection time (see `IdentityBuilders`). No-op when the schema declares
        // neither collection (the coding-assistant schema) or `identities/` is
        // absent.
        let identity_sections =
            crate::response_section::load_identity_sections(&root.join("identities"), &identity);
        for (collection, sections) in [
            ("identity_anchor", &identity_sections.anchors),
            ("identity", &identity_sections.facets),
        ] {
            if sections.is_empty() {
                continue;
            }
            // Only install into a collection the schema actually declares — a
            // workspace with an `identities/` folder but a schema that doesn't
            // declare these collections (e.g. the coding-assistant schema) is a
            // no-op, not a startup failure.
            if proj_builder.id_for_system_collection(collection).is_none() {
                tracing::warn!(
                    collection,
                    n = sections.len(),
                    "identities/ present but schema declares no '{collection}' collection — skipping",
                );
                continue;
            }
            let installed =
                crate::response_section::install_sections(&mut proj_builder, collection, sections)
                    .map_err(|e| anyhow::anyhow!("install '{collection}' sections: {e}"))?;
            tracing::info!(
                collection,
                n = installed.len(),
                "installed identity collection",
            );
        }
        let default_identity = identity.default_identity.clone();

        // `--disable-layer` takes its layer OUT OF THE PROVENANCE GATHER, and the
        // flag has to be set HERE — while there is still exactly one builder.
        // Every schema that resolves a projection later is a copy of this one:
        // `base_conv` is built from `proj_builder` itself, and the clone below
        // feeds the ingest passes, the per-mode turn builders
        // (`ModeBuilders::build`) and the ingest worker's `refresh_builder`. Setting
        // `gathered` after the clone would leave the base conversation — the very
        // schema the live dialogue projects against — still gathering the layer,
        // which is the half that mattered.
        //
        // `gathered` is runtime-only and deliberately not a YAML field: the
        // schema declares what a layer IS, and whether this boot lets it compete
        // is an operator decision for this process only. Nothing is deleted, so
        // dropping the flag restores the layer's turns to the gather intact.
        //
        // A name that is a section collection rather than a layer (`response`,
        // `mood`) matches nothing here; its suppression is the `section_sinks`
        // loop above, so a `false` return is expected, not an error.
        for name in &disabled_layers {
            if proj_builder.set_layer_gathered(name, false) {
                tracing::info!(
                    layer = %name,
                    "--disable-layer: layer excluded from the provenance gather",
                );
            }
        }
        // The dialogue layer's `system_prompt.items` start with a static
        // prelude (mode/frame/grounding/tools_intro) →
        // then the `tools` collection (90+ tool sections, top_k=3) →
        // then `tools_outro`.  The pre-collection prelude is what we
        // pass as the engine's `system_prompt` so it gets ChatML-wrapped
        // for the dialect; everything after the first Collection is
        // expanded by `preemptive_prefill` itself.
        let before_text: String = pre_collection_prelude(&proj_builder);

        // The model pack, from the model cache: shared by every workspace on
        // this model, it survives `--wipe-substrate`, and carries the tokenizer.
        // What every unit this process ingests says it was computed under: a
        // duplicate a better-quantized member wrote replaces this one's.
        crate::branch_ingest::keys::set_model_bits(
            candle_conversation::guest::checkpoint::weight_bits(&pack),
        );
        // zend's engine-wide asks come from `engine_asks`, the one place a
        // host that runs without zend also takes them from.
        let mut builder = engine_asks::ask(
            model
                .builder()
                .system_prompt(&before_text)
                .model_path(pack)
                .workspace_path(root.clone())
                .read_only_substrate(read_only_substrate)
                .tombstone_retention(tombstone_retention)
                .cancel_token(crate::ingest_cancel::token().clone()),
            &proj_builder,
            qsa_selection_budget,
            qsa_strata,
        )
        // Dialogue turns compress at C5 (moderate adaptive quantization).
        // Paired with the removed uniform-K pin (see `ModelBuilder::engine`),
        // so K is adaptive too.
        .compression_level(5)
        // Per-layer corrupt-turn policy (from the projection schema): the
        // startup reload drops the whole conversation for ingest layers and
        // only the corrupt turn for dialogue.
        .corrupt_turn_policies(
            proj_builder
                .schema()
                .layers
                .iter()
                .map(|l| (l.id, l.on_corrupt_turn))
                .collect(),
        )
        // The substrate open is its own step: a store written before
        // identities is stamped here, record by record, and that ran for a
        // minute and a half under "Loading model" at 0%. The stage names
        // the line under the bar; the records fill it.
        .on_substrate_open({
            let progress = Arc::clone(&progress);
            let status = status_tx.clone();
            let shown: Arc<Mutex<Option<OpenStage>>> = Arc::new(Mutex::new(None));
            OpenProgressHook(Arc::new(
                move |stage: OpenStage, done: usize, total: usize| {
                    progress.set_step(LoadStep::Opening);
                    progress.set_step_unit(stage.unit());
                    progress.set_step_progress(done as u64, total as u64);
                    let mut last = shown.lock().unwrap_or_else(|e| e.into_inner());
                    if *last != Some(stage) {
                        *last = Some(stage);
                        status.send(stage.label().to_string()).ok();
                    }
                },
            ))
        });
        // The model load, phase by phase: the builder says when it begins —
        // as soon as the substrate is open — and each phase names the line
        // under the bar and what its counts are.
        let model_progress = Arc::clone(&progress);
        let model_status = status_tx.clone();
        let model_phase: Mutex<Option<LoadPhase>> = Mutex::new(None);
        let model_hook = move |phase: LoadPhase, done: usize, total: usize| {
            if model_progress.current_step() != Some(LoadStep::Model) {
                model_progress.set_step(LoadStep::Model);
            }
            let mut shown = model_phase.lock().unwrap_or_else(|e| e.into_inner());
            if *shown != Some(phase) {
                *shown = Some(phase);
                model_status.send(phase.label().to_string()).ok();
                model_progress.set_step_unit(phase.unit());
            }
            model_progress.set_step_progress(done as u64, total as u64);
        };
        // The engine comes from the process's slot: built here, with every
        // tenant's requests merged, when zend ranks highest of the tenants that
        // arrive; otherwise zend's substrate is bound and attached to the engine
        // the higher one built. Building loads the model weights (blocking,
        // per-layer hook above) AND spawns the scheduler thread, which kicks off
        // the substrate reload (redo-log replay) to run CONCURRENTLY in the
        // background. So this timing is the model-weight load; the reload's cost
        // shows up as the `LoadStep::Substrate` wait below (its overhang past the
        // weight load) — the two together are the opaque "Loading model" span.
        let t_engine = Instant::now();
        let waiting_progress = Arc::clone(&progress);
        let engine = engine_source
            .ticket()
            .engine_with_progress(&mut builder, &device, Some(&model_hook), &|tenants| {
                waiting_progress.set_waiting_for(tenants)
            })
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
        tracing::info!(
            elapsed_ms = t_engine.elapsed().as_millis() as u64,
            "load timing: model weights loaded (engine built; substrate reload now running in background)",
        );

        // AFTER the engine build, never before. The builder resolves the
        // `<think>`/`</think>` ids into its sampling config while loading the
        // tokenizer, so a config taken earlier is a snapshot with both ids still
        // at their `-1` placeholder — and every tier of the think-block close
        // budget is gated on them. Unresolved, `update_segment_state` returns at
        // its first guard, `in_segment` never opens, `segment_len` never counts,
        // and the graceful ramp, the force cutoff and the closer script are all
        // silently unreachable: think blocks then run to the stencil's runaway
        // span cap and get amputated mid-word instead of closing on a clause.
        let mut conv_config = builder.conversation_config();
        // zend decodes code, not prose: every conversation it opens — dialogue,
        // passthrough, ingest — derives from this config. See `coding_sampling`.
        coding_sampling::apply(&mut conv_config.sampling);
        // Every conversation — dialogue, titler, passthrough, ingest — derives
        // from this config, so this one switch covers them all. See
        // `DaemonConfig::summarize` for why summaries are opt-in.
        if !summarize {
            conv_config.tree.disable_summarization();
        }
        tracing::info!(
            enabled = conv_config.tree.summarizes(),
            "background tree summaries (--summarize)",
        );
        if conv_config.sampling.segment_close_token_id < 0
            || conv_config.sampling.segment_open_token_id < 0
        {
            // Loud, because the failure mode is invisible: generation still
            // works, it just ignores every thinking budget.
            anyhow::bail!(
                "thinking token ids unresolved in the dialogue sampling config \
                 (open={}, close={}) — the think-block close budget would be inert",
                conv_config.sampling.segment_open_token_id,
                conv_config.sampling.segment_close_token_id,
            );
        }

        // Compile each tools mode's catalog into its own constrained-decoding
        // stencil, keyed by the `<tool_call>` trigger, so any tool call a turn
        // starts is forced to the exact shape of a tool its mode offers. Only
        // arm a stencil for tools that were actually installed into the
        // dialogue prompt. A tool-free projection, or a mode that offers
        // nothing, gets an empty `TriggerRegistry` (never fires) rather than a
        // compiled stencil, which rejects an empty catalog.
        let t_compile = Instant::now();
        let compile = |mode: ToolMode| -> anyhow::Result<Arc<TriggerRegistry>> {
            let specs = crate::tools::catalog_for(mode);
            if tool_sections.is_empty() || specs.is_empty() {
                return Ok(Arc::new(TriggerRegistry::new()));
            }
            engine
                .compile_tool_stencil(&specs)
                .map_err(|e| anyhow::anyhow!("{} tool stencil compile: {e}", mode.id()))
        };
        let [none, restricted, comprehensive] = ToolMode::ALL.map(compile);
        let tool_stencils = [none?, restricted?, comprehensive?];
        // A file's reading calls `file_read` and nothing else: the Restricted
        // grammar, its calls held to that one tool.
        let read_stencil = engine
            .holding_tools(
                &tool_stencils[ToolMode::Restricted.level() as usize],
                &[FILE_READ_TOOL.to_string()],
            )
            .map_err(|e| anyhow::anyhow!("file reading's tool stencil: {e}"))?;
        // The thinking-block steering trees (one per non-off effort dial),
        // compiled once and reused across turns alongside the tool-call base.
        let think_steering = engine
            .compile_think_steering()
            .map_err(|e| anyhow::anyhow!("think steering compile: {e}"))?;
        tracing::info!(
            elapsed_ms = t_compile.elapsed().as_millis() as u64,
            "load timing: tool-stencil + think-steering compiled",
        );

        // The substrate reload (redo-log replay) runs on the scheduler thread
        // spawned by the engine ctor. As the substrate grows this is no longer
        // instantaneous, so surface real progress and block here until it
        // finishes before advancing to section prefill (which would otherwise
        // stall against the busy scheduler under a misleading step label).
        //
        // This wait's elapsed time is the reload's OVERHANG past the model-weight
        // load — i.e. how much longer the redo-log replay took than loading the
        // weights it ran alongside. It grows with substrate size, so it's the
        // number to watch as the corpus (and its segment count) scales.
        progress.set_step(LoadStep::Substrate);
        let t_reload_wait = Instant::now();
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
        tracing::info!(
            elapsed_ms = t_reload_wait.elapsed().as_millis() as u64,
            "load timing: substrate reload overhang (redo-log replay past the weight load)",
        );
        // Replayed: the substrate views can read it from here, while the rest of
        // the load runs (`ZendSession::substrate_for_views`).
        let _ = loading_substrate.set(engine.conversation());

        // Reclaim is normally fully background: the segmented log's
        // persistence-thread maintenance pass drops / compacts / combines
        // segments incrementally (`docs/archived/segmented_substrate_log.md` §6), so a
        // startup pays no whole-store rewrite. `--compact-substrate` forces the
        // eager path instead — a whole-store rewrite here, after the reload (so
        // the live set is known) and before serving. It always runs when the
        // flag is set (no reclaimable-marker gate): the operator asked for it.
        // A read-only substrate is never rewritten.
        if compact_substrate && !read_only_substrate {
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

        if load_stopped(&engine, "the substrate reload") {
            return Ok(None);
        }

        // With `--cluster`, join the gateways now that the substrate — and the
        // node it carries — is open (`docs/replicated_substrate.md` §7.6). A
        // gateway that refuses this build for good leaves the daemon here, on
        // its loading page with the reason, until it is stopped.
        if let Some(join) = &cluster {
            progress.set_step(LoadStep::Connecting);
            let conv = engine.conversation();
            let writer = conv.clone();
            let reloader = engine.reloader();
            let store = ConversationStore::new(conv.clone(), engine.kv_layers(), move || {
                reloader.reload_and_wait()
            });
            let replication = store.replication()?;
            join.link().set_repos(repo_ads(&workspace));
            // The ingest units still to do, which the gateway's plan moves
            // characters off this card for (`docs/cooperative_admission.md` §4.4).
            join.link().set_bulk(|| {
                crate::ingest_backlog::snapshot().map_or(0, |b| {
                    b.total.saturating_sub(b.processed).min(u64::from(u32::MAX)) as u32
                })
            });
            let report = |views: &[ClusterView]| {
                progress.set_step_progress(admitted(views), views.len() as u64);
                status_tx.send(connecting_detail(views)).ok();
            };
            let link = join.connect(
                conv.node(),
                Some(model_id(engine.model_key())),
                replication,
                move || writer.last_oseq(),
                crate::ingest_cancel::cancelled,
                report,
            );
            if let Link::Stopped(_) = link {
                join.hold(crate::ingest_cancel::cancelled, report);
            }
            // A conversation another member lets go is this one's to take once
            // it holds it complete, and this one's own releases name their
            // conversations' newest commits (`handover`).
            if let Some(member) = join.link().member() {
                crate::handover::install(
                    member.clone(),
                    &store,
                    conv.clone(),
                    conversation_timeline,
                );
            }
            if link == Link::Joined {
                // Serving from a substrate that is behind is the thing to
                // avoid (§7.6): ready only once caught up.
                progress.set_step(LoadStep::CatchUp);
                join.catch_up(crate::ingest_cancel::cancelled, |catchups| {
                    let (done, total) = catchup_progress(catchups);
                    progress.set_step_progress(done, total);
                    status_tx.send(catchup_detail(catchups)).ok();
                });
            }
            if crate::ingest_cancel::cancelled() {
                tracing::warn!("shutdown while connecting to the cluster — draining the engine");
                if let Err(e) = engine.shutdown() {
                    tracing::error!("shutdown-while-connecting drain failed: {e}");
                }
                return Ok(None);
            }
            if let Some(replicator) = join
                .link()
                .member()
                .and_then(|member| member.replicator().cloned())
            {
                // Readings lease a unit before minting its conversation, so a
                // second member that planned the same tree does not read it too.
                ingest_lease::install(replicator);
            }
            status_tx.send(String::new()).ok();
        }

        // Repositories read from the substrate (`docs/substrate_git.md` §7):
        // every store brought up to the log before anything reads a branch,
        // then kept there — and with origin — on a thread of its own, which
        // `ZendSession::shutdown` stops. A workspace naming no repository
        // has nothing to load, and needs no agent.
        let repositories = {
            let names: Vec<String> = workspace
                .repos()
                .iter()
                .map(|r| r.name.clone())
                .filter(|name| name != UPLOADS_REPO)
                .collect();
            if names.is_empty() {
                None
            } else {
                let runtime = host_runtime.clone().ok_or_else(|| {
                    anyhow::anyhow!("no async runtime for the agents' jobs to run on")
                })?;
                let access = substrate_git::access(cluster.as_ref(), agent.as_ref(), runtime)?;
                Some(substrate_git::load(
                    &workspace,
                    &names,
                    engine.conversation().shared(),
                    access,
                    &progress,
                    &status_tx,
                )?)
            }
        };
        // A stop ends the repositories' load between rounds, incomplete — and
        // nothing after it should read a branch they have not brought up. The
        // follower goes with `repositories`, whose drop stops it.
        if load_stopped(&engine, "loading repositories") {
            return Ok(None);
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
            // Cheap (one `GlobalMemoryStatusEx`/`/proc/meminfo` read) at section
            // granularity, not per token — see `vram::sample_available_low_water`.
            // Section prefill is the first of the two boot phases whose combined
            // footprint (tool corpus + the expert warm tier sized against
            // launch-time RAM) has crashed a from-empty-substrate boot with a raw
            // host allocator abort; this is the trough this phase leaves behind.
            vram::sample_available_low_water();
            let scaled = (done * 5_000).checked_div(total).unwrap_or(0);
            section_progress.set_step_progress(scaled, 10_000);
        };
        // A clone of the (tools-installed + templates-tokenised) projection
        // builder is kept for the schema-driven ingest passes (one cheap clone
        // per ingest layer below — schemas are `Arc`-backed) and stays on
        // `InferenceState` afterwards for the ingest worker's passes to mint
        // fresh ingest-layer timelines.
        let proj_builder_refresh = proj_builder.clone();
        let base_conv = engine.new_conversation_with_projection_progress(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config.clone(),
            Some(&section_hook),
        );
        // The section prefill checks the ingest-cancel latch between sections,
        // and a stop asked during it ends the build `IngestCancelled`: the stop,
        // not a failed load.
        if matches!(
            base_conv,
            Err(candle_conversation::ConversationError::IngestCancelled)
        ) && load_stopped(&engine, "prefilling tool sections")
        {
            return Ok(None);
        }
        let mut base_conv = base_conv.map_err(|e| anyhow::anyhow!("base conv create: {e}"))?;

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
            free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
            "base conversation ready (prelude + tool catalog + outro pinned at init)",
        );
        // A stop asked as the last section sealed lands here.
        if load_stopped(&engine, "prefilling tool sections") {
            return Ok(None);
        }

        // Tool-catalog summaries: one per tools mode that projects tools, each
        // listing exactly the tools that mode offers. Each is assembled
        // **deterministically** from the catalog metadata — every tool grouped
        // under its category — so there is no model call and nothing to cache:
        // the text is rebuilt and its section prefilled on every startup, exactly
        // like the tool sections themselves.
        {
            let prelude = pre_tools_section_ids(&proj_builder_refresh);
            let summary_for = |mode: ToolMode| {
                let offered = crate::tools::offered_tool_names(mode);
                let sections: Vec<crate::tool_summary::InstalledTool> = tool_sections
                    .iter()
                    .filter(|(name, _, _)| offered.contains(name))
                    .cloned()
                    .collect();
                crate::tool_summary::build_tool_summary(&sections)
            };
            let summaries: Vec<(String, Reserved, &str)> = ToolMode::ALL
                .into_iter()
                .filter_map(|mode| {
                    Some((summary_for(mode), tool_summary_section(mode)?, mode.id()))
                })
                .collect();
            // Assembly is instant — the step's second half completes immediately.
            progress.set_step_progress(10_000, 10_000);

            // Seal each mode's summary under its reserved section id, prefilled with
            // the same pre-tools prefix so its KV is position-correct for "just
            // before the tools". Each mode's projection points the tools collection
            // at its own summary ([`tool_summary_section`]); None emits none.
            for (text, reserved, label) in summaries {
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
        // every title-generation reuses it. It is background to the dialogue
        // it names, so it decodes at `Normal`: it never holds that dialogue's
        // prompt back.
        let titler_formatted = conv_config
            .dialect
            .format_system_prompt(TITLER_SYSTEM_PROMPT);
        let titler = engine
            .new_reserved_conversation(
                &titler_formatted,
                Reserved::Titler,
                DecodePriority::Normal,
                conv_config.clone(),
            )
            .map_err(|e| anyhow::anyhow!("titler create: {e}"))?;
        let titler_timeline = titler.timeline_id();
        let tokenizer = Arc::new(engine.tokenizer().clone());
        tracing::info!(
            titler_timeline = ?titler_timeline,
            "titler conversation ready",
        );

        // ── Calibrating sections ──────────────────────────────────────────────
        // Prefill each registered tool's authored trajectories and questions into
        // the hidden `Reserved::Calibration` layer, capturing each think→call
        // trajectory (and its per-reprojection wide-Q windows) and each question.
        //
        // All cases share ONE projection: the whole catalog as a name-keyed `tools`
        // collection governed by `SelectionRule::Named`. Each run pins exactly its
        // tool by name (`TurnOptions::selection`), so the model sees a single
        // available tool — the condition under which a free decode reliably calls
        // the right one. The tool sections are sealed once (the first run); every
        // later run reuses them via the idempotent section ingest. One throwaway
        // conversation per (tool, example); a failing case is logged and skipped so
        // it can never break daemon load.
        if load_stopped(&engine, "the tool summaries") {
            return Ok(None);
        }
        progress.set_step(LoadStep::CalibratingSections);
        // Skip entirely for a tool-free projection — nothing to calibrate — and
        // on a read-only substrate, whose calibration corpus is whatever the
        // daemon that writes it has already sealed.
        if !tool_sections.is_empty() && !read_only_substrate {
            // Wall-time attribution for the phase, logged once at its end.
            let calib_start = Instant::now();
            let mut timing = CalibTiming::default();
            let defs = crate::tool_def::all();
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
            // Two kinds of calibration case:
            //
            //  • `examples` — the full trajectory from the definition file:
            //    prompt + `<|im_end|><|im_start|>assistant` + think→call with
            //    projection markers. One submission each.
            //  • `questions` — plain user questions with an EMPTY assistant
            //    turn. Routing happens on the question, and a full trajectory is
            //    ~95% think block and call, so these are exemplars that are only
            //    the part a live probe resembles — the region the phase lens
            //    reads (`docs/provenance_score_normalization.md` §3.4). ALL of a
            //    tool's questions go in as ONE submission.
            //
            // Each case carries its content marker, computed once here from the
            // tool's whole projected definition plus the trajectory, so editing
            // either regenerates exactly that case.
            // The authored trajectories are BORROWED from the definitions, which
            // outlive this vector; only the question trajectories are built here
            // and so have to be owned. `Cow` lets both kinds share one list
            // without copying every authored example — thousands of them across
            // the catalog, each a full think block and call.
            // **A tool's questions are ONE submission, not one each.** They are
            // short, uniform, and all carry the same tool tag and the same
            // pinned selection, so they go in as one turn group — prefilled
            // together, each masked to itself, sealed one turn apiece
            // (`Conversation::submit_prefilled_turn_group`). That takes the
            // corpus from 2,998 submissions to 859 without changing a single
            // exemplar's content.
            //
            // Per TOOL rather than per catalog: a group shares one projection
            // and one selection, so it must contain one tool's cases only. Size
            // is the other reason — a tool's 24 questions are ~1.5k tokens of
            // lossless KV (~55 MiB), while the whole question corpus in one
            // sequence would be ~143k tokens (~5 GB), which would consume the
            // entire KV budget and force exactly the expert-zone concessions
            // this engine works to avoid.
            //
            // Trajectories stay one per submission: they are ~95% think block
            // and call, long and irregular, and each is staged at its own
            // projection points, which a group's cases do not carry. A bare-prompt
            // example has no trajectory and joins the question group
            // (`ToolDef::calibration_questions`), so nothing in this phase is
            // decoded live on the tool files' account.
            let dialect = &conv_config.dialect;
            let cases: Vec<CalibCase<'_>> = defs
                .iter()
                .flat_map(|d| {
                    let full = d.trajectories(dialect.assistant_start).map(move |e| {
                        CalibCase::Trajectory {
                            tool: d.name.as_str(),
                            text: Cow::Borrowed(e),
                            marker: d.calibration_marker(e),
                        }
                    });
                    let questions =
                        d.calibration_questions(dialect.assistant_start, dialect.user_end);
                    let asked = (!questions.is_empty()).then(|| CalibCase::Questions {
                        tool: d.name.as_str(),
                        // One marker for the whole group: the group is
                        // regenerated as a unit, so its resume key must change
                        // when ANY of its questions does. Joined with a
                        // separator that cannot occur inside a question, so two
                        // different splits cannot hash alike.
                        marker: d.calibration_marker(&questions.join("\u{0}")),
                        questions,
                    });
                    full.chain(asked)
                })
                .collect();
            let total: usize = cases.iter().map(CalibCase::exemplars).sum();
            let asks: usize = cases
                .iter()
                .filter(|c| matches!(c, CalibCase::Questions { .. }))
                .map(CalibCase::exemplars)
                .sum();
            // Submissions, not exemplars: every tool's question set is ONE
            // stuffed prefill carved into one turn per question, so the number
            // of forwards the phase costs is trajectories + tools, not the
            // corpus size.
            tracing::info!(
                exemplars = total,
                question_only = asks,
                submissions = cases.len(),
                "calibrating sections: per-tool example prefills (named tool selection)"
            );

            // Retire exemplars that no longer match any current case. A tool whose
            // description, parameters, or example changed keeps its old
            // conversation ARCHIVED under the *previous* marker, where the
            // by-marker lookup below can never find it — so without this sweep it
            // stays tagged `["tool", name]` and keeps answering the belief scan
            // with signatures captured against wording that no longer exists,
            // alongside the freshly calibrated ones. Tombstoning drops it from the
            // gather (`resolver.rs` skips tombstoned timelines), leaving exactly
            // the current corpus.
            let current: HashSet<&str> = cases.iter().map(|c| c.marker()).collect();
            let mut retired = 0usize;
            for (timeline, marker) in engine.conversations_with_metadata_key(CALIB_MARKER_KEY) {
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
            // A case's resume marker is `ToolDef::calibration_marker` — the tool
            // name plus a hash of its projected definition and this example — so
            // rewording a description or a parameter, or editing the example,
            // changes the marker and the case regenerates against the current text.
            let mut done = 0usize;
            let mut to_run: Vec<&CalibCase<'_>> = Vec::new();
            for case in cases.iter() {
                let (name, marker) = (case.tool(), case.marker());
                // Distill-inclusive: a finished exemplar's designed end state is
                // archived + distilled(`ProvenanceOnly`) + tombstoned, and the
                // live-only lookup cannot see it — every such case read as
                // "never ran" and regenerated. Ordinary tombstones stay excluded,
                // so a genuinely retired conversation is still ignored here.
                let prior = engine
                    .find_conversations_by_metadata_including_distilled(CALIB_MARKER_KEY, marker);
                // Done iff a prior conversation for this case is archived.
                if prior.iter().any(|t| engine.is_conversation_archived(*t)) {
                    // The completed corpus only needs its wide-Q sigs, so mark the
                    // archived conversation for distillation — compaction reclaims
                    // its trajectory content (the tool file's example is the source
                    // of truth). Idempotent: skip a timeline already distilled.
                    for t in prior
                        .iter()
                        .filter(|t| engine.is_conversation_archived(**t))
                    {
                        mark_calibration_distill(&engine, *t, name, &mut timing);
                    }
                    // Progress counts EXEMPLARS, not submissions: a resumed
                    // question group already produced all of its turns, so
                    // crediting it with one would make the bar crawl for a
                    // resumed run and then jump.
                    done += case.exemplars();
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
                to_run.push(case);
            }
            // Say up front how much of the corpus is being regenerated and why the
            // step will take as long as it does. A case resumes only when a prior
            // conversation carrying its marker is ARCHIVED; anything else — a
            // reworded tool description, an edited example, or an archive record
            // that did not survive — regenerates. Without this split a partial
            // recalibration is indistinguishable from a full one at a glance.
            // `exemplars` and `submissions` are deliberately both reported: a
            // question group is one submission carrying many exemplars, so a
            // single figure would misrepresent either the work or the corpus.
            tracing::info!(
                exemplars = total,
                resumed = done,
                submissions = to_run.len(),
                exemplars_to_run = to_run.iter().map(|c| c.exemplars()).sum::<usize>(),
                "calibration resume filter"
            );

            // Non-blocking sweep: archive + retire every case finished since the
            // last sweep. Archives only complete trajectories (`</tool_call>`); an
            // incomplete/failed one stays un-archived to retry next load. Returns
            // whether anything was retired.
            let mut inflight: Vec<InFlightCase<'_>> = Vec::new();
            let retire_completed = |inflight: &mut Vec<InFlightCase<'_>>,
                                    done: &mut usize,
                                    sealed: &mut usize,
                                    reclaim: &mut Vec<TimelineId>,
                                    timing: &mut CalibTiming|
             -> bool {
                let mut retired = false;
                let mut i = 0;
                while i < inflight.len() {
                    let mut collected: Vec<ProjectionEvent> = Vec::new();
                    // A terminal the idle wait already absorbed retires this
                    // case without touching the channel again.
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
                    match terminal {
                        Some(result) => {
                            let InFlightCase {
                                mut conv,
                                handle,
                                tool: name,
                                events,
                                finished: _,
                                is_prefill,
                                exemplars,
                            } = inflight.remove(i);
                            match result {
                                Ok(resp) if is_prefill || resp.text.contains("</tool_call>") => {
                                    seal_calibration_turn(
                                        &mut conv, handle, &resp, name, events, timing,
                                    );
                                    if let Err(e) = timing.time(
                                        |t| &mut t.archive,
                                        || {
                                            engine
                                                .set_conversation_archived(conv.timeline_id(), true)
                                        },
                                    ) {
                                        tracing::warn!(
                                            tool = name,
                                            "calibration archive failed: {e}"
                                        );
                                    }
                                    // Only the wide-Q sig is needed henceforth —
                                    // mark for distillation so compaction sheds
                                    // the trajectory content.
                                    mark_calibration_distill(
                                        &engine,
                                        conv.timeline_id(),
                                        name,
                                        timing,
                                    );
                                    *sealed += 1;
                                }
                                Ok(_) => tracing::warn!(
                                    tool = name,
                                    "calibration trajectory incomplete — left un-archived for retry"
                                ),
                                Err(e) => {
                                    tracing::warn!(tool = name, "calibration case failed: {e}")
                                }
                            }
                            // Retired either way, and its sealed K/V is never
                            // attended again — only a persisted wide-Q sig is,
                            // and an un-archived case is regenerated whole next
                            // boot. A turn group that lost one case still sealed
                            // the rest, so its timeline is demoted hot→warm like
                            // any other to keep calibration's VRAM footprint flat.
                            reclaim.push(conv.timeline_id());
                            *done += exemplars;
                            progress.set_step_progress(*done as u64, total as u64);
                            retired = true;
                        }
                        None => i += 1,
                    }
                }
                retired
            };

            // The ChatML markers a calibration example is split on: everything
            // after `assistant_start` is the verbatim body we prefill; the user
            // prompt is what precedes it, minus the trailing `user_end`.
            let assistant_start = conv_config.dialect.assistant_start;
            let user_end = conv_config.dialect.user_end;
            // What the completion line reports, counted rather than derived:
            // exemplars the resume filter found sealed, and submissions — a
            // question group is one submission carrying many exemplars, so the
            // two units are kept apart.
            let resumed_exemplars = done;
            let submissions = to_run.len();
            let mut to_run_iter = to_run.into_iter();
            let mut warmed = false;
            // Timelines of retired calibration cases, sealed or not, awaiting
            // hot→warm demotion. Their sealed K/V is never attended again — only
            // the persisted wide-Q sig is — so dropping the hot copy as they retire
            // keeps the phase's VRAM footprint flat instead of letting 700+ cases'
            // lossless K/V pile up hot and OOM the next phase's first prefill.
            let mut calib_timelines: Vec<TimelineId> = Vec::new();
            // Submissions sealed and archived — counted as they retire, never
            // read off `calib_timelines`, which holds failed ones too.
            let mut sealed = 0usize;
            // Index into `calib_timelines` up to which we've already demoted, so
            // each incremental sweep only re-demotes the fresh tail.
            let mut reclaimed_up_to = 0usize;
            loop {
                // A stop ends the phase between waves: nothing more is
                // submitted, and the cases in flight are dropped un-archived —
                // the scheduler ends a turn whose handle is gone, as it does an
                // ingest decode's, and the next load's resume filter tombstones
                // and regenerates each one.
                if crate::ingest_cancel::cancelled() {
                    inflight.clear();
                    break;
                }
                // Refill the window in ONE pipelined round-trip:
                // `new_conversations_with_projection_batch` fires every slot
                // allocation before awaiting any, so the whole window's cases
                // prefill together in a wide forward instead of trickling in one
                // per wave-latency (which starved the batch to 2–4 wide). The
                // warm-up case is created alone first so the shared tool sections
                // pin once before the concurrent window opens.
                // Refill in blocks, not per retired case. Slot allocation is
                // served by the scheduler between waves, so ANY create call —
                // for one case or sixteen — blocks the caller until the current
                // wave ends. Topping the window up by one or two therefore pays
                // a full wave boundary for one or two cases: measured at 29 such
                // calls costing 67 s of the phase's 84 s of create-blocking,
                // averaging 2.3 s each. Waiting until half the window is free
                // amortises that same boundary over eight cases instead, and the
                // GPU keeps working the remaining in-flight ones meanwhile.
                //
                // `inflight.is_empty()` is the tail case: fewer cases remain
                // than the threshold, so nothing is running to wait behind and
                // the block must be issued at whatever size is left.
                let free = CALIBRATION_BATCH.saturating_sub(inflight.len());
                let want = calibration_window_want(free, warmed, inflight.is_empty());
                let batch: Vec<&CalibCase<'_>> =
                    (0..want).map_while(|_| to_run_iter.next()).collect();
                let created_any = !batch.is_empty();
                let convs = if created_any {
                    timing.time(
                        |t| &mut t.create,
                        || {
                            engine.new_conversations_with_projection_batch(
                                batch.len(),
                                &calib_prelude,
                                &calib_builder,
                                calib_layer,
                                calib_group,
                                &calib_config,
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
                            // The whole case's exemplars, as a failed submit
                            // forfeits — a group is one creation carrying many.
                            done += case.exemplars();
                            progress.set_step_progress(done as u64, total as u64);
                            continue;
                        }
                    };
                    // **This exemplar's KV never goes to disk.** The seal captures
                    // its wide-Q signature — which persists on its own record —
                    // and the case's designed end state is
                    // `DistillMode::ProvenanceOnly`: KV dropped, signature kept.
                    // Cold-persisting it therefore writes bytes the next load's
                    // distillation immediately deletes. Measured on this 2,998-case
                    // corpus before the flag: ~4 GB of redo log per minute, ~108 GB
                    // in total, all of it discarded — enough to fill the disk and
                    // wedge the daemon on failing fsyncs.
                    //
                    // Marked before the turn is submitted so the residence is
                    // flagged at `append_complete` rather than after it has already
                    // been queued for a cold write.
                    engine
                        .conversation()
                        .mark_timeline_transient(conv.timeline_id());
                    // Tag at creation so a half-finished case is findable next load.
                    if let Err(e) = timing.time(
                        |t| &mut t.tag,
                        || {
                            engine.set_conversation_metadata(
                                conv.timeline_id(),
                                CALIB_MARKER_KEY,
                                marker,
                            )
                        },
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
                    opts.selection.select(CALIB_TOOL_SELECTOR, name);
                    // An exemplar is a tool-using turn, so it is sealed behind the
                    // prefix a tool-using chat turn gets — the demonstration
                    // included. Selected explicitly because the schema defaults it
                    // ABSENT for the ingest layers, which cannot select anything
                    // (see `projection.yaml`).
                    opts.selection
                        .set_optional(TOOL_EXAMPLE_SELECTOR, OptionalState::Present);
                    // Fast path: prefill the tool file's ChatML trajectory verbatim
                    // in one batched forward pass instead of decoding it token by
                    // token — the wide-Q is captured identically at seal. Prefill
                    // only a single-turn example (exactly one assistant header):
                    // split into the body (after the header — the think→call we
                    // prefill, projection markers kept) and the user prompt (before
                    // it, minus the trailing `user_end`). A multi-turn lead-in (more
                    // than one header) is decoded live rather than prefilling a
                    // wrong-grid body; a bare prompt never reaches here — it is
                    // calibrated with the tool's questions. `submit_prefilled_turn`
                    // strips the markers for the prefilled text and records each
                    // one's token offset so the staged prefill wave fires a
                    // projection there, reproducing the decode's projection sequence.
                    let (submit_result, is_prefill) = timing.time(
                        |t| &mut t.submit,
                        || match case {
                            // ALL of this tool's question exemplars in ONE
                            // turn group, sealed one turn each. They share this
                            // submission's pinned selection, which is correct
                            // precisely because a group is one tool's.
                            CalibCase::Questions { questions, .. } => {
                                // A question exemplar's assistant half is empty —
                                // the routing happens on the question.
                                let group: Vec<(String, String, Vec<String>)> = questions
                                    .iter()
                                    .map(|q| (q.clone(), String::new(), opts.tags.clone()))
                                    .collect();
                                (
                                    conv.submit_prefilled_turn_group(
                                        &group,
                                        opts.selection.clone(),
                                    )
                                    .map(|(handle, _)| handle),
                                    true,
                                )
                            }
                            CalibCase::Trajectory { text, .. } => {
                                // Prefill the tool file's ChatML trajectory
                                // verbatim in one batched forward instead of
                                // decoding it token by token — the wide-Q is
                                // captured identically at seal. Only a
                                // single-turn example (exactly one assistant
                                // header) qualifies: split into the body (after
                                // the header — the think→call we prefill,
                                // projection markers kept) and the user prompt
                                // (before it, minus the trailing `user_end`). A
                                // multi-turn lead-in is decoded live rather than
                                // prefilling a wrong-grid body.
                                if text.matches(assistant_start).count() == 1 {
                                    let (before, b) =
                                        text.split_once(assistant_start).expect("count == 1");
                                    let p = before.trim_end();
                                    let p = p.strip_suffix(user_end.trim()).unwrap_or(p).trim_end();
                                    (
                                        conv.submit_prefilled_turn(
                                            p,
                                            b,
                                            crate::tool_def::PROJECTION_MARKER,
                                            opts.selection.clone(),
                                            opts.tags.clone(),
                                        ),
                                        true,
                                    )
                                } else {
                                    (conv.submit_turn_with_options(text.trim(), opts), false)
                                }
                            }
                        },
                    );
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
                                        seal_calibration_turn(&mut conv, handle, &resp, name, events, &mut timing);
                                        if let Err(e) = timing.time(
                                            |t| &mut t.archive,
                                            || engine.set_conversation_archived(conv.timeline_id(), true),
                                        ) {
                                            tracing::warn!(tool = name, "calibration archive failed: {e}");
                                        }
                                        // Sig captured — mark for distillation so
                                        // compaction sheds the trajectory content.
                                        mark_calibration_distill(&engine, conv.timeline_id(), name, &mut timing);
                                        sealed += 1;
                                    }
                                    Some(_) => tracing::warn!(
                                        tool = name,
                                        "calibration trajectory incomplete — left un-archived for retry"
                                    ),
                                    None => {}
                                }
                                // Queued for hot→warm demotion whatever its outcome
                                // (see the retire path), so the warm-up case's K/V
                                // is reclaimed like every other case's.
                                calib_timelines.push(conv.timeline_id());
                                done += case.exemplars();
                                progress.set_step_progress(done as u64, total as u64);
                            } else {
                                inflight.push(InFlightCase {
                                    conv,
                                    handle,
                                    tool: name,
                                    events: Vec::new(),
                                    finished: None,
                                    is_prefill,
                                    exemplars: case.exemplars(),
                                });
                            }
                        }
                        Err(e) => {
                            tracing::warn!(tool = name, "calibration submit failed: {e}");
                            // A failed group forfeits all of its exemplars, not
                            // one — otherwise the bar never reaches its total.
                            done += case.exemplars();
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
                // Retire finished cases; if none finished this pass, park until
                // whichever in-flight case speaks next — completion order, no
                // poll cadence. The one event that wait consumes is absorbed
                // into its case (a reprojection into `events`, a terminal into
                // `finished` for the next sweep), so nothing the channel says
                // is lost. Driven by `block_on` because this is the loader
                // thread: the futures wake on the engine's own channels, so no
                // timer or I/O driver is needed.
                if !retire_completed(
                    &mut inflight,
                    &mut done,
                    &mut sealed,
                    &mut calib_timelines,
                    &mut timing,
                ) && !inflight.is_empty()
                {
                    timing.time(
                        |t| &mut t.idle,
                        || {
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
                        },
                    );
                }
                // Incremental hot→warm demotion: once a full window's worth of
                // cases has retired since the last sweep, drop the hot K/V of just
                // the fresh tail (`[reclaimed_up_to..]`). No flush — the
                // persistence thread has already migrated the older cases to warm,
                // so they qualify; any not-yet-warm straggler is caught by the
                // flushing boundary sweep below (which re-demotes the whole list).
                // The call is fire-and-forget, so case submission never stalls.
                // The host-RAM trough is sampled on every event, so a from-empty
                // full recalibration records it as it happens rather than it
                // being inferred after a crash — see
                // `vram::sample_available_low_water`.
                vram::sample_available_low_water();
                if calib_timelines.len() - reclaimed_up_to >= CALIBRATION_BATCH {
                    if let Err(e) = timing.time(
                        |t| &mut t.demote,
                        || engine.demote_timelines_hot(&calib_timelines[reclaimed_up_to..], false),
                    ) {
                        tracing::warn!("calibration hot→warm demote failed: {e}");
                    }
                    reclaimed_up_to = calib_timelines.len();
                    // Logged once per window, not only at the phase's end: a raw
                    // allocator abort kills the process with no unwind, so a line
                    // only at completion is never written when the phase itself
                    // is what crashes — exactly the run this exists to diagnose.
                    tracing::info!(
                        done,
                        total,
                        free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
                        "calibrating sections: window checkpoint"
                    );
                }
            }
            // Stopped mid-phase: no boundary sweep — the drain persists what
            // sealed, and the rest is the next load's.
            if load_stopped(&engine, "calibrating sections") {
                return Ok(None);
            }
            // Boundary sweep: flush the pending hot→warm migration so the final
            // window's just-sealed cases are warm-backed, then demote every
            // calibration timeline. This reclaims the tail before the repo-scan
            // phase's first prefill, which would otherwise hit a card still full
            // of the calibration corpus's hot K/V.
            if let Err(e) = timing.time(
                |t| &mut t.demote,
                || engine.demote_timelines_hot(&calib_timelines, true),
            ) {
                tracing::warn!("calibration boundary hot→warm demote failed: {e}");
            }
            timing.log(calib_start.elapsed());
            // `resumed_exemplars` is counted by the resume filter and `sealed` as
            // each submission seals, never one unit less another. `failed` is the
            // rest of the submissions — every one submitted either sealed or did
            // not, each failure logged with its cause as it retired.
            let failed = submissions.saturating_sub(sealed);
            progress.set_step_progress(total as u64, total as u64);
            tracing::info!(
                exemplars = total,
                resumed_exemplars,
                submissions,
                sealed,
                failed,
                free_low_water_mib = vram::available_low_water().map(|b| b / (1024 * 1024)),
                "calibrating sections complete"
            );
        }

        // Every ingest pass takes the shared `&ConversationEngine`, so the loop
        // below is uniform across injection types. The substrate handle status
        // reads comes out first.
        let substrate = engine.conversation();

        // Structure-derived ingestion. The load plan is derived from the declared
        // schema (see `crate::ingest`), not annotated in it: each turn-sink layer
        // is prepared here, in schema order, skipping any named by
        // `--disable-layer`. The folder and file layers are ingested from every
        // repository's branches (or the `--ingest-dir` scope inside one) by the
        // background worker; a raw layer reads its content folder under the
        // workspace.
        let ingest_layers =
            crate::ingest::ingest_layers(proj_builder_refresh.schema(), &root, &ingest_dirs);
        // Mark EVERY enabled ingest layer append-only UP FRONT — deterministically,
        // before any ingest AND before `warm_ingest_normalization` runs — using
        // the SAME builder (`proj_builder_refresh`) whose layer ids the warm-up
        // checks. Marking via any other builder's id leaves the warm seeing 0
        // append-only layers, every code scope COLD, and the most promiscuous
        // file topping every query at an un-normalized score (the observed 0%
        // retrieval after a re-ingest). Marking here, once, from the warm's own
        // builder keeps the append-only set consistent on every load path.
        //
        // That consistency is what the mark buys, and it is expensive to lose: it
        // is what `warm_ingest_normalization` recognises a layer by, and without it
        // the gather divides raw scores in the millions by the `hit_prior` of 400 —
        // a ~13,000x under-correction that put a repo_map file listing, the most
        // promiscuous member in the corpus, at the top of an unrelated dialogue's
        // context at a score of 13,315,007 against the live conversation's 0.
        //
        // `--disable-layer` means "this layer is out of service". No mark, so no
        // normalization warm; `gathered = false` (set before the builder was
        // cloned, far above), so no gather; and no sweep, because sweeping
        // tombstones turns, which is a MUTATION — the one thing an out-of-service
        // layer must not suffer. Its substrate is left exactly as it stands, so
        // dropping the flag restores the layer whole.
        //
        // `base_conv`'s counterpart for ingestion — one prefilled template per
        // `Folders`/`Files` layer, built here exactly as `base_conv` itself was
        // built above, so `repo_scan`/`code_read` can fork a unit's conversation
        // off it the same way a dialogue turn forks off `base_conv`, instead of
        // each unit independently re-running the schema's "eager section
        // ingestion". See `InferenceState::ingest_bases`.
        //
        // Built unconditionally, even for a `--disable-layer` layer: the upload
        // path (`ZendSession::ingest_uploaded_files`) bypasses the disabled
        // layers' passes entirely and still needs a `Files` layer's base to fork
        // from on the first upload. `Raw` is excluded — it
        // only ever prefills records and never decodes, so it has no reasoning
        // to protect and stays on `raw_read::ingest_raw`'s existing construction.
        let mut ingest_bases: HashMap<String, Sequence> = HashMap::new();
        for il in &ingest_layers {
            if il.mode == IngestMode::Raw {
                continue;
            }
            let layer = proj_builder_refresh
                .id_for_layer(&il.name)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' layer", il.name))?;
            let group = proj_builder_refresh
                .id_for_group(&il.group)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' group", il.group))?;
            let base = engine
                .new_conversation_with_projection(
                    &formatted_prompt,
                    proj_builder_refresh.clone(),
                    layer,
                    group,
                    // Append-only ingest: no mid-decode reprojection. Each turn
                    // still assembles its slot from the substrate — slots are
                    // emptied at every seal — so the priming chain, inherited
                    // through `forked_from`, is in every turn's context.
                    utility_config(conv_config.clone()),
                )
                .map_err(|e| anyhow::anyhow!("{} ingest base create: {e}", il.name))?;
            ingest_bases.insert(il.name.clone(), base);
        }
        if load_stopped(&engine, "the ingest bases") {
            return Ok(None);
        }

        // `--self-check`: every stored conversation asked whether it is intact,
        // before the retire sweep, the priming chain and every ingest — so an
        // ingest unit it tombstones is simply read again by them. Each is
        // written out as a record and judged through the dialogue base, which
        // has no history of its own to mix into the record.
        if self_check != SelfCheck::Off {
            progress.set_step(LoadStep::SelfCheck);
            let mut layers: HashSet<LayerId> = ingest_layers
                .iter()
                .filter(|il| ingest_bases.contains_key(&il.name))
                .filter_map(|il| proj_builder_refresh.id_for_layer(&il.name))
                .collect();
            layers.insert(dialogue_layer);
            self_check::run(
                &engine,
                &conv_config,
                &layers,
                base_conv.unsealed_asker(),
                self_check,
                &|done, total| progress.set_step_progress(done, total),
            )?;
        }

        // Ingest conversations are read, never continued: retire the recurrent
        // snapshot tails they kept from before the mark existed, and mark each
        // so no later seal writes another (`crate::ingest_snapshots`).
        let retired = crate::ingest_snapshots::retire_finished(&engine);
        if retired > 0 {
            tracing::info!(
                retired,
                "retired the recurrent snapshots of finished ingest conversations"
            );
        }
        crate::layers::removed::sweep(&engine);
        for il in &ingest_layers {
            if disabled_layers.contains(&il.name) {
                continue;
            }
            if let Some(layer_id) = proj_builder_refresh.id_for_layer(&il.name) {
                engine.mark_layer_append_only(layer_id);
            }
            // Retire crashed partials before any pool starts — a half-built chain
            // still carries its `WideQSig`s, so until it is tombstoned it competes
            // in the gather with a promiscuous file listing and no summary. Nothing
            // is in flight at this point, which is why the sweep lives here and not
            // inside the ingest passes, which can overlap a pool mid-flight.
            match il.mode {
                IngestMode::Folders => crate::repo_scan::retire_crashed_partials(&engine),
                IngestMode::Files => crate::code_read::retire_crashed_partials(&engine),
                // Raw ingest has no half-built state to retire: a record's turns
                // are prefilled and its content hash written in the same commit,
                // so a conversation is either absent or complete.
                IngestMode::Raw => {}
                IngestMode::Derived => {
                    crate::layers::run::retire_crashed_partials(&engine, &il.name)
                }
            }
            // `--wipe-layer <name>`: tombstone EVERY conversation in this layer,
            // not just crashed partials — a targeted alternative to
            // `--wipe-substrate` for exercising the background ingest worker
            // against a real, full-layer backlog without touching anything else
            // (dialogue, the other ingest layers, uploads). Runs before the ingest
            // worker's first pass, which then finds the layer empty.
            // `--disable-layer` still wins: a disabled layer gets no cleanup of
            // any kind, wipe included (see the `continue` above).
            if wiped_layers.contains(&il.name) {
                match il.mode {
                    IngestMode::Folders => crate::repo_scan::wipe_layer(&engine),
                    IngestMode::Files => crate::code_read::wipe_layer(&engine),
                    IngestMode::Derived => {
                        let e = &engine;
                        let all = crate::layers::run::all_of(e, &il.name);
                        let wiped = all
                            .iter()
                            .filter(|tl| e.tombstone_timeline(**tl).is_ok())
                            .count();
                        tracing::info!(layer = %il.name, wiped, "--wipe-layer: every unit of the layer tombstoned");
                    }
                    IngestMode::Raw => tracing::warn!(
                        layer = %il.name,
                        "--wipe-layer: raw layers are not wipeable this way — use \
                         --wipe-substrate, or remove the layer's content folder by hand",
                    ),
                }
            }
        }
        // The raw-layer registry starts empty: a raw layer's records are read
        // only when a mind's folder is read for them.
        let ingest_convs: HashMap<String, IngestConv> = HashMap::new();

        // The branch layers this boot ingests, each with the scope its walk runs
        // under, and each one's group SCOPED: a conversation may retrieve only
        // the units its own base holds (`docs/zend_branch_ingest.md` §8.2), set
        // before each of its turns by `retrieval_scope`. Marked for a disabled
        // layer too — it is out of the gather already, and scoped it could
        // never offer another branch's content if it came back.
        // Every working-set group's share of the budget, as the schema declares
        // it, so the working set can price what it admits: a conversation in
        // one of these groups draws on its group's share, one in no such group
        // is never a member.
        for group in proj_builder_refresh
            .schema()
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
        {
            if let SelectionRule::WorkingSet { share } = group.selection {
                engine.set_working_set_share(group.id, share);
            }
        }

        let mut branch_layers: Vec<(IngestLayer, IngestScope)> = Vec::new();
        let mut files_group = None;
        let mut folders_group = None;
        let mut derived_groups = Vec::new();
        for il in &ingest_layers {
            if il.mode == IngestMode::Raw {
                continue;
            }
            let group = proj_builder_refresh
                .id_for_group(&il.group)
                .ok_or_else(|| anyhow::anyhow!("projection schema missing '{}' group", il.group))?;
            // A day layer's units are of no branch, so its group is not scoped.
            let of_files = il
                .spec
                .as_ref()
                .is_none_or(|s| s.unit == crate::layers::UnitKind::File);
            if of_files {
                engine.mark_group_scoped(group);
            }
            if disabled_layers.contains(&il.name) {
                continue;
            }
            let scope = IngestScope::new(&il.folder, max_depth);
            match il.mode {
                IngestMode::Files => files_group = Some(group),
                // Retrieval derives folders at full depth: the layer retains
                // a folder past the bound while a branch holds it, and a held
                // folder must stay retrievable. A key found for a folder never
                // ingested simply matches nothing.
                IngestMode::Folders => folders_group = Some((group, scope.full_depth())),
                IngestMode::Derived if of_files => derived_groups.push(group),
                IngestMode::Raw | IngestMode::Derived => {}
            }
            branch_layers.push((il.clone(), scope));
        }
        let retrieval_scope =
            RetrievalScope::new(files_group, folders_group).with_derived(derived_groups);
        // Each layer of units compounds on the layers of units below it: their
        // groups are what its units retrieve across conversations
        // (`Substrate::set_ingest_sees`), the scoped ones on the unit's own
        // branches (`layer_sees`).
        let layer_sees = layer_sees(&proj_builder_refresh, &ingest_layers, &disabled_layers);
        for (name, sees) in &layer_sees {
            let (Some(layer), groups) = (
                proj_builder_refresh.id_for_layer(name),
                sees.iter()
                    .map(|(_, g, _)| *g)
                    .collect::<HashSet<GroupId>>(),
            ) else {
                continue;
            };
            engine.set_ingest_sees(layer, groups);
        }
        let layer_sees: HashMap<String, Vec<(String, GroupId)>> = layer_sees
            .into_iter()
            .map(|(name, sees)| {
                let scoped = sees
                    .into_iter()
                    .filter(|(_, _, scoped)| *scoped)
                    .map(|(l, g, _)| (l, g))
                    .collect();
                (name, scoped)
            })
            .collect();
        retrieval_scope.refresh(&engine);

        // Cooperative shutdown during the (long) startup ingest: if a Ctrl-C
        // arrived while ingesting, the loop above broke early. Don't finish
        // building the InferenceState or mark ready — drain the engine HERE, on
        // this loader thread that owns it, exactly as `engine.shutdown()` does at
        // ready (scheduler join → persistence hot→warm→cold full drain → off-thread
        // writer flush). Then return `None`: the loader thread exits without
        // publishing an engine, and `ZendSession::shutdown` (joining this thread)
        // sees `inference` still `None` and knows the drain already happened. The
        // turns ingested so far are durable; the rest re-ingest on the next start.
        if crate::ingest_cancel::cancelled() {
            tracing::warn!(
                "shutdown during startup ingest — draining the engine on the loader thread"
            );
            if let Err(e) = engine.shutdown() {
                tracing::error!("shutdown-during-ingest drain failed: {e}");
            }
            return Ok(None);
        }

        // The titler runs on a single dedicated task fed by this queue. The
        // task owns the titler `Sequence` exclusively (no shared mutex, so
        // title generation never serialises against the request path), and is
        // awaited on shutdown so its in-flight turn unwinds.
        let (titler_tx, titler_rx) = mpsc::channel(TITLER_QUEUE_DEPTH);
        // Build the per-tools-mode projection builders once, up front, so each
        // turn only pays a cheap `Arc` clone instead of re-cloning the
        // ~93-section schema (see `ModeBuilders`).
        let mode_builders = ModeBuilders::build(&proj_builder_refresh)
            .map_err(|e| anyhow::anyhow!("tools-mode projection builders: {e}"))?;
        // Identity scoping rides the tools-mode builders (see `IdentityBuilders`).
        // Empty when the schema has no `identities/` content — then it's a no-op
        // and conversations use the plain tools-mode projection.
        let identity_builders = IdentityBuilders::new(&identity_sections, default_identity);
        tracing::info!(
            n_identities = identity_builders.keeps.len(),
            default = ?identity_builders.default_identity,
            "identity projection scoping ready",
        );
        let think_closer_phrase = tokenizer
            .encode(THINK_CLOSER_PHRASE, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        // Every conversation keeps its own branch per repository; one with none
        // yet starts on the repository's base.
        let base_branches = conv_branches::base_branches(&workspace);
        let seeded = conv_branches::seed_live(&engine, &base_branches, titler_timeline);
        tracing::info!(seeded, "conversations given their base branches");
        let state = Arc::new(Self {
            decoder,
            engine,
            substrate,
            conversations: Resident::default(),
            tool_modes: Mutex::new(HashMap::new()),
            target_layer: base_conv.target_layer_name(),
            base_conv,
            ingest_bases,
            chain_end: Mutex::new(None),
            titler_tx,
            titler_worker: Mutex::new(None),
            memory_writer: Mutex::new(None),
            memory_stop: CancelToken::new(),
            titler_timeline,
            shutting_down: AtomicBool::new(false),
            ingest_convs: Mutex::new(ingest_convs),
            ingest_layers,
            branch_layers,
            layer_sees,
            branch_ingest: BranchIngest::default(),
            retrieval_scope,
            ingest_wake: OnceLock::new(),
            learning_exclusions: AtomicBool::new(false),
            working_set,
            read_coverage: Coverage::default(),
            refresh_builder: proj_builder_refresh,
            refresh_config: conv_config.clone(),
            formatted_prompt,
            mode_builders,
            identity_builders,
            think_closer_phrase,
            tokenizer,
            tool_host: ToolHost::deployed(&workspace, secrets, cluster.as_ref(), agent.as_ref())?
                .running_parts(parts)
                .guarding_defaults(&base_branches),
            repositories,
            workspace,
            base_branches,
            tool_stencils,
            read_stencil,
            think_steering,
            passthrough: PassthroughCache::new(),
            deletions: Deletions::default(),
            journaling,
            journals: Journals::default(),
        });
        // A conversation is this member's for one turn at a time, and what it
        // keeps of one goes when the conversation does.
        if let Some(member) = cluster.as_ref().and_then(|join| join.link().member()) {
            let held = Arc::downgrade(&state);
            turn_claims::install(Arc::clone(member), move |key| {
                if let Some(state) = held.upgrade() {
                    state.forget_conversation(key);
                }
            });
        }
        // The priming chain, read before `ready` and before the ingest worker's
        // first pass: the first question asked must already have it, and the
        // pass must find its links committed rather than read them unparented.
        progress.set_step(LoadStep::Priming);
        state.prime(&progress)?;
        // The chain stops between links on a stop; the links it read are
        // committed, and the next start reads only the rest.
        if load_stopped(&state.engine, "priming") {
            return Ok(None);
        }
        let worker_state = Arc::clone(&state);
        // Spawned on the daemon's runtime — the loader thread holds an enter
        // guard for exactly this kind of arming (see `start_loading`).
        let worker = tokio::spawn(titler_task(worker_state, titler, titler_rx));
        *state.titler_worker.lock().unwrap() = Some(worker);
        // The idle sweep, on the same runtime. It holds the state weakly and
        // ends with it.
        let swept = Arc::downgrade(&state);
        tokio::spawn(async move {
            let mut every = tokio::time::interval(CONV_SWEEP_EVERY);
            loop {
                every.tick().await;
                let Some(state) = swept.upgrade() else {
                    return;
                };
                let released = state.sweep_idle_conversations(Instant::now());
                if released > 0 {
                    tracing::info!(released, "released idle conversations");
                }
            }
        });
        // The journals' calendar: every open conversation folds the periods
        // that have ended since it last looked — a day turning over while
        // nobody talks to it included.
        if state.journaling {
            let folding = Arc::downgrade(&state);
            tokio::spawn(async move {
                let mut every = tokio::time::interval(FOLD_SWEEP_EVERY);
                loop {
                    every.tick().await;
                    let Some(state) = folding.upgrade() else {
                        return;
                    };
                    for conv_id in state.journals.open_ids() {
                        begin_journal_folds(&state, &conv_id, conversation_timeline(&conv_id));
                    }
                }
            });
        }
        Ok(Some(state))
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
            formatted_prompt: &self.formatted_prompt,
            // `Off`, as repo_map's folder summaries run: a file read is a tool
            // round and a description, with no reasoning block between them.
            // Its calls are held to `file_read`, the one tool it is given.
            think_triggers: match &self.think_steering {
                Some(ts) => ts.registry_for(&self.read_stencil, ThinkMode::Off),
                None => Arc::clone(&self.read_stencil),
            },
            // Ingest reads; it runs no job, so it belongs to no lineage.
            tool_ctx: self.tool_host.context_for(
                ToolMode::Restricted,
                &self.tool_host.conversation_files(),
                "",
            ),
            retrieval: &self.retrieval_scope,
            chain_end: *self.chain_end.lock().unwrap(),
            band: None,
            think_for: Some({
                let steering = self.think_steering.clone();
                // A unit decodes under the grammar of the mode whose catalog
                // holds its tools (`LayerSpec::tool_mode`); its grants stay
                // ingest's, widened only as its layer's tools need.
                let stencils = self.tool_stencils.clone();
                Arc::new(move |mode: ThinkMode, tools: ToolMode| {
                    let stencil = &stencils[tools.level() as usize];
                    match &steering {
                        Some(ts) => ts.registry_for(stencil, mode),
                        None => Arc::clone(stencil),
                    }
                })
            }),
            dialogue: self
                .refresh_builder
                .id_for_group("primary_conversation")
                .map(|group| (group, Some(self.titler_timeline))),
        }
    }

    /// Every branch layer this boot ingests, as the branch ingest takes them.
    fn layer_passes(&self) -> anyhow::Result<Vec<LayerPass<'_>>> {
        self.branch_layers
            .iter()
            .map(|(il, scope)| {
                Ok(LayerPass {
                    name: &il.name,
                    mode: il.mode,
                    scope: scope.clone(),
                    base: self.ingest_base(&il.name)?,
                    spec: il.spec.as_deref(),
                    calibrate: il.calibrate.as_deref(),
                    sees: self
                        .layer_sees
                        .get(&il.name)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                })
            })
            .collect()
    }

    /// Build the priming chain and hang the base conversation off its end, so
    /// every dialogue — a fork of `base_conv` inherits its lineage — starts
    /// with the workspace listing, each repository's root listing and its
    /// anchor documents already read. Every ingest unit read after this
    /// records the same end as its parent (`RefreshContext::chain_end`).
    fn prime(&self, progress: &LoadProgress) -> anyhow::Result<()> {
        if self.branch_layers.is_empty() {
            return Ok(());
        }
        let passes = self.layer_passes()?;
        let end =
            self.branch_ingest
                .prime(&self.refresh_ctx(), &self.workspace, &passes, progress)?;
        if let Some(end) = end {
            let base = self.base_conv.timeline_id();
            self.engine
                .set_forked_from(base, end)
                .map_err(|e| anyhow::anyhow!("base_conv priming-chain parent: {e}"))?;
        }
        *self.chain_end.lock().unwrap() = end;
        Ok(())
    }

    /// Calibrate every branch layer that declares a `calibrate:` block
    /// (`crate::calibrate`): each ready unit without a calibration of its
    /// current version — one its ingest did not calibrate as it committed it —
    /// writes its questions, and they are prefilled as its exemplars. Run ahead
    /// of each pass.
    fn calibrate_layers(&self) {
        for (il, _) in &self.branch_layers {
            if crate::ingest_cancel::cancelled() {
                return;
            }
            let Some(spec) = il.calibrate.as_deref() else {
                continue;
            };
            let Some(group) = self.refresh_builder.id_for_group(&il.group) else {
                tracing::error!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    group = %il.group,
                    "the layer declares a calibration but its group is not in the schema",
                );
                continue;
            };
            let base = match self.ingest_base(&il.name) {
                Ok(base) => base,
                Err(e) => {
                    tracing::error!(
                        target: "zend::calibrate",
                        layer = %il.name,
                        "the layer declares a calibration but cannot be calibrated: {e:#}",
                    );
                    continue;
                }
            };
            calibrate_layer(
                &self.engine,
                &CalibrateLayer { spec, group, base },
                &CalibrateFrame {
                    system_prompt: &self.formatted_prompt,
                    config: &self.refresh_config,
                    thinking: turn_triggers(self, ThinkMode::Balanced, ToolMode::None),
                },
            );
        }
    }

    /// Learn the chunk exclusions ([`Self::learn_chunk_exclusions`]) on a
    /// thread of their own, so the next pass starts without waiting for them —
    /// and not again while a learn is running: the pass after it learns from
    /// what was written meanwhile.
    ///
    /// **Not at the end of the pass.** Learned there, every pass ended with
    /// the scans: on BlackWorld (2026-10-28) a pass that read its units at
    /// 23:53:43 learned until 23:57:01, and the layers read nothing, the GPU
    /// idle, for those 3 m 18 s. The scans are background jobs the scheduler
    /// runs between the next pass's waves.
    pub(crate) fn learn_chunk_exclusions_behind(self: &Arc<Self>) {
        if self.branch_layers.is_empty() || self.learning_exclusions.swap(true, Ordering::SeqCst) {
            return;
        }
        let state = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("zend-exclusions".into())
            .spawn(move || {
                state.learn_chunk_exclusions();
                state.learning_exclusions.store(false, Ordering::SeqCst);
            });
        if let Err(e) = spawned {
            self.learning_exclusions.store(false, Ordering::SeqCst);
            tracing::warn!(
                target: "zend::calibrate",
                "the chunk exclusions were not learned (no thread): {e}"
            );
        }
    }

    /// Learn the chunk exclusion of every branch layer that declares a
    /// `calibrate:` block: the calibrations written are the background its
    /// generic chunks are told apart by. A standing exclusion is kept until its
    /// questions move by a tenth.
    ///
    /// **After the pass, never ahead of it.** The learn is a few hundred scans,
    /// each one background job the scheduler runs between waves; under live
    /// conversations that is many minutes, and a pass waiting on it reads
    /// nothing meanwhile — on BlackWorld (2026-10-26) restarts a quarter of an
    /// hour apart left the ingest reading no file for hours. The price is the
    /// first pass after a restart, which has no exclusion: the units its
    /// derived layers write are projected without the cut. On the
    /// calibration competition the cut measured neutral (MRR 0.251 → 0.250);
    /// on content alone it bought MRR 0.172 → 0.186
    /// (`docs/provenance_scoped_signatures.md` §2.6). Run behind the passes
    /// ([`Self::learn_chunk_exclusions_behind`]).
    fn learn_chunk_exclusions(&self) {
        for (il, _) in &self.branch_layers {
            if crate::ingest_cancel::cancelled() {
                return;
            }
            if il.calibrate.is_none() {
                continue;
            }
            let Some(group) = self.refresh_builder.id_for_group(&il.group) else {
                continue;
            };
            // Each competition in the fold groups its live scans weigh.
            let Some(policy) = self
                .refresh_builder
                .schema()
                .layers
                .iter()
                .flat_map(|l| &l.groups)
                .find(|g| g.id == group)
                .map(|g| &g.policy)
            else {
                continue;
            };
            let content = (
                GroupSet::weighed(&policy.layer_weights, FOLD_GROUPS),
                policy.scan.gallery_keep,
            );
            let calibration = (
                GroupSet::weighed(&policy.calibration_layer_weights, FOLD_GROUPS),
                policy.scan.calibration_gallery_keep,
            );
            let scanner = self.engine.group_scanner();
            match scanner.learn_chunk_exclusion(
                group,
                EXCLUSION_QUESTIONS_PER_FILE,
                content,
                calibration,
            ) {
                Ok(r) => tracing::info!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    files = r.files,
                    calibrated = r.calibrated,
                    questions = r.probes,
                    chunks = r.chunks,
                    cut = r.cut,
                    calibration_chunks = r.calibration_chunks,
                    calibration_cut = r.calibration_cut,
                    kept = r.kept,
                    "the layer's anomalous-chunk exclusion",
                ),
                Err(e) => tracing::warn!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    "learning the layer's chunk exclusion failed (its scans cut nothing): {e:#}",
                ),
            }
        }
    }

    /// The prefilled template a `Folders`/`Files` ingest layer's units fork
    /// their own conversation from — see [`InferenceState::ingest_bases`].
    /// `Err` only for a layer name that isn't one of those two modes (a
    /// caller bug: `ingest_bases` is built for every such layer at boot).
    fn ingest_base(&self, layer_name: &str) -> anyhow::Result<&Sequence> {
        self.ingest_bases
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("no ingest base built for layer '{layer_name}'"))
    }

    /// One pass of the background ingest worker (`crate::ingest_worker`) —
    /// at startup, and after every fetch that moved a branch on origin.
    ///
    /// The branch layers (`repo_map`, `code_reading`) are planned against
    /// every record branch together (`crate::branch_ingest`): what no branch
    /// holds is tombstoned, what nothing holds yet is ingested, each unit at
    /// the commit it was found on. A raw layer in the registry re-reads its
    /// folder's ChatML records; on a content-hash change it mints a fresh
    /// timeline, re-prefills, tombstones the old, and swaps it in.
    ///
    /// Stale-better-than-missing holds throughout: an old generation stays
    /// alive until its replacement commits. Returns `Ok(true)` if any layer
    /// changed.
    pub(crate) fn refresh_ingest_layers(&self) -> anyhow::Result<bool> {
        if self.ingest_layers.is_empty() {
            return Ok(false);
        }
        // Throwaway progress sink (a refresh, not the model-load lifecycle) —
        // silent so a pass doesn't log "load step started Loading model".
        let progress = Arc::new(LoadProgress::silent());
        let mut any = false;
        self.calibrate_layers();
        if !self.branch_layers.is_empty() {
            let passes = self.layer_passes()?;
            let ctx = self.refresh_ctx();
            any |= self
                .branch_ingest
                .pass(&ctx, &self.workspace, &passes, &progress)?;
        }
        for il in &self.ingest_layers {
            // Cooperative shutdown: stop the background reconcile between layers so
            // a Ctrl-C during a post-`ready` re-ingest stops promptly (the per-file
            // worker pool also polls the same cancel). `shutdown` then drains the
            // engine via the ready path.
            if crate::ingest_cancel::cancelled() {
                break;
            }
            if il.mode != IngestMode::Raw {
                continue;
            }
            let snapshot = {
                let convs = self.ingest_convs.lock().unwrap();
                convs
                    .get(&il.name)
                    .map(|c| (c.sequence.timeline_id(), c.state.clone()))
            };
            let Some((old_timeline, prior_state)) = snapshot else {
                continue;
            };
            let ctx = self.refresh_ctx();
            let outcome = crate::raw_read::refresh_raw(
                &ctx,
                &self.workspace.root().join(&il.folder),
                &prior_state,
                old_timeline,
                &progress,
                &il.name,
                &il.group,
            )?;
            if let crate::raw_read::RefreshOutcome::Replaced { sequence, state } = outcome {
                self.ingest_convs
                    .lock()
                    .unwrap()
                    .insert(il.name.clone(), IngestConv { sequence, state });
                any = true;
                tracing::info!(layer = %il.name, "raw ingest layer refreshed");
            }
        }
        Ok(any)
    }

    /// Ingest **only** the given workspace-relative files — in the uploads
    /// repository — into the projection's per-file ingest layer: the upload's
    /// read_file phase. Bounded to those files (no branch walk, no sweep), so
    /// it can't overload the model with the entire repo. Reports per-file
    /// progress into `progress`. Returns `(ingested, failed)`: whether anything
    /// was read, and whether at least one file's ingest tolerated-failed (e.g.
    /// out of KV VRAM) — so the upload can report a real failure.
    ///
    /// If the projection declares no per-file ingest layer (a pure conversational
    /// mind), there is nowhere to read uploads into — a no-op. A per-file layer
    /// suppressed by `--disable-layer` still accepts uploads (they're bounded and
    /// safe).
    pub(crate) fn ingest_uploaded_files(
        &self,
        conv_id: &str,
        rel_paths: &[String],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let Some(files_layer) = self
            .ingest_layers
            .iter()
            .find(|l| l.mode == IngestMode::Files)
        else {
            tracing::debug!(
                "upload ingest: projection has no per-file ingest layer — skipping read_file"
            );
            return Ok((false, false));
        };
        // The uploads are the conversation's: its files, as its events hold
        // them, are what the read reads.
        let files = self.saved_files(conv_id);
        let ctx = RefreshContext {
            tool_ctx: self.tool_host.context_for(ToolMode::Restricted, &files, ""),
            ..self.refresh_ctx()
        };
        let (ingested, n_failed) = crate::code_read::ingest_files(
            &ctx,
            rel_paths,
            progress,
            &files_layer.name,
            self.ingest_base(&files_layer.name)?,
            files_layer.calibrate.as_deref(),
        )?;
        if ingested {
            // An upload is in every conversation's scope once it is committed.
            self.retrieval_scope.refresh(&self.engine);
        }
        Ok((ingested, n_failed > 0))
    }

    /// `conv_id`'s file stores as its saved events hold them: a fresh set,
    /// restored, never saved — what a read of its uploads reads.
    pub(crate) fn saved_files(&self, conv_id: &str) -> Arc<RepoFiles> {
        let files = self.tool_host.conversation_files();
        let mirror = Mutex::new(Mirror::unread());
        conv_overlay::restore(&self.engine, timeline_for(conv_id), &files, &mirror);
        files
    }

    /// Run `change` over `conv_id`'s file stores and save what it changed as
    /// the conversation's file events (§10.2). A conversation open here is
    /// changed under its own lock, so an upload and a turn never interleave;
    /// one that is not is changed over its stores restored from those events,
    /// put on each repository's base first when it has never worked anywhere.
    pub(crate) async fn change_files<T>(
        &self,
        conv_id: &str,
        change: impl FnOnce(&RepoFiles) -> T,
    ) -> T {
        let open = self.conversations.get(conv_id);
        if let Some(conv) = open {
            let cs = conv.lock_owned().await;
            let out = change(&cs.files);
            conv_overlay::save(&self.engine, timeline_for(conv_id), &cs.files, &cs.mirror);
            return out;
        }
        self.change_saved(conv_id, change)
    }

    /// [`Self::change_files`] for a conversation not open here: its stores
    /// restored from its events, changed, and saved back.
    pub(crate) fn change_saved<T>(&self, conv_id: &str, change: impl FnOnce(&RepoFiles) -> T) -> T {
        let timeline = timeline_for(conv_id);
        let files = self.tool_host.conversation_files();
        let mirror = Mutex::new(Mirror::unread());
        {
            let engine = &self.engine;
            conv_branches::seed(engine, timeline, &self.base_branches);
            conv_overlay::restore(engine, timeline, &files, &mirror);
        }
        let out = change(&files);
        conv_overlay::save(&self.engine, timeline, &files, &mirror);
        out
    }

    /// Uploads kept before conversations held them, each put in its
    /// conversation's uploads store and its upload record pointed at it; the
    /// old folder removed once every file is held (§10.2, §15.1). A file that
    /// cannot be stored keeps the folder, so nothing is lost.
    pub(crate) fn take_in_legacy_uploads(&self) {
        let root = self.workspace.root();
        let legacy = match crate::legacy_uploads::read(root) {
            Ok(legacy) => legacy,
            Err(e) => {
                tracing::warn!("the old uploads could not be read, and are left as they are: {e}");
                return;
            }
        };
        if legacy.is_empty() {
            return;
        }
        let mut all_held = true;
        for (conv_id, files) in legacy {
            let wanted = files.len();
            let moved: Vec<(u64, UploadInfo)> = self.change_saved(&conv_id, |repos| {
                let Ok(store) = repos.repo(UPLOADS_REPO) else {
                    return Vec::new();
                };
                files
                    .into_iter()
                    .filter_map(|f| {
                        let size = f.bytes.len() as u64;
                        match crate::uploads::store(&store, &f.name, f.bytes) {
                            Ok(stored) => Some((f.id, upload_record(stored, size))),
                            Err(e) => {
                                tracing::warn!(name = %f.name, "an old upload was not taken in: {e}");
                                None
                            }
                        }
                    })
                    .collect()
            });
            all_held &= moved.len() == wanted;
            let timeline = timeline_for(&conv_id);
            let written = edit_upload_records(&self.engine, timeline, |records| {
                for (old, new) in &moved {
                    match records.iter_mut().find(|r| r.id == *old) {
                        Some(r) => {
                            r.id = new.id;
                            r.path = new.path.clone();
                            r.name = new.name.clone();
                        }
                        None => records.push(new.clone()),
                    }
                }
            });
            if !written {
                tracing::warn!(conv_id = %conv_id, "an old upload's record was not moved");
                all_held = false;
            }
        }
        if !all_held {
            tracing::warn!("some old uploads were not taken in; their folder is kept");
            return;
        }
        match crate::legacy_uploads::remove(root) {
            Ok(()) => tracing::info!("the old uploads are held by their conversations"),
            Err(e) => tracing::warn!("the old uploads folder could not be removed: {e}"),
        }
    }

    /// Tombstone the per-file `code_read` conversation of every uploaded file
    /// that no conversation holds any more.
    ///
    /// Uploads are endpoint-managed and on no branch, so the branch pass never
    /// retires them — that would delete uploaded content on the next pass. But
    /// a deleted upload must still retire its conversation; otherwise deleted
    /// uploads accumulate live turns forever, growing the redo log and the
    /// projection candidate set without bound.
    ///
    /// An upload is held while some conversation's upload record names its
    /// path (§10.2), or while the uploads folder — the lower layer every
    /// conversation's uploads store reads — still has it. Cheap and
    /// self-limiting: a metadata scan, no walk and no re-ingest.
    ///
    /// Runs at startup, to retire uploads deleted since the last one.
    pub(crate) fn reconcile_uploaded_files(&self) {
        let engine = &self.engine;
        let held: HashSet<String> = engine
            .conversations_with_metadata_key("uploads")
            .into_iter()
            .filter_map(|(_, json)| serde_json::from_str::<Vec<UploadInfo>>(&json).ok())
            .flatten()
            .map(|u| u.path)
            .collect();
        let mut tombstoned = 0usize;
        for (tl, path) in engine.conversations_with_metadata_key("path") {
            if !crate::code_read::is_upload_path(&path) {
                continue;
            }
            // `path` is workspace-relative with `/` separators; `Path::join`
            // accepts them on win32, and `exists()` is case-insensitive there.
            if held.contains(&path) || self.workspace.root().join(&path).exists() {
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

/// The reserved section holding `mode`'s tool-catalog summary; `None` projects
/// no tools and has none.
fn tool_summary_section(mode: ToolMode) -> Option<Reserved> {
    match mode {
        ToolMode::None => None,
        ToolMode::Restricted => Some(Reserved::ToolSummaryRestricted),
        ToolMode::Comprehensive => Some(Reserved::ToolSummary),
    }
}

/// Build the projection for one tools mode by cloning `base` and filtering the
/// `tools` collection MEMBERS to the tools the mode offers
/// ([`crate::tools::offered_tool_names`]). These builders control only WHICH
/// members project — the WHOLE tool block (markers, catalog, summary) is gated
/// separately by the `tools_enabled` optional_group, which chat.rs sets
/// `absent` for `None`. Called once per mode at startup by
/// [`ModeBuilders::build`]; the results are cached and handed out as cheap
/// `Arc` clones each turn, since both projection and reprojection just read the
/// swapped builder.
fn build_mode_builder(base: &Builder, mode: ToolMode) -> anyhow::Result<Arc<Builder>> {
    let mut b = base.clone();
    // Keep exactly the tools this mode offers — none in None, everything in
    // Comprehensive; the summary association below points the mode at its listing.
    b.retain_collection_sections("tools", &crate::tools::offered_tool_names(mode))
        .map_err(|e| anyhow::anyhow!("{} tools projection: {e}", mode.id()))?;
    // Associate the sealed tool-catalog summary (built by `build_tool_summary`,
    // prefilled into its reserved section at startup) with the `tools` collection,
    // so projection emits the FULL tool-name listing just before the selected
    // subset. Without this the catalog K/V is sealed but orphaned — the model only
    // ever sees the 1-3 provenance-selected tools and concludes a tool it can't see
    // (e.g. `datetime` on a "what is the time?" turn) doesn't exist. The projection
    // emits it whenever the selection is a proper subset (§ `record` in
    // `emit_system_prompt_items`), which for a 93-tool catalog is every turn.
    // `None` mode has no tools block, so no summary.
    if let Some(reserved) = tool_summary_section(mode) {
        let tools = b
            .id_for_system_collection("tools")
            .ok_or_else(|| anyhow::anyhow!("projection schema missing 'tools' collection"))?;
        b.set_collection_summary_section(tools, SectionId::reserved(reserved))
            .map_err(|e| anyhow::anyhow!("tool summary association ({mode:?}): {e}"))?;
    }
    Ok(Arc::new(b))
}

/// The per-tools-mode projection builders, built once at startup so each turn
/// hands out a cheap `Arc` clone instead of re-cloning the ~93-section schema.
struct ModeBuilders {
    /// Indexed by [`ToolMode::level`].
    builders: [Arc<Builder>; ToolMode::ALL.len()],
}

impl ModeBuilders {
    /// Every mode's builder, or the first failure. A failure is only reachable
    /// through a missing dialogue layer or `tools` collection — a broken schema
    /// the daemon cannot serve — and falling back to another mode's builder
    /// would offer a mode tools it does not grant.
    fn build(base: &Builder) -> anyhow::Result<Self> {
        let [none, restricted, comprehensive] =
            ToolMode::ALL.map(|mode| build_mode_builder(base, mode));
        Ok(Self {
            builders: [none?, restricted?, comprehensive?],
        })
    }

    /// The prebuilt projection for `mode` (a cheap `Arc` clone).
    fn get(&self, mode: ToolMode) -> Arc<Builder> {
        Arc::clone(&self.builders[mode.level() as usize])
    }
}

/// The `identity_anchor` + `identity` members that belong to one identity
/// (namespaced `<name>::…`). The keep-sets a `retain_collection_sections` scopes
/// a conversation down to.
#[derive(Default)]
struct IdentityKeep {
    anchor: HashSet<String>,
    facets: HashSet<String>,
}

/// Which identity scope a turn's projection uses — the three genuine states the
/// resolver distinguishes so an unresolved identity never falls through to the
/// unscoped builder (which would emit every persona's always-visible anchor).
enum IdentityScope {
    /// No identities are installed at all (e.g. the coding-assistant schema whose
    /// projection has no `identity`/`identity_anchor` collections). Use the plain
    /// tools-mode builder; nothing identity-shaped exists to emit.
    Unscoped,
    /// Scope both identity collections to this identity's members.
    Named(String),
    /// Identities ARE installed but none resolved (the conversation set none and
    /// `mind.yaml` declares no default). Scope both collections to empty so no
    /// anchor emits — a mind schema with identities should set a default.
    Empty,
}

/// Lazy per-`(identity, ToolMode)` projection builders. Each is a [`ModeBuilders`]
/// view further scoped — via `retain_collection_sections`, which drops members
/// from projection but never re-seals KV — to a single identity's anchor + facets,
/// so a conversation projects only its own identity. Built on first use and
/// cached; bounded by identities-in-use × tool modes. KV is shared across all of
/// them (every identity's sections seal once on the base conv).
struct IdentityBuilders {
    /// Per-identity keep-sets, computed once at load from the installed members.
    keeps: HashMap<String, IdentityKeep>,
    /// The `mind.yaml` default identity, used when a conversation names none.
    default_identity: Option<String>,
    cache: Mutex<HashMap<(String, ToolMode), Arc<Builder>>>,
}

impl IdentityBuilders {
    /// Build the keep-sets from the loaded identity sections (member ids
    /// `<name>::<stem>`). `default_identity` comes from `mind.yaml`.
    fn new(
        sections: &crate::response_section::IdentitySections,
        default_identity: Option<String>,
    ) -> Self {
        let mut keeps: HashMap<String, IdentityKeep> = HashMap::new();
        for s in &sections.anchors {
            if let Some((name, _)) = s.id.split_once("::") {
                keeps
                    .entry(name.to_string())
                    .or_default()
                    .anchor
                    .insert(s.id.clone());
            }
        }
        for s in &sections.facets {
            if let Some((name, _)) = s.id.split_once("::") {
                keeps
                    .entry(name.to_string())
                    .or_default()
                    .facets
                    .insert(s.id.clone());
            }
        }
        Self {
            keeps,
            default_identity,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve which identity scope a turn's projection uses: the conversation's
    /// identity, else the `mind.yaml` default. Three genuine states — collapsing
    /// them onto `Option` is what leaked every anchor: an unresolved identity when
    /// identities ARE installed must scope to empty, NOT fall through to the
    /// unscoped builder (whose AlwaysVisible `identity_anchor` collection emits
    /// every persona's anchor at once).
    fn resolve(&self, conv_identity: Option<&str>) -> IdentityScope {
        if self.keeps.is_empty() {
            return IdentityScope::Unscoped;
        }
        match conv_identity
            .map(str::to_string)
            .or_else(|| self.default_identity.clone())
        {
            Some(name) => IdentityScope::Named(name),
            None => IdentityScope::Empty,
        }
    }

    /// Retain both identity collections on `b` to `name`'s installed members. An
    /// unknown name — including the reserved empty sentinel — scopes to empty. The
    /// single home of the identity retain, shared by `get`, `get_empty`, and the
    /// hires capture path so scoping can never diverge between them.
    fn scope_to(&self, b: &mut Builder, name: &str) {
        let empty = HashSet::new();
        let (anchor_keep, facet_keep) = match self.keeps.get(name) {
            Some(k) => (&k.anchor, &k.facets),
            None => (&empty, &empty),
        };
        if let Err(e) = b.retain_collection_sections("identity_anchor", anchor_keep) {
            tracing::warn!("scope identity_anchor to {name}: {e}");
        }
        if let Err(e) = b.retain_collection_sections("identity", facet_keep) {
            tracing::warn!("scope identity to {name}: {e}");
        }
    }

    /// Scope BOTH identity collections to empty — no anchor, no facets. Used when
    /// identities are installed but none resolved (see [`IdentityScope::Empty`]).
    /// Cached under a reserved key that no real identity name can collide with.
    fn get_empty(&self, mode_builders: &ModeBuilders, mode: ToolMode) -> Arc<Builder> {
        let key = ("\0empty".to_string(), mode);
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Arc::clone(b);
        }
        let mut b = (*mode_builders.get(mode)).clone();
        self.scope_to(&mut b, "\0empty"); // sentinel is not a real identity → empty
        let arc = Arc::new(b);
        self.cache.lock().unwrap().insert(key, Arc::clone(&arc));
        arc
    }

    /// Apply the conversation's identity scope to an arbitrary builder. The hires
    /// capture path clones the UNSCOPED base, so without this a capture would emit
    /// every persona's always-visible anchor. No-op when no identities are
    /// installed (the coding-assistant schema).
    fn scope_hires(&self, b: &mut Builder, conv_identity: Option<&str>) {
        match self.resolve(conv_identity) {
            IdentityScope::Named(name) => self.scope_to(b, &name),
            IdentityScope::Empty => self.scope_to(b, "\0empty"),
            IdentityScope::Unscoped => {}
        }
    }

    /// The projection scoped to `name` for `mode` — cloned from the tools-mode
    /// builder and retained on first use, then cached. An unknown `name` scopes
    /// both identity collections to EMPTY (no identity surfaces) with a warning,
    /// never the unscoped builder — which would leak every identity's
    /// always-visible anchor at once.
    fn get(&self, mode_builders: &ModeBuilders, name: &str, mode: ToolMode) -> Arc<Builder> {
        let key = (name.to_string(), mode);
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Arc::clone(b);
        }
        if !self.keeps.contains_key(name) {
            tracing::warn!(identity = %name, "unknown identity — scoping to empty (no identity surfaced)");
        }
        let mut b = (*mode_builders.get(mode)).clone();
        self.scope_to(&mut b, name);
        let arc = Arc::new(b);
        self.cache.lock().unwrap().insert(key, Arc::clone(&arc));
        arc
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
fn build_hires_projection(
    state: &InferenceState,
    spec: &str,
    conv_identity: Option<&str>,
) -> anyhow::Result<Arc<Builder>> {
    let mut b = state.refresh_builder.clone();
    if let Some((collection, section)) = spec.split_once('/') {
        b.set_collection_single_section(collection, section)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    } else {
        b.set_collection_selection(spec, SelectionRule::AlwaysVisible)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    }
    // Scope to the conversation's identity so a capture reflects its real
    // projection, not every persona's always-visible anchor — the base
    // `refresh_builder` this clones is unscoped. Independent of the forced
    // collection above, so order does not matter.
    state.identity_builders.scope_hires(&mut b, conv_identity);
    Ok(Arc::new(b))
}

// The submit parameter list, threaded from the HTTP handler through
// `submit`/`submit_with_sampling` to here. Bundling it into a request struct is
// worth doing, but it is a change to the request path rather than a lint fix.
#[allow(clippy::too_many_arguments)]
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
    identity: Option<String>,
    selection: candle_conversation::SelectionState,
    // Not from the HTTP handler: set only by the boot scan, naming a sealed
    // call turn whose tools never delivered. `user_message` is then unused —
    // the turn's message is the tool output, produced below.
    resume_call_turn: Option<u32>,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
    let (tx, rx) = mpsc::channel::<anyhow::Result<StreamItem>>(64);

    // An async task, not a blocking one: every wait in the body is an await —
    // the turn's event stream, the client channel's backpressure — and the
    // per-conversation lock it holds across the decode is async precisely so
    // this task parks instead of pinning a thread. A client that disconnects
    // drops `rx`; the next send fails, the task returns, and dropping the
    // in-flight `TurnHandle` stops the decode at its next step.
    tokio::spawn(async move {
        let msg_preview: String = user_message.chars().take(60).collect();
        tracing::info!(
            conv_id = %conv_id,
            msg_len = user_message.len(),
            "inference start  \"{}{}\"",
            msg_preview,
            if user_message.len() > 60 { "…" } else { "" },
        );

        let timeline = timeline_for(&conv_id);
        // Only a freshly-forked conv uses the stored identity; a reused conv
        // keeps its in-memory identity.
        let stored_identity = state
            .engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get("identity").cloned());
        // What the fork that mints this conversation hands back beside it: the
        // per-conversation exclusivity guard, and the file stores and mirror
        // that are restored from its events once it is open.
        //
        // The guard is taken inside the fork, before `conversations` hands the
        // new state to anyone: a second request for the same, not-yet-resident
        // conv_id awaits the fork and then parks on this guard. Without it, that
        // request could run a full turn against a `ConvState` whose files are
        // not yet restored from the substrate (`upper.base` still unset) — and
        // then have that turn's writes silently overwritten when this request's
        // own restore, below, replaces the store's contents wholesale. See the
        // note where the guard is picked back up, and
        // `zend_run_iteration_traps.md` for the trace that found this.
        type Minted = (
            OwnedMutexGuard<ConvState>,
            (Arc<RepoFiles>, Arc<Mutex<Mirror>>),
        );
        let (minted_tx, mut minted_rx) = mpsc::unbounded_channel::<Minted>();
        let forked = state
            .conversations
            .get_or_try_init(&conv_id, Instant::now(), || {
                let (state, minted_tx) = (Arc::clone(&state), minted_tx.clone());
                let (conv_id, identity) = (conv_id.clone(), stored_identity.clone());
                async move {
                    tracing::info!(conv_id = %conv_id, "forking new conv from base");
                    // Fork onto the timeline derived from `conv_id` — a stable
                    // hash, so a daemon restart reconnects the client to the
                    // turns the substrate reload recovered for this
                    // conversation (§16.12). An unknown conv_id simply forks
                    // empty. On a blocking thread: the fork is a scheduler
                    // round-trip.
                    let fork_state = Arc::clone(&state);
                    let conv = tokio::task::spawn_blocking(move || {
                        fork_state.base_conv.fork_resuming(timeline)
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("fork task: {e}"))?
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let files = state.tool_host.conversation_files();
                    let mirror = Arc::new(Mutex::new(Mirror::unread()));
                    let arc = Arc::new(ConvLock::new(ConvState {
                        conv,
                        identity,
                        files: Arc::clone(&files),
                        mirror: Arc::clone(&mirror),
                        // An earlier run may have left this conversation's
                        // sections: the first reconcile looks for them.
                        journal: Carried::new(false),
                        promise_marks: Vec::new(),
                    }));
                    let guard = Arc::clone(&arc)
                        .try_lock_owned()
                        .expect("a conversation nothing has been handed yet is locked by nothing");
                    let _ = minted_tx.send((guard, (files, mirror)));
                    Ok::<_, anyhow::Error>(arc)
                }
            })
            .await;
        // The last fork this request ran is the one it minted (an earlier one
        // was dropped for an entry that won the key).
        let mut minted_by_us: Option<Minted> = None;
        while let Ok(m) = minted_rx.try_recv() {
            minted_by_us = Some(m);
        }
        let (forked, minted) = match forked {
            Ok((arc, minted)) => {
                if !minted {
                    tracing::debug!(conv_id = %conv_id, "reusing existing conv");
                }
                (Ok(arc), minted)
            }
            Err(e) => (Err(e), false),
        };
        let (minted_guard, minted_files) = match minted_by_us.filter(|_| minted) {
            Some((guard, files)) => (Some(guard), Some(files)),
            None => (None, None),
        };
        // Record the fork's lineage. `base_conv` is itself parented onto the
        // priming chain, so this one hop is what puts the anchor documents in
        // front of a brand new conversation — nothing is copied, and the
        // pointer is persisted metadata, so it survives a restart.
        //
        // Best-effort: a conversation that answers without its inherited
        // context is far better than one that refuses to open.
        if minted {
            let base_timeline = state.base_conv.timeline_id();
            if let Err(e) = state.engine.set_forked_from(timeline, base_timeline) {
                tracing::warn!(
                    conv_id = %conv_id,
                    "recording fork lineage failed, conversation starts unprimed: {e}",
                );
            }
        }
        let conv_arc = match forked {
            Ok(arc) => arc,
            Err(e) => {
                tracing::error!(conv_id = %conv_id, "fork failed: {e}");
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        // Held across every decode await below, deliberately: this guard IS
        // the per-conversation in-flight exclusivity — a second submit for
        // the same conv_id parks here until this turn loop finishes.
        //
        // For a minted conversation this is `minted_guard`, taken inside the
        // fork before `state.conversations` handed the state to anyone (see the
        // note there) — never re-acquired here, only picked back up, so there
        // is no gap between "findable" and "exclusive" for anything freshly
        // minted. A reused conversation had no reason to hold anything
        // earlier, so it acquires its own guard here — which is, as it always
        // was, the first moment this codepath asks for one.
        let mut cs = match minted_guard {
            Some(guard) => guard,
            None => conv_arc.clone().lock_owned().await,
        };
        // The slot was seeded from this (fresh) timeline before the lineage
        // above existed; ask again so the conversation opens with the priming
        // chain's recurrent memory behind it rather than from nothing.
        //
        // On a blocking thread: the seed is a scheduler round-trip that waits
        // on the queue behind whatever wave is running (measured in the
        // hundreds of ms for its sibling `InstallRecurrentState`), and this
        // task's contract is that every wait in it is an await — see the note
        // where it is spawned. Blocking here would pin a runtime worker and
        // stall every other stream sharing it.
        //
        // `cs` is moved into the closure and handed back rather than
        // re-locked through `conv_arc`: it is already held, and
        // `OwnedMutexGuard` is not reentrant — a second lock through the same
        // Arc from this same logical task would deadlock against itself.
        if minted {
            let cid = conv_id.clone();
            cs = tokio::task::spawn_blocking(move || {
                if let Err(e) = cs.conv.seed_recurrent_from_lineage() {
                    tracing::warn!(
                        conv_id = %cid,
                        "seeding recurrent memory from the priming chain failed: {e}",
                    );
                }
                cs
            })
            .await
            .expect("seed_recurrent_from_lineage task panicked");
        }
        // What a draft of this conversation's journal is put through, so one can
        // start without its lock: the asker its questions fork from.
        if minted && state.journaling {
            let open = Open {
                asker: cs.conv.unsealed_asker(),
            };
            state
                .journals
                .opened(&conv_id, open, || journal_record(&state, timeline));
        }

        // Persist the conv_id ↔ timeline mapping *after* `fork_resuming`
        // has registered the timeline in the substrate — otherwise
        // `set_conv_id` no-ops in-RAM and the sidebar wouldn't see this
        // conversation until the next daemon restart. Idempotent on
        // repeat calls (no-op when the substrate already has it).
        {
            let engine = &state.engine;
            if let Err(e) = engine.set_conversation_conv_id(timeline, &conv_id) {
                tracing::warn!(conv_id = %conv_id, "persist conv_id failed: {e}");
            }
            // A new conversation — or one unarchived since startup — starts on
            // each repository's base; one already working somewhere keeps it,
            // and then this writes nothing.
            conv_branches::seed(engine, timeline, &state.base_branches);
            // A turn is a use: the sidebar lists this conversation first.
            engine.touch_conversation(timeline);
        }

        if let Some((files, mirror)) = minted_files {
            // A conversation built again — after an eviction, after a restart
            // — reads each repository through its own branch, and gets back
            // the changes its tool rounds made there.
            conv_overlay::restore(&state.engine, timeline, &files, &mirror);
            // Its working set is in-memory and did not survive. The seeds
            // resolve again on its own branch and enter provenance at the
            // seeding momentum; the locks come back from the marks its turns
            // carry — the promises it was actually given — and the rest of
            // the momentum rebuilds from the scan.
            if let Some(config) = state.working_set.clone() {
                let open_state = Arc::clone(&state);
                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let t_open = Instant::now();
                    open_state
                        .engine
                        .working_set_clear(timeline);
                    let clear_ms = t_open.elapsed().as_millis() as u64;
                    let t_scope = Instant::now();
                    // Scope it first. A seed or a restored lock is admitted only
                    // as the conversation's scope offers it, and the scope is
                    // otherwise first set right before its first submit — after
                    // this — so every seed and every lock was refused: measured
                    // 2026-10-24, no dialogue had carried a seed since scoping
                    // landed.
                    open_state
                        .retrieval_scope
                        .apply(&open_state.engine, timeline, &files);
                    let scope_ms = t_scope.elapsed().as_millis() as u64;
                    let t_seed = Instant::now();
                    let seeded = seed_working_set(&open_state, timeline, &files, &config);
                    let seed_ms = t_seed.elapsed().as_millis() as u64;
                    let t_restore = Instant::now();
                    let restored = restore_locks(&open_state.engine, timeline);
                    let restore_ms = t_restore.elapsed().as_millis() as u64;
                    tracing::info!(
                        target: "zend::working_set",
                        timeline = timeline.raw(),
                        seeded,
                        restored,
                        clear_ms,
                        scope_ms,
                        seed_ms,
                        restore_ms,
                        "working set opened: seeds in provenance, locks restored from the turn marks",
                    );
                })
                .await
                {
                    tracing::warn!(conv_id = %conv_id, "opening the working set panicked: {e}");
                }
            }
        }

        // An explicit request `identity` overrides and is persisted, so later
        // turns (and a daemon restart) keep it. Only writes when it actually
        // changes, to avoid a substrate write every turn.
        if let Some(req_id) = identity {
            // `cs` is already held (see above) — no reason to re-lock through
            // `conv_arc` for what is the same guard under a different name.
            let changed = if cs.identity.as_deref() == Some(req_id.as_str()) {
                false
            } else {
                cs.identity = Some(req_id.clone());
                true
            };
            if changed {
                if let Err(e) = state
                    .engine
                    .set_conversation_metadata(timeline, "identity", &req_id)
                {
                    tracing::warn!(conv_id = %conv_id, "persist identity failed: {e}");
                }
            }
        }

        // The dials this turn ran under, persisted so both a resumed turn and
        // the GUI reopening the conversation use the settings it was last set
        // to rather than the defaults. Written only when they change, as
        // `identity` is — otherwise every turn costs a metadata record.
        {
            let dials = dials_metadata(&selection, tools_mode);
            let engine = &state.engine;
            let stored = engine.conversation_metadata(timeline).unwrap_or_default();
            if dials.iter().any(|(k, v)| stored.get(k) != Some(v)) {
                if let Err(e) = engine.set_conversation_metadata_many(timeline, &dials) {
                    tracing::warn!(conv_id = %conv_id, "persist dials failed: {e}");
                }
            }
        }

        // Lossless capture: seal this conversation's turns WITHOUT KV
        // quantization — K/V persist in native R16/F16 so the provenance work
        // gets full-resolution keys. Set before the first turn's residence is
        // allocated so the hot→warm migration inherits it.
        if lossless_kv {
            state.engine.set_timeline_compression(
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
        let already_labelled = state.engine.conversation_label_of(timeline).is_some();
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

        // Per-conversation projection swap. Both the prefill projection and the
        // reprojection read the swapped builder, so neither re-introduces a
        // filtered section mid-conversation.
        //
        // `force_hires` is the zend capture aid (force one collection to
        // AllVisible) and wins when present; otherwise apply the composer's
        // tools mode — Comprehensive restores the full catalog (so switching a
        // conversation back from Restricted/None works), Restricted drops the
        // high-risk tools, None drops the whole catalog.
        // `tool_modes` records the mode actually applied, so the projection
        // panel shows the matching tool summary.
        if let Some(ref coll) = force_hires {
            match build_hires_projection(&state, coll, cs.identity.as_deref()) {
                Ok(b) => {
                    cs.conv.set_projection(b);
                    // The capture override forces the full catalog (AllVisible),
                    // so the panel should show the comprehensive summary.
                    state
                        .tool_modes
                        .lock()
                        .unwrap()
                        .insert(conv_id.clone(), ToolMode::Comprehensive);
                    tracing::info!(conv_id = %conv_id, collection = %coll,
                        "force-high-resolution: collection forced to AllVisible");
                }
                Err(e) => {
                    tracing::warn!(conv_id = %conv_id, "force-high-resolution failed: {e}")
                }
            }
        } else {
            // Applied every turn so a mid-conversation dial change takes effect,
            // and both projection and reprojection use it.
            let proj = turn_projection(&state, &conv_id, cs.identity.as_deref(), tools_mode);
            cs.conv.set_projection(proj);
            state
                .tool_modes
                .lock()
                .unwrap()
                .insert(conv_id.clone(), tools_mode);
        }

        let original_user_message = user_message.clone();
        let mut current_message = TurnText::from(user_message);

        // A tool round this daemon never finished. Run its tools now and enter
        // the loop at a tool round rather than at a fresh question, so the
        // response turn projects as it would have: without the worked
        // demonstration, and without re-applying the caller's prefill.
        // Every `cs.conv` touch below is its own synchronous statement. A
        // `&Sequence` is not `Send` — a conversation owns its background
        // cognitive tasks — so one held across the await would make this whole
        // task non-`Send`, and the spawn would not compile.
        let mut start_iteration = 0usize;
        // The working-set marks the next submitted turn carries: what the
        // round it answers did to the working set, in the order it happened
        // (`docs/zend_working_set.md` §4.8).
        let mut marks: Vec<String> = Vec::new();
        if let Some(call_turn) = resume_call_turn {
            let calls = resumed_calls(&cs.conv, &conv_id, timeline, call_turn);
            let n_calls = calls.len();
            if calls.is_empty() {
                resume::clear_call_turn(&state.engine, timeline);
                return;
            }
            // The round is finished over the conversation's own files, under
            // the grants of the mode it was held at.
            let ctx = state.tool_host.dialogue_context(
                tools_mode,
                &cs.files,
                &conv_id,
                journal_shelf(&state, &conv_id, timeline),
            );
            let dispatched =
                tokio::task::spawn_blocking(move || run_tool_calls(&ctx, calls, Dispatch::Resumed))
                    .await;
            conv_overlay::save(&state.engine, timeline, &cs.files, &cs.mirror);
            let results = match dispatched {
                Ok(results) => results,
                Err(e) => {
                    tracing::error!(
                        conv_id = %conv_id,
                        call_turn,
                        "resume: tool dispatch panicked: {e}",
                    );
                    resume::clear_call_turn(&state.engine, timeline);
                    return;
                }
            };
            // A resumed round never passed the screen, but it can still have
            // written: then what it wrote ends, exactly as a live round's would.
            marks.extend(end_written_promises(&state, timeline, &results).await);
            let text = format_tool_responses(&results);
            if text.is_blank() {
                tracing::warn!(
                    conv_id = %conv_id,
                    call_turn,
                    n_calls,
                    "resume: the re-issued round produced no response text",
                );
                resume::clear_call_turn(&state.engine, timeline);
                return;
            }
            // The one window in which the coupling is knowable: the tools have
            // returned and the response turn is certain to follow. Nothing
            // recomputes it afterwards, so without this the call turn and its
            // response never become one exchange.
            if let Err(e) = cs.conv.couple_turn(call_turn) {
                tracing::warn!(conv_id = %conv_id, call_turn, "resume: couple_turn: {e}");
            }
            tracing::info!(
                conv_id = %conv_id,
                call_turn,
                n_calls,
                "resume: finishing a tool round the daemon stopped in the middle of",
            );
            current_message = text;
            start_iteration = 1;
        }

        // The reflection-marker suppression ceiling is per-dial: derive the think
        // mode once, materialise the turn's sampling config (the conversation
        // default carries the resolved marker family), and bake in the penalty —
        // Quick/Balanced suppress the "Wait"/"Hmm"/… family in-block, Deep/
        // Exhaustive leave it 0 (reconsideration is wanted there).
        let think_mode = think_mode_from_selection(&selection);
        // Vary the RNG seed per turn from real entropy. The default base seed is
        // a fixed constant, which makes a whole conversation a deterministic
        // replay — the same context always samples the same tokens, so a turn
        // that lands in a bad attractor can never sample its way out. A fresh
        // per-turn seed restores genuine run-to-run variation. (The per-token
        // `rng_offset` still advances within the turn.)
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or_else(|_| cs.conv.default_sampling().seed);
        let sampling = turn_sampling(
            &cs.conv,
            sampling,
            &selection,
            &state.think_closer_phrase,
            seed,
        );

        // The tool loop runs until the model stops emitting tool calls (i.e.
        // produces a final answer). A workflow is as long as it needs to be, so
        // the only bound is [`MAX_TOOL_ROUNDS`], which sits far above any real
        // one — see the note there for why a bound has to exist at all now that
        // a disconnect no longer ends the turn.
        let mut repeat_guard = RepeatGuard::new();
        // Set once the repeat guard closes the loop: the turn that answers the
        // closing round is the last, and no call it makes runs.
        let mut closing = false;
        for iteration in start_iteration.. {
            // The closing turn the cap gives runs below it ([`closes_at_cap`]);
            // a turn resumed at or past the cap ends here.
            if iteration >= MAX_TOOL_ROUNDS {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "tool loop resumed past its round cap — ending the turn on the answer already streamed",
                );
                break;
            }
            // A conversation deleted while its reply runs ends the reply at the
            // next round. A turn runs on with no one listening so a client that
            // reconnects can follow it, but a deleted conversation has no one to
            // come back: its K/V is dead to the compactor, which unlinks the
            // segments holding it, and every further round would reproject turns
            // whose chunks are gone.
            if state.engine.is_timeline_tombstoned(timeline) {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "conversation deleted while its reply ran — ending the turn",
                );
                break;
            }
            tracing::debug!(conv_id = %conv_id, iteration, "submitting turn");
            // Collect this turn's projection events (reprojections + decode-end)
            // so they survive a browser reload (served back on hydrate).
            let mut turn_events: Vec<ProjectionEventOut> = Vec::new();
            // The turn that answers the repeat guard's closing round may not
            // open a tool call: its results said none would run, and a call
            // written anyway would be sealed with no answer.
            // An explicit caller config may carry no resolved id; the
            // conversation's own config always does.
            let mut this_turn = sampling.clone();
            if closing {
                let open = match this_turn.tool_call_open_token_id {
                    id if id >= 0 => id,
                    _ => cs.conv.default_sampling().tool_call_open_token_id,
                };
                repeat_guard::closing_answer(&mut this_turn, open);
            }
            // A tool round's user message is the previous turn's results;
            // it projects without the worked demonstration, which there
            // becomes the question the model answers (`tool_round_selection`).
            let mut round_selection = if iteration == 0 {
                selection.clone()
            } else {
                tool_round_selection(&selection)
            };
            if state.journaling {
                carry_journal(&state, &conv_id, timeline, &mut cs, &mut round_selection).await;
            }
            let options = candle_conversation::TurnOptions {
                max_tokens,
                sampling: Some(this_turn),
                // Apply the caller's assistant prefill only on the first tool
                // iteration — re-prefilling it on every chained iteration would
                // prevent the model ever reaching a final answer.
                assistant_prefill: if iteration == 0 {
                    assistant_prefill.clone()
                } else {
                    None
                },
                selection: round_selection,
                triggers: turn_triggers(&state, think_mode, tools_mode),
                // The promises the last decode found ended come first: they
                // ended before this round's reads were served.
                tags: std::mem::take(&mut cs.promise_marks)
                    .into_iter()
                    .chain(std::mem::take(&mut marks))
                    .collect(),
                ..Default::default()
            };
            // What this turn may retrieve from the branch layers: the units its
            // own base holds. Set before every submit — a tool round can move
            // the base (a commit, a merge, a switch) or change a file. Git reads
            // on a base not yet listed, so on the blocking pool.
            let scope_state = Arc::clone(&state);
            let scope_files = Arc::clone(&cs.files);
            if let Err(e) = tokio::task::spawn_blocking(move || {
                scope_state
                    .retrieval_scope
                    .apply(&scope_state.engine, timeline, &scope_files)
            })
            .await
            {
                tracing::warn!(conv_id = %conv_id, "retrieval scope panicked: {e}");
            }
            let handle = match cs
                .conv
                .submit_turn_with_options(current_message.clone(), options)
            {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "submit_turn failed: {e}");
                    let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
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
            // **A collapsed `<think></think>` must not reach the client.**
            //
            // The engine strips empty think blocks from the text it STORES
            // (`strip_empty_think_blocks`, in `build_turn_layout`), but the
            // stream is assembled from the raw token events and nothing filtered
            // it — so an empty block went out verbatim and rendered as leaked
            // markup. It became the common case with thinking-span projection:
            // older turns have their reasoning windowed out of the K/V, so the
            // model reads a history with no visible reasoning and answers with a
            // collapsed block of its own.
            //
            // Classifying a block needs its `</think>`, which is why this holds
            // rather than filters — but only while the block is still
            // WHITESPACE-ONLY. Real reasoning trips `resolved` on its first
            // non-blank token and streams from then on, so a thinking turn is
            // never held back waiting for its own close.
            let mut think_resolved = false;
            let mut think_progress = ThinkProgress::default();
            let mut done_resp = None;
            let mut turn_error: Option<anyhow::Error> = None;
            // Whether the client stopped reading. Observational only: a turn is
            // not abandoned because nobody is listening. It decodes, runs its
            // tools and seals exactly as it would with the browser attached, so
            // a reload finds the work finished rather than lost — and the
            // substrate never holds half a turn the client's socket decided the
            // length of.
            let mut listener_gone = false;
            // The last "waiting for the GPU" status sent, so a wait reported
            // every second reaches the client only when its text changes.
            let mut last_queued: Option<String> = None;
            // Set when the conversation is deleted while this round runs: the
            // round ends where it stands and the handle goes with it, so the
            // engine stops spending on a reply nobody can come back to.
            let mut deleted = false;
            let mut deletions = state.deletions.watch();

            // Scoped so the stream's borrow of `handle` ends with the loop —
            // `finish_turn` consumes the handle below.
            {
                let mut events_in = std::pin::pin!(handle.stream_async());
                loop {
                    let event = tokio::select! {
                        event = events_in.next() => event,
                        Ok(()) = deletions.changed() => {
                            if state
                                .engine
                                .is_timeline_tombstoned(timeline)
                            {
                                deleted = true;
                                break;
                            }
                            continue;
                        }
                    };
                    let Some(event) = event else {
                        break;
                    };
                    match event {
                        TurnEvent::Token(id) => {
                            if turn_error.is_none() {
                                tokens.push(id);
                                let text = state.decoder.decode(&tokens);
                                if let Some(update) = think_progress.observe(&text, tokens.len()) {
                                    let (count, done) = match update {
                                        ThinkUpdate::Running(n) => (n, false),
                                        ThinkUpdate::Done(n) => (n, true),
                                    };
                                    let progress = StreamItem::Think {
                                        tokens: count,
                                        done,
                                    };
                                    if tx.send(Ok(progress)).await.is_err() {
                                        listener_gone = true;
                                    }
                                }
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
                                // Resolve a leading think block before anything of it
                                // is emitted. Three outcomes, checked in order: the
                                // block closed while empty (skip it entirely), it has
                                // real content (release and never look again), or it
                                // is still opening (hold — at most a few whitespace
                                // tokens).
                                if !think_resolved {
                                    const OPEN: &str = "<think>";
                                    const CLOSE: &str = "</think>";
                                    match text.find(OPEN) {
                                        None => think_resolved = !text.trim().is_empty(),
                                        Some(open) => {
                                            let inner_at = open + OPEN.len();
                                            match text[inner_at..].find(CLOSE) {
                                                Some(rel) => {
                                                    let inner = &text[inner_at..inner_at + rel];
                                                    if inner.trim().is_empty() {
                                                        // Skip the block and the blank
                                                        // run after it, so the answer
                                                        // does not open on a gap.
                                                        let after = inner_at + rel + CLOSE.len();
                                                        let tail = text[after..].trim_start();
                                                        emitted_len = text.len() - tail.len();
                                                    }
                                                    think_resolved = true;
                                                }
                                                // Open but unclosed: real reasoning the
                                                // moment it is not just whitespace.
                                                None => {
                                                    think_resolved =
                                                        !text[inner_at..].trim().is_empty()
                                                }
                                            }
                                        }
                                    }
                                    if !think_resolved {
                                        continue;
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
                                            .send(Ok(StreamItem::Token(new_part.to_string())))
                                            .await
                                            .is_err()
                                        {
                                            // The client closed the connection. The
                                            // decode carries on: breaking here would
                                            // drop the handle on return, stop the
                                            // scheduler at its next token, and seal a
                                            // truncated answer whose length was decided
                                            // by a socket.
                                            listener_gone = true;
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
                            // Report the finished turn — usage + finish_reason —
                            // to the client, async like the rest of this stream.
                            let _ = tx
                                .send(Ok(StreamItem::TurnEnd {
                                    usage: Usage::for_turn(
                                        resp.stats.context_tokens,
                                        resp.stats.tokens_generated,
                                    ),
                                    finish: resp.stats.finish,
                                }))
                                .await;
                            done_resp = Some(resp);
                        }
                        TurnEvent::Error(e) => {
                            let msg = format!("{e}");
                            tracing::error!(conv_id = %conv_id, iteration, "scheduler error: {msg}");
                            // A per-layer KV divergence the engine has already
                            // decided it cannot trust a repair for
                            // (`heal_tail_divergence`/`assert_sealed_layers_aligned`
                            // in candle-nn) is not a transient hiccup — the
                            // sequence's K/V is inconsistent across layers and
                            // every future turn on it fails identically. Rather
                            // than surface the internal diagnostic as if the
                            // assistant said it, tell the user plainly and stop:
                            // no retry here would do anything but repeat it.
                            let shown = if msg.contains("no repair from here can be trusted") {
                                "This conversation's memory has become corrupted and can't \
                                 be recovered. Please start a new conversation."
                                    .to_string()
                            } else {
                                msg.clone()
                            };
                            // Its own item, so the client shows it beside the
                            // reply rather than dropping the connection.
                            let _ = tx.send(Ok(StreamItem::Error(shown.clone()))).await;
                            turn_error = Some(anyhow::anyhow!("{shown}"));
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
                            let _ = tx.send(Ok(StreamItem::Projection(out))).await;
                        }
                        TurnEvent::PrefillProgress {
                            tokens_done,
                            tokens_total,
                        } => {
                            let progress = StreamItem::Prefill {
                                done: tokens_done,
                                total: tokens_total,
                            };
                            if tx.send(Ok(progress)).await.is_err() {
                                listener_gone = true;
                            }
                        }
                        // The prefill is waiting on the GPU: say so, and for how
                        // long, instead of a silent stream.
                        TurnEvent::PrefillQueued {
                            ahead_tokens,
                            ahead_turns,
                            waiting_ms,
                        } => {
                            let status = queued_status(ahead_tokens, ahead_turns, waiting_ms);
                            if last_queued.as_deref() != Some(status.as_str()) {
                                last_queued = Some(status.clone());
                                if tx.send(Ok(StreamItem::Status(status))).await.is_err() {
                                    listener_gone = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            if deleted {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "conversation deleted mid-round — ending the turn now",
                );
                // As a turn that ended without its Done: the scope it was given
                // goes with it.
                state.engine.clear_retrieval_scope(timeline);
                break;
            }

            if listener_gone {
                tracing::info!(
                    conv_id = %conv_id,
                    iteration,
                    "client went away mid-stream — the turn runs to completion",
                );
            }

            let resp = match done_resp {
                Some(r) => r,
                None => {
                    if turn_error.is_none() {
                        // Scheduler closed the channel without Done — the
                        // conversation's turn_in_flight flag is stuck.
                        tracing::error!(
                            conv_id = %conv_id,
                            iteration,
                            "turn ended without Done or Error — evicting conversation",
                        );
                        let _ = tx
                            .send(Ok(StreamItem::Error(
                                "Generation ended unexpectedly. Your next message will start fresh."
                                    .to_string(),
                            )))
                            .await;
                    }
                    // Evict the conversation so the next request forks fresh
                    // rather than hitting the in-flight guard.  Dropping `handle`
                    // here (via return) closes event_rx and stops the scheduler.
                    // The scope it was given goes with it: the next request
                    // sets its own before its first turn.
                    drop(cs);
                    state.conversations.remove(&conv_id);
                    state.journals.closed(&conv_id);
                    state.engine.clear_retrieval_scope(timeline);
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
                        let _ = tx.send(Ok(StreamItem::Token(out))).await;
                    }
                }
            } else if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
                let tail = &resp.text[emitted_len..];
                if !tail.is_empty() {
                    let _ = tx.send(Ok(StreamItem::Token(tail.to_string()))).await;
                }
            }

            // Every call the answer made, in order — including any that could
            // not be read, which are answered with an error rather than dropped
            // (see `tool_round`), so a broken call is a failed round the model
            // and the GUI both see, not a turn that silently reads as final.
            let round = tool_round::plan(&resp.answer);
            // Force-high-resolution is a capture mode: seal the first turn (the
            // tool invocation) into the substrate as the dataset baseline, but
            // do NOT execute the tools — capture-only, so `code_run` / network
            // tools have no real side effects.
            //
            // A turn answering the repeat guard's closing round is final too:
            // its results told the model no further call runs.
            if closing && !round.is_empty() {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    n_calls = round.len(),
                    "the turn after the repeat guard closed the loop made tool calls — not run",
                );
            }
            // A turn that failed ends the reply too: what it wrote is cut off
            // wherever the error landed, and a call in it was never finished.
            let is_final =
                round.is_empty() || force_hires.is_some() || closing || turn_error.is_some();
            // A call the turn stopped writing has no `</tool_call>` in the
            // stream, and the GUI reads everything after an unclosed opener —
            // the rounds that follow included — as that call's JSON. Close it
            // in the stream so its card ends where the call did.
            if !is_final && tool_round::ends_inside_a_call(&resp.answer) {
                let _ = tx
                    .send(Ok(StreamItem::Token("\n</tool_call>".to_string())))
                    .await;
            }

            // If tools will run, tell the GUI *now* — before sealing/persisting
            // this turn — so the in-flight tool cards show their spinner across
            // the whole execution window (seal + persist + dispatch), not just
            // the instant before dispatch (which an instant tool coalesces away,
            // making the card jump straight to a result).
            let tool_names: Vec<String> = if is_final {
                Vec::new()
            } else {
                round.iter().map(|s| s.name().to_string()).collect()
            };
            if !is_final {
                let _ = tx
                    .send(Ok(StreamItem::Tool(ToolStatusOut {
                        phase: "running",
                        tools: tool_names.clone(),
                        results: Vec::new(),
                        tokens: Vec::new(),
                    })))
                    .await;
            }

            if let Err(e) = cs.conv.finish_turn(handle, &resp) {
                tracing::warn!(conv_id = %conv_id, "finish_turn error: {e}");
            }
            // A failed turn is kept — what it wrote is what the user watched
            // arrive — and so is why it stopped, under the turn it belongs to,
            // for the history to show beside it after a reload.
            if let (Some(e), Some(turn)) =
                (&turn_error, resp.seal.as_ref().and_then(|s| s.turn_index))
            {
                if let Err(err) = state.engine.set_conversation_metadata(
                    timeline,
                    &turn_error_key(turn),
                    &e.to_string(),
                ) {
                    tracing::warn!(conv_id = %conv_id, "recording the turn's error failed: {err}");
                }
            }

            // One projection event per decode: the scored reprojection now
            // reflects what provenance materialized for this turn (finish_turn
            // refreshed the BDP scores). Stream it so the GUI drops a timeline
            // dot the moment the turn seals — no separate fetch needed.
            // Which of this conversation's turns it selected is also what tells
            // the journal whether the dialogue has started losing sight of turns
            // no entry covers.
            let mut selected_turns: Option<Vec<u32>> = None;
            if let Some(event) = cs.conv.projection_event(&resp.stats) {
                selected_turns = Some(pressure::selected_turns(&event, timeline));
                let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let out = ProjectionEventOut::answer(seq, event);
                turn_events.push(out.clone());
                let _ = tx.send(Ok(StreamItem::Projection(out))).await;
            }
            // A served read's lock stands while the reply that promised it is
            // read: once this decode's projection left that reply out, the
            // promise is no longer in view and its lock ends — marked on the
            // next turn the conversation submits.
            if let (Some(config), Some(selected), Some(current)) = (
                state.working_set.as_ref(),
                selected_turns.as_deref(),
                resp.seal.as_ref().and_then(|s| s.turn_index),
            ) {
                let engine = &state.engine;
                let gone = promises::departed(engine, timeline, selected, current);
                cs.promise_marks.extend(promises::release(
                    engine,
                    timeline,
                    &gone,
                    config.released_momentum(),
                ));
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

            // This turn asked for tools that have not run yet. Name it before
            // the commit below, so one group-commit makes the call turn and the
            // note of it durable together: an entry still in the writer queue
            // while the tools run would be lost by the very crash it exists to
            // survive. `crate::resume` finds it on the next boot and finishes
            // the round.
            if !is_final {
                if let Some(call_idx) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                    resume::mark_call_turn(&state.engine, timeline, call_idx);
                }
            }

            // Group-commit the substrate redo log: the just-sealed turn is
            // now durable on disk, so a crash or restart resumes it intact.
            // The commit waits for the writes queued before it, and nothing
            // else waits on the commit.
            let conversation = state.engine.conversation();
            if let Err(e) = conversation.commit_persistence() {
                tracing::warn!(conv_id = %conv_id, "persistence commit error: {e}");
            }

            // The sealed turn may make a draft due. The draft runs on its own,
            // on forks of this conversation at `Normal`; this reply does not
            // wait for it.
            if state.journaling {
                if let Some(newest) = resp.seal.as_ref().and_then(|s| s.turn_index) {
                    let cadence = Cadence::Due {
                        newest,
                        selected: selected_turns.as_deref(),
                    };
                    match begin_journal_draft(&state, &conv_id, timeline, cadence) {
                        Ok(span) => tracing::info!(
                            target: "zend::journal",
                            conv_id = %conv_id,
                            from_part = span.from_turn,
                            to_part = span.to_turn,
                            parts = span.turns.len(),
                            "journal draft started",
                        ),
                        // No draft holds the journal: a period that ended
                        // since the last round folds now.
                        Err(_) => {
                            begin_journal_folds(&state, &conv_id, timeline);
                        }
                    }
                }
            }

            if is_final {
                break;
            }

            // Answer what the corpus has already ingested before dispatching the
            // rest. A `file_read` of a file an existing `code_reading`
            // conversation read whole is served by locking that conversation
            // into this one's working set — the K/V exists, so it costs an
            // elevation rather than a prefill and a decode. Every other call,
            // a `file_list` included, runs and is prefilled. Anything
            // unsure (changed file, no complete conversation, the working set
            // full) is left in the round and runs for real, and so is every
            // call from the round's first write on. Each served call marks the
            // turn that carries the results, so a restart restores the lock.
            // On the blocking pool, like the dispatch below: the lookups are
            // git work, and this task's every wait is an await.
            let mut served: Vec<TimelineId> = Vec::new();
            let round = match state.working_set.clone() {
                None => round,
                Some(config) => {
                    let screen_state = Arc::clone(&state);
                    let screen_files = Arc::clone(&cs.files);
                    let unscreened = round.clone();
                    match tokio::task::spawn_blocking(move || {
                        fast_path::screen(
                            &Screen {
                                engine: &screen_state.engine,
                                target: timeline,
                                workspace: &screen_state.workspace,
                                files: &screen_files,
                                config: &config,
                                coverage: &screen_state.read_coverage,
                            },
                            round,
                        )
                    })
                    .await
                    {
                        Ok(screened) => {
                            for hit in &screened.served {
                                tracing::info!(
                                    conv_id = %conv_id,
                                    iteration,
                                    tool = hit.tool,
                                    path = %hit.path,
                                    timeline = hit.timeline.raw(),
                                    "tool call served from the working set — nothing re-read",
                                );
                                served.push(hit.timeline);
                            }
                            screened.steps
                        }
                        Err(e) => {
                            // The fast path is an optimisation; losing it costs
                            // reads, not correctness. Dispatch the round as the
                            // model wrote it.
                            tracing::warn!(conv_id = %conv_id, "fast-path screen panicked: {e}");
                            unscreened
                        }
                    }
                }
            };

            tracing::info!(
                conv_id = %conv_id,
                iteration,
                n_calls = round.len(),
                "dispatching tool calls",
            );
            // The "running" notice was already sent above (before the seal).  Run
            // the tools — file and process work, so on the blocking pool while
            // this task stays parked — then the "done" notice clears the spinner
            // and carries each result so the cards resolve immediately, before
            // the post-stream hydrate.
            let n_calls = round.len();
            let ctx = state.tool_host.dialogue_context(
                tools_mode,
                &cs.files,
                &conv_id,
                journal_shelf(&state, &conv_id, timeline),
            );
            let ran = tokio::task::spawn_blocking(move || tool_round::run(&ctx, round)).await;
            // Whatever the round changed outlives this conversation's state.
            conv_overlay::save(&state.engine, timeline, &cs.files, &cs.mirror);
            let mut results = match ran {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "tool dispatch panicked: {e}");
                    let _ = tx
                        .send(Ok(StreamItem::Token(format!(
                            "\n\n⚠ tool dispatch failed: {e}"
                        ))))
                        .await;
                    break;
                }
            };
            // A round that published to origin or fetched from it moved the
            // tracking refs itself, so no origin probe will see a difference:
            // wake the ingest worker, which reads branches from those refs.
            if results.iter().any(|r| moves_branches(&r.call.name)) {
                if let Some(wake) = state.ingest_wake.get() {
                    crate::ingest_worker::request(wake);
                }
            }
            // The reads the round served, then what its writes ended — so the
            // results turn's marks read in the order the round ran.
            marks.extend(round_marks(&served));
            marks.extend(end_written_promises(&state, timeline, &results).await);
            if repeat_guard.screen(&mut results) == Verdict::Close {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "{} rounds running made only calls already made {} times with the same \
                     result — the next turn answers without tools",
                    repeat_guard::STOP_AFTER,
                    repeat_guard::NOTICE_FROM - 1,
                );
                closing = true;
                let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_REPEATS))).await;
            }
            if !closing && closes_at_cap(iteration) {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    "the next turn is the last the round cap allows — it answers without tools",
                );
                repeat_guard::cap_notice(&mut results);
                closing = true;
                let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_CAP))).await;
            }
            // A worker `/mcp` started reads what Claude sent it — a stop, a
            // nudge — at this boundary, never mid-decode.
            let from_claude = owner_of(&conv_id).map(|_| state.engine.conversation().shared());
            // Claude stopping it closes the loop here, as the cap does. A
            // round the guard or the cap closed too is still Claude's stop,
            // so the worker settles stopped. A round that ran nothing has no
            // results to carry it and ends below; the stop is then reported
            // unreached when the worker settles.
            if let Some(shared) = from_claude.as_ref().filter(|_| !results.is_empty()) {
                if let Some(reason) = stops::take(shared, &conv_id) {
                    tracing::info!(
                        conv_id = %conv_id,
                        iteration,
                        "Claude stopped the task — the next turn answers without tools",
                    );
                    stops::notice(&mut results, &reason);
                    closing = true;
                    let _ = tx.send(Ok(StreamItem::Closing(CLOSED_BY_CLAUDE))).await;
                }
            }
            // Each result measured as its own block of the turn it is about to
            // become — the length its card shows.
            let tokens = results
                .iter()
                .map(|r| {
                    cs.conv
                        .encoded_len(&format_tool_responses(std::slice::from_ref(r)))
                })
                .collect();
            let _ = tx
                .send(Ok(StreamItem::Tool(ToolStatusOut {
                    phase: "done",
                    tools: tool_names,
                    results: results.iter().map(|r| r.response.clone()).collect(),
                    tokens,
                })))
                .await;
            current_message = format_tool_responses(&results);
            // A tool round that produces no response text must NOT spawn a
            // follow-up turn. `format_tool_responses` wraps every real result in
            // `<tool_response>…</tool_response>`, so an empty `current_message`
            // means no tool actually ran (the model emitted a tool call that was
            // filtered/failed). Submitting it would seal a phantom turn whose grid
            // is nothing but boundary glue (no user content), which the model then
            // "answers" with a generic greeting — derailing the conversation. The
            // answer already streamed for this turn stands; end the loop here.
            if current_message.is_blank() {
                tracing::warn!(
                    conv_id = %conv_id,
                    iteration,
                    n_calls,
                    "tool round yielded no response text; ending turn loop instead \
                     of submitting an empty follow-up turn",
                );
                break;
            }
            // Guidance Claude queued for a worker `/mcp` started joins real
            // results only — after the check above, so a nudge never turns an
            // empty round into a turn, and one left queued when the loop ends
            // is reported by the worker as undelivered.
            if let Some(shared) = &from_claude {
                current_message = nudges::follow(current_message, shared, &conv_id);
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

        // The round is over, however it ended — a final answer, a tool round
        // that yielded nothing, a dispatch that panicked. The entry only ever
        // means "tools are in flight", so it goes now. A crash before this
        // reaches disk is covered by the scan's own check: an entry whose turn
        // arrived is tombstoned rather than acted on.
        resume::clear_call_turn(&state.engine, timeline);

        // Title generation has already been fired in parallel from
        // `submit_with_sampling` — it doesn't depend on the main
        // convo's state, so we don't repeat it here.
        let _ = original_user_message;
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

/// End the working-set promises a dispatched round's writes broke, and return
/// the marks that record it: the lock on each file a write named, or every
/// lock after a call whose effects cannot be named (`working_set::promises`).
async fn end_written_promises(
    state: &Arc<InferenceState>,
    timeline: TimelineId,
    results: &[ToolResult],
) -> Vec<String> {
    let Some(config) = state.working_set.clone() else {
        return Vec::new();
    };
    match promises::round_release(&config, results, &state.workspace) {
        Release::Nothing => Vec::new(),
        Release::All => {
            release_working_set(state, timeline).await;
            vec![RELEASE_TAG.to_string()]
        }
        Release::Files(files) => {
            // On the blocking pool, as every release is: it writes through the
            // persistence owner, whose flow control can wait for room.
            let state = Arc::clone(state);
            tokio::task::spawn_blocking(move || {
                let engine = &state.engine;
                let held = promises::holding(engine, timeline, &files);
                promises::release(engine, timeline, &held, config.released_momentum())
            })
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(
                    timeline = timeline.raw(),
                    "ending the written files' locks panicked: {e}"
                );
                Vec::new()
            })
        }
    }
}

/// The journal `timeline`'s metadata records, if it records one.
fn journal_record(state: &InferenceState, timeline: TimelineId) -> Option<Kept> {
    let metadata = state.engine.conversation_metadata(timeline)?;
    kept_in(&metadata)
}

/// Bring the conversation's journal sections in line with its journal, and pin
/// the ones that have sealed in `selection`.
///
/// Nothing waits for a section to seal: a round reads the journal as it last
/// sealed, and a new entry joins the prompt the first round after its section
/// seals (`journal::carry`). Called at each round's head, and by a draft as it
/// ends, so a new entry is already sealing before the next round begins.
async fn carry_journal(
    state: &InferenceState,
    conv_id: &str,
    timeline: TimelineId,
    cs: &mut ConvState,
    selection: &mut SelectionState,
) {
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    let prompt = {
        let held = journal.lock().unwrap();
        held.journal().prompt(held.style())
    };
    // A conversation owned through `/mcp` carries its ancestors' journals
    // ahead of its own, and the reader's line after it (`mcp::lineage`).
    let inherited = inherited_journals(state, conv_id);
    let (before, reader) = match inherited.split_last() {
        Some((reader, before)) => (before, Some(reader)),
        None => (&inherited[..], None),
    };
    let mut wants: Vec<Want<'_>> = before
        .iter()
        .map(|i| Want {
            name: &i.name,
            text: &i.text,
        })
        .collect();
    wants.extend(carry::wants(&prompt));
    wants.extend(reader.map(|i| Want {
        name: &i.name,
        text: &i.text,
    }));
    let lineage_sweep = lineage::sweep_names();
    let mut sweep = prompt.sweep();
    if owner_of(conv_id).is_some() {
        sweep.extend(lineage_sweep.iter().map(String::as_str));
    }
    let ConvState {
        conv,
        journal: carried,
        ..
    } = cs;
    let sealed = carried
        .reconcile(
            conv,
            JOURNAL,
            &wants,
            Linking::Chain,
            &sweep,
            None,
            tokio::time::sleep,
        )
        .await;
    journal.lock().unwrap().journal_mut().swept(&prompt);
    carry::select(selection, &sealed);
}

/// The journals `conv_id` inherits, as the sections that carry them — none for
/// a conversation not owned through `/mcp`: its project's, its thread's, and
/// for a part each worker's above it (`docs/claude_zend_gateway.md` §5.1).
fn inherited_journals(state: &InferenceState, conv_id: &str) -> Vec<lineage::Inherited> {
    let Some((_, thread, name)) = conv_names(conv_id) else {
        return Vec::new();
    };
    let nearest_first: Vec<lineage::Ancestor> = journals_above(state, conv_id)
        .into_iter()
        .map(|a| a.ancestor)
        .collect();
    lineage::sections(&format!("T:{thread}/{name}"), &nearest_first)
}

/// One journal above a conversation: the ancestor it is, the bare name it
/// answers to, and — for a worker's — the conversation that drafts it.
struct JournalAbove {
    ancestor: lineage::Ancestor,
    name: String,
    conv_id: Option<String>,
    /// The entries a citation may name: a worker's journal's ids and those its
    /// folds took, a thread's or project's notes numbered from 1.
    entries: Vec<u64>,
}

/// The journals above `conv_id`, nearest first — none for a conversation not
/// owned through `/mcp`: for a part each worker's above it, its own worker's
/// first, then its thread's notes, then each project's standing rules.
fn journals_above(state: &InferenceState, conv_id: &str) -> Vec<JournalAbove> {
    let Some((owner, thread, name)) = conv_names(conv_id) else {
        return Vec::new();
    };
    let records = state.engine.conversation().shared();
    let mut nearest_first = Vec::new();
    let mut above = names::parent_of(name);
    while !above.is_empty() {
        let id = threads::conv_id(owner, thread, above);
        let (lines, mut entries) = journal_of(state, &id);
        // A fold's entries are still cited by their addresses, which read.
        entries.extend(journal_text(state, conversation_timeline(&id)).folded());
        nearest_first.push(JournalAbove {
            ancestor: lineage::Ancestor {
                path: format!("T:{thread}/{above}"),
                whose: lineage::Whose::Worker,
                lines,
            },
            name: names::leaf(above).to_string(),
            conv_id: Some(id),
            entries,
        });
        above = names::parent_of(above);
    }
    let notes = |journal: &str| threads::note_lines(&threads::notes(&records, owner, journal));
    let numbered = |lines: &[String]| (1..=lines.len() as u64).collect();
    let lines = notes(thread);
    nearest_first.push(JournalAbove {
        entries: numbered(&lines),
        ancestor: lineage::Ancestor {
            path: format!("T:{thread}"),
            whose: lineage::Whose::Thread,
            lines,
        },
        name: thread.to_string(),
        conv_id: None,
    });
    for repo in state
        .workspace
        .repos()
        .iter()
        .filter(|r| r.name != UPLOADS_REPO)
    {
        let project = project_journal(&repo.name);
        let lines = notes(&project);
        nearest_first.push(JournalAbove {
            entries: numbered(&lines),
            ancestor: lineage::Ancestor {
                lines,
                path: project,
                whose: lineage::Whose::Project,
            },
            name: repo.name.clone(),
            conv_id: None,
        });
    }
    nearest_first
}

/// The journals above `conv_id` as its claims may cite them.
fn citable_above(state: &InferenceState, conv_id: &str) -> Vec<Citable> {
    journals_above(state, conv_id)
        .into_iter()
        .map(|a| Citable {
            path: a.ancestor.path,
            entries: a.entries,
        })
        .collect()
}

/// `conv_id`'s journal — one line per entry and one per open item — and its
/// entries' ids.
fn journal_of(state: &InferenceState, conv_id: &str) -> (Vec<String>, Vec<u64>) {
    let journal = state.journals.of(conv_id, || {
        journal_record(state, conversation_timeline(conv_id))
    });
    let held = journal.lock().unwrap();
    let mut lines: Vec<String> = held
        .journal()
        .entries()
        .map(|e| format!("#{} {}", e.id, e.summary))
        .collect();
    let ids = held.journal().entries().map(|e| e.id).collect();
    lines.extend(
        held.journal()
            .open()
            .iter()
            .map(|item| format!("Still open #{}: {}", item.id, item.text)),
    );
    (lines, ids)
}

/// When a journal draft starts.
#[derive(Clone, Copy)]
enum Cadence<'a> {
    /// Turn `newest` has sealed, and its round's projection selected
    /// `selected` of the conversation's turns: start one if the cadence or the
    /// selection pressure asks for it.
    Due {
        newest: u32,
        selected: Option<&'a [u32]>,
    },
    /// Start one now over whatever is not covered (`POST …/journal/draft`).
    Now,
}

/// Why no journal draft started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoDraft {
    /// The conversation is not open on this member, so there is no live slot
    /// to fork the draft's questions from.
    NotOpen,
    /// The cadence does not ask for one yet.
    NotDue,
    /// A draft is running: it may not cover the latest turns, so ask again
    /// once it ends.
    Running,
    /// Nothing has happened since the journal last looked.
    NothingNew,
}

/// Start a journal draft of `conv_id` if `cadence` calls for one, and return
/// the stretch it covers. The draft runs as a task of its own.
fn begin_journal_draft(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
    cadence: Cadence<'_>,
) -> Result<Span, NoDraft> {
    let open = state.journals.open(conv_id).ok_or(NoDraft::NotOpen)?;
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    let covered = {
        let held = journal.lock().unwrap();
        if let Cadence::Due { newest, selected } = cadence {
            let pressed = selected
                .is_some_and(|s| pressure::left_out(s, held.first_uncovered_turn(), newest));
            if !held.due(newest, pressed) {
                return Err(NoDraft::NotDue);
            }
        }
        if held.journal().pending() {
            return Err(NoDraft::Running);
        }
        held.journal().covered_to()
    };
    let history = state.base_conv.recovered_history(timeline, false);
    let newest = match cadence {
        Cadence::Due { newest, .. } => newest,
        Cadence::Now => history
            .iter()
            .map(|m| m.turn)
            .max()
            .ok_or(NoDraft::NothingNew)?,
    };
    let written = state.engine.timeline_turn_wall_ms(timeline);
    let parts = origin::parts_after(&history, &written, covered);
    let span = {
        // Under the same lock as the begin: a draft that started since the
        // check above is running, not nothing new.
        let mut held = journal.lock().unwrap();
        if held.journal().pending() {
            return Err(NoDraft::Running);
        }
        held.begin(parts, newest).ok_or(NoDraft::NothingNew)?
    };
    spawn_journal_draft(
        Arc::clone(state),
        conv_id.to_string(),
        timeline,
        open,
        journal,
        span.clone(),
    );
    Ok(span)
}

/// The desk a draft of `timeline`'s journal puts its questions through: forks
/// of the conversation's slot, each answer held to one call of the offered ones
/// in the dialect's own shape, the reasoning block written closed, decoded
/// greedily.
fn journal_desk(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
    asker: UnsealedAsker,
    max_tokens: usize,
) -> ZendDesk {
    // A conversation with a lineage — the one `style_for` and the journals
    // above it read — is the one whose entries are cited from below.
    let folds_keep_turns = conv_names(conv_id).is_some();
    let dialect = state.refresh_config.dialect.clone();
    let env = ToolCallEnvelope::for_assistant_calls(&dialect);
    let close = ToolCallEnvelope::turn_close(&dialect);
    let prefill = format!(
        "{}{}",
        dialect.no_think_block,
        ToolCallEnvelope::for_dialect(&dialect).marker
    );
    let compiling = Arc::clone(state);
    let compile: Compile = Arc::new(move |specs: &[Value]| {
        let specs = specs
            .iter()
            .map(|spec| serde_json::from_value(spec.clone()))
            .collect::<Result<Vec<ToolSpec>, _>>()?;
        let tree = compile_action_loop(&specs, &env, 1, &close, None)?;
        let compiled = compiling.engine.compile_stencil(&tree)?;
        Ok(Arc::new(compiled))
    });
    let mut sampling = state.refresh_config.sampling.clone();
    sampling.temperature = 0.0;
    let recording = Arc::clone(state);
    let record: Record = Box::new(
        move |prompt: &JournalPrompt, filed: Option<&Filed>, ceased: &[u64]| {
            let conversation = recording.engine.conversation();
            let shared = conversation.shared();
            let text = JournalText::of(shared, timeline);
            if let Some(filed) = filed {
                text.file(filed)?;
            }
            conversation
                .set_conversation_metadata_many(timeline, &metadata_of(prompt))
                .map_err(|e| anyhow::anyhow!("the journal could not be recorded: {e}"))?;
            // Only once the record no longer holds them: a fold whose record
            // fails is abandoned with its entries still in the journal, and
            // their text must still be there to read. A conversation owned
            // through `/mcp` files a fold's entries' turns under the fold, so
            // what its readers cite keeps its evidence
            // (`docs/claude_zend_gateway.md` §5.2).
            match filed {
                Some(fold) if folds_keep_turns && !ceased.is_empty() => {
                    text.refile(ceased, fold.entry)?;
                }
                _ => {
                    for entry in ceased {
                        text.cease_entry(*entry)?;
                    }
                }
            }
            Ok(())
        },
    );
    ZendDesk::new(
        Asking {
            asker,
            compile,
            prefill,
            sampling,
            max_tokens,
        },
        record,
    )
}

/// Cease the text behind `timeline`'s journal entries — the conversation was
/// tombstoned or archived, and nothing reads it again.
fn cease_journal_text(
    engine: &ConversationEngine,
    timeline: TimelineId,
) -> Result<(), ConversationError> {
    JournalText::of(engine.conversation().shared(), timeline)
        .cease_all()
        .map(|_| ())
        .map_err(|e| ConversationError::Other(e.to_string()))
}

/// `timeline`'s journal text: the turns behind its entries.
fn journal_text(state: &InferenceState, timeline: TimelineId) -> JournalText {
    let shared = state.engine.conversation().shared();
    JournalText::of(shared, timeline)
}

/// The journals `conv_id`'s journal tools read: its own, and every journal
/// above it (`docs/claude_zend_gateway.md` §5.3) — read when a tool first
/// opens one, so a round that calls none reads nothing for it.
fn journal_shelf(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
) -> Arc<dyn JournalShelf> {
    let state = Arc::clone(state);
    let conv_id = conv_id.to_string();
    Arc::new(Lazy::new(move || shelf_of(&state, &conv_id, timeline)))
}

/// `conv_id`'s journals as they stand now.
fn shelf_of(state: &InferenceState, conv_id: &str, timeline: TimelineId) -> Shelf {
    let own = Drafted::new(journal_of(state, conv_id).0, journal_text(state, timeline));
    let above = journals_above(state, conv_id)
        .into_iter()
        .map(|a| {
            let book: Arc<dyn JournalBook> = match a.conv_id {
                Some(id) => Arc::new(Drafted::new(
                    a.ancestor.lines,
                    journal_text(state, conversation_timeline(&id)),
                )),
                None => Arc::new(Notes::new(a.ancestor.lines)),
            };
            Above {
                path: a.ancestor.path,
                name: a.name,
                book,
            }
        })
        .collect();
    Shelf::new(Arc::new(own), above)
}

/// Run one journal draft over `span` as a task of its own, and log how it
/// ended. Its record is the journal's too, for the API.
fn spawn_journal_draft(
    state: Arc<InferenceState>,
    conv_id: String,
    timeline: TimelineId,
    open: Open,
    journal: Arc<Mutex<SessionJournal>>,
    span: Span,
) {
    tokio::spawn(async move {
        let mut desk = journal_desk(&state, &conv_id, timeline, open.asker, WRITE_TOKENS);
        let world = JournalsAbove(citable_above(&state, &conv_id));
        let record = draft::draft(&mut desk, &journal, &span, &world).await;
        tracing::info!(
            target: "zend::journal",
            conv_id = %conv_id,
            from_part = record.from_turn,
            to_part = record.to_turn,
            result = ?record.result,
            entry = ?record.entry,
            rounds = record.rounds,
            took_ms = record.took_ms,
            write_ms = record.write_ms,
            detail = ?record.detail,
            checks = ?record.checks,
            "journal draft ended",
        );
        reconcile_journal_now(&state, &conv_id, timeline).await;
        // A draft held the journal; a period that ended meanwhile folds now.
        begin_journal_folds(&state, &conv_id, timeline);
    });
}

/// Hand `conv_id`'s journal sections to its conversation now rather than at
/// the next round's head, so they are sealing — a chain one link after another
/// — while the user is still reading the reply, and the next round reads them.
/// A conversation busy in a round is left to that round's own reconcile, which
/// does the same.
async fn reconcile_journal_now(state: &Arc<InferenceState>, conv_id: &str, timeline: TimelineId) {
    let held = state.conversations.get(conv_id);
    if let Some(conversation) = held {
        if let Ok(mut cs) = conversation.try_lock() {
            let mut unused = SelectionState::default();
            carry_journal(state, conv_id, timeline, &mut cs, &mut unused).await;
        }
    }
}

/// Milliseconds since the Unix epoch, now.
fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Start folding `conv_id`'s journal if a period of it has ended, the
/// conversation is open — the fold asks it, through a fork of its slot — no
/// draft or fold is running, and no fold was given up on lately
/// ([`SessionJournal::may_fold`]).
fn begin_journal_folds(state: &Arc<InferenceState>, conv_id: &str, timeline: TimelineId) {
    let Some(open) = state.journals.open(conv_id) else {
        return;
    };
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    {
        let held = journal.lock().unwrap();
        let entries: Vec<Entry> = held.journal().entries().cloned().collect();
        if held.journal().pending()
            || !held.may_fold(Instant::now())
            || fold::next(&entries, wall_ms(), held.style()).is_none()
        {
            return;
        }
    }
    spawn_journal_folds(
        Arc::clone(state),
        conv_id.to_string(),
        timeline,
        open,
        journal,
    );
}

/// Fold every period of the journal that has ended, one after another, as a
/// task of its own; then hand the journal to the conversation.
fn spawn_journal_folds(
    state: Arc<InferenceState>,
    conv_id: String,
    timeline: TimelineId,
    open: Open,
    journal: Arc<Mutex<SessionJournal>>,
) {
    tokio::spawn(async move {
        let mut desk = journal_desk(&state, &conv_id, timeline, open.asker, FOLD_TOKENS);
        let mut folded = 0usize;
        let style = journal.lock().unwrap().style();
        loop {
            match folding::run(&mut desk, &journal, wall_ms(), style).await {
                Folded::Kept { fold, entry } => {
                    folded += 1;
                    tracing::info!(
                        target: "zend::journal",
                        conv_id = %conv_id,
                        into = fold.into.name(),
                        took = ?fold.ids,
                        entry = entry.id,
                        summary = %entry.summary,
                        "journal fold kept",
                    );
                }
                Folded::Nothing => break,
                Folded::Abandoned(why) => {
                    tracing::warn!(
                        target: "zend::journal",
                        conv_id = %conv_id,
                        why = ?why,
                        "journal fold abandoned",
                    );
                    journal.lock().unwrap().fold_failed(Instant::now());
                    break;
                }
            }
        }
        if folded > 0 {
            reconcile_journal_now(&state, &conv_id, timeline).await;
        }
    });
}

/// Release `timeline`'s working-set locks into provenance
/// (`docs/zend_working_set.md` §4.6). The seeds are not touched: they entered
/// provenance when the conversation opened and fade there like any other file.
/// On the blocking pool: it writes through the persistence owner, whose flow
/// control can wait for room.
async fn release_working_set(state: &Arc<InferenceState>, timeline: TimelineId) {
    let Some(config) = state.working_set.clone() else {
        return;
    };
    let state = Arc::clone(state);
    if let Err(e) = tokio::task::spawn_blocking(move || {
        state
            .engine
            .working_set_release(timeline, config.released_momentum());
    })
    .await
    {
        tracing::warn!(
            timeline = timeline.raw(),
            "releasing the working set panicked: {e}"
        );
    }
}

/// Resolve `config`'s seeds for `timeline` on its own branch and seed its
/// provenance with them. Returns how many were seeded.
fn seed_working_set(
    state: &InferenceState,
    timeline: TimelineId,
    files: &RepoFiles,
    config: &WorkingSetConfig,
) -> usize {
    let folder_of = |repo: &str, inner: &str| {
        state
            .retrieval_scope
            .folder_unit(&state.engine, files, repo, inner)
    };
    Seeding {
        engine: &state.engine,
        files,
        folder_of: &folder_of,
        coverage: &state.read_coverage,
    }
    .apply(timeline, config)
}

/// The tool calls a sealed call turn is still waiting on — the round this
/// daemon did not run, because the process stopped between the call turn's seal
/// and its results.
///
/// Synchronous on purpose, and called as one statement: a `&Sequence` is not
/// `Send`, so reading the turn must not span the await that runs the tools.
/// Empty when there is nothing to finish, which the caller reads as "retract
/// the entry and leave the recorded answer standing".
fn resumed_calls(
    conv: &Sequence,
    conv_id: &str,
    timeline: TimelineId,
    call_turn: u32,
) -> Vec<ToolCall> {
    let Some((_, assistant)) = conv.resolve_turn_text(timeline, call_turn) else {
        tracing::warn!(
            conv_id = %conv_id,
            call_turn,
            "resume: the marked turn is not in the substrate",
        );
        return Vec::new();
    };
    let calls = extract_tool_calls(&assistant);
    if calls.is_empty() {
        tracing::warn!(
            conv_id = %conv_id,
            call_turn,
            "resume: the marked turn holds no tool call — nothing to finish",
        );
    }
    calls
}

/// Rebuild the turn state a conversation was last held at, so a resumed turn
/// runs at the settings its author chose rather than at whatever the defaults
/// happen to be.
///
/// Built through the same [`dial_selection`] / [`apply_tools_dial`] the HTTP
/// path uses, rather than by restating the mapping here.
fn resume_dials(dials: Option<ConversationDials>) -> (SelectionState, ToolMode) {
    let (effort, verbosity, think, tools) = match dials {
        Some(d) => (d.effort, d.verbosity, d.think, d.tools),
        None => resume::DEFAULT_DIALS,
    };
    let mut selection = dial_selection(Some(effort), Some(verbosity), Some(think));
    let tools_mode = ToolMode::from_level(tools);
    apply_tools_dial(&mut selection, tools_mode);
    (selection, tools_mode)
}

/// Finish the tool rounds the previous run stopped in the middle of.
///
/// Spawned once, just after `ready`. Only a conversation carrying a resume
/// entry is touched, so one that ended normally is passed over — and so is one
/// whose turn died before its call turn sealed, which is not resumed at all:
/// nothing ran, and whoever asked can ask again. In practice the set here is
/// whatever the daemon had tools out for when it stopped, usually nothing.
///
/// Resumed one at a time, deliberately: each is a full decode, and starting
/// every one of them at once would hand the scheduler a boot-time burst that
/// competes with the first real request.
fn resume_unfinished_turns(session: Arc<ZendSession>, state: Arc<InferenceState>) {
    tokio::spawn(async move {
        // Passthrough conversations are the client's own turns and are not
        // resumable here; the titler's timeline is not a conversation; an
        // archived one has been distilled and must not take another turn. A
        // deleted one is already absent — `known_conversations` excludes
        // tombstoned timelines. An ingest layer carries no `conv_id` at all and
        // is never listed here to begin with.
        let candidates: Vec<(TimelineId, String)> = {
            let engine = &state.engine;
            let titler = state.titler_timeline;
            engine
                .known_conversations()
                .into_iter()
                .filter(|(tl, conv_id, _, archived, _)| {
                    *tl != titler && !*archived && !conv_id.starts_with(passthrough::CONV_ID_PREFIX)
                })
                .map(|(tl, conv_id, _, _, _)| (tl, conv_id))
                .collect()
        };

        let mut resumed = 0usize;
        for (timeline, conv_id) in candidates {
            let unfinished = {
                let engine = &state.engine;
                match resume::marked_call_turn(engine, timeline) {
                    // The round's response turn never arrived — finish it.
                    Some(call_turn) if !resume::round_answered(engine, timeline, call_turn) => {
                        Some(call_turn)
                    }
                    // The turn after it is in the substrate, so the round
                    // completed and the entry outlived it. Retract it.
                    Some(call_turn) => {
                        tracing::info!(
                            conv_id = %conv_id,
                            call_turn,
                            "resume: the round completed before the daemon stopped — entry retired",
                        );
                        resume::clear_call_turn(engine, timeline);
                        None
                    }
                    // No entry, so nothing was mid-round here. A turn killed
                    // before its call turn sealed is not resumed: nothing ran,
                    // and whoever asked can ask again.
                    None => None,
                }
            };
            let Some(call_turn) = unfinished else {
                continue;
            };

            let (selection, tools_mode) = resume_dials(session.conversation_dials(&conv_id));
            // The turn carries no question of its own: its message is the tool
            // output, built inside the loop from the call turn's own request.
            let user_message = String::new();
            let resume_call_turn = Some(call_turn);

            // Drained to completion with nothing listening: the turn's worth is
            // the seal it writes, and the next client to open the conversation
            // reads it from the substrate like any other.
            let mut turn = run_inference_stream(
                Arc::clone(&state),
                conv_id.clone(),
                user_message,
                None,
                None,
                None,
                None,
                false,
                tools_mode,
                None,
                selection,
                resume_call_turn,
            );
            while turn.next().await.is_some() {}
            resumed += 1;
        }

        if resumed > 0 {
            tracing::info!(resumed, "resume: finished the turns the last run left open");
        }
    });
}

/// One OpenAI-passthrough call, holding its conversation's lock for the whole
/// turn — see [`crate::passthrough`]. Reuses the live conversation when the call
/// extends what it holds; otherwise resumes or opens one. A conversation left
/// in an unknown state by a failed call is dropped, so the next call reopens it
/// from the substrate.
async fn run_passthrough(
    state: Arc<InferenceState>,
    key: String,
    mut conv: OwnedMutexGuard<Option<LiveConv>>,
    transcript: Transcript,
    max_tokens: Option<usize>,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<StreamItem>>,
) {
    let reusable = conv.as_ref().is_some_and(|live| {
        live.system == transcript.system && passthrough::extends(&live.history, &transcript.history)
    });
    if !reusable {
        // Free the stale conversation's slot before opening another.
        *conv = None;
        let opened = {
            let engine = &state.engine;
            passthrough::open(engine, state.refresh_config.clone(), &transcript)
        };
        match opened {
            Ok(live) => *conv = Some(live),
            Err(e) => {
                tracing::error!(key, "passthrough: {e:#}");
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
    let healthy = match conv.as_mut() {
        Some(live) => passthrough_turn(&state, &key, live, transcript, max_tokens, &tx).await,
        None => false,
    };
    if !healthy {
        *conv = None;
    }
    state.passthrough.touch(&key);
}

/// The answer room a passthrough reply gets past its think block — the
/// response-length dial's widest rung. A client's reply can be a whole file
/// written through a tool call, and the client's own `max_tokens` still caps
/// the turn; this only places the failsafe that ends a runaway.
const PASSTHROUGH_RESPONSE_TOKENS: i32 = 7168;

/// Prefill the history `live` does not hold, decode the reply to the new user
/// half, seal it and commit the redo log. Returns whether `live` still matches
/// what the substrate holds.
async fn passthrough_turn(
    state: &InferenceState,
    key: &str,
    live: &mut LiveConv,
    transcript: Transcript,
    max_tokens: Option<usize>,
    tx: &tokio::sync::mpsc::Sender<anyhow::Result<StreamItem>>,
) -> bool {
    let held = live.history.len();
    tracing::info!(
        key,
        timeline = live.seq.timeline_id().raw(),
        held,
        sent = transcript.history.len(),
        "passthrough turn",
    );
    for exchange in &transcript.history[held..] {
        if let Err(e) = live
            .seq
            .insert_turn(exchange.user.clone(), &exchange.assistant)
        {
            tracing::error!(key, "passthrough: prefilling history failed: {e}");
            let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
            return false;
        }
        live.history.push(exchange.clone());
    }

    // Once the model opens a call it is held to the client's tools and their
    // declared arguments — see `LiveConv::tool_stencil`. The `<think>` block is
    // steered atop that, exactly as on a dialogue turn: its tree takes an EOS
    // sampled inside the block as the block's close, so a reply cannot end
    // with its reasoning still open. Measured on a Cline turn without it: the
    // model opened a block, planned in it, and stopped — the client received
    // only reasoning, no answer and no call, and ended the task.
    let triggers = match &state.think_steering {
        Some(ts) => ts.registry_for(&live.tool_stencil, ThinkMode::Balanced),
        None => Arc::clone(&live.tool_stencil),
    };
    // The balanced dial's budget, as a dialogue turn gets it — not the
    // preset's fallback, which cut a Cline plan mid-sentence at 301 thinking
    // tokens and force-ends a reply at 1000 in all.
    let mut sampling = live.seq.default_sampling();
    think_budget::steer(
        &mut sampling,
        ThinkMode::Balanced,
        PASSTHROUGH_RESPONSE_TOKENS,
        &state.think_closer_phrase,
    );
    let options = candle_conversation::TurnOptions {
        max_tokens,
        sampling: Some(sampling),
        triggers,
        ..Default::default()
    };
    let handle = match live
        .seq
        .submit_turn_with_options(transcript.message.clone(), options)
    {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(key, "passthrough: submit failed: {e}");
            let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
            return false;
        }
    };

    // Deltas are cut at byte offsets into the decoded text, and a fragment that
    // still holds U+FFFD waits for the rest of its byte-fallback sequence.
    let mut tokens: Vec<u32> = Vec::new();
    let mut emitted_len = 0usize;
    let mut done = None;
    let mut deletions = state.deletions.watch();
    let timeline = live.seq.timeline_id();
    {
        let mut stream = std::pin::pin!(handle.stream_async());
        loop {
            let event = tokio::select! {
                event = stream.next() => event,
                Ok(()) = deletions.changed() => {
                    if state.engine.is_timeline_tombstoned(timeline) {
                        // Deleted under the reply: dropping the handle withdraws it
                        // from the engine, and the turn never seals.
                        tracing::info!(key, "passthrough: conversation deleted mid-reply");
                        return false;
                    }
                    continue;
                }
                // The client went away before a token was due: a reply still
                // queued for its prefill would otherwise run it, and seal, for
                // nobody — the next send is what used to notice.
                () = tx.closed() => {
                    tracing::info!(key, "passthrough: client disconnected before the reply");
                    return false;
                }
            };
            let Some(event) = event else {
                break;
            };
            match event {
                TurnEvent::Token(id) => {
                    tokens.push(id);
                    let text = state.decoder.decode(&tokens);
                    if text.len() > emitted_len && text.is_char_boundary(emitted_len) {
                        let part = &text[emitted_len..];
                        if !part.contains('\u{FFFD}') {
                            if tx
                                .send(Ok(StreamItem::Token(part.to_string())))
                                .await
                                .is_err()
                            {
                                // The client went away: dropping the handle stops the
                                // decode, and the turn never seals.
                                tracing::info!(key, "passthrough: client disconnected mid-stream");
                                return false;
                            }
                            emitted_len = text.len();
                        }
                    }
                }
                TurnEvent::Done(resp) => done = Some(resp),
                TurnEvent::Error(e) => {
                    tracing::error!(key, "passthrough: scheduler error: {e}");
                    let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
                }
                _ => {}
            }
        }
    }
    // `handle` is free here: the stream that borrowed it was scoped to the block
    // above, so `finish_turn` below can take it.
    let Some(resp) = done else {
        return false;
    };
    if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
        let tail = &resp.text[emitted_len..];
        if !tail.is_empty() {
            let _ = tx.send(Ok(StreamItem::Token(tail.to_string()))).await;
        }
    }
    let _ = tx
        .send(Ok(StreamItem::TurnEnd {
            usage: Usage::for_turn(resp.stats.context_tokens, resp.stats.tokens_generated),
            finish: resp.stats.finish,
        }))
        .await;
    if let Err(e) = live.seq.finish_turn(handle, &resp) {
        tracing::warn!(key, "passthrough: finish_turn: {e}");
    }
    // See the dialogue path's commit.
    let conversation = state.engine.conversation();
    if let Err(e) = conversation.commit_persistence() {
        tracing::warn!(key, "passthrough: persistence commit: {e}");
    }
    live.history.push(Exchange {
        user: transcript.message,
        assistant: resp.text,
    });
    true
}

/// The titler task. Owns the titler [`Sequence`] exclusively and drains title
/// jobs one at a time off `rx`. Each title-gen overlaps the main decode (the
/// scheduler batches both sequences), but the task never blocks the request
/// path and never piles up work per submit. Between jobs it checks the
/// shutdown flag, so once shutdown begins it stops draining queued jobs
/// immediately; on exit it abandons any in-flight turn so the sequence is
/// left clean.
async fn titler_task(
    state: Arc<InferenceState>,
    mut titler: Sequence,
    mut rx: mpsc::Receiver<TitleJob>,
) {
    while let Some(job) = rx.recv().await {
        if state.shutting_down.load(Ordering::Relaxed) {
            break;
        }
        match job {
            TitleJob::Shutdown => break,
            TitleJob::Title { timeline, message } => {
                generate_one_title(&state, &mut titler, timeline, &message).await;
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
async fn generate_one_title(
    state: &Arc<InferenceState>,
    titler: &mut Sequence,
    timeline: TimelineId,
    user_message: &str,
) {
    use candle_conversation::TurnOptions;

    let truncated = head_tail_truncate(
        user_message,
        &state.tokenizer,
        TITLER_HEAD_TOKENS,
        TITLER_TAIL_TOKENS,
    );
    // **The titler must never reason, and the soft switch cannot stop it.**
    //
    // `NO_THINK_SELECTOR` below emits the scheduler's `no_think_current` glue,
    // which renders `Dialect::no_think` — and on the Qwen3.5/3.8 family that is
    // the empty string, because those models suppress by *prefilling* a closed
    // block rather than by a marker (`Dialect::thinking_suppression`). So the
    // glue was a zero-token segment and the `/no_think` in the system prompt was
    // prose: the titler had two suppression mechanisms and, on the model this
    // daemon actually runs, neither did anything.
    //
    // What that cost: the model reasoned into a 24-token budget and the label
    // got whatever survived. A block that never closed stripped to nothing
    // (`strip_think_blocks` drops an unterminated block to end) and the title
    // was skipped; a block that closed just in time left a fragment, which is
    // how "Cas", "The KV", "A" and "P" reached the sidebar — and, since a
    // one-character `String` is not empty, they were written without a warning.
    // Twelve identical prompts produced twelve different lengths, because what
    // varied was the reasoning, not the budget.
    //
    // The trigger below is the layer that cannot be ignored: it binds `<think>`
    // to `ThinkMode::Off`'s tree, whose whole body is the static run
    // `"\n\n</think>"` with no free-decode span, so the block closes on the
    // token after it opens and the budget goes to the title. `apply_think_mode`
    // programs the sampler to match. This is the same three-layer arrangement
    // the ingest summariser settled on after the same defect
    // (`Conversation::submit_ingest_summary`) and the tree compressor's probe
    // after it (`SubmitSummaryProbe`) — the titler was the one left behind.
    let triggers = {
        let en = &state.engine;
        match en.compile_think_steering() {
            // No `<think>`/`</think>` token in this vocabulary, so there is no
            // block to steer and nothing to bind to; `compile_think_steering`
            // has already warned, and the sampler side below still runs.
            Ok(Some(ts)) => ts.registry_for(&TriggerRegistry::new(), ThinkMode::Off),
            Ok(None) => Arc::new(TriggerRegistry::new()),
            Err(e) => {
                tracing::warn!("titler think steering unavailable: {e}");
                Arc::new(TriggerRegistry::new())
            }
        }
    };

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
    // The sampler half of the same suppression: `Off` at this budget collapses
    // the segment-close budget to a forced empty block, so the model is steered
    // out of the block it opened as well as being structurally closed out of it.
    let mut titler_sampling = titler.default_sampling();
    titler_sampling.apply_think_mode(ThinkMode::Off, &state.tokenizer, TITLER_MAX_TOKENS);
    let opts = TurnOptions {
        max_tokens: Some(TITLER_MAX_TOKENS),
        selection: titler_sel,
        sampling: Some(titler_sampling),
        triggers,
        ..Default::default()
    };
    let handle = match titler.submit_turn_with_options(&truncated, opts) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("titler submit failed: {e}");
            return;
        }
    };
    let resp = match handle.wait_async().await {
        Ok(r) => r,
        Err(e) => {
            // No response: scheduler error or shutdown mid-decode. Abandon the
            // turn so it doesn't wedge the sequence (the cause of the shutdown
            // "already has a turn in flight" reset loop).
            tracing::warn!("titler ended without a response: {e}");
            drop(handle);
            titler.abort_turn();
            return;
        }
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
    let result = state.engine.set_conversation_label(timeline, &title);
    match result {
        Ok(()) => tracing::info!("conversation labelled: \"{title}\""),
        Err(e) => tracing::warn!("set_conversation_label failed: {e}"),
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct ZendSession {
    config: DaemonConfig,
    projection_builder: Builder,
    /// Fires `true` once the daemon transitions to fully-ready (every
    /// load step complete) — or once a stop has ended the load, so its
    /// waiters wake and find whether an engine was published. Used by
    /// submit-flow tasks waiting for the engine to come up.
    ready_tx: tokio::sync::watch::Sender<bool>,
    /// Current human-readable detail string for the active load step
    /// (e.g. `"Downloading shard 3/8"`). Surfaced as the loading
    /// section's `detail` field in `GET /v1/status`.
    pub(crate) status_tx: tokio::sync::watch::Sender<String>,
    /// The detail's own reader, held for the session's life. A `watch` channel
    /// stores nothing a `send` hands it while no receiver is open, and the
    /// status snapshot only subscribes for the instant it reads — so without
    /// this every detail sent between two snapshots was dropped, and the
    /// loading screen never named the phase it was in.
    status_rx: tokio::sync::watch::Receiver<String>,
    /// Structured load-state machine — drives the frontend's loading
    /// overlay (current step + progress + completed list).
    load_progress: Arc<LoadProgress>,
    /// Each conversation's reply in flight, recorded so a client that
    /// reconnects mid-turn can follow it (`GET /v1/conversations/{id}/live`).
    live_turns: Arc<LiveTurns>,
    /// Wall-clock time (ms since Unix epoch) when this `ZendSession` was
    /// constructed — i.e. when the daemon process started. Surfaced via
    /// `GET /v1/status` so the frontend can detect daemon restarts and
    /// re-fetch the conversations list.
    started_at_ms: u64,
    /// Populated in the background after construction; None until model loads.
    inference: Arc<ArcSwapOption<InferenceState>>,
    /// The substrate, published by the load once its redo log has replayed —
    /// long before the engine is: what the substrate views read while zend is
    /// still loading (`substrate_for_views`).
    loading_substrate: Arc<OnceLock<Conversation>>,
    /// Handle to the `zend-loader` thread. [`Self::shutdown`] joins it so an
    /// in-progress load has broken its ingest and drained the engine (on the
    /// loader thread, which owns it) before the process exits.
    load_thread: Mutex<Option<JoinHandle<()>>>,
    /// The daemon's single background ingest worker (`crate::ingest_worker`).
    /// Owns every post-startup ingest pass — the startup pass over every
    /// branch, and every pass a fetch or a published branch wakes — plus the
    /// ingest normalization warm-up that follows its first pass. `None` under
    /// `--read-only-substrate` (nothing ingests there) and while shutdown is
    /// tearing it down. [`Self::shutdown`] stops and joins it before draining
    /// the engine, so a stopped session never keeps its model resident — or
    /// its scan on the device — behind the next one.
    ingest_worker: Mutex<Option<crate::ingest_worker::IngestWorker>>,
    /// What `/mcp` holds in memory: its sessions, the tasks this process runs,
    /// and the signal a waiting `repo_task_wait` wakes on (`crate::mcp`).
    mcp: Arc<McpState>,
}

/// Snapshot returned by `GET /v1/status`. `loading` is `None` once the
/// daemon is fully ready (chat unlocked); `Some` while any startup step
/// is still in flight. `started_at_ms` changes only across daemon
/// restarts — the frontend uses it to detect a restart and re-fetch.
pub struct StatusSnapshot {
    pub loading: Option<LoadingSnapshot>,
    pub detail: String,
    pub started_at_ms: u64,
    /// Pending background ingest work, merged across every ingest layer (and
    /// uploads, which share the same per-file pool). `None` when the queue is
    /// empty — the GUI hides its bar on this.
    pub ingest_backlog: Option<crate::ingest_backlog::BacklogSnapshot>,
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

/// Rewrite `timeline`'s upload records with `edit`, as one step against the
/// records it holds — two uploads, or an upload and a delete, each landing
/// whole whatever order they race in. Whether the records were written.
fn edit_upload_records(
    engine: &ConversationEngine,
    timeline: TimelineId,
    mut edit: impl FnMut(&mut Vec<UploadInfo>),
) -> bool {
    engine
        .update_conversation_metadata(timeline, "uploads", |_, current| {
            let mut records: Vec<UploadInfo> = current
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            edit(&mut records);
            serde_json::to_string(&records).ok()
        })
        .is_some()
}

/// One uploaded file recorded against a conversation — the substrate
/// event that survives resume and drives the inline history tile. Stored
/// (JSON-serialized, as an array under the `uploads` custom-metadata key)
/// in the conversation's Label record, so it recovers with the conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UploadInfo {
    /// The handle the GUI opens the file's content by: a hash of its path in
    /// the conversation's uploads store ([`crate::uploads::id_of`]).
    pub id: u64,
    pub name: String,
    /// Workspace-relative path of the file (e.g. `uploads/notes-001.txt`),
    /// with `/` separators: `uploads/` and its path in the uploads store.
    pub path: String,
    pub ext: String,
    pub kind: String,
    /// Display size (e.g. `18.4 KB`).
    pub size: String,
    /// Its length in bytes.
    #[serde(default)]
    pub bytes: u64,
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

/// The record of a file just stored at `stored` in a conversation's uploads
/// store, `size` bytes long: positioned and measured later.
fn upload_record(stored: String, size: u64) -> UploadInfo {
    UploadInfo {
        id: crate::uploads::id_of(&stored),
        path: format!("{UPLOADS_REPO}/{stored}"),
        ext: crate::conv_files::ext_badge(&stored),
        kind: crate::conv_files::kind_for(&stored).as_str().to_string(),
        size: crate::conv_files::fmt_bytes(size),
        bytes: size,
        added: "just now".to_string(),
        name: stored,
        turn_index: 0,
        stats: None,
    }
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
    /// The entry's place in the listing, most recently used first — see
    /// [`crate::conv_order`]. The top entry holds the highest value. A rank,
    /// NOT a millisecond clock: the substrate keeps no time. The wire name is
    /// kept for the frontend's existing sort key.
    pub updated_ms: u64,
}

impl ZendSession {
    /// Who is an admin — see [`crate::access`].
    pub fn roles(&self) -> &Roles {
        &self.config.roles
    }

    /// The projection schema this session runs — its layers, as parsed.
    pub fn projection_builder(&self) -> &Builder {
        &self.projection_builder
    }

    /// A handle that scores probes raw on the scheduler's GPU gallery arena
    /// without holding the engine, or `None` until the model is loaded.
    pub fn group_scanner(&self) -> Option<GroupScanner> {
        let state = self.inference.load_full()?;
        let scanner = state.engine.group_scanner();
        Some(scanner)
    }

    /// The workspace folder — where `substrate/` lives.
    pub fn workspace_root(&self) -> &Path {
        self.config.workspace.root()
    }

    /// This process's membership of zend's cluster, when it was given one and
    /// joined: what the operations tools send their requests up
    /// (`docs/claude_zend_gateway.md` §8).
    pub fn cluster_member(&self) -> Option<Arc<AppMember>> {
        self.config.cluster.as_ref()?.link().member().cloned()
    }

    /// The workspace's repositories by name, the uploads store left out: the
    /// projects a `P:<repo>` journal can be kept for.
    pub fn repository_names(&self) -> Vec<String> {
        self.config
            .workspace
            .repos()
            .iter()
            .map(|r| r.name.clone())
            .filter(|name| name != UPLOADS_REPO)
            .collect()
    }

    pub fn new(config: DaemonConfig) -> Self {
        let projection_builder = build_projection_builder(config.workspace.root());
        tracing::info!(
            workspace = %config.workspace.root().display(),
            repos = %config.workspace.names().join(", "),
            "session initialised",
        );
        let (ready_tx, _) = tokio::sync::watch::channel(false);
        let (status_tx, status_rx) = tokio::sync::watch::channel(String::new());
        let started_at_ms = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mcp = Arc::new(McpState::default());
        // The memory backbone, when the agent holds the journal repository.
        if let Some(journal) = config
            .agent
            .as_ref()
            .and_then(|jobs| jobs.checkouts().journal())
        {
            mcp.memory().start(journal.dir.clone());
        }
        Self {
            inference: Arc::new(ArcSwapOption::empty()),
            loading_substrate: Arc::new(OnceLock::new()),
            config,
            projection_builder,
            ready_tx,
            status_tx,
            status_rx,
            load_progress: Arc::new(LoadProgress::new()),
            live_turns: Arc::new(LiveTurns::default()),
            started_at_ms,
            load_thread: Mutex::new(None),
            ingest_worker: Mutex::new(None),
            mcp,
        }
    }

    fn loaded(&self) -> Option<Arc<InferenceState>> {
        self.inference.load_full()
    }

    /// What `/mcp` holds in memory.
    pub fn mcp(&self) -> &Arc<McpState> {
        &self.mcp
    }

    /// The substrate the model's conversations live in, for records kept
    /// beside them (`crate::mcp::records`). The load publishes the engine's
    /// conversation handle as soon as the redo log has replayed and keeps it
    /// for the process's life, so a caller can read while zend is still
    /// loading. `None` before that.
    pub fn shared_substrate(&self) -> Option<SharedSubstrate> {
        let conv = self.loading_substrate.get().cloned()?;
        Some(conv.shared())
    }

    /// Whether the model and its substrate are loaded: what a conversation's
    /// files need.
    pub fn is_loaded(&self) -> bool {
        self.loaded().is_some()
    }

    /// **Phase 1 of the upload pipeline** — each of `files` into `conv_id`'s
    /// uploads store under a safe, unused name, saved as the conversation's
    /// file events (§10.2): what was stored, as its upload record, or the file
    /// and why not. `None` while the model is not loaded: the uploads are the
    /// substrate's.
    pub async fn store_uploads(
        &self,
        conv_id: &str,
        files: Vec<(String, Vec<u8>)>,
    ) -> Option<Vec<Result<UploadInfo, (String, String)>>> {
        let state = self.loaded()?;
        Some(
            state
                .change_files(conv_id, |repos| {
                    let store = match repos.repo(UPLOADS_REPO) {
                        Ok(store) => store,
                        Err(e) => {
                            return files
                                .into_iter()
                                .map(|(name, _)| Err((name, e.to_string())))
                                .collect();
                        }
                    };
                    files
                        .into_iter()
                        .map(|(name, bytes)| {
                            let size = bytes.len() as u64;
                            let stored = crate::uploads::store(&store, &name, bytes)
                                .map_err(|e| (name.clone(), e.to_string()))?;
                            Ok(upload_record(stored, size))
                        })
                        .collect()
                })
                .await,
        )
    }

    /// The upload record `id` of `conv_id`.
    fn upload_of(&self, conv_id: &str, id: u64) -> Option<UploadInfo> {
        self.conversation_uploads(conv_id)
            .into_iter()
            .find(|u| u.id == id)
    }

    /// Upload `id`'s bytes, from `conv_id`'s uploads store. The outer `None`
    /// while the model is not loaded; the inner when there is no such upload.
    pub fn upload_content(&self, conv_id: &str, id: u64) -> Option<Option<Vec<u8>>> {
        let state = self.loaded()?;
        let Some(upload) = self.upload_of(conv_id, id) else {
            return Some(None);
        };
        let files = state.saved_files(conv_id);
        let inner = upload.path.split_once('/').map_or("", |(_, p)| p);
        Some(
            files
                .repo(UPLOADS_REPO)
                .ok()
                .and_then(|s| s.read_bytes(inner).ok().flatten()),
        )
    }

    /// Delete upload `id` from `conv_id`: from its uploads store, saved, and
    /// from its upload records. Whether it was there; `None` while the model
    /// is not loaded.
    pub async fn delete_upload(&self, conv_id: &str, id: u64) -> Option<bool> {
        let state = self.loaded()?;
        let Some(upload) = self.upload_of(conv_id, id) else {
            return Some(false);
        };
        let inner = upload
            .path
            .split_once('/')
            .map_or("", |(_, p)| p)
            .to_string();
        state
            .change_files(conv_id, |repos| {
                if let Ok(store) = repos.repo(UPLOADS_REPO) {
                    store.delete(&inner);
                }
            })
            .await;
        if !edit_upload_records(&state.engine, timeline_for(conv_id), |records| {
            records.retain(|u| u.id != id)
        }) {
            tracing::warn!(conv_id = %conv_id, "forgetting an upload failed");
        }
        Some(true)
    }

    /// **Phase 2 of the upload pipeline** — read the just-uploaded files into
    /// the substrate via the normal `code_read` ingest, scoped to **only those
    /// files** (`rel_paths`, workspace-relative, in `conv_id`'s uploads).
    /// Bounded to the uploaded files' work — never a whole-workspace re-ingest
    /// — so it can't overload the model and is safe under `--skip-code-read`.
    /// Returns `Ok(true)` when something was ingested, `Ok(false)` for a no-op
    /// (non-code files, or the model not loaded). Per-scope progress streams
    /// into `progress`. Returns `(ingested, failed)` — see
    /// `ingest_uploaded_files`.
    pub fn read_file_phase(
        &self,
        conv_id: &str,
        rel_paths: &[String],
        progress: &Arc<crate::loading::LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let Some(state) = self.loaded() else {
            return Ok((false, false));
        };
        state.ingest_uploaded_files(conv_id, rel_paths, progress)
    }

    /// Kick the async summariser so the just-ingested turns start folding into
    /// their summary trees promptly instead of waiting for the next periodic
    /// tick. **Fire-and-forget**: summarisation is a fully background task,
    /// invisible to the upload — the pipeline never waits on it, shows no
    /// progress for it, and records nothing about it. A no-op when the model
    /// isn't loaded (the periodic tick picks the work up once it is).
    pub fn kick_summariser(&self) {
        if let Some(state) = self.inference.load_full() {
            state.engine.trigger_summariser();
        }
    }

    /// The agents this daemon's commands, pushes and fetches run on
    /// (`docs/replicated_substrate.md` §11.5, `docs/substrate_git.md` §9): the
    /// cluster's, or with no cluster the one in this process. `None` with
    /// neither.
    pub fn agents(&self) -> Option<Agents> {
        substrate_git::agents(self.config.cluster.as_ref(), self.config.agent.as_ref())
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
            detail: self.status_rx.borrow().clone(),
            started_at_ms: self.started_at_ms,
            ingest_backlog: crate::ingest_backlog::snapshot(),
        }
    }

    /// Sidebar entries for `GET /v1/conversations`. Built from the
    /// in-RAM substrate (the `RecordType::Label` records carry every
    /// conversation's `(conv_id, label)` pair), but **gated on the
    /// on-disk redo log still existing**.
    ///
    /// The gate exists because the dev workflow of wiping the
    /// `substrate/` dir behind a running daemon used to leave ghost
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
    pub fn substrate_maintenance(&self) -> Option<MaintenanceStatus> {
        let state = self.inference.load_full()?;
        Some(state.substrate.maintenance_status())
    }

    /// How this boot loaded its prompt sections — restored from the redo log or
    /// prefilled — counted since the engine was built. `None` until the model is
    /// loaded.
    pub fn section_loads(&self) -> Option<SectionLoads> {
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        Some(conv.section_loads())
    }

    /// The substrate, once the model is loaded: what the host's maintenance
    /// commands run on. A cloned handle, taken without holding the engine
    /// lock past the clone, so a command never blocks a turn's commits.
    pub fn conversation(&self) -> Option<Conversation> {
        let state = self.inference.load_full()?;
        let conv = state.engine.conversation();
        Some(conv)
    }

    /// `self-check`, run now (`docs/replicated_substrate.md` §12.4): every
    /// stored conversation asked whether it is intact, as `--self-check
    /// --dry-run` asks at load — reporting, tombstoning nothing. Blocks until
    /// done; `None` until the model is loaded.
    pub fn self_check_now(
        &self,
        progress: &dyn Fn(u64, u64),
    ) -> Option<anyhow::Result<SelfCheckReport>> {
        let state = self.inference.load_full()?;
        let mut layers: HashSet<LayerId> = state
            .ingest_layers
            .iter()
            .filter(|il| state.ingest_bases.contains_key(&il.name))
            .filter_map(|il| state.refresh_builder.id_for_layer(&il.name))
            .collect();
        if let Some(dialogue) = state.refresh_builder.id_for_layer("dialogue") {
            layers.insert(dialogue);
        }
        let base = state.base_conv.unsealed_asker();
        Some(self_check::run(
            &state.engine,
            &state.refresh_config,
            &layers,
            base,
            SelfCheck::DryRun,
            progress,
        ))
    }

    /// What the substrate views read: the substrate, the titler's timeline they
    /// leave out, and the target layer — the loaded engine's, or while zend is
    /// still loading the substrate the load published once its redo log
    /// replayed, with no titler or target layer built yet. `None` before the
    /// replay has finished.
    fn substrate_for_views(&self) -> Option<(Conversation, Option<TimelineId>, String)> {
        if let Some(state) = self.inference.load_full() {
            return Some((
                state.substrate.clone(),
                Some(state.titler_timeline),
                state.target_layer.clone(),
            ));
        }
        let conv = self.loading_substrate.get().cloned()?;
        Some((conv, None, String::new()))
    }

    /// `GET /v1/substrate` — a structural snapshot of the live substrate for the
    /// viewer: on-disk size, live counts, and the projection hierarchy (layers →
    /// conversations). Reads the shared `Substrate` through a cloned
    /// `Conversation` handle (one version of it, held for the walk), never a
    /// rebuild from the log.
    /// Answers while zend is still loading, once its substrate has replayed.
    pub fn substrate_overview(&self) -> Option<SubstrateOverview> {
        let (conv, titler, target_layer) = self.substrate_for_views()?;
        // The builder parsed once at session construction: no lock held, and no
        // per-request re-parse.
        Some(overview(
            &conv,
            self.projection_builder.schema(),
            titler,
            target_layer,
            &self.config.workspace.root().join(SUBSTRATE_DIR),
        ))
    }

    /// Flip a projection layer's runtime kill switch (in-memory, non-persistent).
    /// Returns the layer's NEW disabled state, or `None` if `name` isn't a
    /// declared layer. The exclusion takes effect on the next (re)projection; a
    /// restart clears every toggle. Diagnostic only — for isolating which layer's
    /// projected content is breaking coherence.
    pub fn toggle_projection_layer(&self, name: &str) -> Option<bool> {
        let known = self
            .projection_builder
            .schema()
            .layers
            .iter()
            .any(|l| l.name == name);
        if !known {
            return None;
        }
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        Some(conv.toggle_layer(name))
    }

    /// `GET /v1/substrate/zend/layer/{name}` — the conversations (timelines) that
    /// target one layer. Fetched when the layer card expands. `None` before the
    /// substrate has replayed or when the layer name is unknown.
    pub fn substrate_layer(&self, name: &str) -> Option<LayerConversations> {
        let schema = self.projection_builder.schema();
        // Resolve the layer's groups before touching the substrate.
        let groups = schema
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.groups.clone())?;
        let (conv, titler, _) = self.substrate_for_views()?;
        let read = conv.read();
        Some(layer_conversations(&read, &groups, titler, unit_of))
    }

    /// `GET /v1/substrate/zend/system-prompt` — the target (dialogue) layer's framing:
    /// its ordered item labels, its always-emit sections with authored text +
    /// materialized token cost, and the live tool-catalog size (the catalog
    /// itself loads from `substrate_tools`). Fetched when the system-prompt card
    /// expands.
    pub fn substrate_system_prompt(&self) -> Option<SystemPromptView> {
        let (conv, _, target_layer) = self.substrate_for_views()?;
        let read = conv.read();
        Some(system_prompt(
            &read,
            self.projection_builder.schema(),
            target_layer,
            crate::tool_def::all().len(),
        ))
    }

    /// zend's entries in `GET /v1/tools` — the live tool catalog projected into the
    /// dialogue layer's `tools` collection. Sourced from the bundled tool
    /// definitions (`src/prompts/tools/`, embedded); the schema YAML declares the
    /// collection empty and tools install at daemon start. Fetched when the tools
    /// collection expands. The bundled definitions alone: it answers while zend
    /// is still loading.
    pub fn substrate_tools(&self) -> Option<ToolsView> {
        Some(ToolsView {
            tools: crate::tool_def::all()
                .iter()
                .map(|d| ToolView {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    high_risk: d.high_risk,
                    parameters: d.parameters.clone(),
                })
                .collect(),
        })
    }

    /// The conversation `conv_id` as [`Self::substrate_timeline`] reads it:
    /// every turn decoded. `None` before the substrate has replayed, or when it
    /// has no turns.
    pub fn conversation_detail(&self, conv_id: &str) -> Option<TimelineDetail> {
        self.substrate_timeline(conversation_timeline(conv_id).raw())
    }

    /// `GET /v1/substrate/zend/timeline/{tl}` — one conversation's full summary forest,
    /// every turn decoded. `tl` is the raw timeline id. `None` before the
    /// substrate has replayed or when no such timeline exists.
    pub fn substrate_timeline(&self, tl_raw: u64) -> Option<TimelineDetail> {
        let (conv, _, _) = self.substrate_for_views()?;
        let read = conv.read();
        timeline(&read, self.projection_builder.schema(), tl_raw)
    }

    /// `GET /v1/substrate/zend/timeline/{tl}/selection` — the most recent
    /// score-density selection diagnostic recorded for this timeline. `None`
    /// when the model isn't loaded, `tl` isn't a real timeline, or no
    /// projection has run for it yet (or it used the rule-based path, which
    /// records no diagnostic). In-memory only — last-write-wins per timeline,
    /// nothing here survives a restart. See
    /// `ConversationEngine::last_selection_diagnostics`.
    pub fn substrate_selection(&self, tl_raw: u64) -> Option<SelectionView> {
        let state = self.inference.load_full()?;
        let tl = projection::TimelineId::from_raw(tl_raw)?;
        let diag = state.engine.last_selection_diagnostics(tl)?;
        Some(SelectionView {
            selected: diag
                .selected_nodes
                .iter()
                .zip(diag.origins.iter())
                .map(|(n, o)| SelectedNodeView {
                    node_id: n.0,
                    origin: *o,
                    effective_score: diag.effective_scores.get(n).copied(),
                })
                .collect(),
            pending_count: diag.pending_count,
            selected_tokens: diag.selected_tokens,
            budget: diag.budget,
        })
    }

    /// `POST /v1/substrate/zend/project` — project a typed query against the substrate.
    /// Prefills `text` under the base conversation's system prompt (no seal, no
    /// belief mutation), then resolves the selected turns + scored section members
    /// into score-sorted tiles. `None` until the model is loaded.
    pub fn substrate_project(&self, text: &str) -> Option<ProjectView> {
        let state = self.inference.load_full()?;
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
        let ctx = { state.base_conv.probe_ctx() };
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
        let conv = { state.engine.conversation() };
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

        // Scored section members (the collection tools). Text comes from the
        // bundled tool definitions (the schema declares the collection empty). Cut
        // the noise floor: keep only members the query selected or gave a positive
        // belief score.
        let tool_desc: HashMap<&str, &str> = crate::tool_def::all()
            .iter()
            .map(|d| (d.name.as_str(), d.description.as_str()))
            .collect();
        for item in &event.selection.system {
            if let SystemItem::Collection { name, sections, .. } = item {
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

        // Rank on the RAW belief, then bound what is rendered. Order first is
        // load-bearing: a lock-on rides far above the 0-1000 normalized band, so
        // clamping before the sort ties the leaders and the readout's ordering —
        // the thing the margin is actually read for — becomes arbitrary.
        tiles.sort_by(|a, b| b.score.total_cmp(&a.score));
        for tile in &mut tiles {
            tile.cap_score();
        }

        // Diagnostic: if `scored` is 0 while `query_tokens` > 0, the probe Q was
        // captured but didn't discriminate against the gallery (cold/partial-warm
        // context) — selection fell back to the default fill.
        let max_score = tiles.iter().map(|t| t.score).fold(0.0f32, f32::max);
        tracing::info!(
            query_tokens,
            tiles = tiles.len(),
            scored = tiles
                .iter()
                .filter(|t| t.score.abs() > f32::EPSILON)
                .count(),
            max_score,
            "substrate_project"
        );

        Some(ProjectView {
            query_tokens,
            tiles,
        })
    }

    pub fn list_conversations(&self, include_archived: bool) -> Vec<ConvEntry> {
        // On-disk gate: if the workspace's redo log is gone (no segment files
        // in `substrate/`), return empty regardless of the in-RAM cache. The
        // daemon keeps running and any new turn re-mints the segment set.
        let sub_dir = self.config.workspace.root().join(SUBSTRATE_DIR);
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

        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        // Polled every few seconds by every open tab, so it reads what each
        // listed conversation holds and nothing more: a turn
        // count is its own timeline's, never a walk of every turn the substrate
        // holds (the ingest layers are most of them).
        let engine = &state.engine;
        let titler_timeline = state.titler_timeline;
        let known = engine.known_conversations();
        let passthrough_tagged = engine.conversations_with_metadata_key(passthrough::METADATA_KEY);
        let conv = engine.conversation();
        let view = conv.read();
        let active_of = |tl| view.timeline_entry(tl).map_or(0, |e| e.active);
        let mut entries: Vec<(ConvEntry, conv_order::Ranks)> = known
            .into_iter()
            .filter(|(tl, _, _, _, _)| *tl != titler_timeline)
            .filter(|(_, _, _, archived, _)| include_archived || !*archived)
            .map(|(tl, conv_id, label, archived, order)| {
                let entry = ConvEntry {
                    id: conv_id,
                    label,
                    turn_count: view.turn_count_of(tl),
                    archived,
                    updated_ms: 0,
                };
                (entry, (active_of(tl), order))
            })
            .collect();
        // Passthrough conversations stored before they carried a `conv_id` are
        // found by their tag; the next call to one names it for good.
        entries.extend(passthrough_tagged.into_iter().filter_map(|(tl, _)| {
            let entry = view.timeline_entry(tl)?;
            if entry.conv_id.is_some() || (entry.archived && !include_archived) {
                return None;
            }
            let first_user = view
                .turn_indices(tl)
                .min()
                .map(|i| view.user_text_of(tl, i))
                .unwrap_or_default();
            Some((
                ConvEntry {
                    id: passthrough::conv_id_of(tl),
                    label: passthrough::label_for(&first_user),
                    turn_count: view.turn_count_of(tl),
                    archived: entry.archived,
                    updated_ms: 0,
                },
                (entry.active, entry.order),
            ))
        }));
        conv_order::by_last_use(entries)
    }

    /// Archive a conversation — one-way. Sets the archived lifecycle flag and
    /// marks the timeline for `TextOnly` distillation, so the next compaction
    /// sheds its KV chunks + signatures + projections and leaves a read-only
    /// text record. There is no unarchive: distillation is lossy (the KV is
    /// gone), so an archived conversation can't be resumed. Its journal keeps
    /// its entries, and the text behind them is ceased: no tool runs on an
    /// archived conversation again. Returns `None` when the model isn't loaded
    /// yet.
    pub fn archive_conversation(&self, conv_id: &str) -> Option<EngineResult<()>> {
        let (state, timeline, distilled) = self.distill_archived(conv_id)?;
        let engine = &state.engine;
        Some(distilled.and_then(|()| cease_journal_text(engine, timeline)))
    }

    /// Archive a worker `/mcp` started (`crate::mcp::workers`): distilled like
    /// any archived conversation, but its journal's text is **kept**. The
    /// turns behind a worker's entries are the evidence for its task's outcome
    /// — exactly what Claude reads later, through `journal_read` with an entry
    /// — so they stay until the worker's thread is done with
    /// (`docs/claude_zend_gateway.md` §4). `None` when the model isn't loaded.
    pub fn archive_worker(&self, conv_id: &str) -> Option<EngineResult<()>> {
        self.distill_archived(conv_id)
            .map(|(_, _, distilled)| distilled)
    }

    /// Record that `conv_id` decodes at `band` in place of its layer's
    /// priority: read when its slot is made, so it is set before the turn
    /// that makes it (`docs/claude_zend_gateway.md` §3.4). `None` when the
    /// model isn't loaded.
    pub fn set_decode_band(&self, conv_id: &str, band: DecodePriority) -> Option<EngineResult<()>> {
        let state = self.loaded()?;
        let engine = &state.engine;
        Some(engine.set_conversation_metadata(
            conversation_timeline(conv_id),
            DecodePriority::METADATA_KEY,
            band.name(),
        ))
    }

    /// What every archive does: set the archived flag and mark the timeline
    /// for `TextOnly` distillation. `None` when the model isn't loaded.
    fn distill_archived(
        &self,
        conv_id: &str,
    ) -> Option<(Arc<InferenceState>, TimelineId, EngineResult<()>)> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let distilled = {
            let engine = &state.engine;
            engine
                .set_conversation_archived(timeline, true)
                .and_then(|()| engine.distill_timeline(timeline, DistillMode::TextOnly))
        };
        Some((state, timeline, distilled))
    }

    /// Permanently tombstone a conversation — the delete path behind
    /// `DELETE /v1/conversations/{id}`. Writes a durable `Tombstone` record so the
    /// timeline is hidden from listings/resume immediately, and its records
    /// (turns, KV, provenance) are physically reclaimed at the next compaction
    /// (lazy — the tombstone is the commit, compaction does the delete). Also
    /// drops the conversation from the in-RAM map so it vanishes from the sidebar
    /// at once. Returns `None` when the model isn't loaded yet.
    pub fn tombstone_conversation(&self, conv_id: &str) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let result = {
            let engine = &state.engine;
            engine
                .tombstone_timeline(timeline)
                .and_then(|()| cease_journal_text(engine, timeline))
        };
        if result.is_ok() {
            state.conversations.remove(conv_id);
            state.journals.forget(conv_id);
            // After the tombstone, so a turn woken by it finds its timeline
            // tombstoned.
            state.deletions.deleted();
        }
        Some(result)
    }

    /// `GET /v1/conversations/{id}/journal`: the journal's entries as the
    /// prompt reads them, its open items, what it covers, and its recent drafts.
    /// A conversation that has written nothing has an empty one. `None` while
    /// the model is not loaded.
    pub fn journal(&self, conv_id: &str) -> Option<Value> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let view = match state.journals.get(conv_id) {
            Some(journal) => {
                let held = journal.lock().unwrap();
                journal_view(held.journal(), held.style())
            }
            None => {
                let held = journal_record(&state, timeline)
                    .map_or_else(SessionJournal::new, SessionJournal::restore);
                journal_view(held.journal(), style_for(conv_id))
            }
        };
        Some(view)
    }

    /// `POST /v1/conversations/{id}/journal/draft`: draft an entry now, over
    /// whatever the journal does not cover, without waiting for the cadence.
    /// Answers the stretch it covers; how it ends shows in the journal's
    /// drafts. `None` while the model is not loaded.
    pub fn journal_draft(&self, conv_id: &str) -> Option<Result<Value, NoDraft>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(
            begin_journal_draft(&state, conv_id, timeline, Cadence::Now).map(|span| {
                json!({
                    "from_turn": origin::turn_of(span.from_turn),
                    "to_turn": origin::turn_of(span.to_turn),
                    "parts": span.turns.len(),
                })
            }),
        )
    }

    /// `GET /v1/conversations/{id}/journal/{eid}/turns`: the turns behind entry
    /// `entry` — a step's stretch, a folded entry's digest — exactly as
    /// `journal_read` pages them, `lines` lines from line `from`. `None` while
    /// the model is not loaded.
    pub fn journal_turns(
        &self,
        conv_id: &str,
        entry: u64,
        from: usize,
        lines: usize,
    ) -> Option<Result<Page, Miss>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(journal_text(&state, timeline).read(entry, from, lines))
    }

    /// The lines of `conv_id`'s journal text that hold `pattern`, ignoring
    /// case — in `entry`'s turns, or every entry's — the first `max`, exactly
    /// as `journal_grep` finds them. `None` while the model is not loaded.
    pub fn journal_grep(
        &self,
        conv_id: &str,
        pattern: &str,
        entry: Option<u64>,
        max: usize,
    ) -> Option<Result<Hits, Miss>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(journal_text(&state, timeline).grep(pattern, entry, max))
    }

    /// `GET /v1/conversations/{id}/prompt-tokens`: the tokens the conversation
    /// would be given if it decoded now, piece by piece. `Some(None)` when it
    /// is not open on this member, or does not reproject; `None` while the model
    /// is not loaded. Waits for a round in flight to end.
    pub async fn prompt_dump(&self, conv_id: &str) -> Option<Option<PromptDump>> {
        let state = self.inference.load_full()?;
        let held = state.conversations.get(conv_id);
        let Some(conversation) = held else {
            return Some(None);
        };
        let cs = conversation.lock().await;
        Some(cs.conv.prompt_dump())
    }

    /// `DELETE /v1/conversations/{id}/journal/{eid}`: forget one entry. It
    /// leaves the journal and its record at once, and its section leaves the
    /// conversation's prompt at the next round. Answers the journal as it now
    /// reads; `Some(Ok(None))` when no entry by that number is held (only the
    /// newest few are). `None` while the model is not loaded.
    pub fn journal_forget(
        &self,
        conv_id: &str,
        entry: u64,
    ) -> Option<anyhow::Result<Option<Value>>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let journal = state
            .journals
            .of(conv_id, || journal_record(&state, timeline));
        let prompt = {
            let mut held = journal.lock().unwrap();
            if held.journal_mut().forget(&[entry]).is_empty() {
                return Some(Ok(None));
            }
            held.journal().prompt(held.style())
        };
        let recorded = {
            let engine = &state.engine;
            engine
                .set_conversation_metadata_many(timeline, &metadata_of(&prompt))
                .map_err(|e| anyhow::anyhow!("the journal could not be recorded: {e}"))
                .and_then(|()| {
                    JournalText::of(engine.conversation().shared(), timeline).forget(entry)
                })
        };
        Some(recorded.map(|_| {
            let held = journal.lock().unwrap();
            Some(journal_view(held.journal(), held.style()))
        }))
    }

    /// Tombstone a timeline by its RAW id — the derived-layer counterpart to
    /// [`Self::tombstone_conversation`], which can only address timelines whose
    /// id is `hash(conv_id)`.
    ///
    /// Backs `DELETE /v1/substrate/zend/timeline/{tl}`. Nothing is removed from the
    /// dialogue conversation map, because a derived-layer timeline was never in
    /// it: these are `repo_map` / `code_reading` ingests, addressed only by the
    /// id the substrate views print.
    ///
    /// `None` when the model is not loaded or `raw` is not a valid timeline id
    /// (zero) — the same shape the sibling routes return.
    pub fn tombstone_timeline_raw(&self, raw: u64) -> Option<candle_conversation::Result<()>> {
        let state = self.inference.load_full()?;
        let timeline = projection::TimelineId::from_raw(raw)?;
        let result = state.engine.tombstone_timeline(timeline);
        Some(result)
    }

    /// Retire the persisted copy of every section of the dialogue system
    /// prompt — the schema's sections (prelude, tool catalog, outro, every
    /// section-tree variant) and each tools mode's catalog summary — so the next
    /// start prefills them fresh under the current compression thresholds.
    ///
    /// Backs `DELETE /v1/substrate/zend/system-prompt`. The running daemon keeps
    /// serving from its resident copies
    /// ([`candle_conversation::Conversation::retire_section_generation`]); the
    /// rebuild happens on restart. Ingest frames are not touched: a sealed
    /// ingest turn was computed against its frame as stored.
    ///
    /// `None` when the model is not loaded.
    pub fn retire_system_prompt(&self) -> Option<candle::Result<SystemPromptRetired>> {
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        let summaries = ToolMode::ALL
            .into_iter()
            .filter_map(tool_summary_section)
            .map(SectionId::reserved);
        let mut streams = HashSet::new();
        let mut unregistered = 0usize;
        for id in state
            .refresh_builder
            .schema()
            .system_prompt
            .all_section_ids()
            .chain(summaries)
        {
            match conv.section_stream(id) {
                Some(stream) => {
                    streams.insert(stream);
                }
                None => unregistered += 1,
            }
        }
        for &stream in &streams {
            if let Err(e) = conv.retire_section_generation(stream, "system prompt rebuild") {
                return Some(Err(e));
            }
        }
        tracing::info!(
            retired = streams.len(),
            unregistered,
            "system prompt sections retired; a restart prefills them fresh",
        );
        Some(Ok(SystemPromptRetired {
            retired_streams: streams.len(),
            unregistered_sections: unregistered,
        }))
    }

    /// Decoded turn history for a single recovered conversation — backs
    /// `GET /v1/conversations/{id}`. Returns `None` when the model isn't
    /// loaded yet; an empty `Vec` when the conv_id has no recovered turns.
    pub fn conversation_history(&self, conv_id: &str) -> Option<Vec<RecoveredMessage>> {
        let state = self.inference.load_full()?;
        if let Some(timeline) = passthrough::timeline_of(conv_id) {
            // A passthrough conversation's turns are the client's exchanges.
            let engine = &state.engine;
            // One exchange per turn, in turn order.
            let bubble = |role, text, turn: usize| RecoveredMessage {
                role,
                text,
                turn: turn as u32,
                no_think: false,
                thinking: None,
                tool_tokens: Vec::new(),
            };
            return Some(
                passthrough::held_history(engine, timeline)
                    .into_iter()
                    .enumerate()
                    .flat_map(|(turn, e)| {
                        [
                            bubble(TurnRole::User, e.user.text(), turn),
                            bubble(TurnRole::Assistant, e.assistant, turn),
                        ]
                    })
                    .collect(),
            );
        }
        Some(dialogue_history(&state, conv_id))
    }

    /// The reply in flight for `conv_id`, if there is one.
    pub fn live_turn(&self, conv_id: &str) -> Option<Arc<LiveTurn>> {
        self.live_turns.get(conv_id)
    }

    /// The composer dials this conversation last ran under, for the GUI to set
    /// its dials from when the conversation is opened. `None` when the model
    /// isn't loaded, or when no turn has recorded dials — a conversation that
    /// has never run keeps the client's own defaults.
    /// Why each of the conversation's failed turns stopped, by the turn's
    /// substrate index — recorded as the turn sealed (`turn_error_key`).
    pub fn conversation_turn_errors(&self, conv_id: &str) -> BTreeMap<u32, String> {
        let Some(state) = self.inference.load_full() else {
            return BTreeMap::new();
        };
        let timeline = conversation_timeline(conv_id);
        let meta = state
            .engine
            .conversation_metadata(timeline)
            .unwrap_or_default();
        turn_errors(&meta)
    }

    pub fn conversation_dials(&self, conv_id: &str) -> Option<ConversationDials> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let meta = state.engine.conversation_metadata(timeline)?;
        let level = |key: &str, options: &[&str]| -> Option<u8> {
            let id = meta.get(key)?;
            options.iter().position(|o| o == id).map(|i| i as u8)
        };
        let effort = level(DIAL_EFFORT_KEY, &EFFORT_OPTIONS);
        let verbosity = level(DIAL_LENGTH_KEY, &RESPONSE_LENGTH_OPTIONS);
        let tool_ids = ToolMode::ALL.map(ToolMode::id);
        let tools = level(DIAL_TOOLS_KEY, &tool_ids);
        let think = meta.get(DIAL_THINK_KEY).map(|v| v != "off");
        // Nothing recorded at all: the conversation predates the dial record, or
        // has never taken a turn. Say so rather than inventing levels.
        //
        // The test is whether any dial key is present, NOT whether effort parses
        // to a level. `dials_metadata` writes all four keys on every turn and
        // stores `""` for a selector the turn left unset, so a request carrying
        // a tools dial but no thinking effort records `dial_effort = ""` — a
        // record that plainly exists while parsing to no level. Gating on effort
        // dropped the whole record there, and a resumed turn then ran the
        // defaults rather than the tools mode the conversation was held at.
        let recorded = [
            DIAL_EFFORT_KEY,
            DIAL_LENGTH_KEY,
            DIAL_THINK_KEY,
            DIAL_TOOLS_KEY,
        ]
        .iter()
        .any(|key| meta.contains_key(*key));
        if !recorded {
            return None;
        }
        Some(ConversationDials {
            effort: effort.unwrap_or(2),
            verbosity: verbosity.unwrap_or(2),
            think: think.unwrap_or(true),
            tools: tools.unwrap_or(ToolMode::Comprehensive.level()),
        })
    }

    /// The conversation's sidebar title — the titler's label, or the one an
    /// upload gave it. `None` when the model isn't loaded or nothing has
    /// labelled the conversation yet.
    pub fn conversation_label(&self, conv_id: &str) -> Option<String> {
        let state = self.inference.load_full()?;
        let timeline = passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id));
        let label = state.engine.conversation_label_of(timeline)?;
        (!label.is_empty()).then_some(label)
    }

    /// The branch the conversation works on in each repository, by repository
    /// name — see [`conv_branches`]. Empty when the model isn't loaded or the
    /// conversation is unknown.
    pub fn conversation_branches(&self, conv_id: &str) -> BTreeMap<String, String> {
        let Some(state) = self.inference.load_full() else {
            return BTreeMap::new();
        };
        let timeline = passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id));
        let state = state.engine.conversation_state(timeline);
        state.map(|s| s.branches).unwrap_or_default()
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
        let Some(state) = self.inference.load_full() else {
            return;
        };
        let timeline = timeline_for(conv_id);
        let engine = &state.engine;
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
        conv_branches::seed(engine, timeline, &state.base_branches);
        engine.touch_conversation(timeline);
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
            let base = &state.base_conv;
            base.recovered_history(timeline, false).len() as u32
        };
        // Append the new files to the uploads array it holds.
        if !edit_upload_records(engine, timeline, |records| {
            for f in files {
                let mut f = f.clone();
                f.turn_index = turn_index;
                records.push(f);
            }
        }) {
            tracing::warn!(conv_id = %conv_id, "record_uploads write failed");
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
        let Some(state) = self.inference.load_full() else {
            return;
        };
        if !edit_upload_records(&state.engine, timeline_for(conv_id), |records| {
            for entry in records.iter_mut() {
                if ids.contains(&entry.id) {
                    entry.stats = Some(stats.clone());
                }
            }
        }) {
            tracing::warn!(conv_id = %conv_id, "record_upload_stats write failed");
        }
    }

    /// The upload events recorded against a conversation, in upload order —
    /// backs both the files pane and the inline history tiles on resume.
    /// Empty when the model isn't loaded or nothing was uploaded.
    pub fn conversation_uploads(&self, conv_id: &str) -> Vec<UploadInfo> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let engine = &state.engine;
        engine
            .conversation_metadata(timeline)
            .and_then(|m| m.get("uploads").cloned())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// [`Self::conversation_projections`] as the history carries it: each
    /// point light, without the selection and spine only the projection panel
    /// reads, under a fresh display id. One bucket per assistant turn, in order.
    pub fn conversation_spans(&self, conv_id: &str) -> Vec<Vec<ProjectionSpanOut>> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = &state.base_conv;
            base.recovered_projection_points(timeline)
        };
        raw.into_iter()
            .enumerate()
            .map(|(turn, points)| {
                points
                    .into_iter()
                    .enumerate()
                    .map(|(event, point)| {
                        let seq = PROJ_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        ProjectionSpanOut::of(seq, point, turn, event)
                    })
                    .collect()
            })
            .collect()
    }

    /// Projection-event buckets banked for a conversation this daemon session,
    /// one per assistant turn (in order). Backs the timeline-dot replay on a
    /// browser reload. Empty when the conversation isn't resident (e.g. it was
    /// recovered from disk but hasn't been chatted with since startup).
    pub fn conversation_projections(&self, conv_id: &str) -> Vec<Vec<ProjectionEventOut>> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let raw = {
            let base = &state.base_conv;
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
        let state = self.inference.load_full()?;
        let mut out = {
            let base = &state.base_conv;
            base.section_contents()
        };
        // The `tools` collection's summary section is generated at runtime (not a
        // schema section), so its text isn't in `section_contents`. Serve the
        // summary matching this conversation's tools mode — the tools that mode
        // offers, and none in None (no tools are projected) — under the key the
        // projection event uses (`<collection> summary`) so the panel expands the
        // right list.
        let mode = state
            .tool_modes
            .lock()
            .unwrap()
            .get(conv_id)
            .copied()
            .unwrap_or(ToolMode::Comprehensive);
        if mode == ToolMode::None {
            return Some(out);
        }
        // The tools-collection summary is assembled deterministically from the
        // catalog (the same text the startup seals); rebuild the mode-appropriate
        // one for the panel rather than reading a cache.
        let text = crate::tool_summary::tool_summary_for_mode(mode);
        if !text.is_empty() {
            out.push(("tools summary".to_string(), text));
        }
        Some(out)
    }

    /// The target (dialogue) layer's name, for the panel's conversation prefix.
    pub fn target_layer_name(&self) -> Option<String> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
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
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
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
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
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
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        base.turn_layout(timeline, index)
    }

    /// The dialect's framing markers — the glue wrapped around the system prompt
    /// and each turn. The projection panel renders these between sections/turns.
    pub fn glue_markers(&self) -> Option<GlueMarkers> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        Some(base.glue_markers())
    }

    pub fn start_loading(self: &Arc<Self>) {
        let slot = Arc::clone(&self.inference);
        let loading_substrate = Arc::clone(&self.loading_substrate);
        let proj_builder = self.projection_builder.clone();
        let ready_tx = self.ready_tx.clone();
        let status_tx = self.status_tx.clone();
        let load_progress = Arc::clone(&self.load_progress);
        let workspace = self.config.workspace.clone();
        let secrets = Arc::clone(&self.config.secrets);
        let disabled_layers = self.config.disabled_layers.clone();
        let wiped_layers = self.config.wiped_layers.clone();
        let ingest_dirs = self.config.ingest_dirs.clone();
        let max_depth = self.config.max_depth;
        let agent = self.config.agent.clone();
        let compact_substrate = self.config.compact_substrate;
        let self_check = self.config.self_check;
        let read_only_substrate = self.config.read_only_substrate;
        let qsa_selection_budget = self.config.qsa_selection_budget;
        let qsa_strata = self.config.qsa_strata;
        let summarize = self.config.summarize;
        let tombstone_retention = self.config.tombstone_retention;
        let cluster = self.config.cluster.clone();
        let parts: Arc<dyn SubagentRunner> = Arc::clone(self.mcp.parts()) as _;
        // The memory its writer keeps current once the engine is up.
        let memory = Arc::clone(self.mcp.memory());
        // Handed to both the downloader and the engine builder, so the artifact
        // fetched and the model built are the same one.
        let model = self.config.model.clone();
        let engine_source = self.config.engine.clone();
        // Re-arm zend's ingest-cancel latch for this load: clear any cancel left
        // by a prior load/shutdown before this one's ingest starts polling it.
        // The startup ingest and background reconcile poll
        // `ingest_cancel::cancelled()`; `shutdown` latches it.
        crate::ingest_cancel::token().reset();
        // Handle to the ambient Tokio runtime (if any). The loader runs on a
        // plain OS thread and drops its temporary download runtime before the
        // model load, so the ingest worker's `tokio::spawn` would otherwise
        // panic for lack of a runtime context — which killed the loader thread
        // before `mark_ready()`, wedging the loading screen. We re-enter this
        // handle after the download phase so the worker can spawn.
        let rt_handle = Handle::try_current().ok();
        // The runner a worker's `sub_run` starts its parts with, bound to this
        // session by a weak handle so no tool context keeps it alive. A session
        // loaded with no runtime around it serves no `/mcp`, so has no parts.
        // The same runtime runs the sweep that archives the `/mcp` workers and
        // threads left idle a day.
        if let Some(rt) = &rt_handle {
            self.mcp.parts().bind(Arc::downgrade(self), rt.clone());
            archiving::start_sweeping(self, rt);
            resuming::start_resuming(self, rt);
        }
        // Held by the spawned thread, which stores the ingest worker on the
        // session and hands it to the resumed turns.
        let session_for_loader: Arc<Self> = Arc::clone(self);
        // The same runtime, where the cluster's sockets live: what an agent's
        // fetch for a repository read from the substrate is made on.
        let host_runtime = rt_handle.clone();
        // OS thread, not `tokio::spawn` — `start_loading` may be
        // called from contexts without an ambient Tokio runtime
        // (integration tests, alternative binaries).  The bits that
        // need an async runtime (HF download) drive a local
        // current-thread runtime built inside this thread.
        let loader_handle = std::thread::Builder::new()
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
                        status_tx.send(format!("Runtime build failed: {e}")).ok();
                        fatal(
                            "zend",
                            &format!("local tokio runtime for download failed: {e:#}"),
                        );
                    }
                };
                // The device the pack is resolved for — its int8 mode and layer
                // narrowing are this card's — and then loaded on.
                let device = match Device::cuda_if_available(0) {
                    Ok(d) => d,
                    Err(e) => {
                        status_tx.send(format!("Device init failed: {e}")).ok();
                        fatal("zend", &format!("device init failed: {e:#}"));
                    }
                };
                let pack = match ensure_model_pack(&model, &device, &download_runtime, &status_tx) {
                    Ok(p) => p,
                    Err(e) => {
                        // A missing model is fatal — the daemon cannot serve
                        // anything without it. Fail hard rather than limping
                        // into a fake-ready state that errors on first submit.
                        status_tx.send(format!("Model pack failed: {e}")).ok();
                        fatal("zend", &format!("model pack failed: {e:#}"));
                    }
                };
                // Drop the runtime; the model load below is sync.
                drop(download_runtime);

                // Re-enter the main Tokio runtime for the rest of this thread so the
                // ingest worker's `tokio::spawn` has a runtime context. Must come
                // *after* the download runtime is dropped (no nested `block_on`).
                // The model load is synchronous, so holding the enter guard is safe.
                let _rt_guard = rt_handle.as_ref().map(|h| h.enter());

                status_tx.send("Loading model…".into()).ok();
                // Named from the choice rather than spelled out, so the line
                // cannot go on claiming a model zend has stopped running.
                tracing::info!(
                    "loading inference engine ({}) …",
                    model.clone().spec().model_filename,
                );
                let load_progress_for_blocking = Arc::clone(&load_progress);
                // `InferenceState::load` is fully synchronous (CUDA model
                // load + substrate recovery + ingestion).  Call directly
                // on this thread — no `spawn_blocking` needed.
                match InferenceState::load(
                    proj_builder,
                    model,
                    engine_source,
                    device,
                    pack,
                    workspace,
                    secrets,
                    disabled_layers,
                    wiped_layers,
                    ingest_dirs,
                    max_depth,
                    host_runtime,
                    agent,
                    compact_substrate,
                    self_check,
                    read_only_substrate,
                    qsa_selection_budget,
                    qsa_strata,
                    summarize,
                    tombstone_retention,
                    cluster,
                    status_tx.clone(),
                    load_progress_for_blocking,
                    loading_substrate,
                    parts,
                ) {
                    Ok(Some(state)) => {
                        slot.store(Some(Arc::clone(&state)));
                        tracing::info!("inference engine ready");
                        // A stop asked since the load's last boundary: the engine
                        // is published, so `ZendSession::shutdown` drains it —
                        // skip the warm-ups, the worker and the resumed turns, and
                        // release whatever waits on the load.
                        if crate::ingest_cancel::cancelled() {
                            ready_tx.send(true).ok();
                            return;
                        }
                        // The last load step: the tool catalog's normalization hit
                        // levels, relearned from its corpus in batch on the GPU.
                        // Before `ready`, so no query is ever scored against cold
                        // levels — and no live turn competes with the warm-up.
                        load_progress.set_step(LoadStep::Normalizing);
                        let schema = state.refresh_builder.schema().clone();
                        state.engine.warm_collection_normalization(&schema);
                        if crate::ingest_cancel::cancelled() {
                            ready_tx.send(true).ok();
                            return;
                        }
                        load_progress.set_step(LoadStep::Preloading);
                        let t_preload = Instant::now();
                        let preload = state.engine.preload_galleries(&schema);
                        tracing::info!(
                            turns = preload.turns,
                            resident = preload.resident,
                            tokens = preload.tokens,
                            ms = t_preload.elapsed().as_millis() as u64,
                            "projection galleries preloaded"
                        );
                        status_tx.send(String::new()).ok();
                        // The memory's writer, when this zend serves the
                        // memory and may write: summaries, folds and closes,
                        // at `Low`.
                        // A session loaded with no runtime around it has
                        // nothing to run the writer on, and no `/mcp` to serve
                        // the memory through.
                        if let Some(rt) = rt_handle.as_ref() {
                            if memory.serves() && !read_only_substrate {
                                let writer = rt.spawn(memory_writer::run(
                                    Arc::downgrade(&state),
                                    Arc::clone(&memory),
                                    state.memory_stop.clone(),
                                ));
                                *state.memory_writer.lock().unwrap() = Some(writer);
                            }
                        }
                        // Substrate persistence runs in the engine's own
                        // thread (`PersistenceThread`) — 5 s tick + per-turn
                        // trigger from the scheduler — so no periodic-flush
                        // task is needed at the daemon level here.
                        //
                        // The schema-driven `LoadStep::Ingesting` step is
                        // advanced inside `InferenceState::load` itself — the
                        // ingest passes own it and report sub-step progress
                        // through the same `LoadProgress` handle.

                        // Uploads kept before conversations held them are taken
                        // in once. Uploads are endpoint-managed, but a deletion
                        // of an uploaded file still has to retire its substrate
                        // conversation: the cheap tombstone-if-unheld reconcile
                        // runs once at startup, for uploads deleted since the
                        // last one. A read-only substrate takes none of
                        // this upkeep, here or below: it is written by the daemon
                        // beside it.
                        if !read_only_substrate {
                            state.take_in_legacy_uploads();
                            state.reconcile_uploaded_files();
                        }

                        // The single background ingest worker. Its pass at
                        // startup reads every branch as the tracking refs hold
                        // it (`crate::branch_ingest`), and is followed by the
                        // ingest normalization warm-up; after that it runs when
                        // an origin watcher fetched a moved branch, or a tool
                        // round published one — and on a clock while a layer
                        // changes with time.
                        let wake = Arc::new(Notify::new());
                        if !read_only_substrate {
                            let pass_state = Arc::clone(&state);
                            let pass: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                                match pass_state.refresh_ingest_layers() {
                                    Ok(true) => tracing::info!("ingest pass: layers updated"),
                                    Ok(false) => {
                                        tracing::debug!("ingest pass: no layer changed")
                                    }
                                    Err(e) => tracing::warn!("ingest pass failed: {e:#}"),
                                }
                                // Learned whether or not the pass succeeded: a
                                // failed pass is read again only on the next
                                // branch move or layer clock, and the scans until
                                // then would run without an exclusion.
                                pass_state.learn_chunk_exclusions_behind();
                            });
                            let warm_state = Arc::clone(&state);
                            let after_first_pass: Arc<dyn Fn() + Send + Sync> =
                                Arc::new(move || {
                                    // The first pass is now DONE, so warm the
                                    // per-file normalization hit levels HERE —
                                    // off the load path and, crucially, with no
                                    // concurrent ingest writer to starve (running
                                    // the heavy self-match scan during ingest
                                    // freezes the scheduler). Covers the
                                    // first-run ingest and every restart's
                                    // reconcile. It scores on the scheduler's GPU
                                    // arena a slice of files at a time. The tool
                                    // catalog's own levels are already warmed by
                                    // the load's `Normalizing` step, before ready.
                                    let warmer = warm_state.engine.ingest_warmer();
                                    let schema = warm_state.refresh_builder.schema().clone();
                                    warmer.run(&schema);
                                });
                            let worker = crate::ingest_worker::spawn(
                                Arc::clone(&wake),
                                pass,
                                after_first_pass,
                            );
                            // The startup handoff: the initial backlog — first-ever
                            // ingest, or whatever drifted while the daemon was down
                            // — starts NOW, in the background, while the lines
                            // below take the daemon to `ready` without waiting on it.
                            worker.wake();
                            *session_for_loader.ingest_worker.lock().unwrap() = Some(worker);
                            let _ = state.ingest_wake.set(Arc::clone(&wake));
                            // A record branch that moves in the log — imported
                            // from origin through an agent, or another
                            // member's push — wakes the worker
                            // (`docs/substrate_git.md` §5). Registering wakes it
                            // once too: the follower has run since the load, and
                            // a move in between is reported to no one.
                            if let Some(repositories) = state.repositories.as_ref() {
                                // The wake on registering does not preempt: the
                                // startup pass woken above is the pass it asks
                                // for, and stopping it would read no layer above
                                // the branches until a second one.
                                let on_change: Arc<dyn Fn() + Send + Sync> = {
                                    let wake = Arc::clone(&wake);
                                    let registered = AtomicBool::new(false);
                                    Arc::new(move || {
                                        if registered.swap(true, Ordering::SeqCst) {
                                            crate::ingest_worker::request(&wake);
                                        } else {
                                            wake.notify_one();
                                        }
                                    })
                                };
                                repositories.on_change(on_change);
                            }
                            // A day layer's unit is a day once it is over, and a
                            // unit of a layer that expires goes stale with time;
                            // no branch move announces either. While a layer
                            // passed this boot does either, ask for a pass now
                            // and then. A pass that finds nothing new to read
                            // costs its plans.
                            let on_a_clock = state.branch_layers.iter().any(|(l, _)| {
                                l.spec.as_ref().is_some_and(|s| {
                                    s.unit == crate::layers::UnitKind::Day || s.expiry.is_some()
                                })
                            });
                            if on_a_clock {
                                let wake = Arc::clone(&wake);
                                tokio::spawn(async move {
                                    let mut every = tokio::time::interval(LAYER_CLOCK_EVERY);
                                    every.tick().await;
                                    loop {
                                        every.tick().await;
                                        if crate::ingest_cancel::cancelled() {
                                            return;
                                        }
                                        // A wake, not a request: a day closing or a
                                        // unit expiring is no work for the layers
                                        // below, so a pass in flight is not stopped
                                        // for it — the next pass reads it.
                                        wake.notify_one();
                                    }
                                });
                            }
                        }
                        // The engine is up — only NOW mark ready and unblock
                        // submit-flow waiters. Skipped on the shutdown-during-ingest
                        // path (the `Ok(None)` arm below).
                        load_progress.mark_ready();
                        ready_tx.send(true).ok();

                        // Whatever the last run was in the middle of, finish it
                        // — after `ready`, so the loading screen never waits on
                        // a decode, and never against a read-only substrate,
                        // which is written by the daemon beside this one.
                        if !read_only_substrate {
                            resume_unfinished_turns(
                                Arc::clone(&session_for_loader),
                                Arc::clone(&state),
                            );
                        }
                    }
                    Ok(None) => {
                        // Shutdown arrived during the load — at any step's
                        // boundary, the startup ingest among them: `load` already
                        // drained the engine on this thread (scheduler join →
                        // persistence hot→warm→cold → writer flush). Do NOT mark
                        // ready or publish an engine — the loader thread just exits;
                        // `ZendSession::shutdown`, which is joining it, then returns
                        // and the process exits with everything durable.
                        //
                        // The waiters on the load are released all the same: a
                        // request parked on it would otherwise hold the server's
                        // drain open for good. Each wakes, finds no engine, and
                        // answers so.
                        ready_tx.send(true).ok();
                        tracing::info!(
                            "zend-loader: shutdown during the load — engine drained, loader exiting"
                        );
                    }
                    // A step that ended on the stop's latch — a conversation
                    // create or ingest refused `IngestCancelled` between two
                    // `load_stopped` checks — is the stop, not a failed load:
                    // exiting 1 here would skip the drain and read as a crash.
                    Err(e) if crate::ingest_cancel::cancelled() => {
                        ready_tx.send(true).ok();
                        tracing::info!("zend-loader: the load ended by the stop: {e:#}");
                    }
                    Err(e) => {
                        status_tx.send(format!("Load failed: {e}")).ok();
                        fatal("zend", &format!("inference engine failed to load: {e:#}"));
                    }
                }
            })
            .expect("spawn zend-loader thread");
        // Hand the loader thread's handle to the session so `shutdown` can join it
        // — that join is what guarantees an in-progress load has broken its ingest
        // and finished draining the engine before the process exits.
        *self.load_thread.lock().unwrap() = Some(loader_handle);
    }

    /// Graceful shutdown: stop any in-flight ingest, durably drain + commit the
    /// substrate (hot→warm→cold + off-thread writer flush), then stop the
    /// scheduler. Idempotent, and — crucially — correct DURING loading too: a
    /// Ctrl-C mid-ingest (when `inference` is still `None`) is not a no-op. Runs
    /// the blocking work on the blocking pool since it does `fsync` I/O + joins.
    pub async fn shutdown(&self) {
        // 1. Signal any in-flight ingest — the blocking `raw` ingest inside
        //    `InferenceState::load` AND the background ingest worker's current
        //    pass — to stop submitting. Both poll zend's ingest-cancel latch
        //    cooperatively (as do the deep shared ingest hot paths — per-file
        //    worker pool, cluster scan, raw docs), so no thread is torn down
        //    mid-turn; they break at a safe boundary. `start_loading` clears it
        //    before each load.
        crate::ingest_cancel::token().request();
        // 2. Join the loader thread. If a load is IN PROGRESS, it breaks its ingest
        //    and drains the engine on that thread (the `Ok(None)` path in
        //    `start_loading`), and the join returns only once that drain is done.
        //    If the load already completed (or never started), the thread has ended
        //    and this returns immediately.
        let loader = self.load_thread.lock().unwrap().take();
        if let Some(h) = loader {
            let _ = tokio::task::spawn_blocking(move || {
                if h.join().is_err() {
                    tracing::warn!("shutdown: loader thread panicked");
                }
            })
            .await;
        }
        //    Then the repositories' follower, so no record branch that moves
        //    wakes a worker being stopped, and no import writes past the drain;
        //    the round in flight finishes first. A load that never
        //    published its state dropped its repositories, which stops theirs.
        let followed = self.inference.load_full();
        if let Some(state) = followed {
            let _ = tokio::task::spawn_blocking(move || {
                if let Some(repositories) = state.repositories.as_ref() {
                    repositories.stop();
                }
            })
            .await;
        }
        //    Then the ingest worker. It holds the inference state for its whole
        //    life, so nothing outside this call may keep the engine alive past
        //    the drain below. `stop` sets its flag, wakes it, and AWAITS the
        //    task: if a pass is in flight, the pool breaks at its next unit
        //    boundary on the cancel raised in step 1 (`repo_scan::run_dir_pool`,
        //    `code_read::ingest_jobs`) and the warm-up scan stops between
        //    probes, so the await is bounded. It is an await and never an
        //    `abort()`: Tokio cannot cancel the `spawn_blocking` a pass runs on,
        //    so aborting would return here with a pool still minting
        //    conversations — and the drain below would race it.
        let worker = self.ingest_worker.lock().unwrap().take();
        if let Some(w) = worker {
            w.stop().await;
        }
        // 3. Re-read `inference` AFTER the join. If the load COMPLETED (ready path,
        //    or it finished racing our cancel), the engine is published here and
        //    has NOT been drained — drain it below. If the loader drained mid-ingest
        //    (step 2), `inference` is still `None` and there is nothing left to do.
        //    Exactly-once: the drain runs either on the loader thread OR here, never
        //    both.
        let state = self.inference.load_full();
        let Some(state) = state else {
            tracing::info!(
                "shutdown: loader drained mid-ingest (or model never loaded) — nothing more to persist"
            );
            return;
        };
        // Signal the titler task to stop draining and wake it if it's idle.
        // This must happen before the scheduler stops so the task skips any
        // queued jobs (rather than failing each one against a dead scheduler).
        state.shutting_down.store(true, Ordering::Relaxed);
        let _ = state.titler_tx.try_send(TitleJob::Shutdown);
        // The memory's writer stops on its token: a turn in flight — which may
        // sit behind every other band for minutes — is abandoned at once, and
        // what it began on the blocking pool is awaited, so its question's
        // slot, timeline and frame are all let go while the scheduler still
        // answers.
        // A write waiting on the store waits for the sync thread's push or
        // fetch; past `memory_writer::GRACE` the writer is left to the
        // process's end rather than holding the drain.
        state.memory_stop.request();
        let writer = state.memory_writer.lock().unwrap().take();
        if let Some(mut writer) = writer {
            match tokio::time::timeout(memory_writer::GRACE, &mut writer).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => tracing::warn!("shutdown: memory writer panicked"),
                Err(_) => {
                    tracing::warn!("shutdown: memory writer still busy; left behind");
                    writer.abort();
                }
            }
        }
        let engine_state = Arc::clone(&state);
        let _ = tokio::task::spawn_blocking(move || {
            // Passthrough conversations hold scheduler slots; free them while
            // the scheduler still answers.
            engine_state.passthrough.clear();
            let engine = &engine_state.engine;
            match engine.commit_persistence() {
                Ok(()) => tracing::info!("shutdown: substrate committed"),
                Err(e) => tracing::error!("shutdown: commit failed: {e}"),
            }
            if let Err(e) = engine.shutdown() {
                tracing::error!("shutdown: scheduler stop failed: {e}");
            }
        })
        .await;

        // Await the titler task so any in-flight title turn unwinds (via
        // `abort_turn`, against the now-stopped scheduler) before the process
        // exits. The guard is released before the await — the take is all it
        // covers.
        let worker = state.titler_worker.lock().unwrap().take();
        if let Some(worker) = worker {
            if worker.await.is_err() {
                tracing::warn!("shutdown: titler task panicked");
            }
        }
    }

    /// Run an OpenAI-passthrough call — the client's own context, as-is — and
    /// return a stream of its reply tokens. See [`crate::passthrough`].
    ///
    /// Waits for the engine, then for the conversation's own lock: a call on a
    /// conversation another call is running queues behind it, while calls on
    /// other conversations go ahead.
    pub async fn submit_passthrough(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
        max_tokens: Option<usize>,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);
        let transcript = match Transcript::from_messages(&messages, &tools) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("passthrough: {e}"))).await;
                return Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
            }
        };
        let inference = Arc::clone(&self.inference);
        let mut ready_rx = self.ready_tx.subscribe();
        tokio::spawn(async move {
            loop {
                // Loaded, or the load is over without an engine (a stop): the
                // read below tells the two apart.
                let loaded = inference.load().is_some();
                if loaded || *ready_rx.borrow_and_update() {
                    break;
                }
                if ready_rx.changed().await.is_err() {
                    return;
                }
            }
            let state = { inference.load_full() };
            let Some(state) = state else {
                return;
            };
            state.passthrough.ensure_sweeper();
            let key = transcript.key();
            // The conversation stays this member's until the reply is sealed.
            let claim = turn_claims::hold(&key);
            let conv = state.passthrough.conversation(&key).lock_owned().await;
            run_passthrough(state, key.clone(), conv, transcript, max_tokens, tx).await;
            turn_claims::release(claim, &key).await;
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Submit the latest user message and return a stream of status + token items.
    // See the note on `run_inference_stream`.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        force_hires: Option<String>,
        assistant_prefill: Option<String>,
        lossless_kv: bool,
        tools_mode: ToolMode,
        identity: Option<String>,
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
            identity,
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
    // See the note on `run_inference_stream`.
    #[allow(clippy::too_many_arguments)]
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
        identity: Option<String>,
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
        let live_turns = Arc::clone(&self.live_turns);

        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);

        tokio::spawn(async move {
            // Phase 1 — emit status updates while model is loading.
            if inference.load().is_none() {
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
            let state = inference.load_full();
            if let Some(state) = state {
                let history_before = dialogue_history(&state, &conv_id).len();
                // The conversation stays this member's until the turn's last
                // record is written, however soon its client goes.
                let claim = turn_claims::hold(&conv_id);
                let live = live_turns.begin(&conv_id, last_user.clone(), history_before);
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
                    identity,
                    selection,
                    None,
                );
                relay_live(&mut ts, &tx, &live, &conv_id).await;
                live_turns.end(&conv_id, &live);
                // The reply is over: its stream ends now, and the claim waits on
                // the turn's commit behind it, where no client waits.
                drop(ts);
                drop(tx);
                turn_claims::release(claim, &conv_id).await;
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

/// The timeline a sidebar `id` names: a passthrough conversation's id carries
/// its timeline ([`passthrough::timeline_of`]); any other is [`timeline_for`].
fn conversation_timeline(conv_id: &str) -> projection::TimelineId {
    passthrough::timeline_of(conv_id).unwrap_or_else(|| timeline_for(conv_id))
}

/// The [`projection::TimelineId`] a client `conv_id` maps to — a stable hash
/// of the id. Deterministic across daemon restarts, so a reconnecting client
/// resolves to the same timeline the substrate reload recovered (§16.12).
pub fn timeline_for(conv_id: &str) -> projection::TimelineId {
    let h = content_hash::hash_bytes(conv_id.as_bytes());
    projection::TimelineId::from_raw(h.lo.max(1)).expect("timeline id is non-zero")
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

/// What a committed ingested unit is keyed by, workspace-relative: a file's
/// reading by its path, a folder's by its folder, a layer's unit by its
/// subject — the file or folder it analyses. `None` for a conversation that is
/// no committed unit.
fn committed_unit_key(meta: &POrdMap<String, String>) -> Option<&String> {
    meta.get(PATH_KEY)
        .or_else(|| meta.get(DIR_KEY))
        .or_else(|| meta.get(SUBJECT_KEY))
        .filter(|_| meta.contains_key(CONTENT_KEY))
}

/// How zend names one of its conversations in a layer's list. An ingested unit
/// — the internal content layers (repo_map, code_reading, the analysis layers)
/// — the way the tools address it: its repository, the path inside it, and the
/// branches or commit that tell one version from another.
fn unit_of(s: &Substrate, tl: TimelineId) -> Unit {
    // Title precedence: the sidebar label, then the conv_id, then the
    // conversation's `path` metadata — code_reading (and any file-scoped
    // ingest) records the file path there at creation, so those otherwise
    // conv_id-less timelines read as the file they cover instead of
    // "(untitled)".
    let custom = s.custom_of(tl);
    let label = title(s, tl)
        .or_else(|| {
            custom
                .and_then(|m| m.get(PATH_KEY).or_else(|| m.get(SUBJECT_KEY)))
                .filter(|p| !p.is_empty())
                .cloned()
        })
        .unwrap_or_default();
    let branches = custom
        .and_then(|m| m.get(BRANCHES_KEY))
        .map(|v| {
            v.split(',')
                .filter(|b| !b.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let commit = custom
        .and_then(|m| m.get(COMMIT_KEY))
        .cloned()
        .unwrap_or_default();
    // An ingested unit is keyed workspace-relative (`candle/CLAUDE.md`); the
    // view shows it the way the tools address it — its repository, and the
    // path inside it.
    let (repo, label) = match custom.and_then(committed_unit_key) {
        Some(key) => {
            let (repo, inner) = shown_unit(key);
            (repo.to_string(), inner)
        }
        None => (String::new(), label),
    };
    Unit {
        label,
        repo,
        branches,
        commit,
    }
}

/// The status a waiting prefill shows (`TurnEvent::PrefillQueued`): what it
/// waits behind and for how long, in whole seconds, so the text changes once a
/// second at most — "Waiting for the GPU — 2 turns (18K tokens) ahead, 4 s".
fn queued_status(ahead_tokens: usize, ahead_turns: usize, waiting_ms: u64) -> String {
    let secs = waiting_ms / 1000;
    if ahead_turns == 0 {
        return format!("Waiting for the GPU — {secs} s");
    }
    let tokens = if ahead_tokens >= 1000 {
        format!("{}K", ahead_tokens / 1000)
    } else {
        ahead_tokens.to_string()
    };
    let turns = if ahead_turns == 1 { "turn" } else { "turns" };
    format!("Waiting for the GPU — {ahead_turns} {turns} ({tokens} tokens) ahead, {secs} s")
}

/// The workspace's projection schema — its `projection.yaml` or the embedded
/// default. A schema that does not parse stops zend at its load: there is no
/// daemon to run without one.
fn build_projection_builder(workspace: &Path) -> Builder {
    try_projection_builder(workspace).unwrap_or_else(|e| panic!("{e}"))
}

/// [`build_projection_builder`], with a schema that does not parse as an
/// error — for a host reading zend's settings without running zend.
pub(crate) fn try_projection_builder(workspace: &Path) -> Result<Builder, String> {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    // Qwen3-30B-A3B is a ChatML-family model; the dialect is required
    // so the schema's `kind: template` items (system_open / system_close,
    // tool_block_open/close, no_think_prefix) resolve to the right
    // structural-token strings at parse time.
    let dialect = Dialect::chat_ml();
    // A `projection.yaml` in the workspace OVERRIDES the embedded schema, so a
    // dedicated (uncommitted) mind directory — e.g. `--working-dir ../mind` — can
    // tune its own cognitive substrate without touching the checked-in default.
    let override_path = workspace.join("projection.yaml");
    let overridden = std::fs::read_to_string(&override_path).ok();
    let (yaml, source): (&str, String) = match &overridden {
        Some(s) => {
            tracing::info!(path = %override_path.display(), "projection schema: workspace override");
            (s.as_str(), override_path.display().to_string())
        }
        None => (PROJECTION_SCHEMA_TEMPLATE, "embedded default".to_string()),
    };
    Builder::from_yaml_with_vars_and_dialect(yaml, &[("workspace", name)], Some(&dialect))
        .map_err(|e| format!("projection.yaml ({source}) failed to parse — check YAML syntax and {{workspace}} placeholder: {e:#}"))
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
    let mut out = String::new();
    for item in &builder.schema().system_prompt.items {
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
mod queued_status_tests {
    use super::queued_status;

    /// What a waiting prefill shows: whole seconds, tokens in K past a
    /// thousand, the ahead part only when something is ahead.
    #[test]
    fn a_wait_reads_as_what_is_ahead_and_how_long() {
        assert_eq!(
            queued_status(18_400, 2, 4_900),
            "Waiting for the GPU — 2 turns (18K tokens) ahead, 4 s"
        );
        assert_eq!(
            queued_status(512, 1, 1_000),
            "Waiting for the GPU — 1 turn (512 tokens) ahead, 1 s"
        );
        assert_eq!(queued_status(0, 0, 3_200), "Waiting for the GPU — 3 s");
    }
}

#[cfg(test)]
mod projection_schema_tests {
    use super::build_projection_builder;
    use candle_conversation::projection::{SelectionRule, WorkingSetShare};
    use host::substrate_views::system_prompt_labels;
    use std::collections::HashMap;
    use std::path::Path;

    /// The dialogue layer's system-prompt labels name its framing sections, its
    /// tools collection (with a member count), and its section tree — the
    /// structure the viewer lists under "Target layer".
    #[test]
    fn dialogue_system_prompt_labels_cover_sections_collection_and_tree() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let labels = system_prompt_labels(&builder.schema().system_prompt);

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

    /// The embedded schema carries the working set (`docs/zend_working_set.md`)
    /// onto the right nodes: `repo_map/structure` and `code_reading/scopes`
    /// fill from the dialogue's working set, in tool rounds too, with the
    /// scan's settings kept (they build the momentum) and the belief-only
    /// tuning — locality, anchor, adaptive budgets — gone; the dialogue layer
    /// declares the budget; the tools collection stays additive (the measured
    /// per-axis split).
    #[test]
    fn embedded_schema_carries_the_working_set_config() {
        use crate::calibrate::CalibrateSpec;
        use candle_conversation::projection::{ScanCases, SystemPromptItem};
        use candle_conversation::provenance::FusionMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        let schema = builder.schema();

        let layer = |name: &str| {
            schema
                .layers
                .iter()
                .find(|l| l.name == name)
                .unwrap_or_else(|| panic!("layer {name} present"))
        };
        for (name, share) in [
            ("repo_map", WorkingSetShare::Folders),
            ("code_reading", WorkingSetShare::Remainder),
        ] {
            let l = layer(name);
            let g = &l.groups[0];
            assert_eq!(g.selection, SelectionRule::WorkingSet { share }, "{name}");
            assert!(l.in_tool_rounds, "{name} is in tool rounds");
            // Content axes are ADDITIVE — content_gated was reverted after live
            // testing showed it zeros NL→code retrieval (the domain gap).
            assert_eq!(g.policy.scan.fusion, FusionMode::Additive, "{name}");
            assert!(g.policy.scan.question_pin, "{name}");
            assert!(g.locality.is_none() && g.anchor.is_none(), "{name}");
            assert!(g.budget_adaptive.is_none(), "{name}");
            assert!(l.budget.adaptive.is_none(), "{name}");
        }

        // A whole file is one case of one competition, content weighed on fold
        // groups 1-2 and the calibration competition on the top group alone,
        // at a quarter of the content's weight.
        let g = &layer("code_reading").groups[0];
        assert_eq!(g.policy.scan.cases, ScanCases::File);
        assert_eq!(g.policy.layer_weights, vec![0.0, 1.0, 1.0]);
        assert_eq!(g.policy.calibration_layer_weights, vec![0.0, 0.0, 1.0]);
        assert_eq!(g.policy.scan.calibration_weight, 0.25);
        let scan = &g.policy.scan;
        assert_eq!((scan.level_prior_base, scan.level_prior_cap), (2.0, 16.0));
        assert_eq!(
            layer("repo_map").groups[0].policy.scan.cases,
            ScanCases::Exchange
        );
        // Each file's reading is calibrated with the questions it answers,
        // each opening the thinking it starts.
        let calibrate = CalibrateSpec::of(layer("code_reading"))
            .unwrap()
            .expect("code_reading declares a calibration");
        assert_eq!((calibrate.questions, calibrate.think), (24, 64));
        assert!(calibrate.prompt.contains("<<path>>"));

        let ws = layer("dialogue")
            .working_set
            .as_ref()
            .expect("the dialogue declares a working set");
        assert_eq!(ws.budget_tokens, 150_000);
        assert_eq!(ws.lock_tokens, 150_000);
        assert_eq!(ws.folder_tokens, 10_000);
        assert_eq!(ws.beta, 0.2);
        assert_eq!(ws.min_momentum, 100.0);
        assert_eq!(ws.max_file_tokens, 100_000);
        assert_eq!(ws.seeds.len(), 17);
        assert_eq!(ws.seeds[0], "candle/zend/");

        let tools = schema
            .system_prompt
            .items
            .iter()
            .find_map(|i| match i {
                SystemPromptItem::Collection(c) if c.name == "tools" => Some(c),
                _ => None,
            })
            .expect("tools collection");
        assert_eq!(
            tools.policy.scan.fusion,
            FusionMode::Additive,
            "tools MUST stay additive (results doc §25.8: gating collapses Top-1 97.3% → 32.9%)"
        );
        // Strict normalized-band gate (reverted from the too-permissive min 112).
        assert_eq!(tools.policy.config.min_score, 800.0);
        assert_eq!(tools.policy.config.budget_max, 3);

        // Dialogue stays recency-driven — no scan concepts apply.
        let dialogue = layer("dialogue");
        assert!(!dialogue.groups[0].is_belief_driven());
    }

    /// The turn-sink load plan is DERIVED from the embedded schema's structure —
    /// no `ingest:` annotations: the built-in `repo_map` (folder scan) and
    /// `code_reading` (file read) over every repository — an empty scope — the
    /// live `dialogue` layer excluded.
    #[test]
    fn embedded_turn_sinks_are_derived_from_structure() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        // "demo-project" doesn't exist on disk, so no folder-backed raw sinks
        // resolve — only the two built-in pipelines.
        let layers = crate::ingest::ingest_layers(
            builder.schema(),
            Path::new("demo-project"),
            &HashMap::new(),
        );
        let got: Vec<(&str, IngestMode, &str)> = layers
            .iter()
            .map(|l| (l.name.as_str(), l.mode, l.folder.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("repo_map", IngestMode::Folders, ""),
                ("code_reading", IngestMode::Files, ""),
            ],
            "derived turn-sinks: {got:?}"
        );
        // Every layer keeps the id its conversations were persisted under.
        let ids: Vec<(&str, u32)> = builder
            .schema()
            .layers
            .iter()
            .map(|l| (l.name.as_str(), l.id.raw()))
            .collect();
        assert_eq!(
            ids,
            vec![("repo_map", 1), ("code_reading", 2), ("dialogue", 10),]
        );
        // The live dialogue layer is never a turn-sink.
        assert!(
            !layers.iter().any(|l| l.name == "dialogue"),
            "dialogue must be excluded as the live conversation layer",
        );
        // The coding-assistant schema has no folder-backed section collections:
        // its only collection is the registry-backed `tools`, which is excluded.
        assert!(
            crate::ingest::section_sinks(builder.schema()).is_empty(),
            "embedded schema should derive no section-collection sinks",
        );
    }

    /// `--ingest-dir <layer>=<path>` replaces a derived layer's content root, so
    /// a rebuild can be scoped to a subtree instead of sweeping the workspace.
    /// Layers without an override keep their derived root.
    #[test]
    fn ingest_dir_overrides_a_layers_content_root() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        let mut dirs = HashMap::new();
        dirs.insert("code_reading".to_string(), "candle/zend/src".to_string());

        let layers =
            crate::ingest::ingest_layers(builder.schema(), Path::new("demo-project"), &dirs);
        let code = layers
            .iter()
            .find(|l| l.name == "code_reading")
            .expect("code_reading derives");
        assert_eq!(code.folder, "candle/zend/src", "the scope is the override");
        assert_eq!(code.mode, IngestMode::Files, "mode is unchanged");
        assert!(
            code.display.contains("candle/zend/src"),
            "the scoped root shows in the loading phase label: {:?}",
            code.display
        );

        // An un-overridden layer keeps its derived scope: every repository.
        let repo = layers
            .iter()
            .find(|l| l.name == "repo_map")
            .expect("repo_map derives");
        assert_eq!(repo.folder, "");
        assert_eq!(repo.display, "Scanning repositories");
    }

    /// A coding-agent workspace with a folder matching a declared-but-pipeline-fed
    /// layer name (e.g. `code_reading/`) must NOT be ingested as a raw sink; only a
    /// mind (a workspace with its own `projection.yaml`) draws raw sinks from its
    /// folders. Guards the folder-name-collision fix.
    #[test]
    fn coding_workspace_folder_never_ingests_a_pipeline_layer() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        let ws = std::env::temp_dir().join(format!("zend_ingestws_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(ws.join("code_reading")).unwrap();

        // Not a mind (no projection.yaml): the colliding `code_reading/` folder
        // is never a raw sink — the layer is read by its own pipeline.
        let coding = crate::ingest::ingest_layers(builder.schema(), &ws, &HashMap::new());
        assert!(
            !coding.iter().any(|l| l.mode == IngestMode::Raw),
            "no raw sinks in a coding-agent workspace"
        );
        assert!(coding
            .iter()
            .any(|l| l.name == "code_reading" && l.mode == IngestMode::Files));

        let _ = std::fs::remove_dir_all(&ws);
    }

    /// The shipped `projection.yaml` parses, and the repo map a dialogue sees is
    /// its working set's folders — no group-level threshold or default: a cold
    /// probe selects only what is pinned, because a folder is a provenance
    /// candidate only once its momentum clears `min_momentum`.
    #[test]
    fn projection_yaml_parses_and_repo_map_is_the_working_sets_folders() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let structure = builder
            .id_for_group("structure")
            .expect("repo_map declares a 'structure' group");
        let group = builder.group(structure).expect("group schema present");
        assert_eq!(
            group.selection,
            SelectionRule::WorkingSet {
                share: WorkingSetShare::Folders
            },
        );
        assert_eq!(group.score_threshold, None);
    }

    /// The shipped schema floors every sequence at the ×2 YaRN rung — the
    /// value `load` hands to `ModelBuilder::min_rope_factor`.
    #[test]
    fn projection_yaml_sets_the_min_yarn_factor() {
        let builder = build_projection_builder(Path::new("demo-project"));
        assert_eq!(builder.schema().min_yarn_factor, Some(2.0));
    }
}

#[cfg(test)]
mod status_detail_tests {
    use super::ZendSession;
    use crate::config::DaemonConfig;
    use crate::workspace::single_repo;
    use tempfile::TempDir;

    /// **A detail sent while nobody is reading is the next snapshot's.** The
    /// load sends each phase's name once, as the phase begins; when the
    /// channel had no reader open, the send was refused and dropped, and every
    /// snapshot read an empty detail.
    #[test]
    fn a_detail_sent_between_snapshots_is_the_next_snapshots() {
        let tmp = TempDir::new().unwrap();
        let session = ZendSession::new(DaemonConfig::new(
            single_repo(tmp.path(), "project").unwrap(),
        ));
        session
            .status_tx
            .send("Reading experts into host memory".into())
            .unwrap();
        assert_eq!(
            session.status_snapshot().detail,
            "Reading experts into host memory"
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
mod round_cap_tests {
    use super::{closes_at_cap, MAX_TOOL_ROUNDS};

    /// **The round cap closes the loop rather than stopping it**: the results
    /// of the round before the cap are answered by a closing turn, which still
    /// runs under the cap, and no earlier round's are.
    #[test]
    fn the_last_answerable_round_is_closed() {
        assert!(!closes_at_cap(0));
        assert!(!closes_at_cap(MAX_TOOL_ROUNDS - 3));
        assert!(closes_at_cap(MAX_TOOL_ROUNDS - 2));
        let closing_turn = MAX_TOOL_ROUNDS - 2 + 1;
        assert!(closing_turn < MAX_TOOL_ROUNDS, "the closing turn runs");
    }
}

#[cfg(test)]
mod turn_cap_tests {
    use super::MAX_TURN_TOKENS;
    use candle_conversation::stencil::{ThinkMode, MAX_STRING_VALUE_TOKENS};

    /// **A turn can think as long as any dial allows and still write one whole
    /// string value.** Otherwise the turn's own cap, not the value's, is what
    /// cuts a large file — which is the failure this cap exists to remove.
    #[test]
    fn a_turn_holds_the_longest_think_block_and_a_full_string_value() {
        let longest_think = [
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ]
        .into_iter()
        .map(|m| m.span_cap() as usize)
        .max()
        .unwrap();
        assert!(
            MAX_TURN_TOKENS > longest_think + MAX_STRING_VALUE_TOKENS as usize,
            "{MAX_TURN_TOKENS} cannot hold {longest_think} thinking + {MAX_STRING_VALUE_TOKENS} of value",
        );
    }
}

#[cfg(test)]
mod unit_key_tests {
    use candle_conversation::substrate::pcoll::POrdMap;

    use super::committed_unit_key;

    fn meta(pairs: &[(&str, &str)]) -> POrdMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// **A committed unit is named by what it covers**: a file's reading by
    /// its path, a folder by its folder, a layer's unit by its subject — and a
    /// conversation without a content key by none of them.
    #[test]
    fn a_committed_unit_is_named_by_what_it_covers() {
        let read = meta(&[("path", "candle/zend/src/rate.rs"), ("content_key", "k1")]);
        assert_eq!(
            committed_unit_key(&read).map(String::as_str),
            Some("candle/zend/src/rate.rs")
        );
        let review = meta(&[
            ("layer", "code_review"),
            ("subject", "candle/zend/src/rate.rs"),
            ("content_key", "u2:code_review:ab"),
        ]);
        assert_eq!(
            committed_unit_key(&review).map(String::as_str),
            Some("candle/zend/src/rate.rs")
        );
        let partial = meta(&[("layer", "code_review"), ("subject", "candle/x.rs")]);
        assert_eq!(committed_unit_key(&partial), None);
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
