//! The daemon's chat session: one loaded model, the substrate it writes to, and
//! the conversations that run on them.
//!
//! `InferenceState` owns the engine and the per-conversation state, keyed by the
//! client's conversation id. Engine methods take `&self`, so no reader waits on
//! another. Each conversation's reply is serialised by an async lock held across
//! its decode, and an idle sweep releases conversations no one uses. Turns fork
//! from prefilled base conversations, and ingest units fork from their own
//! prefilled templates. `ZendSession` wraps the engine with the load state and
//! the background workers, and shutdown stops ingest and the titler before the
//! engine drains.
//!
//! - `builders`: the projection each turn runs under, by tools mode and identity.
//! - `conversations`: listing, archiving and tombstoning conversations.
//! - `core`: access to the model, substrate, cluster and roles.
//! - `dials`: composer dials recorded per conversation, and their settings.
//! - `history`: a conversation's recovered turns and their errors.
//! - `inference_stream`: one chat turn as an async task.
//! - `ingesting`: background ingest passes and calibration.
//! - `journaling`: journal sections, drafts and folds.
//! - `lifecycle`: leave, idle release and ordered shutdown.
//! - `load`: the startup load on the loader thread.
//! - `passthrough_turns`: OpenAI-passthrough calls.
//! - `prompt_text`: prompt and section text for the projection panel.
//! - `status`: the status snapshot and maintenance state.
//! - `submission`: submit entry points and worker turns.
//! - `substrate_view_routes`: reads for the substrate views and projection query.
//! - `timeline_ids`: client ids to timelines, and unit naming.
//! - `title_worker`: the sidebar titler task.
//! - `triggers`: token triggers for tool calls and thinking.
//! - `turn_types`: stream items, turn starts and resumes.
//! - `unfinished_turns`: resuming requests a previous run left.
//! - `uploads`: upload storage, records and ingest.
//! - `working_reads`: working-set withholding and seeding.
//! - `workspace_tools`: repository tool contexts.
//! - `tests_projection`, `tests_status`, `tests_timeline`, `tests_turn`: tests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(feature = "memory-census")]
use heap_census::{set_thread, Scope};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use agent::client::Agents;
use agent::jobs::Jobs;
use arc_swap::{ArcSwap, ArcSwapOption};
use cluster::member::{AppMember, ClusterView, Link};
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

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use candle::{vram, Device};
use candle_conversation::models::{Dialect, Model, StrataTokens};
use candle_conversation::persistence::record::DistillMode;
use candle_conversation::persistence::{
    content_hash, OpenProgressHook, OpenStage, SharedSubstrate, SUBSTRATE_DIR,
};
use candle_conversation::projection::{
    self, Builder, Conversation, DecodePriority, ExclusionLearner, GroupId, LayerId, Reserved,
    SectionId, SectionLoads, SelectionRule, SystemItem, SystemPromptItem, TimelineId, TurnIndex,
};
use candle_conversation::provenance::gallery_arena::GroupSet;
use candle_conversation::provenance::{encode_wide_sigs, FOLD_GROUPS};
use candle_conversation::stencil::{
    compile_action_loop, OpeningCall, ThinkMode, ToolCallEnvelope, ToolSpec, TriggerRegistry,
    MAX_TOOL_CALLS_PER_TURN,
};
use candle_conversation::substrate::pcoll::POrdMap;
use candle_conversation::substrate::Substrate;
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::working_set::WorkingSetConfig;
use candle_conversation::FinishReason;
use candle_conversation::LoadPhase;
use candle_conversation::RecoveredMessage;
use candle_conversation::Result as EngineResult;
use candle_conversation::Role as TurnRole;
use candle_conversation::TurnText;
use candle_conversation::{
    CancelToken, Carried, ConversationEngine, ConversationError, EngineSource, GlueMarkers,
    GroupScanner, Linking, Probed, PromptDump, SamplingConfig, SelectionState, Sequence, Sharing,
    ThinkSteering, TokenDecoder, TurnEvent, UnsealedAsker, Want,
};
use host::calibration;
use host::download::ensure_model_pack;
use host::model::ModelView;
use host::process::fatal;
use host::substrate_views::{
    layer_conversations, overview, system_prompt, timeline, title, LayerConversations,
    SubstrateOverview, SystemPromptView, TimelineDetail, Unit,
};
use journal_core::entry::Entry;
use journal_core::fold;
use journal_core::folding::{self, Folded};
use journal_core::journal::Holds;
use journal_core::journal::Journal;
use journal_core::record::view as journal_view;
use journal_core::section::{
    entry_section, JournalPrompt, Kept, Section, CLOSE_SECTION, HISTORY_SECTION, JOURNAL,
};
use journal_core::span::Span;
use journal_core::transcript::{Filed, Hits, Miss, Page};
use journal_core::workflow::Skip;
use journals::ops::wire::path_from_wire;
use journals::path::JournalPath;
use journals::time::Stamp;
use mcp_server::refs::{self, Refs};
use serde_json::{json, Value};
use web::auth::{Role as AccessRole, Roles};
use zend_tools::run_watch::{self, RunWatch};
use zend_tools::state::Secrets;
use zend_tools::tools::journal::{JournalBook, JournalShelf};
use zend_tools::{OpsRunner, ToolContext};
use zend_vfs::{RepoFiles, Workspace};

use crate::answer_steps;
use crate::api::chat::{
    apply_tools_dial, dial_selection, tool_round_selection, EFFORT_OPTIONS, RESPONSE_LENGTH_OPTIONS,
};
use crate::api::substrate::{
    ProjectTile, ProjectView, SelectedNodeView, SelectionView, SystemPromptRetired, ToolView,
    ToolsView,
};
use crate::branch_ingest::filter::IngestScope;
use crate::branch_ingest::ignore::Ignore;
use crate::branch_ingest::keys::{BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY};
use crate::branch_ingest::{moves_branches, BranchIngest, LayerPass};
use crate::calibrate::thinking::thinking_of;
use crate::calibrate::{
    calibrate_layer, mark_sealed, Frame as CalibrateFrame, Layer as CalibrateLayer,
};
use crate::code_read::{read_opening, reading_tools, PATH_KEY, READ_CALLS_PER_REPLY};
use crate::coding_sampling;
use crate::config::{DaemonConfig, SelfCheck};
use crate::conv_branches::{self, BaseBranches};
use crate::conv_inherit;
use crate::conv_order;
use crate::conv_overlay::{self, Mirror};
use crate::conv_sweep::{CONV_IDLE_TTL, CONV_SWEEP_EVERY};
use crate::deletions::Deletions;
use crate::engine_asks;
use crate::guidance;
use crate::hand_records::{self, Carrying};
use crate::held_values::FilesSeen;
use crate::images;
use crate::ingest::{IngestConv, IngestLayer, IngestMode};
use crate::ingest_cancel;
use crate::ingest_lease;
use crate::journal::desk::{Asking, Compile, Record, ZendDesk, FOLD_TOKENS, WRITE_TOKENS};
use crate::journal::notes::{own_notes, JOURNAL_NOTE_TOOL};
use crate::journal::opening::{self, OPENING_SEAL_WAIT};
use crate::journal::record::{kept_in, metadata_of, KEY as JOURNAL_KEY};
use crate::journal::registry::{Journals, Open};
use crate::journal::sharing::chain_sharing;
use crate::journal::shelf::{Above, Drafted, Lazy, Notes, Shelf};
use crate::journal::state::{Request, SessionJournal, FOLD_SWEEP_EVERY};
use crate::journal::style::style_for;
use crate::journal::text::JournalText;
use crate::journal::world::{Citable, JournalsAbove};
use crate::journal::{carry, draft, origin, pressure};
use crate::layers::expiry::now_ms;
use crate::layers::keys::SUBJECT_KEY;
use crate::leaving::{Leaving, Running, LEAVE_TURNS_WAIT};
use crate::live_turn::{LiveTurn, LiveTurns};
use crate::loading::{LoadProgress, LoadStep, LoadingSnapshot};
use crate::mcp::caller::Caller;
use crate::mcp::chats::sweep as chat_sweep;
use crate::mcp::diaries::placement as diary_placement;
use crate::mcp::memory::readers::MemoryReaders;
use crate::mcp::memory::Memory;
use crate::mcp::ownership::owner_of;
use crate::mcp::resuming::RESUME_SWEEP;
use crate::mcp::state::McpState;
use crate::mcp::threads::{conv_names, project_journal};
use crate::mcp::zen_tasks::{level_of, withhold_by_level};
use crate::mcp::{
    archiving, lineage, memory_sections, nudges, resuming, stops, tending, thread_export, threads,
    workers,
};
use crate::memory_paths;
use crate::passthrough::{self, Exchange, LiveConv, PassthroughCache, Transcript};
use crate::projection_event::{ProjectionEventOut, ProjectionSpanOut};
use crate::refresh_ctx::RefreshContext;
use crate::repeat_guard::{self, RepeatGuard, Verdict};
use crate::repo_ads::repo_ads;
use crate::repo_path::shown as shown_unit;
use crate::repo_scan::{utility_config, DIR_KEY};
use crate::resident::Resident;
use crate::resume;
use crate::resume::{Unfinished, Wholeness};
use crate::retrieval_scope::{Readers, RetrievalScope};
use crate::round_timing::{self, RoundTiming};
use crate::self_check::{self, Report as SelfCheckReport};
use crate::substrate_git::{self, Repositories};
use crate::think_budget;
use crate::think_progress::{answer_start, ThinkProgress, ThinkUpdate};
use crate::tool_availability::{announce_returned, unavailable, Unavailable};
use crate::tool_def;
use crate::tool_round;
use crate::tools::{
    calibration_frame, extract_tool_calls, format_tool_responses, install_tool_catalog,
    memory_hidden, run_tool_calls, Dispatch, ToolCall, ToolHost, ToolResult,
};
use crate::turn_claims::{self, NotTaken};
use crate::types::{ChatMessage, Role, ToolMode, Usage};
use crate::worker_hop;
use crate::working_set::coverage::Coverage;
use crate::working_set::seeds::Seeding;
use crate::working_set::{corpus, dialogue_config};
use crate::workspace::UPLOADS_REPO;
use candle_conversation::vision::tenancy::{holds, note_use, release_conversation, sweep_unused};
use candle_conversation::vision::{load_original, load_thumb, marker, split_markers, ImageSource};

pub mod diag;
mod kv_census;
mod memory_writer;
mod prewarm;
mod replay;
mod titler;
use prewarm::Prewarm;
use titler::{retire_titler_timelines, Titler};
pub mod restore;
mod titles;

pub use replay::{
    RecordedTurn, ReplayOutcome, ReplayReport, ReplaySampling, ReplaySpec, TurnComparison,
};
use titles::{
    clean_title, head_tail_truncate, retitle_prompt, Finished, PendingRetitles, FIRST_HEAD_TOKENS,
    FIRST_TAIL_TOKENS, TITLER_SYSTEM_PROMPT,
};

pub const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("../prompts/projection.yaml");

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
    /// The priming chain's final link (`BranchIngest::prime`, built at boot
    /// before `ready` and again whenever a link changes): this member's seed,
    /// which every dialogue's
    /// `forked_from` alias resolves to (`Substrate::SEED_PARENT`), and the
    /// parent every ingest unit's conversation records before its reading
    /// starts. `None` when there is no chain — no git repository, or no
    /// ingest layer.
    chain_end: Mutex<Option<TimelineId>>,
    /// Queue feeding the dedicated titler task. The request path enqueues a
    /// [`TitleJob`] (non-blocking, dropped if the task is backed up) instead
    /// of spawning per submit, so title generation runs in the background —
    /// concurrently with main decode, never serialised against the request
    /// path — and drains cleanly on shutdown.
    titler_tx: mpsc::Sender<TitleJob>,
    /// The re-titles queued and not yet started, one per conversation — see
    /// [`titles`].
    retitles: PendingRetitles,
    /// Handle to the titler task, awaited during [`ZendSession::shutdown`]
    /// so any in-flight title turn unwinds before the process exits.
    titler_worker: Mutex<Option<TaskHandle<()>>>,
    /// The memory's writer (`memory_writer`), awaited during
    /// [`ZendSession::shutdown`] once `memory_stop` is requested, while the
    /// scheduler still answers.
    memory_writer: Mutex<Option<TaskHandle<()>>>,
    /// What the memory's writer stops on — between jobs, and mid-turn.
    memory_stop: CancelToken,
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
    /// The memory a worker's sections read its thread's topics from
    /// (`mcp::memory_sections`).
    memory: Arc<Memory>,
    /// The ingest worker's wake, once it is running: a `git_*` tool round
    /// that can move a branch wakes it, since a push moves the tracking refs
    /// itself and no origin probe will see a difference afterwards.
    ingest_wake: OnceLock<Arc<Notify>>,
    /// Raised while the branch layers' chunk exclusions are being folded on
    /// their own thread ([`InferenceState::fold_chunk_exclusions_behind`]),
    /// so a pass ending while one folds does not start a second.
    learning_exclusions: AtomicBool,
    /// Each branch layer's exclusion learner, its state kept between passes
    /// (`docs/incremental_chunk_exclusion.md`).
    exclusion_learners: Mutex<HashMap<GroupId, ExclusionLearner>>,
    /// The dialogue layer's working set (`docs/zend_working_set.md`). `None`
    /// when the schema declares none: nothing is seeded.
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
    /// The turn triggers with tools out of the grammar ([`round_triggers`]), by
    /// mode, dial and the tools taken out.
    held_stencils: ArcSwap<HeldStencils>,
    /// The Restricted stencil with every call held to `file_read` — what a
    /// file's reading decodes under, so it can call no other tool. Built once
    /// at startup beside the stencils it narrows.
    read_stencil: Arc<TriggerRegistry>,
    /// A file reading's opening call, which its first turn begins inside
    /// (`RefreshContext::read_opening`). Built once at startup beside
    /// `read_stencil`.
    read_opening: Option<OpeningCall>,
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
    /// Raised when the member leaves its clusters: every live turn stops where
    /// it stands for another member to resume (`docs/turn_handover.md` §4).
    leaving: Leaving,
    /// The conversations this member is resuming now (`crate::resume`).
    resuming: resume::Resuming,
    /// Whether the schema declares a journal (`journal::carry::install`). A
    /// workspace schema without one journals nothing.
    journaling: bool,
    /// Every dialogue's journal (`docs/zend_journal.md`).
    journals: Journals,
    /// The scopes whose shared journal run is sealed ahead of use, and what was
    /// sealed for each (`docs/shared_journal_sections.md` §7).
    prewarm: Prewarm,
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
    /// and the signal a task's waiting answer wakes on (`crate::mcp`).
    mcp: Arc<McpState>,
}

mod builders;
mod conversations;
mod core;
mod dials;
mod history;
mod inference_stream;
mod ingesting;
mod journaling;
mod kept_images;
mod lifecycle;
mod load;
mod passthrough_turns;
mod prompt_text;
mod status;
mod submission;
mod substrate_view_routes;
#[cfg(test)]
mod tests_projection;
#[cfg(test)]
mod tests_status;
#[cfg(test)]
mod tests_timeline;
#[cfg(test)]
mod tests_turn;
mod timeline_ids;
mod title_worker;
mod triggers;
mod turn_types;
mod unfinished_turns;
mod uploads;
mod working_reads;
mod workspace_tools;

pub(crate) use builders::*;
pub use conversations::*;
pub use dials::*;
use history::*;
use inference_stream::*;
pub use journaling::*;
pub use kept_images::{ImageRead, KeptImage};
pub use status::*;
pub use timeline_ids::*;
use title_worker::*;
use triggers::*;
pub use turn_types::*;
pub use unfinished_turns::*;
pub use uploads::*;
use working_reads::*;
pub use workspace_tools::*;
