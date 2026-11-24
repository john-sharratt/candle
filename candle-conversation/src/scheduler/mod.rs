//! The scheduler: the single thread that owns every GPU resource and runs the
//! continuous fair-wave loop.
//!
//! Each wave drains submitted work, runs VRAM-governor admission and relief, and
//! then fires one co-batched forward. In that forward a fast decode cursor sweeps
//! every layer once per token, while a large prefill cursor creeps through the
//! layers at a throttled rate and shares each layer's expert load with decode, so
//! decode never goes cold behind a large prefill. Sealing, cold-tail eviction and
//! the KV-pool trim run at the tail of the wave. The module is organised around
//! the requests callers send, the per-turn decode and in-flight state, the
//! admission budget that throttles prefill by bytes, and the projection that
//! rebuilds a slot's prefix from the substrate for each turn. The `Scheduler`
//! struct and its limits are defined here.
//!
//! - `budgets`: the AIMD admission budget and its pass caps.
//! - `cleanup`: disposes of finished and abandoned turns.
//! - `compression`: the decode-only pass that writes a summary's assistant half.
//! - `compression_turn`: the summary probe over a summary-tree node.
//! - `construct`: startup, the provenance fold, and substrate replay.
//! - `counters`: process-wide hot-path counters and timers.
//! - `decode_state`: per-turn decode state from first token to seal.
//! - `handle_request_now`: the SubmitTurn handler.
//! - `in_flight`: turns, views, compression jobs and sections in flight.
//! - `perform_seal`: writes a finished turn or section to the substrate.
//! - `projection_turns`: projects a turn's selection onto its slot.
//! - `provenance_capture`: per-token provenance signatures from the Q values.
//! - `recurrent`: restore and fork of recurrent state.
//! - `reproject_prepare`: the prepare phase of a reprojection.
//! - `reprojection`: continuous reprojection and its completion.
//! - `requests`: the requests callers send and their dispatch.
//! - `seal_turn`: seals a finished turn and sends its `Done`.
//! - `section_finish`: seals sections and quantizes pending ones.
//! - `sections`: restores and begins section ingests.
//! - `slots`: slot creation, freeing and detaching.
//! - `telemetry`: per-window wave statistics.
//! - `wave_flush`: emits a window's summary and phase breakdown.
//! - tests: `tests_admission`, `tests_branch_checkpoint`,
//!   `tests_ephemeral_forks`, `tests_fixtures`, `tests_forks`, `tests_pages`,
//!   `tests_recurrent_slots`, `tests_sealing`, `tests_slots`, `tests_toy_model`,
//!   `tests_turns`, `tests_waves`.

mod admission;
mod admission_split;

/// Static backstop on the ragged prefill forward's width — how many in-flight
/// prefills may coalesce into one `forward_batched`, regardless of what the
/// byte budget would allow.
///
/// The admission budget's setpoint (`Scheduler::admit_budget`) is the real
/// throttle; this is the dumb ceiling beneath it. It exists so an error in the
/// cost model (`admission::per_block_kv_bytes` and friends) costs throughput
/// rather than the daemon: a cost underestimate can only ever admit this many
/// sequences before it is capped, whatever it believes they are worth. It is
/// also what a caller that queues many turns at once sizes its window by, so
/// it never queues more than the scheduler can run.
pub const MAX_PREFILL_WIDTH: usize = 24;

/// The decode counterpart of [`MAX_PREFILL_WIDTH`], and a backstop of exactly
/// the same kind: the rate model is the real throttle, and this is the dumb
/// ceiling beneath it so an error in the cost model costs throughput rather
/// than the daemon.
///
/// Sixty-four, which is the width the aggregate benchmark runs at — the engine
/// is known to carry that many sessions, so a cap below it would refuse work
/// the hardware has already been shown to do.
pub const MAX_DECODE_WIDTH: usize = 64;

/// Rate-based wave budgeting — **compiled and tested, not yet wired.**
///
/// Admission: [`admit::fill`] is the path, and [`admit_ground::AdmitPass`] is
/// the engine's answer to its `Ground` trait. An offer joins the wave while the
/// wave goes *faster* carrying it — [`admit::rate`] models what a wave will
/// actually achieve and the rest prices and orders candidates against that
/// model rather than against free bytes.
///
/// What remains in [`admission`] is the arithmetic that survived the change: the
/// per-block and per-prefill costing the new path still prices with, and the
/// AIMD setpoint the *ingest* regulator moves.
mod admit;
mod admit_ground;
mod answered;
mod band;
mod block_guard;
mod census_stage;
mod compaction_split;
mod compaction_stall;
mod creep_slice;
mod decode;
mod detached;
mod diag;
mod warm_split;

mod ephemeral_fork;
mod expert_counters;
pub mod exported_state;
mod first_reading;
mod first_sample;
mod frame;
mod gallery_sweep;
mod gang;
mod gap_residence;
mod gap_wait;
mod graph_window;
mod guest_room;
mod held_turn;
mod high_drain;
mod hit_density;
mod idle_wait;
mod injected_piece;
mod interleave;
mod kv_gap;
#[cfg(feature = "kv-zero-check")]
pub(crate) mod kv_zero_check;
pub mod memory_report;
mod named_tool;
mod norm_warm;
mod page_cut;
mod paused_creep;
mod pending_ranges;
pub mod phase_ring;
mod phase_split;
#[cfg(test)]
mod phase_window_tests;
mod piece_identity;
mod piece_splice;
mod pinned_report;
pub mod pool;
mod pool_fetch;
mod pool_linger;
mod pool_refresh;
mod pooled_rounds;
mod prefill;
mod prefill_wait;
pub(crate) mod profile;
pub(crate) mod projection_assembler;
mod projection_identity;
mod rates_store;
mod recurrent_park;
mod relief_pacing;
pub mod relief_trace;
mod remote_arena;
mod reproject_abandon;
mod reproject_census;
mod request_laps;
mod run;
mod sample;
mod section_pack;
mod section_pages;
mod section_timing;
mod segment_rearm;
mod settle;
mod spec_chooser;
pub mod state_census;
mod stencil_wave;
mod step_census;
pub mod store_holders;
mod submit_admit;
mod sweep_forecast;
#[cfg(test)]
mod test_substrate;
mod think_gate;
mod turn_airtime;
mod turn_group;
mod turn_probe;
mod turn_seal;
pub mod turn_status;
mod urgent_hold;
mod wave_logits;
mod wave_retry;
mod wave_telemetry;
mod wave_timing;
mod whole_conversations;

use band::recorded_band;
use census_stage::Stage;
use detached::Detached;
use expert_counters::ExpertCounters;
use frame::{framed_segments, frames};
use graph_window::graph_window;
use held_turn::HeldTurn;
use phase_split::cobatched_step_shares;
use reproject_abandon::or_abandon;
use section_pages::SectionPages;
use section_timing::SectionTiming;
use turn_probe::TurnProbe;

use self::paused_creep::PausedCreep;
use self::phase_ring::{ProjectedKind, VramBands};
use self::relief_pacing::ReliefPacing;
use self::reproject_census::{Outcome as ReprojectOutcome, Part as ReprojectPart, ReprojectCensus};
use self::request_laps::RequestLaps;
use self::wave_timing::WaveTiming;
use crate::batched_sampler::{BatchedSampler, SequenceSamplingState};
use crate::config::{DecodeHealthConfig, SamplingConfig};
use crate::conversation::{is_shared_stream_name, slice_per_layer_sealed, Frame};
use crate::decode_health::DecodeHealthState;
use crate::decoded_text::{decode_turn_text, TagIds};
use crate::dry_reference::reference_flags;
use crate::error::ConversationError;
use crate::guest::Guests;
use crate::handle::{SealResult, TurnEvent, TurnResponse};
use crate::index_pages::{self, StoredPages};
use crate::persistence::cold_load::{
    preallocate_pinned_scratch, ColdLoadStager, PINNED_PREALLOC_BYTES,
};
use crate::persistence::content_hash::{section_stream_id, turn_stream_id, ContentChain};
use crate::provenance::boundaries::{BoundaryRule, BoundaryState, Step};
use crate::working_set::PoolSize;
// Only the K/V-digest request carries these, and it is a test-helper probe.
#[cfg(any(test, feature = "test-helpers"))]
use crate::persistence::content_hash::{ContentHash, ContentHasher};
use crate::persistence::elevate::{elevate_to_hot, sealed_total_bytes};
use crate::persistence::record::{SnapshotLayer, SnapshotPayload};
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::persistence::thread::PersistenceTrigger;
use crate::persistence::triggers::{PersistenceTriggers, FIRST_ATTACHMENT};
use crate::projection::event::{group_name_of, layer_name_of_group};
use crate::projection::scan_census::take as take_scan_census;
use crate::projection::Content;
use crate::projection::{
    decode_events, encode_events, summary_node_event, Builder, CompressionPrompt, Conversation,
    GroupKey, Observe, PriorBelief, ProjectionMode, ProjectionSegment, ProjectionTarget,
    ResolvedSection, ResolvedTurn, SealedKind, SectionId, SelectionState, SystemPromptItem,
    TimelineId, TurnId, TurnIndex, TurnKey,
};
use crate::projection::{
    CollectionWarm, DecodePriority, GalleryPreload, GroupId, GroupVotes, LayerId, Schema,
    SelectionRule, SubstrateRef, LINEAGE_BASE,
};
use crate::provenance::gallery_arena::keep_price::KeepPrice;
use crate::provenance::gallery_arena::{BeliefArena, GroupSet, HeldScan, Keep, ScanReply};
use crate::provenance::{
    encode_wide_sigs_with, extract_q_vector_r16, fold_fits, fold_provenance_fitted, FoldParams,
    GalleryArena, WideQSig,
};
use crate::recorded_reply::{RecordedReply, Replay};
use crate::recurrent_span::TokenSpan;
use crate::vision::{engine_ids, ImageSpan};
use crate::sequence_handle::{BlockCount, BlockRange, SequenceId};
use crate::stencil::{
    Healed, StencilDriver, StepMask, TriggerRegistry, TurnGrammar, TOOL_CALL_TREE_LABEL,
};
use crate::substrate::{ContentResolver, HotKv, ProjectionScores, ResidenceIndex, TurnPartWrite};
use crate::summary_tree::scope::Scope;
use crate::summary_tree::{
    leaf_skeleton, structural_rollup, ProbeError, SelectionDiagnostics, SummariserTrigger, TurnKind,
};
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::turn_layout::{GlueKind, KvSpan, TurnLayout, TurnSegment};
use crate::turn_text::TurnText;
use crate::{FinishReason, SubstrateReloadStatus, TurnStats};
use admit::pass_budget::{RateWidth, RefusedDepth, RefusedWidth};
use first_sample::PrefillEnd;

use candle::cuda_backend::graph::CaptureStats;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, Tensor};
use candle_nn::kv_cache::{quantize_sealed_in_place, QuantFormat, SealedSequence, WarmLiftRing};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel, ProvSignPacked, SequenceStats,
    WINDOW_DIVERGENCE_MARKER,
};
use candle_transformers::models::decode_diag;
use candle_transformers::models::delta_net::ExportedLayerState;
use candle_transformers::models::draft_depth::DraftDepth;
use candle_transformers::models::piece_key::PieceKey;
use candle_transformers::models::residual_rows::CreepResidual;
use candle_transformers::models::sealed_injection::reconcile_sealed_prefix;

use self::exported_state::{ExportedState, SharedState};
use self::norm_warm::NormWarm;
use self::projection_identity::{
    section_content_stamp, segments_identity, submit_identity, turn_start_identity,
};
pub(crate) use self::remote_arena::RemoteArena;
pub use self::settle::SettleReport;
use self::settle::SettledGround;
use self::turn_group::{GroupCaseId, TurnGroup};
use candle::engine_events::note;
use flume::{Receiver, Sender, TryRecvError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::take;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a request may hold the wave thread before it is named in the log —
/// a quarter of a typical two-second wave under load.
const SLOW_REQUEST: Duration = Duration::from_millis(250);

/// Chunks of the turn's head (the user query) that stay in every reprojection
/// probe once the trailing `max_probe_tokens` window has slid past them. The
/// query is the strongest intent signal in the belief-gallery domain, so long
/// turns keep scoring it alongside the most recent reasoning.
const QUERY_HEAD_CHUNKS: usize = 2;

/// Factor the previous turn's carried tool belief is scaled by when it seeds
/// the next turn's opening projection ([`PriorBelief::decay_scores`]). Halving
/// makes the carry a soft prior rather than a hard pin, so a topic-changed
/// turn's fresh decode-Q can overtake the prior turn's tool within a few
/// tokens instead of being suppressed for the whole opening window.
const CARRIED_BELIEF_TURN_DECAY: f32 = 0.5;

/// Minimum NEW prefill tokens per evidence tick. A tick certifies "the current
/// admission budget survives this pressure", which a handful of tiny interactive
/// turns cannot — one chunk-sized batch per tick is the floor. Small forwards
/// accumulate toward it rather than being discarded.
const EVIDENCE_MIN_PREFILL_TOKENS: u64 = 256;

/// Rolling-window size (turns) for append-only ingest prefills (the
/// `disable_reprojection` path — `code_reading` / `repo_map`): each such turn
/// attends only the system prompt + the last `CODE_READ_WINDOW_TURNS` sealed
/// turns, bounding otherwise-unbounded ingest context that would exhaust the
/// window and blow up KV VRAM (see
/// [`crate::projection::resolver::Conversation::windowed_ingest_ranges`]). These
/// turns skip projection, so without the window each would borrow the whole
/// parent and prefill against every prior turn.
/// `0` = unbounded (whole-parent borrow).
const CODE_READ_WINDOW_TURNS: usize = 8;

/// The wave window: [`WaveStats`] closes one this often, busy or idle, and
/// each closing is one telemetry sample (`phase_ring::push_wave`).
const WAVE_WINDOW: Duration = Duration::from_secs(2);

/// Runs a continuous loop:
/// 1. Drain pending submissions (non-blocking)
/// 2. If idle, block waiting for work
/// 3. Process one prefill (complete, single-mode)
/// 4. Run one decode step for all active sequences
/// 5. Clean up finished sequences, send `Done` events
pub(crate) struct Scheduler {
    /// Channel receiving work from caller threads.
    rx: Receiver<SchedulerRequest>,
    /// The model (trait object for architecture independence).
    model: Box<dyn ManagedBatchedModel + Send>,
    /// Session owning all VRAM (arenas, sequences, page tables).
    session: BatchedInferenceSession,
    /// Shared tokenizer for streaming decode.
    tokenizer: tokenizers::Tokenizer,
    /// The tokenizer's tag ids, which split a finished turn's text into what
    /// the model wrote as tags and what it wrote as text
    /// ([`TurnResponse::answer`]).
    tag_ids: TagIds,
    /// The `</think>` token id, when the tokenizer has a single token for it.
    ///
    /// **The reasoning span's end, measured rather than re-derived.** A turn's
    /// `<think>…</think>` block is located by watching this id go past during
    /// decode ([`DecodeState::think_close_at`]), not by decoding the assistant
    /// text and re-tokenising the block — a round trip whose result is only
    /// approximately the span it describes.
    ///
    /// `None` when the tokenizer spells the marker as several tokens. That is a
    /// genuine absence: the block cannot be located this way, so the turn seals
    /// with no reasoning span and is projected whole. Resolved by NAME, matching
    /// `Engine::compile_think_steering`, which gates the same way.
    think_close: Option<u32>,
    /// Tokens that end an index page wherever they appear — `<think>`,
    /// `</think>`, and the turn closer.
    ///
    /// **A property of the token stream, not of the decode state.** The boundary
    /// that this replaced was derived from [`DecodeState::think_close_at`], so it
    /// could only be found for a marker the sampler *decoded* — and a turn's
    /// `<think>…</think>` is usually **prefilled**, arriving before any
    /// `DecodeState` exists. `reasoning_split` therefore looked the state up,
    /// found nothing, and split nothing: measured at **0 splits and 6 late cuts
    /// across 822 turns**, which is why almost no turn's reasoning occupied
    /// pages of its own.
    ///
    /// Reading the ids straight off the stream removes the dependency: a marker
    /// breaks a page whether it was decoded, prefilled, or played by a steering
    /// tree. Three entries, so a linear scan beats a hash.
    ///
    /// The cut falls **after** the token, so a marker stays in the page it
    /// closes rather than heading the next one.
    page_break_tokens: Vec<u32>,
    /// How large a dialogue's pool is behind its slot
    /// (`docs/hit_driven_projection.md` §7.2 step 1), by the card's size.
    pool_size: PoolSize,
    /// EOS token ID (stops generation).
    eos_tokens: TokenBuffer,
    /// Device the model lives on.
    device: Device,
    /// Resident VRAM gallery arena for the paged belief scan (one per device,
    /// shared across conversations). `None` on non-CUDA devices — the reproject
    /// then falls back to the CPU per-file scan.
    gallery_arena: Option<Arc<GalleryArena>>,
    /// The state band's last census — the gallery's, and each substrate's
    /// group names — reused by the memory report until it is
    /// [`state_census::CENSUS_EVERY`] old.
    state_census: Option<state_census::CensusCache>,
    /// What the gallery sweep has seen and owes — see [`gallery_sweep`].
    gallery_sweep: gallery_sweep::SweepState,
    /// Active decode state per sequence ID.
    active_decodes: HashMap<SequenceId, DecodeState>,
    /// Persistent sampling state per sequence ID (survives across turns).
    /// DRY penalty needs the recent-token window to span turn boundaries.
    sampling_states: HashMap<SequenceId, SequenceSamplingState>,
    /// Prefill queue (FIFO) — newly submitted, not yet started.
    prefill_queue: VecDeque<PrefillWork>,
    /// Submitted turns whose projection is chosen but not yet on the device,
    /// offered to the admission fill with their lift priced (`submit_admit`).
    pub(super) pending_submits: VecDeque<submit_admit::PendingSubmit>,
    /// When a turn waiting long for admission was last reported
    /// (`submit_admit::Scheduler::report_waiting_submits`).
    pub(super) waiting_told: Option<Instant>,
    /// Whether a slot has released its ground since the last admission pass.
    ///
    /// **Admission opportunities are created only by completions.** Nothing new
    /// can fit that did not fit before unless something freed ground, so a pass
    /// over an engine where nothing finished re-prices every queued item to
    /// reach the answer it reached last time. Set where a slot's ground goes
    /// back — a decode cleaned up, a section sealed, a turn freed, an eviction
    /// that shed something — and cleared by the pass that acts on it.
    ///
    /// Starts `true`: nothing has completed on a fresh engine, and one that
    /// waited for a completion before its first admission would never take one.
    pub(super) settled_since_admit: bool,
    /// Whether forwards have run since the device was last settled
    /// ([`Self::settle_device`]). The idle branch settles once when it is set,
    /// so the weight side grows back into what the finished work left behind
    /// while there is nothing in flight to pace against.
    pub(super) settle_owed: bool,
    /// The partition the last settle left — what the idle branch compares the
    /// ground with to tell whether something freed since owes another
    /// (`settle::resettle_owed`). `None` before the first settle.
    pub(super) settled_ground: Option<SettledGround>,
    /// Callers waiting on that settle ([`SchedulerRequest::Settle`]), each sent
    /// its report when the idle branch runs it.
    pub(super) settle_waiters: Vec<Sender<SettleReport>>,
    /// The dialogue normalization warm-up's replayed turns, scored one per loop
    /// iteration (see [`norm_warm`]).
    norm_warm: NormWarm,
    /// When the last KV compaction pass ran, so the cheap per-wave gate can hold a
    /// floor on the interval. `None` before the first pass.
    pub(super) last_kv_compaction: Option<std::time::Instant>,
    /// The pool shape the last KV compaction pass failed to improve, which holds
    /// the gate shut until that shape changes. See [`compaction_stall`].
    kv_compaction_stall: compaction_stall::CompactionStall,
    /// What the gap quantize did this second, for the `kv_gap` line.
    kv_gap: kv_gap::KvGapLine,
    /// Forwards this wave iteration has run, of either kind.
    ///
    /// The divisor for the loop's per-forward overhead: a wave's non-forward wall clock
    /// is amortised across the forwards it carried, and that quotient is what the rate
    /// planner needs to price width correctly. See `WaveRate::observe_overhead`.
    pub(super) wave_forwards: usize,
    /// Microseconds this wave iteration spent **outside** its forwards.
    ///
    /// Every non-forward phase, not just the one that was easiest to reach. Feeding only
    /// the housekeeping left the submission drain out — 7% of the run on its own — and the
    /// planner then under-priced width in proportion to what was missing, which is the
    /// same error as omitting the term altogether, only smaller.
    pub(super) wave_overhead_us: u64,
    /// The engine's one wave throughput planner, carried across admission
    /// passes because what it learns — the effective copy rate, the decode
    /// layer time, the hit coefficient — is a property of the machine rather
    /// than of any one wave.
    ///
    /// `None` until the model can describe its weight side
    /// ([`ManagedBatchedModel::weight_plan`]), which on a streaming MoE stack is
    /// after the first classify. A dense model never reports one and never
    /// plans: admission then falls through to the width backstop alone.
    pub(super) wave_rate: Option<admit::WaveRate>,
    /// The expert cache's hit and miss counters as of the last observation.
    ///
    /// They are cumulative for the life of the process — nothing in the daemon
    /// resets them — so the planner is taught the *delta* over each interval.
    /// See `Scheduler::observe_expert_hit_rate`.
    pub(super) expert_hits_seen: usize,
    pub(super) expert_misses_seen: usize,
    /// In-flight prefills (partially advanced across loop iterations).
    /// Promoted from `prefill_queue` by `promote_new_prefills` and drained
    /// by `promote_finished_prefills_to_decodes` once their offset reaches
    /// `work.tokens.len()`.
    pub(super) active_prefills: Vec<ActivePrefill>,
    /// Section ingests waiting for admission, in arrival order — see
    /// [`QueuedSection`].
    pub(super) section_queue: VecDeque<QueuedSection>,
    /// In-flight section ingests (batched prefill-only, no decode).
    /// Populated by [`Self::begin_section`] once admission takes a
    /// [`QueuedSection`] and its slot is prepared (truncate, prefix inject,
    /// push writer chunk). Advanced by the wave step once each entry's offset
    /// reaches its token count, then finalised by
    /// [`Self::finalize_done_section_ingests`].
    pub(super) active_section_ingests: Vec<ActiveSectionIngest>,
    /// Each sealed section's per-position model state, by the section's
    /// content address — see [`section_pages`].
    pub(super) section_positional: SectionPages,
    /// Batched sampler for token generation.
    sampler: BatchedSampler,
    /// When `true`, special tokens are included in streamed text.
    show_special_tokens: bool,
    /// Decode health monitoring configuration (always present; checks are
    /// compiled only with the `decode-health` feature).
    health_config: DecodeHealthConfig,
    /// Chunk size from the main session config (used to compute write offsets).
    chunk_size: usize,
    /// Maximum tokens per prefill chunk (chunked prefill).
    /// When a submission exceeds this, it is split into multiple forward passes
    /// so intermediate activation buffers stay bounded.
    max_prefill_pass_tokens: usize,
    /// The forward width the rate model chose for the prefill cohort in flight
    /// — the rows admitted when it judged that more would not make the forward
    /// faster — and how many prefills that cohort held when it did.
    ///
    /// The turns admitted past that point join the cohort rather than widening
    /// its forward (`Ground::close_rows`), so the forwards they share are
    /// composed at this width — [`Self::prefill_pass_budget`] is bounded by it.
    /// Without it they would be composed against the deployment's whole target,
    /// which is a width the rate model has just refused.
    ///
    /// **A verdict on one cohort, not a setting.** It lapses when that cohort
    /// shrinks — fewer members share the forward, so the trade it weighed is no
    /// longer the one being made — and is replaced whenever a later pass offers
    /// rows: one that stops on the rate sets a new width, one whose rows all pay
    /// clears it. A busy daemon, where some prefill is always running, would
    /// otherwise compose every later forward at a width judged for a cohort
    /// long gone.
    rate_width: Option<RateWidth>,
    /// The widest forward the partition will take since it refused one — see
    /// [`RefusedWidth`]. [`Self::prefill_pass_budget`] is bounded by it until
    /// the ground a tier can be placed in grows ([`Self::settle_refused_width`]).
    refused_width: Option<RefusedWidth>,
    /// The deepest a decode step drafts since the partition refused a deeper
    /// one — see [`RefusedDepth`]. Bounds the model's draft ceiling until the
    /// ground a tier can be placed in grows by what the refused step was short.
    refused_depth: Option<RefusedDepth>,
    /// Per-turn view ownership: `view_id → ViewState`.
    ///
    /// Populated when [`SchedulerRequest::SubmitTurn`] creates a view
    /// internally; consumed in `cleanup_finished` when the decode for that
    /// view completes — `Session::finalize_view` fires automatically so
    /// the conversation never has to manage view ownership.  Mid-decode
    /// swaps update this map (old entry removed when its view is
    /// finalized, new entry inserted for the replacement view, with
    /// `turn_start_parent_blocks` carried across unchanged).
    turn_views: HashMap<SequenceId, ViewState>,
    /// Turn groups in flight, keyed by [`GroupCaseId::group`] — see
    /// [`Scheduler::start_turn_group`].
    turn_groups: HashMap<u64, TurnGroup>,
    /// The id the next turn group is filed under.
    next_turn_group: u64,
    /// Each conversation's belief as of its last completed turn, keyed by the
    /// conversation's parent slot. Harvested in `cleanup_finished` when a turn
    /// seals, and seeded into the NEXT turn's submit-time projection and
    /// decode state — the belief is conversation-state that evolves across
    /// turns, never resetting at a turn boundary (a tool-retry turn opens with
    /// the prior turn's committed tool already materialized, not the catalog
    /// fallback). In-memory only: a daemon restart begins from an empty belief
    /// and the first turn's reprojections rebuild it.
    carried_beliefs: HashMap<SequenceId, PriorBelief>,
    /// The co-resident models that borrow the card between waves, and the
    /// backlog they are waiting on.
    ///
    /// Polled once per pass of the loop — one atomic load, answering "no" on
    /// almost every pass — and drained between forwards when it is not empty.
    /// Shared with the engine, which is what the API submits through. A
    /// deployment with no guests configured carries an empty registry and a
    /// queue nothing ever pushes to; see [`crate::guest`].
    guests: Arc<Guests>,

    /// Pending mid-decode view swaps, queued during `batch_decode_step`
    /// (which holds shared/exclusive borrows on `active_decodes`) and
    /// drained immediately after the batch completes.  Values are
    /// `view_id`s whose decoded count crossed an `every_n_tokens` boundary
    /// this batch.
    pending_reprojections: Vec<SequenceId>,

    /// A request's first-reading scores, by the parent slot its turn is
    /// submitted again on (`first_reading`): the second reading's opening
    /// projection is scored by them, so the question's files rank beside the
    /// seeds rather than every round reading zero.
    first_readings: HashMap<SequenceId, ProjectionScores>,

    /// Per-slot `Conversation` (workspace) registry.  Registered when
    /// [`SchedulerRequest::NewSequence`] allocates the slot; removed
    /// when [`SchedulerRequest::FreeSequence`] frees it.  The
    /// [`SchedulerRequest::SubmitTurn`] handler resolves the
    /// conversation by `parent_id` so the projection step reads from
    /// the right substrate — the scheduler hosts conversations on
    /// many substrates concurrently.
    slot_conversations: HashMap<SequenceId, Conversation>,

    /// The slots and substrates detached from this engine, whose handles'
    /// requests are dropped — see [`detached`].
    detached: Detached,

    /// Per-slot projection target — `(layer, group, timeline)` pinned at
    /// [`SchedulerRequest::NewSequence`] / [`SchedulerRequest::ResumeSequence`]
    /// time and consumed by both the
    /// [`SchedulerRequest::SubmitTurn`] handler (running the projection)
    /// and the seal step (writing turns into the substrate).  Replaces
    /// the per-request `seal_target` / `projection_inputs.target`
    /// threading: callers no longer pass the target on every submit.
    slot_targets: HashMap<SequenceId, ProjectionTarget>,

    /// Slots created ephemeral ([`SchedulerRequest::NewEphemeralSequence`]) — the
    /// interactive projection probe. They project (materialize the full warm
    /// system prompt) exactly like a real turn, but resolve to `SealAction::None`
    /// so nothing is written to the substrate. At turn completion the query's
    /// warm wide-Q window is gathered off the parent slot and stashed in
    /// [`Self::ephemeral_sigs`] for the caller to retrieve via `ProbeWideSigs`.
    ephemeral_slots: std::collections::HashSet<SequenceId>,

    /// The band a slot decodes at in place of its target layer's
    /// `decode_priority`: an ephemeral fork's, named by its asker
    /// ([`SchedulerRequest::ForkEphemeralSequence`]), or the one its
    /// conversation records for itself under [`DecodePriority::METADATA_KEY`].
    /// Dropped when the slot is freed.
    slot_bands: HashMap<SequenceId, DecodePriority>,

    /// Wide-Q window gathered for each ephemeral slot's turn at completion, keyed
    /// by the parent slot. Drained by the `ProbeWideSigs` request; dropped on
    /// `FreeSequence`.
    ephemeral_sigs: HashMap<SequenceId, Vec<WideQSig>>,

    /// **Diagnostic**: per-slot record of every token that has been
    /// committed to the slot's K/V — in the exact order it landed
    /// in the kernel's view.  Updated by every write path:
    ///
    ///   - `apply_projection` appends each injected section's
    ///     tokens (looked up from substrate).
    ///   - `run_prefill` appends the prefilled tokens.
    ///   - prefill's first sampled token gets appended.
    ///   - each decode step's sampled token gets appended.
    ///   - `run_one_section_ingest_chunk` appends the section's tokens
    ///     (the entry is cleared on `FreeSequence` immediately
    ///     afterwards, so this is only observable mid-ingest).
    ///
    /// Cleared on `FreeSequence`.  When the `context-dump` cargo
    /// feature is enabled, at turn-complete the full token stream
    /// is decoded and logged at `INFO` on target
    /// `candle_conversation::scheduler::context_dump`.  With the
    /// feature off (default) the map is never written and the dump
    /// emit is compiled out — recording cost is zero in production
    /// builds.
    slot_tokens: HashMap<SequenceId, Vec<u32>>,

    /// Per-slot projection-assembler state — holds the in-flight turn's
    /// `pending_user_part` capture (reserved infrastructure; no current
    /// code path emits `NewUserMessage`).  Cleared on `FreeSequence`.
    slot_projection_state: HashMap<SequenceId, projection_assembler::SlotState>,

    /// Reusable pinned host scratch for the cold→hot HtoD leg used
    /// by `elevate_to_hot` (cuMemHostAlloc'd once, grown on demand).
    cold_load_stager: ColdLoadStager,

    /// Trigger handles for every attached substrate's persistence thread.
    /// Fired after every turn-seal so each thread runs its hot→warm→cold
    /// drain promptly instead of waiting up to 5 s on its tick; their gauges
    /// are read summed — see [`PersistenceTriggers`].
    persist_triggers: PersistenceTriggers,

    /// Every substrate attached to this engine core, slot or no slot: the
    /// holders a compaction sweep and a relief pass must visit. A substrate
    /// with no live slot still holds resident KV in its residences — on a
    /// shared engine, the tenant that is between conversations — and a pass
    /// that walked only [`Self::slot_conversations`] relocated the chunks they
    /// name without rewriting them: the sweep's `unwitnessed` count.
    attached_substrates: Vec<Conversation>,

    /// Trigger handle for the async summariser thread.  Fired after
    /// every turn-seal
    /// so the freshly-pending Normal turn is absorbed into the immutable
    /// summary forest on the next
    /// pass instead of waiting up to 250 ms
    /// for the periodic tick.  Backpressure-clearing is purely a
    /// latency optimisation — the tick alone is correct.
    summariser_trigger: SummariserTrigger,

    /// Sections that have been ingested (their native K/V is installed
    /// in `substrate.section.hot`) but haven't been quantized to the
    /// configured `compression_policy` yet.  Drained synchronously by
    /// the next `SealAction::Turn` handler, *after* the turn record
    /// completes — so the in-flight conversation build (priming +
    /// sysprompt prefill) computes against the original native section
    /// K/V, and only post-turn-boundary reads see the quantized form.
    /// Empty when no `compression_policy` is configured (the queue is
    /// effectively dead code on engines without quantize).
    pending_section_quantize: Vec<PendingSectionQuantize>,

    /// Re-used pinned host scratch for the section quantize's staging.
    /// Grows on demand; stays allocated for the scheduler's lifetime.
    quantize_pinned_scratch: Option<PinnedBuf>,

    /// The warm lift's pinned ring (`docs/warm_lift.md`): fixed, allocated at
    /// the first lift and held for the scheduler's lifetime. The lift runs on
    /// the device's **main inference stream** (`dev.cuda_stream()`), so the
    /// prefill and decode kernels after it see the lifted bytes by stream
    /// order, with no fence.
    warm_lift_ring: Option<WarmLiftRing>,

    /// Pre-tokenised role markers for the engine's dialect. Built once at
    /// engine construction and passed into the scheduler.
    ///
    /// Turns own their own boundaries — `Sequence::submit_prefill_unit` bakes
    /// the opener and closer into the turn's grid — so the assembler no longer
    /// emits anything around a `Sealed::Turn` and reads only `causal_only_glue`
    /// from here. What still reads the token forms is the scheduler itself: the
    /// compression turn, which builds its grid directly and therefore has to
    /// bake its own head and tail, and the summary probe, which frames a
    /// synthetic user→assistant exchange so the model responds rather than
    /// continuing the prompt.
    boundary_markers: projection_assembler::BoundaryMarkers,

    /// The provenance fold this process's model produces, derived once from its
    /// geometry at construction.
    ///
    /// **One source for both the bits and the stamp.** A signature is
    /// comparable only to another folded the same way, so the parameters that
    /// fold a capture and the parameters stamped onto the stored record have to
    /// be the same value — not two derivations that happen to agree. They did
    /// not agree: the stamp was derived while the fold ran on the locked
    /// 48-layer/4-head/128-dim constants, which is an identity on Qwen3-30B and
    /// silently wrong on anything else. Holding the value here is what makes
    /// them one thing.
    prov_fold: crate::provenance::FoldParams,
    /// KV layers whose Q carries provenance — the model's
    /// `provenance_capture_layers`, cached beside the fold that was derived
    /// from it.
    ///
    /// Cached together on purpose: the fold's geometry is `layers × kv_heads`,
    /// so a gather taken over a different count than the fold was built for
    /// yields different bits, not merely slower ones. One field each, read from
    /// one place, is what keeps them from drifting.
    prov_capture_layers: usize,

    /// Periodic forward-pass batch-size telemetry (diagnostic; one line / 2 s).
    wave_stats: WaveStats,

    /// In-flight compression nodes, keyed by job id. A `SubmitSummaryProbe`
    /// registers one entry holding both half-passes; each pass's decode rides
    /// the normal wave (`active_decodes`) and completes in `cleanup_finished`,
    /// which stitches the node and fires `response_tx` once both halves land.
    compression_jobs: HashMap<u64, CompressionJob>,

    /// Compressed turns whose marker-framed text is re-prefilling on the shared
    /// wave, keyed by `job_id`. Drained when the prefill completes — see
    /// [`PendingCompressionSeal`] and `complete_compression_turn`.
    pending_compression_seals: HashMap<u64, PendingCompressionSeal>,

    /// Monotonic id source for `compression_jobs`.
    next_compression_job_id: u64,

    /// AIMD congestion budget over prefill admission, in **bytes** of VRAM the
    /// inference working set may occupy — the single throttle every pressure
    /// signal moves and [`Self::promote_new_prefills`] reads. See the
    /// [`admission`] module header for why the regulated quantity is bytes and
    /// not sequence count.
    ///
    /// A wide ragged prefill forward's transient VRAM peak (MoE expert gather +
    /// per-sequence activations) isn't visible at admission time, so a fixed cap
    /// that's fine on an idle card OOMs a busy one. The budget closes
    /// multiplicatively — halving toward one quantum — whenever VRAM pressure
    /// survives an eviction pass, a forward reports device-OOM, host RAM falls
    /// below the ingest floor, or the hot→warm drain falls behind; it reopens
    /// additively, one quantum at a time, on proven out-of-memory-free
    /// throughput. Clamped at every read to what the card can actually deliver
    /// ([`admission::available_bytes`]), so the setpoint can be optimistic
    /// without the card ever being oversubscribed.
    admit_budget: u64,

    /// Cached OS memory probe for host-tier ingest backpressure, as
    /// `(checked_at, available_bytes, total_bytes)`. The warm (RAM) KV tier can
    /// fill host memory faster than warm→cold demotion drains it; when free RAM
    /// runs low, `regulate_ingest_admission` throttles ingest so the hot→warm
    /// migration always finds a staging buffer it can allocate (the host-OOM that
    /// aborted a full overnight load). `sysinfo` is a syscall, so this is
    /// refreshed at most once per `HOST_RAM_PROBE_INTERVAL`, never per wave.
    host_ram_probe: Option<(std::time::Instant, u64, u64)>,

    /// Consecutive progress observations (new prefill tokens forwarded
    /// OOM-free) while growth was blocked ONLY by the VRAM-pressure bit — the
    /// evidence streak that reopens a wedged budget. Consumed by BOTH reopen
    /// paths (the ingest regulator via [`admission::evidence_admit_grow`], and
    /// the non-ingest run-loop reopen); the phases are mutually exclusive, so
    /// one streak serves both. Reset by any budget cut.
    admit_grow_streak: usize,

    /// [`PREFILL_OK_TOKENS`] as of the last evidence check.
    admit_ok_tokens_seen: u64,

    /// When a standing-condition signal last cut the admission budget — see
    /// [`Self::cut_admit_budget_leveled`]. `None` until the first level cut.
    last_level_cut: Option<std::time::Instant>,

    /// [`PREFILL_OK_TOKENS`] as of the last promote-side pressure episode —
    /// the "forwards are still completing" evidence that distinguishes chronic
    /// nominal pressure (hold the width) from a genuine stall (halve it). See
    /// `promote_new_prefills`.
    promote_ok_tokens_seen: u64,

    /// When prefill throughput last advanced, as observed by the promote path.
    /// A shrink fires only after [`PROMOTE_STALL_GRACE`] of complete silence —
    /// promote runs many times per second while forwards take seconds, so a
    /// per-pass "no progress" reading is meaningless. `None` until the first
    /// completed forward (never throttle a pipeline that hasn't started).
    promote_last_progress: Option<std::time::Instant>,

    /// Timelines being ingested append-only (`disable_reprojection` submits, e.g.
    /// `code_reading` / `repo_map`). Their sealed turns are never re-attended
    /// until query time, so the gentle-early ladder rung
    /// (`demote_cold_ingest_if_pressured`) demotes their warm-backed hot KV to
    /// RAM at ~50% capacity — long before the near-cap eviction ladder — keeping
    /// only a small rolling hot window resident during a bulk repo ingest.
    ingest_timelines: HashSet<TimelineId>,

    /// Set while `drain_submissions` runs: `apply_projection` defers each
    /// no-deferred-user gap-fill (ingest / compression) into `deferred_glue_fires`
    /// instead of firing a single-slot forward, so they batch into ONE forward at
    /// drain end (amortising the per-forward GPU launch floor across the whole
    /// drain). See [`projection_assembler::apply_segments`].
    batch_drain_gap_fills: bool,
    deferred_glue_fires: Vec<projection_assembler::GapFillPlan>,
    /// The expert cache's tallies when last read, and their rates since,
    /// averaged — the weight side's half of the gallery's keep time
    /// (`update_gallery_keep`).
    gallery_price: KeepPrice,
    /// When the gallery keep trade's inputs were last logged (`update_gallery_keep`).
    gallery_logged_at: Option<Instant>,
    /// When the gallery last gave up the turns past its keep time.
    gallery_evicted_at: Option<Instant>,
    /// The held cohort's own claims as its fit priced them, and of them the K/V
    /// its windows write layer by layer — see `held_claims_outstanding`.
    held_fit_claims: Option<(u64, u64)>,
    /// When the learned rates were last logged (`publish_learned_rates`).
    rates_logged_at: Option<Instant>,
    /// When the learned rates were last priced (`publish_learned_rates`).
    rates_priced_at: Option<Instant>,
    /// The widths the last pricing's three trades settled at — the placed
    /// prefill's, one decode more's, a burst's — each the next one's start
    /// (`WaveRate::traded_width_near`).
    rates_width_hints: [Option<usize>; 4],
    /// Since when each idle slot has been idle (`slot_idle`), as the rates'
    /// pricing last saw — what a keep trade for idle conversations' KV will
    /// be judged on, measured first.
    idle_since: HashMap<SequenceId, Instant>,
    /// The `High` backlog's measured drain per wall second — the prefill rate
    /// a member advertises while one stands (`high_drain`).
    high_drain: high_drain::HighDrain,
    /// `TURN_CONTENT_TOKENS` as `publish_learned_rates` last read it.
    turn_content_seen: u64,
    /// Static runs the stencil has played and no wave has written yet
    /// (`stencil_wave`). A sequence holding one rides the next decode step as
    /// that run instead of as a decode row, and keeps it until a wave writes it.
    stencil_rows: Vec<stencil_wave::StencilRow>,
    /// Whether this decode step's `Low` rows ride beside an `Urgent` decode
    /// (`urgent_hold`).
    urgent_hold: urgent_hold::UrgentHold,
    /// Rows the last decode step carried — decode rows and static runs — the
    /// denominator of the tokens each row commits a step.
    step_rows: usize,
    /// Ephemeral-fork requests whose parent had a turn in flight when they
    /// arrived, answered at the turn boundary. See [`ephemeral_fork`].
    parked_forks: Vec<ephemeral_fork::ParkedFork>,
    /// Each slot's last sealed snapshot, `(timeline, turn)`, until the
    /// persistence layer reports it durable and the slot's store is parked
    /// (`recurrent_park`).
    sealed_snapshots: HashMap<SequenceId, (TimelineId, u32)>,
    /// The `(timeline, turn)` each parked slot's state was dropped at, which
    /// its own record holds (`recurrent_park`).
    parked_at: HashMap<SequenceId, (TimelineId, u32)>,

    /// Continuous-fair-wave prefill cohort (`docs/continuous_fair_waves.md`): the
    /// in-flight prefill batch's inter-layer residual, held between steps in its
    /// arena (`docs/wave_step.md` §3), the next layer it resumes at, and the
    /// sequences in the cohort. One cohort advances in lockstep — held across
    /// waves so decode sweeps every layer between the cohort's throttled layer
    /// advances, keeping the decode expert set hot. `cursor == 0` with no
    /// residual means no cohort is in flight.
    wave_prefill_residual: Option<CreepResidual>,
    wave_prefill_cursor: usize,
    /// The in-flight wave group's members in a STABLE order (dialogue prefills
    /// then section chunks). The order must not change between waves: `forward_wave`
    /// re-partitions the same inputs into the same internal order each resume, and
    /// the held residual only lines up if the input order is identical. Fixed at
    /// formation — new prefills / sections wait for the next group.
    wave_prefill_members: Vec<WaveMember>,
    /// The rows each of [`Self::wave_prefill_members`] holds in
    /// [`Self::wave_prefill_residual`], aligned with it: member `k`'s rows follow
    /// members `0..k`'s. What lets a member leave mid-sweep without the sweep
    /// being abandoned ([`Self::drop_wave_members`]).
    wave_prefill_rows: Vec<usize>,
    /// A creep set aside mid-sweep for a higher band's prompt, resumed at the
    /// same layer once nothing above its band waits (`paused_creep`).
    paused_creep: Option<PausedCreep>,
    /// The background prompts being prefilled as one gang, by sequence id, so
    /// they reach their first tokens together and decode side by side
    /// (`docs/prefill_gangs.md`). Fixed from formation until every member's
    /// prefill is done.
    prefill_gang: Vec<usize>,
    /// The steps of the request in hand, named in the log when it holds the
    /// wave thread long ([`Self::handle_request`]).
    request_laps: RequestLaps,
    /// Consecutive failures of the creep's current mid-sweep window, retried in
    /// place up to `prefill::MID_SWEEP_RETRIES`.
    wave_sweep_failures: u32,
    /// The KV the decodes will add over a creep cohort's sweep, which its
    /// width is priced against at formation (`sweep_forecast`).
    sweep_forecast: sweep_forecast::SweepForecast,
    /// When each waiting turn last moved and was last told so
    /// (`prefill_wait`).
    prefill_waits: prefill_wait::PrefillWaits,
    /// A wave step swept in segments, open on the model for its full-sweep
    /// members (`ManagedBatchedModel::begin_step`): the trunk layers its
    /// segments have committed so far. Closed by `decode_forward_cobatched`.
    wave_step_open: Option<usize>,
    /// The prefill cohort may not advance on this wave step: it already did, or
    /// decode is still owed airtime for the last step that carried it
    /// ([`Self::creep_owed`]). Set per decode step, so the cohort advances its
    /// budget of layers on every step its airtime allows — a wave is one sweep.
    wave_cohort_advanced: bool,
    /// Decode-only airtime still owed before the creep, or a section chunk, may
    /// ride a decode step again: a co-batched step owes its creep's share — its
    /// length less [`Self::decode_only_step`] — times the airtime ratio `R`,
    /// paid down by each decode-only step. Kept across quanta, so the creep
    /// rides as often as its ratio allows, not once per quantum.
    creep_owed: Duration,
    /// The band of the work that ran [`Self::creep_owed`] up — what decode is
    /// protected from by it. Work of a higher band is not held by it.
    creep_owed_band: DecodePriority,
    /// The band the last wave step's co-batched work ran at.
    wave_step_band: DecodePriority,
    /// What a decode-only step takes now — an average over the recent ones, so
    /// it follows the decode width and depth. `None` until one has run.
    decode_only_step: Option<Duration>,
    /// The recent step at each decode width, the bound every published step
    /// is held to (`step_census::WidthSteps`).
    width_steps: step_census::WidthSteps,
    /// What one row of creep through one layer costs beside decode rows — an
    /// average over the steps it rode (`creep_slice::next_layer_row_cost`), so
    /// a slice can be bounded in time. `None` until a creep has ridden one.
    creep_layer_row_cost: Option<Duration>,
    /// What one layer of creep costs whatever its rows — the same average over
    /// the steps it rode, per layer swept — the floor a slice's per-row price
    /// cannot go under (`creep_slice::time_bounded_layers`).
    creep_layer_cost: Option<Duration>,
    /// Rows times layers of creep the last wave step carried beside decode
    /// rows; zero when it carried none, or carried no decode.
    wave_step_creep_layer_rows: usize,
    /// The layers of creep the last wave step swept; zero as above.
    wave_step_creep_layers: usize,
    /// The last wave step swept a creep of short prompts whole
    /// (`creep_slice::sweeps_whole`), so its cost teaches the slice nothing.
    wave_step_short_sweep: bool,
    /// When the section gate's relief may run next (`relief_pacing`).
    section_relief_pacing: ReliefPacing,
    /// Every live reprojection's parts, summed and written once a minute.
    reproject_census: ReprojectCensus,
    /// The query×key pairs the creep attended through those layers this step.
    wave_step_creep_layer_pairs: f64,
    /// Consecutive failures of the decode-less wave step and the wait owed
    /// before the next — see `wave_retry`.
    wave_retry: wave_retry::WaveRetry,
    /// Consecutive decode steps that did not run — a refused tier, an expert
    /// stall — and the wait owed before the next; see `decode::retry_step`.
    decode_retry: wave_retry::WaveRetry,
    /// The last wave step carried rows other than decode and verify — a prefill
    /// creep, a section chunk or glue — so the profile can tell an 8-session
    /// decode step from one that also paid for ingest.
    wave_cobatched: bool,
    /// The airtime ratio `R` the last wave step's co-batched work yielded to —
    /// the ratio of the decodes that outrank it (and, for `High` work, of the
    /// `High` decodes), 1 when there are none. Decode is
    /// owed that share of a co-batched step ([`Self::creep_owed`]).
    wave_step_ratio: f64,
    /// How long the last step's creep slice could run beside decode rows, in
    /// decode-only steps (`turn_airtime::slice_steps`).
    wave_step_slice: f64,
    /// The split the last `High`-beside-`High` step derived (`turn_airtime`), for
    /// the rates line: `wave_step_ratio` is whatever band the last step ran in.
    high_airtime: Option<f64>,
    /// The creep's rows per second of its own airtime (`turn_airtime`).
    creep_airtime: turn_airtime::AirtimeRate,
    /// The mean decode run of a `High` turn segment (`turn_airtime`).
    segment_mean: turn_airtime::SegmentMean,
    /// The mean prompt of the turns this card has begun — what a turn placed
    /// here is priced at, learned from the turns it is sent rather than set.
    prompt_mean: turn_airtime::SegmentMean,
    /// The tokens a decoding row commits a step (`run_decode_until_budget`).
    tokens_per_step: turn_airtime::SegmentMean,
    /// Every step by what it carried, reported with the learned rates
    /// (`step_census`).
    step_census: step_census::StepCensus,
    /// The `High` prompt rows outstanding, as `publish_learned_rates` last
    /// counted them.
    high_backlog_rows: usize,
    /// Set once per wave after the co-batched decode wave folded the active
    /// section-ingest chunk into its full sweep (section rides decode's `[0, N)`
    /// as a prefill-group member — one shared MoE grouped GEMM per layer serves
    /// decode + section + cohort). Skips the standalone `run_one_section_ingest_chunk`
    /// pass that wave (`docs/continuous_fair_waves.md` §4).
    wave_section_advanced: bool,
    /// Set when a mid-wave admission drain (inside the time-sliced decode quantum)
    /// observes a shutdown / channel-disconnect. The request is consumed there, so
    /// the top-of-loop drain can't re-read it; the main loop breaks on this flag
    /// before it would otherwise block on the idle recv.
    shutdown_requested: bool,

    /// Each section's symbolic `debug_name`, by its stream id, so the promote
    /// tracker's name lookup is O(1) after the first sighting (sections are
    /// stable). By stream id rather than `SectionId`, which is its schema's
    /// own and names a different section in each tenant's.
    section_name_cache: HashMap<StreamId, String>,

    /// Per-slot live receiver for a compression pass's private event channel.
    /// A compression decode's `DecodeState::event_tx` reports through the job,
    /// not to a caller, so its `TurnEvent`s drain into this sink. Holding the
    /// receiver alive keeps the per-token sends from failing (a failed send
    /// marks a decode finished after one token). Removed when the pass's slot
    /// is freed.
    compression_event_sinks: HashMap<SequenceId, Receiver<TurnEvent>>,

    /// Per-timeline projection `Builder`, captured from each `SubmitTurn`'s
    /// `projection_inputs`. The summariser's compression probes carry only a
    /// `timeline`, so this is how `handle_summary_probe` reaches the target
    /// layer's `summary` block (compression prompts + decode cap).
    timeline_projections: HashMap<TimelineId, Arc<Builder>>,

    /// Fallback projection `Builder` for `handle_summary_probe` when a timeline
    /// isn't in `timeline_projections` — the common case for timelines reloaded
    /// from the substrate on restart (whose builders were only ever registered in
    /// a prior run's in-memory map). The projection **schema** is workspace-global
    /// (one YAML: the layer set and each layer's `summary` block are identical
    /// across every conversation), so any registered builder resolves the target
    /// layer's compression config. Set from the first registration; without it a
    /// reloaded timeline's reconcile would fail "no projection/summary" and disarm
    /// that timeline's summary tree forever.
    workspace_projection: Option<Arc<Builder>>,
}

// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// Scheduler unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
//
// The `DummyModel` is a minimal `ManagedBatchedModel` that returns dummy CPU
// tensors so no GPU is needed. `BatchedInferenceSession` is constructed on
// `Device::Cpu` with tiny arena dimensions so these tests run without CUDA.

#[cfg(test)]
mod tests {
    // Expected token totals are written as the sum of the per-item terms they
    // came from, including the zero ones, so the arithmetic under test is
    // visible in the assertion.
    #![allow(clippy::identity_op)]

    use super::*;

    use candle::{DType, Tensor};

    use std::sync::Mutex;

    use candle_transformers::models::batched_inference::{
        BatchedInferenceSession, ManagedBatchedModel, WaveResult,
    };

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use std::time::Duration;

    pub(super) use super::tests_fixtures::tests::*;

    pub(super) use super::tests_toy_model::tests::*;

    pub(super) use super::tests_admission::tests::*;

    pub(super) use super::tests_pages::tests::*;

    pub(super) use super::tests_slots::tests::*;

    pub(super) mod creep_window_progress;

    pub(super) mod faulted_turn;

    pub(super) mod idle_telemetry;

    pub(super) mod positional_ledger;

    pub(super) mod refused_decode_tier;

    pub(super) mod reproject_abandoned;

    pub(super) mod section_page_refused;

    pub(super) mod speculation_exits;

    pub(super) mod speculation_in_a_grammar;

    pub(super) mod splice_rebuild;

    pub(super) mod urgent_hold_steps;

    pub(super) mod whole_turns;
}

mod counters;
mod provenance_capture;
mod recurrent;
mod requests;
#[cfg(test)]
use recurrent::*;
mod decode_state;
mod in_flight;
mod projection_turns;
mod reproject_prepare;
mod reprojection;
mod telemetry;
mod wave_flush;
#[cfg(test)]
use wave_flush::*;
mod budgets;
mod cleanup;
mod compression;
mod compression_turn;
mod construct;
mod handle_request_now;
mod perform_seal;
mod seal_turn;
mod section_finish;
mod sections;
mod slots;
#[cfg(test)]
mod tests_admission;
#[cfg(test)]
mod tests_branch_checkpoint;
#[cfg(test)]
mod tests_ephemeral_forks;
#[cfg(test)]
mod tests_fixtures;
#[cfg(test)]
mod tests_forks;
#[cfg(test)]
mod tests_pages;
#[cfg(test)]
mod tests_recurrent_slots;
#[cfg(test)]
mod tests_sealing;
#[cfg(test)]
mod tests_slots;
#[cfg(test)]
mod tests_toy_model;
#[cfg(test)]
mod tests_turns;
#[cfg(test)]
mod tests_waves;

pub use counters::*;
pub(super) use decode_state::*;
pub(crate) use in_flight::*;
pub(crate) use projection_turns::*;
use provenance_capture::*;
pub(crate) use reprojection::*;
pub(crate) use requests::*;
use seal_turn::*;
use telemetry::*;
pub(crate) use think_gate::DecodeReprojection;
