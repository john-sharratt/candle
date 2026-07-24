//! The scheduler: single thread that owns all GPU resources.
//!
//! Runs a continuous loop alternating between prefill and decode.
//! Phase 1 uses single-mode prefill (no small/large split).
mod decode;
#[cfg(feature = "kv-zero-check")]
pub(crate) mod kv_zero_check;
pub mod phase_ring;
mod prefill;
pub(crate) mod profile;
pub(crate) mod projection_assembler;
mod run;
mod sample;

use crate::batched_sampler::{BatchedSampler, SequenceSamplingState};
use crate::config::{DecodeHealthConfig, SamplingConfig};
use crate::conversation::slice_per_layer_sealed;
use crate::decode_health::DecodeHealthState;
use crate::error::ConversationError;
use crate::handle::{SealResult, TurnEvent, TurnResponse};
use crate::persistence::cold_load::{
    preallocate_pinned_scratch, ColdLoadStager, PINNED_PREALLOC_BYTES,
};
use crate::persistence::content_hash::{section_stream_id, turn_stream_id, ContentChain};
use crate::persistence::elevate::{elevate_to_hot, sealed_total_bytes};
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::persistence::thread::PersistenceTrigger;
use crate::projection::event::{group_name_of, layer_name_of_group};
use crate::projection::Content;
use crate::projection::{
    encode_events, summary_node_event, Builder, CompressionPrompt, Conversation, GeneratedIdentity,
    GroupKey, OptionalState, PriorBelief, ProjectionMode, ProjectionSegment, ProjectionTarget,
    ResolvedSection, ResolvedTurn, SealedKind, SectionId, SelectionState, SystemPromptItem,
    TimelineId, TurnId, TurnIndex, TurnKey, NO_THINK_SELECTOR,
};
use crate::provenance::{encode_wide_sigs, extract_q_vector_r16, fold_provenance, WideQSig};
use crate::sequence_handle::{BlockCount, BlockRange, SequenceId};
use crate::stencil::{Healed, StencilDriver, StepMask, TriggerRegistry, TOOL_CALL_TREE_LABEL};
use crate::substrate::{ResidenceIndex, TurnPartWrite};
use crate::summary_tree::scope::Scope;
use crate::summary_tree::{
    leaf_skeleton, structural_rollup, ProbeError, SelectionDiagnostics, SummariserTrigger, TurnKind,
};
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::turn_layout::{GlueKind, KvSpan, TurnLayout, TurnSegment};
use crate::{SubstrateReloadStatus, TurnStats};

use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, IndexOp, Tensor};
use candle_nn::kv_cache::{quantize_sealed_in_place, QuantFormat, SealedSequence};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel, ProvSignPacked,
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ————————————————————————————————————————————————————————————————————————————
// Scheduler request types (sent from caller threads)
// ————————————————————————————————————————————————————————————————————————————

/// Requests sent from caller threads to the scheduler.
pub(crate) enum SchedulerRequest {
    /// Allocate a fresh empty GPU slot bound to a conversation
    /// (workspace) and pinned to the given projection [`ProjectionTarget`].
    ///
    /// The target — `(layer, group, timeline)` — is recorded in the
    /// scheduler's `slot_targets` map and consumed by both the
    /// [`Self::SubmitTurn`] handler (when running the projection) and
    /// the seal step (when writing the new turn into the substrate).
    /// Callers must mint the timeline up-front via
    /// [`Conversation::mint_timeline`] so the
    /// substrate's registry is in sync before the first `SubmitTurn`.
    ///
    /// No prefill; system content is pinned in the substrate ahead of
    /// time as a section, and `apply_projection` materialises it onto
    /// the slot at the first `SubmitTurn`.
    NewSequence {
        conversation: Conversation,
        /// `Some` when the slot is for projection-driven turns or
        /// section ingestion (the seal step writes into
        /// `target.timeline`).  `None` for raw paths (RULER eval,
        /// summarisation) that allocate a slot purely as a GPU scratch
        /// resource and never touch the substrate.
        target: Option<ProjectionTarget>,
        response_tx: Sender<Result<SequenceId, ConversationError>>,
    },

    /// Allocate a slot for the interactive projection PROBE. It is bound to
    /// `target` so `apply_projection` materializes the full warm system prompt
    /// exactly like a real turn — but the slot is marked ephemeral, so its turn
    /// resolves to `SealAction::None` (nothing is written to the substrate) and
    /// its query wide-Q window is gathered at completion and stashed for a
    /// following `ProbeWideSigs` to drain. Used by `Sequence::probe`.
    NewEphemeralSequence {
        conversation: Conversation,
        target: ProjectionTarget,
        response_tx: Sender<Result<SequenceId, ConversationError>>,
    },

    /// Allocate a fresh GPU slot bound to an **existing** timeline.
    ///
    /// The substrate is expected to already have `timeline` registered
    /// (typically from a previous session restored via
    /// [`Conversation::open`]).  The handler looks
    /// up `(layer, group)` from the substrate's registry to construct
    /// the slot's `ProjectionTarget`, then proceeds like
    /// [`Self::NewSequence`].
    ///
    /// Returns `Err(...)` if the timeline is not registered.
    #[allow(dead_code)] // public scheduler API; used by Phase 2 resume callers
    ResumeSequence {
        conversation: Conversation,
        timeline: TimelineId,
        response_tx: Sender<Result<SequenceId, ConversationError>>,
    },

    /// Submit a turn for prefill + decode.
    ///
    /// The scheduler owns the turn's lifecycle:
    ///
    /// 1. Reset `parent_id` to empty.
    /// 2. Run the projection: when `projection_inputs` is `Some`,
    ///    call `projection.project(target, &substrate)`,
    ///    prepend the sequence's pinned
    ///    `system_section_id`, and translate the result to a list
    ///    of substrate sections + turns to inject.  Each unit's
    ///    `Arc<Vec<SealedSequence>>` is fetched from the substrate
    ///    and run through the per-sequence upload cache (CPU→GPU
    ///    materialisation, deduped against currently-cached
    ///    chunks).  The resulting GPU sealed chunks are injected
    ///    onto `parent_id` in order.  When `projection_inputs` is
    ///    `None` (raw RULER / summarisation paths), the parent is
    ///    just left empty.  The substrate IS the turn's parent —
    ///    `parent_id` is a GPU scratch slot the projection
    ///    materialises into.
    /// 3. Carve a view borrowing the full parent.
    /// 4. Queue prefill of `prefill_tokens` on the view.
    /// 5. Decode until EOS or `max_decode_tokens`.
    /// 6. Auto-finalize the view (transferring newly-written blocks
    ///    back to the parent) when decode completes — `Done` event
    ///    reaches the caller after the finalize fires.
    ///
    /// Mid-decode view swaps (continuous re-projection at chunk
    /// boundaries) happen entirely inside the scheduler — the caller
    /// never sees the view ids change.  The caller just streams events
    /// from `event_tx`.
    SubmitTurn {
        sequence_id: SequenceId,
        /// Projection inputs.  When `Some`, the scheduler runs
        /// `projection.project(target, &substrate)` at handler
        /// entry, prepends `system_section_id`, and
        /// materialises the resulting sections + turns onto the
        /// parent.  `None` skips the projection step entirely —
        /// used by the bare RULER eval and summarisation paths
        /// that just want a fresh parent slot.
        projection_inputs: Option<ProjectionInputs>,
        prefill_tokens: TokenBuffer,
        /// The formatted text being prefilled (emitted as TurnEvent::Prefill).
        prefill_text: String,
        /// Raw user message string — passed through verbatim to the
        /// substrate's `TurnPart::user_text` at seal time so the
        /// sidebar reload path renders it without re-tokenising.
        /// Empty for non-turn paths (RULER, summarisation) that
        /// have no user message.
        user_text: String,
        /// Gather-scope tags for this turn (e.g. `["tool"]` on calibration
        /// turns). Threaded to the turn's `TurnDecl` at seal so a projection
        /// policy's `tags:` filter can scope its provenance gallery.
        tags: Vec<String>,
        /// Content boundaries that frame the user-message body and the
        /// assistant-response body inside the sealed grid — computed CPU-side
        /// at submit time (the tokenizer lives on the conversation handle).
        /// Used at seal time to build the turn's [`TurnLayout`] so the
        /// compressor can window content-only halves.
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        /// Thinking suppressed at submit (the `/no_think` dial) — recorded into
        /// the turn's [`TurnLayout`] at seal so prior turns re-render their
        /// switch.
        no_think: bool,
        /// Marker-delimited projection points — token offsets into `prefill_tokens`
        /// where a staged calibration prefill fires a projection. The prefill wave
        /// stops its per-pass advance on each offset and emits a `ProjectionEvent`,
        /// reproducing the per-segment projection sequence a real decode produced
        /// (`docs/tool_provenance_distillation.md`). Empty for every normal prefill
        /// (one projection, single forward). The last offset is the trajectory end.
        projection_offsets: Vec<u32>,
        /// The assistant half supplied by a PREFILL turn (e.g. repo_map /
        /// code_reading ingest), stored verbatim as `TurnPart::assistant_text`
        /// at seal time. A prefill never decodes, so the seal's decoded `text`
        /// is empty; this carries the real content. Empty for decode turns,
        /// where the seal uses the decoded text instead.
        prefill_assistant_text: String,
        /// Trailing structural tokens written into the slot **after**
        /// decode finishes, before the seal.  The model didn't emit
        /// these — the scheduler appends them as if they were part of
        /// the assistant's emission so the turn's pinned KV closes
        /// out its own structural brackets (e.g. ChatML's `\n` after
        /// `<|im_end|>`).  Empty for paths that don't need a closing
        /// tail (RULER eval, summarisation).
        post_decode_tokens: TokenBuffer,
        max_decode_tokens: usize,
        sampling: SamplingConfig,
        event_tx: Sender<TurnEvent>,
        /// Optional continuous-re-projection policy.  When `Some`, the
        /// scheduler re-runs provenance + projection mid-decode and swaps the
        /// view's borrowed ranges in place; see [`ReprojectionPolicy`]
        /// for the full contract.  `None` skips re-projection entirely
        /// (used by single-shot paths like RULER eval and summarisation).
        reprojection: Option<ReprojectionPolicy>,
        /// When `true`, skip the per-turn projection rebuild
        /// (`apply_projection` reset + re-project) as long as the slot
        /// already holds content, and append the prefill onto the
        /// cumulative slot instead.  The `seal_action` is unaffected, so
        /// the turn still seals into the substrate.  The system prompt is
        /// seeded by PrimingProjection at conversation creation (so the
        /// slot is already non-empty and this stays true from turn 1);
        /// see the handler for the gated-section trade-off.  Used by
        /// append-only utility ingests (e.g. `code_reading`) where
        /// re-projecting the whole trunk every turn is both unnecessary
        /// and O(n²) — see `zend::code_read`.
        disable_reprojection: bool,
        /// Tool-call stencils that may fire during this turn's decode, keyed by
        /// their trigger token (e.g. `<tool_call>`).  An empty registry means no
        /// constrained decoding — the turn free-decodes.
        triggers: Arc<TriggerRegistry>,
    },

    /// Free a sequence slot.
    FreeSequence { sequence_id: SequenceId },

    /// Reset a sequence's KV cache back to the empty state (offset 0).
    ///
    /// The sequence slot is reused (same `sequence_id`). All KV data is cleared
    /// so the next prefill starts from scratch.
    ResetSequence {
        sequence_id: SequenceId,
        response_tx: Sender<Result<(), ConversationError>>,
    },

    /// Ingest a substrate section: fresh slot in, sealed section out.
    ///
    /// Synchronously prefills `tokens` into `sequence_id` (which must
    /// be a fresh empty slot — typically a fork allocated via
    /// [`Self::NewSequence`]), seals the slot, writes the result into
    /// the substrate as `section_id` via `set_section_data` +
    /// `set_section_sealed`, and returns the [`SealResult`] on
    /// `response_tx`.
    ///
    /// Distinct from [`Self::SubmitTurn`] — section ingestion has no
    /// view, no decode, no continuous reprojection, no event stream.
    /// One forward pass, one seal, one substrate write.
    IngestSection {
        sequence_id: SequenceId,
        section_id: SectionId,
        /// Substrate-pinned sections to Arc-clone onto the scratch slot
        /// **before** prefilling `tokens` — the cumulative-ingest
        /// prefix.  Each prefix section's `SealedSequence` is fetched
        /// from substrate and injected via `inject_sealed_at_tail`
        /// (pure metadata clone, no DMA).  After all prefix sections
        /// are injected, the handler force-pushes a fresh empty
        /// writer chunk so the upcoming prefill writes into a
        /// writer-owned chunk instead of extending the prefix's
        /// shared partial tail.
        ///
        /// Empty `Vec` ⇒ isolated ingest (no prefix), identical to
        /// the original section-ingest flow.
        prefix_section_ids: Vec<SectionId>,
        tokens: TokenBuffer,
        /// Content-addressed `(prefix_hash, section_hash)` for this
        /// section, computed by the caller via a `ContentChain` walked
        /// alongside the schema items.  The scheduler derives the
        /// persistence stream id from this and uses it to declare the
        /// SectionDecl + write Tokens/Signatures records at seal time
        /// — see `SealAction::Section`.
        address: ContentAddress,
        /// Section's symbolic name (schema item id or tool name) used
        /// purely as diagnostic metadata on the SectionDecl record.
        debug_name: String,
        /// `true` when this section is a member of a schema
        /// `Collection` (e.g. one tool inside a tool catalog) — its
        /// K/V can absorb more aggressive quantization because the
        /// projection's top-k selection downsamples collection
        /// members at every turn, so per-member precision matters
        /// less than for boundary sections (role markers, opening /
        /// closing tags) which the slot always carries.  Drives
        /// section-quantize policy selection at seal time.
        in_collection: bool,
        response_tx: Sender<Result<SealResult, ConversationError>>,
    },

    /// Recover a previously-persisted section directly from the redo
    /// log instead of running a fresh prefill.  Used when an ingest
    /// caller has computed a section's content-addressed stream id
    /// and confirmed that the persistence manifest has durable chunks
    /// for it.  The scheduler cold-loads the chunks into hot VRAM via
    /// the same pipeline used for turn cold-loads, restores the
    /// section into the substrate (`SectionEntryData` + residence
    /// with both hot and cold installed), and replies with `Ok(())`.
    ///
    /// On any failure the scheduler falls back to a normal
    /// `IngestSection` would be issued by the caller — this request
    /// just reports `Err`.
    RestoreSection {
        /// Workspace conversation the section lands in.  Sections are
        /// shared substrate state — every conversation in the
        /// workspace sees the same map — but the handle is needed so
        /// the scheduler can call `restore_section` + `recover_*` on
        /// the right `Arc<RwLock<Substrate>>`.
        conversation: Conversation,
        section_id: SectionId,
        stream_id: StreamId,
        address: ContentAddress,
        chunks_per_layer: usize,
        /// Pre-tokenised section content — the same byte sequence
        /// the original prefill used.  Reused verbatim so we don't
        /// need to read the `Tokens` record back from disk just to
        /// repopulate `SectionEntryData::tokens`.
        tokens: TokenBuffer,
        response_tx: Sender<Result<(), ConversationError>>,
    },

    /// Pre-warm a freshly-allocated slot by injecting the static system-prompt
    /// sections before the user submits the first turn.
    ///
    /// The scheduler calls [`Scheduler::apply_projection`] with the supplied
    /// section IDs and no history turns.  After this returns the slot's offset
    /// is non-zero, so the first [`Self::SubmitTurn`]'s `apply_projection`
    /// call sees `slot_already_populated = true` and skips the injection
    /// entirely — eliminating the injection latency from the first turn's
    /// critical path.
    ///
    /// Safe to call only once, immediately after slot creation and after all
    /// section ingestion in the `new_with_schema` flow has completed.
    PrimingProjection {
        sequence_id: SequenceId,
        /// Sections to inject, in the exact order `apply_projection` would
        /// use at first-turn time (declaration order from the schema loop,
        /// including collection members).
        section_ids: Vec<SectionId>,
        response_tx: Sender<Result<(), ConversationError>>,
    },

    /// Quantize + offload the collection-member sections sealed so far, freeing
    /// their VRAM mid-build.  Sent by the section-ingest loop after each batch
    /// of prefix-transparent members (e.g. one `no_think` branch of the tool
    /// catalog) so the native catalog never piles up past one batch.  The
    /// handler quantizes the pending members, then blocks on a persistence pass
    /// so their cold copies land and `install_cold` frees the hot VRAM before
    /// the next batch prefills.
    OffloadCollectionMembers {
        conversation: Conversation,
        response_tx: Sender<Result<(), ConversationError>>,
    },

    /// Extract raw K and Q float vectors from the KV cache for a list of
    /// layer indices, synchronously on the scheduler thread.
    ///
    /// Returns one entry per requested layer index: `(layer_idx,
    /// Vec<(block_idx, k_flat, v_flat, q_flat)>)` where k/v/q layouts match
    /// `dump_r16_kv_for_provenance`.  Used by data-generation tools to capture
    /// raw KVQ data for offline signature-strategy experimentation.
    ///
    /// Call this after `finish_turn` and before dropping the sequence —
    /// the parent slot's KV cache remains live in that window.
    ExtractRawKvq {
        sequence_id: SequenceId,
        /// Layer indices to extract.  May include duplicates or be out of order.
        layer_indices: Vec<usize>,
        /// Block range `(lo, hi)` — only blocks `[lo, hi)` are extracted.
        /// Pass `None` to extract all blocks for the sequence.
        block_range: Option<(usize, usize)>,
        response_tx: Sender<
            Result<Vec<(usize, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>)>, ConversationError>,
        >,
    },

    /// Drain the query wide-Q window that an ephemeral probe slot
    /// ([`Self::NewEphemeralSequence`]) gathered at its turn's completion — the
    /// interactive projection probe (`POST /v1/substrate/project`). Fire after the
    /// probe turn's `Done`; returns the belief-domain [`WideQSig`] window
    /// `score_beliefs` consumes (empty if the turn produced none). Nothing was
    /// written to the substrate.
    ProbeWideSigs {
        sequence_id: SequenceId,
        response_tx: Sender<Result<Vec<WideQSig>, ConversationError>>,
    },

    /// Run a §6 summary probe over a list of child substrate turns.
    ///
    /// The scheduler builds a probe slot per half — synthetic "compressor"
    /// system section + the children's content + the compression instruction —
    /// decodes a faithful, much shorter rewrite, re-prefills it into a clean
    /// turn, and seals it into the substrate as `kind = SummaryOfTurns` (run of
    /// Normal turns) or `kind = SummaryOfSummaries` (`children.len() == 2`).
    /// Replies with the new substrate `TurnIndex`.
    ///
    /// The compression prompts come from the target layer's `summary`
    /// block in the projection schema; the scheduler reaches it via the
    /// per-timeline `Builder` it captured on `SubmitTurn`. The system-prompt
    /// framing is sealed once per conversation and prefixed to every probe
    /// for free (cached K/V).
    SubmitSummaryProbe {
        timeline: TimelineId,
        kind: TurnKind,
        /// Structural tree children recorded on the sealed node, and the turns
        /// whose two halves the compression pass reads.
        children: Vec<TurnIndex>,
        /// The node's tree height — drives the structural roll-up's depth prune.
        height: u8,
        /// `Ok(turn_index)` on success — the sealed substrate `TurnIndex` of the
        /// new compressed turn.  `Err(ProbeError::Soft)` on a transient failure
        /// (GPU contention, snapshot/record I/O) the summariser retries;
        /// `Err(ProbeError::Permanent)` when retrying is futile (an empty
        /// compression half under argmax) — the summariser gives up.
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    },

    /// Ingest one code scope of a file, prefilled in parallel with its siblings.
    ///
    /// Each scope cold-prefills on its own scratch slot so a file's N scopes (and
    /// scopes across concurrently-ingesting files) batch on the shared prefill
    /// wave instead of prefilling one-at-a-time on a single slot. The scheduler
    /// buffers each scope's snapshotted K/V by `(timeline, scope_index)`; when all
    /// `scope_total` scopes of a file have landed it records them into the file
    /// timeline in scope order and answers each `response_tx` with the recorded
    /// `TurnIndex`. Fairness across files and the scratch-slot bound are owned by
    /// `pump_scope_prefills`, so the client submits every scope up-front without
    /// managing backpressure itself.
    PrefillScope {
        /// File timeline the scope records into. The scheduler resolves the
        /// owning conversation + projection target from it.
        timeline: TimelineId,
        /// This timeline's projection Builder, registered in
        /// `timeline_projections` so the summariser's compression probes can read
        /// the target layer's `summary`. `SubmitTurn` captures this for dialogue
        /// turns; the parallel scope path must too, or code_read turns can never be
        /// summarised (the probe fails "no projection/summary for this timeline").
        projection: Arc<Builder>,
        /// Position within the file (record order).
        scope_index: u32,
        /// Total scopes in the file — the batch's flush trigger. Every scope of a
        /// file passes the same value.
        scope_total: u32,
        /// Cold-prefill grid tokens: `[/no_think][user][user_end][assistant_start][assistant]`.
        tokens: TokenBuffer,
        /// User-message body span within `tokens` (for the sealed turn's layout).
        user_content_start: u32,
        user_content_end: u32,
        /// Assistant content start within `tokens` — the prefix before it seals as
        /// assistant content.
        assistant_content_start: u32,
        /// Verbatim user / assistant text for the turn layout (sidebar + compressor
        /// read these without re-tokenising).
        user_text: String,
        assistant_text: String,
        /// Gather-scope tags applied to each recorded scope turn's `TurnDecl`
        /// (e.g. `["code", <path>]`) so tag-scoped provenance galleries admit it,
        /// matching the serial `insert_turn_staged` path.
        tags: Vec<String>,
        /// `Ok(turn_index)` once the scope is recorded into the file timeline;
        /// `Err` on snapshot / record failure or a lost timeline.
        response_tx: Sender<Result<TurnIndex, ConversationError>>,
        /// Fired with the scope's token count the moment it lands on the wave
        /// (in `complete_scope_ingest`), so the ingest can report per-scope
        /// progress live — the parallel batch would otherwise only surface it
        /// once every scope of the file flushed. Every scope of a file carries
        /// the same callback; the batch keeps the first.
        on_prefilled: ScopeProgressFn,
    },

    /// Re-run substrate reconstruction on the scheduler thread — used after a
    /// compaction rewrites the redo log, so the scheduler-side view (KV residence
    /// + offsets) is rebuilt from the new log. Marks `status` finished when done.
    ReconstructSubstrate {
        conversation: Conversation,
        status: Arc<SubstrateReloadStatus>,
    },

    /// Reclaim VRAM by demoting the hot K/V of specific (already-sealed,
    /// retired) timelines to the warm tier, keeping their warm copy. Used by
    /// the calibration phase to hold VRAM flat: each throwaway case's K/V is
    /// dropped as it retires rather than accumulating hot until the next phase's
    /// first prefill hits an exhausted card. Only already-warm-backed turns are
    /// demoted; any not-yet-warm are skipped (a hot→warm flush, when needed, is
    /// done caller-side before this request — see [`ConversationEngine::
    /// demote_timelines_hot`] — so it can't stall this thread). Runs on the
    /// scheduler thread so the GPU-pool free-list mutation stays single-owner.
    /// Replies with the number of turn residences demoted.
    DemoteTimelinesHot {
        conversation: Conversation,
        timelines: Vec<TimelineId>,
        response_tx: Sender<Result<usize, ConversationError>>,
    },

    /// Shut down the scheduler.
    Shutdown,
}

/// Per-scope progress callback for [`SchedulerRequest::PrefillScope`], invoked
/// with a scope's token count as it lands. `Arc<dyn Fn>` so it's `'static` +
/// `Send` for the request channel and cheap to clone across a file's scopes.
pub type ScopeProgressFn = Arc<dyn Fn(usize) + Send + Sync>;

/// Hot-path timing instrumentation for the scheduler.
///
/// Two ergonomic shapes for the same underlying mechanism — both emit
/// one `tracing::trace!` event with elapsed-ms.  Enable observation
/// with `RUST_LOG=candle_conversation::scheduler::timing=trace`.
///
/// 1. RAII guard for "whole-scope" timing — drops on scope exit:
///    ```ignore
///    let _t = PhaseTimer::new("phase_name");
///    // ... work ...                              // event emitted here
///    ```
///
/// 2. Explicit start + finish for sub-phase timing inside one function
///    that needs intermediate values out of the timed block:
///    ```ignore
///    let t = Instant::now();
///    // ... work ...
///    record_phase(t, "phase_name");
///    ```
///
/// Cost when the target is filtered out is one `Instant::now()` and
/// one short `tracing::trace!` call (which short-circuits before
/// formatting when the subscriber rejects the event).
struct PhaseTimer {
    phase: &'static str,
    start: Instant,
}

impl PhaseTimer {
    #[inline]
    fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: Instant::now(),
        }
    }
}

impl Drop for PhaseTimer {
    #[inline]
    fn drop(&mut self) {
        record_phase(self.start, self.phase);
    }
}

/// Per-wave accumulators (µs) splitting the scope-ingest provenance sig cost into
/// its three stages, logged in the wave phase breakdown: `resolve` (host-side
/// per-layer Q-pointer resolution), `kernel` (HtoD + sign-pack launch + D2H), and
/// `assemble` (host XOR-fold of the packed bits). Reset each wave in `flush`.
static PROV_RESOLVE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROV_KERNEL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROV_ASSEMBLE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Scheduler-thread time spent blocked on a deliberate GPU/persistence **wait** —
/// `device.synchronize()` (draining the GPU queue) and `flush_blocking` (waiting
/// on the persistence thread's hot→warm drain). Accumulated across a wave and
/// swapped into the Sync phase in `flush`, so a multi-second stall shows as its own
/// band instead of hiding inside decode/prefill or the opaque "blocked" remainder.
static WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Persistence-maintenance stall (µs): a segment compaction on the persistence
/// thread holds the persistence lock while the scheduler thread's seal writes
/// block on it. Kept SEPARATE from [`WAIT_US`] because it is a pure off-thread
/// *blocked* wait: it must carve only from the `blocked` remainder, never spill
/// into decode/prefill/section the way an on-thread GPU sync legitimately can.
static MAINT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reprojection sub-phase timers (µs), swapped into the projection-decomposition
/// panel each wave: `scan` = probe-extract + belief scan (provenance re-selection),
/// `glue` = the shared gap-fill forward (boundary-seam tokens), `layout` = view
/// project + swap. The remainder of `reproj_ms` is "other".
static REPROJ_SCAN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPROJ_GLUE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPROJ_LAYOUT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `true` while `drain_submissions` is running, so the projection-assembler
/// sub-timers (which run in BOTH the drain and reproject paths) attribute their
/// cost to the drain buckets only when it's actually a submission drain.
static IN_DRAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Drain (submission-projection) sub-phase timers (µs) + prefilled token count,
/// accumulated only while [`IN_DRAIN`]: `elevate` = sealed-prefix inject / warm→hot
/// lift, `prefill` = the turn-content forward (its tokens land in `DRAIN_PREFILL_TOKENS`
/// and are re-attributed to the Prefill phase), `glue` = the submit-time gap-fill.
static DRAIN_ELEVATE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAIN_PREFILL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAIN_GLUE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAIN_PREFILL_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Add `us` to a drain sub-timer, but only during a submission drain (so the
/// shared assembler helpers don't miscount reproject-path work). Used by
/// `projection_assembler`.
pub(super) fn drain_add_us(atom: &std::sync::atomic::AtomicU64, us: u64) {
    if IN_DRAIN.load(std::sync::atomic::Ordering::Relaxed) {
        atom.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `device.synchronize()`, timed into [`WAIT_US`]. Use at every deliberate GPU
/// drain on the scheduler thread so the wait surfaces as the Sync phase.
fn timed_synchronize(device: &Device) {
    let t = Instant::now();
    let _ = device.synchronize();
    WAIT_US.fetch_add(
        t.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Record a persistence-side stall into [`MAINT_US`] from another module (the
/// segment-compaction I/O in `projection::resolver` holds the persistence lock
/// while the scheduler thread's seal writes block on it — an off-thread wait the
/// scheduler can't time itself). Surfaces as the Sync phase, but carved
/// **blocked-only** in `push_phase_window`: the wall-clock added here is off-thread
/// blocked time, so if it exceeds this window's real `blocked` remainder the excess
/// is dropped rather than wrongly shrinking decode/prefill/section (which would
/// inflate their tok/s).
pub(crate) fn note_persistence_maint_us(us: u64) {
    MAINT_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

/// Run a blocking wait (e.g. `flush_blocking`) and add its wall-clock to
/// [`WAIT_US`], so a multi-second persistence stall surfaces as the Sync phase.
fn timed_wait<T>(f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    WAIT_US.fetch_add(
        t.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

#[inline]
fn record_phase(start: Instant, phase: &'static str) {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::trace!(
        target: "candle_conversation::scheduler::timing",
        phase,
        ms,
        "phase",
    );
}

/// Assemble per-token raw wide `sign(Q)` from the GPU-packed sub-band bits
/// ([`ProvSignPacked`]) and fold each to the compact provenance signature. This
/// is the on-CPU tail of the GPU fast path in [`Scheduler::gather_wide_sigs`] and
/// is **bit-identical** to `fold_provenance(WideQSig::from_band(..))`: a head's
/// `head_dim` sign bits are its `n_palette` sub-band u32s laid down at global
/// dims `[p*sub_head_dim, (p+1)*sub_head_dim)` — exactly how `from_band` packs
/// them (bit `i` → word `i/64`, bit `i%64`). Only real tokens (per the chunk
/// `layout`) are emitted, skipping partial-chunk padding.
fn assemble_folded_prov_sigs(
    packed: &ProvSignPacked,
    layout: &[(u16, u16, usize)],
    head_dim: usize,
) -> Vec<WideQSig> {
    let chunk = candle_nn::CHUNK_SIZE;
    let wph = head_dim.div_ceil(64);
    let n_layers = packed.n_layers;
    let n_kv_head = packed.n_kv_head;
    let n_palette = packed.n_palette;
    let sub = packed.sub_head_dim;
    let n_blocks = packed.block_indices.len();
    let n_raw_heads = n_layers * n_kv_head;
    if wph == 0 || n_raw_heads == 0 || sub == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (block_pos, &block_idx) in packed.block_indices.iter().enumerate() {
        let Some(&(offset, len, _cum)) = layout.get(block_idx) else {
            continue; // no metadata — can't place this block's tokens
        };
        let offset = offset as usize;
        for j in 0..len as usize {
            let t = offset + j; // physical token slot within the chunk
            if t >= chunk {
                break;
            }
            let mut words = vec![0u64; n_raw_heads * wph];
            for layer in 0..n_layers {
                for head in 0..n_kv_head {
                    let hidx = layer * n_kv_head + head;
                    for p in 0..n_palette {
                        let warp =
                            ((layer * n_blocks + block_pos) * n_kv_head + head) * n_palette + p;
                        let Some(&bits_u32) = packed.packed.get(warp * chunk + t) else {
                            continue;
                        };
                        let bits = bits_u32 as u64;
                        // Palette p → global head-dims [p*sub, (p+1)*sub); place its
                        // `sub` bits at that offset within the head's `wph` words,
                        // splitting across the word boundary if it straddles one.
                        let bit_off = p * sub;
                        let w = bit_off / 64;
                        let sh = bit_off % 64;
                        if let Some(word) = words.get_mut(hidx * wph + w) {
                            *word |= bits << sh;
                        }
                        if sh + sub > 64 {
                            if let Some(word) = words.get_mut(hidx * wph + w + 1) {
                                *word |= bits >> (64 - sh);
                            }
                        }
                    }
                }
            }
            out.push(fold_provenance(&WideQSig {
                n_heads: n_raw_heads as u16,
                words,
            }));
        }
    }
    out
}

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

/// All the per-conversation context the scheduler needs to do continuous
/// re-projection on its own thread, without round-tripping back to the
/// caller.  Built once at [`SchedulerRequest::SubmitTurn`] time and stored
/// on the active `DecodeState`.
///
/// When `every_n_tokens > 0`, the scheduler:
///
/// 1. Tracks decoded-token count per-view.
/// 2. After every `every_n_tokens` tokens, extracts live Q sigs from the
///    view's R16 backing at the three provenance layers.
/// 3. Runs provenance against the substrate corpus and writes fresh per-turn
///    scores under [`Conversation::write`].
/// 4. Re-runs `Builder::project` for `target` to obtain a new
///    visible-block set.
/// 5. Drops the current view (transferring its decoded blocks to the
///    parent via `finalize_view`) and carves a fresh view borrowing the
///    new ranges plus the active turn's own decoded tail.  Decode state
///    (sampler, generated-token buffer, health) is re-keyed onto the new
///    view id, invisible to the caller.
#[derive(Clone)]
pub(crate) struct ReprojectionPolicy {
    pub(crate) target: ProjectionTarget,
    pub(crate) projection: Arc<Builder>,
    /// Section-tree selection (the composer dials), so decode reprojection emits
    /// the same selector options as the initial prefill.
    pub(crate) selection: SelectionState,
    pub(crate) substrate: Conversation,
    /// Cadence trigger: re-project after every `every_n_tokens` decoded
    /// tokens.  `0` disables the cadence trigger (punctuation triggers
    /// can still fire).
    pub(crate) every_n_tokens: usize,
    /// Maximum tokens looking back from the current decode position to
    /// include in the wide-Q probe.  Caps the "thought window" — beyond
    /// this many tokens the prior reprojection already captured the
    /// older intent.  Must be `>= 1`.  Default: 64.
    pub(crate) max_probe_tokens: usize,
    /// Token IDs that, when sampled, fire an immediate reprojection in
    /// addition to the every-`every_n_tokens` cadence.  Use for
    /// paragraph/sentence boundaries (`\n`, `. `, etc) so attention
    /// re-orients at semantic transition points rather than waiting
    /// for the next fixed-cadence trigger.
    pub(crate) trigger_token_ids: Arc<Vec<u32>>,
    /// Token id of the `<tool_call>` open tag (when the tokenizer has it as a
    /// single token). When sampled, one lock-in reprojection fires — committing
    /// the tool from the reasoning so far — and reprojection is then suppressed
    /// until `tool_call_close_id`, so the generic call body doesn't re-orient the
    /// selection. `None` disables the gate (tag not a single token).
    pub(crate) tool_call_open_id: Option<u32>,
    /// Token id of the `</tool_call>` close tag — re-enables reprojection.
    pub(crate) tool_call_close_id: Option<u32>,
}

impl ReprojectionPolicy {
    /// Whether the target layer declares any belief-driven collection. The
    /// immediate first-reprojection at prefill promotion only pays off when
    /// there is a belief scan to run against the query's wide-Q — a
    /// plain-prompt layer (e.g. the titler's single-section schema) gains
    /// nothing from the extra view swap, so its turns skip it. Cadence and
    /// punctuation reprojections are unaffected.
    pub(crate) fn has_belief_collections(&self) -> bool {
        self.projection
            .schema()
            .system_prompt
            .items
            .iter()
            .any(|i| matches!(i, SystemPromptItem::Collection(_)))
    }
}

/// Inputs the scheduler needs to run `Builder::project()` itself for a
/// `SubmitTurn` request.  The target — `(layer, group, timeline)` — is
/// **not** carried here; the scheduler reads it from
/// [`Scheduler::slot_targets`] (pinned at `NewSequence` /
/// `ResumeSequence` time).  Callers only need to supply the schema and
/// the conversation's pinned system-section id.
#[derive(Clone)]
pub(crate) struct ProjectionInputs {
    pub projection: Arc<Builder>,
    /// Section-tree selection for this turn's projection (the composer dials).
    pub selection: SelectionState,
}

// ————————————————————————————————————————————————————————————————————————————
// Internal state
// ————————————————————————————————————————————————————————————————————————————

/// In-flight reprojection state carried from
/// [`Scheduler::reproject_view_prepare`] to
/// [`Scheduler::reproject_view_complete`], so the cross-conversation wave can
/// prepare many views (scan + project + inject sealed + build the gap-fill
/// descriptor), fire ONE batched multi-slot gap-fill forward, then complete
/// each view (finish + carve new view + re-key) independently.
struct ReprojectInFlight {
    view_id: SequenceId,
    parent_id: SequenceId,
    tail_per_layer: Vec<candle_nn::kv_cache::SealedSequence>,
    decode_state: DecodeState,
    sampling_state: Option<SequenceSamplingState>,
    sections_len: usize,
    segments_len: usize,
    n_turns_selected: usize,
    /// The materialized-context composition this reprojection selected
    /// (buckets + materialized/substrate totals; the decode-span timing is
    /// filled in at `complete` from the per-sequence anchor). Emitted as a
    /// [`TurnEvent::Projection`] so the GUI drops a timeline dot per reproject.
    composition: crate::projection::ProjectionEvent,
    plan: projection_assembler::GapFillPlan,
    build_ms: u64,
    /// Wall-clock start of the whole reproject (set at the top of `prepare`);
    /// drives `total_ms` reported at the end of `complete`.
    t_repro: Instant,
    /// Pure swap work in `prepare` (tail snapshot + free + truncate) — disjoint
    /// from `elevate_ms`/`glue_ms`/`apply_ms`, unlike the old umbrella that
    /// spanned the whole reproject and hid the glue wave inside it.
    swap_ms: u64,
    probe_ms: u64,
    scan_ms: u64,
    project_ms: u64,
    elevate_ms: u64,
}

/// Per-sequence state while actively generating tokens.
struct DecodeState {
    /// Channel back to the caller.
    event_tx: Sender<TurnEvent>,
    /// Tokens generated so far.
    generated_tokens: TokenBuffer,
    /// Maximum tokens to generate.
    max_tokens: usize,
    /// Sampling configuration for this turn.
    sampling_config: SamplingConfig,
    /// What the scheduler does in `cleanup_finished` — see [`SealAction`].
    /// The `(layer, group, timeline)` to write to is looked up from
    /// [`Scheduler::slot_targets`] at seal time, not carried here.
    seal_action: SealAction,
    /// Whether this sequence has finished (EOS or max_tokens).
    finished: bool,
    /// Decode start time (for stats).
    decode_start: Instant,
    /// Prefill duration (for stats).
    prefill_ms: f64,
    /// Number of tokens in the prefill prompt for this turn.
    /// Recorded for the health diagnostic dump so that step depth can be
    /// interpreted relative to the context window size.
    prefill_token_count: usize,
    /// Turn start time (for total_ms stat).
    turn_start: Instant,
    /// Per-sequence decode health tracking state.
    health: DecodeHealthState,
    /// Continuous re-projection policy.  When `Some` and
    /// `policy.every_n_tokens > 0`, the scheduler fires a view swap each
    /// time `generated_tokens.len()` becomes a multiple of
    /// `every_n_tokens`.  Carried across mid-decode swaps unchanged.
    reprojection: Option<ReprojectionPolicy>,
    /// Count of non-trigger tokens decoded since the last reprojection.
    /// A punctuation trigger only fires a reprojection once this exceeds 16,
    /// so short lines / runs of trigger tokens don't each re-project. Reset to
    /// 0 whenever a reprojection (cadence or punctuation) is queued.
    non_punct_since_reproject: usize,
    /// Generated-token index of the previous projection point, so a reprojection
    /// event's span is `[last_projection_end, current_gen)`. Lives on the decode
    /// state (which migrates correctly across reprojection view swaps), unlike the
    /// separate view-keyed anchor map.
    last_projection_end: u32,
    /// Trailing structural tokens forwarded into the slot after
    /// decode finishes, before the seal.  Today's seal path leaves
    /// this empty (the `assistant_end` boundary is a live
    /// `Generated` segment at projection time, not part of the
    /// persisted turn).  Kept on `DecodeState` for the no-decode
    /// `insert_turn` path that still needs to close its own brackets.
    post_decode_tokens: TokenBuffer,
    /// Online belief accumulated across this turn's reprojections. Each
    /// reprojection seeds `project()` from it and writes the result back, so the
    /// RelLeak decay/reinforcement carries across the turn (§80). Migrates with
    /// the decode state through view swaps.
    belief: PriorBelief,
    /// Full prefill token sequence pinned into the slot at this
    /// turn's submit:
    /// `[user_msg][user_end][assistant_start]` — the assistant header is clean,
    /// and the `insert_turn` path additionally prepends `/no_think`.  The think
    /// block is NEVER baked here: a suppressed turn decodes its own empty
    /// `<think></think>` and a thinking turn opens its own `<think>`, so either
    /// way it lands in `generated_tokens`.
    /// Concatenated with `generated_tokens` and `post_decode_tokens`
    /// at seal time to form `TurnContent::token_ids` — the
    /// cross-process replay sequence persisted to the redo log.
    prefill_tokens: TokenBuffer,
    /// The raw user message string — passed through from
    /// `submit_turn` verbatim (no role-marker envelope, no
    /// `/no_think` prefix).  Stored directly on the substrate at
    /// seal time as `TurnPart::user_text` so the sidebar reload
    /// path never has to re-tokenise or scan for boundary markers.
    user_text: String,
    /// Gather-scope tags for this turn, from the submit request. Persisted onto
    /// the turn's `TurnDecl` at seal so a projection policy's `tags:` filter can
    /// scope its provenance gallery. Empty for live/untagged turns.
    tags: Vec<String>,
    /// Content boundaries that frame the user-message body and the
    /// assistant-response body inside the sealed grid.  Set from the submit
    /// path (CPU-tokenised against the prefill prefix strings) and used at seal
    /// time to build the turn's [`TurnLayout`] so the compressor can window
    /// content-only halves on demand.
    user_content_start: u32,
    user_content_end: u32,
    assistant_content_start: u32,
    /// Whether this turn was submitted with thinking SUPPRESSED (the composer
    /// `/no_think` dial active).  Recorded into the turn's [`TurnLayout`] at
    /// seal so the projection re-injects the `/no_think` soft-switch into this
    /// turn's user opener when it re-renders the turn as history.
    no_think: bool,
    /// The prefilled assistant half (repo_map / code_reading ingest), stored
    /// verbatim as `TurnPart::assistant_text` at seal time. Empty for decode
    /// turns, where the seal uses the model's decoded text instead.
    prefill_assistant_text: String,
    /// `true` while decoding inside a `<tool_call>…</tool_call>` block. Set when
    /// the open tag is sampled (which also fires one lock-in reprojection) and
    /// cleared at the close tag. While set, cadence/punctuation reprojection is
    /// suppressed so the generic call body can't re-orient the committed tool.
    /// Migrates with the decode state across view swaps.
    in_tool_call: bool,
    /// Tool-call stencils available this turn, keyed by trigger token.  Empty
    /// registry = free decode.  Carried from the turn's `SubmitTurn`.
    triggers: Arc<TriggerRegistry>,
    /// The constrained-decoding walk currently in progress (a `<tool_call>` is
    /// mid-emission).  `None` whenever the model is free-decoding — a genuine
    /// present/absent, not a feature flag.
    stencil: Option<StencilDriver>,
    /// The decode action (`Branch`/`Free`/`Done`) the stencil yielded for the
    /// upcoming decode step.  Set by `inject_stencil_prefills` once any preceding
    /// static runs are injected; consumed by `batch_decode_step` (apply the mask,
    /// then `accept` the sampled token and clear).  `None` ⇒ the driver needs
    /// advancing again.
    pending_mask: Option<StepMask>,
}

/// Per-turn view bookkeeping carried in the scheduler's `turn_views` map.
///
/// Distinct from a single view's `(parent, original_borrowed)` because
/// the **turn** outlives any single view: under continuous re-projection
/// each swap allocates a fresh view, so we need to remember turn-level
/// invariants (where the parent stood when the turn started) so the new
/// view can include the active turn's own decoded tail.
#[derive(Debug, Clone, Copy)]
struct ViewState {
    /// Parent sequence the current view borrows from.  Stable across
    /// mid-decode swaps within one turn.
    parent_id: SequenceId,
    /// Number of blocks **this** view borrowed at carve time.  Used by
    /// `finalize_view` to know which blocks were borrowed (and therefore
    /// must remain in the parent unchanged) versus which were freshly
    /// written by the view (and therefore must transfer back).
    original_borrowed: BlockCount,
    /// The parent's total block count at the moment **the very first**
    /// view of this turn was carved.  Stable across swaps; copied
    /// verbatim from the swapped-from `ViewState` into the swapped-to
    /// one.  Lets the re-projection helper compute the active-turn tail
    /// range — the just-decoded blocks at parent indices
    /// `[turn_start_parent_blocks, current_parent_blocks)`.
    turn_start_parent_blocks: usize,
}

/// A model-decode compression node in flight — the [`Content::Decode`]
/// summariser. The node's scratch slot holds the sealed originals
/// (`[full user | full assistant]`, natural roles, zero re-prefill) plus a tiny
/// summarise instruction, and the decode produces the node's **assistant half**
/// only. When it finishes, `complete_compression_pass` pairs that body with the
/// scope derived up front and seals both halves through
/// [`Scheduler::seal_compression_turn`], firing `response_tx` with the new
/// `TurnIndex`.
///
/// The decode rides the normal wave; the deterministic structural path
/// (repo_map) skips it and derives both halves, but reaches the same seal.
struct CompressionJob {
    /// Conversation that owns the node's timeline.
    conversation: Conversation,
    /// Projection target (layer, group, timeline) the node records into.
    target: ProjectionTarget,
    /// Node kind — carried to the seal, which records it on the sealed turn.
    kind: TurnKind,
    /// The node's user half, derived from the children's scopes before the decode
    /// ran (`summary_tree::scope`). Paired with the decoded assistant half at the
    /// seal — the decode is never asked to write a question.
    scope_tokens: Vec<u32>,
    /// The child turns this pass lifted to hot to attend over. Demoted back to
    /// warm in `complete_compression_pass` once the summary has decoded, so a long
    /// run of summary passes doesn't accumulate transient hot residency.
    children: Vec<TurnKey>,
    /// Summariser channel — receives the recorded node's `TurnIndex`, or an `Err`
    /// if the decode produced no forwarded tokens.
    response_tx: Sender<Result<TurnIndex, ProbeError>>,
}

/// A compressed turn whose marker-framed text has been enqueued for re-prefill
/// on the shared prefill wave, awaiting its seal. Keyed by `job_id` in
/// [`Scheduler::pending_compression_seals`]. When the prefill completes,
/// `complete_compression_turn` snapshots the slot's freshly-computed
/// (role-coherent) K/V, records the turn, and fires `response_tx`.
struct PendingCompressionSeal {
    conversation: Conversation,
    target: ProjectionTarget,
    /// The compressed turn's segment-vector layout (user / assistant text +
    /// spans), built when the exchange was framed.
    layout: TurnLayout,
    token_ids: Vec<u32>,
    /// Forest kind of the node — see [`CompressionJob::kind`].
    kind: TurnKind,
    /// The turns this node compresses — referenced by the node's synthesized
    /// projection event so a wide-Q hit on the summary resolves to the
    /// covered turns.
    children: Vec<TurnIndex>,
    /// Gather-scope tags inherited as the union of the children's TurnDecl
    /// tags: a code_read leaf carries its scan turn's `["code", <path>]`, a
    /// SoS the union of its children's; untagged (dialogue) children yield
    /// an untagged summary, preserving the tag-partition invariant.
    tags: Vec<String>,
    response_tx: Sender<Result<TurnIndex, ProbeError>>,
}

/// A finished dialogue turn whose reasoning-free tokens have been enqueued for
/// re-prefill on the shared wave, awaiting its seal. Keyed by `pending_id` in
/// [`Scheduler::pending_turn_seals`]. When the prefill completes,
/// `complete_turn_reprefill` snapshots the clean K/V, seals the turn (via the
/// normal `SealAction::Turn` write), and fires the deferred `Done` — the seal
/// therefore lands one wave after decode, but the client's streamed tokens are
/// unaffected and `Done` still carries the seal result.
struct PendingTurnSeal {
    /// Parent slot the clean turn is re-prefilled onto (the view was already
    /// finalized in `cleanup_finished`, so this is the plain parent sequence).
    parent_id: SequenceId,
    /// First block of the turn's own region on `parent_id` — the clean re-prefill
    /// appends here, and the seal captures `[seal_block_from, block_count)`.
    seal_block_from: usize,
    /// The turn's segment layout: the `<think>…</think>` block is an ETHEREAL
    /// `Thinking` segment (its text is kept for display, its K/V dropped).
    layout: TurnLayout,
    /// Clean replay tokens (reasoning stripped) — pinned as the turn's `token_ids`
    /// so they match the reasoning-free sealed K/V.
    token_ids: Vec<u32>,
    /// Gather-scope tags carried from the decode's `DecodeState`, re-stamped onto
    /// the sealed turn. (Wide-Q provenance sigs are NOT carried — the seal
    /// re-gathers them from the reasoning-free re-prefilled grid via
    /// `gather_wide_sigs`, so they match the sealed K/V.)
    tags: Vec<String>,
    /// The caller's event channel — the deferred `Done` fires here once sealed.
    event_tx: Sender<TurnEvent>,
    /// `Done` payload, captured at decode-end: the FULL decoded reply (reasoning
    /// included, exactly as streamed) and the decode stats. Only the SEALED K/V
    /// is reasoning-free; the client's view is unchanged.
    done_text: String,
    done_token_ids: TokenBuffer,
    stats: TurnStats,
    /// The re-prefill `PrefillWork`'s private event sink. The prefill machinery
    /// sends progress/errors on the paired `Sender`; keeping the receiver alive
    /// here stops those sends from failing before the wave completes. Dropped
    /// when the pending seal is drained.
    _sink_rx: Receiver<TurnEvent>,
}

/// A code-scope ingest unit awaiting its scratch slot on the fair pump.
///
/// The client submits every scope of a file as a [`SchedulerRequest::PrefillScope`];
/// each becomes a `QueuedScope` in [`Scheduler::scope_pending`]. `pump_scope_prefills`
/// drains them fairly (least-advanced file first) onto scratch slots, bounded so no
/// more than [`Scheduler::MAX_SCOPE_SLOTS`] scratch slots exist at once.
/// The inputs `build_turn_layout` needs to (re)build a scope turn's segment
/// layout. Carried unbuilt through the queue + prefill because the closing
/// assistant segment is DECODED: the layout can only be finalised once
/// `complete_scope_summary_decoded` has the summary tokens, so the offsets + texts are
/// threaded through and the layout is built once, post-decode. `assistant_text`
/// is the prefilled prefix only — `<tool_call>` … `<tool_response>` up to the
/// tool_response's user_end — with the `assistant_start` + decoded summary
/// appended at seal.
struct ScopeLayoutInputs {
    user_content_start: u32,
    user_content_end: u32,
    /// Grid offset of the FIRST assistant segment (`<tool_call>`), after the
    /// user request's `user_end` + `assistant_start`.
    assistant_content_start: u32,
    user_text: String,
    assistant_text: String,
}

struct QueuedScope {
    /// Position within the file (record order). Indexes the file batch's slots.
    scope_index: u32,
    /// Cold-prefill grid: `[/no_think][user][user_end][assistant_start][assistant]`,
    /// where `assistant` stops at the tool_response's user_end (the summary is decoded).
    tokens: Vec<u32>,
    /// Inputs to (re)build the turn layout post-decode. See [`ScopeLayoutInputs`].
    layout_inputs: ScopeLayoutInputs,
    /// Token count of `tokens` — the prefill grid length, before the decode.
    token_count: usize,
    /// Gather-scope tags carried to the recorded turn's `TurnDecl`.
    tags: Vec<String>,
}

/// A scope prefill in flight on the wave, keyed by its scratch slot in
/// [`Scheduler::pending_scope_prefills`]. Carries what `complete_scope_summary_decoded`
/// needs to snapshot the scope + rebuild its layout into its file batch — the
/// batch itself is keyed by `timeline` in [`Scheduler::scope_batches`].
struct PendingScopePrefill {
    timeline: TimelineId,
    scope_index: u32,
    layout_inputs: ScopeLayoutInputs,
    /// The prefill grid tokens (up to the tool_response's user_end). The decode's
    /// `assistant_start` + summary tokens are appended in
    /// `complete_scope_summary_decoded`.
    token_ids: Vec<u32>,
    token_count: usize,
    /// Gather-scope tags carried to the recorded turn's `TurnDecl`.
    tags: Vec<String>,
    /// The reasoning-free layout + token ids built in
    /// `complete_scope_summary_decoded` (summary `<think>` stripped) and consumed
    /// by `complete_scope_summary_sealed` once the clean re-prefill lands. `None`
    /// until the summary has decoded.
    sealed_layout: Option<TurnLayout>,
    sealed_token_ids: Option<Vec<u32>>,
}

/// One scope's snapshotted K/V, buffered in its file batch until the flush cursor
/// reaches it and `advance_scope_flush` records it (in scope order).
struct SealedScope {
    /// Per-layer sealed K/V (RAII `ChunkGid` clones keep it alive across the slot free).
    sealed_gpu: Vec<SealedSequence>,
    /// Per-token wide-Q provenance sigs captured over the fresh (R16) blocks — the
    /// gallery entries a provenance scan matches against.
    sigs: Vec<WideQSig>,
    /// Gather-scope tags (e.g. `["code", <path>]`) stamped on the recorded turn's
    /// `TurnDecl` so tag-scoped provenance galleries admit it.
    tags: Vec<String>,
    layout: TurnLayout,
    token_ids: Vec<u32>,
    token_count: usize,
    /// Block count of the sealed slice — the recorded turn's `block_end`.
    block_end: usize,
}

/// A file's parallel scope ingest, keyed by `timeline` in
/// [`Scheduler::scope_batches`]. Scopes prefill in any order on the wave and land
/// in `sealed[scope_index]`; [`Scheduler::advance_scope_flush`] records the
/// contiguous run of landed scopes at the front (`flushed` onward) into the file
/// timeline **as they land**, in scope order, answering each `responder`.
///
/// Recording incrementally — rather than holding every scope's sealed K/V in VRAM
/// until the whole file completes — is what keeps a large file's ingest from
/// pinning gigabytes of un-evictable K/V: each recorded scope becomes an ordinary
/// timeline turn under the persist/eviction pipeline the moment its prefix is
/// reached, so its VRAM is reclaimable while later scopes are still prefilling.
struct PendingScopeBatch {
    /// The conversation that owns the file timeline (records the turns).
    conversation: Conversation,
    /// Total scopes in the file — recording is complete at `flushed == total`.
    total: u32,
    /// Snapshotted scopes, indexed by `scope_index`; `Some` once that scope lands
    /// successfully (taken when recorded), `None` for a pending or failed scope.
    sealed: Vec<Option<SealedScope>>,
    /// Whether each scope has landed (succeeded **or** failed), indexed by
    /// `scope_index` — the flush cursor advances over the contiguous landed prefix.
    /// Distinct from `sealed.is_some()` so a failed scope (which leaves `sealed`
    /// `None`) doesn't stall the prefix, and an unsubmitted scope (also `None`)
    /// isn't mistaken for landed.
    landed: Vec<bool>,
    /// Per-scope reply channels, indexed by `scope_index`; answered as each scope
    /// is recorded with its `TurnIndex` (or on failure with the error).
    responders: Vec<Option<Sender<Result<TurnIndex, ConversationError>>>>,
    /// Next scope index to record — the low water mark of the contiguous run of
    /// landed-and-recorded scopes. Recording is done when it reaches `total`.
    flushed: u32,
    /// Fired with each scope's token count as it lands — live per-scope ingest
    /// progress (see [`SchedulerRequest::PrefillScope::on_prefilled`]).
    on_prefilled: ScopeProgressFn,
}

/// Max-min fair pick among `(timeline_raw, submitted_count)` candidates — each a
/// file with at least one pending scope. Returns the file that has had the fewest
/// scopes pumped onto scratch slots so far, ties broken by the lowest raw id.
/// `None` when there are no candidates. Pure core of
/// [`Scheduler::fairest_scope_timeline`]; the max-min rule is what lets a lone
/// file with many scopes claim all the slots while several files still each get a
/// fair turn.
fn pick_fair_scope(candidates: &[(u64, u32)]) -> Option<u64> {
    candidates
        .iter()
        .min_by_key(|(raw, submitted)| (*submitted, *raw))
        .map(|(raw, _)| *raw)
}

/// AIMD multiplicative-decrease of the prefill admission window: halve `w` but
/// never below `floor`. Pure core of [`Scheduler::shrink_admit_window`] —
/// repeated application converges to `floor` (never 0, so the engine always
/// keeps ≥`floor` prefills in flight).
fn narrow_window(w: usize, floor: usize) -> usize {
    (w / 2).max(floor)
}

/// AIMD additive-increase of the prefill admission window: `w + 1` capped at
/// `ceil`. Pure core of [`Scheduler::grow_admit_window`].
fn widen_window(w: usize, ceil: usize) -> usize {
    (w + 1).min(ceil)
}

/// Admission action chosen by the drain-backlog controller.
#[derive(Debug, PartialEq, Eq)]
enum BacklogAction {
    Shrink,
    Grow,
    Hold,
}

/// Pure decision core of [`Scheduler::regulate_ingest_admission`]: from the live
/// hot→warm `backlog`, its `target`, the current `window`/`ceil`, and whether
/// VRAM is under pressure, decide whether to narrow, widen, or hold ingest
/// admission. Hysteresis — shrink above `target`, grow only below `target / 2`,
/// hold in the deadband between — keeps the window from flapping around the
/// target as the backlog jitters.
fn backlog_admit_action(
    backlog: usize,
    target: usize,
    window: usize,
    ceil: usize,
    vram_pressure: bool,
) -> BacklogAction {
    if backlog > target {
        BacklogAction::Shrink
    } else if backlog < target / 2 && window < ceil && !vram_pressure {
        BacklogAction::Grow
    } else {
        BacklogAction::Hold
    }
}

/// What the scheduler does after a [`SubmitTurn`] decode completes,
/// just before sending `Done`.
///
/// Encodes the substrate write that previously lived on the
/// conversation side as a follow-up round trip.
#[derive(Clone, Debug)]
pub(crate) enum SealAction {
    /// Seal the parent slot and append the result to the workspace
    /// substrate at `(target.layer, target.group)` (read from the
    /// turn's `projection_inputs`) as a new turn.
    Turn,
    /// Seal the parent slot and pin the result on the workspace
    /// substrate under `section_id` via `set_section_full`.
    ///
    /// `address` is the content-addressed `(prefix_hash, section_hash)`
    /// pair computed at IngestSection time — used to derive the
    /// stream id under which the section's chunks persist.
    /// `debug_name` is the section's symbolic name (e.g. tool name or
    /// schema item id) carried into the SectionDecl record purely
    /// for diagnostic surfacing in the manifest dump.
    Section {
        section_id: SectionId,
        tokens: Arc<Vec<u32>>,
        address: ContentAddress,
        debug_name: String,
        /// Propagated from `SchedulerRequest::IngestSection` — `true`
        /// when this section is a schema-Collection member (a tool in
        /// the tool catalog, an entry in a hits list, etc.), `false`
        /// for boundary sections (role markers, opening / closing
        /// tags).  Drives section-quantize policy selection: members
        /// use the turn-level adaptive policy (much smaller VRAM at a
        /// small precision cost — top-k projection masks the
        /// per-member noise anyway); boundary sections use the
        /// near-lossless C0 / Q8_KS / Q8_0 policy because every
        /// later token of every later turn attends back over them.
        in_collection: bool,
    },
    /// Skip the seal entirely.  Used by raw RULER eval and
    /// summarisation paths that don't write to the substrate.
    None,
    /// The summarise decode for `job_id`, which produces the node's assistant
    /// half. When it finishes, `complete_compression_pass` pairs the decoded body
    /// with the scope derived before the decode and seals both halves through
    /// `seal_compression_turn`, which replies to the summariser.
    CompressionPass { job_id: u64 },
    /// The re-prefilled compressed turn for `job_id`. Once the prefill wave
    /// finishes the marker-framed `[question][user_end][assistant_start][answer]`
    /// on its scratch slot, `promote_finished_prefills_to_decodes` snapshots the
    /// freshly-computed (role-coherent) K/V, records the turn from the pending
    /// seal stashed in [`Scheduler::pending_compression_seals`], and replies to
    /// the summariser. `max_decode_tokens` is 0 — prefill + seal, no decode.
    CompressionTurn { job_id: u64 },
    /// The clean re-prefill of a finished dialogue turn, keyed by `pending_id`
    /// in [`Scheduler::pending_turn_seals`]. The decode's K/V carried the
    /// `<think>…</think>` reasoning; this unit re-prefills the turn with the
    /// reasoning stripped so the SEALED K/V never lets a future projection attend
    /// its own thoughts. Rides the shared prefill wave (batched with the live
    /// turn + summaries); once it finishes,
    /// `promote_finished_prefills_to_decodes` snapshots the clean K/V, seals the
    /// turn (reasoning kept as ethereal text), and fires the deferred `Done`.
    /// `max_decode_tokens` is 0 — prefill + seal, no decode.
    TurnReprefill { pending_id: u64 },
    /// A parallel code-scope ingest unit: one scope of a file, cold-prefilled on
    /// its own scratch slot so N scopes of a file (and scopes across files) batch
    /// on the shared wave instead of prefilling serially. `max_decode_tokens` is
    /// 0 — prefill + summary decode, no decode ON the prefill unit itself. When
    /// the prefill finishes, `promote_finished_prefills_to_decodes` calls
    /// `begin_scope_summary_decode`, which frames `assistant_start` and registers
    /// the [`Self::ScopeSummary`] decode on the same slot. The slot is the routing
    /// key — its [`Scheduler::pending_scope_prefills`] entry carries the timeline +
    /// index + layout inputs.
    ScopeIngest,
    /// The two-sentence summary decode for a code-scope, riding on the SAME slot
    /// the scope prefilled onto. When the `ScopeIngest` prefill finishes,
    /// `begin_scope_summary_decode` frames `assistant_start` and registers this
    /// bounded decode (mirroring [`Self::CompressionPass`]); the decode attends
    /// over the excerpt in-slot, so the summary is anchored to the code. On
    /// completion `cleanup_finished` routes to `complete_scope_summary_decoded`, which
    /// snapshots the whole slot (excerpt + summary), rebuilds the turn layout
    /// with the decoded summary as its closing assistant segment, gathers wide-Q
    /// sigs over the summary-inclusive grid (the provenance anchor), and records
    /// the scope through `advance_scope_flush` exactly like `ScopeIngest`. The
    /// slot is the routing key — its [`Scheduler::pending_scope_prefills`] entry
    /// carries the timeline + index + layout inputs.
    ScopeSummary,
    /// The reasoning-free RE-PREFILL of a code-scope after its summary decoded,
    /// mirroring [`Self::TurnReprefill`] for the scope path. The summary decode
    /// may emit a `<think>…</think>` block (empty under `/no_think`, or a full
    /// reasoning leak); `complete_scope_summary_decoded` strips it and re-prefills
    /// the clean `[excerpt tool-exchange][assistant_start][stripped summary]` grid
    /// on the same scratch slot, so the SEALED K/V and its wide-Q provenance
    /// signature never carry reasoning. When this prefill finishes,
    /// `promote_finished_prefills_to_decodes` calls `complete_scope_summary_sealed`,
    /// which snapshots the clean K/V into the file batch. The slot is the routing
    /// key — its [`Scheduler::pending_scope_prefills`] entry carries the rebuilt
    /// clean layout + token ids.
    ScopeReprefill,
}

/// Content the substrate pins on a `SealAction::Turn` write — the
/// user message, the assistant reply, and the full token sequence
/// assembled from the prefill + decoded generation + post-decode tail.
///
/// Defaulting (empty strings, empty tokens) is fine for paths that
/// lack a content trail (test fixtures, in-process seals where the
/// caller doesn't care about cross-process restore).
#[derive(Debug, Default, Clone)]
pub(crate) struct TurnContent {
    pub role: Role,
    /// Gather-scope tags for this turn, carried from submit to seal.
    pub tags: Vec<String>,
    /// The turn's segment-vector layout — user / thinking / assistant text and
    /// each real segment's K/V span, built at the seal site.
    pub layout: TurnLayout,
    /// The combined token sequence pinned onto the slot, in slot
    /// order.  Must match the K/V chunk grid 1-1; consumed by
    /// `persist_tokens_only` for cross-process replay.
    pub token_ids: TokenBuffer,
}

/// A section whose hot bytes are in their native (prefill-output) form
/// and need to be quantized to the configured `compression_policy` at
/// the next turn-seal boundary.  The conversation handle picks up the
/// section's current hot residence via `section_residence(section_id)`
/// when the drain runs.
struct PendingSectionQuantize {
    section_id: SectionId,
    /// Propagated from `SealAction::Section::in_collection` — drives
    /// the drain's choice of compression policy.  Collection members
    /// (tools in a tool catalog, hits in a retrieval list, etc.) take
    /// the turn-level adaptive policy because the projection's top-k
    /// already masks their per-member K/V noise; boundary sections
    /// (role markers, opening/closing tags) take the conservative
    /// near-lossless policy because every later token attends back
    /// over them.
    in_collection: bool,
}

/// A unit of prefill work queued for processing.
pub(super) struct PrefillWork {
    pub(super) sequence_id: SequenceId,
    pub(super) tokens: TokenBuffer,
    /// Text to emit as TurnEvent::Prefill before starting decode.
    pub(super) prefill_text: String,
    /// Raw user message string — see [`DecodeState::user_text`].
    pub(super) user_text: String,
    /// Gather-scope tags — see [`DecodeState::tags`].
    pub(super) tags: Vec<String>,
    /// Content boundaries — see [`DecodeState`].
    pub(super) user_content_start: u32,
    pub(super) user_content_end: u32,
    pub(super) assistant_content_start: u32,
    /// Thinking suppressed at submit — see [`DecodeState::no_think`].
    pub(super) no_think: bool,
    /// Prefilled assistant half — see [`DecodeState::prefill_assistant_text`].
    pub(super) prefill_assistant_text: String,
    pub(super) event_tx: Sender<TurnEvent>,
    pub(super) max_decode_tokens: usize,
    pub(super) sampling: SamplingConfig,
    pub(super) submitted_at: Instant,
    /// Carried through prefill so it can be installed onto `DecodeState`
    /// when the prefill promotes to decode.  `None` when the caller
    /// disabled re-projection.
    pub(super) reprojection: Option<ReprojectionPolicy>,
    /// The turn's opening belief — the conversation's carried belief stepped
    /// through the submit-time projection. Installed onto `DecodeState` so the
    /// first mid-decode reprojection evolves it rather than starting empty.
    /// Default for paths with no belief-driven selection (sections, resume,
    /// compression).
    pub(super) belief: PriorBelief,
    /// Carried through prefill so the post-Done substrate write fires
    /// on the right key.  The substrate target is looked up from
    /// [`Scheduler::slot_targets`] at seal time, not carried here.
    pub(super) seal_action: SealAction,
    /// Trailing structural tokens written into the slot after decode
    /// finishes, before the seal.  Carried through prefill so the
    /// post-decode forward pass in `cleanup_finished` can run.  Empty
    /// for paths that don't append a closing tail.
    pub(super) post_decode_tokens: TokenBuffer,
    /// Staged-prefill projection points — token offsets into `tokens` where the
    /// wave stops and emits a projection. Empty for a normal (single-projection)
    /// prefill. See [`SchedulerRequest::SubmitTurn::projection_offsets`].
    pub(super) projection_offsets: Vec<u32>,
    /// The projection composition emitted at each staged projection point (the
    /// calibration projection is pinned, so one composition serves every segment;
    /// only its `start_token`/`end_token` span differs per emission). `None` for a
    /// normal prefill.
    pub(super) staged_composition: Option<crate::projection::ProjectionEvent>,
    /// Tool-call stencils carried through prefill and installed on the
    /// [`DecodeState`] at decode start.  Empty registry = no constrained decode.
    pub(super) triggers: Arc<TriggerRegistry>,
}

/// An in-flight prefill, partially advanced. Lives across scheduler
/// iterations until `offset` reaches `work.tokens.len()`, at which point it
/// is promoted to `active_decodes`.
pub(super) struct ActivePrefill {
    pub(super) work: PrefillWork,
    /// Tokens consumed so far.
    pub(super) offset: usize,
    /// Index of the next `work.projection_offsets` entry the wave has yet to
    /// reach and emit. Advances as the prefill crosses each staged projection
    /// point. Unused (stays 0) for a normal prefill with no offsets.
    pub(super) next_projection: usize,
    /// Set once `offset >= work.tokens.len()` by the chunk runner.
    /// Drained by `promote_finished_prefills_to_decodes`.
    pub(super) final_logits: Option<Tensor>,
    /// Set if any chunk for this prefill failed.
    pub(super) error: Option<ConversationError>,
    /// Wall-clock when prefill processing actually started (first chunk).
    pub(super) prefill_start: Option<Instant>,
}

/// An in-flight section ingest — CPU setup is done, awaiting the batched
/// forward pass(es).  Multiple concurrent entries are batched together in
/// [`Scheduler::run_one_section_ingest_chunk`] so collection members (which
/// share the same prefix) prefill in parallel rather than serially.
pub(super) struct ActiveSectionIngest {
    pub(super) sequence_id: SequenceId,
    pub(super) section_id: SectionId,
    pub(super) tokens: TokenBuffer,
    pub(super) offset: usize,
    pub(super) seal_block_from: usize,
    /// Content address derived from the section's tokens + cumulative
    /// prefix at IngestSection time.  Used to derive the persistence
    /// stream id at seal — see `SealAction::Section::address`.
    pub(super) address: ContentAddress,
    /// Section's symbolic name for the SectionDecl record — see
    /// `SealAction::Section::debug_name`.
    pub(super) debug_name: String,
    /// Carried from the `IngestSection` request through to
    /// `SealAction::Section` — see that variant's docstring for the
    /// quantize-policy implications.
    pub(super) in_collection: bool,
    pub(super) response_tx: Sender<Result<SealResult, ConversationError>>,
    pub(super) error: Option<ConversationError>,
}

/// A member of the in-flight continuous-fair-wave prefill group
/// (`docs/continuous_fair_waves.md`). The group creeps through the layers
/// together with its residual held whole between waves; members are separated
/// only at the head, where `forward_wave`'s logits are in caller order. A member
/// is resolved live by `seq_id` (not a stored index) so `swap_remove` in either
/// backing collection can't invalidate it mid-creep.
#[derive(Clone, Copy, Debug)]
pub(super) enum WaveMember {
    /// Dialogue-turn prefill — resolved in `active_prefills`; its full token set
    /// flows through the layers and it is promoted to decode at the head.
    Prefill { seq_id: usize },
    /// Section-ingest chunk — resolved in `active_section_ingests`; covers
    /// `[offset, offset + advance)` (offset is stable until the head, where the
    /// chunk is advanced + sealed). Rides the cohort's creep so its expert loads
    /// co-batch with decode + the dialogue prefills.
    Section { seq_id: usize, advance: usize },
}

// ————————————————————————————————————————————————————————————————————————————
// Scheduler
// ————————————————————————————————————————————————————————————————————————————

/// The scheduler: single thread that owns all GPU resources.
///
/// One forward-pass "channel" (prefill or decode) accumulator for [`WaveStats`].
/// Sum of not-yet-processed prefill tokens across pending/in-flight work — the
/// scheduler's prefill **backlog**. `items` yields `(total_tokens,
/// consumed_offset)` per work unit (queued work has `consumed == 0`; an
/// in-flight prefill/section has `consumed == offset`). This is the signal the
/// unified-wave engine's large-batch trigger keys on (design
/// `docs/unified_wave_inference_engine.md` §4.5). Pure over its input so it is
/// unit-testable without a live scheduler.
pub(super) fn sum_pending_prefill_tokens(items: impl IntoIterator<Item = (usize, usize)>) -> u64 {
    items
        .into_iter()
        .map(|(total, consumed)| total.saturating_sub(consumed) as u64)
        .sum()
}

/// Rolling-window size (turns) for append-only ingest prefills (the
/// `disable_reprojection` path — `code_reading` / `repo_map`): each such turn
/// attends only the system prompt + the last `CODE_READ_WINDOW_TURNS` sealed
/// turns, bounding otherwise-unbounded ingest context that would exhaust the
/// window and blow up KV VRAM (design `docs/unified_wave_inference_engine.md`
/// §4.7 and [`crate::projection::resolver::Conversation::windowed_ingest_ranges`]).
/// `0` = unbounded (whole-parent borrow).
const CODE_READ_WINDOW_TURNS: usize = 8;

#[derive(Default)]
struct WaveChannel {
    fwds: u64,
    seq_sum: u64,
    seq_max: usize,
    tok_sum: u64,
    /// Σ attended-KV length (each forward's prefix/context length, summed over
    /// the batch) over the window. `kv_sum/fwds` is the avg KV a forward swept
    /// — the number to watch for a context-growth slowdown.
    kv_sum: u64,
    ms_sum: u64,
}

/// Periodic aggregator for forward-pass batch ("wave") sizes. Lets us see
/// whether the scheduler is actually batching wide under load without
/// per-forward log spam — it emits a single INFO line at most every 2 s and
/// resets. Single-threaded (scheduler thread only), so no synchronization.
struct WaveStats {
    window_start: std::time::Instant,
    prefill: WaveChannel,
    decode: WaveChannel,
    /// Section-ingest forwards (startup code-read / repo-map prefill). These run
    /// through `forward_batched` like prefills but on the section-ingest path, so
    /// they get their own channel rather than being invisible behind `section_ms`.
    section: WaveChannel,
    /// Per-phase wall-clock spent on the scheduler thread this window, in ms.
    /// These account for where the wall-clock goes when it is NOT a forward:
    /// `drain` includes the synchronous per-turn SubmitTurn handling
    /// (projection + elevate + apply_segments gap-fill + view create), `reproj`
    /// is the mid-decode continuous reprojection drain, the rest are the
    /// forward quanta themselves. A wave whose elapsed ≫ Σ(phase) was blocked
    /// off-thread (e.g. waiting on the persistence thread / a lock).
    drain_ms: u64,
    promote_ms: u64,
    decode_ms: u64,
    prefill_ms: u64,
    section_ms: u64,
    reproj_ms: u64,
    /// Per-scope code-read seal sub-slices of the Prefill phase (microseconds,
    /// summed over the window). These are NOT separate phases — they live inside
    /// `prefill_ms` — but split the non-forward seal cost so we can see whether
    /// the wave-dominating overhead is the GPU snapshot, the wide-Q provenance
    /// sig gather, or the record/compress/persist flush. `seal_count` is scopes
    /// sealed this window.
    seal_snapshot_us: u64,
    seal_sig_us: u64,
    seal_flush_us: u64,
    seal_count: u64,
    /// Eviction phase this window: resident KV bytes freed by the relief ladder
    /// (cold-tail evict + ingest demote + footprint reclaim), the residence count,
    /// and the wall-clock spent doing it. Fed the GUI's phase timeline.
    evict_bytes: u64,
    evict_count: u64,
    evict_ms: u64,
    /// Idle phase this window: wall-clock the loop spent blocked on `rx.recv()`
    /// with NO work to run — waiting for the next request. Carved out of the
    /// window remainder so it isn't mislabeled as "blocked" (which is reserved for
    /// off-thread stalls *during* active work).
    idle_ms: u64,
    /// Sync phase this window: deliberate GPU/persistence waits ([`WAIT_US`],
    /// swapped in at flush). Carved out of the compute phases + blocked so a
    /// device-sync / hot→warm flush stall is its own band, not hidden decode/prefill.
    wait_ms: u64,
    /// Persistence-maintenance stall this window ([`MAINT_US`], swapped in at
    /// flush) — a segment compaction blocking the scheduler's seal writes. Folded
    /// into the Sync band but carved blocked-only (never spills into compute).
    maint_ms: u64,
    /// Drain sub-timers (ms) + prefilled tokens, swapped from the `DRAIN_*` atoms.
    /// `prefill` (time + tokens) is re-attributed OUT of the Projection/drain band
    /// INTO the Prefill phase so the ingest prefill shows real throughput; `elevate`
    /// and `glue` decompose the remaining drain for the projection panel.
    drain_prefill_ms: u64,
    drain_prefill_tokens: u64,
    drain_elevate_ms: u64,
    drain_glue_ms: u64,
}

impl WaveStats {
    fn new() -> Self {
        Self {
            window_start: std::time::Instant::now(),
            prefill: WaveChannel::default(),
            decode: WaveChannel::default(),
            section: WaveChannel::default(),
            drain_ms: 0,
            promote_ms: 0,
            decode_ms: 0,
            prefill_ms: 0,
            section_ms: 0,
            reproj_ms: 0,
            seal_snapshot_us: 0,
            seal_sig_us: 0,
            seal_flush_us: 0,
            seal_count: 0,
            evict_bytes: 0,
            evict_count: 0,
            evict_ms: 0,
            idle_ms: 0,
            wait_ms: 0,
            maint_ms: 0,
            drain_prefill_ms: 0,
            drain_prefill_tokens: 0,
            drain_elevate_ms: 0,
            drain_glue_ms: 0,
        }
    }

    /// Accumulate one eviction event (relief-ladder shed of resident KV).
    fn add_evict(&mut self, bytes: u64, count: u64, ms: u64) {
        self.evict_bytes += bytes;
        self.evict_count += count;
        self.evict_ms += ms;
    }

    /// Accumulate one idle wait — the loop blocked on `rx.recv()` with no work.
    fn add_idle(&mut self, ms: u64) {
        self.idle_ms += ms;
    }

    /// Accumulate one scope-ingest seal's sub-step timings (microseconds).
    fn add_seal(&mut self, snapshot_us: u64, sig_us: u64, flush_us: u64) {
        self.seal_snapshot_us += snapshot_us;
        self.seal_sig_us += sig_us;
        self.seal_flush_us += flush_us;
        self.seal_count += 1;
    }

    /// Record one forward. `n_tokens` is the total tokens in the batch
    /// (seqs × per-seq chunk for prefill, seqs × 1 for decode). `kv_len` is the
    /// total attended-KV length the forward swept (Σ per-seq prefix/context
    /// length, captured before the forward advanced the sequences).
    fn record(
        &mut self,
        prefill: bool,
        n_seqs: usize,
        n_tokens: usize,
        kv_len: usize,
        fwd_ms: u64,
    ) {
        let ch = if prefill {
            &mut self.prefill
        } else {
            &mut self.decode
        };
        ch.fwds += 1;
        ch.seq_sum += n_seqs as u64;
        ch.seq_max = ch.seq_max.max(n_seqs);
        ch.tok_sum += n_tokens as u64;
        ch.kv_sum += kv_len as u64;
        ch.ms_sum += fwd_ms;
        // NB: do NOT flush here. Flushing mid-forward (record runs inside the
        // forward) would fire before the enclosing `timed_*` wrapper attributes
        // the phase ms, leaking the forward's wall-clock into `unaccounted`.
        // The run loop calls `flush_if_due` once per iteration, after all phase
        // attribution for that iteration is complete.
    }

    /// Record one co-batched section-ingest advance (its own throughput channel so
    /// the section panel isn't invisible behind decode). `fwd_ms` is the shared
    /// co-batch wave-step time — sections ride decode's sweep, so this reflects the
    /// concurrent throughput, not a serial section forward.
    fn record_section(&mut self, n_seqs: usize, n_tokens: usize, kv_len: usize, fwd_ms: u64) {
        let ch = &mut self.section;
        ch.fwds += 1;
        ch.seq_sum += n_seqs as u64;
        ch.seq_max = ch.seq_max.max(n_seqs);
        ch.tok_sum += n_tokens as u64;
        ch.kv_sum += kv_len as u64;
        ch.ms_sum += fwd_ms;
    }

    /// Accumulate wall-clock spent in a named scheduler-loop phase.
    fn add_phase(&mut self, phase: WavePhase, ms: u64) {
        match phase {
            WavePhase::Drain => self.drain_ms += ms,
            WavePhase::Promote => self.promote_ms += ms,
            WavePhase::Decode => self.decode_ms += ms,
            WavePhase::Prefill => self.prefill_ms += ms,
            WavePhase::Section => self.section_ms += ms,
            WavePhase::Reproject => self.reproj_ms += ms,
        }
    }

    /// Whether the 2 s wave window has elapsed and [`Self::flush`] should run.
    /// Split from `flush` so the caller only pays for the VRAM query (a couple
    /// of CUDA driver calls) on the wave it actually emits, not every loop
    /// iteration.
    fn due(&self) -> bool {
        self.window_start.elapsed() >= std::time::Duration::from_secs(2)
    }

    /// Emit the wave summary + phase breakdown, then reset. `kv_vram` is
    /// `(pool_budget_available, pool_used)` in bytes — the pool-accounting
    /// numbers our eviction gate keys on (NOT the driver's raw free, which the
    /// pool's retained reservation pins low). `None` on non-CUDA / query miss.
    /// `backlog` is the point-in-time prefill backlog in tokens
    /// ([`Scheduler::pending_prefill_tokens`]) — the signal the unified-wave
    /// engine's large-batch trigger keys on (see
    /// `docs/unified_wave_inference_engine.md` §4.5); the line reports
    /// *executed* forwards, so this is the only view of pending *work*.
    /// Call only when [`Self::due`] — windows with NO forwards still flush so
    /// stalls surface their phase split. `fmt` is the resident-arena format split
    /// `(float_arenas, float_reserved_mib, float_live_mib, quant_arenas,
    /// quant_reserved_mib, quant_live_mib)` for the arena panel. `vram` is the
    /// whole-card decomposition `(pool_reserved_mib, driver_total_mib,
    /// driver_free_mib)` for the VRAM-decomposition panel.
    fn flush(
        &mut self,
        kv_vram: Option<(usize, usize)>,
        backlog: u64,
        fmt: Option<(u32, u64, u64, u32, u64, u64)>,
        vram_decomp: (u64, u64, u64),
        slots: (u32, u32, u32, u32),
    ) {
        let elapsed = self.window_start.elapsed();
        let avg = |sum: u64, n: u64| if n > 0 { sum as f64 / n as f64 } else { 0.0 };
        let phase_sum =
            self.drain_ms + self.promote_ms + self.decode_ms + self.prefill_ms + self.section_ms;
        let unaccounted = (elapsed.as_millis() as u64).saturating_sub(phase_sum);
        // Only emit channels that actually ran this window — an all-zero
        // `decode`/`section` block during a pure-prefill insert was just noise.
        // `tok total` is the tokens fed this window; `kv/fwd avg` is the
        // attended-KV length each forward swept — watch it climb to spot a
        // context-growth slowdown (vs. a fixed paged-glue prefix staying flat).
        let mut parts: Vec<String> = Vec::new();
        let p = &self.prefill;
        if p.fwds > 0 {
            parts.push(format!(
                "prefill fwds={} seqs avg={:.1} max={} tok/fwd avg={:.0} tok total={} kv/fwd avg={:.0} kv total={} fwd avg={:.0}ms",
                p.fwds, avg(p.seq_sum, p.fwds), p.seq_max,
                avg(p.tok_sum, p.fwds), p.tok_sum, avg(p.kv_sum, p.fwds), p.kv_sum, avg(p.ms_sum, p.fwds),
            ));
        }
        let d = &self.decode;
        if d.fwds > 0 {
            parts.push(format!(
                "decode fwds={} seqs avg={:.1} max={} kv/fwd avg={:.0} fwd avg={:.0}ms",
                d.fwds,
                avg(d.seq_sum, d.fwds),
                d.seq_max,
                avg(d.kv_sum, d.fwds),
                avg(d.ms_sum, d.fwds),
            ));
        }
        let s = &self.section;
        if s.fwds > 0 {
            parts.push(format!(
                "section fwds={} seqs avg={:.1} max={} tok/fwd avg={:.0} tok total={} kv/fwd avg={:.0} kv total={} fwd avg={:.0}ms",
                s.fwds, avg(s.seq_sum, s.fwds), s.seq_max, avg(s.tok_sum, s.fwds), s.tok_sum, avg(s.kv_sum, s.fwds), s.kv_sum, avg(s.ms_sum, s.fwds),
            ));
        }
        let body = if parts.is_empty() {
            "(no forwards)".to_string()
        } else {
            parts.join(" | ")
        };
        // The pool budget our eviction defends and our pool_used — watch
        // `budget` fall toward the band (eviction fires) and `used` cap out
        // rather than climb unbounded.
        let vram = match kv_vram {
            Some((budget, used)) => format!(
                " | kv-vram budget={}MiB used={}MiB",
                budget / (1 << 20),
                used / (1 << 20),
            ),
            None => String::new(),
        };
        // Prefill backlog (pending, not-yet-executed tokens) — the large-batch
        // trigger signal. Only emitted when non-zero to keep the line quiet at
        // idle.
        let backlog_str = if backlog > 0 {
            format!(" | backlog={backlog}tok")
        } else {
            String::new()
        };
        tracing::info!(
            "wave {:.1}s: {body}{vram}{backlog_str}",
            elapsed.as_secs_f64()
        );
        // Phase breakdown: where the wall-clock went on the scheduler thread.
        // `drain` rising over the run ⇒ per-turn reprojection/elevate growing;
        // `reproj` rising ⇒ continuous-reproject (provenance scan/glue) growing;
        // `unaccounted` large ⇒ blocked off-thread (persistence thread / lock).
        // Detailed per-wave breakdown — the live GUI panels carry the same numbers,
        // so this stays at debug and the `wave {}s` heartbeat above is the info line.
        tracing::debug!(
            target: "candle_conversation::scheduler::timing",
            drain_ms = self.drain_ms,
            promote_ms = self.promote_ms,
            decode_ms = self.decode_ms,
            prefill_ms = self.prefill_ms,
            section_ms = self.section_ms,
            reproj_ms = self.reproj_ms,
            unaccounted_ms = unaccounted,
            // Sub-slices INSIDE prefill_ms (not additive to the phases above):
            // where the non-forward per-scope seal cost goes during code-read.
            seal_count = self.seal_count,
            seal_snapshot_ms = self.seal_snapshot_us / 1000,
            seal_sig_ms = self.seal_sig_us / 1000,
            seal_flush_ms = self.seal_flush_us / 1000,
            // seal_sig split: host pointer-resolve / GPU kernel(HtoD+launch+D2H) / host fold.
            prov_resolve_ms =
                PROV_RESOLVE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            prov_kernel_ms = PROV_KERNEL_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            prov_assemble_ms =
                PROV_ASSEMBLE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            "wave phase breakdown (scheduler-thread wall-clock; watch which grows)"
        );
        // Swap in this window's accumulated GPU/persistence wait for the Sync phase,
        // plus the drain sub-timers (prefill re-attributed to Prefill; elevate/glue
        // decompose the drain).
        self.wait_ms = WAIT_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.maint_ms = MAINT_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_prefill_ms =
            DRAIN_PREFILL_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_prefill_tokens =
            DRAIN_PREFILL_TOKENS.swap(0, std::sync::atomic::Ordering::Relaxed);
        self.drain_elevate_ms =
            DRAIN_ELEVATE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_glue_ms = DRAIN_GLUE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        // Feed the live GUI's phase timeline: one measurement per phase that ran
        // this window. Volume is tokens for the inference phases and bytes for the
        // memory phases; the GUI colors by kind and shows the flip over time.
        self.push_phase_window(elapsed.as_millis() as u32);
        // Feed the instrumented generic panels (VRAM / throughput / backlog / wave
        // latency / arenas) — the same numbers the log line carries, straight from
        // the ring so the dashboard needs no log tail.
        let tok = self.prefill.tok_sum
            + self.decode.tok_sum
            + self.section.tok_sum
            + self.drain_prefill_tokens;
        let fwds = self.prefill.fwds + self.decode.fwds + self.section.fwds;
        let fwd_sum = self.prefill.ms_sum + self.decode.ms_sum + self.section.ms_sum;
        let fwd_ms = if fwds > 0 { fwd_sum / fwds } else { 0 };
        let (budget_mib, used_mib) = kv_vram
            .map(|(b, u)| ((b / (1 << 20)) as u64, (u / (1 << 20)) as u64))
            .unwrap_or((0, 0));
        // Projection decomposition. The drain-path PREFILL is re-attributed to the
        // Prefill phase (below), so the projection band's drain contribution is
        // `drain_ms - drain_prefill`, decomposed into elevate / glue / other.
        // Reproject decomposes into scan / glue / layout / other.
        let use_ms = |a: &std::sync::atomic::AtomicU64| {
            a.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000
        };
        let pdrain = self.drain_ms.saturating_sub(self.drain_prefill_ms);
        let proj = (
            pdrain,
            self.drain_elevate_ms.min(pdrain),
            self.drain_glue_ms.min(pdrain),
            self.reproj_ms,
            use_ms(&REPROJ_SCAN_US),
            use_ms(&REPROJ_GLUE_US),
            use_ms(&REPROJ_LAYOUT_US),
        );
        phase_ring::push_wave(phase_ring::wave_sample(
            elapsed.as_millis() as u32,
            fwd_ms as u32,
            tok,
            budget_mib,
            used_mib,
            backlog,
            fmt,
            vram_decomp,
            proj,
            slots,
        ));
        self.window_start = std::time::Instant::now();
        self.prefill = WaveChannel::default();
        self.decode = WaveChannel::default();
        self.section = WaveChannel::default();
        self.drain_ms = 0;
        self.promote_ms = 0;
        self.decode_ms = 0;
        self.prefill_ms = 0;
        self.section_ms = 0;
        self.reproj_ms = 0;
        self.seal_snapshot_us = 0;
        self.seal_sig_us = 0;
        self.seal_flush_us = 0;
        self.seal_count = 0;
        self.evict_bytes = 0;
        self.evict_count = 0;
        self.evict_ms = 0;
        self.idle_ms = 0;
        self.wait_ms = 0;
        self.maint_ms = 0;
        self.drain_prefill_ms = 0;
        self.drain_prefill_tokens = 0;
        self.drain_elevate_ms = 0;
        self.drain_glue_ms = 0;
    }

    /// Emit this window's per-phase measurements to the GUI ring as a **disjoint**
    /// decomposition of the wave's wall-clock — the phase durations sum to
    /// `window_ms`, so the GUI can stack them as "time in phase" without
    /// over-counting.
    ///
    /// The reliable top-level split is the five timed quanta (drain, promote,
    /// decode, prefill, section) plus the leftover "blocked". Two costs are
    /// SUB-SLICES that run *inside* those quanta and would double-count if drawn as
    /// their own segment on top:
    /// - `reproj_ms` runs inside the decode quantum → carved out of decode into
    ///   Projection.
    /// - `seal_ms` runs inside prefill (`complete_turn_reprefill`) and decode
    ///   (`perform_seal_and_write`) → carved out of those into Sealing.
    /// - `evict_ms` runs partly in the flush block (reclaim/demote → the blocked
    ///   remainder) and partly inside the quanta (relief) → carved out, blocked
    ///   first, into Eviction.
    ///
    /// Carving redistributes ms between buckets (never adds), so the sum is
    /// preserved at `window_ms`.
    fn push_phase_window(&self, window_ms: u32) {
        use phase_ring::{PhaseKind, PhaseMeasure};

        // Disjoint base buckets (sum to window_ms). The drain-path prefill (the
        // ingest turn's own content forward) is real prefill work that happens
        // inside `drain_submissions`, so move its time OUT of the projection band
        // and INTO Prefill — with its tokens (below) — so ingest throughput isn't
        // hidden as token-less projection time.
        let drain_prefill = self.drain_prefill_ms.min(self.drain_ms);
        let proj_dur = self.drain_ms.saturating_sub(drain_prefill) + self.reproj_ms;
        let mut decode_dur = self.decode_ms.saturating_sub(self.reproj_ms);
        // Continuous-fair-wave co-batching folds the prefill cohort and section
        // chunks INTO the decode quantum — one shared forward per wave — so their
        // PHASE wall-clock (`prefill_ms`/`section_ms`) is ~0; the time lives in
        // `decode_ms`. The panel derives each phase's tok/s as `tok / dur`, so a
        // per-phase duration of 0 renders prefill as an invisible zero-width bar,
        // while stealing time from `decode_dur` to give prefill a slice inflates the
        // decode rate (fewer ms for the same decode tokens). Instead give prefill
        // and section their OWN co-batched forward time — the channel `ms_sum`
        // recorded per wave in `decode_forward_cobatched` — as the display duration.
        // These overlap the decode quantum (they ran concurrently inside it), which
        // the stacked timeline normalizes to its own sum; the win is that `tok / dur`
        // yields each class's real CONCURRENT rate, and decode keeps its full
        // duration so its rate stays truthful.
        let mut prefill_dur = self.prefill.ms_sum.max(self.prefill_ms) + drain_prefill;
        let mut section_dur = self.section.ms_sum.max(self.section_ms);
        let alloc_dur = self.promote_ms;
        let accounted =
            self.drain_ms + self.promote_ms + self.decode_ms + self.prefill_ms + self.section_ms;
        let mut blocked_dur = (window_ms as u64).saturating_sub(accounted);

        // Sealing lives inside prefill + decode; carve it out (prefill first).
        let seal_ms = (self.seal_snapshot_us + self.seal_sig_us + self.seal_flush_us) / 1000;
        let seal_dur = carve_ms(seal_ms, &mut [&mut prefill_dur, &mut decode_dur]);
        // Eviction lives in the flush block (blocked) + relief inside the quanta;
        // carve blocked first, then the quanta.
        let evict_dur = carve_ms(
            self.evict_ms,
            &mut [&mut blocked_dur, &mut decode_dur, &mut prefill_dur],
        );
        // Sync (deliberate GPU `synchronize` + persistence `flush_blocking` waits)
        // happens both in the flush block (→ blocked remainder) and inside the
        // quanta (relief mid-decode/prefill) → carve blocked first, then the
        // compute quanta, so a multi-second stall reads as Sync rather than
        // inflating decode/prefill or hiding in blocked.
        let mut sync_dur = carve_ms(
            self.wait_ms,
            &mut [
                &mut blocked_dur,
                &mut decode_dur,
                &mut prefill_dur,
                &mut section_dur,
            ],
        );
        // Persistence-maintenance stall (a segment compaction blocking seal writes)
        // is off-thread blocked time, so it carves ONLY from the blocked remainder
        // — never spilling into decode/prefill/section like an on-thread GPU sync
        // can. Any excess over this window's blocked is dropped rather than wrongly
        // shrinking a compute phase's duration (which would inflate its tok/s).
        sync_dur += carve_ms(self.maint_ms, &mut [&mut blocked_dur]);
        // Idle (loop blocked on `rx.recv()` with no work) is unaccounted time, so
        // it sits in the blocked remainder — carve it out so a scheduler sitting
        // idle between requests reads as Idle, not Blocked. What's left in
        // `blocked_dur` is genuinely unattributed remainder (lock contention, the
        // cheap on-thread flush-block housekeeping).
        let idle_dur = carve_ms(self.idle_ms, &mut [&mut blocked_dur]);

        let mut phases: Vec<PhaseMeasure> = Vec::new();
        let mut inference = |kind: PhaseKind, ch: &WaveChannel, dur_ms: u64| {
            if ch.fwds == 0 && dur_ms == 0 {
                return;
            }
            phases.push(PhaseMeasure {
                kind,
                dur_ms: dur_ms as u32,
                tokens: ch.tok_sum,
                bytes: 0,
                seqs: ch.seq_max as u32,
                count: ch.fwds as u32,
            });
        };
        inference(PhaseKind::Decode, &self.decode, decode_dur);
        inference(PhaseKind::Section, &self.section, section_dur);
        // Prefill emitted directly (not via `inference`) so the drain-path ingest
        // prefill's tokens are folded in with the async-queue prefill's — matching
        // the `drain_prefill` time already folded into `prefill_dur`.
        let prefill_tok = self.prefill.tok_sum + self.drain_prefill_tokens;
        let prefill_seqs = self
            .prefill
            .seq_max
            .max(if drain_prefill > 0 { 1 } else { 0 });
        if self.prefill.fwds > 0 || prefill_dur > 0 || prefill_tok > 0 {
            phases.push(PhaseMeasure {
                kind: PhaseKind::Prefill,
                dur_ms: prefill_dur as u32,
                tokens: prefill_tok,
                bytes: 0,
                seqs: prefill_seqs as u32,
                count: self.prefill.fwds as u32,
            });
        }
        let push = |phases: &mut Vec<PhaseMeasure>, kind, dur: u64, bytes, count| {
            phases.push(PhaseMeasure {
                kind,
                dur_ms: dur as u32,
                tokens: 0,
                bytes,
                seqs: 0,
                count,
            });
        };
        if proj_dur > 0 {
            push(&mut phases, PhaseKind::Projection, proj_dur, 0, 0);
        }
        if self.seal_count > 0 || seal_dur > 0 {
            push(
                &mut phases,
                PhaseKind::Sealing,
                seal_dur,
                0,
                self.seal_count as u32,
            );
        }
        if self.evict_count > 0 || self.evict_bytes > 0 {
            push(
                &mut phases,
                PhaseKind::Eviction,
                evict_dur,
                self.evict_bytes,
                self.evict_count as u32,
            );
        }
        if alloc_dur > 0 {
            push(&mut phases, PhaseKind::Allocation, alloc_dur, 0, 0);
        }
        if sync_dur > 0 {
            push(&mut phases, PhaseKind::Sync, sync_dur, 0, 0);
        }
        if idle_dur > 0 {
            push(&mut phases, PhaseKind::Idle, idle_dur, 0, 0);
        }
        if blocked_dur > 0 {
            push(&mut phases, PhaseKind::Blocked, blocked_dur, 0, 0);
        }
        phase_ring::push_window(window_ms, phases);
    }
}

/// Carve a sub-slice of `amt` ms out of `buckets` in priority order, draining each
/// before moving to the next. Returns the amount actually carved — bounded by the
/// total the buckets hold, so a carved-out phase segment can never push the stacked
/// decomposition past the window. Redistributes ms (subtracts from a bucket, hands
/// it to the caller's segment), so the overall sum across buckets + segment is
/// preserved.
fn carve_ms(amt: u64, buckets: &mut [&mut u64]) -> u64 {
    let mut rem = amt;
    let mut taken = 0;
    for b in buckets.iter_mut() {
        let t = rem.min(**b);
        **b -= t;
        rem -= t;
        taken += t;
        if rem == 0 {
            break;
        }
    }
    taken
}

/// Named scheduler-loop phases for [`WaveStats::add_phase`].
#[derive(Clone, Copy)]
enum WavePhase {
    Drain,
    Promote,
    Decode,
    Prefill,
    Section,
    Reproject,
}

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
    /// EOS token ID (stops generation).
    eos_tokens: TokenBuffer,
    /// Device the model lives on.
    device: Device,
    /// Active decode state per sequence ID.
    active_decodes: HashMap<SequenceId, DecodeState>,
    /// Persistent sampling state per sequence ID (survives across turns).
    /// DRY penalty needs the recent-token window to span turn boundaries.
    sampling_states: HashMap<SequenceId, SequenceSamplingState>,
    /// Prefill queue (FIFO) — newly submitted, not yet started.
    prefill_queue: VecDeque<PrefillWork>,
    /// In-flight prefills (partially advanced across loop iterations).
    /// Promoted from `prefill_queue` by `promote_new_prefills` and drained
    /// by `promote_finished_prefills_to_decodes` once their offset reaches
    /// `work.tokens.len()`.
    pub(super) active_prefills: Vec<ActivePrefill>,
    /// In-flight section ingests (batched prefill-only, no decode).
    /// Populated from [`SchedulerRequest::IngestSection`] after the cheap
    /// CPU setup (truncate, prefix inject, push writer chunk, pin tokens)
    /// completes.  Drained by [`Self::run_one_section_ingest_chunk`] once
    /// each entry's offset reaches its token count, then finalised by
    /// [`Self::finalize_done_section_ingests`].
    pub(super) active_section_ingests: Vec<ActiveSectionIngest>,
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
    /// Each conversation's belief as of its last completed turn, keyed by the
    /// conversation's parent slot. Harvested in `cleanup_finished` when a turn
    /// seals, and seeded into the NEXT turn's submit-time projection and
    /// decode state — the belief is conversation-state that evolves across
    /// turns, never resetting at a turn boundary (a tool-retry turn opens with
    /// the prior turn's committed tool already materialized, not the catalog
    /// fallback). In-memory only: a daemon restart begins from an empty belief
    /// and the first turn's reprojections rebuild it.
    carried_beliefs: HashMap<SequenceId, PriorBelief>,
    /// Pending mid-decode view swaps, queued during `batch_decode_step`
    /// (which holds shared/exclusive borrows on `active_decodes`) and
    /// drained immediately after the batch completes.  Values are
    /// `view_id`s whose decoded count crossed an `every_n_tokens` boundary
    /// this batch.
    pending_reprojections: Vec<SequenceId>,

    /// Per-slot `Conversation` (workspace) registry.  Registered when
    /// [`SchedulerRequest::NewSequence`] allocates the slot; removed
    /// when [`SchedulerRequest::FreeSequence`] frees it.  The
    /// [`SchedulerRequest::SubmitTurn`] handler resolves the
    /// conversation by `parent_id` so the projection step reads from
    /// the right substrate — the scheduler hosts conversations on
    /// many substrates concurrently.
    slot_conversations: HashMap<SequenceId, Conversation>,

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

    /// Trigger handle for the substrate persistence thread. Fired
    /// after every turn-seal so the thread runs its hot→warm→cold
    /// drain promptly instead of waiting up to 5 s on its tick.
    persist_trigger: PersistenceTrigger,

    /// Trigger handle for the async summariser thread.  Fired after
    /// every turn-seal (`docs/archived/infinite_conversations.md` §4 step ③)
    /// so the freshly-pending Normal turn is absorbed into the immutable
    /// summary forest (`docs/immutable_summary_forest.md`) on the next
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

    /// Re-used pinned host scratch for the cold→hot leg inside
    /// `elevate_to_hot`. Grows on demand across submits; stays
    /// allocated for the scheduler's lifetime.
    ///
    /// The elevate path itself runs on the device's **main inference
    /// stream** (`dev.cuda_stream()`) — the scheduler thread is
    /// single-owner and is blocking on the result before
    /// `apply_projection` runs prefill/decode, so a dedicated stream
    /// would only add a stream-sync without buying any overlap.
    /// Scattering on the main stream lets the subsequent prefill
    /// kernels see the writes for free (same-stream serialisation).
    elevate_pinned_scratch: Option<PinnedBuf>,

    /// Pre-tokenised inter-turn boundary markers (`user_start` /
    /// `assistant_end`) for the engine's dialect.  Built once at
    /// engine construction and passed into the scheduler; the
    /// projection assembler reads them via `ApplyContext` to wrap
    /// every `Sealed::Turn` segment in a live-prefilled boundary
    /// run, and the `SubmitTurn` handler reads `user_start` to
    /// append the trailing `Generated(UserStart)` ahead of the
    /// current turn's prefill.
    boundary_markers: projection_assembler::BoundaryMarkers,

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

    /// Finished dialogue turns whose reasoning-free tokens are re-prefilling on
    /// the shared wave, keyed by `pending_id`. Drained when the prefill completes
    /// — see [`PendingTurnSeal`] and `complete_turn_reprefill`.
    pending_turn_seals: HashMap<u64, PendingTurnSeal>,

    /// Monotonic id source for `pending_turn_seals`.
    next_turn_seal_id: u64,

    /// Per-file parallel scope ingests, keyed by file `timeline`. Each holds the
    /// file's total scope count, the per-scope reply channels, and the scopes'
    /// snapshotted K/V as they land; `advance_scope_flush` records each landed
    /// contiguous prefix and retires the entry once the file is done. See
    /// [`PendingScopeBatch`].
    scope_batches: HashMap<TimelineId, PendingScopeBatch>,

    /// Scopes awaiting a scratch slot, bucketed by file `timeline`.
    /// `pump_scope_prefills` drains them fairly (least-advanced file first),
    /// bounded by [`Self::MAX_SCOPE_SLOTS`]. See [`QueuedScope`].
    scope_pending: HashMap<TimelineId, VecDeque<QueuedScope>>,

    /// How many scopes each file has had pumped onto a scratch slot so far — the
    /// max-min fairness key: the pump always advances the least-advanced file.
    scope_submitted: HashMap<TimelineId, u32>,

    /// Scope prefills in flight on the wave, keyed by scratch slot. Drained by
    /// `complete_scope_ingest` when the prefill finishes. See [`PendingScopePrefill`].
    pending_scope_prefills: HashMap<SequenceId, PendingScopePrefill>,

    /// Live scratch slots held by in-flight scope prefills — the bound the pump
    /// respects so a wide fan-out never exhausts the model's sequence slots.
    active_scope_slots: usize,

    /// AIMD congestion window over concurrent prefill admission — the dynamic
    /// ceiling both [`Self::promote_new_prefills`] (forward width) and
    /// [`Self::pump_scope_prefills`] (pinned scope working set) clamp to, on top
    /// of their static caps ([`Self::MAX_PREFILL_WIDTH`] / [`Self::MAX_SCOPE_SLOTS`]).
    ///
    /// A wide ragged prefill forward's transient VRAM peak (MoE expert gather +
    /// per-sequence activations) scales with the batch width but isn't visible at
    /// admission time, so a fixed cap that's fine on an idle card OOMs a busy one.
    /// The window closes multiplicatively — halving toward [`Self::MIN_PREFILL_WIDTH`]
    /// — whenever VRAM pressure survives an eviction pass or a forward actually
    /// reports device-OOM, and reopens additively (one slot per idle-ish wave) once
    /// pressure clears. Under sustained pressure it converges toward 1 (throughput
    /// floor that still makes progress); with headroom it returns to the full width.
    admit_window: usize,

    /// Cached OS memory probe for host-tier ingest backpressure, as
    /// `(checked_at, available_bytes, total_bytes)`. The warm (RAM) KV tier can
    /// fill host memory faster than warm→cold demotion drains it; when free RAM
    /// runs low, `regulate_ingest_admission` throttles ingest so the hot→warm
    /// migration always finds a staging buffer it can allocate (the host-OOM that
    /// aborted a full overnight load). `sysinfo` is a syscall, so this is
    /// refreshed at most once per `HOST_RAM_PROBE_INTERVAL`, never per wave.
    host_ram_probe: Option<(std::time::Instant, u64, u64)>,

    /// When the footprint reclaim last ran, to rate-limit it. Without this, a
    /// `reserved` pinned just over the compact-ceiling (a fragmented gap the
    /// engine keeps reusing, which compaction can't lower) trips the pressure
    /// gate on every scheduler-loop iteration and fires relief many times/second.
    /// See `Scheduler::reclaim_footprint` / `vram_under_pressure_for`.
    last_footprint_relief: Option<std::time::Instant>,

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

    /// Continuous-fair-wave prefill cohort (`docs/continuous_fair_waves.md`): the
    /// packed inter-layer residual stream of the in-flight prefill batch, the next
    /// layer it resumes at, and the sequences in the cohort. One cohort advances
    /// in lockstep — held across waves so decode sweeps every layer between the
    /// cohort's throttled layer advances, keeping the decode expert set hot.
    /// `cursor == 0` with no residual means no cohort is in flight.
    wave_prefill_residual: Option<Tensor>,
    wave_prefill_cursor: usize,
    /// The in-flight wave group's members in a STABLE order (dialogue prefills
    /// then section chunks). The order must not change between waves: `forward_wave`
    /// re-partitions the same inputs into the same internal order each resume, and
    /// the held residual only lines up if the input order is identical. Fixed at
    /// formation — new prefills / sections wait for the next group.
    wave_prefill_members: Vec<WaveMember>,
    /// Set once per decode quantum after the co-batched decode wave advances the
    /// prefill cohort, so only the FIRST decode step of the quantum folds the
    /// cohort in (one budget-sized layer advance per wave, not per decode step).
    wave_cohort_advanced: bool,
    /// Set once per wave after the co-batched decode wave folded the active
    /// section-ingest chunk into its full sweep (section rides decode's `[0, N)`
    /// as a prefill-group member — one shared MoE grouped GEMM per layer serves
    /// decode + section + cohort). Skips the standalone `run_one_section_ingest_chunk`
    /// pass that wave (`docs/continuous_fair_waves.md` §4).
    wave_section_advanced: bool,

    /// Cache of `SectionId` → symbolic `debug_name`, so the promote tracker's name
    /// lookup is O(1) after the first sighting (sections are stable).
    section_name_cache: HashMap<SectionId, String>,

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

impl Scheduler {
    /// Create a new scheduler. Called once by `ConversationEngine`.
    pub fn new(
        rx: Receiver<SchedulerRequest>,
        model: Box<dyn ManagedBatchedModel + Send>,
        session: BatchedInferenceSession,
        tokenizer: tokenizers::Tokenizer,
        eos_tokens: TokenBuffer,
        vocab_size: usize,
        max_recent_len: usize,
        show_special_tokens: bool,
        penalty_log_path: Option<PathBuf>,
        health_config: DecodeHealthConfig,
        max_prefill_pass_tokens: usize,
        persist_trigger: PersistenceTrigger,
        summariser_trigger: SummariserTrigger,
        boundary_markers: projection_assembler::BoundaryMarkers,
    ) -> Self {
        let device = model.device().clone();

        // Force this thread to bind to the device's CUDA context
        // BEFORE we allocate pinned host memory below. The model and
        // session were created on another thread; the CUDA context is
        // per-thread on Windows, and `cuMemHostAlloc` returns
        // `CUDA_ERROR_NOT_INITIALIZED` if called from a thread that
        // hasn't bound to a context yet. A failed alloc silently falls
        // back to non-pinned heap memory, which turns every cold-load
        // HtoD into a synchronous copy at ~1 GB/s instead of an
        // async-pinned DMA at PCIe rate.
        if let Device::Cuda(d) = &device {
            let _ = d.cuda_context().bind_to_thread();
        }

        let sampler = BatchedSampler::new(
            device.clone(),
            vocab_size,
            max_recent_len,
            eos_tokens.clone(),
            penalty_log_path,
        );

        let chunk_size = CHUNK_SIZE;
        Self {
            rx,
            model,
            session,
            tokenizer,
            eos_tokens,
            device,
            active_decodes: HashMap::new(),
            sampling_states: HashMap::new(),
            prefill_queue: VecDeque::new(),
            active_prefills: Vec::new(),
            active_section_ingests: Vec::new(),
            sampler,
            show_special_tokens,
            health_config,
            chunk_size,
            max_prefill_pass_tokens,
            slot_conversations: HashMap::new(),
            slot_targets: HashMap::new(),
            ephemeral_slots: std::collections::HashSet::new(),
            ephemeral_sigs: HashMap::new(),
            turn_views: HashMap::new(),
            carried_beliefs: HashMap::new(),
            pending_reprojections: Vec::new(),
            slot_tokens: HashMap::new(),
            slot_projection_state: HashMap::new(),
            cold_load_stager: ColdLoadStager::with_preallocation(PINNED_PREALLOC_BYTES),
            persist_trigger,
            summariser_trigger,
            pending_section_quantize: Vec::new(),
            elevate_pinned_scratch: preallocate_pinned_scratch(
                PINNED_PREALLOC_BYTES,
                "scheduler::elevate_pinned_scratch",
            ),
            boundary_markers,
            wave_stats: WaveStats::new(),
            compression_jobs: HashMap::new(),
            pending_compression_seals: HashMap::new(),
            next_compression_job_id: 0,
            pending_turn_seals: HashMap::new(),
            next_turn_seal_id: 0,
            scope_batches: HashMap::new(),
            scope_pending: HashMap::new(),
            scope_submitted: HashMap::new(),
            pending_scope_prefills: HashMap::new(),
            active_scope_slots: 0,
            admit_window: Self::MAX_PREFILL_WIDTH,
            host_ram_probe: None,
            last_footprint_relief: None,
            ingest_timelines: HashSet::new(),
            batch_drain_gap_fills: false,
            deferred_glue_fires: Vec::new(),
            wave_prefill_residual: None,
            wave_prefill_cursor: 0,
            wave_prefill_members: Vec::new(),
            wave_cohort_advanced: false,
            wave_section_advanced: false,
            section_name_cache: HashMap::new(),
            compression_event_sinks: HashMap::new(),
            timeline_projections: HashMap::new(),
            workspace_projection: None,
        }
    }

    /// Append `tokens` to slot `slot`'s diagnostic token log used by
    /// the `context-dump` feature.  Compiled to a no-op when the
    /// feature is disabled so callers can sprinkle the call freely.
    ///
    /// Takes `&mut HashMap` directly instead of `&mut self` so callers
    /// can hold another mutable borrow into a different `Scheduler`
    /// field (e.g. `active_prefills[i]`) simultaneously.  Use the
    /// associated-function form: `Self::record_slot_tokens(&mut self.slot_tokens, slot, tokens)`.
    #[inline(always)]
    pub(crate) fn record_slot_tokens(
        slot_tokens: &mut HashMap<SequenceId, Vec<u32>>,
        slot: SequenceId,
        tokens: &[u32],
    ) {
        #[cfg(feature = "context-dump")]
        slot_tokens
            .entry(slot)
            .or_default()
            .extend_from_slice(tokens);
        #[cfg(not(feature = "context-dump"))]
        {
            let _ = (slot_tokens, slot, tokens);
        }
    }

    // —— Submission handling —————————————————————————————————————————————

    /// Drain all pending submissions. Returns `false` if shutdown requested.
    /// Point-in-time prefill **backlog** in tokens: not-yet-processed tokens
    /// across the queued FIFO plus the in-flight prefills and section ingests.
    /// The unified-wave large-batch trigger (design §4.5) reads this; today it
    /// only feeds the wave log line. Cheap (a few small iterations), computed
    /// once per emitted wave.
    pub(super) fn pending_prefill_tokens(&self) -> u64 {
        let queued = self
            .prefill_queue
            .iter()
            .map(|w| (w.tokens.token_count(), 0));
        let active = self
            .active_prefills
            .iter()
            .map(|p| (p.work.tokens.token_count(), p.offset));
        let sections = self
            .active_section_ingests
            .iter()
            .map(|s| (s.tokens.token_count(), s.offset));
        sum_pending_prefill_tokens(queued.chain(active).chain(sections))
    }

    fn drain_submissions(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(req) => {
                    if !self.handle_request(req) {
                        return false;
                    }
                }
                Err(crossbeam::channel::TryRecvError::Empty) => return true,
                Err(crossbeam::channel::TryRecvError::Disconnected) => return false,
            }
        }
    }

    /// Handle a single request. Returns `false` if shutdown.
    fn handle_request(&mut self, req: SchedulerRequest) -> bool {
        match req {
            SchedulerRequest::NewSequence {
                conversation,
                target,
                response_tx,
            } => {
                let result = self.create_sequence(conversation, target);
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::NewEphemeralSequence {
                conversation,
                target,
                response_tx,
            } => {
                // Bind the target (so `apply_projection` runs → warm system
                // prompt), then mark ephemeral so the turn never seals.
                let result = self.create_sequence(conversation, Some(target));
                if let Ok(slot) = &result {
                    self.ephemeral_slots.insert(*slot);
                }
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ResumeSequence {
                conversation,
                timeline,
                response_tx,
            } => {
                // Look up `(layer, group)` from the substrate registry
                // to construct the slot's target, then create the
                // sequence as if it were a fresh `NewSequence` —
                // `slot_targets` ends up the same shape regardless of
                // which entry point produced it.
                let result = match conversation.timeline_target(timeline) {
                    Some((layer, group)) => {
                        let target = ProjectionTarget {
                            layer,
                            group,
                            timeline,
                        };
                        self.create_sequence(conversation, Some(target))
                    }
                    None => Err(ConversationError::Channel(format!(
                        "ResumeSequence: timeline {timeline} not registered in substrate",
                    ))),
                };
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::SubmitTurn {
                sequence_id,
                projection_inputs,
                prefill_tokens,
                prefill_text,
                user_text,
                tags,
                user_content_start,
                user_content_end,
                assistant_content_start,
                no_think,
                projection_offsets,
                prefill_assistant_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
                triggers,
            } => {
                // The sequence acts as the parent slot for a carved
                // view inside this handler — rebind for clarity.
                let parent_id = sequence_id;
                // Derive the post-Done seal directly from the
                // projection inputs.  When projection is supplied AND
                // the slot has a registered target the request is by
                // definition a turn submission and the result lands in
                // the substrate at `slot_targets[parent_id]`.  When
                // either is absent the request is a raw prefill+decode
                // (RULER, summarisation) and no substrate write
                // happens.  Section ingestion uses
                // [`SchedulerRequest::IngestSection`] instead.
                let slot_target = self.slot_targets.get(&parent_id).copied();
                // Capture this timeline's projection Builder so the summariser's
                // compression probes can read the target layer's `summary`.
                if let (Some(inputs), Some(tgt)) = (&projection_inputs, slot_target.as_ref()) {
                    self.timeline_projections
                        .insert(tgt.timeline, inputs.projection.clone());
                    self.workspace_projection
                        .get_or_insert_with(|| inputs.projection.clone());
                }
                let seal_action = if self.ephemeral_slots.contains(&parent_id) {
                    // Ephemeral probe: project (warm) but NEVER write back — the
                    // query's wide-Q is gathered at completion instead of sealed.
                    SealAction::None
                } else {
                    match (&projection_inputs, slot_target) {
                        (Some(_), Some(_)) => SealAction::Turn,
                        _ => SealAction::None,
                    }
                };
                // Append-only ingests (e.g. code_reading) opt out of the
                // per-turn projection rebuild once their slot is seeded:
                // skip the reset + re-project and prefill straight onto the
                // cumulative slot. The seal is unaffected — turns still land
                // in the substrate.
                //
                // The system prompt is laid down at conversation creation by
                // PrimingProjection (see `Conversation::new_*`), so the slot
                // already has blocks before turn 1 and `skip_projection` is
                // true from the first turn — `apply_projection` never runs for
                // these ingests. That's intentional: priming injects the
                // `fixed_prefix` (system) sections. The trade-off is that
                // collection-member / `depends_on`-gated sections (which only
                // materialize via per-turn `apply_projection`) are NOT added;
                // utility-layer system prompts (code_reading, repo_map) use
                // only fixed sections, so this is exact for them.
                let skip_projection = disable_reprojection
                    && self.session.sequence_block_count(parent_id.0).unwrap_or(0) > 0;
                // Mark this as an append-only ingest timeline so the gentle-early
                // ladder can demote its sealed, warm-backed KV to RAM at ~50%
                // capacity — it is never re-attended until query time. See
                // `demote_cold_ingest_if_pressured`.
                if disable_reprojection {
                    if let Some(tgt) = slot_target {
                        self.ingest_timelines.insert(tgt.timeline);
                    }
                }
                // Step 1: run projection (if requested) and apply it
                // — reset `parent_id` to empty and write the
                // projected sections + projected turns from the
                // substrate onto it.
                //
                // `turn_keys_for_elevate` is the parallel
                // `Vec<TurnKey>` we feed to `elevate_to_hot` before
                // `apply_projection`. Built inside the same closure
                // so the resolved `timeline` for each `(g, t)` pair
                // doesn't get thrown away.
                let mut turn_keys_for_elevate: Vec<TurnKey> = Vec::new();
                // Captured from the projection if it ran the
                // score-density path; written to the substrate's
                // side-channel below, after the read guard drops.
                let mut diag_to_write: Option<(TimelineId, SelectionDiagnostics)> = None;
                // Staged calibration prefill: one projection composition, cloned
                // per segment with its own span, emitted as the wave crosses each
                // `projection_offsets` point. Computed inside the projection block
                // (below) while the view read guard is live; `None` for a normal
                // prefill.
                let mut staged_composition: Option<crate::projection::ProjectionEvent> = None;
                // The turn's opening belief — assigned by the projection path
                // below (carried belief stepped through the submit projection);
                // default when projection is skipped.
                let mut turn_belief = PriorBelief::default();
                let (projected_sections, projected_segments) = if let (Some(inputs), Some(target)) = (
                    projection_inputs.as_ref().filter(|_| !skip_projection),
                    slot_target,
                ) {
                    let conversation = match self.slot_conversations.get(&parent_id) {
                        Some(c) => c.clone(),
                        None => {
                            let _ = event_tx.send(TurnEvent::Error(ConversationError::Channel(
                                format!(
                                    "submit_turn: no conversation registered for slot {parent_id}"
                                ),
                            )));
                            return true;
                        }
                    };
                    // Target-aware read: the projection sees only
                    // `target.timeline` within `target.group`,
                    // masking sibling timelines.
                    let view = conversation.read_for(target);
                    // Prefill mode: section scoring uses the calibrated
                    // prefill profile (Max / semantic depth, no threshold)
                    // against the prefill-Q section corpus.  The sink
                    // captures score-density diagnostics for the test
                    // harness (§10.8.4) on a substrate-side side-channel.
                    // Last-write-wins per timeline; the recall test reads
                    // it via `Conversation::last_selection_diagnostics`
                    // after `send_turn` returns.  Sink is never invoked
                    // when the rule-based path runs (no tree on the
                    // target timeline).
                    // Submit-time projection seeds from the conversation's belief
                    // as of its last completed turn (`carried_beliefs`) — empty
                    // only for a genuinely fresh conversation. The belief then
                    // evolves through this turn's reprojections and is harvested
                    // back at seal, so provenance selection is continuous across
                    // turn boundaries instead of resetting to catalog order.
                    //
                    // Decayed at the boundary so it seeds as a SOFT PRIOR, not a
                    // hard pin: the incoming query has no decode-Q evidence yet
                    // (its prefill-Q hits the call↔definition domain gap), so a
                    // full-strength prior would let the previous turn's tool own
                    // the whole opening window and the model could commit to a
                    // wrong framing before the correct tool is selected. Halving
                    // lets the fresh decode-Q overtake a stale tool within a few
                    // tokens; a real continuation re-accumulates just as fast.
                    let mut carried_belief = self
                        .carried_beliefs
                        .get(&parent_id)
                        .cloned()
                        .unwrap_or_default();
                    carried_belief.decay_scores(CARRIED_BELIEF_TURN_DECAY);
                    let projection = inputs.projection.project_with_mode_and_sink(
                        target,
                        &view,
                        ProjectionMode::Prefill,
                        &inputs.selection,
                        &carried_belief,
                        // Opening projection: decode position 0. The early-decode
                        // window excludes position 0 (its prior is the previous
                        // turn's decayed belief), so submit selects on the steady
                        // band; the grace window engages once decode starts.
                        Some(0),
                        &mut |diag| {
                            diag_to_write = Some((target.timeline, diag));
                        },
                    );
                    // Emit the OPENING projection as a timeline event: a POINT at
                    // token 0 (nothing decoded yet) carrying the initial composition
                    // this turn decodes against. A projection governs everything
                    // forward until the next reprojection supersedes it, so even a
                    // short / no-think turn that never reprojects still gets this one
                    // clickable record. `start_token`/`seconds` stay 0 — the turn's
                    // opening point.
                    {
                        let mut opening = crate::projection::from_projection_with_origins(
                            &projection.segments,
                            &projection.selection_origins,
                            inputs.projection.schema(),
                            &view,
                            &projection.selection_scores,
                            view.total_token_count(target.timeline) as u32,
                            0,
                            0.0,
                        );
                        // The turn's decode state seeds its belief from what this
                        // opening projection actually selected (the carried belief
                        // stepped against this turn's selection), so the first
                        // mid-decode reprojection continues the evolution.
                        turn_belief = PriorBelief::from_selection(&opening.selection);
                        opening.materialized = projection_assembler::materialize_conversation(
                            &projection.segments,
                            &self.boundary_markers,
                            &projection.selection_origins,
                            &view,
                            inputs.projection.schema(),
                            |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
                        );
                        let _ = event_tx.send(TurnEvent::Projection(opening));
                    }
                    // The schema's projection is the single source
                    // of truth for system-side sections — emit
                    // exactly what the projection picked, in
                    // declaration order.  The legacy
                    // "always prepend `system_section_id`" path
                    // (a monolithic ChatML-wrapped duplicate of
                    // the schema fragments) has been removed; the
                    // schema items now compose the full system
                    // prompt on their own.
                    // Single walk of `projection.segments` builds three
                    // outputs:
                    //   - `projected_sections` / `turn_keys_for_elevate`
                    //     feed `elevate_to_hot` (per-id form expected
                    //     by the persistence API).
                    //   - `segments` carries the same items in
                    //     declaration order for `apply_projection`.
                    // Each turn already carries its resolved timeline
                    // (stamped at projection), so no group→timeline
                    // resolution happens here.
                    let mut sections: Vec<SectionId> = Vec::new();
                    let mut segments: Vec<ProjectionSegment> = Vec::new();
                    for seg in &projection.segments {
                        match seg {
                            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                                // Any-tier filter — cold-marker
                                // sections (post-reload, before
                                // elevate) must survive this filter
                                // so the `elevate_to_hot` call below
                                // can lift them.  Filtering on
                                // `section_sealed_of` (hot-only)
                                // would silently drop every
                                // not-yet-elevated section the
                                // projection selected.
                                if view.section_tier_state(rs.id).is_some() {
                                    sections.push(rs.id);
                                    segments.push(seg.clone());
                                }
                            }
                            ProjectionSegment::Sealed(SealedKind::Turn(rt, _))
                            | ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => {
                                // The turn carries its conversation (stamped at
                                // projection); read it directly — no group→timeline
                                // re-derivation.  `elevate_to_hot` below brings
                                // cold-marker turns into hot before apply runs; a
                                // timeline-less (mock/untracked) turn is skipped, a
                                // `TurnHalf` elevates the same underlying turn.
                                let Some(timeline) = rt.timeline else {
                                    continue;
                                };
                                turn_keys_for_elevate.push(TurnKey::new(timeline, rt.index()));
                                segments.push(seg.clone());
                            }
                            ProjectionSegment::Generated { .. }
                            | ProjectionSegment::NewUserMessage { .. } => {
                                // Live-prefill runs and new-user-message
                                // captures don't need an elevate side-list
                                // entry — their K/V is produced on the
                                // slot, not loaded from the substrate.
                                segments.push(seg.clone());
                            }
                        }
                    }

                    // Append a trailing `Generated(UserStart)` so the
                    // current turn's user-side prefill begins behind a
                    // live `<|im_start|>user\n` opener.  This is the
                    // boundary between the most recent past turn (or
                    // the system block) and the new turn — adjacent to
                    // the previous turn's assembler-emitted
                    // `assistant_end` boundary, so the assembler
                    // batches them into a single live-prefill run.
                    let pos = segments.len();
                    segments.push(ProjectionSegment::Generated {
                        tokens: self.boundary_markers.user_start.clone(),
                        identity: GeneratedIdentity {
                            name: "user_start_current".into(),
                            position: pos,
                        },
                    });
                    // Build the composition for a staged prefill's per-segment
                    // projection events now, while the view guard is live. The
                    // calibration projection is pinned (`SelectionRule::Named`), so
                    // one composition serves every segment — the wave overrides its
                    // point `start_token` per emission.
                    if !projection_offsets.is_empty() {
                        let total = view.total_token_count(target.timeline) as u32;
                        staged_composition = Some(crate::projection::from_projection(
                            &projection.segments,
                            inputs.projection.schema(),
                            &view,
                            &projection.selection_scores,
                            total,
                            0,
                            0.0,
                        ));
                    }
                    // When the composer dial suppresses thinking, follow the user
                    // opener with a live `/no_think` run.  Qwen3 only honours the
                    // soft-switch from the user turn (not the system prompt), and
                    // emitting it as GLUE here — re-decided from the current dial
                    // every projection, never sealed into a turn — keeps it out of
                    // the substrate and prevents a past suppressed turn from leaking
                    // a stale switch onto a later thinking-on turn.  The assembler
                    // batches it into the same live-prefill run as `user_start`.
                    let suppress = matches!(
                        inputs.selection.optional(NO_THINK_SELECTOR),
                        Some(OptionalState::Present)
                    );
                    if suppress && !self.boundary_markers.no_think.is_empty() {
                        let pos = segments.len();
                        segments.push(ProjectionSegment::Generated {
                            tokens: self.boundary_markers.no_think.clone(),
                            identity: GeneratedIdentity {
                                name: "no_think_current".into(),
                                position: pos,
                            },
                        });
                    }
                    (sections, segments)
                } else {
                    (Vec::new(), Vec::new())
                };

                // Read guard dropped above when the projection block
                // exited; now safe to take a write guard for the
                // diagnostic side-channel.  No-op when projection used
                // the rule-based path (no tree on the target timeline).
                // Ephemeral probe slots write NOTHING to the substrate — not even
                // this diagnostic side-channel on the (shared) target timeline.
                if let Some((timeline, diag)) = diag_to_write {
                    if !self.ephemeral_slots.contains(&parent_id) {
                        if let Some(conv) = self.slot_conversations.get(&parent_id) {
                            conv.write().set_last_selection(timeline, diag);
                        }
                    }
                }

                // Step 1.5: select-promote (cold → warm → hot).
                //
                // Batched warm→hot scatter (one `kv_migrate_on` per
                // layer on `elevate_copy_stream`) + per-item
                // cold→hot recover for any turn whose sealed bytes
                // live in the redo log. After this returns,
                // `apply_projection`'s per-unit `ensure_*_hot`
                // calls hit the hot branch immediately.
                //
                // Skipped when nothing was projected — `elevate_to_hot`
                // would no-op anyway, but the substrate read-lock
                // snapshot is cheap to avoid.
                if let Some(conversation) = self.slot_conversations.get(&parent_id).cloned() {
                    // Free VRAM held by yesterday's working set (warm-backed hot
                    // residences NOT in the incoming projection), then batch
                    // select-promote the projected sections/turns into hot before
                    // `apply_projection` injects them.
                    self.elevate_projection_working_set(
                        &conversation,
                        &projected_sections,
                        &turn_keys_for_elevate,
                        "submit",
                    );
                }

                // When reprojection is disabled and the slot is already
                // seeded, leave the cumulative slot intact — prefill appends
                // onto it below. `apply_projection(_, BlockCount(0), _)` would
                // reset it to empty, so it must be skipped here (not just fed
                // empty segments, which is the RULER/summarisation reset path).
                if !skip_projection {
                    if let Err(e) =
                        self.apply_projection(parent_id, BlockCount(0), &projected_segments)
                    {
                        let _ = event_tx.send(TurnEvent::Error(e));
                        return true;
                    }
                }

                // Step 2: borrowed ranges cover the whole
                // post-injection parent — the projection itself
                // already selected what's visible.  Use the
                // *chunk-count* API (`sequence_block_count`) rather
                // than `offset / CHUNK_SIZE`: each projected section
                // ends in its own partial trailing chunk that the
                // divided value silently under-counts, leaving the
                // tail invisible to the view.  See
                // `BatchedInferenceSession::sequence_block_count`.
                let parent_block_count =
                    self.session.sequence_block_count(parent_id.0).unwrap_or(0);
                let parent_offset_for_log = self.session.sequence_offset(parent_id.0).unwrap_or(0);
                tracing::trace!(
                    target: "candle_conversation::scheduler::view_create",
                    parent = parent_id.0,
                    parent_block_count,
                    parent_offset = parent_offset_for_log,
                    offset_div_ceil = parent_offset_for_log.div_ceil(self.chunk_size),
                    "view borrow plan",
                );
                // Rolling window over an append-only ingest (design §4.7): on the
                // disable-reprojection path (`skip_projection`), borrow just the
                // system prompt + the last `CODE_READ_WINDOW_TURNS` sealed turns
                // instead of the whole growing parent — bounding otherwise-
                // unbounded ingest context. `window == 0` is the whole-parent
                // borrow.
                let window = if skip_projection {
                    CODE_READ_WINDOW_TURNS
                } else {
                    0
                };
                let effective_ranges: Vec<BlockRange> = if window > 0 {
                    self.slot_targets
                        .get(&parent_id)
                        .map(|t| t.timeline)
                        .zip(self.slot_conversations.get(&parent_id))
                        .map(|(tl, c)| c.windowed_ingest_ranges(tl, window, parent_block_count))
                        .unwrap_or_else(|| {
                            if parent_block_count == 0 {
                                Vec::new()
                            } else {
                                vec![(0, parent_block_count)]
                            }
                        })
                        .into_iter()
                        .map(|(s, e)| BlockRange::new(s, e))
                        .collect()
                } else if parent_block_count == 0 {
                    Vec::new()
                } else {
                    vec![BlockRange::new(0, parent_block_count)]
                };

                // Step 3: carve the view sequence.
                let (view_id, borrowed) = match self.create_view(parent_id, &effective_ranges) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = event_tx.send(TurnEvent::Error(e));
                        return true;
                    }
                };

                // Step 4: anchor the active turn's start index in the
                // view's chunk list.
                //
                // Under the read-only projection model, all borrowed
                // chunks (including any partial tail) are shared Arc
                // clones; the view's writable tail is a fresh active
                // chunk pushed at view index `borrowed.0` by
                // `create_view_sequence`. So `borrowed.0` is the
                // anchor for mid-turn captures and final seals.
                let turn_start_parent_blocks = borrowed.0;

                // Step 5: register the view so cleanup_finished can auto-finalize.
                self.turn_views.insert(
                    view_id,
                    ViewState {
                        parent_id,
                        original_borrowed: borrowed,
                        turn_start_parent_blocks,
                    },
                );

                // Step 6: queue prefill on the view sequence, carrying the
                // reprojection policy through to DecodeState.
                self.prefill_queue.push_back(PrefillWork {
                    sequence_id: view_id,
                    tokens: prefill_tokens,
                    prefill_text,
                    user_text,
                    tags,
                    user_content_start,
                    user_content_end,
                    assistant_content_start,
                    no_think,
                    prefill_assistant_text,
                    event_tx,
                    max_decode_tokens,
                    sampling,
                    submitted_at: Instant::now(),
                    reprojection,
                    belief: turn_belief,
                    seal_action,
                    post_decode_tokens,
                    projection_offsets,
                    staged_composition,
                    triggers,
                });
                true
            }

            SchedulerRequest::FreeSequence { sequence_id } => {
                tracing::debug!(target: "sched", "free_sequence {}", sequence_id);
                if let Err(e) = self.session.free_sequence(sequence_id.0) {
                    tracing::warn!("failed to free sequence {}: {}", sequence_id, e);
                }
                // Clean up persistent sampling state.
                self.sampling_states.remove(&sequence_id);
                // Drop the conversation handle and projection target
                // bound to this slot.
                self.slot_conversations.remove(&sequence_id);
                let freed_target = self.slot_targets.remove(&sequence_id);
                self.ephemeral_slots.remove(&sequence_id);
                self.ephemeral_sigs.remove(&sequence_id);
                self.carried_beliefs.remove(&sequence_id);
                self.slot_tokens.remove(&sequence_id);
                self.slot_projection_state.remove(&sequence_id);
                self.prune_ingest_timeline(freed_target);
                true
            }

            SchedulerRequest::ResetSequence {
                sequence_id,
                response_tx,
            } => {
                let result = self
                    .session
                    .reset_sequence(sequence_id.0)
                    .map_err(ConversationError::Model);
                if result.is_ok() {
                    // Reset sampling state counters while preserving the shape.
                    // end_turn() clears per-turn frequency/presence counts; the
                    // context tokens fed in the next prefill will re-seed DRY.
                    if let Some(state) = self.sampling_states.get_mut(&sequence_id) {
                        state.end_turn();
                    }
                    // A reset slot is reused for NEW content (the titler resets
                    // between title jobs): the previous occupant's belief must
                    // not seed the next occupant's projections.
                    self.carried_beliefs.remove(&sequence_id);
                }
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::IngestSection {
                sequence_id,
                section_id,
                prefix_section_ids,
                tokens,
                address,
                debug_name,
                in_collection,
                response_tx,
            } => {
                match self.prepare_section_ingest(
                    sequence_id,
                    section_id,
                    &prefix_section_ids,
                    &tokens,
                ) {
                    Ok(seal_block_from) => {
                        if tokens.is_empty() {
                            // No forward pass needed — seal immediately.
                            let result = self
                                .perform_seal_and_write(
                                    sequence_id,
                                    seal_block_from,
                                    &SealAction::Section {
                                        section_id,
                                        tokens: Arc::new(tokens.to_vec()),
                                        address,
                                        debug_name: debug_name.clone(),
                                        in_collection,
                                    },
                                    None,
                                )
                                .and_then(|opt| {
                                    opt.ok_or_else(|| {
                                        ConversationError::Channel(
                                            "ingest_section: seal returned None".into(),
                                        )
                                    })
                                });
                            let _ = response_tx.send(result);
                        } else {
                            self.active_section_ingests.push(ActiveSectionIngest {
                                sequence_id,
                                section_id,
                                tokens,
                                offset: 0,
                                seal_block_from,
                                address,
                                debug_name,
                                in_collection,
                                response_tx,
                                error: None,
                            });
                        }
                    }
                    Err(e) => {
                        let _ = response_tx.send(Err(e));
                    }
                }
                true
            }

            SchedulerRequest::RestoreSection {
                conversation,
                section_id,
                stream_id,
                address,
                chunks_per_layer,
                tokens,
                response_tx,
            } => {
                let result = self.restore_section_from_persistence(
                    &conversation,
                    section_id,
                    stream_id,
                    address,
                    chunks_per_layer,
                    tokens,
                );
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::PrimingProjection {
                sequence_id,
                section_ids,
                response_tx,
            } => {
                let result = if section_ids.is_empty() {
                    Ok(())
                } else {
                    // Elevate priming sections cold→hot before
                    // `apply_projection`.  Cold-marker sections
                    // restored from the redo log have `hot = None`
                    // until elevate lifts them; without this call
                    // `inject_sealed_section` would warn-and-skip
                    // and the primed slot would be missing every
                    // persisted section from the schema's prelude.
                    // Load-phase VRAM gate: elevation scatters sealed KV into
                    // fresh GPU arenas BEFORE the projection attends over it — a
                    // "load KV into VRAM before attention" phase. The freed-float
                    // free-list is the destination of that load, so relieve first
                    // (compress/evict make real headroom) rather than allocating
                    // straight into space the load will consume. Without this the
                    // elevation had no scheduler pressure gate at all — only the
                    // per-arena `vram_has_room` check, which OOMs per-arena instead
                    // of shedding ahead of the load.
                    if self.vram_under_pressure_for(prefill::VramPhase::Load) {
                        self.relieve_vram_pressure("elevate", prefill::VramPhase::Load);
                    }
                    if let Some(conversation) = self.slot_conversations.get(&sequence_id).cloned() {
                        let backings = self.session.backings().to_vec();
                        let device = self.session.device().clone();
                        let main_stream = match &device {
                            Device::Cuda(d) => d.cuda_stream(),
                            _ => panic!("scheduler: requires a CUDA device"),
                        };
                        let no_turns: Vec<TurnKey> = Vec::new();
                        if let Err(e) = elevate_to_hot(
                            &conversation,
                            &backings,
                            &device,
                            &main_stream,
                            &mut self.elevate_pinned_scratch,
                            &mut self.cold_load_stager,
                            &section_ids,
                            &no_turns,
                        ) {
                            tracing::warn!(
                                "priming elevate failed for slot {sequence_id}: {e} — \
                                 apply_projection will warn-and-skip cold sections"
                            );
                        }
                    }
                    let segments: Vec<ProjectionSegment> = section_ids
                        .iter()
                        .map(|&id| {
                            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id }))
                        })
                        .collect();
                    let r = self.apply_projection(sequence_id, BlockCount(0), &segments);
                    // End-of-build boundary: every section the schema
                    // ingested earlier in `new_with_projection` is now
                    // installed in `substrate.section.hot` (native) and
                    // has been injected into this slot's block table.
                    // The slot's `Arc<ChunkGid>` refs keep the native
                    // chunks alive even after we swap the residence's
                    // hot to its quantized form, so swap is safe here.
                    // Any conversation forked from this point on will
                    // re-project sections fresh from `section.hot` and
                    // see the quantized form.
                    if r.is_ok() && !self.pending_section_quantize.is_empty() {
                        if let Some(turn_policy) = self.session.compression_policy() {
                            if let Some(conversation) =
                                self.slot_conversations.get(&sequence_id).cloned()
                            {
                                let boundary_policy = Self::section_compression_policy_boundary();
                                let member_policy =
                                    Self::section_compression_policy_member(&turn_policy);
                                if let Err(e) = self.quantize_pending_sections(
                                    &conversation,
                                    &boundary_policy,
                                    &member_policy,
                                ) {
                                    tracing::warn!(
                                        "post-priming section quantize drain failed: {e:?}"
                                    );
                                }
                            } else {
                                tracing::warn!(
                                    "post-priming drain: no conversation handle for slot {sequence_id}"
                                );
                                self.pending_section_quantize.clear();
                            }
                        } else {
                            self.pending_section_quantize.clear();
                        }
                    }
                    r
                };
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::OffloadCollectionMembers {
                conversation,
                response_tx,
            } => {
                let result = (|| -> Result<(), ConversationError> {
                    if let Some(turn_policy) = self.session.compression_policy() {
                        let boundary_policy = Self::section_compression_policy_boundary();
                        let member_policy = Self::section_compression_policy_member(&turn_policy);
                        // Quantize just the prefix-transparent members + flag
                        // them for offload; boundary sections stay pending for
                        // the turn-seal boundary.
                        self.quantize_pending_collection_members(
                            &conversation,
                            &boundary_policy,
                            &member_policy,
                        )?;
                    }
                    // Block on a full persistence pass so the members' cold
                    // copies land and `install_cold` frees their VRAM before the
                    // next batch prefills.  The wave-batched prefill within a
                    // batch keeps its own concurrency; this seam only gates
                    // between batches so the native catalog can't outrun the
                    // offload under VRAM pressure.
                    timed_wait(|| {
                        self.persist_trigger
                            .flush_blocking(std::time::Duration::from_secs(30))
                    });
                    Ok(())
                })();
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ExtractRawKvq {
                sequence_id,
                layer_indices,
                block_range,
                response_tx,
            } => {
                let result =
                    self.handle_extract_raw_kvq(sequence_id.0, &layer_indices, block_range);
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ProbeWideSigs {
                sequence_id,
                response_tx,
            } => {
                // Hand back the warm query wide-Q the ephemeral probe turn stashed
                // at its completion; drain it so a re-probe of a recycled slot id
                // can't read a stale window.
                let sigs = self.ephemeral_sigs.remove(&sequence_id).unwrap_or_default();
                let _ = response_tx.send(Ok(sigs));
                true
            }

            SchedulerRequest::SubmitSummaryProbe {
                timeline,
                kind,
                children,
                height,
                response_tx,
            } => {
                // Dispatches the node to its compression: a structural (repo-map)
                // node seals synchronously; a model-decode node enqueues ONE
                // decode-only pass that rides `batch_decode_step` concurrently
                // with foreground turns and is recorded by
                // `complete_compression_pass` when it lands. Either way the
                // handler returns immediately; a setup failure replies with `Err`
                // here for a soft retry on the summariser's next pass.
                if let Err(e) =
                    self.handle_summary_probe(timeline, kind, children, height, response_tx.clone())
                {
                    // "no projection/summary" means this layer has no
                    // summary-of-summaries projection to compress into — a permanent
                    // structural condition, not a transient setup failure. Classify
                    // it Permanent so the summariser disarms this timeline's reconcile
                    // instead of re-submitting a doomed probe every pass forever
                    // (which starves the summariser and floods the log).
                    let pe = if e.contains("no projection/summary") {
                        ProbeError::Permanent(e)
                    } else {
                        ProbeError::Soft(e)
                    };
                    let _ = response_tx.send(Err(pe));
                }
                true
            }

            SchedulerRequest::PrefillScope {
                timeline,
                projection,
                scope_index,
                scope_total,
                tokens,
                user_content_start,
                user_content_end,
                assistant_content_start,
                user_text,
                assistant_text,
                tags,
                response_tx,
                on_prefilled,
            } => {
                // Register the timeline's projection so the summariser can compress
                // its turns (the parallel path's analog of SubmitTurn's capture).
                self.workspace_projection
                    .get_or_insert_with(|| projection.clone());
                self.timeline_projections
                    .entry(timeline)
                    .or_insert(projection);
                // Register the scope into its file batch and queue it for the fair
                // pump; a setup failure (lost timeline) replies `Err` immediately.
                // The pump owns scratch-slot allocation + cross-file fairness, so
                // this returns without touching the GPU.
                if let Err(e) = self.handle_prefill_scope(
                    timeline,
                    scope_index,
                    scope_total,
                    tokens,
                    user_content_start,
                    user_content_end,
                    assistant_content_start,
                    user_text,
                    assistant_text,
                    tags,
                    response_tx.clone(),
                    on_prefilled,
                ) {
                    let _ = response_tx.send(Err(e));
                }
                true
            }

            SchedulerRequest::ReconstructSubstrate {
                conversation,
                status,
            } => {
                self.reconstruct_substrate(&conversation, &status);
                true
            }

            SchedulerRequest::DemoteTimelinesHot {
                conversation,
                timelines,
                response_tx,
            } => {
                // Drop the hot copy of each timeline's turns that already hold a
                // warm copy (a turn without one is left hot — see
                // `demote_turns_to_warm`). Any needed hot→warm flush was run
                // caller-side before this request, so this thread never blocks on
                // the persistence pass.
                let demoted = {
                    let mut view = conversation.write();
                    let keys: Vec<TurnKey> = timelines
                        .iter()
                        .flat_map(|&tl| {
                            view.turn_indices(tl)
                                .map(move |idx| TurnKey::new(tl, idx))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    view.demote_turns_to_warm(&keys)
                };
                // The demote returned the hot chunks to the pool free-list;
                // release now-empty arenas so `pool_used` actually drops.
                let _ = self.session.release_empty_arenas();
                let _ = response_tx.send(Ok(demoted));
                true
            }

            SchedulerRequest::Shutdown => false,
        }
    }

    /// Compress a §6 summary node over its children into a single summary turn.
    ///
    /// A structural (repo-map) node is built deterministically from its inputs
    /// and sealed synchronously ([`Self::seal_structural_turn`]) — no model
    /// decode. A model-decode node runs ONE decode-only pass
    /// ([`Self::enqueue_compression_pass`]): the children's full turns are
    /// injected sealed (zero re-prefill) after being lifted hot, a tiny
    /// summarise instruction is appended, and the model decodes one summary into
    /// a fresh chunk range that `complete_compression_pass` snapshots and
    /// records — no stitch, no re-prefill. Nothing blocks the scheduler: the
    /// handler returns as soon as the pass is enqueued, and a setup failure
    /// replies `Err` for a soft retry. The decode is argmax, capped at the
    /// `summary` level's `max_tokens`.
    ///
    /// `children` are the node's structural children carrying the content to
    /// compress.
    fn handle_summary_probe(
        &mut self,
        timeline: TimelineId,
        kind: TurnKind,
        children: Vec<TurnIndex>,
        height: u8,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        tracing::trace!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            children = ?children,
            "compression probe: enqueuing compression pass",
        );
        // Resolve the conversation + projection target that own this timeline.
        // We look up through any slot registered against the same target.
        let conv = self
            .slot_conversations
            .values()
            .find(|c| c.read().timeline_target(timeline).is_some())
            .cloned()
            .ok_or_else(|| {
                format!("SubmitSummaryProbe: no conversation registered for timeline {timeline}")
            })?;
        let (layer, group) = conv
            .read()
            .timeline_target(timeline)
            .ok_or_else(|| format!("SubmitSummaryProbe: timeline {timeline} has no target"))?;
        let target = ProjectionTarget {
            layer,
            group,
            timeline,
        };
        if children.is_empty() {
            return Err("SubmitSummaryProbe: node has no children to compress".to_string());
        }

        // The compression prompts + decode cap come from the target layer's
        // summary config, read off the projection Builder the scheduler captured
        // for this timeline on `SubmitTurn`. A node selects its tree-level via
        // `LayerSummary::for_kind` — `summaries` for a SummaryOfSummaries node
        // (falling back to `turns`), else `turns` — and within that level the
        // user pass uses the `question` prompt (user-message half) and the
        // assistant pass the `answer` prompt (assistant-response half). Cloned so
        // it is owned for the rest of this handler (the two passes borrow
        // `&mut self`).
        // Resolve the target layer's summary config from this timeline's builder,
        // falling back to the workspace builder for timelines reloaded from the
        // substrate (not in the live per-timeline map) — the schema is
        // workspace-global, so either resolves the same layer config.
        let turn_summary = self
            .timeline_projections
            .get(&timeline)
            .or(self.workspace_projection.as_ref())
            .and_then(|b| {
                b.schema()
                    .layers
                    .iter()
                    .find(|l| l.id == target.layer)
                    .map(|l| {
                        l.summary
                            .for_kind(matches!(kind, TurnKind::SummaryOfSummaries))
                            .clone()
                    })
            })
            .ok_or_else(|| {
                format!(
                    "SubmitSummaryProbe: no projection/summary for timeline {timeline} layer {:?}",
                    target.layer
                )
            })?;

        // Structural layers (repo_map directory trees) are built deterministically
        // from their inputs — no model decode at any level. A leaf strips the size
        // annotations off its one scan turn; a summary-of-summaries reconstructs
        // and height-truncates the children's directory paths. Decoding either is
        // pure cost and invites fabrication. `seal_structural_turn` dispatches on
        // kind and derives both halves.
        if turn_summary.content == Content::Structural {
            return self.seal_structural_turn(&conv, target, kind, &children, height, response_tx);
        }

        // Backpressure: a model-decode compression pass lifts its children to hot
        // and adds VRAM churn. The summariser is a background task, so when VRAM is
        // under pressure defer this node — a Soft error it retries on its next
        // reconcile pass — rather than piling onto a tight card and starving
        // foreground decode + the persist thread's hot→warm drain. (The structural
        // path above is deterministic and cheap, so it is deliberately not gated.)
        if self.vram_under_pressure() {
            return Err("SubmitSummaryProbe: deferred — VRAM under pressure".to_string());
        }

        // Reserve the job id up front so both passes tag their `SealAction`
        // with it; the job entry registers once both passes are set up.
        let job_id = self.next_compression_job_id;
        self.next_compression_job_id += 1;

        // Derive the node's user half — its scope — before the decode. It is a
        // pure function of the children's scopes, so it needs neither the model
        // nor the decode's result, and the decode is never asked to write it.
        let scope_tokens =
            self.derive_scope_tokens(&conv, target, &children, turn_summary.scope, height)?;

        // One decode, for the assistant half only. Inject the children's FULL
        // original turns (natural user→assistant roles, sealed — zero re-prefill),
        // add a tiny summarise instruction, open a fresh writer chunk, and decode
        // ONE summary body. `complete_compression_pass` then pairs it with the
        // derived scope above and seals both halves through the shared
        // `seal_compression_turn`, which strips the think block and re-prefills the
        // marker-framed exchange into role-coherent K/V.
        let slot = self.enqueue_compression_pass(
            &conv,
            target,
            &children,
            &turn_summary.assistant,
            turn_summary.max_tokens,
            job_id,
        )?;

        self.compression_jobs.insert(
            job_id,
            CompressionJob {
                conversation: conv,
                target,
                kind,
                scope_tokens,
                children: children
                    .iter()
                    .map(|&c| TurnKey::new(target.timeline, c))
                    .collect(),
                response_tx,
            },
        );
        // `slot` is carried into `active_decodes` by the pass; the completion
        // receives it from `cleanup_finished`, so the job need not hold it.
        let _ = slot;
        Ok(())
    }

    /// Derive a summary node's user half — its scope — from its children's, and
    /// encode it (`summary_tree::scope`).
    ///
    /// A child's scope is its user-half tokens: for a `Normal` turn that is the
    /// real question the user asked, and for a summary node it is the scope this
    /// same derivation gave it one level down. So a node's scope is grounded in
    /// real user text at every height, and no decode ever writes one — a decode
    /// always speaks as the assistant, so asking it for the question half is
    /// asking it to invent a question that was never asked.
    ///
    /// A `SummaryOfTurns` leaf covers one **exchange**, which answers exactly one
    /// question — the head turn's user half. The remaining members are tool
    /// round-trips whose user halves are `<tool_response>` output, not questions,
    /// so only the head is consulted at the leaf: a single-turn exchange keeps its
    /// question verbatim, and a tool exchange is scoped by the question that
    /// started it rather than by the JSON that served it.
    fn derive_scope_tokens(
        &self,
        conv: &Conversation,
        target: ProjectionTarget,
        children: &[TurnIndex],
        scope: Scope,
        height: u8,
    ) -> Result<Vec<u32>, String> {
        // A node always summarises at least one child; an empty set means the
        // caller built a degenerate node, and deriving a scope from nothing would
        // silently seal an empty (question-less) summary. Fail loudly instead.
        if children.is_empty() {
            return Err("derive_scope_tokens: node has no children".to_string());
        }
        // At a leaf the children are one exchange's turns, and only its head
        // carries the question (see above). Above the leaf the children are
        // summary nodes, each already scoped — those all contribute.
        let scoped: &[TurnIndex] = if height <= 1 {
            &children[..1]
        } else {
            children
        };
        let child_scopes: Vec<String> = {
            let view = conv.read();
            scoped
                .iter()
                .map(|&c| {
                    view.turn_user_token_ids(target.timeline, c)
                        .unwrap_or_default()
                })
                .collect::<Vec<Vec<u32>>>()
        }
        .into_iter()
        .map(|toks| self.tokenizer.decode(&toks, true).unwrap_or_default())
        .collect();
        let combined = scope.combine(&child_scopes, height);
        Ok(self
            .tokenizer
            .encode(combined.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode derived scope: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Build and seal a structural node deterministically, skipping the model
    /// decode entirely (`content: structural`, the repo_map directory layer). A
    /// `SummaryOfTurns` leaf strips the size annotations off its one scan turn,
    /// keeping the full skeleton; a `SummaryOfSummaries` reconstructs the
    /// children's directory paths and truncates them by tree height (coarser
    /// toward the root). See `summary_tree::structural`.
    fn seal_structural_turn(
        &mut self,
        conv: &Conversation,
        target: ProjectionTarget,
        kind: TurnKind,
        children: &[TurnIndex],
        height: u8,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        // Read each child's assistant-half *tokens* and decode them. Normal scan
        // turns are prefill-only (no stored `assistant_text`), so we slice the
        // assistant body off their token ids rather than relying on the text field.
        let child_tokens: Vec<Vec<u32>> = {
            let view = conv.read();
            children
                .iter()
                .map(|&c| {
                    view.turn_assistant_token_ids(target.timeline, c)
                        .unwrap_or_default()
                })
                .collect()
        };
        let child_texts: Vec<String> = child_tokens
            .iter()
            .map(|toks| self.tokenizer.decode(toks, true).unwrap_or_default())
            .collect();
        let rollup = match kind {
            // Leaf: its one child is the raw Normal scan turn — strip annotations.
            TurnKind::SummaryOfTurns => leaf_skeleton(child_texts.first().map_or("", |s| s)),
            // SoS: its children are already-built directory skeletons.
            _ => structural_rollup(&child_texts, height),
        };
        // An empty structural node IS the "empty repo_map" symptom. Log only when
        // it happens, with enough to place blame: no children, a child whose
        // assistant K/V read back empty (upstream seal problem), or a non-empty
        // input the transform reduced to nothing (a parsing regression). Quiet on
        // the healthy path.
        if rollup.skeleton.trim().is_empty() {
            let child_lens: Vec<(u32, usize, usize)> = children
                .iter()
                .zip(&child_tokens)
                .zip(&child_texts)
                .map(|((c, toks), text)| (c.0, toks.len(), text.trim().len()))
                .collect();
            tracing::warn!(
                target: "candle_conversation::summariser",
                timeline = target.timeline.raw(),
                ?kind,
                height,
                n_children = children.len(),
                // (turn_index, assistant_token_len, decoded_char_len) per child.
                children = ?child_lens,
                scope_empty = rollup.scope.trim().is_empty(),
                "structural seal produced an EMPTY skeleton — this becomes an empty repo_map node",
            );
        }
        let user_tokens = self
            .tokenizer
            .encode(rollup.scope.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode structural scope: {e}"))?
            .get_ids()
            .to_vec();
        let assistant_tokens = self
            .tokenizer
            .encode(rollup.skeleton.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode structural skeleton: {e}"))?
            .get_ids()
            .to_vec();
        let job_id = self.next_compression_job_id;
        self.next_compression_job_id += 1;
        self.seal_compression_turn(
            job_id,
            conv.clone(),
            target,
            kind,
            children.to_vec(),
            user_tokens,
            assistant_tokens,
            response_tx,
        )
    }

    /// Seal a compression prompt's system-prompt framing as a content section in
    /// `conv`'s substrate and return its [`SectionId`], reusing the real section
    /// ingest path (`prepare_section_ingest` + `finalize_section_ingest`) rather
    /// than a bespoke seal. Idempotent and lazy: the first compression probe to
    /// use this prompt seals it; every later probe returns the same id with no
    /// work (the `section_sealed_of` check is the authoritative guard).
    ///
    /// The section id is the one the schema allocated for `prompt.system_prompt`
    /// (a normal `1..n` schema id, but never added to `system_prompt.items`, so
    /// it only ever materialises via this seal — never in a normal projection).
    /// The K/V is pinned in the conversation's own substrate, so a scheduler
    /// hosting several conversations re-seals into each one the first time that
    /// conversation runs a compression pass.
    fn ensure_summary_section(
        &mut self,
        conv: &Conversation,
        prompt: &CompressionPrompt,
    ) -> Result<SectionId, ConversationError> {
        let section_id = prompt.system_prompt.id;

        // Already pinned in this conversation's substrate → nothing to do.
        if conv.read().section_sealed_of(section_id).is_some() {
            return Ok(section_id);
        }

        // Tokenize the prompt once and derive a stable content address so the
        // section gets its own persistence stream, disjoint from the schema's.
        let tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt.system_prompt.content.as_str(), false)
            .map_err(|e| ConversationError::Channel(format!("summary section: encode: {e}")))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err(ConversationError::Channel(
                "summary section: prompt encoded to zero tokens".into(),
            ));
        }
        let address = ContentChain::new().push_section(&tokens);
        let tokens = TokenBuffer::from(tokens);

        // Seal on a throwaway scratch slot through the section ingest path: lay
        // down the prefix (none) + writer chunk, prefill the prompt tokens, then
        // seal `[seal_block_from..block_count)` into the substrate via
        // `SealAction::Section`. The slot is freed regardless of outcome.
        let slot = self.create_sequence(conv.clone(), None)?;
        let result = (|| -> Result<(), ConversationError> {
            let seal_block_from = self.prepare_section_ingest(slot, section_id, &[], &tokens)?;
            self.run_prefill(slot, &tokens[..])?;
            self.finalize_section_ingest(
                slot,
                section_id,
                seal_block_from,
                Arc::new(tokens.to_vec()),
                address,
                "summary_system_prompt".to_string(),
                false,
            )?;
            Ok(())
        })();
        self.free_summary_slot(slot);
        result?;

        Ok(section_id)
    }

    /// Set up the summarise decode — which produces the node's assistant half —
    /// returning its scratch slot.
    ///
    /// Assembles `[compressor system][full child turns, natural roles][user_open]
    /// [instruction][user_end]` — the children's ORIGINAL user + assistant content
    /// injected as sealed K/V (`SealedKind::Turn`, zero re-prefill, natural
    /// user→assistant framing), then a follow-up user turn asking the model to
    /// summarise them. `setup_compression_decode` prefills `assistant_start` and
    /// registers the decode. The decode advances in `batch_decode_step` and
    /// completes in `cleanup_finished` → `complete_compression_pass`, which takes
    /// the decoded body as the node's assistant half and seals it with the derived
    /// scope. The decode's own K/V is not kept — the seal re-prefills the cleaned,
    /// marker-framed exchange into role-coherent K/V.
    fn enqueue_compression_pass(
        &mut self,
        conv: &Conversation,
        target: ProjectionTarget,
        children: &[TurnIndex],
        prompt: &CompressionPrompt,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<SequenceId, String> {
        // Lazily seal the summary system-prompt framing as a content section the
        // first time it is used, then inject it (zero-copy Arc clone, zero
        // re-prefill) at the head of the segment list.
        let summary_section = self
            .ensure_summary_section(conv, prompt)
            .map_err(|e| format!("SubmitSummaryProbe: seal summary section: {e}"))?;

        // Segment list: compressor system section, then the children's full turns
        // in natural roles (sealed), then the instruction user-turn.
        let mut segments: Vec<ProjectionSegment> = vec![ProjectionSegment::Sealed(
            SealedKind::Section(ResolvedSection {
                id: summary_section,
            }),
        )];
        for &child in children {
            segments.push(ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: target.layer,
                        group_id: target.group,
                        index: child,
                    },
                    timeline: Some(target.timeline),
                },
                // Role is display-only for injection (`inject_sealed_turn` ignores
                // it and places the whole turn's sealed K/V); the turn's own baked
                // boundary markers carry the real user→assistant roles.
                Role::Assistant,
            )));
        }
        // The instruction user-turn opener: `user_start` + `/no_think`. `/no_think`
        // rides it as live glue — the same mechanism the dialogue's
        // `no_think_current` uses — so the compressor never reasons (the summary
        // budget is tiny). Unconditional: a summary pass always suppresses.
        segments.push(ProjectionSegment::Generated {
            tokens: Arc::new(self.boundary_markers.user_start.as_ref().clone()),
            identity: GeneratedIdentity {
                name: "compress_open".to_string(),
                position: 0,
            },
        });
        if !self.boundary_markers.no_think.is_empty() {
            segments.push(ProjectionSegment::Generated {
                tokens: self.boundary_markers.no_think.clone(),
                identity: GeneratedIdentity {
                    name: "compress_no_think".to_string(),
                    position: 0,
                },
            });
        }
        // Close glue: the summarise instruction followed by `user_end`. No
        // prior-summary anchor — the pass sees only its children and the prompt, so
        // there is nothing for it to reproduce verbatim (a leak we hit when
        // anchoring on prior summaries).
        let close: Vec<u32> = {
            let mut t = self
                .tokenizer
                .encode(format!("\n\n{}", prompt.user_prompt), false)
                .map_err(|e| format!("SubmitSummaryProbe: encode instruction: {e}"))?
                .get_ids()
                .to_vec();
            t.extend_from_slice(&self.boundary_markers.user_end);
            t
        };
        segments.push(ProjectionSegment::Generated {
            tokens: Arc::new(close),
            identity: GeneratedIdentity {
                name: "compress_close".to_string(),
                position: 1,
            },
        });

        // Scratch slot bound to the timeline (so sealed-turn injection can resolve
        // it from `slot_target`). Freed once the pass completes.
        let slot = self
            .create_sequence(conv.clone(), Some(target))
            .map_err(|e| format!("SubmitSummaryProbe: create slot: {e}"))?;
        // Lift the children (and the compressor system section) into hot VRAM
        // before `setup_compression_decode` injects them. The children are
        // just-recorded scope/turn K/V that the tier system may already have
        // evicted to warm/cold; without this the compression pass's
        // `apply_projection` finds no hot residence and drops them, so the
        // summary would be decoded over missing content. The other
        // apply_projection callers (SubmitTurn, reproject) already elevate;
        // this path is the one that didn't.
        let child_keys: Vec<TurnKey> = children
            .iter()
            .map(|&c| TurnKey::new(target.timeline, c))
            .collect();
        self.elevate_projection_working_set(conv, &[summary_section], &child_keys, "summary");
        match self.setup_compression_decode(slot, &segments, max_tokens, job_id) {
            Ok(()) => {}
            Err(e) => {
                self.free_summary_slot(slot);
                // Setup failed before decode, so `complete_compression_pass` (which
                // normally demotes) never runs — demote the just-lifted children
                // here too, else they linger hot until ordinary eviction.
                conv.write().demote_turns_to_warm(&child_keys);
                return Err(e);
            }
        }
        Ok(slot)
    }

    /// Assemble + prefill + sample-first for one compression pass, then install
    /// its [`DecodeState`]. Split out so `enqueue_compression_pass` can free the
    /// scratch slot on every failure path.
    fn setup_compression_decode(
        &mut self,
        slot: SequenceId,
        segments: &[ProjectionSegment],
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        self.apply_projection(slot, BlockCount(0), segments)
            .map_err(|e| format!("SubmitSummaryProbe: assemble summary pass: {e}"))?;
        self.finish_compression_pass_setup(slot, max_tokens, job_id)
    }

    /// Prefill `assistant_start`, sample the first token, and register the
    /// `CompressionPass` decode.
    fn finish_compression_pass_setup(
        &mut self,
        slot: SequenceId,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        let turn_start = Instant::now();
        // Prefill `assistant_start` to get the first-token logits and frame the
        // model to *answer* rather than continue the prompt.
        let asst_start = self.boundary_markers.assistant_start.as_ref().clone();
        let prefill_logits = self
            .run_prefill(slot, &asst_start)
            .map_err(|e| format!("SubmitSummaryProbe: prefill assistant_start: {e}"))?;
        let prefill_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let prefill_token_count = self.session.sequence_offset(slot.0).unwrap_or(0);

        let config = SamplingConfig::compression();
        let mut sstate = self
            .sampling_states
            .remove(&slot)
            .ok_or_else(|| "SubmitSummaryProbe: missing sampling state".to_string())?;
        let first = self
            .sample_single(&prefill_logits, &config, &mut sstate)
            .map_err(|e| {
                self.sampling_states.insert(slot, sstate.clone());
                format!("SubmitSummaryProbe: sample first: {e}")
            })?;
        self.sampling_states.insert(slot, sstate);

        if self.is_eos(first) {
            return Err(
                "SubmitSummaryProbe: summary produced no tokens (immediate EOS)".to_string(),
            );
        }

        // Compression passes report completion through the job, not through
        // `TurnEvent`, so the event channel is a private sink. `batch_decode_step`
        // sends streamed `Token` events into it; the dropped receiver makes those
        // sends fail, which marks the decode finished — harmless, because
        // `cleanup_finished` reaps it the same way EOS / max_tokens would.
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        // Keep the receiver alive for the decode's lifetime so per-token sends
        // succeed; it is dropped when the pass's slot is freed.
        self.compression_event_sinks.insert(slot, event_rx);
        let health = {
            let mut hs = DecodeHealthState::new(
                self.health_config.repetition_window,
                self.health_config.health_log_capacity,
            );
            hs.apply_baseline_config(
                self.health_config.entropy_baseline_window,
                self.health_config.entropy_trend_relative_factor,
                self.health_config.entropy_trend_absolute_min_nats,
            );
            // Argmax: temperature ~0, so entropy checks self-disable (peaked
            // distributions are expected). Repetition / phrase-loop checks stay
            // active — the normal-wave health contract a foreground decode gets.
            hs.skip_entropy_checks = config.temperature <= 0.01;
            hs
        };
        self.active_decodes.insert(
            slot,
            DecodeState {
                event_tx,
                generated_tokens: TokenBuffer::from(vec![first]),
                max_tokens,
                sampling_config: config,
                seal_action: SealAction::CompressionPass { job_id },
                prefill_assistant_text: String::new(),
                finished: false,
                decode_start: Instant::now(),
                prefill_ms,
                prefill_token_count,
                turn_start,
                health,
                reprojection: None,
                non_punct_since_reproject: 0,
                last_projection_end: 0,
                post_decode_tokens: TokenBuffer::default(),
                belief: PriorBelief::default(),
                prefill_tokens: TokenBuffer::default(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                in_tool_call: false,
                triggers: Arc::new(TriggerRegistry::new()),
                stencil: None,
                pending_mask: None,
            },
        );
        Ok(())
    }

    /// Complete the finished summarise pass: take its decoded tokens as the
    /// node's assistant half, pair them with the scope derived before the decode,
    /// and seal both halves through the shared [`Self::seal_compression_turn`] —
    /// which strips the think block and re-prefills the marker-framed exchange
    /// into role-coherent K/V.
    ///
    /// The decode's own K/V is deliberately dropped rather than snapshotted. It
    /// is assistant-role K/V for a bare body with no question in front of it, and
    /// it covers the raw generated tokens — including any `<think>` block. Keeping
    /// it would mean either storing the think block or storing K/V that no longer
    /// matches the stored token ids. The re-prefill costs one short prefill and
    /// buys a node that is a real exchange: a question in user-role position, its
    /// answer in assistant-role position.
    fn complete_compression_pass(&mut self, slot: SequenceId, job_id: u64, generated: TokenBuffer) {
        let Some(job) = self.compression_jobs.remove(&job_id) else {
            // Job already torn down — just reclaim the slot.
            self.free_summary_slot(slot);
            return;
        };
        let CompressionJob {
            conversation,
            target,
            kind,
            scope_tokens,
            children,
            response_tx,
        } = job;
        let timeline = target.timeline;

        // The children were lifted to hot only so this pass's decode could attend
        // over them. Now that the summary has decoded, drop their hot copies
        // (keeping warm) so a long run of summary passes can't accumulate
        // transient hot residency and exhaust VRAM — the failure mode where the
        // persist thread's hot→warm migrate then can't get scratch and OOMs.
        // Cheap: no migrate, the warm copy already exists from before the lift.
        if !children.is_empty() {
            let demoted = conversation.write().demote_turns_to_warm(&children);
            if demoted > 0 {
                tracing::trace!(
                    target: "candle_conversation::persistence::tier",
                    demoted,
                    timeline = timeline.raw(),
                    "summary pass: demoted lifted children back to warm"
                );
            }
        }

        // The wave decode pushes the final token (EOS / max_tokens-th) WITHOUT
        // forwarding it, so its K/V never landed. Drop it so the token_ids align
        // 1:1 with the snapshotted chunk grid.
        let summary_tokens: Vec<u32> = generated
            .split_last()
            .map(|(_, rest)| rest.to_vec())
            .unwrap_or_default();
        if summary_tokens.is_empty() {
            let _ = response_tx.send(Err(ProbeError::Soft(
                "SubmitSummaryProbe: summary produced no forwarded tokens".to_string(),
            )));
            self.free_summary_slot(slot);
            return;
        }

        // The decode's K/V is not kept — the seal re-prefills the cleaned,
        // marker-framed exchange — so the scratch slot can go now.
        self.free_summary_slot(slot);

        // Pair the decoded body (the assistant half) with the scope derived
        // before the decode (the user half) and seal both through the shared
        // path: think-strip, marker-frame, re-prefill to role-coherent K/V,
        // record, persist tokens + wide-Q sigs, and reply to the summariser.
        if let Err(e) = self.seal_compression_turn(
            job_id,
            conversation,
            target,
            kind,
            children.iter().map(|k| k.index).collect(),
            scope_tokens,
            summary_tokens,
            response_tx,
        ) {
            // `seal_compression_turn` has already replied to the summariser on
            // every error path it owns; log for the operator and move on.
            tracing::warn!(timeline = timeline.raw(), "summary seal failed: {e}");
        }
    }

    /// Frame a compressed exchange (a user-half + assistant-half, already as
    /// token ids) as one clean marker-framed turn and enqueue it for the seal.
    /// Decode `tokens`, strip any `<think>…</think>` block, and re-encode the
    /// cleaned summary text. Returns the original tokens unchanged when there is
    /// no think tag (the common case pays only a decode + scan), so the
    /// deterministic structural path — which never emits a think block — is a
    /// no-op.
    fn strip_think_from_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        let Ok(text) = self.tokenizer.decode(tokens, true) else {
            return tokens.to_vec();
        };
        let lower = text.to_ascii_lowercase();
        if !lower.contains("<think>") && !lower.contains("</think>") {
            return tokens.to_vec();
        }
        let stripped = crate::think_strip::strip_think_blocks(&text);
        self.tokenizer
            .encode(stripped.as_str(), false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_else(|_| tokens.to_vec())
    }

    /// As [`Self::strip_think_from_tokens`], but PRESERVES the surviving answer's
    /// formatting (newlines, indentation, code blocks) — only the
    /// `<think>…</think>` block and the whitespace immediately around it are
    /// removed. Used for a dialogue turn's clean re-prefill, where collapsing the
    /// answer's whitespace (as the summary path does) would mangle its layout in
    /// the re-injected K/V.
    fn strip_think_from_tokens_keep_layout(&self, tokens: &[u32]) -> Vec<u32> {
        let Ok(text) = self.tokenizer.decode(tokens, true) else {
            return tokens.to_vec();
        };
        let lower = text.to_ascii_lowercase();
        if !lower.contains("<think>") && !lower.contains("</think>") {
            return tokens.to_vec();
        }
        let stripped = crate::think_strip::strip_think_blocks_keep_layout(&text);
        self.tokenizer
            .encode(stripped.as_str(), false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_else(|_| tokens.to_vec())
    }

    /// Shared by the model-decode path ([`Self::enqueue_compression_turn`]) and
    /// the deterministic structural path ([`Self::handle_summary_probe`]).
    #[allow(clippy::too_many_arguments)]
    fn seal_compression_turn(
        &mut self,
        job_id: u64,
        conversation: Conversation,
        target: ProjectionTarget,
        kind: TurnKind,
        children: Vec<TurnIndex>,
        user_tokens: Vec<u32>,
        assistant_tokens: Vec<u32>,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        // The node inherits the union of its children's gather-scope tags —
        // exact per node and recursive by construction (a SoS's children are
        // earlier summary nodes carrying their own inherited tags).
        let tags = conversation.union_turn_tags(target.timeline, &children);
        // Despite the `/no_think` directive the model may still emit an (empty)
        // `<think></think>` block before the summary. Summaries are stored as
        // plain content, so strip the block from each half here — before the
        // halves are stitched, re-prefilled, and recorded — so it never leaks
        // into the substrate or the GUI's summary view.
        let user_tokens = self.strip_think_from_tokens(&user_tokens);
        let assistant_tokens = self.strip_think_from_tokens(&assistant_tokens);

        // Frame the compressed exchange as a clean, marker-framed turn — the
        // question body, then `[user_end][assistant_start]`, then the answer body.
        // No leading `no_think` / `user_start` head: those are live `Generated`
        // segments the assembler re-emits around the sealed turn on every future
        // projection (matching the persisted form of a normal turn).
        let user_end = self.boundary_markers.user_end.as_ref().clone();
        let assistant_start = self.boundary_markers.assistant_start.as_ref().clone();
        let mut token_ids: Vec<u32> = Vec::with_capacity(
            user_tokens.len() + user_end.len() + assistant_start.len() + assistant_tokens.len(),
        );
        token_ids.extend_from_slice(&user_tokens);
        let user_end_at = token_ids.len();
        token_ids.extend_from_slice(&user_end);
        token_ids.extend_from_slice(&assistant_start);
        let asst_start_at = token_ids.len();
        token_ids.extend_from_slice(&assistant_tokens);
        let token_count = token_ids.len();

        // Decode both halves' display text. On failure reply to the summariser
        // before returning — otherwise `response_tx` drops and its `recv()` turns
        // a soft, retryable error into a hard one.
        let user_text = match self.tokenizer.decode(&user_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode user-half: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };
        let assistant_text = match self.tokenizer.decode(&assistant_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode assistant-half: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };

        // Create the scratch slot the wave will re-prefill the marker-framed turn
        // onto. Its question body lands in user-message position (user-role K/V),
        // its answer body after `assistant_start` (assistant-role K/V) — the
        // role-coherent K/V that replaces the passes' decode-time K/V.
        let slot = match self.create_sequence(conversation.clone(), Some(target)) {
            Ok(s) => s,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: reproject slot: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };

        // Private event sink, kept alive so the prefill machinery's sends never
        // fail. This path has no decode, but `PrefillWork` still carries an
        // `event_tx`; the receiver is dropped with the slot in `free_summary_slot`.
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        self.compression_event_sinks.insert(slot, event_rx);

        // Stash the turn content for the deferred seal, then enqueue the
        // marker-framed tokens as a `max_decode=0` prefill unit. The shared wave
        // prefills it alongside the live turn and other summaries;
        // `complete_compression_turn` snapshots + records it when the prefill
        // finishes.
        // Build the compressed turn's segment layout. The exchange is framed
        // `[question][user_end][assistant_start][answer]` with no leading head,
        // so the user body spans `[0, user_end_at)` and the answer body starts at
        // `asst_start_at`. Summaries strip their think block, so there is no
        // thinking split here.
        let layout = TurnLayout::from_flat_grid(
            0,
            user_end_at as u32,
            asst_start_at as u32,
            token_count as u32,
            user_end.len() as u32,
            assistant_start.len() as u32,
            user_text,
            Some(assistant_text),
            false,
        );
        self.pending_compression_seals.insert(
            job_id,
            PendingCompressionSeal {
                conversation,
                target,
                layout,
                token_ids: token_ids.clone(),
                kind,
                children,
                tags,
                response_tx,
            },
        );
        self.prefill_queue.push_back(PrefillWork {
            sequence_id: slot,
            tokens: TokenBuffer::from(token_ids),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            prefill_assistant_text: String::new(),
            event_tx,
            max_decode_tokens: 0,
            sampling: SamplingConfig::compression(),
            submitted_at: Instant::now(),
            reprojection: None,
            belief: PriorBelief::default(),
            seal_action: SealAction::CompressionTurn { job_id },
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
            triggers: Arc::new(TriggerRegistry::new()),
        });
        Ok(())
    }

    /// Build a turn's [`TurnLayout`] at seal time from the submit-time content
    /// boundaries and the per-half display text. The dialect marker lengths come
    /// from the scheduler's `boundary_markers`; when the assistant body carries a
    /// `<think>…</think>` block it is split into a real `Thinking` segment whose
    /// token length is measured by re-tokenising the block (the answer span
    /// absorbs any tokeniser round-trip remainder, so the layout still tiles).
    #[allow(clippy::too_many_arguments)]
    ///
    /// `ethereal_thinking` chooses how the `<think>…</think>` block is
    /// represented: `false` keeps its K/V (the grid still contains the reasoning
    /// tokens — a REAL `Thinking` span); `true` drops its K/V (the grid was
    /// re-prefilled reasoning-free, so the block is an ETHEREAL `Thinking`
    /// segment whose prose is kept for display but never materializes into the
    /// slot). The clean-reprefill seal passes `true`.
    #[allow(clippy::too_many_arguments)]
    fn build_turn_layout(
        &self,
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        total: u32,
        user_text: String,
        assistant_text: String,
        no_think: bool,
        ethereal_thinking: bool,
    ) -> TurnLayout {
        let im_end_len = self.boundary_markers.user_end.len() as u32;
        let assistant_start_len = self.boundary_markers.assistant_start.len() as u32;
        let layout = TurnLayout::from_flat_grid(
            user_content_start,
            user_content_end,
            assistant_content_start,
            total,
            im_end_len,
            assistant_start_len,
            user_text,
            (!assistant_text.is_empty()).then(|| assistant_text.clone()),
            no_think,
        );
        // Code_read tool-exchange turns bake their assistant→user→assistant
        // sub-structure into `assistant_text` as dialect role markers; split the
        // single Assistant span into the real sub-segments so the layout — and
        // the details view built from it — reflects the true role boundaries.
        //
        // Checked BEFORE the `<think>` split: a tool-exchange turn is identified
        // unambiguously by its baked role boundaries, and its DECODED closing
        // summary may itself contain a `<think>` block (the scope grid carries no
        // `/no_think`). Were the think split to run first, that summary-internal
        // reasoning would hijack the layout and the tool exchange would never be
        // split. A normal assistant reply never contains the role-boundary marker,
        // so this branch is a no-op for ordinary turns and they fall through to the
        // think split below.
        if let Some(subs) =
            self.tool_exchange_segments(&assistant_text, assistant_content_start, total)
        {
            return layout.with_assistant_split(subs);
        }
        // Split the `<think>…</think>` reasoning out of the assistant body. Its
        // token length is measured by re-tokenising the block; `ethereal_thinking`
        // decides whether that length is a real K/V span or dropped.
        if let (Some(o), Some(c)) = (
            assistant_text.find("<think>"),
            assistant_text.find("</think>"),
        ) {
            if c >= o {
                let block = assistant_text[o..c + "</think>".len()].to_string();
                let think_len = self
                    .tokenizer
                    .encode(block.as_str(), false)
                    .map(|t| t.get_ids().len() as u32)
                    .unwrap_or(0);
                return layout.with_thinking_split(block, think_len, ethereal_thinking);
            }
        }
        layout
    }

    /// If `asst_text` is a code_read tool exchange — `<tool_call>`, the
    /// call→response boundary (`assistant_end` + `user_start`), `<tool_response>`,
    /// the response→close boundary (`user_end` + `assistant_start`), then the
    /// confirmation — split its single `[asst_start, total)` assistant span into
    /// the real `Assistant → ImEnd → UserStart → User → ImEnd → AssistantStart →
    /// Assistant` sub-segments. Boundary token offsets come from re-tokenising the
    /// byte prefixes, clamped monotonic into the span so the pieces always tile
    /// exactly (`validate_tiling` guards the seal). Returns `None` for an ordinary
    /// single-body assistant turn (a normal reply never contains the boundary).
    fn tool_exchange_segments(
        &self,
        asst_text: &str,
        asst_start: u32,
        total: u32,
    ) -> Option<Vec<TurnSegment>> {
        let g = &self.boundary_markers;
        let call_to_resp = format!("{}{}", g.assistant_end_str, g.user_start_str);
        let resp_to_close = format!("{}{}", g.user_end_str, g.assistant_start_str);
        if call_to_resp.is_empty() || resp_to_close.is_empty() {
            return None;
        }
        // The tool_call is the read_file JSON (no markers), so the FIRST
        // call→response boundary is the real one. The tool_response is arbitrary
        // source, which could itself contain the response→close marker string, so
        // take the LAST occurrence: the real boundary sits just before the short,
        // marker-free confirmation, so `rfind` can't be fooled by embedded markup.
        let b1 = asst_text.find(&call_to_resp)?; // tool_call ends here
        let after1 = b1 + call_to_resp.len(); // tool_response starts here
        let b2 = after1 + asst_text[after1..].rfind(&resp_to_close)?; // tool_response ends
        let after2 = b2 + resp_to_close.len(); // confirmation starts here

        let span = total.saturating_sub(asst_start);
        let tlen = |s: &str| {
            self.tokenizer
                .encode(s, false)
                .map(|t| t.get_ids().len() as u32)
                .unwrap_or(0)
        };
        // Absolute grid offset of the byte prefix `asst_text[..byte]`, clamped
        // into the span so a tokenizer merge across a join can never break tiling.
        let at = |byte: usize| asst_start + tlen(&asst_text[..byte]).min(span);
        let im_end_len = tlen(&g.user_end_str);

        let a1 = at(b1); // end of tool_call
        let a2 = at(after1).max(a1); // end of call→response boundary
        let a3 = at(b2).max(a2); // end of tool_response
        let a4 = at(after2).max(a3).min(total); // end of response→close boundary
        let im1 = im_end_len.min(a2 - a1);
        let im2 = im_end_len.min(a4 - a3);

        Some(vec![
            TurnSegment::Assistant {
                text: Some(asst_text[..b1].to_string()),
                kv: KvSpan::new(asst_start, a1 - asst_start),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(a1, im1)),
            },
            TurnSegment::Glue {
                marker: GlueKind::UserStart,
                kv: Some(KvSpan::new(a1 + im1, (a2 - a1) - im1)),
            },
            TurnSegment::User {
                text: asst_text[after1..b2].to_string(),
                kv: KvSpan::new(a2, a3 - a2),
            },
            TurnSegment::Glue {
                marker: GlueKind::ImEnd,
                kv: Some(KvSpan::new(a3, im2)),
            },
            TurnSegment::Glue {
                marker: GlueKind::AssistantStart,
                kv: Some(KvSpan::new(a3 + im2, (a4 - a3) - im2)),
            },
            TurnSegment::Assistant {
                text: Some(asst_text[after2..].to_string()),
                kv: KvSpan::new(a4, total.saturating_sub(a4)),
            },
        ])
    }

    /// Defer a finished dialogue turn's seal: re-prefill it with the
    /// `<think>…</think>` reasoning stripped so the SEALED K/V is reasoning-free
    /// (a future projection of the turn can no longer attend its own thoughts),
    /// then seal + fire the deferred `Done` once the re-prefill wave completes.
    /// The reasoning TEXT is kept as an ethereal `Thinking` segment, so
    /// display / history / summaries are unchanged. The caller has already
    /// truncated the slot to `seal_block_from` (its go/no-go); this only builds
    /// the clean grid, stashes the [`PendingTurnSeal`], and enqueues the unit.
    fn enqueue_clean_turn_reprefill(
        &mut self,
        parent_id: SequenceId,
        seal_block_from: usize,
        state: DecodeState,
        text: String,
        stats: TurnStats,
    ) {
        // Forwarded generated: drop the last, un-forwarded sampled token (its K/V
        // never landed in the slot), matching the immediate-seal path.
        let forwarded_generated: &[u32] = state
            .generated_tokens
            .split_last()
            .map(|(_, rest)| rest)
            .unwrap_or(&[]);
        // Reasoning-free answer: strip `<think>…</think>` (+ the whitespace around
        // it) while keeping the answer's own formatting. A turn with no think
        // block re-prefills byte-identical (a clean no-op).
        let clean_answer = self.strip_think_from_tokens_keep_layout(forwarded_generated);
        // Clean grid: [user_msg][user_end][assistant_start] (already
        // `/no_think`-free — that glue lives in the prefix, before
        // `seal_block_from`) + the reasoning-free answer + the closing tail.
        let mut clean_tokens: Vec<u32> = Vec::with_capacity(
            state.prefill_tokens.len() + clean_answer.len() + state.post_decode_tokens.len(),
        );
        clean_tokens.extend_from_slice(&state.prefill_tokens);
        clean_tokens.extend_from_slice(&clean_answer);
        clean_tokens.extend_from_slice(&state.post_decode_tokens);
        let total = clean_tokens.len() as u32;

        // Display text (verbatim, reasoning included): prefill turns supply it,
        // decode turns fall back to the streamed text.
        let assistant_text = if state.prefill_assistant_text.is_empty() {
            text.clone()
        } else {
            state.prefill_assistant_text.clone()
        };
        // The `<think>` block becomes an ETHEREAL `Thinking` segment over the
        // reasoning-free grid — text kept, K/V dropped.
        let layout = self.build_turn_layout(
            state.user_content_start,
            state.user_content_end,
            state.assistant_content_start,
            total,
            state.user_text.clone(),
            assistant_text,
            state.no_think,
            true,
        );

        let pending_id = self.next_turn_seal_id;
        self.next_turn_seal_id += 1;

        // Private sink for the re-prefill unit's `PrefillWork`; the real caller
        // channel (`state.event_tx`) fires `Done` from `complete_turn_reprefill`.
        let (sink_tx, sink_rx) = crossbeam::channel::unbounded();

        self.pending_turn_seals.insert(
            pending_id,
            PendingTurnSeal {
                parent_id,
                seal_block_from,
                layout,
                token_ids: clean_tokens.clone(),
                tags: state.tags,
                event_tx: state.event_tx,
                done_text: text,
                done_token_ids: state.generated_tokens,
                stats,
                _sink_rx: sink_rx,
            },
        );

        // Enqueue the clean grid as a `max_decode=0` prefill unit — it rides the
        // SAME wave as the next turn's prefill and any summaries, so the
        // re-prefill batches for maximum parallelism. `complete_turn_reprefill`
        // seals + fires the deferred `Done`.
        self.prefill_queue.push_back(PrefillWork {
            sequence_id: parent_id,
            tokens: TokenBuffer::from(clean_tokens),
            prefill_text: String::new(),
            user_text: String::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            tags: Vec::new(),
            prefill_assistant_text: String::new(),
            event_tx: sink_tx,
            max_decode_tokens: 0,
            sampling: SamplingConfig::compression(),
            submitted_at: Instant::now(),
            reprojection: None,
            belief: PriorBelief::default(),
            seal_action: SealAction::TurnReprefill { pending_id },
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
            triggers: Arc::new(TriggerRegistry::new()),
        });
    }

    /// Seal a finished dialogue turn once its reasoning-free re-prefill completes
    /// on the wave: snapshot the clean K/V (via the normal `SealAction::Turn`
    /// write), drop the slot's chunks, and fire the deferred `Done` (the client's
    /// full reply + the seal result). Mirrors `complete_compression_turn`.
    fn complete_turn_reprefill(&mut self, pending_id: u64) {
        let Some(pending) = self.pending_turn_seals.remove(&pending_id) else {
            return;
        };
        let PendingTurnSeal {
            parent_id,
            seal_block_from,
            layout,
            token_ids,
            tags,
            event_tx,
            done_text,
            done_token_ids,
            stats,
            _sink_rx,
        } = pending;

        let turn_content = TurnContent {
            role: Role::Assistant,
            tags,
            layout,
            token_ids: TokenBuffer::from(token_ids),
        };
        let seal_result = self
            .perform_seal_and_write(
                parent_id,
                seal_block_from,
                &SealAction::Turn,
                Some(turn_content),
            )
            .unwrap_or_else(|e| {
                tracing::warn!("clean turn seal failed for slot {}: {}", parent_id, e);
                None
            });

        // Drop the slot's chunks now the residence owns them (the next projection
        // rebuilds from the substrate) — same housekeeping as the immediate seal.
        if let Err(e) = self.session.truncate_sequence_to_blocks(parent_id.0, 0) {
            tracing::warn!(
                "post-seal slot truncate failed for slot {}: {}",
                parent_id,
                e
            );
        }

        let _ = event_tx.send(TurnEvent::Done(TurnResponse {
            text: done_text,
            token_ids: done_token_ids,
            stats,
            seal: seal_result,
        }));
    }

    /// Seal a re-prefilled compressed turn once the shared wave finishes its
    /// prefill. Snapshots the slot's freshly-computed (role-coherent) K/V,
    /// records the turn from its [`PendingCompressionSeal`], frees the slot, and
    /// replies to the summariser with the new `TurnIndex`.
    fn complete_compression_turn(&mut self, slot: SequenceId, job_id: u64) {
        let pending = match self.pending_compression_seals.remove(&job_id) {
            Some(p) => p,
            None => {
                // No pending entry (e.g. enqueue raced a teardown) — just reclaim.
                self.free_summary_slot(slot);
                return;
            }
        };
        let block_count = self.session.sequence_block_count(slot.0).unwrap_or(0);
        let sealed_gpu = match self.session.snapshot_sequence_per_layer(slot.0) {
            Ok(snap) => slice_per_layer_sealed(&snap, 0, block_count),
            Err(e) => {
                let _ = pending.response_tx.send(Err(ProbeError::Soft(format!(
                    "SubmitSummaryProbe: reproject snapshot: {e}"
                ))));
                self.free_summary_slot(slot);
                return;
            }
        };
        // Capture the node's wide per-token `sign(Q)` while the slot is still
        // live — the re-prefill's K/V is freshly R16 here, exactly as at a
        // normal turn seal. The whole slot is the marker-framed turn
        // (`reprojection: None`), so the range is 1:1 with the Tokens record.
        // Live seal (code-read roundtrip summary turns) — count it for the GUI's
        // sealing phase, timing the dominant sig-gather cost.
        let t_sig = Instant::now();
        let wide_sigs = self.gather_wide_sigs(slot, (0, block_count));
        self.wave_stats
            .add_seal(0, t_sig.elapsed().as_micros() as u64, 0);
        // The slice holds RAII `ChunkGid` clones, so the K/V survives the free.
        self.free_summary_slot(slot);
        let block_end = sealed_gpu.first().map(|s| s.chunks.len()).unwrap_or(0);
        let token_count = pending.token_ids.len();

        let timeline = pending.target.timeline;
        let conversation = pending.conversation;
        let response_tx = pending.response_tx;
        // Kept for the post-injection log + token persistence below — the rest
        // move into the write.
        let summary_question = pending.layout.user_text().to_string();
        let summary_answer = pending.layout.assistant_text().unwrap_or_default();
        // A summary node stands in for the real turns it compressed and is injected
        // as a synthetic user→assistant exchange. If EITHER half's decode pass
        // produced no text, the node is hollow — it would silently drop that slice
        // of history behind an empty slot. Refuse to seal it and surface the
        // failure to the summariser rather than committing a malformed summary.
        if summary_question.trim().is_empty() || summary_answer.trim().is_empty() {
            let msg = format!(
                "summary seal aborted for timeline {timeline}: a compression pass produced an \
                 empty half (question {} chars, answer {} chars) — refusing to seal a hollow \
                 summary node",
                summary_question.trim().chars().count(),
                summary_answer.trim().chars().count(),
            );
            tracing::error!(
                target: "candle_conversation::summariser",
                timeline = %timeline,
                "{msg}",
            );
            let _ = response_tx.send(Err(ProbeError::Permanent(msg)));
            return;
        }
        let persist_token_ids = pending.token_ids.clone();
        let write = TurnPartWrite {
            layout: pending.layout,
            token_ids: TokenBuffer::from(pending.token_ids),
            token_count,
            // The node inherits the union of its children's gather-scope
            // tags, so tag-scoped provenance galleries admit the summary in
            // place of the turns it compresses; untagged (dialogue) children
            // yield an untagged summary.
            tags: pending.tags.clone(),
            block_start: 0,
            block_end: block_end as u64,
            sealed_gpu: Some(Arc::new(sealed_gpu)),
        };
        let idx = match conversation
            .record_turn(timeline, Role::Assistant, write, |seqs| Ok(seqs.to_vec()))
        {
            Ok(idx) => idx,
            Err(e) => {
                let _ = response_tx.send(Err(ProbeError::Soft(format!(
                    "SubmitSummaryProbe: record compressed turn: {e}"
                ))));
                return;
            }
        };

        // Persist the summary's token ids. `record_turn` only declared the turn
        // stream; the normal seal path writes the `Tokens` record separately, so
        // the compression path must too — otherwise the compressed text is
        // unrecoverable from the substrate (only its K/V chunks survive).
        let stream_id = turn_stream_id(timeline.raw(), idx.0);
        if let Err(e) = conversation.persist_tokens_only(stream_id, &persist_token_ids) {
            tracing::warn!("persist summary tokens failed: {e}");
        }
        // Persist the node's wide-Q signature under the same stream — the
        // gallery entry a provenance scan matches against the summary.
        if !wide_sigs.is_empty() {
            if let Err(e) =
                conversation.persist_wide_q_sigs(stream_id, &encode_wide_sigs(&wide_sigs))
            {
                tracing::warn!("persist summary wide-Q sigs failed: {e}");
            }
        }
        // Synthesize + persist the node's projection event: the mandatory
        // provenance linkage naming the node itself and the turns it covers
        // by `(timeline, index)`, so a wide-Q hit on the summary resolves to
        // real turns. Compression frames run no projection — the event's
        // system list is honestly empty.
        {
            let (layer_name, group_name) = self
                .timeline_projections
                .get(&timeline)
                .map(|b| {
                    let schema = b.schema();
                    (
                        layer_name_of_group(schema, pending.target.group)
                            .unwrap_or_default()
                            .to_string(),
                        group_name_of(schema, pending.target.group)
                            .unwrap_or_default()
                            .to_string(),
                    )
                })
                .unwrap_or_default();
            let children_meta: Vec<(TurnIndex, TurnKind, u32)> = {
                let read = conversation.read();
                pending
                    .children
                    .iter()
                    .map(|&c| {
                        let kind = read
                            .tree_meta_of(timeline, c)
                            .map(|m| m.kind)
                            .unwrap_or(TurnKind::Normal);
                        (c, kind, read.turn_token_count_of(timeline, c) as u32)
                    })
                    .collect()
            };
            let event = summary_node_event(
                &layer_name,
                &group_name,
                timeline.raw(),
                (idx, pending.kind, token_count as u32),
                children_meta,
            );
            if let Err(e) =
                conversation.persist_projection_events(stream_id, &encode_events(&[event]))
            {
                tracing::warn!("persist summary projection event failed: {e}");
            }
        }
        self.persist_trigger.fire();

        tracing::info!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            index = idx.0,
            tokens = token_count,
            blocks = block_end,
            question = %summary_question,
            answer = %summary_answer,
            "compression: summary turn decoded and injected into substrate",
        );
        let _ = response_tx.send(Ok(idx));
    }

    /// The most scratch slots the scope-ingest pump keeps live at once — the
    /// parallelism ceiling for code-scope prefills. Bounds the model's sequence
    /// slots a wide fan-out can consume; when `create_sequence` refuses (the
    /// model is at capacity anyway), the pump re-queues and waits for a slot to
    /// free, so this is an aspiration, not a hard requirement.
    ///
    /// 20 (was 24): each live scope pins its KV in VRAM, so a wide fan-out is a
    /// major peak-VRAM driver on a memory-tight card. Trimming to 20 leaves a
    /// little more headroom for the compress-to-free path (see the eviction
    /// reserve in `candle-nn`'s `alloc.rs`) without meaningfully slowing ingest.
    const MAX_SCOPE_SLOTS: usize = 20;

    /// Token cap on a code-scope's two-sentence summary decode. Loose enough that
    /// a genuine two-sentence summary is never truncated mid-thought; EOS ends it
    /// earlier in the common case.
    const SCOPE_SUMMARY_MAX_TOKENS: usize = 100;

    /// Static ceiling on the ragged prefill forward's width — how many in-flight
    /// prefills coalesce into one `forward_batched`. The [`Self::admit_window`]
    /// congestion window rides below this and shrinks it under VRAM pressure.
    pub(super) const MAX_PREFILL_WIDTH: usize = 24;

    /// Throughput floor the congestion window never closes past — one prefill
    /// always in flight so the engine keeps making progress even under sustained
    /// pressure (a lone oversized turn is then bounded by the per-arena VRAM gate).
    const MIN_PREFILL_WIDTH: usize = 1;

    /// Multiplicative-decrease the admission window: halve it toward
    /// [`Self::MIN_PREFILL_WIDTH`]. Called when VRAM pressure survives an eviction
    /// pass (`promote`/`pump`) or a prefill forward reports device-OOM — the
    /// current width is unsustainable, so back off hard and let the additive
    /// reopen ([`Self::grow_admit_window`]) probe back up once pressure clears.
    fn shrink_admit_window(&mut self) {
        let before = self.admit_window;
        self.admit_window = narrow_window(self.admit_window, Self::MIN_PREFILL_WIDTH);
        if self.admit_window != before {
            tracing::info!(
                target: "candle_conversation::scheduler::timing",
                admit_window = self.admit_window,
                was = before,
                "admission window narrowed under VRAM pressure"
            );
        }
    }

    /// Additive-increase the admission window by one, toward
    /// [`Self::MAX_PREFILL_WIDTH`]. Called once per scheduler loop when VRAM is
    /// not under pressure, so a transient pressure episode's backoff heals
    /// gradually (AIMD) rather than snapping straight back to full width and
    /// re-tripping on the next wide wave.
    fn grow_admit_window(&mut self) {
        self.admit_window = widen_window(self.admit_window, Self::MAX_PREFILL_WIDTH);
    }

    /// Register a submitted scope into its file batch and queue it for the fair
    /// pump. Builds the scope turn's layout from the content offsets; the batch
    /// is created (sized to `scope_total`) on the file's first scope. Returns
    /// `Err` only for a lost timeline or a malformed / duplicate submission — the
    /// GPU work happens later, in `pump_scope_prefills`.
    #[allow(clippy::too_many_arguments)]
    fn handle_prefill_scope(
        &mut self,
        timeline: TimelineId,
        scope_index: u32,
        scope_total: u32,
        tokens: TokenBuffer,
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        user_text: String,
        assistant_text: String,
        tags: Vec<String>,
        response_tx: Sender<Result<TurnIndex, ConversationError>>,
        on_prefilled: ScopeProgressFn,
    ) -> Result<(), ConversationError> {
        if scope_total == 0 || scope_index >= scope_total {
            return Err(ConversationError::Channel(format!(
                "PrefillScope: scope_index {scope_index} out of range for total {scope_total}"
            )));
        }
        // Resolve the conversation that owns this timeline — the same slot-registry
        // lookup the summary probe uses. Needed to record the turns at flush.
        let conversation = self
            .slot_conversations
            .values()
            .find(|c| c.read().timeline_target(timeline).is_some())
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "PrefillScope: no conversation registered for timeline {timeline}"
                ))
            })?;
        // Carry the layout inputs (offsets + texts) unbuilt: the scope's closing
        // assistant segment is DECODED, so the layout is finalised only once
        // `complete_scope_summary_decoded` has the summary tokens. Clamp the offsets to
        // the prefill grid here (monotonic) so a tokenizer merge across a join
        // can't invert the windows. Scopes cold-prefill under `/no_think`.
        let total = tokens.len() as u32;
        let user_content_start = user_content_start.min(total);
        let user_content_end = user_content_end.min(total).max(user_content_start);
        let assistant_content_start = assistant_content_start.min(total).max(user_content_end);
        let layout_inputs = ScopeLayoutInputs {
            user_content_start,
            user_content_end,
            assistant_content_start,
            user_text,
            assistant_text,
        };
        let token_ids: Vec<u32> = tokens[..].to_vec();
        let token_count = token_ids.len();

        // Get-or-create the file batch. Every scope of a file passes the same
        // `scope_total`; the first submit sizes the batch.
        let batch = self
            .scope_batches
            .entry(timeline)
            .or_insert_with(|| PendingScopeBatch {
                conversation,
                total: scope_total,
                sealed: (0..scope_total).map(|_| None).collect(),
                landed: (0..scope_total).map(|_| false).collect(),
                responders: (0..scope_total).map(|_| None).collect(),
                flushed: 0,
                on_prefilled,
            });
        // The batch was sized by the file's first scope; a later scope disagreeing
        // on the total (client bug) would index out of range — reject it instead.
        if scope_index >= batch.total {
            return Err(ConversationError::Channel(format!(
                "PrefillScope: scope {scope_index} exceeds this file's batch total {}",
                batch.total
            )));
        }
        if batch.responders[scope_index as usize].is_some() {
            return Err(ConversationError::Channel(format!(
                "PrefillScope: scope {scope_index} of timeline {timeline} submitted twice"
            )));
        }
        batch.responders[scope_index as usize] = Some(response_tx);

        self.scope_pending
            .entry(timeline)
            .or_default()
            .push_back(QueuedScope {
                scope_index,
                tokens: token_ids,
                layout_inputs,
                token_count,
                tags,
            });
        self.scope_submitted.entry(timeline).or_insert(0);
        // A scope-ingest file timeline is append-only by construction — its sealed
        // turns are never re-attended until query time, so mark it for the
        // gentle-early ladder (`demote_cold_ingest_if_pressured`) to demote its
        // warm-backed hot KV at ~50% capacity.
        self.ingest_timelines.insert(timeline);
        self.pump_scope_prefills();
        Ok(())
    }

    /// Drain queued scopes onto scratch slots — fairly (least-advanced file
    /// first) and bounded by [`Self::MAX_SCOPE_SLOTS`]. Each scope cold-prefills
    /// on a fresh slot as a `max_decode=0` unit on the shared wave, so a file's
    /// scopes (and scopes across files) batch instead of prefilling serially.
    /// Called on every submit, every scope completion (a freed slot), and once
    /// per scheduler loop.
    fn pump_scope_prefills(&mut self) {
        // Each live scope pins its KV in VRAM, so the concurrent scope working set
        // is a primary peak-VRAM driver during an upload / code-read burst. Cap it
        // by the AIMD `admit_window` as well as the static `MAX_SCOPE_SLOTS`, so a
        // big file's fan-out can't pin more KV than a busy card can hold alongside
        // the forward's transient peak.
        let cap = Self::MAX_SCOPE_SLOTS.min(self.admit_window.max(Self::MIN_PREFILL_WIDTH));
        while self.active_scope_slots < cap {
            // Before pinning another scope's KV, if VRAM is already tight, shed hot
            // KV to the substrate first; if pressure survives that, narrow the
            // window and stop admitting scopes this pass (the ≥1 kept in flight
            // elsewhere still makes progress). This is the anticipatory half of the
            // fix: bound the pinned working set *before* the wide forward runs,
            // rather than only reacting once its transient peak has OOM'd.
            if self.active_scope_slots > 0 && self.vram_under_pressure() {
                if self.relieve_vram_pressure("pump", prefill::VramPhase::Load) {
                    self.shrink_admit_window();
                    break;
                }
            }
            let Some(timeline) = self.fairest_scope_timeline() else {
                break;
            };
            let Some(queued) = self
                .scope_pending
                .get_mut(&timeline)
                .and_then(|q| q.pop_front())
            else {
                break;
            };
            // Drop the bucket once empty so the fair scan skips it.
            if self
                .scope_pending
                .get(&timeline)
                .map(|q| q.is_empty())
                .unwrap_or(false)
            {
                self.scope_pending.remove(&timeline);
            }

            // Resolve the conversation + projection target for the scratch slot.
            let (conversation, target) = match self.scope_batches.get(&timeline) {
                Some(b) => {
                    let conv = b.conversation.clone();
                    let tgt = conv.read().timeline_target(timeline).map(|(layer, group)| {
                        ProjectionTarget {
                            layer,
                            group,
                            timeline,
                        }
                    });
                    (conv, tgt)
                }
                None => continue, // batch gone (failed) — drop the queued scope
            };
            let Some(target) = target else {
                self.fail_scope(
                    timeline,
                    queued.scope_index,
                    ConversationError::Channel(format!(
                        "PrefillScope: timeline {timeline} lost its projection target"
                    )),
                );
                continue;
            };
            let slot = match self.create_sequence(conversation, Some(target)) {
                Ok(s) => s,
                Err(_) => {
                    // The model is at sequence capacity. Re-queue at the front and
                    // stop pumping — a scope completion (or the per-loop pump) frees
                    // a slot and retries. Undo the fairness increment we haven't
                    // applied yet (we increment only on a successful slot below).
                    self.scope_pending
                        .entry(timeline)
                        .or_default()
                        .push_front(queued);
                    break;
                }
            };
            *self.scope_submitted.entry(timeline).or_insert(0) += 1;
            self.active_scope_slots += 1;

            let prefill_tokens: Vec<u32> = queued.tokens.clone();
            self.pending_scope_prefills.insert(
                slot,
                PendingScopePrefill {
                    timeline,
                    scope_index: queued.scope_index,
                    layout_inputs: queued.layout_inputs,
                    token_ids: queued.tokens,
                    token_count: queued.token_count,
                    tags: queued.tags,
                    sealed_layout: None,
                    sealed_token_ids: None,
                },
            );
            // Private event sink so the prefill's progress/error sends never fail;
            // dropped when the slot is freed (`free_summary_slot`).
            let (event_tx, event_rx) = crossbeam::channel::unbounded();
            self.compression_event_sinks.insert(slot, event_rx);
            self.prefill_queue.push_back(PrefillWork {
                sequence_id: slot,
                tokens: TokenBuffer::from(prefill_tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                prefill_assistant_text: String::new(),
                event_tx,
                max_decode_tokens: 0,
                sampling: SamplingConfig::compression(),
                submitted_at: Instant::now(),
                reprojection: None,
                belief: PriorBelief::default(),
                seal_action: SealAction::ScopeIngest,
                post_decode_tokens: TokenBuffer::default(),
                projection_offsets: Vec::new(),
                staged_composition: None,
                triggers: Arc::new(TriggerRegistry::new()),
            });
        }
    }

    /// The file `timeline` whose pending scopes have been pumped the FEWEST times
    /// — max-min fairness so every file advances rather than the first-submitted
    /// draining to completion first. Ties break on the lowest raw id. `None` when
    /// no file has queued scopes.
    fn fairest_scope_timeline(&self) -> Option<TimelineId> {
        let candidates: Vec<(u64, u32)> = self
            .scope_pending
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(tl, _)| (tl.raw(), self.scope_submitted.get(tl).copied().unwrap_or(0)))
            .collect();
        let chosen = pick_fair_scope(&candidates)?;
        self.scope_pending
            .keys()
            .find(|tl| tl.raw() == chosen)
            .copied()
    }

    /// A scope prefill errored on the wave: answer its responder and free the
    /// slot, then re-pump (a freed slot may admit a queued scope).
    fn fail_scope_ingest(&mut self, slot: SequenceId, err: ConversationError) {
        self.active_scope_slots = self.active_scope_slots.saturating_sub(1);
        let pending = self.pending_scope_prefills.remove(&slot);
        self.free_summary_slot(slot);
        if let Some(p) = pending {
            self.fail_scope(p.timeline, p.scope_index, err);
        }
        self.pump_scope_prefills();
    }

    /// A scope's cold-prefill (`<tool_call>` … `<tool_response>` up to the
    /// tool_response's user_end) finished on the wave. Frame `assistant_start` and
    /// register a bounded two-sentence summary decode on the SAME slot, so the
    /// decode attends over the excerpt in-slot (anchoring the summary to the
    /// code). Mirrors [`Self::finish_compression_pass_setup`], but the decode's
    /// K/V is KEPT (it becomes the scope's closing assistant segment), it seals
    /// through the scope-batch path, and the slot stays counted as an active scope
    /// slot until `complete_scope_summary_sealed` frees it. `SamplingConfig::compression`
    /// is argmax; the grid is already `/no_think`, so the summary is reasoning-free.
    /// On a prefill/sample error, fail just this scope so its siblings still flush.
    fn begin_scope_summary_decode(&mut self, slot: SequenceId) {
        let asst_start = self.boundary_markers.assistant_start.as_ref().clone();
        let turn_start = Instant::now();
        let prefill_logits = match self.run_prefill(slot, &asst_start) {
            Ok(l) => l,
            Err(e) => {
                self.fail_scope_ingest(slot, e);
                return;
            }
        };
        let prefill_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let prefill_token_count = self.session.sequence_offset(slot.0).unwrap_or(0);

        let config = SamplingConfig::compression();
        let mut sstate = match self.sampling_states.remove(&slot) {
            Some(s) => s,
            None => {
                self.fail_scope_ingest(
                    slot,
                    ConversationError::Channel("ScopeSummary: missing sampling state".into()),
                );
                return;
            }
        };
        let first = match self.sample_single(&prefill_logits, &config, &mut sstate) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states.insert(slot, sstate.clone());
                self.fail_scope_ingest(slot, e);
                return;
            }
        };
        self.sampling_states.insert(slot, sstate);

        if self.is_eos(first) {
            // Immediate EOS — the model has nothing to add. Seal the scope with an
            // empty decode buffer: the `assistant_start` framing already gives it a
            // (terse) closing assistant turn, so the alternation still closes cleanly.
            self.complete_scope_summary_decoded(slot, TokenBuffer::default());
            return;
        }

        // Private event sink: the decode's per-token sends go nowhere (a scope
        // summary reports through the batch, not `TurnEvent`), so a dropped
        // receiver is fine — it just marks the decode finished the same way EOS /
        // max_tokens would, and `cleanup_finished` reaps it.
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        self.compression_event_sinks.insert(slot, event_rx);
        let health = {
            let mut hs = DecodeHealthState::new(
                self.health_config.repetition_window,
                self.health_config.health_log_capacity,
            );
            hs.apply_baseline_config(
                self.health_config.entropy_baseline_window,
                self.health_config.entropy_trend_relative_factor,
                self.health_config.entropy_trend_absolute_min_nats,
            );
            hs.skip_entropy_checks = config.temperature <= 0.01;
            hs
        };
        self.active_decodes.insert(
            slot,
            DecodeState {
                event_tx,
                generated_tokens: TokenBuffer::from(vec![first]),
                max_tokens: Self::SCOPE_SUMMARY_MAX_TOKENS,
                sampling_config: config,
                seal_action: SealAction::ScopeSummary,
                prefill_assistant_text: String::new(),
                finished: false,
                decode_start: Instant::now(),
                prefill_ms,
                prefill_token_count,
                turn_start,
                health,
                reprojection: None,
                non_punct_since_reproject: 0,
                last_projection_end: 0,
                post_decode_tokens: TokenBuffer::default(),
                belief: PriorBelief::default(),
                prefill_tokens: TokenBuffer::default(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                in_tool_call: false,
                triggers: Arc::new(TriggerRegistry::new()),
                stencil: None,
                pending_mask: None,
            },
        );
    }

    /// A scope's two-sentence summary decode finished on the SAME slot it
    /// prefilled onto. Strip any `<think>…</think>` block from the summary (empty
    /// under `/no_think`, or a full reasoning leak) and RE-PREFILL the clean
    /// `[excerpt][assistant_start][stripped summary]` grid on the slot — mirroring
    /// the dialogue clean re-prefill ([`Self::enqueue_clean_turn_reprefill`]) and
    /// the compression pass, so the sealed K/V and its wide-Q provenance signature
    /// never carry reasoning. The reasoning-free snapshot + record happens in
    /// [`Self::complete_scope_summary_sealed`] once the re-prefill lands. The slot
    /// stays a live scope slot until then. `generated`'s last sampled token is
    /// dropped (never forwarded).
    fn complete_scope_summary_decoded(&mut self, slot: SequenceId, generated: TokenBuffer) {
        let Some(mut pending) = self.pending_scope_prefills.remove(&slot) else {
            self.active_scope_slots = self.active_scope_slots.saturating_sub(1);
            self.free_summary_slot(slot);
            self.pump_scope_prefills();
            return;
        };
        // Reasoning-free summary: drop the decode's last (un-forwarded) token, then
        // strip the `<think>` block so the sealed summary is plain content.
        let summary_forwarded: &[u32] = generated.split_last().map(|(_, rest)| rest).unwrap_or(&[]);
        let clean_summary = self.strip_think_from_tokens_keep_layout(summary_forwarded);
        // The clean grid the re-prefill forwards: the prefill tokens (up to the
        // tool_response's user_end), the `assistant_start` marker, then the
        // think-stripped summary. A prefill forwards ALL tokens, so this aligns 1:1
        // with the re-prefilled chunk grid (no un-forwarded tail like a decode).
        let asst_start_ids: &[u32] = self.boundary_markers.assistant_start.as_ref();
        let mut clean_tokens = Vec::with_capacity(
            pending.token_ids.len() + asst_start_ids.len() + clean_summary.len(),
        );
        clean_tokens.extend_from_slice(&pending.token_ids);
        clean_tokens.extend_from_slice(asst_start_ids);
        clean_tokens.extend_from_slice(&clean_summary);
        let total = clean_tokens.len() as u32;
        // Rebuild the layout with the STRIPPED summary as the closing assistant
        // segment: `tool_exchange_segments` splits the reconstructed assistant text
        // into its real role sub-segments; the summary is the final Assistant span,
        // filling to `total`. `no_think = false` — the `/no_think` soft-switch is
        // baked in the user body (see `prepare_scope_grid`), not re-emitted glue.
        let summary_text = self
            .tokenizer
            .decode(&clean_summary, true)
            .unwrap_or_default();
        let full_assistant_text = format!(
            "{}{}{}",
            pending.layout_inputs.assistant_text,
            self.boundary_markers.assistant_start_str,
            summary_text,
        );
        let layout = self.build_turn_layout(
            pending.layout_inputs.user_content_start,
            pending.layout_inputs.user_content_end,
            pending.layout_inputs.assistant_content_start,
            total,
            pending.layout_inputs.user_text.clone(),
            full_assistant_text,
            false,
            false,
        );
        let timeline = pending.timeline;
        let scope_index = pending.scope_index;
        pending.sealed_layout = Some(layout);
        pending.sealed_token_ids = Some(clean_tokens.clone());
        self.pending_scope_prefills.insert(slot, pending);

        // Clear the decode's K/V (with its `<think>` tokens) and re-prefill the
        // clean grid on the same slot as a `max_decode=0` unit that rides the
        // shared wave. On a truncate error, fail just this scope.
        if let Err(e) = self.session.truncate_sequence_to_blocks(slot.0, 0) {
            self.active_scope_slots = self.active_scope_slots.saturating_sub(1);
            self.pending_scope_prefills.remove(&slot);
            self.free_summary_slot(slot);
            self.fail_scope(timeline, scope_index, ConversationError::Model(e));
            self.pump_scope_prefills();
            return;
        }
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        self.compression_event_sinks.insert(slot, event_rx);
        self.prefill_queue.push_back(PrefillWork {
            sequence_id: slot,
            tokens: TokenBuffer::from(clean_tokens),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            prefill_assistant_text: String::new(),
            event_tx,
            max_decode_tokens: 0,
            sampling: SamplingConfig::compression(),
            submitted_at: Instant::now(),
            reprojection: None,
            belief: PriorBelief::default(),
            seal_action: SealAction::ScopeReprefill,
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
            triggers: Arc::new(TriggerRegistry::new()),
        });
    }

    /// A scope's reasoning-free re-prefill landed on the wave. Snapshot the clean
    /// K/V (excerpt + `<think>`-stripped summary), gather wide-Q sigs over it (the
    /// provenance anchor — now free of reasoning tokens), and buffer the scope into
    /// its file batch with the clean layout + token ids stashed by
    /// [`Self::complete_scope_summary_decoded`]. `advance_scope_flush` records the
    /// contiguous run of landed scopes. Then free the slot and re-pump.
    fn complete_scope_summary_sealed(&mut self, slot: SequenceId) {
        self.active_scope_slots = self.active_scope_slots.saturating_sub(1);
        let Some(pending) = self.pending_scope_prefills.remove(&slot) else {
            self.free_summary_slot(slot);
            self.pump_scope_prefills();
            return;
        };
        let PendingScopePrefill {
            timeline,
            scope_index,
            sealed_layout,
            sealed_token_ids,
            tags,
            ..
        } = pending;
        let (Some(layout), Some(full_token_ids)) = (sealed_layout, sealed_token_ids) else {
            // The decoded phase always stashes both before enqueuing this unit.
            self.free_summary_slot(slot);
            self.fail_scope(
                timeline,
                scope_index,
                ConversationError::Channel("scope re-prefill missing sealed layout".into()),
            );
            self.pump_scope_prefills();
            return;
        };
        let block_count = self.session.sequence_block_count(slot.0).unwrap_or(0);
        let t_snap = Instant::now();
        let sealed_gpu = match self.session.snapshot_sequence_per_layer(slot.0) {
            Ok(snap) => slice_per_layer_sealed(&snap, 0, block_count),
            Err(e) => {
                self.free_summary_slot(slot);
                self.fail_scope(timeline, scope_index, ConversationError::Model(e));
                self.pump_scope_prefills();
                return;
            }
        };
        let snap_us = t_snap.elapsed().as_micros() as u64;
        // Wide-Q sigs over the fresh (still-R16) clean blocks BEFORE the free — the
        // sealed slice holds RAII `ChunkGid` clones, so the K/V survives too. The
        // grid spans the excerpt AND the reasoning-free summary, so the summary's
        // Q vectors anchor the scope semantically without think-block pollution.
        let t_sig = Instant::now();
        let sigs = self.gather_wide_sigs(slot, (0, block_count));
        let sig_us = t_sig.elapsed().as_micros() as u64;
        self.free_summary_slot(slot);
        let block_end = sealed_gpu.first().map(|s| s.chunks.len()).unwrap_or(0);
        let full_token_count = full_token_ids.len();
        if let Some(batch) = self.scope_batches.get_mut(&timeline) {
            batch.sealed[scope_index as usize] = Some(SealedScope {
                sealed_gpu,
                sigs,
                tags,
                layout,
                token_ids: full_token_ids,
                token_count: full_token_count,
                block_end,
            });
            batch.landed[scope_index as usize] = true;
            // Report this scope's arrival live so the ingest bar + token count
            // climb per scope instead of jumping only when the whole file flushes.
            (batch.on_prefilled)(full_token_count);
        }
        // Record the contiguous run of landed scopes now — its VRAM K/V flows into
        // the persist/eviction pipeline instead of piling up until the file ends.
        let t_flush = Instant::now();
        self.advance_scope_flush(timeline);
        let flush_us = t_flush.elapsed().as_micros() as u64;
        self.wave_stats.add_seal(snap_us, sig_us, flush_us);
        self.pump_scope_prefills();
    }

    /// Fail one scope: answer its responder with `err` and count it landed (as a
    /// failure) so its file batch can still flush its successful siblings. Flushes
    /// now if it was the last outstanding scope.
    fn fail_scope(&mut self, timeline: TimelineId, scope_index: u32, err: ConversationError) {
        {
            let Some(batch) = self.scope_batches.get_mut(&timeline) else {
                return;
            };
            if let Some(tx) = batch.responders[scope_index as usize].take() {
                let _ = tx.send(Err(err));
            }
            // A failed scope leaves `sealed[i] == None` but still counts as landed,
            // so the flush cursor advances past it (its responder is already
            // answered above) instead of stalling the file's remaining scopes.
            batch.landed[scope_index as usize] = true;
        }
        self.advance_scope_flush(timeline);
    }

    /// Record every landed scope of a file into its timeline in scope order, then
    /// answer each responder with the recorded `TurnIndex`. Failed scopes (already
    /// answered with `Err`, `sealed == None`) are skipped, leaving their siblings
    /// contiguous. Fires the persist + summariser triggers so the new turns
    /// migrate and get absorbed into the summary tree (the bridging "summaries in
    /// between" scopes) without waiting for a periodic tick.
    /// Record the contiguous run of *landed* scopes at the front of the file's
    /// batch (from `flushed` onward) as timeline turns, **as they land**, then
    /// retire the batch once the whole file is recorded. Preserves scope order —
    /// only the completed prefix is recorded, so an out-of-order gap (a later
    /// scope landing before an earlier one) waits for the earlier scope.
    ///
    /// Recording a scope hands its sealed GPU K/V to `record_turn`, putting it
    /// under the persist/eviction pipeline immediately — so a large file's scope
    /// K/V is reclaimed incrementally instead of pinning VRAM until the file ends
    /// (the accumulation that otherwise exhausts a memory-tight card, where the
    /// pinned-but-unrecorded snapshots are invisible to hot-turn eviction).
    fn advance_scope_flush(&mut self, timeline: TimelineId) {
        // What to do with the scope at the flush cursor, decided under a short
        // borrow so the `&mut self` record call doesn't overlap the batch borrow.
        enum FlushStep {
            Record {
                sealed: SealedScope,
                responder: Option<Sender<Result<TurnIndex, ConversationError>>>,
                conversation: Conversation,
            },
            Skip, // failed scope: already answered, advance past it
            Wait, // reorder gap: the cursor's scope hasn't landed yet
            Done, // whole file recorded: retire the batch
        }
        let mut any_recorded = false;
        loop {
            let step = {
                let Some(batch) = self.scope_batches.get_mut(&timeline) else {
                    return;
                };
                if batch.flushed >= batch.total {
                    FlushStep::Done
                } else {
                    let i = batch.flushed as usize;
                    if !batch.landed[i] {
                        FlushStep::Wait
                    } else {
                        batch.flushed += 1;
                        match batch.sealed[i].take() {
                            None => FlushStep::Skip,
                            Some(sealed) => FlushStep::Record {
                                sealed,
                                responder: batch.responders[i].take(),
                                conversation: batch.conversation.clone(),
                            },
                        }
                    }
                }
            };
            match step {
                FlushStep::Record {
                    sealed,
                    responder,
                    conversation,
                } => {
                    if self.record_scope_turn(timeline, &conversation, sealed, responder) {
                        any_recorded = true;
                    }
                }
                FlushStep::Skip => {}
                FlushStep::Wait => break,
                FlushStep::Done => {
                    self.scope_batches.remove(&timeline);
                    self.scope_submitted.remove(&timeline);
                    self.scope_pending.remove(&timeline);
                    break;
                }
            }
        }
        if any_recorded {
            self.persist_trigger.fire();
            self.summariser_trigger.fire();
        }
    }

    /// Record one landed scope as its own timeline turn, persisting its token ids
    /// and BDP sigs and answering its responder. Handing the sealed GPU K/V to
    /// `record_turn` places it under the persist pipeline (hot→warm→cold) and
    /// makes it an ordinary evictable turn — the point of recording scopes
    /// incrementally. Returns whether the turn was recorded (`false` on a
    /// record error, which is surfaced on the responder). The turn's block range
    /// is its own `[0..scope]` — scopes never share a cumulative slot; the file
    /// summary is the async tree's root, built from these turns.
    fn record_scope_turn(
        &mut self,
        timeline: TimelineId,
        conversation: &Conversation,
        sealed: SealedScope,
        responder: Option<Sender<Result<TurnIndex, ConversationError>>>,
    ) -> bool {
        let SealedScope {
            sealed_gpu,
            sigs,
            tags,
            layout,
            token_ids,
            token_count,
            block_end,
        } = sealed;
        let persist_token_ids = token_ids.clone();
        let write = TurnPartWrite {
            layout,
            token_ids: TokenBuffer::from(token_ids),
            token_count,
            block_start: 0,
            block_end: block_end as u64,
            sealed_gpu: Some(Arc::new(sealed_gpu)),
            tags,
        };
        let idx = match conversation
            .record_turn(timeline, Role::Assistant, write, |seqs| Ok(seqs.to_vec()))
        {
            Ok(idx) => idx,
            Err(e) => {
                if let Some(tx) = responder {
                    let _ = tx.send(Err(ConversationError::Model(e)));
                }
                return false;
            }
        };
        // Persist the turn's token ids — `record_turn` only declared the turn
        // stream; without this the scope's text is unrecoverable from disk.
        let stream_id = turn_stream_id(timeline.raw(), idx.0);
        if let Err(e) = conversation.persist_tokens_only(stream_id, &persist_token_ids) {
            tracing::warn!("scope ingest persist tokens failed: {e}");
        }
        // Persist the scope turn's wide-Q signature (in-RAM + redo log) — the
        // gallery entry a provenance scan matches against.
        if !sigs.is_empty() {
            if let Err(e) = conversation.persist_wide_q_sigs(stream_id, &encode_wide_sigs(&sigs)) {
                tracing::warn!("scope ingest persist wide-Q sigs failed: {e}");
            }
        }
        if let Some(tx) = responder {
            let _ = tx.send(Ok(idx));
        }
        true
    }

    /// Free a compression scratch slot and its per-slot bookkeeping. Called on
    /// every path out of a compression pass (success or error). The decoded
    /// delta's RAII `ChunkGid`s keep the arena chunks alive after this returns.
    fn free_summary_slot(&mut self, slot: SequenceId) {
        let _ = self.session.free_sequence(slot.0);
        self.slot_conversations.remove(&slot);
        let freed_target = self.slot_targets.remove(&slot);
        self.sampling_states.remove(&slot);
        self.slot_projection_state.remove(&slot);
        self.compression_event_sinks.remove(&slot);
        self.prune_ingest_timeline(freed_target);
    }

    /// Drop a freed slot's timeline from [`Self::ingest_timelines`] once no
    /// other live slot still targets it. Without this, the set grows by one
    /// entry per ingested file for the daemon's lifetime (a slow leak, and a
    /// growing scan for [`demote_cold_ingest_if_pressured`]). Called from every
    /// slot-teardown path; a `None` target (raw/non-projecting slot) is a no-op.
    fn prune_ingest_timeline(&mut self, freed_target: Option<ProjectionTarget>) {
        let Some(tgt) = freed_target else {
            return;
        };
        if !self.ingest_timelines.contains(&tgt.timeline) {
            return;
        }
        if !self
            .slot_targets
            .values()
            .any(|t| t.timeline == tgt.timeline)
        {
            self.ingest_timelines.remove(&tgt.timeline);
        }
    }

    // —— Sequence creation ——————————————————————————————————————————

    /// Allocate a fresh empty GPU slot bound to a `Conversation`
    /// (workspace) and an optional projection target.
    /// Registers the conversation handle in
    /// [`Self::slot_conversations`]; when `target` is `Some`, also
    /// registers it in [`Self::slot_targets`] so the [`SubmitTurn`]
    /// handler can resolve it without re-derivation on every submit.
    /// `None` is for raw paths (RULER eval, summarisation) that don't
    /// project from the substrate.
    fn create_sequence(
        &mut self,
        conversation: Conversation,
        target: Option<ProjectionTarget>,
    ) -> Result<SequenceId, ConversationError> {
        let raw_id = self
            .session
            .create_sequence()
            .map_err(ConversationError::Model)?;
        let slot_id = SequenceId(raw_id);

        // Create persistent sampling state for this sequence.
        // This state survives across turns so that DRY penalty can
        // see a rolling window of recent tokens spanning turn boundaries.
        self.sampling_states.insert(
            slot_id,
            SequenceSamplingState::new(self.sampler.vocab_size(), self.sampler.max_recent_len()),
        );

        // Register the conversation handle and (optional) projection
        // target for this slot — see [`Self::slot_conversations`] and
        // [`Self::slot_targets`].
        self.slot_conversations.insert(slot_id, conversation);
        if let Some(target) = target {
            self.slot_targets.insert(slot_id, target);
        }

        tracing::trace!(
            kind = if target.is_some() {
                "targeted"
            } else {
                "scratch"
            },
            "allocated empty slot {}",
            slot_id,
        );
        Ok(slot_id)
    }

    /// Handle the case where generation finishes on the first token (EOS or max=0).
    fn finish_immediately(
        &self,
        seq_id: SequenceId,
        token: u32,
        event_tx: &Sender<TurnEvent>,
        prefill_ms: f64,
        turn_start: Instant,
        prefill_token_count: usize,
    ) {
        let skip = !self.show_special_tokens;
        // Persist verbatim (see the main finish path) — no think-stripping here.
        let text = self.tokenizer.decode(&[token], skip).unwrap_or_default();
        let total_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let _ = event_tx.send(TurnEvent::Done(TurnResponse {
            text,
            token_ids: vec![token].into(),
            stats: TurnStats {
                prefill_ms,
                decode_ms: 0.0,
                total_ms,
                tokens_generated: 1,
                tokens_per_second: 0.0,
                prefill_token_count,
                sequence: self.session.get_sequence_stats(seq_id.0),
            },
            // `finish_immediately` fires before any decode starts and
            // does no seal — the substrate write would have nothing
            // to capture (zero new blocks).
            seal: None,
        }));
    }

    // —— Projection helpers ———————————————————————————————————————————

    /// Write the per-turn projection onto `parent_id`.
    ///
    /// Sets `parent_id`'s contents to the system-prompt baseline
    /// (`system_block_count` blocks), followed by each projected
    /// section's sealed chunks (in `projected_sections` order),
    /// followed by each projected turn's sealed chunks (in
    /// `projected_turns` order).  Every unit's
    /// `Arc<Vec<SealedSequence>>` is fetched from the substrate and
    /// run through the per-sequence upload cache (CPU→GPU
    /// materialisation, deduped against currently-cached chunks);
    /// the resulting GPU sealed chunks are appended to the parent's
    /// tail in one batch.  The substrate's `block_range` for each
    /// injected section / turn is updated to reflect its new
    /// position so `reproject_view`'s mid-decode lookups resolve
    /// correctly.
    ///
    /// Implementation truncates the parent back to
    /// `system_block_count` and appends; the post-condition is the
    /// state above regardless of what was on the parent before.
    /// Inject the projected sections + turns onto `parent_id`'s
    /// slot as windowed sealed chunks.
    ///
    /// Each `SealedChunk` is a `(gid, offset, token_count)` window
    /// into a physical KV chunk.  Sharing a partial chunk with the
    /// source (the conversation that originally wrote it) is safe:
    /// the substrate's record asserts only the bytes the window
    /// covers; subsequent writers extending the chunk land beyond
    /// the recorded window in the R16/F16 storage and don't disturb
    /// the bytes the record references.  Arc-clone keeps the
    /// chunk's memory alive across freed slots; substrate metadata
    /// is atomic; projection sees a consistent snapshot.
    pub(crate) fn apply_projection(
        &mut self,
        parent_id: SequenceId,
        system_block_count: BlockCount,
        segments: &[ProjectionSegment],
    ) -> Result<(), ConversationError> {
        let _ = system_block_count;

        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        // During a submission drain, defer this slot's gap-fill into the one
        // batched forward at drain end (disjoint field borrow from the ctx below).
        // If this slot ALREADY has a deferred plan, that plan is now STALE: the
        // `apply_segments_build` below snapshots + truncates the slot to zero and
        // rebuilds it, so the old plan's reserved gaps — addressed by slot-relative
        // block index — no longer exist. Firing it at drain end would scatter old
        // glue K/V into whatever this rebuild placed at those indices. Drop the
        // superseded plan and defer only this latest projection, so exactly one
        // (current) plan fires against the slot as this call built it.
        let defer = if self.batch_drain_gap_fills {
            self.deferred_glue_fires
                .retain(|p| p.parent_id != parent_id);
            Some(&mut self.deferred_glue_fires)
        } else {
            None
        };
        let state = self.slot_projection_state.entry(parent_id).or_default();
        // Record the sealed working set this projection attends over, so relief
        // eviction can protect it (see `evict_cold_tail`).
        state.working_set = projection_assembler::working_set_from_segments(segments);

        profile::reset();
        let r = projection_assembler::apply_segments(
            state,
            projection_assembler::ApplyContext {
                session: &mut self.session,
                model: &mut self.model,
                device: &self.device,
                conversation: &conversation,
                slot_target,
                parent_id,
                chunk_size: self.chunk_size,
                max_prefill_pass_tokens: self.max_prefill_pass_tokens,
                tokenizer: &self.tokenizer,
                slot_tokens: &mut self.slot_tokens,
                boundary_markers: &self.boundary_markers,
            },
            segments,
            defer,
        );
        profile::report("apply_projection");
        r
    }


    /// Build phase of [`Self::apply_projection`] for the cross-conversation wave:
    /// inject the sealed prefix + collect the glue descriptor, but do NOT fire
    /// the gap-fill forward. The caller fires (batched across slots via
    /// [`projection_assembler::fire_gap_fill_batch`]) then calls
    /// [`Self::apply_projection_finish`].
    fn apply_projection_build(
        &mut self,
        parent_id: SequenceId,
        segments: &[ProjectionSegment],
    ) -> Result<projection_assembler::GapFillPlan, ConversationError> {
        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection_build: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        // Record the sealed working set this projection attends over, so relief
        // eviction can protect it (see `evict_cold_tail`). Same as the single-slot
        // `apply_projection`; the wave path threads build/finish separately, so we
        // stamp it here where the segments are in hand.
        self.slot_projection_state
            .entry(parent_id)
            .or_default()
            .working_set = projection_assembler::working_set_from_segments(segments);
        profile::reset();
        let mut ctx = projection_assembler::ApplyContext {
            session: &mut self.session,
            model: &mut self.model,
            device: &self.device,
            conversation: &conversation,
            slot_target,
            parent_id,
            chunk_size: self.chunk_size,
            max_prefill_pass_tokens: self.max_prefill_pass_tokens,
            tokenizer: &self.tokenizer,
            slot_tokens: &mut self.slot_tokens,
            boundary_markers: &self.boundary_markers,
        };
        projection_assembler::apply_segments_build(&mut ctx, segments)
    }

    /// Finish phase of [`Self::apply_projection`] for the wave: prefill the
    /// deferred user message against the now-committed `[sealed | glue]` and
    /// re-attach the writer tail. Call after the batched fire.
    fn apply_projection_finish(
        &mut self,
        parent_id: SequenceId,
        plan: projection_assembler::GapFillPlan,
    ) -> Result<(), ConversationError> {
        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection_finish: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        let state = self.slot_projection_state.entry(parent_id).or_default();
        let mut ctx = projection_assembler::ApplyContext {
            session: &mut self.session,
            model: &mut self.model,
            device: &self.device,
            conversation: &conversation,
            slot_target,
            parent_id,
            chunk_size: self.chunk_size,
            max_prefill_pass_tokens: self.max_prefill_pass_tokens,
            tokenizer: &self.tokenizer,
            slot_tokens: &mut self.slot_tokens,
            boundary_markers: &self.boundary_markers,
        };
        let r = projection_assembler::apply_segments_finish(state, &mut ctx, plan);
        profile::report("apply_projection");
        r
    }

    // —— Cleanup ————————————————————————————————————————————————————————

    /// Remove finished sequences and send `Done` events.
    fn cleanup_finished(&mut self) {
        let finished_seq_ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| s.finished)
            .map(|(&id, _)| id)
            .collect();

        for seq_id in finished_seq_ids {
            if let Some(state) = self.active_decodes.remove(&seq_id) {
                // The summarise decode completes through the job registry, not
                // the substrate seal path: its body becomes the node's assistant
                // half and is sealed with the derived scope. No view to finalize,
                // no Done event — the summariser blocks on the job's `response_tx`,
                // fired by `complete_compression_pass`.
                if let SealAction::CompressionPass { job_id } = state.seal_action {
                    self.complete_compression_pass(seq_id, job_id, state.generated_tokens);
                    continue;
                }

                // A code-scope's summary decode finished: strip its `<think>` block
                // and re-prefill the clean grid (no view, no Done event — the ingest
                // caller is answered per-scope via the batch channel once the clean
                // re-prefill lands and `complete_scope_summary_sealed` records it).
                if let SealAction::ScopeSummary = state.seal_action {
                    self.complete_scope_summary_decoded(seq_id, state.generated_tokens);
                    continue;
                }

                let decode_ms = state.decode_start.elapsed().as_secs_f64() * 1000.0;
                let total_ms = state.turn_start.elapsed().as_secs_f64() * 1000.0;
                let tokens_generated = state.generated_tokens.len();
                let tokens_per_second = if decode_ms > 0.0 {
                    tokens_generated as f64 / (decode_ms / 1000.0)
                } else {
                    0.0
                };

                let skip = !self.show_special_tokens;
                // Persist the reply VERBATIM — including its `<think>…</think>`
                // reasoning — so the stored text matches the sealed token_ids and
                // a substrate reload renders identically to the live stream. The
                // `<think>` block is part of the response; consumers that don't
                // want it strip at the point of use (the summariser strips its own
                // output), and the KV already carries the reasoning tokens either
                // way. Stripping here would silently drop reasoning on F5/reload.
                let text = self
                    .tokenizer
                    .decode(&state.generated_tokens, skip)
                    .unwrap_or_default();
                // Snapshot stats before any view finalize, since the view
                // slot is dropped during finalize and its sequence stats
                // become unavailable.
                let sequence_stats = self.session.get_sequence_stats(seq_id.0);

                // Auto-finalize: if this sequence is a scheduler-owned view
                // (created by SubmitTurn), transfer its newly-written blocks
                // back to the parent and drop the view slot before sending
                // Done.  This keeps the entire view lifecycle invisible to
                // the caller.
                let finalized_view = self.turn_views.remove(&seq_id);

                // KV-zero check: scan the VIEW slot right BEFORE finalize_view. If this
                // is clean but `substrate-write-seal` (post-finalize) is not, the seal
                // re-org drops blocks; if this is already dirty, the live decode arena
                // held zeros during generation.
                #[cfg(feature = "kv-zero-check")]
                {
                    let layers: Vec<usize> = (0..self.session.num_layers()).collect();
                    let n_real = self.session.sequence_offset(seq_id.0).unwrap_or(0);
                    // Boundary = where THIS turn's decode began = offset − generated,
                    // so `region="own"` = decoded by this turn, `region="prefix"` =
                    // present before decode (inherited projection or this turn's prefill).
                    let turn_start = n_real.saturating_sub(tokens_generated);
                    let layout = self.session.provenance_chunk_layout(seq_id.0, n_real);
                    if let Ok(dump) = self
                        .session
                        .gather_r16_kv_provenance_layers(seq_id.0, &layers, None)
                    {
                        kv_zero_check::scan_gathered(
                            "decode-final-view",
                            seq_id.0,
                            &dump,
                            &layout,
                            self.session.n_kv_head(),
                            self.session.head_dim(),
                            turn_start,
                        );
                    }
                }

                let (seal_slot, seal_block_from) = if let Some(view_state) = finalized_view {
                    if let Err(e) = self.session.finalize_view(
                        seq_id.0,
                        view_state.parent_id.0,
                        view_state.original_borrowed.0,
                    ) {
                        tracing::warn!(
                            "auto-finalize failed for view {} parent {}: {}",
                            seq_id,
                            view_state.parent_id,
                            e,
                        );
                    }
                    self.sampling_states.remove(&seq_id);
                    // The view's KV blocks just got transferred onto
                    // the parent — mirror the same merge in the
                    // diagnostic log so `slot_tokens[parent]`
                    // continues to be a faithful 1:1 mirror of the
                    // parent's KV contents.  Gated on `context-dump`
                    // since the map is empty when the feature is off.
                    #[cfg(feature = "context-dump")]
                    if let Some(view_toks) = self.slot_tokens.remove(&seq_id) {
                        self.slot_tokens
                            .entry(view_state.parent_id)
                            .or_default()
                            .extend(view_toks);
                    }
                    (view_state.parent_id, view_state.turn_start_parent_blocks)
                } else {
                    // Non-view path (rare; not produced by SubmitTurn today
                    // but kept for safety).  Treat the slot itself as parent
                    // and seal from offset 0.
                    (seq_id, 0)
                };

                // Ephemeral probe: the parent slot now holds the full warm KV —
                // system prompt at `[0, seal_block_from)`, the query (+ the single
                // decoded token) at `[seal_block_from, end)`. Gather the query's
                // warm wide-Q here, where the blocks are guaranteed present (the
                // same point a real seal gathers), and stash it for the probe's
                // `ProbeWideSigs` to drain. The turn seals nothing (`SealAction::None`).
                if self.ephemeral_slots.contains(&seal_slot) {
                    let bc = self.session.sequence_block_count(seal_slot.0).unwrap_or(0);
                    let sigs = self.gather_wide_sigs(seal_slot, (seal_block_from, bc));
                    self.ephemeral_sigs.insert(seal_slot, sigs);
                }

                // Harvest the turn's final belief onto the conversation's parent
                // slot: the NEXT turn's submit-time projection seeds from it, so
                // provenance selection evolves across turn boundaries instead of
                // resetting to catalog order.
                //
                // MERGE, never replace: a turn whose projection lacked a
                // collection (tools dial off, projection skipped) harvests a
                // belief without that collection's key, and overwriting would
                // erase the conversation's accumulated lock-on for it. Gated on
                // the slot still being registered — a mid-decode FreeSequence
                // (client disconnect) must not resurrect a dead conversation's
                // belief under a slot id the allocator is about to recycle.
                if finalized_view.is_some() && self.slot_conversations.contains_key(&seal_slot) {
                    self.carried_beliefs
                        .entry(seal_slot)
                        .or_default()
                        .merge_from(&state.belief);
                }

                tracing::info!(
                    target: "sched",
                    tokens = tokens_generated, tps = tokens_per_second as u32,
                    prefill_ms = state.prefill_ms as u64, decode_ms = decode_ms as u64,
                    "turn complete",
                );

                // Post-decode forward pass: append the turn's
                // closing structural tokens (e.g. ChatML's `\n` after
                // `<|im_end|>`) into the slot before sealing, so the
                // turn's pinned KV closes its own brackets.  The
                // model didn't emit these — we synthesise them as if
                // it did.
                if !state.post_decode_tokens.is_empty() {
                    if let Err(e) = self.run_prefill(seal_slot, &state.post_decode_tokens[..]) {
                        tracing::warn!("post-decode prefill failed for slot {}: {}", seal_slot, e,);
                    }
                }

                // Diagnostic: dump the *entire* token stream the
                // kernel saw for this turn — every injected system
                // section, every projected turn, the user prefill,
                // every decoded token, and the post-decode tail.
                // Decoded to text so a human can eyeball it for
                // formatting bugs (missing role markers, malformed
                // ChatML, etc.) at the moment generation completes.
                //
                // Compiled in only when the `context-dump` cargo
                // feature is enabled (default: off).  With the
                // feature off the per-token recording is a no-op
                // and this entire emit is removed by the compiler
                // — no tokenizer.decode() in the hot path of a
                // production build.  When enabled, also requires
                // `RUST_LOG=candle_conversation::scheduler::context_dump=info`
                // to actually emit.
                #[cfg(feature = "context-dump")]
                if tracing::enabled!(
                    target: "candle_conversation::scheduler::context_dump",
                    tracing::Level::INFO,
                ) {
                    if let Some(toks) = self.slot_tokens.get(&seal_slot) {
                        let token_count = toks.len();
                        let decoded = self
                            .tokenizer
                            .decode(toks, false)
                            .unwrap_or_else(|e| format!("<decode error: {e}>"));
                        tracing::info!(
                            target: "candle_conversation::scheduler::context_dump",
                            seq_id = seal_slot.0,
                            token_count,
                            "=== KV cache token dump (turn complete) ===\n{decoded}\n=== end dump ==="
                        );
                    }
                }

                // Clean-reprefill defer (dialogue turns only). The decode's K/V
                // carries the `<think>…</think>` reasoning; sealing it as-is would
                // let a future projection of this turn attend its own thoughts. So
                // re-prefill the turn reasoning-free and seal THAT instead. The
                // truncate that resets the slot to the turn boundary is the
                // go/no-go: on success the seal + `Done` defer to the re-prefill
                // wave (batched with the next wave's normal prefills); on the rare
                // truncate failure we fall through to the immediate,
                // reasoning-bearing seal so the turn is never lost.
                if matches!(state.seal_action, SealAction::Turn)
                    && self
                        .session
                        .truncate_sequence_to_blocks(seal_slot.0, seal_block_from)
                        .is_ok()
                {
                    let stats = TurnStats {
                        prefill_ms: state.prefill_ms,
                        decode_ms,
                        total_ms,
                        tokens_generated,
                        tokens_per_second,
                        prefill_token_count: state.prefill_token_count,
                        sequence: sequence_stats,
                    };
                    self.enqueue_clean_turn_reprefill(
                        seal_slot,
                        seal_block_from,
                        state,
                        text,
                        stats,
                    );
                    continue;
                }

                // Seal-and-write step.  When `seal_action != None`, we
                // snapshot `seal_slot` and apply the appropriate substrate
                // write (turn append or section pin).  The resulting
                // `SealResult` rides along on the Done event so the
                // conversation-side post-actions (cold store) can run
                // without a second round trip.
                let seal_result = match &state.seal_action {
                    SealAction::None => None,
                    action => {
                        // Bundle the per-half display text and the
                        // combined token sequence the seal pinned
                        // into the slot.  Text and tokens carry
                        // distinct shapes here: the substrate stores
                        // `user_text` / `assistant_text` as the
                        // human-readable strings the caller supplied,
                        // while `token_ids` carries the full slot
                        // token sequence (prefill + decoded body +
                        // post-decode tail) so cross-process replay
                        // reconstructs the exact K/V the kernel saw.
                        let turn_content = if matches!(action, SealAction::Turn) {
                            // user_text comes through verbatim from
                            // submit_turn (raw user message, no role
                            // markers, no /no_think prefix).
                            // assistant_text is the model's decoded
                            // body — special tokens skipped, just the
                            // reply the user sees streamed.  The
                            // substrate stores both halves verbatim;
                            // no caller assembles a combined string.
                            // The last entry in state.generated_tokens
                            // was sampled from the most recent forward
                            // pass but never forwarded itself — the
                            // loop terminated (EOS or max_tokens) before
                            // another forward could write its K/V into
                            // the slot.  Drop it so token_ids aligns
                            // 1:1 with the K/V chunk grid.
                            let forwarded_generated: &[u32] = state
                                .generated_tokens
                                .split_last()
                                .map(|(_, rest)| rest)
                                .unwrap_or(&[]);
                            let mut full_tokens: Vec<u32> = Vec::with_capacity(
                                state.prefill_tokens.len()
                                    + forwarded_generated.len()
                                    + state.post_decode_tokens.len(),
                            );
                            full_tokens.extend_from_slice(&state.prefill_tokens);
                            full_tokens.extend_from_slice(forwarded_generated);
                            full_tokens.extend_from_slice(&state.post_decode_tokens);

                            // Prefill turns (repo_map / code_reading) supply the
                            // assistant half verbatim and never decode, so `text`
                            // is empty — store the supplied content. Decode turns
                            // leave it empty and fall back to the decoded `text`.
                            let assistant_text = if state.prefill_assistant_text.is_empty() {
                                text.clone()
                            } else {
                                state.prefill_assistant_text.clone()
                            };
                            let total = full_tokens.len() as u32;
                            // Immediate (non-deferred) seal: no re-prefill ran, so
                            // the grid still contains the reasoning tokens — the
                            // `<think>` block is a REAL span (`ethereal = false`).
                            let layout = self.build_turn_layout(
                                state.user_content_start,
                                state.user_content_end,
                                state.assistant_content_start,
                                total,
                                state.user_text.clone(),
                                assistant_text,
                                state.no_think,
                                false,
                            );
                            Some(TurnContent {
                                role: Role::Assistant,
                                tags: state.tags.clone(),
                                layout,
                                token_ids: TokenBuffer::from(full_tokens),
                            })
                        } else {
                            None
                        };
                        self.perform_seal_and_write(
                            seal_slot,
                            seal_block_from,
                            action,
                            turn_content,
                        )
                        .unwrap_or_else(|e| {
                            tracing::warn!("post-Done seal failed for slot {}: {}", seal_slot, e,);
                            None
                        })
                    }
                };

                // Stateless-slot housekeeping: drop the slot's chunks
                // now that `perform_seal_and_write` has captured them
                // into the substrate residence. The next turn's
                // `apply_projection` rebuilds the slot from substrate
                // anyway, so holding onto these `Arc<ChunkGid>`s
                // between turns just pins arena slots that nothing
                // reads. Truncating to 0 drops the slot's Arc refs;
                // the residence keeps the chunks alive for the next
                // projection inject.
                if let Err(e) = self.session.truncate_sequence_to_blocks(seal_slot.0, 0) {
                    tracing::warn!(
                        "post-seal slot truncate failed for slot {}: {}",
                        seal_slot,
                        e
                    );
                }

                let _ = state.event_tx.send(TurnEvent::Done(TurnResponse {
                    text,
                    token_ids: state.generated_tokens,
                    stats: TurnStats {
                        prefill_ms: state.prefill_ms,
                        decode_ms,
                        total_ms,
                        tokens_generated,
                        tokens_per_second,
                        prefill_token_count: state.prefill_token_count,
                        sequence: sequence_stats,
                    },
                    seal: seal_result,
                }));
            }
        }
    }

    /// Install a persisted section as a **cold-marker** — `cold = Some`,
    /// `hot = None`.  No disk I/O for the chunks at restore time; the
    /// K/V is materialised into VRAM on the first projection that
    /// selects this section, via `elevate_to_hot`'s
    /// `PromotionItemKind::Section` branch.
    ///
    /// Lazy reload pays cold-load only for sections that actually get
    /// projected.  For a `top_k=3` tools collection over 90 entries
    /// that's 3 cold-loads at first projection vs. 90 at startup —
    /// the trade we want once selection is honoured.
    fn restore_section_from_persistence(
        &mut self,
        conversation: &Conversation,
        section_id: SectionId,
        stream_id: StreamId,
        _address: ContentAddress,
        _chunks_per_layer: usize,
        tokens: TokenBuffer,
    ) -> Result<(), ConversationError> {
        let n_layers = self.session.num_layers();

        // 1. Resolve cold refs from the manifest — these point at
        //    each chunk's `(log_offset, record_len, token_count)` in
        //    the redo log.  Installed alone (with hot = None) so the
        //    elevate path can lift the section when a projection
        //    needs it.
        let cold_refs = conversation
            .recover_section_cold_refs(stream_id, n_layers)
            .map_err(ConversationError::Model)?
            .unwrap_or_default();

        // 2. Install as a cold-marker.  `sealed_hot = Vec::new()`
        //    leaves `residence.hot = None`; `cold_refs` lands in
        //    `residence.cold` so `elevate_to_hot` can lift the
        //    section on the first projection that selects it.
        //    `token_count` comes from the cold refs (sum of
        //    per-chunk token counts across one layer's
        //    StoredSequence).
        let token_count = cold_refs.first().map(|s| s.token_count).unwrap_or(0);
        let tokens_arc = Arc::new(tokens[..].to_vec());
        let mut view = conversation.write();
        view.restore_section(
            section_id,
            stream_id,
            token_count,
            Vec::new(),
            cold_refs,
            tokens_arc,
        );
        Ok(())
    }

    /// CPU-only setup for a section ingest: truncate slot, inject prefix,
    /// capture `seal_block_from`, push writer chunk, pin tokens on substrate.
    /// Returns `seal_block_from` (the chunk index lower bound for the seal).
    /// No forward pass — the caller queues an [`ActiveSectionIngest`] entry
    /// so the batched prefill loop can interleave multiple sections.
    pub(super) fn prepare_section_ingest(
        &mut self,
        sequence_id: SequenceId,
        #[cfg_attr(not(feature = "context-dump"), allow(unused_variables))] section_id: SectionId,
        prefix_section_ids: &[SectionId],
        #[cfg_attr(not(feature = "context-dump"), allow(unused_variables))] tokens: &TokenBuffer,
    ) -> Result<usize, ConversationError> {
        // 1. Defensive truncate.
        self.session
            .truncate_sequence_to_blocks(sequence_id.0, 0)
            .map_err(ConversationError::Model)?;

        // 2. Inject substrate-pinned prefix sections (zero-copy Arc clone).
        let has_prefix = !prefix_section_ids.is_empty();
        if has_prefix {
            if let Some(conversation) = self.slot_conversations.get(&sequence_id).cloned() {
                // Ensure every prefix section is hot-resident before reading its
                // sealed KV. A large collection ingest (e.g. the 93-tool catalog)
                // can evict earlier prelude sections from the hot tier, and
                // `section_sealed_of` is hot-only — without this, those sections
                // silently drop out of the prefix and the section being ingested
                // seals against a truncated context. `elevate_to_hot` is a no-op
                // when already resident, and re-lifts from cold otherwise.
                {
                    let backings = self.session.backings().to_vec();
                    let device = self.session.device().clone();
                    let main_stream = match &device {
                        Device::Cuda(d) => d.cuda_stream(),
                        _ => panic!("scheduler: requires a CUDA device"),
                    };
                    let no_turns: Vec<TurnKey> = Vec::new();
                    if let Err(e) = elevate_to_hot(
                        &conversation,
                        &backings,
                        &device,
                        &main_stream,
                        &mut self.elevate_pinned_scratch,
                        &mut self.cold_load_stager,
                        prefix_section_ids,
                        &no_turns,
                    ) {
                        tracing::warn!(
                            "prepare_section_ingest: prefix elevate failed: {e} — \
                             some prefix sections may still be cold"
                        );
                    }
                }
                let n_layers = self.session.num_layers();
                let mut per_layer_chunks: Vec<Vec<candle_nn::kv_cache::SealedChunk>> =
                    (0..n_layers).map(|_| Vec::new()).collect();
                let mut per_layer_token_count: Vec<usize> = vec![0; n_layers];
                let chunk_size = self.chunk_size;
                {
                    let view = conversation.read();
                    for &prefix_id in prefix_section_ids {
                        let Some(sealed) = view.section_sealed_of(prefix_id) else {
                            tracing::warn!(
                                "prepare_section_ingest: prefix section {:?} has no sealed substrate entry — skipping",
                                prefix_id
                            );
                            continue;
                        };
                        if sealed.len() != n_layers {
                            tracing::warn!(
                                "prepare_section_ingest: prefix section {:?} has {} layers, expected {} — skipping",
                                prefix_id,
                                sealed.len(),
                                n_layers
                            );
                            continue;
                        }
                        for layer_idx in 0..n_layers {
                            let layer_seq = &sealed[layer_idx];
                            per_layer_chunks[layer_idx].extend(layer_seq.chunks.iter().cloned());
                            per_layer_token_count[layer_idx] += layer_seq.token_count;
                        }
                    }
                }
                let total_tokens = per_layer_token_count.first().copied().unwrap_or(0);
                if total_tokens > 0 {
                    let per_layer_sealed: Vec<candle_nn::kv_cache::SealedSequence> =
                        per_layer_chunks
                            .into_iter()
                            .zip(per_layer_token_count)
                            .map(|(chunks, toks)| candle_nn::kv_cache::SealedSequence {
                                chunks,
                                token_count: toks,
                                chunk_size,
                                location: candle_nn::kv_cache::ArenaLocation::Gpu,
                            })
                            .collect();
                    self.session
                        .inject_sealed_at_tail(sequence_id.0, &per_layer_sealed)
                        .map_err(ConversationError::Model)?;
                }
            }
        }

        // 3. Capture seal lower bound before pushing the fresh writer chunk.
        let seal_block_from = self
            .session
            .sequence_block_count(sequence_id.0)
            .unwrap_or(0);

        // 4. Push fresh writer chunk so section N doesn't alias the prefix's
        //    partial Arc-shared tail.  Not needed for empty prefix (slot is
        //    empty; ensure_for_batch_entries allocates the first chunk).
        if has_prefix {
            self.session
                .push_empty_writer_chunk(sequence_id.0)
                .map_err(ConversationError::Model)?;
        }

        #[cfg(feature = "context-dump")]
        if tracing::enabled!(
            target: "candle_conversation::scheduler::section_dump",
            tracing::Level::INFO,
        ) {
            let decoded = self
                .tokenizer
                .decode(&tokens[..], false)
                .unwrap_or_else(|e| format!("<decode error: {e}>"));
            tracing::info!(
                target: "candle_conversation::scheduler::section_dump",
                section_id = ?section_id,
                prefix_section_count = prefix_section_ids.len(),
                seal_block_from,
                token_count = tokens.len(),
                "=== section ingest dump ===\n{decoded}\n=== end section ==="
            );
        }

        Ok(seal_block_from)
    }

    /// Seal and write a completed section ingest. Called after the batched
    /// forward pass has consumed all of the section's tokens.
    pub(super) fn finalize_section_ingest(
        &mut self,
        sequence_id: SequenceId,
        section_id: SectionId,
        seal_block_from: usize,
        tokens: Arc<Vec<u32>>,
        address: ContentAddress,
        debug_name: String,
        in_collection: bool,
    ) -> Result<SealResult, ConversationError> {
        #[cfg(feature = "context-dump")]
        if tracing::enabled!(
            target: "candle_conversation::scheduler::section_dump",
            tracing::Level::INFO,
        ) {
            let slot_chunk_count_at_seal = self
                .session
                .sequence_block_count(sequence_id.0)
                .unwrap_or(0);
            let slot_offset_at_seal = self.session.sequence_offset(sequence_id.0).unwrap_or(0);
            tracing::info!(
                target: "candle_conversation::scheduler::section_dump",
                section_id = ?section_id,
                seal_block_from,
                slot_chunk_count_at_seal,
                slot_offset_at_seal,
                "section seal: pre-seal slot stats"
            );
        }
        let seal = self.perform_seal_and_write(
            sequence_id,
            seal_block_from,
            &SealAction::Section {
                section_id,
                tokens,
                address,
                debug_name,
                in_collection,
            },
            None,
        )?;
        let seal = seal.ok_or_else(|| {
            ConversationError::Channel(
                "ingest_section: seal returned None (slot had no content?)".into(),
            )
        })?;

        Ok(seal)
    }

    /// Snapshot `seal_slot`, seal the new blocks
    /// `[seal_block_from, block_count)`, and apply the substrate write
    /// described by `seal_action`.  Returns the [`SealResult`] payload,
    /// or `Ok(None)` when there are no new blocks to seal.
    ///
    /// Near-lossless compression policy for **boundary** sections —
    /// role markers, opening/closing tags, anything outside a schema
    /// `Collection`.
    ///
    /// Boundary sections sit at the head of every slot and every
    /// later token of every later turn attends back over them, so any
    /// K *or* V noise on them is amplified through 48 layers of
    /// attention.  At Q4_KS K + level-5 V the model produced raw
    /// gibberish; at Q8_KS K + level-5 V it stayed coherent but the
    /// model's tool-detection heads dropped below their decision
    /// threshold.  So we override both sides — K to Q8_KS, V to
    /// Q8_0, `compression_level = 0` — every K and V block is 8-bit
    /// with no adaptive Q2/Q4 selection.  ~2.5× smaller than native,
    /// effectively lossless for downstream attention.
    fn section_compression_policy_boundary() -> candle_nn::kv_cache::CompressionPolicy {
        candle_nn::kv_cache::CompressionPolicy::new(0)
            .with_override_k_quant(Some(QuantFormat::Q8_KS))
            .with_override_v_quant(Some(QuantFormat::Q8_0))
    }

    /// Compression policy for **collection-member** sections —
    /// individual tools in a tool catalog, hits in a retrieval list,
    /// anything inside a schema `Collection`.
    ///
    /// The projection's per-turn top-k selection already masks
    /// per-member precision: of N tools in the catalog only k make it
    /// into the slot at any given turn, and the scorer picks them on
    /// relevance, so per-member K/V noise gets averaged out by the
    /// selection itself.  Members tolerate aggressive compression
    /// better than boundary sections, but not *as* aggressive as
    /// turns — a turn is read by a handful of decode steps right
    /// after its seal, a collection member is read by every later
    /// token of every later turn the projection re-selects it into.
    ///
    /// So members sit between the boundary near-lossless policy
    /// (C0 / Q8 / Q8) and the dialogue turn policy: **C4, fully adaptive
    /// for both K and V** — the uniform-K pin removed to match the
    /// engine-wide adaptive-K stress config.
    /// `_turn_policy` is taken to keep the call sites future-proof
    /// for a per-engine override knob; today it's ignored.
    fn section_compression_policy_member(
        _turn_policy: &candle_nn::kv_cache::CompressionPolicy,
    ) -> candle_nn::kv_cache::CompressionPolicy {
        candle_nn::kv_cache::CompressionPolicy::new(4)
    }

    /// Drain [`Self::pending_section_quantize`]: re-read each section's
    /// current native hot bytes from the substrate, quantize them
    /// per-layer with the section policy, and atomically replace
    /// the residence's hot with the quantized form.
    ///
    /// Called from the `SealAction::Turn` handler, on the main scheduler
    /// thread, *after* the turn record has been committed.  At that
    /// instant every in-flight reader of the native sections has
    /// finished — no other thread (including the persistence thread)
    /// holds a stale assumption about the section's contents — so the
    /// hot-replace is safe.  See the call site for the full ordering
    /// argument.
    fn quantize_pending_sections(
        &mut self,
        conversation: &Conversation,
        boundary_policy: &candle_nn::kv_cache::CompressionPolicy,
        member_policy: &candle_nn::kv_cache::CompressionPolicy,
    ) -> Result<(), ConversationError> {
        let drained: Vec<PendingSectionQuantize> =
            self.pending_section_quantize.drain(..).collect();
        self.quantize_section_batch(conversation, drained, boundary_policy, member_policy, false)
    }

    /// Quantize + offload ONLY the collection-member sections currently
    /// pending, leaving boundary sections pending for the turn-seal boundary.
    ///
    /// Collection members are prefix-transparent — nothing attends back over
    /// them during the build (priming uses the collections-empty projection),
    /// so it is safe to quantize them mid-build the moment their ingest wave
    /// has sealed.  This is what lets a large catalog (e.g. the tool list,
    /// sealed ×no_think) be shrunk and offloaded as the build runs instead of
    /// piling up native until the end.  Each member's residence is flagged
    /// `evict_when_cold` so the persistence thread frees its VRAM the instant
    /// the redo-log write lands.
    fn quantize_pending_collection_members(
        &mut self,
        conversation: &Conversation,
        boundary_policy: &candle_nn::kv_cache::CompressionPolicy,
        member_policy: &candle_nn::kv_cache::CompressionPolicy,
    ) -> Result<(), ConversationError> {
        let mut members: Vec<PendingSectionQuantize> = Vec::new();
        let mut keep: Vec<PendingSectionQuantize> = Vec::new();
        for p in self.pending_section_quantize.drain(..) {
            if p.in_collection {
                members.push(p);
            } else {
                keep.push(p);
            }
        }
        self.pending_section_quantize = keep;
        self.quantize_section_batch(conversation, members, boundary_policy, member_policy, true)
    }

    /// Quantize an already-drained pending list and atomically swap each
    /// residence's hot to its quantized form.  `mark_evict` additionally flags
    /// every section's residence for offload-on-persist (collection-member path).
    fn quantize_section_batch(
        &mut self,
        conversation: &Conversation,
        pending_list: Vec<PendingSectionQuantize>,
        boundary_policy: &candle_nn::kv_cache::CompressionPolicy,
        member_policy: &candle_nn::kv_cache::CompressionPolicy,
        mark_evict: bool,
    ) -> Result<(), ConversationError> {
        // Snapshot per-section native hot + residence + which policy
        // applies under a brief read lock; the heavy GPU work runs
        // unlocked.  Pending list is partitioned into (residence,
        // sealed, in_collection); the per-layer loop below groups by
        // in_collection so each policy gets its own batched launch.
        let pending: Vec<(SectionId, ResidenceIndex, Vec<SealedSequence>, bool)> = {
            let view = conversation.read();
            pending_list
                .into_iter()
                .filter_map(|p| {
                    let residence = view.section_residence(p.section_id)?;
                    let sealed = view.section_sealed_of(p.section_id)?;
                    Some((p.section_id, residence, (*sealed).clone(), p.in_collection))
                })
                .collect()
        };
        if pending.is_empty() {
            return Ok(());
        }

        let n_layers = self.session.num_layers();
        let backings = self.session.backings();
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => {
                return Err(ConversationError::Channel(
                    "section quantize requires a CUDA device".into(),
                ))
            }
        };

        // Split pending into two groups so each gets its own batched
        // per-layer launch under its own policy.  `indices_*` records
        // the position of each section in the original `pending` list
        // so we can stitch the per-group results back together in the
        // install loop without losing the original order.
        let mut indices_boundary: Vec<usize> = Vec::new();
        let mut indices_member: Vec<usize> = Vec::new();
        for (i, (_, _, _, in_collection)) in pending.iter().enumerate() {
            if *in_collection {
                indices_member.push(i);
            } else {
                indices_boundary.push(i);
            }
        }
        tracing::debug!(
            target: "candle_conversation::scheduler::section_quantize",
            boundary = indices_boundary.len(),
            member = indices_member.len(),
            "quantize drain: boundary vs collection-member split"
        );

        // Batch across sections per layer for each group.
        // `quantize_sealed_in_place` returns one output SealedSequence
        // per input SealedSequence in the same order, so the per-group
        // result slot is filled positionally before being scattered
        // back into the global `quantized_per_section` Vec.
        let mut quantized_per_section: Vec<Vec<SealedSequence>> = (0..pending.len())
            .map(|_| Vec::with_capacity(n_layers))
            .collect();
        let groups: [(&[usize], &candle_nn::kv_cache::CompressionPolicy); 2] = [
            (indices_boundary.as_slice(), boundary_policy),
            (indices_member.as_slice(), member_policy),
        ];
        for (group_indices, group_policy) in groups {
            if group_indices.is_empty() {
                continue;
            }
            for layer in 0..n_layers {
                let inputs: Vec<&SealedSequence> = group_indices
                    .iter()
                    .map(|&idx| {
                        let hot = &pending[idx].2;
                        debug_assert_eq!(
                            hot.len(),
                            n_layers,
                            "pending section hot must have one SealedSequence per layer",
                        );
                        &hot[layer]
                    })
                    .collect();
                let out = quantize_sealed_in_place(
                    &backings[layer],
                    &inputs,
                    group_policy,
                    &device,
                    &copy_stream,
                    &mut self.elevate_pinned_scratch,
                )
                .map_err(ConversationError::Model)?;
                for (slot_idx, qi) in out.into_iter().enumerate() {
                    let global_idx = group_indices[slot_idx];
                    quantized_per_section[global_idx].push(qi);
                }
            }
        }

        copy_stream
            .synchronize()
            .map_err(|e| ConversationError::Channel(format!("section quantize sync: {e}")))?;

        // Atomic swap: take the substrate write lock once, replace every
        // pending residence's hot with its quantized form, clear the
        // pending-quantize flag (so the persistence thread can now
        // gather and write the final bytes), then release.  Dropping
        // the (in-Vec) native SealedSequences after the lock is
        // released decrements the source chunks' refcounts to zero;
        // their arena slots return to the pool for reuse by the next
        // prefill.
        {
            let mut view = conversation.write();
            for ((_, residence, _native, _in_collection), q_per_layer) in
                pending.into_iter().zip(quantized_per_section.into_iter())
            {
                if q_per_layer.len() != n_layers {
                    tracing::warn!(
                        "quantize_pending_sections: layer-count mismatch ({} vs {}), skipping section {:?}",
                        q_per_layer.len(),
                        n_layers,
                        residence
                    );
                    continue;
                }
                view.replace_section_hot(residence, q_per_layer);
                view.clear_section_pending_quantize(residence);
                // Collection-member path: free this section's VRAM as soon as
                // the persistence thread lands its cold copy.
                if mark_evict {
                    view.mark_section_evict_when_cold(residence);
                }
            }
        }
        // Now that the substrate holds the final (quantized) form for
        // every drained section, wake the persistence thread so it
        // gathers them and appends to the redo log without waiting for
        // its 5 s tick.
        self.persist_trigger.fire();
        Ok(())
    }

    /// `turn_content`, when `seal_action == SealAction::Turn`, carries
    /// the role / text / token IDs the substrate pins on the new turn
    /// entry so the on-disk record can be reconstructed later without
    /// re-tokenising.  Ignored for `SealAction::Section` and `None`.
    fn perform_seal_and_write(
        &mut self,
        seal_slot: SequenceId,
        seal_block_from: usize,
        seal_action: &SealAction,
        turn_content: Option<TurnContent>,
    ) -> Result<Option<SealResult>, ConversationError> {
        // The substrate target (where a `SealAction::Turn` write
        // lands) is read from `slot_targets` rather than threaded
        // through the request — see [`Self::slot_targets`].
        let seal_target = self.slot_targets.get(&seal_slot).copied();
        let snapshot = self
            .session
            .snapshot_sequence(seal_slot.0)
            .map_err(ConversationError::Model)?;
        // Authoritative chunk total — `offset / CHUNK_SIZE` would
        // under-count when the slot's chunks include partials from
        // back-to-back section injection.  See
        // `BatchedInferenceSession::sequence_block_count`.
        let block_count = self.session.sequence_block_count(seal_slot.0).unwrap_or(0);
        if block_count <= seal_block_from {
            return Ok(None);
        }

        // GPU-resident sealed sequences.  No CPU round-trip: the
        // substrate stores `Arc<Vec<SealedSequence>>` with the same
        // GPU `ChunkGid`s the slot just prefilled into.  The
        // chunks stay live via the substrate's Arc refs even after
        // the ingest slot is freed (RAII on `ChunkGid`).  Errors
        // abort the seal entirely — a missing snapshot would leave
        // the substrate entry without KV data.
        let sealed_per_layer = match self.session.snapshot_sequence_per_layer(seal_slot.0) {
            Ok(sealed) => std::sync::Arc::new(sealed),
            Err(e) => {
                tracing::error!(
                    "snapshot_per_layer failed: seal_slot={} err={}",
                    seal_slot.0,
                    e,
                );
                return Ok(None);
            }
        };

        let chunk_size = self.chunk_size;
        let block_to = block_count.min(snapshot.chunks.len());
        let block_from = seal_block_from.min(block_to);
        let turn_token_count: usize = snapshot
            .chunks
            .get(block_from..block_to)
            .map(|s| s.iter().map(|c| c.token_count as usize).sum())
            .unwrap_or(0);

        // Capture the whole turn's wide per-token `sign(Q)` from R16 NOW — before
        // `record_turn` (below) detaches the sealed KV. All heads / all layers,
        // un-folded. Complete while the turn's KV is R16 (`kv_lossless`); a block
        // whose R16 is already gone (compressed) simply contributes nothing.
        // This is the LIVE turn seal (dialogue + code-read roundtrip turns) — count
        // it for the GUI's sealing phase, timing the dominant sig-gather cost.
        let t_sig = Instant::now();
        let wide_sigs = self.gather_wide_sigs(seal_slot, (block_from, block_to));
        self.wave_stats
            .add_seal(0, t_sig.elapsed().as_micros() as u64, 0);

        // KV-zero check: scan the PARENT slot at the seal range, right before the
        // chunks are persisted — i.e. exactly what gets written to the substrate.
        #[cfg(feature = "kv-zero-check")]
        {
            let layers: Vec<usize> = (0..self.session.num_layers()).collect();
            let seq_off = self.session.sequence_offset(seal_slot.0).unwrap_or(0);
            let layout = self.session.provenance_chunk_layout(seal_slot.0, seq_off);
            // The sealed range [block_from, block_to) is entirely this turn's own
            // content, so the boundary is the real position where block_from begins.
            let turn_start = layout.get(block_from).map(|l| l.2).unwrap_or(0);
            if let Ok(dump) = self.session.gather_r16_kv_provenance_layers(
                seal_slot.0,
                &layers,
                Some((block_from, block_to)),
            ) {
                kv_zero_check::scan_gathered(
                    "substrate-write-seal",
                    seal_slot.0,
                    &dump,
                    &layout,
                    self.session.n_kv_head(),
                    self.session.head_dim(),
                    turn_start,
                );
            }
        }

        // Apply the substrate write on the workspace conversation
        // bound to this slot.
        let conversation = self
            .slot_conversations
            .get(&seal_slot)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "perform_seal_and_write: no conversation registered for slot {seal_slot}"
                ))
            })?;
        // The substrate TurnIndex a `SealAction::Turn` records — surfaced on
        // the SealResult so per-turn persists key by the exact index instead
        // of racing `turn_count - 1` against the async summariser.
        let mut recorded_turn_index: Option<u32> = None;
        match seal_action {
            SealAction::Turn => {
                let target = seal_target.ok_or_else(|| {
                    ConversationError::Channel("SealAction::Turn missing seal_target".into())
                })?;
                let TurnContent {
                    role,
                    tags,
                    layout,
                    token_ids,
                } = turn_content.unwrap_or_default();
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to);
                // Snapshot what the resume path needs before the substrate
                // consumes `delta_gpu` / `token_ids` (Â§16.12 seal-time gather).
                let persist_token_ids: Vec<u32> = token_ids[..].to_vec();
                debug_assert_eq!(
                    persist_token_ids.len(),
                    turn_token_count,
                    "persisted token_ids must align 1:1 with the K/V chunk grid \
                     (off-by-one usually means an unforwarded token slipped through)"
                );

                // The turn is sealed as one indivisible K/V block; its
                // segment-vector `layout` carries the per-half text + spans the
                // sidebar and compressor read without re-tokenising.
                let write = TurnPartWrite {
                    layout,
                    token_ids,
                    token_count: turn_token_count,
                    tags,
                    block_start: block_from as u64,
                    block_end: block_to as u64,
                    sealed_gpu: Some(Arc::new(delta_gpu)),
                };
                let idx = conversation
                    .record_turn(target.timeline, role, write, |seqs| Ok(seqs.to_vec()))
                    .map_err(ConversationError::Model)?;
                recorded_turn_index = Some(idx.0);

                // Drain pending section quantizations.  Every section in
                // the queue was ingested earlier in this conversation
                // build with its native (prefill-output) K/V installed in
                // `substrate.section.hot`, and every reader up to this
                // point — priming projection's top-k selection, this
                // turn's prefill kernel attending back over the injected
                // sections — has consumed that native form.  This
                // turn-seal boundary is the first moment where no
                // in-flight operation depends on those native bytes any
                // more, which makes it the only safe place to swap the
                // residence's hot to its quantized form.  Synchronous,
                // on the main scheduler thread, batched across all
                // pending sections so the per-launch kernel overhead
                // amortises.  See the bisect history in
                // `tests/section_quantize_real_model.rs` for why earlier
                // / asynchronous attempts corrupt sysprompt K/V.
                if !self.pending_section_quantize.is_empty() {
                    if let Some(turn_policy) = self.session.compression_policy() {
                        let boundary_policy = Self::section_compression_policy_boundary();
                        let member_policy = Self::section_compression_policy_member(&turn_policy);
                        self.quantize_pending_sections(
                            &conversation,
                            &boundary_policy,
                            &member_policy,
                        )?;
                    } else {
                        self.pending_section_quantize.clear();
                    }
                }
                // Persist the wide per-token sign(Q) captured above (pre-detach) as the
                // turn's `WideQSig` record — continuous per-token shape, keyed to the
                // same turn stream.
                if !wide_sigs.is_empty() {
                    let stream_id = turn_stream_id(target.timeline.raw(), idx.0);
                    if let Err(e) =
                        conversation.persist_wide_q_sigs(stream_id, &encode_wide_sigs(&wide_sigs))
                    {
                        tracing::warn!("persist wide-Q sigs failed: {e}");
                    }
                }
                // Synchronously persist the turn's `Tokens` record —
                // tiny and load-bearing for substrate reconstruction.
                // The heavy `Chunks` records + the matching `Commit`
                // are deferred to the persistence thread; we fire its
                // trigger below.
                {
                    use turn_stream_id;
                    let stream_id = turn_stream_id(target.timeline.raw(), idx.0);
                    if let Err(e) = conversation.persist_tokens_only(stream_id, &persist_token_ids)
                    {
                        tracing::warn!("persist tokens failed: {e}");
                    }
                }
                // Wake the persistence thread so it drains the new
                // turn (hot→warm migrate, warm→cold redo-log write,
                // group fsync) without waiting for its 5 s tick.
                self.persist_trigger.fire();
                // Wake the summariser thread
                // (`docs/archived/infinite_conversations.md` §4 step ③) so the
                // freshly-pending Normal turn gets absorbed into the immutable
                // summary forest (`docs/immutable_summary_forest.md`) on its
                // next pass instead of waiting up to 250 ms for the periodic tick.
                self.summariser_trigger.fire();
                // Trim the slot's live-prefill capture cache down to
                // the working set this projection actually touched.
                // Without trimming, each turn would append at least
                // one new boundary-run entry (the trailing
                // `user_start_current`) and pin GPU arenas via its
                // `Arc<Vec<SealedSequence>>` values indefinitely.
                // Retaining the active keys keeps the boundary K/V
                // hot for next-turn reprojection while bounding the
                // cache at ~one projection's worth of entries — see
                // `SlotState::trim_post_turn`.
                if let Some(state) = self.slot_projection_state.get_mut(&seal_slot) {
                    state.trim_post_turn();
                }
            }
            SealAction::Section {
                section_id,
                tokens,
                address,
                debug_name,
                in_collection,
            } => {
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to);
                let stream_id = section_stream_id(*address);
                let policy_active = self.session.compression_policy().is_some();
                {
                    let mut view = conversation.write();
                    view.set_section_full(
                        *section_id,
                        stream_id,
                        turn_token_count,
                        Arc::new(delta_gpu),
                        |seqs| Ok(seqs.to_vec()),
                        Arc::clone(tokens),
                    )
                    .map_err(ConversationError::Model)?;
                    if policy_active {
                        if let Some(residence) = view.section_residence(*section_id) {
                            view.mark_section_pending_quantize(residence);
                        }
                    }
                }
                // Queue the section for quantization at the next safe
                // boundary (end of `PrimingProjection` for sections
                // ingested during a base-conv build, or `SealAction::
                // Turn` for sections ingested mid-conversation).  Until
                // the drain runs, `mark_section_pending_quantize` keeps
                // the persistence thread from writing the interim
                // native bytes to disk — otherwise the cold tier would
                // hold a form that diverges from the post-drain hot
                // tier, and a daemon restart would resume in an
                // inconsistent state.
                //
                // Sections **must** be left in their native form for the
                // in-flight build: priming projection reads
                // `substrate.section.hot` to pick top-k tools, the user
                // prefill kernel attends back over those injected K/V
                // to compute the prompt's own hidden states.  If we
                // quantize before that attention runs, the small Q4_KS
                // K error compounds across 48 layers into corrupted
                // prompt K/V — and everything downstream reads from a
                // poisoned cache.  See the bisect history in
                // `tests/section_quantize_real_model.rs`.
                if policy_active {
                    self.pending_section_quantize.push(PendingSectionQuantize {
                        section_id: *section_id,
                        in_collection: *in_collection,
                    });
                }
                // Declare the section stream in the redo log so the
                // manifest knows the (address, debug_name) before any
                // chunks land.  Mirrors record_turn's StreamDecl write.
                if let Err(e) = conversation.declare_section_stream(*address, debug_name) {
                    tracing::warn!("declare section stream failed: {e}");
                }
                // Persist the section's token ids, then fire the persistence
                // trigger so the chunks land on disk in the next pass.
                if let Err(e) = conversation.persist_tokens_only(stream_id, tokens) {
                    tracing::warn!("persist section tokens failed: {e}");
                }
                self.persist_trigger.fire();
            }
            SealAction::None => unreachable!("filtered above"),
            SealAction::CompressionPass { .. } => {
                unreachable!("compression passes complete in cleanup_finished, not here")
            }
            SealAction::CompressionTurn { .. } => {
                unreachable!(
                    "compression turns seal in promote_finished_prefills_to_decodes, not here"
                )
            }
            SealAction::TurnReprefill { .. } => {
                unreachable!(
                    "clean turn re-prefill seals via SealAction::Turn in complete_turn_reprefill"
                )
            }
            SealAction::ScopeIngest => {
                unreachable!(
                    "scope ingests snapshot in complete_scope_ingest and record in \
                     advance_scope_flush, not through perform_seal_and_write"
                )
            }
            SealAction::ScopeSummary => {
                unreachable!(
                    "scope summaries complete in cleanup_finished → complete_scope_summary_decoded, \
                     not through perform_seal_and_write"
                )
            }
            SealAction::ScopeReprefill => {
                unreachable!(
                    "scope re-prefills snapshot in complete_scope_summary_sealed and record in \
                     advance_scope_flush, not through perform_seal_and_write"
                )
            }
        }

        Ok(Some(SealResult {
            block_count,
            block_from,
            block_to,
            turn_token_count,
            chunk_size,
            turn_index: recorded_turn_index,
        }))
    }

    /// Rebuild the workspace substrate from the persistence redo log on
    /// daemon startup (Â§16.12 substrate reload).
    ///
    /// **Cold-only restart.** Every persisted turn stream is recovered in
    /// `(timeline, turn_index)` order; for each, tokens + wide-Q signatures
    /// are replayed into the in-RAM substrate and the turn is registered
    /// cold-marker (`hot/warm = None`, `cold = Some(...)` from the
    /// manifest). The KV bytes stay on disk until the runtime inject
    /// path demand-materialises them via [`elevate_to_hot`].
    pub fn reconstruct_substrate(
        &self,
        conversation: &Conversation,
        status: &SubstrateReloadStatus,
    ) {
        let n_layers = self.session.backings().len();
        if n_layers == 0 {
            // Nothing to replay (CPU/test backings) — unblock the waiter.
            status.finish();
            return;
        }
        let progress = |done: usize, total: usize| status.record(done, total);
        let result = conversation.reconstruct_from_log(n_layers, Some(&progress));
        // Always unblock the daemon's loading-screen waiter, success or not.
        status.finish();
        match result {
            Ok(0) => {}
            Ok(n) => tracing::info!("substrate reload: {n} turns restored from redo log"),
            Err(e) => tracing::error!("substrate reload failed: {e}"),
        }
    }

    // —— Raw KVQ extraction ————————————————————————————————————————————

    /// Extract raw K/V/Q float data for multiple layer indices.
    ///
    /// Calls the R16 dump path for each layer and returns results in the
    /// same order as `layer_indices`.  Each element is
    /// `(layer_idx, Vec<(block_idx, k_flat, v_flat, q_flat)>)`.
    fn handle_extract_raw_kvq(
        &self,
        seq_idx: usize,
        layer_indices: &[usize],
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>)>, ConversationError> {
        layer_indices
            .iter()
            .map(|&layer_idx| {
                let blocks = self
                    .session
                    .gather_r16_kv_for_provenance(seq_idx, layer_idx, block_range)
                    .map_err(ConversationError::Model)?;
                Ok((layer_idx, blocks))
            })
            .collect()
    }

    /// Capture the turn's per-token provenance signature from R16, one [`WideQSig`] per
    /// REAL token in token order. It walks each chunk's real-token window
    /// `[offset, offset+len)` from [`provenance_chunk_layout`] — the exact slots attention
    /// reads — so interior partial chunks (section / glue / reproject boundaries)
    /// contribute only their real tokens, never their zero padding; the result is 1:1
    /// aligned with the turn's `Tokens` record. Every token of each block is captured (no
    /// structural filter; filtering is a query-time concern). Each token's raw all-heads /
    /// all-layers `sign(Q)` is [`fold_provenance`]-folded to the compact locked signature
    /// (46,1,1 layer groups, heads separate, 1536 bits) before storing. Blocks whose R16
    /// backing is gone (compressed) contribute nothing — capture a turn while its KV is
    /// still R16 (e.g. `kv_lossless`, or at seal before the bg-quantizer runs).
    fn gather_wide_sigs(&self, seq_id: SequenceId, range: (usize, usize)) -> Vec<WideQSig> {
        let n_layers = self.session.num_layers();
        let n_kv_head = self.session.n_kv_head();
        let head_dim = self.session.head_dim();
        if n_layers == 0 || n_kv_head == 0 || head_dim == 0 || range.1 <= range.0 {
            return Vec::new();
        }
        let seq_off = self.session.sequence_offset(seq_id.0).unwrap_or(0);
        let layout = self.session.provenance_chunk_layout(seq_id.0, seq_off);

        // GPU fast path: read R16 Q + sign + bit-pack on device in ONE launch
        // across all layers, then D2H only the packed bits (a few KB) and XOR-fold
        // them here — bit-identical to the CPU path below. Falls through to the CPU
        // gather on non-CUDA / unsupported geometry / no R16 blocks. Split into
        // resolve / kernel / assemble so the wave breakdown shows where the
        // per-scope cost lives (see the `PROV_*_US` accumulators).
        use std::sync::atomic::Ordering::Relaxed;
        let sub = self.session.prov_sub_head_dim();
        if sub > 0 {
            let t_res = Instant::now();
            let resolved = self
                .session
                .resolve_provenance_q_ptrs(seq_id.0, Some(range))
                .ok()
                .flatten();
            PROV_RESOLVE_US.fetch_add(t_res.elapsed().as_micros() as u64, Relaxed);
            if let Some((all_ptrs, block_indices)) = resolved {
                let t_run = Instant::now();
                let packed = self
                    .session
                    .run_prov_sign_pack(&all_ptrs, sub)
                    .unwrap_or_default();
                PROV_KERNEL_US.fetch_add(t_run.elapsed().as_micros() as u64, Relaxed);
                if !packed.is_empty() {
                    let t_asm = Instant::now();
                    let ps = ProvSignPacked {
                        packed,
                        block_indices,
                        n_layers,
                        n_kv_head,
                        n_palette: candle_nn::kv_cache::N_PALETTE,
                        sub_head_dim: sub,
                    };
                    let sigs = assemble_folded_prov_sigs(&ps, &layout, head_dim);
                    PROV_ASSEMBLE_US.fetch_add(t_asm.elapsed().as_micros() as u64, Relaxed);
                    return sigs;
                }
            }
        }

        let layers: Vec<usize> = (0..n_layers).collect();
        let dump =
            match self
                .session
                .gather_r16_kv_provenance_layers(seq_id.0, &layers, Some(range))
            {
                Ok(d) if !d.is_empty() && !d[0].is_empty() => d,
                _ => return Vec::new(),
            };
        let chunk = candle_nn::CHUNK_SIZE;
        let band_len = n_layers * n_kv_head * head_dim;
        let n_blocks = dump[0].len();
        let mut out = Vec::with_capacity(n_blocks * chunk);
        for bi in 0..n_blocks {
            // Absolute chunk index of this gathered block → its real-token window.
            let block_idx = dump[0][bi].0;
            let Some(&(offset, len, _cum)) = layout.get(block_idx) else {
                continue; // no metadata for this chunk — skip (cannot place its tokens)
            };
            let offset = offset as usize;
            for j in 0..len as usize {
                let t = offset + j; // physical slot of real token j
                if t >= chunk {
                    break;
                }
                let mut band = Vec::with_capacity(band_len);
                let mut ok = true;
                for layer in &dump {
                    if bi >= layer.len() {
                        ok = false;
                        break;
                    }
                    let q_flat = &layer[bi].3;
                    for h in 0..n_kv_head {
                        band.extend_from_slice(&extract_q_vector_r16(
                            q_flat, t, h, n_kv_head, head_dim, chunk,
                        ));
                    }
                }
                if ok && band.len() == band_len {
                    // Fold the raw all-heads/all-layers sign(Q) into the compact locked
                    // provenance signature (46,1,1 layer groups, 32-bit stagger, heads
                    // separate) before storing — 16× smaller than the full wide-Q, and
                    // what the z-score late-fusion retrieval scores. See docs §23.
                    out.push(fold_provenance(&WideQSig::from_band(&band, head_dim)));
                }
            }
        }
        out
    }

    /// Carve a view sequence borrowing the requested ranges from
    /// `parent_id` (or every block if `visible_block_ranges` is
    /// empty).  Returns `(view_id, original_borrowed_block_count,
    /// borrowed_partial_usage)` — the partial usage is forwarded
    /// from the underlying `create_view_sequence` so callers can
    /// store it on the view's [`ViewState`] for zero-copy reproject.
    ///
    /// Called from the [`SchedulerRequest::SubmitTurn`] handler
    /// after [`apply_projection`] has populated the
    /// parent with the projected turns' sealed chunks.
    fn create_view(
        &mut self,
        parent_id: SequenceId,
        visible_block_ranges: &[BlockRange],
    ) -> Result<(SequenceId, BlockCount), ConversationError> {
        // Empty range list = sentinel "use all blocks of the parent".
        // Use `sequence_block_count` rather than `offset / CHUNK_SIZE`
        // — chunks with `usage < CHUNK_SIZE` from back-to-back section
        // injection are not visible to the divided value.  See
        // `BatchedInferenceSession::sequence_block_count`.
        let raw_ranges: Vec<(usize, usize)> = if visible_block_ranges.is_empty() {
            let total_blocks = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
            if total_blocks == 0 {
                vec![]
            } else {
                vec![(0, total_blocks)]
            }
        } else {
            visible_block_ranges.iter().map(|r| r.to_raw()).collect()
        };

        let vs = self
            .session
            .create_view_sequence(parent_id.0, &raw_ranges)
            .map_err(ConversationError::Model)?;
        let view_id = SequenceId(vs.view_idx);
        let borrowed = BlockCount(vs.borrowed_block_count);

        // Seed sampling state for the view (clone parent's state so
        // the DRY window survives the carve).
        if let Some(parent_state) = self.sampling_states.get(&parent_id).cloned() {
            self.sampling_states.insert(view_id, parent_state);
        } else {
            self.sampling_states.insert(
                view_id,
                SequenceSamplingState::new(
                    self.sampler.vocab_size(),
                    self.sampler.max_recent_len(),
                ),
            );
        }

        Ok((view_id, borrowed))
    }

    /// Run provenance scan + projection for the active view's policy, then
    /// rebuild the parent + view around the freshly-projected
    /// `(sections, turns)` selection, preserving the active turn's
    /// in-flight tokens **without any data copies or forward-pass
    /// work**.
    ///
    /// # The zero-copy rebuild
    ///
    /// The previous implementation tried to keep the same parent slot
    /// and just narrow the view's borrow window onto it.  That assumed
    /// the new projection's selection was already materialised on
    /// parent in a contiguous prefix.  provenance-driven `top_k` swaps broke
    /// that assumption: newly-picked sections only existed in the
    /// substrate, not in parent, so they got filtered out of the new
    /// borrow ranges, the view ended up borrowing a gappy subset,
    /// `finalize_view`'s truncate-to-borrowed step then dropped the
    /// in-parent sections the new selection didn't keep, and the
    /// next reproject's range request landed past parent's now-
    /// shrunken end.
    ///
    /// The rebuild is fully metadata-only.  The chunked backing's
    /// primitives (`inject_sealed_at_tail`, `create_view_sequence`,
    /// `extend_chunks`) all move chunks via `Arc<ChunkGid>` clones —
    /// no DMA, no kernel work, no forward pass.  Steps:
    ///
    /// 1. **Probe + scan + project**: probe live Q from the recent
    ///    decode window, run provenance against substrate sigs to refresh
    ///    section / turn scores, project the new
    ///    `(sections, turns)` selection.  Unchanged from the old
    ///    code.
    /// 2. **Capture the active turn's tail**: snapshot the view per
    ///    layer as `SealedSequence`s and slice off
    ///    `chunks[original_borrowed..]`.  Those chunks hold every
    ///    K/V byte computed since the view was carved (user prefill +
    ///    decoded-so-far).  The Q vectors and provenance signatures for the
    ///    decoded tokens are already captured by provenance — nothing
    ///    in the tail needs regenerating.
    /// 3. **Free the old view**: drops its `ChunkGid` refs.  The
    ///    captured tail Arc-refs keep the underlying chunks alive
    ///    across the free, so the GPU bytes survive untouched.
    /// 4. **Reset parent**: truncate to 0 chunks.  Substrate-pinned
    ///    section chunks survive via the substrate's own Arc refs;
    ///    parent just drops its metadata pointers.
    /// 5. **Re-run `apply_projection`**: Arc-clones the new
    ///    selection's sealed chunks onto parent and patches their
    ///    `block_range` in the substrate to the new layout.
    /// 6. **`inject_sealed_at_tail` the captured tail onto parent**:
    ///    Arc-clones again, so parent now holds `[new prefix] +
    ///    [active-turn tail]` with the tail's exact K/V bytes
    ///    preserved.
    /// 7. **Carve a fresh view borrowing all of parent**: the new
    ///    view's writer-owned chunk is the standard CoW copy of
    ///    parent's trailing partial chunk, ready for the next
    ///    decoded token to extend.
    /// 8. **Re-key per-view state** (`active_decodes`,
    ///    `sampling_states`, `turn_views`) onto the new view id.
    ///
    /// # RoPE is recomputed transparently
    ///
    /// K bytes are stored un-rotated.  Per-chunk `rope_base` is
    /// rederived every forward pass by
    /// [`SlotStateHost::from_sealed_chunks`] from the destination
    /// slot's cumulative chunk usage.  Moving chunks between slots
    /// or shifting the prefix size automatically yields the right
    /// rotated K when the kernel reads them.
    ///
    /// # The one real semantic compromise: stale K_raw
    ///
    /// The tail's stored K and V bytes were computed by the model
    /// against the *old* prefix.  Under the new prefix, a fresh
    /// forward pass would have produced slightly different K/V
    /// because the residual stream at every layer absorbed
    /// different attention output.  Re-prefilling would burn a
    /// forward pass to "fix" this; the zero-copy path doesn't.
    ///
    /// For typical reproject use (small provenance-driven catalog swaps
    /// of semantically-adjacent sections), the staleness is
    /// negligible.  Compounded over many reprojects fired in quick
    /// succession (e.g. every newline via `trigger_token_ids`),
    /// the drift can degrade generation quality — set the
    /// reproject cadence conservatively (`every_n_tokens` ~50+,
    /// or trigger on semantic-paragraph boundaries) to keep K_raw
    /// staleness within tolerance.
    ///
    /// No-op (returns the same view id) when:
    /// - The view has no [`ReprojectionPolicy`] attached, **or**
    /// - The R16 dump returns no sealed blocks (nothing to probe
    ///   with), **or**
    /// - The substrate corpus is empty.
    ///
    /// Errors propagate from the underlying session ops; the caller
    /// (decode loop) marks the view as finished on failure.

    /// Budget-aware pre-elevate eviction (replaces the unconditional
    /// `evict_from_hot`). Frees only enough of **this** conversation's
    /// least-recently-promoted hot KV to fit the incoming cold-load within the
    /// *accurate* VRAM budget — so on a big GPU the working set stays resident
    /// instead of being dropped (and reloaded from cold) every reproject.
    ///
    /// Order: estimate the incoming cold-load size → if it already fits the free
    /// budget, evict nothing → otherwise reclaim partial-arena free space
    /// (compaction, cheap unless fragmented past the 0.20 threshold) → only then
    /// evict the oldest non-selected turns, and only as many bytes as still
    /// needed. Per-conversation scoped: `evict_hot_to_free` walks only
    /// `conversation`'s own residence, never a parallel conversation's hot KV.
    fn evict_to_fit_incoming(
        &mut self,
        conversation: &Conversation,
        sections: &[SectionId],
        turns: &[TurnKey],
    ) -> crate::substrate::EvictionReport {
        // Incoming hot VRAM footprint of the working set that `elevate_to_hot`
        // is about to lift. BOTH warm→hot and cold→hot allocate fresh hot
        // arenas, so both must be budgeted for. Counting only cold left a
        // warm-resident selection — the common case once hot KV has been evicted
        // to the RAM tier — with zero reserved headroom, so its warm→hot batch
        // migrate hit the VRAM gate and failed; that drops the *whole* warm batch
        // (the migrate is all-or-nothing per layer), and `apply_projection` then
        // discards every selected turn as "no hot sealed K/V".
        let incoming_bytes: u64 = {
            let view = conversation.read();
            let plan = view.snapshot_promotion_state(sections, turns);
            let cold: u64 = plan
                .cold_to_hot
                .iter()
                .flat_map(|c| c.cold.iter())
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.record_len)
                .sum();
            let warm: u64 = plan
                .warm_to_hot
                .iter()
                .map(|w| sealed_total_bytes(&w.warm))
                .sum();
            cold + warm
        };
        if incoming_bytes == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        let device = self.session.device().clone();
        let avail = match candle_nn::kv_cache::vram_budget_available(&device) {
            // Budget unknown (non-CUDA / query failure) — don't evict on a guess.
            None => return crate::substrate::EvictionReport { count: 0, bytes: 0 },
            Some(a) => a as u64,
        };
        if incoming_bytes <= avail {
            // Ample VRAM — keep the whole working set hot.
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        // Tight: reclaim partial-arena free space first, then re-measure.
        let _ = self.session.compact();
        let avail = candle_nn::kv_cache::vram_budget_available(&device)
            .map(|a| a as u64)
            .unwrap_or(avail);
        let needed = incoming_bytes.saturating_sub(avail);
        if needed == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        conversation
            .write()
            .evict_hot_to_free(sections, turns, needed)
    }

    /// Promote the projected working set (`sections` + `turns`) into hot VRAM so
    /// [`Self::apply_projection`] can inject it: budget-aware evict of the
    /// non-incoming hot residences, then a batched select-promote (warm/cold →
    /// hot). Every path that hands sealed sections/turns to `apply_projection`
    /// MUST call this first — `apply_projection` only reads the *hot* residence
    /// and silently drops any selected unit that isn't hot, so a missing elevate
    /// step means the turn decodes/summarises without that content.
    ///
    /// A promote miss is not fatal (the decode proceeds without the dropped
    /// units), but it IS surfaced: when any selected item fails to reach hot the
    /// per-bucket report is logged at WARN with `whence` identifying the caller,
    /// so the drop has a visible cause rather than only the downstream
    /// `apply_projection: ... dropping it` symptom.
    fn elevate_projection_working_set(
        &mut self,
        conversation: &Conversation,
        sections: &[SectionId],
        turns: &[TurnKey],
        whence: &str,
    ) {
        // A `section_<n>` fallback name (see `Conversation::section_debug_name`) —
        // an unnamed content/probe section, not a schema-declared unit.
        fn is_generic_section_name(name: &str) -> bool {
            name.strip_prefix("section_")
                .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
                || name.starts_with("section#")
        }
        if sections.is_empty() && turns.is_empty() {
            return;
        }
        // PER-ITEM SKIP: only lift the units NOT already hot. A unit already hot
        // needs no work — and the machinery below (`snapshot_promotion_state`, run
        // twice, + a `backings` clone) costs ~100 ms/turn for a fixed system prompt
        // that never leaves hot. Cheap O(1) tier check per unit builds the actual
        // lift-list. `evict_to_fit_incoming` still gets the FULL working set as its
        // keep-list (it must protect the already-hot units from eviction), but only
        // budgets for the non-hot bytes. The promote tracker records only ACTUAL
        // lifts, so its leaderboard is a waste monitor — empty is good.
        let (secs_lift, turns_lift) = {
            let read = conversation.read();
            let mut sl: Vec<SectionId> = Vec::new();
            let mut tl: Vec<TurnKey> = Vec::new();
            for &sid in sections {
                if read.section_tier_state(sid).is_some_and(|t| t.hot) {
                    continue;
                }
                let name = self
                    .section_name_cache
                    .entry(sid)
                    .or_insert_with(|| {
                        // Generic auto-names (`section_<n>`) are the content /
                        // probe sections with no schema name — meaningless
                        // per-id on a churn leaderboard, so bucket them together
                        // and let the named always-hot units (system prompt,
                        // tools) stand out.
                        match read.section_debug_name(sid) {
                            Some(n) if !is_generic_section_name(&n) => n,
                            _ => "(content §)".to_string(),
                        }
                    })
                    .clone();
                phase_ring::record_promote(&name);
                sl.push(sid);
            }
            for &t in turns {
                if !read
                    .turn_tier_state(t.timeline, t.index)
                    .is_some_and(|ts| ts.hot)
                {
                    tl.push(t);
                }
            }
            if !tl.is_empty() {
                phase_ring::record_promote("(turns)");
            }
            (sl, tl)
        };
        // Everything already hot → nothing to evict or lift.
        if secs_lift.is_empty() && turns_lift.is_empty() {
            return;
        }
        let backings = self.session.backings().to_vec();
        let device = self.session.device().clone();
        // Scheduler is single-owner and blocks here before re-applying, so the
        // main inference stream is the right place for the scatter — no extra
        // sync, and subsequent prefill kernels serialise behind it for free.
        let main_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => panic!("scheduler: requires a CUDA device"),
        };
        // Keep-list = FULL working set (protect the already-hot units too).
        let evicted = self.evict_to_fit_incoming(conversation, sections, turns);
        if evicted.count > 0 {
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                whence, count = evicted.count, bytes = evicted.bytes,
                "select-evict complete"
            );
        }
        // Lift only the non-hot subset; already-hot units are left untouched.
        match elevate_to_hot(
            conversation,
            &backings,
            &device,
            &main_stream,
            &mut self.elevate_pinned_scratch,
            &mut self.cold_load_stager,
            &secs_lift,
            &turns_lift,
        ) {
            Ok(report) => {
                if report.missing > 0 || report.failed > 0 {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        whence,
                        already_hot = report.already_hot,
                        warm_to_hot = report.warm_to_hot,
                        cold_to_hot = report.cold_to_hot,
                        missing = report.missing,
                        failed = report.failed,
                        n_sections = sections.len(),
                        n_turns = turns.len(),
                        "select-promote: some selected units could not be lifted to hot; \
                         apply_projection will drop them from the assembled context"
                    );
                } else {
                    tracing::trace!(
                        target: "candle_conversation::persistence::tier",
                        whence,
                        already_hot = report.already_hot,
                        warm_to_hot = report.warm_to_hot,
                        cold_to_hot = report.cold_to_hot,
                        bytes_warm_to_hot = report.bytes_warm_to_hot,
                        bytes_cold_to_hot = report.bytes_cold_to_hot,
                        "select-promote complete"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "candle_conversation::persistence::tier",
                    whence,
                    "select-promote failed: {e} — apply_projection will drop non-hot selected units"
                );
            }
        }
    }

    /// Prepare phase of reprojection: BDP scan + projection + tier elevate +
    /// inject the sealed prefix + build the gap-fill descriptor. Returns the
    /// in-flight state for [`Self::reproject_view_complete`] (after the caller
    /// fires the batched gap-fill), or `None` if the view needs no reprojection.
    /// Removes the view's `DecodeState` into the in-flight on success.
    fn reproject_view_prepare(
        &mut self,
        view_id: SequenceId,
    ) -> Result<Option<ReprojectInFlight>, ConversationError> {
        let _t_total = PhaseTimer::new("reproject_total");
        // Clone the policy out — `swap_view_with_new_ranges` mutates
        // `active_decodes` so we cannot hold a borrow over it.
        let policy = match self
            .active_decodes
            .get(&view_id)
            .and_then(|s| s.reprojection.clone())
        {
            Some(p) => p,
            None => return Ok(None),
        };
        // Seed this reprojection's belief from the last one on the decode state
        // (empty on the first) — the RelLeak decay/reinforcement carries across
        // the turn. The new belief is written back in `reproject_view_complete`.
        let mut prior_belief = self
            .active_decodes
            .get(&view_id)
            .map(|s| s.belief.clone())
            .unwrap_or_default();
        // Tokens generated so far this turn — drives the early-decode grace window
        // (lowered band + carried-belief floor) so a correct pick whose decode-Q is
        // still accruing over the first ~64 tokens isn't evicted at token 1.
        let decode_pos = self
            .active_decodes
            .get(&view_id)
            .map(|s| s.generated_tokens.len());
        // This turn's first reprojection is the turn boundary: no projection has
        // run against the decoded content yet (`last_projection_end == 0`).
        let is_turn_boundary = self
            .active_decodes
            .get(&view_id)
            .map(|s| s.last_projection_end == 0)
            .unwrap_or(false);

        // 1. Compute the probe's CHUNK range — the turn's own content, in the
        //    same chunk coordinates the seal captures for the belief gallery.
        //
        //    The seal gathers `gather_wide_sigs(seal_slot, (seal_block_from,
        //    block_count))` where `seal_block_from == turn_start_parent_blocks`
        //    (see the turn-complete handler) and `block_count ==
        //    sequence_block_count`.  Those stored signatures are the belief
        //    gallery, so the live probe covers the same turn-content domain:
        //      • It never reaches before the turn boundary into the materialized
        //        system-prompt / tools prefix — a probe that swept the prefix
        //        would score the selected tool's own definition against its
        //        gallery and pin the selection to it with a turn-independent
        //        constant.
        //      • Chunk indices are partial-aware: `sequence_block_count` is the
        //        authoritative chunk total (the same call the seal uses), so
        //        partial chunks from glue/section boundaries can't shrink the
        //        probe below the turn's real tokens.
        //
        //    For turns longer than `max_probe_tokens`, the probe is the most
        //    recent chunks capped to that budget PLUS the turn's first chunks
        //    (the user query): the query is the strongest intent signal in the
        //    gallery domain, so it stays in every scan of the turn rather than
        //    sliding out of a purely trailing window.
        let view_state = self.turn_views.get(&view_id).copied().ok_or_else(|| {
            ConversationError::Channel(format!("reproject: missing view state {view_id}"))
        })?;
        let turn_start_chunk = view_state.turn_start_parent_blocks;
        let cur_chunks = self.session.sequence_block_count(view_id.0).unwrap_or(0);
        if cur_chunks <= turn_start_chunk {
            return Ok(None);
        }
        let max_probe_chunks = policy.max_probe_tokens.max(1).div_ceil(self.chunk_size);
        let tail_lo = turn_start_chunk.max(cur_chunks.saturating_sub(max_probe_chunks));

        // Wall-clock start of the whole reproject — drives `total_ms` so the log
        // reports the real end-to-end cost, not a sum of (partly overlapping)
        // phase fields.
        let t_repro = Instant::now();

        // 2. Gather the live wide-Q probe — folded per-token sign(Q): the query
        //    head (when the trailing window has slid past it) followed by the
        //    most recent turn chunks. A gather covering no real tokens means
        //    nothing to score.
        let t_probe = Instant::now();
        let mut probe = Vec::new();
        if tail_lo > turn_start_chunk {
            let head_hi = (turn_start_chunk + QUERY_HEAD_CHUNKS).min(tail_lo);
            probe.extend(self.gather_wide_sigs(view_id, (turn_start_chunk, head_hi)));
        }
        probe.extend(self.gather_wide_sigs(view_id, (tail_lo, cur_chunks)));
        if probe.is_empty() {
            return Ok(None);
        }
        let probe_ms = t_probe.elapsed().as_millis() as u64;
        record_phase(t_probe, "reproject_probe_extract");
        REPROJ_SCAN_US.fetch_add(
            t_probe.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // 4. Wide-Q belief scoring: scan the probe against each belief-driven
        //    collection's tag-scoped gallery of past turns→selected-section AND
        //    each belief-driven turn group's own turns (self-match), producing the
        //    per-section / per-turn scores the belief policy selects from in
        //    `project()`. No provenance scan, no persisted scores — projection-local.
        //    `group_candidates` carries each turn group's freshly-scored turns for
        //    the turn-boundary challenger below.
        let t_scan = Instant::now();
        let schema = policy.projection.schema();
        // observe = false: a live reprojection only READS the normalization hit
        // levels; learning happens once per turn at seal (last_turn_belief_scores).
        let (projection_scores, group_candidates) =
            policy
                .substrate
                .score_beliefs(schema, policy.target, &probe, false);

        let scan_ms = t_scan.elapsed().as_millis() as u64;
        record_phase(t_scan, "reproject_belief_scan");
        REPROJ_SCAN_US.fetch_add(
            t_scan.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Turn-boundary challenger: on this turn's FIRST reprojection, give each
        // top-N belief collection's strongest fresh signal a slot even when it's
        // below `min_score`, evicting the weakest carried incumbent only if the
        // selection is full. Lets a topic-changed query's new intent break in
        // without lowering the threshold (which would admit noise mid-turn); the
        // strong carried signals survive, and RelLeak decays the challenger back
        // out over the turn if its fresh score doesn't hold up.
        if is_turn_boundary {
            // Collections (tool catalog) live in the shared system prompt:
            // challenge on section fresh scores.
            {
                for item in &schema.system_prompt.items {
                    if let SystemPromptItem::Collection(coll) = item {
                        let budget_max = coll.policy.config.budget_max;
                        if budget_max < 3 {
                            continue;
                        }
                        let fresh: Vec<(String, f32)> = coll
                            .sections
                            .iter()
                            .map(|s| (s.name.clone(), projection_scores.section(s.id)))
                            .collect();
                        // Seed the challenger at the *windowed* selection bar so, in
                        // the opening grace window, it lands beside the floored
                        // carried picks (250) rather than dominating them at 1000.
                        let (wcfg, _) = coll.policy.config.windowed(decode_pos);
                        prior_belief.seat_turn_boundary_challenger(
                            GroupKey::Collection(coll.name.clone()),
                            &fresh,
                            budget_max,
                            wcfg.min_score,
                        );
                    }
                }
            }
            // Belief-driven turn groups span every layer (memory tiers, repo
            // map): look each candidate group up across the whole schema and
            // challenge on its per-turn fresh scores, keyed by turn index.
            for (gid, cands) in &group_candidates {
                let Some(group) = schema
                    .layers
                    .iter()
                    .flat_map(|l| l.groups.iter())
                    .find(|g| g.id == *gid)
                else {
                    continue;
                };
                let cfg = group.belief_config(cands.len());
                if cfg.budget_max < 3 {
                    continue;
                }
                let (wcfg, _) = cfg.windowed(decode_pos);
                let fresh: Vec<(String, f32)> = cands
                    .iter()
                    .map(|(idx, score)| (idx.0.to_string(), *score))
                    .collect();
                prior_belief.seat_turn_boundary_challenger(
                    GroupKey::TurnGroup(group.name.clone()),
                    &fresh,
                    cfg.budget_max,
                    wcfg.min_score,
                );
            }
        }

        // 5. Re-project against the freshly-scored substrate.
        //
        //    Produce the same `(projected_sections, projected_turns)`
        //    pair that `SchedulerRequest::SubmitTurn` builds: only
        //    sealed substrate entries make the cut.  These two lists
        //    drive the zero-copy rebuild in step 6.
        let t_project = Instant::now();
        let parent_id = view_state.parent_id;
        let mut turn_keys_for_elevate: Vec<TurnKey> = Vec::new();
        let (projected_sections, projected_segments, composition) = {
            let view = policy
                .substrate
                .read_for_scored(policy.target, &projection_scores);
            // Prefill mode: the section corpus is prefill-Q of tool
            // descriptions, so reprojection scores it with the calibrated
            // prefill profile (Max / semantic, no threshold gate).
            let projection = policy.projection.project_with_mode_and_sink(
                policy.target,
                &view,
                ProjectionMode::Prefill,
                &policy.selection,
                &prior_belief,
                decode_pos,
                &mut |_| {},
            );
            // Walk segments once, populating both the elevate side-lists
            // and the segment list `apply_projection` will diff against
            // the slot's previous projection.
            let mut sections: Vec<SectionId> = Vec::new();
            let mut segments: Vec<ProjectionSegment> = Vec::new();
            for seg in &projection.segments {
                match seg {
                    ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                        // Any-tier filter — same logic as the
                        // SubmitTurn path.  Cold-marker sections
                        // survive the filter; `elevate_to_hot`
                        // below lifts them before
                        // `apply_projection` injects them.
                        if view.section_tier_state(rs.id).is_some() {
                            sections.push(rs.id);
                            segments.push(seg.clone());
                        }
                    }
                    ProjectionSegment::Sealed(SealedKind::Turn(rt, _))
                    | ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => {
                        // The turn carries its conversation (stamped at projection),
                        // so read it directly — no group→timeline re-derivation,
                        // which is what once resolved the first-registered timeline
                        // and dropped every non-first conversation's turns here (the
                        // reproject `turns=0` history loss).  A timeline-less turn
                        // (mock/untracked) is skipped; a genuinely-missing one is
                        // surfaced loudly at inject, not silently filtered.
                        let Some(timeline) = rt.timeline else {
                            continue;
                        };
                        turn_keys_for_elevate.push(TurnKey::new(timeline, rt.index()));
                        segments.push(seg.clone());
                    }
                    ProjectionSegment::Generated { .. } => {
                        // Keep prefix structural / live-prefill runs in the
                        // segment chain, exactly as the SubmitTurn path does
                        // (see the matching arm where submit builds its
                        // `segments`). These structural tokens are part of the
                        // assembled prefix; dropping them here would make the
                        // reproject's rebuilt prefix diverge from submit's.
                        segments.push(seg.clone());
                    }
                    ProjectionSegment::NewUserMessage { .. } => {
                        // The active turn's user message is captured as the
                        // writer tail (snapshot/restore in the swap), so it must
                        // NOT be re-injected into the rebuilt prefix here.
                    }
                }
            }
            // Bucket the materialized segments by category (system / section
            // groups / turns) for the GUI. Timing is zeroed here and filled in
            // at `complete` from the per-sequence span anchor.
            //
            // Build the GUI composition from the FULL `projection.segments`, not
            // the tier-filtered `segments` used for apply: the filtered list
            // drops the active turn's user message (captured as the writer tail,
            // not re-injected) and any tier-less turn, which would leave the
            // panel's conversation section empty on a first turn. The decode-end
            // path (`projection_event`) buckets the full segments too, so this
            // keeps the live reproject view consistent with it.
            let mut composition = crate::projection::from_projection_with_origins(
                &projection.segments,
                &projection.selection_origins,
                policy.projection.schema(),
                &view,
                &projection.selection_scores,
                view.total_token_count(policy.target.timeline) as u32,
                0,
                0.0,
            );
            // The dialogue glue, sourced from the SAME `assemble_pieces` decision
            // the engine injects from — so the panel shows the real boundary
            // markers, never a reconstruction. `decode` keeps special tokens (the
            // markers ARE special tokens).
            composition.materialized = projection_assembler::materialize_conversation(
                &projection.segments,
                &self.boundary_markers,
                &projection.selection_origins,
                &view,
                policy.projection.schema(),
                |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
            );
            (sections, segments, composition)
        };
        let project_ms = t_project.elapsed().as_millis() as u64;
        record_phase(t_project, "reproject_project");
        REPROJ_LAYOUT_US.fetch_add(
            t_project.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Count of sealed turns this projection selected — the reproject cost
        // scales linearly with it (each adds one boundary-glue prefill).
        let n_turns_selected = projected_segments
            .iter()
            .filter(|s| matches!(s, ProjectionSegment::Sealed(SealedKind::Turn(..))))
            .count();

        // 6. Zero-copy rebuild.
        //
        //    The previous implementation tried to *narrow* the view's
        //    borrow window over a still-correct parent.  That assumed
        //    the new projection's selected sections / turns were
        //    already materialised on parent in a contiguous prefix.
        //    provenance-driven `top_k` swaps invalidated that assumption mid-
        //    decode (newly-picked sections only exist in the substrate,
        //    not in parent), `finalize_view`'s truncate-to-borrowed
        //    step then dropped the in-parent sections that the new
        //    selection didn't keep, and the next reproject borrowed
        //    block indices past parent's now-shrunken end.
        //
        //    The rebuild is fully metadata-only:
        //
        //      a. Snapshot the view per-layer to capture every
        //         `ChunkGid` it holds — the borrowed prefix *and*
        //         the writer-owned tail.  Slice off chunks beyond
        //         `original_borrowed`; those are the active turn's
        //         writes (user prefill + decoded-so-far).  Their K
        //         bytes are un-rotated, V bytes are
        //         position-independent, Q vectors and provenance signatures
        //         for the decoded tokens are already captured by
        //         provenance — nothing needs regenerating.
        //
        //      b. Free the old view slot.  The captured tail
        //         `ChunkGid`s keep the underlying chunks alive via
        //         their Arc refs even after the view drops.
        //
        //      c. Truncate parent to 0.  The old prefix's section
        //         chunks survive in the substrate (which still holds
        //         Arc refs to them); parent's metadata pointers are
        //         the only thing dropped.
        //
        //      d. Re-run `apply_projection` against parent with the
        //         new `(sections, turns)` list.  This Arc-clones
        //         each selected section / turn's sealed chunks onto
        //         parent and patches their `block_range` in the
        //         substrate so future lookups resolve to the new
        //         layout.
        //
        //      e. `inject_sealed_at_tail` the captured tail onto
        //         parent — Arc-clones again, no DMA.  Parent now
        //         holds `[new prefix] + [active-turn tail]`.
        //
        //      f. Carve a fresh view borrowing all of parent.  The
        //         view's writer-owned tail will be the CoW copy of
        //         parent's trailing partial chunk (create_view's
        //         normal partial-tail handling), ready for the next
        //         decoded token to extend.
        //
        //      g. Re-key every per-view map (`active_decodes`,
        //         `sampling_states`, `turn_views`) onto the new
        //         view id.
        //
        //    RoPE just works: each chunk's `rope_base` is rederived
        //    from cumulative usage at every forward pass by
        //    `SlotStateHost::from_sealed_chunks` — the same un-rotated
        //    K bytes yield the right rotated K at the new positions
        //    when the kernel reads them.  Zero bytes copied; the only
        //    side effect is the chunk pool's refcounts rebalancing.
        let t_swap = Instant::now();
        // Capture the active turn's tail from `turn_start_parent_blocks`,
        // not `original_borrowed`.
        //
        // The active turn's content begins at `turn_start_parent_blocks`
        // in the view's chunk list: for a freshly-submitted turn that
        // index is the COW chunk (`= original_borrowed` when the
        // parent had a partial tail); for a view that's already been
        // through one or more zero-copy reprojects, the active turn's
        // earliest content (the original COW, then later writes) lives
        // in the *borrowed* prefix of the view (Arc'd from parent),
        // *not* in the writer-owned tail.  Using `original_borrowed`
        // here would silently drop everything between
        // `turn_start_parent_blocks..original_borrowed` — i.e. the
        // user prefill and all decoding up to the previous reproject —
        // leaving only the latest CoW + most recent writer chunks.
        // That dropped content is exactly the "the model has lost the
        // user message" symptom that derails generation past the
        // first reproject.
        let tail_per_layer = {
            let snapshot = self
                .session
                .snapshot_sequence_per_layer(view_id.0)
                .map_err(ConversationError::Model)?;
            let tail_start = view_state.turn_start_parent_blocks;
            snapshot
                .into_iter()
                .map(|seq| {
                    let chunks: Vec<candle_nn::kv_cache::SealedChunk> =
                        if tail_start < seq.chunks.len() {
                            seq.chunks[tail_start..].to_vec()
                        } else {
                            Vec::new()
                        };
                    let token_count: usize = chunks.iter().map(|c| c.token_count as usize).sum();
                    candle_nn::kv_cache::SealedSequence {
                        chunks,
                        token_count,
                        chunk_size: seq.chunk_size,
                        location: seq.location,
                    }
                })
                .collect::<Vec<_>>()
        };
        // Pull DecodeState / sampling_state off the old view id before
        // freeing the slot, then re-bind to the new view id after the
        // rebuild.  Failing to find them is an internal invariant
        // violation — the caller (decode loop) only fires reproject on
        // an active view.
        let decode_state = self.active_decodes.remove(&view_id).ok_or_else(|| {
            ConversationError::Channel(format!(
                "reproject: missing decode state for view {view_id}"
            ))
        })?;
        let sampling_state = self.sampling_states.remove(&view_id);
        self.turn_views.remove(&view_id);
        // Diagnostic log: drop the freed view's per-slot token mirror.
        // The kernel will recycle this slot index for the new view we
        // carve below; without this drop, the new view's `slot_tokens`
        // would carry every decode-step input ever pushed under this
        // slot id across the turn's reprojects, producing duplicated
        // tokens in the turn-complete context dump that don't reflect
        // anything actually in the KV cache.
        self.slot_tokens.remove(&view_id);
        self.session
            .free_sequence(view_id.0)
            .map_err(ConversationError::Model)?;

        // Reset parent and re-project onto it.  `apply_projection`'s
        // populated-slot guard returns early when the slot is non-empty,
        // so truncate first.
        self.session
            .truncate_sequence_to_blocks(parent_id.0, 0)
            .map_err(ConversationError::Model)?;
        // Reset the slot_tokens diagnostic log so it stays in sync with
        // the post-rebuild slot contents.  apply_projection re-populates
        // it with the new prefix's tokens; the active turn tokens get
        // re-added below.  Gated on `context-dump` — the map is empty
        // when the feature is off, so the get_mut would no-op anyway.
        #[cfg(feature = "context-dump")]
        if let Some(toks) = self.slot_tokens.get_mut(&parent_id) {
            toks.clear();
        }
        // The pure swap work (tail snapshot + free + truncate) ends here —
        // captured disjoint from elevate/glue/complete so `swap_ms` reflects
        // only the chunk-pool rebalance, not the whole reproject.
        let swap_ms = t_swap.elapsed().as_millis() as u64;
        record_phase(t_swap, "reproject_swap");
        REPROJ_LAYOUT_US.fetch_add(
            t_swap.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Select-promote (warm/cold → hot) the new working set before re-applying,
        // turning the per-unit `ensure_*_hot` calls inside `apply_projection` into
        // hot-hit no-ops — see `elevate_projection_working_set`.
        let t_elevate = Instant::now();
        self.elevate_projection_working_set(
            &policy.substrate,
            &projected_sections,
            &turn_keys_for_elevate,
            "reproject",
        );
        let elevate_ms = t_elevate.elapsed().as_millis() as u64;

        // Time the per-slot apply work (segment build now + finish later),
        // EXCLUDING the shared gap-fill wave — that wait is reported as `glue_ms`.
        let t_build = Instant::now();
        // Build the gap-fill descriptor + inject the sealed prefix, but DON'T
        // fire the forward — the caller batches it across conversations.
        let plan = self.apply_projection_build(parent_id, &projected_segments)?;
        let build_ms = t_build.elapsed().as_millis() as u64;
        Ok(Some(ReprojectInFlight {
            view_id,
            parent_id,
            tail_per_layer,
            decode_state,
            sampling_state,
            sections_len: projected_sections.len(),
            segments_len: projected_segments.len(),
            n_turns_selected,
            composition,
            plan,
            build_ms,
            t_repro,
            swap_ms,
            probe_ms,
            scan_ms,
            project_ms,
            elevate_ms,
        }))
    }

    /// Complete phase of reprojection — runs after the batched gap-fill forward
    /// has fired + committed every prepared slot's glue. Finishes the projection
    /// (deferred user prefill + restore tail), appends the active turn's captured
    /// tail, carves a fresh view from the rebuilt parent, and re-keys the
    /// per-view maps onto the new view id.
    fn reproject_view_complete(
        &mut self,
        inflight: ReprojectInFlight,
        glue_ms: u64,
    ) -> Result<SequenceId, ConversationError> {
        let ReprojectInFlight {
            view_id,
            parent_id,
            tail_per_layer,
            mut decode_state,
            sampling_state,
            sections_len,
            segments_len,
            n_turns_selected,
            composition,
            plan,
            build_ms,
            t_repro,
            swap_ms,
            probe_ms,
            scan_ms,
            project_ms,
            elevate_ms,
        } = inflight;

        // Carry the belief forward: the next reprojection seeds from what this one
        // selected, so the RelLeak decay/reinforcement accumulates across the turn.
        // Migrates with the decode state into `new_view_id` below.
        decode_state.belief = PriorBelief::from_selection(&composition.selection);

        // A projection is a POINT, not a span: it is selected here (by the Q of
        // the tokens decoded so far) and governs everything forward until the next
        // projection supersedes it. So emit it the instant it occurs — `start_token`
        // is the generated-token position at which it was selected, `seconds` is the
        // wall-clock elapsed since decode start. The consumer reconstructs each
        // projection's effective interval `[pos_i, pos_{i+1})` and its throughput
        // from the *sequence* of events; there is nothing to close or flush.
        let repro_now = Instant::now();
        let repro_gen = decode_state.generated_tokens.len() as u32;
        {
            let elapsed = repro_now
                .duration_since(decode_state.decode_start)
                .as_secs_f64();
            let event = crate::projection::ProjectionEvent {
                start_token: repro_gen,
                seconds: elapsed,
                ..composition
            };
            let _ = decode_state.event_tx.send(TurnEvent::Projection(event));
        }
        // Advance the span cursor for the next reprojection / final seal. (The wide
        // `sign(Q)` history is captured continuously at decode time and persisted at
        // seal — it is not tied to reprojections; the event is just a marker.)
        decode_state.last_projection_end = repro_gen;

        let t_finish = Instant::now();
        self.apply_projection_finish(parent_id, plan)?;
        let new_prefix_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        // apply_ms = this slot's segment build + finish, NOT the shared gap-fill
        // wave (reported separately as glue_ms), so it no longer absorbs the
        // cross-slot wave wait.
        let apply_ms = build_ms + t_finish.elapsed().as_millis() as u64;

        // Under the read-only projection model there's no COW prefix
        // duplication to guard against: the view never copied bytes from
        // the parent's partial — it borrowed the partial read-only and
        // wrote new K/V into a fresh active chunk. So the captured tail
        // can be re-injected directly on top of the freshly-projected
        // parent with no truncate dance.

        // Append the captured tail to parent (metadata-only Arc clone).
        let tail_token_count: usize = tail_per_layer.first().map(|s| s.token_count).unwrap_or(0);
        let t_inject = Instant::now();
        if tail_token_count > 0 {
            self.session
                .inject_sealed_at_tail(parent_id.0, &tail_per_layer)
                .map_err(ConversationError::Model)?;
            // Mirror the tail's tokens back into the slot_tokens log
            // (we recorded them at original prefill / decode time
            // under the now-freed view id, but the parent's log was
            // also wiped above).  Reconstruct from DecodeState's
            // authoritative `prefill_tokens` + `generated_tokens`.
            Self::record_slot_tokens(
                &mut self.slot_tokens,
                parent_id,
                &decode_state.prefill_tokens,
            );
            Self::record_slot_tokens(
                &mut self.slot_tokens,
                parent_id,
                &decode_state.generated_tokens,
            );
        }

        let inject_ms = t_inject.elapsed().as_millis() as u64;

        // Carve a fresh view from the rebuilt parent.  Empty
        // `effective_ranges` means "borrow every chunk".
        let parent_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        let effective_ranges: Vec<BlockRange> = if parent_block_count == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_block_count)]
        };
        let t_view = Instant::now();
        let (new_view_id, new_borrowed) = self.create_view(parent_id, &effective_ranges)?;
        let view_ms = t_view.elapsed().as_millis() as u64;
        // Diagnostic log: the new view's slot index may have been
        // recycled from the freed old view's slot; drop any stale
        // entry under the new id before decode resumes.
        self.slot_tokens.remove(&new_view_id);
        self.turn_views.insert(
            new_view_id,
            ViewState {
                parent_id,
                original_borrowed: new_borrowed,
                // The active turn's content starts where the new
                // prefix ends — `cleanup_finished`'s seal extracts
                // `parent[turn_start_parent_blocks..end]` to produce
                // the substrate turn entry, so this anchor must
                // exclude the freshly-injected sections / turns and
                // include the tail.
                turn_start_parent_blocks: new_prefix_block_count,
            },
        );

        // Re-bind DecodeState and sampling state to the new view id.
        // `create_view` already seeded a fresh sampling state from
        // parent for the new view; replace it with the live one
        // carried across from the old view so DRY history and
        // per-turn counters survive.
        self.active_decodes.insert(new_view_id, decode_state);
        if let Some(state) = sampling_state {
            self.sampling_states.insert(new_view_id, state);
        }
        // True end-to-end wall-clock of the whole reproject (prepare → wave →
        // complete), so the phase fields below — which are individually disjoint
        // but separated by the shared glue wave — can be read against a real
        // total instead of summed by eye.
        let total_ms = t_repro.elapsed().as_millis() as u64;

        // Per-reproject timing breakdown — a diagnostic detail line, not a
        // heartbeat; debug so a continuous-reproject run doesn't flood info.
        tracing::debug!(
            target: "candle_conversation::scheduler::reproject",
            from_view = view_id.0,
            to_view = new_view_id.0,
            new_prefix_blocks = new_prefix_block_count,
            tail_tokens = tail_token_count,
            parent_blocks_after = parent_block_count,
            new_borrowed = new_borrowed.0,
            total_ms,
            probe_ms,
            scan_ms,
            project_ms,
            swap_ms,
            elevate_ms,
            glue_ms,
            apply_ms,
            inject_ms,
            view_ms,
            sections = sections_len,
            segments = segments_len,
            turns = n_turns_selected,
            "reproject (zero-copy rebuild)",
        );

        Ok(new_view_id)
    }
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
    use super::*;
    use candle::{DType, Tensor};

    #[test]
    fn carve_ms_redistributes_and_bounds() {
        // Carves in priority order, draining each bucket before the next.
        let (mut a, mut b) = (100u64, 50u64);
        let taken = carve_ms(120, &mut [&mut a, &mut b]);
        assert_eq!((taken, a, b), (120, 0, 30)); // 100 from a, 20 from b

        // Bounded by what the buckets hold — never carves more than available, so a
        // segment can't push the stack past the window.
        let (mut a, mut b) = (10u64, 5u64);
        let taken = carve_ms(1000, &mut [&mut a, &mut b]);
        assert_eq!((taken, a, b), (15, 0, 0));

        // Sum preservation: carved segment + remaining buckets == original total,
        // for any request. This is the invariant the phase decomposition relies on.
        for req in [0u64, 7, 33, 80, 200] {
            let orig = 80u64;
            let mut only = orig;
            let seg = carve_ms(req, &mut [&mut only]);
            assert_eq!(seg + only, orig, "req={req}");
        }
    }

    /// The GPU provenance path's on-CPU tail (`assemble_folded_prov_sigs`) must be
    /// bit-identical to the CPU path (`fold_provenance(WideQSig::from_band(..))`).
    /// This drives both from the same synthetic signs — the kernel just produces
    /// the packed `u32`s this test hand-packs — so it validates the whole host
    /// mapping (palette→word, warp indexing, layout) without a GPU.
    #[test]
    fn gpu_assembled_prov_sigs_match_cpu_fold() {
        use crate::provenance::WideQSig;
        const N_LAYERS: usize = 48;
        const N_KV_HEAD: usize = 4; // == PROV_HEADS_PER_LAYER
        const N_PALETTE: usize = 4;
        const SUB: usize = 32; // sub_head_dim
        const HEAD_DIM: usize = N_PALETTE * SUB; // 128
        let chunk = candle_nn::CHUNK_SIZE;
        let n_real = 5usize; // real tokens in the single block

        // Deterministic sign for (layer, head, token, global-dim): true = bit set.
        let sgn = |l: usize, h: usize, t: usize, d: usize| -> bool {
            let x = ((((l * 131 + h) * 131 + t) * 131 + d) as u64).wrapping_mul(2654435761);
            (x >> 13) & 1 == 0
        };

        // CPU reference: build the f32 band, from_band, fold — per token.
        let mut cpu: Vec<WideQSig> = Vec::new();
        for t in 0..n_real {
            let mut band = Vec::with_capacity(N_LAYERS * N_KV_HEAD * HEAD_DIM);
            for l in 0..N_LAYERS {
                for h in 0..N_KV_HEAD {
                    for d in 0..HEAD_DIM {
                        band.push(if sgn(l, h, t, d) { 1.0f32 } else { -1.0f32 });
                    }
                }
            }
            cpu.push(fold_provenance(&WideQSig::from_band(&band, HEAD_DIM)));
        }

        // Hand-pack the kernel's warp-major u32 output for the same signs.
        let n_blocks = 1usize;
        let n_warps = N_LAYERS * n_blocks * N_KV_HEAD * N_PALETTE;
        let mut packed = vec![0u32; n_warps * chunk];
        for l in 0..N_LAYERS {
            for h in 0..N_KV_HEAD {
                for p in 0..N_PALETTE {
                    let warp = ((l * n_blocks) * N_KV_HEAD + h) * N_PALETTE + p;
                    for t in 0..n_real {
                        let mut bits = 0u32;
                        for d in 0..SUB {
                            if sgn(l, h, t, p * SUB + d) {
                                bits |= 1u32 << d;
                            }
                        }
                        packed[warp * chunk + t] = bits;
                    }
                }
            }
        }
        let ps = ProvSignPacked {
            packed,
            block_indices: vec![0],
            n_layers: N_LAYERS,
            n_kv_head: N_KV_HEAD,
            n_palette: N_PALETTE,
            sub_head_dim: SUB,
        };
        // Block 0: real tokens at physical slots 0..n_real.
        let layout = vec![(0u16, n_real as u16, 0usize)];
        let gpu = assemble_folded_prov_sigs(&ps, &layout, HEAD_DIM);

        assert_eq!(gpu.len(), cpu.len(), "token count");
        for (t, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g.n_heads, c.n_heads, "token {t}: n_heads");
            assert_eq!(
                g.words, c.words,
                "token {t}: assembled words != from_band+fold"
            );
        }
    }

    #[test]
    fn pending_prefill_backlog_sums_remaining_tokens() {
        // queued (offset 0), partially-advanced, and fully-consumed units.
        let items = [
            (512, 0),   // queued: all 512 pending
            (300, 120), // in-flight: 180 remaining
            (64, 64),   // finished but not yet drained: 0 remaining
            (0, 0),     // empty unit
        ];
        assert_eq!(sum_pending_prefill_tokens(items), 512 + 180 + 0 + 0);
        // Empty backlog.
        assert_eq!(sum_pending_prefill_tokens(std::iter::empty()), 0);
        // Defensive: consumed > total (should never happen) saturates to 0, not
        // underflow.
        assert_eq!(sum_pending_prefill_tokens([(10, 25)]), 0);
    }

    #[test]
    fn fair_scope_picks_least_advanced_file() {
        // No candidates → nothing to pump.
        assert_eq!(pick_fair_scope(&[]), None);
        // A lone file claims the slot regardless of how many it's had.
        assert_eq!(pick_fair_scope(&[(7, 99)]), Some(7));
        // Among several files, the one pumped the fewest times wins — max-min
        // fairness so every file advances, not the first-submitted to completion.
        assert_eq!(pick_fair_scope(&[(1, 5), (2, 2), (3, 8)]), Some(2));
        // Tie on submitted count → the lowest raw id, for deterministic order.
        assert_eq!(pick_fair_scope(&[(9, 3), (4, 3), (6, 3)]), Some(4));
        // A single file with many parts (the only candidate) keeps winning every
        // pump, so it hits maximum parallelism.
        assert_eq!(pick_fair_scope(&[(42, 0)]), Some(42));
        assert_eq!(pick_fair_scope(&[(42, 23)]), Some(42));
    }

    #[test]
    fn admit_window_aimd_converges_and_recovers() {
        let floor = Scheduler::MIN_PREFILL_WIDTH; // 1
        let ceil = Scheduler::MAX_PREFILL_WIDTH; // 24

        // Multiplicative decrease halves toward the floor and stops there —
        // sustained pressure drives the window to 1, never 0 (always ≥1 in flight).
        let mut w = ceil;
        let descent: Vec<usize> = (0..6)
            .map(|_| {
                w = narrow_window(w, floor);
                w
            })
            .collect();
        assert_eq!(descent, vec![12, 6, 3, 1, 1, 1]);
        assert_eq!(narrow_window(floor, floor), floor);

        // Additive increase climbs by one and saturates at the ceiling — gradual
        // recovery, so a cleared episode doesn't snap straight back to full width.
        let mut w = floor;
        for _ in 0..(ceil * 2) {
            w = widen_window(w, ceil);
        }
        assert_eq!(w, ceil);
        assert_eq!(widen_window(ceil, ceil), ceil);
        assert_eq!(widen_window(5, ceil), 6);

        // A shrink is undone by exactly `n` grows for an n-halving descent — the
        // window is a plain saturating counter with no hidden state.
        assert_eq!(widen_window(narrow_window(2, floor), ceil), 2);
    }

    #[test]
    fn backlog_admit_action_hysteresis() {
        use BacklogAction::{Grow, Hold, Shrink};
        let ceil = Scheduler::MAX_PREFILL_WIDTH;
        let target = 8000;

        // Above target → shrink, regardless of window position.
        assert_eq!(
            backlog_admit_action(8001, target, ceil, ceil, false),
            Shrink
        );
        assert_eq!(backlog_admit_action(20000, target, 1, ceil, false), Shrink);

        // Deadband [target/2, target] → hold — no flapping as the backlog jitters.
        assert_eq!(backlog_admit_action(target, target, 4, ceil, false), Hold);
        assert_eq!(
            backlog_admit_action(target / 2, target, 4, ceil, false),
            Hold
        );
        assert_eq!(backlog_admit_action(5000, target, 4, ceil, false), Hold);

        // Below target/2 with headroom and no VRAM pressure → grow.
        assert_eq!(backlog_admit_action(3999, target, 4, ceil, false), Grow);
        assert_eq!(backlog_admit_action(0, target, 1, ceil, false), Grow);

        // Grow is suppressed at the ceiling (nothing to reopen)…
        assert_eq!(backlog_admit_action(0, target, ceil, ceil, false), Hold);
        // …and while VRAM is under pressure (the hard floor wins over reopening).
        assert_eq!(backlog_admit_action(0, target, 4, ceil, true), Hold);
        // But a high backlog still shrinks even under VRAM pressure.
        assert_eq!(backlog_admit_action(9000, target, 4, ceil, true), Shrink);
    }

    use candle_transformers::models::batched_inference::{
        BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
    };
    use std::str::FromStr;

    // —— Dummy model ——————————————————————————————————————————————————————————

    #[derive(Clone)]
    struct DummyModel {
        device: candle::Device,
        vocab_size: usize,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                device: candle::Device::Cpu,
                vocab_size: 64,
            }
        }

        fn dummy_logits(&self, n: usize) -> candle::Result<Vec<Tensor>> {
            (0..n)
                .map(|_| Tensor::zeros((1, self.vocab_size), DType::F32, &self.device))
                .collect()
        }
    }

    impl ManagedBatchedModel for DummyModel {
        fn num_layers(&self) -> usize {
            1
        }
        fn n_kv_head(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            16
        }
        fn device(&self) -> &candle::Device {
            &self.device
        }

        #[allow(clippy::too_many_arguments)]
        fn forward_wave(
            &self,
            _session: &mut BatchedInferenceSession,
            decode_seqs: &[usize],
            _decode_inputs: &[Tensor],
            prefill_seqs: &[usize],
            _prefill_inputs: &[Tensor],
            _glue_seqs: &[usize],
            _glue_inputs: &[Tensor],
            _layer_start: usize,
            layer_end: usize,
            _residual_in: Option<Tensor>,
        ) -> candle::Result<candle_transformers::models::batched_inference::WaveStep> {
            use candle_transformers::models::batched_inference::WaveStep;
            if layer_end >= self.num_layers() {
                Ok(WaveStep {
                    residual: None,
                    logits: Some(self.dummy_logits(decode_seqs.len() + prefill_seqs.len())?),
                })
            } else {
                Ok(WaveStep {
                    residual: Some(Tensor::zeros((1, 1, 1), DType::F32, &self.device)?),
                    logits: None,
                })
            }
        }

        fn prune(&self) -> candle::Result<()> {
            Ok(())
        }
    }

    // —— Helpers ——————————————————————————————————————————————————————————————

    /// Minimal CPU-backed session: 1 layer, 1 KV head, head_dim=16.
    fn make_test_session() -> BatchedInferenceSession {
        BatchedInferenceSession::new(
            1,  // num_layers
            1,  // n_kv_head
            16, // head_dim
            &candle::Device::Cpu,
            BatchedConfig::default()
                .with_initial_seq_len(32)
                .with_dtype(DType::F32),
        )
        .expect("failed to create test BatchedInferenceSession on CPU")
    }

    /// Trivial WordLevel tokenizer with an empty vocab.
    /// The scheduler dispatch tests do not call encode/decode, so any valid
    /// Tokenizer object suffices here.
    fn make_dummy_tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_str(
            r#"{"version":"1.0","truncation":null,"padding":null,
               "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
               "post_processor":null,"decoder":null,
               "model":{"type":"WordLevel","vocab":{},"unk_token":"[UNK]"}}"#,
        )
        .expect("failed to build dummy tokenizer")
    }

    /// A scheduler over the CPU test session and a `DummyModel`, plus its
    /// request sender — for tests that drive handler-level state (belief
    /// lifecycle) rather than forwards.
    fn make_test_scheduler() -> (Scheduler, crossbeam::channel::Sender<SchedulerRequest>) {
        let (tx, rx) = crossbeam::channel::bounded(16);
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let scheduler = Scheduler::new(
            rx,
            Box::new(DummyModel::new()),
            session,
            tokenizer,
            vec![0u32].into(), // eos_tokens
            64,                // vocab_size
            8,                 // max_recent_len
            false,             // show_special_tokens
            None,              // penalty_log_path
            DecodeHealthConfig::default(),
            512, // max_prefill_pass_tokens
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
        );
        (scheduler, tx)
    }

    // —— Tests ————————————————————————————————————————————————————————————————

    // —— view-creation tests —————————————————————————————————————————————————

    /// Explicit `visible_block_ranges` over a populated parent must create a valid view.
    #[test]
    fn create_view_with_explicit_ranges_creates_view() {
        let model = DummyModel::new();
        let (_tx, rx) = crossbeam::channel::bounded(16);
        let model_box = Box::new(model) as Box<dyn ManagedBatchedModel + Send>;
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let mut scheduler = Scheduler::new(
            rx,
            model_box,
            session,
            tokenizer,
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
        );

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);
        // Populate one full chunk on the parent so the view has a block to borrow.
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let input = candle::Tensor::new(&tokens[..], &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[parent_raw], tokens.len())
            .unwrap();
        let nl = scheduler.model.num_layers().max(1);
        scheduler
            .model
            .forward_wave(
                &mut scheduler.session,
                &[],
                &[],
                &[parent_raw],
                &[input],
                &[],
                &[],
                0,
                nl,
                None,
            )
            .unwrap();
        scheduler
            .session
            .advance_sequence(parent_raw, tokens.len())
            .unwrap();

        let result = scheduler.create_view(parent_id, &[BlockRange::new(0, 1)]);
        assert!(result.is_ok(), "view creation failed: {:?}", result.err());
    }

    /// Sentinel (empty) ranges + zero-block parent yields a
    /// zero-block view (borrowing zero blocks is valid).
    #[test]
    fn create_view_sentinel_with_zero_block_parent_yields_empty_view() {
        let model = DummyModel::new();
        let (_tx, rx) = crossbeam::channel::bounded(16);
        let model_box = Box::new(model) as Box<dyn ManagedBatchedModel + Send>;
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let mut scheduler = Scheduler::new(
            rx,
            model_box,
            session,
            tokenizer,
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
        );

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);

        let result = scheduler.create_view(parent_id, &[]);
        assert!(
            result.is_ok(),
            "sentinel + empty parent should succeed: {:?}",
            result.err()
        );
    }

    // —— View swap tests ——————————————————————————————————————————————————————
    //
    // The previous `swap_view_with_new_ranges` mid-decode view-swap path
    // has been replaced by the zero-copy reproject rebuild in
    // [`Scheduler::reproject_view`].  Its dedicated unit test
    // (`swap_view_re_keys_state_and_preserves_turn_start_parent_blocks`)
    // has been removed alongside the function.  The same re-keying
    // invariants — `active_decodes` / `sampling_states` / `turn_views`
    // migrate to the new view id, `DecodeState.event_tx` survives —
    // are now exercised end-to-end by the `zend` coherence integration
    // test, which fires reproject many times in a real decode loop.

    // —— Carried-belief slot lifecycle ————————————————————————————————————————
    //
    // The belief carried across a conversation's turns is keyed by its parent
    // slot; slot teardown and reuse must never let one occupant's belief seed
    // another's projections.

    /// A carried belief for a slot.
    fn seeded_belief() -> PriorBelief {
        let mut b = PriorBelief::default();
        b.set("tools", "calculator", 2500.0, true);
        b
    }

    #[test]
    fn reset_sequence_clears_the_slots_carried_belief() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let raw_id = scheduler.session.create_sequence().expect("create");
        let seq_id = SequenceId(raw_id);
        scheduler.carried_beliefs.insert(seq_id, seeded_belief());

        let (rtx, rrx) = crossbeam::channel::bounded(1);
        scheduler.handle_request(SchedulerRequest::ResetSequence {
            sequence_id: seq_id,
            response_tx: rtx,
        });
        rrx.recv().expect("reset response").expect("reset ok");

        assert!(
            !scheduler.carried_beliefs.contains_key(&seq_id),
            "a reset slot is reused for new content — the previous occupant's \
             belief must not survive the reset"
        );
    }

    #[test]
    fn free_sequence_clears_the_slots_carried_belief() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let raw_id = scheduler.session.create_sequence().expect("create");
        let seq_id = SequenceId(raw_id);
        scheduler.carried_beliefs.insert(seq_id, seeded_belief());

        scheduler.handle_request(SchedulerRequest::FreeSequence {
            sequence_id: seq_id,
        });

        assert!(
            !scheduler.carried_beliefs.contains_key(&seq_id),
            "a freed slot id is recycled by the allocator — the dead \
             conversation's belief must not seed the next occupant"
        );
    }
}
