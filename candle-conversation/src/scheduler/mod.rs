//! The scheduler: single thread that owns all GPU resources.
//!
//! Runs a continuous loop alternating between prefill and decode.
//! Phase 1 uses single-mode prefill (no small/large split).
mod decode;
mod prefill;
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
use crate::persistence::elevate::{elevate_to_hot, evict_from_hot};
use crate::persistence::thread::PersistenceTrigger;
use crate::projection::{
    Builder, Conversation, GroupId, ProjectionMode, ProjectionTarget, SectionId, TurnIndex, TurnKey,
};
#[cfg(feature = "sig-trace")]
use crate::provenance::TokenSignature;
use crate::sequence_handle::{BlockCount, BlockRange, SequenceId};
use crate::think_strip::strip_think_blocks;
use crate::token_buffer::TokenBuffer;
use crate::{ProvenanceFile, TurnStats};

use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, IndexOp, Tensor};
#[cfg(feature = "sig-trace")]
use candle_nn::kv_cache::SealedChunk;
use candle_nn::kv_cache::SealedSequence;
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, ProvenanceLayerIndices,
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// Scheduler request types (sent from caller threads)
// ────────────────────────────────────────────────────────────────────────────

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
    /// [`crate::projection::Conversation::mint_timeline`] so the
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

    /// Allocate a fresh GPU slot bound to an **existing** timeline.
    ///
    /// The substrate is expected to already have `timeline` registered
    /// (typically from a previous session restored via
    /// [`crate::projection::Conversation::open`]).  The handler looks
    /// up `(layer, group)` from the substrate's registry to construct
    /// the slot's `ProjectionTarget`, then proceeds like
    /// [`Self::NewSequence`].
    ///
    /// Returns `Err(...)` if the timeline is not registered.
    #[allow(dead_code)] // public scheduler API; used by Phase 2 resume callers
    ResumeSequence {
        conversation: Conversation,
        timeline: crate::projection::TimelineId,
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
        /// scheduler re-runs BDP + projection mid-decode and swaps the
        /// view's borrowed ranges in place; see [`ReprojectionPolicy`]
        /// for the full contract.  `None` skips re-projection entirely
        /// (used by single-shot paths like RULER eval and summarisation).
        reprojection: Option<ReprojectionPolicy>,
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
        response_tx: Sender<Result<SealResult, ConversationError>>,
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

    /// Shut down the scheduler.
    Shutdown,
}

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
/// 3. Runs BDP against the substrate corpus and writes fresh per-turn
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
    pub(crate) substrate: Conversation,
    pub(crate) provenance: Arc<crate::provenance::ProvenanceFile>,
    pub(crate) provenance_layer_indices: ProvenanceLayerIndices,
    /// Cadence trigger: re-project after every `every_n_tokens` decoded
    /// tokens.  `0` disables the cadence trigger (punctuation triggers
    /// can still fire).
    pub(crate) every_n_tokens: usize,
    /// Maximum tokens looking back from the current decode position to
    /// include in the BDP probe.  Caps the "thought window" — beyond
    /// this many tokens the prior reprojection already captured the
    /// older intent.  Must be `>= 1`.  Default: 64.
    pub(crate) max_probe_tokens: usize,
    /// Token IDs to drop from the BDP probe.  Includes formatting
    /// characters (whitespace, markdown punctuation) and chat-template
    /// scaffolding (role markers, think-block boundaries).  Without
    /// this filter every historical turn would inflate by roughly the
    /// same amount on shared-structure matches rather than ranking by
    /// shared content.
    pub(crate) probe_filter_token_ids: Arc<Vec<u32>>,
    /// Token IDs that, when sampled, fire an immediate reprojection in
    /// addition to the every-`every_n_tokens` cadence.  Use for
    /// paragraph/sentence boundaries (`\n`, `. `, etc) so attention
    /// re-orients at semantic transition points rather than waiting
    /// for the next fixed-cadence trigger.
    pub(crate) trigger_token_ids: Arc<Vec<u32>>,
    /// Span α for the BDP scanner.  Must match the α in `FIXED_FORMULA` so
    /// scores produced during reprojection are consistent with the scores the
    /// projection engine reads when computing group scores.
    pub(crate) span_alpha: f32,
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
}

// ────────────────────────────────────────────────────────────────────────────
// Internal state
// ────────────────────────────────────────────────────────────────────────────

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
    /// Provenance `SigEntry` records accumulated during decode by
    /// `extract_prov_after_step`.  Each entry covers one 32-token
    /// block extracted immediately after the forward pass that completed
    /// it, while the R16 backing is still intact.  Passed to
    /// `perform_seal_and_write` so seal-time extraction covers only the
    /// residual partial block (and any post-decode tail tokens), not the
    /// bulk that the bg_quantizer may have already compressed.
    prov_sig_entries: Vec<crate::provenance::SigEntry>,
    /// Trailing structural tokens written into the slot after decode
    /// finishes, before the seal.  Lifted to a forward pass in
    /// `cleanup_finished` so the turn's pinned KV closes its own
    /// brackets (e.g. ChatML's `\n` after `<|im_end|>`).
    post_decode_tokens: TokenBuffer,
    /// Full prefill token sequence (the user/system content the turn
    /// opened with).  Carried verbatim into the substrate's
    /// `TurnEntryData` at seal time so the on-disk record can be
    /// reconstructed without re-tokenising.
    prefill_tokens: TokenBuffer,
    /// Full prefill text in its ChatML-formatted form.  Combined with
    /// the decoded generation and the post-decode tail to form the
    /// substrate entry's `text` at seal time.
    prefill_text: String,
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
    Section {
        section_id: SectionId,
        tokens: Arc<Vec<u32>>,
    },
    /// Skip the seal entirely.  Used by raw RULER eval and
    /// summarisation paths that don't write to the substrate.
    None,
}

/// Content the substrate pins on a `SealAction::Turn` write — the
/// role, full ChatML text, and full token sequence assembled from
/// the prefill + decoded generation + post-decode tail.
///
/// Defaulting (`Role::Assistant`, empty text, empty tokens) is fine
/// for paths that lack a content trail (test fixtures, in-process
/// seals where the caller doesn't care about cross-process restore).
#[derive(Debug, Default, Clone)]
pub(crate) struct TurnContent {
    pub role: crate::turn::Role,
    pub text: String,
    pub token_ids: TokenBuffer,
}

/// A unit of prefill work queued for processing.
pub(super) struct PrefillWork {
    pub(super) sequence_id: SequenceId,
    pub(super) tokens: TokenBuffer,
    /// Text to emit as TurnEvent::Prefill before starting decode.
    pub(super) prefill_text: String,
    pub(super) event_tx: Sender<TurnEvent>,
    pub(super) max_decode_tokens: usize,
    pub(super) sampling: SamplingConfig,
    pub(super) submitted_at: Instant,
    /// Carried through prefill so it can be installed onto `DecodeState`
    /// when the prefill promotes to decode.  `None` when the caller
    /// disabled re-projection.
    pub(super) reprojection: Option<ReprojectionPolicy>,
    /// Carried through prefill so the post-Done substrate write fires
    /// on the right key.  The substrate target is looked up from
    /// [`Scheduler::slot_targets`] at seal time, not carried here.
    pub(super) seal_action: SealAction,
    /// Trailing structural tokens written into the slot after decode
    /// finishes, before the seal.  Carried through prefill so the
    /// post-decode forward pass in `cleanup_finished` can run.  Empty
    /// for paths that don't append a closing tail.
    pub(super) post_decode_tokens: TokenBuffer,
}

/// An in-flight prefill, partially advanced. Lives across scheduler
/// iterations until `offset` reaches `work.tokens.len()`, at which point it
/// is promoted to `active_decodes`.
pub(super) struct ActivePrefill {
    pub(super) work: PrefillWork,
    /// Tokens consumed so far.
    pub(super) offset: usize,
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
    pub(super) response_tx: Sender<Result<SealResult, ConversationError>>,
    pub(super) error: Option<ConversationError>,
}

// ────────────────────────────────────────────────────────────────────────────
// Scheduler
// ────────────────────────────────────────────────────────────────────────────

/// The scheduler: single thread that owns all GPU resources.
///
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
    max_prefill_chunk: usize,
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

    /// Per-slot count of provenance blocks already extracted and
    /// appended to the workspace [`ProvenanceFile`].  The seal step
    /// in `cleanup_finished` extracts only the new blocks
    /// `[sig_blocks_processed, block_count)` and updates this counter.
    /// Cleared on `FreeSequence`.
    slot_sig_blocks_processed: HashMap<SequenceId, usize>,

    /// **Diagnostic**: per-slot record of every token that has been
    /// committed to the slot's K/V — in the exact order it landed
    /// in the kernel's view.  Updated by every write path:
    ///
    ///   - `apply_projection` appends each injected section's
    ///     tokens (looked up from substrate).
    ///   - `run_prefill_with_shift` appends the prefilled tokens.
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

    /// Workspace-shared provenance signature file.  All seals across
    /// all slots append into this same mmap-backed file.
    provenance: Arc<crate::provenance::ProvenanceFile>,

    /// Static model properties captured at engine construction.
    model_core: ModelCoreProperties,

    /// Reusable pinned host scratch for the cold→hot HtoD leg used
    /// by `elevate_to_hot` (cuMemHostAlloc'd once, grown on demand).
    cold_load_stager: ColdLoadStager,

    /// Trigger handle for the substrate persistence thread. Fired
    /// after every turn-seal so the thread runs its hot→warm→cold
    /// drain promptly instead of waiting up to 5 s on its tick.
    persist_trigger: PersistenceTrigger,

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
}

/// Per-chunk debug trace of the BDP signatures emitted at seal time.
///
/// Each depth's per-token signatures are XOR-folded into a single chunk-level
/// fingerprint (same semantics as `TokenSignature::from_q_multi`), so each
/// chunk emits exactly three hex strings — one per depth.  The fold range is
/// limited to the chunk's actual valid token count; slots beyond that are
/// zero-initialised padding from the pre-allocated 32-slot signature buffer.
///
/// Enable with `RUST_LOG=candle_conversation::scheduler::signatures=trace`.
///
/// When the target is filtered out, the whole body is skipped via the
/// `tracing::enabled!` gate — no lookups, no folds, no formatting.  The
/// `#[inline]` hint lets the compiler hoist the gate to the call site.
#[cfg(feature = "sig-trace")]
fn trace_chunk_signatures(
    sequence_id: usize,
    chunk_idx: usize,
    slot_count: usize,
    chunks: &[SealedChunk],
    syn_sigs: &[TokenSignature],
    sem_sigs: &[TokenSignature],
    prag_sigs: &[TokenSignature],
) {
    if !tracing::enabled!(
        target: "candle_conversation::scheduler::signatures",
        tracing::Level::TRACE,
    ) {
        return;
    }
    let valid = chunks
        .get(chunk_idx)
        .map(|c| c.token_count as usize)
        .unwrap_or(slot_count)
        .min(slot_count);
    let fold = |sigs: &[TokenSignature]| -> u128 {
        sigs.iter()
            .take(valid)
            .fold(0u128, |acc, s| acc ^ s.as_u128())
    };
    tracing::trace!(
        target: "candle_conversation::scheduler::signatures",
        sequence_id,
        chunk = chunk_idx,
        token_count = valid,
        syn  = format!("{:032x}", fold(syn_sigs)),
        sem  = format!("{:032x}", fold(sem_sigs)),
        prag = format!("{:032x}", fold(prag_sigs)),
        "chunk sealed",
    );
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
        max_prefill_chunk: usize,
        provenance: Arc<ProvenanceFile>,
        model_core: ModelCoreProperties,
        persist_trigger: PersistenceTrigger,
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
            max_prefill_chunk,
            slot_conversations: HashMap::new(),
            slot_targets: HashMap::new(),
            slot_sig_blocks_processed: HashMap::new(),
            provenance,
            model_core,
            turn_views: HashMap::new(),
            pending_reprojections: Vec::new(),
            slot_tokens: HashMap::new(),
            cold_load_stager: ColdLoadStager::with_preallocation(PINNED_PREALLOC_BYTES),
            persist_trigger,
            elevate_pinned_scratch: preallocate_pinned_scratch(
                PINNED_PREALLOC_BYTES,
                "scheduler::elevate_pinned_scratch",
            ),
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

    // ── Submission handling ─────────────────────────────────────────────

    /// Drain all pending submissions. Returns `false` if shutdown requested.
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
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
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
                let seal_action = match (&projection_inputs, slot_target) {
                    (Some(_), Some(_)) => SealAction::Turn,
                    _ => SealAction::None,
                };
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
                let (projected_sections, projected_turns) = if let (Some(inputs), Some(target)) =
                    (projection_inputs.as_ref(), slot_target)
                {
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
                    // against the prefill-Q section corpus.
                    let projection =
                        inputs
                            .projection
                            .project_with_mode(target, &view, ProjectionMode::Prefill);
                    // The schema's projection is the single source
                    // of truth for system-side sections — emit
                    // exactly what the projection picked, in
                    // declaration order.  The legacy
                    // "always prepend `system_section_id`" path
                    // (a monolithic ChatML-wrapped duplicate of
                    // the schema fragments) has been removed; the
                    // schema items now compose the full system
                    // prompt on their own.
                    let mut sections: Vec<SectionId> =
                        Vec::with_capacity(projection.system_prompt.len());
                    for sec in &projection.system_prompt {
                        if view.section_sealed_of(sec.id).is_some() {
                            sections.push(sec.id);
                        }
                    }
                    // Translate each projected (group, idx) to its
                    // owning timeline before checking for sealed
                    // bytes. For the target's own group route via
                    // `target.timeline`; for other groups fall back
                    // to first-of-group (a Phase-1 single-timeline
                    // assumption that holds for cross-group
                    // references in our current schema).
                    let resolve_timeline = |g: GroupId| -> Option<crate::projection::TimelineId> {
                        if g == target.group {
                            Some(target.timeline)
                        } else {
                            view.timelines_for_group(g).next()
                        }
                    };
                    let turns: Vec<(GroupId, TurnIndex)> = projection
                        .turns
                        .iter()
                        .filter_map(|resolved| {
                            let g = resolved.group();
                            let t = resolved.index();
                            let timeline = resolve_timeline(g)?;
                            // Include any tracked turn, regardless of
                            // which tier holds its KV right now. Cold-
                            // marker turns (post-restart, before this
                            // submit's elevate_to_hot runs) must
                            // survive the filter — `elevate_to_hot`
                            // below is precisely what brings them into
                            // hot. Filtering on `turn_sealed_of` here
                            // would silently drop every resumed-from-
                            // disk turn from the projection because
                            // `hot` is None until elevation lands.
                            view.turn_tier_state(timeline, t)?;
                            turn_keys_for_elevate.push(TurnKey::new(timeline, t));
                            Some((g, t))
                        })
                        .collect();
                    (sections, turns)
                } else {
                    (Vec::new(), Vec::new())
                };

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
                if !projected_sections.is_empty() || !turn_keys_for_elevate.is_empty() {
                    if let Some(conversation) = self.slot_conversations.get(&parent_id).cloned() {
                        let backings = self.session.backings().to_vec();
                        let device = self.session.device().clone();
                        // Run on the device's main inference stream:
                        // the scheduler thread blocks on the result
                        // before `apply_projection` schedules prefill,
                        // so a dedicated copy stream would only force
                        // an extra sync. Same-stream serialisation is
                        // free.
                        let main_stream = match &device {
                            Device::Cuda(d) => d.cuda_stream(),
                            _ => panic!("scheduler: requires a CUDA device"),
                        };
                        // Free VRAM held by yesterday's working set
                        // (warm-backed hot residences NOT in the
                        // incoming projection) before the scatter
                        // runs. Items that are about to be re-elevated
                        // are excluded so we don't churn bytes
                        // through DMA for no reason.
                        let evicted = evict_from_hot(
                            &conversation,
                            &projected_sections,
                            &turn_keys_for_elevate,
                        );
                        if evicted.count > 0 {
                            tracing::debug!(
                                target: "candle_conversation::persistence::tier",
                                count = evicted.count,
                                bytes = evicted.bytes,
                                "select-evict complete (submit)"
                            );
                        }
                        match elevate_to_hot(
                            &conversation,
                            &backings,
                            &device,
                            &main_stream,
                            &mut self.elevate_pinned_scratch,
                            &mut self.cold_load_stager,
                            &projected_sections,
                            &turn_keys_for_elevate,
                        ) {
                            Ok(report) => {
                                tracing::debug!(
                                    target: "candle_conversation::persistence::tier",
                                    already_hot = report.already_hot,
                                    warm_to_hot = report.warm_to_hot,
                                    cold_to_hot = report.cold_to_hot,
                                    missing = report.missing,
                                    failed = report.failed,
                                    bytes_warm_to_hot = report.bytes_warm_to_hot,
                                    bytes_cold_to_hot = report.bytes_cold_to_hot,
                                    "select-promote complete (submit)"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "select-promote failed for slot {parent_id}: {e} — \
                                     apply_projection will fall back to per-unit ensure_*_hot"
                                );
                            }
                        }
                    }
                }

                if let Err(e) = self.apply_projection(
                    parent_id,
                    BlockCount(0),
                    &projected_sections,
                    &projected_turns,
                ) {
                    let _ = event_tx.send(TurnEvent::Error(e));
                    return true;
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
                tracing::info!(
                    target: "candle_conversation::scheduler::view_create",
                    parent = parent_id.0,
                    parent_block_count,
                    parent_offset = parent_offset_for_log,
                    offset_div_ceil = parent_offset_for_log.div_ceil(self.chunk_size),
                    "view borrow plan",
                );
                let effective_ranges: Vec<BlockRange> = if parent_block_count == 0 {
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
                    event_tx,
                    max_decode_tokens,
                    sampling,
                    submitted_at: Instant::now(),
                    reprojection,
                    seal_action,
                    post_decode_tokens,
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
                self.slot_targets.remove(&sequence_id);
                self.slot_tokens.remove(&sequence_id);
                self.slot_sig_blocks_processed.remove(&sequence_id);
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
                }
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::IngestSection {
                sequence_id,
                section_id,
                prefix_section_ids,
                tokens,
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
                                    },
                                    None,
                                    vec![],
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

            SchedulerRequest::PrimingProjection {
                sequence_id,
                section_ids,
                response_tx,
            } => {
                let result = if section_ids.is_empty() {
                    Ok(())
                } else {
                    self.apply_projection(sequence_id, BlockCount(0), &section_ids, &[])
                };
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

            SchedulerRequest::Shutdown => false,
        }
    }

    // ── Sequence creation ──────────────────────────────────────────

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

        tracing::debug!(
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
    ) {
        let skip = !self.show_special_tokens;
        let text = self.tokenizer.decode(&[token], skip).unwrap_or_default();
        let text = strip_think_blocks(&text);
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
                sequence: self.session.get_sequence_stats(seq_id.0),
            },
            // `finish_immediately` fires before any decode starts and
            // does no seal — the substrate write would have nothing
            // to capture (zero new blocks).
            seal: None,
        }));
    }

    // ── Projection helpers ───────────────────────────────────────────

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
        projected_sections: &[SectionId],
        projected_turns: &[(GroupId, TurnIndex)],
    ) -> Result<(), ConversationError> {
        // Stateless rebuild model: the slot's prefix is always the
        // current projection — `[projected_sections ++ projected_turns]`
        // — re-fetched from substrate residences on every call. Any
        // in-flight decode chunks ([writer_start_idx..end)) are
        // preserved across the rebuild so a mid-turn reprojection
        // doesn't drop the user's tokens. The substrate residence is
        // the single source of truth for the prefix's contents and
        // format; the slot's chunks are a transient view of it.
        let _ = system_block_count;

        // Snapshot any writer-owned tail (work-in-progress decode
        // chunks). At turn-submit boundaries this is empty; mid-turn
        // reprojection paths have a non-empty tail that must survive
        // the prefix rewrite. The snapshot holds RAII refs that keep
        // the underlying arena chunks alive across the truncate.
        let tail_per_layer: Vec<candle_nn::kv_cache::WriterTail> = {
            let mut out: Vec<candle_nn::kv_cache::WriterTail> =
                Vec::with_capacity(self.session.backings().len());
            for backing in self.session.backings() {
                out.push(
                    backing
                        .split_off_writer_tail(parent_id.0)
                        .map_err(ConversationError::Model)?,
                );
            }
            out
        };

        // Reset the slot to empty. Borrowed prefix chunks drop their
        // Arc refs (residence keeps them alive); the tail snapshot
        // above keeps its chunks alive independently.
        self.session
            .truncate_sequence_to_blocks(parent_id.0, 0)
            .map_err(ConversationError::Model)?;

        if projected_sections.is_empty() && projected_turns.is_empty() {
            // Nothing to inject — just restore the tail (if any) and return.
            for (backing, tail) in self.session.backings().iter().zip(tail_per_layer) {
                backing
                    .extend_writer_tail(parent_id.0, tail)
                    .map_err(ConversationError::Model)?;
            }
            return Ok(());
        }

        let n_layers = self.session.num_layers();

        // Identifies which substrate entry contributed each unit's
        // chunks to the concatenated injection — used to patch
        // `block_range` in the substrate after inject so
        // `reproject_view`'s lookups resolve to the new parent layout.
        enum InjectedUnit {
            Section(SectionId),
            Turn(GroupId, TurnIndex),
        }

        // Resolve the conversation handle bound to this slot — every
        // substrate read in this method goes through it so multi-
        // workspace setups read from the right store.
        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection: no conversation registered for slot {parent_id}"
                ))
            })?;

        // Gather per-unit per-layer sealed sequences in injection
        // order: sections first, then turns. Each `SealedSequence`
        // is a vector of windowed `SealedChunk`s (any may be partial
        // — sharing the underlying physical chunk with the writer is
        // safe because windows assert only the bytes they cover).
        //
        // The working set has already been brought into VRAM by
        // `elevate_to_hot` ahead of this call (§Step 1.5 in the
        // SubmitTurn handler); these lookups just pull the
        // residence's hot bytes out of the substrate. Any item the
        // elevator missed will surface as `None` here and gets
        // skipped — the elevate report's `missing` / `failed`
        // counters are where to look if turns silently drop.
        //
        // Pull the slot's target so we resolve `g == target.group` to
        // `target.timeline` rather than the first-registered timeline
        // (which is base_conv's empty timeline and would silently drop
        // every projected turn).
        let slot_target = self.slot_targets.get(&parent_id).copied();
        let mut section_list: Vec<SectionId> = Vec::with_capacity(projected_sections.len());
        let mut turn_list: Vec<(GroupId, TurnIndex, crate::projection::TimelineId)> =
            Vec::with_capacity(projected_turns.len());
        {
            let view = conversation.read();
            for &sid in projected_sections {
                if view.section_sealed_of(sid).is_some() {
                    section_list.push(sid);
                }
            }
            for &(g, t) in projected_turns {
                // Same target-aware lookup as the SubmitTurn handler
                // uses when assembling `projected_turns`.
                let timeline = match slot_target {
                    Some(tgt) if g == tgt.group => Some(tgt.timeline),
                    _ => view.timelines_for_group(g).next(),
                };
                if let Some(timeline) = timeline {
                    if view.turn_sealed_of(timeline, t).is_some() {
                        turn_list.push((g, t, timeline));
                    }
                }
            }
        }
        tracing::debug!(
            slot = parent_id.0,
            section_list = section_list.len(),
            turn_list = turn_list.len(),
            "apply_projection materialisation list",
        );
        let mut units: Vec<(InjectedUnit, Arc<Vec<SealedSequence>>)> =
            Vec::with_capacity(section_list.len() + turn_list.len());
        for sid in section_list {
            if let Some(s) = Self::hot_section_or_skip(&conversation, sid) {
                units.push((InjectedUnit::Section(sid), s));
            } else {
                tracing::warn!(
                    target: "candle_conversation::persistence::tier",
                    section = sid.raw(),
                    slot = parent_id.0,
                    "apply_projection: section not hot — elevate missed it; skipping borrow"
                );
            }
        }
        for (g, t, timeline) in turn_list {
            if let Some(s) = Self::hot_turn_or_skip(&conversation, timeline, t) {
                units.push((InjectedUnit::Turn(g, t), s));
            } else {
                tracing::warn!(
                    target: "candle_conversation::persistence::tier",
                    timeline = timeline.raw(),
                    turn = t.0,
                    slot = parent_id.0,
                    "apply_projection: turn not hot — elevate missed it; skipping borrow"
                );
            }
        }
        if units.is_empty() {
            return Ok(());
        }

        // Concatenate per-layer + remember each unit's block extent
        // so we can patch substrate `block_range` entries after
        // inject.
        let chunk_size = self.chunk_size;
        let mut per_layer_chunks: Vec<Vec<candle_nn::kv_cache::SealedChunk>> =
            (0..n_layers).map(|_| Vec::new()).collect();
        let mut per_layer_token_count: Vec<usize> = vec![0; n_layers];
        let mut unit_extents: Vec<(InjectedUnit, usize)> = Vec::with_capacity(units.len());
        for (unit, sealed) in units {
            if sealed.len() != n_layers {
                tracing::warn!(
                    "apply_projection: unit has {} layers, expected {}; skipping",
                    sealed.len(),
                    n_layers
                );
                continue;
            }
            // Layer 0's block count is the canonical extent (every
            // layer's chunk count matches for a well-formed sealed
            // entry).
            let block_count = sealed[0].chunks.len();
            for layer_idx in 0..n_layers {
                let layer_seq = &sealed[layer_idx];
                per_layer_chunks[layer_idx].extend(layer_seq.chunks.iter().cloned());
                per_layer_token_count[layer_idx] += layer_seq.token_count;
            }
            unit_extents.push((unit, block_count));
        }
        let gpu_per_layer: Vec<SealedSequence> = per_layer_chunks
            .into_iter()
            .zip(per_layer_token_count.into_iter())
            .map(|(chunks, tokens)| SealedSequence {
                chunks,
                token_count: tokens,
                chunk_size,
                location: candle_nn::kv_cache::ArenaLocation::Gpu,
            })
            .collect();

        let (start_block, _end_block) = self
            .session
            .inject_sealed_at_tail(parent_id.0, &gpu_per_layer)
            .map_err(ConversationError::Model)?;

        // Record each injected unit's `(start, end)` block range in
        // the substrate so `reproject_view`'s `block_range_of`
        // lookups resolve against the current parent layout.  Also
        // mirror each unit's token IDs into the slot's diagnostic
        // log so the turn-complete dump can reconstruct the exact
        // context the kernel saw.
        let mut injected_tokens: Vec<u32> = Vec::new();
        // Per-unit injection log (unit description + decoded text +
        // token count).  Built inside the same view borrow so we get
        // a consistent snapshot of what projection actually picked,
        // and emitted afterwards so the trace fires whether or not
        // any single decode succeeds.
        {
            let mut view = conversation.write();
            let mut cursor = start_block;
            for (unit, block_count) in unit_extents {
                let next = cursor + block_count;
                match unit {
                    InjectedUnit::Section(sid) => {
                        view.set_section_block_range(sid, cursor as u64, next as u64);
                        let toks = view.section_tokens_of(sid);
                        #[cfg(feature = "context-dump")]
                        if tracing::enabled!(
                            target: "candle_conversation::scheduler::projection_dump",
                            tracing::Level::INFO,
                        ) {
                            let decoded = self
                                .tokenizer
                                .decode(&toks, false)
                                .unwrap_or_else(|e| format!("<decode error: {e}>"));
                            tracing::info!(
                                target: "candle_conversation::scheduler::projection_dump",
                                slot = parent_id.0,
                                label = %format!("Section({sid:?})"),
                                token_count = toks.len(),
                                "{decoded}\n---"
                            );
                        }
                        injected_tokens.extend_from_slice(&toks);
                    }
                    InjectedUnit::Turn(g, t) => {
                        // Resolve the timeline before reborrowing `view`
                        // mutably for `set_block_range`.
                        let timeline = view.timelines_for_group(g).next();
                        if let Some(timeline) = timeline {
                            view.set_block_range(timeline, t, cursor as u64, next as u64);
                            let toks = view.token_ids_of(timeline, t).to_vec();
                            #[cfg(feature = "context-dump")]
                            if tracing::enabled!(
                                target: "candle_conversation::scheduler::projection_dump",
                                tracing::Level::INFO,
                            ) {
                                let decoded = self
                                    .tokenizer
                                    .decode(&toks, false)
                                    .unwrap_or_else(|e| format!("<decode error: {e}>"));
                                tracing::info!(
                                    target: "candle_conversation::scheduler::projection_dump",
                                    slot = parent_id.0,
                                    label = %format!("Turn(group={g:?}, index={t:?})"),
                                    token_count = toks.len(),
                                    "{decoded}\n---"
                                );
                            }
                            injected_tokens.extend_from_slice(&toks);
                        }
                    }
                }
                cursor = next;
            }
        }
        #[cfg(feature = "context-dump")]
        if tracing::enabled!(
            target: "candle_conversation::scheduler::projection_dump",
            tracing::Level::INFO,
        ) {
            tracing::info!(
                target: "candle_conversation::scheduler::projection_dump",
                slot = parent_id.0,
                total_tokens = injected_tokens.len(),
                "=== apply_projection injection order complete ==="
            );
        }
        Self::record_slot_tokens(&mut self.slot_tokens, parent_id, &injected_tokens);

        // Restore the writer-owned tail snapshot taken at the top of
        // this function. The prefix is now the new projection; the
        // tail picks up where it left off. RoPE is derived from
        // cumulative usage at decode-sync time, so the kernel will
        // re-rotate the tail's tokens against their new absolute
        // positions automatically — no byte copy or re-encoding
        // needed (see `SealedChunk`'s position-agnostic doc).
        for (backing, tail) in self.session.backings().iter().zip(tail_per_layer) {
            backing
                .extend_writer_tail(parent_id.0, tail)
                .map_err(ConversationError::Model)?;
        }

        Ok(())
    }

    /// (Dead code, retained for future use; rewritten to take a
    /// `Conversation` parameter so it can read from the right
    /// substrate under the multi-workspace scheduler model.)
    #[allow(dead_code)]
    pub(crate) fn assemble_projected_parent(
        &mut self,
        conversation: &Conversation,
        projected_turns: &[(GroupId, TurnIndex)],
    ) -> Result<SequenceId, ConversationError> {
        let n_layers = self.session.num_layers();

        // 1. Read substrate; collect per-turn per-layer sealed sequences.
        // Expects the caller to have run `elevate_to_hot` upstream so
        // every projected turn is hot-resident — any miss is logged
        // and skipped.
        let turn_keys: Vec<(GroupId, TurnIndex, crate::projection::TimelineId)> = {
            let view = conversation.read();
            projected_turns
                .iter()
                .filter_map(|&(g, t)| view.timelines_for_group(g).next().map(|tl| (g, t, tl)))
                .collect()
        };
        let mut per_turn_sealed: Vec<Arc<Vec<SealedSequence>>> =
            Vec::with_capacity(turn_keys.len());
        for (_g, t, timeline) in turn_keys {
            if let Some(s) = Self::hot_turn_or_skip(conversation, timeline, t) {
                per_turn_sealed.push(s);
            }
        }

        // 2. Concatenate per-layer.  Result: Vec<Arc<SealedSequence>>
        // of length `n_layers`, where each SealedSequence's chunks are
        // the concatenation of every projected turn's chunks for that
        // layer (in projection order).
        let mut per_layer_chunks: Vec<Vec<candle_nn::kv_cache::SealedChunk>> =
            (0..n_layers).map(|_| Vec::new()).collect();
        let mut per_layer_token_count: Vec<usize> = vec![0; n_layers];
        let chunk_size = self.chunk_size;
        for turn in &per_turn_sealed {
            if turn.len() != n_layers {
                tracing::warn!(
                    "assemble_projected_parent: turn has {} layers, expected {}; skipping",
                    turn.len(),
                    n_layers
                );
                continue;
            }
            for layer_idx in 0..n_layers {
                let layer_seq = &turn[layer_idx];
                per_layer_chunks[layer_idx].extend(layer_seq.chunks.iter().cloned());
                per_layer_token_count[layer_idx] += layer_seq.token_count;
            }
        }
        let gpu_per_layer: Vec<SealedSequence> = per_layer_chunks
            .into_iter()
            .zip(per_layer_token_count.into_iter())
            .map(|(chunks, tokens)| SealedSequence {
                chunks,
                token_count: tokens,
                chunk_size,
                location: candle_nn::kv_cache::ArenaLocation::Gpu,
            })
            .collect();

        // 3. Allocate the fresh GPU sequence; it'll receive the
        // projected chunks via inject_sealed_at_tail.
        let new_seq_idx = self
            .session
            .create_sequence()
            .map_err(ConversationError::Model)?;
        let new_seq_id = SequenceId(new_seq_idx);

        // 4. Inject as live ChunkWindows on the new sequence.
        self.session
            .inject_sealed_at_tail(new_seq_idx, &gpu_per_layer)
            .map_err(ConversationError::Model)?;

        Ok(new_seq_id)
    }

    // ── Cleanup ────────────────────────────────────────────────────────

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
                let decode_ms = state.decode_start.elapsed().as_secs_f64() * 1000.0;
                let total_ms = state.turn_start.elapsed().as_secs_f64() * 1000.0;
                let tokens_generated = state.generated_tokens.len();
                let tokens_per_second = if decode_ms > 0.0 {
                    tokens_generated as f64 / (decode_ms / 1000.0)
                } else {
                    0.0
                };

                let skip = !self.show_special_tokens;
                let text = self
                    .tokenizer
                    .decode(&state.generated_tokens, skip)
                    .unwrap_or_default();
                let text = strip_think_blocks(&text);
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
                    if let Err(e) =
                        self.run_prefill_with_shift(seal_slot, &state.post_decode_tokens[..], 0)
                    {
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

                // Seal-and-write step.  When `seal_action != None`, we
                // snapshot `seal_slot`, extract sigs for the new blocks,
                // append them to the workspace `ProvenanceFile`, and apply
                // the appropriate substrate write (turn append or section
                // pin).  The resulting `SealResult` rides along on the Done
                // event so the conversation-side post-actions (cold store,
                // BDP scan) can run without a second round trip.
                //
                // `pre_sigs` carries SigEntry records already extracted
                // during decode (per-step, while R16 was intact).  Move them
                // out before borrowing `state.seal_action` for the match.
                let pre_sigs = state.prov_sig_entries;
                let seal_result = match &state.seal_action {
                    SealAction::None => None,
                    action => {
                        // Assemble the full turn content (prefill + decoded
                        // generation + post-decode tail) so the substrate
                        // entry holds the verbatim ChatML text and token
                        // sequence that the seal pinned into the slot.
                        let turn_content = if matches!(action, SealAction::Turn) {
                            let post_decode_text = if state.post_decode_tokens.is_empty() {
                                String::new()
                            } else {
                                self.tokenizer
                                    .decode(&state.post_decode_tokens, skip)
                                    .unwrap_or_default()
                            };
                            let mut full_text = String::with_capacity(
                                state.prefill_text.len() + text.len() + post_decode_text.len(),
                            );
                            full_text.push_str(&state.prefill_text);
                            full_text.push_str(&text);
                            full_text.push_str(&post_decode_text);

                            let mut full_tokens: Vec<u32> = Vec::with_capacity(
                                state.prefill_tokens.len()
                                    + state.generated_tokens.len()
                                    + state.post_decode_tokens.len(),
                            );
                            full_tokens.extend_from_slice(&state.prefill_tokens);
                            full_tokens.extend_from_slice(&state.generated_tokens);
                            full_tokens.extend_from_slice(&state.post_decode_tokens);

                            Some(TurnContent {
                                role: crate::turn::Role::Assistant,
                                text: full_text,
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
                            pre_sigs,
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
                        sequence: sequence_stats,
                    },
                    seal: seal_result,
                }));
            }
        }
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
                            .zip(per_layer_token_count.into_iter())
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
            &SealAction::Section { section_id, tokens },
            None,
            vec![],
        )?;
        seal.ok_or_else(|| {
            ConversationError::Channel(
                "ingest_section: seal returned None (slot had no content?)".into(),
            )
        })
    }

    /// Snapshot `seal_slot`, extract sigs for the new blocks
    /// `[seal_block_from, block_count)`, append them to the workspace
    /// `ProvenanceFile`, and apply the substrate write described by
    /// `seal_action`.  Returns the [`SealResult`] payload, or
    /// `Ok(None)` when there are no new blocks to seal.
    ///
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
        pre_sigs: Vec<crate::provenance::SigEntry>,
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

        // Per-slot sig-blocks-processed counter — extracted only for
        // new blocks since the last seal.
        let prev_processed = self
            .slot_sig_blocks_processed
            .get(&seal_slot)
            .copied()
            .unwrap_or(0);
        let sig_from = prev_processed.max(seal_block_from);
        let sig_range = if block_count > sig_from {
            Some((sig_from, block_count))
        } else {
            None
        };

        // Extract sigs at all three depths (MH_XOR_QQ_l0xl4: dual-layer, all heads).
        // `pre_sigs` carries entries already extracted during decode (while R16
        // was intact); seal-time extraction covers only the residual range that
        // wasn't reached before the bg_quantizer compressed earlier blocks.
        let ProvenanceLayerIndices {
            syn_l0,
            syn_l4,
            sem_l0,
            sem_l4,
            prag_l0,
            prag_l4,
        } = self.model_core.provenance_layer_indices;
        let mut new_sig_entries: Vec<crate::provenance::SigEntry> = pre_sigs;
        let mut new_processed = prev_processed;

        // Actual fill count of the last block — may be < CHUNK_SIZE for the
        // partial tail.  Used below to strip zero-padded garbage signatures
        // that the gather kernel reads from uninitialized arena slots.
        let tail_tokens = snapshot
            .chunks
            .get(block_count.saturating_sub(1))
            .map(|c| c.token_count as usize)
            .filter(|&t| t > 0)
            .unwrap_or(candle_nn::CHUNK_SIZE);

        if let Some(range) = sig_range {
            let syn =
                self.handle_extract_mh_dual_signatures(seal_slot.0, syn_l0, syn_l4, Some(range));
            let sem =
                self.handle_extract_mh_dual_signatures(seal_slot.0, sem_l0, sem_l4, Some(range));
            let prag =
                self.handle_extract_mh_dual_signatures(seal_slot.0, prag_l0, prag_l4, Some(range));
            if let (Ok(syn_b), Ok(sem_b), Ok(prag_b)) = (syn, sem, prag) {
                let total = syn_b.len().min(sem_b.len()).min(prag_b.len());
                for j in 0..total {
                    let raw_n = syn_b[j]
                        .sigs
                        .len()
                        .min(sem_b[j].sigs.len())
                        .min(prag_b[j].sigs.len());
                    // Cap the final block to its actual fill to avoid appending
                    // zero-padded signatures from uninitialized arena slots.
                    let n = if j + 1 == total {
                        raw_n.min(tail_tokens)
                    } else {
                        raw_n
                    };
                    match self.provenance.append(
                        &syn_b[j].sigs[..n],
                        &sem_b[j].sigs[..n],
                        &prag_b[j].sigs[..n],
                    ) {
                        Ok(entry) => new_sig_entries.push(entry),
                        Err(e) => tracing::warn!(
                            "provenance append failed for block {}: {e}",
                            sig_from + j,
                        ),
                    }
                }
                new_processed = sig_from + total;
            }
        }
        self.slot_sig_blocks_processed
            .insert(seal_slot, new_processed);

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
        match seal_action {
            SealAction::Turn => {
                let target = seal_target.ok_or_else(|| {
                    ConversationError::Channel("SealAction::Turn missing seal_target".into())
                })?;
                let TurnContent {
                    role,
                    text,
                    token_ids,
                } = turn_content.unwrap_or_default();
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to);
                // Snapshot what the resume path needs before the substrate
                // consumes `delta_gpu` / `token_ids` (§16.12 seal-time gather).
                let persist_token_ids: Vec<u32> = token_ids[..].to_vec();

                let idx = conversation
                    .record_turn(
                        target.timeline,
                        role,
                        text,
                        token_ids,
                        turn_token_count,
                        block_from as u64,
                        block_to as u64,
                        Arc::new(delta_gpu),
                        |seqs| Ok(seqs.to_vec()),
                    )
                    .map_err(ConversationError::Model)?;
                if !new_sig_entries.is_empty() {
                    {
                        let mut view = conversation.write();
                        view.set_sig_entries(target.timeline, idx, new_sig_entries.clone());
                    }
                    // Persist the BDP provenance signatures to the redo log
                    // so attentional retrieval survives a restart. SigEntry
                    // only references the (ephemeral) provenance file, so
                    // read each entry's bytes and embed them in a
                    // `Signatures` record.
                    let stream_id = crate::persistence::content_hash::turn_stream_id(
                        target.timeline.raw(),
                        idx.0,
                    );
                    let mut sig_bytes = Vec::with_capacity(new_sig_entries.len());
                    for e in &new_sig_entries {
                        match self.provenance.read_entry(*e) {
                            Ok((syn, sem, prag)) => {
                                let mut bytes = Vec::with_capacity(e.byte_len());
                                for s in syn.iter().chain(sem.iter()).chain(prag.iter()) {
                                    bytes.extend_from_slice(s.as_bytes());
                                }
                                sig_bytes.push((e.token_count, bytes));
                            }
                            Err(err) => {
                                tracing::warn!("read provenance entry for persistence: {err}")
                            }
                        }
                    }
                    let payload = crate::persistence::resume::encode_signatures(&sig_bytes);
                    if let Err(e) = conversation.persist_signatures(stream_id, &payload) {
                        tracing::warn!("persist signatures failed: {e}");
                    }
                }
                // Synchronously persist the turn's `Tokens` record —
                // tiny and load-bearing for substrate reconstruction.
                // The heavy `Chunks` records + the matching `Commit`
                // are deferred to the persistence thread; we fire its
                // trigger below.
                {
                    use crate::persistence::content_hash::turn_stream_id;
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
            }
            SealAction::Section { section_id, tokens } => {
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to);
                let mut view = conversation.write();
                view.set_section_full(
                    *section_id,
                    turn_token_count,
                    new_sig_entries.clone(),
                    Arc::new(delta_gpu),
                    |seqs| Ok(seqs.to_vec()),
                    Arc::clone(tokens),
                )
                .map_err(ConversationError::Model)?;
            }
            SealAction::None => unreachable!("filtered above"),
        }

        Ok(Some(SealResult {
            block_count,
            block_from,
            block_to,
            turn_token_count,
            new_sig_entries,
            chunk_size,
            sig_blocks_processed: new_processed,
        }))
    }

    /// Substrate lookup: a turn's hot-resident sealed sequences, or
    /// `None` if not hot. Post-elevate this is the working contract —
    /// `elevate_to_hot` ahead of `apply_projection` populates every
    /// projected turn's hot tier, so a `None` here means the
    /// elevation orchestrator missed it (logged as `missing` or
    /// `failed` in the [`ElevationReport`]) and the caller skips
    /// the inject borrow rather than trying to recover.
    fn hot_turn_or_skip(
        conversation: &Conversation,
        timeline: crate::projection::TimelineId,
        index: crate::projection::TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        conversation.read().turn_sealed_of(timeline, index)
    }

    /// Section-side counterpart. Sections are pinned at conversation
    /// setup and the elevate path keeps them hot; a `None` return
    /// means either the section isn't registered or elevation hasn't
    /// run yet for this slot.
    fn hot_section_or_skip(
        conversation: &Conversation,
        section: crate::projection::SectionId,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        conversation.read().section_sealed_of(section)
    }

    /// Rebuild the workspace substrate from the persistence redo log on
    /// daemon startup (§16.12 substrate reload).
    ///
    /// **Cold-only restart.** Every persisted turn stream is recovered in
    /// `(timeline, turn_index)` order; for each, tokens + BDP signatures
    /// are replayed into the in-RAM substrate and the turn is registered
    /// cold-marker (`hot/warm = None`, `cold = Some(...)` from the
    /// manifest). The KV bytes stay on disk until the runtime inject
    /// path demand-materialises them via [`elevate_to_hot`].
    pub fn reconstruct_substrate(&self, conversation: &Conversation) {
        use crate::provenance::{SigEntry, TokenSignature};

        let n_layers = self.session.backings().len();
        if n_layers == 0 {
            return;
        }
        // Re-append each persisted chunk's signatures into the fresh
        // provenance file; `(token_count, syn‖sem‖prag bytes)` → a
        // SigEntry at the new offset, so attentional retrieval works
        // after the reload.
        let restore_sigs = |sigs: &[(u16, Vec<u8>)]| -> candle::Result<Vec<SigEntry>> {
            let n_bytes = TokenSignature::BYTE_LEN;
            let mut out = Vec::with_capacity(sigs.len());
            for (token_count, bytes) in sigs {
                let tc = *token_count as usize;
                let want = tc * n_bytes * 3;
                if bytes.len() != want {
                    candle::bail!(
                        "restore_sigs: chunk has {} sig bytes, expected {want}",
                        bytes.len()
                    );
                }
                let depth = |d: usize| -> Vec<TokenSignature> {
                    bytes[d * tc * n_bytes..(d + 1) * tc * n_bytes]
                        .chunks_exact(n_bytes)
                        .map(|c| {
                            let arr: [u8; TokenSignature::BYTE_LEN] = c.try_into().unwrap();
                            TokenSignature::from_bytes(&arr)
                        })
                        .collect()
                };
                out.push(
                    self.provenance
                        .append(&depth(0), &depth(1), &depth(2))
                        .map_err(|e| candle::Error::Msg(format!("restore sig: {e}")))?,
                );
            }
            Ok(out)
        };
        match conversation.reconstruct_from_log(n_layers, restore_sigs) {
            Ok(0) => {}
            Ok(n) => tracing::info!("substrate reload: {n} turns restored from redo log"),
            Err(e) => tracing::error!("substrate reload failed: {e}"),
        }
    }

    // ── Raw KVQ extraction ────────────────────────────────────────────

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

    // ── Provenance signature extraction ───────────────────────────────

    /// Dual-layer multi-head variant for MH_XOR_QQ_l0xl4.
    ///
    /// Extracts R16 Q data for `layer_a` (band-start, l0) and `layer_b`
    /// (band-centre, l4), builds a multi-head [`TurnSignatures`] for each,
    /// then XOR-folds them token-by-token.  This is the production path for
    /// the `MH_XOR_QQ_l0xl4` strategy.
    fn handle_extract_mh_dual_signatures(
        &self,
        seq_idx: usize,
        layer_a: usize,
        layer_b: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<crate::provenance::TurnSignatures>, ConversationError> {
        let (blocks_a, blocks_b) = {
            let mut layers = self
                .session
                .gather_r16_kv_provenance_layers(seq_idx, &[layer_a, layer_b], block_range)
                .map_err(ConversationError::Model)?
                .into_iter();
            let a = layers.next().unwrap_or_default();
            let b = layers.next().unwrap_or_default();
            (a, b)
        };

        if blocks_a.is_empty() {
            return Ok(vec![]);
        }

        let n_kv_head = self.session.n_kv_head();
        let head_dim = self.session.head_dim();
        let chunk = candle_nn::CHUNK_SIZE;

        let sigs_a = crate::provenance::extract_mh_signatures_from_r16_dump(
            &blocks_a, n_kv_head, head_dim, chunk,
        );
        let sigs_b = crate::provenance::extract_mh_signatures_from_r16_dump(
            &blocks_b, n_kv_head, head_dim, chunk,
        );

        let merged = sigs_a
            .iter()
            .zip(sigs_b.iter())
            .map(|(a, b)| crate::provenance::merge_turn_signatures_xor(a, b))
            .collect();
        Ok(merged)
    }

    /// Extract Q-vector provenance signatures for newly-completed 32-token
    /// blocks across all active decode sequences, immediately after each
    /// forward pass while the R16 backing is still intact.
    ///
    /// Called right after `forward_batched` returns.  Results are accumulated
    /// in `DecodeState::prov_sig_entries` and passed wholesale to
    /// `perform_seal_and_write` at Done time, so seal-time extraction only
    /// covers the residual partial block (guaranteed to still be R16 since
    /// the bg_quantizer never compresses the active block).
    pub(super) fn extract_prov_after_step(&mut self, seq_ids: &[SequenceId]) {
        let ProvenanceLayerIndices {
            syn_l0,
            syn_l4,
            sem_l0,
            sem_l4,
            prag_l0,
            prag_l4,
        } = self.model_core.provenance_layer_indices;
        let provenance = Arc::clone(&self.provenance);

        for &seq_id in seq_ids {
            // Only view-based sequences have entries in turn_views.
            let view_state = match self.turn_views.get(&seq_id).copied() {
                Some(v) => v,
                None => continue,
            };
            let parent_id = view_state.parent_id;
            let turn_start = view_state.turn_start_parent_blocks;

            // Complete blocks = tokens written / CHUNK_SIZE.  The active
            // (partial) block is excluded — not yet finalized, and the
            // bg_quantizer never compresses it, so it's safely extractable
            // at seal time.
            let view_offset = self.session.sequence_offset(seq_id.0).unwrap_or(0);
            let complete_view_blocks = view_offset / candle_nn::CHUNK_SIZE;

            // prev = high-water mark from prior steps.  Also skip any
            // borrowed blocks (indices < turn_start) which belong to the
            // projected context, not this turn.
            let prev = self
                .slot_sig_blocks_processed
                .get(&parent_id)
                .copied()
                .unwrap_or(0);
            let extract_from = prev.max(turn_start);

            if complete_view_blocks <= extract_from {
                continue;
            }
            let range = (extract_from, complete_view_blocks);

            // Extract Q-sigs from R16 blocks on the view slot (MH_XOR_QQ_l0xl4).
            let syn = self.handle_extract_mh_dual_signatures(seq_id.0, syn_l0, syn_l4, Some(range));
            let sem = self.handle_extract_mh_dual_signatures(seq_id.0, sem_l0, sem_l4, Some(range));
            let prag =
                self.handle_extract_mh_dual_signatures(seq_id.0, prag_l0, prag_l4, Some(range));

            let (syn_b, sem_b, prag_b) = match (syn, sem, prag) {
                (Ok(s), Ok(sm), Ok(p)) => (s, sm, p),
                _ => continue,
            };

            let total = syn_b.len().min(sem_b.len()).min(prag_b.len());
            if total == 0 {
                continue;
            }

            let mut new_entries: Vec<crate::provenance::SigEntry> = Vec::with_capacity(total);
            let mut new_high = prev;
            for j in 0..total {
                let n = syn_b[j]
                    .sigs
                    .len()
                    .min(sem_b[j].sigs.len())
                    .min(prag_b[j].sigs.len());
                match provenance.append(
                    &syn_b[j].sigs[..n],
                    &sem_b[j].sigs[..n],
                    &prag_b[j].sigs[..n],
                ) {
                    Ok(entry) => {
                        new_entries.push(entry);
                        new_high = extract_from + j + 1;
                    }
                    Err(e) => tracing::warn!(
                        "prov extract: parent={} block={}: {e}",
                        parent_id,
                        extract_from + j,
                    ),
                }
            }

            // Update high-water mark so the next step and seal skip
            // already-extracted blocks.
            if new_high > prev {
                self.slot_sig_blocks_processed.insert(parent_id, new_high);
            }
            // Append new entries to the sequence's accumulated list.
            if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                state.prov_sig_entries.extend(new_entries);
            }
        }
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

    /// Run BDP scan + projection for the active view's policy, then
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
    /// parent in a contiguous prefix.  BDP-driven `top_k` swaps broke
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
    ///    decode window, run BDP against substrate sigs to refresh
    ///    section / turn scores, project the new
    ///    `(sections, turns)` selection.  Unchanged from the old
    ///    code.
    /// 2. **Capture the active turn's tail**: snapshot the view per
    ///    layer as `SealedSequence`s and slice off
    ///    `chunks[original_borrowed..]`.  Those chunks hold every
    ///    K/V byte computed since the view was carved (user prefill
    ///    + decoded-so-far).  The Q vectors and BDP signatures for
    ///    the decoded tokens are already captured by provenance —
    ///    nothing in the tail needs regenerating.
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
    /// For typical reproject use (small BDP-driven catalog swaps
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
    fn reproject_view(&mut self, view_id: SequenceId) -> Result<SequenceId, ConversationError> {
        let _t_total = PhaseTimer::new("reproject_total");
        // Clone the policy out — `swap_view_with_new_ranges` mutates
        // `active_decodes` so we cannot hold a borrow over it.
        let policy = match self
            .active_decodes
            .get(&view_id)
            .and_then(|s| s.reprojection.clone())
        {
            Some(p) => p,
            None => return Ok(view_id),
        };

        // 1. Compute the probe window.
        //
        //    R16 captures Q on every forward pass (prefill *and* decode),
        //    so live signatures cover the full view — both the user's
        //    prefill query and any decode tokens emitted so far.
        //
        //    The window is: min(max_probe_tokens, view_offset)
        //
        //    No lower bound on the decoded-delta; structural/turn-boundary
        //    tokens are already excluded by `probe_filter_token_ids`, so
        //    there is no need to clip to decode-only positions.  Including
        //    the prefill query is essential: at first reprojection (e.g.
        //    after `<tool_call>\n`) the decode delta is tiny and would
        //    miss the user's intent entirely.
        let view_offset = self.session.sequence_offset(view_id.0).unwrap_or(0);
        if view_offset == 0 {
            return Ok(view_id);
        }

        let (decoded_count, generated_tokens_snapshot, prefill_tokens_snapshot) = {
            let state = self.active_decodes.get(&view_id).ok_or_else(|| {
                ConversationError::Channel(format!("reproject: missing decode state {view_id}"))
            })?;
            (
                state.generated_tokens.len(),
                state.generated_tokens.clone(),
                state.prefill_tokens.clone(),
            )
        };

        let max_probe = policy.max_probe_tokens.max(1);
        let window = max_probe.min(view_offset);
        if window == 0 {
            return Ok(view_id);
        }
        let probe_lo = view_offset - window; // inclusive
        let probe_hi = view_offset; // exclusive

        // 2. Gather live Q from a NARROW block range covering only the probe window.
        //    The fast path (CUDA) launches one kernel per layer and does a single
        //    DtoH copy, replacing O(n_head × N_PALETTE × n_blocks) memcpy_dtov stalls.
        let t_probe = Instant::now();
        let block_lo = probe_lo / self.chunk_size;
        let block_hi = probe_hi.div_ceil(self.chunk_size);
        let range = Some((block_lo, block_hi));

        let raw_layers = {
            let mut layers = self
                .session
                .gather_r16_kv_provenance_layers(
                    view_id.0,
                    &policy.provenance_layer_indices.as_array(),
                    range,
                )
                .map_err(ConversationError::Model)?
                .into_iter();
            let raw_syn_l0 = layers.next().unwrap_or_default();
            let raw_syn_l4 = layers.next().unwrap_or_default();
            let raw_sem_l0 = layers.next().unwrap_or_default();
            let raw_sem_l4 = layers.next().unwrap_or_default();
            let raw_prag_l0 = layers.next().unwrap_or_default();
            let raw_prag_l4 = layers.next().unwrap_or_default();
            (
                raw_syn_l0,
                raw_syn_l4,
                raw_sem_l0,
                raw_sem_l4,
                raw_prag_l0,
                raw_prag_l4,
            )
        };
        let (raw_syn_l0, raw_syn_l4, raw_sem_l0, raw_sem_l4, raw_prag_l0, raw_prag_l4) = raw_layers;
        if raw_syn_l0.is_empty() {
            return Ok(view_id);
        }

        let n_kv_head = self.session.n_kv_head();
        let head_dim = self.session.head_dim();
        let chunk = self.chunk_size;
        let block_indices: Vec<usize> = raw_syn_l0.iter().map(|(idx, _, _, _)| *idx).collect();

        let merge = |a: &[_], b: &[_]| {
            let sa = crate::provenance::extract_mh_signatures_from_r16_dump(
                a, n_kv_head, head_dim, chunk,
            );
            let sb = crate::provenance::extract_mh_signatures_from_r16_dump(
                b, n_kv_head, head_dim, chunk,
            );
            sa.iter()
                .zip(sb.iter())
                .map(|(x, y)| crate::provenance::merge_turn_signatures_xor(x, y))
                .collect::<Vec<_>>()
        };
        let syn_blocks = merge(&raw_syn_l0, &raw_syn_l4);
        let sem_blocks = merge(&raw_sem_l0, &raw_sem_l4);
        let prag_blocks = merge(&raw_prag_l0, &raw_prag_l4);

        // 3. Build per-depth probe vectors from the window + structural
        //    filter.  For a view position `p ∈ [probe_lo, probe_hi)`:
        //      - block_idx = p / chunk_size
        //      - slot       = p % chunk_size
        //      - decoded_idx = decoded_count - (view_offset - p)
        //      - skip if `generated_tokens[decoded_idx]` is in the
        //        probe-filter set (whitespace, markdown, chat-template
        //        scaffolding) — those tokens match across turns purely
        //        on shared structure rather than on shared content,
        //        biasing the BDP scan.
        let filter: &[u32] = &policy.probe_filter_token_ids;
        let chunk_size = self.chunk_size;
        let extract_window = |sigs_per_block: &[crate::provenance::TurnSignatures]|
                              -> Vec<crate::provenance::TokenSignature> {
            let mut out: Vec<crate::provenance::TokenSignature> = Vec::with_capacity(window);
            for (block_idx, block_sigs) in block_indices.iter().zip(sigs_per_block.iter()) {
                let block_start = block_idx * chunk_size;
                let block_end = block_start + chunk_size;
                if block_end <= probe_lo || block_start >= probe_hi {
                    continue;
                }
                let lo = probe_lo.max(block_start);
                let hi = probe_hi.min(block_end);
                for p in lo..hi {
                    let slot = p - block_start;
                    // `view_offset - p` is the distance from the current
                    // decode head back to position `p`.  If that distance
                    // is within the decode buffer the token is a generated
                    // token and may be filtered; positions further back
                    // are prefill tokens — include them unconditionally.
                    let dist = view_offset - p;
                    if dist <= decoded_count {
                        let decoded_idx = decoded_count - dist;
                        if let Some(&tok) = generated_tokens_snapshot.get(decoded_idx) {
                            if filter.contains(&tok) {
                                continue;
                            }
                        }
                    }
                    if let Some(&sig) = block_sigs.sigs.get(slot) {
                        out.push(sig);
                    }
                }
            }
            out
        };

        let probe_syn = extract_window(&syn_blocks);
        let probe_sem = extract_window(&sem_blocks);
        let probe_prag = extract_window(&prag_blocks);
        if probe_syn.is_empty() {
            return Ok(view_id);
        }
        let probe_ms = t_probe.elapsed().as_millis() as u64;
        record_phase(t_probe, "reproject_probe_extract");

        // 4. Snapshot corpus (turns + sections) from substrate, run BDP
        //    scan against both, write scores back.  Sections (e.g. tool
        //    definitions prefilled at startup) compete on the same probe
        //    as historical turns, so an "I want to compute" intent
        //    surfaces the calculator section's sigs the same way it
        //    surfaces a previously-asked computation turn.
        let t_scan = Instant::now();
        let (turn_corpus, section_corpus): (
            Vec<(crate::projection::TurnKey, Vec<crate::provenance::SigEntry>)>,
            Vec<(SectionId, Vec<crate::provenance::SigEntry>)>,
        ) = {
            let view = policy.substrate.read();
            // BdpScanner is keyed by `TurnKey` natively — no group/timeline
            // translation at the boundary.
            let turns: Vec<_> = view
                .all_turns()
                .filter_map(|key| {
                    let entries = view.sig_entries_of(key.timeline, key.index).to_vec();
                    if entries.is_empty() {
                        None
                    } else {
                        Some((key, entries))
                    }
                })
                .collect();
            let sections: Vec<_> = view
                .all_sections()
                .map(|sid| (sid, view.section_sig_entries(sid).to_vec()))
                .filter(|(_, e)| !e.is_empty())
                .collect();
            (turns, sections)
        };

        if turn_corpus.is_empty() && section_corpus.is_empty() {
            return Ok(view_id);
        }

        // ── Trace-only validation: probe health + section corpus health ────────
        if tracing::enabled!(tracing::Level::TRACE) {
            // Probe stats
            let probe_n = probe_syn.len();
            let probe_nonzero_syn = probe_syn.iter().filter(|s| s.as_u128() != 0).count();
            let probe_nonzero_prag = probe_prag.iter().filter(|s| s.as_u128() != 0).count();
            tracing::trace!(
                probe_tokens = probe_n,
                probe_window = format!("{}..{}", probe_lo, probe_hi),
                nonzero_syn = probe_nonzero_syn,
                nonzero_prag = probe_nonzero_prag,
                "reproject probe health"
            );

            // Decode probe positions to text: decode + prefill regions only,
            // positions that fall in the borrowed-parent region are marked <parent>.
            let pf_len = prefill_tokens_snapshot.len();
            let probe_text: Vec<String> = (probe_lo..probe_hi)
                .map(|p| {
                    let dist = view_offset - p;
                    let tok = if dist <= decoded_count {
                        generated_tokens_snapshot.get(decoded_count - dist).copied()
                    } else {
                        let pf_dist = dist - decoded_count;
                        if pf_dist <= pf_len {
                            prefill_tokens_snapshot.get(pf_len - pf_dist).copied()
                        } else {
                            None
                        }
                    };
                    match tok {
                        Some(id) => self
                            .tokenizer
                            .decode(&[id], true)
                            .unwrap_or_else(|_| format!("[{}]", id)),
                        None => "<parent>".to_string(),
                    }
                })
                .collect();
            tracing::trace!(
                tokens = ?probe_text,
                "reproject probe tokens"
            );

            // Section corpus health
            for (sid, entries) in &section_corpus {
                let total_tokens: usize = entries.iter().map(|e| e.token_count as usize).sum();
                let nonzero_entries = entries.iter().filter(|e| e.token_count > 0).count();
                let name = policy
                    .projection
                    .section(*sid)
                    .map(|s| s.name.as_str())
                    .unwrap_or("<unknown>");
                tracing::trace!(
                    section = name,
                    sig_entries = entries.len(),
                    nonzero_entries,
                    total_tokens,
                    "reproject section corpus"
                );
            }
        }

        let mut scanner = crate::provenance::BdpScanner::new().with_span_alpha(policy.span_alpha);
        scanner.scan(
            &policy.provenance,
            &probe_syn,
            &probe_sem,
            &probe_prag,
            &turn_corpus,
        )?;
        scanner.scan_sections(
            &policy.provenance,
            &probe_syn,
            &probe_sem,
            &probe_prag,
            &section_corpus,
        )?;

        // Build a transient per-projection scores cache from the scanner
        // output. This lives on the stack for the duration of the
        // re-projection; it is never stored in the substrate (§Phase-3
        // BDP: scores are projection-local, not session-persistent).
        let projection_scores = scanner.to_projection_scores();
        let scan_ms = t_scan.elapsed().as_millis() as u64;
        record_phase(t_scan, "reproject_bdp_scan");

        // 5. Re-project against the freshly-scored substrate.
        //
        //    Produce the same `(projected_sections, projected_turns)`
        //    pair that `SchedulerRequest::SubmitTurn` builds: only
        //    sealed substrate entries make the cut.  These two lists
        //    drive the zero-copy rebuild in step 6.
        let t_project = Instant::now();
        let view_state = self.turn_views.get(&view_id).copied().ok_or_else(|| {
            ConversationError::Channel(format!("reproject: missing view state {view_id}"))
        })?;
        let parent_id = view_state.parent_id;
        let mut turn_keys_for_elevate: Vec<TurnKey> = Vec::new();
        let (projected_sections, projected_turns): (Vec<SectionId>, Vec<(GroupId, TurnIndex)>) = {
            let view = policy
                .substrate
                .read_for_scored(policy.target, &projection_scores);
            // Prefill mode: the section corpus is prefill-Q of tool
            // descriptions, so reprojection scores it with the calibrated
            // prefill profile (Max / semantic, no threshold gate).
            let projection =
                policy
                    .projection
                    .project_with_mode(policy.target, &view, ProjectionMode::Prefill);
            let mut sections: Vec<SectionId> = Vec::with_capacity(projection.system_prompt.len());
            for sec in &projection.system_prompt {
                if view.section_sealed_of(sec.id).is_some() {
                    sections.push(sec.id);
                }
            }
            let turns: Vec<(GroupId, TurnIndex)> = projection
                .turns
                .iter()
                .filter_map(|resolved| {
                    let g = resolved.group();
                    let t = resolved.index();
                    let timeline = view.timelines_for_group(g).next()?;
                    // Tier-agnostic existence check (see comment at
                    // the SubmitTurn site). Cold-marker turns must
                    // pass — elevate_to_hot below will pull them up.
                    view.turn_tier_state(timeline, t)?;
                    turn_keys_for_elevate.push(TurnKey::new(timeline, t));
                    Some((g, t))
                })
                .collect();
            (sections, turns)
        };
        let project_ms = t_project.elapsed().as_millis() as u64;
        record_phase(t_project, "reproject_project");

        // 6. Zero-copy rebuild.
        //
        //    The previous implementation tried to *narrow* the view's
        //    borrow window over a still-correct parent.  That assumed
        //    the new projection's selected sections / turns were
        //    already materialised on parent in a contiguous prefix.
        //    BDP-driven `top_k` swaps invalidated that assumption mid-
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
        //         position-independent, Q vectors and BDP signatures
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
        self.slot_sig_blocks_processed.insert(parent_id, 0);
        // Reset the slot_tokens diagnostic log so it stays in sync with
        // the post-rebuild slot contents.  apply_projection re-populates
        // it with the new prefix's tokens; the active turn tokens get
        // re-added below.  Gated on `context-dump` — the map is empty
        // when the feature is off, so the get_mut would no-op anyway.
        #[cfg(feature = "context-dump")]
        if let Some(toks) = self.slot_tokens.get_mut(&parent_id) {
            toks.clear();
        }
        // Select-promote (cold → warm → hot) the new working set
        // before re-applying. Same shape as the SubmitTurn path: a
        // single batched scatter per layer + per-item cold-recover
        // turns the per-unit `ensure_*_hot` calls inside
        // `apply_projection` into hot-hit no-ops.
        if !projected_sections.is_empty() || !turn_keys_for_elevate.is_empty() {
            let backings = self.session.backings().to_vec();
            let device = self.session.device().clone();
            // Same rationale as the SubmitTurn path: scheduler is
            // single-owner and blocks here before re-applying, so
            // the main inference stream is the right place for the
            // scatter — no extra sync, subsequent prefill kernels
            // serialise behind it for free.
            let main_stream = match &device {
                Device::Cuda(d) => d.cuda_stream(),
                _ => panic!("scheduler: requires a CUDA device"),
            };
            // Working-set-aware evict before elevate (see SubmitTurn
            // site for rationale).
            let evicted = evict_from_hot(
                &policy.substrate,
                &projected_sections,
                &turn_keys_for_elevate,
            );
            if evicted.count > 0 {
                tracing::debug!(
                    target: "candle_conversation::persistence::tier",
                    count = evicted.count,
                    bytes = evicted.bytes,
                    "select-evict complete (reproject)"
                );
            }
            match elevate_to_hot(
                &policy.substrate,
                &backings,
                &device,
                &main_stream,
                &mut self.elevate_pinned_scratch,
                &mut self.cold_load_stager,
                &projected_sections,
                &turn_keys_for_elevate,
            ) {
                Ok(report) => {
                    tracing::debug!(
                        target: "candle_conversation::persistence::tier",
                        already_hot = report.already_hot,
                        warm_to_hot = report.warm_to_hot,
                        cold_to_hot = report.cold_to_hot,
                        missing = report.missing,
                        failed = report.failed,
                        bytes_warm_to_hot = report.bytes_warm_to_hot,
                        bytes_cold_to_hot = report.bytes_cold_to_hot,
                        "select-promote complete (reproject)"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "select-promote (reproject) failed for parent {parent_id}: {e} — \
                         apply_projection will fall back to per-unit ensure_*_hot"
                    );
                }
            }
        }

        self.apply_projection(
            parent_id,
            BlockCount(0),
            &projected_sections,
            &projected_turns,
        )?;
        let new_prefix_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);

        // Under the read-only projection model there's no COW prefix
        // duplication to guard against: the view never copied bytes from
        // the parent's partial — it borrowed the partial read-only and
        // wrote new K/V into a fresh active chunk. So the captured tail
        // can be re-injected directly on top of the freshly-projected
        // parent with no truncate dance.

        // Append the captured tail to parent (metadata-only Arc clone).
        let tail_token_count: usize = tail_per_layer.first().map(|s| s.token_count).unwrap_or(0);
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

        // Carve a fresh view from the rebuilt parent.  Empty
        // `effective_ranges` means "borrow every chunk".
        let parent_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        let effective_ranges: Vec<BlockRange> = if parent_block_count == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_block_count)]
        };
        let (new_view_id, new_borrowed) = self.create_view(parent_id, &effective_ranges)?;
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
        let swap_ms = t_swap.elapsed().as_millis() as u64;
        record_phase(t_swap, "reproject_swap");

        tracing::info!(
            target: "candle_conversation::scheduler::reproject",
            from_view = view_id.0,
            to_view = new_view_id.0,
            new_prefix_blocks = new_prefix_block_count,
            tail_tokens = tail_token_count,
            parent_blocks_after = parent_block_count,
            new_borrowed = new_borrowed.0,
            probe_ms,
            scan_ms,
            project_ms,
            swap_ms,
            sections = projected_sections.len(),
            turns = projected_turns.len(),
            "reproject (zero-copy rebuild)",
        );

        Ok(new_view_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scheduler dispatch unit tests
// ═══════════════════════════════════════════════════════════════════════════
//
// These tests verify that `run_prefill_with_shift` dispatches to
// `forward_batched_with_write_shifts` when shift ≠ 0, and to plain
// `forward_batched` when shift = 0.  The bug was previously invisible
// because the mock model used the default trait impl which silently
// drops the shifts.
//
// The `RecordingModel` is a minimal `ManagedBatchedModel` that:
//  - records which dispatch method was used last
//  - records the write_offset_shifts slice it received
//  - returns dummy CPU tensors so no GPU is needed
//
// `BatchedInferenceSession` is constructed on `Device::Cpu` with tiny arena
// dimensions so these tests run without CUDA.

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Tensor};
    use candle_transformers::models::batched_inference::{
        BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
    };
    use std::str::FromStr;
    use std::sync::{Arc, Mutex};

    // ── Recording model ──────────────────────────────────────────────────────

    #[derive(Clone)]
    struct RecordingModel {
        device: candle::Device,
        vocab_size: usize,
        /// Stores the name of the last dispatch method called.
        last_call: Arc<Mutex<Option<String>>>,
        /// Stores the write_offset_shifts passed to `forward_batched_with_write_shifts`,
        /// or `None` if `forward_batched` was called instead.
        last_shifts: Arc<Mutex<Option<Vec<u32>>>>,
    }

    impl RecordingModel {
        fn new() -> Self {
            Self {
                device: candle::Device::Cpu,
                vocab_size: 64,
                last_call: Arc::new(Mutex::new(None)),
                last_shifts: Arc::new(Mutex::new(None)),
            }
        }

        fn record(&self, method: &str, shifts: Option<Vec<u32>>) {
            *self.last_call.lock().unwrap() = Some(method.to_owned());
            *self.last_shifts.lock().unwrap() = shifts;
        }

        fn last_call(&self) -> Option<String> {
            self.last_call.lock().unwrap().clone()
        }

        fn last_shifts(&self) -> Option<Vec<u32>> {
            self.last_shifts.lock().unwrap().clone()
        }

        fn dummy_logits(&self, n: usize) -> candle::Result<Vec<Tensor>> {
            (0..n)
                .map(|_| Tensor::zeros((1, self.vocab_size), DType::F32, &self.device))
                .collect()
        }
    }

    impl ManagedBatchedModel for RecordingModel {
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

        fn forward_batched(
            &self,
            _session: &mut BatchedInferenceSession,
            seq_indices: &[usize],
            _inputs: &[Tensor],
        ) -> candle::Result<Vec<Tensor>> {
            self.record("forward_batched", None);
            self.dummy_logits(seq_indices.len())
        }

        fn forward_batched_with_write_shifts(
            &self,
            _session: &mut BatchedInferenceSession,
            seq_indices: &[usize],
            _inputs: &[Tensor],
            write_offset_shifts: &[u32],
        ) -> candle::Result<Vec<Tensor>> {
            self.record(
                "forward_batched_with_write_shifts",
                Some(write_offset_shifts.to_vec()),
            );
            self.dummy_logits(seq_indices.len())
        }

        fn prune(&self) -> candle::Result<()> {
            Ok(())
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

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

    fn make_test_scheduler(
        model: RecordingModel,
    ) -> (Scheduler, crossbeam::channel::Sender<SchedulerRequest>) {
        let (tx, rx) = crossbeam::channel::bounded(16);
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let scheduler = Scheduler::new(
            rx,
            Box::new(model),
            session,
            tokenizer,
            vec![0u32].into(), // eos_tokens
            64,                // vocab_size
            8,                 // max_recent_len
            false,             // show_special_tokens
            None,              // penalty_log_path
            DecodeHealthConfig::default(),
            512, // max_prefill_chunk
            Arc::new(crate::provenance::ProvenanceFile::new().unwrap()),
            ModelCoreProperties {
                num_layers: 6,
                n_kv_heads: 4,
                head_dim: 128,
                provenance_layer_indices: ProvenanceLayerIndices {
                    syn_l0: 0,
                    syn_l4: 1,
                    sem_l0: 2,
                    sem_l4: 3,
                    prag_l0: 4,
                    prag_l4: 5,
                },
                k_hi_error_threshold_factor: 1.0,
                k_low_error_threshold_factor: 1.0,
                v_hi_error_threshold_factor: 1.0,
                v_low_error_threshold_factor: 1.0,
            },
            PersistenceTrigger::noop(),
        );
        (scheduler, tx)
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// `run_prefill_with_shift` with a non-zero shift must dispatch to
    /// `forward_batched_with_write_shifts` and carry the exact shift value.
    ///
    /// Before the fix the `_wos` field was silently discarded and the call
    /// fell through to `forward_batched`.
    #[test]
    fn run_prefill_nonzero_shift_dispatches_with_write_shifts() {
        let model = RecordingModel::new();
        let model_ref = model.clone();
        let (mut scheduler, _tx) = make_test_scheduler(model);

        let raw_id = scheduler
            .session
            .create_sequence()
            .expect("create_sequence failed");
        let seq_id = SequenceId(raw_id);
        let shift = 7usize;

        let result = scheduler.run_prefill_with_shift(seq_id, &[10u32, 20, 30], shift);

        assert_eq!(
            model_ref.last_call().as_deref(),
            Some("forward_batched_with_write_shifts"),
            "run_prefill_with_shift(non-zero) must dispatch to \
             forward_batched_with_write_shifts"
        );
        assert_eq!(
            model_ref.last_shifts(),
            Some(vec![shift as u32]),
            "run_prefill_with_shift must pass the exact shift value ({shift})"
        );
        assert!(
            result.is_ok(),
            "run_prefill_with_shift failed: {:?}",
            result.err()
        );
    }

    /// `run_prefill_with_shift` with shift = 0 must use plain `forward_batched`
    /// (no pointless overhead for the common case).
    #[test]
    fn run_prefill_zero_shift_dispatches_to_forward_batched() {
        let model = RecordingModel::new();
        let model_ref = model.clone();
        let (mut scheduler, _tx) = make_test_scheduler(model);

        let raw_id = scheduler
            .session
            .create_sequence()
            .expect("create_sequence failed");
        let seq_id = SequenceId(raw_id);

        let result = scheduler.run_prefill_with_shift(seq_id, &[1u32, 2], 0);

        assert_eq!(
            model_ref.last_call().as_deref(),
            Some("forward_batched"),
            "run_prefill_with_shift(0) must dispatch to plain forward_batched, \
             not forward_batched_with_write_shifts"
        );
        assert_eq!(
            model_ref.last_shifts(),
            None,
            "plain forward_batched does not receive a shifts slice"
        );
        assert!(
            result.is_ok(),
            "run_prefill_with_shift(0) failed: {:?}",
            result.err()
        );
    }

    // ── view-creation tests ─────────────────────────────────────────────────

    /// Explicit `visible_block_ranges` over a populated parent must create a valid view.
    #[test]
    fn create_view_with_explicit_ranges_creates_view() {
        let model = RecordingModel::new();
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
            Arc::new(crate::provenance::ProvenanceFile::new().unwrap()),
            ModelCoreProperties {
                num_layers: 6,
                n_kv_heads: 4,
                head_dim: 128,
                provenance_layer_indices: ProvenanceLayerIndices {
                    syn_l0: 0,
                    syn_l4: 1,
                    sem_l0: 2,
                    sem_l4: 3,
                    prag_l0: 4,
                    prag_l4: 5,
                },
                k_hi_error_threshold_factor: 1.0,
                k_low_error_threshold_factor: 1.0,
                v_hi_error_threshold_factor: 1.0,
                v_low_error_threshold_factor: 1.0,
            },
            PersistenceTrigger::noop(),
        );

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);
        // Populate one full chunk on the parent so the view has a block to borrow.
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let input = candle::Tensor::new(&tokens[..], &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[parent_raw], tokens.len())
            .unwrap();
        scheduler
            .model
            .forward_batched(&mut scheduler.session, &[parent_raw], &[input])
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
        let model = RecordingModel::new();
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
            Arc::new(crate::provenance::ProvenanceFile::new().unwrap()),
            ModelCoreProperties {
                num_layers: 6,
                n_kv_heads: 4,
                head_dim: 128,
                provenance_layer_indices: ProvenanceLayerIndices {
                    syn_l0: 0,
                    syn_l4: 1,
                    sem_l0: 2,
                    sem_l4: 3,
                    prag_l0: 4,
                    prag_l4: 5,
                },
                k_hi_error_threshold_factor: 1.0,
                k_low_error_threshold_factor: 1.0,
                v_hi_error_threshold_factor: 1.0,
                v_low_error_threshold_factor: 1.0,
            },
            PersistenceTrigger::noop(),
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

    // ── View swap tests ──────────────────────────────────────────────────────
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
}
