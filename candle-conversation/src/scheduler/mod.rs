//! The scheduler: single thread that owns all GPU resources.
//!
//! Runs a continuous loop alternating between prefill and decode.
//! Phase 1 uses single-mode prefill (no small/large split).
mod decode;
#[cfg(feature = "kv-zero-check")]
pub(crate) mod kv_zero_check;
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
use crate::persistence::elevate::elevate_to_hot;
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::persistence::thread::PersistenceTrigger;
use crate::projection::{
    Builder, CompressionPrompt, Conversation, GeneratedIdentity, GroupId, ProjectionMode,
    ProjectionSegment, ProjectionTarget, ResolvedSection, ResolvedTurn, SealedKind, SectionId,
    SelectionState, SummaryMode, TimelineId, TurnId, TurnIndex, TurnKey,
};
use crate::provenance::{encode_wide_sigs, extract_q_vector_r16, fold_provenance, WideQSig};
use crate::sequence_handle::{BlockCount, BlockRange, SequenceId};
use crate::substrate::{ResidenceIndex, TurnContentBounds, TurnPartWrite};
use crate::summary_tree::{
    leaf_skeleton, structural_rollup, SelectionDiagnostics, SummariserTrigger, TurnKind,
};
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::{SubstrateReloadStatus, TurnStats};

use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, IndexOp, Tensor};
use candle_nn::kv_cache::{quantize_sealed_in_place, QuantFormat, SealedSequence};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel,
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::{HashMap, VecDeque};
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
        /// Carried into `TurnPart::content_bounds` at seal time so the
        /// compressor can window content-only halves. See
        /// [`TurnContentBounds`].
        content_bounds: TurnContentBounds,
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
        /// `Ok(turn_index)` on success — the sealed substrate
        /// `TurnIndex` of the new compressed turn.  `Err(msg)` on a
        /// soft failure (model output unparseable, transient GPU
        /// error); the summariser retries.
        response_tx: Sender<Result<TurnIndex, String>>,
    },

    /// Re-run substrate reconstruction on the scheduler thread — used after a
    /// compaction rewrites the redo log, so the scheduler-side view (KV residence
    /// + offsets) is rebuilt from the new log. Marks `status` finished when done.
    ReconstructSubstrate {
        conversation: Conversation,
        status: Arc<SubstrateReloadStatus>,
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
    belief: crate::projection::PriorBelief,
    /// Full prefill token sequence pinned into the slot at this
    /// turn's submit:
    /// `[user_msg][user_end][assistant_start]` — a *suppressed* turn bakes an
    /// empty `<think></think>` right after `assistant_start`, and the
    /// `insert_turn` path additionally prepends `/no_think`.  A *thinking*
    /// turn's `<think>` is NOT here: the model emits it into `generated_tokens`.
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
    /// path (CPU-tokenised against the prefill prefix strings) and recorded
    /// into `TurnPart::content_bounds` at seal time so the compressor can
    /// window content-only halves on demand. See [`TurnContentBounds`].
    content_bounds: TurnContentBounds,
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

/// One decoded compression half, captured in `cleanup_finished` when the
/// pass's decode finishes. Holds only the decoded token ids — the stitch
/// re-prefills them into a clean turn to compute coherent K/V, so the decode's
/// own (role/context-wrong) K/V is never kept.
struct CompressionPassResult {
    tokens: Vec<u32>,
}

/// A compression node in flight. Both half-passes (user, assistant) are
/// registered together so they ride the same decode wave; each completes
/// independently in `cleanup_finished`, depositing its
/// [`CompressionPassResult`] here. When both halves are present the node is
/// stitched into one turn, recorded, and `response_tx` fires with the new
/// `TurnIndex`.
///
/// The compression DESIGN is unchanged from the synchronous path — the same
/// per-half prompt, `TurnHalf` injection, content-bounds windowing, and
/// `record_turn` write. Only the orchestration is async: the two passes are
/// normal wave-loop decodes rather than two inline synchronous decode loops.
struct CompressionJob {
    /// Conversation that owns the node's timeline.
    conversation: Conversation,
    /// Projection target (layer, group, timeline) the node records into.
    target: ProjectionTarget,
    /// Scratch slot driving the user-half pass; freed once that half completes.
    user_slot: SequenceId,
    /// Scratch slot driving the assistant-half pass.
    assistant_slot: SequenceId,
    /// Decoded user-half tokens (filled when the user pass finishes).
    user_result: Option<CompressionPassResult>,
    /// Decoded assistant-half tokens.
    assistant_result: Option<CompressionPassResult>,
    /// Summariser channel — receives the stitched node's `TurnIndex` once both
    /// halves complete, or an `Err` if either pass produces no forwarded tokens.
    response_tx: Sender<Result<TurnIndex, String>>,
}

/// A compressed turn whose marker-framed text has been enqueued for re-prefill
/// on the shared prefill wave, awaiting its seal. Keyed by `job_id` in
/// [`Scheduler::pending_compression_seals`]. When the prefill completes,
/// `complete_compression_turn` snapshots the slot's freshly-computed
/// (role-coherent) K/V, records the turn, and fires `response_tx`.
struct PendingCompressionSeal {
    conversation: Conversation,
    target: ProjectionTarget,
    user_text: String,
    assistant_text: String,
    token_ids: Vec<u32>,
    content_bounds: TurnContentBounds,
    response_tx: Sender<Result<TurnIndex, String>>,
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
    /// One half of a compression node. `cleanup_finished` captures the decoded
    /// tokens, deposits them into [`Scheduler::compression_jobs`]`[job_id]`, and
    /// — once both halves are present — stitches the node (re-prefilling the
    /// compressed exchange into coherent K/V), records it, and replies to the
    /// summariser. No substrate write happens per-pass; the stitched node's
    /// `record_turn` is the single write.
    CompressionPass {
        job_id: u64,
        /// Which half this pass decoded — `User` or `Assistant`.
        half: Role,
    },
    /// The re-prefilled compressed turn for `job_id`. Once the prefill wave
    /// finishes the marker-framed `[question][user_end][assistant_start][answer]`
    /// on its scratch slot, `promote_finished_prefills_to_decodes` snapshots the
    /// freshly-computed (role-coherent) K/V, records the turn from the pending
    /// seal stashed in [`Scheduler::pending_compression_seals`], and replies to
    /// the summariser. `max_decode_tokens` is 0 — prefill + seal, no decode.
    CompressionTurn { job_id: u64 },
    /// The assistant half's `[content … instruction user_end]` prefill, riding
    /// the shared wave so its (potentially large) content batches with the live
    /// turn instead of stalling the loop. `max_decode_tokens` is 0; once the
    /// wave finishes, `promote_finished_prefills_to_decodes` runs
    /// `finish_compression_pass_setup` — prefill `assistant_start`, sample the
    /// first token, and register the half's `CompressionPass` decode.
    CompressionSetup {
        job_id: u64,
        half: Role,
        max_tokens: usize,
    },
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
    /// The user's message text — exactly what `submit_turn`
    /// received, no role-marker envelope, no `/no_think` prefix.
    pub user_text: String,
    /// The assistant's decoded reply text — the model's response
    /// body with special tokens skipped.
    pub assistant_text: String,
    /// The combined token sequence pinned onto the slot, in slot
    /// order.  Must match the K/V chunk grid 1-1; consumed by
    /// `persist_tokens_only` for cross-process replay.
    pub token_ids: TokenBuffer,
    /// Content boundaries — see [`DecodeState::content_bounds`].
    pub content_bounds: TurnContentBounds,
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
    /// Content boundaries — see [`DecodeState::content_bounds`].
    pub(super) content_bounds: TurnContentBounds,
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

// ————————————————————————————————————————————————————————————————————————————
// Scheduler
// ————————————————————————————————————————————————————————————————————————————

/// The scheduler: single thread that owns all GPU resources.
///
/// One forward-pass "channel" (prefill or decode) accumulator for [`WaveStats`].
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
        }
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

    /// Record one section-ingest forward (startup code-read / repo-map prefill).
    /// Kept separate so the section phase isn't invisible behind `section_ms`.
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
    /// Call only when [`Self::due`] — windows with NO forwards still flush so
    /// stalls surface their phase split.
    fn flush(&mut self, kv_vram: Option<(usize, usize)>) {
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
        tracing::info!("wave {:.1}s: {body}{vram}", elapsed.as_secs_f64());
        // Phase breakdown: where the wall-clock went on the scheduler thread.
        // `drain` rising over the run ⇒ per-turn reprojection/elevate growing;
        // `reproj` rising ⇒ continuous-reproject (provenance scan/glue) growing;
        // `unaccounted` large ⇒ blocked off-thread (persistence thread / lock).
        tracing::info!(
            target: "candle_conversation::scheduler::timing",
            drain_ms = self.drain_ms,
            promote_ms = self.promote_ms,
            decode_ms = self.decode_ms,
            prefill_ms = self.prefill_ms,
            section_ms = self.section_ms,
            reproj_ms = self.reproj_ms,
            unaccounted_ms = unaccounted,
            "wave phase breakdown (scheduler-thread wall-clock; watch which grows)"
        );
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
    }
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
    /// Per-decode projection-span anchor, keyed by the decode's parent
    /// sequence (stable across reprojections): `(generated-token index, time)`
    /// of the last projection event. Each reproject (and the final seal) closes
    /// the span `[anchor.token, now]` and re-anchors. Drives the t/s on each
    /// `TurnEvent::Projection`. Pruned in `cleanup_finished`.
    reproj_anchor: HashMap<SequenceId, (u32, Instant)>,
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
    /// every turn-seal (`docs/infinite_conversations.md` §4 step ③)
    /// so the freshly-pending Normal turn is absorbed into the AVL
    /// summary tree on the next pass instead of waiting up to 250 ms
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
            reproj_anchor: HashMap::new(),
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
            turn_views: HashMap::new(),
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
            compression_event_sinks: HashMap::new(),
            timeline_projections: HashMap::new(),
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
                user_text,
                tags,
                content_bounds,
                projection_offsets,
                prefill_assistant_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
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
                }
                let seal_action = match (&projection_inputs, slot_target) {
                    (Some(_), Some(_)) => SealAction::Turn,
                    _ => SealAction::None,
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
                    // Submit-time projection is the first of the turn — the belief
                    // starts fresh (no prior). Mid-decode reprojections seed from
                    // the prior projection event to carry the online belief.
                    let projection = inputs.projection.project_with_mode_and_sink(
                        target,
                        &view,
                        ProjectionMode::Prefill,
                        &inputs.selection,
                        &crate::projection::PriorBelief::default(),
                        &mut |diag| {
                            diag_to_write = Some((target.timeline, diag));
                        },
                    );
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
                    // For the target's own group route via
                    // `target.timeline`; for other groups fall back to
                    // first-of-group (a Phase-1 single-timeline
                    // assumption that holds for cross-group references
                    // in our current schema).
                    let resolve_timeline = |g: GroupId| -> Option<TimelineId> {
                        if g == target.group {
                            Some(target.timeline)
                        } else {
                            view.timelines_for_group(g).next()
                        }
                    };
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
                                let g = rt.group();
                                let t = rt.index();
                                let Some(timeline) = resolve_timeline(g) else {
                                    continue;
                                };
                                // Include any tracked turn regardless
                                // of tier — cold-marker turns must
                                // survive this filter; `elevate_to_hot`
                                // below brings them into hot before
                                // apply_projection runs.  A `TurnHalf`
                                // elevates the same underlying turn; the
                                // half-window is derived after the turn's
                                // chunks are hot.
                                if view.turn_tier_state(timeline, t).is_some() {
                                    turn_keys_for_elevate.push(TurnKey::new(timeline, t));
                                    segments.push(seg.clone());
                                }
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
                    // `start_token`/`end_token` span per emission.
                    if !projection_offsets.is_empty() {
                        let total = view.total_token_count(target.timeline) as u32;
                        staged_composition = Some(crate::projection::from_projection(
                            &projection.segments,
                            inputs.projection.schema(),
                            &view,
                            &projection.selection_scores,
                            total,
                            0,
                            0,
                            0.0,
                        ));
                    }
                    (sections, segments)
                } else {
                    (Vec::new(), Vec::new())
                };

                // Read guard dropped above when the projection block
                // exited; now safe to take a write guard for the
                // diagnostic side-channel.  No-op when projection used
                // the rule-based path (no tree on the target timeline).
                if let Some((timeline, diag)) = diag_to_write {
                    if let Some(conv) = self.slot_conversations.get(&parent_id) {
                        conv.write().set_last_selection(timeline, diag);
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
                        let evicted = self.evict_to_fit_incoming(
                            &conversation,
                            &projected_sections,
                            &turn_keys_for_elevate,
                        );
                        if evicted.count > 0 {
                            tracing::trace!(
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
                                tracing::trace!(
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
                    user_text,
                    tags,
                    content_bounds,
                    prefill_assistant_text,
                    event_tx,
                    max_decode_tokens,
                    sampling,
                    submitted_at: Instant::now(),
                    reprojection,
                    seal_action,
                    post_decode_tokens,
                    projection_offsets,
                    staged_composition,
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
                self.slot_projection_state.remove(&sequence_id);
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

            SchedulerRequest::SubmitSummaryProbe {
                timeline,
                kind,
                children,
                height,
                response_tx,
            } => {
                // Enqueues the two per-half compression passes as normal
                // wave-loop decodes and returns immediately — the decodes ride
                // `batch_decode_step` concurrently with foreground turns, and
                // `cleanup_finished` stitches the node and fires `response_tx`
                // once both halves complete. A setup failure replies with `Err`
                // here for a soft retry on the summariser's next pass.
                if let Err(e) =
                    self.handle_summary_probe(timeline, kind, children, height, response_tx.clone())
                {
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

            SchedulerRequest::Shutdown => false,
        }
    }

    /// Enqueue a §6 compression probe over a node's children as two
    /// wave-driven passes.
    ///
    /// A node's compression is two passes selected from the target layer's
    /// `summary` config by `(kind, role)`: the **question** pass over the
    /// children's user-halves and the **answer** pass over their
    /// assistant-halves (see [`Self::enqueue_compression_pass`]). The user pass
    /// is set up synchronously (sealed halves, cheap glue); the assistant pass
    /// sends its content prefill through the shared wave. Both then decode under
    /// `batch_decode_step` concurrently with foreground turns;
    /// `complete_compression_pass` collects each, and once both land the node is
    /// re-prefilled into role-coherent K/V and sealed by
    /// `complete_compression_turn`, which replies on `response_tx`. Nothing
    /// blocks the scheduler — the handler returns as soon as both passes are
    /// enqueued, and a setup failure replies `Err` for a soft retry. The decode
    /// is argmax, capped at the `summary` level's `max_tokens`.
    ///
    /// Binary fan-in: the node's structural `children` carry the content the
    /// halves compress.
    fn handle_summary_probe(
        &mut self,
        timeline: TimelineId,
        kind: TurnKind,
        children: Vec<TurnIndex>,
        height: u8,
        response_tx: Sender<Result<TurnIndex, String>>,
    ) -> Result<(), String> {
        tracing::debug!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            children = ?children,
            "compression probe: enqueuing per-half passes",
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
        let turn_summary = self
            .timeline_projections
            .get(&timeline)
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
        // kind.
        if turn_summary.mode == SummaryMode::Structural {
            return self.seal_structural_turn(&conv, target, kind, &children, height, response_tx);
        }

        // Reserve the job id up front so both passes tag their `SealAction`
        // with it; the job entry registers once both passes are set up.
        let job_id = self.next_compression_job_id;
        self.next_compression_job_id += 1;

        // Set up both half-passes. Each returns its scratch slot id (now a
        // live `DecodeState` in `active_decodes`). On any failure, tear down
        // whatever was already enqueued so no orphan slot or job lingers.
        let user_slot = match self.enqueue_compression_pass(
            &conv,
            target,
            &children,
            Role::User,
            &turn_summary.user,
            turn_summary.max_tokens,
            job_id,
        ) {
            Ok(slot) => slot,
            Err(e) => return Err(e),
        };
        let assistant_slot = match self.enqueue_compression_pass(
            &conv,
            target,
            &children,
            Role::Assistant,
            &turn_summary.assistant,
            turn_summary.max_tokens,
            job_id,
        ) {
            Ok(slot) => slot,
            Err(e) => {
                // The user pass is already an active decode; abandon it so it
                // doesn't seal into a job that will never complete.
                self.active_decodes.remove(&user_slot);
                self.free_summary_slot(user_slot);
                return Err(e);
            }
        };

        // Register the job. Both passes now ride the wave concurrently; the
        // node stitches in `cleanup_finished` when both halves land.
        self.compression_jobs.insert(
            job_id,
            CompressionJob {
                conversation: conv,
                target,
                user_slot,
                assistant_slot,
                user_result: None,
                assistant_result: None,
                response_tx,
            },
        );
        Ok(())
    }

    /// Build and seal a structural node deterministically, skipping the model
    /// decode entirely (`mode: structural`, the repo_map directory layer). A
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
        response_tx: Sender<Result<TurnIndex, String>>,
    ) -> Result<(), String> {
        // Read each child's assistant-half *tokens* and decode them. Normal scan
        // turns are prefill-only (no stored `assistant_text`), so we slice the
        // assistant body off their token ids rather than relying on the text field.
        let child_texts: Vec<String> = {
            let view = conv.read();
            children
                .iter()
                .map(|&c| {
                    view.turn_assistant_token_ids(target.timeline, c)
                        .unwrap_or_default()
                })
                .collect::<Vec<Vec<u32>>>()
        }
        .into_iter()
        .map(|toks| self.tokenizer.decode(&toks, true).unwrap_or_default())
        .collect();
        let rollup = match kind {
            // Leaf: its one child is the raw Normal scan turn — strip annotations.
            TurnKind::SummaryOfTurns => leaf_skeleton(child_texts.first().map_or("", |s| s)),
            // SoS: its children are already-built directory skeletons.
            _ => structural_rollup(&child_texts, height),
        };
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
            self.run_prefill_with_shift(slot, &tokens[..], 0)?;
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

    /// Set up one compression half-pass, returning its scratch slot id.
    ///
    /// Both passes share the head `[Section(compressor) + user_start]` and the
    /// close glue `[instruction + user_end]`, differing in the content between:
    ///
    /// - **User half** — the source user bodies inject as sealed K/V
    ///   (role-matched, zero re-prefill). Only the glue prefills, so the whole
    ///   pass is set up synchronously via [`Self::setup_compression_decode`]
    ///   (prefill `assistant_start`, sample first, register the `CompressionPass`
    ///   decode).
    /// - **Assistant half** — the source assistant bodies must be re-prefilled
    ///   (their original K/V are role/context-wrong here). Only the cheap prefix
    ///   is laid down synchronously; the `[content… close]` run is enqueued on
    ///   the **shared prefill wave** (a [`SealAction::CompressionSetup`] unit), so
    ///   its potentially-large content batches with the live turn. The setup tail
    ///   ([`Self::finish_compression_pass_setup`]) runs in the wave-completion
    ///   hook, then the decode advances in `batch_decode_step` and completes in
    ///   `cleanup_finished`.
    fn enqueue_compression_pass(
        &mut self,
        conv: &Conversation,
        target: ProjectionTarget,
        children: &[TurnIndex],
        role: Role,
        prompt: &CompressionPrompt,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<SequenceId, String> {
        // Lazily seal this half's summary system-prompt framing as a content
        // section the first time it is used, then inject it (zero-copy Arc clone,
        // zero re-prefill) at the head of the segment list. The user pass
        // therefore prefills only the framing glue.
        let summary_section = self
            .ensure_summary_section(conv, prompt)
            .map_err(|e| format!("SubmitSummaryProbe: seal summary section: {e}"))?;

        // The compressor system prompt + the `user_start` opener head every pass.
        let mut prefix: Vec<ProjectionSegment> = Vec::new();
        prefix.push(ProjectionSegment::Sealed(SealedKind::Section(
            ResolvedSection {
                id: summary_section,
            },
        )));
        let open = self.boundary_markers.user_start.as_ref().clone();
        prefix.push(ProjectionSegment::Generated {
            tokens: Arc::new(open),
            identity: GeneratedIdentity {
                name: "compress_open".to_string(),
                position: 0,
            },
        });

        // Rolling anchor: the most recent sealed *summary* turn before these
        // children — the running thread of the conversation so far, itself
        // bounded (it is a prior summary, not raw history, so this stays O(1)).
        // Prepended as context so each half-summary is grounded in what came
        // before instead of confabulating a request/answer from a bare turn.
        //
        // BOTH halves of the prior leaf are carried — its question summary AND
        // its answer summary, separated. A stitched leaf is stored as
        // `[question][user_end][assistant_start][answer]`; reading only the
        // assistant half would amputate the running *question* thread, so the
        // original ask would survive exactly one turn and then vanish.
        let anchor_sep = self
            .tokenizer
            .encode("\n\n", false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        let prior_summary: Vec<u32> = {
            let first_child = children.iter().map(|t| t.0).min().unwrap_or(0);
            let view = conv.read();
            view.turn_indices(target.timeline)
                .filter(|idx| {
                    idx.0 < first_child
                        && view
                            .tree_meta_of(target.timeline, *idx)
                            .is_some_and(|m| m.kind.is_summary())
                })
                .max_by_key(|idx| idx.0)
                .map(|idx| {
                    let mut q = view
                        .turn_user_token_ids(target.timeline, idx)
                        .unwrap_or_default();
                    let a = view
                        .turn_assistant_token_ids(target.timeline, idx)
                        .unwrap_or_default();
                    if !q.is_empty() && !a.is_empty() {
                        q.extend_from_slice(&anchor_sep);
                    }
                    q.extend(a);
                    q
                })
                .unwrap_or_default()
        };
        if !prior_summary.is_empty() {
            prefix.push(ProjectionSegment::Generated {
                tokens: Arc::new(prior_summary),
                identity: GeneratedIdentity {
                    name: "compress_prior".to_string(),
                    position: 1,
                },
            });
        }

        // Close glue: the compression instruction followed by `user_end`.
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

        // Scratch slot bound to the dialogue timeline (so half-injection can
        // resolve the timeline from `slot_target`). Freed once its pass completes.
        let slot = self
            .create_sequence(conv.clone(), Some(target))
            .map_err(|e| format!("SubmitSummaryProbe: create slot: {e}"))?;

        match role {
            // The user half injects as sealed K/V (role-matched, zero re-prefill):
            // user content reused in a user-role slot is coherent. Only the glue
            // prefills, so the whole setup is cheap and runs synchronously.
            Role::User => {
                let mut segments = prefix;
                for &child in children {
                    segments.push(ProjectionSegment::Sealed(SealedKind::TurnHalf(
                        ResolvedTurn {
                            id: TurnId {
                                layer_id: target.layer,
                                group_id: target.group,
                                index: child,
                            },
                        },
                    )));
                }
                // Ground with the assistant half (text) so the user-request
                // summary sees the response it produced — a bare tool-result
                // turn is otherwise context-free and confabulates.
                let asst_ctx: Vec<u32> = {
                    let view = conv.read();
                    let mut acc: Vec<u32> = Vec::new();
                    for &c in children {
                        let half = view
                            .turn_assistant_token_ids(target.timeline, c)
                            .unwrap_or_default();
                        if half.is_empty() {
                            continue;
                        }
                        // Separate consecutive children's halves (same `\n\n` as
                        // the rolling anchor); single-child leaves are unchanged.
                        if !acc.is_empty() {
                            acc.extend_from_slice(&anchor_sep);
                        }
                        acc.extend(half);
                    }
                    acc
                };
                if !asst_ctx.is_empty() {
                    segments.push(ProjectionSegment::Generated {
                        tokens: Arc::new(asst_ctx),
                        identity: GeneratedIdentity {
                            name: "compress_asst_ctx".to_string(),
                            position: 2,
                        },
                    });
                }
                segments.push(ProjectionSegment::Generated {
                    tokens: Arc::new(close),
                    identity: GeneratedIdentity {
                        name: "compress_close".to_string(),
                        position: 3,
                    },
                });
                if let Err(e) =
                    self.setup_compression_decode(slot, &segments, role, max_tokens, job_id)
                {
                    self.free_summary_slot(slot);
                    return Err(e);
                }
            }
            // The assistant content is text-prefilled — its original K/V were
            // computed in the assistant region of each source turn (role- and
            // context-mismatched here), so they must be recomputed. Lay down only
            // the cheap `[Section | user_start]` prefix synchronously, then send
            // the (potentially large) `[content… close]` through the shared
            // prefill wave so it batches with the live turn instead of stalling
            // the loop. `finish_compression_pass_setup` runs in the
            // wave-completion hook (`promote_finished_prefills_to_decodes`).
            Role::Assistant => {
                if let Err(e) = self
                    .apply_projection(slot, BlockCount(0), &prefix)
                    .map_err(|e| format!("SubmitSummaryProbe: assemble Assistant prefix: {e}"))
                {
                    self.free_summary_slot(slot);
                    return Err(e);
                }
                let mut wave_tokens: Vec<u32> = Vec::new();
                {
                    let view = conv.read();
                    // Ground with the user half (text) first so the response
                    // summary sees the request it answers, then the assistant
                    // half it actually summarizes. Consecutive children's halves
                    // within each block are separated (same `\n\n` as the rolling
                    // anchor); a single-child leaf is unchanged.
                    let mut first = true;
                    for &child in children {
                        let half = view
                            .turn_user_token_ids(target.timeline, child)
                            .unwrap_or_default();
                        if half.is_empty() {
                            continue;
                        }
                        if !first {
                            wave_tokens.extend_from_slice(&anchor_sep);
                        }
                        first = false;
                        wave_tokens.extend(half);
                    }
                    let mut first = true;
                    for &child in children {
                        let half = view
                            .turn_assistant_token_ids(target.timeline, child)
                            .unwrap_or_default();
                        if half.is_empty() {
                            continue;
                        }
                        if !first {
                            wave_tokens.extend_from_slice(&anchor_sep);
                        }
                        first = false;
                        wave_tokens.extend(half);
                    }
                }
                wave_tokens.extend_from_slice(&close);

                // Private event sink, overwritten by the decode's own when
                // `finish_compression_pass_setup` runs; cleaned up with the slot.
                let (event_tx, event_rx) = crossbeam::channel::unbounded();
                self.compression_event_sinks.insert(slot, event_rx);
                self.prefill_queue.push_back(PrefillWork {
                    sequence_id: slot,
                    tokens: TokenBuffer::from(wave_tokens),
                    prefill_text: String::new(),
                    user_text: String::new(),
                    tags: Vec::new(),
                    content_bounds: TurnContentBounds::default(),
                    prefill_assistant_text: String::new(),
                    event_tx,
                    max_decode_tokens: 0,
                    sampling: SamplingConfig::compression(),
                    submitted_at: Instant::now(),
                    reprojection: None,
                    seal_action: SealAction::CompressionSetup {
                        job_id,
                        half: role,
                        max_tokens,
                    },
                    post_decode_tokens: TokenBuffer::default(),
                    projection_offsets: Vec::new(),
                    staged_composition: None,
                });
            }
            Role::System => {
                self.free_summary_slot(slot);
                return Err(
                    "SubmitSummaryProbe: compression pass role must be User or Assistant".into(),
                );
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
        role: Role,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        self.apply_projection(slot, BlockCount(0), segments)
            .map_err(|e| format!("SubmitSummaryProbe: assemble {role:?}-half: {e}"))?;
        self.finish_compression_pass_setup(slot, role, max_tokens, job_id)
    }

    /// Prefill `assistant_start`, sample the first token, and register the half's
    /// `CompressionPass` decode. Shared by the synchronous user-pass setup
    /// ([`Self::setup_compression_decode`]) and the wave-driven assistant pass,
    /// whose large content prefill completes on the wave before this tail runs.
    fn finish_compression_pass_setup(
        &mut self,
        slot: SequenceId,
        role: Role,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        let turn_start = Instant::now();
        // Prefill `assistant_start` to get the first-token logits and frame the
        // model to *answer* rather than continue the prompt.
        let asst_start = self.boundary_markers.assistant_start.as_ref().clone();
        let prefill_logits = self
            .run_prefill_with_shift(slot, &asst_start, 0)
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
            return Err("SubmitSummaryProbe: half produced no tokens (immediate EOS)".to_string());
        }

        // The decode writes its K/V into the slot, but that K/V is never kept —
        // only the generated tokens are captured (the stitch re-prefills them).
        // So there is no need to start the decode on a clean block boundary.

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
                seal_action: SealAction::CompressionPass { job_id, half: role },
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
                belief: crate::projection::PriorBelief::default(),
                prefill_tokens: TokenBuffer::default(),
                user_text: String::new(),
                tags: Vec::new(),
                content_bounds: TurnContentBounds::default(),
                in_tool_call: false,
            },
        );
        Ok(())
    }

    /// Complete one finished compression half-pass: capture its decoded tokens
    /// and deposit them into the job, then — if both halves are now present —
    /// stitch the node (re-prefilling the compressed exchange into coherent K/V),
    /// record it, and reply to the summariser. Frees each pass's scratch slot
    /// here; only the tokens are carried forward, so the decode's K/V can go.
    fn complete_compression_pass(
        &mut self,
        slot: SequenceId,
        job_id: u64,
        half: Role,
        generated: TokenBuffer,
    ) {
        // The wave decode terminates by pushing the final token (EOS or the
        // max_tokens-th) into `generated_tokens` WITHOUT forwarding it. Drop that
        // last token so the captured tokens are exactly the committed output. The
        // decode's K/V is discarded — the stitch re-prefills these tokens.
        let result: Result<CompressionPassResult, String> = (|| {
            let forwarded: Vec<u32> = generated
                .split_last()
                .map(|(_, rest)| rest.to_vec())
                .unwrap_or_default();
            if forwarded.is_empty() {
                return Err("SubmitSummaryProbe: half produced no forwarded tokens".to_string());
            }
            Ok(CompressionPassResult { tokens: forwarded })
        })();

        // The slot is done either way — free it. The delta's `ChunkGid` clones
        // keep its arena chunks alive past this free.
        self.free_summary_slot(slot);

        let Some(job) = self.compression_jobs.get_mut(&job_id) else {
            // The job was already torn down by the partner pass's failure.
            return;
        };
        match result {
            Ok(r) => match half {
                Role::User => job.user_result = Some(r),
                _ => job.assistant_result = Some(r),
            },
            Err(e) => {
                // One half failed → the whole node cannot stitch. Report the
                // soft error and tear down the job + the partner slot.
                let job = self.compression_jobs.remove(&job_id).unwrap();
                let _ = job.response_tx.send(Err(e));
                let partner = match half {
                    Role::User => job.assistant_slot,
                    _ => job.user_slot,
                };
                self.active_decodes.remove(&partner);
                // The partner may be the assistant half still mid-`CompressionSetup`
                // on the wave — purge its pending prefill before freeing the slot,
                // or a stale forward would hit the freed (and possibly reused) id.
                self.discard_pending_prefill(partner);
                self.free_summary_slot(partner);
                return;
            }
        }

        // Stitch only once both halves have landed.
        if self.compression_jobs[&job_id].user_result.is_none()
            || self.compression_jobs[&job_id].assistant_result.is_none()
        {
            return;
        }
        let job = self.compression_jobs.remove(&job_id).unwrap();
        if let Err(e) = self.enqueue_compression_turn(job_id, job) {
            tracing::warn!(
                target: "candle_conversation::summariser",
                err = %e,
                "compression turn enqueue failed",
            );
        }
    }

    /// Stitch a node's two decoded halves into ONE turn and enqueue it for the
    /// seal.
    ///
    /// The two passes generated their compressed question/answer as
    /// assistant-decode output, attending to the compression frame — so their
    /// decode-time K/V is role- and context-wrong for a stored turn (on
    /// re-projection the question half would inject assistant-region K/V into a
    /// user slot). Rather than keep it, frame the compressed exchange as a clean
    /// turn `[question][user_end][assistant_start][answer]` and **re-prefill it**
    /// so the question half computes user-role K/V and the answer half
    /// assistant-role K/V — coherent under any future re-projection (the same
    /// reproject-then-inject `insert_turn` does for a supplied turn). Bounds:
    /// user `[0, user_end_at)`, assistant `[asst_start_at, total)`.
    ///
    /// The re-prefill rides the **shared prefill wave** rather than a synchronous
    /// inline forward, so it batches with the live turn and other summaries
    /// instead of stalling the loop. The marker-framed turn is stashed in
    /// [`Scheduler::pending_compression_seals`] and enqueued as a `max_decode=0`
    /// [`SealAction::CompressionTurn`] prefill unit; `complete_compression_turn`
    /// snapshots the K/V, records the turn (`Role::Assistant`), and replies to
    /// the summariser once the wave finishes it.
    fn enqueue_compression_turn(&mut self, job_id: u64, job: CompressionJob) -> Result<(), String> {
        let CompressionJob {
            conversation,
            target,
            user_result,
            assistant_result,
            response_tx,
            ..
        } = job;
        let user = user_result.expect("user half present at stitch");
        let assistant = assistant_result.expect("assistant half present at stitch");
        self.seal_compression_turn(
            job_id,
            conversation,
            target,
            user.tokens,
            assistant.tokens,
            response_tx,
        )
    }

    /// Frame a compressed exchange (a user-half + assistant-half, already as
    /// token ids) as one clean marker-framed turn and enqueue it for the seal.
    /// Shared by the model-decode path ([`Self::enqueue_compression_turn`]) and
    /// the deterministic structural path ([`Self::handle_summary_probe`]).
    fn seal_compression_turn(
        &mut self,
        job_id: u64,
        conversation: Conversation,
        target: ProjectionTarget,
        user_tokens: Vec<u32>,
        assistant_tokens: Vec<u32>,
        response_tx: Sender<Result<TurnIndex, String>>,
    ) -> Result<(), String> {
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
        let content_bounds = TurnContentBounds::clamped(0, user_end_at, asst_start_at, token_count);

        // Decode both halves' display text. On failure reply to the summariser
        // before returning — otherwise `response_tx` drops and its `recv()` turns
        // a soft, retryable error into a hard one.
        let user_text = match self.tokenizer.decode(&user_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode user-half: {e}");
                let _ = response_tx.send(Err(e.clone()));
                return Err(e);
            }
        };
        let assistant_text = match self.tokenizer.decode(&assistant_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode assistant-half: {e}");
                let _ = response_tx.send(Err(e.clone()));
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
                let _ = response_tx.send(Err(e.clone()));
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
        self.pending_compression_seals.insert(
            job_id,
            PendingCompressionSeal {
                conversation,
                target,
                user_text,
                assistant_text,
                token_ids: token_ids.clone(),
                content_bounds,
                response_tx,
            },
        );
        self.prefill_queue.push_back(PrefillWork {
            sequence_id: slot,
            tokens: TokenBuffer::from(token_ids),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            content_bounds: TurnContentBounds::default(),
            prefill_assistant_text: String::new(),
            event_tx,
            max_decode_tokens: 0,
            sampling: SamplingConfig::compression(),
            submitted_at: Instant::now(),
            reprojection: None,
            seal_action: SealAction::CompressionTurn { job_id },
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
        });
        Ok(())
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
                let _ = pending
                    .response_tx
                    .send(Err(format!("SubmitSummaryProbe: reproject snapshot: {e}")));
                self.free_summary_slot(slot);
                return;
            }
        };
        // The slice holds RAII `ChunkGid` clones, so the K/V survives the free.
        self.free_summary_slot(slot);
        let block_end = sealed_gpu.first().map(|s| s.chunks.len()).unwrap_or(0);
        let token_count = pending.token_ids.len();

        let timeline = pending.target.timeline;
        let conversation = pending.conversation;
        let response_tx = pending.response_tx;
        // Kept for the post-injection log + token persistence below — the rest
        // move into the write.
        let summary_question = pending.user_text.clone();
        let summary_answer = pending.assistant_text.clone();
        let persist_token_ids = pending.token_ids.clone();
        let write = TurnPartWrite {
            user_text: pending.user_text,
            assistant_text: pending.assistant_text,
            token_ids: TokenBuffer::from(pending.token_ids),
            token_count,
            content_bounds: pending.content_bounds,
            // Compression/summary turns are never a gather target — untagged.
            tags: Vec::new(),
            block_start: 0,
            block_end: block_end as u64,
            sealed_gpu: Some(Arc::new(sealed_gpu)),
        };
        let idx = match conversation
            .record_turn(timeline, Role::Assistant, write, |seqs| Ok(seqs.to_vec()))
        {
            Ok(idx) => idx,
            Err(e) => {
                let _ = response_tx.send(Err(format!(
                    "SubmitSummaryProbe: record compressed turn: {e}"
                )));
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

    /// Tear down a compression node whose setup or decode failed: reply to the
    /// summariser with the error and reclaim both half-passes' slots (whichever
    /// of them is live — one may still be decoding, the other not yet
    /// registered). Used when the wave-driven assistant setup can't complete.
    fn abort_compression_job(&mut self, job_id: u64, err: String) {
        if let Some(job) = self.compression_jobs.remove(&job_id) {
            let _ = job.response_tx.send(Err(err));
            for slot in [job.user_slot, job.assistant_slot] {
                self.active_decodes.remove(&slot);
                self.discard_pending_prefill(slot);
                self.free_summary_slot(slot);
            }
        }
    }

    /// Drop any not-yet-run prefill for `slot` — queued or in-flight on the wave
    /// — so a torn-down compression slot can never be forwarded after it is
    /// freed (the assistant half's `CompressionSetup` content prefill may still
    /// be pending when its partner fails). `free_summary_slot` clears the rest of
    /// the per-slot state (incl. the event sink).
    fn discard_pending_prefill(&mut self, slot: SequenceId) {
        self.prefill_queue.retain(|w| w.sequence_id != slot);
        self.active_prefills.retain(|p| p.work.sequence_id != slot);
    }

    /// Free a compression scratch slot and its per-slot bookkeeping. Called on
    /// every path out of a compression pass (success or error). The decoded
    /// delta's RAII `ChunkGid`s keep the arena chunks alive after this returns.
    fn free_summary_slot(&mut self, slot: SequenceId) {
        let _ = self.session.free_sequence(slot.0);
        self.slot_conversations.remove(&slot);
        self.slot_targets.remove(&slot);
        self.sampling_states.remove(&slot);
        self.slot_projection_state.remove(&slot);
        self.compression_event_sinks.remove(&slot);
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
        let state = self.slot_projection_state.entry(parent_id).or_default();

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
        let turn_keys: Vec<(GroupId, TurnIndex, TimelineId)> = {
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
            .zip(per_layer_token_count)
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

        // KV-zero check: scan the freshly-injected projected context — catches a
        // corrupted turn's KV being reused as provenance context for a new turn.
        #[cfg(feature = "kv-zero-check")]
        {
            let layers: Vec<usize> = (0..self.session.num_layers()).collect();
            let n_real = self.session.sequence_offset(new_seq_idx).unwrap_or(0);
            let layout = self.session.provenance_chunk_layout(new_seq_idx, n_real);
            if let Ok(dump) =
                self.session
                    .gather_r16_kv_provenance_layers(new_seq_idx, &layers, None)
            {
                kv_zero_check::scan_gathered(
                    "projection-inject",
                    new_seq_idx,
                    &dump,
                    &layout,
                    self.session.n_kv_head(),
                    self.session.head_dim(),
                    n_real, // pure projected context — everything is "prefix"
                );
            }
        }

        Ok(new_seq_id)
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
            // Drop the projection-span anchor for this finished decode.
            self.reproj_anchor.remove(&seq_id);
            if let Some(state) = self.active_decodes.remove(&seq_id) {
                // Compression half-passes complete through the job registry,
                // not the substrate seal path: slice the decoded delta, deposit
                // it, and stitch the node once both halves land. No view to
                // finalize, no Done event — the summariser blocks on the job's
                // `response_tx`, fired by `complete_compression_pass`.
                if let SealAction::CompressionPass { job_id, half } = state.seal_action {
                    self.complete_compression_pass(seq_id, job_id, half, state.generated_tokens);
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
                            Some(TurnContent {
                                role: Role::Assistant,
                                tags: state.tags.clone(),
                                user_text: state.user_text.clone(),
                                assistant_text,
                                token_ids: TokenBuffer::from(full_tokens),
                                content_bounds: state.content_bounds,
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
    /// (C0 / Q8 / Q8) and the dialogue turn policy: **C3 with K
    /// overridden to Q8_KS**, V uses adaptive level-3 selection.
    /// `_turn_policy` is taken to keep the call sites future-proof
    /// for a per-engine override knob; today it's ignored.
    fn section_compression_policy_member(
        _turn_policy: &candle_nn::kv_cache::CompressionPolicy,
    ) -> candle_nn::kv_cache::CompressionPolicy {
        candle_nn::kv_cache::CompressionPolicy::new(3)
            .with_override_k_quant(Some(QuantFormat::Q8_KS))
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
        // Snapshot per-section native hot + residence + which policy
        // applies under a brief read lock; the heavy GPU work runs
        // unlocked.  Pending list is partitioned into (residence,
        // sealed, in_collection); the per-layer loop below groups by
        // in_collection so each policy gets its own batched launch.
        let pending: Vec<(SectionId, ResidenceIndex, Vec<SealedSequence>, bool)> = {
            let view = conversation.read();
            self.pending_section_quantize
                .drain(..)
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
        let wide_sigs = self.gather_wide_sigs(seal_slot, (block_from, block_to));

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
        match seal_action {
            SealAction::Turn => {
                let target = seal_target.ok_or_else(|| {
                    ConversationError::Channel("SealAction::Turn missing seal_target".into())
                })?;
                let TurnContent {
                    role,
                    tags,
                    user_text,
                    assistant_text,
                    token_ids,
                    content_bounds,
                } = turn_content.unwrap_or_default();
                // Re-clamp the content boundaries to the sealed token count
                // so `window_sealed_tokens` can never window past the turn's
                // own grid.  The assistant content end is the total token
                // count; the three boundaries stay monotonic.
                let content_bounds = TurnContentBounds::clamped(
                    content_bounds.user_start as usize,
                    content_bounds.user_end as usize,
                    content_bounds.asst_start as usize,
                    turn_token_count,
                );
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

                // The turn is sealed as one indivisible K/V block but
                // the substrate stores the user and assistant text
                // separately — clean strings the sidebar can render
                // without re-tokenising at read time.
                let write = TurnPartWrite {
                    user_text,
                    assistant_text,
                    token_ids,
                    token_count: turn_token_count,
                    content_bounds,
                    tags,
                    block_start: block_from as u64,
                    block_end: block_to as u64,
                    sealed_gpu: Some(Arc::new(delta_gpu)),
                };
                let idx = conversation
                    .record_turn(target.timeline, role, write, |seqs| Ok(seqs.to_vec()))
                    .map_err(ConversationError::Model)?;

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
                // Wake the summariser thread (`docs/infinite_conversations.md`
                // §4 step ③) so the freshly-pending Normal turn gets
                // absorbed into the AVL summary tree on its next pass
                // instead of waiting up to 250 ms for the periodic tick.
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
            SealAction::CompressionSetup { .. } => {
                unreachable!(
                    "compression setup finishes in promote_finished_prefills_to_decodes, not here"
                )
            }
        }

        Ok(Some(SealResult {
            block_count,
            block_from,
            block_to,
            turn_token_count,
            chunk_size,
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
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        conversation.read().turn_sealed_of(timeline, index)
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
        // Incoming cold-load VRAM footprint (≈ the on-disk record size).
        let cold_bytes: u64 = {
            let view = conversation.read();
            let plan = view.snapshot_promotion_state(sections, turns);
            plan.cold_to_hot
                .iter()
                .flat_map(|c| c.cold.iter())
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.record_len)
                .sum()
        };
        if cold_bytes == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        let device = self.session.device().clone();
        let avail = match candle_nn::kv_cache::vram_budget_available(&device) {
            // Budget unknown (non-CUDA / query failure) — don't evict on a guess.
            None => return crate::substrate::EvictionReport { count: 0, bytes: 0 },
            Some(a) => a as u64,
        };
        if cold_bytes <= avail {
            // Ample VRAM — keep the whole working set hot.
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        // Tight: reclaim partial-arena free space first, then re-measure.
        let _ = self.session.compact();
        let avail = candle_nn::kv_cache::vram_budget_available(&device)
            .map(|a| a as u64)
            .unwrap_or(avail);
        let needed = cold_bytes.saturating_sub(avail);
        if needed == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        conversation
            .write()
            .evict_hot_to_free(sections, turns, needed)
    }

    /// Prepare phase of reprojection: provenance scan + projection + tier elevate +
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
        let prior_belief = self
            .active_decodes
            .get(&view_id)
            .map(|s| s.belief.clone())
            .unwrap_or_default();

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
            return Ok(None);
        }

        let max_probe = policy.max_probe_tokens.max(1);
        let window = max_probe.min(view_offset);
        if window == 0 {
            return Ok(None);
        }
        let probe_lo = view_offset - window; // inclusive
        let probe_hi = view_offset; // exclusive

        // Wall-clock start of the whole reproject — drives `total_ms` so the log
        // reports the real end-to-end cost, not a sum of (partly overlapping)
        // phase fields.
        let t_repro = Instant::now();

        // 2. Block range covering only the probe window.
        let t_probe = Instant::now();
        let block_lo = probe_lo / self.chunk_size;
        let block_hi = probe_hi.div_ceil(self.chunk_size);

        // 3. Gather the live wide-Q probe over the window — folded per-token
        //    sign(Q). A gather covering no real tokens means nothing to score.
        let probe = self.gather_wide_sigs(view_id, (block_lo, block_hi));
        if probe.is_empty() {
            return Ok(None);
        }
        let probe_ms = t_probe.elapsed().as_millis() as u64;
        record_phase(t_probe, "reproject_probe_extract");

        // 4. Wide-Q belief scoring: scan the probe against each belief-driven
        //    collection's tag-scoped gallery of past turns→selected-section,
        //    producing the per-section scores the belief policy selects from in
        //    `project()`. No provenance scan, no persisted scores — projection-local.
        let t_scan = Instant::now();
        let schema = policy.projection.schema();
        let projection_scores = match schema.layers.iter().find(|l| l.id == policy.target.layer) {
            Some(layer) => policy.substrate.score_belief_collections(layer, &probe),
            None => crate::substrate::ProjectionScores::new(),
        };

        let scan_ms = t_scan.elapsed().as_millis() as u64;
        record_phase(t_scan, "reproject_belief_scan");

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
                        let g = rt.group();
                        let t = rt.index();
                        let Some(timeline) = view.timelines_for_group(g).next() else {
                            continue;
                        };
                        if view.turn_tier_state(timeline, t).is_some() {
                            turn_keys_for_elevate.push(TurnKey::new(timeline, t));
                            segments.push(seg.clone());
                        }
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
            let composition = crate::projection::from_projection(
                &projection.segments,
                policy.projection.schema(),
                &view,
                &projection.selection_scores,
                view.total_token_count(policy.target.timeline) as u32,
                0,
                0,
                0.0,
            );
            (sections, segments, composition)
        };
        let project_ms = t_project.elapsed().as_millis() as u64;
        record_phase(t_project, "reproject_project");

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

        // Select-promote (cold → warm → hot) the new working set
        // before re-applying. Same shape as the SubmitTurn path: a
        // single batched scatter per layer + per-item cold-recover
        // turns the per-unit `ensure_*_hot` calls inside
        // `apply_projection` into hot-hit no-ops.
        let t_elevate = Instant::now();
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
            // Budget-aware evict before elevate: keep the working set hot on a
            // big GPU, evicting only enough (oldest first) to fit the incoming
            // cold-load within the accurate VRAM budget (see SubmitTurn site).
            let evicted = self.evict_to_fit_incoming(
                &policy.substrate,
                &projected_sections,
                &turn_keys_for_elevate,
            );
            if evicted.count > 0 {
                tracing::trace!(
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
                    tracing::trace!(
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
        decode_state.belief =
            crate::projection::PriorBelief::from_selection(&composition.selection);

        // Close the decode span that ran under the *previous* projection and
        // emit it as a timeline event: `[anchor.token, now]` tokens at the
        // measured t/s, paired with the composition this reprojection selected.
        // The anchor is keyed by view id and migrated to `new_view_id` below
        // (like the decode state), so the next reproject / final seal measures
        // from here. First reproject has no anchor → measures from decode start.
        let repro_now = Instant::now();
        let repro_gen = decode_state.generated_tokens.len() as u32;
        {
            // Span start = the previous projection's end, carried on the decode state
            // (which migrates correctly across view swaps) — NOT the view-keyed anchor
            // map, whose migrated `from_token` was doubling. The anchor is still used
            // only for the wall-clock `since` (its `Instant` is fine).
            let from_token = decode_state.last_projection_end;
            let since = self
                .reproj_anchor
                .remove(&view_id)
                .map(|(_, t)| t)
                .unwrap_or(decode_state.decode_start);
            let seconds = repro_now.duration_since(since).as_secs_f64();
            let span = repro_gen.saturating_sub(from_token);
            let event = crate::projection::ProjectionEvent {
                start_token: from_token,
                end_token: repro_gen,
                seconds,
                tokens_per_second: if seconds > 0.0 {
                    span as f64 / seconds
                } else {
                    0.0
                },
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
        // Migrate the projection-span anchor onto the new view id so the next
        // reproject / final seal measures its span from this reprojection.
        self.reproj_anchor
            .insert(new_view_id, (repro_gen, repro_now));
        // True end-to-end wall-clock of the whole reproject (prepare → wave →
        // complete), so the phase fields below — which are individually disjoint
        // but separated by the shared glue wave — can be read against a real
        // total instead of summed by eye.
        let total_ms = t_repro.elapsed().as_millis() as u64;

        tracing::info!(
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
// Scheduler dispatch unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
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

    // —— Recording model ——————————————————————————————————————————————————————

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
            512, // max_prefill_pass_tokens
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
        );
        (scheduler, tx)
    }

    // —— Tests ————————————————————————————————————————————————————————————————

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

    // —— view-creation tests —————————————————————————————————————————————————

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
}
