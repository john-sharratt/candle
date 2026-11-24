//! The requests the scheduler's wave thread accepts from caller threads, and the
//! dispatch that answers them.
//!
//! Each request either carries a reply channel or has a fire-and-forget effect.
//! A `SubmitTurn` runs its projection, carves its view and queues its prefill. A
//! request that arrives while its parent has a turn running is answered at the
//! turn boundary rather than locking the parent. Queued submissions are drained
//! ahead of the work already in flight, so a turn submitted on a parent right
//! after its last one finished cannot start ahead of the forks waiting on that
//! gap. The prefill backlog is the sum of the queued and in-flight units that
//! have not yet run, and it is read by the large-batch trigger.

use super::*;

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
        /// The slot this one is forking from, when it is a fork of a **live**
        /// conversation.
        ///
        /// A fork's spliced K/V covers the parent's history as sealed, and its
        /// recurrent state has to correspond to the same token stream. When the
        /// parent is resident the state comes straight across device-to-device;
        /// when it is not (daemon restart, parent already freed) the slot falls
        /// back to the snapshot read, which is what `None` selects. This is a
        /// real hit/miss, not a mode.
        parent: Option<SequenceId>,
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

    /// Fork the live slot `parent` into an ephemeral one, between its turns.
    ///
    /// The fork borrows the parent's K/V zero-copy, copies its per-sequence
    /// model state and clones its projection caches, so a turn run on it
    /// projects over the prefix the parent already placed instead of rebuilding
    /// it. Like [`Self::NewEphemeralSequence`] it is marked ephemeral: its turn
    /// seals nothing. While a turn runs on `parent` the request is parked and
    /// answered at the turn boundary; it fails if `parent` is freed first.
    /// Used by `Sequence::ask_unsealed`.
    ///
    /// The fork decodes at `band`, not at its parent's layer priority: a
    /// question put beside a conversation is the asker's work, and how urgent
    /// it is is the asker's to say.
    ForkEphemeralSequence {
        parent: SequenceId,
        band: DecodePriority,
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
    ///
    /// Reached through [`crate::ConversationEngine::resume_conversation_with_projection`],
    /// which is what a mind that never stops living opens its conversation with
    /// after a restart.
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
        /// Present when this submission is a TURN GROUP: many prefilled turns,
        /// each prefilled on its own view of the projected parent and sealed as
        /// a turn of its own ([`Scheduler::start_turn_group`]). Each entry's
        /// `token_ids` is the whole of what that case prefills, and
        /// `prefill_tokens` is then empty. `None` is the ordinary one-turn
        /// submission, whose turn is the slot's tail.
        seal_group: Option<Vec<TurnContent>>,
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
        /// laid down by the first turn from its frame
        /// ([`ProjectionInputs::frame`]), so the slot is non-empty from
        /// turn 2; see the handler for the gated-section trade-off.  Used by
        /// append-only utility ingests (e.g. `code_reading`) where
        /// re-projecting the whole trunk every turn is both unnecessary
        /// and O(n²) — see `zend::code_read`.
        disable_reprojection: bool,
        /// Tool-call stencils that may fire during this turn's decode, keyed by
        /// their trigger token (e.g. `<tool_call>`).  An empty registry means no
        /// constrained decoding — the turn free-decodes.
        triggers: Arc<TriggerRegistry>,
        /// A grammar this turn *begins inside* — see [`TurnOptions::turn_grammar`].
        /// Its opening scaffold is already appended to `tokens`; the driver is
        /// armed at the prefill boundary so the first sampled token is masked.
        turn_grammar: Option<TurnGrammar>,
        /// Whether the schema asked for repetition penalties to be lifted inside
        /// tool calls — see
        /// [`crate::projection::Schema::free_tool_calls_from_penalties`].
        free_tool_calls_from_penalties: bool,
        /// The reply this turn decodes in place of sampling — a recorded turn
        /// replayed ([`crate::TurnOptions::recorded_turn`]). Committed one id
        /// per step, then the end of turn or free decoding. `None` = an ordinary
        /// sampled decode.
        recorded_reply: Option<RecordedReply>,
        /// Spans of `prefill_tokens` whose recurrent state is captured or
        /// replayed — see [`crate::TurnOptions::recurrent_spans`].
        recurrent_spans: Vec<TokenSpan>,
        /// The images in `prefill_tokens`: where each one's pads are, and the
        /// rows they name to the engine ([`crate::vision::engine_ids`]).
        image_spans: Vec<ImageSpan>,
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

    /// Choose the LoRA adapter this sequence decodes through, by name, or
    /// `None` for the base model.
    ///
    /// Sent before the sequence's first turn. Changing it mid-conversation is
    /// refused by the handler: the KV already written was produced by the
    /// projections in force when it was written, and continuing with different
    /// ones would attend those keys with queries from another model.
    SetSequenceAdapter {
        sequence_id: SequenceId,
        adapter: Option<Arc<str>>,
        response_tx: Sender<Result<(), ConversationError>>,
    },

    /// Run one belief-scan launch on the gallery arena, for a scan running on
    /// another thread ([`RemoteArena`]).
    ///
    /// The arena has exactly one scan thread — a re-seal frees a turn's pages
    /// on the grounds that nobody else can be holding a scan pin on them — so a
    /// scan elsewhere sends its launches here and does the rest of its work,
    /// the substrate reads and the normalization, on its own thread. A seal
    /// scan held the wave thread ~575 ms a turn, all but ~50 ms of it host
    /// work (BlackWorld, 2026-11-16).
    ///
    /// A `background` launch — a chunk-exclusion learner's — is not run as it
    /// arrives but queued with the loop's other background scans, one a pass
    /// (`norm_warm`), so a run of them never stands between two waves.
    ArenaScan {
        scan: HeldScan,
        response_tx: Sender<ScanReply>,
        background: bool,
    },

    /// The gallery arena's fold-group count, `None` when there is no arena —
    /// what a [`RemoteArena`] lays its segments out by.
    ArenaGroups { response_tx: Sender<Option<usize>> },

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
        /// The recurrent state at the section's first token — the state the
        /// prefix leaves behind. Staged on the scratch slot before the prefill,
        /// so the section's recurrence continues from its prefix rather than
        /// starting from zero, and with it every layer the recurrence feeds,
        /// which is what the section's K/V is computed from. `None` for a model
        /// that carries no recurrent state.
        start: Option<SharedState>,
        /// The band its forwards run at. A schema's sections are background
        /// ingest, `Low`. A section a conversation submits for itself — a
        /// journal entry — is read by that conversation's next turn, so it
        /// runs at the band the conversation decodes at: a `High` one rides
        /// `High` groups and pauses an ingest creep, as a prompt does.
        band: DecodePriority,
        response_tx: Sender<Result<SealResult, ConversationError>>,
    },

    /// Install a recurrent state onto an existing slot — the branch-checkpoint
    /// restore. The model stages it (`stage_recurrent`): the store is built
    /// when the slot's state is first wanted, not here.
    ///
    /// The conversation-resume path installs from a timeline snapshot inside
    /// `create_sequence`, before any wave. A branch checkpoint cannot: the slot
    /// has to be primed with its prompt K/V first, and priming is its own
    /// request. So this arrives after, on a slot that exists and has not yet
    /// run a turn.
    ///
    /// Installs onto EVERY slot in `sequence_ids` from one shared `layers`.
    /// Conversations opened on the same prompt branch install byte-identical
    /// state, and the request's cost is dominated by the queue wait before the
    /// scheduler drains it — measured at 157 ms of a 175 ms round-trip, against
    /// 17 ms of actual device work per slot. Paying that wait once per branch
    /// instead of once per conversation is why this takes a list.
    ///
    /// Replies `Ok(false)` when the model carries no recurrent state.
    InstallRecurrentState {
        sequence_ids: Vec<SequenceId>,
        state: SharedState,
        response_tx: Sender<Result<bool, ConversationError>>,
    },

    /// Re-run a slot's create-time recurrent seeding, now that its lineage is
    /// known — see [`Substrate::FORKED_FROM_KEY`].
    ///
    /// `create_sequence` seeds from the timeline's own snapshot, and a fork is
    /// minted on a FRESH timeline that has none. Its memory is its parent's,
    /// but the `forked_from` pointer is recorded by the caller *after* the fork
    /// returns — by which time the slot has already been seeded from nothing.
    /// Rather than reorder the fork (the child's timeline does not exist until
    /// it returns), the caller records the pointer and then asks for the seed
    /// again. Like [`Self::InstallRecurrentState`], this lands on a slot that
    /// exists and has not yet run a turn.
    ///
    /// Replies `Ok(false)` when the model carries no recurrent state, or no
    /// ancestor has a snapshot to seed from.
    SeedRecurrentFromLineage {
        sequence_id: SequenceId,
        response_tx: Sender<Result<bool, ConversationError>>,
    },

    /// Advance a conversation's recurrent memory over tokens whose K/V it
    /// already holds — the catch-up after a splice adopts turns by reference.
    ///
    /// Replies `Ok(false)` when the model carries no recurrent memory, so the
    /// splice can ask unconditionally.
    MemoryCatchUp {
        sequence_id: SequenceId,
        tokens: TokenBuffer,
        /// The first adopted turn index. Turns before it are this
        /// conversation's own history and are injected onto the replay slot as
        /// context; turns from it on are the span being replayed.
        adopted_from: u32,
        /// The images in `tokens`, whose pads replay as their rows.
        image_spans: Vec<ImageSpan>,
        response_tx: Sender<Result<bool, ConversationError>>,
    },

    /// Prefill an ordered token stream on a throwaway slot and hand back the
    /// recurrent state it produced — the **branch checkpoint pass**.
    ///
    /// **A genuine prefill, not an injection, and that is the whole point.**
    /// Section ingest Arc-injects its prefix and prefills only the new
    /// section's own content, which is right for K/V — the prefix's K/V already
    /// exists, so copying it is free and the forward attends to real preceding
    /// context. A recurrence cannot be Arc-injected: the state that a prefix
    /// would have produced exists nowhere until the tokens actually pass
    /// through the model, in order. So this request runs the whole branch.
    ///
    /// Replies `Ok(None)` when the model carries no recurrent state, so the
    /// caller can ask unconditionally and pay nothing on a plain transformer.
    BranchCheckpointPass {
        /// The state the stream continues from: `None` for a stream that starts
        /// the prompt, the state at its first token for one that continues it.
        start: Option<SharedState>,
        /// The ordered token stream — every sealed section of the system prompt
        /// under one selector assignment, concatenated, or the sections a
        /// continuation runs on from `start`.
        tokens: TokenBuffer,
        response_tx: Sender<Result<Option<ExportedState>, ConversationError>>,
    },

    /// Recover a previously-persisted section directly from the redo
    /// log instead of running a fresh prefill.  Used when an ingest
    /// caller has computed a section's content-addressed stream id
    /// and found durable chunks for it.  The scheduler checks those
    /// chunks form a whole grid over the session's KV backings,
    /// installs the section as a cold marker (the elevate path lifts
    /// it when a projection needs it), reinstalls its index page for a
    /// model that keeps per-position state, and replies with the
    /// section's chunks per layer.
    ///
    /// An `Err` means "prefill it instead": the caller issues a normal
    /// `IngestSection`.
    RestoreSection {
        /// Workspace conversation the section lands in.  Sections are
        /// shared substrate state — every conversation in the
        /// workspace sees the same map — but the handle is needed so
        /// the scheduler can call `restore_section` + `recover_*` on
        /// the right `Arc<SubstrateLock>`.
        conversation: Conversation,
        section_id: SectionId,
        stream_id: StreamId,
        /// Pre-tokenised section content — the same byte sequence
        /// the original prefill used.  Reused verbatim so we don't
        /// need to read the `Tokens` record back from disk just to
        /// repopulate `SectionEntryData::tokens`.
        tokens: TokenBuffer,
        /// The section's chunks per layer once restored.
        response_tx: Sender<Result<usize, ConversationError>>,
    },

    /// Remove transient sections — a throwaway conversation's frame — once
    /// their last user has finished with them.
    ///
    /// On the wave thread because the scheduler holds per-section state of its
    /// own (the index page, the name, a queued quantize) that has to go with the
    /// substrate entry, and because between waves nothing is attending it.
    /// Fire-and-forget: the sender has already let go, and a section that is not
    /// transient is left alone — see
    /// [`crate::substrate::Substrate::retire_section`].
    RetireSections {
        conversation: Conversation,
        sections: Vec<SectionId>,
    },

    /// Set the band a slot decodes at in place of its target layer's, from
    /// its next admission on; `None` returns it to the band its conversation
    /// records for itself, or with none recorded to its layer's.
    ///
    /// A conversation's band is otherwise fixed when its slot is made. A
    /// character carrying out what a person is waiting on is the same
    /// conversation as the one that idles: what changes is who waits on its
    /// next turn. Fire-and-forget.
    SetBand {
        sequence_id: SequenceId,
        band: Option<DecodePriority>,
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
    /// interactive projection probe (`POST /v1/substrate/zend/project`). Fire after the
    /// probe turn's `Done`; returns the belief-domain [`WideQSig`] window
    /// `score_beliefs` consumes (empty if the turn produced none). Nothing was
    /// written to the substrate.
    ProbeWideSigs {
        sequence_id: SequenceId,
        response_tx: Sender<Result<Vec<WideQSig>, ConversationError>>,
    },

    /// Read a **live** sequence's recurrent memory, without sealing.
    ///
    /// Test observability, and it exists because the alternative is worse than
    /// inconvenient. Memory is otherwise visible only through the record a seal
    /// writes, so any behavioural test has to advance the conversation a whole
    /// turn to look at it — which changes the thing it is measuring — and a
    /// leak test (does a freed slot still hold memory?) cannot be written at
    /// all, because a freed slot never seals again.
    ///
    /// Reply is `None` for a model that carries no such memory, and for a slot
    /// that has none yet.
    ReadRecurrentMemory {
        sequence_id: SequenceId,
        response_tx: Sender<Result<Option<Vec<ExportedLayerState>>, ConversationError>>,
    },

    /// Digest a sealed turn's **hot K/V content** — per layer, per chunk: the
    /// window offset, every band's format tag, palette maps, outer scales, and
    /// the gathered arena bytes. Test observability for the continuity oracles:
    /// after every coarser instrument (layout blocks, selection items,
    /// materialized glue, recurrent state) compared equal across a live/resumed
    /// divergence, the K/V bytes the engine actually serves are the last place
    /// the difference can live, and nothing else can see them.
    ///
    /// Reply is `None` when the turn holds no hot sealed K/V (not yet
    /// materialised, or no such turn).
    #[cfg(any(test, feature = "test-helpers"))]
    ReadTurnKvDigest {
        sequence_id: SequenceId,
        timeline: TimelineId,
        index: TurnIndex,
        response_tx: Sender<Result<Option<ContentHash>, ConversationError>>,
    },

    /// How many sequences the model currently holds recurrent memory for.
    ///
    /// The leak gauge. A slot's memory is dropped on free, and a slot id is a
    /// recycled pool index, so a leak here is not merely wasted VRAM — it is the
    /// next conversation on that id inheriting a stranger's memory.
    #[cfg(any(test, feature = "test-helpers"))]
    CountRecurrentMemories { response_tx: Sender<usize> },

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

    /// Re-run substrate reconstruction on the scheduler thread — used after a
    /// compaction rewrites the redo log, so the scheduler-side view (KV residence
    /// + offsets) is rebuilt from the new log. Marks `status` finished when done.
    ReconstructSubstrate {
        conversation: Conversation,
        status: Arc<SubstrateReloadStatus>,
    },

    /// Attach another substrate to this engine: register its persistence
    /// thread's trigger under attachment `id`, then reload the substrate from
    /// its redo log, marking `status` finished when done
    /// (`docs/shared_model_host.md` §3.2).
    AttachSubstrate {
        id: u64,
        trigger: PersistenceTrigger,
        conversation: Conversation,
        status: Arc<SubstrateReloadStatus>,
    },

    /// Detach attachment `id`: free every slot bound to `conversation`'s
    /// substrate, forget its timelines' scheduler state, and drop its
    /// persistence trigger. Replies once nothing on this thread refers to the
    /// substrate, so the caller can stop its persistence thread and flush.
    DetachSubstrate {
        id: u64,
        conversation: Conversation,
        response_tx: Sender<()>,
    },

    /// Warm the belief-driven section collections' backgrounds from their own
    /// tag-scoped corpus — on this thread, because the GPU gallery arena that
    /// scores the corpus in batch lives here. Replies with what was warmed.
    WarmCollectionNormalization {
        conversation: Conversation,
        schema: Box<Schema>,
        response_tx: Sender<CollectionWarm>,
    },

    /// Queue the dialogue normalization replay ([`Conversation::take_warm_replay`])
    /// for `target` when it is armed, so a load re-learns the score backgrounds
    /// without waiting for a first reprojection to ask. Scored in the gaps a
    /// dialogue leaves, one turn per loop iteration (`norm_warm`); no reply.
    WarmDialogueNormalization {
        conversation: Conversation,
        projection: Arc<Builder>,
        target: ProjectionTarget,
        max_probe_tokens: usize,
    },

    /// Decode and place on this thread's GPU gallery arena everything a
    /// projection's belief scan reads ([`Conversation::preload_galleries`]),
    /// so the first projection after a start finds it all resident.
    PreloadGalleries {
        conversation: Conversation,
        schema: Box<Schema>,
        response_tx: Sender<GalleryPreload>,
    },

    /// Warm one slice of an ingest group's per-file backgrounds on this thread's
    /// GPU gallery arena ([`Conversation::warm_ingest_timelines`]). Sent a slice
    /// at a time by [`crate::IngestWarmer`], and scored at ingest priority — in
    /// the gaps a dialogue leaves, one slice per loop iteration (`norm_warm`).
    /// Replies with the timelines warmed, or `None` when this scheduler has no
    /// arena and the caller must score on the host.
    WarmIngestSlice {
        conversation: Conversation,
        schema: Arc<Schema>,
        layer: LayerId,
        group: GroupId,
        timelines: Vec<TimelineId>,
        response_tx: Sender<Option<usize>>,
    },

    /// Score a batch of probes raw against every file of a belief group on
    /// this thread's GPU gallery arena ([`Conversation::scan_group_votes`]),
    /// at the same priority as an ingest slice. Replies with the votes, or
    /// `None` when this scheduler has no arena.
    ScanGroupVotes {
        conversation: Conversation,
        group: GroupId,
        calibrations: bool,
        masked: HashSet<StreamId>,
        excluded: HashMap<StreamId, Vec<Range<usize>>>,
        probes: Vec<Vec<WideQSig>>,
        /// The fold groups to scan; `None` for every group.
        groups: Option<GroupSet>,
        /// Which of the turns' tokens the card holds for the scan.
        keep: Keep,
        response_tx: Sender<Option<candle::Result<GroupVotes>>>,
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

    /// Nothing to do — wake the loop so it looks around.
    ///
    /// **Guest work needs this and nothing else does.** An idle scheduler parks
    /// in `rx.recv()`, and everything else that produces work arrives *through*
    /// that channel, so the block is also the wake. The guest queue is the one
    /// producer that is not a `SchedulerRequest`: a job submitted to an idle
    /// daemon sat in the queue forever, because the loop was parked below the
    /// point that polls it and nothing was ever going to arrive to unpark it.
    ///
    /// Handled as a no-op that returns to the top of the loop, where the guest
    /// poll is. Sent after the job is queued, so the loop either sees the work
    /// on its own or is woken to it — there is no ordering in which both miss.
    Wake,

    /// Settle the device: sweep the empty arenas, pack the pools and let the
    /// weight side grow back into what freed — see `Scheduler::settle_device`.
    /// Replies with the partition as it then stands, once the scheduler has
    /// fallen idle and run the settle: work in flight is finished first.
    ///
    /// Sent once a caller knows the work it put on the card has left it — an
    /// evicted timeline whose turns no longer hold a device copy — so the next
    /// request starts on ground the last one gave back, rather than finding the
    /// housekeeping still owed inside its own first forwards.
    Settle { response_tx: Sender<SettleReport> },

    /// Shut down the scheduler.
    Shutdown,
}

impl SchedulerRequest {
    /// The profile span a request's handling is recorded under — one per kind,
    /// so a drain that costs the decode loop half a second says which request
    /// did it.
    pub(super) fn span_name(&self) -> &'static str {
        match self {
            Self::NewSequence { .. } => "req:new_sequence",
            Self::NewEphemeralSequence { .. } => "req:new_ephemeral_sequence",
            Self::ForkEphemeralSequence { .. } => "req:fork_ephemeral_sequence",
            Self::ResumeSequence { .. } => "req:resume_sequence",
            Self::SubmitTurn { .. } => "req:submit_turn",
            Self::FreeSequence { .. } => "req:free_sequence",
            Self::ResetSequence { .. } => "req:reset_sequence",
            Self::SetSequenceAdapter { .. } => "req:set_sequence_adapter",
            Self::ArenaScan { .. } => "req:arena_scan",
            Self::ArenaGroups { .. } => "req:arena_groups",
            Self::IngestSection { .. } => "req:ingest_section",
            Self::InstallRecurrentState { .. } => "req:install_recurrent_state",
            Self::SeedRecurrentFromLineage { .. } => "req:seed_recurrent_from_lineage",
            Self::MemoryCatchUp { .. } => "req:memory_catch_up",
            Self::BranchCheckpointPass { .. } => "req:branch_checkpoint_pass",
            Self::RestoreSection { .. } => "req:restore_section",
            Self::RetireSections { .. } => "req:retire_sections",
            Self::SetBand { .. } => "req:set_band",
            Self::OffloadCollectionMembers { .. } => "req:offload_collection_members",
            Self::ExtractRawKvq { .. } => "req:extract_raw_kvq",
            Self::ProbeWideSigs { .. } => "req:probe_wide_sigs",
            Self::ReadRecurrentMemory { .. } => "req:read_recurrent_memory",
            #[cfg(any(test, feature = "test-helpers"))]
            Self::ReadTurnKvDigest { .. } => "req:read_turn_kv_digest",
            #[cfg(any(test, feature = "test-helpers"))]
            Self::CountRecurrentMemories { .. } => "req:count_recurrent_memories",
            Self::SubmitSummaryProbe { .. } => "req:submit_summary_probe",
            Self::ReconstructSubstrate { .. } => "req:reconstruct_substrate",
            Self::AttachSubstrate { .. } => "req:attach_substrate",
            Self::DetachSubstrate { .. } => "req:detach_substrate",
            Self::WarmCollectionNormalization { .. } => "req:warm_collection_normalization",
            Self::WarmDialogueNormalization { .. } => "req:warm_dialogue_normalization",
            Self::PreloadGalleries { .. } => "req:preload_galleries",
            Self::WarmIngestSlice { .. } => "req:warm_ingest_slice",
            Self::ScanGroupVotes { .. } => "req:scan_group_votes",
            Self::DemoteTimelinesHot { .. } => "req:demote_timelines_hot",
            Self::Wake => "req:wake",
            Self::Settle { .. } => "req:settle",
            Self::Shutdown => "req:shutdown",
        }
    }
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
/// in-flight prefill/section has `consumed == offset`). The wave log line
/// reports executed forwards only, so this sum is the one view of pending work.
/// Pure over its input so it is unit-testable without a live scheduler.
pub(in super::super) fn sum_pending_prefill_tokens(
    items: impl IntoIterator<Item = (usize, usize)>,
) -> u64 {
    items
        .into_iter()
        .map(|(total, consumed)| total.saturating_sub(consumed) as u64)
        .sum()
}

impl Scheduler {
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
    pub(in super::super) fn pending_prefill_tokens(&self) -> u64 {
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
        let queued_sections = self
            .section_queue
            .iter()
            .map(|s| (s.tokens.token_count(), 0));
        sum_pending_prefill_tokens(queued.chain(active).chain(sections).chain(queued_sections))
    }

    pub(super) fn drain_submissions(&mut self) -> bool {
        // Before any queued request: a turn submitted on a parent right after its
        // last one finished must not start ahead of the forks waiting on that gap.
        self.drain_parked_forks();
        loop {
            match self.rx.try_recv() {
                Ok(req) => {
                    // Timed for the same reason as the idle-path handler in
                    // `run.rs`: this is the hot drain during active work, so an
                    // ingest opening thousands of conversations spends real
                    // scheduler time here and it must not read as Blocked.
                    let t_req = Instant::now();
                    let req_span = profile::span(req.span_name());
                    let keep_going = self.handle_request(req);
                    req_span.end();
                    self.wave_stats
                        .add_requests(t_req.elapsed().as_micros() as u64);
                    if !keep_going {
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    /// Handle a single request. Returns `false` if shutdown.
    /// Handle one request, saying so when it held the wave thread long.
    ///
    /// **Named, because the phase ring cannot.** On WW under four Makers the
    /// ring put 12% of the scheduler's time in `requests` — one window 3.4 s of
    /// 4 — with no forward running meanwhile, and nothing said which request.
    pub(super) fn handle_request(&mut self, req: SchedulerRequest) -> bool {
        let _stage = census_stage::enter(Stage::Requests);
        let name = req.span_name();
        let began = Instant::now();
        self.request_laps.start();
        let keep_going = self.handle_request_now(req);
        let took = began.elapsed();
        if took >= SLOW_REQUEST {
            // With its steps, where the handler laps them (`request_laps`).
            if self.request_laps.any() {
                tracing::info!(
                    request = name,
                    took_ms = took.as_millis() as u64,
                    steps = %self.request_laps,
                    "scheduler: a request held the wave thread"
                );
            } else {
                tracing::info!(
                    request = name,
                    took_ms = took.as_millis() as u64,
                    "scheduler: a request held the wave thread"
                );
            }
        }
        keep_going
    }
}
