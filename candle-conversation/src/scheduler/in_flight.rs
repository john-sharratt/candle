//! Bookkeeping for the turns, views, compression jobs and sections in flight on
//! the scheduler.
//!
//! A turn's view state records the parent it borrows from and what it borrowed,
//! so a reprojection can rebuild the view and the seal can tell the turn's own
//! blocks from the prefix. A view is forked at admission rather than at submission,
//! because the fork holds a whole recurrent store that admission has to price. After
//! a reprojection the parent holds the turn's state itself, and a discarded view
//! hands it back. A compression job, its pending seal and a section ingest each
//! carry what is needed to finish after the forward that carries them, including
//! the scope derived before the decode and the layout of the reprefilled turn.

use super::*;

/// Per-turn view bookkeeping carried in the scheduler's `turn_views` map.
///
/// Distinct from a single view's `(parent, original_borrowed)` because
/// the **turn** outlives any single view: under continuous re-projection
/// each swap allocates a fresh view, so we need to remember turn-level
/// invariants (where the parent stood when the turn started) so the new
/// view can include the active turn's own decoded tail.
#[derive(Debug, Clone, Copy)]
pub(super) struct ViewState {
    /// Parent sequence the current view borrows from.  Stable across
    /// mid-decode swaps within one turn.
    pub(super) parent_id: SequenceId,
    /// Number of blocks **this** view borrowed at carve time.  Used by
    /// `finalize_view` to know which blocks were borrowed (and therefore
    /// must remain in the parent unchanged) versus which were freshly
    /// written by the view (and therefore must transfer back).
    pub(super) original_borrowed: BlockCount,
    /// The parent's total block count at the moment **the very first**
    /// view of this turn was carved.  Stable across swaps; copied
    /// verbatim from the swapped-from `ViewState` into the swapped-to
    /// one.  Lets the re-projection helper compute the active-turn tail
    /// range — the just-decoded blocks at parent indices
    /// `[turn_start_parent_blocks, current_parent_blocks)`.
    pub(super) turn_start_parent_blocks: usize,
    /// Token length of the turn's user half (`user_content_end`), captured at
    /// turn open and carried across swaps — the Concept F question-window
    /// boundary (`docs/provenance_adaptive_projection.md` §8). The pinned
    /// Q-window is the turn's first `question_tokens` real tokens.
    pub(super) question_tokens: usize,
    /// The identity ([`projection_identity`]) of the segment list this turn's
    /// last reprojection rebuilt the parent from — `None` until the turn's first
    /// reprojection, and whenever that rebuild left a piece out. A reprojection
    /// that arrives at the same identity would rebuild the parent it already
    /// has, so it leaves the parent and the view as they are.
    pub(super) applied_identity: Option<u64>,
    /// Whether the view holds its own copy of the parent's per-sequence model
    /// state yet. A submitted turn's view is carved at submit but forked at
    /// admission ([`Scheduler::begin_admitted`]): the fork is a whole recurrent
    /// store on a hybrid, ground admission prices and can make the turn wait
    /// for. Forked at submit, a window of turns would claim their stores at once
    /// and fail the ones the span had no room for.
    pub(super) state_forked: bool,
    /// Whether the view holds the parent's state itself rather than a copy of
    /// it: a reprojection moves the turn's store onto the view it re-carves
    /// (`reproject_view_complete`), leaving the parent none. A discarded view
    /// moves it back ([`Scheduler::discard_turn_view`]) instead of releasing
    /// it, or the parent's next turn would start from a store with none of the
    /// conversation in it.
    pub(super) holds_parent_state: bool,
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
pub(super) struct CompressionJob {
    /// Conversation that owns the node's timeline.
    pub(super) conversation: Conversation,
    /// Projection target (layer, group, timeline) the node records into.
    pub(super) target: ProjectionTarget,
    /// Node kind — carried to the seal, which records it on the sealed turn.
    pub(super) kind: TurnKind,
    /// The node's user half, derived from the children's scopes before the decode
    /// ran (`summary_tree::scope`). Paired with the decoded assistant half at the
    /// seal — the decode is never asked to write a question.
    pub(super) scope_tokens: Vec<u32>,
    /// The child turns this pass lifted to hot to attend over. Demoted back to
    /// warm in `complete_compression_pass` once the summary has decoded, so a long
    /// run of summary passes doesn't accumulate transient hot residency.
    pub(super) children: Vec<TurnKey>,
    /// Summariser channel — receives the recorded node's `TurnIndex`, or an `Err`
    /// if the decode produced no forwarded tokens.
    pub(super) response_tx: Sender<Result<TurnIndex, ProbeError>>,
}

/// A compressed turn whose marker-framed text has been enqueued for re-prefill
/// on the shared prefill wave, awaiting its seal. Keyed by `job_id` in
/// [`Scheduler::pending_compression_seals`]. When the prefill completes,
/// `complete_compression_turn` snapshots the slot's freshly-computed
/// (role-coherent) K/V, records the turn, and fires `response_tx`.
pub(super) struct PendingCompressionSeal {
    pub(super) conversation: Conversation,
    pub(super) target: ProjectionTarget,
    /// The compressed turn's segment-vector layout (user / assistant text +
    /// spans), built when the exchange was framed.
    pub(super) layout: TurnLayout,
    pub(super) token_ids: Vec<u32>,
    /// Forest kind of the node — see [`CompressionJob::kind`].
    pub(super) kind: TurnKind,
    /// The turns this node compresses — referenced by the node's synthesized
    /// projection event so a wide-Q hit on the summary resolves to the
    /// covered turns.
    pub(super) children: Vec<TurnIndex>,
    /// Gather-scope tags inherited as the union of the children's TurnDecl
    /// tags: a code_read leaf carries its scan turn's `["code", <path>]`, a
    /// SoS the union of its children's; untagged (dialogue) children yield
    /// an untagged summary, preserving the tag-partition invariant.
    pub(super) tags: Vec<String>,
    pub(super) response_tx: Sender<Result<TurnIndex, ProbeError>>,
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
        /// the tool catalog, an entry in a hits list, etc.) at its
        /// collection's level, or a boundary section (role markers,
        /// opening / closing tags) near-lossless — see [`SectionSeal`].
        seal: SectionSeal,
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
    /// One case of a turn group, prefilled on a view of its own. When its
    /// prefill finishes, `cleanup_finished` hands it to its group
    /// ([`Scheduler::finish_group_case`]) instead of finalizing it onto the
    /// parent: the group seals its cases in order, each from its own view.
    GroupCase(GroupCaseId),
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
pub(super) struct PendingSectionQuantize {
    /// The substrate the section was sealed into. Section ids are a schema's
    /// own, so two tenants sharing the engine can each hold a section `7`;
    /// a drain takes only the sealing substrate's entries.
    pub(super) substrate: SubstrateRef,
    pub(super) section_id: SectionId,
    /// Propagated from `SealAction::Section::seal` — the policy the drain
    /// quantizes the section under ([`SectionSeal::policy`]).
    pub(super) seal: SectionSeal,
}

/// A unit of prefill work queued for processing.
pub(in super::super) struct PrefillWork {
    pub(in super::super) sequence_id: SequenceId,
    pub(in super::super) tokens: TokenBuffer,
    /// Text to emit as TurnEvent::Prefill before starting decode.
    pub(in super::super) prefill_text: String,
    /// Raw user message string — see [`DecodeState::user_text`].
    pub(in super::super) user_text: String,
    /// Gather-scope tags — see [`DecodeState::tags`].
    pub(in super::super) tags: Vec<String>,
    /// Content boundaries — see [`DecodeState`].
    pub(in super::super) user_content_start: u32,
    pub(in super::super) user_content_end: u32,
    pub(in super::super) assistant_content_start: u32,
    /// Thinking suppressed at submit — see [`DecodeState::no_think`].
    pub(in super::super) no_think: bool,
    /// Prefilled assistant half — see [`DecodeState::prefill_assistant_text`].
    pub(in super::super) prefill_assistant_text: String,
    pub(in super::super) event_tx: Sender<TurnEvent>,
    pub(in super::super) max_decode_tokens: usize,
    pub(in super::super) sampling: SamplingConfig,
    pub(in super::super) submitted_at: Instant,
    /// Carried through prefill so it can be installed onto `DecodeState`
    /// when the prefill promotes to decode.  `None` when the caller
    /// disabled re-projection.
    pub(in super::super) reprojection: Option<ReprojectionPolicy>,
    /// The turn's opening belief — the conversation's carried belief stepped
    /// through the submit-time projection. Installed onto `DecodeState` so the
    /// first mid-decode reprojection evolves it rather than starting empty.
    /// Default for paths with no belief-driven selection (sections, resume,
    /// compression).
    pub(in super::super) belief: PriorBelief,
    /// Carried through prefill so the post-Done substrate write fires
    /// on the right key.  The substrate target is looked up from
    /// [`Scheduler::slot_targets`] at seal time, not carried here.
    pub(in super::super) seal_action: SealAction,
    /// Trailing structural tokens written into the slot after decode
    /// finishes, before the seal.  Carried through prefill so the
    /// post-decode forward pass in `cleanup_finished` can run.  Empty
    /// for paths that don't append a closing tail.
    pub(in super::super) post_decode_tokens: TokenBuffer,
    /// Staged-prefill projection points — token offsets into `tokens` where the
    /// wave stops and emits a projection. Empty for a normal (single-projection)
    /// prefill. See [`SchedulerRequest::SubmitTurn::projection_offsets`].
    pub(in super::super) projection_offsets: Vec<u32>,
    /// The projection composition emitted at each staged projection point (the
    /// calibration projection is pinned, so one composition serves every segment;
    /// only its `start_token`/`end_token` span differs per emission). `None` for a
    /// normal prefill.
    pub(in super::super) staged_composition: Option<crate::projection::ProjectionEvent>,
    /// Tool-call stencils carried through prefill and installed on the
    /// [`DecodeState`] at decode start.  Empty registry = no constrained decode.
    pub(in super::super) triggers: Arc<TriggerRegistry>,
    /// A grammar the turn begins inside — see [`TurnOptions::turn_grammar`].
    /// Armed at the prefill boundary, before the first token is sampled.
    pub(in super::super) turn_grammar: Option<TurnGrammar>,
    /// Whether this turn's schema asked for repetition penalties to be lifted
    /// while a tool call is emitted — see
    /// [`crate::projection::Schema::free_tool_calls_from_penalties`].
    pub(in super::super) free_tool_calls_from_penalties: bool,
    /// The recorded reply this turn replays — see [`DecodeState::recorded_reply`].
    pub(in super::super) recorded_reply: Option<RecordedReply>,
    /// Spans of `tokens` each prefilled as a chunk of its own, armed on the
    /// model for a capture or a replay of their recurrent state.
    pub(in super::super) recurrent_spans: Vec<TokenSpan>,
    /// The images in `tokens`, whose pads reach the engine as their rows'
    /// virtual ids ([`crate::vision::engine_ids`]); their handles hold the rows
    /// for as long as this work can be prefilled.
    pub(in super::super) image_spans: Vec<ImageSpan>,
    /// A request's first reading (`first_reading`): the question is
    /// prefilled to its last token against a projection without the seeds,
    /// scored, and submitted again with them. Set at submit for a turn a new
    /// user message opens.
    pub(in super::super) rereads: bool,
}

/// An in-flight prefill, partially advanced. Lives across scheduler
/// iterations until `offset` reaches `work.tokens.len()`, at which point it
/// is promoted to `active_decodes`.
pub(in super::super) struct ActivePrefill {
    pub(in super::super) work: PrefillWork,
    /// Tokens consumed so far.
    pub(in super::super) offset: usize,
    /// Index of the next `work.projection_offsets` entry the wave has yet to
    /// reach and emit. Advances as the prefill crosses each staged projection
    /// point. Unused (stays 0) for a normal prefill with no offsets.
    pub(in super::super) next_projection: usize,
    /// How the last chunk ended — for a turn, its first token sampled in the
    /// wave that delivered that chunk (`first_sample`). Drained by
    /// `promote_finished_prefills_to_decodes`.
    pub(in super::super) finished: Option<PrefillEnd>,
    /// Set if any chunk for this prefill failed.
    pub(in super::super) error: Option<ConversationError>,
    /// Wall-clock when prefill processing actually started (first chunk).
    pub(in super::super) prefill_start: Option<Instant>,
    /// Every how many tokens the opening stops for a boundary that re-chooses
    /// the slot (`docs/hit_driven_projection.md` §7.4): its target's
    /// `reserve_tokens` on an engine that selects. `None` for a prefill whose
    /// slot is not filled — no working set, an engine that attends every
    /// position, a section, a calibration's staged or captured prefill — which
    /// runs to its end in one stride.
    pub(in super::super) stops_every: Option<usize>,
    /// Stopped at a boundary, its reprojection queued: no wave takes it until
    /// the reprojection hands it back on its new view.
    pub(in super::super) awaiting: bool,
}

impl ActivePrefill {
    /// Where the prefill stops: its last token, or for a first reading the
    /// token before it, so the reading is scored before anything is sampled.
    pub(super) fn end(&self) -> usize {
        let total = self.work.tokens.len();
        match self.work.rereads {
            true => total - 1,
            false => total,
        }
    }

    /// The most the next chunk may advance: the tokens left, cut at the next
    /// boundary.
    pub(in super::super) fn advance_cap(&self) -> usize {
        let left = self.end() - self.offset;
        match self.stops_every {
            Some(every) => left.min(every - self.offset % every),
            None => left,
        }
    }

    /// Whether the opening, now at `offset`, stands at a boundary it stops at:
    /// a whole number of strides in, with tokens still to come — or a first
    /// reading at its end.
    pub(in super::super) fn at_boundary(&self) -> bool {
        if self.work.rereads {
            return self.offset == self.end();
        }
        self.stops_every
            .is_some_and(|every| self.offset > 0 && self.offset.is_multiple_of(every))
            && self.offset < self.work.tokens.len()
    }
}

/// Resolve a seal's block range against the slot as it actually stands.
///
/// A seal is always the tail of its slot: `[from, end)`, where the end is the
/// last block the slot holds. A turn group's cases each seal from a view of
/// their own, so no slot ever holds two sealable regions.
///
/// `block_count` is the slot's authoritative chunk total and `chunk_len` the
/// snapshot's; the range is clamped to both because a snapshot taken a moment
/// earlier can be the shorter of the two, and indexing past it would panic a
/// slice. Returns `None` when the range is empty — the caller treats that as
/// "nothing to seal" and, for a turn, warns loudly, because an empty range means
/// the turn's K/V is gone and the seal would silently drop it.
///
/// Pure, so the clamping is testable without a session or a device.
pub(super) fn resolve_seal_range(
    from: usize,
    block_count: usize,
    chunk_len: usize,
) -> Option<(usize, usize)> {
    let to = block_count.min(chunk_len);
    (to > from).then_some((from, to))
}

/// A section ingest waiting for admission.
///
/// **A section is a first admission exactly as a prefill is**: fresh K/V on its
/// slot, a recurrent store on a hybrid, and rows in the wave's tier. So it waits
/// here until an admission pass prices it and finds the wave better off
/// carrying it (`AdmitPass`, at its own band), and only then has its slot
/// prepared and joins the forwards. Sections put into flight as their requests
/// drain take a minute to push Qwen3.6's weight zone from 10,398 MiB to its
/// hold, and on Qwen3.8-Flash-Next on a 16 GB card a calibration's sections
/// and cases fill the ground below the weight floor until no wave's tier can be
/// placed and nothing in flight can finish.
pub(in super::super) struct QueuedSection {
    pub(in super::super) sequence_id: SequenceId,
    pub(in super::super) section_id: SectionId,
    /// See [`SchedulerRequest::IngestSection::prefix`].
    pub(in super::super) prefix: Vec<PrefixItem>,
    pub(in super::super) tokens: TokenBuffer,
    pub(in super::super) address: ContentAddress,
    pub(in super::super) debug_name: String,
    pub(in super::super) seal: SectionSeal,
    /// See [`SchedulerRequest::IngestSection::start`].
    pub(in super::super) start: Option<SharedState>,
    /// See [`SchedulerRequest::IngestSection::band`].
    pub(in super::super) band: DecodePriority,
    pub(in super::super) response_tx: Sender<Result<SealResult, ConversationError>>,
    /// When it was queued — what a long wait for admission is reported from
    /// (`submit_admit::Scheduler::report_waiting_submits`).
    pub(in super::super) queued_at: Instant,
}

/// An in-flight section ingest — CPU setup is done, awaiting the batched
/// forward pass(es).  Multiple concurrent entries are batched together in
/// [`Scheduler::run_one_section_ingest_chunk`] so collection members (which
/// share the same prefix) prefill in parallel rather than serially.
pub(in super::super) struct ActiveSectionIngest {
    pub(in super::super) sequence_id: SequenceId,
    pub(in super::super) section_id: SectionId,
    pub(in super::super) tokens: TokenBuffer,
    pub(in super::super) offset: usize,
    pub(in super::super) seal_block_from: usize,
    /// Content address derived from the section's tokens + cumulative
    /// prefix at IngestSection time.  Used to derive the persistence
    /// stream id at seal — see `SealAction::Section::address`.
    pub(in super::super) address: ContentAddress,
    /// Section's symbolic name for the SectionDecl record — see
    /// `SealAction::Section::debug_name`.
    pub(in super::super) debug_name: String,
    /// Carried from the `IngestSection` request through to
    /// `SealAction::Section` — see that variant's docstring for the
    /// quantize-policy implications.
    pub(in super::super) seal: SectionSeal,
    /// Whether the seal hands back the state the prefill ended on — set for a
    /// section that was given the state it starts from.
    pub(in super::super) export_end: bool,
    /// See [`SchedulerRequest::IngestSection::band`].
    pub(in super::super) band: DecodePriority,
    pub(in super::super) response_tx: Sender<Result<SealResult, ConversationError>>,
    pub(in super::super) error: Option<ConversationError>,
    /// When it was queued, admitted and prepared — written beside its seal.
    pub(in super::super) timing: SectionTiming,
}

/// A member of the in-flight continuous-fair-wave prefill group
/// (`docs/continuous_fair_waves.md`). The group creeps through the layers
/// together with its residual held whole between waves; members are separated
/// only at the head, where `forward_wave`'s logits are in caller order. A member
/// is resolved live by `seq_id` (not a stored index) so `swap_remove` in either
/// backing collection can't invalidate it mid-creep.
#[derive(Clone, Copy, Debug)]
pub(in super::super) enum WaveMember {
    /// Dialogue-turn prefill chunk — resolved in `active_prefills`; covers
    /// `[offset, offset + advance)` of its token set (offset is stable until the
    /// head, where the chunk is committed). The final chunk is promoted to
    /// decode; an earlier one leaves the prefill active for the next group.
    Prefill { seq_id: usize, advance: usize },
    /// Section-ingest chunk — resolved in `active_section_ingests`; covers
    /// `[offset, offset + advance)` (offset is stable until the head, where the
    /// chunk is advanced + sealed). Rides the cohort's creep so its expert loads
    /// co-batch with decode + the dialogue prefills.
    Section { seq_id: usize, advance: usize },
}

impl WaveMember {
    /// The session sequence this member occupies, whichever kind it is.
    pub(in super::super) fn seq_id(&self) -> usize {
        match self {
            WaveMember::Prefill { seq_id, .. } => *seq_id,
            WaveMember::Section { seq_id, .. } => *seq_id,
        }
    }
}
