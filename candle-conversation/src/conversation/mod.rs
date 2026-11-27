//! Client-side conversation handle that lays out turns and submits them to the
//! scheduler.
//!
//! All CPU work, tokenisation and formatting, happens on the caller's thread, and
//! the scheduler only runs GPU work. A `Sequence` reaches its submitter's fields
//! through `Deref`, so a handle that outlives one borrow of the conversation can
//! still submit a turn. Turns are recorded in the substrate and the redo log, and
//! the conversation's frame, branch checkpoint and selection are kept on the
//! handle between turns.
//!
//! - `probe_assembly`: framing for the projection panel and the interactive probe.
//! - `sequence_construction`: system-prompt ingest and prompt-branch checkpoints.
//! - `sequence_fork`: forks, and whether they inherit recurrent memory.
//! - `sequence_history`: recorded history read back for display and projection.
//! - `sequence_ingest`: prefilled content turns and tool round-trip summaries.
//! - `sequence_lifecycle`: a turn from submission to seal, including cancellation.
//! - `sequence_prefilled`: calibration turns prefilled from a trajectory.
//! - `sequence_sections`: system-prompt section ingest into the substrate.
//! - `sequence_state`: live state, observables and between-turn settings.
//! - `turn_submitter`: turn grid layout and submission to the scheduler.
//! - `window_slices`: key-value window cuts and oversized-probe caps.
//! - `tests_fork`: fork and branch-swap predicates.
//! - `tests_turn_assembly`: probe and glue framing per dialect.
//! - `tests_windowing`: windowed turns, with K/V and index pages checked together.

use crate::config::{SamplingConfig, SequenceConfig};
use crate::error::ConversationError;
use crate::handle::{SealResult, TurnEvent, TurnHandle, TurnResponse};
use crate::index_pages::StoredPages;
use crate::provenance::boundaries::BoundaryRule;
use crate::recorded_reply::{recorded_reply, RecordedReply};
use crate::recovered::recovered_from;
use crate::recovered_message::{RecoveredMessage, TOOL_RESPONSE_OPEN};
use candle::engine_events::note;
use candle_transformers::models::delta_net::ExportedLayerState;

use crate::persistence::content_hash::{
    hash_tokens, section_stream_id, turn_stream_id, ContentChain, ContentHash,
};
use crate::persistence::record::BranchCheckpointPayload;
use crate::persistence::streams::ContentAddress;
use crate::projection::{
    decode_events, decode_points, from_projection_with_origins, runs_are_empty, sections_prefix,
    Builder, Conversation, DecodePriority, Observe, PrefixItem, ProjectionEvent, ProjectionMode,
    ProjectionPoint, ProjectionTarget, RangeFill, SectionId, SectionTree, SelectionState,
    SystemPromptItem, TimelineId, TokenRun, TurnIndex, TurnKey,
};
use crate::provenance::gallery_arena::BeliefArena;
use crate::provenance::{decode_wide_sigs, WideQSig};
use crate::reply_wait::replied;
use crate::scheduler::projection_assembler::materialize_conversation;
use crate::scheduler::{
    note_branch_checkpoint_computed, note_branch_checkpoint_installed, DecodeReprojection,
    ProjectionInputs, RemoteArena, ReprojectionPolicy, SchedulerRequest, TurnContent,
};
use crate::sealed_turn::{SealedPages, SealedTurn};
use crate::sequence_handle::{BlockCount, SequenceId};
use crate::substrate::ProjectionScores;
use crate::token_buffer::TokenBuffer;
use crate::tree::token_text::TokenizedText;
use crate::tree::{ConversationTree, TurnType};
use crate::turn::{Role, Turn, TurnOptions};
use crate::turn_group::{plan_cases, CaseGrid};
use crate::turn_layout::TurnLayout;
use crate::working_set::marks::gather_tags;
use candle_nn::kv_cache::{SealedChunk, SealedSequence};
use candle_transformers::models::batched_inference::ModelCoreProperties;
use candle_transformers::models::dialect::Dialect;
use candle_transformers::models::qwen_vision::VisionTowerSource;

use crate::recurrent_span::TokenSpan;
use crate::stencil::{ThinkMode, TriggerRegistry, TurnGrammar};
use crate::turn_text::{encode_pieces, token_span, TurnText};
use crate::vision::{marker, stored_spans, ImageSpan, TurnImage, VisionTokens};
use flume::{Receiver, Sender};
use std::sync::{Arc, Mutex};

mod carried;
pub use carried::{Carried, Changes, Linking, Want};
mod birth_range;
mod ephemeral_turn;
pub(crate) use birth_range::BirthRange;
mod frame;
mod member_seal;
pub use ephemeral_turn::{AskScope, UnsealedAsker};
pub(crate) use frame::Frame;
mod prefix_state;
mod prompt_dump;
pub(crate) mod section_end_state;
mod sharing;
mod submitted_sections;
mod turn_bans;

pub use member_seal::{MemberSpec, MembersPrefix, SealAs, SealedMember};
pub(crate) use prefix_state::run_state;
use prefix_state::{chain_of, checkpoint, shared_held, state_after};
pub use prompt_dump::{PieceKind, PromptDump, PromptPiece};
pub use sharing::{is_shared_stream_name, shared_stream_name, Sharing, SHARED_SECTION_NAME_PREFIX};
use submitted_sections::SubmittedSections;
pub use submitted_sections::{SectionRef, SectionState};

/// What it takes to lay out a turn's grid and hand it to the scheduler: the
/// channel, the tokenizers and the config, none of which a conversation changes
/// after it opens. Split out of [`Sequence`] so a handle that outlives any one
/// borrow of the conversation ([`UnsealedAsker`]) can submit a turn without it.
/// A `Sequence` reaches these fields and methods through `Deref`.
#[derive(Clone)]
pub struct TurnSubmitter {
    /// Channel to submit GPU work to the scheduler.
    pub(crate) scheduler_tx: Sender<SchedulerRequest>,

    /// Shared tokenizer.
    pub(crate) tokenizer: Arc<tokenizers::Tokenizer>,

    /// The tokenizer for literal text, which reads every chat tag as plain
    /// characters — see [`crate::turn_text`].
    pub(crate) literal_tokenizer: Arc<tokenizers::Tokenizer>,

    /// Per-conversation config.
    pub(crate) config: SequenceConfig,
}

/// The two things a conversation changes between turns that an
/// [`UnsealedAsker`] must see current: the projection schema (its submitted
/// sections come and go) and the selection the last turn ran under.
pub(crate) struct AskState {
    pub(crate) projection: Arc<Builder>,
    pub(crate) selection: SelectionState,
}

impl std::ops::Deref for Sequence {
    type Target = TurnSubmitter;

    fn deref(&self) -> &TurnSubmitter {
        &self.submitter
    }
}

pub struct Sequence {
    /// How this conversation lays out and submits a turn.
    submitter: TurnSubmitter,

    /// The projection and selection as of the last change, shared with every
    /// [`UnsealedAsker`] taken from this conversation.
    ask_state: Arc<Mutex<AskState>>,

    /// The sequence slot allocated by the scheduler.  This is the
    /// sequence's identifier — there is no separate logical id,
    /// since persistence is workspace-scoped (substrate-keyed) and
    /// every other use of "which sequence" is in-memory addressing
    /// that this `SequenceId` covers.
    id: SequenceId,

    /// Sequence tree — the canonical turn history for this conversation.
    /// Holds system prompt (with token ids), paired user↔assistant exchanges
    /// (each with text + token ids), temporal markers, and phase 2+ segment nodes.
    tree: ConversationTree,

    /// Pending user turn set by `submit_turn_with_options`, consumed by
    /// `finish_turn`. Holds raw user text + token ids for commit to the tree.
    pending_user: Option<TokenizedText>,

    /// Monotonic counter tracking the total number of individual entries
    /// (system, user, assistant) appended to cold store.  Also used by
    /// `turn_count()` for informational display.

    /// Monotonic turn counter.
    turn_counter: u64,

    /// Current section-tree selection (e.g. the composer dials).  Set from each
    /// turn's [`TurnOptions::selection`] and used by every projection until the
    /// next turn changes it.  Empty = the schema's authored defaults.
    selection: SelectionState,

    /// True while a submitted turn is being processed by the scheduler.
    turn_in_flight: bool,

    /// True after `close()` has been called (prevents double-free in Drop).
    freed: bool,

    /// High-water mark: cumulative sealed blocks at the end of the
    /// most recent seal.  Per-turn block ranges live on the substrate
    /// (one entry per appended turn); this field is just the running
    /// tail pointer used for fork-origin and snapshot bookkeeping.
    current_blocks: BlockCount,

    /// KV chunk size in tokens (mirrors `BatchedConfig::chunk_size`).
    chunk_size: usize,

    /// Static model properties captured at engine construction.
    model_core: ModelCoreProperties,

    /// The engine's vision tower, whose rows a stored turn's images are read
    /// back as; `None` for a model that reads no images.
    vision: Option<&'static VisionTowerSource>,

    /// Projection schema for this conversation.
    ///
    /// Held as an `Arc` so the scheduler can hold a reference for
    /// continuous re-projection without repeatedly deep-cloning the
    /// underlying `Schema`.
    ///
    /// This is `base_projection` plus the sections this conversation has
    /// submitted and sealed — see [`SubmittedSections`].
    pub(crate) projection: Arc<Builder>,
    /// The schema as the caller supplied it, without the submitted sections.
    base_projection: Arc<Builder>,
    /// Sections submitted to this conversation after it opened.
    submitted: SubmittedSections,
    /// Workspace-shared substrate handle (Arc-cloneable).  Holds per-turn
    /// metadata (token counts, scores, sig entries, restoration sources)
    /// for every conversation in the engine.  Read-locked during projection,
    /// write-locked at seal time.
    pub(crate) substrate: Conversation,
    /// `(layer, group)` this conversation is for.  Determines which group
    /// `seal_and_register_turn` appends into and which target is passed to
    /// `projection.project()`.  Different cognitive activities (dialogue,
    /// bug analysis, dream log, …) point at different targets.
    pub(crate) target: ProjectionTarget,
    /// The birth-time section list — the schema's fixed sections, each tree at
    /// its default branch — and the branch windows inside it, retained because
    /// `prompt_branch` needs both to rebuild a checkpoint when the FIRST turn's
    /// dials select a different branch than the create-time default.
    ///
    /// One window per [`SectionTree`] the schema declares, in item order: a
    /// schema may declare several (zend's does), each contributing its own
    /// run of ids to the birth list.
    ///
    /// Each reserved range is a hole in it, left empty, so everything after a
    /// range is sealed and read at the positions it has however full the range
    /// is (`docs/placed_section_layout.md` §3.1).
    birth_prefix: Arc<Vec<PrefixItem>>,
    /// Where each reserved range stands in [`Self::birth_prefix`].
    birth_ranges: Vec<BirthRange>,
    /// The sections of the frame an append-only conversation's first turn
    /// lays onto an empty slot: [`Self::birth_prefix`], sent with the first
    /// submit — its lineage's turns after them ([`Frame::of`]) — and gone after
    /// it ([`ProjectionInputs::frame`]). `None` for a conversation that
    /// re-projects every turn, and once the first turn has been submitted. A
    /// fork carries its parent's, so a unit forked from an ingest base that
    /// never ran a turn is framed exactly as the base would have been, behind
    /// whatever lineage it has been given by its first submit.
    frame: Option<Arc<Vec<PrefixItem>>>,
    /// The turns the pending frame lays down in place of the inherited chain
    /// ([`Self::frame_window`]); `None` lays down the chain.
    frame_window: Option<Vec<TurnKey>>,
    branch_spans: Vec<(usize, usize)>,
    /// True when this conversation was **born with a prompt checkpoint** — as
    /// opposed to a fork, whose state is the parent's or the timeline's and
    /// must never be swapped.
    ///
    /// Half of the branch-swap window; the other half (has anything advanced
    /// that state since?) is derived from the timeline's turn count at the use
    /// site in [`Sequence::submit_turn_with_options`], because every path that
    /// advances the state lands a turn on the timeline and a tracked flag
    /// would have to be cleared by each of them.
    state_is_prompt_only: bool,
    /// The prompt branch whose checkpoint this conversation's state is — the one
    /// built at create, or the one a dialed first turn swapped in. `None` when no
    /// checkpoint was built (a model without recurrent state, an empty prompt, a
    /// fork). A first turn whose dials name this same branch has nothing to swap.
    installed_branch: Option<ContentHash>,
}

impl Drop for Sequence {
    fn drop(&mut self) {
        if !self.freed {
            // Best-effort: free the sequence slot.
            let _ = self.scheduler_tx.send(SchedulerRequest::FreeSequence {
                sequence_id: self.id,
            });
        }
    }
}

mod sequence_construction;
mod sequence_fork;
mod sequence_ingest;
mod sequence_prefilled;
mod sequence_sections;
mod turn_submitter;
mod window_slices;
#[cfg(test)]
use sequence_fork::*;
mod probe_assembly;
mod sequence_history;
mod sequence_lifecycle;
mod sequence_state;
#[cfg(test)]
mod tests_fork;
#[cfg(test)]
mod tests_turn_assembly;
#[cfg(test)]
mod tests_windowing;

pub use probe_assembly::*;
pub(crate) use sequence_construction::*;
pub use sequence_ingest::*;
use sequence_lifecycle::*;
use sequence_sections::*;
pub(crate) use window_slices::*;
