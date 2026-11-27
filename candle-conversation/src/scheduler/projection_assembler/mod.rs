//! The per-slot projection assembler. It rebuilds a slot's prefix K/V for each
//! projection and keeps every piece the slot already holds.
//!
//! The slot is cut after its last held piece that is still wanted. Each change
//! before that point is spliced in where it falls, and the rest of the assembled
//! pieces are walked onto the tail in logical order. A sealed piece is kept by its
//! own identity, since its K/V is the same wherever it sits. A forwarded piece is
//! kept only while the identities before it still match, since its K/V was
//! computed against all of them. Glue islands are cached separately, under their
//! prefix identity.
//!
//! - `apply`: applies a projection's segments, inline or deferred.
//! - `apply_build`: builds a slot's prefix, keeping the pieces it holds.
//! - `assembly`: lays the pieces out in logical order.
//! - `gap_fill`: fires the batched forward that fills glue gaps.
//! - `inject`: injects sealed sections and turns with their index pages.
//! - `prefill`: prefills a slot's own tokens, and re-injects forwarded spans.
//! - `slot_state`: the state a slot carries between projections.
//! - `walk`: walks sealed pieces, and reserves glue islands in place.
//! - Tests: `tests_assembly`, `tests_fixtures`, `tests_reasoning`, `tests_slots`.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use candle::engine_events::note;
use candle::{Device, Tensor};
use candle_nn::kv_cache::{ArenaLocation, SealedSequence, WriterTail, CHUNK_SIZE};
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel, PendingGlue, WINDOW_DIVERGENCE_MARKER,
};
use candle_transformers::models::expert_lre::is_expert_stall;
use candle_transformers::models::piece_key::PieceKey;
use candle_transformers::models::positional_piece::PositionalPiece;
use candle_transformers::models::sealed_injection::SealedInjection;

use crate::conversation::slice_per_layer_sealed;
use crate::error::ConversationError;
use crate::index_pages::{self, StoredPages};
use crate::persistence::content_hash::turn_stream_id;
use crate::projection::event::{group_name_of, layer_name_of_group, role_str};
use crate::projection::rounds::{self, RoundKey};
use crate::projection::{
    ContentResolver, Conversation, GroupId, HitMomentum, MaterializedPiece, Pin, ProjectionSegment,
    ProjectionTarget, Schema, SealedKind, SectionId, SelectedTurn, TimelineId, TurnIndex, TurnKey,
};
use crate::scheduler::hit_density::HitPart;
use crate::scheduler::injected_piece::InjectedPiece;
use candle_transformers::models::decode_diag::PieceSpan;
use candle_transformers::models::hit_layers::PieceHits;

use crate::files::scope::{band, EvictWatch};
use crate::files::SectionKey;
use crate::scheduler::pending_ranges::PendingRanges;
use crate::scheduler::piece_identity::{
    alignment_identities, chained_identities, piece_label, PieceReads,
};
use crate::scheduler::piece_splice::{plan_splice, HeldPiece, PieceKind, SplicePlan, WantedPiece};
use crate::scheduler::pool_linger::PoolLinger;
use crate::scheduler::pooled_rounds::{pooled_rounds, PoolPages};
use crate::scheduler::profile;
use crate::scheduler::projection_identity::section_content_stamp;
use crate::scheduler::section_pages::SectionPages;
use crate::sequence_handle::SequenceId;
use crate::substrate::Substrate;
use crate::summary_tree::SelectionOrigin;
use crate::summary_tree::TurnKind;

mod apply;
mod apply_build;
mod assembly;
mod gap_fill;
mod inject;
mod kb_sections;
mod prefill;
mod slot_state;
#[cfg(test)]
mod tests_assembly;
#[cfg(test)]
mod tests_fixtures;
#[cfg(test)]
mod tests_holes;
#[cfg(test)]
mod tests_kb_rounds;
#[cfg(test)]
mod tests_reasoning;
#[cfg(test)]
mod tests_slots;
mod walk;

pub(super) use apply::*;
pub(super) use apply_build::*;
pub(crate) use assembly::*;
pub(super) use gap_fill::*;
use inject::*;
use kb_sections::kb_sections;
pub(super) use prefill::*;
pub(super) use slot_state::*;
use walk::*;
