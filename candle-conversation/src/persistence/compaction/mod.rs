//! Store compaction: rewrites only the live records into a fresh segment, which
//! reclaims the space of dead records.
//!
//! An append-only store only grows, so superseded snapshots, stale singletons
//! and the chunks of deleted streams accumulate. Compaction plans the live set in
//! dependency order from the index, and writes it with verbatim copies of the
//! records that sit in the log and encoded copies of resident metadata. The
//! orchestration that adopts the result as the sole segment is
//! `SubstratePersistence::compact`.
//!
//! - `branch_checkpoints`: the cap on prompt branch checkpoints.
//! - `collect`: planning of the live set.
//! - `compact_item`: the items staged into the new log.
//! - `write_log`: writing the live set to a fresh log.
//! - `tests_cast_and_files`: tests for characters and file events.
//! - `tests_couplings_tombstones`: tests for couplings and tombstones.
//! - `tests_distillation`: tests for distilled timelines.
//! - `tests_live_set`: tests for the live set and recovery.
//! - `tests_snapshots`: tests for recurrent-state snapshots.
//! - `tests_support`: fixtures for the compaction tests.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::adoption::Adoption;
use super::blob::BlobIndex;
use super::content_hash::snapshot_stream_id;
use super::deleted::DeletedPayload;
use super::header_index::{encode_index_payload, IndexEntry, INDEX_FLUSH_ENTRIES};
use super::identity::MarkerIds;
use super::log_file::LogFile;
use super::manifest::{encode_conv_state_payload, Manifest, RecordLoc};
use super::rebase::RebaseMap;
use super::record::{
    decode_header, decode_record, encode_record, encode_record_written, DebugIdPayload,
    DistillMode, DistillPayload, RecordHeader, RecordType, SectionTombstonePayload,
    TombstonePayload, TurnCouplingPayload,
};
use super::replica::ReplicaPayload;
use super::segment::{SegmentId, FIRST_SEGMENT};
use super::stamped_locs::StampedLocs;
use super::streams::{StreamDecl, StreamId};
use super::stripes::{stripes, MAX_STRIPE_BYTES};
use super::tombstone_expiry::TombstoneExpiry;
use super::vfs::{carried, VfsIndex};
use super::Result;
use crate::projection::TimelineId;
use crate::substrate::{IndexPage, Located, Substrate};

mod branch_checkpoints;
mod collect;
mod compact_item;
#[cfg(test)]
mod tests_cast_and_files;
#[cfg(test)]
mod tests_couplings_tombstones;
#[cfg(test)]
mod tests_distillation;
#[cfg(test)]
mod tests_live_set;
#[cfg(test)]
mod tests_snapshots;
#[cfg(test)]
mod tests_support;
mod write_log;

pub(super) use branch_checkpoints::*;
pub use collect::*;
pub use compact_item::*;
pub use write_log::*;
