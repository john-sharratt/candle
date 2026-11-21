//! Background maintenance for the segmented redo log: drop, compact or combine a
//! sealed segment.
//!
//! Each op relocates a sealed segment's live records into the active segment and
//! only then unlinks the source, so no needed record is lost. Ops run one at a
//! time on the persistence thread and release the substrate lock during the slow
//! I/O. Resident records are rebuilt from RAM and re-emitted before a drop, while
//! read-back records are moved by location. Writes that land between phases are
//! detected, so a stale copy never replaces a newer record.
//!
//! - `policy`: choosing the op from segment stats.
//! - `plan`: gathering an op's plan under a brief read lock.
//! - `worklist`: which records an op relocates.
//! - `shed`: what a stream sheds when counted live.
//! - `resident`: the resident records re-emitted before a drop.
//! - `relocate`: lock-bounded batches of relocation.
//! - `result`: applying relocated locations to the index.
//! - `apply`: running one op inline under a held lock.
//! - `tests_batches`: batched ops keep the newest write between batches.
//! - `tests_compaction`: single-segment compaction, and its benchmark.
//! - `tests_policy`: the pure op policy.
//! - `tests_resident`: which resident records a sweep re-emits.
//! - `tests_support`: shared fixtures for the tests.
//! - `tests_survival`: characters, objects and markers survive maintenance.
//! - `tests_tails`: the single snapshot tail through reload and relocation.
//! - `tests_weight`: what counts as live weight.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use super::appended::Appended;
use super::blob::BlobKey;
use super::identity::{MarkerIds, RecordIdentity};
use super::live_indexes::{LiveIndexes, SegmentSizes};
use super::liveness_audit::{keyed_by_payload, resident_key};
use super::manifest::{encode_conv_state_payload, encode_label_payload, ChunkLoc, RecordLoc};
use super::record::{
    DebugIdPayload, DistillMode, DistillPayload, RecordType, SectionTombstonePayload,
    TombstonePayload, TurnCouplingPayload,
};
use super::relocation_watch::{Reemit, RelocationWatch};
use super::retired::RetiredSegments;
use super::segment::SegmentId;
use super::stranded::{Stranded, StrandedKind, StrandedRecord};
use super::streams::{StreamDecl, StreamId};
use super::stripes::{stripes, MAX_STRIPE_BYTES};
use super::survival::RecordCensus;
use super::vfs::carried as vfs_carried;
use super::vfs::VfsEventId;
use super::{
    is_tracked_metadata, metadata_key_parts, PersistenceError, Result, SubstratePersistence,
};
use crate::substrate::{IndexPage, StreamKey, StreamRuntime, Substrate};

/// The streams maintenance carries forward, keyed `(stream_id, writer)`, each
/// mapped to whether its signature records go with it. A stream absent from the
/// map carries nothing.
type CarriedStreams = HashMap<(u64, u64), bool>;

/// The maintenance op chosen for one background pass (§6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaintenanceOp {
    /// Drop a fully-dead sealed segment (no live read-back records).
    Drop(SegmentId),
    /// Relocate one sealed segment's live records into the active, then drop it.
    Compact(SegmentId),
    /// Relocate two adjacent small sealed segments, then drop both.
    Combine(SegmentId, SegmentId),
}

impl MaintenanceOp {
    /// The segment(s) this op relocates-then-drops.
    pub fn targets(&self) -> Vec<SegmentId> {
        match *self {
            MaintenanceOp::Drop(a) | MaintenanceOp::Compact(a) => vec![a],
            MaintenanceOp::Combine(a, b) => vec![a, b],
        }
    }

    /// A short human-readable label for the daemon status / GUI indicator.
    pub fn label(&self) -> String {
        match *self {
            MaintenanceOp::Drop(a) => format!("dropped segment {a}"),
            MaintenanceOp::Compact(a) => format!("compacted segment {a}"),
            MaintenanceOp::Combine(a, b) => format!("combined segments {a}+{b}"),
        }
    }
}

mod apply;
mod plan;
mod policy;
mod relocate;
mod resident;
mod result;
mod shed;
#[cfg(test)]
mod tests_batches;
#[cfg(test)]
mod tests_compaction;
#[cfg(test)]
mod tests_policy;
#[cfg(test)]
mod tests_resident;
#[cfg(test)]
mod tests_support;
#[cfg(test)]
mod tests_survival;
#[cfg(test)]
mod tests_tails;
#[cfg(test)]
mod tests_weight;
mod worklist;

use apply::*;
pub use plan::*;
pub use policy::*;
pub use relocate::*;
pub(super) use resident::*;
pub use result::*;
use shed::*;
pub use worklist::*;
