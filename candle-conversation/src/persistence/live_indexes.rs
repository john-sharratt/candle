//! The log's location indexes as one value maintenance plans against, taken
//! on the log's owner and read on the caller's thread
//! (`docs/substrate_lock_free.md` §9.5).
//!
//! Choosing a maintenance op is a liveness scan of every live record, and
//! planning it walks the same records again for what the op carries. Both read
//! the substrate — a version any thread holds without waiting — and the owner's
//! own indexes: where each stream's metadata, each resident record, each
//! character, each file event and each blob currently is. Run on the owner,
//! the scan held every write, replication apply and cold load behind it for as
//! long as it took: 493 s on WhiteWorld on 2026-11-01, which stalled the shared
//! engine's wave and every character's turn with it.
//!
//! Every index here is a persistent map, so taking this value is a handful of
//! pointer copies. The owner hands it over in one short call; the scan and the
//! plan run on the maintenance thread against it. A write after it was taken is
//! not in it, which is what the relocation watch is for: it starts in the same
//! call the plan's indexes are taken in, so every write the plan cannot see is
//! one the relocation steps around.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::blob::BlobIndex;
use super::dead_ratio::dead_ratio_of;
use super::identity::MarkerIds;
use super::liveness_audit::ResidentKey;
use super::log_facts::LogFacts;
use super::manifest::Manifest;
use super::rebase::RebaseMap;
use super::replica::ReplicaPayload;
use super::segment::SegmentId;
use super::stamped_locs::StampedLocs;
use super::stranded::Stranded;
use super::vfs::VfsIndex;
use super::{MetadataKey, Result, SubstratePersistence};

/// The owner's location indexes, as they stood when taken. See the module.
#[derive(Clone)]
pub struct LiveIndexes {
    pub(super) node: u64,
    pub(super) read_only: bool,
    pub(super) dir: PathBuf,
    pub(super) facts: LogFacts,
    pub(super) manifest: Manifest,
    pub(super) metadata_locs: StampedLocs<MetadataKey>,
    pub(super) resident_locs: StampedLocs<ResidentKey>,
    pub(super) npc_locs: StampedLocs<u64>,
    pub(super) vfs_index: VfsIndex,
    pub(super) blob_index: BlobIndex,
    pub(super) marker_ids: MarkerIds,
    /// The `Replica` record an op re-emits, as it would be written now.
    pub(super) replica: ReplicaPayload,
    pub(super) rebases: RebaseMap,
    /// Segments an earlier op kept for a record the index named there; an op
    /// is chosen around the ones whose record the index still names.
    pub(super) stranded: Stranded,
}

/// One sealed segment's size and age, as choosing an op weighs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SealedSize {
    pub id: SegmentId,
    /// Record bytes on disk, the superblock excluded.
    pub total_bytes: u64,
    /// Seconds since the segment's file was last written.
    pub age_secs: u64,
}

/// Every sealed segment's size and age, and the record bytes of the whole log
/// — what choosing an op needs from the segment set beside the indexes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SegmentSizes {
    pub sealed: Vec<SealedSize>,
    pub total_record_bytes: u64,
}

impl LiveIndexes {
    /// The node the indexes' orphan gates judge by.
    pub fn node(&self) -> u64 {
        self.node
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// The log's directory, which the plan's line names.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Publish the dead ratio of `total` record bytes of which `live` are live,
    /// as the log's own scan does, and answer it.
    pub(super) fn note_dead_ratio(&self, total: u64, live: u64) -> f32 {
        let ratio = dead_ratio_of(total, live);
        self.facts.set_dead_ratio(ratio);
        ratio
    }

    /// The `Replica` record an op re-emits.
    pub(super) fn replica_record(&self) -> ReplicaPayload {
        self.replica.clone()
    }
}

impl SubstratePersistence {
    /// The location indexes as they stand, for a scan or a plan off the
    /// owner. O(1) in the indexes' size: each is a persistent map.
    pub fn live_indexes(&self) -> LiveIndexes {
        LiveIndexes {
            node: self.node(),
            read_only: self.is_read_only(),
            dir: self.dir().to_path_buf(),
            facts: self.facts.clone(),
            manifest: self.manifest.clone(),
            metadata_locs: self.metadata_locs.clone(),
            resident_locs: self.resident_locs.clone(),
            npc_locs: self.npc_locs.clone(),
            vfs_index: self.vfs_index.clone(),
            blob_index: self.blob_index.clone(),
            marker_ids: self.marker_ids.clone(),
            replica: self.replica_record(),
            rebases: self.rebases.clone(),
            stranded: self.stranded.clone(),
        }
    }

    /// Every sealed segment's size and age, and the log's record bytes. One
    /// file-metadata read per sealed segment; nothing is scanned.
    pub fn segment_sizes(&self) -> Result<SegmentSizes> {
        let now = SystemTime::now();
        let mut sealed = Vec::new();
        for &id in self.segments.sealed_ids() {
            sealed.push(SealedSize {
                id,
                total_bytes: self.segments.sealed_record_bytes(id)?,
                age_secs: self.segments.sealed_age_secs(id, now)?,
            });
        }
        Ok(SegmentSizes {
            sealed,
            total_record_bytes: self.segments.total_record_bytes()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::Substrate;

    /// **What the indexes were when taken is what the value holds**: an
    /// append on the owner after it was taken reaches the log's indexes and
    /// not the value, so a plan against it never sees half a write.
    #[test]
    fn a_write_after_the_take_is_not_in_the_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut p =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        p.write_section_tombstone(0x5000, None, None).unwrap();
        p.commit().unwrap();
        let taken = p.live_indexes();
        let held = taken.resident_locs.len();
        assert_eq!(held, 1, "the first tombstone is located");

        p.write_section_tombstone(0x5001, None, None).unwrap();
        p.commit().unwrap();
        assert_eq!(p.live_indexes().resident_locs.len(), 2);
        assert_eq!(taken.resident_locs.len(), held, "the value is unchanged");
        assert_eq!(taken.node(), p.node());
        assert_eq!(taken.dir(), p.dir());
    }

    /// **The sizes name every sealed segment**, and none while nothing is
    /// sealed.
    #[test]
    fn the_sizes_name_every_sealed_segment() {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut p =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        assert!(p.segment_sizes().unwrap().sealed.is_empty());
        p.write_section_tombstone(0x5000, None, None).unwrap();
        p.commit().unwrap();
        p.seal_active().unwrap();
        let sizes = p.segment_sizes().unwrap();
        assert_eq!(sizes.sealed.len(), 1);
        assert!(sizes.sealed[0].total_bytes > 0);
        assert!(sizes.total_record_bytes >= sizes.sealed[0].total_bytes);
    }
}
