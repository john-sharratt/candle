//! Records where each kind of record currently lives, so segment liveness counts the current copy and maintenance can find it.
//!
//! A metadata or resident record's location is kept by the greatest stamp, so a re-write supersedes the prior location and a maintenance re-emission keeps the identity while moving the record. Snapshot locations are offered by both the append path and the relocation path, and a retraction withdraws the tail. Each append path reports where its record landed, so the map follows the record.

use super::*;

impl SubstratePersistence {
    /// The location of a record just placed at `(segment, offset)`.
    pub(super) fn placed_loc(
        h: &RecordHeader,
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) -> RecordLoc {
        RecordLoc {
            segment,
            offset,
            payload_len: h.payload_len,
            record_size: size,
        }
    }

    /// The runtime twin of [`record_resident_loc`]: offer where a just-appended
    /// payload-keyed resident record landed — the greatest stamp per key holds.
    pub(super) fn track_resident_loc(
        &mut self,
        h: &RecordHeader,
        payload: &[u8],
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) {
        if !keyed_by_payload(h.record_type) {
            return;
        }
        let id = h.identity();
        if let Some(key) = resident_key(h.record_type, h.stream_id, payload, &id) {
            self.resident_locs
                .offer(key, Some(Self::placed_loc(h, segment, offset, size)), id);
        }
    }

    /// Offer the on-disk location of a per-stream metadata record
    /// (`StreamDecl` / `ProjectionEvents` / `WideQSig` / `Commit`) so
    /// [`segment_liveness`](Self::segment_liveness) counts the current copy as
    /// live weight. The greatest stamp per [`MetadataKey`] holds: a
    /// re-write supersedes the prior location, which then reads as dead, and a
    /// maintenance re-emission keeps the identity and moves it. A no-op for
    /// every other record type. Called from
    /// [`append_record`](Self::append_record) (runtime writes) and the
    /// load / compact walk so the map covers both.
    pub(super) fn track_metadata_loc(
        &mut self,
        h: &RecordHeader,
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) {
        if is_tracked_metadata(h.record_type) {
            self.metadata_locs.offer(
                metadata_key(h),
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }

    /// Offer the on-disk location of a `Snapshot` record to
    /// [`snapshot_locs`](Self::snapshot_locs) — a retraction withdraws it, and
    /// the greatest stamp per snapshot stream holds, so the map always names
    /// the stream's current tail. A no-op for every other record type. Called
    /// from both append paths ([`append_record`](Self::append_record) and
    /// [`append_raw_record`](Self::append_raw_record)) so runtime snapshot
    /// writes AND maintenance relocations keep the map current; the load /
    /// compact walk uses the free-function mirror [`record_snapshot_loc`].
    pub(super) fn track_snapshot_loc(
        &mut self,
        h: &RecordHeader,
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) {
        if h.retracts_snapshot() {
            self.snapshot_locs.offer(h.stream_id, None, h.identity());
        } else if matches!(
            h.record_type,
            RecordType::Snapshot | RecordType::BranchCheckpoint
        ) {
            self.snapshot_locs.offer(
                h.stream_id,
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }

    /// The runtime-append twin of [`record_npc_loc`].
    pub(super) fn track_npc_loc(
        &mut self,
        h: &RecordHeader,
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) {
        if h.record_type == RecordType::Npc {
            self.npc_locs.offer(
                h.stream_id,
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }
}
