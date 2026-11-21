//! Fixtures for the compaction tests: records, chunk and snapshot payloads, and
//! an empty store.
//!
//! `record` and `chunk_record` build a header and its bytes for the record type
//! under test. A chunk carries a valid checksum, so reading it back passes
//! verification. `empty_store` gives the tests about the character map a store
//! with no other records.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};

    use crate::persistence::record::encode_record;

    use crate::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;

    /// A clock at the epoch: no tombstone is ever old enough to shrink.
    pub(in super::super) const NO_EXPIRY: TombstoneExpiry = TombstoneExpiry {
        now_ms: 0,
        retention: DEFAULT_TOMBSTONE_RETENTION,
    };

    /// True iff any record of type `rt` carries `payload` (only `Synth` items
    /// hold a payload; `Raw` read-back records don't).
    pub(in super::super) fn has_synth(
        live: &[CompactItem],
        rt: RecordType,
        payload: &[u8],
    ) -> bool {
        live.iter()
            .any(|it| it.header().record_type == rt && it.synth_payload() == Some(payload))
    }

    /// True iff any record of type `rt` is present (by header, `Raw` or `Synth`).
    pub(in super::super) fn has_type(live: &[CompactItem], rt: RecordType) -> bool {
        live.iter().any(|it| it.header().record_type == rt)
    }

    pub(in super::super) fn record(
        rt: RecordType,
        stream_id: u64,
        chunk_index: u64,
        payload: &[u8],
    ) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index,
                token_count: if rt == RecordType::Chunk { 32 } else { 0 },
                ..RecordHeader::NONE
            },
            payload,
        )
    }

    // A valid `Chunk` record whose KV bytes carry `content`, with the golden set
    // (Fletcher-32 over the KV bytes) — so `read_record_at`'s chunk verification
    // passes. Used where a test reads a chunk's payload back; the header-only
    // liveness tests can keep the opaque `record(RecordType::Chunk, …)`.
    pub(in super::super) fn chunk_record(
        stream_id: u64,
        chunk_index: u64,
        content: &[u8],
    ) -> Vec<u8> {
        use crate::persistence::record::ChunkPayload;
        let cp = ChunkPayload {
            offset: 0,
            k_formats: vec![4],
            v_formats: vec![5],
            k_pal: vec![],
            v_pal: vec![],
            k_scale: vec![],
            v_scale: vec![],
            kv_bytes: content.to_vec(),
        };
        let payload = cp.encode();
        encode_record(
            &RecordHeader {
                record_type: RecordType::Chunk,
                format: 4,
                payload_len: payload.len() as u64,
                crc: candle::fletcher::fletcher32(&cp.kv_bytes),
                stream_id,
                chunk_index,
                token_count: 32,
                ..RecordHeader::NONE
            },
            &payload,
        )
    }

    /// Encoded snapshot payload for `timeline` at `turn` with a recognisable
    /// `fill` byte — the same fixture shape the maintenance relocation tests
    /// use, kept local because that module's test helpers are private to it.
    pub(in super::super) fn snapshot_payload(timeline: u64, turn: u32, fill: u8) -> Vec<u8> {
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};
        SnapshotPayload {
            timeline_id: timeline,
            turn_index: turn,
            schedule_hash: 0xA5A5,
            layers: vec![SnapshotLayer {
                layer_index: 0,
                n_v_heads: 1,
                d_v: 4,
                d_k: 4,
                state: vec![fill; 64],
                conv_channels: 2,
                conv_tail_cols: 2,
                conv_tail: vec![fill; 16],
            }],
            // Filled, not empty: a relocation that carried the layers and
            // dropped the model's own state would still pass every assertion
            // that only inspects the layers.
            aux: vec![fill; 24],
        }
        .encode()
    }

    /// The failure this guards against is silent and total: NPC payloads are
    /// not in this process's RAM, so nothing can re-synthesise them, and a
    /// compaction that does not carry them forward deletes every character in
    /// the store without an error, a warning, or a failing test anywhere else.
    /// An empty store, for tests whose subject is the NPC map rather than the
    /// log. `collect_live_records` is planning-only and reads no bytes, so the
    /// locations it echoes need not point at anything real.
    pub(in super::super) fn empty_store() -> (Manifest, Substrate) {
        let mut mem = MemLog::with_records(&[]);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        (manifest, substrate)
    }

    pub(in super::super) fn loc_at(offset: u64, size: u64) -> RecordLoc {
        RecordLoc {
            segment: FIRST_SEGMENT,
            offset,
            payload_len: size - 64,
            record_size: size,
        }
    }
}
