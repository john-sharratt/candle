//! Tests the byte layout and the refusals of the snapshot and checkpoint payloads.
//!
//! The snapshot's encoding is pinned byte for byte, and its pre-sized length must
//! match what it writes. A truncated aux blob and a version-one record are refused,
//! since each would otherwise read as a plausible state. The checkpoint's header is
//! pinned too, and the two payloads are shown not to decode as one another.

use super::*;

#[cfg(test)]
pub(super) mod snapshot_payload_tests {
    use super::*;

    pub(in super::super) fn tiny() -> SnapshotPayload {
        SnapshotPayload {
            timeline_id: 7,
            turn_index: 3,
            schedule_hash: 0xDEAD_BEEF_CAFE_F00D,
            layers: vec![SnapshotLayer {
                layer_index: 1,
                n_v_heads: 1,
                d_v: 2,
                d_k: 2,
                state: vec![0u8; 16],
                conv_channels: 3,
                conv_tail_cols: 1,
                conv_tail: vec![1u8; 12],
            }],
            aux: vec![9u8; 5],
        }
    }

    /// Byte-exact pin: durable state must decode identically forever.
    #[test]
    fn encode_is_byte_stable() {
        let bytes = tiny().encode();
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(&2u32.to_le_bytes()); // version
        expect.extend_from_slice(&7u64.to_le_bytes()); // timeline
        expect.extend_from_slice(&3u32.to_le_bytes()); // turn
        expect.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes());
        expect.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        expect.extend_from_slice(&1u32.to_le_bytes()); // layer_index
        expect.push(0); // dtype
        expect.extend_from_slice(&1u32.to_le_bytes()); // n_v_heads
        expect.extend_from_slice(&2u32.to_le_bytes()); // d_v
        expect.extend_from_slice(&2u32.to_le_bytes()); // d_k
        expect.extend_from_slice(&16u32.to_le_bytes()); // state blob len
        expect.extend_from_slice(&[0u8; 16]);
        expect.extend_from_slice(&3u32.to_le_bytes()); // conv_channels
        expect.extend_from_slice(&1u32.to_le_bytes()); // conv_tail_cols
        expect.extend_from_slice(&12u32.to_le_bytes()); // tail blob len
        expect.extend_from_slice(&[1u8; 12]);
        expect.extend_from_slice(&5u32.to_le_bytes()); // aux blob len
        expect.extend_from_slice(&[9u8; 5]);
        assert_eq!(bytes, expect);
    }

    /// The parts a seal writes from borrow the state and conv-tail blobs — the
    /// whole point of writing from parts — and frame to the same record image
    /// `encode_record` builds from the joined payload, CRC included.
    #[test]
    fn the_parts_borrow_the_state_and_frame_to_the_encoded_record() {
        let p = tiny();
        let parts = p.parts();
        let slices = parts.slices();
        assert!(slices
            .iter()
            .any(|s| std::ptr::eq(s.as_ptr(), p.layers[0].state.as_ptr())));
        assert!(slices
            .iter()
            .any(|s| std::ptr::eq(s.as_ptr(), p.layers[0].conv_tail.as_ptr())));

        let header = RecordHeader {
            record_type: RecordType::Snapshot,
            format: 0,
            payload_len: p.encoded_len() as u64,
            crc: 0,
            stream_id: 4,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let head = record_head(&header, &slices);
        let mut framed = head.line;
        for s in &slices {
            framed.extend_from_slice(s);
        }
        framed.resize(framed.len() + head.pad, 0);
        assert_eq!(framed, encode_record(&header, &p.encode()));
    }

    /// The pre-sized length is exactly what `encode` writes — 98 bytes for
    /// `tiny`: a 28-byte header, a 61-byte layer and a 9-byte aux blob, counted
    /// field by field in `encode_is_byte_stable` above.
    #[test]
    fn encoded_len_is_exactly_what_encode_writes() {
        let p = tiny();
        assert_eq!(p.encoded_len(), 98);
        assert_eq!(p.encode().len(), p.encoded_len());
        let mut empty = tiny();
        empty.layers.clear();
        empty.aux.clear();
        assert_eq!(empty.encode().len(), empty.encoded_len());
    }

    #[test]
    fn roundtrip_and_dim_validation() {
        let p = tiny();
        let back = SnapshotPayload::decode(&p.encode()).unwrap();
        assert_eq!(p, back);

        // A state blob that disagrees with its dims is corrupt, not clamped.
        let mut bad = tiny();
        bad.layers[0].state.pop();
        let err = SnapshotPayload::decode(&bad.encode()).unwrap_err();
        assert!(err.to_string().contains("state blob"));
    }

    /// A model with no auxiliary state writes an empty blob, and it round-trips
    /// as empty rather than as a missing field.
    #[test]
    fn an_absent_aux_blob_round_trips_as_empty() {
        let mut p = tiny();
        p.aux.clear();
        let back = SnapshotPayload::decode(&p.encode()).unwrap();
        assert!(back.aux.is_empty());
        assert_eq!(p, back);
    }

    /// The aux blob is the record's LAST field, so a truncation that loses it
    /// is the one corruption a length-prefixed reader can meet at the end of
    /// the buffer. Refused, not read as empty: an empty blob and a lost blob
    /// mean opposite things — no state to install, versus state that was
    /// installed and is now silently gone.
    #[test]
    fn a_truncated_aux_blob_is_refused() {
        let bytes = tiny().encode();
        for cut in 1..=6 {
            let err = SnapshotPayload::decode(&bytes[..bytes.len() - cut]).unwrap_err();
            assert!(err.to_string().contains("truncated input"), "{err}");
        }
    }

    /// **A record written before `aux` existed is REFUSED, not misread.**
    ///
    /// This is the one case a live substrate actually meets: version 1 had no
    /// trailing blob, so under this layout its bytes end exactly where the
    /// length word would begin. Read leniently it would either take four bytes
    /// of whatever followed as a length, or — for a record that happened to sit
    /// at the end of a segment — decode to a plausible state with an empty aux,
    /// which is the same shape as "this model carries no auxiliary state" and
    /// is indistinguishable from it.
    ///
    /// The refusal is what makes an old substrate degrade honestly:
    /// `read_recurrent_snapshot` returns the error,
    /// `Scheduler::restore_recurrent_state` logs RECURRENT RESUME FAILED and
    /// continues, and the conversation opens with no memory of its history
    /// instead of a wrong one.
    #[test]
    fn a_version_one_record_is_refused() {
        // Version 1 is this payload's bytes with the version word set to 1 and
        // the trailing aux blob absent.
        let mut v1 = tiny().encode();
        v1.truncate(v1.len() - (4 + 5)); // aux length word + aux bytes
        v1[..4].copy_from_slice(&1u32.to_le_bytes());

        let err = SnapshotPayload::decode(&v1).unwrap_err();
        assert!(
            err.to_string().contains("version 1"),
            "a version-1 record must be named as such, got: {err}"
        );
    }

    #[test]
    fn wire_tag_is_twenty() {
        assert_eq!(RecordType::Snapshot as u8, 20);
        assert_eq!(RecordType::from_tag(20), RecordType::Snapshot);
    }

    /// The branch checkpoint's own tag. Durable, so it is pinned like every
    /// other: a tag that moved would make old records decode as a different
    /// kind of state.
    #[test]
    fn branch_checkpoint_wire_tag_is_twenty_one() {
        assert_eq!(RecordType::BranchCheckpoint as u8, 21);
        assert_eq!(RecordType::from_tag(21), RecordType::BranchCheckpoint);
        assert_ne!(RecordType::BranchCheckpoint, RecordType::Snapshot);
    }
}

#[cfg(test)]
pub(super) mod branch_checkpoint_tests {
    use super::*;

    /// One layer at the 35B's real DeltaNet geometry — 32 V heads, d_v 128,
    /// d_k 128, conv_dim 8192 with a 3-column tail. 2 MiB of state and 96 KiB of
    /// tail per layer, which is what makes the whole-stack figure ~63 MiB.
    pub(in super::super) fn real_layer(idx: u32) -> SnapshotLayer {
        SnapshotLayer {
            layer_index: idx,
            n_v_heads: 32,
            d_v: 128,
            d_k: 128,
            state: vec![idx as u8; 32 * 128 * 128 * 4],
            conv_channels: 8192,
            conv_tail_cols: 3,
            conv_tail: vec![!(idx as u8); 8192 * 3 * 4],
        }
    }

    pub(in super::super) fn real_checkpoint() -> BranchCheckpointPayload {
        BranchCheckpointPayload {
            prefix_hash: ContentHash {
                lo: 0x0123_4567_89AB_CDEF,
                hi: 0xFEDC_BA98_7654_3210,
            },
            schedule_hash: 0xDEAD_BEEF_CAFE_F00D,
            layers: (0..30).map(real_layer).collect(),
            aux: vec![0xABu8; 4096],
        }
    }

    /// Byte-exact header pin. The body reuses the layer encoding
    /// [`SnapshotPayload`] already pins byte-for-byte, so only the fields this
    /// payload adds need their own assertion — and they need it for the same
    /// reason: this is durable state, and a silently-shifted field decodes as a
    /// plausible state rather than as an error.
    #[test]
    fn header_encodes_to_exact_bytes() {
        let p = real_checkpoint();
        let bytes = p.encode();
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(b"BRCK"); // magic
        expect.extend_from_slice(&2u32.to_le_bytes()); // version
        expect.extend_from_slice(&0x0123_4567_89AB_CDEFu64.to_le_bytes()); // prefix lo
        expect.extend_from_slice(&0xFEDC_BA98_7654_3210u64.to_le_bytes()); // prefix hi
        expect.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes()); // schedule
        expect.extend_from_slice(&30u32.to_le_bytes()); // n_layers
        assert_eq!(&bytes[..expect.len()], &expect[..]);
    }

    /// Round-trip at the real geometry, byte-identical. Not a tolerance: the
    /// state is copied, never recomputed, so any difference is a layout bug.
    #[test]
    fn round_trips_at_real_35b_geometry() {
        let p = real_checkpoint();
        let back = BranchCheckpointPayload::decode(&p.encode()).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.layers.len(), 30, "every recurrent layer survives");
        assert_eq!(
            back.aux.len(),
            4096,
            "and so does the model's own state, which for an architecture that \
             keeps its whole recurrence there is the entire checkpoint"
        );
    }

    /// A truncated state blob is corrupt, not clamped — restoring a partial
    /// state would seed a conversation with a prompt memory that is real for
    /// some layers and zero for others, which is invisible from the outside.
    #[test]
    fn a_short_state_blob_is_refused() {
        let mut bad = real_checkpoint();
        bad.layers[0].state.pop();
        let err = BranchCheckpointPayload::decode(&bad.encode()).unwrap_err();
        assert!(err.to_string().contains("state blob"), "{err}");
    }

    /// The two payloads that share `RecordType::Snapshot` do **not** decode as
    /// each other.
    ///
    /// They are told apart by stream id, so nothing routes a conversation
    /// snapshot into a branch read today. This is the guard for the day
    /// something does — and it is not hypothetical tidiness: without the magic
    /// byte the two are wire-*compatible*. `(version, timeline, turn, schedule,
    /// n_layers)` and `(version, prefix_lo, prefix_hi, schedule, n_layers)` are
    /// the same widths in the same order, so a conversation snapshot decoded as
    /// a branch checkpoint cleanly, with every field landing somewhere
    /// plausible. The first version of this test asserted the refusal and
    /// failed, which is how the magic came to exist.
    #[test]
    fn the_two_snapshot_payloads_do_not_decode_as_each_other() {
        let conv = SnapshotPayload {
            timeline_id: 7,
            turn_index: 3,
            schedule_hash: 11,
            layers: vec![real_layer(0)],
            aux: Vec::new(),
        };
        let err = BranchCheckpointPayload::decode(&conv.encode()).unwrap_err();
        assert!(err.to_string().contains("not a branch checkpoint"), "{err}");
        assert!(SnapshotPayload::decode(&real_checkpoint().encode()).is_err());
    }
}
