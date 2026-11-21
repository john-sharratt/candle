//! Tests the chunk golden and the chunk and distillation payload codecs.
//!
//! The golden must cover the KV bytes only, so flipping the host prefix leaves it
//! unchanged. Chunk records are excluded from the metadata CRC check. Round trips
//! must be exact, and a distillation marker written before the mode field decodes
//! to its original behaviour.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    #[test]
    fn recompute_chunk_golden_tracks_kv_not_prefix() {
        // The golden is Fletcher-32 over the KV bytes only.
        let cp = ChunkPayload {
            offset: 3,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
            k_pal: vec![0xAA; 3],
            v_pal: vec![0x55; 3],
            k_scale: vec![1.0, 2.0],
            v_scale: vec![3.0],
            kv_bytes: (0..96u32).map(|i| (i * 7 + 1) as u8).collect(),
        };
        let payload = cp.encode();
        let golden = candle::fletcher::fletcher32(&cp.kv_bytes);
        assert_eq!(recompute_chunk_golden(&payload).unwrap(), golden);

        // Flipping a KV byte changes the golden (a mismatch would be flagged).
        let (_, kv_range) = ChunkPayload::decode_with_kv_range(&payload).unwrap();
        let mut kv_flip = payload.clone();
        kv_flip[kv_range.start] ^= 0x01;
        assert_ne!(recompute_chunk_golden(&kv_flip).unwrap(), golden);

        // Flipping the host-authored prefix (the `offset` low byte) leaves the
        // golden unchanged — it covers only the arena KV bytes.
        let mut prefix_flip = payload.clone();
        prefix_flip[0] ^= 0x01;
        assert_eq!(recompute_chunk_golden(&prefix_flip).unwrap(), golden);
    }

    #[test]
    fn verify_record_crc_skips_chunk_records() {
        // Chunk integrity is the golden's job (checked at write/read), not this
        // metadata verifier — a chunk with a deliberately wrong crc still passes.
        let cp = ChunkPayload {
            offset: 0,
            k_formats: vec![4],
            v_formats: vec![5],
            k_pal: vec![],
            v_pal: vec![],
            k_scale: vec![],
            v_scale: vec![],
            kv_bytes: vec![1, 2, 3, 4],
        };
        let payload = cp.encode();
        let header = RecordHeader {
            record_type: RecordType::Chunk,
            format: 4,
            payload_len: payload.len() as u64,
            crc: 0xDEAD_BEEF,
            stream_id: 1,
            chunk_index: 0,
            token_count: 32,
            ..RecordHeader::NONE
        };
        assert!(verify_record_crc(&header, &payload).is_ok());
    }

    #[test]
    fn chunk_payload_roundtrip() {
        let payload = ChunkPayload {
            offset: 4,
            k_formats: vec![2, 7, 2, 7],
            v_formats: vec![7, 7, 7, 7],
            k_pal: vec![1, 2, 3, 4, 5],
            v_pal: vec![9, 8, 7],
            k_scale: vec![0.5, 1.0, 2.0],
            v_scale: vec![3.25],
            kv_bytes: (0..1000u32).map(|i| (i % 256) as u8).collect(),
        };
        let bytes = payload.encode();
        assert_eq!(ChunkPayload::decode(&bytes).unwrap(), payload);
    }

    #[test]
    fn chunk_payload_empty_metadata_roundtrip() {
        let payload = ChunkPayload {
            offset: 0,
            k_formats: Vec::new(),
            v_formats: Vec::new(),
            k_pal: Vec::new(),
            v_pal: Vec::new(),
            k_scale: Vec::new(),
            v_scale: Vec::new(),
            kv_bytes: vec![0xAB; 64],
        };
        assert_eq!(ChunkPayload::decode(&payload.encode()).unwrap(), payload);
    }

    #[test]
    fn chunk_payload_rejects_trailing_bytes() {
        let payload = ChunkPayload {
            offset: 1,
            k_formats: vec![1, 1, 1, 1],
            v_formats: vec![1, 1, 1, 1],
            k_pal: vec![1],
            v_pal: vec![2],
            k_scale: vec![1.0],
            v_scale: vec![1.0],
            kv_bytes: vec![3],
        };
        let mut bytes = payload.encode();
        bytes.push(0);
        assert!(matches!(
            ChunkPayload::decode(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }

    #[test]
    fn distill_payload_roundtrips_both_modes() {
        for mode in [DistillMode::ProvenanceOnly, DistillMode::TextOnly] {
            let p = DistillPayload {
                timeline_id: 42,
                mode,
            };
            assert_eq!(DistillPayload::decode(&p.encode()).unwrap(), p);
        }
    }

    #[test]
    fn distill_payload_pre_mode_records_decode_provenance_only() {
        // Records written before `mode` existed carry only `timeline_id`; the
        // serde default keeps them decoding to the original behaviour so live
        // substrates (whose calibration corpus is already distilled) load.
        let legacy = br#"{"timeline_id":7}"#;
        let decoded = DistillPayload::decode(legacy).unwrap();
        assert_eq!(decoded.timeline_id, 7);
        assert_eq!(decoded.mode, DistillMode::ProvenanceOnly);
    }
}
