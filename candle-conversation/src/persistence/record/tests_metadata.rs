//! Tests the byte encoding and tags of the coupling and metadata payloads.
//!
//! The coupling encoding is asserted in exact bytes, because it is durable state,
//! and its wire tag is pinned. Corrupt payloads must be errors rather than panics,
//! so an unknown tag in an older log stays skippable. Tree metadata, debug ids and
//! the model record's wire name all round trip through a real record frame.

use super::*;

#[cfg(test)]
pub(super) mod turn_coupling_tests {
    use super::*;

    /// Exact wire bytes — a coupling is durable state that must decode
    /// identically in every future build, so this asserts the encoding itself,
    /// not merely that it round-trips.
    #[test]
    fn payload_encodes_to_exact_bytes() {
        let p = TurnCouplingPayload {
            timeline_id: 42,
            from_turn: 7,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":42,"from_turn":7}"#.to_vec());
    }

    #[test]
    fn payload_round_trips() {
        let p = TurnCouplingPayload {
            timeline_id: u64::MAX,
            from_turn: u32::MAX,
        };
        assert_eq!(TurnCouplingPayload::decode(&p.encode()).unwrap(), p);
    }

    /// The wire tag is durable: changing it silently reinterprets every existing
    /// coupling record as some other type.
    #[test]
    fn wire_tag_is_nineteen() {
        assert_eq!(RecordType::TurnCoupling.tag(), 19);
        assert_eq!(RecordType::from_tag(19), RecordType::TurnCoupling);
    }

    /// A log written before couplings existed has none of these records, and an
    /// unknown tag must stay skippable rather than abort the walk.
    #[test]
    fn corrupt_payload_is_an_error_not_a_panic() {
        assert!(TurnCouplingPayload::decode(b"not json").is_err());
        assert!(TurnCouplingPayload::decode(b"").is_err());
    }
}

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    #[test]
    fn tree_metadata_payload_round_trips() {
        let p = TreeMetadataPayload {
            timeline_id: 42,
            turn_index: 17,
            kind: 1, // SummaryOfTurns
            tree_height: 1,
            children: vec![10],
        };
        let bytes = p.encode();
        let back = TreeMetadataPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);

        // Through a real record frame.
        let header = RecordHeader {
            record_type: RecordType::TreeMetadata,
            format: 0,
            payload_len: bytes.len() as u64,
            crc: crc32(&bytes),
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let frame = encode_record(&header, &bytes);
        let (hdr2, payload2, _total) = decode_record(&frame).unwrap();
        assert_eq!(hdr2.record_type, RecordType::TreeMetadata);
        let back2 = TreeMetadataPayload::decode(payload2).unwrap();
        assert_eq!(back2, p);
    }

    #[test]
    fn tree_metadata_sos_round_trips() {
        let p = TreeMetadataPayload {
            timeline_id: 7,
            turn_index: 99,
            kind: 2, // SummaryOfSummaries
            tree_height: 4,
            children: vec![50, 75, 88],
        };
        let bytes = p.encode();
        let back = TreeMetadataPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn debug_id_payload_round_trips() {
        let p = DebugIdPayload {
            timeline_id: 123,
            debug_id: "coherent-50".to_string(),
        };
        let bytes = p.encode();
        let back = DebugIdPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn record_header_serialises_tree_metadata_variant() {
        // The new variants must serialise as JSON strings the decoder
        // can round-trip — protects against typo'd `rename_all` collisions.
        let h = RecordHeader {
            record_type: RecordType::TreeMetadata,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let line = serde_json::to_string(&h).unwrap();
        let back: RecordHeader = serde_json::from_str(&line).unwrap();
        assert_eq!(back.record_type, RecordType::TreeMetadata);

        let h2 = RecordHeader {
            record_type: RecordType::DebugId,
            ..h
        };
        let line2 = serde_json::to_string(&h2).unwrap();
        let back2: RecordHeader = serde_json::from_str(&line2).unwrap();
        assert_eq!(back2.record_type, RecordType::DebugId);
    }

    /// The model record keeps the wire name the spec record had, so a header
    /// written before model keys decodes as `ModelKey` rather than `Unknown`.
    #[test]
    fn the_model_record_is_named_model_spec_on_the_wire() {
        let h = RecordHeader {
            record_type: RecordType::ModelKey,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        };
        let js = serde_json::to_string(&h).unwrap();
        assert!(js.contains(r#""type":"model_spec""#), "{js}");
        let older: RecordHeader = serde_json::from_str(&js).unwrap();
        assert_eq!(older.record_type, RecordType::ModelKey);
    }
}
