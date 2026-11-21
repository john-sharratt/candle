//! Tests the bytes of timeline, turn and section tombstone payloads.
//!
//! A tombstone with no reason must match the format written before reasons existed.
//! A turn tombstone's key and layout are pinned, since a serde change could demote
//! it to a timeline tombstone. The section tombstone names its generation ahead of
//! its reason, and an older record without a reason decodes to none.

use super::*;

#[cfg(test)]
pub(super) mod tombstone_payload_tests {
    use super::TombstonePayload;

    #[test]
    fn reasonless_tombstone_is_byte_identical_to_pre_reason_format() {
        // A None reason must serialise to exactly `{"timeline_id":N}` so existing
        // on-disk tombstone records (written before the reason field) read back
        // identically and new reason-less tombstones don't change the byte layout.
        let p = TombstonePayload {
            timeline_id: 77,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":77}"#.to_vec());
    }

    #[test]
    fn reason_is_serialised_when_present() {
        let p = TombstonePayload {
            timeline_id: 42,
            turn_index: None,
            turn_writer: 0,
            reason: Some("corrupt reload (turn 1): chunk mismatch".to_string()),
        };
        assert_eq!(
            p.encode(),
            br#"{"timeline_id":42,"reason":"corrupt reload (turn 1): chunk mismatch"}"#.to_vec()
        );
    }

    #[test]
    fn old_record_without_reason_decodes_to_none() {
        // A record persisted before the field existed still decodes cleanly.
        let decoded = TombstonePayload::decode(br#"{"timeline_id":9}"#).unwrap();
        assert_eq!(decoded.timeline_id, 9);
        assert_eq!(decoded.reason, None);
    }

    #[test]
    fn round_trips_with_reason() {
        let p = TombstonePayload {
            timeline_id: 5,
            turn_index: None,
            turn_writer: 0,
            reason: Some("corrupt reload".to_string()),
        };
        assert_eq!(TombstonePayload::decode(&p.encode()).unwrap(), p);
    }

    #[test]
    fn turn_scoped_tombstone_bytes_are_pinned() {
        // The turn-scoped form is the reload's ONLY signal to hole-restore the
        // turn instead of renumbering the timeline — pin its exact on-disk key
        // and layout so a serde tweak can't silently demote it to a
        // timeline-level tombstone (which drops the whole conversation).
        let p = TombstonePayload {
            timeline_id: 77,
            turn_index: Some(2),
            turn_writer: 0,
            reason: None,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":77,"turn_index":2}"#.to_vec());
        assert_eq!(TombstonePayload::decode(&p.encode()).unwrap(), p);
        // And a reasoned turn tombstone keeps field order stable.
        let q = TombstonePayload {
            timeline_id: 3,
            turn_index: Some(9),
            turn_writer: 0,
            reason: Some("corrupt reload (turn 9): chunk mismatch".to_string()),
        };
        assert_eq!(
            q.encode(),
            br#"{"timeline_id":3,"turn_index":9,"reason":"corrupt reload (turn 9): chunk mismatch"}"#
                .to_vec()
        );
        assert_eq!(TombstonePayload::decode(&q.encode()).unwrap(), q);
    }

    /// A turn tombstone names its turn's writer beside the index that writer
    /// gave it, between the two in the record.
    #[test]
    fn a_turn_tombstone_names_its_writer() {
        let p = TombstonePayload {
            timeline_id: 77,
            turn_index: Some(2),
            turn_writer: 0xab,
            reason: None,
        };
        assert_eq!(
            p.encode(),
            br#"{"timeline_id":77,"turn_index":2,"turn_writer":171}"#.to_vec()
        );
        assert_eq!(TombstonePayload::decode(&p.encode()).unwrap(), p);
    }
}

#[cfg(test)]
pub(super) mod section_tombstone_payload_tests {
    use super::{SectionGeneration, SectionTombstonePayload};

    #[test]
    fn reasonless_payload_is_byte_identical_minimal_form() {
        let p = SectionTombstonePayload {
            generation: None,
            reason: None,
        };
        assert_eq!(p.encode(), br#"{}"#.to_vec());
    }

    /// A tombstone names the generation it ends, ahead of its reason.
    #[test]
    fn the_generation_is_serialised_when_named() {
        let p = SectionTombstonePayload {
            generation: Some(SectionGeneration {
                writer: 171,
                oseq: 9,
            }),
            reason: Some("rebuild".to_string()),
        };
        assert_eq!(
            p.encode(),
            br#"{"generation":{"writer":171,"oseq":9},"reason":"rebuild"}"#.to_vec()
        );
        assert_eq!(SectionTombstonePayload::decode(&p.encode()).unwrap(), p);
    }

    #[test]
    fn reason_is_serialised_when_present() {
        let p = SectionTombstonePayload {
            generation: None,
            reason: Some(
                "chunk 2: (offset 0, 32 tokens) on layers [0,1]; (offset 0, 0 tokens) \
                          on layers [2..47]"
                    .to_string(),
            ),
        };
        assert_eq!(
            p.encode(),
            br#"{"reason":"chunk 2: (offset 0, 32 tokens) on layers [0,1]; (offset 0, 0 tokens) on layers [2..47]"}"#.to_vec()
        );
    }

    #[test]
    fn round_trips() {
        let p = SectionTombstonePayload {
            generation: None,
            reason: Some("window divergence".to_string()),
        };
        assert_eq!(SectionTombstonePayload::decode(&p.encode()).unwrap(), p);
        let q = SectionTombstonePayload {
            generation: None,
            reason: None,
        };
        assert_eq!(SectionTombstonePayload::decode(&q.encode()).unwrap(), q);
    }

    #[test]
    fn old_record_without_reason_decodes_to_none() {
        let decoded = SectionTombstonePayload::decode(b"{}").unwrap();
        assert_eq!(decoded.reason, None);
    }
}
