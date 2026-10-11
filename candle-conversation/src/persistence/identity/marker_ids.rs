//! The identity of each marker record a rewrite writes back.
//!
//! Markers — a timeline or turn `Tombstone`, a `SectionTombstone`, a
//! `Distilled` marker — are facts about what they name, not values, so the
//! substrate holds them as flags without a record behind them. When a rewrite
//! re-encodes one, the copy must keep the identity of the record it copies:
//! the same `origin` and `oseq`, so every member still recognises it, and the
//! same stamp, so it ranks where it did. This table holds that identity per
//! marker key — the greatest stamp that stated the marker, so it is the same
//! whatever order the records arrived in.
//!
//! The substrate keeps the identity of every keyed *value* beside the value
//! ([`crate::substrate::StreamIds`], [`crate::substrate::TimelineIds`]); this
//! table is the markers' counterpart.

use std::collections::HashMap;

use super::super::liveness_audit::{resident_key, ResidentKey};
use super::super::record::RecordType;
use super::record_identity::RecordIdentity;

/// Whether `rt` is a marker this table keeps.
pub fn is_marker(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::Tombstone | RecordType::SectionTombstone | RecordType::Distilled
    )
}

/// Identity per marker key.
#[derive(Clone, Debug, Default)]
pub struct MarkerIds {
    ids: HashMap<ResidentKey, RecordIdentity>,
}

impl MarkerIds {
    pub fn new() -> MarkerIds {
        MarkerIds::default()
    }

    /// Record that the marker `(rt, stream_id, payload)` was written as `id`.
    /// A type that is not a marker, or a payload that does not decode, is
    /// ignored; of two records stating one marker, the greater stamp is kept.
    pub fn record(&mut self, rt: RecordType, stream_id: u64, payload: &[u8], id: RecordIdentity) {
        if !is_marker(rt) {
            return;
        }
        if let Some(key) = resident_key(rt, stream_id, payload, &id) {
            let held = self.ids.entry(key).or_default();
            if id.supersedes(held) {
                *held = id;
            }
        }
    }

    /// The identity a re-emission of the marker `(rt, stream_id, payload)`
    /// carries, or none for a marker no stamped record stated.
    pub fn identity_for(&self, rt: RecordType, stream_id: u64, payload: &[u8]) -> RecordIdentity {
        resident_key(rt, stream_id, payload, &RecordIdentity::NONE)
            .and_then(|key| self.ids.get(&key).copied())
            .unwrap_or(RecordIdentity::NONE)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::{SectionTombstonePayload, TombstonePayload};

    fn id(oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 3,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        }
    }

    fn tombstone(timeline_id: u64) -> Vec<u8> {
        TombstonePayload {
            timeline_id,
            turn_index: None,
            reason: None,
        }
        .encode()
    }

    /// The greatest record stating a marker is the one a re-emission carries,
    /// whichever arrived last.
    #[test]
    fn the_greatest_record_of_a_marker_is_kept() {
        let mut t = MarkerIds::new();
        t.record(RecordType::Tombstone, 0, &tombstone(9), id(2));
        t.record(RecordType::Tombstone, 0, &tombstone(9), id(1));
        t.record(RecordType::Tombstone, 0, &tombstone(10), id(3));
        assert_eq!(
            t.identity_for(RecordType::Tombstone, 0, &tombstone(9)),
            id(2)
        );
        assert_eq!(
            t.identity_for(RecordType::Tombstone, 0, &tombstone(10)),
            id(3)
        );
        let section = SectionTombstonePayload { reason: None }.encode();
        t.record(RecordType::SectionTombstone, 44, &section, id(4));
        assert_eq!(
            t.identity_for(RecordType::SectionTombstone, 44, &section),
            id(4)
        );
        assert_eq!(t.len(), 3);
    }

    /// A value record is not a marker and carries no identity here.
    #[test]
    fn values_are_not_markers() {
        let mut t = MarkerIds::new();
        t.record(RecordType::Commit, 9, &[], id(1));
        t.record(RecordType::Chunk, 9, &[], id(1));
        assert!(t.is_empty());
        assert_eq!(
            t.identity_for(RecordType::Commit, 9, &[]),
            RecordIdentity::NONE
        );
    }
}
