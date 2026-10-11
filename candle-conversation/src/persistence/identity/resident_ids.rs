//! The identity of each resident value's current record.
//!
//! A resident record is re-encoded from the substrate's RAM whenever a rewrite
//! carries it forward ([`crate::persistence::survival::Survival::Resident`]),
//! and the copy must keep the identity of the record it copies: the same
//! `origin` and `oseq`, so every member still recognises it, and the same
//! stamp, so it still ranks where it did. This table holds that identity per
//! supersession key, kept in step with the records the walk and the appends
//! see, and a re-emission looks it up by the key of the payload it encodes.

use std::collections::HashMap;

use super::super::liveness_audit::{resident_key, ResidentKey};
use super::super::record::RecordType;
use super::record_identity::RecordIdentity;

/// Identity per resident supersession key.
#[derive(Clone, Debug, Default)]
pub struct ResidentIds {
    ids: HashMap<ResidentKey, RecordIdentity>,
}

impl ResidentIds {
    pub fn new() -> ResidentIds {
        ResidentIds::default()
    }

    /// Record that the record `(rt, stream_id, payload)` carries `id`. A type
    /// that is not resident, or a payload that does not decode, is ignored.
    pub fn record(&mut self, rt: RecordType, stream_id: u64, payload: &[u8], id: RecordIdentity) {
        if let Some(key) = resident_key(rt, stream_id, payload) {
            self.ids.insert(key, id);
        }
    }

    /// The identity a re-emission of `(rt, stream_id, payload)` carries: that
    /// of the record holding the value now, or none for a value no stamped
    /// record set.
    pub fn identity_for(&self, rt: RecordType, stream_id: u64, payload: &[u8]) -> RecordIdentity {
        resident_key(rt, stream_id, payload)
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

    fn id(oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 3,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        }
    }

    /// The latest record of a key is the one a re-emission carries.
    #[test]
    fn the_latest_record_of_a_key_is_remembered() {
        let mut t = ResidentIds::new();
        t.record(RecordType::Commit, 9, &[], id(1));
        t.record(RecordType::Commit, 9, &[], id(2));
        t.record(RecordType::Commit, 10, &[], id(3));
        assert_eq!(t.identity_for(RecordType::Commit, 9, &[]), id(2));
        assert_eq!(t.identity_for(RecordType::Commit, 10, &[]), id(3));
        assert_eq!(t.len(), 2);
    }

    /// A key no record set, and a type that is not resident, carry none.
    #[test]
    fn unknown_keys_and_types_carry_no_identity() {
        let mut t = ResidentIds::new();
        t.record(RecordType::Chunk, 9, &[], id(1));
        assert!(t.is_empty());
        assert_eq!(
            t.identity_for(RecordType::Commit, 9, &[]),
            RecordIdentity::NONE
        );
    }
}
