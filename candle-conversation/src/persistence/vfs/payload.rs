//! The payloads of a conversation's file events and their tombstones.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::persistence::{PersistenceError, Result};

/// One change a conversation made to a repository's files — the payload of a
/// [`RecordType::VfsEvent`](crate::persistence::record::RecordType::VfsEvent).
///
/// `repo` and `key` say what the event belongs to — a repository, and a path
/// in it or the empty key for the store's base — and `body` what it is. The
/// substrate reads none of the three: it keys the event by timeline and
/// `seq`, which the record header carries too.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfsEventPayload {
    pub timeline_id: u64,
    pub seq: u64,
    pub repo: String,
    pub key: String,
    pub body: Value,
}

impl VfsEventPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("VfsEventPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("VfsEvent JSON parse: {e}")))
    }
}

/// Kills a set of a timeline's file events — every event of one key, up to
/// the last — the payload of a
/// [`RecordType::VfsTombstone`](crate::persistence::record::RecordType::VfsTombstone).
///
/// `kills` names the sequence numbers outright, so the index drops them
/// without having read the events; `repo` and `key` say whose they were.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfsTombstonePayload {
    pub timeline_id: u64,
    pub seq: u64,
    pub repo: String,
    pub key: String,
    pub kills: Vec<u64>,
}

impl VfsTombstonePayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("VfsTombstonePayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("VfsTombstone JSON parse: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_event_encodes_to_exact_bytes_and_back() {
        let event = VfsEventPayload {
            timeline_id: 7,
            seq: 3,
            repo: "candle".into(),
            key: "src/lib.rs".into(),
            body: json!({"kind": "state", "size": 12, "conflict": false}),
        };
        assert_eq!(
            event.encode(),
            br#"{"timeline_id":7,"seq":3,"repo":"candle","key":"src/lib.rs","body":{"kind":"state","size":12,"conflict":false}}"#
                .to_vec()
        );
        assert_eq!(VfsEventPayload::decode(&event.encode()).unwrap(), event);
    }

    #[test]
    fn a_tombstone_encodes_to_exact_bytes_and_back() {
        let tomb = VfsTombstonePayload {
            timeline_id: 7,
            seq: 9,
            repo: "candle".into(),
            key: "".into(),
            kills: vec![1, 4, 8],
        };
        assert_eq!(
            tomb.encode(),
            br#"{"timeline_id":7,"seq":9,"repo":"candle","key":"","kills":[1,4,8]}"#.to_vec()
        );
        assert_eq!(VfsTombstonePayload::decode(&tomb.encode()).unwrap(), tomb);
    }

    #[test]
    fn a_payload_that_is_not_one_is_refused() {
        assert!(VfsEventPayload::decode(b"{}").is_err());
        assert!(VfsTombstonePayload::decode(b"not json").is_err());
    }
}
