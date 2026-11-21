//! Payloads for summary-tree metadata, debug ids and host-defined custom objects.
//!
//! Tree metadata is JSON with one entry per timeline turn, and the latest record
//! for a key wins on reload. A custom object is keyed by a stable hash of its key,
//! so a later write supersedes the earlier one and compaction keeps the live copy.
//! A tombstone on the key hides the object and then expires into the deleted set.
//! The stream ids come from SHA-256, which is stable across builds.

use super::*;

// ── TreeMetadata + DebugId structured payloads ──────────────────────────────

/// JSON payload for a [`RecordType::TreeMetadata`] record — one entry
/// per `(timeline, turn_index)`.  Last-writer-wins on reload: a later
/// record for the same key supersedes the earlier one.
///
/// Stored as JSON because it's small (a handful of integer fields plus a
/// children list bounded by `MERGE_FANOUT`) and because the redo-log already
/// has a JSON-header culture; binary would buy nothing here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeMetadataPayload {
    /// `TimelineId.raw()` — opaque u64 key.
    pub timeline_id: u64,
    /// `TurnIndex.0` — the turn this metadata belongs to.
    pub turn_index: u32,
    /// Discriminant matching `summary_tree::TurnKind`:
    ///   0 = Normal, 1 = SummaryOfTurns, 2 = SummaryOfSummaries.
    pub kind: u8,
    /// Forest level: `SummaryOfTurns` = 1, a `SummaryOfSummaries` over level-`h`
    /// children = `h + 1`; 0 for Normal sub-leaves.
    pub tree_height: u8,
    /// For `SummaryOfTurns`: Normal-child indices in chronological order.  For
    /// `SummaryOfSummaries`: exactly `MERGE_FANOUT` same-level child indices.
    /// For `Normal`: empty.
    pub children: Vec<u32>,
}

impl TreeMetadataPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TreeMetadataPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("TreeMetadata JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::DebugId`] record.  One entry per
/// `(timeline)`, last-writer-wins on replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugIdPayload {
    pub timeline_id: u64,
    pub debug_id: String,
}

impl DebugIdPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("DebugIdPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("DebugId JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::CustomObject`] record — a generic keyed
/// object anything may persist: a `key`, a metadata bag, and a JSON `blob`.
///
/// Last-writer-wins by `key`: the header's `stream_id` is [`Self::stream_id`], a
/// stable hash of the key, so a later record with the same key supersedes the
/// older one exactly the way a re-written [`NpcPayload`] supersedes its previous
/// record. The substrate keeps the live set in RAM and re-emits it on
/// compaction, so it outlives a compaction pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomObjectPayload {
    /// The identity of the object. Same key → same record stream → LWW.
    pub key: String,
    /// A free-form string bag for small attributes that do not belong in the
    /// blob (a kind, a schema version, a world id). Omitted on the wire when
    /// empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// The object itself, as arbitrary JSON.
    pub blob: serde_json::Value,
    /// A tombstone on the key: what the object was is gone, such as a world
    /// entity that ceased (`docs/replicated_substrate.md` §6.8, §9.1). It wins
    /// by stamp like any write, hides the key, and expires to an entry in the
    /// deleted set. Omitted on the wire when false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub ceased: bool,
}

pub(super) fn is_false(b: &bool) -> bool {
    !*b
}

impl CustomObjectPayload {
    /// A new object with no metadata.
    pub fn new(key: impl Into<String>, blob: serde_json::Value) -> Self {
        Self {
            key: key.into(),
            metadata: BTreeMap::new(),
            blob,
            ceased: false,
        }
    }

    /// The tombstone that ends `key`.
    pub fn ceased(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            metadata: BTreeMap::new(),
            blob: serde_json::Value::Null,
            ceased: true,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("CustomObjectPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("CustomObject JSON parse: {e}")))
    }

    /// The header `stream_id` this object is keyed on — a stable hash of `key`.
    pub fn stream_id(&self) -> u64 {
        Self::stream_id_for(&self.key)
    }

    /// The `stream_id` a given key hashes to. Stable across builds (SHA-256), so
    /// a key written by one run is found by the next.
    ///
    /// Two distinct keys that collide on the low 8 bytes of their SHA-256 share
    /// a stream, so the later write LWW-shadows the earlier — the objects are
    /// silently merged into one. At 64 bits the birthday bound is ~4 billion
    /// live keys for a 50% chance of one collision; the command table holds tens
    /// of keys, so this is not a practical risk here, but a caller that mints
    /// keys from unbounded external input should carry its own namespace.
    pub fn stream_id_for(key: &str) -> u64 {
        let d = Sha256::digest(key.as_bytes());
        let v = u64::from_le_bytes(
            d[..8]
                .try_into()
                .expect("32-byte digest is at least 8 bytes"),
        );
        // Never zero: zero is the header's "no stream" sentinel.
        if v == 0 {
            1
        } else {
            v
        }
    }
}
