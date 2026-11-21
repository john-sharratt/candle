//! The payloads of timeline and section tombstones, which mark streams logically
//! deleted.
//!
//! A timeline tombstone kills every record of its timeline, or only one turn when it
//! names a turn index, and replay applies it whatever its reason, which is
//! diagnostic only. A section tombstone is keyed by the section's stream id in the
//! header, and it ends one generation, so a later re-prefill of the same content is
//! not ended. A turn tombstone also names the writer that gave the turn its index.
//! Fields without a reason are omitted, so the records stay byte-identical to older
//! ones.

use super::*;

/// JSON payload for a [`RecordType::Tombstone`] record.  Naming
/// `timeline_id` marks it as logically deleted: every
/// `StreamDecl::Turn`, `Chunk`, `Tokens`, `Commit`,
/// `TreeMetadata`, `Label`, `ConvState`, and `DebugId` record bound
/// to that timeline becomes inert on replay, and the compactor
/// drops them from disk on the next compaction pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstonePayload {
    pub timeline_id: u64,
    /// When `Some`, this tombstone kills only ONE TURN — the
    /// `(timeline_id, turn_index)` pair — instead of the whole timeline. Used by
    /// the per-layer `drop_turn` corrupt-turn policy so a single unrecoverable
    /// turn (e.g. a partial write) doesn't take its whole conversation with it.
    /// `None` tombstones the entire timeline (the default, and every pre-existing
    /// tombstone). Skipped from the serialized record when `None`, so a
    /// timeline-level tombstone stays byte-identical to the pre-`turn_index` format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<u32>,
    /// For a turn tombstone, the origin of the member that wrote the turn:
    /// `turn_index` is the index that writer gave it, which another writer's
    /// turn can move it above (`docs/replicated_substrate.md` §6.6), so the
    /// pair names the turn wherever it now sits. Zero for a turn with no
    /// identity, and skipped from the serialized record then, so such a
    /// tombstone stays byte-identical to the format before this field.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub turn_writer: u64,
    /// Why the timeline was tombstoned, when known — e.g.
    /// `"corrupt reload (turn N): <detail>"` for a turn dropped during substrate
    /// reconstruction because its persisted state was inconsistent. Diagnostic
    /// only: the runtime treats any tombstone as dead regardless. `None` for the
    /// ordinary deletions (file removed, superseded generation, spliced fork).
    /// Skipped from the serialized record when `None`, so a reason-less tombstone
    /// stays byte-identical to the pre-reason format on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl TombstonePayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TombstonePayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("Tombstone JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::SectionTombstone`] record. The section's
/// `StreamId` lives in the record header (`stream_id`), not here — the same
/// convention [`RecordType::WideQSig`] and [`RecordType::TurnIndexPage`] use,
/// so the walker never has to decode this payload to know which stream it
/// names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionTombstonePayload {
    /// The generation the tombstone ends (`docs/replicated_substrate.md`
    /// §6.5): the writer whose section it is and the `oseq` of the declaration
    /// that started it. A later generation of the same section — a fresh
    /// prefill of the same content, under the same stream id — is not ended,
    /// whatever order the records arrive in. `None` on a tombstone written
    /// before generations, which ends the unstamped stream as it stands in the
    /// log at that point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<SectionGeneration>,
    /// Why the section was tombstoned, when known — e.g. the chunk index and
    /// layer split a confirmed per-layer window divergence reported.
    /// Diagnostic only: replay treats any record of this type as dead
    /// regardless. `None` skips the field so a reason-less tombstone stays
    /// minimal on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// One generation of a section stream: the writer, and the `oseq` of the
/// declaration that started it, which every record of the generation carries
/// in its `epoch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SectionGeneration {
    pub writer: u64,
    pub oseq: u64,
}

impl SectionTombstonePayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("SectionTombstonePayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("SectionTombstone JSON parse: {e}")))
    }
}
