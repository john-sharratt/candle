//! Payloads for distillation markers and turn couplings.
//!
//! A distillation marker names a timeline and a mode. `ProvenanceOnly` keeps the
//! declaration and the signatures, for retrieval by signature. `TextOnly` keeps
//! the declaration and the tokens, for a read-only archive. Both drop the KV
//! chunks. A coupling joins a turn to the tool response that follows it, and it is
//! written before that response exists, so the pair is durable before the
//! response turn is submitted.

use super::*;

/// The degree to which a distilled timeline's turns shed content at compaction.
/// Both modes drop the KV chunks (the bulk of the on-disk cost); they differ on
/// what else survives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistillMode {
    /// Keep `StreamDecl` + `WideQSig` + projection events; drop tokens + KV
    /// chunks. The provenance corpus (calibration exemplars): retrievable by
    /// signature, no verbatim text, not resumable. The default so pre-`mode`
    /// on-disk `Distilled` records decode to the original behaviour.
    #[default]
    ProvenanceOnly,
    /// Keep `StreamDecl` + tokens; drop KV chunks + `WideQSig` + projection
    /// events. Archived conversations: a plain read-only text record — not
    /// retrievable by provenance, not resumable.
    TextOnly,
}

/// JSON payload for a [`RecordType::TurnCoupling`] record — joins turn
/// `from_turn` to the tool response that follows it.
///
/// Only `from_turn` is needed: a tool response is always the immediately
/// following turn, so the record can be written before that turn exists (and
/// therefore before the summariser can observe it). `from_turn ∈ set` reads as
/// "turn `from_turn + 1` is the tool response to turn `from_turn`".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCouplingPayload {
    pub timeline_id: u64,
    pub from_turn: u32,
}

impl TurnCouplingPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TurnCouplingPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("TurnCoupling JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::Distilled`] record — names the timeline whose
/// turns should shed content at compaction, and the [`DistillMode`] degree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillPayload {
    pub timeline_id: u64,
    /// Absent on pre-`mode` records → [`DistillMode::ProvenanceOnly`].
    #[serde(default)]
    pub mode: DistillMode,
}

impl DistillPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("DistillPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("Distill JSON parse: {e}")))
    }
}
