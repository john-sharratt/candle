//! Stream identity and the `StreamDecl` record payload.
//!
//! Every chunk in the redo log belongs to a *stream*: a turn stream
//! (one conversation turn) or a content-addressed prompt-section
//! stream. A conversation is the emergent set of turn streams
//! sharing a `timeline_id`, anchored to the section streams that
//! form its prefix.
//!
//! The `StreamDecl` payload is JSON — adding/removing fields is
//! forward-compat for both writer and reader, courtesy of
//! `#[serde(default)]`.

use serde::{Deserialize, Serialize};

use super::content_hash::{section_stream_id, turn_stream_id, ContentHash};
use super::{PersistenceError, Result};

/// Per-`(timeline, turn)` count of cognitive-depth scores carried in
/// a turn declaration: 3 depths (syntactic / semantic / pragmatic) ×
/// 7 score fields (max, sum, mean, top-k mean, count, span,
/// per-token excess).
pub const SCORE_LANES: usize = 21;

/// Globally unique stream identifier. `0` is reserved as the header's
/// "not stream-scoped" sentinel and is never a real stream id.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct StreamId(pub u64);

/// The two kinds of stream.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamKind {
    /// One conversation turn — identity-addressed, immutable once sealed.
    Turn,
    /// One system-prompt section — content-addressed, write-once.
    PromptSection,
}

/// A resolved reference to a stream — its id and kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamRef {
    pub stream_id: StreamId,
    pub kind: StreamKind,
}

/// The content address of a prompt-section stream.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub struct ContentAddress {
    #[serde(default)]
    pub prefix_hash: ContentHash,
    #[serde(default)]
    pub section_hash: ContentHash,
}

/// The cognitive-depth relevance scores of a turn — the
/// `PerDepthScores` flattened to [`SCORE_LANES`] lanes.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PerDepthScores(pub [f32; SCORE_LANES]);

impl Default for PerDepthScores {
    fn default() -> PerDepthScores {
        PerDepthScores([0.0; SCORE_LANES])
    }
}

/// Declaration of a content-addressed prompt-section stream.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct SectionDecl {
    #[serde(default)]
    pub address: ContentAddress,
    #[serde(default)]
    pub debug_name: String,
}

/// Declaration of an identity-addressed conversation-turn stream — all
/// the per-turn substrate metadata the redo log must carry to
/// reconstruct the substrate.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TurnDecl {
    #[serde(default)]
    pub timeline_id: u64,
    #[serde(default)]
    pub turn_index: u32,
    /// `tree::TurnId` coordinates — elapsed days and the monotonic counter.
    #[serde(default)]
    pub turn_id_day: i32,
    #[serde(default)]
    pub turn_id_seq: u32,
    /// `turn::Role` tag (System / User / Assistant).
    #[serde(default)]
    pub role: u8,
    /// KV block span `(start, end)` this turn occupies.
    #[serde(default)]
    pub block_start: u64,
    #[serde(default)]
    pub block_end: u64,
    /// Projection `(layer, group)` the turn's timeline is registered against.
    #[serde(default)]
    pub layer_id: u32,
    #[serde(default)]
    pub group_id: u32,
    /// Ordered prefix streams this turn is anchored to.
    #[serde(default)]
    pub anchored_prefix: Vec<StreamId>,
    /// The projection `view` — turn indices selected as this turn's context.
    #[serde(default)]
    pub view: Vec<u32>,
    #[serde(default)]
    pub scores: PerDepthScores,
}

/// The payload of a `StreamDecl` record — declares a stream and
/// carries its structural metadata.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamDecl {
    PromptSection(SectionDecl),
    Turn(TurnDecl),
}

impl StreamDecl {
    /// The kind of stream this declares.
    pub fn kind(&self) -> StreamKind {
        match self {
            StreamDecl::PromptSection(_) => StreamKind::PromptSection,
            StreamDecl::Turn(_) => StreamKind::Turn,
        }
    }

    /// The derived [`StreamId`] this declaration addresses — content-derived
    /// for a section, identity-derived for a turn.
    pub fn stream_id(&self) -> StreamId {
        match self {
            StreamDecl::PromptSection(s) => section_stream_id(s.address),
            StreamDecl::Turn(t) => turn_stream_id(t.timeline_id, t.turn_index),
        }
    }

    /// Encode to the `StreamDecl` record payload bytes — UTF-8 JSON.
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("StreamDecl JSON encoding is infallible")
    }

    /// Decode from the `StreamDecl` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<StreamDecl> {
        serde_json::from_slice(payload)
            .map_err(|e| PersistenceError::Corrupt(format!("StreamDecl JSON decode: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_decl_roundtrip() {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash {
                    lo: 0x1111,
                    hi: 0x2222,
                },
                section_hash: ContentHash {
                    lo: 0x3333,
                    hi: 0x4444,
                },
            },
            debug_name: "system_framing".to_string(),
        });
        let bytes = decl.encode();
        assert_eq!(StreamDecl::decode(&bytes).unwrap(), decl);
        assert_eq!(decl.kind(), StreamKind::PromptSection);
    }

    #[test]
    fn turn_decl_roundtrip() {
        let mut lanes = [0.0f32; SCORE_LANES];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = i as f32 * 0.25;
        }
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 0xABCD,
            turn_index: 12,
            turn_id_day: -3,
            turn_id_seq: 99,
            role: 2,
            block_start: 64,
            block_end: 96,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: vec![StreamId(7), StreamId(8), StreamId(123)],
            view: vec![0, 2, 5],
            scores: PerDepthScores(lanes),
        });
        let bytes = decl.encode();
        let decoded = StreamDecl::decode(&bytes).unwrap();
        assert_eq!(decoded, decl);
        assert_eq!(decl.kind(), StreamKind::Turn);
    }

    #[test]
    fn turn_decl_empty_vecs_roundtrip() {
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 1,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 0,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            scores: PerDepthScores::default(),
        });
        assert_eq!(StreamDecl::decode(&decl.encode()).unwrap(), decl);
    }

    /// A newer writer adds a field we don't model. The older
    /// reader must ignore it (serde's default behaviour for unknown
    /// keys) rather than fail.
    #[test]
    fn stream_decl_ignores_unknown_field() {
        // A Turn with an extra `future_field`. serde drops unknown keys.
        let payload =
            br#"{"kind":"turn","timeline_id":1,"turn_index":0,"role":1,"future_field":"opaque"}"#;
        let decoded = StreamDecl::decode(payload).unwrap();
        match decoded {
            StreamDecl::Turn(t) => {
                assert_eq!(t.timeline_id, 1);
                assert_eq!(t.role, 1);
            }
            _ => panic!("expected Turn variant"),
        }
    }

    /// An older writer omits fields the newer reader expects;
    /// `#[serde(default)]` covers the gap.
    #[test]
    fn stream_decl_defaults_missing_fields() {
        let payload = br#"{"kind":"prompt_section"}"#;
        let decoded = StreamDecl::decode(payload).unwrap();
        match decoded {
            StreamDecl::PromptSection(s) => {
                assert_eq!(s.debug_name, "");
                assert_eq!(s.address, ContentAddress::default());
            }
            _ => panic!("expected PromptSection variant"),
        }
    }

    #[test]
    fn invalid_kind_tag_is_an_error() {
        // serde's enum-tag matcher rejects unknown variants by
        // default — the catch-all forward-compat lever for record
        // kinds lives in `RecordType::Unknown`, not here.
        let payload = br#"{"kind":"future_kind","timeline_id":1}"#;
        assert!(StreamDecl::decode(payload).is_err());
    }
}
