//! Stream identity and the `StreamDecl` record payload (§5.2–§5.3 of
//! `docs/kv_tier_migration.md`).
//!
//! Every chunk in the redo log belongs to a *stream*: a turn stream (one
//! conversation turn) or a content-addressed prompt-section stream. A
//! conversation is the emergent set of turn streams sharing a `timeline_id`,
//! anchored to the section streams that form its prefix.

use super::content_hash::{section_stream_id, turn_stream_id, ContentHash};
use super::record::{ByteReader, ByteWriter};
use super::{PersistenceError, Result};

/// Per-`(timeline, turn)` count of cognitive-depth scores carried in a turn
/// declaration: 3 depths (syntactic / semantic / pragmatic) × 7 score
/// fields (max, sum, mean, top-k mean, count, span, per-token excess).
pub const SCORE_LANES: usize = 21;

/// Globally unique stream identifier. `0` is reserved as the header's "not
/// stream-scoped" sentinel and is never a real stream id.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct StreamId(pub u64);

/// The two kinds of stream.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StreamKind {
    /// One conversation turn — identity-addressed, immutable once sealed.
    Turn,
    /// One system-prompt section — content-addressed, write-once.
    PromptSection,
}

impl StreamKind {
    fn tag(self) -> u8 {
        match self {
            StreamKind::Turn => 1,
            StreamKind::PromptSection => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<StreamKind> {
        match tag {
            1 => Ok(StreamKind::Turn),
            2 => Ok(StreamKind::PromptSection),
            other => Err(PersistenceError::UnknownStreamKind(other)),
        }
    }
}

/// A resolved reference to a stream — its id and kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamRef {
    pub stream_id: StreamId,
    pub kind: StreamKind,
}

/// The content address of a prompt-section stream: the hash of the prefix
/// that precedes it, and the hash of its own tokens (§5.2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ContentAddress {
    pub prefix_hash: ContentHash,
    pub section_hash: ContentHash,
}

/// The cognitive-depth relevance scores of a turn (the `PerDepthScores`
/// flattened to [`SCORE_LANES`] lanes — persisted, not recomputed; §15 q7).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PerDepthScores(pub [f32; SCORE_LANES]);

impl Default for PerDepthScores {
    fn default() -> PerDepthScores {
        PerDepthScores([0.0; SCORE_LANES])
    }
}

/// Declaration of a content-addressed prompt-section stream.
#[derive(Clone, PartialEq, Debug)]
pub struct SectionDecl {
    pub address: ContentAddress,
    /// Human-readable section name, for debugging and log inspection.
    pub debug_name: String,
}

/// Declaration of an identity-addressed conversation-turn stream — all the
/// per-turn substrate metadata the redo log must carry to reconstruct the
/// substrate (§5.7).
#[derive(Clone, PartialEq, Debug)]
pub struct TurnDecl {
    pub timeline_id: u64,
    pub turn_index: u32,
    /// `tree::TurnId` coordinates — elapsed days and the monotonic counter.
    pub turn_id_day: i32,
    pub turn_id_seq: u32,
    /// `turn::Role` tag (System / User / Assistant).
    pub role: u8,
    /// KV block span `(start, end)` this turn occupies.
    pub block_start: u64,
    pub block_end: u64,
    /// Projection `(layer, group)` the turn's timeline is registered against
    /// — persisted so the substrate-reload path can re-register the timeline
    /// (`LayerId` / `GroupId` raw values).
    pub layer_id: u32,
    pub group_id: u32,
    /// Ordered prefix streams this turn is anchored to — its prompt-section
    /// streams followed by every prior turn stream.
    pub anchored_prefix: Vec<StreamId>,
    /// The projection `view` — the turn indices selected as this turn's
    /// context.
    pub view: Vec<u32>,
    pub scores: PerDepthScores,
}

/// The payload of a `StreamDecl` record — declares a stream and carries its
/// structural metadata.
#[derive(Clone, PartialEq, Debug)]
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

    /// Encode to the `StreamDecl` record payload bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.put_u8(self.kind().tag());
        match self {
            StreamDecl::PromptSection(s) => {
                w.put_raw(&s.address.prefix_hash.to_bytes());
                w.put_raw(&s.address.section_hash.to_bytes());
                w.put_str(&s.debug_name);
            }
            StreamDecl::Turn(t) => {
                w.put_u64(t.timeline_id);
                w.put_u32(t.turn_index);
                w.put_i32(t.turn_id_day);
                w.put_u32(t.turn_id_seq);
                w.put_u8(t.role);
                w.put_u64(t.block_start);
                w.put_u64(t.block_end);
                w.put_u32(t.layer_id);
                w.put_u32(t.group_id);
                w.put_u32(t.anchored_prefix.len() as u32);
                for id in &t.anchored_prefix {
                    w.put_u64(id.0);
                }
                w.put_u32(t.view.len() as u32);
                for &v in &t.view {
                    w.put_u32(v);
                }
                for lane in t.scores.0 {
                    w.put_f32(lane);
                }
            }
        }
        w.into_bytes()
    }

    /// Decode from the `StreamDecl` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<StreamDecl> {
        let mut r = ByteReader::new(payload);
        let kind = StreamKind::from_tag(r.get_u8()?)?;
        let decl = match kind {
            StreamKind::PromptSection => {
                let mut hb = [0u8; 16];
                hb.copy_from_slice(r.get_raw(16)?);
                let prefix_hash = ContentHash::from_bytes(hb);
                hb.copy_from_slice(r.get_raw(16)?);
                let section_hash = ContentHash::from_bytes(hb);
                let debug_name = r.get_str()?;
                StreamDecl::PromptSection(SectionDecl {
                    address: ContentAddress {
                        prefix_hash,
                        section_hash,
                    },
                    debug_name,
                })
            }
            StreamKind::Turn => {
                let timeline_id = r.get_u64()?;
                let turn_index = r.get_u32()?;
                let turn_id_day = r.get_i32()?;
                let turn_id_seq = r.get_u32()?;
                let role = r.get_u8()?;
                let block_start = r.get_u64()?;
                let block_end = r.get_u64()?;
                let layer_id = r.get_u32()?;
                let group_id = r.get_u32()?;
                let n_prefix = r.get_u32()? as usize;
                let mut anchored_prefix = Vec::with_capacity(n_prefix);
                for _ in 0..n_prefix {
                    anchored_prefix.push(StreamId(r.get_u64()?));
                }
                let n_view = r.get_u32()? as usize;
                let mut view = Vec::with_capacity(n_view);
                for _ in 0..n_view {
                    view.push(r.get_u32()?);
                }
                let mut lanes = [0.0f32; SCORE_LANES];
                for lane in lanes.iter_mut() {
                    *lane = r.get_f32()?;
                }
                StreamDecl::Turn(TurnDecl {
                    timeline_id,
                    turn_index,
                    turn_id_day,
                    turn_id_seq,
                    role,
                    block_start,
                    block_end,
                    layer_id,
                    group_id,
                    anchored_prefix,
                    view,
                    scores: PerDepthScores(lanes),
                })
            }
        };
        if !r.is_done() {
            return Err(PersistenceError::Corrupt(format!(
                "StreamDecl payload has {} trailing bytes",
                r.remaining()
            )));
        }
        Ok(decl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_kind_tag_roundtrip() {
        for k in [StreamKind::Turn, StreamKind::PromptSection] {
            assert_eq!(StreamKind::from_tag(k.tag()).unwrap(), k);
        }
        assert!(matches!(
            StreamKind::from_tag(7),
            Err(PersistenceError::UnknownStreamKind(7))
        ));
    }

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

    #[test]
    fn trailing_bytes_rejected() {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "x".to_string(),
        });
        let mut bytes = decl.encode();
        bytes.push(0);
        assert!(matches!(
            StreamDecl::decode(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }

    #[test]
    fn truncated_payload_rejected() {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "x".to_string(),
        });
        let bytes = decl.encode();
        assert!(StreamDecl::decode(&bytes[..5]).is_err());
    }
}
