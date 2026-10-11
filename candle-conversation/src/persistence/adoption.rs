//! The one-time adoption of a store written before identities
//! (`docs/replicated_substrate.md` §6.2).
//!
//! A store written before this design holds records with no origin. Its first
//! writable open rewrites it whole, by a compaction that stamps every one of
//! them as this node's own: its origin, a fresh `oseq`, the current generation
//! and a clock reading. There is no mixed mode — a record with no identity
//! cannot be named, replicated, fetched or ordered against another member's —
//! so adoption happens at open, before anything is served.
//!
//! The stamps follow the order the compaction writes records in, and that
//! order keeps every ranking the store already had:
//!
//! - a timeline's turns are written in turn order, so their declarations'
//!   stamps rise with the index they were written at, and each turn stays at
//!   its position (§6.6);
//! - labels are written in conversation creation order, so the record that
//!   first named each conversation keeps its place among them;
//! - a section's declaration comes before its records, which take its new
//!   `oseq` as their generation (§6.5).
//!
//! What the stamp changes inside a payload changes with it: a turn tombstone
//! names the turn's writer, which is now this node, and file events and their
//! tombstones written before events were named by writer gain this node as
//! theirs.
//!
//! Local housekeeping keeps no identity, as it never had one: the header
//! index, the model binding, the branch-checkpoint cache, and the member's own
//! `Replica`, `Deleted` and `Rebase` records.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use super::identity::Minter;
use super::record::{RecordHeader, RecordType, TombstonePayload};
use super::streams::StreamDecl;
use super::vfs::{VfsEventId, VfsEventPayload, VfsTombstonePayload};
use super::{PersistenceError, Result};

/// Whether a record of type `rt` is one an origin writes — what adoption
/// stamps.
fn originated(rt: RecordType) -> bool {
    !matches!(
        rt,
        RecordType::HeaderIndex
            | RecordType::ModelKey
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::BranchCheckpoint
            | RecordType::Replica
            | RecordType::Deleted
            | RecordType::Rebase
            | RecordType::Unknown
    )
}

/// Whether a walked record says the store predates identities: a record an
/// origin writes, with no identity. A tombstone with none says nothing — this
/// member writes its repair tombstones that way (§6.5).
pub fn predates_identities(header: &RecordHeader) -> bool {
    header.identity().is_none()
        && originated(header.record_type)
        && !matches!(
            header.record_type,
            RecordType::Tombstone | RecordType::SectionTombstone
        )
}

/// Whether a record of type `rt` belongs to the generation of the section it
/// is written under (`identity::Minter::mint_for`).
fn generational(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Commit
            | RecordType::TurnIndexPage
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
    )
}

/// A file event as it was written before events were named by writer.
#[derive(Deserialize)]
struct UnnamedVfsEvent {
    timeline_id: u64,
    seq: u64,
    repo: String,
    key: String,
    body: Value,
}

/// A file-event tombstone as it was written before events were named by
/// writer: it killed sequence numbers of one timeline's single counter.
#[derive(Deserialize)]
struct UnnamedVfsTombstone {
    timeline_id: u64,
    seq: u64,
    repo: String,
    key: String,
    kills: Vec<u64>,
}

/// The events a file-event tombstone written before events were named by
/// writer kills, as the store names them until it is adopted: writer zero,
/// like every record of its time. `None` for a payload of any other form.
pub fn unnamed_tombstone_kills(payload: &[u8]) -> Option<Vec<VfsEventId>> {
    let t: UnnamedVfsTombstone = serde_json::from_slice(payload).ok()?;
    Some(
        t.kills
            .into_iter()
            .map(|seq| VfsEventId { origin: 0, seq })
            .collect(),
    )
}

/// The stamping state of one adoption — see the module documentation.
pub struct Adoption {
    minter: Minter,
    node: u64,
    /// Each section's new generation: the `oseq` its declaration took.
    sections: HashMap<u64, u64>,
}

impl Adoption {
    /// An adoption minting through `minter`, this node's.
    pub fn new(minter: Minter) -> Adoption {
        let node = minter.node();
        Adoption {
            minter,
            node,
            sections: HashMap::new(),
        }
    }

    /// The record `header` heads, holding `payload`, stamped as this node's
    /// own — or `None` when it already carries an identity or is housekeeping
    /// no origin writes.
    pub fn restamp(
        &mut self,
        header: &RecordHeader,
        payload: &[u8],
    ) -> Result<Option<(RecordHeader, Vec<u8>)>> {
        if !header.identity().is_none() || !originated(header.record_type) {
            return Ok(None);
        }
        let mut id = self.minter.mint(0);
        match header.record_type {
            RecordType::StreamDecl => {
                if let Ok(StreamDecl::PromptSection(_)) = StreamDecl::decode(payload) {
                    self.sections.insert(header.stream_id, id.oseq);
                    self.minter.start_section(header.stream_id, id.oseq);
                }
            }
            rt if generational(rt) => {
                if let Some(&generation) = self.sections.get(&header.stream_id) {
                    id.epoch = generation;
                }
            }
            _ => {}
        }
        let payload = self.payload(header.record_type, payload)?;
        let mut stamped = header.with_identity(id);
        stamped.payload_len = payload.len() as u64;
        Ok(Some((stamped, payload)))
    }

    /// `payload` as the stamped record holds it.
    fn payload(&self, rt: RecordType, payload: &[u8]) -> Result<Vec<u8>> {
        match rt {
            RecordType::Tombstone => {
                let mut p = TombstonePayload::decode(payload)?;
                if p.turn_index.is_some() && p.turn_writer == 0 {
                    p.turn_writer = self.node;
                }
                Ok(p.encode())
            }
            RecordType::VfsEvent if VfsEventPayload::decode(payload).is_err() => {
                let e: UnnamedVfsEvent = serde_json::from_slice(payload)
                    .map_err(|e| PersistenceError::Corrupt(format!("VfsEvent JSON parse: {e}")))?;
                Ok(VfsEventPayload {
                    timeline_id: e.timeline_id,
                    origin: self.node,
                    seq: e.seq,
                    repo: e.repo,
                    key: e.key,
                    body: e.body,
                }
                .encode())
            }
            RecordType::VfsTombstone if VfsTombstonePayload::decode(payload).is_err() => {
                let t: UnnamedVfsTombstone = serde_json::from_slice(payload).map_err(|e| {
                    PersistenceError::Corrupt(format!("VfsTombstone JSON parse: {e}"))
                })?;
                Ok(VfsTombstonePayload {
                    timeline_id: t.timeline_id,
                    origin: self.node,
                    seq: t.seq,
                    repo: t.repo,
                    key: t.key,
                    kills: t
                        .kills
                        .into_iter()
                        .map(|seq| VfsEventId {
                            origin: self.node,
                            seq,
                        })
                        .collect(),
                }
                .encode())
            }
            _ => Ok(payload.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::{hash_bytes, ContentHash};
    use crate::persistence::identity::Origination;
    use crate::persistence::streams::{ContentAddress, SectionDecl};

    const NODE: u64 = 0x5EED;

    fn adoption() -> Adoption {
        Adoption::new(Minter::new(Origination::new(NODE)))
    }

    fn header(rt: RecordType, stream_id: u64) -> RecordHeader {
        RecordHeader {
            record_type: rt,
            stream_id,
            ..RecordHeader::NONE
        }
    }

    /// Each record takes this node's next `oseq`; one with an identity, and
    /// housekeeping, are left as they are.
    #[test]
    fn records_take_this_nodes_sequence_in_write_order() {
        let mut a = adoption();
        let (first, _) = a
            .restamp(&header(RecordType::Tokens, 9), b"t")
            .unwrap()
            .unwrap();
        let (second, _) = a
            .restamp(&header(RecordType::Label, 0), b"l")
            .unwrap()
            .unwrap();
        assert_eq!((first.origin, first.oseq), (NODE, 1));
        assert_eq!((second.origin, second.oseq), (NODE, 2));
        assert!(second.identity().stamp() > first.identity().stamp());
        let stamped = header(RecordType::Tokens, 9).with_identity(first.identity());
        assert!(a.restamp(&stamped, b"t").unwrap().is_none());
        for rt in [
            RecordType::HeaderIndex,
            RecordType::Replica,
            RecordType::ModelKey,
        ] {
            assert!(a.restamp(&header(rt, 0), b"").unwrap().is_none(), "{rt:?}");
        }
    }

    /// A section's records take its declaration's new `oseq` as their
    /// generation; a turn's records take none.
    #[test]
    fn a_sections_records_take_its_new_generation() {
        let mut a = adoption();
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 1, hi: 0 },
                section_hash: hash_bytes(b"s"),
            },
            debug_name: "s".into(),
        });
        let sid = decl.stream_id().0;
        let (d, _) = a
            .restamp(&header(RecordType::StreamDecl, sid), &decl.encode())
            .unwrap()
            .unwrap();
        let (chunk, _) = a
            .restamp(&header(RecordType::Chunk, sid), b"kv")
            .unwrap()
            .unwrap();
        let (other, _) = a
            .restamp(&header(RecordType::Chunk, sid + 1), b"kv")
            .unwrap()
            .unwrap();
        assert_eq!(d.epoch, 0);
        assert_eq!(chunk.epoch, d.oseq);
        assert_eq!(other.epoch, 0);
    }

    /// A turn tombstone names this node as the turn's writer; a timeline
    /// tombstone names none.
    #[test]
    fn a_turn_tombstone_names_this_node() {
        let mut a = adoption();
        let turn = TombstonePayload {
            timeline_id: 7,
            turn_index: Some(2),
            turn_writer: 0,
            reason: None,
        };
        let (_, p) = a
            .restamp(&header(RecordType::Tombstone, 0), &turn.encode())
            .unwrap()
            .unwrap();
        assert_eq!(TombstonePayload::decode(&p).unwrap().turn_writer, NODE);
        let whole = TombstonePayload {
            turn_index: None,
            ..turn
        };
        let (_, p) = a
            .restamp(&header(RecordType::Tombstone, 0), &whole.encode())
            .unwrap()
            .unwrap();
        assert_eq!(TombstonePayload::decode(&p).unwrap().turn_writer, 0);
    }

    /// File events and their tombstones written before events were named by
    /// writer become this node's, to exact bytes.
    #[test]
    fn unnamed_file_records_become_this_nodes() {
        let mut a = adoption();
        let event = br#"{"timeline_id":7,"seq":3,"repo":"candle","key":"a.rs","body":{"n":1}}"#;
        let (h, p) = a
            .restamp(&header(RecordType::VfsEvent, 7), event)
            .unwrap()
            .unwrap();
        assert_eq!(
            p,
            br#"{"timeline_id":7,"origin":24301,"seq":3,"repo":"candle","key":"a.rs","body":{"n":1}}"#
                .to_vec()
        );
        assert_eq!(h.payload_len, p.len() as u64);
        let tomb = br#"{"timeline_id":7,"seq":4,"repo":"candle","key":"a.rs","kills":[3]}"#;
        let (_, p) = a
            .restamp(&header(RecordType::VfsTombstone, 7), tomb)
            .unwrap()
            .unwrap();
        assert_eq!(
            p,
            br#"{"timeline_id":7,"origin":24301,"seq":4,"repo":"candle","key":"a.rs","kills":[{"origin":24301,"seq":3}]}"#
                .to_vec()
        );
    }

    /// What says a store predates identities: a record an origin writes, with
    /// none — not a tombstone, not housekeeping, not a stamped record.
    #[test]
    fn what_says_a_store_predates_identities() {
        assert!(predates_identities(&header(RecordType::Chunk, 1)));
        assert!(predates_identities(&header(RecordType::Label, 0)));
        assert!(!predates_identities(&header(RecordType::Tombstone, 0)));
        assert!(!predates_identities(&header(RecordType::HeaderIndex, 0)));
        assert!(!predates_identities(&header(RecordType::Replica, 0)));
        let mut a = adoption();
        let (stamped, _) = a
            .restamp(&header(RecordType::Chunk, 1), b"kv")
            .unwrap()
            .unwrap();
        assert!(!predates_identities(&stamped));
    }
}
