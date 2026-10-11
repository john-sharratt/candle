//! Records that arrive before the timeline they belong to
//! (`docs/replicated_substrate.md` §6.5).
//!
//! A timeline is registered by its first turn declaration. Records about the
//! timeline — its label, its state, its summary nodes, its debug id, its
//! couplings — can reach the substrate first: zend writes a conversation's
//! label before its first turn, and records from other members arrive in any
//! order. Such a record waits here, with its identity, until the declaration
//! registers the timeline, and is then applied as if it had just arrived.
//! Nothing is dropped for arriving early.
//!
//! The records waiting here are durable state like any other, so a rewrite of
//! the log carries them ([`PendingChildren::records`]).

use std::collections::HashMap;

use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::{
    encode_conv_state_payload, encode_label_payload, ConvMeta, ConvState,
};
use crate::persistence::record::{
    DebugIdPayload, RecordType, TreeMetadataPayload, TurnCouplingPayload,
};

/// One record waiting for its timeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PendingChild {
    Label(ConvMeta),
    State(ConvState),
    TreeMeta(TreeMetadataPayload),
    DebugId(String),
    Coupling(u32),
}

impl PendingChild {
    /// The record type and payload this child was decoded from.
    fn encode(&self, timeline: u64) -> (RecordType, Vec<u8>) {
        match self {
            PendingChild::Label(meta) => (
                RecordType::Label,
                encode_label_payload(timeline, &meta.conv_id, &meta.label, &meta.custom),
            ),
            PendingChild::State(state) => (
                RecordType::ConvState,
                encode_conv_state_payload(timeline, state),
            ),
            PendingChild::TreeMeta(p) => (RecordType::TreeMetadata, p.encode()),
            PendingChild::DebugId(id) => (
                RecordType::DebugId,
                DebugIdPayload {
                    timeline_id: timeline,
                    debug_id: id.clone(),
                }
                .encode(),
            ),
            PendingChild::Coupling(from_turn) => (
                RecordType::TurnCoupling,
                TurnCouplingPayload {
                    timeline_id: timeline,
                    from_turn: *from_turn,
                }
                .encode(),
            ),
        }
    }
}

/// The records waiting, by the raw id of the timeline they name.
#[derive(Debug, Clone, Default)]
pub struct PendingChildren {
    by_timeline: HashMap<u64, Vec<(PendingChild, RecordIdentity)>>,
}

impl PendingChildren {
    /// Hold `child`, written as record `id`, until `timeline` registers. The
    /// same record held twice — a copy relocated past the original — is held
    /// once.
    pub fn hold(&mut self, timeline: u64, child: PendingChild, id: RecordIdentity) {
        let waiting = self.by_timeline.entry(timeline).or_default();
        if !id.is_none() && waiting.iter().any(|(_, held)| *held == id) {
            return;
        }
        waiting.push((child, id));
    }

    /// Everything waiting for `timeline`, which it now takes.
    pub fn release(&mut self, timeline: u64) -> Vec<(PendingChild, RecordIdentity)> {
        self.by_timeline.remove(&timeline).unwrap_or_default()
    }

    /// Every waiting record as a rewrite writes it back: `(timeline, type,
    /// payload, identity)`, in timeline order then arrival order.
    pub fn records(&self) -> Vec<(u64, RecordType, Vec<u8>, RecordIdentity)> {
        let mut timelines: Vec<&u64> = self.by_timeline.keys().collect();
        timelines.sort_unstable();
        let mut out = Vec::new();
        for tl in timelines {
            for (child, id) in &self.by_timeline[tl] {
                let (rt, payload) = child.encode(*tl);
                out.push((*tl, rt, payload, *id));
            }
        }
        out
    }

    pub fn clear(&mut self) {
        self.by_timeline.clear();
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

    /// A record held twice is held once; a different record beside it is kept.
    #[test]
    fn a_record_is_held_once() {
        let mut p = PendingChildren::default();
        p.hold(9, PendingChild::DebugId("a".into()), id(1));
        p.hold(9, PendingChild::DebugId("a".into()), id(1));
        p.hold(9, PendingChild::Coupling(4), id(2));
        assert_eq!(p.release(9).len(), 2);
        assert!(p.release(9).is_empty(), "a release takes everything held");
    }

    /// What a rewrite writes back is the record each child came from.
    #[test]
    fn records_re_encode_each_child() {
        let mut p = PendingChildren::default();
        p.hold(9, PendingChild::Coupling(4), id(2));
        let records = p.records();
        assert_eq!(records.len(), 1);
        let (tl, rt, payload, rid) = &records[0];
        assert_eq!((*tl, *rt, *rid), (9, RecordType::TurnCoupling, id(2)));
        assert_eq!(
            TurnCouplingPayload::decode(payload).unwrap(),
            TurnCouplingPayload {
                timeline_id: 9,
                from_turn: 4
            }
        );
    }
}
