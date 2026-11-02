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

use serde::Serialize;

use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::{
    encode_conv_state_payload, encode_label_payload, ConvMeta, ConvState,
};
use crate::persistence::record::{
    DebugIdPayload, RecordType, TreeMetadataPayload, TurnCouplingPayload,
};

use super::pcoll::{PMap, PVec};

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

/// What [`PendingChildren`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct PendingGauge {
    pub timelines: u64,
    pub records: u64,
    pub payload_bytes: u64,
}

/// One record waiting: what it is, its identity, and the bytes it encodes to,
/// measured once as it is held.
type Waiting = (PendingChild, RecordIdentity, u64);

/// The records waiting, by the raw id of the timeline they name.
#[derive(Debug, Clone, Default)]
pub struct PendingChildren {
    by_timeline: PMap<u64, PVec<Waiting>>,
    /// Records held, kept as they are held and released, so the census reads
    /// it without a walk.
    records: u64,
    /// The bytes those records encode to.
    payload_bytes: u64,
}

impl PendingChildren {
    /// Hold `child`, written as record `id`, until `timeline` registers. The
    /// same record held twice — a copy relocated past the original — is held
    /// once.
    pub fn hold(&mut self, timeline: u64, child: PendingChild, id: RecordIdentity) {
        let waiting = self.by_timeline.entry(timeline).or_default();
        if !id.is_none() && waiting.iter().any(|(_, held, _)| *held == id) {
            return;
        }
        let len = child.encode(timeline).1.len() as u64;
        waiting.push_back((child, id, len));
        self.records += 1;
        self.payload_bytes += len;
    }

    /// Everything waiting for `timeline`, which it now takes.
    pub fn release(&mut self, timeline: u64) -> Vec<(PendingChild, RecordIdentity)> {
        let Some(waiting) = self.by_timeline.remove(&timeline) else {
            return Vec::new();
        };
        waiting
            .into_iter()
            .map(|(child, id, len)| {
                self.records -= 1;
                self.payload_bytes -= len;
                (child, id)
            })
            .collect()
    }

    /// What is waiting, for the memory census (`docs/memory_census.md` §5):
    /// the timelines not yet registered, the records held for them, and the
    /// bytes those records encode to. Nothing here is released until its
    /// timeline registers, so a timeline that never does keeps its records for
    /// good. Read from counts kept as records are held, so it walks nothing.
    pub fn gauge(&self) -> PendingGauge {
        PendingGauge {
            timelines: self.by_timeline.len() as u64,
            records: self.records,
            payload_bytes: self.payload_bytes,
        }
    }

    /// Every waiting record as a rewrite writes it back: `(timeline, type,
    /// payload, identity)`, in timeline order then arrival order.
    pub fn records(&self) -> Vec<(u64, RecordType, Vec<u8>, RecordIdentity)> {
        let mut timelines: Vec<&u64> = self.by_timeline.keys().collect();
        timelines.sort_unstable();
        let mut out = Vec::new();
        for tl in timelines {
            for (child, id, _) in &self.by_timeline[tl] {
                let (rt, payload) = child.encode(*tl);
                out.push((*tl, rt, payload, *id));
            }
        }
        out
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

    /// **The gauge counts every record still waiting, and its bytes**; a
    /// release takes its timeline's out of the count.
    #[test]
    fn the_gauge_counts_what_waits() {
        let mut p = PendingChildren::default();
        p.hold(9, PendingChild::Coupling(4), id(1));
        p.hold(9, PendingChild::DebugId("abc".into()), id(2));
        p.hold(11, PendingChild::Coupling(7), id(3));
        // Each child is sized as it encodes under its own timeline.
        let size = |c: PendingChild, tl: u64| c.encode(tl).1.len() as u64;
        let first = size(PendingChild::Coupling(4), 9);
        let debug = size(PendingChild::DebugId("abc".into()), 9);
        let other = size(PendingChild::Coupling(7), 11);
        assert_eq!(
            p.gauge(),
            PendingGauge {
                timelines: 2,
                records: 3,
                payload_bytes: first + debug + other
            }
        );
        p.release(9);
        assert_eq!(
            p.gauge(),
            PendingGauge {
                timelines: 1,
                records: 1,
                payload_bytes: other
            }
        );
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
