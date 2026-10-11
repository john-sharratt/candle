//! The payload of a [`RecordType::Replica`](super::super::record::RecordType::Replica)
//! record (`docs/replicated_substrate.md` §6.2, §6.5).
//!
//! What a member has applied of every **other** origin — its own records are
//! covered by its own `oseq`s, which the walk recovers — and the records it
//! deferred because this build does not decode their type. Local housekeeping:
//! written at a commit in which it changed, carried by every rewrite, never
//! replicated, with no identity.
//!
//! Layout, little-endian: `u32` origin count, then per origin its `u64` origin,
//! `u64` through, `u32` range count and each range's `u64` start and end; `u32`
//! deferred count, then per record its `u64` origin, `u64` oseq, `u16` type
//! length and the type's name as its header spells it.

use std::collections::BTreeSet;

use super::super::le_reader::LeReader;
use super::super::Result;
use super::applied_set::{AppliedSet, OriginApplied};

/// A record not written because this build does not decode its type: its
/// identity, and its type as the header names it. A build that knows the type
/// takes it off the deferred set when it opens the substrate, and a catch-up
/// then fetches it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeferredRecord {
    pub origin: u64,
    pub oseq: u64,
    pub record_type: String,
}

/// See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplicaPayload {
    pub applied: AppliedSet,
    pub deferred: BTreeSet<DeferredRecord>,
}

impl ReplicaPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let origins: Vec<(u64, &OriginApplied)> = self.applied.origins().collect();
        out.extend_from_slice(&(origins.len() as u32).to_le_bytes());
        for (origin, applied) in origins {
            out.extend_from_slice(&origin.to_le_bytes());
            out.extend_from_slice(&applied.through().to_le_bytes());
            let ranges: Vec<(u64, u64)> = applied.ranges_above().collect();
            out.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
            for (start, end) in ranges {
                out.extend_from_slice(&start.to_le_bytes());
                out.extend_from_slice(&end.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.deferred.len() as u32).to_le_bytes());
        for d in &self.deferred {
            out.extend_from_slice(&d.origin.to_le_bytes());
            out.extend_from_slice(&d.oseq.to_le_bytes());
            let name = d.record_type.as_bytes();
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name);
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Result<ReplicaPayload> {
        let mut r = LeReader::new(buf, "Replica");
        let mut applied = AppliedSet::default();
        for _ in 0..r.u32()? {
            let origin = r.u64()?;
            let through = r.u64()?;
            let n = r.u32()?;
            let mut ranges = Vec::with_capacity((n as usize).min(buf.len() / 16));
            for _ in 0..n {
                ranges.push((r.u64()?, r.u64()?));
            }
            applied.set_origin(origin, OriginApplied::from_parts(through, ranges));
        }
        let mut deferred = BTreeSet::new();
        for _ in 0..r.u32()? {
            let origin = r.u64()?;
            let oseq = r.u64()?;
            let len = r.u16()? as usize;
            let record_type = String::from_utf8_lossy(r.bytes(len)?).into_owned();
            deferred.insert(DeferredRecord {
                origin,
                oseq,
                record_type,
            });
        }
        r.finish()?;
        Ok(ReplicaPayload { applied, deferred })
    }

    pub fn is_empty(&self) -> bool {
        self.applied.is_empty() && self.deferred.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_to_exact_bytes() {
        let mut applied = AppliedSet::default();
        applied.insert_range(0xA, 1, 3);
        applied.insert(0xA, 6);
        let p = ReplicaPayload {
            applied,
            deferred: [DeferredRecord {
                origin: 0xB,
                oseq: 2,
                record_type: "blob".into(),
            }]
            .into(),
        };
        let mut want = vec![1, 0, 0, 0];
        want.extend_from_slice(&[0xA, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0xB, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[4, 0]);
        want.extend_from_slice(b"blob");
        assert_eq!(p.encode(), want);
        assert_eq!(ReplicaPayload::decode(&want).unwrap(), p);
    }

    #[test]
    fn an_empty_replica_is_eight_bytes() {
        let p = ReplicaPayload::default();
        assert!(p.is_empty());
        assert_eq!(p.encode(), vec![0; 8]);
        assert_eq!(ReplicaPayload::decode(&[0; 8]).unwrap(), p);
        assert!(ReplicaPayload::decode(&[0; 7]).is_err());
    }
}
