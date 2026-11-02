//! What this store has applied from other members, by record type: how many,
//! and how many payload bytes (`docs/memory_census.md` §5). Each record is
//! counted the moment it is written, so it says what the growth that tracks
//! records applied is made of.

use std::collections::BTreeMap;

use serde::Serialize;

use super::record::RecordType;

/// One record type's share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct AppliedTally {
    pub count: u64,
    pub payload_bytes: u64,
}

/// One line of the mix, as reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AppliedLine {
    pub record_type: RecordType,
    pub count: u64,
    pub payload_bytes: u64,
}

/// Every record type applied since the store opened.
#[derive(Debug, Clone, Default)]
pub struct AppliedMix {
    by_type: BTreeMap<RecordType, AppliedTally>,
}

impl AppliedMix {
    /// Count one record of `record_type`, `payload_bytes` long, applied.
    pub fn note(&mut self, record_type: RecordType, payload_bytes: usize) {
        let t = self.by_type.entry(record_type).or_default();
        t.count += 1;
        t.payload_bytes += payload_bytes as u64;
    }

    /// The mix, by payload bytes, largest first.
    pub fn lines(&self) -> Vec<AppliedLine> {
        let mut out: Vec<AppliedLine> = self
            .by_type
            .iter()
            .map(|(&record_type, t)| AppliedLine {
                record_type,
                count: t.count,
                payload_bytes: t.payload_bytes,
            })
            .collect();
        out.sort_by(|a, b| {
            b.payload_bytes
                .cmp(&a.payload_bytes)
                .then(a.record_type.cmp(&b.record_type))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Five chunks of 30,000 B and two labels are counted by type, the
    /// heavier type first** — the mix that would confirm BlackWorld's growth is
    /// chunk-sized records.
    #[test]
    fn the_mix_counts_by_type_heaviest_first() {
        let mut m = AppliedMix::default();
        for _ in 0..5 {
            m.note(RecordType::Chunk, 30_000);
        }
        m.note(RecordType::Label, 120);
        m.note(RecordType::Label, 80);
        assert_eq!(
            m.lines(),
            [
                AppliedLine {
                    record_type: RecordType::Chunk,
                    count: 5,
                    payload_bytes: 150_000
                },
                AppliedLine {
                    record_type: RecordType::Label,
                    count: 2,
                    payload_bytes: 200
                },
            ]
        );
    }

    #[test]
    fn nothing_applied_is_an_empty_mix() {
        assert!(AppliedMix::default().lines().is_empty());
    }
}
