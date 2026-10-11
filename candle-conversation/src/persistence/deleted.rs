//! The payload of a [`RecordType::Deleted`](super::record::RecordType::Deleted)
//! record — a member's **deleted set** (`docs/replicated_substrate.md` §6.8).
//!
//! A timeline tombstone is a 4 KiB record. Once a complete compaction has run
//! since it — removing every record of the timeline the member held — and it
//! is older than the member's retention, a compaction shrinks it to an entry
//! here: the timeline and the `(origin, oseq)` of the tombstone, 24 bytes. The
//! timeline stays dead for good: a timeline id is never reused, and a record
//! for it arriving however late dies exactly as it would against the
//! tombstone.
//!
//! The record also names the tombstones a complete compaction has already
//! run past (`swept`), which is the first of the two conditions for expiry.
//! It is local housekeeping, rewritten whole by every compaction and never
//! replicated, so it carries no identity.
//!
//! Layout, little-endian: `u32` entry count, then per entry `timeline`,
//! `origin`, `oseq` as `u64`s; `u32` swept count, then each swept timeline as a
//! `u64`.

use super::le_reader::LeReader;
use super::Result;

/// One expired timeline tombstone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeletedTimeline {
    pub timeline: u64,
    /// The identity of the tombstone it replaced.
    pub origin: u64,
    pub oseq: u64,
}

/// See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeletedPayload {
    pub timelines: Vec<DeletedTimeline>,
    /// Timelines whose tombstone a complete compaction has run past.
    pub swept: Vec<u64>,
}

impl DeletedPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 24 * self.timelines.len() + 8 * self.swept.len());
        out.extend_from_slice(&(self.timelines.len() as u32).to_le_bytes());
        for t in &self.timelines {
            out.extend_from_slice(&t.timeline.to_le_bytes());
            out.extend_from_slice(&t.origin.to_le_bytes());
            out.extend_from_slice(&t.oseq.to_le_bytes());
        }
        out.extend_from_slice(&(self.swept.len() as u32).to_le_bytes());
        for tl in &self.swept {
            out.extend_from_slice(&tl.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Result<DeletedPayload> {
        let mut r = LeReader::new(buf, "Deleted");
        let entries = r.u32()? as usize;
        let mut timelines = Vec::with_capacity(entries.min(buf.len() / 24));
        for _ in 0..entries {
            timelines.push(DeletedTimeline {
                timeline: r.u64()?,
                origin: r.u64()?,
                oseq: r.u64()?,
            });
        }
        let swept_n = r.u32()? as usize;
        let mut swept = Vec::with_capacity(swept_n.min(buf.len() / 8));
        for _ in 0..swept_n {
            swept.push(r.u64()?);
        }
        r.finish()?;
        Ok(DeletedPayload { timelines, swept })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_to_exact_bytes() {
        let p = DeletedPayload {
            timelines: vec![DeletedTimeline {
                timeline: 0x0102,
                origin: 0xA,
                oseq: 7,
            }],
            swept: vec![0x0304],
        };
        let mut want = vec![1, 0, 0, 0];
        want.extend_from_slice(&[0x02, 0x01, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x0A, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0x04, 0x03, 0, 0, 0, 0, 0, 0]);
        assert_eq!(p.encode(), want);
        assert_eq!(DeletedPayload::decode(&want).unwrap(), p);
    }

    #[test]
    fn an_empty_set_is_eight_bytes() {
        let p = DeletedPayload::default();
        assert_eq!(p.encode(), vec![0; 8]);
        assert_eq!(DeletedPayload::decode(&[0; 8]).unwrap(), p);
    }

    #[test]
    fn a_truncated_or_padded_payload_is_refused() {
        let bytes = DeletedPayload {
            timelines: vec![DeletedTimeline {
                timeline: 1,
                origin: 2,
                oseq: 3,
            }],
            swept: vec![],
        }
        .encode();
        assert!(DeletedPayload::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(DeletedPayload::decode(&padded).is_err());
    }
}
