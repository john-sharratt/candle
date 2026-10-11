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
//! **A keyed value's tombstone** — a custom object marked ceased, such as a
//! world entity that is gone (§9.1) — shrinks the same way, to
//! `(key hash, stamp, origin, oseq)`: 48 bytes. The stamp is kept because a
//! keyed write is ordered by it: a later write with a greater stamp (a body
//! re-entering the world) still wins, as it would against the tombstone.
//!
//! Layout, little-endian: `u32` entry count, then per entry `timeline`,
//! `origin`, `oseq` as `u64`s; `u32` swept count, then each swept timeline as a
//! `u64`; `u32` keyed-entry count, then per entry `key`, `gen`, `epoch`,
//! `hlc`, `origin`, `oseq` as `u64`s; `u32` swept-key count, then each swept
//! key hash as a `u64`.

use super::identity::{RecordIdentity, VersionStamp};
use super::le_reader::LeReader;
use super::record::RecordType;
use super::{Result, SubstratePersistence};

/// One expired timeline tombstone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeletedTimeline {
    pub timeline: u64,
    /// The identity of the tombstone it replaced.
    pub origin: u64,
    pub oseq: u64,
}

/// One expired keyed-value tombstone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeletedKey {
    /// The key's hash: the custom object's `stream_id`.
    pub key: u64,
    /// The tombstone's version stamp, origin aside.
    pub gen: u64,
    pub epoch: u64,
    pub hlc: u64,
    /// The identity of the tombstone it replaced.
    pub origin: u64,
    pub oseq: u64,
}

impl DeletedKey {
    /// The stamp a write to the key must exceed to stand.
    pub fn stamp(&self) -> VersionStamp {
        VersionStamp {
            gen: self.gen,
            epoch: self.epoch,
            hlc: self.hlc,
            origin: self.origin,
        }
    }

    /// The entry an expired tombstone on `key` written as `id` leaves.
    pub fn of(key: u64, id: RecordIdentity) -> DeletedKey {
        DeletedKey {
            key,
            gen: id.gen,
            epoch: id.epoch,
            hlc: id.hlc,
            origin: id.origin,
            oseq: id.oseq,
        }
    }
}

/// An entry of the deleted set, of either kind: what a catch-up's headers
/// carry for an `oseq` that was an expired tombstone (§7.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeletedEntry {
    Timeline(DeletedTimeline),
    Key(DeletedKey),
}

impl DeletedEntry {
    pub fn origin(&self) -> u64 {
        match self {
            DeletedEntry::Timeline(t) => t.origin,
            DeletedEntry::Key(k) => k.origin,
        }
    }

    pub fn oseq(&self) -> u64 {
        match self {
            DeletedEntry::Timeline(t) => t.oseq,
            DeletedEntry::Key(k) => k.oseq,
        }
    }
}

/// See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeletedPayload {
    pub timelines: Vec<DeletedTimeline>,
    /// Timelines whose tombstone a complete compaction has run past.
    pub swept: Vec<u64>,
    pub keys: Vec<DeletedKey>,
    /// Keys whose tombstone a complete compaction has run past, by hash.
    pub swept_keys: Vec<u64>,
}

impl DeletedPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            16 + 24 * self.timelines.len()
                + 8 * self.swept.len()
                + 48 * self.keys.len()
                + 8 * self.swept_keys.len(),
        );
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
        out.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for k in &self.keys {
            for w in [k.key, k.gen, k.epoch, k.hlc, k.origin, k.oseq] {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.swept_keys.len() as u32).to_le_bytes());
        for k in &self.swept_keys {
            out.extend_from_slice(&k.to_le_bytes());
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
        let keys_n = r.u32()? as usize;
        let mut keys = Vec::with_capacity(keys_n.min(buf.len() / 48));
        for _ in 0..keys_n {
            keys.push(DeletedKey {
                key: r.u64()?,
                gen: r.u64()?,
                epoch: r.u64()?,
                hlc: r.u64()?,
                origin: r.u64()?,
                oseq: r.u64()?,
            });
        }
        let swept_keys_n = r.u32()? as usize;
        let mut swept_keys = Vec::with_capacity(swept_keys_n.min(buf.len() / 8));
        for _ in 0..swept_keys_n {
            swept_keys.push(r.u64()?);
        }
        r.finish()?;
        Ok(DeletedPayload {
            timelines,
            swept,
            keys,
            swept_keys,
        })
    }
}

impl SubstratePersistence {
    /// Make the deleted set durable now, whole: an entry a catch-up brought in
    /// must survive a restart before the next compaction rewrites the record
    /// (§6.8, §7.2).
    pub fn write_deleted(&mut self, deleted: &DeletedPayload) -> Result<()> {
        self.append_carried(
            RecordType::Deleted,
            0,
            0,
            0,
            0,
            0,
            &deleted.encode(),
            RecordIdentity::NONE,
        )?;
        self.commit()
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
            keys: vec![DeletedKey {
                key: 0x55,
                gen: 1,
                epoch: 2,
                hlc: 3,
                origin: 0xB,
                oseq: 9,
            }],
            swept_keys: vec![0x66],
        };
        let mut want = vec![1, 0, 0, 0];
        want.extend_from_slice(&[0x02, 0x01, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0x0A, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0x04, 0x03, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, 0, 0]);
        for w in [0x55u8, 1, 2, 3, 0x0B, 9] {
            want.extend_from_slice(&[w, 0, 0, 0, 0, 0, 0, 0]);
        }
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0x66, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(p.encode(), want);
        assert_eq!(DeletedPayload::decode(&want).unwrap(), p);
    }

    #[test]
    fn an_empty_set_is_sixteen_bytes() {
        let p = DeletedPayload::default();
        assert_eq!(p.encode(), vec![0; 16]);
        assert_eq!(DeletedPayload::decode(&[0; 16]).unwrap(), p);
    }

    /// A keyed entry outranks the writes its tombstone did: its stamp is the
    /// tombstone's.
    #[test]
    fn a_keyed_entry_keeps_its_tombstones_stamp() {
        let id = RecordIdentity {
            origin: 4,
            oseq: 8,
            gen: 1,
            epoch: 900,
            hlc: 77,
        };
        let entry = DeletedKey::of(0x55, id);
        assert_eq!(entry.stamp(), id.stamp());
        assert_eq!((entry.origin, entry.oseq), (4, 8));
        assert_eq!(DeletedEntry::Key(entry).oseq(), 8);
    }

    #[test]
    fn a_truncated_or_padded_payload_is_refused() {
        let bytes = DeletedPayload {
            timelines: vec![DeletedTimeline {
                timeline: 1,
                origin: 2,
                oseq: 3,
            }],
            ..DeletedPayload::default()
        }
        .encode();
        assert!(DeletedPayload::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(DeletedPayload::decode(&padded).is_err());
    }
}
