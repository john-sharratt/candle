//! The payload of a [`RecordType::Deleted`](super::record::RecordType::Deleted)
//! record (`docs/replicated_substrate.md` §6.8): the tombstones a complete
//! compaction has already run past (*swept*), the first of the two conditions
//! for a tombstone to expire. A timeline's tombstone is named by its timeline,
//! a keyed value's — a custom object marked ceased, such as a world entity
//! that is gone (§9.1) — by its key hash.
//!
//! An expired tombstone leaves nothing behind: a member away long enough to
//! miss one drops what it did not author before it syncs again (§7.8).
//!
//! It is local housekeeping, rewritten whole by every compaction and never
//! replicated, so it carries no identity.
//!
//! Layout, little-endian: `u32` count of timeline entries, then 24 bytes per
//! entry; `u32` swept count, then each swept timeline as a `u64`; `u32` count of
//! keyed entries, then 48 bytes per entry; `u32` swept-key count, then each
//! swept key hash as a `u64`. Entries are what an expired tombstone once left
//! behind: a record of this build writes none, and one it reads has them
//! skipped, so the next compaction rewrites it without them.

use super::identity::RecordIdentity;
use super::le_reader::LeReader;
use super::record::RecordType;
use super::{Result, SubstratePersistence};

/// The bytes of one timeline entry, and of one keyed entry, as the layout
/// holds them.
const TIMELINE_ENTRY_BYTES: usize = 24;
const KEY_ENTRY_BYTES: usize = 48;

/// See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeletedPayload {
    /// Timelines whose tombstone a complete compaction has run past.
    pub swept: Vec<u64>,
    /// Keys whose tombstone a complete compaction has run past, by hash.
    pub swept_keys: Vec<u64>,
}

impl DeletedPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + 8 * (self.swept.len() + self.swept_keys.len()));
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(self.swept.len() as u32).to_le_bytes());
        for tl in &self.swept {
            out.extend_from_slice(&tl.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(self.swept_keys.len() as u32).to_le_bytes());
        for k in &self.swept_keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Result<DeletedPayload> {
        let mut r = LeReader::new(buf, "Deleted");
        let entries = r.u32()? as usize;
        r.bytes(entries * TIMELINE_ENTRY_BYTES)?;
        let swept_n = r.u32()? as usize;
        let mut swept = Vec::with_capacity(swept_n.min(buf.len() / 8));
        for _ in 0..swept_n {
            swept.push(r.u64()?);
        }
        let keys_n = r.u32()? as usize;
        r.bytes(keys_n * KEY_ENTRY_BYTES)?;
        let swept_keys_n = r.u32()? as usize;
        let mut swept_keys = Vec::with_capacity(swept_keys_n.min(buf.len() / 8));
        for _ in 0..swept_keys_n {
            swept_keys.push(r.u64()?);
        }
        r.finish()?;
        Ok(DeletedPayload { swept, swept_keys })
    }
}

impl SubstratePersistence {
    /// Make the swept set durable now, whole.
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
            swept: vec![0x0304],
            swept_keys: vec![0x66],
        };
        let mut want = vec![0, 0, 0, 0];
        want.extend_from_slice(&[1, 0, 0, 0]);
        want.extend_from_slice(&[0x04, 0x03, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[0, 0, 0, 0]);
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

    /// **A record holding entries reads as its swept sets alone**: the
    /// entries an expired tombstone once left are skipped, byte for byte.
    #[test]
    fn entries_are_skipped_on_read() {
        let mut bytes = vec![1, 0, 0, 0];
        bytes.extend_from_slice(&[0xEE; TIMELINE_ENTRY_BYTES]);
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&[0xDD; KEY_ENTRY_BYTES]);
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&[9, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            DeletedPayload::decode(&bytes).unwrap(),
            DeletedPayload {
                swept: vec![7],
                swept_keys: vec![9],
            }
        );
    }

    #[test]
    fn a_truncated_or_padded_payload_is_refused() {
        let bytes = DeletedPayload {
            swept: vec![1],
            swept_keys: Vec::new(),
        }
        .encode();
        assert!(DeletedPayload::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(DeletedPayload::decode(&padded).is_err());
    }
}
