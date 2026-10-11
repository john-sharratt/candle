//! What a member holds of one origin's range, for the headers of a catch-up's
//! answer (`docs/replicated_substrate.md` §7.2): each record it holds, by its
//! header alone, and each `oseq` it covers that no record holds any more.
//!
//! The caller lists only a range it covers completely — applied, gone, or its
//! own written records — so an `oseq` with no location is one compaction or a
//! skip on arrival dropped (§6.5, §6.8), never one this member lacks.

use super::record::{decode_header, HEADER_PROBE_SIZE};
use super::{Result, SubstratePersistence};

/// One record a catch-up's headers list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedRecord {
    pub oseq: u64,
    /// The record's type tag.
    pub record_type: u8,
    pub stream_id: u64,
    pub chunk_index: u64,
    /// The record as it travels: its header line and payload, unpadded.
    pub len: u32,
    pub digest: [u8; 16],
}

/// One origin's range, as this member holds it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Listing {
    pub records: Vec<ListedRecord>,
    /// The `oseq`s in the range no record holds any more.
    pub gone: Vec<u64>,
}

impl SubstratePersistence {
    /// What this member holds of `origin`'s `from..=to`. See the module
    /// documentation.
    pub fn list(&mut self, origin: u64, from: u64, to: u64) -> Result<Listing> {
        self.flush_staged()?;
        let mut listing = Listing::default();
        let mut probe = vec![0u8; HEADER_PROBE_SIZE];
        for oseq in from..=to {
            let Some(at) = self.location_of(origin, oseq) else {
                listing.gone.push(oseq);
                continue;
            };
            self.segments.read_into(at.segment, at.offset, &mut probe)?;
            let (header, header_bytes) = decode_header(&probe)?;
            listing.records.push(ListedRecord {
                oseq,
                record_type: header.record_type.tag(),
                stream_id: header.stream_id,
                chunk_index: header.chunk_index,
                len: (header_bytes as u64 + header.payload_len) as u32,
                digest: header.digest.to_bytes(),
            });
        }
        Ok(listing)
    }
}

#[cfg(test)]
mod tests {
    use crate::persistence::record::RecordType;
    use crate::persistence::SubstratePersistence;
    use crate::substrate::Substrate;

    /// Every record held is listed by its header; an `oseq` whose record a
    /// rewrite dropped is gone.
    #[test]
    fn held_records_are_listed_and_dropped_ones_gone() {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let node = sp.node();
        let a = sp
            .append_record(RecordType::Tokens, 0, 9, 0, 0, 0, &[1, 2, 3])
            .unwrap();
        let b = sp
            .append_record(RecordType::Tokens, 0, 9, 1, 0, 0, &[4, 5])
            .unwrap();
        let listing = sp.list(node, a.identity.oseq, b.identity.oseq + 1).unwrap();
        assert_eq!(listing.records.len(), 2);
        let first = &listing.records[0];
        assert_eq!(
            (
                first.oseq,
                first.record_type,
                first.stream_id,
                first.chunk_index
            ),
            (a.identity.oseq, RecordType::Tokens.tag(), 9, 0)
        );
        let bytes = sp.read_event(node, a.identity.oseq).unwrap().unwrap();
        let line = bytes.iter().position(|b| *b == b'\n').unwrap() + 1;
        assert_eq!(
            first.len as usize,
            line + 3,
            "unpadded: header line and payload"
        );
        assert_eq!(
            listing.gone,
            vec![b.identity.oseq + 1],
            "past the last record, nothing holds the oseq"
        );
    }
}
