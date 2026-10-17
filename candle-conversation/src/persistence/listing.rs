//! What a member holds of one origin's range, for the headers of a catch-up's
//! answer (`docs/replicated_substrate.md` §7.2): each record it holds, by its
//! header alone, and each `oseq` it covers that no record holds any more.
//!
//! The caller lists only a range it covers completely — applied, gone, or its
//! own written records — so an `oseq` with no location is one compaction or a
//! skip on arrival dropped (§6.5, §6.8), never one this member lacks.
//!
//! **Listed from memory.** Each header comes from the location index
//! ([`ListedHeader`]), never off disk, so a listing holds the persistence lock
//! for a walk over memory. Read off disk, a catch-up's headers were a 4 KiB
//! read per record under the lock, and a peer catching up kept the scheduler
//! out of it for 75–93 s at a time.

use super::replica::ListedHeader;
use super::SubstratePersistence;

/// One record a catch-up's headers list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedRecord {
    pub oseq: u64,
    /// The record's type tag.
    pub record_type: u8,
    pub stream_id: u64,
    pub chunk_index: u64,
    /// The record as it travels — its header line and payload, unpadded — with
    /// the line's `crc` counted at its widest: at most nine bytes over.
    pub len: u32,
    pub digest: [u8; 16],
}

impl ListedRecord {
    fn new(oseq: u64, h: ListedHeader) -> ListedRecord {
        ListedRecord {
            oseq,
            record_type: h.record_type,
            stream_id: h.stream_id,
            chunk_index: h.chunk_index,
            len: h.len,
            digest: h.digest,
        }
    }
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
    pub fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        let mut listing = Listing::default();
        for oseq in from..=to {
            match self.locations.listed(origin, oseq) {
                Some(h) => listing.records.push(ListedRecord::new(oseq, h)),
                None => listing.gone.push(oseq),
            }
        }
        listing
    }
}

#[cfg(test)]
mod tests {
    use crate::persistence::record::{decode_header, RecordType};
    use crate::persistence::SubstratePersistence;
    use crate::substrate::Substrate;

    /// Every record held is listed by its header; an `oseq` no record holds is
    /// gone.
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
        let listing = sp.list(node, a.identity.oseq, b.identity.oseq + 1);
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
        assert_eq!(
            listing.gone,
            vec![b.identity.oseq + 1],
            "past the last record, nothing holds the oseq"
        );
    }

    /// What is listed of a record is what reading it gives: the header's
    /// fields, and its travelling length with `crc` at its widest — whether
    /// the record was just appended, or found again at reopen.
    #[test]
    fn a_listing_matches_the_records_it_lists_before_and_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let (node, top, appended) = {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
            for (i, payload) in [&[1u8, 2, 3][..], &[9u8; 5000], &[]].iter().enumerate() {
                sp.append_record(RecordType::Tokens, 0, 4, i as u64, 0, 0, payload)
                    .unwrap();
            }
            sp.commit().unwrap();
            let node = sp.node();
            let top = sp.applied_through(node);
            let listing = sp.list(node, 1, top);
            for r in &listing.records {
                let bytes = sp.read_event(node, r.oseq).unwrap().unwrap();
                let (h, head) = decode_header(&bytes).unwrap();
                let travelling = head + h.payload_len as usize;
                let crc_digits = h.crc.to_string().len();
                assert_eq!(r.len as usize, travelling + 10 - crc_digits);
                assert_eq!(r.digest, h.digest.to_bytes());
                assert_eq!(
                    (r.record_type, r.stream_id, r.chunk_index),
                    (h.record_type.tag(), h.stream_id, h.chunk_index)
                );
            }
            // Sealed, the segment is indexed: the reopen replays its records
            // from header-index digests, which carry no `crc`.
            sp.seal_active().unwrap();
            (node, top, listing)
        };
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        assert_eq!(sp.list(node, 1, top), appended);
    }
}
