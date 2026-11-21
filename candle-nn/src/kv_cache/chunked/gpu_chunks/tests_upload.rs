//! Tests for the byte ranges a slot upload copies, and for the staging buffer an
//! upload waits on.
//!
//! Dirty chunks become ranges over the headers and over the records. A run that
//! covers the whole slot is one copy, because its headers run on to where the
//! records begin, and a run that does not touch its records is two ranges. The
//! staging tests cover every combination of which buffer a queued copy may still
//! read and which one the previous upload used.

use super::*;

#[cfg(test)]
pub(super) mod upload_range_tests {
    use super::upload_ranges;

    /// A slot of four chunks with 100-byte records in a layout of four: headers
    /// fill `[0, 64)`, records `[64, 464)`.
    const N: usize = 4;
    const REC: usize = 100;

    /// **A full rebuild is one copy.** Its headers end at 64 where its records
    /// begin, so the two halves are one range — the case a prefill hits for every
    /// `(layer, slot)` it serialises from nothing.
    #[test]
    fn a_whole_slot_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1, 2, 3], N, N, REC), vec![0..464]);
    }

    /// In a layout wider than the live entries the records begin at
    /// `capacity · 16`; a whole-slot run carries its headers on through the
    /// unused header room to them, and is still one copy. Three entries in a
    /// layout of eight: headers `[0, 48)`, room to 128, records `[128, 428)`.
    #[test]
    fn a_whole_slot_in_a_wider_layout_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1, 2], 3, 8, REC), vec![0..428]);
    }

    /// An extend's run — the previous writer and the chunk appended after it —
    /// is its headers and its records, at the layout's record offset. Entries
    /// 2 and 3 of four in a layout of eight: headers `[32, 64)`, records
    /// `[128 + 200, 128 + 400)`.
    #[test]
    fn an_extend_run_is_its_headers_and_its_records() {
        assert_eq!(upload_ranges(&[2, 3], 4, 8, REC), vec![32..64, 328..528]);
    }

    /// A run in the middle of the slot has headers and records far apart: two
    /// ranges, each exactly its run's bytes.
    #[test]
    fn an_interior_run_is_two_ranges() {
        assert_eq!(upload_ranges(&[1, 2], N, N, REC), vec![16..48, 164..364]);
    }

    /// Runs that do not touch stay separate, in run order: each run's headers,
    /// then its records.
    #[test]
    fn separated_runs_stay_separate() {
        assert_eq!(
            upload_ranges(&[0, 2], N, N, REC),
            vec![0..16, 64..164, 32..48, 264..364]
        );
    }

    /// A run that reaches the end of the slot but starts later does not touch
    /// its records: the headers end at 64, the records begin at 64 + 3·100.
    #[test]
    fn a_tail_run_is_two_ranges() {
        assert_eq!(upload_ranges(&[3], N, N, REC), vec![48..64, 364..464]);
    }

    /// A slot of two chunks, whole: headers `[0, 32)` run into records
    /// `[32, 232)`.
    #[test]
    fn a_two_chunk_slot_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1], 2, 2, REC), vec![0..232]);
    }

    /// With no out-of-line record, only the headers travel — the live ones,
    /// never the unused room.
    #[test]
    fn a_slot_without_records_carries_headers_alone() {
        assert_eq!(upload_ranges(&[0, 1, 2, 3], N, N, 0), vec![0..64]);
        assert_eq!(upload_ranges(&[0, 1, 2], 3, 8, 0), vec![0..48]);
        assert_eq!(upload_ranges(&[1], N, N, 0), vec![16..32]);
    }

    /// Nothing dirty, nothing to upload.
    #[test]
    fn no_dirty_chunks_is_no_ranges() {
        assert!(upload_ranges(&[], N, N, REC).is_empty());
    }
}

#[cfg(test)]
pub(super) mod staging_tests {
    use super::choose_staging;

    /// Every combination of which staging buffer may still be read by a queued
    /// copy and which one the previous upload used: the buffer the next upload
    /// takes, and whether it waits.
    #[test]
    fn an_upload_waits_only_when_both_staging_buffers_are_in_flight() {
        // No upload yet: the first buffer, no wait.
        assert_eq!(choose_staging([false, false], None), (0, false));
        // The previous upload's buffer is free: reuse it, so the second buffer
        // is never grown on a path that never collides.
        assert_eq!(choose_staging([false, false], Some(0)), (0, false));
        assert_eq!(choose_staging([false, true], Some(0)), (0, false));
        assert_eq!(choose_staging([false, false], Some(1)), (1, false));
        assert_eq!(choose_staging([true, false], Some(1)), (1, false));
        // The previous upload's buffer is still read by a queued copy: the
        // other one, which is free — no wait.
        assert_eq!(choose_staging([true, false], Some(0)), (1, false));
        assert_eq!(choose_staging([false, true], Some(1)), (0, false));
        // Both are read by queued copies: the other one is the older of the
        // two, and the upload waits for it.
        assert_eq!(choose_staging([true, true], Some(0)), (1, true));
        assert_eq!(choose_staging([true, true], Some(1)), (0, true));
    }
}
