//! Grouping records to relocate into the reads that fetch them.
//!
//! Maintenance and compaction move live records verbatim, and reading one
//! record at a time is a syscall per record — so adjacent records are read
//! together, one stripe per run. A run is bounded, though: a segment of
//! back-to-back live KV chunks is one unbroken run of gigabytes, and reading
//! it whole put the entire run in one buffer. Measured live: a 4.3 GB segment
//! with 3.4 GB live asked the allocator for 3.3 GB in one piece, which failed,
//! and the failure aborted the daemon. A stripe therefore ends at
//! [`MAX_STRIPE_BYTES`], at a record boundary.

/// The most one stripe read covers — unless a single record is larger, which
/// is read alone.
pub(super) const MAX_STRIPE_BYTES: u64 = 64 * 1024 * 1024;

/// The stripes over `spans` — `(offset, size)` of each record, sorted by
/// offset — as half-open index ranges `(first, past_last)`: a stripe holds a
/// run of records each starting where the one before ended, and grows no
/// larger than `cap` bytes past its first record.
pub(super) fn stripes(spans: &[(u64, u64)], cap: u64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        let start = spans[i].0;
        let mut end = start + spans[i].1;
        let mut j = i + 1;
        while j < spans.len() && spans[j].0 == end && end + spans[j].1 - start <= cap {
            end += spans[j].1;
            j += 1;
        }
        out.push((i, j));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A contiguous run is one stripe; a gap starts another.**
    #[test]
    fn a_contiguous_run_is_one_stripe_and_a_gap_starts_another() {
        let spans = [(0, 10), (10, 10), (20, 5), (40, 10), (50, 10)];
        assert_eq!(stripes(&spans, 1000), [(0, 3), (3, 5)]);
    }

    /// **A run longer than the cap is cut at a record boundary**, each stripe
    /// no larger than the cap.
    #[test]
    fn a_run_past_the_cap_is_cut_at_a_record_boundary() {
        let spans: Vec<(u64, u64)> = (0..10).map(|i| (i * 10, 10)).collect();
        assert_eq!(stripes(&spans, 30), [(0, 3), (3, 6), (6, 9), (9, 10)]);
        assert_eq!(
            stripes(&spans, 29),
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        );
    }

    /// A record larger than the cap is a stripe of its own, never skipped.
    #[test]
    fn a_record_past_the_cap_is_read_alone() {
        let spans = [(0, 5), (5, 100), (105, 5)];
        assert_eq!(stripes(&spans, 20), [(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn nothing_to_read_is_no_stripe() {
        assert!(stripes(&[], 64).is_empty());
    }
}
