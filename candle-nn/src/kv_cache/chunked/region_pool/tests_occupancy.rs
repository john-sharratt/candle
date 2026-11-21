//! Tests that the span's bands sum to the span and that the frontier is correct.
//!
//! The partition reports every byte of the span in exactly one band, and a
//! tier's peak is read as at most the free ground it can occupy. The frontier is
//! one past the highest region not free. Free regions at the top bring it down
//! one by one, while a hole below a live region leaves it where it is.

use super::*;

#[cfg(test)]
pub(super) mod partition_tests {
    use super::{partition, SpanPartition, BOOTSTRAP_HELD_BYTES, PERSIST_SPAN_BYTES, REGION_BYTES};

    const R: usize = REGION_BYTES;

    /// **Every byte of the span is in exactly one band.** A span of the fixed
    /// block, a dense block of two and a half regions, forty regions and a
    /// half-region tail, then eight regions of weight zone: ten regions held by
    /// KV, three by span tenants, a tier of four and a half regions at its peak.
    #[test]
    fn the_bands_sum_to_the_span() {
        let dense = 2 * R + R / 2;
        let span = PERSIST_SPAN_BYTES + 3 * R + 40 * R + R / 2 + 8 * R;
        let floor = span - 8 * R;
        let p = partition(
            PERSIST_SPAN_BYTES,
            BOOTSTRAP_HELD_BYTES,
            span,
            dense,
            floor,
            13,
            3,
            4 * R + R / 2,
        );
        assert_eq!(
            p,
            SpanPartition {
                held_ground: BOOTSTRAP_HELD_BYTES,
                persist_staging: PERSIST_SPAN_BYTES - BOOTSTRAP_HELD_BYTES,
                dense,
                dense_padding: R / 2,
                kv: 10 * R,
                tenants: 3 * R,
                transient: 4 * R + R / 2,
                regions_free: 27 * R - (4 * R + R / 2),
                slack: R / 2,
                weight_zone: 8 * R,
            }
        );
        assert_eq!(p.sum(), span);
    }

    /// A tier can only stand on unheld ground, so a peak larger than that is
    /// read as all of it, and the bands still close.
    #[test]
    fn a_tier_never_reads_wider_than_the_free_ground() {
        let span = PERSIST_SPAN_BYTES + 6 * R;
        let p = partition(
            PERSIST_SPAN_BYTES,
            BOOTSTRAP_HELD_BYTES,
            span,
            0,
            span,
            4,
            0,
            10 * R,
        );
        assert_eq!((p.kv, p.transient, p.regions_free), (4 * R, 2 * R, 0));
        assert_eq!(p.sum(), span);
    }
}

#[cfg(test)]
pub(super) mod watermark_tests {
    use super::watermark_of;

    /// **The frontier is one past the highest region not free.** Free regions
    /// at the top bring it down one by one; a hole below a live region does
    /// not; every region free is a frontier of zero.
    #[test]
    fn the_frontier_is_one_past_the_highest_region_not_free() {
        assert_eq!(watermark_of(10, [9, 8, 3].into_iter()), 8);
        assert_eq!(watermark_of(10, [3, 1].into_iter()), 10, "holes only");
        assert_eq!(watermark_of(10, [7, 9].into_iter()), 9, "8 is live");
        assert_eq!(watermark_of(4, [0, 1, 2, 3].into_iter()), 0, "all free");
        assert_eq!(watermark_of(0, [].into_iter()), 0);
        assert_eq!(
            watermark_of(5, [12, 4, 6].into_iter()),
            4,
            "free indices past `next` are not in the walk"
        );
    }
}
