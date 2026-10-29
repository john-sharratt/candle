//! The fraction of a store's record bytes that are dead weight.
//!
//! `live` is what the substrate's index still references — read-back records,
//! the current copy of each metadata and resident record; everything else in
//! the segments (superseded records, tombstoned timelines' records) is dead.
//! `live` can't exceed `total` for this log's own records, so the ratio is a
//! real fraction; the clamp covers inherited-log streams whose bytes are not
//! in this log's `total`.

/// `total` record bytes of which `live` are live, as a dead fraction in
/// `[0, 1]`; an empty log is none dead.
pub fn dead_ratio_of(total: u64, live: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (total.saturating_sub(live) as f32 / total as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dead_fraction_of_the_bytes() {
        assert_eq!(dead_ratio_of(1_000, 250), 0.75);
        assert_eq!(dead_ratio_of(1_000, 1_000), 0.0);
        assert_eq!(dead_ratio_of(0, 0), 0.0);
        assert_eq!(dead_ratio_of(1_000, 4_000), 0.0, "inherited live bytes");
    }
}
