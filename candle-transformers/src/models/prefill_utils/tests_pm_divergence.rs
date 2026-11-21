//! Tests the rules that decide when a cached position map is stale, and when a
//! stale map is worth a loud line.
//!
//! An unchanged layout reuses the map. A trailing empty chunk from windowed creep
//! makes the map stale but is not anomalous. A covered-token move, or a change in
//! the slots a wave holds, is anomalous, because creep cannot produce it. A writer
//! index the cached map never described does not by itself force a rebuild.

use super::*;

/// The position-map staleness rules, tested off the GPU. See
/// [`classify_pm_divergence`] and [`SharedPm`].
#[cfg(test)]
pub(super) mod pm_divergence_tests {
    use super::classify_pm_divergence;

    /// An unchanged layout is neither stale nor anomalous — the common case, a
    /// later layer of a settled forward reusing the cached map.
    #[test]
    fn an_unchanged_layout_reuses_the_cached_map() {
        let cached = [(4u32, 128u32), (3, 96)];
        let write = [3u32, 2];
        let live = [(4u32, 128u32, 3u32), (3, 96, 2)];
        assert_eq!(
            classify_pm_divergence(&cached, &write, &live),
            (false, false)
        );
    }

    /// **The benign windowed-creep skew: a trailing 0-token chunk.** The slice
    /// count moves by one but the covered-token count does not — stale (rebuild),
    /// but NOT anomalous (no loud line).
    #[test]
    fn a_trailing_empty_chunk_is_stale_but_not_anomalous() {
        let cached = [(4u32, 128u32)];
        let write = [3u32];
        // One more slice, same covered tokens — the phantom writer chunk.
        let live = [(5u32, 128u32, 3u32)];
        assert_eq!(
            classify_pm_divergence(&cached, &write, &live),
            (true, false)
        );
    }

    /// A covered-token move is a real layout divergence, not creep — stale AND
    /// anomalous, so it is worth a loud, attributed line.
    #[test]
    fn a_covered_token_move_is_anomalous() {
        let cached = [(4u32, 128u32)];
        let write = [3u32];
        let live = [(4u32, 160u32, 3u32)];
        assert_eq!(classify_pm_divergence(&cached, &write, &live), (true, true));
    }

    /// A change in the wave's slot membership between layers can never be a creep
    /// artefact — stale AND anomalous.
    #[test]
    fn a_membership_change_is_anomalous() {
        let cached = [(4u32, 128u32), (3, 96)];
        let write = [3u32, 2];
        let live = [(4u32, 128u32, 3u32)];
        assert_eq!(classify_pm_divergence(&cached, &write, &live), (true, true));
    }

    /// The writer slipping into a chunk the map does not describe is stale (the
    /// map's write region names the wrong slice) but not, on its own, anomalous.
    #[test]
    fn a_writer_move_within_bounds_is_stale_but_not_anomalous() {
        let cached = [(5u32, 128u32)];
        let write = [2u32]; // cached writer 2 < live n_slices 5, and it moved
        let live = [(5u32, 128u32, 3u32)];
        assert_eq!(
            classify_pm_divergence(&cached, &write, &live),
            (true, false)
        );
    }

    /// A cached writer index that is out of the live slice array's bounds does
    /// not by itself force a rebuild — the guard requires the cached writer to be
    /// a real slice the map describes.
    #[test]
    fn a_writer_move_the_cached_map_never_described_is_not_stale() {
        let cached = [(4u32, 128u32)];
        // Cached writer 4 is not < live n_slices 4, so the writer-move arm is
        // suppressed; nothing else moved.
        let write = [4u32];
        let live = [(4u32, 128u32, 1u32)];
        assert_eq!(
            classify_pm_divergence(&cached, &write, &live),
            (false, false)
        );
    }
}
