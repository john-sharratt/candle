//! How many calibration cases are in flight at once, and when the window refills.

/// In-flight calibration cases held in the sliding window. Cases are
/// prefill-only (the authored trajectory is prefilled, not decoded), so the
/// whole window co-batches into one prefill forward shape; a wider window just
/// packs more sequences per forward, up to the prefill pass's token budget. 16
/// keeps the peak VRAM footprint of the concurrently-resident lossless case K/V
/// within a tight (16 GB) card while still filling every forward — the window
/// is populated in one pipelined batch, so it actually reaches this width
/// instead of trickling in one case per wave-latency.
pub const BATCH: usize = 16;

/// How many of [`BATCH`]'s slots must be free before the window refills.
/// Creation is served between waves, so every refill costs one wave boundary
/// regardless of how many cases it creates — refilling at half the window
/// amortises that boundary over eight cases rather than paying it for one.
pub const REFILL: usize = BATCH / 2;

/// How many new calibration cases to submit this iteration.
///
/// The warm-up case (`!warmed`) always goes in alone, before anything else is
/// asked of the window: skipping it would leave the shared tool sections
/// unpinned and break every case after it.
///
/// Free host RAM is not an input. The phase runs the whole corpus; a host
/// that runs low is the OS's to page, not a reason to leave cases unrun.
pub fn want(free: usize, warmed: bool, inflight_empty: bool) -> usize {
    if !warmed {
        1
    } else if free >= REFILL || inflight_empty {
        free
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::want;

    #[test]
    fn the_warm_up_case_goes_in_alone() {
        assert_eq!(want(16, false, true), 1);
        assert_eq!(want(16, false, false), 1);
    }

    #[test]
    fn the_ordinary_refill_cadence_holds() {
        // Half the window free: refills.
        assert_eq!(want(8, true, false), 8);
        // Fewer than half free, something still running: waits.
        assert_eq!(want(3, true, false), 0);
        // Nothing in flight regardless of the count: the tail case.
        assert_eq!(want(3, true, true), 3);
    }
}
