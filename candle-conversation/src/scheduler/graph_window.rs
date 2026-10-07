//! The wave line's graph-capture field: one window's share of the device's
//! cumulative capture counters.

use candle::cuda_backend::graph::CaptureStats;

/// ` | graphs waves=… segs=… fold=…/…ms reshape=…/…ms inst=…/…ms rec=…ms` for
/// the window between two snapshots of the device's counters, or nothing when
/// no wave was captured in it. Each `n/ms` pair is a count of segments and the
/// host time they cost: folded into an executable of their own shape, folded
/// into one of another shape, and instantiated afresh. `rec` is the host time
/// the window spent recording segments.
pub(super) fn graph_window(before: &CaptureStats, after: &CaptureStats) -> String {
    let waves = after.waves.saturating_sub(before.waves);
    if waves == 0 {
        return String::new();
    }
    let d = |a: u64, b: u64| a.saturating_sub(b);
    format!(
        " | graphs waves={waves} segs={} fold={}/{}ms reshape={}/{}ms inst={}/{}ms rec={}ms",
        d(after.segments, before.segments),
        d(after.updated, before.updated).saturating_sub(d(after.reshaped, before.reshaped)),
        d(after.in_place_us, before.in_place_us) / 1000,
        d(after.reshaped, before.reshaped),
        d(after.reshaped_us, before.reshaped_us) / 1000,
        d(after.instantiated, before.instantiated),
        d(after.instantiated_us, before.instantiated_us) / 1000,
        d(after.recording_us, before.recording_us) / 1000,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(waves: u64, segments: u64, instantiated: u64, recording_us: u64) -> CaptureStats {
        CaptureStats {
            waves,
            segments,
            instantiated,
            recording_us,
            ..CaptureStats::default()
        }
    }

    #[test]
    fn reports_the_window_not_the_lifetime() {
        let before = CaptureStats {
            updated: 500,
            reshaped: 20,
            in_place_us: 10_000,
            reshaped_us: 4_000,
            instantiated_us: 30_000,
            ..stats(100, 900, 40, 50_000)
        };
        let after = CaptureStats {
            updated: 977,
            reshaped: 22,
            in_place_us: 19_500,
            reshaped_us: 5_100,
            instantiated_us: 36_250,
            ..stats(160, 1_380, 43, 81_500)
        };
        // 477 updates in the window, 2 of them reshapes: 475 plain folds.
        assert_eq!(
            graph_window(&before, &after),
            " | graphs waves=60 segs=480 fold=475/9ms reshape=2/1ms inst=3/6ms rec=31ms"
        );
    }

    #[test]
    fn a_window_without_captured_waves_says_nothing() {
        assert_eq!(
            graph_window(&stats(7, 70, 1, 900), &stats(7, 70, 1, 900)),
            ""
        );
    }
}
