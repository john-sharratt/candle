//! The gate's one-line summary of what the wave chains did across a decode.

use candle::cuda_backend::graph::CaptureStats;

/// The difference `after - before`, as the gate prints it: forwards, how
/// finely each was cut into graph launches, how big those were, how many
/// segments were folded into their executable in place, and the host time each
/// kind of fold cost on average.
pub(crate) fn graph_line(before: &CaptureStats, after: &CaptureStats) -> String {
    let waves = after.waves - before.waves;
    let segments = after.segments - before.segments;
    let nodes = after.nodes - before.nodes;
    let updated = after.updated - before.updated;
    let reshaped = after.reshaped - before.reshaped;
    let instantiated = after.instantiated - before.instantiated;
    let in_place_us = after.in_place_us - before.in_place_us;
    let reshaped_us = after.reshaped_us - before.reshaped_us;
    let instantiated_us = after.instantiated_us - before.instantiated_us;
    let per = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    format!(
        "  graphs: {waves} waves, {:.1} segments/wave, {:.1} launches/segment, \
         {updated} updated in place ({reshaped} reshaped), {instantiated} instantiated; \
         fold µs each: in place {:.1}, reshaped {:.1}, instantiated {:.1}",
        per(segments, waves),
        per(nodes, segments),
        per(in_place_us, updated - reshaped),
        per(reshaped_us, reshaped),
        per(instantiated_us, instantiated),
    )
}

pub(crate) fn print_graph_line(before: &CaptureStats, after: &CaptureStats) {
    println!("{}", graph_line(before, after));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_reports_the_difference_across_the_decode() {
        let before = CaptureStats {
            waves: 3,
            segments: 100,
            updated: 90,
            reshaped: 30,
            instantiated: 10,
            nodes: 2_000,
            in_place_us: 1_000,
            reshaped_us: 9_000,
            instantiated_us: 5_000,
        };
        let after = CaptureStats {
            waves: 7,
            segments: 264,
            updated: 252,
            reshaped: 37,
            instantiated: 12,
            nodes: 6_100,
            in_place_us: 8_750,
            reshaped_us: 12_500,
            instantiated_us: 6_000,
        };
        assert_eq!(
            graph_line(&before, &after),
            "  graphs: 4 waves, 41.0 segments/wave, 25.0 launches/segment, \
             162 updated in place (7 reshaped), 2 instantiated; \
             fold µs each: in place 50.0, reshaped 500.0, instantiated 500.0"
        );
    }

    #[test]
    fn no_waves_reads_as_zero_rather_than_dividing_by_it() {
        let s = CaptureStats::default();
        assert_eq!(
            graph_line(&s, &s),
            "  graphs: 0 waves, 0.0 segments/wave, 0.0 launches/segment, \
             0 updated in place (0 reshaped), 0 instantiated; \
             fold µs each: in place 0.0, reshaped 0.0, instantiated 0.0"
        );
    }
}
