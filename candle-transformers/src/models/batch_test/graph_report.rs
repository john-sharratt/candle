//! The gate's one-line summary of what the wave chains did across a decode.

use candle::cuda_backend::graph::CaptureStats;

/// The difference `after - before`, as the gate prints it: forwards, how
/// finely each was cut into graph launches, how big those were, and how many
/// segments were folded into their executable in place.
pub(crate) fn graph_line(before: &CaptureStats, after: &CaptureStats) -> String {
    let waves = after.waves - before.waves;
    let segments = after.segments - before.segments;
    let nodes = after.nodes - before.nodes;
    let updated = after.updated - before.updated;
    let instantiated = after.instantiated - before.instantiated;
    let per = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    format!(
        "  graphs: {waves} waves, {:.1} segments/wave, {:.1} launches/segment, \
         {updated} updated in place, {instantiated} instantiated",
        per(segments, waves),
        per(nodes, segments),
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
            instantiated: 10,
            nodes: 2_000,
        };
        let after = CaptureStats {
            waves: 7,
            segments: 264,
            updated: 252,
            instantiated: 12,
            nodes: 6_100,
        };
        assert_eq!(
            graph_line(&before, &after),
            "  graphs: 4 waves, 41.0 segments/wave, 25.0 launches/segment, \
             162 updated in place, 2 instantiated"
        );
    }

    #[test]
    fn no_waves_reads_as_zero_rather_than_dividing_by_it() {
        let s = CaptureStats::default();
        assert_eq!(
            graph_line(&s, &s),
            "  graphs: 0 waves, 0.0 segments/wave, 0.0 launches/segment, \
             0 updated in place, 0 instantiated"
        );
    }
}
