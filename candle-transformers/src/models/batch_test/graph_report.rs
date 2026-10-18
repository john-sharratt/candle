//! The gate's one-line summary of what the wave chains did across a decode.

use std::cmp::Reverse;

use candle::cuda_backend::graph::CaptureStats;

use crate::models::expert_lre::cpu_pool::stats::{stats as pool_stats, ClassStats};

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
    let recording_us = after.recording_us - before.recording_us;
    let eager = after.eager_segments - before.eager_segments;
    let leased = after.leased_eager - before.leased_eager;
    let banned = after.banned - before.banned;
    let benched = after.benched - before.benched;
    let reopened = after.reopened - before.reopened;
    let (phase, held) = (after.phase.name(), after.execs_held);
    let per = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    format!(
        "  graphs: {waves} waves, {:.1} segments/wave, {:.1} launches/segment, \
         {updated} updated in place ({reshaped} reshaped), {instantiated} instantiated; \
         fold µs each: in place {:.1}, reshaped {:.1}, instantiated {:.1}; \
         recording µs/segment {:.1}; {eager} run as issued ({leased} behind a lease); \
         library {phase}, {held} held, {banned} banned, {benched} benched, {reopened} reopened",
        per(segments, waves),
        per(nodes, segments),
        per(in_place_us, updated - reshaped),
        per(reshaped_us, reshaped),
        per(instantiated_us, instantiated),
        per(recording_us, segments),
    )
}

pub(crate) fn print_graph_line(before: &CaptureStats, after: &CaptureStats) {
    println!("{}", graph_line(before, after));
}

/// What the CPU expert pool served (`expert_lre::cpu_pool`), as of one moment.
pub(crate) struct CpuServed {
    layers: u64,
    experts: u64,
    rows: u64,
    ns: u64,
    phases: [ClassStats; 2],
}

impl CpuServed {
    pub(crate) fn now() -> Self {
        use crate::models::expert_lre::cpu_pool::service as s;
        Self {
            layers: s::served_total(),
            experts: s::served_experts(),
            rows: s::served_rows(),
            ns: s::served_ns(),
            phases: pool_stats(),
        }
    }

    /// One class's phases since `before`, per layer and per expert (µs).
    fn phase_line(name: &str, now: &ClassStats, before: &ClassStats) {
        let c = now.since(before);
        if c.layers == 0 {
            return;
        }
        let per = |ns: u64, n: u64| {
            if n == 0 {
                0.0
            } else {
                ns as f64 / n as f64 / 1e3
            }
        };
        println!(
            "    {name}: {} layers, {:.1} experts/layer, {:.1} rows/layer · µs a layer: parse {:.1}, \
             compute {:.1}, cold wait {:.1}, publish {:.1}, device wait {:.1} · compute µs an expert \
             {:.1} · runs {:.2}/layer, {:.1} participants, join {:.1} µs, occupancy {:.0}%",
            c.layers,
            c.experts as f64 / c.layers as f64,
            c.rows as f64 / c.layers as f64,
            per(c.parse_ns, c.layers),
            per(c.compute_ns, c.layers),
            per(c.cold_wait_ns, c.layers),
            per(c.publish_ns, c.layers),
            per(c.device_wait_ns, c.layers),
            per(c.compute_ns, c.experts),
            c.runs as f64 / c.layers as f64,
            if c.runs == 0 { 0.0 } else { c.participants as f64 / c.runs as f64 },
            per(c.join_ns, c.runs),
            if c.in_job_ns == 0 { 0.0 } else { 100.0 * c.work_ns as f64 / c.in_job_ns as f64 },
        );
    }

    /// Print what the pool served since `self`, and whether it is calibrated.
    pub(crate) fn print_since(&self) {
        let after = Self::now();
        let p = *crate::models::expert_lre::cpu_pool::params::shared()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let layers = after.layers - self.layers;
        let ms = (after.ns - self.ns) as f64 / 1e6;
        println!(
            "  cpu experts: {} · {layers} layers served, {} experts, {} rows, {ms:.1} ms pool time ({:.1} µs a layer)",
            match (p.enabled, p.measured.is_some()) {
                (false, _) => "off",
                (true, false) => "on, not yet calibrated",
                (true, true) => "on, calibrated",
            },
            after.experts - self.experts,
            after.rows - self.rows,
            if layers == 0 { 0.0 } else { ms * 1e3 / layers as f64 },
        );
        Self::phase_line("decode-sized", &after.phases[0], &self.phases[0]);
        Self::phase_line("prompt-sized", &after.phases[1], &self.phases[1]);
        let s = p.split();
        println!(
            "    split: cut n≤{} · cpu µs n1/n2/n4 {:.0}/{:.0}/{:.0} · gpu µs n1 {:.0} · copy µs {:.0} · \
             handoff µs {:.0} · read µs {:.0} · max experts {} · cap rows {}",
            s.max_n,
            s.cpu_n_ns[1] / 1e3,
            s.cpu_n_ns[2] / 1e3,
            s.cpu_n_ns[4] / 1e3,
            s.gpu_n_ns[1] / 1e3,
            s.copy_ns / 1e3,
            s.handoff_ns / 1e3,
            s.read_shared_ns / 1e3,
            s.max_experts,
            s.cap_rows,
        );
    }
}

/// Stream leases that waited for another thread's recording segment
/// (`record_gate`), as of one moment: the wait threads off the forward pay,
/// which the graph line does not show.
pub(crate) struct LeaseWaits {
    total: (u64, u64),
    sites: Vec<(String, u64, u64)>,
}

impl LeaseWaits {
    pub(crate) fn now() -> Self {
        Self {
            total: candle::cuda_backend::graph::lease_wait_total(),
            sites: candle::cuda_backend::graph::lease_wait_sites(),
        }
    }

    /// Print what waited since `self`, the busiest sites first.
    pub(crate) fn print_since(&self) {
        let after = Self::now();
        let n = after.total.0 - self.total.0;
        let us = after.total.1 - self.total.1;
        println!(
            "  lease waits: {n} leases waited {:.1} ms in all ({:.1} µs each)",
            us as f64 / 1_000.0,
            if n == 0 { 0.0 } else { us as f64 / n as f64 },
        );
        let mut sites: Vec<(String, u64, u64)> = after
            .sites
            .into_iter()
            .map(|(site, n, us)| {
                let (n0, us0) = self
                    .sites
                    .iter()
                    .find(|(s, _, _)| *s == site)
                    .map_or((0, 0), |(_, n, us)| (*n, *us));
                (site, n - n0, us - us0)
            })
            .filter(|(_, n, _)| *n > 0)
            .collect();
        sites.sort_by_key(|s| Reverse(s.2));
        for (site, n, us) in sites.iter().take(8) {
            println!("    {:>9.1} ms over {n:>6} at {site}", *us as f64 / 1_000.0);
        }
    }
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
            recording_us: 10_000,
            eager_segments: 5,
            leased_eager: 1,
            ..CaptureStats::default()
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
            recording_us: 26_400,
            eager_segments: 12,
            leased_eager: 4,
            ..CaptureStats::default()
        };
        assert_eq!(
            graph_line(&before, &after),
            "  graphs: 4 waves, 41.0 segments/wave, 25.0 launches/segment, \
             162 updated in place (7 reshaped), 2 instantiated; \
             fold µs each: in place 50.0, reshaped 500.0, instantiated 500.0; \
             recording µs/segment 100.0; 7 run as issued (3 behind a lease); \
             library learn, 0 held, 0 banned, 0 benched, 0 reopened"
        );
    }

    #[test]
    fn no_waves_reads_as_zero_rather_than_dividing_by_it() {
        let s = CaptureStats::default();
        assert_eq!(
            graph_line(&s, &s),
            "  graphs: 0 waves, 0.0 segments/wave, 0.0 launches/segment, \
             0 updated in place (0 reshaped), 0 instantiated; \
             fold µs each: in place 0.0, reshaped 0.0, instantiated 0.0; \
             recording µs/segment 0.0; 0 run as issued (0 behind a lease); \
             library learn, 0 held, 0 banned, 0 benched, 0 reopened"
        );
    }
}
