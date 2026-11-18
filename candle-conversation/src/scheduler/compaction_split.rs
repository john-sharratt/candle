//! What a window's `compaction` band was spent on: the gate that decides
//! whether to pack, the KV pools' packs, and the span tenants' — the
//! recurrent state, the QSA index, the gallery, the creeps' residuals.
//!
//! The phase chart draws compaction as one band; which pass costs it is what
//! making it faster needs, so the window's close logs this beside it.

use std::time::Duration;

/// The stages a KV pass times itself by (`CompactionReport::timings`), in the
/// order [`CompactionSplit::kv_stage`] takes them.
pub(super) const KV_STAGES: [&str; 9] = [
    "quiesce",
    "plan",
    "claim",
    "copy",
    "barrier",
    "sweep",
    "mint",
    "invalidate",
    "publish",
];

/// The parts a span tenants' pass is timed by, in the order it runs them: each
/// tenant's pack, then reading the frontier it left.
pub(super) const TENANT_PARTS: [&str; 4] = ["recurrent", "creep", "gallery", "judge"];

/// A part of a span tenants' pass, as an index into [`TENANT_PARTS`].
#[derive(Debug, Clone, Copy)]
pub(super) enum TenantPart {
    Recurrent,
    Creep,
    Gallery,
    Judge,
}

/// One window's compaction, by part, in microseconds, and how many passes of
/// each kind ran.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompactionSplit {
    pub gate_us: u64,
    pub kv_us: u64,
    pub kv_passes: u32,
    /// The KV passes' own stages, by [`KV_STAGES`].
    pub kv_stage_us: [u64; KV_STAGES.len()],
    /// Chunks the KV passes moved.
    pub kv_moves: u64,
    pub tenants_us: u64,
    pub tenant_passes: u32,
    /// The span tenants' passes by part, by [`TENANT_PARTS`].
    pub tenant_part_us: [u64; TENANT_PARTS.len()],
    /// Regions the frontier fell by across the KV passes — what they bought.
    pub kv_frontier_drop: u64,
    /// Regions the frontier fell by across the span tenants' passes.
    pub tenants_frontier_drop: u64,
}

impl CompactionSplit {
    /// The gate's reading of the pools, on a wave it was consulted.
    pub fn gate(&mut self, took: Duration) {
        self.gate_us += took.as_micros() as u64;
    }

    /// One KV pack.
    pub fn kv(&mut self, took: Duration) {
        self.kv_us += took.as_micros() as u64;
        self.kv_passes += 1;
    }

    /// One KV pass's own stage times, by [`KV_STAGES`], and the chunks it moved.
    pub fn kv_stage(&mut self, stages: [Duration; KV_STAGES.len()], moves: usize) {
        for (sum, took) in self.kv_stage_us.iter_mut().zip(stages) {
            *sum += took.as_micros() as u64;
        }
        self.kv_moves += moves as u64;
    }

    /// One span tenants' pack.
    pub fn tenants(&mut self, took: Duration) {
        self.tenants_us += took.as_micros() as u64;
        self.tenant_passes += 1;
    }

    /// One part of a span tenants' pass.
    pub fn tenant_part(&mut self, part: TenantPart, took: Duration) {
        self.tenant_part_us[part as usize] += took.as_micros() as u64;
    }

    /// The span tenants' parts as `name=ms` pairs, for the log.
    pub fn tenant_parts_ms(&self) -> String {
        TENANT_PARTS
            .iter()
            .zip(&self.tenant_part_us)
            .map(|(name, us)| format!("{name}={}", us / 1000))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// How far a KV pass lowered the frontier, from `before` to `after` regions.
    pub fn kv_lowered(&mut self, before: usize, after: usize) {
        self.kv_frontier_drop += before.saturating_sub(after) as u64;
    }

    /// How far a span tenants' pass lowered the frontier.
    pub fn tenants_lowered(&mut self, before: usize, after: usize) {
        self.tenants_frontier_drop += before.saturating_sub(after) as u64;
    }

    /// The KV stages as `name=ms` pairs, for the log.
    pub fn kv_stages_ms(&self) -> String {
        KV_STAGES
            .iter()
            .zip(&self.kv_stage_us)
            .map(|(name, us)| format!("{name}={}", us / 1000))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each part accumulates its own time, and a pass is counted where it ran.
    #[test]
    fn each_part_counts_its_own_time_and_passes() {
        let mut s = CompactionSplit::default();
        s.gate(Duration::from_micros(40));
        s.gate(Duration::from_micros(35));
        s.kv(Duration::from_millis(12));
        s.tenants(Duration::from_micros(2_500));
        s.tenants(Duration::from_micros(1_500));
        assert_eq!(
            s,
            CompactionSplit {
                gate_us: 75,
                kv_us: 12_000,
                kv_passes: 1,
                kv_stage_us: [0; KV_STAGES.len()],
                kv_moves: 0,
                tenants_us: 4_000,
                tenant_passes: 2,
                tenant_part_us: [0; TENANT_PARTS.len()],
                kv_frontier_drop: 0,
                tenants_frontier_drop: 0,
            }
        );
    }

    /// The frontier's fall is summed per kind of pass, and a pass that raised
    /// it — ground claimed while it ran — counts as no fall.
    #[test]
    fn the_frontiers_fall_is_summed_per_kind_of_pass() {
        let mut s = CompactionSplit::default();
        s.kv_lowered(455, 450);
        s.kv_lowered(450, 452);
        s.tenants_lowered(450, 447);
        s.tenants_lowered(447, 447);
        assert_eq!((s.kv_frontier_drop, s.tenants_frontier_drop), (5, 3));
    }

    /// A span tenants' pass's parts sum part by part, and read back as
    /// `name=ms`.
    #[test]
    fn a_tenant_passs_parts_sum_and_read_back_by_name() {
        let ms = Duration::from_millis;
        let mut s = CompactionSplit::default();
        s.tenant_part(TenantPart::Recurrent, ms(80));
        s.tenant_part(TenantPart::Gallery, ms(12));
        s.tenant_part(TenantPart::Gallery, ms(9));
        s.tenant_part(TenantPart::Judge, ms(1));
        assert_eq!(
            s.tenant_parts_ms(),
            "recurrent=80 creep=0 gallery=21 judge=1"
        );
    }

    /// A KV pass's stages sum stage by stage, and read back as `name=ms`.
    #[test]
    fn a_kv_passs_stages_sum_and_read_back_by_name() {
        let ms = Duration::from_millis;
        let mut s = CompactionSplit::default();
        s.kv_stage([30, 9, 2, 1, 4, 3, 0, 1, 2].map(ms), 120);
        s.kv_stage([10, 1, 0, 1, 0, 1, 1, 0, 0].map(ms), 30);
        assert_eq!(s.kv_moves, 150);
        assert_eq!(
            s.kv_stages_ms(),
            "quiesce=40 plan=10 claim=2 copy=2 barrier=4 sweep=4 mint=1 invalidate=1 publish=2"
        );
    }
}
