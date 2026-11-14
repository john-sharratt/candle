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
            }
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
