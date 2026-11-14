//! What a window's `compaction` band was spent on: the gate that decides
//! whether to pack, the KV pools' packs, and the span tenants' — the
//! recurrent state, the QSA index, the gallery, the creeps' residuals.
//!
//! The phase chart draws compaction as one band; which pass costs it is what
//! making it faster needs, so the window's close logs this beside it.

use std::time::Duration;

/// One window's compaction, by part, in microseconds, and how many passes of
/// each kind ran.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompactionSplit {
    pub gate_us: u64,
    pub kv_us: u64,
    pub kv_passes: u32,
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

    /// One span tenants' pack.
    pub fn tenants(&mut self, took: Duration) {
        self.tenants_us += took.as_micros() as u64;
        self.tenant_passes += 1;
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
                tenants_us: 4_000,
                tenant_passes: 2,
            }
        );
    }
}
