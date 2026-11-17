//! Where a warm lift's host time goes, phase by phase.
//!
//! A lift returns once its work is enqueued, so these are the scheduler
//! thread's microseconds: resolving and claiming (`plan`), the host copy into
//! the ring (`fill`), waiting for a ring slot's last scatter (`wait`), the
//! launches (`launch`), and rebuilding the sealed sequences at their new
//! placement (`finish`). The GPU's own copy time is not here — it runs after
//! the lift returns, in stream order.

/// See the module documentation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LiftSplit {
    pub plan_us: u64,
    /// Of `plan_us`: reading the source bands' arenas and lengths, claiming
    /// their destination slots, and resolving the slots' device addresses.
    pub plan_resolve_us: u64,
    pub plan_claim_us: u64,
    pub plan_place_us: u64,
    pub fill_us: u64,
    pub wait_us: u64,
    pub launch_us: u64,
    pub finish_us: u64,
    /// Of `finish_us`: building the lifted chunks' resident records.
    pub finish_records_us: u64,
    /// Ring slots filled.
    pub batches: u64,
    /// Descriptor entries launched.
    pub entries: u64,
    /// Payload bytes lifted.
    pub bytes: u64,
}

impl LiftSplit {
    /// The lift's host time, every phase.
    pub fn total_us(&self) -> u64 {
        self.plan_us + self.fill_us + self.wait_us + self.launch_us + self.finish_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The total is the phases, not their parts: a part is already inside its
    /// phase.
    #[test]
    fn the_total_is_every_phase() {
        let s = LiftSplit {
            plan_us: 1,
            plan_resolve_us: 7,
            plan_claim_us: 7,
            plan_place_us: 7,
            fill_us: 20,
            wait_us: 300,
            launch_us: 4_000,
            finish_us: 50_000,
            finish_records_us: 7,
            batches: 2,
            entries: 9,
            bytes: 1 << 20,
        };
        assert_eq!(s.total_us(), 54_321);
    }
}
