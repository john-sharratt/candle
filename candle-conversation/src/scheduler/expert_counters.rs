//! The expert pipeline's running counters the wave line reports, as a window's
//! own: hits and misses, how the misses were served, what was promoted, and
//! what read-ahead did.

use candle_transformers::models::expert_lre::PipelineStats;

/// The counters the wave line reads off [`PipelineStats`], each a running
/// total; [`Self::since`] turns two readings into a window's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExpertCounters {
    pub hits: usize,
    pub misses: usize,
    /// Misses the GEMM workers promoted into VRAM as they served them.
    pub promoted: usize,
    /// Misses that waited on the stager's pack read.
    pub cold: usize,
    /// Misses the host's expert pool computed, crossing no link.
    pub cpu: usize,
    /// Experts read ahead of their row into a ring slot.
    pub ahead_claims: usize,
    /// Passes the read-ahead gate decided at, and those it read ahead through.
    pub ahead_passes: usize,
    pub ahead_passes_on: usize,
    /// Times the gate changed its decision.
    pub ahead_switches: usize,
}

impl ExpertCounters {
    pub(super) fn of(s: &PipelineStats) -> Self {
        Self {
            hits: s.expert_hits,
            misses: s.expert_misses,
            promoted: s.worker_promotions,
            cold: s.worker_cold,
            cpu: s.worker_cpu,
            ahead_claims: s.ahead_claims,
            ahead_passes: s.ahead_passes,
            ahead_passes_on: s.ahead_passes_on,
            ahead_switches: s.ahead_switches,
        }
    }

    /// What was counted between `then` and this reading.
    pub(super) fn since(&self, then: &Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(then.hits),
            misses: self.misses.saturating_sub(then.misses),
            promoted: self.promoted.saturating_sub(then.promoted),
            cold: self.cold.saturating_sub(then.cold),
            cpu: self.cpu.saturating_sub(then.cpu),
            ahead_claims: self.ahead_claims.saturating_sub(then.ahead_claims),
            ahead_passes: self.ahead_passes.saturating_sub(then.ahead_passes),
            ahead_passes_on: self.ahead_passes_on.saturating_sub(then.ahead_passes_on),
            ahead_switches: self.ahead_switches.saturating_sub(then.ahead_switches),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field of a window is the difference of its two readings.
    #[test]
    fn a_window_is_each_counter_since_the_last_reading() {
        let then = ExpertCounters {
            hits: 100,
            misses: 10,
            promoted: 4,
            cold: 1,
            cpu: 3,
            ahead_claims: 7,
            ahead_passes: 20,
            ahead_passes_on: 5,
            ahead_switches: 2,
        };
        let now = ExpertCounters {
            hits: 190,
            misses: 25,
            promoted: 9,
            cold: 1,
            cpu: 8,
            ahead_claims: 40,
            ahead_passes: 50,
            ahead_passes_on: 30,
            ahead_switches: 3,
        };
        assert_eq!(
            now.since(&then),
            ExpertCounters {
                hits: 90,
                misses: 15,
                promoted: 5,
                cold: 0,
                cpu: 5,
                ahead_claims: 33,
                ahead_passes: 30,
                ahead_passes_on: 25,
                ahead_switches: 1,
            }
        );
    }

    /// Read off the pipeline's stats field for field.
    #[test]
    fn read_off_the_pipelines_stats() {
        let s = PipelineStats {
            expert_hits: 1,
            expert_misses: 2,
            worker_promotions: 3,
            worker_cold: 4,
            worker_cpu: 5,
            ahead_claims: 6,
            ahead_passes: 7,
            ahead_passes_on: 8,
            ahead_switches: 9,
            ..PipelineStats::default()
        };
        assert_eq!(
            ExpertCounters::of(&s),
            ExpertCounters {
                hits: 1,
                misses: 2,
                promoted: 3,
                cold: 4,
                cpu: 5,
                ahead_claims: 6,
                ahead_passes: 7,
                ahead_passes_on: 8,
                ahead_switches: 9,
            }
        );
    }
}
