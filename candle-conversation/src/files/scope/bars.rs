//! The bars a range's step decides against (`docs/placed_section_layout.md`
//! §3.4): on the 0–1000 band of a section's hit rate, entry above eviction —
//! hysteresis, so a section near the line does not flap — and the steps a
//! section must stay under the eviction bar before it leaves.

use super::nucleus::EVICT_OF_EVEN;

/// One range's bars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bars {
    /// The rate at or above which a section out of scope enters.
    pub entry: f32,
    /// The rate below which a section in scope counts a step against it.
    pub evict: f32,
    /// The consecutive steps below [`Self::evict`] after which it leaves:
    /// one reprojection cadence.
    pub evict_steps: u32,
}

impl Bars {
    /// The tools' belief selection's bars (`min_score` 800, `evict_score`
    /// 600) and one 64-step cadence.
    pub const DEFAULT: Bars = Bars {
        entry: 800.0,
        evict: 600.0,
        evict_steps: 64,
    };

    /// The knowledge base's bars for one step, taken from the range's shares
    /// of attention (`docs/placed_section_layout.md` §3.4,
    /// `super::nucleus`): its `placed` sections rated `total` together on the
    /// band this step, and `edge` the share of the weakest one locked. A
    /// section out of scope enters at a rate that would hold `edge` — it would
    /// be among what the range is read by — and a placed one counts a step
    /// toward leaving under [`EVICT_OF_EVEN`] of an even share, for one
    /// 64-step cadence.
    pub fn knowledge_base(total: f32, placed: usize, edge: f32) -> Bars {
        let even = match placed {
            0 => 0.0,
            n => total / n as f32,
        };
        Bars {
            entry: (edge * total).min(1000.0),
            evict: (EVICT_OF_EVEN * even).min(1000.0),
            evict_steps: 64,
        }
    }

    /// Why these bars cannot be used, if they cannot.
    pub fn invalid(&self) -> Option<String> {
        if !(0.0..=1000.0).contains(&self.evict) || !(0.0..=1000.0).contains(&self.entry) {
            return Some("bars lie on the 0–1000 band".into());
        }
        if self.evict > self.entry {
            return Some(format!(
                "the eviction bar ({}) is above the entry bar ({})",
                self.evict, self.entry
            ));
        }
        if self.evict_steps == 0 {
            return Some("a section stays under the eviction bar at least one step".into());
        }
        None
    }
}

/// `hits` of a section's `blocks` selected over `queries` queries, on the
/// 0–1000 band: 1000 when every block was selected by every query.
pub fn band(hits: u64, blocks: u64, queries: u64) -> f32 {
    if blocks == 0 || queries == 0 {
        return 0.0;
    }
    (1000.0 * hits as f64 / (blocks * queries) as f64).min(1000.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A rate is hits over blocks and queries, on the 0–1000 band.**
    #[test]
    fn a_rate_is_hits_over_blocks_and_queries() {
        assert_eq!(band(30, 10, 3), 1000.0);
        assert_eq!(band(6, 10, 3), 200.0);
        assert_eq!(band(0, 10, 3), 0.0);
        assert_eq!(band(5, 0, 3), 0.0);
        assert_eq!(band(5, 10, 0), 0.0);
        assert_eq!(band(99, 10, 3), 1000.0, "held to the band");
    }

    #[test]
    fn bars_must_be_on_the_band_with_entry_above_eviction() {
        assert_eq!(Bars::DEFAULT.invalid(), None);
        let kb = Bars::knowledge_base(400.0, 4, 0.375);
        assert_eq!((kb.entry, kb.evict, kb.evict_steps), (150.0, 25.0, 64));
        assert_eq!(kb.invalid(), None);
        let flipped = Bars {
            entry: 500.0,
            evict: 600.0,
            evict_steps: 64,
        };
        assert!(flipped.invalid().unwrap().contains("above the entry bar"));
        let off = Bars {
            entry: 1200.0,
            ..Bars::DEFAULT
        };
        assert!(off.invalid().is_some());
        let none = Bars {
            evict_steps: 0,
            ..Bars::DEFAULT
        };
        assert!(none.invalid().is_some());
    }
}
