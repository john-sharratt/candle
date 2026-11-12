//! Choosing a slot's rung (`docs/progressive_yarn.md` §8).
//!
//! The rung is a pure function of the slot's reach — the deepest position the
//! wave writes, less any sliding-window base — so there is no rung state to
//! keep: a slot's reach only grows between rebuilds, so its rung only rises,
//! and it switches exactly when it crosses a ceiling. A sequence inside the
//! trained window is on rung 0, the trained RoPE.

use super::schedule::RopeSchedule;

/// Everything a rung is chosen by: the ceilings. The model's rung set and
/// every session that writes headers for it hold the same one, so they pick
/// the same rung for the same reach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RungSelect {
    ceilings: Vec<usize>,
}

impl RungSelect {
    /// Rungs chosen by `ceilings` (ascending).
    pub fn new(ceilings: Vec<usize>) -> candle::Result<Self> {
        if ceilings.is_empty() || ceilings.windows(2).any(|w| w[0] >= w[1]) {
            candle::bail!("rope ceilings must be non-empty and ascending, got {ceilings:?}");
        }
        Ok(Self { ceilings })
    }

    /// One unbounded rung — for a model whose kernels read none.
    pub fn unbounded() -> Self {
        Self {
            ceilings: vec![usize::MAX],
        }
    }

    /// The rung a sequence reaching `reach` positions rotates by: the lowest
    /// whose ceiling covers it. Past the last ceiling is refused.
    pub fn rung_for(&self, reach: usize) -> candle::Result<u32> {
        rung_of(&self.ceilings, reach)
    }

    /// The ceilings, ascending.
    pub fn ceilings(&self) -> &[usize] {
        &self.ceilings
    }

    /// The deepest reach any sequence may have: the last ceiling.
    pub fn reach(&self) -> usize {
        self.ceilings.last().copied().unwrap_or(0)
    }
}

/// The lowest rung whose ceiling covers `reach` positions.
///
/// A reach past the schedule's supported maximum is refused, naming the model's
/// limit: no rung was published for it, and extrapolating plain RoPE beyond it
/// is what the schedule exists to avoid.
pub fn rung_for(schedule: &RopeSchedule, reach: usize) -> candle::Result<u32> {
    rung_of(&schedule.ceilings(), reach)
}

/// [`rung_for`] over a schedule's `ceilings` — what a header writer holds.
pub fn rung_of(ceilings: &[usize], reach: usize) -> candle::Result<u32> {
    match ceilings.iter().position(|&c| reach <= c) {
        Some(r) => Ok(r as u32),
        None => candle::bail!(
            "rope schedule: a reach of {reach} positions is past the supported maximum of {}",
            ceilings.last().copied().unwrap_or(0)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope_schedule::schedule::Rung;

    fn qwen3() -> RopeSchedule {
        RopeSchedule::yarn(
            128,
            1e6,
            32_768,
            vec![
                Rung {
                    ceiling: 32_768,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 65_536,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 131_072,
                    factor: 4.0,
                },
            ],
            true,
        )
        .unwrap()
    }

    /// Each ceiling is its own rung's last reach; one past it is the next rung.
    #[test]
    fn each_ceiling_and_one_past_it() {
        let s = qwen3();
        assert_eq!(rung_for(&s, 0).unwrap(), 0);
        assert_eq!(rung_for(&s, 32_768).unwrap(), 0);
        assert_eq!(rung_for(&s, 32_769).unwrap(), 1);
        assert_eq!(rung_for(&s, 65_536).unwrap(), 1);
        assert_eq!(rung_for(&s, 65_537).unwrap(), 2);
        assert_eq!(rung_for(&s, 131_072).unwrap(), 2);
    }

    /// A short reach is on the trained rung, each ceiling is its own rung's
    /// last reach, and the maximum is refused.
    #[test]
    fn a_selection_climbs_only_past_a_ceiling() {
        let s = RungSelect::new(vec![32_768, 65_536, 131_072]).unwrap();
        assert_eq!(s.rung_for(1).unwrap(), 0);
        assert_eq!(s.rung_for(32_768).unwrap(), 0);
        assert_eq!(s.rung_for(32_769).unwrap(), 1);
        assert_eq!(s.rung_for(65_537).unwrap(), 2);
        assert!(s.rung_for(131_073).is_err());
        assert_eq!(s.reach(), 131_072);
    }

    /// Malformed ceilings are refused.
    #[test]
    fn malformed_ceilings_are_refused() {
        assert!(RungSelect::new(vec![20, 10]).is_err());
        assert!(RungSelect::new(vec![10, 10]).is_err());
        assert!(RungSelect::new(vec![]).is_err());
        assert_eq!(RungSelect::unbounded().rung_for(1 << 30).unwrap(), 0);
    }

    /// Past the supported maximum is refused, for every kind of schedule.
    #[test]
    fn past_the_supported_maximum_is_refused() {
        assert!(rung_for(&qwen3(), 131_073).is_err());
        let plain = RopeSchedule::plain(64, 1e6, 32_768);
        assert_eq!(rung_for(&plain, 32_768).unwrap(), 0);
        assert!(rung_for(&plain, 32_769).is_err());
        let lin = RopeSchedule::linear(128, 1e6, 2.0, 65_536);
        assert!(rung_for(&lin, 65_537).is_err());
    }
}
