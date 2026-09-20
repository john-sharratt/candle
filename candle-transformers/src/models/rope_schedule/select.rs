//! Choosing a slot's rung (`docs/progressive_yarn.md` §8).
//!
//! The rung is a pure function of the slot's reach — the deepest position the
//! wave writes, less any sliding-window base — so there is no rung state to
//! keep: a slot's reach only grows between rebuilds, so its rung only rises,
//! and it switches exactly when it crosses a ceiling.

use super::schedule::RopeSchedule;

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
