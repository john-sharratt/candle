//! The escalation ladder: which step to take next with a character that stays
//! unwell, and when to stop pressing.
//!
//! Pure, driven by a clock the caller passes in, so the whole sequence of
//! nudge, settle, verify and escalate is testable without waiting.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// One step of force, from gentlest to last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rung {
    /// A generic thought that puts the errand back in front of the character.
    Nudge,
    /// The errand again, naming whoever is pulling the character off it.
    Restate,
    /// The mission's standing text, delivered as a fresh instruction.
    Refresh,
    /// Stop pressing and surface the character to an operator.
    Flag,
}

impl Rung {
    pub fn as_str(self) -> &'static str {
        match self {
            Rung::Nudge => "nudge",
            Rung::Restate => "restate",
            Rung::Refresh => "refresh",
            Rung::Flag => "flag",
        }
    }
}

/// How a rung that was taken turned out once the character had settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// The character is well again.
    Held,
    /// It is not; the next rung is due.
    Failed,
}

/// What one scan of one character calls for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Advance {
    pub resolution: Option<Resolution>,
    pub act: Option<Rung>,
}

/// One character's place on the ladder.
#[derive(Debug, Default)]
pub struct NpcState {
    step: usize,
    acted_at: Option<Duration>,
    awaiting: bool,
    flagged: bool,
}

impl NpcState {
    /// Whether a flag has been raised and nothing more will be done.
    pub fn flagged(&self) -> bool {
        self.flagged
    }
}

pub struct Ladder {
    rungs: Vec<Rung>,
    cooldown: Duration,
    settle: Duration,
}

impl Ladder {
    /// `rungs` must not be empty.
    pub fn new(rungs: Vec<Rung>, cooldown: Duration, settle: Duration) -> Self {
        assert!(!rungs.is_empty(), "a ladder needs a rung");
        Self {
            rungs,
            cooldown,
            settle,
        }
    }

    /// Whether the last rung is still being given time to work, in which case
    /// the character is neither asked nor pressed.
    pub fn settling(&self, state: &NpcState, now: Duration) -> bool {
        state.awaiting && !self.settled(state, now)
    }

    fn settled(&self, state: &NpcState, now: Duration) -> bool {
        state.acted_at.is_some_and(|at| now >= at + self.settle)
    }

    /// Take one scan's reading of a character: `unwell` is whether any module
    /// raised a concern.
    pub fn advance(&self, state: &mut NpcState, now: Duration, unwell: bool) -> Advance {
        let mut resolution = None;
        if state.awaiting && self.settled(state, now) {
            state.awaiting = false;
            if unwell {
                resolution = Some(Resolution::Failed);
                state.step = (state.step + 1).min(self.rungs.len() - 1);
            } else {
                resolution = Some(Resolution::Held);
            }
        }
        if !unwell {
            state.step = 0;
            state.flagged = false;
            return Advance {
                resolution,
                act: None,
            };
        }
        let holding = state.awaiting
            || state.flagged
            || state.acted_at.is_some_and(|at| now < at + self.cooldown);
        if holding {
            return Advance {
                resolution,
                act: None,
            };
        }
        let rung = self.rungs[state.step];
        state.acted_at = Some(now);
        match rung {
            Rung::Flag => state.flagged = true,
            _ => state.awaiting = true,
        }
        Advance {
            resolution,
            act: Some(rung),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn ladder() -> Ladder {
        Ladder::new(
            vec![Rung::Nudge, Rung::Restate, Rung::Flag],
            secs(120),
            secs(30),
        )
    }

    #[test]
    fn a_well_character_is_left_alone() {
        let l = ladder();
        let mut s = NpcState::default();
        assert_eq!(
            l.advance(&mut s, secs(0), false),
            Advance {
                resolution: None,
                act: None
            }
        );
    }

    #[test]
    fn the_first_concern_takes_the_first_rung() {
        let l = ladder();
        let mut s = NpcState::default();
        assert_eq!(l.advance(&mut s, secs(10), true).act, Some(Rung::Nudge));
    }

    #[test]
    fn the_character_is_not_pressed_while_a_rung_settles() {
        let l = ladder();
        let mut s = NpcState::default();
        l.advance(&mut s, secs(10), true);
        assert!(l.settling(&s, secs(39)));
        assert!(!l.settling(&s, secs(40)));
        assert_eq!(l.advance(&mut s, secs(39), true).act, None);
    }

    #[test]
    fn a_character_well_after_settling_resets_the_ladder() {
        let l = ladder();
        let mut s = NpcState::default();
        l.advance(&mut s, secs(10), true);
        let a = l.advance(&mut s, secs(40), false);
        assert_eq!(a.resolution, Some(Resolution::Held));
        assert_eq!(a.act, None);
        assert_eq!(l.advance(&mut s, secs(500), true).act, Some(Rung::Nudge));
    }

    #[test]
    fn a_character_still_unwell_is_taken_up_a_rung_once_the_cooldown_passes() {
        let l = ladder();
        let mut s = NpcState::default();
        l.advance(&mut s, secs(10), true);
        let early = l.advance(&mut s, secs(40), true);
        assert_eq!(early.resolution, Some(Resolution::Failed));
        assert_eq!(early.act, None, "the cooldown runs from the last act");
        assert_eq!(l.advance(&mut s, secs(130), true).act, Some(Rung::Restate));
    }

    #[test]
    fn a_flag_is_raised_once_and_then_nothing_more_is_done() {
        let l = ladder();
        let mut s = NpcState::default();
        let mut now = 0;
        let mut acts = Vec::new();
        for _ in 0..12 {
            if let Some(r) = l.advance(&mut s, secs(now), true).act {
                acts.push(r);
            }
            now += 130;
        }
        assert_eq!(acts, vec![Rung::Nudge, Rung::Restate, Rung::Flag]);
        assert!(s.flagged());
    }

    #[test]
    fn health_clears_a_flag_and_starts_over() {
        let l = ladder();
        let mut s = NpcState::default();
        let mut now = 0;
        for _ in 0..3 {
            l.advance(&mut s, secs(now), true);
            now += 130;
        }
        assert!(s.flagged());
        l.advance(&mut s, secs(now), false);
        assert!(!s.flagged());
        assert_eq!(
            l.advance(&mut s, secs(now + 1), true).act,
            Some(Rung::Nudge)
        );
    }

    #[test]
    fn a_ladder_with_one_repeating_rung_repeats_it_each_cooldown() {
        let l = Ladder::new(vec![Rung::Nudge], secs(100), secs(10));
        let mut s = NpcState::default();
        assert_eq!(l.advance(&mut s, secs(0), true).act, Some(Rung::Nudge));
        assert_eq!(l.advance(&mut s, secs(50), true).act, None);
        assert_eq!(l.advance(&mut s, secs(100), true).act, Some(Rung::Nudge));
    }
}
