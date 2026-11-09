//! Whether each bulb is there (§9.7): asked what it shows every
//! [`EVERY`], one bulb at a time and staggered so twenty do not answer at
//! once, and marked dead after [`DEAD_AFTER`] questions in a row go
//! unanswered. A dead bulb is left out of every scene; one that answers again
//! is back, and is sent its state afresh.

use std::time::{Duration, Instant};

/// How often each bulb is asked.
pub const EVERY: Duration = Duration::from_secs(10);
/// How many unanswered questions make a bulb dead.
pub const DEAD_AFTER: u32 = 3;

#[derive(Debug, Clone, Default)]
struct Bulb {
    /// When it was asked and has not answered since.
    waiting: Option<Instant>,
    missed: u32,
    dead: bool,
}

/// What became of a reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Bulb `i` stopped answering.
    Died(usize),
    /// Bulb `i` answered again.
    Back(usize),
}

#[derive(Debug, Clone)]
pub struct Health {
    bulbs: Vec<Bulb>,
    next: usize,
    last: Option<Instant>,
}

impl Health {
    pub fn new(bulbs: usize) -> Health {
        Health {
            bulbs: vec![Bulb::default(); bulbs],
            next: 0,
            last: None,
        }
    }

    /// The bulb to ask at `now`, if one is due — one every [`EVERY`] divided
    /// among them — and a bulb that has just died for not answering.
    pub fn ask(&mut self, now: Instant) -> (Option<usize>, Option<Change>) {
        let n = self.bulbs.len();
        if n == 0 {
            return (None, None);
        }
        let gap = EVERY / n as u32;
        if self
            .last
            .is_some_and(|l| now.saturating_duration_since(l) < gap)
        {
            return (None, None);
        }
        self.last = Some(now);
        let i = self.next;
        self.next = (i + 1) % n;
        let b = &mut self.bulbs[i];
        let mut change = None;
        if b.waiting.is_some() {
            b.missed += 1;
            if b.missed >= DEAD_AFTER && !b.dead {
                b.dead = true;
                change = Some(Change::Died(i));
            }
        }
        b.waiting = Some(now);
        (Some(i), change)
    }

    /// Bulb `i` answered at `now`: its round trip, when it was waited for,
    /// and whether it came back.
    pub fn answered(&mut self, i: usize, now: Instant) -> (Option<Duration>, Option<Change>) {
        let Some(b) = self.bulbs.get_mut(i) else {
            return (None, None);
        };
        let rtt = b.waiting.take().map(|at| now.saturating_duration_since(at));
        b.missed = 0;
        let change = std::mem::take(&mut b.dead).then_some(Change::Back(i));
        (rtt, change)
    }

    pub fn is_dead(&self, i: usize) -> bool {
        self.bulbs.get(i).is_some_and(|b| b.dead)
    }

    pub fn dead(&self) -> Vec<usize> {
        (0..self.bulbs.len()).filter(|&i| self.is_dead(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The bulbs are asked in turn, staggered across [`EVERY`].**
    #[test]
    fn the_bulbs_are_asked_in_turn_and_staggered() {
        let t = Instant::now();
        let mut h = Health::new(4);
        assert_eq!(h.ask(t).0, Some(0));
        assert_eq!(h.ask(t + Duration::from_millis(100)).0, None, "too soon");
        assert_eq!(h.ask(t + EVERY / 4).0, Some(1));
        assert_eq!(Health::new(0).ask(t), (None, None));
    }

    /// **A bulb is dead after [`DEAD_AFTER`] unanswered questions, and back
    /// when it answers.**
    #[test]
    fn a_bulb_dies_unanswered_and_comes_back_answering() {
        let t = Instant::now();
        let mut h = Health::new(1);
        let mut at = t;
        let mut died = None;
        for _ in 0..=DEAD_AFTER {
            let (asked, change) = h.ask(at);
            assert_eq!(asked, Some(0));
            died = died.or(change);
            at += EVERY;
        }
        assert_eq!(died, Some(Change::Died(0)));
        assert_eq!(h.dead(), [0]);
        let asked = at - EVERY;
        assert_eq!(
            h.answered(0, asked + Duration::from_millis(30)),
            (Some(Duration::from_millis(30)), Some(Change::Back(0))),
            "its round trip, and back"
        );
        assert!(!h.is_dead(0));
        assert_eq!(h.answered(0, at), (None, None), "not asked since");
    }
}
