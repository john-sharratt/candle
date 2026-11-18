//! The steps of the request in progress (`docs/request_steps.md` §2): the
//! notes this conversation wrote to its own journal within the request, in
//! order, the newest [`MAX_STEPS`].
//!
//! A note is visible only while the turn that wrote it is projected, and a
//! long request sheds its middle tool turns first, so its notes were the first
//! thing to go. Measured 2026-11-18 on a 27-round worker loop: the same finding
//! noted six times, the plan derived again each round. The steps carry them in
//! the system prompt for as long as the request runs; its entry replaces them
//! once it finishes.
//!
//! Nothing new is stored: a step's words are the note call in its turn. The
//! steps grow from each sealed reply ([`RequestSteps::sealed`]) and are read
//! again from the request's turns once after a restart, a takeover or a resume
//! ([`RequestSteps::rebuild`]).

use std::collections::VecDeque;

use journal_core::section::Step;

/// How many steps a request shows at most, newest kept.
pub const MAX_STEPS: usize = 64;

/// See the module documentation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestSteps {
    /// The first turn of the request the steps are of.
    from: Option<u32>,
    steps: VecDeque<Step>,
    /// The number the next step takes, from 1 within its request.
    next: u32,
    /// Whether the steps stand for this conversation as it is: false until the
    /// first seal or rebuild of a process.
    known: bool,
}

impl RequestSteps {
    /// A round of the request opened at turn `from` sealed, its reply writing
    /// `notes`; `finished` when that reply ended the request, whose entry then
    /// replaces its steps.
    pub fn sealed(&mut self, from: u32, notes: Vec<String>, finished: bool) {
        self.known = true;
        if finished {
            self.clear();
            return;
        }
        if self.from != Some(from) {
            self.clear();
            self.from = Some(from);
        }
        for text in notes {
            self.next += 1;
            self.steps.push_back(Step { n: self.next, text });
            while self.steps.len() > MAX_STEPS {
                self.steps.pop_front();
            }
        }
    }

    /// The steps read again from the request in progress — its first turn and
    /// each round's notes, in order — or `None` when no request is in progress.
    pub fn rebuild(&mut self, request: Option<(u32, Vec<Vec<String>>)>) {
        self.clear();
        self.known = true;
        if let Some((from, rounds)) = request {
            for notes in rounds {
                self.sealed(from, notes, false);
            }
        }
    }

    /// Whether the steps stand for the conversation as it is.
    pub fn known(&self) -> bool {
        self.known
    }

    /// The steps, oldest first.
    pub fn steps(&self) -> Vec<Step> {
        self.steps.iter().cloned().collect()
    }

    fn clear(&mut self) {
        self.from = None;
        self.steps.clear();
        self.next = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    fn texts(s: &RequestSteps) -> Vec<(u32, String)> {
        s.steps().into_iter().map(|s| (s.n, s.text)).collect()
    }

    /// **A request's notes become its steps, numbered from 1, in order**; a
    /// round with no note adds none; a new request starts again from 1; the
    /// round that finishes a request leaves no step.
    #[test]
    fn notes_become_steps_and_a_finished_request_leaves_none() {
        let mut s = RequestSteps::default();
        assert!(!s.known());
        s.sealed(4, notes(&["a", "b"]), false);
        s.sealed(4, notes(&[]), false);
        s.sealed(4, notes(&["c"]), false);
        assert!(s.known());
        assert_eq!(
            texts(&s),
            [(1, "a".into()), (2, "b".into()), (3, "c".into())]
        );
        s.sealed(4, notes(&["last"]), true);
        assert!(s.steps().is_empty(), "its entry replaces them");
        s.sealed(9, notes(&["new request"]), false);
        assert_eq!(texts(&s), [(1, "new request".into())]);
        s.sealed(12, notes(&["another"]), false);
        assert_eq!(
            texts(&s),
            [(1, "another".into())],
            "a request it did not see end"
        );
    }

    /// **Past [`MAX_STEPS`] the oldest go and the numbers stay**: the 65th
    /// note drops step 1 and is step 65.
    #[test]
    fn the_newest_sixty_four_are_kept_and_keep_their_numbers() {
        let mut s = RequestSteps::default();
        for i in 1..=MAX_STEPS + 1 {
            s.sealed(0, vec![format!("n{i}")], false);
        }
        let steps = s.steps();
        assert_eq!(steps.len(), MAX_STEPS);
        assert_eq!((steps[0].n, steps[0].text.as_str()), (2, "n2"));
        assert_eq!(steps.last().map(|s| s.n), Some(MAX_STEPS as u32 + 1));
    }

    /// A rebuild from the request's turns reads as the steps grown round by
    /// round; with no request in progress it holds none.
    #[test]
    fn a_rebuild_equals_the_steps_grown_round_by_round() {
        let rounds = vec![notes(&["a"]), notes(&[]), notes(&["b", "c"])];
        let mut grown = RequestSteps::default();
        for r in rounds.clone() {
            grown.sealed(3, r, false);
        }
        let mut rebuilt = RequestSteps::default();
        rebuilt.rebuild(Some((3, rounds)));
        assert_eq!(rebuilt, grown);
        rebuilt.rebuild(None);
        assert!(rebuilt.known());
        assert!(rebuilt.steps().is_empty());
    }
}
