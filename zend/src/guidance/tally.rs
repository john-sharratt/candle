//! Whether placing guidance helps: per turn, which drawn units were placed and
//! which were read (`docs/guidance_layer.md` §3).
//!
//! - **hit**: placed, and read in the same turn;
//! - **missed**: read by the turn while not placed;
//! - **idle**: placed, and the turn never read it.
//!
//! The counts are logged once per turn (`zend::guidance`, "guidance this
//! turn"), where the self-improvement mission reads them: the missed count
//! should fall as use-exemplars accumulate, and a unit idle turn after turn is
//! placed on a calibration that no longer fits.

use std::collections::BTreeSet;

/// One turn's placements and reads of drawn units, by workspace-relative path.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Tally {
    placed: BTreeSet<String>,
    read: BTreeSet<String>,
}

/// What a turn's [`Tally`] comes to, each list in path order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Counts {
    pub hit: Vec<String>,
    pub missed: Vec<String>,
    pub idle: Vec<String>,
}

impl Tally {
    /// Note the drawn units placed for one of the turn's rounds.
    pub fn place(&mut self, paths: impl IntoIterator<Item = String>) {
        self.placed.extend(paths);
    }

    /// Note a drawn unit one of the turn's rounds read.
    pub fn read(&mut self, path: String) {
        self.read.insert(path);
    }

    /// Whether the turn placed or read any drawn unit.
    pub fn is_empty(&self) -> bool {
        self.placed.is_empty() && self.read.is_empty()
    }

    /// The turn's counts.
    pub fn counts(&self) -> Counts {
        Counts {
            hit: self.placed.intersection(&self.read).cloned().collect(),
            missed: self.read.difference(&self.placed).cloned().collect(),
            idle: self.placed.difference(&self.read).cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    /// **A placed unit the turn read is a hit, one it did not is idle, and one
    /// it read unplaced is missed**, over every round of the turn.
    #[test]
    fn placed_and_read_make_hit_missed_and_idle() {
        let mut t = Tally::default();
        t.place(paths(&["g/deploy.md", "g/test.md"]));
        t.read("g/deploy.md".into());
        t.place(paths(&["g/deploy.md", "j/notes.md"]));
        t.read("g/release.md".into());
        assert_eq!(
            t.counts(),
            Counts {
                hit: paths(&["g/deploy.md"]),
                missed: paths(&["g/release.md"]),
                idle: paths(&["g/test.md", "j/notes.md"]),
            }
        );
    }

    #[test]
    fn a_turn_that_touched_no_guidance_is_empty() {
        let mut t = Tally::default();
        assert!(t.is_empty());
        t.place(Vec::new());
        assert!(t.is_empty());
        t.read("g/deploy.md".into());
        assert!(!t.is_empty());
    }
}
