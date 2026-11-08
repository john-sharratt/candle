//! Whether placing guidance helps: per turn, which drawn units were placed,
//! how strongly the turn's scans matched each while it was, and which were
//! read (`docs/guidance_layer.md` §3).
//!
//! - **missed**: read by the turn while not placed;
//! - **read anyway**: placed, and read in the same turn — placement offered
//!   it and did not spare the read;
//! - **relevant**: placed, never read, and the turn's scans kept matching it —
//!   its peak fresh score reached the working set's `min_momentum`;
//! - **idle**: placed, never read, and no scan matched it.
//!
//! The counts are logged once per turn (`zend::guidance`, "guidance this
//! turn"), where the self-improvement mission reads them: "missed" should fall
//! as use-exemplars accumulate, and "relevant" should rise against "read
//! anyway".

use std::collections::{BTreeMap, BTreeSet};

/// One turn's placements and reads of drawn units, by workspace-relative path.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Tally {
    /// Each unit placed, with the best fresh score a scan gave it while it was.
    placed: BTreeMap<String, f32>,
    read: BTreeSet<String>,
}

/// What a turn's [`Tally`] comes to, each list in path order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Counts {
    pub missed: Vec<String>,
    pub read_anyway: Vec<String>,
    pub relevant: Vec<String>,
    pub idle: Vec<String>,
}

impl Tally {
    /// Note the drawn units placed for one of the turn's rounds, each with the
    /// fresh score that round's scan gave it.
    pub fn place(&mut self, placed: impl IntoIterator<Item = (String, f32)>) {
        for (path, fresh) in placed {
            let peak = self.placed.entry(path).or_insert(fresh);
            *peak = peak.max(fresh);
        }
    }

    /// Note a drawn unit one of the turn's rounds read.
    pub fn read(&mut self, path: String) {
        self.read.insert(path);
    }

    /// Whether the turn placed or read any drawn unit.
    pub fn is_empty(&self) -> bool {
        self.placed.is_empty() && self.read.is_empty()
    }

    /// The turn's counts, a placed unit unread counting as relevant once its
    /// peak fresh score reached `min_momentum`.
    pub fn counts(&self, min_momentum: f32) -> Counts {
        let mut counts = Counts {
            missed: self
                .read
                .iter()
                .filter(|p| !self.placed.contains_key(*p))
                .cloned()
                .collect(),
            ..Counts::default()
        };
        for (path, peak) in &self.placed {
            let list = if self.read.contains(path) {
                &mut counts.read_anyway
            } else if *peak >= min_momentum {
                &mut counts.relevant
            } else {
                &mut counts.idle
            };
            list.push(path.clone());
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    fn at(p: &[(&str, f32)]) -> Vec<(String, f32)> {
        p.iter().map(|(s, f)| (s.to_string(), *f)).collect()
    }

    /// **A placed unit the turn read is read anyway; one it did not read is
    /// relevant once a scan matched it to `min_momentum`, idle otherwise; one
    /// read unplaced is missed** — over every round, on each unit's peak.
    #[test]
    fn placed_read_and_matched_make_the_four_counts() {
        let mut t = Tally::default();
        t.place(at(&[("g/deploy.md", 400.0), ("g/test.md", 20.0)]));
        t.read("g/deploy.md".into());
        t.place(at(&[
            ("g/test.md", 150.0),
            ("j/notes.yaml", 30.0),
            ("g/deploy.md", 0.0),
        ]));
        t.read("g/release.md".into());
        assert_eq!(
            t.counts(100.0),
            Counts {
                missed: paths(&["g/release.md"]),
                read_anyway: paths(&["g/deploy.md"]),
                relevant: paths(&["g/test.md"]),
                idle: paths(&["j/notes.yaml"]),
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
