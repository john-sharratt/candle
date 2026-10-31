//! Missions one character orders of others, and what came back
//! (`docs/zen_moon_mind.md` §8).
//!
//! A character that takes no orders gives them: it posts a mission on the order
//! table ([`super::missions::Missions::post`]), a character that takes orders
//! carries it, and its report comes back here, to the poster. What this holds is
//! the poster's side of that — the reports filed to it, kept for it to read at
//! the table, and the notices the world owes: a mission posted wakes whoever
//! waits for work, and a report filed tells its poster.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::mission::Outcome;

/// How many filed reports are kept: the newest, for their posters to read.
pub const KEPT: usize = 50;

/// A report filed on a mission somebody posted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filed {
    /// The body that posted the mission.
    pub poster: String,
    /// The body that carried it.
    pub carrier: String,
    /// The mission's brief.
    pub brief: String,
    pub outcome: Outcome,
    /// What the carrier reported: what it found, or what stopped it.
    pub notes: String,
}

/// Something the world owes a character's attention, delivered at its next
/// moment (`Runtime::moment`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Notice {
    /// A mission was posted: for the body named, or, with none, for whoever
    /// waits at the table.
    Posted { to: Option<String>, brief: String },
    /// A mission somebody posted was reported.
    Reported(Filed),
}

/// The poster's side of the order table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orders {
    /// Reports filed, oldest first, at most [`KEPT`], each with the number it
    /// was filed under.
    filed: Vec<(u64, Filed)>,
    /// The number the next report is filed under.
    next: u64,
    /// How far each poster has read: the number of the last report it read.
    read: BTreeMap<String, u64>,
    /// Notices not yet delivered, oldest first.
    notices: Vec<Notice>,
}

impl Orders {
    /// A mission was posted for `to`, or for whoever waits.
    pub fn posted(&mut self, to: Option<&str>, brief: &str) {
        self.notices.push(Notice::Posted {
            to: to.map(str::to_string),
            brief: brief.to_string(),
        });
    }

    /// A report on a posted mission: kept for its poster, who is told.
    pub fn filed(&mut self, report: Filed) {
        self.notices.push(Notice::Reported(report.clone()));
        self.next += 1;
        self.filed.push((self.next, report));
        let over = self.filed.len().saturating_sub(KEPT);
        self.filed.drain(..over);
    }

    /// The notices owed since this was last asked, oldest first.
    pub fn take_notices(&mut self) -> Vec<Notice> {
        std::mem::take(&mut self.notices)
    }

    /// Whether a report waits that `poster` has not read.
    pub fn unread_for(&self, poster: &str) -> bool {
        let through = self.read.get(poster).copied().unwrap_or(0);
        self.filed
            .iter()
            .any(|(n, f)| *n > through && f.poster == poster)
    }

    /// The reports `poster` has not read, newest first — read now.
    ///
    /// **What was read is not read again.** A reading that came back the same
    /// every time it was asked for was asked for again and again: an answer
    /// changes when something new is filed, and only then is there anything to
    /// read.
    pub fn read_by(&mut self, poster: &str) -> Vec<Filed> {
        let through = self.read.get(poster).copied().unwrap_or(0);
        let unread: Vec<Filed> = self
            .filed
            .iter()
            .rev()
            .filter(|(n, f)| *n > through && f.poster == poster)
            .map(|(_, f)| f.clone())
            .collect();
        self.read.insert(poster.to_string(), self.next);
        unread
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filed(poster: &str, brief: &str) -> Filed {
        Filed {
            poster: poster.to_string(),
            carrier: "wick".to_string(),
            brief: brief.to_string(),
            outcome: Outcome::Pass,
            notes: "the blast door stands locked".to_string(),
        }
    }

    /// **A report is kept for its poster and owed to it**, newest first, and a
    /// post is owed to whoever it is for.
    #[test]
    fn a_report_is_kept_for_its_poster_and_it_is_told() {
        let mut o = Orders::default();
        o.posted(None, "check the blast door");
        o.filed(filed("zen", "check the blast door"));
        o.filed(filed("someone-else", "read the board"));
        o.filed(filed("zen", "watch earth"));
        assert!(o.unread_for("zen"));
        let briefs: Vec<String> = o.read_by("zen").into_iter().map(|f| f.brief).collect();
        assert_eq!(briefs, ["watch earth", "check the blast door"]);
        assert!(!o.unread_for("zen"), "read once");
        assert!(o.read_by("zen").is_empty());
        assert!(o.unread_for("someone-else"), "each poster reads its own");
        o.filed(filed("zen", "read a cradle"));
        let briefs: Vec<String> = o.read_by("zen").into_iter().map(|f| f.brief).collect();
        assert_eq!(briefs, ["read a cradle"], "only what came since");
        assert_eq!(
            o.take_notices(),
            [
                Notice::Posted {
                    to: None,
                    brief: "check the blast door".to_string()
                },
                Notice::Reported(filed("zen", "check the blast door")),
                Notice::Reported(filed("someone-else", "read the board")),
                Notice::Reported(filed("zen", "watch earth")),
                Notice::Reported(filed("zen", "read a cradle")),
            ]
        );
        assert!(o.take_notices().is_empty(), "each is owed once");
    }

    /// **Only the newest [`KEPT`] reports are kept.**
    #[test]
    fn only_the_newest_reports_are_kept() {
        let mut o = Orders::default();
        for n in 0..KEPT + 3 {
            o.filed(filed("zen", &format!("job {n}")));
        }
        let kept: Vec<String> = o.read_by("zen").into_iter().map(|f| f.brief).collect();
        assert_eq!(kept.len(), KEPT);
        assert_eq!(kept[0], format!("job {}", KEPT + 2));
        assert_eq!(kept[KEPT - 1], "job 3");
    }
}
