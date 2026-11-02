//! What a mission on a character's agenda carries beside the mission itself
//! (`docs/zen_moon_mind.md` §17.1): the title it is named by and how soon it is
//! to be done. A character that takes no orders keeps an agenda of several
//! missions and works the highest; everybody else carries one mission and no
//! agenda.

use serde::{Deserialize, Serialize};

/// Most missions an agenda holds, the one worked among them.
pub const KEEP: usize = 8;

/// How soon a mission on the agenda is to be done. Ordered: `Now` is highest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Later,
    Soon,
    Now,
}

impl Priority {
    /// Every priority, highest first — the grammar's set.
    pub const ALL: [Priority; 3] = [Priority::Now, Priority::Soon, Priority::Later];

    /// The word the grammar and the prompt use for it.
    pub fn as_str(self) -> &'static str {
        match self {
            Priority::Now => "now",
            Priority::Soon => "soon",
            Priority::Later => "later",
        }
    }

    /// The priority a word names, or `None`.
    pub fn parse(word: &str) -> Option<Priority> {
        Priority::ALL
            .into_iter()
            .find(|p| p.as_str().eq_ignore_ascii_case(word.trim()))
    }
}

/// A mission's place on an agenda.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Agendum {
    /// What it is named by, in the grammar and the prompt.
    pub title: String,
    pub priority: Priority,
    /// When it came onto the agenda, by the agenda's own count — what orders
    /// missions of one priority, oldest first.
    pub since: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Now outranks soon outranks later**, and each reads as its word.
    #[test]
    fn priorities_are_ordered_and_named() {
        assert!(Priority::Now > Priority::Soon && Priority::Soon > Priority::Later);
        let words: Vec<&str> = Priority::ALL.iter().map(|p| p.as_str()).collect();
        assert_eq!(words, ["now", "soon", "later"]);
        assert_eq!(Priority::parse(" Soon "), Some(Priority::Soon));
        assert_eq!(Priority::parse("urgent"), None);
        assert_eq!(
            serde_json::to_string(&Priority::Later).unwrap(),
            "\"later\""
        );
    }
}
