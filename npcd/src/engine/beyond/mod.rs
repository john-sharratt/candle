//! What a character reaches beyond its world (`docs/zen_moon_mind.md` §16):
//! the weather, the web, a sum, its own journal and the world's notebook, the
//! guides, a picture — offered by its personality's `reaches:`, wherever it
//! stands, each act its tool's own definition.
//!
//! | Module | |
//! |---|---|
//! | [`acts`] | the acts, built from their tools' definitions |
//! | [`carry`] | how a parameter travels through the grammar, and back |
//! | [`call`] | where each act runs, and with what arguments |
//! | [`answer`] | a tool's answer as the mind reads it |
//! | [`notes`] | the world's notebook, in the substrate |
//! | [`within`] | the acts a turn is offered |
//!
//! The running itself reaches the network and is the runtime's
//! (`crate::authority::beyond`).

pub mod acts;
pub mod answer;
pub mod call;
pub mod carry;
pub mod notes;
pub mod within;

use serde_json::Value;

pub use acts::BEYOND_ACTS;

/// A group of acts a personality names in `reaches:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// `outside`: the weather, the web, a request and its session, the time
    /// and the DNS.
    Outside,
    /// `reckoning`: a sum, a conversion, a draw at random.
    Reckoning,
    /// `memory`: the character's own journal, the world's notebook and the
    /// guides.
    Memory,
    /// `making`: a picture drawn or changed.
    Making,
}

impl Group {
    pub const ALL: [Group; 4] = [
        Group::Outside,
        Group::Reckoning,
        Group::Memory,
        Group::Making,
    ];

    /// The word `reaches:` names it by.
    pub fn name(self) -> &'static str {
        match self {
            Group::Outside => "outside",
            Group::Reckoning => "reckoning",
            Group::Memory => "memory",
            Group::Making => "making",
        }
    }

    /// The group `word` names.
    pub fn parse(word: &str) -> Option<Group> {
        Group::ALL.into_iter().find(|r| r.name() == word.trim())
    }

    /// The acts it offers, each named as its tool.
    pub fn acts(self) -> &'static [&'static str] {
        match self {
            Group::Outside => &[
                "weather",
                "web_search",
                "web_fetch",
                "http_session_open",
                "http_request",
                "http_session_list",
                "http_session_close",
                "datetime",
                "dns_lookup",
            ],
            Group::Reckoning => &["calculator", "unit_convert", "random"],
            Group::Memory => &[
                "journal_read",
                "journal_grep",
                "notes_list",
                "notes_read",
                "notes_search",
                "notes_write",
                "guide_read",
                "guide_list",
            ],
            Group::Making => &["image_generate", "image_edit"],
        }
    }
}

/// The groups `words` name, in [`Group::ALL`]'s order and each once, and the
/// words that name none.
pub fn reaches_of(words: &[String]) -> (Vec<Group>, Vec<String>) {
    let mut known: Vec<Group> = Vec::new();
    let mut unknown = Vec::new();
    for word in words {
        match Group::parse(word) {
            Some(r) => known.push(r),
            None => unknown.push(word.clone()),
        }
    }
    known.sort_unstable();
    known.dedup();
    (known, unknown)
}

/// The groups a personality's record `body` names in `reaches:`, and the
/// words there that name none. A personality that writes none reaches nothing.
pub fn declared(body: &Value) -> (Vec<Group>, Vec<String>) {
    let words: Vec<String> = body
        .get("reaches")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|w| w.as_str().map_or_else(|| w.to_string(), str::to_string))
        .collect();
    reaches_of(&words)
}

/// Whether `tool` is one of these acts.
pub fn is_mine(tool: &str) -> bool {
    Group::ALL.iter().any(|g| g.acts().contains(&tool))
}

/// Whether `groups` reach `tool`.
pub fn reached(groups: &[Group], tool: &str) -> bool {
    groups.iter().any(|r| r.acts().contains(&tool))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **The groups are the catalogue, exactly**: every act of every group is
    /// in it once, and nothing in it belongs to no group.
    #[test]
    fn the_groups_are_the_catalogue() {
        let grouped: Vec<&str> = Group::ALL.iter().flat_map(|r| r.acts()).copied().collect();
        let catalogue: Vec<&str> = BEYOND_ACTS.iter().map(|t| t.name).collect();
        assert_eq!(grouped, catalogue);
        let mut once = grouped.clone();
        once.sort_unstable();
        once.dedup();
        assert_eq!(once.len(), grouped.len(), "an act is in two groups");
    }

    /// **A group is named by its word**; one that names none is kept apart so
    /// it can be said, and a group named twice is one.
    #[test]
    fn reaches_are_read_from_their_words() {
        let words = ["making", " outside ", "outside", "magic"].map(String::from);
        assert_eq!(
            reaches_of(&words),
            (
                vec![Group::Outside, Group::Making],
                vec!["magic".to_string()]
            )
        );
        assert_eq!(Group::parse("memory"), Some(Group::Memory));
        assert_eq!(Group::parse("Memory"), None);
    }

    /// **A personality's `reaches:` is read off its record**; one that writes
    /// none, or writes it as no list, reaches nothing.
    #[test]
    fn a_personalitys_reaches_are_read_off_its_record() {
        assert_eq!(
            declared(&json!({"id": "zen-moon", "reaches": ["outside", "memory", 7]})),
            (vec![Group::Outside, Group::Memory], vec!["7".to_string()])
        );
        assert_eq!(declared(&json!({"id": "maker"})), (Vec::new(), Vec::new()));
        assert_eq!(
            declared(&json!({"reaches": "outside"})),
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn a_group_reaches_its_acts_and_no_other() {
        assert!(reached(&[Group::Outside], "weather"));
        assert!(reached(&[Group::Outside], "http_session_open"));
        assert!(!reached(&[Group::Outside], "calculator"));
        assert!(reached(&[Group::Reckoning, Group::Making], "image_edit"));
        assert!(!reached(&[], "weather"));
        assert!(is_mine("notes_write"));
        assert!(!is_mine("cluster_status"));
    }
}
