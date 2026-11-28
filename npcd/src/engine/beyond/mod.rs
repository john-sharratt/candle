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

    /// What it lets the character do, as a conversation is told it.
    pub fn doing(self) -> &'static str {
        match self {
            Group::Outside => {
                "look up the weather and its forecast anywhere, search and read the web, \
                 make requests to the internet, tell the time anywhere and look up a name"
            }
            Group::Reckoning => "work out sums, conversions and random draws",
            Group::Memory => {
                "read your own journal and the guides, and keep and read notes in the \
                 world's notebook"
            }
            Group::Making => "draw a picture, or change one",
        }
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

/// What a conversation is told `groups` let the character do once the call is
/// over — `None` for a character that reaches nothing.
///
/// **A conversation is shown nothing of the world, so it has to be told.**
/// Asked the weather, Zen's conversation answered "I do not have access to
/// real-time weather data" while its mind held `weather`.
pub fn away_from_the_phone(groups: &[Group]) -> Option<String> {
    let doing: Vec<&str> = groups.iter().map(|g| g.doing()).collect();
    let (last, rest) = doing.split_last()?;
    let all = match rest.is_empty() {
        true => last.to_string(),
        false => format!("{}; and {last}", rest.join("; ")),
    };
    Some(format!(
        "Away from the phone you can also {all}. Asked for one of these, say you will — \
         never that you cannot, or have no way to know."
    ))
}

/// The acts beyond the world a turn always describes when it offers them,
/// however they rank: what the commonest questions from outside — the
/// weather, the time, a sum, anything else — are answered with.
///
/// **Ranked with the rest, the narrow ones lost to the broad ones.** A turn
/// describes ten acts by rank beside a connected terminal's verbs; asked the
/// weather, the time in Hanoi and a sum, Zen called `web_fetch` and
/// `web_search` for all three and never once `weather`, `datetime` or
/// `calculator` — and its conversation answered that it had no live data.
pub const ALWAYS_SHOWN: [&str; 4] = ["weather", "datetime", "calculator", "web_search"];

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
    use crate::engine::tools::shown_wherever_offered;

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

    /// **A conversation is told what its character reaches, in order, and of
    /// one that reaches nothing, nothing.**
    #[test]
    fn a_conversation_is_told_what_its_character_reaches() {
        assert_eq!(away_from_the_phone(&[]), None);
        assert_eq!(
            away_from_the_phone(&[Group::Reckoning]).unwrap(),
            "Away from the phone you can also work out sums, conversions and random draws. \
             Asked for one of these, say you will — never that you cannot, or have no way to \
             know."
        );
        assert_eq!(
            away_from_the_phone(&[Group::Reckoning, Group::Making]).unwrap(),
            "Away from the phone you can also work out sums, conversions and random draws; and \
             draw a picture, or change one. Asked for one of these, say you will — never that \
             you cannot, or have no way to know."
        );
    }

    /// **The acts always shown are reached acts, each in a group**, so none
    /// is shown to a character whose personality does not reach it.
    #[test]
    fn the_acts_always_shown_are_reached_acts() {
        for name in ALWAYS_SHOWN {
            assert!(is_mine(name), "{name}");
            let act = acts::of(name).unwrap();
            assert!(shown_wherever_offered(&act.tool), "{name}");
        }
        assert!(!shown_wherever_offered(
            &acts::of("web_fetch").unwrap().tool
        ));
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
