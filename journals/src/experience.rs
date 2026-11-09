//! What an episode npcd writes for a bot was to that bot
//! (`docs/npc_conversations.md` §5.1): something it heard, said, wrote down,
//! promised, a change to its mission, an act of its own, or a change in the
//! world that concerns it — and, when there is one, the journal of the party
//! it was with.
//!
//! A typed field rather than words in the text, so a conversation's catch-up
//! can rank what it reads (§6.1) without parsing prose.

use std::fmt;

use crate::path::{JournalPath, Viewer};

/// What happened, to the bot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExperienceKind {
    /// A message or speech addressed to it landed.
    Heard,
    /// A line it voiced landed.
    Said,
    /// A conversation turn's text went to its journal, verbatim.
    Wrote,
    /// A promise of its landed.
    Promised,
    /// Its mission changed, through a conversation or an order.
    Mission,
    /// An act of its own changed the world.
    Did,
    /// The world changed in a way that concerns it.
    Changed,
}

impl ExperienceKind {
    pub const ALL: [ExperienceKind; 7] = [
        Self::Heard,
        Self::Said,
        Self::Wrote,
        Self::Promised,
        Self::Mission,
        Self::Did,
        Self::Changed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Heard => "heard",
            Self::Said => "said",
            Self::Wrote => "wrote",
            Self::Promised => "promised",
            Self::Mission => "mission",
            Self::Did => "did",
            Self::Changed => "changed",
        }
    }

    pub fn parse(s: &str) -> Result<ExperienceKind, String> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| {
                format!(
                    "`{s}` is not an experience: heard, said, wrote, promised, mission, did or \
                     changed"
                )
            })
    }

    /// Which a catch-up keeps first when its budget is short (§6.1): what the
    /// bot said, wrote, promised and what changed for it, then what it heard,
    /// then its own acts. Lower is kept first.
    pub fn rank(self) -> u8 {
        match self {
            Self::Said | Self::Wrote | Self::Promised | Self::Mission | Self::Changed => 0,
            Self::Heard => 1,
            Self::Did => 2,
        }
    }
}

impl fmt::Display for ExperienceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An episode's experience: what it was, and whose journal it was with.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Experience {
    pub kind: ExperienceKind,
    /// The party's journal — `people/<p>` or `bots/<other>` — when there was
    /// a party.
    pub about: Option<JournalPath>,
}

impl Experience {
    pub fn of(kind: ExperienceKind) -> Experience {
        Experience { kind, about: None }
    }

    pub fn about(kind: ExperienceKind, about: JournalPath) -> Experience {
        Experience {
            kind,
            about: Some(about),
        }
    }
}

/// `about` as a line's file writes it: `@group/…`, which names the journal
/// whoever reads the file.
pub(crate) fn about_to_text(about: &JournalPath) -> String {
    about.show()
}

/// `about` read back from a line's file.
pub(crate) fn about_from_text(s: &str) -> Result<JournalPath, String> {
    if !s.starts_with('@') {
        return Err(format!("`{s}` names no group: an experience's journal is `@group/…`"));
    }
    Viewer {
        group: None,
        here: None,
    }
    .parse(s)
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::p;

    #[test]
    fn every_kind_reads_back_as_itself() {
        for k in ExperienceKind::ALL {
            assert_eq!(ExperienceKind::parse(k.name()), Ok(k));
        }
        assert!(ExperienceKind::parse("saw")
            .unwrap_err()
            .contains("not an experience"));
    }

    #[test]
    fn what_matters_to_a_conversation_ranks_ahead_of_what_was_heard_and_done() {
        let ranked: Vec<(&str, u8)> = ExperienceKind::ALL
            .iter()
            .map(|k| (k.name(), k.rank()))
            .collect();
        assert_eq!(
            ranked,
            [
                ("heard", 1),
                ("said", 0),
                ("wrote", 0),
                ("promised", 0),
                ("mission", 0),
                ("did", 2),
                ("changed", 0),
            ]
        );
    }

    #[test]
    fn about_is_written_with_its_group_and_read_back() {
        let path = p("@moon/people/mira");
        assert_eq!(about_to_text(&path), "@moon/people/mira");
        assert_eq!(about_from_text("@moon/people/mira"), Ok(path));
        assert!(about_from_text("people/mira").unwrap_err().contains("names no group"));
    }
}
