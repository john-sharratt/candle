//! What is fixed at compile time (§2.1): the sections under a group's master,
//! how each section's journals are named, and the channels a conversation
//! comes through.

use std::fmt;
use std::ops::Range;

/// The sections: the journals directly under a group's master.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Section {
    /// The people the group talks to and about, a journal each.
    People,
    /// Each conversation, from any client, a journal each.
    Conversations,
    /// Everything durable the group talks about: projects, themes, places.
    Topics,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// The group's bots, and what each did.
    Bots,
}

/// How a section's journals are named (§2.2): for what they are, or for when
/// they began.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Naming {
    /// A name for what it is: `candle`, `yen`, `house-extension`.
    Named,
    /// When it began and the channel it began on, `2026-10-31-0712-grok`,
    /// and what it was about if it says: `2026-10-31-0712-grok-permits`. Its
    /// siblings list in the order they began.
    Dated,
}

/// The minute a dated name begins with, `yyyy-mm-dd-hhmm`, is `name`'s first
/// fifteen bytes.
fn minute_prefix(b: &[u8]) -> bool {
    let digits = |r: Range<usize>| b.get(r).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    b.len() >= 15
        && [4, 7, 10].iter().all(|&i| b[i] == b'-')
        && digits(0..4)
        && digits(5..7)
        && digits(8..10)
        && digits(11..15)
}

/// `name` as a conversation's: the minute it began, then `-<channel>`, then
/// nothing, `-<what it was about>`, or the `.<n>` a taken name is numbered
/// with. `None` when it is not one.
pub fn conversation_channel(name: &str) -> Option<Channel> {
    let b = name.as_bytes();
    if !minute_prefix(b) || b.get(15) != Some(&b'-') {
        return None;
    }
    let rest = &name[16..];
    Channel::ALL.into_iter().find(|c| {
        rest.strip_prefix(c.name())
            .is_some_and(|after| matches!(after.as_bytes().first(), None | Some(b'-' | b'.')))
    })
}

/// The name a conversation begun at `minute` (`yyyy-mm-dd-hhmm`) on
/// `channel` is given.
pub fn conversation_name(minute: &str, channel: Channel) -> String {
    format!("{minute}-{}", channel.name())
}

impl Section {
    pub const ALL: [Section; 5] = [
        Section::People,
        Section::Conversations,
        Section::Topics,
        Section::Threads,
        Section::Bots,
    ];

    /// Its folder and the name a path uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::People => "people",
            Self::Conversations => "conversations",
            Self::Topics => "topics",
            Self::Threads => "threads",
            Self::Bots => "bots",
        }
    }

    pub fn parse(s: &str) -> Option<Section> {
        Self::ALL.into_iter().find(|x| x.name() == s)
    }

    /// The cue its index line shows.
    pub fn cue(self) -> &'static str {
        match self {
            Self::People => "open for the people the group talks to and about",
            Self::Conversations => "open for each conversation, from any client",
            Self::Topics => "open for everything durable the group talks about",
            Self::Threads => "open for Zen feature threads, their workers and parts",
            Self::Bots => "open for the group's bots and what each did",
        }
    }

    /// Who writes its journals, as a README says it.
    pub fn kept_by(self) -> &'static str {
        match self {
            Self::People => "a journal each, kept by every member's AIs",
            Self::Conversations => "a conversation each, opened by the connector as it begins",
            Self::Topics => "a journal each, kept by every member's AIs",
            Self::Threads => "kept by zend for Zen's threads and workers",
            Self::Bots => "kept by npcd for the group's bots",
        }
    }

    /// How a journal at `depth` below the section is named: 1 is a child of
    /// the section itself.
    pub fn naming(self, depth: usize) -> Naming {
        match (self, depth) {
            (Self::Conversations, 1) => Naming::Dated,
            _ => Naming::Named,
        }
    }

    /// What an eleventh live child of a journal in this section does
    /// (§2.3). `depth` is the journal's depth below the section: 0 is the
    /// section itself.
    pub fn eleventh(self, depth: usize) -> Eleventh {
        match (self, depth) {
            (Self::Conversations, _) => Eleventh::FoldOldest,
            (Self::Threads, _) => Eleventh::FoldIdlestClosed,
            (Self::Bots, _) => Eleventh::FoldIdlest,
            (Self::Topics | Self::People, _) => Eleventh::Refuse,
        }
    }
}

impl fmt::Display for Section {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What happens when a journal that has ten live children gets an eleventh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eleventh {
    /// The oldest live child folds: conversations, a sliding window.
    FoldOldest,
    /// The least recently active closed child folds; never an open one.
    FoldIdlestClosed,
    /// The least recently active child folds.
    FoldIdlest,
    /// Refused: making room is a restructuring, a coding task (§2.6).
    Refuse,
}

/// A channel: the client app a conversation comes through, named as
/// `mcp_server::Client::name` names it. Not a journal: it is recorded on the
/// lines it wrote and in each conversation's name (§2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Channel {
    ClaudeCode,
    Grok,
    /// claude.ai, Claude Desktop, or any other app.
    Other,
}

impl Channel {
    pub const ALL: [Channel; 3] = [Channel::ClaudeCode, Channel::Grok, Channel::Other];

    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Grok => "grok",
            Self::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Option<Channel> {
        Self::ALL.into_iter().find(|c| c.name() == s)
    }

    /// How a line written through it reads: "via Claude Code".
    pub fn spoken(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Grok => "Grok",
            Self::Other => "another app",
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_read_back_as_themselves() {
        for s in Section::ALL {
            assert_eq!(Section::parse(s.name()), Some(s));
        }
        for c in Channel::ALL {
            assert_eq!(Channel::parse(c.name()), Some(c));
        }
        for gone in ["Topics", "channels", "projects", "clients"] {
            assert_eq!(Section::parse(gone), None, "{gone}");
        }
        assert_eq!(Channel::parse("claude_code"), None);
    }

    #[test]
    fn each_section_has_its_eleventh_rule() {
        assert_eq!(Section::Conversations.eleventh(0), Eleventh::FoldOldest);
        assert_eq!(Section::Threads.eleventh(0), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Threads.eleventh(2), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Bots.eleventh(0), Eleventh::FoldIdlest);
        for s in [Section::Topics, Section::People] {
            assert_eq!(s.eleventh(0), Eleventh::Refuse);
            assert_eq!(s.eleventh(4), Eleventh::Refuse);
        }
    }

    /// **A conversation is named for when it began and the channel it began
    /// on; everything else for what it is.**
    #[test]
    fn only_conversations_are_dated_and_name_their_channel() {
        assert_eq!(Section::Conversations.naming(1), Naming::Dated);
        for (s, d) in [
            (Section::Conversations, 2),
            (Section::People, 1),
            (Section::Topics, 1),
            (Section::Topics, 3),
            (Section::Threads, 1),
            (Section::Bots, 1),
        ] {
            assert_eq!(s.naming(d), Naming::Named, "{s} {d}");
        }
        for (name, channel) in [
            ("2026-10-31-0712-grok", Channel::Grok),
            ("2026-10-31-0930-claude-code-permits", Channel::ClaudeCode),
            ("2026-10-31-0930-other.2", Channel::Other),
        ] {
            assert_eq!(conversation_channel(name), Some(channel), "{name}");
        }
        for not in [
            "permits",
            "2026-10-31-0712",
            "2026-10-31-0712-tesla",
            "2026-10-31-0712-grokking",
            "2026-10-31-0712-zen",
            "2026-10-31-0712grok",
            "20261031-0712-grok",
        ] {
            assert_eq!(conversation_channel(not), None, "{not}");
        }
        assert_eq!(
            conversation_name("2026-10-31-0712", Channel::ClaudeCode),
            "2026-10-31-0712-claude-code"
        );
    }
}
