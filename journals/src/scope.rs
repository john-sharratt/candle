//! What is fixed at compile time (§2.1): the scopes, the sections under a
//! root and which scope each is in, how each section's journals are named,
//! and the channels a person speaks through.

use std::fmt;
use std::ops::Range;

/// Where a section's journals live, and so who reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// The person's own folder, `persons/<person>/`: read by their AIs alone.
    Personal,
    /// A group's folder, `groups/<group>/`: read by every member's AIs, and
    /// its bots.
    Group,
}

/// The sections: the journals directly under a root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Section {
    /// Each channel the person speaks through, and its conversations.
    Channels,
    /// The people in the person's life, as the person sees them.
    People,
    /// Repositories and durable projects.
    Projects,
    /// Themes of life and work that span conversations.
    Topics,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// The group's bots, and what each did.
    Bots,
}

/// How a section's journals are named (§2.3): for what they are, or for when
/// they began.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Naming {
    /// A name for what it is: `candle`, `yen`, `house-build`.
    Named,
    /// When it began, `2026-10-30-2105`, and what it was about if it says:
    /// `2026-10-30-2105-permits`. Its siblings list in the order they began.
    Dated,
}

/// `name` is a dated journal's: `yyyy-mm-dd-hhmm`, then nothing, `-<what it
/// was about>`, or the `.<n>` a taken name is numbered with.
pub fn is_dated(name: &str) -> bool {
    let b = name.as_bytes();
    let digits = |r: Range<usize>| b.get(r).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    b.len() >= 15
        && [4, 7, 10].iter().all(|&i| b[i] == b'-')
        && digits(0..4)
        && digits(5..7)
        && digits(8..10)
        && digits(11..15)
        && matches!(b.get(15), None | Some(b'-' | b'.'))
}

impl Section {
    pub const ALL: [Section; 6] = [
        Section::Channels,
        Section::People,
        Section::Projects,
        Section::Topics,
        Section::Threads,
        Section::Bots,
    ];

    pub fn scope(self) -> Scope {
        match self {
            Self::Channels | Self::People => Scope::Personal,
            Self::Projects | Self::Topics | Self::Threads | Self::Bots => Scope::Group,
        }
    }

    /// The sections under a root of `scope`, in the order an index lists
    /// them.
    pub fn of(scope: Scope) -> impl Iterator<Item = Section> {
        Self::ALL.into_iter().filter(move |s| s.scope() == scope)
    }

    /// Its folder and the name a path uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Channels => "channels",
            Self::People => "people",
            Self::Projects => "projects",
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
            Self::Channels => "open for each channel they speak through and its conversations",
            Self::People => "open for the people in their life",
            Self::Projects => "open for repositories and durable projects",
            Self::Topics => "open for themes that span conversations",
            Self::Threads => "open for Zen feature threads, their workers and parts",
            Self::Bots => "open for the group's bots and what each did",
        }
    }

    /// Who writes its journals, as a README says it: any member's AI, or a
    /// service on their behalf.
    pub fn kept_by(self) -> &'static str {
        match self {
            Self::Channels => "a conversation each, opened by the connector as it begins",
            Self::People => "a journal each, kept by your AIs",
            Self::Projects | Self::Topics => "a journal each, kept by every member's AIs",
            Self::Threads => "kept by zend for Zen's threads and workers",
            Self::Bots => "kept by npcd for the group's bots",
        }
    }

    /// How a journal at `depth` below the section is named: 1 is a child of
    /// the section itself.
    pub fn naming(self, depth: usize) -> Naming {
        match (self, depth) {
            // `channels/<channel>/<conversation>`: a conversation is when it
            // began.
            (Self::Channels, 2) => Naming::Dated,
            _ => Naming::Named,
        }
    }

    /// What an eleventh live child of a journal in this section does
    /// (§2.3). `depth` is the journal's depth below the section: 0 is the
    /// section itself.
    pub fn eleventh(self, depth: usize) -> Eleventh {
        match (self, depth) {
            // The channels' own journals are fixed; below each, conversations.
            (Self::Channels, 0) => Eleventh::Fixed,
            (Self::Channels, _) => Eleventh::FoldOldest,
            (Self::Threads, _) => Eleventh::FoldIdlestClosed,
            (Self::Bots, _) => Eleventh::FoldIdlest,
            (Self::Projects | Self::Topics | Self::People, _) => Eleventh::Refuse,
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
    /// Its children are fixed in code; none is created at runtime.
    Fixed,
    /// The oldest live child folds: conversations, a sliding window.
    FoldOldest,
    /// The least recently active closed child folds; never an open one.
    FoldIdlestClosed,
    /// The least recently active child folds.
    FoldIdlest,
    /// Refused until one is folded or merged.
    Refuse,
}

/// A channel a person speaks through — one journal each under `channels`,
/// named as `mcp_server::Client::name` names the client app.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Channel {
    ClaudeCode,
    Grok,
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

    pub fn cue(self) -> &'static str {
        match self {
            Self::ClaudeCode => "open for their Claude Code conversations",
            Self::Grok => "open for their conversations with Grok, in its apps and the car",
            Self::Other => "open for their claude.ai and other conversations",
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
    fn everything_is_shared_but_the_personal_sections() {
        assert_eq!(
            Section::of(Scope::Personal).collect::<Vec<_>>(),
            [Section::Channels, Section::People]
        );
        assert_eq!(
            Section::of(Scope::Group).collect::<Vec<_>>(),
            [
                Section::Projects,
                Section::Topics,
                Section::Threads,
                Section::Bots
            ]
        );
    }

    #[test]
    fn names_read_back_as_themselves() {
        for s in Section::ALL {
            assert_eq!(Section::parse(s.name()), Some(s));
        }
        for c in Channel::ALL {
            assert_eq!(Channel::parse(c.name()), Some(c));
        }
        assert_eq!(Section::parse("Projects"), None);
        assert_eq!(Section::parse("clients"), None);
        assert_eq!(Channel::parse("claude_code"), None);
    }

    #[test]
    fn each_section_has_its_eleventh_rule() {
        assert_eq!(Section::Channels.eleventh(0), Eleventh::Fixed);
        assert_eq!(Section::Channels.eleventh(1), Eleventh::FoldOldest);
        assert_eq!(Section::Channels.eleventh(3), Eleventh::FoldOldest);
        assert_eq!(Section::Threads.eleventh(0), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Threads.eleventh(2), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Bots.eleventh(0), Eleventh::FoldIdlest);
        for s in [Section::Projects, Section::Topics, Section::People] {
            assert_eq!(s.eleventh(0), Eleventh::Refuse);
            assert_eq!(s.eleventh(4), Eleventh::Refuse);
        }
    }

    /// **Conversations are named for when they began; everything else for
    /// what it is.**
    #[test]
    fn only_conversations_are_dated() {
        assert_eq!(Section::Channels.naming(2), Naming::Dated);
        for (s, d) in [
            (Section::Channels, 3),
            (Section::People, 1),
            (Section::Projects, 1),
            (Section::Projects, 3),
            (Section::Topics, 2),
            (Section::Threads, 1),
            (Section::Bots, 1),
        ] {
            assert_eq!(s.naming(d), Naming::Named, "{s} {d}");
        }
        for dated in [
            "2026-10-30-2105",
            "2026-10-30-2105-permits",
            "2026-10-30-2105.2",
        ] {
            assert!(is_dated(dated), "{dated}");
        }
        for not in [
            "permits",
            "2026-10-30",
            "2026-10-30-21",
            "20261030-2105",
            "2026-10-30-2105x",
        ] {
            assert!(!is_dated(not), "{not}");
        }
    }
}
