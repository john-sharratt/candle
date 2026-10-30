//! What is fixed at compile time (§2.1): the scopes, the sections under a
//! root and which scope each is in, and the clients a person speaks through.

use std::fmt;

/// Where a section's journals live, and so who reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// The person's own folder, `persons/<person>/`: read by their AIs alone.
    Personal,
    /// A family's folder, `families/<family>/`: read by every member's AIs.
    Family,
}

/// The sections: the journals directly under a root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Section {
    /// Each client the person speaks through, and its conversations.
    Clients,
    /// The people in the person's life, as the person sees them.
    People,
    /// Repositories and durable projects.
    Projects,
    /// Zen feature threads, their workers and parts.
    Threads,
    /// Themes of life and work that span conversations.
    Topics,
    /// The family's bots, and what each did.
    Bots,
}

impl Section {
    pub const ALL: [Section; 6] = [
        Section::Clients,
        Section::People,
        Section::Projects,
        Section::Threads,
        Section::Topics,
        Section::Bots,
    ];

    pub fn scope(self) -> Scope {
        match self {
            Self::Clients | Self::People => Scope::Personal,
            Self::Projects | Self::Threads | Self::Topics | Self::Bots => Scope::Family,
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
            Self::Clients => "clients",
            Self::People => "people",
            Self::Projects => "projects",
            Self::Threads => "threads",
            Self::Topics => "topics",
            Self::Bots => "bots",
        }
    }

    pub fn parse(s: &str) -> Option<Section> {
        Self::ALL.into_iter().find(|x| x.name() == s)
    }

    /// The cue its index line shows.
    pub fn cue(self) -> &'static str {
        match self {
            Self::Clients => "open for each client they speak through and its conversations",
            Self::People => "open for the people in their life",
            Self::Projects => "open for repositories and durable projects",
            Self::Threads => "open for Zen feature threads, their workers and parts",
            Self::Topics => "open for themes that span conversations",
            Self::Bots => "open for the family's bots and what each did",
        }
    }

    /// What an eleventh live child of a journal in this section does
    /// (§2.3). `depth` is the journal's depth below the section: 0 is the
    /// section itself.
    pub fn eleventh(self, depth: usize) -> Eleventh {
        match (self, depth) {
            // The clients' own journals are fixed; below each, conversations.
            (Self::Clients, 0) => Eleventh::Fixed,
            (Self::Clients, _) => Eleventh::FoldOldest,
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

/// A client a person speaks through — one journal each under `clients`, named
/// as `mcp_server::Client::name` names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Client {
    ClaudeCode,
    Grok,
    Other,
}

impl Client {
    pub const ALL: [Client; 3] = [Client::ClaudeCode, Client::Grok, Client::Other];

    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Grok => "grok",
            Self::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Option<Client> {
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

impl fmt::Display for Client {
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
            [Section::Clients, Section::People]
        );
        assert_eq!(
            Section::of(Scope::Family).collect::<Vec<_>>(),
            [
                Section::Projects,
                Section::Threads,
                Section::Topics,
                Section::Bots
            ]
        );
    }

    #[test]
    fn names_read_back_as_themselves() {
        for s in Section::ALL {
            assert_eq!(Section::parse(s.name()), Some(s));
        }
        for c in Client::ALL {
            assert_eq!(Client::parse(c.name()), Some(c));
        }
        assert_eq!(Section::parse("Projects"), None);
        assert_eq!(Client::parse("claude_code"), None);
    }

    #[test]
    fn each_section_has_its_eleventh_rule() {
        assert_eq!(Section::Clients.eleventh(0), Eleventh::Fixed);
        assert_eq!(Section::Clients.eleventh(1), Eleventh::FoldOldest);
        assert_eq!(Section::Clients.eleventh(3), Eleventh::FoldOldest);
        assert_eq!(Section::Threads.eleventh(0), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Threads.eleventh(2), Eleventh::FoldIdlestClosed);
        assert_eq!(Section::Bots.eleventh(0), Eleventh::FoldIdlest);
        for s in [Section::Projects, Section::Topics, Section::People] {
            assert_eq!(s.eleventh(0), Eleventh::Refuse);
            assert_eq!(s.eleventh(4), Eleventh::Refuse);
        }
    }
}
