//! Where a journal is (§6.5): its group and the names from the group's master
//! down, and how a caller writes it: `@group/…`, a bare path in the
//! conversation's group, or `here`, the conversation itself.

use std::fmt;

use crate::error::{Error, Result};
use crate::line::LineId;
use crate::scope::Section;
use crate::slug::Slug;

/// The most levels below a section (§2.2).
pub const MAX_DEPTH: usize = 6;

/// What a caller writes for the conversation's own journal.
pub const HERE: &str = "here";

/// A journal: its group, then the names below the group's master — first a
/// section, then named or dated journals.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JournalPath {
    group: Slug,
    names: Vec<Slug>,
}

/// What a level of a path is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// The group's master.
    Root,
    Section(Section),
    /// A journal `depth` levels below its section (1 is a section's child).
    Named {
        section: Section,
        depth: usize,
    },
}

impl Place {
    /// A conversation's own journal, `conversations/<began>-<channel>`.
    pub fn is_conversation(self) -> bool {
        self == Place::Named {
            section: Section::Conversations,
            depth: 1,
        }
    }
}

impl JournalPath {
    /// `group`'s master.
    pub fn root(group: Slug) -> JournalPath {
        JournalPath {
            group,
            names: Vec::new(),
        }
    }

    /// `group`'s `section`.
    pub fn of_section(group: &Slug, section: Section) -> JournalPath {
        JournalPath {
            group: group.clone(),
            names: vec![Slug::parse(section.name()).expect("a section's name is a name")],
        }
    }

    /// `person`'s journal in `group`, `people/<person>`; an error for a name
    /// no journal may take.
    pub fn person(group: &Slug, person: &Slug) -> Result<JournalPath> {
        JournalPath::of_section(group, Section::People).child(person.clone())
    }

    /// `group`'s journal at `names`, which must be a place in the tree.
    pub fn new(group: Slug, names: Vec<Slug>) -> Result<JournalPath> {
        let p = JournalPath { group, names };
        p.place()?;
        Ok(p)
    }

    pub fn group(&self) -> &Slug {
        &self.group
    }

    pub fn names(&self) -> &[Slug] {
        &self.names
    }

    pub fn section(&self) -> Option<Section> {
        self.names.first().and_then(|s| Section::parse(s.as_str()))
    }

    /// The journal's place, or why these names are no place in the tree.
    pub fn place(&self) -> Result<Place> {
        let Some(first) = self.names.first() else {
            return Ok(Place::Root);
        };
        let section = Section::parse(first.as_str())
            .ok_or_else(|| Error::invalid(format!("`{first}` is not a section")))?;
        let rest = &self.names[1..];
        if rest.is_empty() {
            return Ok(Place::Section(section));
        }
        if let Some(r) = rest.iter().find(|s| s.is_reserved()) {
            return Err(Error::invalid(format!(
                "`{r}` is reserved: no journal takes it"
            )));
        }
        if rest.len() > MAX_DEPTH {
            return Err(Error::invalid(format!(
                "{} levels below {section}: at most {MAX_DEPTH}",
                rest.len()
            )));
        }
        Ok(Place::Named {
            section,
            depth: rest.len(),
        })
    }

    pub fn is_root(&self) -> bool {
        self.names.is_empty()
    }

    pub fn parent(&self) -> Option<JournalPath> {
        if self.names.is_empty() {
            return None;
        }
        Some(JournalPath {
            group: self.group.clone(),
            names: self.names[..self.names.len() - 1].to_vec(),
        })
    }

    pub fn child(&self, name: Slug) -> Result<JournalPath> {
        let mut names = self.names.clone();
        names.push(name);
        JournalPath::new(self.group.clone(), names)
    }

    pub fn last(&self) -> Option<&Slug> {
        self.names.last()
    }

    /// Its folder in the repository.
    pub fn folder(&self) -> String {
        let mut out = format!("groups/{}", self.group);
        for n in &self.names {
            out.push('/');
            out.push_str(n.as_str());
        }
        out
    }

    /// `self` is `other` or above it.
    pub fn holds(&self, other: &JournalPath) -> bool {
        self.group == other.group && other.names.starts_with(&self.names)
    }

    /// How a reader writes it: `@group/…`.
    pub fn show(&self) -> String {
        let mut out = format!("@{}", self.group);
        for n in &self.names {
            out.push('/');
            out.push_str(n.as_str());
        }
        out
    }

    /// The path below the group, as a reference's `to` and a focus's
    /// `about` write it: `topics/candle`. Empty for the master.
    pub fn below(&self) -> String {
        self.names
            .iter()
            .map(Slug::as_str)
            .collect::<Vec<_>>()
            .join("/")
    }
}

impl fmt::Display for JournalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.show())
    }
}

/// Where a caller writes paths from: the conversation's group, which a bare
/// path is in, and the conversation itself, which `here` is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Viewer {
    pub group: Option<Slug>,
    pub here: Option<JournalPath>,
}

impl Viewer {
    /// A path as a caller wrote it, before any reference in it is followed
    /// (`Tree::resolve` does that).
    pub fn parse(&self, text: &str) -> Result<JournalPath> {
        let text = text.trim().trim_end_matches('/');
        if text.is_empty() {
            return Err(Error::invalid("a journal path is empty"));
        }
        if text.starts_with('~') {
            return Err(Error::invalid(format!(
                "`{text}` names no journal: the memory starts at a group's master, such as `@zen`, \
                 and a person's own journal is `people/<name>`"
            )));
        }
        if text == HERE {
            return self.here.clone().ok_or_else(|| {
                Error::invalid(
                    "`here` is a conversation's own journal, and this is no conversation",
                )
            });
        }
        let (group, rest) = if let Some(rest) = text.strip_prefix('@') {
            let (group, rest) = rest.split_once('/').map_or((rest, ""), |(g, r)| (g, r));
            (Slug::parse(group).map_err(Error::Invalid)?, rest)
        } else {
            let group = self.group.clone().ok_or_else(|| {
                Error::invalid(format!(
                    "`{text}` names no group, and you are in none: write `@group/{text}`"
                ))
            })?;
            (group, text)
        };
        let names = rest
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| Slug::parse(s).map_err(Error::Invalid))
            .collect::<Result<Vec<_>>>()?;
        JournalPath::new(group, names)
    }

    /// An address as a caller wrote it: `path#line-id`.
    pub fn parse_address(&self, text: &str) -> Result<Address> {
        let (path, id) = text.rsplit_once('#').ok_or_else(|| {
            Error::invalid(format!("`{text}` is not an address: write `path#line-id`"))
        })?;
        Ok(Address {
            path: self.parse(path)?,
            id: LineId::parse(id.trim()).map_err(Error::Invalid)?,
        })
    }
}

/// A path below a group's master, as a file in that group writes it: a
/// reference's `to`, a focus's `about`, a bot's `reads` — `topics/candle`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Below(Vec<Slug>);

impl Below {
    pub fn parse(text: &str) -> Result<Below, String> {
        let names = text
            .split('/')
            .map(Slug::parse)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|why| format!("`{text}` is not a path below a group: {why}"))?;
        Ok(Below(names))
    }

    /// Where it is in `group`, or why it is no place there.
    pub fn at(&self, group: &Slug) -> Result<JournalPath> {
        JournalPath::new(group.clone(), self.0.clone())
    }

    pub fn names(&self) -> &[Slug] {
        &self.0
    }
}

impl From<&JournalPath> for Below {
    fn from(p: &JournalPath) -> Below {
        Below(p.names().to_vec())
    }
}

impl fmt::Display for Below {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.0.iter().map(Slug::as_str).collect();
        f.write_str(&names.join("/"))
    }
}

/// One line: its journal and id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub path: JournalPath,
    pub id: LineId,
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.path, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    fn in_zen() -> Viewer {
        Viewer {
            group: Some(slug("zen")),
            here: Some(
                JournalPath::new(
                    slug("zen"),
                    vec![slug("conversations"), slug("2026-10-31-0712-grok")],
                )
                .unwrap(),
            ),
        }
    }

    fn shown(v: &Viewer, s: &str) -> String {
        v.parse(s)
            .map(|p| p.show())
            .unwrap_or_else(|e| format!("error: {e}"))
    }

    #[test]
    fn every_way_of_writing_a_path_resolves() {
        let v = in_zen();
        for (written, is) in [
            ("@zen", "@zen"),
            ("@zen/", "@zen"),
            ("@zen/topics/candle/", "@zen/topics/candle"),
            ("topics/candle", "@zen/topics/candle"),
            (
                "  topics/house-extension/permits ",
                "@zen/topics/house-extension/permits",
            ),
            ("people/yen", "@zen/people/yen"),
            ("here", "@zen/conversations/2026-10-31-0712-grok"),
            ("@battle-cities/topics/canon", "@battle-cities/topics/canon"),
            (
                "@zen/threads/review/verify.2",
                "@zen/threads/review/verify.2",
            ),
        ] {
            assert_eq!(shown(&v, written), is, "{written}");
        }
        assert_eq!(
            v.parse("topics/candle").unwrap().folder(),
            "groups/zen/topics/candle"
        );
        assert_eq!(v.parse("topics/candle").unwrap().below(), "topics/candle");
        assert_eq!(v.parse("@zen").unwrap().below(), "");
    }

    #[test]
    fn a_path_in_no_place_is_refused() {
        let v = in_zen();
        for (written, says) in [
            ("", "empty"),
            ("channels/grok", "`channels` is not a section"),
            ("projects/candle", "`projects` is not a section"),
            ("~", "a person's own journal is `people/<name>`"),
            ("~/channels/grok", "names no journal"),
            ("candle", "`candle` is not a section"),
            ("@Zen/topics", "not a name"),
            ("topics/House Build", "not a name"),
            ("topics/candle/archive", "reserved"),
            ("topics/a/b/c/d/e/f/g", "at most 6"),
        ] {
            let got = shown(&v, written);
            assert!(
                got.starts_with("error:") && got.contains(says),
                "{written}: {got}"
            );
        }
        assert!(v.parse("topics/a/b/c/d/e/f").is_ok());
    }

    /// **A bare path is the conversation's group's**; `here` needs a
    /// conversation.
    #[test]
    fn a_bare_path_needs_a_group_and_here_a_conversation() {
        let none = Viewer {
            group: None,
            here: None,
        };
        assert_eq!(
            shown(&none, "topics/x"),
            "error: `topics/x` names no group, and you are in none: write `@group/topics/x`"
        );
        assert_eq!(
            shown(&none, "here"),
            "error: `here` is a conversation's own journal, and this is no conversation"
        );
        assert_eq!(shown(&none, "@zen/topics/x"), "@zen/topics/x");
    }

    #[test]
    fn places_parents_and_children() {
        let v = in_zen();
        let p = v.parse("here").unwrap();
        assert_eq!(
            p.place().unwrap(),
            Place::Named {
                section: Section::Conversations,
                depth: 1
            }
        );
        assert_eq!(
            p.parent().unwrap().place().unwrap(),
            Place::Section(Section::Conversations)
        );
        assert_eq!(v.parse("@zen").unwrap().place().unwrap(), Place::Root);
        assert_eq!(v.parse("@zen").unwrap().parent(), None);
        let candle = v.parse("topics/candle").unwrap();
        assert_eq!(
            candle.child(slug("cluster")).unwrap().show(),
            "@zen/topics/candle/cluster"
        );
        assert!(candle.child(slug("archive")).is_err());
        assert!(v.parse("@zen/topics").unwrap().holds(&candle));
        assert!(!candle.holds(&v.parse("@zen/topics").unwrap()));
        assert!(!v.parse("@battle-cities").unwrap().holds(&candle));
    }

    #[test]
    fn addresses_are_a_path_and_a_line_id() {
        let v = in_zen();
        let a = v
            .parse_address("topics/candle#2026-10-30-0912-7e3a")
            .unwrap();
        assert_eq!(a.to_string(), "@zen/topics/candle#2026-10-30-0912-7e3a");
        assert_eq!(
            v.parse_address("here#2026-10-30-0912-7e3a")
                .unwrap()
                .path
                .show(),
            "@zen/conversations/2026-10-31-0712-grok"
        );
        assert!(v.parse_address("topics/candle").is_err());
        assert!(v.parse_address("topics/candle#nope").is_err());
    }
}
