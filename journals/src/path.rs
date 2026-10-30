//! Where a journal is (§6.5): its owner — a person or a family — and the names
//! from that owner's root down, and how a caller writes it: `~/…` and
//! `@family/…`, or a section alone when that is unambiguous.

use std::fmt;

use crate::error::{Error, Result};
use crate::line::LineId;
use crate::scope::{Client, Scope, Section};
use crate::slug::Slug;

/// The most levels below a section (§2.2).
pub const MAX_DEPTH: usize = 6;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Owner {
    Person(Slug),
    Family(Slug),
}

impl Owner {
    pub fn scope(&self) -> Scope {
        match self {
            Owner::Person(_) => Scope::Personal,
            Owner::Family(_) => Scope::Family,
        }
    }

    pub fn slug(&self) -> &Slug {
        match self {
            Owner::Person(s) | Owner::Family(s) => s,
        }
    }

    /// Its folder in the repository: `persons/<p>` or `families/<f>`.
    pub fn folder(&self) -> String {
        match self {
            Owner::Person(p) => format!("persons/{p}"),
            Owner::Family(f) => format!("families/{f}"),
        }
    }
}

/// A journal: its owner, then the names below the owner's root — first a
/// section, then for `clients` a client, then named journals.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JournalPath {
    owner: Owner,
    names: Vec<Slug>,
}

/// What a level of a path is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Root,
    Section(Section),
    Client(Client),
    /// A named journal, `depth` levels below its section (1 is a section's
    /// child; under `clients`, a client's child is depth 2).
    Named {
        section: Section,
        depth: usize,
    },
}

impl JournalPath {
    pub fn root(owner: Owner) -> JournalPath {
        JournalPath {
            owner,
            names: Vec::new(),
        }
    }

    /// `owner`'s journal at `names`, which must be a place in the tree.
    pub fn new(owner: Owner, names: Vec<Slug>) -> Result<JournalPath> {
        let p = JournalPath { owner, names };
        p.place()?;
        Ok(p)
    }

    pub fn owner(&self) -> &Owner {
        &self.owner
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
        if section.scope() != self.owner.scope() {
            return Err(Error::invalid(match section.scope() {
                Scope::Personal => format!("{section} is personal: it is under `~`, not a family"),
                Scope::Family => format!("{section} is shared: it is under a family, not `~`"),
            }));
        }
        let rest = &self.names[1..];
        if rest.is_empty() {
            return Ok(Place::Section(section));
        }
        if section == Section::Clients {
            let client = Client::parse(rest[0].as_str())
                .ok_or_else(|| Error::invalid(format!("`{}` is not a client", rest[0])))?;
            if rest.len() == 1 {
                return Ok(Place::Client(client));
            }
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
            owner: self.owner.clone(),
            names: self.names[..self.names.len() - 1].to_vec(),
        })
    }

    pub fn child(&self, name: Slug) -> Result<JournalPath> {
        let mut names = self.names.clone();
        names.push(name);
        JournalPath::new(self.owner.clone(), names)
    }

    pub fn last(&self) -> Option<&Slug> {
        self.names.last()
    }

    /// Its folder in the repository.
    pub fn folder(&self) -> String {
        let mut out = self.owner.folder();
        for n in &self.names {
            out.push('/');
            out.push_str(n.as_str());
        }
        out
    }

    /// `self` is `other` or above it.
    pub fn holds(&self, other: &JournalPath) -> bool {
        self.owner == other.owner && other.names.starts_with(&self.names)
    }

    /// How a reader writes it: `~/…` for their own person, `@family/…`.
    pub fn show(&self) -> String {
        let mut out = match &self.owner {
            Owner::Person(_) => "~".to_string(),
            Owner::Family(f) => format!("@{f}"),
        };
        for n in &self.names {
            out.push('/');
            out.push_str(n.as_str());
        }
        out
    }
}

impl fmt::Display for JournalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.show())
    }
}

/// Whom a path is read for: their person and families, which `~`, a bare
/// personal section and a bare shared section resolve against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Viewer {
    pub person: Slug,
    pub families: Vec<Slug>,
}

impl Viewer {
    /// A path as a caller wrote it.
    pub fn parse(&self, text: &str) -> Result<JournalPath> {
        let text = text.trim().trim_end_matches('/');
        if text.is_empty() {
            return Err(Error::invalid("a journal path is empty"));
        }
        let (owner, rest) = if let Some(rest) = text.strip_prefix('~') {
            if !(rest.is_empty() || rest.starts_with('/')) {
                return Err(Error::invalid(format!(
                    "`{text}`: `~` is your own journal; write `~/clients/…`"
                )));
            }
            (Owner::Person(self.person.clone()), rest)
        } else if let Some(rest) = text.strip_prefix('@') {
            let (fam, rest) = rest.split_once('/').map_or((rest, ""), |(f, r)| (f, r));
            let fam = Slug::parse(fam).map_err(Error::Invalid)?;
            (Owner::Family(fam), rest)
        } else {
            let first = text.split('/').next().unwrap_or_default();
            let owner = match Section::parse(first).map(Section::scope) {
                Some(Scope::Personal) => Owner::Person(self.person.clone()),
                Some(Scope::Family) => Owner::Family(self.only_family(text)?),
                None => {
                    return Err(Error::invalid(format!(
                        "`{text}` starts with neither `~`, `@family` nor a section"
                    )))
                }
            };
            (owner, text)
        };
        let names = rest
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| Slug::parse(s).map_err(Error::Invalid))
            .collect::<Result<Vec<_>>>()?;
        JournalPath::new(owner, names)
    }

    fn only_family(&self, text: &str) -> Result<Slug> {
        match self.families.as_slice() {
            [one] => Ok(one.clone()),
            [] => Err(Error::invalid(format!(
                "`{text}` is shared, and you belong to no family"
            ))),
            many => Err(Error::invalid(format!(
                "`{text}` is shared and you belong to {}: write `@{}/{text}` or another",
                many.iter()
                    .map(|f| format!("@{f}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                many[0]
            ))),
        }
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

    fn john() -> Viewer {
        Viewer {
            person: slug("john"),
            families: vec![slug("sharratt")],
        }
    }

    fn shown(v: &Viewer, s: &str) -> String {
        v.parse(s)
            .map(|p| p.show())
            .unwrap_or_else(|e| format!("error: {e}"))
    }

    #[test]
    fn every_way_of_writing_a_path_resolves() {
        let v = john();
        for (written, is) in [
            ("~", "~"),
            ("~/", "~"),
            (
                "~/clients/grok/2026-10-30-2105",
                "~/clients/grok/2026-10-30-2105",
            ),
            ("clients/grok", "~/clients/grok"),
            ("people/builder", "~/people/builder"),
            ("@sharratt", "@sharratt"),
            ("@sharratt/projects/candle/", "@sharratt/projects/candle"),
            ("projects/candle", "@sharratt/projects/candle"),
            (
                "  topics/house-build/permits ",
                "@sharratt/topics/house-build/permits",
            ),
            (
                "@work/threads/review/verify.2",
                "@work/threads/review/verify.2",
            ),
        ] {
            assert_eq!(shown(&v, written), is, "{written}");
        }
        assert_eq!(
            v.parse("projects/candle").unwrap().folder(),
            "families/sharratt/projects/candle"
        );
        assert_eq!(
            v.parse("~/clients").unwrap().folder(),
            "persons/john/clients"
        );
    }

    #[test]
    fn a_path_in_the_wrong_scope_or_place_is_refused() {
        let v = john();
        for (written, says) in [
            ("", "empty"),
            ("~/projects/candle", "projects is shared"),
            ("@sharratt/clients/grok", "clients is personal"),
            ("~/clients/tesla", "`tesla` is not a client"),
            ("~/notes", "`notes` is not a section"),
            ("candle", "starts with neither"),
            ("~john/clients", "`~` is your own"),
            ("@Sharratt/projects", "not a name"),
            ("projects/House Build", "not a name"),
            ("projects/candle/archive", "reserved"),
            ("projects/a/b/c/d/e/f/g", "at most 6"),
        ] {
            let got = shown(&v, written);
            assert!(
                got.starts_with("error:") && got.contains(says),
                "{written}: {got}"
            );
        }
        assert!(v.parse("projects/a/b/c/d/e/f").is_ok());
        // Under clients the client is a level of its own.
        assert!(v.parse("~/clients/grok/a/b/c/d/e").is_ok());
        assert!(v.parse("~/clients/grok/a/b/c/d/e/f").is_err());
    }

    #[test]
    fn a_bare_shared_section_needs_exactly_one_family() {
        let none = Viewer {
            person: slug("solo"),
            families: vec![],
        };
        assert!(shown(&none, "topics/x").contains("belong to no family"));
        assert_eq!(shown(&none, "clients/grok"), "~/clients/grok");
        let two = Viewer {
            person: slug("john"),
            families: vec![slug("sharratt"), slug("work")],
        };
        assert_eq!(
            shown(&two, "topics/x"),
            "error: `topics/x` is shared and you belong to @sharratt, @work: write `@sharratt/topics/x` or another"
        );
        assert_eq!(shown(&two, "@work/topics/x"), "@work/topics/x");
    }

    #[test]
    fn places_parents_and_children() {
        let v = john();
        let p = v.parse("~/clients/grok/2026-10-30-2105").unwrap();
        assert_eq!(
            p.place().unwrap(),
            Place::Named {
                section: Section::Clients,
                depth: 2
            }
        );
        assert_eq!(
            p.parent().unwrap().place().unwrap(),
            Place::Client(Client::Grok)
        );
        assert_eq!(
            v.parse("~/clients").unwrap().place().unwrap(),
            Place::Section(Section::Clients)
        );
        assert_eq!(v.parse("~").unwrap().place().unwrap(), Place::Root);
        assert_eq!(v.parse("~").unwrap().parent(), None);
        let candle = v.parse("projects/candle").unwrap();
        assert_eq!(
            candle.child(slug("cluster")).unwrap().show(),
            "@sharratt/projects/candle/cluster"
        );
        assert!(candle.child(slug("archive")).is_err());
        assert!(v.parse("@sharratt/projects").unwrap().holds(&candle));
        assert!(!candle.holds(&v.parse("@sharratt/projects").unwrap()));
        assert!(!v.parse("~").unwrap().holds(&candle));
    }

    #[test]
    fn addresses_are_a_path_and_a_line_id() {
        let v = john();
        let a = v
            .parse_address("projects/candle#20261030-0912-7e3a")
            .unwrap();
        assert_eq!(
            a.to_string(),
            "@sharratt/projects/candle#20261030-0912-7e3a"
        );
        assert_eq!(
            v.parse_address("~#20261030-0912-7e3a").unwrap().path.show(),
            "~"
        );
        assert!(v.parse_address("projects/candle").is_err());
        assert!(v.parse_address("projects/candle#nope").is_err());
    }
}
