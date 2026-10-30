//! `groups/<group>/group.yaml` (§3.5): what kind of group it is, who is in
//! it, and as what. A member is a person, or a bot that reads the group's
//! journals and nothing else.

use std::collections::BTreeSet;
use std::fmt;

use crate::error::Problem;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{item_line, parse_doc, quote, scalar, Fields};

/// What a group is, for whoever reads about it. Every kind has the same
/// sections and the same rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// A household.
    Family,
    /// People working together.
    Team,
    /// A game's world: its people, and the bots that live in it.
    World,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Family, Kind::Team, Kind::World];

    pub fn name(self) -> &'static str {
        match self {
            Self::Family => "family",
            Self::Team => "team",
            Self::World => "world",
        }
    }

    pub fn parse(s: &str) -> Result<Kind, String> {
        Self::ALL
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| format!("`{s}` is not a kind of group: family, team or world"))
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Adds and removes members; edits and forgets any line of the group's.
    Owner,
    /// Reads and writes the group's journals; edits and forgets their own lines.
    Member,
    /// Reads the group's journals and writes nothing: every bot's role.
    Reader,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::Owner => "owner",
            Self::Member => "member",
            Self::Reader => "reader",
        }
    }

    pub fn parse(s: &str) -> Result<Role, String> {
        match s {
            "owner" => Ok(Self::Owner),
            "member" => Ok(Self::Member),
            "reader" => Ok(Self::Reader),
            _ => Err(format!("`{s}` is not a role: owner, member or reader")),
        }
    }

    /// Whether a member in this role writes the group's journals.
    pub fn writes(self) -> bool {
        self != Self::Reader
    }
}

/// Who a member is.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Who {
    Person(Slug),
    /// A bot, by the id its journal under `bots` is named with.
    Bot(Slug),
}

impl fmt::Display for Who {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Who::Person(p) => write!(f, "{p}"),
            Who::Bot(b) => write!(f, "bot {b}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub who: Who,
    pub role: Role,
    pub joined: Stamp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupFile {
    pub group: Slug,
    pub name: String,
    pub kind: Kind,
    pub members: Vec<Member>,
}

impl GroupFile {
    /// `person`'s membership.
    pub fn member(&self, person: &Slug) -> Option<&Member> {
        self.members
            .iter()
            .find(|m| matches!(&m.who, Who::Person(p) if p == person))
    }

    /// `bot`'s membership.
    pub fn bot(&self, bot: &Slug) -> Option<&Member> {
        self.members
            .iter()
            .find(|m| matches!(&m.who, Who::Bot(b) if b == bot))
    }

    /// The people in it, in the file's order.
    pub fn persons(&self) -> impl Iterator<Item = (&Slug, Role)> {
        self.members.iter().filter_map(|m| match &m.who {
            Who::Person(p) => Some((p, m.role)),
            Who::Bot(_) => None,
        })
    }

    pub fn owners(&self) -> impl Iterator<Item = &Slug> {
        self.persons()
            .filter(|(_, r)| *r == Role::Owner)
            .map(|(p, _)| p)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("`name` is empty".into());
        }
        let mut seen = BTreeSet::new();
        for m in &self.members {
            if !seen.insert(&m.who) {
                return Err(format!("{} is a member twice", m.who));
            }
            if matches!(m.who, Who::Bot(_)) && m.role != Role::Reader {
                return Err(format!(
                    "{} is a {}: a bot is only ever a reader",
                    m.who,
                    m.role.name()
                ));
            }
        }
        if self.owners().next().is_none() {
            return Err("no member is an owner".into());
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "group: {}\nname: {}\nkind: {}\nmembers:\n",
            scalar(self.group.as_str()),
            quote(&self.name),
            self.kind.name()
        );
        for m in &self.members {
            let who = match &m.who {
                Who::Person(p) => format!("person: {}", scalar(p.as_str())),
                Who::Bot(b) => format!("bot: {}", scalar(b.as_str())),
            };
            out.push_str(&format!(
                "  - {{{who}, role: {}, joined: {}}}\n",
                m.role.name(),
                m.joined
            ));
        }
        out
    }

    pub fn read(file: &str, text: &str) -> Result<GroupFile, Problem> {
        let v = parse_doc(file, text)?;
        let mut f = Fields::of(&v, file, None, Some(text))?;
        let group = f.parsed("group", Slug::parse)?;
        let name = f.str("name")?;
        let kind = f.parsed("kind", Kind::parse)?;
        let mut members = Vec::new();
        for (i, m) in f.opt_seq("members")?.into_iter().flatten().enumerate() {
            let line = item_line(text, "members", i);
            let mut mf = Fields::of(m, file, line, None)?;
            let person = mf.opt_parsed("person", Slug::parse)?;
            let bot = mf.opt_parsed("bot", Slug::parse)?;
            let who = match (person, bot) {
                (Some(p), None) => Who::Person(p),
                (None, Some(b)) => Who::Bot(b),
                _ => {
                    return Err(Problem::new(
                        file,
                        line,
                        "a member is a `person` or a `bot`, one of the two",
                    ))
                }
            };
            let role = mf.parsed("role", Role::parse)?;
            let joined = mf.parsed("joined", Stamp::parse)?;
            mf.done()?;
            members.push(Member { who, role, joined });
        }
        let out = GroupFile {
            group,
            name,
            kind,
            members,
        };
        out.check().map_err(|why| f.problem("members", why))?;
        f.done()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    fn joined() -> Stamp {
        Stamp::parse("2026-10-30T00:00:00Z").unwrap()
    }

    fn person(p: &str, role: Role) -> Member {
        Member {
            who: Who::Person(slug(p)),
            role,
            joined: joined(),
        }
    }

    fn sharratt() -> GroupFile {
        GroupFile {
            group: slug("sharratt"),
            name: "Sharratt".into(),
            kind: Kind::Family,
            members: vec![
                person("john", Role::Owner),
                person("yen", Role::Owner),
                person("thanhlong", Role::Member),
                person("yen-nhi", Role::Member),
            ],
        }
    }

    #[test]
    fn group_yaml_is_written_exactly_and_read_back() {
        let g = sharratt();
        assert_eq!(
            g.render(),
            "group: sharratt\nname: \"Sharratt\"\nkind: family\nmembers:\n\
             \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: yen-nhi, role: member, joined: 2026-10-30T00:00:00Z}\n"
        );
        assert_eq!(
            GroupFile::read("groups/sharratt/group.yaml", &g.render()).unwrap(),
            g
        );
        assert_eq!(g.owners().count(), 2);
        assert_eq!(g.member(&slug("yen-nhi")).unwrap().role, Role::Member);
        assert!(g.member(&slug("stranger")).is_none());
    }

    /// **A world holds its bots as readers**, beside the people who run it.
    #[test]
    fn a_world_holds_bots_as_readers() {
        let g = GroupFile {
            group: slug("battle-cities"),
            name: "Battle Cities".into(),
            kind: Kind::World,
            members: vec![
                person("john", Role::Owner),
                Member {
                    who: Who::Bot(slug("c3f1a07e")),
                    role: Role::Reader,
                    joined: joined(),
                },
            ],
        };
        assert_eq!(
            g.render(),
            "group: battle-cities\nname: \"Battle Cities\"\nkind: world\nmembers:\n\
             \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z}\n"
        );
        assert_eq!(GroupFile::read("f", &g.render()).unwrap(), g);
        assert_eq!(g.bot(&slug("c3f1a07e")).unwrap().role, Role::Reader);
        assert!(g.member(&slug("c3f1a07e")).is_none(), "a bot is no person");
        assert_eq!(g.persons().count(), 1);
        assert!(!Role::Reader.writes());
        assert!(Role::Member.writes() && Role::Owner.writes());
    }

    #[test]
    fn names_yaml_would_misread_round_trip() {
        let mut g = sharratt();
        g.group = slug("null");
        g.members[2].who = Who::Person(slug("true"));
        g.members[3].who = Who::Person(slug("1e3"));
        assert_eq!(GroupFile::read("f", &g.render()).unwrap(), g);
    }

    #[test]
    fn the_rules_of_a_group_are_errors_naming_the_line() {
        let good = sharratt().render();
        for (from, to, line, says) in [
            (
                "person: yen, role: owner",
                "person: john, role: owner",
                None,
                "john is a member twice",
            ),
            (
                "role: member, joined: 2026-10-30T00:00:00Z}\n  - {person: yen-nhi",
                "role: chief, joined: 2026-10-30T00:00:00Z}\n  - {person: yen-nhi",
                Some(7),
                "`chief` is not a role: owner, member or reader",
            ),
            (
                "joined: 2026-10-30T00:00:00Z}\n",
                "joined: today}\n",
                Some(5),
                "not a time",
            ),
            (
                "{person: john,",
                "{who: x, person: john,",
                Some(5),
                "`who` is not a field",
            ),
            (
                "{person: john,",
                "{person: john, bot: b1,",
                Some(5),
                "a member is a `person` or a `bot`, one of the two",
            ),
            (
                "{person: john, role: owner, joined: 2026-10-30T00:00:00Z}",
                "{role: owner, joined: 2026-10-30T00:00:00Z}",
                Some(5),
                "a member is a `person` or a `bot`, one of the two",
            ),
            (
                "{person: yen-nhi, role: member,",
                "{bot: b1, role: member,",
                None,
                "bot b1 is a member: a bot is only ever a reader",
            ),
            (
                "kind: family",
                "kind: clan",
                None,
                "`clan` is not a kind of group: family, team or world",
            ),
            ("name: \"Sharratt\"", "name: \"\"", None, "`name` is empty"),
        ] {
            assert!(good.contains(from), "{from}");
            let err = GroupFile::read("group.yaml", &good.replacen(from, to, 1)).unwrap_err();
            assert!(err.what.contains(says), "{from}: {err}");
            if line.is_some() {
                assert_eq!(err.line, line, "{from}: {err}");
            }
        }
        let no_owner = good.replace("role: owner", "role: member");
        assert!(GroupFile::read("f", &no_owner)
            .unwrap_err()
            .what
            .contains("no member is an owner"));
        let no_kind = good.replace("kind: family\n", "");
        assert!(GroupFile::read("f", &no_kind)
            .unwrap_err()
            .what
            .contains("`kind`"));
    }
}
