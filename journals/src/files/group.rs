//! `groups/<group>/group.yaml` (§3.5): what the group is for, who is in it,
//! and as what. A member is a person, who reads the whole group, or a bot,
//! which reads only the subtrees its `reads` grants.

use std::collections::BTreeSet;
use std::fmt;

use crate::error::Problem;
use crate::path::Below;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{item_line, parse_doc, scalar, scalar_list, text_field, Fields};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Adds and removes members; edits and forgets any line of the group's.
    Owner,
    /// Reads and writes the group's journals; edits and forgets their own lines.
    Member,
    /// Reads what it is granted and writes nothing: every bot's role.
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
    /// A bot's grants: the subtrees it reads. A person reads the whole group
    /// and has none.
    pub reads: Vec<Below>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupFile {
    pub group: Slug,
    pub name: String,
    /// What the group is for, as its README and every frame open with it.
    pub about: Option<String>,
    pub members: Vec<Member>,
}

/// The longest a group's `about` may be, in characters.
pub const MAX_ABOUT: usize = 400;

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
        if let Some(a) = &self.about {
            let n = a.chars().count();
            if a.trim().is_empty() {
                return Err("`about` is empty".into());
            }
            if n > MAX_ABOUT {
                return Err(format!("`about` is {n} characters, over {MAX_ABOUT}"));
            }
        }
        let mut seen = BTreeSet::new();
        for m in &self.members {
            if !seen.insert(&m.who) {
                return Err(format!("{} is a member twice", m.who));
            }
            match &m.who {
                Who::Bot(_) if m.role != Role::Reader => {
                    return Err(format!(
                        "{} is a {}: a bot is only ever a reader",
                        m.who,
                        m.role.name()
                    ))
                }
                Who::Person(_) if !m.reads.is_empty() => {
                    return Err(format!(
                        "{} has `reads`: a person reads the whole group, and only a bot is granted part of it",
                        m.who
                    ))
                }
                _ => {}
            }
        }
        if self.owners().next().is_none() {
            return Err("no member is an owner".into());
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let mut out = format!("group: {}\n", scalar(self.group.as_str()));
        out.push_str(&text_field("", "name", &self.name));
        if let Some(a) = &self.about {
            out.push_str(&text_field("", "about", a));
        }
        out.push_str("members:\n");
        for m in &self.members {
            let who = match &m.who {
                Who::Person(p) => format!("person: {}", scalar(p.as_str())),
                Who::Bot(b) => format!("bot: {}", scalar(b.as_str())),
            };
            out.push_str(&format!(
                "  - {who}\n    role: {}\n    joined: {}\n",
                m.role.name(),
                m.joined
            ));
            if !m.reads.is_empty() {
                let reads: Vec<String> = m.reads.iter().map(Below::to_string).collect();
                out.push_str(&format!("    reads: {}\n", scalar_list(&reads)));
            }
        }
        out
    }

    pub fn read(file: &str, text: &str) -> Result<GroupFile, Problem> {
        let v = parse_doc(file, text)?;
        let mut f = Fields::of(&v, file, None, Some(text))?;
        let group = f.parsed("group", Slug::parse)?;
        let name = f.str("name")?;
        let about = f.opt_str("about")?;
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
            let reads = mf
                .opt_strs("reads")?
                .unwrap_or_default()
                .iter()
                .map(|s| Below::parse(s))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|why| mf.problem("reads", format!("`reads`: {why}")))?;
            mf.done()?;
            members.push(Member {
                who,
                role,
                joined,
                reads,
            });
        }
        let out = GroupFile {
            group,
            name,
            about,
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
            reads: vec![],
        }
    }

    fn zen() -> GroupFile {
        GroupFile {
            group: slug("zen"),
            name: "Zen".into(),
            about: None,
            members: vec![
                person("john", Role::Owner),
                person("yen", Role::Owner),
                person("thanhlong", Role::Member),
                person("zen", Role::Member),
            ],
        }
    }

    #[test]
    fn group_yaml_is_written_exactly_and_read_back() {
        let g = zen();
        assert_eq!(
            g.render(),
            "group: zen\nname: Zen\nmembers:\n\
             \x20 - person: john\n    role: owner\n    joined: 2026-10-30T00:00:00Z\n\
             \x20 - person: yen\n    role: owner\n    joined: 2026-10-30T00:00:00Z\n\
             \x20 - person: thanhlong\n    role: member\n    joined: 2026-10-30T00:00:00Z\n\
             \x20 - person: zen\n    role: member\n    joined: 2026-10-30T00:00:00Z\n"
        );
        assert_eq!(
            GroupFile::read("groups/zen/group.yaml", &g.render()).unwrap(),
            g
        );
        assert_eq!(g.owners().count(), 2);
        assert_eq!(g.member(&slug("zen")).unwrap().role, Role::Member);
        assert!(g.member(&slug("stranger")).is_none());
    }

    /// **A bot reads only what its group grants it.**
    #[test]
    fn a_bot_is_a_reader_of_its_grants() {
        let g = GroupFile {
            group: slug("battle-cities"),
            name: "Battle Cities".into(),
            about: Some("The game's world: its canon, its economy, its NPCs.".into()),
            members: vec![
                person("john", Role::Owner),
                Member {
                    who: Who::Bot(slug("c3f1a07e")),
                    role: Role::Reader,
                    joined: joined(),
                    reads: vec![Below::parse("topics/canon").unwrap()],
                },
            ],
        };
        assert_eq!(
            g.render(),
            "group: battle-cities\nname: Battle Cities\n\
             about: \"The game's world: its canon, its economy, its NPCs.\"\nmembers:\n\
             \x20 - person: john\n    role: owner\n    joined: 2026-10-30T00:00:00Z\n\
             \x20 - bot: c3f1a07e\n    role: reader\n    joined: 2026-10-30T00:00:00Z\n\
             \x20   reads: [topics/canon]\n"
        );
        assert_eq!(GroupFile::read("f", &g.render()).unwrap(), g);
        assert_eq!(g.bot(&slug("c3f1a07e")).unwrap().reads.len(), 1);
        assert!(g.member(&slug("c3f1a07e")).is_none(), "a bot is no person");
        assert_eq!(g.persons().count(), 1);
        assert!(!Role::Reader.writes());
        assert!(Role::Member.writes() && Role::Owner.writes());
    }

    #[test]
    fn names_yaml_would_misread_round_trip() {
        let mut g = zen();
        g.group = slug("null");
        g.members[2].who = Who::Person(slug("true"));
        g.members[3].who = Who::Person(slug("1e3"));
        assert_eq!(GroupFile::read("f", &g.render()).unwrap(), g);
    }

    #[test]
    fn the_rules_of_a_group_are_errors_naming_the_line() {
        let good = zen().render();
        for (from, to, line, says) in [
            (
                "  - person: yen\n",
                "  - person: john\n",
                None,
                "john is a member twice",
            ),
            (
                "  - person: thanhlong\n    role: member",
                "  - person: thanhlong\n    role: chief",
                Some(10),
                "`chief` is not a role: owner, member or reader",
            ),
            (
                "    joined: 2026-10-30T00:00:00Z\n",
                "    joined: today\n",
                Some(4),
                "not a time",
            ),
            (
                "  - person: john\n",
                "  - who: x\n    person: john\n",
                Some(4),
                "`who` is not a field",
            ),
            (
                "  - person: john\n",
                "  - person: john\n    bot: b1\n",
                Some(4),
                "a member is a `person` or a `bot`, one of the two",
            ),
            (
                "  - person: zen\n",
                "  - bot: b1\n",
                None,
                "bot b1 is a member: a bot is only ever a reader",
            ),
            (
                "  - person: zen\n    role: member\n    joined: 2026-10-30T00:00:00Z\n",
                "  - person: zen\n    role: member\n    joined: 2026-10-30T00:00:00Z\n    reads: [topics]\n",
                None,
                "zen has `reads`: a person reads the whole group",
            ),
            (
                "  - person: zen\n    role: member\n    joined: 2026-10-30T00:00:00Z\n",
                "  - bot: b1\n    role: reader\n    joined: 2026-10-30T00:00:00Z\n    reads: [Topics]\n",
                Some(13),
                "`reads`: `Topics` is not a path below a group",
            ),
            ("name: Zen", "name: \"\"", None, "`name` is empty"),
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
        // A group has no kind: it is what `about` says.
        let kind = good.replacen("name: Zen\n", "name: Zen\nkind: family\n", 1);
        assert!(GroupFile::read("f", &kind)
            .unwrap_err()
            .what
            .contains("`kind` is not a field"));
    }
}
