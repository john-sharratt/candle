//! `families/<family>/family.yaml` (§3.5): who is in a family, and as what.

use std::collections::BTreeSet;

use crate::error::Problem;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{item_line, parse_doc, quote, scalar, Fields};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Adds and removes members; edits and forgets any line of the family's.
    Owner,
    /// Reads and writes the family's journals; edits and forgets their own lines.
    Member,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::Owner => "owner",
            Self::Member => "member",
        }
    }

    pub fn parse(s: &str) -> Result<Role, String> {
        match s {
            "owner" => Ok(Self::Owner),
            "member" => Ok(Self::Member),
            _ => Err(format!("`{s}` is not a role: owner or member")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub person: Slug,
    pub role: Role,
    pub joined: Stamp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyFile {
    pub family: Slug,
    pub name: String,
    pub members: Vec<Member>,
}

impl FamilyFile {
    pub fn member(&self, person: &Slug) -> Option<&Member> {
        self.members.iter().find(|m| &m.person == person)
    }

    pub fn owners(&self) -> impl Iterator<Item = &Member> {
        self.members.iter().filter(|m| m.role == Role::Owner)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("`name` is empty".into());
        }
        let mut seen = BTreeSet::new();
        for m in &self.members {
            if !seen.insert(&m.person) {
                return Err(format!("{} is a member twice", m.person));
            }
        }
        if self.owners().next().is_none() {
            return Err("no member is an owner".into());
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "family: {}\nname: {}\nmembers:\n",
            scalar(self.family.as_str()),
            quote(&self.name)
        );
        for m in &self.members {
            out.push_str(&format!(
                "  - {{person: {}, role: {}, joined: {}}}\n",
                scalar(m.person.as_str()),
                m.role.name(),
                m.joined
            ));
        }
        out
    }

    pub fn read(file: &str, text: &str) -> Result<FamilyFile, Problem> {
        let v = parse_doc(file, text)?;
        let mut f = Fields::of(&v, file, None, Some(text))?;
        let family = f.parsed("family", Slug::parse)?;
        let name = f.str("name")?;
        let mut members = Vec::new();
        for (i, m) in f.opt_seq("members")?.into_iter().flatten().enumerate() {
            let mut mf = Fields::of(m, file, item_line(text, "members", i), None)?;
            let person = mf.parsed("person", Slug::parse)?;
            let role = mf.parsed("role", Role::parse)?;
            let joined = mf.parsed("joined", Stamp::parse)?;
            mf.done()?;
            members.push(Member {
                person,
                role,
                joined,
            });
        }
        let out = FamilyFile {
            family,
            name,
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

    fn sharratt() -> FamilyFile {
        let joined = Stamp::parse("2026-10-30T00:00:00Z").unwrap();
        let m = |p: &str, role| Member {
            person: slug(p),
            role,
            joined,
        };
        FamilyFile {
            family: slug("sharratt"),
            name: "Sharratt".into(),
            members: vec![
                m("john", Role::Owner),
                m("yen", Role::Owner),
                m("thanhlong", Role::Member),
                m("yen-nhi", Role::Member),
            ],
        }
    }

    #[test]
    fn family_yaml_is_written_exactly_and_read_back() {
        let fam = sharratt();
        assert_eq!(
            fam.render(),
            "family: sharratt\nname: \"Sharratt\"\nmembers:\n\
             \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n\
             \x20 - {person: yen-nhi, role: member, joined: 2026-10-30T00:00:00Z}\n"
        );
        assert_eq!(
            FamilyFile::read("families/sharratt/family.yaml", &fam.render()).unwrap(),
            fam
        );
        assert_eq!(fam.owners().count(), 2);
        assert_eq!(fam.member(&slug("yen-nhi")).unwrap().role, Role::Member);
        assert!(fam.member(&slug("stranger")).is_none());
    }

    #[test]
    fn names_yaml_would_misread_round_trip() {
        let mut fam = sharratt();
        fam.family = slug("null");
        fam.members[2].person = slug("true");
        fam.members[3].person = slug("1e3");
        assert_eq!(FamilyFile::read("f", &fam.render()).unwrap(), fam);
    }

    #[test]
    fn the_rules_of_a_family_are_errors_naming_the_line() {
        let good = sharratt().render();
        for (from, to, line, says) in [
            (
                "person: yen, role: owner",
                "person: john, role: owner",
                None,
                "a member twice",
            ),
            (
                "role: member, joined: 2026-10-30T00:00:00Z}\n  - {person: yen-nhi",
                "role: chief, joined: 2026-10-30T00:00:00Z}\n  - {person: yen-nhi",
                Some(6),
                "not a role",
            ),
            (
                "joined: 2026-10-30T00:00:00Z}\n",
                "joined: today}\n",
                Some(4),
                "not a time",
            ),
            (
                "{person: john,",
                "{who: x, person: john,",
                Some(4),
                "`who` is not a field",
            ),
            ("name: \"Sharratt\"", "name: \"\"", None, "`name` is empty"),
        ] {
            assert!(good.contains(from), "{from}");
            let err = FamilyFile::read("family.yaml", &good.replacen(from, to, 1)).unwrap_err();
            assert!(err.what.contains(says), "{from}: {err}");
            if line.is_some() {
                assert_eq!(err.line, line, "{from}: {err}");
            }
        }
        let no_owner = good.replace("role: owner", "role: member");
        assert!(FamilyFile::read("f", &no_owner)
            .unwrap_err()
            .what
            .contains("no member is an owner"));
    }
}
