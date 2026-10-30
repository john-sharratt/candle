//! `persons/<person>/person.yaml` (§3.4): identity alone — the accounts that
//! are one person, the groups they belong to, and the one their
//! conversations open in. A person's journals are their groups'.

use std::collections::BTreeSet;
use std::fmt;

use crate::error::Problem;
use crate::slug::Slug;
use crate::yaml::{item_line, parse_doc, scalar, scalar_list, text_field, Fields};

/// The most groups a person may belong to (§3.5).
pub const MAX_GROUPS: usize = 10;

/// One sign-in at the gateway: zend's owner key, `sha256(provider \0 sub)` as
/// 16 lowercase hex digits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AccountKey(String);

impl AccountKey {
    pub fn parse(s: &str) -> Result<AccountKey, String> {
        if s.len() == 16
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            Ok(AccountKey(s.into()))
        } else {
            Err(format!(
                "`{s}` is not an account key: 16 lowercase hex digits"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AccountKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An email address an account is known by before it binds: lowercase, one
/// `@`, something either side, no spaces.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Email(String);

impl Email {
    pub fn parse(s: &str) -> Result<Email, String> {
        let bad = |why: &str| format!("`{s}` is not an email address: {why}");
        if s.len() > 254 {
            return Err(bad("it is too long"));
        }
        if s.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(bad("it has spaces in it"));
        }
        if s.chars().any(char::is_uppercase) {
            return Err(bad("write it in lowercase"));
        }
        match s.split_once('@') {
            Some((local, domain))
                if !local.is_empty()
                    && !domain.is_empty()
                    && !domain.contains('@')
                    && domain.contains('.')
                    && !domain.starts_with('.')
                    && !domain.ends_with('.') =>
            {
                Ok(Email(s.into()))
            }
            _ => Err(bad("it needs a name, `@` and a domain")),
        }
    }

    /// An address as a person gave it: trimmed and lowercased.
    pub fn normalise(s: &str) -> Result<Email, String> {
        Self::parse(&s.trim().to_lowercase())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Email {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Account {
    pub email: Email,
    /// Its key, once it has signed in with that verified email.
    pub key: Option<AccountKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonFile {
    pub person: Slug,
    pub name: String,
    pub accounts: Vec<Account>,
    pub groups: Vec<Slug>,
    /// The group their conversations open in (§3.4).
    pub default_group: Option<Slug>,
}

impl PersonFile {
    /// The file's own rules, apart from what other files say.
    pub fn check(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("`name` is empty".into());
        }
        let mut emails = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for a in &self.accounts {
            if !emails.insert(&a.email) {
                return Err(format!("{} is listed twice", a.email));
            }
            if let Some(k) = &a.key {
                if !keys.insert(k) {
                    return Err(format!("the account {k} is listed twice"));
                }
            }
        }
        if self.groups.len() > MAX_GROUPS {
            return Err(format!(
                "{} groups: a person belongs to at most {MAX_GROUPS}",
                self.groups.len()
            ));
        }
        let unique: BTreeSet<&Slug> = self.groups.iter().collect();
        if unique.len() != self.groups.len() {
            return Err("a group is listed twice".into());
        }
        // No default is a person's to fix by choosing one: taken out of
        // their default by an owner, they are left without one rather than
        // given whichever group that owner left them in.
        if let Some(d) = &self.default_group {
            if !self.groups.contains(d) {
                return Err(format!(
                    "`default_group` is {d}, which is not one of their groups"
                ));
            }
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let mut out = format!("person: {}\n", scalar(self.person.as_str()));
        out.push_str(&text_field("", "name", &self.name));
        if self.accounts.is_empty() {
            out.push_str("accounts: []\n");
        } else {
            out.push_str("accounts:\n");
            for a in &self.accounts {
                out.push_str(&format!("  - email: {}\n", scalar(a.email.as_str())));
                if let Some(k) = &a.key {
                    out.push_str(&format!("    key: {}\n", scalar(k.as_str())));
                }
            }
        }
        let groups: Vec<&str> = self.groups.iter().map(Slug::as_str).collect();
        out.push_str(&format!("groups: {}\n", scalar_list(&groups)));
        if let Some(d) = &self.default_group {
            out.push_str(&format!("default_group: {}\n", scalar(d.as_str())));
        }
        out
    }

    pub fn read(file: &str, text: &str) -> Result<PersonFile, Problem> {
        let v = parse_doc(file, text)?;
        let mut f = Fields::of(&v, file, None, Some(text))?;
        let person = f.parsed("person", Slug::parse)?;
        let name = f.str("name")?;
        let mut accounts = Vec::new();
        for (i, a) in f.opt_seq("accounts")?.into_iter().flatten().enumerate() {
            let mut af = Fields::of(a, file, item_line(text, "accounts", i), None)?;
            let email = af.parsed("email", Email::parse)?;
            let key = af.opt_parsed("key", AccountKey::parse)?;
            af.done()?;
            accounts.push(Account { email, key });
        }
        let mut slugs = |key: &str| {
            f.opt_strs(key)?
                .unwrap_or_default()
                .iter()
                .map(|s| Slug::parse(s))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|why| f.problem(key, format!("`{key}`: {why}")))
        };
        let groups = slugs("groups")?;
        let default_group = f.opt_parsed("default_group", Slug::parse)?;
        let out = PersonFile {
            person,
            name,
            accounts,
            groups,
            default_group,
        };
        out.check().map_err(|why| f.problem("person", why))?;
        f.done()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    pub(crate) fn yen() -> PersonFile {
        PersonFile {
            person: slug("yen"),
            name: "Yen".into(),
            accounts: vec![
                Account {
                    email: Email::parse("yen@example.com").unwrap(),
                    key: Some(AccountKey::parse("3b1f09c2d47e8a60").unwrap()),
                },
                Account {
                    email: Email::parse("yen.other@example.net").unwrap(),
                    key: None,
                },
            ],
            groups: vec![slug("zen")],
            default_group: Some(slug("zen")),
        }
    }

    #[test]
    fn person_yaml_is_written_exactly_and_read_back() {
        let p = yen();
        assert_eq!(
            p.render(),
            "person: yen\nname: Yen\naccounts:\n\
             \x20 - email: yen@example.com\n\
             \x20   key: 3b1f09c2d47e8a60\n\
             \x20 - email: yen.other@example.net\n\
             groups: [zen]\ndefault_group: zen\n"
        );
        assert_eq!(
            PersonFile::read("persons/yen/person.yaml", &p.render()).unwrap(),
            p
        );
    }

    /// **Zen is a person with no accounts**: nobody signs in as it.
    #[test]
    fn a_person_with_no_accounts_round_trips() {
        let p = PersonFile {
            person: slug("zen"),
            name: "Zen".into(),
            accounts: vec![],
            groups: vec![slug("zen")],
            default_group: Some(slug("zen")),
        };
        assert_eq!(
            p.render(),
            "person: zen\nname: Zen\naccounts: []\ngroups: [zen]\ndefault_group: zen\n"
        );
        assert_eq!(PersonFile::read("f", &p.render()).unwrap(), p);
    }

    /// **Groups without a default are theirs to choose from**: an owner
    /// taking them out of their default leaves them so.
    #[test]
    fn a_person_in_groups_with_no_default_round_trips() {
        let mut p = yen();
        p.groups = vec![slug("work")];
        p.default_group = None;
        assert!(p.render().ends_with("groups: [work]\n"), "{}", p.render());
        assert_eq!(PersonFile::read("f", &p.render()).unwrap(), p);
    }

    #[test]
    fn names_and_keys_yaml_would_misread_round_trip() {
        let mut p = yen();
        p.person = slug("0123");
        p.groups = vec![slug("true"), slug("null")];
        p.default_group = Some(slug("null"));
        p.accounts[0].key = Some(AccountKey::parse("1234567890123e45").unwrap());
        assert_eq!(PersonFile::read("f", &p.render()).unwrap(), p);
    }

    #[test]
    fn emails_are_lowercase_addresses() {
        assert!(Email::parse("yen.nhi.sharratt@gmail.com").is_ok());
        assert!(Email::parse("a+tag@sub.example.co.uk").is_ok());
        for bad in [
            "",
            "yen",
            "@x.com",
            "yen@",
            "yen@x",
            "Yen@x.com",
            "y n@x.com",
            "a@b@c.com",
            "a@.com",
            "a@com.",
        ] {
            assert!(Email::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            Email::normalise("  YenSharratt@Hotmail.com ")
                .unwrap()
                .as_str(),
            "yensharratt@hotmail.com"
        );
    }

    #[test]
    fn the_rules_of_a_person_are_errors() {
        let good = yen().render();
        for (from, to, says) in [
            ("3b1f09c2d47e8a60", "3B1F", "not an account key"),
            ("yen.other@example.net", "yen@example.com", "listed twice"),
            (
                "groups: [zen]",
                "groups: [zen, zen]",
                "a group is listed twice",
            ),
            (
                "groups: [zen]",
                "groups: [zen, a, b, c, d, e, f, g, h, i, j]",
                "11 groups: a person belongs to at most 10",
            ),
            (
                "default_group: zen",
                "default_group: other",
                "`default_group` is other, which is not one of their groups",
            ),
            (
                "default_group: zen\n",
                "default_group: zen\nframed: [work]\n",
                "`framed` is not a field",
            ),
            ("name: Yen", "name: \"  \"", "`name` is empty"),
            ("person: yen", "person: Yen", "not a name"),
            (
                "  - email: yen.other@example.net\n",
                "  - email: yen.other@example.net\n    phone: 1\n",
                "`phone` is not a field",
            ),
            ("groups:", "colour: red\ngroups:", "`colour` is not a field"),
            (
                "groups: [zen]\ndefault_group: zen\n",
                "families: [zen]\ndefault_family: zen\n",
                "`families` is not a field",
            ),
        ] {
            assert!(good.contains(from), "{from}");
            let err = PersonFile::read("persons/yen/person.yaml", &good.replacen(from, to, 1))
                .unwrap_err();
            assert!(err.what.contains(says), "{from} → {to}: {err}");
            assert_eq!(err.file, "persons/yen/person.yaml");
        }
    }

    #[test]
    fn an_account_error_names_the_accounts_line() {
        let bad = yen().render().replace("3b1f09c2d47e8a60", "nope");
        let err = PersonFile::read("f", &bad).unwrap_err();
        assert_eq!(err.line, Some(4));
    }
}
