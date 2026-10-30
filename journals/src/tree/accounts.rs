//! Which person a call is for (§3.4): its account's key, or — the first time
//! an account signs in — the email its provider verified.

use crate::error::{Error, Result};
use crate::files::{AccountKey, Email};
use crate::slug::Slug;
use crate::tree::Tree;

/// Whose an account is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Known {
    /// Bound to a person already.
    Person(Slug),
    /// Not bound, but a person lists its verified email with no key: bind it,
    /// then it is theirs.
    Binds { person: Slug, email: Email },
    /// No person lists it: no memory until an owner adds it.
    Stranger,
}

/// The person `key` is, using `verified` — an email the provider verified,
/// never one a client claims — when the key is not yet bound.
pub fn resolve(tree: &Tree, key: &AccountKey, verified: Option<&Email>) -> Result<Known> {
    let mut by_key = Vec::new();
    let mut by_email = Vec::new();
    for (slug, person) in &tree.persons {
        let Ok(file) = &person.file else { continue };
        for a in &file.accounts {
            if a.key.as_ref() == Some(key) {
                by_key.push(slug.clone());
            } else if a.key.is_none() && Some(&a.email) == verified {
                by_email.push((slug.clone(), a.email.clone()));
            }
        }
    }
    match (by_key.as_slice(), by_email.as_slice()) {
        ([one], _) => Ok(Known::Person(one.clone())),
        ([_, ..], _) => Err(Error::conflict(format!(
            "the account {key} is listed by {}: an account is one person's",
            by_key
                .iter()
                .map(Slug::as_str)
                .collect::<Vec<_>>()
                .join(" and ")
        ))),
        ([], [(person, email)]) => Ok(Known::Binds {
            person: person.clone(),
            email: email.clone(),
        }),
        ([], [_, ..]) => Err(Error::conflict(format!(
            "{} is listed by {}: an account is one person's",
            by_email[0].1,
            by_email
                .iter()
                .map(|(p, _)| p.as_str())
                .collect::<Vec<_>>()
                .join(" and ")
        ))),
        ([], []) => Ok(Known::Stranger),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::testing::slug;
    use crate::tree::load::load;

    fn tree(persons: &[(&str, &str)]) -> Tree {
        let files: BTreeMap<String, Vec<u8>> = persons
            .iter()
            .map(|(p, accounts)| {
                (
                    format!("persons/{p}/person.yaml"),
                    format!("person: {p}\nname: \"{p}\"\naccounts:\n{accounts}families: []\n")
                        .into_bytes(),
                )
            })
            .collect();
        load(&files)
    }

    fn key(s: &str) -> AccountKey {
        AccountKey::parse(s).unwrap()
    }

    fn email(s: &str) -> Email {
        Email::parse(s).unwrap()
    }

    #[test]
    fn one_person_with_two_accounts_is_known_by_either() {
        let t = tree(&[(
            "yen",
            "  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\n  - {email: \"yen.other@example.net\", key: \"1111111111111111\"}\n",
        )]);
        assert_eq!(
            resolve(&t, &key("3b1f09c2d47e8a60"), None),
            Ok(Known::Person(slug("yen")))
        );
        assert_eq!(
            resolve(&t, &key("1111111111111111"), Some(&email("x@y.com"))),
            Ok(Known::Person(slug("yen")))
        );
    }

    #[test]
    fn an_unbound_account_binds_by_its_verified_email_only() {
        let t = tree(&[("yen", "  - {email: \"yen.other@example.net\"}\n")]);
        let k = key("2222222222222222");
        assert_eq!(
            resolve(&t, &k, Some(&email("yen.other@example.net"))),
            Ok(Known::Binds {
                person: slug("yen"),
                email: email("yen.other@example.net")
            })
        );
        assert_eq!(resolve(&t, &k, None), Ok(Known::Stranger));
        assert_eq!(
            resolve(&t, &k, Some(&email("someone@else.com"))),
            Ok(Known::Stranger)
        );
    }

    #[test]
    fn a_bound_email_is_not_bound_again_by_another_account() {
        let t = tree(&[(
            "yen",
            "  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\n",
        )]);
        assert_eq!(
            resolve(
                &t,
                &key("4444444444444444"),
                Some(&email("yen@example.com"))
            ),
            Ok(Known::Stranger)
        );
    }

    #[test]
    fn a_key_of_digits_left_bare_by_hand_is_an_error_not_a_number() {
        let t = tree(&[(
            "yen",
            "  - {email: \"yen@example.com\", key: 1111111111111111}\n",
        )]);
        let err = t.persons[&slug("yen")].file.as_ref().unwrap_err();
        assert!(err.what.contains("quote it"), "{err}");
    }

    #[test]
    fn an_account_two_people_list_is_refused() {
        let t = tree(&[
            (
                "a",
                "  - {email: \"a@example.com\", key: \"5555555555555555\"}\n",
            ),
            (
                "b",
                "  - {email: \"b@example.com\", key: \"5555555555555555\"}\n",
            ),
        ]);
        assert!(resolve(&t, &key("5555555555555555"), None)
            .unwrap_err()
            .to_string()
            .contains("listed by a and b"));
        let t = tree(&[
            ("a", "  - {email: \"same@example.com\"}\n"),
            ("b", "  - {email: \"same@example.com\"}\n"),
        ]);
        assert!(resolve(
            &t,
            &key("6666666666666666"),
            Some(&email("same@example.com"))
        )
        .is_err());
    }
}
