//! Which person a caller is (§3.4): their account's key, or — at an account's
//! first call — the email the gateway names, which binds the account to the
//! one person who lists it.

use journals::actor::Actor;
use journals::files::{AccountKey, Email};
use journals::ops::Op;
use journals::slug::Slug;
use journals::tree::accounts::{resolve, Known};
use journals::tree::Tree;

use super::Memory;
use crate::mcp::caller::Caller;

/// What a caller no person lists is told.
pub const STRANGER: &str = "this account is in no family yet, so nothing is remembered for it: \
     an owner of a family adds it with family_add";

/// The person `key` and `email` are, from `tree` alone: who, and the bind to
/// make first when the account is new.
pub fn known(tree: &Tree, key: &str, email: &str) -> Result<(Slug, Option<Op>), String> {
    let key = AccountKey::parse(key)?;
    let email = Email::normalise(email).ok();
    match resolve(tree, &key, email.as_ref()).map_err(|e| e.to_string())? {
        Known::Person(p) => Ok((p, None)),
        Known::Binds { person, email } => {
            Ok((person.clone(), Some(Op::Bind { person, email, key })))
        }
        Known::Stranger => Err(STRANGER.into()),
    }
}

/// The person `caller` is, binding their account first when it is new.
///
/// Two calls of a new account arrive together — the aggregator reads the
/// context beside a conversation's first call — and both find it unbound; the
/// second bind is refused as already made. That call reads the tree again and
/// finds the account bound.
pub fn person(memory: &Memory, caller: &Caller) -> Result<Slug, String> {
    let tree = memory.tree()?;
    let (person, bind) = known(&tree, &caller.key(), &caller.person.email)?;
    let Some(op) = bind else { return Ok(person) };
    match memory.write(&Actor::System, &op) {
        Ok(_) => {
            tracing::info!(person = %person, "journals: bound an account at its first call");
            Ok(person)
        }
        Err(why) => match known(memory.tree()?.as_ref(), &caller.key(), &caller.person.email)? {
            (p, None) => Ok(p),
            (_, Some(_)) => Err(why),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;

    use super::*;

    fn tree() -> Tree {
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([(
            "persons/yen/person.yaml".to_string(),
            b"person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\n  - {email: \"yensharratt@hotmail.com\"}\nfamilies: []\n".to_vec(),
        )]);
        load(&files)
    }

    #[test]
    fn a_bound_account_is_its_person_and_a_new_one_binds_by_email() {
        let t = tree();
        assert_eq!(
            known(&t, "3b1f09c2d47e8a60", "").unwrap(),
            (Slug::parse("yen").unwrap(), None)
        );
        let (p, bind) = known(&t, "1111111111111111", " YenSharratt@Hotmail.com ").unwrap();
        assert_eq!(p.as_str(), "yen");
        assert!(matches!(bind, Some(Op::Bind { .. })));
        assert_eq!(
            known(&t, "2222222222222222", "who@else.com").unwrap_err(),
            STRANGER
        );
        assert!(known(&t, "not-a-key", "").is_err());
    }
}
