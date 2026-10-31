//! Which person a caller is (§3.4): their account's key, or — at an account's
//! first call — the email the gateway names, which binds the account to the
//! one person who lists it. An account no person lists is a guest (§3.6).

use journals::actor::Actor;
use journals::files::{AccountKey, Email};
use journals::ops::Op;
use journals::slug::Slug;
use journals::tree::accounts::{resolve, Known};
use journals::tree::Tree;

use super::Memory;
use crate::mcp::caller::Caller;

/// Who a caller is on the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Who {
    Person(Slug),
    /// An account no person lists: it reads `@guest` and writes nothing.
    Guest,
}

/// Who `key` and `email` are, from `tree` alone, and the bind to make first
/// when the account is new.
pub fn known(tree: &Tree, key: &str, email: &str) -> Result<(Who, Option<Op>), String> {
    let key = AccountKey::parse(key)?;
    let email = Email::normalise(email).ok();
    match resolve(tree, &key, email.as_ref()).map_err(|e| e.to_string())? {
        Known::Person(p) => Ok((Who::Person(p), None)),
        Known::Binds { person, email } => Ok((
            Who::Person(person.clone()),
            Some(Op::Bind { person, email, key }),
        )),
        Known::Stranger => Ok((Who::Guest, None)),
    }
}

/// Who `caller` is, read from the tree alone: a new account is already its
/// person, and its bind is left to the first call that writes (§3.1).
pub fn read(memory: &Memory, caller: &Caller) -> Result<Who, String> {
    let tree = memory.tree()?;
    Ok(known(&tree, &caller.key(), &caller.person.email)?.0)
}

/// Who `caller` is, binding their account first when it is new.
///
/// Two calls of a new account arrive together — the aggregator reads the
/// context beside a conversation's first call — and both find it unbound; the
/// second bind is refused as already made. That call reads the tree again and
/// finds the account bound.
pub fn who(memory: &Memory, caller: &Caller) -> Result<Who, String> {
    let tree = memory.tree()?;
    let (who, bind) = known(&tree, &caller.key(), &caller.person.email)?;
    let Some(op) = bind else { return Ok(who) };
    match memory.write(&Actor::System, &op) {
        Ok(_) => {
            tracing::info!(?who, "journals: bound an account at its first call");
            Ok(who)
        }
        Err(why) => match known(memory.tree()?.as_ref(), &caller.key(), &caller.person.email)? {
            (w, None) => Ok(w),
            (_, Some(_)) => Err(why.into()),
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
            b"person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\n  - {email: \"yensharratt@hotmail.com\"}\ngroups: []\n".to_vec(),
        )]);
        load(&files)
    }

    #[test]
    fn a_bound_account_is_its_person_a_new_one_binds_and_a_stranger_is_a_guest() {
        let t = tree();
        assert_eq!(
            known(&t, "3b1f09c2d47e8a60", "").unwrap(),
            (Who::Person(Slug::parse("yen").unwrap()), None)
        );
        let (w, bind) = known(&t, "1111111111111111", " YenSharratt@Hotmail.com ").unwrap();
        assert_eq!(w, Who::Person(Slug::parse("yen").unwrap()));
        assert!(matches!(bind, Some(Op::Bind { .. })));
        assert_eq!(
            known(&t, "2222222222222222", "who@else.com").unwrap(),
            (Who::Guest, None)
        );
        assert!(known(&t, "not-a-key", "").is_err());
    }
}
