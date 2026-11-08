//! Whom each conversation reads the memory as, for its retrieval scope
//! (`docs/zen_memory_and_guides.md` D2): a task's or a chat's owner, named by
//! its conversation id; a dialogue's, the signed-in person whose message began
//! its turn. Their account is looked up in the tree as the tools look it up,
//! and they read the groups the tree says they are in. An account the tree
//! lists nowhere reads `@guest`; a conversation nobody signed in to, nothing.

use std::sync::Arc;

use journals::actor::{groups_of, GUEST};
use journals::files::AccountKey;
use journals::tree::accounts::{resolve, Known};
use journals::tree::Tree;

use super::Memory;
use crate::mcp::ops_tools::OpsTools;
use crate::mcp::ownership::owner_of;
use crate::retrieval_scope::{Reader, Readers};

/// The readers of one session's conversations.
pub struct MemoryReaders {
    memory: Arc<Memory>,
    ops: Arc<OpsTools>,
}

impl MemoryReaders {
    pub fn new(memory: Arc<Memory>, ops: Arc<OpsTools>) -> Self {
        Self { memory, ops }
    }
}

impl Readers for MemoryReaders {
    fn of(&self, conv_id: &str) -> Reader {
        let key = match owner_of(conv_id) {
            Some(owner) => owner.to_string(),
            None => match self.ops.dialogue_key(conv_id) {
                Some(key) => key,
                None => return Reader::nobody(),
            },
        };
        match self.memory.tree() {
            Ok(tree) => reader_for(&tree, &key),
            Err(_) => Reader::nobody(),
        }
    }
}

/// Whom the account `key` reads the memory as in `tree`.
pub fn reader_for(tree: &Tree, key: &str) -> Reader {
    let Ok(key) = AccountKey::parse(key) else {
        return Reader::nobody();
    };
    match resolve(tree, &key, None) {
        Ok(Known::Person(person)) => Reader::person(
            person.as_str(),
            groups_of(tree, &person)
                .iter()
                .map(|g| g.as_str().to_string()),
        ),
        Ok(Known::Stranger | Known::Binds { .. }) => Reader::guest([GUEST.to_string()]),
        Err(_) => Reader::nobody(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;

    use super::*;

    /// Yen, bound by her key, a member of `zen`.
    fn tree() -> Tree {
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            (
                "persons/yen/person.yaml".to_string(),
                b"person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\ngroups: [zen]\n".to_vec(),
            ),
            (
                "groups/zen/group.yaml".to_string(),
                b"group: zen\nname: \"Zen\"\nmembers:\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n".to_vec(),
            ),
        ]);
        load(&files)
    }

    /// **A bound account reads its person's groups and file; an account the
    /// tree lists nowhere reads `@guest`; a key that is no key reads nothing.**
    #[test]
    fn an_account_reads_as_its_person_a_stranger_as_a_guest() {
        let t = tree();
        let yen = reader_for(&t, "3b1f09c2d47e8a60");
        assert_eq!(yen, Reader::person("yen", ["zen".to_string()]));
        assert!(yen.reads("journals/persons/yen/person.yaml"));
        assert!(!yen.reads("journals/groups/guest/topics/a.yaml"));
        assert_eq!(
            reader_for(&t, "1111111111111111"),
            Reader::guest(["guest".to_string()])
        );
        assert_eq!(reader_for(&t, "not-a-key"), Reader::nobody());
    }
}
