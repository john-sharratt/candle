//! Whom each conversation reads the memory as, for its retrieval scope
//! (`docs/zen_memory_and_guides.md` D2): a task's or a chat's person, as its
//! worker's record keeps them; a dialogue's, the signed-in person whose
//! message began its turn, as its record keeps them. Both are records, so a
//! turn resumed after a restart reads as the same person. Their account is
//! looked up in the tree as the tools look it up — its key, or the email the
//! gateway verified when the key is not bound yet — and they read the groups
//! the tree says they are in. An account the tree lists nowhere reads
//! `@guest`; a conversation nobody signed in to, nothing.

use std::sync::Arc;

use journals::actor::{groups_of, GUEST};
use journals::tree::Tree;

use super::who::{known, Who};
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
        let (key, email) = match self.ops.reader_person(conv_id) {
            Some(person) => (person.key(), person.email),
            // A task whose record is not read here yet is its owner's, by
            // key alone.
            None => match owner_of(conv_id) {
                Some(owner) => (owner.to_string(), String::new()),
                None => return Reader::nobody(),
            },
        };
        match self.memory.tree() {
            Ok(tree) => reader_for(&tree, &key, &email),
            Err(_) => Reader::nobody(),
        }
    }
}

/// Whom the account `key`, whose provider verified `email`, reads the memory
/// as in `tree`: the person it is bound to, or the one listing `email` it
/// will bind to at its first write.
pub fn reader_for(tree: &Tree, key: &str, email: &str) -> Reader {
    match known(tree, key, email) {
        Ok((Who::Person(person), _)) => Reader::person(
            person.as_str(),
            groups_of(tree, &person)
                .iter()
                .map(|g| g.as_str().to_string()),
        ),
        Ok((Who::Guest, _)) => Reader::guest([GUEST.to_string()]),
        Err(_) => Reader::nobody(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;

    use super::*;

    /// Yen, bound by her key and listing a second email not yet bound, a
    /// member of `zen`.
    fn tree() -> Tree {
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            (
                "persons/yen/person.yaml".to_string(),
                b"person: yen\nname: \"Yen\"\naccounts:\n  - {email: \"yen@example.com\", key: 3b1f09c2d47e8a60}\n  - {email: \"yensharratt@hotmail.com\"}\ngroups: [zen]\n".to_vec(),
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
        let yen = reader_for(&t, "3b1f09c2d47e8a60", "");
        assert_eq!(yen, Reader::person("yen", ["zen".to_string()]));
        assert!(yen.reads("journals/persons/yen/person.yaml"));
        assert!(!yen.reads("journals/groups/guest/topics/a.yaml"));
        assert_eq!(
            reader_for(&t, "1111111111111111", "who@else.com"),
            Reader::guest(["guest".to_string()])
        );
        assert_eq!(reader_for(&t, "not-a-key", ""), Reader::nobody());
    }

    /// **An account listed by its verified email, not yet bound, reads as its
    /// person**, as the tools already see it.
    #[test]
    fn an_unbound_account_reads_as_the_person_listing_its_email() {
        let t = tree();
        assert_eq!(
            reader_for(&t, "1111111111111111", "YenSharratt@Hotmail.com"),
            Reader::person("yen", ["zen".to_string()])
        );
    }
}
