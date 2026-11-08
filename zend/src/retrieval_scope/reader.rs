//! Whose conversation a scope is worked out for, and so which of the memory's
//! files it may retrieve (`docs/zen_memory_and_guides.md` D2): a journal of a
//! group its person reads, that person's own file, and the repository's own
//! files — on every route a unit reaches a prompt by, files, folders, the
//! layers derived from files and the groups that draw them.
//!
//! The ingest and calibration conversations work no scope out at all, so they
//! retrieve every group's units; only a dialogue — a person's, a task's, a
//! chat's — is scoped, and one nobody signed in to reads no journal.

use std::collections::BTreeSet;

use crate::memory_paths::{self, is_memory_repo, Owner};

/// The person a conversation reads the memory as, and the groups they read.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Reader {
    person: Option<String>,
    groups: BTreeSet<String>,
}

/// How a conversation's reader is found, by its id: the person its turn is
/// for, and the groups the memory says they read.
pub trait Readers: Send + Sync {
    fn of(&self, conv_id: &str) -> Reader;
}

impl Reader {
    /// `person`, reading `groups`.
    pub fn person(person: &str, groups: impl IntoIterator<Item = String>) -> Self {
        Self {
            person: Some(person.to_string()),
            groups: groups.into_iter().collect(),
        }
    }

    /// Someone the memory lists as nobody, reading `groups` — a guest.
    pub fn guest(groups: impl IntoIterator<Item = String>) -> Self {
        Self {
            person: None,
            groups: groups.into_iter().collect(),
        }
    }

    /// A conversation nobody signed in to: it reads no journal.
    pub fn nobody() -> Self {
        Self::default()
    }

    /// Whether this reader retrieves the workspace-relative `at` — a file
    /// (`journals/groups/zen/…/x.yaml`) or a folder (`journals/groups/zen/`).
    /// Anything outside the memory is everyone's.
    pub fn reads(&self, at: &str) -> bool {
        match at.split_once('/') {
            Some((repo, path)) if is_memory_repo(repo) => self.reads_memory(path),
            _ => true,
        }
    }

    /// Whether this reader retrieves the memory's file at repository-relative
    /// `path`.
    pub fn reads_memory(&self, path: &str) -> bool {
        match memory_paths::owner(path) {
            Owner::Group(group) => self.groups.contains(group),
            Owner::Person(person) => self.person.as_deref() == Some(person),
            Owner::Repository => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn john() -> Reader {
        Reader::person("john", ["zen".to_string(), "battle-cities".to_string()])
    }

    /// **A person reads their groups' journals and their own file**, and no
    /// other group's or person's.
    #[test]
    fn a_person_reads_their_groups_and_their_own_file() {
        let r = john();
        assert!(r.reads("journals/groups/zen/topics/candle/2026-10-31-0034-7c25.yaml"));
        assert!(r.reads("journals/groups/battle-cities/README.md"));
        assert!(r.reads("journals/persons/john/person.yaml"));
        assert!(!r.reads("journals/groups/guest/topics/a.yaml"));
        assert!(!r.reads("journals/persons/yen/person.yaml"));
    }

    /// **A folder is read as its files are**: a group's folder by its members,
    /// the listings above the groups by everyone.
    #[test]
    fn a_folder_is_read_as_its_files_are() {
        let r = john();
        assert!(r.reads("journals/groups/zen/"));
        assert!(!r.reads("journals/groups/guest/"));
        assert!(r.reads("journals/groups/"));
        assert!(r.reads("journals/"));
    }

    /// **Outside the memory everything is everyone's**, the repository's own
    /// files included.
    #[test]
    fn outside_the_memory_everything_is_read() {
        let nobody = Reader::nobody();
        assert!(nobody.reads("candle/zend/src/lib.rs"));
        assert!(nobody.reads("candle/mcp/src/guides/memory.md"));
        assert!(nobody.reads("journals/README.md"));
        assert!(nobody.reads("journals"));
    }

    /// **Nobody reads no journal; a guest reads only theirs.**
    #[test]
    fn nobody_reads_no_journal_and_a_guest_reads_only_theirs() {
        let nobody = Reader::nobody();
        assert!(!nobody.reads("journals/groups/zen/topics/a.yaml"));
        assert!(!nobody.reads("journals/persons/john/person.yaml"));
        let guest = Reader::guest(["guest".to_string()]);
        assert!(guest.reads("journals/groups/guest/topics/a.yaml"));
        assert!(!guest.reads("journals/groups/zen/topics/a.yaml"));
        assert!(!guest.reads("journals/persons/john/person.yaml"));
    }
}
