//! Where the memory's files sit in the `journals` repository, as zend's
//! ingest and retrieval read them (`docs/memory_backbone.md` §3.1): a group's
//! journals under `groups/<group>/…`, a person's own file under
//! `persons/<person>/…`, and each conversation's journal under its group's
//! `conversations` section.
//!
//! Paths here are repository-relative: what a branch walk lists for the
//! memory's repository.
//!
//! Which workspace repository is the memory is the session's to say
//! ([`name_repo`]): the one whose folder is the agent's `journal: true`
//! checkout, whatever the workspace calls it. A tree of the same shape under
//! any other name is no one's memory, and nothing here treats it as one.

use std::sync::OnceLock;

/// What the memory's repository is called when the session names none.
pub const REPO: &str = "journals";

static NAMED: OnceLock<String> = OnceLock::new();

/// Name the workspace repository that is the memory. Set once, as the
/// session starts; a second call changes nothing.
pub fn name_repo(name: &str) {
    let _ = NAMED.set(name.to_string());
}

/// Whether the workspace repository `repo` is the memory.
pub fn is_memory_repo(repo: &str) -> bool {
    repo == NAMED.get().map_or(REPO, String::as_str)
}

/// The section every conversation's journal sits under.
const CONVERSATIONS: &str = "conversations";

/// Whose a file of the memory is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner<'a> {
    /// A journal of this group.
    Group(&'a str),
    /// This person's own file.
    Person(&'a str),
    /// The repository's own files — its README, its tooling — nobody's.
    Repository,
}

/// Whose the file at `path` is.
pub fn owner(path: &str) -> Owner<'_> {
    let mut parts = path.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("groups"), Some(group), Some(_)) => Owner::Group(group),
        (Some("persons"), Some(person), Some(_)) => Owner::Person(person),
        _ => Owner::Repository,
    }
}

/// Whether `path` is a file of a conversation's journal: what a session's
/// calls wrote, never read by provenance (`docs/zen_memory_and_guides.md` D3).
pub fn is_conversation(path: &str) -> bool {
    let mut parts = path.split('/');
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("groups"), Some(_), Some(CONVERSATIONS), Some(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A file is its group's, its person's, or the repository's.**
    #[test]
    fn a_file_belongs_to_its_group_its_person_or_the_repository() {
        assert_eq!(
            owner("groups/zen/topics/candle/2026-10-31-0034-7c25.yaml"),
            Owner::Group("zen")
        );
        assert_eq!(owner("groups/zen/README.md"), Owner::Group("zen"));
        assert_eq!(owner("persons/yen/person.yaml"), Owner::Person("yen"));
        assert_eq!(owner("README.md"), Owner::Repository);
        assert_eq!(owner("groups"), Owner::Repository);
        assert_eq!(
            owner("groups/zen"),
            Owner::Repository,
            "a bare folder name holds nothing"
        );
    }

    /// **The memory is one repository**: until the session names it, the one
    /// called `journals`; a tree of the same shape under another name is not.
    #[test]
    fn the_memory_is_the_repository_named() {
        assert!(is_memory_repo(REPO));
        assert!(!is_memory_repo("memory"));
        assert!(!is_memory_repo("candle"));
    }

    /// **Only a file below a group's `conversations` is a conversation's.**
    #[test]
    fn a_conversation_file_is_one_below_a_groups_conversations() {
        assert!(is_conversation(
            "groups/zen/conversations/2026-11-08-0308-claude-code/2026-11-08-0315-1989.yaml"
        ));
        assert!(is_conversation("groups/zen/conversations/journal.yaml"));
        assert!(!is_conversation("groups/zen/conversations"));
        assert!(!is_conversation("groups/zen/topics/conversations/a.yaml"));
        assert!(!is_conversation("persons/zen/conversations/a.yaml"));
    }
}
