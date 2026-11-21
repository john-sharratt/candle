//! Tests for path normalisation.
//!
//! Backslash separators are accepted, so a path echoed back from a Windows host
//! addresses the same entry. A `..` pops the stack and can never climb above the
//! root. No segment name is special, and the immediate-child rule respects
//! directory boundaries, so `src` does not match `srcx/...`.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn a_short_name_is_a_tilde_and_a_digit() {
        assert!(is_short_name("SECRET~1"));
        assert!(is_short_name("PROGRA~2.TXT"));
        assert!(!is_short_name("notes~"));
        assert!(!is_short_name("~draft.md"));
        assert!(!is_short_name("plain.rs"));
    }

    // ── normalize ────────────────────────────────────────────────────────────

    #[test]
    fn normalize_collapses_to_one_canonical_key() {
        for spelling in [
            "src/main.rs",
            "/src/main.rs",
            "./src/main.rs",
            "src/./main.rs",
            "src//main.rs",
            "src/util/../main.rs",
            "/./src/../src/main.rs",
        ] {
            assert_eq!(
                VfsStore::normalize(spelling),
                "src/main.rs",
                "spelling {spelling:?}",
            );
        }
    }

    /// Windows-style separators are accepted, so a model echoing a path back from
    /// a Windows-hosted daemon still addresses the same entry.
    #[test]
    fn normalize_accepts_backslash_separators() {
        assert_eq!(VfsStore::normalize(r"src\main.rs"), "src/main.rs");
        assert_eq!(VfsStore::normalize(r"\src\main.rs"), "src/main.rs");
    }

    /// `..` can never climb above the root: popping an empty stack is a no-op, so
    /// a traversal attempt lands back inside the workspace.
    #[test]
    fn normalize_cannot_escape_the_root() {
        assert_eq!(VfsStore::normalize("../../../etc/passwd"), "etc/passwd");
        assert_eq!(VfsStore::normalize("/../../.."), "");
        assert_eq!(VfsStore::normalize(".."), "");
    }

    /// **No segment is special.** A folder named `workspace` is an ordinary
    /// folder wherever it sits, and the root spells as the empty key.
    #[test]
    fn normalize_treats_every_segment_as_a_plain_name() {
        assert_eq!(
            VfsStore::normalize("workspace/src/a.rs"),
            "workspace/src/a.rs"
        );
        assert_eq!(VfsStore::normalize("/workspace"), "workspace");
        assert_eq!(VfsStore::normalize("/"), "");
        assert_eq!(VfsStore::normalize(""), "");
    }

    #[test]
    fn immediate_child_collapses_deeper_paths_to_their_subdirectory() {
        assert_eq!(
            VfsStore::immediate_child("", "README.md"),
            Some(("README.md".to_string(), false)),
        );
        assert_eq!(
            VfsStore::immediate_child("", "src/main.rs"),
            Some(("src".to_string(), true)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/main.rs"),
            Some(("src/main.rs".to_string(), false)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/util/helper.rs"),
            Some(("src/util".to_string(), true)),
        );
        assert_eq!(
            VfsStore::immediate_child("src", "src/util/deep/leaf.rs"),
            Some(("src/util".to_string(), true)),
            "three levels down still collapses to the immediate subdirectory, not the leaf",
        );
    }

    /// A directory boundary, not a string prefix: `src` must not falsely match
    /// `srcx/…` the way a plain `starts_with` would.
    #[test]
    fn immediate_child_respects_the_directory_boundary() {
        assert_eq!(VfsStore::immediate_child("src", "srcx/file.rs"), None);
        assert_eq!(VfsStore::immediate_child("src", "docs/readme.md"), None);
        assert_eq!(
            VfsStore::immediate_child("src", "src"),
            None,
            "src is dir itself, not a child of it"
        );
    }
}
