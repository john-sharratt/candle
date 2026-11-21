//! Tests for edit and write deltas.
//!
//! An edit costs only its changed lines, and successive edits replay in order. A
//! write supersedes every earlier edit and no longer reads the workspace copy. An
//! edit over a file changed on disk is refused rather than spliced into the wrong
//! place. An edit that rewrites most of a file is held as the whole file, and an
//! edit needs a file to edit and records nothing when it changes nothing.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_upper::tests::{put, store_with_tree};

    // ── Deltas ───────────────────────────────────────────────────────────────

    /// A workspace file of `n` numbered lines.
    pub(in super::super) fn numbered(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}\n")).collect()
    }

    /// **An edit holds only the lines it changed**, not the file it landed
    /// in: one changed line of a 500-line workspace file costs that line, and
    /// every read replays it onto the workspace's copy.
    #[test]
    fn an_edit_holds_only_its_changed_lines() {
        let (dir, s) = store_with_tree();
        let original = numbered(500);
        put(dir.path(), "big.txt", &original);
        let edited = original.replace("line 250\n", "line two-fifty\n");

        assert!(s.edit("big.txt", edited.clone()).unwrap());
        assert_eq!(
            s.total_bytes(),
            "line 250\n".len() + "line two-fifty\n".len()
        );
        assert_eq!(s.read("big.txt").unwrap().as_deref(), Some(edited.as_str()));
        assert!(s.is_modified("big.txt"));
        let listed = s.list_dir("").unwrap().unwrap();
        let big = listed.iter().find(|e| e.path == "big.txt").unwrap();
        assert_eq!((big.bytes, big.modified), (Some(edited.len()), true));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("big.txt")).unwrap(),
            original,
            "the workspace is untouched"
        );
    }

    /// Edits stack: each is positioned in the text the one before it left,
    /// and all of them replay in order.
    #[test]
    fn successive_edits_replay_in_order() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "cfg.toml", "a = 1\nb = 2\nc = 3\n");
        s.edit("cfg.toml", "a = 1\nb = 20\nc = 3\n".into()).unwrap();
        s.edit("cfg.toml", "a = 1\nb = 20\nc = 3\nd = 4\n".into())
            .unwrap();
        s.edit("cfg.toml", "a = 10\nb = 20\nc = 3\nd = 4\n".into())
            .unwrap();
        assert_eq!(
            s.read("cfg.toml").unwrap().as_deref(),
            Some("a = 10\nb = 20\nc = 3\nd = 4\n")
        );
    }

    /// **A write replaces the whole file and supersedes its edits**: what is
    /// held afterwards is the one replacement, and it no longer depends on the
    /// workspace's copy.
    #[test]
    fn a_write_supersedes_every_earlier_edit() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "notes.md", "one\ntwo\n");
        s.edit("notes.md", "one\n2\n".into()).unwrap();
        assert!(!s.write("notes.md", "fresh\n".into()).unwrap());
        assert_eq!(s.total_bytes(), "fresh\n".len());
        put(dir.path(), "notes.md", "changed underneath\n");
        assert_eq!(s.read("notes.md").unwrap().as_deref(), Some("fresh\n"));
    }

    /// An edit over a session-written file stacks on the write, and never
    /// touches the workspace.
    #[test]
    fn an_edit_stacks_on_a_write() {
        let s = VfsStore::new();
        s.write("new.rs", "fn a() {}\nfn b() {}\n".into()).unwrap();
        s.edit("new.rs", "fn a() {}\nfn c() {}\n".into()).unwrap();
        assert_eq!(
            s.read("new.rs").unwrap().as_deref(),
            Some("fn a() {}\nfn c() {}\n")
        );
        assert_eq!(
            s.total_bytes(),
            "fn a() {}\nfn b() {}\n".len() + "fn b() {}\n".len() + "fn c() {}\n".len()
        );
    }

    /// **An edit whose workspace copy changed underneath it is refused**, not
    /// spliced into the wrong place — and a whole-file write settles it.
    #[test]
    fn an_edit_over_a_file_changed_on_disk_diverges() {
        let (dir, s) = store_with_tree();
        put(dir.path(), "lib.rs", "fn one() {}\nfn two() {}\n");
        s.edit("lib.rs", "fn one() {}\nfn TWO() {}\n".into())
            .unwrap();
        put(dir.path(), "lib.rs", "fn zero() {}\n");

        assert!(matches!(s.read("lib.rs"), Err(VfsError::Diverged(_))));
        assert!(matches!(
            s.read_page("lib.rs", 0),
            Err(VfsError::Diverged(_))
        ));
        s.write("lib.rs", "fn settled() {}\n".into()).unwrap();
        assert_eq!(
            s.read("lib.rs").unwrap().as_deref(),
            Some("fn settled() {}\n")
        );
    }

    /// **An edit that rewrites most of a file is held as the whole file**:
    /// the same one replacement a write makes, superseding the edits before
    /// it and no longer depending on the workspace's copy.
    #[test]
    fn an_edit_that_rewrites_most_of_a_file_is_held_whole() {
        let (dir, s) = store_with_tree();
        put(
            dir.path(),
            "small.rs",
            "fn a() {}\nfn b() {}\nfn c() {}\nfn d() {}\n",
        );
        s.edit(
            "small.rs",
            "fn a() {}\nfn B() {}\nfn c() {}\nfn d() {}\n".into(),
        )
        .unwrap();
        let rewritten = "fn w() {}\nfn x() {}\nfn y() {}\nfn d() {}\n";
        s.edit("small.rs", rewritten.into()).unwrap();
        assert_eq!(
            s.total_bytes(),
            rewritten.len(),
            "one replacement, nothing else"
        );

        put(dir.path(), "small.rs", "changed underneath\n");
        assert_eq!(s.read("small.rs").unwrap().as_deref(), Some(rewritten));
    }

    /// An edit needs a file to edit, and one that changes nothing records
    /// nothing.
    #[test]
    fn an_edit_needs_a_file_and_a_change() {
        let (_dir, s) = store_with_tree();
        assert!(matches!(
            s.edit("absent.rs", "x".into()),
            Err(VfsError::Unreadable(_))
        ));
        assert!(!s.edit("README.md", "# project\n".into()).unwrap());
        assert!(!s.is_modified("README.md"));
        assert_eq!(s.total_bytes(), 0);

        s.delete("README.md");
        assert!(matches!(
            s.edit("README.md", "back".into()),
            Err(VfsError::Unreadable(_))
        ));
    }
}
