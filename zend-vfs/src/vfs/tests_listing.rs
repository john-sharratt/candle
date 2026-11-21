//! Tests for what listings, path search and grep show and leave out.
//!
//! A listing shows only files a read can open, so a file whose name has a
//! short-name tail is left out of listing, path search and grep alike. Listing
//! never opens a file, so a non-UTF-8 or oversize workspace file still lists and
//! only its read is refused. The tests also pin the one-level and sorted listing,
//! whiteouts applying to every spelling of a path, and ignored or hidden files
//! staying out of listings while still reading by exact path.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_upper::tests::{listed, put, store_with_tree};

    /// **A listing shows only files a read can open.** `notes~2.md` has the
    /// shape of a Windows short name, so reads refuse it — and the listing,
    /// the path search and grep leave it out rather than show a file that
    /// cannot be read.
    #[test]
    fn a_file_the_store_cannot_open_is_not_listed() {
        let (dir, store) = store_with_tree();
        put(dir.path(), "notes~2.md", "tilde\n");
        assert_eq!(store.read("notes~2.md").unwrap(), None);
        assert!(!listed(&store, "").contains(&"notes~2.md".to_string()));
        assert!(!store.paths("").contains(&"notes~2.md".to_string()));
        assert!(store.paths("").contains(&"src/main.rs".to_string()));
    }

    /// A directory resolves as absent rather than erroring — `read` answers about
    /// files.
    #[test]
    fn reading_a_directory_path_is_absent() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.read("src").unwrap(), None);
        assert_eq!(s.read("src/util").unwrap(), None);
        assert_eq!(s.read("").unwrap(), None, "the root itself is not a file");
    }

    /// **`list_dir` never opens a file.** A listing that had to decode every
    /// file's bytes to report on it would refuse exactly where `read` does; it
    /// does not, because it never reads past `entry.metadata()` — the whole
    /// point of dropping `ListEntry::lines`, which used to force exactly that
    /// open.
    #[test]
    fn non_utf8_lower_file_still_lists() {
        let (dir, s) = store_with_tree();
        std::fs::write(dir.path().join("blob.bin"), [0xff, 0xfe, 0x00]).unwrap();
        let entries = s.list_dir("").unwrap().unwrap();
        let entry = entries
            .iter()
            .find(|e| e.path == "blob.bin")
            .expect("a non-UTF-8 file still lists");
        assert_eq!(entry.bytes, Some(3));
    }

    #[test]
    fn oversize_lower_file_lists_but_refuses_to_read() {
        let (dir, s) = store_with_tree();
        let big = (MAX_LOWER_FILE_BYTES + 1) as usize;
        std::fs::write(dir.path().join("huge.txt"), vec![b'a'; big]).unwrap();

        let err = s.read("huge.txt").unwrap_err();
        assert!(matches!(err, VfsError::Unreadable(_)), "{err:?}");

        let entries = s.list_dir("").unwrap().unwrap();
        let entry = entries
            .iter()
            .find(|e| e.path == "huge.txt")
            .expect("oversize files still list");
        assert_eq!(entry.bytes, Some(big), "with their true size");
    }

    // ── Lower layer: listing ─────────────────────────────────────────────────

    #[test]
    fn list_is_sorted_and_one_level_deep() {
        let (_dir, s) = store_with_tree();
        assert_eq!(listed(&s, ""), vec!["README.md", "src"]);
        for dir in ["src", "src/", "/src"] {
            assert_eq!(
                listed(&s, dir),
                vec!["src/main.rs", "src/util"],
                "dir {dir:?}",
            );
        }
        assert_eq!(listed(&s, "src/util"), vec!["src/util/helper.rs"]);
    }

    #[test]
    fn list_dir_of_a_missing_directory_is_not_found() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.list_dir("nothing/here").unwrap(), None);
    }

    /// A real directory path only — the argument is a directory to list, not a
    /// string prefix, so a partial segment or an exact file name no longer
    /// resolves to anything.
    #[test]
    fn list_dir_of_a_file_path_is_not_found() {
        let (_dir, s) = store_with_tree();
        assert_eq!(s.list_dir("README.md").unwrap(), None);
        assert_eq!(s.list_dir("src/ma").unwrap(), None);
    }

    /// A conflicting write elsewhere in the upper layer must not make a real
    /// file resolve as a directory too — `"README.md"` stays `None` even
    /// though `"README.md/notes.txt"` nonsensically implies it is one.
    #[test]
    fn list_dir_of_a_file_stays_not_found_despite_a_conflicting_deeper_write() {
        let (_dir, s) = store_with_tree();
        s.write("README.md/notes.txt", "oops".into()).unwrap();
        assert_eq!(s.list_dir("README.md").unwrap(), None);
    }

    /// A plain upper file and another upper file that collapses to a
    /// directory of the same name are two different things sharing one
    /// name — both must survive the listing, not silently collide into one.
    #[test]
    fn a_file_and_a_same_named_collapsed_directory_both_list() {
        let s = VfsStore::new();
        s.write("src", "plain file".into()).unwrap();
        s.write("src/foo.rs", "nested".into()).unwrap();
        let entries = s.list_dir("").unwrap().unwrap();
        let file = entries
            .iter()
            .find(|e| e.path == "src" && !e.dir)
            .expect("the plain file at \"src\" must still be listed");
        assert_eq!(file.bytes, Some("plain file".len()));
        let dir = entries
            .iter()
            .find(|e| e.path == "src" && e.dir)
            .expect("the collapsed directory named \"src\" must still be listed");
        assert_eq!(dir.bytes, None);
    }

    #[test]
    fn subdirectory_entries_carry_no_size() {
        let (_dir, s) = store_with_tree();
        let entries = s.list_dir("").unwrap().unwrap();
        let src = entries.iter().find(|e| e.path == "src").unwrap();
        assert!(src.dir);
        assert_eq!(src.bytes, None);
        assert!(!src.modified);
    }

    #[test]
    fn upper_shadows_lower_exactly_once() {
        let (_dir, s) = store_with_tree();
        s.write("src/main.rs", "fn main() { /* mine */ }\n".into())
            .unwrap();
        let entries = s.list_dir("src").unwrap().unwrap();
        assert_eq!(
            entries.iter().filter(|e| e.path == "src/main.rs").count(),
            1,
            "a shadowed path must appear once, not twice",
        );
        let main = entries.iter().find(|e| e.path == "src/main.rs").unwrap();
        assert!(main.modified);
        assert_eq!(main.bytes, Some("fn main() { /* mine */ }\n".len()));
        let util = entries.iter().find(|e| e.path == "src/util").unwrap();
        assert!(util.dir);
        assert!(!util.modified, "a directory entry is never itself modified");
    }

    #[test]
    fn a_session_only_file_lists_alongside_workspace_files() {
        let (_dir, s) = store_with_tree();
        s.write("src/scratch.rs", "// draft\n".into()).unwrap();
        assert_eq!(
            listed(&s, "src"),
            vec!["src/main.rs", "src/scratch.rs", "src/util"],
        );
    }

    /// A directory that exists only because the session wrote into it — never
    /// present on disk — still lists, synthesised purely from the upper layer.
    #[test]
    fn a_session_only_directory_lists_even_without_a_workspace_counterpart() {
        let (_dir, s) = store_with_tree();
        s.write("gen/a.txt", "hi".into()).unwrap();
        assert_eq!(listed(&s, "gen"), vec!["gen/a.txt"]);
        assert!(listed(&s, "").contains(&"gen".to_string()));
    }

    // ── Whiteouts ────────────────────────────────────────────────────────────

    #[test]
    fn deleting_a_lower_file_whiteouts_it_without_touching_disk() {
        let (dir, s) = store_with_tree();
        assert!(s.delete("README.md"));

        assert_eq!(s.read("README.md").unwrap(), None);
        assert!(!listed(&s, "").contains(&"README.md".to_string()));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "# project\n",
            "the file on disk must be untouched",
        );
        assert!(!s.delete("README.md"), "already whiteouted");
    }

    /// The whiteout must survive the alias: deleting by one spelling hides the
    /// path under every other.
    #[test]
    fn a_whiteout_applies_to_every_spelling_of_the_path() {
        let (_dir, s) = store_with_tree();
        s.delete("/src/main.rs");
        assert_eq!(s.read("src/main.rs").unwrap(), None);
        assert_eq!(s.read("./src/main.rs").unwrap(), None);
    }

    #[test]
    fn writing_over_a_whiteout_clears_it_and_counts_as_creation() {
        let (_dir, s) = store_with_tree();
        s.delete("README.md");
        assert!(
            s.write("README.md", "# mine\n".into()).unwrap(),
            "the path did not resolve while the whiteout stood",
        );
        assert_eq!(s.read("README.md").unwrap().as_deref(), Some("# mine\n"));
        assert!(listed(&s, "").contains(&"README.md".to_string()));
        // And deleting again re-hides it, since the lower file is still there.
        assert!(s.delete("README.md"));
        assert_eq!(s.read("README.md").unwrap(), None);
    }

    #[test]
    fn deleting_a_shadowed_path_hides_both_layers() {
        let (_dir, s) = store_with_tree();
        s.write("src/main.rs", "mine".into()).unwrap();
        assert!(s.delete("src/main.rs"));
        assert_eq!(
            s.read("src/main.rs").unwrap(),
            None,
            "the workspace file must not resurface once the shadow is removed",
        );
        assert_eq!(s.total_bytes(), 0);
        assert!(!listed(&s, "src").contains(&"src/main.rs".to_string()));
    }

    /// Shadowing a lower file is an overwrite, not a creation — the path already
    /// resolved before the call.
    #[test]
    fn first_write_over_a_lower_file_reports_overwrite() {
        let (_dir, s) = store_with_tree();
        assert!(!s.write("README.md", "changed".into()).unwrap());
        assert!(s.write("brand-new.md", "fresh".into()).unwrap());
    }

    // ── Ignore rules ─────────────────────────────────────────────────────────

    #[test]
    fn ignored_and_hidden_paths_are_omitted_from_listings_but_still_read() {
        let (dir, s) = store_with_tree();
        put(dir.path(), ".gitignore", "ignored/\n");
        put(dir.path(), "ignored/secret.txt", "shh\n");
        put(dir.path(), ".env", "TOKEN=1\n");

        let all = listed(&s, "");
        assert!(!all.contains(&"ignored".to_string()), "{all:?}");
        assert!(!all.contains(&".env".to_string()), "{all:?}");
        assert!(!all.contains(&".gitignore".to_string()), "{all:?}");

        // Hidden files are `ls`-invisible, not unreachable.
        assert_eq!(s.read(".env").unwrap().as_deref(), Some("TOKEN=1\n"));
        // An ignored file is genuinely out of scope for listing, but reading it by
        // exact path still works — `read` never consults the ignore rules.
        assert_eq!(
            s.read("ignored/secret.txt").unwrap().as_deref(),
            Some("shh\n")
        );
    }
}
