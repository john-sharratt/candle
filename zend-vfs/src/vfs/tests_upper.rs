//! Tests for the session layer's own behaviour.
//!
//! A binary file is held as its bytes and counted against the cap at its length,
//! and is refused only when read as text. An upload past the cap is refused, and
//! a write the cap rejects leaves the overlay exactly as it was, including a
//! whiteout it was about to clear. An overwrite is measured against the slot it
//! replaces. Shared behind an `Arc` by concurrent tool calls, writes to distinct
//! paths all survive, and writers to one path leave exactly one winner. A store
//! never touches the disk.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use std::path::Path;

    use tempfile::TempDir;

    pub(in super::super) fn store_with_tree() -> (TempDir, VfsStore) {
        let dir = tempfile::tempdir().unwrap();
        put(dir.path(), "README.md", "# project\n");
        put(dir.path(), "src/main.rs", "fn main() {}\n");
        put(dir.path(), "src/util/helper.rs", "pub fn h() {}\n");
        let store = VfsStore::with_root(dir.path());
        (dir, store)
    }

    pub(in super::super) fn put(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    pub(in super::super) fn listed(store: &VfsStore, dir: &str) -> Vec<String> {
        store
            .list_dir(dir)
            .unwrap()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.path)
            .collect()
    }

    // ── binary files ─────────────────────────────────────────────────────────

    /// **A binary file is held as its bytes** — one `ReplaceBinary`, counted
    /// against the cap at its length, read back byte for byte, refused only as
    /// text; text written as bytes is an ordinary `Replace`.
    #[test]
    fn a_binary_file_is_held_as_its_bytes() {
        let store = VfsStore::new();
        let png = vec![0x89, b'P', b'N', b'G', 0xff, 0x00];
        assert!(store.write_bytes("logo.png", png.clone()).unwrap());
        assert_eq!(
            store.deltas("logo.png").unwrap()[0].delta,
            FileDelta::ReplaceBinary {
                content: png.clone()
            }
        );
        assert_eq!(store.read_bytes("logo.png").unwrap(), Some(png.clone()));
        assert!(matches!(
            store.read("logo.png"),
            Err(VfsError::Unreadable(_))
        ));
        assert_eq!(store.total_bytes(), png.len());
        store.write_bytes("notes.txt", b"plain\n".to_vec()).unwrap();
        assert_eq!(
            store.deltas("notes.txt").unwrap()[0].delta,
            FileDelta::Replace {
                content: "plain\n".into()
            }
        );
        // A binary delta applied — a sandbox's capture — is held the same way.
        let other = VfsStore::new();
        other
            .apply("logo.png", &store.deltas("logo.png").unwrap())
            .unwrap();
        assert_eq!(other.read_bytes("logo.png").unwrap(), Some(png));
    }

    /// **An upload past the cap is refused**, the store left as it was.
    #[test]
    fn bytes_past_the_cap_are_refused() {
        let store = VfsStore::new();
        let big = vec![0xffu8; MAX_BYTES + 1];
        assert!(matches!(
            store.write_bytes("big.bin", big),
            Err(VfsError::Full)
        ));
        assert_eq!(store.read_bytes("big.bin").unwrap(), None);
    }

    // ── the changes a checkout replays ───────────────────────────────────────

    /// **`changes` is every chain the session holds, delta for delta** — and
    /// replayed onto the workspace's copies it gives exactly what the store
    /// reads.
    #[test]
    fn changes_are_every_chain_and_replay_to_what_the_store_reads() {
        let (dir, store) = store_with_tree();
        store.edit("README.md", "# project\nmore\n".into()).unwrap();
        store.write("src/new.rs", "new\n".into()).unwrap();
        store.delete("src/main.rs");

        let changes = store.changes().unwrap();
        assert_eq!(
            changes.paths().collect::<Vec<_>>(),
            ["README.md", "src/main.rs", "src/new.rs"]
        );
        for path in ["README.md", "src/main.rs", "src/new.rs"] {
            assert_eq!(changes.chain(path).unwrap(), store.deltas(path).unwrap());
            let lower = std::fs::read(dir.path().join(path)).ok();
            assert_eq!(
                changes.replay(path, lower).unwrap(),
                store.read(path).unwrap().map(String::into_bytes),
                "{path}"
            );
        }
        assert!(VfsStore::new().changes().unwrap().is_empty());
    }

    /// **A store never touches the disk** — a write, an overwrite and a
    /// delete all land in the session's own layer.
    #[test]
    fn a_write_never_reaches_disk() {
        let (dir, s) = store_with_tree();
        s.write("new.txt", "x".into()).unwrap();
        s.write("README.md", "changed".into()).unwrap();
        assert!(s.delete("src/main.rs"));
        assert!(!dir.path().join("new.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "# project\n"
        );
        assert!(dir.path().join("src/main.rs").exists());
    }

    // ── Upper layer alone ────────────────────────────────────────────────────

    #[test]
    fn upper_only_store_round_trips_and_accounts_bytes() {
        let s = VfsStore::new();
        assert_eq!(s.total_bytes(), 0);
        assert!(s.root().is_none());

        assert!(
            s.write("a.txt", "hello".into()).unwrap(),
            "first write creates"
        );
        assert_eq!(s.read("a.txt").unwrap().as_deref(), Some("hello"));
        assert_eq!(s.total_bytes(), 5);

        assert!(
            !s.write("a.txt", "hi".into()).unwrap(),
            "second write overwrites",
        );
        assert_eq!(
            s.total_bytes(),
            2,
            "overwriting with less must release budget",
        );
        assert_eq!(s.read("missing.txt").unwrap(), None);
    }

    #[test]
    fn upper_only_delete_removes_without_leaving_a_whiteout() {
        let s = VfsStore::new();
        s.write("a.txt", "x".into()).unwrap();
        assert!(s.delete("a.txt"));
        assert_eq!(s.read("a.txt").unwrap(), None);
        assert_eq!(s.total_bytes(), 0);
        assert!(!s.delete("a.txt"), "second delete finds nothing");
        // With nothing below to hide, the path is simply gone — a later write is
        // an ordinary creation.
        assert!(s.write("a.txt", "y".into()).unwrap());
    }

    #[test]
    fn different_spellings_address_one_entry() {
        let s = VfsStore::new();
        s.write("/src/main.rs", "one".into()).unwrap();
        s.write("src/main.rs", "two".into()).unwrap();
        assert_eq!(
            s.read("./src/../src/main.rs").unwrap().as_deref(),
            Some("two")
        );
        assert_eq!(listed(&s, ""), vec!["src"], "one collapsed entry, not two");
        assert_eq!(listed(&s, "src"), vec!["src/main.rs"]);
        assert_eq!(s.total_bytes(), 3, "one entry, not two");
    }

    // ── Capacity ─────────────────────────────────────────────────────────────

    #[test]
    fn write_beyond_the_cap_is_rejected_and_changes_nothing() {
        let s = VfsStore::new();
        s.write("big.bin", "x".repeat(MAX_BYTES - 10)).unwrap();
        let before = s.total_bytes();

        let err = s.write("more.bin", "y".repeat(64)).unwrap_err();
        assert!(matches!(err, VfsError::Full));
        assert_eq!(s.total_bytes(), before, "a rejected write must not consume");
        assert_eq!(s.read("more.bin").unwrap(), None);

        // What does fit still succeeds.
        s.write("more.bin", "y".repeat(10)).unwrap();
        assert_eq!(s.total_bytes(), MAX_BYTES);
    }

    /// A write that the cap rejects must leave the overlay exactly as it was —
    /// including a whiteout it was about to clear. Otherwise a failed write
    /// resurrects a file the session had deleted.
    #[test]
    fn a_rejected_write_does_not_clear_a_whiteout() {
        let (_dir, s) = store_with_tree();
        s.write("filler.bin", "x".repeat(MAX_BYTES - 10)).unwrap();
        assert!(s.delete("README.md"), "whiteout the lower file");
        assert_eq!(s.read("README.md").unwrap(), None);

        let err = s.write("README.md", "y".repeat(4096)).unwrap_err();
        assert!(matches!(err, VfsError::Full), "{err:?}");

        assert_eq!(
            s.read("README.md").unwrap(),
            None,
            "the failed write must not have resurrected the workspace file",
        );
        assert!(!listed(&s, "").contains(&"README.md".to_string()));
    }

    /// The cap counts the *replacement*, not the sum: overwriting a large file
    /// with a large file is fine even though their total exceeds the budget.
    #[test]
    fn overwrite_is_measured_against_the_slot_it_replaces() {
        let s = VfsStore::new();
        s.write("big.bin", "x".repeat(MAX_BYTES - 100)).unwrap();
        s.write("big.bin", "y".repeat(MAX_BYTES - 100)).unwrap();
        assert_eq!(s.total_bytes(), MAX_BYTES - 100);
    }

    // ── Lower layer: read-through ────────────────────────────────────────────

    #[test]
    fn read_falls_through_and_costs_no_budget() {
        let (_dir, s) = store_with_tree();
        assert_eq!(
            s.read("src/main.rs").unwrap().as_deref(),
            Some("fn main() {}\n")
        );
        assert_eq!(
            s.read("/src/util/helper.rs").unwrap().as_deref(),
            Some("pub fn h() {}\n"),
        );
        assert_eq!(s.total_bytes(), 0, "reading through retains nothing");
        assert_eq!(s.read("src/nope.rs").unwrap(), None);
    }

    // ── Sharing ──────────────────────────────────────────────────────────────

    /// One store is shared across the whole daemon behind an `Arc`, so concurrent
    /// tool calls hit it in parallel. Distinct paths must all survive, byte
    /// accounting must stay exact, and reads that race writes must never observe
    /// a torn value.
    #[test]
    fn concurrent_writes_and_reads_stay_consistent() {
        use std::sync::Arc;

        let (_dir, store) = store_with_tree();
        let store = Arc::new(store);
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let s = Arc::clone(&store);
                std::thread::spawn(move || {
                    for i in 0..50 {
                        s.write(&format!("gen/{t}-{i}.txt"), format!("{t}:{i}"))
                            .unwrap();
                        // Racing the workspace layer at the same time.
                        assert_eq!(
                            s.read("src/util/helper.rs").unwrap().as_deref(),
                            Some("pub fn h() {}\n"),
                        );
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(listed(&store, "gen/").len(), 8 * 50);
        let expected: usize = (0..8)
            .flat_map(|t| (0..50).map(move |i| format!("{t}:{i}").len()))
            .sum();
        assert_eq!(store.total_bytes(), expected, "byte accounting drifted");
    }

    /// Concurrent writers to one path resolve to some single winner — never a
    /// blend of the two, and never double-counted bytes.
    #[test]
    fn concurrent_writes_to_one_path_leave_exactly_one_winner() {
        use std::sync::Arc;

        let store = Arc::new(VfsStore::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let s = Arc::clone(&store);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        s.write("contended.txt", format!("writer-{t}")).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        let final_value = store.read("contended.txt").unwrap().unwrap();
        assert!(
            (0..8).any(|t| final_value == format!("writer-{t}")),
            "torn value {final_value:?}",
        );
        assert_eq!(store.list_dir("").unwrap().unwrap().len(), 1);
        assert_eq!(store.total_bytes(), final_value.len());
    }

    #[test]
    fn unicode_content_survives_both_layers() {
        let (dir, s) = store_with_tree();
        let text = "こんにちは 🌍 — overlay\n";
        put(dir.path(), "uni.txt", text);
        assert_eq!(s.read("uni.txt").unwrap().as_deref(), Some(text));

        let edited = format!("{text}さようなら\n");
        s.write("uni.txt", edited.clone()).unwrap();
        assert_eq!(s.read("uni.txt").unwrap().as_deref(), Some(edited.as_str()));
        assert_eq!(s.total_bytes(), edited.len(), "bytes, not chars");
    }
}
