//! Tests for saving and restoring session state.
//!
//! A snapshot restored through its wire form reads exactly what the saved store
//! did, with each delta's moment kept and nothing replayed. Restoring replaces
//! whatever the store held, and a bad snapshot, one naming a protected path or
//! past the cap, is refused with the store left as it was.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_upper::tests::{listed, store_with_tree};

    // ── saving and restoring ─────────────────────────────────────────────────

    /// **A snapshot restores exactly, through its wire form**: a new store
    /// over the same folder reads what the old one did — an edit's chain, a
    /// write, a deletion — with every delta's moment kept and nothing
    /// replayed to get there.
    #[test]
    fn a_snapshot_restores_exactly_through_its_wire_form() {
        let (dir, store) = store_with_tree();
        store.edit("README.md", "# project\nmore\n".into()).unwrap();
        store
            .edit("README.md", "# project\nmore\nand more\n".into())
            .unwrap();
        store.write("src/new.rs", "new\n".into()).unwrap();
        store.delete("src/main.rs");

        let wire = serde_json::to_string(&store.snapshot()).unwrap();
        let back: Snapshot = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, store.snapshot());

        let restored = VfsStore::with_root(dir.path());
        restored.restore(back).unwrap();
        for path in [
            "README.md",
            "src/new.rs",
            "src/main.rs",
            "src/util/helper.rs",
        ] {
            assert_eq!(
                restored.read(path).unwrap(),
                store.read(path).unwrap(),
                "{path}"
            );
            assert_eq!(restored.deltas(path), store.deltas(path), "{path}");
        }
        assert_eq!(listed(&restored, "src"), listed(&store, "src"));
        assert_eq!(restored.total_bytes(), store.total_bytes());
    }

    /// **Restoring replaces what the store held**, and an empty snapshot
    /// leaves an empty store empty.
    #[test]
    fn restoring_replaces_the_session_layer() {
        let (dir, store) = store_with_tree();
        store.write("a.txt", "a\n".into()).unwrap();
        let saved = store.snapshot();
        let other = VfsStore::with_root(dir.path());
        other.write("b.txt", "b\n".into()).unwrap();
        other.restore(saved).unwrap();
        assert!(other.is_modified("a.txt"));
        assert!(!other.is_modified("b.txt"));
        assert!(VfsStore::new().snapshot().is_empty());
        let empty = VfsStore::new();
        empty.restore(Snapshot::default()).unwrap();
        assert!(empty.snapshot().is_empty());
    }

    /// **A snapshot naming a protected path, or past the cap, is refused**
    /// and the store is left as it was.
    #[test]
    fn a_bad_snapshot_is_refused_whole() {
        let (_dir, store) = store_with_tree();
        store.write("kept.txt", "kept\n".into()).unwrap();
        let protected: Snapshot =
            serde_json::from_str(r#"{"chains":{"secrets/key.txt":{"deltas":[],"size":1}}}"#)
                .unwrap();
        assert!(matches!(
            store.restore(protected),
            Err(VfsError::Forbidden(_))
        ));
        let huge = "x".repeat(MAX_BYTES + 1);
        let too_big: Snapshot = serde_json::from_value(serde_json::json!({
            "chains": {
                "big.txt": {
                    "deltas": [{ "at_ns": 1, "kind": "replace", "content": huge }],
                    "size": MAX_BYTES + 1
                }
            }
        }))
        .unwrap();
        let no_tree: Snapshot = serde_json::from_value(serde_json::json!({
            "base": { "tree": "not an object id" },
            "chains": { "a.txt": { "deltas": [], "size": 1 } }
        }))
        .unwrap();
        assert!(matches!(
            store.restore(no_tree),
            Err(VfsError::Unreadable(_))
        ));
        assert!(
            serde_json::from_str::<Snapshot>(r#"{"a.txt":{"deltas":[],"size":1}}"#).is_err(),
            "a map of paths with no `chains` is not a snapshot"
        );
        assert!(matches!(store.restore(too_big), Err(VfsError::Full)));
        assert!(
            store.is_modified("kept.txt"),
            "the store was left as it was"
        );
    }
}
