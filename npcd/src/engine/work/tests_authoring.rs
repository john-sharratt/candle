//! Tests that character and place authoring reach the disk, survive a reload, and
//! stay in memory when there is no mind folder.
//!
//! What a character is lands in its sheet only at the commit, and a memory lands
//! in the character's own memory layer. A place's entries land in its locations
//! layer rather than its history, and they survive a reload. Without a mind folder,
//! every verb keeps its in-memory draft and does not panic. Craft fields change in
//! place, so the comments around them survive.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use serde_json::json;

    use super::super::tests_mission::tests::{on_operation, stored, vault};

    use super::super::tests_documents::tests::{act, vault_with_documents};

    // ── character and place authoring reach the disk ─────────────────────────
    //
    // D.6 #1: `character_write_*` and `place_write_*` used to fall to an in-RAM
    // `Record.body` that a restart threw away. They write the mind folder now,
    // through the bench working copy, and survive a reload.

    /// The vault, with a mind folder holding a courier's sheet — everything the
    /// character and place authoring tests write into or read back.
    pub(in super::super) fn vault_authoring(name: &str) -> (Hosted, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("npcd-author-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("personalities")).unwrap();
        std::fs::write(
            root.join("personalities/the-courier.yaml"),
            "# the courier — identity definition.\nid: the-courier\nname: The Courier\n",
        )
        .unwrap();
        // The record already names the courier and the eastern flats (the vault
        // seed); the world/ layers are made on the first commit into them.
        let h = vault();
        h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
            &root,
        ))));
        on_operation(&h, &root);
        (h, root)
    }

    /// **Who a character is reaches the sheet in the operation's store, and
    /// not before the commit.** The `anchor` field, written the way a
    /// portrait's words are.
    #[test]
    fn character_identity_is_written_to_the_sheet_and_survives_a_commit() {
        let (h, root) = vault_authoring("identity");
        let out = perform(
            &h,
            "m1",
            &act(
                "character_write_identity",
                json!({"of":"the courier","what":"steady so long that being doubted is the thing they cannot take"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            !stored(&h, "personalities/the-courier.yaml")
                .unwrap()
                .contains("cannot take"),
            "the sheet moved before the commit"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"cast the courier"}))
        )
        .happened());
        assert!(
            !std::fs::read_to_string(root.join("personalities/the-courier.yaml"))
                .unwrap()
                .contains("cannot take"),
            "only a publish reaches the record"
        );
        let sheet = stored(&h, "personalities/the-courier.yaml").unwrap();
        let doc: Value = serde_yaml::from_str(&sheet).unwrap();
        assert_eq!(
            doc["anchor"],
            json!("steady so long that being doubted is the thing they cannot take")
        );
        // The splice kept the sheet's own header and its other fields.
        assert!(
            sheet.contains("# the courier — identity definition."),
            "the header was lost: {sheet}"
        );
        assert_eq!(
            doc["name"],
            json!("The Courier"),
            "an untouched field moved"
        );
    }

    /// **What a character wants becomes a new top-level field on the sheet**,
    /// beside the anchor, and the anchor written first is still there.
    #[test]
    fn character_wants_becomes_a_new_field_beside_the_anchor() {
        let (h, _) = vault_authoring("wants");
        perform(
            &h,
            "m1",
            &act(
                "character_write_identity",
                json!({"of":"the courier","what":"reliable to a fault"}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "character_write_wants",
                json!({"of":"the courier","what":"to be trusted again by the one house that stopped"}),
            ),
        );
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"gave the courier a reason"})),
        );

        let sheet = stored(&h, "personalities/the-courier.yaml").unwrap();
        let doc: Value = serde_yaml::from_str(&sheet).unwrap();
        assert_eq!(doc["anchor"], json!("reliable to a fault"));
        assert_eq!(
            doc["wants"],
            json!("to be trusted again by the one house that stopped")
        );
    }

    /// **A memory lands as a document in the character's own memory layer**, not
    /// on the sheet — and not before the commit.
    #[test]
    fn a_memory_lands_in_the_characters_memory_layer() {
        let (h, root) = vault_authoring("memory");
        let out = perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"the afternoon they waited four hours at a gate that was never going to open"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        let key = "layers/memory/the-courier/memories.md";
        assert_eq!(
            stored(&h, key),
            None,
            "the memory was kept before the commit"
        );

        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"remembered the gate"})),
        );
        assert!(
            !root.join(key).exists(),
            "only a publish reaches the record"
        );
        let written = stored(&h, key).unwrap();
        assert!(
            written.contains("waited four hours at a gate that was never going to open"),
            "{written}"
        );
    }

    /// **A second memory is added to the first**, the way an era's entries are.
    #[test]
    fn a_second_memory_is_added_rather_than_replacing_the_first() {
        let (h, _) = vault_authoring("memory-append");
        perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"First."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "character_write_memories",
                json!({"of":"the courier","what":"Second."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"two afternoons"})),
        );
        let written = stored(&h, "layers/memory/the-courier/memories.md").unwrap();
        assert!(written.contains("First."), "{written}");
        assert!(
            written.find("First.") < written.find("Second."),
            "the memories came back out of order: {written}"
        );
    }

    /// Land what operation [`OP`]'s store holds at `key` in the mind folder at
    /// `root`, as its publish does.
    pub(in super::super) fn publish(h: &Hosted, root: &std::path::Path, key: &str) {
        let text = stored(h, key).unwrap();
        let full = root.join(key);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, text).unwrap();
    }

    /// **A place entry lands in the locations layer and, published, survives a
    /// reload.** A fresh, empty record reads the published mind and the place
    /// is part of the record again.
    #[test]
    fn a_place_entry_lands_in_the_locations_layer_and_survives_a_reload() {
        let (h, root) = vault_authoring("place-entry");
        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"ground fused smooth, no cover anywhere on it, and a wind that does not stop"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        let key = "layers/world/locations/the-eastern-flats.md";
        assert_eq!(stored(&h, key), None, "the store moved before the commit");

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"surveyed the flats"}))
        )
        .happened());
        assert!(
            stored(&h, key)
                .unwrap()
                .contains("a wind that does not stop"),
            "the entry did not reach the locations layer"
        );
        assert!(
            !root.join(key).exists(),
            "only a publish reaches the record"
        );

        // Published, then a fresh process with an empty record: it reads the
        // mind and the place is filed, at its path, again.
        publish(&h, &root, key);
        let mut fresh = crate::sim::record::Record::new();
        fresh.index_canon(&crate::mind::files::MindFiles::folder(&root));
        assert_eq!(
            fresh.path_of("the eastern flats").as_deref(),
            Some("layers/world/locations/the-eastern-flats.md")
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().state,
            State::Filed
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().kind,
            Kind::Place
        );
    }

    /// **A place's local history lands in geography, not on its entry**, and
    /// survives a reload the same way.
    #[test]
    fn a_places_local_history_lands_in_geography_and_survives_a_reload() {
        let (h, root) = vault_authoring("place-history");
        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_local_history",
                json!({"of":"the eastern flats","what":"why the road stops here — what came across it, and the year it stopped being worth rebuilding"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"gave the flats a reason"}))
        )
        .happened());
        let key = "layers/world/geography/the-eastern-flats.md";
        assert!(
            stored(&h, key)
                .unwrap()
                .contains("the year it stopped being worth rebuilding"),
            "the history did not reach the geography layer"
        );
        publish(&h, &root, key);

        // A place known only by its history is part of the record again, but its
        // history is NOT its entry path: `Item.path` names the entry, which is
        // still unwritten here, and the history is found name-derived instead.
        let mut fresh = crate::sim::record::Record::new();
        fresh.index_canon(&crate::mind::files::MindFiles::folder(&root));
        assert!(
            fresh.path_of("the eastern flats").is_none(),
            "the geography history was adopted as the entry path"
        );
        assert_eq!(
            fresh.history_path("the eastern flats").as_deref(),
            Some("layers/world/geography/the-eastern-flats.md")
        );
        assert_eq!(
            fresh.by_name("the eastern flats").unwrap().state,
            State::Filed
        );
    }

    /// **A place with BOTH documents writes its entries into the entry file, not
    /// its history.** After both are on disk and the record has re-indexed them,
    /// `Item.path` names the locations entry — so `place_write_entry` appends
    /// there, and the geography history it also has is left untouched. This is
    /// the defect the geography-pass fix closes: an entry that leaked into the
    /// history because indexing re-pointed `Item.path` at geography.
    #[test]
    fn a_place_with_both_documents_writes_entries_to_the_entry_not_the_history() {
        let (h, root) = vault_authoring("place-both");
        std::fs::create_dir_all(root.join("layers/world/locations")).unwrap();
        std::fs::create_dir_all(root.join("layers/world/geography")).unwrap();
        std::fs::write(
            root.join("layers/world/locations/the-eastern-flats.md"),
            "# the eastern flats\n\nGround fused smooth.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("layers/world/geography/the-eastern-flats.md"),
            "# the eastern flats\n\nWhy the road stops here.\n",
        )
        .unwrap();
        // Re-index off the disk so the record adopts both documents.
        h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
            &root,
        ))));
        h.sim(|s| {
            assert_eq!(
                s.record.path_of("the eastern flats").as_deref(),
                Some("layers/world/locations/the-eastern-flats.md"),
                "the entry path was overwritten by the geography history"
            );
        });

        let out = perform(
            &h,
            "m1",
            &act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"a new survey line about the wind"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"surveyed the flats again"}))
        )
        .happened());

        let entry = stored(&h, "layers/world/locations/the-eastern-flats.md").unwrap();
        assert!(
            entry.contains("a new survey line about the wind"),
            "the entry prose did not reach the entry file: {entry}"
        );
        let history = stored(&h, "layers/world/geography/the-eastern-flats.md").unwrap();
        assert!(
            !history.contains("a new survey line about the wind"),
            "the entry prose leaked into the history file: {history}"
        );
    }

    /// **The mind-less daemon still works.** With no mind folder, every one of
    /// the five verbs keeps its in-RAM draft rather than panicking — the shipped
    /// behaviour, preserved for the world (Battle Cities) that has no disk.
    #[test]
    fn character_and_place_authoring_without_a_mind_folder_stays_in_ram() {
        let h = vault();
        for a in [
            act(
                "character_write_identity",
                json!({"of":"the courier","what":"steady"}),
            ),
            act(
                "character_write_wants",
                json!({"of":"the courier","what":"to be trusted"}),
            ),
            act(
                "character_write_memories",
                json!({"of":"the courier","what":"an afternoon at a gate"}),
            ),
            act(
                "place_write_entry",
                json!({"of":"the eastern flats","what":"smooth ground and a wind"}),
            ),
            act(
                "place_write_local_history",
                json!({"of":"the eastern flats","what":"why the road stops"}),
            ),
        ] {
            let out = perform(&h, "m1", &a);
            assert!(
                out.happened(),
                "{} did not survive the no-root path: {out:?}",
                a.tool
            );
        }
        h.sim(|s| {
            assert!(
                s.record
                    .by_name("the courier")
                    .unwrap()
                    .body
                    .contains("steady"),
                "the courier's draft was lost"
            );
            assert!(
                s.record
                    .by_name("the eastern flats")
                    .unwrap()
                    .body
                    .contains("smooth ground"),
                "the flats' draft was lost"
            );
        });
    }

    // ── the craft libraries ─────────────────────────────────────────────────

    #[test]
    fn a_mood_can_be_read_and_changed_and_lands_in_the_store() {
        let (h, root) = vault_with_documents("mood");
        let read = perform(
            &h,
            "m1",
            &act("library_read", json!({"kind":"mood","id":"undone"})),
        );
        assert!(
            read.line().unwrap().contains("Something has been opened"),
            "{read:?}"
        );

        let out = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"description",
                    "text":"So thoroughly opened that the whole interior has rearranged."
                }),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why":"it read as two things at once"}),
            ),
        );

        let kept = stored(&h, "moods/undone.yaml").unwrap();
        assert!(kept.contains("whole interior has rearranged"), "{kept}");
        assert!(!std::fs::read_to_string(root.join("moods/undone.yaml"))
            .unwrap()
            .contains("whole interior"));
    }

    /// **The comments survive.** This is the entire reason the write goes
    /// through the splice: a `serde_yaml` round trip still produces a document
    /// that loads perfectly, with the half a person wrote silently gone.
    #[test]
    fn changing_a_field_keeps_everything_written_around_it() {
        let (h, _) = vault_with_documents("mood-comments");
        perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"description","text":"Rearranged."
                }),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));

        let on_disk = stored(&h, "moods/undone.yaml").unwrap();
        assert!(
            on_disk.contains("# The felt register — its KV is loaded"),
            "the comment was lost: {on_disk}"
        );
        assert!(
            on_disk.contains("category: mood"),
            "an untouched field moved: {on_disk}"
        );
        assert!(
            on_disk.contains("Something has been opened"),
            "the template was rewritten"
        );
    }

    #[test]
    fn a_library_that_does_not_exist_is_refused_by_name() {
        let (h, _) = vault_with_documents("mood-bad-kind");
        let out = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({"kind":"weather","id":"x","field":"template","text":"y"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("mood"), "{out:?}");
    }

    /// `id` and `category` are how everything else finds this piece, so they
    /// are not a Maker's to change from inside the world.
    #[test]
    fn only_the_fields_that_are_a_makers_to_change_are_accepted() {
        let (h, _) = vault_with_documents("mood-bad-field");
        for field in ["id", "category", "anything"] {
            let out = perform(
                &h,
                "m1",
                &act(
                    "library_write",
                    json!({
                        "kind":"mood","id":"undone","field":field,"text":"x"
                    }),
                ),
            );
            assert!(!out.happened(), "{field} was accepted");
        }
        let ok = perform(
            &h,
            "m1",
            &act(
                "library_write",
                json!({
                    "kind":"mood","id":"undone","field":"template","text":"Steady."
                }),
            ),
        );
        assert!(ok.happened(), "{ok:?}");
    }

    // ── likenesses ──────────────────────────────────────────────────────────

    #[test]
    fn the_words_a_likeness_is_drawn_from_can_be_read_and_changed() {
        let (h, _) = vault_with_documents("portrait");
        let read = perform(
            &h,
            "m1",
            &act("portrait_prompt_read", json!({"of":"ash-the-drifter"})),
        );
        assert!(
            read.line().unwrap().contains("lean sun-darkened man"),
            "{read:?}"
        );

        let out = perform(
            &h,
            "m1",
            &act(
                "portrait_prompt_edit",
                json!({
                    "of":"ash-the-drifter",
                    "carrying":"the same man after the winter, quieter and harder to read"
                }),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"the winter changed him"})),
        );

        let on_disk = stored(&h, "personalities/ash-the-drifter.yaml").unwrap();
        assert!(on_disk.contains("after the winter"), "{on_disk}");
        // The picture it names is untouched — a redraw follows the words.
        assert!(
            on_disk.contains("image: portraits/ash-the-drifter.png"),
            "{on_disk}"
        );
        assert!(
            on_disk.contains("# Biography is NOT here."),
            "the header was lost"
        );
    }

    #[test]
    fn drawing_somebody_the_world_has_never_heard_of_is_refused() {
        let (h, _) = vault_with_documents("portrait-absent");
        let out = perform(
            &h,
            "m1",
            &act("portrait_prompt_read", json!({"of":"nobody at all"})),
        );
        assert!(!out.happened(), "{out:?}");
    }
}
