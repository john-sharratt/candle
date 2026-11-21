//! Tests that a second writable handle is refused, and that a runtime-created record survives compaction through the one shared writer.
//!
//! The shape reproduced is the daemon's: a registry and an engine each opening the same directory, which loses the characters created at runtime. With one shared handle the compaction sees the whole cast. The custom-object test covers last-writer-wins for a durable keyed value across a write, a compaction and a reopen, where the superseded value does not come back.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{cast_on_disk, tmp_dir};

    /// **Two writable handles on one substrate lose the records neither told
    /// the other about.**
    ///
    /// This is the `npcd` daemon's shape reproduced exactly. The character
    /// registry opens the substrate to rebuild its cast; the engine opens the
    /// *same* directory a moment later for conversation turns. Both hold a
    /// writable handle, and each built its [`SubstratePersistence::npc_locs`]
    /// from the records present at *its own* open.
    ///
    /// Nothing tells one handle about the other's appends. So a character
    /// created while the daemon runs is written through the registry's handle
    /// and is invisible to the engine's — and [`compaction::collect_live_records`]
    /// reads only the `npc_locs` of the handle doing the compacting. The
    /// character is not carried forward, and the next restart has never heard of
    /// it.
    ///
    /// Characters that predate *both* opens are in both views and survive, which
    /// is why the loss presented as intermittent rather than total: it is
    /// precisely the runtime-created ones that go.
    ///
    /// This is the failure [`SharedSubstrate`] and [`writer_lock::WriterLock`]
    /// exist to make unrepresentable — asserted in the shape the daemon actually
    /// failed in: the second writable open is now refused, and the character
    /// created through the one writer survives that writer's compaction.
    #[test]
    fn a_second_writable_handle_is_refused() {
        let dir = tmp_dir("two_owners");

        // A character that predates both opens — the control.
        {
            let mut registry = SubstratePersistence::open_in(&dir).unwrap();
            registry.write_npc(&test_npc(1, "Perrin")).unwrap();
            registry.commit().unwrap();
        }

        // Boot, in the daemon's old order: the registry opens, then the engine
        // tries to open the same directory.
        let mut registry_substrate = Substrate::new();
        let mut registry =
            SubstratePersistence::open_in_with_substrate(&dir, &mut registry_substrate).unwrap();
        let mut engine_substrate = Substrate::new();
        assert!(matches!(
            SubstratePersistence::open_in_with_substrate(&dir, &mut engine_substrate),
            Err(PersistenceError::Locked { .. })
        ));
        // A read-only handle — a tool reading the workspace — still opens.
        let mut reader_substrate = Substrate::new();
        assert!(SubstratePersistence::open_in_with_substrate_read_only(
            &dir,
            &mut reader_substrate
        )
        .is_ok());

        // Runtime: a character is created through the one writer, which then
        // runs a routine compaction.
        registry.write_npc(&test_npc(2, "Wyneth")).unwrap();
        registry.commit().unwrap();
        registry.compact(&mut registry_substrate, None).unwrap();
        drop(registry);

        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "with one writer, the character created at runtime survives its compaction"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The fix: one handle, shared, and the compaction carries the whole cast.
    ///
    /// Identical to the test above except that the engine *adopts* the
    /// registry's [`SharedSubstrate`] instead of opening the directory again.
    /// One append cursor, one `npc_locs` — so the character created at runtime
    /// is in the view the compactor reads.
    #[test]
    fn one_shared_handle_carries_a_runtime_created_record_through_compaction() {
        let dir = tmp_dir("one_owner");

        {
            let mut registry = SubstratePersistence::open_in(&dir).unwrap();
            registry.write_npc(&test_npc(1, "Perrin")).unwrap();
            registry.commit().unwrap();
        }

        // Boot: the registry opens once, and the engine takes that same handle.
        let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
        let engine = shared.clone();

        // Runtime: a character is created, through the registry's side.
        shared.persistence.call(|p| {
            p.write_npc(&test_npc(2, "Wyneth")).unwrap();
            p.commit().unwrap();
        });

        // The engine compacts — the same owner, the same view.
        let substrate = &engine.substrate;
        engine
            .persistence
            .call(|p| p.compact_shared(substrate, None))
            .unwrap();
        drop(shared);
        drop(engine);

        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "a shared handle compacts against a view that includes every \
             character, whenever it was created"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A custom object is durable, last-writer-wins, and survives compaction.**
    ///
    /// The whole point of the type: `npcd` writes the command-table flag once and
    /// it comes back after a reboot, keeps the newest value when re-written, and
    /// is not lost when the log is rewritten (the silent-drop failure mode
    /// `survival.rs` documents). Exercises the `SharedSubstrate` write/read pair
    /// across an open → re-write → compact → reopen cycle.
    #[test]
    fn a_custom_object_is_durable_lww_and_survives_compaction() {
        let dir = tmp_dir("custom_object");

        // Write once, read it straight back from the same handle.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let mut obj =
                CustomObjectPayload::new("command-table", serde_json::json!({ "open": true }));
            obj.metadata
                .insert("world".to_string(), "vault".to_string());
            shared.put_custom_object(obj).unwrap();

            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(back.blob, serde_json::json!({ "open": true }));
            assert_eq!(
                back.metadata.get("world").map(String::as_str),
                Some("vault")
            );
            assert!(
                shared.custom_object("never-written").is_none(),
                "an unknown key has no object"
            );
        }

        // A fresh open replays the log — the object is durable.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(back.blob, serde_json::json!({ "open": true }));

            // Re-write the same key: last-writer-wins.
            shared
                .put_custom_object(CustomObjectPayload::new(
                    "command-table",
                    serde_json::json!({ "open": false }),
                ))
                .unwrap();
            assert_eq!(
                shared.custom_object("command-table").unwrap().blob,
                serde_json::json!({ "open": false }),
                "the newest write wins in RAM"
            );

            // Compact against this view, then confirm the winner survived.
            let substrate = &shared.substrate;
            shared
                .persistence
                .call(|p| p.compact_shared(substrate, None))
                .unwrap();
        }

        // Reopen after compaction: only the last value is present, and the
        // superseded first write did not resurrect.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(
                back.blob,
                serde_json::json!({ "open": false }),
                "the last-written value survives compaction and reload"
            );
            assert!(
                back.metadata.is_empty(),
                "the second write carried no metadata, so the winner has none"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
