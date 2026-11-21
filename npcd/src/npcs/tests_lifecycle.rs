//! Checks what must survive a restart: characters, owners, moods, and the
//! newest record of each character.
//!
//! The registry exists so a character outlives the process, so each test
//! writes through one registry and reads through a second over the same
//! directory. The covered cases are reassignment, creation at runtime followed
//! by the engine's compaction, moods, record ordering on reload, edits,
//! deletion and renaming a world.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use std::path::PathBuf;

    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// Two accounts, in the shape `accounts::with_public_id` mints: a `u_`
    /// prefix over the account file's key. Not a number — see `NpcPayload`.
    pub(in super::super) const ME: &str = "u_1a2b3c4d";

    pub(in super::super) const OTHER: &str = "u_99887766";

    pub(in super::super) fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-npcs-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    pub(in super::super) fn ident(sub: &str) -> Identity {
        Identity {
            provider: "google".to_string(),
            sub: sub.to_string(),
            email: String::new(),
            name: String::new(),
            picture: String::new(),
            exp: 0,
        }
    }

    pub(in super::super) fn body(name: &str) -> Value {
        json!({ "name": name, "world_id": "battle-cities", "personality_id": "commander" })
    }

    /// A wire record's `npc_id`, decoded back to the `u64` the tests key on.
    pub(in super::super) fn nid(v: &Value) -> u64 {
        npc_id_of_wire(v["npc_id"].as_str().unwrap()).unwrap()
    }

    /// The property the whole module exists for: a character outlives the
    /// process. Written by one registry, read back by another over the same
    /// directory, with no engine in between.
    #[test]
    fn a_character_survives_a_restart() {
        let dir = tmp();
        let created = {
            let n = Npcs::load(&dir).unwrap();
            n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap()
        };
        let npc_id: u64 = nid(&created);

        // A second registry over the same log — a restart, in effect.
        let reopened = Npcs::load(&dir).unwrap().view();
        let back = reopened.get(npc_id, ME).unwrap();
        assert_eq!(back["name"], "Varek");
        assert_eq!(back["owner_id"], ME);
        assert_eq!(back["revision"], 1);
        assert_eq!(reopened.list(ME, &Filter::default()).len(), 1);
    }

    /// **A character handed to another account is theirs, and stays theirs.**
    /// The new owner reads and lists it, the old one no longer does, the owners
    /// listing says so, and a restart keeps it.
    #[test]
    fn a_reassigned_character_belongs_to_its_new_owner() {
        let dir = tmp();
        let npc_id = {
            let n = Npcs::load(&dir).unwrap();
            let npc_id = nid(&n.create(&ident("u1"), OTHER, &body("Ilsa"), 1_000).unwrap());
            let handed = n.reassign(npc_id, ME, 2_000).unwrap();
            assert_eq!(handed["owner_id"], ME);
            assert_eq!(handed["access"], "owner");
            assert_eq!(handed["revision"], 2);
            assert_eq!(n.view().owners()[0]["owner_id"], ME);
            assert_eq!(n.view().owners()[0]["name"], "Ilsa");
            assert_eq!(
                n.reassign(npc_id, ME, 3_000).unwrap()["revision"],
                2,
                "handing it to its owner writes nothing"
            );
            assert!(matches!(
                n.reassign(npc_id + 1, ME, 3_000),
                Err(NpcError::NotFound)
            ));
            npc_id
        };
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(reopened.get(npc_id, ME).unwrap()["name"], "Ilsa");
        assert!(reopened.get(npc_id, OTHER).is_err());
        assert_eq!(reopened.list(OTHER, &Filter::default()).len(), 0);
    }

    /// **A character created while the daemon runs survives the engine's
    /// maintenance.**
    ///
    /// The regression guard for the bug that lost Makers overnight. The engine
    /// used to open `--data/substrate/` for itself, so it held a second
    /// writable handle whose record index had been built before the character
    /// existed — and its next compaction carried forward only what that index
    /// knew, dropping every character created since. Characters present at boot
    /// survived, so it read as an intermittent fault rather than a mechanical
    /// one.
    ///
    /// Here the engine takes the registry's handle, which is the whole fix: the
    /// compaction runs against the same view the character was written into.
    #[test]
    fn a_character_created_at_runtime_survives_an_engine_compaction() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            // Whatever the engine does to this log, it does through this handle
            // — the one the registry is writing through.
            let engine = n.substrate();

            let created = n.create(&ident("u1"), ME, &body("Wyneth"), 1_000).unwrap();

            // The engine's routine compaction, on the handle it was handed.
            let substrate = &engine.substrate;
            engine
                .persistence
                .call(|p| p.compact_shared(substrate, None))
                .unwrap();
            nid(&created)
        };

        // A restart.
        let reopened = Npcs::load(&dir).unwrap().view();
        assert_eq!(
            reopened.get(npc_id, ME).unwrap()["name"],
            "Wyneth",
            "a character created at runtime must outlive a compaction"
        );
    }

    /// **How a character feels survives a restart.**
    ///
    /// A mood is what a character *is* between one thought and the next, and it
    /// otherwise lives only in the window — which a restart empties. Without
    /// this, one that had spent the afternoon getting angrier came back in no
    /// register at all while its own transcript said otherwise, which is the
    /// same wrong as returning it to the arrival door.
    #[test]
    fn how_a_character_feels_survives_a_restart() {
        let dir = tmp();
        let id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let id: u64 = nid(&n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap());
            // Absent, not calm. Nobody has asked it yet, and those are
            // different facts about a character.
            assert_eq!(n.view().mood_of(id), None);

            assert!(n.remember_mood(id, "wary"));
            assert_eq!(n.view().mood_of(id), Some("wary"));
            // Feeling the same thing again costs nothing, however often it is
            // said — which is the whole of the write gate, because a register
            // is named at most once a pause and is usually last time's word.
            assert!(!n.remember_mood(id, "wary"));
            assert!(n.remember_mood(id, "cornered"));
            id
        };

        let back = Npcs::load(&dir).unwrap().view();
        assert_eq!(
            back.mood_of(id),
            Some("cornered"),
            "the register did not survive the restart"
        );
        // And it rides along with the cast, which is what the loader reads to
        // put the character back into it.
        let casting = back.cast().into_iter().find(|c| c.npc_id == id).unwrap();
        assert_eq!(casting.mood.as_deref(), Some("cornered"));
    }

    /// **The cast loads each character's newest record, not replay's last.**
    /// Another member's older record replicated in after this member's newer
    /// one: the log holds them in that order, and a reload must hold the
    /// newer — the one a refresh compares against, so nothing would ever read
    /// it again.
    #[test]
    fn a_reload_holds_the_newest_record_whatever_order_the_log_holds() {
        let (dir_a, dir_b) = (tmp(), tmp());
        let id = {
            let a = Npcs::load(&dir_a).unwrap();
            let b = Npcs::load(&dir_b).unwrap();
            let id = nid(&a.create(&ident("u1"), ME, &body("Umbra"), 1_000).unwrap());
            let older = a.view().payload(id).unwrap().clone();

            std::thread::sleep(std::time::Duration::from_millis(5));
            let mut newer = older.clone();
            newer.state = SUSPENDED.to_string();
            b.shared.persistence.call(|p| {
                p.write_npc(&newer).unwrap();
                p.commit().unwrap();
            });

            // A's records reach B after B wrote its own.
            let bytes: Vec<Vec<u8>> = a.shared.persistence.call(|p| {
                let node = p.node();
                (1..)
                    .map_while(|q| p.read_event(node, q).unwrap())
                    .collect()
            });
            let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
            let checks = SubstratePersistence::check_remote(&refs);
            let lock = Arc::clone(&b.shared.substrate);
            b.shared.persistence.call(|p| {
                p.apply_remote_shared(&lock, &refs, &checks, 0).unwrap();
                p.commit().unwrap();
            });
            id
        };

        let b = Npcs::load(&dir_b).unwrap();
        assert_eq!(b.view().payload(id).unwrap().state, SUSPENDED);
        assert!(b.refresh().is_empty(), "the cast agrees with the log");
    }

    /// **A write does not put an older record back over a newer one.** A
    /// record replicated into the log after the cast was last refreshed is
    /// what the next write is computed from: a mood named from the stale copy
    /// would otherwise land as the character's whole record, its suspension
    /// gone, on every member.
    #[test]
    fn a_write_does_not_put_an_older_record_back_over_a_newer_one() {
        let n = Npcs::load(&tmp()).unwrap();
        let id = nid(&n.create(&ident("u1"), ME, &body("Umbra"), 1_000).unwrap());

        // Another member's suspension reaches this member's log, and the cast
        // has not been refreshed since.
        let mut theirs = n.view().payload(id).unwrap().clone();
        theirs.state = SUSPENDED.to_string();
        n.shared.persistence.call(|p| {
            p.write_npc(&theirs).unwrap();
            p.commit().unwrap();
        });
        assert_eq!(n.view().payload(id).unwrap().state, "idle", "not refreshed");

        // The character names a mood from here: the write is computed from the
        // log, so the suspension stands under the mood.
        assert!(n.remember_mood(id, "wary"));
        let view = n.view();
        let now = view.payload(id).unwrap();
        assert_eq!(now.state, SUSPENDED);
        assert_eq!(now.mood.as_deref(), Some("wary"));
        drop(view);
        let logged = n
            .shared
            .persistence
            .call(|p| p.read_npc(id).unwrap().unwrap().state);
        assert_eq!(logged, SUSPENDED, "the log holds the suspension too");

        // The record it took in is reported to the next refresh, once.
        assert_eq!(n.refresh(), vec![id]);
        assert!(n.refresh().is_empty());
    }

    /// Feeling something is not an authoring act: a revision bump and an
    /// `updated_ms` describe what somebody *edited*, and a roster that sorted by
    /// "recently edited" would report whoever happened to be feeling something.
    #[test]
    fn feeling_something_is_not_an_edit() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let made = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&made);
        let before = n.view().payload(id).unwrap().clone();

        assert!(n.remember_mood(id, "bitter"));
        let view = n.view();
        let after = view.payload(id).unwrap();
        assert_eq!(
            after.revision, before.revision,
            "a mood bumped the revision"
        );
        assert_eq!(
            after.updated_ms, before.updated_ms,
            "a mood counted as an edit"
        );
    }

    /// An edit supersedes rather than accumulating: the reopened registry sees
    /// exactly one character, at the newest revision.
    #[test]
    fn an_edit_supersedes_the_previous_record() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = nid(&c);
            n.patch(id, ME, &json!({ "name": "Varek the Elder" }), 2_000)
                .unwrap();
            n.patch(id, ME, &json!({ "state": "active" }), 3_000)
                .unwrap();
            id
        };

        let reopened = Npcs::load(&dir).unwrap().view();
        let list = reopened.list(ME, &Filter::default());
        assert_eq!(list.len(), 1, "three records, one character");
        assert_eq!(list[0]["name"], "Varek the Elder");
        assert_eq!(list[0]["state"], "active");
        assert_eq!(list[0]["revision"], 3);
        assert_eq!(reopened.get(npc_id, ME).unwrap()["updated_ms"], 3_000);
    }

    /// Deleting writes a tombstoned record. It disappears from every read path
    /// but the id stays taken — the acts it committed still name it.
    #[test]
    fn deleting_hides_the_character_but_keeps_its_id() {
        let dir = tmp();
        let npc_id: u64 = {
            let n = Npcs::load(&dir).unwrap();
            let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
            let id: u64 = nid(&c);
            n.delete(id, ME, 2_000).unwrap();
            assert!(n.view().list(ME, &Filter::default()).is_empty());
            assert!(matches!(n.view().get(id, ME), Err(NpcError::NotFound)));
            id
        };

        let reopened = Npcs::load(&dir).unwrap().view();
        assert!(reopened.list(ME, &Filter::default()).is_empty());
        assert!(
            reopened.by_id.contains_key(&npc_id),
            "the record survives so the id cannot be reused"
        );
    }

    /// Ownership is authorization, and a stranger's character reads as absent
    /// rather than forbidden — a 403 would confirm the id exists.
    #[test]
    fn another_accounts_character_is_invisible_not_forbidden() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        let c = n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let id: u64 = nid(&c);

        assert!(matches!(n.view().get(id, OTHER), Err(NpcError::NotFound)));
        assert!(n.view().list(OTHER, &Filter::default()).is_empty());
        assert!(matches!(
            n.patch(id, OTHER, &json!({ "name": "Stolen" }), 2_000),
            Err(NpcError::NotFound)
        ));
        assert!(matches!(
            n.delete(id, OTHER, 2_000),
            Err(NpcError::NotFound)
        ));
        // And the original is untouched by the attempts.
        assert_eq!(n.view().get(id, ME).unwrap()["name"], "Varek");
    }

    /// Hidden characters stay out of the default listing (§8.3).
    #[test]
    fn hidden_characters_are_opt_in() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Seen"), 1_000).unwrap();
        let mut hidden_body = body("Unseen");
        hidden_body["hidden"] = json!(true);
        n.create(&ident("u1"), ME, &hidden_body, 1_001).unwrap();

        let n = n.view();
        assert_eq!(n.list(ME, &Filter::default()).len(), 1);
        let all = n.list(
            ME,
            &Filter {
                include_hidden: true,
                ..Default::default()
            },
        );
        assert_eq!(all.len(), 2);
    }

    /// **A renamed world takes its characters with it**: each of the old id
    /// moves to the new, keeping its name and revision line; another world's
    /// stay; a second move finds none.
    #[test]
    fn a_renamed_world_takes_its_characters_with_it() {
        let dir = tmp();
        let n = Npcs::load(&dir).unwrap();
        n.create(&ident("u1"), ME, &body("Varek"), 1_000).unwrap();
        let ilse = json!({ "name": "Ilse", "world_id": "earth", "personality_id": "commander" });
        n.create(&ident("u1"), ME, &ilse, 1_001).unwrap();
        let before = n.view().export_all();
        let varek = before.iter().find(|p| p.name == "Varek").unwrap().clone();

        assert_eq!(n.rehome(&varek.world_id, "moon", 2_000).unwrap(), 1);
        let after = n.view();
        let moved = after.payload(varek.npc_id).unwrap();
        assert_eq!(moved.world_id, "moon");
        assert_eq!(moved.name, "Varek");
        assert_eq!(moved.revision, varek.revision + 1);
        let ilse = after
            .export_all()
            .into_iter()
            .find(|p| p.name == "Ilse")
            .unwrap();
        assert_eq!(ilse.world_id, "earth", "another world's character stays");
        assert_eq!(n.rehome(&varek.world_id, "moon", 3_000).unwrap(), 0);
    }
}
