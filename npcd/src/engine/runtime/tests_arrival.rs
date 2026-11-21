//! Tests for where a character enters its world and where its body is put back.
//!
//! A character is grounded and set to a working pace the moment it is embodied. On
//! a restart its body returns to the room it was last in, and it arrives at the
//! door when that room is no longer on the map. A character is told the building
//! it works in, and a reflection opens on the room it is in.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{embody, run, vaulted, window, WORLD};

    #[tokio::test]
    async fn embodying_a_character_grounds_it_and_quickens_it() {
        // Both, at once. A character that can act before it has been told where
        // it is would act blind; one left at an ambient pace would think about
        // its work every two minutes.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");

        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::WORKING));
        run(&rt, 1);
        let read = window(&rt, 1);
        assert!(read.iter().any(|t| t.contains("band one")), "{read:?}");
    }

    #[tokio::test]
    async fn what_is_within_reach_arrives_with_where_the_body_is() {
        // The two are one fact — both are functions of where it stands — so
        // they arrive together and go stale together.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        run(&rt, 1);

        let here = window(&rt, 1)
            .into_iter()
            .find(|t| t.starts_with("You are"))
            .expect("grounded");
        assert!(here.contains("Within reach"), "{here}");
        assert!(here.contains("terminal"), "{here}");
    }

    #[tokio::test]
    async fn a_corridor_says_nothing_about_what_is_within_reach() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "ring-north");
        run(&rt, 1);

        let here = window(&rt, 1)
            .into_iter()
            .find(|t| t.starts_with("You are"))
            .expect("grounded");
        assert!(!here.contains("Within reach"), "{here}");
    }

    /// **A character is told about the building it is in, and a reflection
    /// opens on the room it is in.** Both failed together: every character in
    /// the world was handed the whole world as "the building you work in" —
    /// six vault levels and not one room of the Redoubt — and a reflection
    /// opened on the persona's situation, which is empty for every authored
    /// character. Asked where it was, a character in the Redoubt named a vault
    /// room; asked to reflect, it dreamed of an office.
    #[tokio::test]
    async fn a_character_knows_the_building_it_stands_in_and_the_room_it_is_in() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with(|w| {
            w.enter("m2", "Maker-02", Where::new("tower-redoubt", "muster-hall"))
                .unwrap()
        });
        rt.scheduler.wake(2, 0, 0);
        rt.embody(2, WORLD, "m2", 0).expect("bound");

        let (vault_key, vault) = rt.building_of(&hosted, "m1").expect("in the vault");
        let (redoubt_key, redoubt) = rt.building_of(&hosted, "m2").expect("in the tower");
        assert_eq!(vault_key, identity::building_key(WORLD, "creators-vault"));
        assert_eq!(redoubt_key, identity::building_key(WORLD, "tower-redoubt"));
        assert!(redoubt.to_lowercase().contains("muster hall"), "{redoubt}");
        assert!(!vault.to_lowercase().contains("muster hall"), "{vault}");

        let here = rt.situation_of(2).expect("a body is somewhere");
        assert!(here.contains("muster hall"), "{here}");
        assert!(rt.situation_of(99).is_none(), "nobody is nowhere");
    }

    // ── a new character, put into its world ─────────────────────────────────

    #[tokio::test]
    async fn a_new_character_arrives_at_the_way_in_and_starts_thinking() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);

        assert!(rt
            .embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(body, Runtime::body_id(1));
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            npc_map::world::Where::new("vault-command", "command-room"),
            "it did not arrive at the way in"
        );
        // And at a pace that keeps it working rather than settling.
        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::WORKING));
    }

    #[tokio::test]
    async fn a_character_whose_world_has_no_map_is_left_without_a_body() {
        // Most worlds have none. Lore and no body is a character, not a
        // failure, and the caller has to be able to tell that from an error.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert!(!rt
            .embody_in_world(1, "unmapped", Kind::default(), "Maker-01", None, 0)
            .unwrap());
        assert!(rt.body_of(1).is_none());
    }

    #[tokio::test]
    async fn putting_a_character_back_in_the_body_it_had_changes_nothing() {
        // The restart path and the create path are the same call, so it has to
        // be safe to make twice — once when the character was created, once
        // every boot after.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        rt.embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap();

        // It walks off the command level's arrival room — one stop, same level.
        let body = Runtime::body_id(1);
        let elsewhere = npc_map::world::Where::new("vault-command", "anteroom");
        let world = rt.hosted.get(WORLD).unwrap();
        world.with(|w| w.set_off(&body, elsewhere.clone()).unwrap());
        world.tick();
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            elsewhere
        );

        assert!(rt
            .embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap());
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            elsewhere,
            "it was sent back to the door"
        );
    }

    /// **A restart puts a body back where it was, not at the front door.**
    ///
    /// The world's own state — who is standing where — is held in RAM and goes
    /// with the process, so the character record's remembered room is the only
    /// thing that survives to rebuild it from.
    #[tokio::test]
    async fn a_body_returns_to_the_room_it_was_remembered_in() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let green = "vault-casting/green-room";

        assert!(rt
            .embody_in_world(1, WORLD, Kind::default(), "Maker-01", Some(green), 0)
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            Where::new("vault-casting", "green-room"),
            "it was sent to the arrival door instead of where it was"
        );
    }

    /// A remembered room the map no longer has is not an error — maps are
    /// authored and rooms get renamed. The character arrives at the door, which
    /// is what somebody whose room was demolished should do.
    #[tokio::test]
    async fn a_body_whose_remembered_room_is_gone_arrives_at_the_door() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);

        assert!(rt
            .embody_in_world(
                1,
                WORLD,
                Kind::default(),
                "Maker-01",
                Some("vault-casting/no-such-room"),
                0
            )
            .unwrap());
        let (world, body) = rt.body_of(1).expect("it has a body");
        assert_eq!(
            world.read(|w| w.actor(&body).unwrap().at.clone()),
            Where::new("vault-command", "command-room"),
            "a room the map does not have should fall through to the way in"
        );
    }
}
