//! The one-time conversion of a world kept before world state records
//! (`docs/replicated_substrate.md` §15.1, step 3): the three whole-world
//! custom objects a daemon checkpointed — `missions/<world>`,
//! `benches/<world>` and `command-table` — are taken into the world, which then
//! writes them as its own records, and the old objects are tombstoned.
//!
//! Read only for a world with no records of its own, so once a world has
//! written its records nothing here is consulted again.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::{Result as PersistenceResult, SharedSubstrate};

use crate::sim::bench::Benches;
use crate::sim::missions::Missions;
use crate::world::Hosted;

/// The one key the command table's open flag was kept under, for every world.
pub const TABLE_OPEN: &str = "command-table";

pub fn missions_key(world: &str) -> String {
    format!("missions/{world}")
}

pub fn benches_key(world: &str) -> String {
    format!("benches/{world}")
}

/// Take what was kept for `hosted` before world state records into it; whether
/// there was anything.
pub fn take_in(shared: &SharedSubstrate, hosted: &Hosted) -> bool {
    let id = hosted.id();
    let open = shared
        .custom_object(TABLE_OPEN)
        .and_then(|o| o.blob.get("open").and_then(|v| v.as_bool()));
    let missions = shared
        .custom_object(&missions_key(id))
        .and_then(|o| serde_json::from_value::<Missions>(o.blob).ok());
    let benches = shared
        .custom_object(&benches_key(id))
        .and_then(|o| serde_json::from_value::<Benches>(o.blob).ok());
    if open.is_none() && missions.is_none() && benches.is_none() {
        return false;
    }
    hosted.with_sim(|s| {
        if let Some(open) = open {
            s.table_open = open;
        }
        if let Some(mut m) = missions {
            m.adopt_workflows(&mut s.missions);
            s.missions = m;
        }
        if let Some(b) = benches {
            s.bench.restore(b);
        }
    });
    true
}

/// Tombstone what was kept for `world`, once its records are written. The
/// command table's flag was one key for every world, so it goes with the last
/// of them, `last`.
pub fn retire(shared: &SharedSubstrate, world: &str, last: bool) -> PersistenceResult<()> {
    let mut keys = vec![missions_key(world), benches_key(world)];
    if last {
        keys.push(TABLE_OPEN.to_string());
    }
    for key in keys {
        if shared.custom_object(&key).is_some() {
            shared.put_custom_object(CustomObjectPayload::ceased(key))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::SubstratePersistence;
    use candle_conversation::substrate::Substrate;
    use serde_json::json;
    use tempfile::TempDir;

    use super::super::snapshot::tests::vault;
    use super::*;
    use crate::engine::mission::{Mission, Origin};

    fn shared(dir: &TempDir) -> SharedSubstrate {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        SharedSubstrate::new(substrate, persistence)
    }

    /// A world kept the old way comes up as it was, and once retired its old
    /// objects are gone; a world with nothing kept is left as seeded.
    #[test]
    fn a_world_kept_the_old_way_is_taken_in_once() {
        let dir = TempDir::new().unwrap();
        let s = shared(&dir);
        let h = vault();
        assert!(!take_in(&s, &h), "nothing kept");

        let mut missions = Missions::default();
        missions.lodge(
            "m1",
            Mission::new(
                "fetch the ledger",
                vec![],
                Origin::Lodged { by: "op".into() },
            ),
        );
        s.put_custom_object(CustomObjectPayload::new(
            missions_key("creators-vault"),
            serde_json::to_value(&missions).unwrap(),
        ))
        .unwrap();
        s.put_custom_object(CustomObjectPayload::new(TABLE_OPEN, json!({"open": false})))
            .unwrap();
        assert!(take_in(&s, &h));
        assert!(!h.sim(|s| s.table_open));
        assert_eq!(h.sim(|s| s.missions.lodged_count("m1")), 1);

        retire(&s, "creators-vault", false).unwrap();
        assert!(s.custom_object(&missions_key("creators-vault")).is_none());
        assert!(s.custom_object(TABLE_OPEN).is_some(), "not the last world");
        retire(&s, "creators-vault", true).unwrap();
        assert!(s.custom_object(TABLE_OPEN).is_none());
    }
}
