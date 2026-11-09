//! What a hosted world *is*, as entity records (`docs/replicated_substrate.md`
//! §9.1), named relative to its prefix:
//!
//! | Entity | Record |
//! |---|---|
//! | each body | `body/<body>`: its place, the station it holds, and the journey it is on |
//! | the lift car | `lift`: its floor, doors and stops, who is aboard and who is waiting for it, each with the floor it is bound for |
//! | each entry of the world's entity maps — a mission, an operation, a body's working set, a pack, a device, a thread, a posting… | `sim/<map path>/<key>` ([`SIM_MAPS`]) |
//! | the tower's clock, and the contact closing on it — each changing every beat | `sim/tower/clock`, `sim/tower/contact` ([`SIM_FACETS`]) |
//! | everything else the world holds — the command table, the rest of the tower, the roster, the ledger's lists | `sim/meta` |
//!
//! A body on its way between rooms is written at the end of each leg with the
//! rest of its way: a member that runs its character without running the
//! world builds the turn it is given there from its own copy, and a copy that
//! had it still standing where it set out offered it the rooms it had left.
//! What the records hold is [`World::state`], which names every field of the
//! world, so nothing a copy builds a turn from is left out of them. Where this
//! machine keeps the benches' documents is not part of the world, so it is not
//! in the record.
//!
//! [`World::state`]: npc_map::world::World::state

use std::collections::BTreeMap;

use npc_map::lift::Doors;
use npc_map::state::{BodyState, LiftState, State, WalkState};
use npc_map::world::Where;
use serde_json::{json, Value};

use super::keys::escape;
use super::split::split;
use crate::sim::Sim;

/// The prefix of the records the world's [`Sim`] is split into.
pub const SIM: &str = "sim/";

/// The paths in a [`Sim`]'s JSON of the maps that hold one entity per entry:
/// each changes entry by entry, and each entry is its own record.
pub const SIM_MAPS: &[&str] = &[
    "packs",
    "bench/open",
    "bench/stashed",
    "bench/last",
    "devices/devices",
    "field/deposits",
    "field/hostiles",
    "field/stances",
    "ledger/sleepers",
    "ledger/orders_read",
    "missions/active",
    "missions/lodged",
    "missions/done",
    "missions/targets",
    "missions/operations/ops",
    "missions/closed",
    "missions/last_closed",
    "record/items",
    "threads/threads",
    "postings/postings",
];

/// The paths in a [`Sim`]'s JSON of the fields that change every beat while
/// the rest of their object does not: each is a record of its own
/// (`sim/<path>`), so the meta record is not rewritten for them.
pub const SIM_FACETS: &[&str] = &["tower/clock", "tower/contact"];

/// A world's state: every entity record, by its name under the world.
pub type Records = BTreeMap<String, Value>;

pub fn doors_name(d: Doors) -> &'static str {
    match d {
        Doors::Open => "open",
        Doors::Closed => "closed",
    }
}

/// A place as a record names it: `<area>/<node>`.
pub fn place_name(at: &Where) -> String {
    format!("{}/{}", at.area, at.node)
}

/// The records of every body and of the lift — everything the world's copies
/// share — from `state`, the copy [`World::state`] makes under the world lock,
/// so the records are built after it is let go.
///
/// [`World::state`]: npc_map::world::World::state
pub fn world_records(state: State) -> Records {
    let State { bodies, lift } = state;
    let mut out = Records::new();
    for (id, body) in bodies {
        let BodyState { at, hold, walk } = body;
        out.insert(
            format!("body/{}", escape(&id)),
            json!({
                "at": place_name(&at),
                "hold": hold.map(|(part, subject)| json!({
                    "part": part,
                    "subject": subject,
                })),
                "walk": walk.map(|WalkState { toward, ahead }| json!({
                    "toward": place_name(&toward),
                    "ahead": ahead.iter().map(place_name).collect::<Vec<_>>(),
                })),
            }),
        );
    }
    if let Some(LiftState {
        floor,
        doors,
        stops,
        riders,
        waiting,
    }) = lift
    {
        out.insert(
            "lift".into(),
            json!({
                "floor": floor,
                "doors": doors_name(doors),
                "stops": stops,
                "riders": riders,
                "waiting": waiting,
            }),
        );
    }
    out
}

/// The records of everything the world holds beyond its map:
/// [`sim_records_of`] [`sim_value`].
pub fn sim_records(s: &Sim) -> Records {
    sim_value(s).map(sim_records_of).unwrap_or_default()
}

/// Everything `s` holds, as the JSON its records are split from — the one
/// part of taking a world's state that needs its lock. The benches' root is
/// this machine's, so it is left out. `None`, and the reason logged, for a
/// state that would not encode.
pub fn sim_value(s: &Sim) -> Option<Value> {
    let mut value = match serde_json::to_value(s) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("a world's state would not encode: {e}");
            return None;
        }
    };
    if let Some(bench) = value.get_mut("bench").and_then(Value::as_object_mut) {
        bench.remove("root");
    }
    Some(value)
}

/// [`sim_value`]'s JSON split into one record per entity, outside the lock.
pub fn sim_records_of(value: Value) -> Records {
    split(value, SIM_MAPS, SIM_FACETS)
        .into_iter()
        .map(|(name, v)| (format!("{SIM}{name}"), v))
        .collect()
}

#[cfg(test)]
pub(super) mod tests {
    use std::time::Duration;

    use npc_map::world::{Where, World};
    use npc_map::MapSet;

    use super::super::apply::sim_of;
    use super::super::diff::{diff, Changes};
    use super::*;
    use crate::engine::watch::Watch;
    use crate::sim::seed;
    use crate::world::Hosted;

    pub fn vault() -> Hosted {
        Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load")
    }

    pub fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// A body standing is its place; one on its way is where it has got to
    /// and the rest of its way.
    #[test]
    fn a_walking_body_s_record_holds_the_rest_of_its_way() {
        let records = vault().with(|w| {
            w.enter("a", "A", at("band-one")).unwrap();
            w.enter("b", "B", at("band-one")).unwrap();
            w.set_off("b", at("green-room")).unwrap();
            world_records(w.state())
        });
        assert_eq!(
            records.get("body/a"),
            Some(&json!({"at": "vault-casting/band-one", "hold": null, "walk": null}))
        );
        assert_eq!(
            records.get("body/b"),
            Some(&json!({
                "at": "vault-casting/band-one",
                "hold": null,
                "walk": {
                    "toward": "vault-casting/green-room",
                    "ahead": [
                        "vault-casting/ring-north",
                        "vault-casting/ring-east",
                        "vault-casting/ring-south",
                        "vault-casting/green-room",
                    ],
                },
            }))
        );
    }

    /// The command table is in the world's meta record; each thread, each
    /// handset, each mission is a record of its own; the benches' root is not
    /// anywhere.
    #[test]
    fn the_world_s_state_is_one_record_per_entity() {
        let mut s = seed::for_world("w", None);
        s.table_open = false;
        s.set_bench_root(crate::mind::Mind::new(Some(
            "/somewhere/on/this/machine".into(),
        )));
        seed::issue_handset(&mut s, "m1", "Maker-01");
        let records = sim_records(&s);
        let meta = records.get("sim/meta").expect("a meta record");
        assert_eq!(meta["table_open"], json!(false));
        assert!(
            meta["bench"].get("root").is_none(),
            "the root is this machine's"
        );
        assert!(records.contains_key("sim/packs/m1"), "{:?}", records.keys());
        assert!(records.keys().all(|k| k.starts_with(SIM)));
    }

    fn shipped() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load")
    }

    /// **The tower's clock and contact are records of their own**: the
    /// battle-cities seed splits into its meta, holding the tower with
    /// neither, the two facets, and its map records — and joins back into
    /// the seed. The vault, with no tower, has no tower records.
    #[test]
    fn the_tower_s_clock_and_contact_are_records_of_their_own() {
        let map = shipped();
        let seed = seed::battle_cities(Some(&map));
        let records = sim_records(&seed);
        let tower = &records["sim/meta"]["tower"];
        assert!(tower.is_object(), "{tower}");
        assert!(tower.get("clock").is_none() && tower.get("contact").is_none());
        assert_eq!(
            records.get("sim/tower/clock"),
            Some(&json!({"carry": 0, "tap_carry": 0, "quiet_ms": 0, "sightings": 0}))
        );
        assert_eq!(records.get("sim/tower/contact"), Some(&Value::Null));
        let tower_records: Vec<&str> = records
            .keys()
            .filter(|k| k.starts_with("sim/tower/"))
            .map(String::as_str)
            .collect();
        assert_eq!(tower_records, ["sim/tower/clock", "sim/tower/contact"]);
        let back = sim_of(&records).expect("the seed decodes");
        assert_eq!(sim_records(&back), records);

        let vault = sim_records(&seed::vault(Some(&map)));
        assert!(vault.keys().all(|k| !k.starts_with("sim/tower/")));
    }

    /// **A meta that still holds the tower's clock, with no clock record,
    /// decodes with that clock** — what a build writing the tower whole
    /// leaves behind.
    #[test]
    fn a_meta_holding_the_clock_decodes_with_it() {
        let map = shipped();
        let mut records = sim_records(&seed::battle_cities(Some(&map)));
        records.remove("sim/tower/clock");
        records.remove("sim/tower/contact");
        let clock = json!({"carry": 7, "tap_carry": 1, "quiet_ms": 30, "sightings": 2});
        records.get_mut("sim/meta").unwrap()["tower"]["clock"] = clock.clone();
        let decoded = sim_of(&records).expect("decodes");
        assert_eq!(sim_records(&decoded)["sim/tower/clock"], clock);
    }

    /// **A beat of the tower changes only its clock's record**: with a body
    /// in the foundry and no contact, half a second puts exactly
    /// `sim/tower/clock`, and nothing in meta churns.
    #[test]
    fn a_tower_beat_writes_only_its_clock() {
        let map = shipped();
        let mut sim = seed::battle_cities(Some(&map));
        let mut w = World::new(map);
        w.enter("c1", "Vael", Where::new("tower-redoubt", "foundry"))
            .unwrap();
        let before = sim_records(&sim);
        Watch::new().watch_at(&mut w, &mut sim, Duration::from_millis(500));
        assert_eq!(
            diff(&before, &sim_records(&sim)),
            Changes {
                put: vec![(
                    "sim/tower/clock".into(),
                    json!({"carry": 10000, "tap_carry": 0, "quiet_ms": 500, "sightings": 0})
                )],
                ceased: vec![],
            }
        );
    }
}
