//! What a hosted world *is*, as entity records (`docs/replicated_substrate.md`
//! §9.1), named relative to its prefix:
//!
//! | Entity | Record |
//! |---|---|
//! | each body standing still | `body/<body>`: its place and the station it holds |
//! | the lift car | `lift`: its floor, doors and stops |
//! | each entry of the world's entity maps — a mission, an operation, a body's working set, a pack, a device, a thread, a posting… | `sim/<map path>/<key>` ([`SIM_MAPS`]) |
//! | everything else the world holds — the command table, the tower, the roster, the ledger's lists | `sim/meta` |
//!
//! A body on its way between rooms has no record written: its walk is what the
//! world is doing, and the steps go as pulses. Its record is written where it
//! stops. Where this machine keeps the benches' documents is not part of the
//! world, so it is not in the record.

use std::collections::BTreeMap;

use npc_map::lift::Doors;
use npc_map::world::World;
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

/// A world's state: every entity record, by its name under the world.
pub type Records = BTreeMap<String, Value>;

pub fn doors_name(d: Doors) -> &'static str {
    match d {
        Doors::Open => "open",
        Doors::Closed => "closed",
    }
}

/// The records of the bodies standing still, and the lift.
pub fn world_records(w: &World) -> Records {
    let mut out = Records::new();
    for actor in w.actors().filter(|a| a.walk.is_none()) {
        out.insert(
            format!("body/{}", escape(&actor.id)),
            json!({
                "at": format!("{}/{}", actor.at.area, actor.at.node),
                "hold": actor.hold.as_ref().map(|h| json!({
                    "part": h.part,
                    "subject": h.subject,
                })),
            }),
        );
    }
    if let Some(lift) = w.lift() {
        out.insert(
            "lift".into(),
            json!({
                "floor": lift.floor(),
                "doors": doors_name(lift.doors()),
                "stops": lift.stops(),
            }),
        );
    }
    out
}

/// The records of everything the world holds beyond its map.
pub fn sim_records(s: &Sim) -> Records {
    let mut value = match serde_json::to_value(s) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("a world's state would not encode: {e}");
            return Records::new();
        }
    };
    if let Some(bench) = value.get_mut("bench").and_then(Value::as_object_mut) {
        bench.remove("root");
    }
    split(&value, SIM_MAPS)
        .into_iter()
        .map(|(name, v)| (format!("{SIM}{name}"), v))
        .collect()
}

#[cfg(test)]
pub(super) mod tests {
    use npc_map::world::Where;

    use super::*;
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

    #[test]
    fn a_standing_body_is_a_record_and_a_walking_one_is_not() {
        let records = vault().with(|w| {
            w.enter("a", "A", at("band-one")).unwrap();
            w.enter("b", "B", at("band-one")).unwrap();
            w.set_off("b", at("green-room")).unwrap();
            world_records(w)
        });
        assert_eq!(
            records.get("body/a"),
            Some(&json!({"at": "vault-casting/band-one", "hold": null}))
        );
        assert!(!records.contains_key("body/b"), "on its way");
    }

    /// The command table is in the world's meta record; each thread, each
    /// handset, each mission is a record of its own; the benches' root is not
    /// anywhere.
    #[test]
    fn the_world_s_state_is_one_record_per_entity() {
        let mut s = seed::for_world("w", None);
        s.table_open = false;
        s.set_bench_root(crate::mind::Mind::new(Some("/somewhere/on/this/machine".into())));
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
}
