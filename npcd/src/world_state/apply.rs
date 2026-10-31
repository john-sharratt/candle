//! A world's state records, put into a hosted world
//! (`docs/replicated_substrate.md` §9.1): what a world is hosted from, what a
//! replica takes in as its authority writes, and what a world reloads from when
//! it leaves disconnected mode.
//!
//! The world's [`Sim`] is taken in whole when the records hold one — what this
//! machine keeps of its own aside ([`Sim::take_in`]) — and left as the world has
//! it when they do not, so a world hosted for the first time keeps its seed. A
//! body record places a body that is in the world; one that is not yet is kept
//! for when it is embodied ([`place_of`]).

use npc_map::lift::Doors;
use npc_map::world::{Where, World};
use serde_json::Value;

use super::keys::unescape;
use super::snapshot::{Records, SIM, SIM_MAPS};
use super::split::{join, META};
use crate::sim::Sim;
use crate::world::Hosted;

/// The records under `prefix`, renamed relative to it.
fn under(records: &Records, prefix: &str) -> Records {
    records
        .iter()
        .filter_map(|(k, v)| Some((k.strip_prefix(prefix)?.to_string(), v.clone())))
        .collect()
}

/// The place a body record holds for `body`, if there is one.
pub fn place_of(records: &Records, body: &str) -> Option<String> {
    records
        .iter()
        .find(|(k, _)| k.strip_prefix("body/").is_some_and(|b| unescape(b) == body))
        .and_then(|(_, v)| v.get("at")?.as_str().map(String::from))
}

/// The world's [`Sim`] the records hold, if they hold one.
pub fn sim_of(records: &Records) -> Option<Sim> {
    let sim = under(records, SIM);
    if !sim.contains_key(META) {
        return None;
    }
    match serde_json::from_value::<Sim>(join(&sim, SIM_MAPS)) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::error!("a world's state records would not decode: {e}");
            None
        }
    }
}

fn apply_world(w: &mut World, records: &Records) {
    for (name, v) in records {
        let Some(body) = name.strip_prefix("body/").map(unescape) else {
            continue;
        };
        if w.actor(&body).is_none() {
            continue;
        }
        let at = v
            .get("at")
            .and_then(Value::as_str)
            .and_then(Where::parse)
            .filter(|at| w.map().node_at(at).is_some());
        if let Some(at) = at {
            if w.actor(&body).is_some_and(|a| a.at != at) {
                let _ = w.restore_place(&body, at);
            }
        }
        let hold = v.get("hold").filter(|h| !h.is_null()).and_then(|h| {
            let part = h.get("part")?.as_str()?.to_string();
            let subject = h.get("subject").and_then(Value::as_str).map(String::from);
            Some((part, subject))
        });
        let held = w
            .actor(&body)
            .and_then(|a| a.hold.as_ref())
            .map(|h| (h.part.clone(), h.subject.clone()));
        if held != hold {
            let _ = w.restore_hold(&body, hold);
        }
    }
    if let Some(lift) = records.get("lift") {
        let floor = lift.get("floor").and_then(Value::as_u64).unwrap_or(0) as usize;
        let doors = match lift.get("doors").and_then(Value::as_str) {
            Some("closed") => Doors::Closed,
            _ => Doors::Open,
        };
        let stops: Vec<usize> = lift
            .get("stops")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|f| f as usize)
                    .collect()
            })
            .unwrap_or_default();
        w.restore_lift(floor, doors, &stops);
    }
}

/// Put `records` into `hosted`, under one hold of its lock — and its bots'
/// DMs (`docs/npcd_discord.md` §4.2), which are not under it.
pub fn apply(hosted: &Hosted, records: &Records) {
    let sim = sim_of(records);
    hosted.with_both(|w, s| {
        apply_world(w, records);
        if let Some(sim) = sim {
            s.take_in(sim);
        }
    });
    hosted.with_dms(|dms| dms.take_in(records));
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::snapshot::tests::{at, vault};
    use super::super::snapshot::{sim_records, world_records};
    use super::*;
    use crate::discord::dm::record::Dm;
    use crate::sim::seed;
    use crate::world_state::records_of;

    /// What one world's snapshot says, another world built from the same map
    /// takes in: the bodies where they stood, the table, the handsets — and it
    /// keeps where its own documents live.
    #[test]
    fn a_snapshot_applied_reproduces_the_world() {
        let source = vault();
        source.with(|w| {
            w.enter("a", "A", at("green-room")).unwrap();
        });
        source.with_sim(|s| {
            s.table_open = false;
            seed::issue_handset(s, "a", "A");
        });
        let mut records = source.read(world_records);
        records.extend(source.sim(sim_records));

        let replica = vault();
        replica.with(|w| w.enter("a", "A", at("band-one")).unwrap());
        let own = crate::mind::Mind::new(Some("/the/replica/s/own".into()));
        replica.with_sim(|s| s.bench.set_mind(own.clone()));
        apply(&replica, &records);
        assert_eq!(replica.place_of("a"), "vault-casting/green-room");
        assert!(!replica.sim(|s| s.table_open));
        assert!(replica.sim(|s| s.has_phone("a")));
        assert_eq!(replica.sim(|s| s.bench.mind().cloned()), Some(own));
        assert_eq!(
            replica.sim(sim_records),
            source.sim(sim_records),
            "the same state"
        );
    }

    /// **A bot's DM is one of the world's records**: written with them,
    /// and taken in by a world hosted from them — replacing what it held.
    #[test]
    fn a_bots_dm_travels_with_the_worlds_records() {
        let dm = Dm {
            channel_id: 31,
            user_id: 77,
            mirrored_through: 4,
            last_seen_discord: 900,
        };
        let source = vault();
        source.with_dms(|d| d.put(42, "0a1b", dm));
        let records = records_of(&source);
        assert_eq!(
            records.get("discord/dm/42/0a1b"),
            Some(&json!({
                "channel_id": 31,
                "user_id": 77,
                "mirrored_through": 4,
                "last_seen_discord": 900,
            }))
        );
        let replica = vault();
        replica.with_dms(|d| d.put(9, "gone", dm));
        apply(&replica, &records);
        assert_eq!(replica.dms(|d| d.of(42)), [("0a1b".to_string(), dm)]);
        assert_eq!(replica.dms(|d| d.of(9)), []);
    }

    /// A body not in the world yet keeps its place for when it is embodied;
    /// a place the map no longer has is left alone; records with no world
    /// state in them leave the seed.
    #[test]
    fn a_body_not_yet_here_keeps_its_place() {
        let mut records = Records::new();
        records.insert("body/b".into(), json!({"at": "vault-casting/green-room"}));
        records.insert("body/c".into(), json!({"at": "nowhere/at-all"}));
        assert_eq!(
            place_of(&records, "b").as_deref(),
            Some("vault-casting/green-room")
        );
        let h = vault();
        h.with(|w| w.enter("c", "C", at("band-one")).unwrap());
        let seeded = h.sim(sim_records);
        apply(&h, &records);
        assert_eq!(h.place_of("c"), "vault-casting/band-one");
        assert_eq!(h.sim(sim_records), seeded);
        assert!(sim_of(&records).is_none());
    }
}
