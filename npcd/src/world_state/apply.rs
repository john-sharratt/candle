//! A world's state records, put into a hosted world
//! (`docs/replicated_substrate.md` §9.1): what a world is hosted from, what a
//! replica takes in as its authority writes, and what a world reloads from when
//! it leaves disconnected mode.
//!
//! The world's [`Sim`] is taken in whole when the records hold one — what this
//! machine keeps of its own aside ([`Sim::take_in`]) — and left as the world has
//! it when they do not, so a world hosted for the first time keeps its seed. A
//! body record places a body that is in the world; one that is not yet is kept
//! for when it is embodied ([`place_of`]). A body's record carries the journey
//! it is on and the lift's who is aboard and waiting, so a replica building a
//! turn for a character it runs builds it from where the authority has it.

use std::collections::BTreeMap;

use npc_map::lift::Doors;
use npc_map::state::{BodyState, LiftState, WalkState};
use npc_map::world::{Where, World};
use serde_json::Value;

use super::keys::unescape;
use super::snapshot::{Records, SIM, SIM_MAPS};
use super::split::{join, META};
use crate::engine::build::replay::replay;
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

/// Each body the records place, then the lift — after the bodies, so a rider
/// the lift's record has aboard is aboard however its body's record moved it.
/// A body not in the world, or placed somewhere the map does not have, is
/// left as it is.
fn apply_world(w: &mut World, records: &Records) {
    for (name, v) in records {
        let Some(id) = name.strip_prefix("body/").map(unescape) else {
            continue;
        };
        if w.actor(&id).is_none() {
            continue;
        }
        if let Some(body) = body_of(v) {
            let _ = w.restore_body(&id, &body);
        }
    }
    if let Some(car) = records.get("lift").map(lift_of) {
        w.restore_lift(&car);
    }
}

/// A place a record names (`<area>/<node>`).
fn place(v: &Value) -> Option<Where> {
    v.as_str().and_then(Where::parse)
}

/// A body's record, read back.
fn body_of(v: &Value) -> Option<BodyState> {
    let at = place(v.get("at")?)?;
    let hold = v.get("hold").filter(|h| !h.is_null()).and_then(|h| {
        let part = h.get("part")?.as_str()?.to_string();
        let subject = h.get("subject").and_then(Value::as_str).map(String::from);
        Some((part, subject))
    });
    let walk = v.get("walk").filter(|w| !w.is_null()).and_then(|w| {
        let toward = place(w.get("toward")?)?;
        let ahead = w
            .get("ahead")?
            .as_array()?
            .iter()
            .map(place)
            .collect::<Option<Vec<_>>>()?;
        Some(WalkState { toward, ahead })
    });
    Some(BodyState { at, hold, walk })
}

/// The lift's record, read back.
fn lift_of(v: &Value) -> LiftState {
    let floors = |field: &str| -> Vec<usize> {
        v.get(field)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|f| f as usize)
                    .collect()
            })
            .unwrap_or_default()
    };
    let bound_for = |field: &str| -> BTreeMap<String, usize> {
        v.get(field)
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(body, f)| Some((body.clone(), f.as_u64()? as usize)))
                    .collect()
            })
            .unwrap_or_default()
    };
    LiftState {
        floor: v.get("floor").and_then(Value::as_u64).unwrap_or(0) as usize,
        doors: match v.get("doors").and_then(Value::as_str) {
            Some("closed") => Doors::Closed,
            _ => Doors::Open,
        },
        stops: floors("stops"),
        riders: bound_for("riders"),
        waiting: bound_for("waiting"),
    }
}

/// Put `records` into `hosted`, under one hold of its lock — and its bots'
/// DMs, servers and channels, and its own Discord server and what is
/// mirrored there (`docs/npcd_discord.md` §4.2), which are not under it.
pub fn apply(hosted: &Hosted, records: &Records) {
    let sim = sim_of(records);
    hosted.with_both(|w, s| {
        // The state first, and what the world's builders made laid over the map
        // from it (`docs/zen_moon_mind.md` §14) — so a body the records place in
        // a built room finds the room there.
        if let Some(sim) = sim {
            s.take_in(sim);
        }
        replay(w, &s.built.unlaid());
        apply_world(w, records);
    });
    hosted.with_dms(|dms| dms.take_in(records));
    hosted.with_servers(|servers| servers.take_in(records));
    hosted.with_mirror(|mirror| mirror.take_in(records));
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::snapshot::tests::{at, vault};
    use super::super::snapshot::{sim_records, world_records};
    use super::*;
    use crate::discord::dm::record::Dm;
    use crate::discord::server::record::{Channel, Server};
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

    /// **Who is riding the lift travels with it.** A body the source has
    /// waiting on its landing is riding on a replica that takes the records
    /// in — so the replica offers it nowhere to walk — and one the replica
    /// thought was riding, the source's record says is not.
    #[test]
    fn who_rides_the_lift_travels_with_its_record() {
        let source = vault();
        let (from, dest) = source.read(|w| (w.shaft()[0].clone(), w.shaft().len() - 1));
        source.with(|w| {
            w.call_lift(dest);
            while !w.lift().unwrap().boardable_at(dest) {
                w.tick();
            }
            w.enter("a", "A", from.clone()).unwrap();
            w.enter("b", "B", from.clone()).unwrap();
            assert!(w.ride_lift("a", dest));
        });
        let records = source.read(world_records);
        assert_eq!(records["lift"]["waiting"], json!({"a": dest}));
        assert_eq!(records["lift"]["riders"], json!({}));

        let replica = vault();
        replica.with(|w| {
            w.enter("a", "A", from.clone()).unwrap();
            w.enter("b", "B", from.clone()).unwrap();
            assert!(w.ride_lift("b", dest));
        });
        apply(&replica, &records);
        replica.read(|w| {
            assert_eq!(w.riding("a"), Some(dest), "aboard at the source");
            assert_eq!(w.riding("b"), None, "not riding at the source");
            assert_eq!(w.lift_within("a"), (false, false, Vec::new()));
        });
        assert_eq!(replica.read(world_records), records, "the same lift");
    }

    /// **A replica given the records is the world the authority has.**
    /// Everything the copies share ([`World::state`]) — a body standing, one
    /// at a station, one on its way to another level, the lift with a body
    /// aboard and one waiting — written by one world and taken in by another
    /// built from the same map reads back the same, and so writes the same
    /// records.
    #[test]
    fn a_replica_given_the_records_has_the_world_the_authority_has() {
        let source = vault();
        let (shaft, top) = source.read(|w| (w.shaft().to_vec(), w.shaft().len() - 1));
        source.with(|w| {
            for (id, place) in [
                ("stands", at("green-room")),
                ("holds", at("band-one")),
                ("walks", at("green-room")),
                ("aboard", shaft[0].clone()),
                ("waits", shaft[top].clone()),
            ] {
                w.enter(id, id, place).unwrap();
            }
            w.take("holds", Some("cindy")).unwrap();
            w.call_lift(0);
            while !w.lift().unwrap().boardable_at(0) {
                w.tick();
            }
            assert!(w.ride_lift("aboard", top));
            assert!(w.ride_lift("waits", 0));
            w.set_off("walks", shaft[top].clone()).unwrap();
        });
        let records = source.read(world_records);
        assert!(records["body/walks"]["walk"].is_object(), "{records:?}");

        let replica = vault();
        replica.with(|w| {
            for id in ["stands", "holds", "walks", "aboard", "waits"] {
                w.enter(id, id, at("green-room")).unwrap();
            }
        });
        apply(&replica, &records);
        assert_eq!(replica.read(World::state), source.read(World::state));
        assert_eq!(replica.read(world_records), records);
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

    /// **A bot's servers and channels are world records** too, taken in by
    /// a world hosted from them in place of what it held.
    #[test]
    fn a_bots_servers_and_channels_travel_with_the_worlds_records() {
        let channel = Channel {
            guild_id: 600,
            thread: "th7".into(),
            mirrored_through: 2,
            last_seen_discord: 901,
        };
        let source = vault();
        source.with_servers(|s| {
            s.put_channel(42, 610, channel.clone());
            s.put_server(
                42,
                600,
                Server {
                    name: "zenworld".into(),
                },
            );
        });
        let records = records_of(&source);
        assert_eq!(
            records.get("discord/channel/42/610"),
            Some(&json!({
                "guild_id": 600,
                "thread": "th7",
                "mirrored_through": 2,
                "last_seen_discord": 901,
            }))
        );
        assert_eq!(
            records.get("discord/server/42/600"),
            Some(&json!({ "name": "zenworld" }))
        );
        let replica = vault();
        replica.with_servers(|s| s.put_channel(9, 1, channel.clone()));
        apply(&replica, &records);
        assert_eq!(replica.servers(|s| s.channels_of(42)), [(610, channel)]);
        assert_eq!(replica.servers(|s| s.channels_of(9)), []);
        assert_eq!(
            replica.servers(|s| s.server(42, 600)),
            Some(Server {
                name: "zenworld".into()
            })
        );
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
