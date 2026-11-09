//! A world's beat, as this daemon's role in it has it
//! (`docs/replicated_substrate.md` §9.1–§9.4), and the restore of a world being
//! hosted from its state records.
//!
//! - **The authority** runs the world: its moment, then the entities that
//!   changed written under its lease's epoch, with the leases it lends the
//!   cast; then what happened since its last pulse, and what it has to tell
//!   characters that run elsewhere, as pulses.
//! - **A replica** follows: the world's records whenever they change, the
//!   authority's pulses as they arrive (see [`super::follow`]), and
//!   perception for the characters that run here.
//! - **A world run cut off** has its moment and writes nothing.

use std::sync::Arc;

use cluster::frame::Pulse;

use super::npc_lease::{hold_still, lend, moves, with_leases};
use super::pulse::{self, EVENTS};
use super::role::Role;
use super::told::{self, TOLD};
use super::world_lease::epoch_for;
use super::written::Written;
use crate::api::now_ms;
use crate::engine::environment::{self, Reach};
use crate::engine::event::{EventKind, Salience};
use crate::engine::runtime::Runtime;
use crate::world::Hosted;
use crate::world_state::{apply, diff, legacy, records_of, renamed, store};

fn nobody_else(_: u64, _: u64, _: Salience, _: EventKind) {}

impl Runtime {
    /// Put a world being hosted into the state its records hold — or, for a
    /// world kept before records, the state the old objects held, once. A
    /// world with neither is as its map seeds it, with its command table open:
    /// a world that hands out no missions leaves its cast with nothing asked
    /// of it.
    pub(crate) fn restore(&self, hosted: &Hosted) {
        hosted.with_sim(|s| s.table_open = true);
        let Some(shared) = self.substrate() else {
            return;
        };
        let id = hosted.id();
        // A daemon with no cluster is the world's authority, under an epoch
        // from its clock above every one the world was written under — the
        // rule a member gaining a world follows (§9.2), so a world removed and
        // hosted again writes above its own tombstones.
        if !self.standing.clustered() {
            let epoch = epoch_for(now_ms(), store::highest_epoch(&shared, id));
            self.standing
                .set_role(id, Role::Authority { epoch }, now_ms());
        }
        let records = store::read(&shared, id);
        if records.is_empty() {
            // **A world renamed comes up as it was under its former id**
            // (`docs/world_rename.md` §3); its former is retired once this
            // world has written its own records.
            if let Some(old) = renamed::take_in(&shared, hosted, &self.standing.formerly_of(id)) {
                tracing::info!("world `{id}`: taken in from its former id `{old}`");
                self.standing.mark_renamed(id, &old);
            } else if legacy::take_in(&shared, hosted) {
                tracing::info!("world `{id}`: taken in from the objects kept before world records");
                self.standing.mark_legacy(id);
            }
        } else {
            apply(hosted, &records);
        }
        self.standing
            .set_followed(id, shared.custom_objects_version());
        self.standing.set_written(id, Written::of(records));
    }

    /// One beat of a world's metronome.
    pub fn beat(self: &Arc<Self>, hosted: &Hosted) {
        match self.standing.role(hosted.id()) {
            Role::Authority { epoch } => {
                self.moment(hosted);
                self.write_world(hosted, epoch);
                self.pulse_world(hosted);
            }
            Role::Replica => self.follow_world(hosted),
            Role::Disconnected => {
                self.moment(hosted);
            }
        }
        // Its own cast only: nothing in the reconcile crosses worlds, and
        // every hosted world beats, so each is reconciled once a beat.
        self.reconcile_cast(hosted.id());
    }

    /// Write what changed of `hosted` since it was last written, stamped with
    /// `epoch`, with the leases the cast is lent now.
    pub(crate) fn write_world(&self, hosted: &Hosted, epoch: u64) {
        let world = hosted.id();
        let Some(shared) = self.substrate() else {
            return;
        };
        let before = self.standing.written(world);
        let mut records = records_of(hosted);
        let handed = self.standing.take_handed(world);
        // The leases the records carry once written: lent now when clustered,
        // otherwise whatever lease records the world's own state holds.
        let mut leases = None;
        if self.standing.clustered() {
            let mut current = before.leases.clone();
            current.extend(handed.clone());
            let cast: Vec<u64> = self
                .bodies
                .in_world(world)
                .into_iter()
                .map(|(npc, _)| npc)
                .collect();
            let (members, joined) = self.lending_members();
            let mut lent = lend(&cast, &current, &members, &joined, now_ms());
            // A suspended character stays where it is, by its record.
            if let Some(npcs) = self.npcs() {
                hold_still(&mut lent, &npcs.view().suspended());
            }
            let moved = moves(&current, &lent);
            if !moved.is_empty() {
                let weights: Vec<String> = members
                    .iter()
                    .map(|m| format!("{:016x}={}", m.node, m.weight))
                    .collect();
                for m in moved {
                    tracing::info!(
                        target: "npcd::authority::lease",
                        world,
                        npc = m.nid,
                        from = m.from,
                        holder = m.holder,
                        next = m.next.as_deref().unwrap_or(""),
                        weights = %weights.join(" "),
                        "lent"
                    );
                }
            }
            with_leases(&mut records, &lent);
            leases = Some(lent);
        }
        let changes = diff(&before.records, &records);
        if changes.is_empty() {
            return;
        }
        match store::write(&shared, world, changes, epoch) {
            Ok(()) => {
                let written = match leases {
                    Some(lent) => Written::new(records, lent),
                    None => Written::of(records),
                };
                self.standing.set_written(world, written);
                self.standing
                    .set_followed(world, shared.custom_objects_version());
                if let Some(last) = self.standing.legacy_written(world) {
                    if let Err(e) = legacy::retire(&shared, world, last) {
                        tracing::error!("world `{world}`: the old objects were not retired: {e}");
                    }
                }
                if let Some(old) = self.standing.renamed_written(world) {
                    let above = epoch_for(now_ms(), store::highest_epoch(&shared, &old));
                    match renamed::retire(&shared, &old, above) {
                        Ok(n) => tracing::info!(
                            "world `{world}`: its former id `{old}` retired, {n} records tombstoned"
                        ),
                        Err(e) => tracing::error!(
                            "world `{world}`: its former id `{old}` was not retired: {e}"
                        ),
                    }
                }
            }
            Err(e) => {
                tracing::error!("world `{world}`: its state was not written: {e}");
                for (npc, lease) in handed {
                    self.standing.handed_back(world, npc, lease);
                }
            }
        }
    }

    /// Send what happened in `hosted` since its last pulse, and what it has
    /// to tell characters that run elsewhere, to the other members.
    pub(crate) fn pulse_world(&self, hosted: &Hosted) {
        let world = hosted.id();
        let told = self.standing.take_told(world);
        let Some(r) = self.standing.member().and_then(|m| m.replicator().cloned()) else {
            return;
        };
        let after = self.standing.pulsed(world);
        let (events, last) = hosted.read(|w| pulse::since(w.log(), after));
        if !events.is_empty() {
            r.pulse(Pulse {
                world: world.to_string(),
                kind: EVENTS.to_string(),
                body: pulse::encode(&events),
            });
            self.standing.set_pulsed(world, last);
        }
        if !told.is_empty() {
            r.pulse(Pulse {
                world: world.to_string(),
                kind: TOLD.to_string(),
                body: told::encode(&told),
            });
        }
    }

    /// A beat of a world this daemon follows: its records taken in when they
    /// have changed, and perception for the characters that run here.
    fn follow_world(self: &Arc<Self>, hosted: &Hosted) {
        let world = hosted.id();
        if let Some(shared) = self.substrate() {
            let version = shared.custom_objects_version();
            if self.standing.followed(world) != Some(version) {
                let records = store::read(&shared, world);
                // **An unchanged world is not taken in again.** The version is
                // substrate-wide, so most moves of it are some other object's
                // write; with nothing changed here and the records as the last
                // take-in had them, applying them would put back what is there.
                let unchanged =
                    !hosted.is_local() && self.standing.written(world).records == records;
                if !unchanged {
                    apply(hosted, &records);
                    self.standing.set_written(world, Written::of(records));
                }
                self.standing.set_followed(world, version);
            }
        }
        let here = |npc_id: u64| self.standing.runs(npc_id);
        let reach = Reach {
            here: &here,
            elsewhere: &nobody_else,
        };
        environment::perceive(
            hosted,
            &self.bodies,
            &self.scheduler,
            &|id| self.world_ms(id),
            &reach,
        );
        self.standing_tasks(hosted);
        self.follow_visitors(hosted);
        self.show_out_the_expired();
    }

    /// Remove world `id`: stop it, and tombstone each of its records above
    /// every epoch they were written under, so nothing of it comes back
    /// (§9.1). Whether it was hosted.
    pub fn remove_world(&self, id: &str) -> bool {
        let hosted = self.unhost(id);
        if let Some(shared) = self.substrate() {
            let epoch = epoch_for(now_ms(), store::highest_epoch(&shared, id));
            match store::cease_world(&shared, id, epoch) {
                Ok(n) => tracing::info!("world `{id}`: removed, {n} records tombstoned"),
                Err(e) => tracing::error!("world `{id}`: its records were not tombstoned: {e}"),
            }
        }
        hosted
    }

    /// Reload a world from its records, scrapping what it became here: a
    /// world run cut off rejoining, or one taken over (§9.4, §9.6).
    pub(crate) fn reload_world(&self, hosted: &Hosted) {
        let world = hosted.id();
        let Some(shared) = self.substrate() else {
            return;
        };
        // What its operations' stores held here goes with the local world: a
        // publish it landed is in the record, and the rest is the events'.
        self.ops.scrap(world);
        let records = store::read(&shared, world);
        apply(hosted, &records);
        self.standing.set_written(world, Written::of(records));
        self.standing
            .set_followed(world, shared.custom_objects_version());
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
    use candle_conversation::substrate::Substrate;
    use npc_map::world::Where;
    use tempfile::TempDir;

    use std::collections::BTreeMap;

    use super::*;
    use crate::discord::dm::record::Dm;
    use crate::mind::Mind;
    use crate::world_state::{records_of, sim_records, Records};

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    fn shared(dir: &Path) -> SharedSubstrate {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
        SharedSubstrate::new(substrate, persistence)
    }

    fn daemon(dir: &TempDir) -> (Arc<Runtime>, SharedSubstrate) {
        let rt = Runtime::new(Mind::new(None), dir.path()).unwrap();
        let s = shared(&dir.path().join("substrate"));
        rt.set_substrate(s.clone());
        (rt, s)
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// **A world comes back as it was left.** Its authority writes what
    /// changed each beat; a daemon hosting it afresh from the same substrate
    /// finds the bodies where they stood and the world's state as it was.
    #[tokio::test]
    async fn a_world_is_hosted_again_from_what_its_authority_wrote() {
        let dir = TempDir::new().unwrap();
        let (rt, s) = daemon(&dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        world.with(|w| w.enter("m1", "Maker-01", at("green-room")).unwrap());
        world.with_sim(|s| s.table_open = false);
        rt.beat(&world);
        assert_eq!(rt.standing.written(WORLD).records, records_of(&world));

        let again = Runtime::new(Mind::new(None), dir.path()).unwrap();
        again.set_substrate(s.clone());
        let hosted = again.host(WORLD, Path::new(ROOMS)).unwrap();
        again.hold_world(WORLD, true);
        hosted.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        again.restore(&hosted);
        assert_eq!(hosted.place_of("m1"), "vault-casting/green-room");
        assert!(!hosted.sim(|s| s.table_open));
        assert_eq!(hosted.sim(sim_records), world.sim(sim_records));
    }

    /// A removed world stops, and its records are gone: hosted again, it
    /// comes up as its map seeds it.
    #[tokio::test]
    async fn a_removed_world_stops_and_leaves_no_state() {
        let dir = TempDir::new().unwrap();
        let (rt, s) = daemon(&dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        world.with_sim(|s| s.table_open = false);
        rt.beat(&world);
        assert!(!store::read(&s, WORLD).is_empty());

        assert!(rt.remove_world(WORLD));
        assert!(rt.hosted.get(WORLD).is_none());
        assert!(store::read(&s, WORLD).is_empty());
        let again = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        assert!(again.sim(|s| s.table_open), "seeded afresh");
        rt.beat(&again);
        assert!(
            !store::read(&s, WORLD).is_empty(),
            "hosted again, it writes above its own tombstones"
        );
    }

    /// **A world renamed comes up as it was under its former id, and the former
    /// is retired after its first write** (`docs/world_rename.md` §3): hosted
    /// with no records of its own, it takes in the former's — the bodies where
    /// they stood, the table as it was left — writes them as its own on its
    /// first beat, and the former's are then gone.
    #[tokio::test]
    async fn a_renamed_world_comes_up_as_it_was_and_retires_its_former() {
        let dir = TempDir::new().unwrap();
        let (rt, s) = daemon(&dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        world.with(|w| w.enter("m1", "Maker-01", at("green-room")).unwrap());
        world.with_sim(|s| s.table_open = false);
        let kept = records_of(&world);
        store::write(&s, "old-vault", diff(&Records::new(), &kept), 5).unwrap();
        assert!(rt.remove_world(WORLD));

        rt.standing.set_formers(BTreeMap::from([(
            WORLD.to_string(),
            vec!["old-vault".to_string()],
        )]));
        let renamed = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        renamed.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        rt.restore(&renamed);
        assert_eq!(renamed.place_of("m1"), "vault-casting/green-room");
        assert!(!renamed.sim(|s| s.table_open));

        rt.beat(&renamed);
        assert_eq!(store::read(&s, WORLD), records_of(&renamed));
        assert!(
            store::read(&s, "old-vault").is_empty(),
            "the former is retired"
        );
        assert_eq!(rt.standing.renamed_written(WORLD), None, "once");
    }

    /// An authority with `m1` standing in the green room, beaten once so its
    /// records are written; and a second daemon on the same substrate
    /// following the same world, restored from them.
    fn authority_and_replica(
        dir: &TempDir,
    ) -> (
        Arc<Runtime>,
        Arc<Hosted>,
        Arc<Runtime>,
        Arc<Hosted>,
        SharedSubstrate,
    ) {
        let (rt, s) = daemon(dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        world.with(|w| w.enter("m1", "Maker-01", at("green-room")).unwrap());
        rt.beat(&world);

        let follower = Runtime::new(Mind::new(None), dir.path()).unwrap();
        follower.set_substrate(s.clone());
        let hosted = follower.host(WORLD, Path::new(ROOMS)).unwrap();
        follower.hold_world(WORLD, true);
        hosted.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        follower.restore(&hosted);
        follower.standing.set_role(WORLD, Role::Replica, now_ms());
        (rt, world, follower, hosted, s)
    }

    /// A custom object written that is no part of `WORLD`, moving the
    /// substrate-wide version.
    fn unrelated_write(s: &SharedSubstrate, n: u64) {
        let mut records = Records::new();
        records.insert("n".into(), serde_json::json!(n));
        store::write(s, "elsewhere", diff(&Records::new(), &records), n).unwrap();
    }

    /// **A replica takes in nothing for another object's write.** Once its
    /// world is taken in and not changed here, a version moved by a write
    /// that is no part of it leaves the world as it is — and the follow is
    /// still at the new version.
    #[tokio::test]
    async fn an_unrelated_write_takes_nothing_in() {
        let dir = TempDir::new().unwrap();
        let (_rt, _world, follower, hosted, s) = authority_and_replica(&dir);
        unrelated_write(&s, 1);
        follower.beat(&hosted);
        assert!(!hosted.is_local(), "a follow beat changes nothing here");
        let taken = hosted.times_taken_in();

        unrelated_write(&s, 2);
        follower.beat(&hosted);
        assert_eq!(hosted.times_taken_in(), taken, "nothing taken in");
        assert_eq!(
            follower.standing.followed(WORLD),
            Some(s.custom_objects_version())
        );
        assert_eq!(hosted.place_of("m1"), "vault-casting/green-room");
    }

    /// **A change made here is put right by the next follow**: a body moved
    /// locally stands where its record says again, and a DM put on the
    /// replica is gone — though the records themselves did not change.
    #[tokio::test]
    async fn a_local_change_is_taken_back_by_the_next_follow() {
        let dir = TempDir::new().unwrap();
        let (_rt, _world, follower, hosted, s) = authority_and_replica(&dir);
        unrelated_write(&s, 1);
        follower.beat(&hosted);

        hosted.with(|w| w.place("m1", at("band-one")).unwrap());
        assert_eq!(hosted.place_of("m1"), "vault-casting/band-one");
        assert!(hosted.is_local());
        unrelated_write(&s, 2);
        follower.beat(&hosted);
        assert_eq!(hosted.place_of("m1"), "vault-casting/green-room");

        let dm = Dm {
            channel_id: 31,
            user_id: 77,
            mirrored_through: 4,
            last_seen_discord: 900,
        };
        hosted.with_dms(|d| d.put(42, "0a1b", dm));
        unrelated_write(&s, 3);
        follower.beat(&hosted);
        assert_eq!(hosted.dms(|d| d.of(42)), []);
    }

    /// **A pulse and perception leave a replica's world not local**: the log
    /// a pulse is witnessed into, and the sweep a follow beat pushes, are not
    /// in the records.
    #[tokio::test]
    async fn a_pulse_and_a_push_leave_the_world_not_local() {
        let dir = TempDir::new().unwrap();
        let (_rt, world, follower, hosted, s) = authority_and_replica(&dir);
        unrelated_write(&s, 1);
        follower.beat(&hosted);
        assert!(!hosted.is_local());
        let (events, _) = world.read(|w| pulse::since(w.log(), 0));
        assert!(!events.is_empty());
        hosted.witness(events);
        environment::push_one(&hosted, &follower.scheduler, 1, "m1", &|_| 0);
        assert!(!hosted.is_local());
    }

    /// **The world its authority beats is local**, so handed over to be
    /// followed, its first follow beat takes the records in.
    #[tokio::test]
    async fn a_world_handed_over_takes_its_records_in_on_its_first_follow() {
        let dir = TempDir::new().unwrap();
        let (rt, world, _follower, _hosted, s) = authority_and_replica(&dir);
        rt.beat(&world);
        assert!(world.is_local());
        rt.standing.set_role(WORLD, Role::Replica, now_ms());
        let taken = world.times_taken_in();
        unrelated_write(&s, 1);
        rt.beat(&world);
        assert_eq!(world.times_taken_in(), taken + 1);
    }

    /// **A character lent back here is grounded afresh**: this member showed
    /// its body where it stands before it was lent away, and on waking it the
    /// character is still handed its percept on its first delta.
    #[tokio::test]
    async fn a_character_lent_back_receives_its_percept() {
        let dir = TempDir::new().unwrap();
        let (rt, _s) = daemon(&dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        world.with(|w| w.enter("m1", "Maker-01", at("green-room")).unwrap());
        rt.bodies.bind(1, WORLD, "m1").unwrap();
        assert!(world.delta("m1").percept.is_some(), "shown here once");

        rt.wake_character(1);
        let at_ms = 10_000_000;
        for id in rt.scheduler.due_now(at_ms) {
            rt.scheduler.tick(id, at_ms, at_ms, |_, _| Vec::new());
        }
        let turns = rt
            .scheduler
            .window_of(1, |win| {
                win.turns().map(|t| t.text.clone()).collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert!(turns.iter().any(|t| t.starts_with("You are")), "{turns:?}");
    }

    /// A beat that changed nothing writes nothing.
    #[tokio::test]
    async fn a_still_world_writes_nothing() {
        let dir = TempDir::new().unwrap();
        let (rt, s) = daemon(&dir);
        let world = rt.host(WORLD, Path::new(ROOMS)).unwrap();
        rt.hold_world(WORLD, true);
        rt.beat(&world);
        let version = s.custom_objects_version();
        rt.beat(&world);
        rt.beat(&world);
        assert_eq!(s.custom_objects_version(), version);
    }
}
