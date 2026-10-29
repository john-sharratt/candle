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

use super::npc_lease::{leases_of, lend, moves, with_leases, LEASES};
use super::pulse::{self, EVENTS};
use super::role::Role;
use super::told::{self, TOLD};
use super::world_lease::epoch_for;
use crate::api::now_ms;
use crate::engine::environment::{self, Reach};
use crate::engine::event::{EventKind, Salience};
use crate::engine::runtime::Runtime;
use crate::world::Hosted;
use crate::world_state::{apply, diff, legacy, records_of, store};

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
            if legacy::take_in(&shared, hosted) {
                tracing::info!("world `{id}`: taken in from the objects kept before world records");
                self.standing.mark_legacy(id);
            }
        } else {
            apply(hosted, &records);
        }
        self.standing
            .set_followed(id, shared.custom_objects_version());
        self.standing.set_written(id, records);
        self.standing.recompute_held();
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
        self.reconcile_cast();
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
        if self.standing.clustered() {
            let mut current = leases_of(&before);
            current.extend(handed.clone());
            let cast: Vec<u64> = self
                .bodies
                .in_world(world)
                .into_iter()
                .map(|(npc, _)| npc)
                .collect();
            let (members, joined) = self.lending_members();
            let lent = lend(&cast, &current, &members, &joined, now_ms());
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
        }
        let changes = diff(&before, &records);
        if changes.is_empty() {
            return;
        }
        let leases_changed = changes
            .put
            .iter()
            .map(|(k, _)| k)
            .chain(changes.ceased.iter())
            .any(|k| k.starts_with(LEASES));
        match store::write(&shared, world, &changes, epoch) {
            Ok(()) => {
                self.standing.set_written(world, records);
                self.standing
                    .set_followed(world, shared.custom_objects_version());
                if leases_changed {
                    self.standing.recompute_held();
                }
                if let Some(last) = self.standing.legacy_written(world) {
                    if let Err(e) = legacy::retire(&shared, world, last) {
                        tracing::error!("world `{world}`: the old objects were not retired: {e}");
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
                apply(hosted, &records);
                self.standing.set_written(world, records);
                self.standing.set_followed(world, version);
                self.standing.recompute_held();
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
        self.standing.set_written(world, records);
        self.standing
            .set_followed(world, shared.custom_objects_version());
        self.standing.recompute_held();
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
    use candle_conversation::substrate::Substrate;
    use npc_map::world::Where;
    use tempfile::TempDir;

    use super::*;
    use crate::mind::Mind;
    use crate::world_state::{records_of, sim_records};

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
        assert_eq!(rt.standing.written(WORLD), records_of(&world));

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
