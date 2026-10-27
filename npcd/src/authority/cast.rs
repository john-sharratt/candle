//! The characters this member runs, kept to its NPC leases
//! (`docs/replicated_substrate.md` §9.3): each one lent here is woken — once
//! what its last holder wrote has been applied — each lent elsewhere is put
//! down, and each whose lease names a next holder is handed back at its next
//! turn boundary.
//!
//! Every member hosts every world and embodies every character in it, so each
//! world's copy here has every body; what differs is which minds think here.

use std::collections::BTreeSet;
use std::sync::Arc;

use cluster::consensus::rendezvous::Weighted;
use cluster::frame::MemberState;

use super::npc_lease::{hex, leases_of};
use super::ops::{OpReply, WorldOp};
use crate::engine::environment;
use crate::engine::event::{EventKind, Salience};
use crate::engine::runtime::Runtime;
use crate::engine::tick::Pace;
use crate::world_state::place_of;

impl Runtime {
    /// The npcd members ready now, weighted as rendezvous weighs them, and
    /// every member in the group (§9.5).
    pub(crate) fn npcd_members(&self) -> (Vec<Weighted>, BTreeSet<u64>) {
        let Some(member) = self.standing.member() else {
            return (Vec::new(), BTreeSet::new());
        };
        let group = member.group();
        let ready = group
            .iter()
            .filter(|m| m.state == MemberState::Ready)
            .map(|m| Weighted {
                node: m.node,
                weight: m.weight as f64,
            })
            .collect();
        let joined = group.iter().map(|m| m.node).collect();
        (ready, joined)
    }

    /// Take in characters created, changed or deleted on other members. The
    /// cast replicates as records; every member embodies every character of
    /// the worlds it hosts, in its own copy of the world, so each world's
    /// authority can lend it and each member can see it.
    pub(crate) async fn follow_cast(self: &Arc<Self>) {
        let Some(npcs) = self.npcs() else {
            return;
        };
        let changed = npcs.write().await.refresh();
        for npc in changed {
            let living = npcs.read().await.is_living(npc);
            if !living {
                if self.bodies.bound(npc).is_some() {
                    self.forget_character(npc);
                }
                continue;
            }
            if self.bodies.bound(npc).is_some() {
                continue;
            }
            let Some(p) = self.persona_of_async(npc).await else {
                continue;
            };
            let remembered = place_of(&self.standing.written(&p.world_id), &Runtime::body_id(npc));
            match self.embody_in_world(
                npc,
                &p.world_id,
                p.home.as_deref(),
                &p.name,
                remembered.as_deref(),
                0,
            ) {
                Ok(true) => {
                    tracing::info!(npc, world = %p.world_id, "a character made elsewhere is embodied here")
                }
                Ok(false) => {}
                Err(e) => tracing::error!(npc, "no body in `{}` — {e:#}", p.world_id),
            }
        }
    }

    /// Wake each character lent here and put down each lent elsewhere; hand
    /// back each the authority means to move, between its turns.
    pub(crate) fn reconcile_cast(self: &Arc<Self>) {
        if !self.standing.clustered() {
            return;
        }
        let Some(me) = self.standing.node() else {
            return;
        };
        let me = hex(me);
        let applied = |origin: u64| {
            self.standing
                .member()
                .and_then(|m| m.replicator().map(|r| r.applied_through(origin)))
                .unwrap_or(0)
        };
        for world in self.hosted.ids() {
            let leases = leases_of(&self.standing.written(&world));
            for (npc, _) in self.bodies.in_world(&world) {
                let lease = leases.get(&npc);
                let mine = lease.is_some_and(|l| l.holder == me);
                let awake = self.scheduler.pace_of(npc).is_some();
                let handing = self.standing.handing(npc);
                if !self.standing.characters() {
                    // **A member that runs no characters runs none**, whatever
                    // its leases say. One still naming it — lent before it was
                    // told so — is handed back once the authority names where
                    // it goes next; until then it is held and not run.
                    if awake {
                        self.put_down(npc);
                    }
                    let next = lease.is_some_and(|l| l.next.is_some());
                    if mine && next && !handing && self.standing.begin_handing(npc) {
                        self.hand_back(&world, npc);
                    } else if !mine && handing {
                        self.standing.end_handing(npc);
                    }
                    continue;
                }
                match (mine, awake) {
                    (true, false) if !handing => {
                        let ready = lease
                            .and_then(|l| l.after)
                            .is_none_or(|a| applied(a.origin) >= a.through);
                        if ready {
                            self.wake_character(npc);
                        }
                    }
                    (true, true) => {
                        let moving = lease.is_some_and(|l| l.next.is_some());
                        if moving && !self.standing.turning(npc) && self.standing.begin_handing(npc)
                        {
                            self.hand_back(&world, npc);
                        }
                    }
                    (false, true) if !handing => self.put_down(npc),
                    (false, _) if handing => self.standing.end_handing(npc),
                    _ => {}
                }
            }
        }
    }

    /// Bring a character lent here into the scheduler, grounded in its body,
    /// back in the register it was last in, and set it thinking.
    pub(crate) fn wake_character(self: &Arc<Self>, npc: u64) {
        let world_ms = self.world_ms(npc);
        self.scheduler.wake(npc, 0, world_ms);
        self.scheduler.set_pace(npc, Pace::WORKING, 0);
        if let Some((hosted, body)) = self.body_of(npc) {
            environment::push_one(&hosted, &self.scheduler, npc, &body, &|id| {
                self.world_ms(id)
            });
        }
        self.spawn_character(npc);
        tracing::info!(npc, "character lent here: running it");
        let (Some(npcs), Some(handle)) = (self.npcs(), self.spawner()) else {
            return;
        };
        let rt = Arc::clone(self);
        handle.spawn(async move {
            let mood = npcs.read().await.mood_of(npc).map(String::from);
            if let Some(mood) = mood {
                rt.scheduler.deliver(
                    npc,
                    world_ms,
                    Salience::IDLE,
                    EventKind::Description {
                        text: format!(
                            "What you were feeling, when you last stopped to notice, was {mood}."
                        ),
                    },
                );
            }
        });
    }

    /// Stop running a character here: its lease names another member.
    pub(crate) fn put_down(&self, npc: u64) {
        self.scheduler.retire(npc);
        self.stop_character(npc);
        tracing::info!(npc, "character lent elsewhere: no longer running it here");
    }

    /// Hand `npc` back to `world`'s authority, with how far this member's
    /// writes reach, and put it down once the authority has it.
    fn hand_back(self: &Arc<Self>, world: &str, npc: u64) {
        let Some(handle) = self.spawner() else {
            self.standing.end_handing(npc);
            return;
        };
        let through = self
            .standing
            .member()
            .and_then(|m| m.replicator().map(|r| r.own_through()))
            .unwrap_or(0);
        let rt = Arc::clone(self);
        let world = world.to_string();
        handle.spawn(async move {
            match rt
                .world_write(&world, WorldOp::HandBack { npc, through })
                .await
            {
                Ok(OpReply::Done { ok: true }) => rt.put_down(npc),
                Ok(other) => {
                    tracing::warn!(npc, "a hand-back was not taken: {other:?}");
                    rt.standing.end_handing(npc);
                }
                Err(e) => {
                    tracing::warn!(npc, "a hand-back did not reach the world: {e:#}");
                    rt.standing.end_handing(npc);
                }
            }
        });
    }
}
