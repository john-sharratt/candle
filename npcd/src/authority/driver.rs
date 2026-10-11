//! A member's world leases (`docs/replicated_substrate.md` §9.2, §9.4, §9.6),
//! driven twice a second and on every lease event.
//!
//! - **Each world's lease** is asked for by the member rendezvous names as its
//!   proposer, once the records of the member that handed it back are applied;
//!   renewed by its holder every 10 s while rendezvous still names it; and
//!   handed over — the beat's state written, then released with this member's
//!   own `oseq` — when rendezvous names another.
//! - **Gaining a world** reloads it from its records, under an epoch above
//!   every one its records carry, and claims its id so its routes come here.
//! - **The cast** is read again every 2 s for characters written on other
//!   members, so each world's authority knows every character it lends.
//! - **Cut off**, or with its lease run out unrenewed, a member runs each world
//!   it has characters in locally and persists nothing of it. **Back** — joined,
//!   and the world held again by some member — it scraps that world and reloads
//!   it from its records.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cluster::consensus::rendezvous::owner;
use cluster::member::{AppMember, Link};
use cluster::replicate::{LeaseEvent, Phase, Replicator, LEASE_TTL, RENEW_EVERY};
use tokio::sync::broadcast::error::RecvError;
use tokio::time::{interval, MissedTickBehavior};

use super::npc_lease::hex;
use super::role::Role;
use super::world_lease::{epoch_for, lease_key, step, world_of, Step};
use crate::api::now_ms;
use crate::engine::runtime::Runtime;
use crate::world::Hosted;
use crate::world_state::store;

/// How often the leases are looked at between events.
pub const DRIVE_EVERY: Duration = Duration::from_millis(500);

/// How often the cast is read again for characters written on other members.
pub const CAST_EVERY: Duration = Duration::from_secs(2);

/// Drive `member`'s world leases until the process stops.
pub async fn run(rt: Arc<Runtime>, member: Arc<AppMember>) {
    let Some(r) = member.replicator().cloned() else {
        return;
    };
    let mut events = r.lease_events();
    let mut asked: HashMap<String, Instant> = HashMap::new();
    let mut tick = interval(DRIVE_EVERY);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut cast_read = Instant::now();
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            e = events.recv() => match e {
                Ok(e) => rt.lease_event(&member, &r, e),
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => return,
            },
        }
        if rt.stopping() {
            return;
        }
        if cast_read.elapsed() >= CAST_EVERY {
            rt.follow_cast().await;
            cast_read = Instant::now();
        }
        rt.drive_leases(&member, &r, &mut asked);
    }
}

impl Runtime {
    /// Whether any character in `world` runs here.
    fn runs_any_in(&self, world: &str) -> bool {
        self.bodies
            .in_world(world)
            .iter()
            .any(|(npc, _)| self.standing.runs(*npc))
    }

    /// What a member cut off, or whose lease ran out, makes of `world`: its
    /// own to run locally when characters of it run here, else only followed.
    fn cut_off_role(&self, world: &str) -> Role {
        match self.runs_any_in(world) {
            true => Role::Disconnected,
            false => Role::Replica,
        }
    }

    fn drive_leases(
        self: &Arc<Self>,
        member: &AppMember,
        r: &Arc<Replicator>,
        asked: &mut HashMap<String, Instant>,
    ) {
        let now = now_ms();
        let cut_off = member.link() == Link::CutOff;
        let joined = r.phase() == Phase::Joined && r.caught_up();
        let me = member.node();
        let (ready, group) = self.npcd_members();
        for world in self.hosted.ids() {
            let Some(hosted) = self.hosted.get(&world) else {
                continue;
            };
            let role = self.standing.role(&world);
            if cut_off {
                let next = self.cut_off_role(&world);
                if role != next {
                    tracing::warn!(
                        node = %hex(me),
                        "world `{world}`: cut off from the cluster — {}",
                        match next {
                            Role::Disconnected =>
                                "running it locally for the characters here, persisting nothing",
                            _ => "following nothing until it is back",
                        }
                    );
                    self.standing.set_role(&world, next, now);
                    if role.is_authority() {
                        member.let_go(world.as_bytes(), r.own_through());
                    }
                }
                continue;
            }
            if !joined {
                continue;
            }
            let key = lease_key(&world);
            let held = r.holds(&key);
            if role == Role::Disconnected {
                // Back once some member runs the world: its records are what
                // the world is, and what was lived here is scrapped.
                let elsewhere = member.holder_of(world.as_bytes()).is_some_and(|h| h != me);
                if elsewhere {
                    self.reload_world(&hosted);
                    self.standing.set_role(&world, Role::Replica, now);
                    tracing::info!(
                        node = %hex(me),
                        "world `{world}`: back with the cluster — reloaded from its records"
                    );
                    continue;
                }
            }
            let proposer = owner(world.as_bytes(), &ready);
            let since = asked.get(&world).map_or(Duration::MAX, |at| at.elapsed());
            match step(me, proposer, held, since, RENEW_EVERY) {
                Step::Acquire => {
                    // **A lease is never taken from its holder** (§9.5): while
                    // the member holding it is in the group, the proposer waits
                    // for it to hand the world over. Asking again meanwhile
                    // would only raise the ballot under the holder's renewal
                    // and run its lease out.
                    //
                    // The holder is known two ways. Its claim on the world's id
                    // reaches every member in its admission, before it may
                    // propose — which matters most when the group has just
                    // grown: a lease taken under a smaller quorum has no
                    // acceptor among the newcomers, and a majority of them alone
                    // would grant it a second time. And a round that found it
                    // held says so.
                    let holder = member
                        .holder_of(world.as_bytes())
                        .or_else(|| self.standing.busy(&world));
                    tracing::debug!(
                        node = %hex(me),
                        holder = ?holder.map(hex),
                        group = ?group.iter().map(|n| hex(*n)).collect::<Vec<_>>(),
                        "world `{world}`: named its proposer"
                    );
                    if holder.is_some_and(|h| h != me && group.contains(&h)) {
                        continue;
                    }
                    if let Some((by, through)) = self.standing.waiting(&world) {
                        if r.applied_through(by) < through {
                            continue;
                        }
                    }
                    if r.acquire(&key, LEASE_TTL) {
                        asked.insert(world.clone(), Instant::now());
                    }
                }
                Step::Renew => {
                    r.acquire(&key, LEASE_TTL);
                    asked.insert(world.clone(), Instant::now());
                }
                Step::HandOver => self.hand_over(&hosted, member, r),
                Step::Wait => {}
            }
        }
    }

    /// What became of a world lease this member asked for, held, or saw
    /// handed back.
    fn lease_event(self: &Arc<Self>, member: &AppMember, r: &Arc<Replicator>, e: LeaseEvent) {
        let Some(world) = world_of(e.key()).map(String::from) else {
            return;
        };
        let Some(hosted) = self.hosted.get(&world) else {
            return;
        };
        match e {
            LeaseEvent::Held { .. } => {
                self.standing.set_busy(&world, None);
                if !self.standing.role(&world).is_authority() {
                    self.take_world(&hosted, member);
                }
            }
            LeaseEvent::Lost { .. } => {
                if self.standing.role(&world).is_authority() {
                    let next = self.cut_off_role(&world);
                    tracing::warn!(
                        node = %hex(member.node()),
                        "world `{world}`: its lease ran out unrenewed — {}",
                        next.name()
                    );
                    self.standing.set_role(&world, next, now_ms());
                    member.let_go(world.as_bytes(), r.own_through());
                }
            }
            LeaseEvent::Busy { holder, .. } => self.standing.set_busy(&world, Some(holder)),
            LeaseEvent::Released { by, through, .. } => {
                self.standing.set_busy(&world, None);
                self.standing.set_waiting(&world, Some((by, through)));
            }
        }
    }

    /// This member holds `hosted`'s lease: it becomes the world's authority,
    /// the world reloaded from its records and written from now on under an
    /// epoch above every one they carry, its id claimed.
    pub(crate) fn take_world(&self, hosted: &Hosted, member: &AppMember) {
        let world = hosted.id();
        let highest = self
            .substrate()
            .map_or(0, |shared| store::highest_epoch(&shared, world));
        self.reload_world(hosted);
        let epoch = epoch_for(now_ms(), highest);
        let last = hosted.read(|w| w.log().last().map_or(0, |e| e.at));
        self.standing.set_pulsed(world, last);
        self.standing.set_waiting(world, None);
        self.standing
            .set_role(world, Role::Authority { epoch }, now_ms());
        member.hold(world.as_bytes());
        tracing::info!(
            node = %hex(member.node()),
            "world `{world}`: this member is its authority, epoch {epoch}"
        );
    }

    /// Rendezvous names another member for `hosted`: write what is pending,
    /// follow the world from now on, and release the lease with how far this
    /// member's writes reach.
    fn hand_over(&self, hosted: &Hosted, member: &AppMember, r: &Arc<Replicator>) {
        let world = hosted.id();
        if let Role::Authority { epoch } = self.standing.role(world) {
            self.write_world(hosted, epoch);
            self.pulse_world(hosted);
        }
        self.standing.set_role(world, Role::Replica, now_ms());
        let through = r.own_through();
        member.let_go(world.as_bytes(), through);
        r.release_lease(&lease_key(world), through);
        tracing::info!(
            node = %hex(member.node()),
            "world `{world}`: handed over, its writes reaching {through}"
        );
    }
}
