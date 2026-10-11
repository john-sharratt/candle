//! A member following its worlds (`docs/replicated_substrate.md` §9.1–§9.4):
//! joined, it reports its worlds in `COORD`, takes in the pulses each world's
//! authority sends, and drives its world leases ([`super::driver`]).

use std::sync::Arc;

use cluster::frame::Pulse;
use cluster::member::AppMember;
use tokio::sync::broadcast::error::RecvError;

use super::driver;
use super::pulse::{self, EVENTS};
use super::report;
use super::role::Role;
use super::told::{self, TOLD};
use crate::engine::runtime::Runtime;

impl Runtime {
    /// This daemon joined its cluster as `member`: from now on it follows its
    /// worlds as a member.
    pub fn join_cluster(self: &Arc<Self>, member: Arc<AppMember>) {
        self.standing.set_member(Arc::clone(&member));
        self.standing.recompute_held();
        let Some(r) = member.replicator().cloned() else {
            return;
        };
        let weak = Arc::downgrade(self);
        r.set_app_coord(Box::new(move || {
            weak.upgrade()
                .map(|rt| report::app_coord(&rt))
                .unwrap_or_default()
        }));
        let Some(handle) = self.spawner() else {
            return;
        };
        let rt = Arc::clone(self);
        let mut pulses = r.pulses();
        handle.spawn(async move {
            loop {
                match pulses.recv().await {
                    Ok(p) => rt.pulsed(p),
                    Err(RecvError::Lagged(missed)) => {
                        tracing::warn!("{missed} world pulses missed: those moments are lost");
                    }
                    Err(RecvError::Closed) => return,
                }
                if rt.stopping() {
                    return;
                }
            }
        });
        handle.spawn(driver::run(Arc::clone(self), member));
    }

    /// A pulse from a world's authority. Taken in only for a world this
    /// daemon follows: one it runs has its own events, and one it runs cut off
    /// is its own world until it rejoins.
    pub(crate) fn pulsed(&self, p: Pulse) {
        if self.standing.role(&p.world) != Role::Replica {
            return;
        }
        let Some(hosted) = self.hosted.get(&p.world) else {
            return;
        };
        match p.kind.as_str() {
            EVENTS => match pulse::decode(&p.body) {
                Ok(events) => {
                    hosted.with(|w| w.witness(events));
                }
                Err(e) => tracing::warn!("world `{}`: a pulse would not decode: {e}", p.world),
            },
            TOLD => match told::decode(&p.body) {
                Ok(told) => {
                    for t in told.into_iter().filter(|t| self.standing.runs(t.npc)) {
                        self.scheduler.deliver(t.npc, t.at_ms, t.salience(), t.kind);
                    }
                }
                Err(e) => tracing::warn!("world `{}`: a delivery would not decode: {e}", p.world),
            },
            other => tracing::debug!("world `{}`: a pulse of kind `{other}` ignored", p.world),
        }
    }
}
