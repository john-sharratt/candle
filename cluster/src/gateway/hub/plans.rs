//! The cast plan, and the log lines that record it.
//!
//! Each socket's tick asks for a plan, and the plan itself runs at most once per
//! `PLAN_EVERY`, so the members' ticks do not plan twice. A changed plan is sent
//! to every npcd member at once. The plan's log lines are written only when the
//! cards or the weights changed since the line last written, since an unchanged
//! line every few seconds would bury everything else in the log.

use super::*;

/// Plan the cast now, under the state's hold, and send a changed plan to every
/// npcd member (`docs/static_placement.md` §5).
pub(super) fn plan_in(s: &mut State, now: Instant) {
    let views = s.registry.card_views();
    let planned = s.planner.plan(now, &views);
    if planned.changed {
        let frame = Frame::Plan(PlanFrame {
            epoch: planned.epoch,
            weights: planned.weights.clone(),
        });
        for ((a, _), socket) in &s.sockets {
            if a == HOSTING_APP {
                let _ = socket.handle.control(frame.clone());
            }
        }
        for ((a, _), p) in s.pending.iter_mut() {
            if a == HOSTING_APP {
                p.relay(frame.clone());
            }
        }
    }
    log_plan(&planned, &mut s.cards_logged);
}

/// The plan's log lines (`docs/static_placement.md` §5): the cards when any
/// of them changed since the line last written (`logged`) — the plan runs
/// every few seconds, and an unchanged line each time buried every other —
/// and the weights when they moved.
pub(super) fn log_plan(planned: &Planned, logged: &mut String) {
    let cards = planned
        .cards
        .iter()
        .map(|c| format!("{:?} latency={} busy={}", c.key, c.latency, c.busy))
        .collect::<Vec<String>>()
        .join("; ");
    if cards != *logged {
        tracing::info!(
            target: "cluster::gateway::plan",
            epoch = planned.epoch,
            cards = %cards,
            "cards"
        );
        *logged = cards;
    }
    if planned.changed {
        let weights: Vec<String> = planned
            .weights
            .iter()
            .map(|(node, w)| format!("{node:016x}={w}"))
            .collect();
        tracing::info!(
            target: "cluster::gateway::plan",
            epoch = planned.epoch,
            weights = %weights.join(" "),
            "plan"
        );
    }
}

impl Hub {
    /// Plan the cast, once each [`super::planner::PLAN_EVERY`] whichever
    /// socket's tick asks (`docs/static_placement.md` §5): a changed plan goes
    /// to every npcd member.
    pub(in super::super) fn plan_tick(&self) {
        let mut s = self.state();
        let now = Instant::now();
        if !s.planner.due(now) {
            return;
        }
        plan_in(&mut s, now);
    }
}
