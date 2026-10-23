//! Lowering `max_members` when the group stays short (§5.9).
//!
//! `max_members` only ever rises by itself, so a member that leaves for good —
//! a laptop off the network — leaves the quorum counted against a group it is
//! no longer part of. With two of three gone the rest cannot decide anything:
//! a member that wrote cut off cannot take its generation, and no verdict,
//! rollout or lease can be decided.
//!
//! The governor lowers it. While the ready members the gateways report stay
//! fewer than `max_members`, it **proposes** a lower value after
//! [`PROPOSE_AFTER`], saying so in its status and log, and **executes** it —
//! writes the *set* (§5.9) — once the shortfall has lasted [`EXECUTE_AFTER`].
//! The members' return at any point withdraws it, and a raise puts the value
//! back the moment they are seen ready again.
//!
//! - **Never below [`FLOOR`].** At two the gateway's tie-break vote completes a
//!   majority, so one member and the voting gateway decide; at one a lone
//!   member would decide everything by itself.
//! - **Not before every lease could have lapsed.** [`EXECUTE_AFTER`] is the
//!   longest term a lease is taken for, the upgrade install's. An absent member
//!   cannot renew without a quorum, so by then every lease it held has run out
//!   on every acceptor, and no lease is granted twice.
//! - **Only from a member that sees the whole group.** It acts only while it is
//!   joined, in no fault, not drained, and admitted and ready on every cluster
//!   it is configured for. A member cut off from a gateway sees part of the
//!   group, and lowering from a partial view would let both sides of a
//!   partition decide.

use std::time::{Duration, Instant};

use serde::Serialize;

use super::lease::LEASE_TTL as WORLD_LEASE_TTL;
use crate::upgrade::runtime::LEASE_TTL as UPGRADE_LEASE_TTL;

/// How long the group must stay short before a lower value is proposed: past
/// a few dropped sockets, which recover within a minute or two.
pub const PROPOSE_AFTER: Duration = Duration::from_secs(5 * 60);

/// How long the group must stay short before the lower value is written: the
/// longest lease term in use.
pub const EXECUTE_AFTER: Duration = if UPGRADE_LEASE_TTL.as_secs() > WORLD_LEASE_TTL.as_secs() {
    UPGRADE_LEASE_TTL
} else {
    WORLD_LEASE_TTL
};

/// The lowest value the governor writes.
pub const FLOOR: u32 = 2;

/// What the governor asks for after one look at the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Nothing to do.
    Hold,
    /// The group has been short long enough to say so: `max_members` will go
    /// from `from` to `to` once [`EXECUTE_AFTER`] has run, `executes_in` from
    /// now.
    Propose {
        from: u32,
        to: u32,
        executes_in: Duration,
    },
    /// Write `max_members` down from `from` to `to`.
    Execute { from: u32, to: u32 },
    /// The group is whole again, or this member can no longer see it: the
    /// proposal to go to `to` stands no more.
    Withdraw { to: u32 },
}

/// A downgrade under way, for `/v1/status`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Downgrade {
    pub from: u32,
    pub to: u32,
    /// How long the group has been short.
    pub short_for_s: u64,
    /// Whether it has been proposed yet.
    pub proposed: bool,
    pub executes_in_s: u64,
}

/// The shortfall being watched.
#[derive(Debug, Default)]
pub struct Governor {
    short_since: Option<Instant>,
    /// The value proposed, once it has been.
    proposed: Option<u32>,
}

/// The value `max_members` goes to with `ready` members seen.
fn target(ready: u32) -> u32 {
    ready.max(FLOOR)
}

impl Governor {
    /// Look at the group once: `max` is `max_members` as the substrate holds
    /// it, `ready` the ready members seen, and `eligible` whether this member
    /// sees the whole group (module documentation).
    pub fn observe(&mut self, now: Instant, max: u32, ready: u32, eligible: bool) -> Step {
        let to = target(ready);
        if !eligible || to >= max {
            self.short_since = None;
            return match self.proposed.take() {
                Some(to) => Step::Withdraw { to },
                None => Step::Hold,
            };
        }
        let since = *self.short_since.get_or_insert(now);
        let short_for = now.saturating_duration_since(since);
        if short_for >= EXECUTE_AFTER {
            self.short_since = None;
            self.proposed = None;
            return Step::Execute { from: max, to };
        }
        if short_for >= PROPOSE_AFTER && self.proposed != Some(to) {
            self.proposed = Some(to);
            return Step::Propose {
                from: max,
                to,
                executes_in: EXECUTE_AFTER - short_for,
            };
        }
        Step::Hold
    }

    /// The downgrade under way at `now`, if the group is short.
    pub fn view(&self, now: Instant, max: u32, ready: u32) -> Option<Downgrade> {
        let since = self.short_since?;
        let short_for = now.saturating_duration_since(since);
        Some(Downgrade {
            from: max,
            to: target(ready),
            short_for_s: short_for.as_secs(),
            proposed: self.proposed.is_some(),
            executes_in_s: EXECUTE_AFTER.saturating_sub(short_for).as_secs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: Duration = Duration::from_secs(60);

    #[test]
    fn the_execution_waits_out_the_longest_lease() {
        assert_eq!(EXECUTE_AFTER, UPGRADE_LEASE_TTL);
        assert_eq!(EXECUTE_AFTER, Duration::from_secs(20 * 60));
        assert!(EXECUTE_AFTER >= WORLD_LEASE_TTL);
        assert!(PROPOSE_AFTER < EXECUTE_AFTER);
    }

    /// A group of three with one ready member: proposed at five minutes,
    /// executed at twenty, down to two — never to one.
    #[test]
    fn a_short_group_is_proposed_then_lowered_to_the_floor() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        assert_eq!(g.observe(t0, 3, 1, true), Step::Hold);
        assert_eq!(g.observe(t0 + 4 * MIN, 3, 1, true), Step::Hold);
        assert_eq!(
            g.observe(t0 + 5 * MIN, 3, 1, true),
            Step::Propose {
                from: 3,
                to: 2,
                executes_in: 15 * MIN
            }
        );
        assert_eq!(
            g.observe(t0 + 6 * MIN, 3, 1, true),
            Step::Hold,
            "proposed once"
        );
        assert_eq!(
            g.observe(t0 + 20 * MIN, 3, 1, true),
            Step::Execute { from: 3, to: 2 }
        );
        assert!(g.view(t0 + 20 * MIN, 2, 1).is_none(), "done with it");
    }

    #[test]
    fn a_group_whole_again_withdraws_the_proposal() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(t0, 3, 2, true);
        assert!(matches!(
            g.observe(t0 + 5 * MIN, 3, 2, true),
            Step::Propose { to: 2, .. }
        ));
        assert_eq!(
            g.observe(t0 + 7 * MIN, 3, 3, true),
            Step::Withdraw { to: 2 }
        );
        assert_eq!(
            g.observe(t0 + 30 * MIN, 3, 2, true),
            Step::Hold,
            "the clock started again"
        );
    }

    #[test]
    fn a_member_that_cannot_see_the_whole_group_never_lowers() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(t0, 3, 1, true);
        assert_eq!(g.observe(t0 + 10 * MIN, 3, 1, false), Step::Hold);
        assert_eq!(
            g.observe(t0 + 21 * MIN, 3, 1, true),
            Step::Hold,
            "a break in the view starts the clock again"
        );
        assert!(g.view(t0 + 21 * MIN, 3, 1).is_some());
    }

    #[test]
    fn a_group_at_the_floor_or_whole_is_left_alone() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        for (max, ready) in [(2, 1), (2, 0), (3, 3), (4, 5)] {
            assert_eq!(g.observe(t0, max, ready, true), Step::Hold);
            assert_eq!(g.observe(t0 + 30 * MIN, max, ready, true), Step::Hold);
            assert!(g.view(t0 + 30 * MIN, max, ready).is_none());
        }
    }

    /// A five-member group with three ready lowers to three, keeping a
    /// majority of the members that remain.
    #[test]
    fn a_larger_group_lowers_to_the_members_it_still_has() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(t0, 5, 3, true);
        assert_eq!(
            g.observe(t0 + 20 * MIN, 5, 3, true),
            Step::Execute { from: 5, to: 3 }
        );
    }

    /// A member that leaves after the proposal changes what is proposed, and
    /// the new value is proposed again; the clock runs from the first shortfall.
    #[test]
    fn a_further_loss_is_proposed_again_on_the_same_clock() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(t0, 5, 4, true);
        assert!(matches!(
            g.observe(t0 + 5 * MIN, 5, 4, true),
            Step::Propose { to: 4, .. }
        ));
        assert_eq!(
            g.observe(t0 + 8 * MIN, 5, 3, true),
            Step::Propose {
                from: 5,
                to: 3,
                executes_in: 12 * MIN
            }
        );
    }

    #[test]
    fn the_view_counts_down_to_the_execution() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(t0, 3, 1, true);
        assert_eq!(
            g.view(t0 + 2 * MIN, 3, 1),
            Some(Downgrade {
                from: 3,
                to: 2,
                short_for_s: 120,
                proposed: false,
                executes_in_s: 18 * 60,
            })
        );
        g.observe(t0 + 6 * MIN, 3, 1, true);
        assert!(g.view(t0 + 6 * MIN, 3, 1).unwrap().proposed);
    }
}
