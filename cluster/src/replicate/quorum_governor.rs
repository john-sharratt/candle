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
//! writes the *set* (§5.9) — once no lease granted under the old quorum can be
//! granted again under the new one. The members' return at any point withdraws
//! it, and a raise puts the value back the moment they are seen ready again.
//!
//! **When it executes.**
//!
//! - **Down to [`FLOOR`] through the voting gateway: as soon as the absent
//!   members' leases have lapsed.** This member is joined through one cluster,
//!   and a reply flagged as the tie-break has arrived on its socket lately, so
//!   its gateway is the voter: it answers every round that reaches it, and so
//!   accepted every lease and decision the group was asked for. Every quorum of
//!   the lowered group is two of the remaining members, or one and that gateway.
//!   A lease granted under the old quorum was accepted by its holder and another
//!   member, and by the gateway — so a new quorum always holds an acceptor that
//!   accepted it, and an acceptor reports a lease for its whole term: the round
//!   finds it busy rather than granting it twice. The write waits besides for
//!   the leases this member's own acceptor holds for members not ready: none,
//!   at once; a world lease, at most its 30 s term.
//! - **Otherwise, after [`EXECUTE_AFTER`]**, the longest term a lease is taken
//!   for (the upgrade install's). A larger group lowered to more than two, or
//!   one joined through several clusters, may have granted a lease through
//!   acceptors none of whose votes this member can read; by then every lease an
//!   absent member held has run out on every acceptor, since it cannot renew
//!   without a quorum.
//!
//! **What it never does.**
//!
//! - Lower below [`FLOOR`]. At two the voting gateway's tie-break completes a
//!   majority, so one member and the gateway decide; at one a lone member would
//!   decide everything by itself.
//! - Act from a partial view. It acts only while this member is joined, in no
//!   fault, not drained, and admitted and ready on every cluster it is
//!   configured for. A member cut off from a gateway sees part of the group, and
//!   lowering from a partial view would let both sides of a partition decide.

use std::time::{Duration, Instant};

use serde::Serialize;

use super::lease::LEASE_TTL as WORLD_LEASE_TTL;
use crate::upgrade::runtime::LEASE_TTL as UPGRADE_LEASE_TTL;

/// How long the group must stay short before a lower value is proposed: past
/// a few dropped sockets, which recover within a minute or two.
pub const PROPOSE_AFTER: Duration = Duration::from_secs(5 * 60);

/// How long the group must stay short before the lower value is written when
/// the leases cannot be read off this member's acceptor: the longest lease
/// term in use.
pub const EXECUTE_AFTER: Duration = if UPGRADE_LEASE_TTL.as_secs() > WORLD_LEASE_TTL.as_secs() {
    UPGRADE_LEASE_TTL
} else {
    WORLD_LEASE_TTL
};

/// The lowest value the governor writes.
pub const FLOOR: u32 = 2;

/// One look at the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Look {
    pub now: Instant,
    /// `max_members` as the substrate holds it.
    pub max: u32,
    /// The distinct ready members seen, this one included.
    pub ready: u32,
    /// Whether this member sees the whole group (module documentation).
    pub eligible: bool,
    /// Whether this member is joined through one cluster whose gateway is the
    /// tie-break voter — a reply flagged as its vote arrived lately — and so
    /// the acceptor of every round.
    pub one_gateway: bool,
    /// The longest term left on a lease this member's acceptor holds for a
    /// member that is not ready; `None` when it holds none.
    pub absent_lease_left: Option<Duration>,
}

impl Look {
    /// The value `max_members` goes to.
    fn target(&self) -> u32 {
        self.ready.max(FLOOR)
    }

    /// Whether the absent members' leases can be read off this member's
    /// acceptor (module documentation).
    fn leases_known(&self) -> bool {
        self.target() == FLOOR && self.one_gateway
    }
}

/// What the governor asks for after one look at the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Nothing to do.
    Hold,
    /// The group has been short long enough to say so: `max_members` will go
    /// from `from` to `to`, `executes_in` from now.
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
    /// Whether it waits on the absent members' leases as this member's
    /// acceptor holds them, rather than on the longest lease term.
    pub by_leases: bool,
}

/// The shortfall being watched.
#[derive(Debug, Default)]
pub struct Governor {
    short_since: Option<Instant>,
    /// The value proposed, once it has been.
    proposed: Option<u32>,
    /// The last look while short, for the view.
    last: Option<Look>,
}

/// How long after `look` the write is due, given how long the group has been
/// short.
fn executes_in(look: &Look, short_for: Duration) -> Duration {
    if look.leases_known() {
        let earliest = PROPOSE_AFTER.saturating_sub(short_for);
        earliest.max(look.absent_lease_left.unwrap_or_default())
    } else {
        EXECUTE_AFTER.saturating_sub(short_for)
    }
}

impl Governor {
    /// Look at the group once.
    pub fn observe(&mut self, look: Look) -> Step {
        let to = look.target();
        if !look.eligible || to >= look.max {
            self.short_since = None;
            self.last = None;
            return match self.proposed.take() {
                Some(to) => Step::Withdraw { to },
                None => Step::Hold,
            };
        }
        let since = *self.short_since.get_or_insert(look.now);
        self.last = Some(look);
        let short_for = look.now.saturating_duration_since(since);
        if short_for < PROPOSE_AFTER {
            return Step::Hold;
        }
        let due_in = executes_in(&look, short_for);
        if self.proposed != Some(to) {
            self.proposed = Some(to);
            return Step::Propose {
                from: look.max,
                to,
                executes_in: due_in,
            };
        }
        if due_in.is_zero() {
            self.short_since = None;
            self.proposed = None;
            self.last = None;
            return Step::Execute { from: look.max, to };
        }
        Step::Hold
    }

    /// The downgrade under way, as of the last look, if the group is short.
    pub fn view(&self, now: Instant) -> Option<Downgrade> {
        let since = self.short_since?;
        let look = self.last?;
        let short_for = now.saturating_duration_since(since);
        Some(Downgrade {
            from: look.max,
            to: look.target(),
            short_for_s: short_for.as_secs(),
            proposed: self.proposed.is_some(),
            executes_in_s: executes_in(&look, short_for).as_secs(),
            by_leases: look.leases_known(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: Duration = Duration::from_secs(60);
    const SEC: Duration = Duration::from_secs(1);

    /// A three-member group through one gateway with one member ready and no
    /// absent leases.
    fn look(now: Instant) -> Look {
        Look {
            now,
            max: 3,
            ready: 1,
            eligible: true,
            one_gateway: true,
            absent_lease_left: None,
        }
    }

    #[test]
    fn the_fallback_waits_out_the_longest_lease() {
        assert_eq!(EXECUTE_AFTER, UPGRADE_LEASE_TTL);
        assert_eq!(EXECUTE_AFTER, Duration::from_secs(20 * 60));
        assert!(EXECUTE_AFTER >= WORLD_LEASE_TTL);
        assert!(PROPOSE_AFTER < EXECUTE_AFTER);
    }

    /// With no lease held for an absent member, a three-member group is
    /// proposed down to two at five minutes and lowered on the next look.
    #[test]
    fn with_no_absent_lease_it_lowers_right_after_proposing() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        assert_eq!(g.observe(look(t0)), Step::Hold);
        assert_eq!(g.observe(look(t0 + 4 * MIN)), Step::Hold);
        assert_eq!(
            g.observe(look(t0 + 5 * MIN)),
            Step::Propose {
                from: 3,
                to: 2,
                executes_in: Duration::ZERO
            }
        );
        assert_eq!(
            g.observe(look(t0 + 5 * MIN + SEC)),
            Step::Execute { from: 3, to: 2 }
        );
        assert!(g.view(t0 + 6 * MIN).is_none(), "done with it");
    }

    /// An absent member's lease holds the write until its term has run out.
    #[test]
    fn an_absent_members_lease_holds_the_write_until_it_lapses() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        let held = |now, left| Look {
            absent_lease_left: Some(left),
            ..look(now)
        };
        g.observe(held(t0, 30 * SEC));
        assert_eq!(
            g.observe(held(t0 + 5 * MIN, 20 * SEC)),
            Step::Propose {
                from: 3,
                to: 2,
                executes_in: 20 * SEC
            }
        );
        assert_eq!(
            g.observe(held(t0 + 5 * MIN + 10 * SEC, 10 * SEC)),
            Step::Hold
        );
        let v = g.view(t0 + 5 * MIN + 10 * SEC).unwrap();
        assert!(v.by_leases && v.proposed);
        assert_eq!(v.executes_in_s, 10);
        assert_eq!(
            g.observe(look(t0 + 5 * MIN + 20 * SEC)),
            Step::Execute { from: 3, to: 2 }
        );
    }

    /// Lowering to more than two, or through several clusters, cannot read the
    /// leases off this member's acceptor: it waits the longest term.
    #[test]
    fn otherwise_it_waits_out_the_longest_lease() {
        for (max, ready, one_gateway) in [(5, 3, true), (3, 1, false)] {
            let mut g = Governor::default();
            let t0 = Instant::now();
            let l = |now| Look {
                max,
                ready,
                one_gateway,
                ..look(now)
            };
            g.observe(l(t0));
            assert_eq!(
                g.observe(l(t0 + 5 * MIN)),
                Step::Propose {
                    from: max,
                    to: ready.max(FLOOR),
                    executes_in: 15 * MIN
                }
            );
            assert_eq!(g.observe(l(t0 + 19 * MIN)), Step::Hold);
            assert!(!g.view(t0 + 19 * MIN).unwrap().by_leases);
            assert_eq!(
                g.observe(l(t0 + 20 * MIN)),
                Step::Execute {
                    from: max,
                    to: ready.max(FLOOR)
                }
            );
        }
    }

    #[test]
    fn a_group_whole_again_withdraws_the_proposal() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        let held = |now| Look {
            absent_lease_left: Some(MIN),
            ..look(now)
        };
        g.observe(held(t0));
        assert!(matches!(
            g.observe(held(t0 + 5 * MIN)),
            Step::Propose { to: 2, .. }
        ));
        assert_eq!(
            g.observe(Look {
                ready: 3,
                ..look(t0 + 5 * MIN + SEC)
            }),
            Step::Withdraw { to: 2 }
        );
        assert_eq!(
            g.observe(look(t0 + 6 * MIN)),
            Step::Hold,
            "the clock started again"
        );
    }

    #[test]
    fn a_member_that_cannot_see_the_whole_group_never_lowers() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(look(t0));
        assert_eq!(
            g.observe(Look {
                eligible: false,
                ..look(t0 + 4 * MIN)
            }),
            Step::Hold
        );
        assert_eq!(
            g.observe(look(t0 + 6 * MIN)),
            Step::Hold,
            "a break in the view starts the clock again"
        );
        assert!(g.view(t0 + 6 * MIN).is_some());
    }

    #[test]
    fn a_group_at_the_floor_or_whole_is_left_alone() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        for (max, ready) in [(2, 1), (2, 0), (3, 3), (4, 5)] {
            let l = |now| Look {
                max,
                ready,
                ..look(now)
            };
            assert_eq!(g.observe(l(t0)), Step::Hold);
            assert_eq!(g.observe(l(t0 + 30 * MIN)), Step::Hold);
            assert!(g.view(t0 + 30 * MIN).is_none());
        }
    }

    /// A member that leaves after the proposal changes what is proposed, and
    /// the new value is proposed again on the same clock.
    #[test]
    fn a_further_loss_is_proposed_again_on_the_same_clock() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        let l = |now, ready| Look {
            max: 5,
            ready,
            ..look(now)
        };
        g.observe(l(t0, 4));
        assert!(matches!(
            g.observe(l(t0 + 5 * MIN, 4)),
            Step::Propose { to: 4, .. }
        ));
        assert_eq!(
            g.observe(l(t0 + 8 * MIN, 3)),
            Step::Propose {
                from: 5,
                to: 3,
                executes_in: 12 * MIN
            }
        );
    }

    #[test]
    fn the_view_counts_down_before_the_proposal() {
        let mut g = Governor::default();
        let t0 = Instant::now();
        g.observe(look(t0));
        assert_eq!(
            g.view(t0 + 2 * MIN),
            Some(Downgrade {
                from: 3,
                to: 2,
                short_for_s: 120,
                proposed: false,
                executes_in_s: 3 * 60,
                by_leases: true,
            })
        );
    }
}
