//! The gateway's upgrade state (§14.1 build floor, certification, forced
//! rollout): candidates, the grace period, verdicts, the floor rising, and the
//! members a rollout leaves behind.

use super::stats::Sample;
use super::*;
use crate::coord::{HoldLine, RolloutLine, VerdictLine};
use crate::upgrade::decisions::{Rollout, Verdict as Decided};

const MIN: u64 = 60_000;

fn b(count: u32) -> BuildId {
    BuildId {
        count,
        hash: format!("h{count}"),
        local: String::new(),
    }
}

fn local(count: u32) -> BuildId {
    BuildId {
        local: "abcd".into(),
        ..b(count)
    }
}

fn upgrades() -> Upgrades {
    Upgrades::new(UpgradeConfig {
        grace: Duration::from_secs(30 * 60),
        canary_pct: 10,
    })
}

fn verdicts(a: &[Action]) -> Vec<(u64, VerdictResult)> {
    a.iter()
        .filter_map(|x| match x {
            Action::Verdict { node, verdict, .. } => Some((*node, verdict.result)),
            _ => None,
        })
        .collect()
}

fn floors(a: &[Action]) -> Vec<BuildId> {
    a.iter()
        .filter_map(|x| match x {
            Action::Floor { build, .. } => Some(build.clone()),
            _ => None,
        })
        .collect()
}

fn sample(candidate: bool, failed: bool, ms: u32) -> Sample {
    Sample {
        route: "GET /v1/tools".into(),
        mirror: Mirror::None,
        candidate,
        failed,
        latency_ms: ms,
    }
}

fn settled(build: &str, result: Settled) -> VerdictLine {
    VerdictLine {
        build: build.into(),
        n: 1,
        verdict: Decided {
            result,
            by: "certification".into(),
            at_ms: 0,
            reason: "GET /v1/tools: too slow".into(),
            figures: vec![],
        },
    }
}

/// A higher build is the candidate, loading; ready, its grace starts. An even
/// newer one takes over the pending floor and keeps the deadline. A local
/// build, an agent, the failsafe and a build at or below the floor are never
/// candidates.
#[test]
fn a_newer_build_is_a_candidate_whose_grace_starts_when_ready() {
    let mut u = upgrades();
    let floor = b(10);
    assert_eq!(u.admitted("zend", 1, &b(10), Some(&floor), 0).len(), 0);
    assert_eq!(u.admitted("zend", 2, &local(12), Some(&floor), 0).len(), 0);
    assert_eq!(u.admitted("agent", 3, &b(12), Some(&floor), 0).len(), 0);
    assert_eq!(u.admitted("keep", 6, &b(12), Some(&floor), 0).len(), 0);
    assert_eq!(u.admitted("zend", 4, &b(12), Some(&floor), 0).len(), 1);
    assert_eq!(u.canary("zend"), None, "loading: no traffic, no grace");
    assert_eq!(u.view("zend").deadline_ms, None);
    u.capacity("zend", 4, MemberState::Ready, 5 * MIN);
    assert_eq!(u.canary("zend"), Some(4));
    assert_eq!(u.view("zend").deadline_ms, Some(35 * MIN));
    // An even newer member inside the window: pending floor up, deadline kept.
    u.admitted("zend", 5, &b(13), Some(&floor), 10 * MIN);
    assert_eq!(u.pending()["zend"], (b(13), Some(35 * MIN)));
    u.capacity("zend", 5, MemberState::Ready, 12 * MIN);
    assert_eq!(u.view("zend").deadline_ms, Some(35 * MIN));
    // Another app is untouched.
    assert!(!u.pending().contains_key("npcd"));
}

/// Before the deadline nothing is sent; a significant failure is sent at
/// once; at the deadline the verdict is sent, a pass with gaps as a pass.
#[test]
fn the_verdict_goes_at_the_deadline_or_at_once_on_a_failure() {
    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    assert!(u.tick(10 * MIN).is_empty());
    assert_eq!(verdicts(&u.tick(30 * MIN)), [(4, VerdictResult::NoOpinion)]);

    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    for _ in 0..300 {
        u.record("zend", sample(true, false, 10));
        u.record("zend", sample(false, false, 10));
    }
    assert!(u.tick(29 * MIN).is_empty());
    assert_eq!(verdicts(&u.tick(30 * MIN)), [(4, VerdictResult::Pass)]);
    assert!(u.tick(30 * MIN + 1_000).is_empty(), "judged once");
    assert_eq!(u.canary("zend"), None, "certified here: no more share");

    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    for _ in 0..300 {
        u.record("zend", sample(true, true, 10));
        u.record("zend", sample(false, false, 10));
    }
    assert_eq!(verdicts(&u.tick(MIN)), [(4, VerdictResult::Fail)]);
}

/// A verdict goes again every ten seconds until the group decides: a
/// candidate that missed it still hears it.
#[test]
fn the_verdict_goes_again_until_the_group_decides() {
    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    assert_eq!(verdicts(&u.tick(30 * MIN)), [(4, VerdictResult::NoOpinion)]);
    assert!(verdicts(&u.tick(30 * MIN + 5_000)).is_empty());
    assert_eq!(
        verdicts(&u.tick(30 * MIN + 10_000)),
        [(4, VerdictResult::NoOpinion)]
    );
    let pass = UpgradeView {
        verdicts: vec![settled("12-h12", Settled::Pass)],
        ..UpgradeView::default()
    };
    u.coord("zend", &pass, Some(&b(10)), 31 * MIN);
    assert!(verdicts(&u.tick(40 * MIN)).is_empty(), "decided: no more");
}

/// The group certifying the pending build raises the floor; rejecting it
/// cancels the pending floor and raises nothing.
#[test]
fn the_groups_verdict_raises_the_floor_or_rejects() {
    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    let pass = UpgradeView {
        verdicts: vec![settled("12-h12", Settled::Pass)],
        ..UpgradeView::default()
    };
    assert_eq!(
        floors(&u.coord("zend", &pass, Some(&b(10)), 40 * MIN)),
        [b(12)]
    );
    assert!(u.pending().is_empty());
    assert_eq!(u.view("zend").rollout_since_ms, Some(40 * MIN));
    assert!(
        floors(&u.coord("zend", &pass, Some(&b(12)), 41 * MIN)).is_empty(),
        "once"
    );

    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    let fail = UpgradeView {
        verdicts: vec![settled("12-h12", Settled::Fail)],
        ..UpgradeView::default()
    };
    assert!(floors(&u.coord("zend", &fail, Some(&b(10)), MIN)).is_empty());
    assert!(u.pending().is_empty());
    assert!(u.rejected("zend", &b(12)));
    // A rejected build is never a candidate again.
    assert!(u
        .admitted("zend", 4, &b(12), Some(&b(10)), 2 * MIN)
        .is_empty());
}

/// `raise-floor` ends the grace now; `cancel-floor` discards the pending
/// floor; both are refused without one.
#[test]
fn the_grace_can_be_ended_or_discarded_by_hand() {
    let mut u = upgrades();
    assert!(u.raise_floor("zend", 0).is_err());
    assert!(u.cancel_floor("zend").is_err());
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    assert!(u
        .raise_floor("zend", 0)
        .unwrap_err()
        .contains("still loading"));
    u.capacity("zend", 4, MemberState::Ready, 0);
    assert_eq!(
        verdicts(&u.raise_floor("zend", MIN).unwrap()),
        [(4, VerdictResult::NoOpinion)]
    );
    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    assert_eq!(u.cancel_floor("zend").unwrap().len(), 1);
    assert!(u.pending().is_empty());
}

/// A forced rollout moves the floor at once, in either direction, cancels
/// another candidate's certification, and marks what a downgrade rejects.
#[test]
fn a_forced_rollout_moves_the_floor_either_way() {
    let mut u = upgrades();
    u.admitted("zend", 4, &b(13), Some(&b(12)), 0);
    let down = UpgradeView {
        rollout: Some(RolloutLine {
            n: 1,
            decision: Rollout {
                build: "10-h10".into(),
                from: "0000000000000001".into(),
                targets: vec![],
                parallel: 1,
                artifacts: vec![],
                rejects: vec!["12-h12".into(), "13-h13".into()],
                by: "ops".into(),
                at_ms: 0,
            },
        }),
        ..UpgradeView::default()
    };
    assert_eq!(floors(&u.coord("zend", &down, Some(&b(12)), MIN)), [b(10)]);
    assert!(u.pending().is_empty());
    assert!(u.rejected("zend", &b(12)) && u.rejected("zend", &b(13)));
    assert!(
        floors(&u.coord("zend", &down, Some(&b(10)), 2 * MIN)).is_empty(),
        "once"
    );
}

/// Once the floor has risen and settled, a member below it that is stranded,
/// refused the build or fetches none of it is closed 4410 with its reason;
/// one installing is left alone; a local build is never closed. An upgrading
/// member stranded, or restored after its install failed, is closed at once.
/// With no one below, the rollout ends.
#[test]
fn members_left_behind_are_closed_when_the_rollout_settles() {
    let mut u = upgrades();
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    let pass = UpgradeView {
        verdicts: vec![settled("12-h12", Settled::Pass)],
        ..UpgradeView::default()
    };
    u.coord("zend", &pass, Some(&b(10)), 0);
    let stranded = UpgradeView {
        stranded: Some("build 12-h12 has no zend for linux-aarch64".into()),
        ..UpgradeView::default()
    };
    let idle = UpgradeView::default();
    let waiting = UpgradeView {
        step: "waiting".into(),
        holds: vec![HoldLine {
            build: "12-h12".into(),
            source: "verified".into(),
            ..HoldLine::default()
        }],
        ..UpgradeView::default()
    };
    let (b10, l9) = (b(10), local(9));
    let members = [
        Below {
            node: 1,
            build: &b10,
            state: MemberState::Ready,
            upgrade: Some(&stranded),
        },
        Below {
            node: 2,
            build: &b10,
            state: MemberState::Ready,
            upgrade: Some(&idle),
        },
        Below {
            node: 3,
            build: &b10,
            state: MemberState::Ready,
            upgrade: Some(&waiting),
        },
        Below {
            node: 5,
            build: &l9,
            state: MemberState::Ready,
            upgrade: Some(&idle),
        },
    ];
    assert!(
        u.left_behind("zend", &b(12), &members, MIN).is_empty(),
        "not settled yet"
    );
    let closed: Vec<(u64, u16, String)> = u
        .left_behind("zend", &b(12), &members, 3 * MIN)
        .into_iter()
        .filter_map(|a| match a {
            Action::Close { node, close, .. } => Some((node, close.code, close.reason)),
            _ => None,
        })
        .collect();
    assert_eq!(
        closed,
        [
            (
                1,
                4410,
                "build 10-h10 is below the zend floor 12-h12, and stranded: build 12-h12 has no \
                 zend for linux-aarch64; rebuild"
                    .to_string()
            ),
            (
                2,
                4410,
                "build 10-h10 is below the zend floor 12-h12, and it holds no artifact of 12-h12 \
                 and fetches none; rebuild"
                    .to_string()
            ),
        ]
    );
    let restored = UpgradeView {
        refused: vec!["12-h12".into()],
        ..UpgradeView::default()
    };
    let upgrading = [
        Below {
            node: 6,
            build: &b10,
            state: MemberState::Upgrading,
            upgrade: Some(&stranded),
        },
        Below {
            node: 7,
            build: &b10,
            state: MemberState::Upgrading,
            upgrade: Some(&restored),
        },
        Below {
            node: 8,
            build: &b10,
            state: MemberState::Upgrading,
            upgrade: Some(&idle),
        },
    ];
    let mut fresh = upgrades();
    let at_once: Vec<(u64, String)> = fresh
        .left_behind("zend", &b(12), &upgrading, 0)
        .into_iter()
        .filter_map(|a| match a {
            Action::Close { node, close, .. } => Some((node, close.reason)),
            _ => None,
        })
        .collect();
    assert_eq!(
        at_once,
        [
            (
                6,
                "build 10-h10 is below the zend floor 12-h12, and stranded: build 12-h12 has no \
                 zend for linux-aarch64; rebuild"
                    .to_string()
            ),
            (
                7,
                "build 10-h10 is below the zend floor 12-h12, and an install of 12-h12 failed \
                 here and was restored; rebuild"
                    .to_string()
            ),
        ]
    );
    let ended = u.left_behind("zend", &b(12), &[], 4 * MIN);
    assert!(matches!(&ended[..], [Action::Note { text, .. }] if text == "rollout of 12-h12 ended"));
}

/// Every tenth pick goes to the candidate while it certifies.
#[test]
fn the_candidate_takes_its_share() {
    let mut u = upgrades();
    assert!(!u.take_share("zend"));
    u.admitted("zend", 4, &b(12), Some(&b(10)), 0);
    u.capacity("zend", 4, MemberState::Ready, 0);
    let taken = (0..100).filter(|_| u.take_share("zend")).count();
    assert_eq!(taken, 10);
}
