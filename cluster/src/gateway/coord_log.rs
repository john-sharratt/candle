//! What a member's `COORD` adds to the coordination log (§12.3): the decisions
//! only the member sees, read off the difference between two successive
//! reports — a mode change, entering and leaving fault mode, the generation a
//! rejoin took, a reload, catch-ups starting and finishing, a lease held and
//! let go, `max_members` moving, retransmits, tombstones expiring, and the
//! maintenance commands run on it; and what changed of its worlds
//! ([`world_log`]).

use super::world_log;
use crate::coord::{CommandLine, CoordState};
use crate::replicate::{CatchupView, Purpose};

const MIB: u64 = 1 << 20;

fn purpose(p: Purpose) -> String {
    match p {
        Purpose::Admission => "admission".into(),
        Purpose::Gap(o) => format!("a gap in {o:016x}"),
        Purpose::Periodic => "periodic".into(),
        Purpose::Offer(o) => format!("an offer from {o:016x}"),
    }
}

fn entered(s: &CoordState) -> Option<String> {
    let f = s.fault.as_ref()?;
    Some(format!(
        "entered fault mode: {}; {} MiB free against a floor of {} MiB; gave up {} claims; still voting",
        f.reason,
        s.disk.free / MIB,
        s.disk.floor / MIB,
        f.claims_released
    ))
}

/// The lines for the commands in `after` that `before` did not have, or had
/// still running.
fn command_lines(before: &[CommandLine], after: &[CommandLine]) -> Vec<String> {
    let mut lines = Vec::new();
    for c in after {
        let was = before.iter().find(|b| b.job == c.job);
        if was.is_none() {
            lines.push(format!(
                "command {} (job {}) by {}, with {}",
                c.command, c.job, c.by, c.args
            ));
        }
        if let Some(outcome) = &c.outcome {
            if was.is_none_or(|b| b.outcome.is_none()) {
                lines.push(format!(
                    "command {} (job {}) ended: {outcome}",
                    c.command, c.job
                ));
            }
        }
    }
    lines
}

/// The log lines `after` adds to `before`, the member's previous report to
/// this gateway — `None` for its first.
pub fn changes(before: Option<&CoordState>, after: &CoordState) -> Vec<String> {
    let Some(before) = before else {
        let mut lines = vec![format!(
            "reporting: {}, generation {}, max_members {}",
            after.mode, after.gen.current, after.paxos.max_members
        )];
        lines.extend(entered(after));
        lines.extend(
            after
                .gen
                .last_rejoin
                .map(|k| format!("rejoined: took generation {k}")),
        );
        lines.extend(command_lines(&[], &after.commands));
        lines.extend(world_log::first(&after.worlds));
        return lines;
    };
    let mut lines = Vec::new();
    match (&before.fault, &after.fault) {
        (None, Some(_)) => lines.extend(entered(after)),
        (Some(_), None) => lines.push(format!(
            "left fault mode: {} MiB free, {} MiB freed; catching up",
            after.disk.free / MIB,
            before.fault.as_ref().map_or(0, |f| f.freed) / MIB
        )),
        _ => {
            if before.mode != after.mode {
                lines.push(format!("mode {} → {}", before.mode, after.mode));
            }
        }
    }
    if after.gen.last_rejoin != before.gen.last_rejoin {
        if let Some(k) = after.gen.last_rejoin {
            lines.push(format!("rejoined: took generation {k}"));
        }
    }
    let reloads = after.stats.reloads.saturating_sub(before.stats.reloads);
    if reloads > 0 {
        lines.push(format!(
            "reloaded{}: a turn arrived ranked below one it held",
            if reloads > 1 {
                format!(" {reloads} times")
            } else {
                String::new()
            }
        ));
    }
    let key = |c: &CatchupView| (c.cluster, c.purpose);
    for c in &after.catchups {
        if !before.catchups.iter().any(|b| key(b) == key(c)) {
            lines.push(format!(
                "catch-up started on cluster {}: {}",
                c.cluster,
                purpose(c.purpose)
            ));
        }
    }
    for c in &before.catchups {
        if !after.catchups.iter().any(|a| key(a) == key(c)) {
            lines.push(format!(
                "catch-up finished on cluster {}: {}, {} records fetched, {} already held",
                c.cluster,
                purpose(c.purpose),
                c.progress.fetched,
                c.progress.skipped
            ));
        }
    }
    let held = |s: &CoordState, key: &str| {
        s.paxos
            .keys
            .iter()
            .any(|k| k.key == key && k.held_ms.is_some())
    };
    for k in after.paxos.keys.iter().filter(|k| k.held_ms.is_some()) {
        if !held(before, &k.key) {
            lines.push(format!("lease on {} held", k.key));
        }
    }
    for k in before.paxos.keys.iter().filter(|k| k.held_ms.is_some()) {
        if !held(after, &k.key) {
            lines.push(format!("lease on {} released or run out", k.key));
        }
    }
    if after.paxos.max_members != before.paxos.max_members {
        lines.push(format!(
            "max_members {} → {}, quorum {}",
            before.paxos.max_members, after.paxos.max_members, after.paxos.quorum
        ));
    }
    let resent = after.stats.resent.saturating_sub(before.stats.resent);
    if resent > 0 {
        lines.push(format!("{resent} events retransmitted"));
    }
    // A held tombstone only ever leaves by expiring (§6.8).
    let expired = before.tombstones.held.saturating_sub(after.tombstones.held);
    if expired > 0 {
        lines.push(format!("{expired} tombstones expired"));
    }
    lines.extend(command_lines(&before.commands, &after.commands));
    lines.extend(world_log::changes(&before.worlds, &after.worlds));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{DiskView, FaultView, GenView, KeyLine, PaxosView, TombstoneView};
    use crate::replicate::{Progress, Stats};

    const GIB: u64 = 1 << 30;

    fn joined() -> CoordState {
        CoordState {
            mode: "joined".into(),
            gen: GenView {
                current: 1,
                last_rejoin: None,
                decided: 1,
            },
            paxos: PaxosView {
                max_members: 3,
                quorum: 2,
                keys: vec![],
            },
            disk: DiskView {
                free: 50 * GIB,
                total: 100 * GIB,
                floor: 2 * GIB,
            },
            ..CoordState::default()
        }
    }

    #[test]
    fn a_first_report_is_logged_once() {
        assert_eq!(
            changes(None, &joined()),
            ["reporting: joined, generation 1, max_members 3"]
        );
        assert!(changes(Some(&joined()), &joined()).is_empty());
    }

    #[test]
    fn entering_and_leaving_fault_mode() {
        let mut faulted = joined();
        faulted.mode = "fault".into();
        faulted.disk.free = GIB;
        faulted.fault = Some(FaultView {
            reason: "1024 MiB free, under the floor of 2048 MiB".into(),
            since_ms: 5,
            claims_released: 2,
            voting: true,
            freed: 0,
        });
        assert_eq!(
            changes(Some(&joined()), &faulted),
            [
                "entered fault mode: 1024 MiB free, under the floor of 2048 MiB; \
              1024 MiB free against a floor of 2048 MiB; gave up 2 claims; still voting"
            ]
        );
        // A member admitted already faulted says so in its first report.
        assert_eq!(changes(None, &faulted).len(), 2);

        faulted.fault.as_mut().unwrap().freed = 3 * GIB;
        let mut back = joined();
        back.disk.free = 5 * GIB;
        assert_eq!(
            changes(Some(&faulted), &back),
            ["left fault mode: 5120 MiB free, 3072 MiB freed; catching up"]
        );
    }

    #[test]
    fn a_drain_a_rejoin_and_a_reload() {
        let mut after = joined();
        after.mode = "draining".into();
        after.gen.last_rejoin = Some(4);
        after.stats = Stats {
            reloads: 2,
            ..Stats::default()
        };
        assert_eq!(
            changes(Some(&joined()), &after),
            [
                "mode joined → draining",
                "rejoined: took generation 4",
                "reloaded 2 times: a turn arrived ranked below one it held",
            ]
        );
    }

    #[test]
    fn catch_ups_quorum_retransmits_and_tombstones() {
        let line = |cluster, purpose| CatchupView {
            cluster,
            purpose,
            progress: Progress {
                fetched: 7,
                skipped: 3,
                ..Progress::default()
            },
            waiting: None,
        };
        let mut before = joined();
        before.catchups = vec![line(0, Purpose::Admission)];
        let mut after = joined();
        after.catchups = vec![line(1, Purpose::Gap(0xab))];
        after.paxos.max_members = 2;
        after.paxos.quorum = 2;
        after.stats.resent = 12;
        before.tombstones = TombstoneView {
            held: 6,
            waiting_age: 6,
        };
        after.tombstones = TombstoneView {
            held: 1,
            waiting_age: 0,
        };
        assert_eq!(
            changes(Some(&before), &after),
            [
                "catch-up started on cluster 1: a gap in 00000000000000ab",
                "catch-up finished on cluster 0: admission, 7 records fetched, 3 already held",
                "max_members 3 → 2, quorum 2",
                "12 events retransmitted",
                "5 tombstones expired",
            ]
        );
    }

    /// A lease the member gains, and one it no longer holds, as it reports
    /// them.
    #[test]
    fn leases_are_logged_as_they_are_gained_and_lost() {
        let lease = |held_ms| KeyLine {
            key: "world/w1".into(),
            held_ms,
            ..KeyLine::default()
        };
        let mut holding = joined();
        holding.paxos.keys = vec![lease(Some(9_000))];
        assert_eq!(
            changes(Some(&joined()), &holding),
            ["lease on world/w1 held"]
        );
        let mut renewed = joined();
        renewed.paxos.keys = vec![lease(Some(4_000))];
        assert!(changes(Some(&holding), &renewed).is_empty());
        let mut lost = joined();
        lost.paxos.keys = vec![lease(None)];
        assert_eq!(
            changes(Some(&renewed), &lost),
            ["lease on world/w1 released or run out"]
        );
    }

    /// A command is logged when it starts — who ran it, with what — and again
    /// when it ends; one seen only once it ended is logged both ways at once.
    #[test]
    fn commands_are_logged_as_they_start_and_end() {
        let line = |job, outcome: Option<&str>| CommandLine {
            job,
            command: "compact".into(),
            args: r#"{"app":"zend"}"#.into(),
            by: "ops@example.com".into(),
            started_ms: 1,
            outcome: outcome.map(String::from),
        };
        let mut running = joined();
        running.commands = vec![line(4, None)];
        assert_eq!(
            changes(Some(&joined()), &running),
            [r#"command compact (job 4) by ops@example.com, with {"app":"zend"}"#]
        );
        let mut ended = joined();
        ended.commands = vec![line(4, Some("ok"))];
        assert_eq!(
            changes(Some(&running), &ended),
            ["command compact (job 4) ended: ok"]
        );
        assert!(changes(Some(&ended), &ended).is_empty());
        assert_eq!(changes(Some(&joined()), &ended).len(), 2);
        // A first report says what ran before it, too.
        assert_eq!(changes(None, &ended).len(), 3);
    }
}
