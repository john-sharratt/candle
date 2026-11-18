//! The installs that failed here, and when each build may be tried again
//! (`docs/cluster_status_story.md` §2.1–§2.2).
//!
//! An install that does not prove itself is rolled back and the old build
//! restored (§3.9). That used to refuse the build here for good, so one missed
//! proof window — a trial on a machine short of memory while a person was
//! using it, WhiteWorld 2026-11-18 — left the member below the floor for as
//! long as its data directory lived. Each failure is now an attempt, kept in
//! `<data>/upgrade/attempts.json` with when it ended and how:
//!
//! - **slow** — the trial ran and did not prove itself in its window, keep's
//!   rollback; tried again after [`RETRY_AFTER_MS`], three times;
//! - **died** — the trial process exited first: tried again, and refused at
//!   once when a second death ends on the same last line, the crash being the
//!   build's own;
//! - **cancelled** — stopped by hand: no attempt is counted.
//!
//! A build is refused while its next try is not due, and for good once its
//! tries are spent. A build the member runs is forgiven; a forced rollout
//! naming the member tries it whatever its attempts say (§3.11).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::coord::RetryLine;

pub const ATTEMPTS: &str = "attempts.json";

/// After the first, second and third failed attempt: how long before the next.
pub const RETRY_AFTER_MS: [u64; 3] = [10 * 60_000, 30 * 60_000, 90 * 60_000];

/// How a failed attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailKind {
    Slow,
    Died,
    Cancelled,
}

impl FailKind {
    pub fn name(self) -> &'static str {
        match self {
            FailKind::Slow => "slow",
            FailKind::Died => "died",
            FailKind::Cancelled => "cancelled",
        }
    }
}

/// One failed attempt to install a build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    /// When the restored process recorded it: the retry waits are counted
    /// from here.
    pub ended_ms: u64,
    pub kind: FailKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_line: Option<String>,
    /// The install plan's own time, so a process restored once and started
    /// again by hand records its failure once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_ms: Option<u64>,
}

/// When a build may be tried again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextTry {
    /// Nothing holds it back.
    Now,
    /// Not before this instant.
    At(u64),
    /// Its tries are spent.
    Never,
}

/// Every build's failed attempts, oldest first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Attempts(BTreeMap<String, Vec<Attempt>>);

impl Attempts {
    /// What `<upgrade_dir>` holds; none when the file is absent or unreadable.
    pub fn read(upgrade_dir: &Path) -> Self {
        fs::read(upgrade_dir.join(ATTEMPTS))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Write it to `<upgrade_dir>`, whole, through a rename.
    pub fn write(&self, upgrade_dir: &Path) -> io::Result<()> {
        fs::create_dir_all(upgrade_dir)?;
        let tmp = upgrade_dir.join(format!("{ATTEMPTS}.tmp"));
        fs::write(
            &tmp,
            serde_json::to_vec(self).expect("attempts always serialize"),
        )?;
        fs::rename(tmp, upgrade_dir.join(ATTEMPTS))
    }

    /// An install of `build` failed: `attempt`. Whether it was new: an attempt
    /// naming a plan already recorded is the same failure seen again.
    pub fn record(&mut self, build: &str, attempt: Attempt) -> bool {
        let held = self.0.entry(build.to_string()).or_default();
        if attempt.plan_ms.is_some() && held.iter().any(|a| a.plan_ms == attempt.plan_ms) {
            return false;
        }
        held.push(attempt);
        true
    }

    /// The builds to report as retrying: each with its counted attempts, the
    /// newest's kind, and when it may be tried.
    pub fn retrying(&self) -> Vec<RetryLine> {
        self.schedule()
            .into_iter()
            .filter_map(|(build, next)| {
                let NextTry::At(at_ms) = next else {
                    return None;
                };
                let (attempts, newest) = self.of(&build);
                Some(RetryLine {
                    attempts: attempts as u32,
                    kind: newest.map_or("slow", |a| a.kind.name()).to_string(),
                    at_ms,
                    build,
                })
            })
            .collect()
    }

    /// The builds never to be tried again.
    pub fn refused_for_good(&self) -> Vec<String> {
        self.schedule()
            .into_iter()
            .filter(|(_, n)| *n == NextTry::Never)
            .map(|(b, _)| b)
            .collect()
    }

    /// `build` runs here: its failures no longer stand. Whether it had any.
    pub fn forgive(&mut self, build: &str) -> bool {
        self.0.remove(build).is_some()
    }

    /// When `build` may be tried again.
    pub fn next_try(&self, build: &str) -> NextTry {
        let counted: Vec<&Attempt> = self
            .0
            .get(build)
            .map(|a| a.iter().filter(|a| a.kind != FailKind::Cancelled).collect())
            .unwrap_or_default();
        let Some(last) = counted.last() else {
            return NextTry::Now;
        };
        if counted.len() > RETRY_AFTER_MS.len() {
            return NextTry::Never;
        }
        if let [.., before, last] = counted.as_slice() {
            if before.kind == FailKind::Died
                && last.kind == FailKind::Died
                && before.last_line.is_some()
                && before.last_line == last.last_line
            {
                return NextTry::Never;
            }
        }
        NextTry::At(last.ended_ms + RETRY_AFTER_MS[counted.len() - 1])
    }

    /// The builds not to be tried at `now_ms`, `count-hash`.
    pub fn refused_at(&self, now_ms: u64) -> BTreeSet<String> {
        self.0
            .keys()
            .filter(|b| match self.next_try(b) {
                NextTry::Now => false,
                NextTry::At(at) => now_ms < at,
                NextTry::Never => true,
            })
            .cloned()
            .collect()
    }

    /// `build`'s counted failures and the newest, for its report.
    pub fn of(&self, build: &str) -> (usize, Option<&Attempt>) {
        let all = self.0.get(build).map(Vec::as_slice).unwrap_or_default();
        let counted = all.iter().filter(|a| a.kind != FailKind::Cancelled).count();
        (counted, all.last())
    }

    /// Every build with an attempt, and its next try.
    pub fn schedule(&self) -> Vec<(String, NextTry)> {
        self.0
            .keys()
            .map(|b| (b.clone(), self.next_try(b)))
            .collect()
    }
}

/// How a restored process's failure record names its end: a trial that
/// exited is `died`, a stopped one `cancelled`, anything else — keep's
/// rollback for a missed proof window, and a restore with no record at all —
/// `slow`.
pub fn kind_of(reason: Option<&str>) -> FailKind {
    let Some(reason) = reason.map(str::to_ascii_lowercase) else {
        return FailKind::Slow;
    };
    if reason.contains("cancel") {
        FailKind::Cancelled
    } else if reason.contains("exited") || reason.contains("died") || reason.contains("crash") {
        FailKind::Died
    } else {
        FailKind::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u64 = 60_000;

    fn slow(at: u64) -> Attempt {
        Attempt {
            ended_ms: at,
            kind: FailKind::Slow,
            last_line: None,
            plan_ms: None,
        }
    }

    fn died(at: u64, line: &str) -> Attempt {
        Attempt {
            ended_ms: at,
            kind: FailKind::Died,
            last_line: Some(line.into()),
            plan_ms: None,
        }
    }

    /// **One install's failure is recorded once**, however often the process
    /// it restored is started again; another install's is its own.
    #[test]
    fn a_failure_is_recorded_once_per_install() {
        let mut a = Attempts::default();
        let of_plan = |at, plan| Attempt {
            plan_ms: Some(plan),
            ..slow(at)
        };
        assert!(a.record("5-h", of_plan(100, 7)));
        assert!(!a.record("5-h", of_plan(900, 7)), "the same install");
        assert!(a.record("5-h", of_plan(2_000_000, 8)));
        assert_eq!(a.of("5-h").0, 2);
    }

    /// The report names a build waiting for its retry, with when, and keeps
    /// one refused for good apart: only that one the gateway closes a member
    /// over.
    #[test]
    fn the_report_tells_a_retry_from_a_refusal() {
        let mut a = Attempts::default();
        a.record("5-h", slow(0));
        for at in [0, MIN, 2 * MIN, 3 * MIN] {
            a.record("6-h", slow(at));
        }
        assert_eq!(
            a.retrying(),
            [RetryLine {
                build: "5-h".into(),
                attempts: 1,
                kind: "slow".into(),
                at_ms: 10 * MIN,
            }]
        );
        assert_eq!(a.refused_for_good(), ["6-h"]);
    }

    /// **A slow failure is tried again three times, at 10, 30 and 90
    /// minutes**, then never.
    #[test]
    fn a_slow_build_is_tried_three_more_times_then_refused() {
        let mut a = Attempts::default();
        assert_eq!(a.next_try("5-h"), NextTry::Now);
        a.record("5-h", slow(0));
        assert_eq!(a.next_try("5-h"), NextTry::At(10 * MIN));
        assert!(a.refused_at(10 * MIN - 1).contains("5-h"));
        assert!(!a.refused_at(10 * MIN).contains("5-h"));
        a.record("5-h", slow(20 * MIN));
        assert_eq!(a.next_try("5-h"), NextTry::At(50 * MIN));
        a.record("5-h", slow(60 * MIN));
        assert_eq!(a.next_try("5-h"), NextTry::At(150 * MIN));
        a.record("5-h", slow(200 * MIN));
        assert_eq!(a.next_try("5-h"), NextTry::Never);
        assert!(a.refused_at(u64::MAX).contains("5-h"));
    }

    /// **Two deaths on the same last line refuse the build at once**; deaths
    /// on different lines are retried like any failure.
    #[test]
    fn a_repeated_death_is_the_builds_own() {
        let mut a = Attempts::default();
        a.record("7-h", died(0, "thread 'main' panicked at x.rs:1"));
        assert_eq!(a.next_try("7-h"), NextTry::At(10 * MIN));
        a.record("7-h", died(11 * MIN, "thread 'main' panicked at x.rs:1"));
        assert_eq!(a.next_try("7-h"), NextTry::Never);
        let mut b = Attempts::default();
        b.record("7-h", died(0, "out of memory"));
        b.record("7-h", died(11 * MIN, "a different line"));
        assert_eq!(b.next_try("7-h"), NextTry::At(41 * MIN));
    }

    /// A cancelled trial counts no attempt; a forgiven build has none.
    #[test]
    fn a_cancel_counts_nothing_and_running_forgives() {
        let mut a = Attempts::default();
        a.record(
            "9-h",
            Attempt {
                ended_ms: 5,
                kind: FailKind::Cancelled,
                last_line: None,
                plan_ms: None,
            },
        );
        assert_eq!(a.next_try("9-h"), NextTry::Now);
        assert_eq!(a.of("9-h").0, 0);
        a.record("9-h", slow(6));
        assert!(a.forgive("9-h"));
        assert_eq!(a.next_try("9-h"), NextTry::Now);
        assert!(!a.forgive("9-h"));
    }

    /// The file is the record, read back exactly, in this raw form.
    #[test]
    fn the_attempts_read_back_from_their_file() {
        let t = tempfile::TempDir::new().unwrap();
        assert_eq!(Attempts::read(t.path()), Attempts::default());
        let mut a = Attempts::default();
        a.record("5-h", slow(1));
        a.record("5-h", died(2, "x"));
        a.write(t.path()).unwrap();
        assert_eq!(
            fs::read_to_string(t.path().join(ATTEMPTS)).unwrap(),
            r#"{"5-h":[{"ended_ms":1,"kind":"slow"},{"ended_ms":2,"kind":"died","last_line":"x"}]}"#
        );
        assert_eq!(Attempts::read(t.path()), a);
    }

    /// A restore's record names its kind; none at all is keep's rollback.
    #[test]
    fn a_failures_reason_names_its_kind() {
        assert_eq!(kind_of(None), FailKind::Slow);
        assert_eq!(kind_of(Some("not ready in time")), FailKind::Slow);
        assert_eq!(
            kind_of(Some("the new process exited (code 3)")),
            FailKind::Died
        );
        assert_eq!(kind_of(Some("cancelled by an admin")), FailKind::Cancelled);
    }
}
