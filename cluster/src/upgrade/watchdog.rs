//! The upgrade's watchdog (§3.9 step 4): a process apart from the daemon, so the
//! old daemon can exit — releasing its VRAM, which only process exit does —
//! before the new one loads. It is the old executable, from its kept-build
//! directory, run as `--upgrade-watchdog <plan>`; it loads no model, opens no
//! substrate and dials no cluster.
//!
//! It waits for the old daemon to be gone, starts the new build with the same
//! launch, and watches it:
//!
//! - **ready** — the new process writes `<data>/upgrade/ok` once admitted and
//!   ready: the swap stands, the kept builds beyond `--keep-builds` are pruned,
//!   and the watchdog exits;
//! - **failed** — the new process exits, or is not ready within
//!   [`READY_WITHIN`]: it is stopped, the old build is put back at its paths,
//!   the failure is written to `<data>/upgrade/failed.json` with the new
//!   process's last log lines, and the old build is started again with
//!   `--upgrade-failed <build>`, so it never tries that build again.
//!
//! The process operations are behind [`Processes`], so the state machine is
//! tested by asserting what it spawns, never by spawning.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::kept::{builds_dir, prune};
use super::swap::{restore, Swapped};
use crate::frame::BuildId;
use crate::launch::{value_of, LaunchSpec, UPGRADE_FAILED, UPGRADE_WATCHDOG};

/// How long the new build has to become ready.
pub const READY_WITHIN: Duration = Duration::from_secs(15 * 60);

/// How long the old daemon has to exit after starting the watchdog.
pub const OLD_EXIT_WITHIN: Duration = Duration::from_secs(120);

/// How often the watchdog looks.
pub const LOOK: Duration = Duration::from_millis(100);

/// How many of the new process's last log lines a failure carries.
pub const LAST_LINES: usize = 40;

/// The upgrade's files, under `<data>/upgrade`.
pub const OK: &str = "ok";
pub const FAILED: &str = "failed.json";
pub const PLAN: &str = "watch.json";

/// What the old daemon hands its watchdog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchPlan {
    /// The daemon that started the watchdog, which exits first.
    pub old_pid: u32,
    /// The launch the new build starts with: the old one's, at the same path.
    pub launch: LaunchSpec,
    pub swapped: Swapped,
    /// The build installed, `count-hash`.
    pub installing: String,
    /// `<data>/upgrade`.
    pub upgrade_dir: PathBuf,
    /// The new process's log, whose last lines a failure reports.
    pub log: Option<PathBuf>,
    pub ready_within_ms: u64,
    /// `--keep-builds`, and the builds never pruned (a rollout's target).
    pub keep_builds: usize,
    pub protect: Vec<BuildId>,
    /// The upgrade leases held for the install, which the process that starts
    /// next — new or restored — takes back and releases once it is ready.
    pub leases: Vec<String>,
}

/// A failed install, as the restored member reports it (§3.9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    pub build: String,
    pub reason: String,
    pub last_lines: Vec<String>,
}

/// How the watch ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Ready,
    Restored(String),
}

/// The process operations the watchdog uses.
pub trait Processes {
    type Child;
    fn alive(&self, pid: u32) -> bool;
    fn kill_pid(&self, pid: u32);
    fn spawn(&self, launch: &LaunchSpec) -> io::Result<Self::Child>;
    /// Whether `child` has exited.
    fn exited(&self, child: &mut Self::Child) -> bool;
    fn kill(&self, child: &mut Self::Child);
    /// Let `d` pass.
    fn sleep(&self, d: Duration);
}

/// The last `n` lines of the file at `path`.
pub fn last_lines(path: &Path, n: usize) -> Vec<String> {
    let Ok(f) = File::open(path) else {
        return Vec::new();
    };
    let mut lines = VecDeque::with_capacity(n);
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if lines.len() == n {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    lines.into()
}

/// Run `plan` to its end.
pub fn watch<P: Processes>(plan: &WatchPlan, p: &P) -> Outcome {
    let ok = plan.upgrade_dir.join(OK);
    let mut waited = Duration::ZERO;
    while p.alive(plan.old_pid) {
        if waited >= OLD_EXIT_WITHIN {
            p.kill_pid(plan.old_pid);
            break;
        }
        p.sleep(LOOK);
        waited += LOOK;
    }
    let _ = fs::remove_file(&ok);
    let mut child = match p.spawn(&plan.launch) {
        Ok(c) => c,
        Err(e) => return fail(plan, p, format!("the new build did not start: {e}")),
    };
    let within = Duration::from_millis(plan.ready_within_ms);
    let mut waited = Duration::ZERO;
    loop {
        if ok.exists() {
            let builds = builds_dir(&plan.swapped.exe);
            if let Err(e) = prune(&builds, plan.keep_builds, &plan.protect) {
                tracing::warn!("the kept builds were not pruned: {e}");
            }
            return Outcome::Ready;
        }
        if p.exited(&mut child) {
            return fail(plan, p, "the new build exited before it was ready".into());
        }
        if waited >= within {
            p.kill(&mut child);
            return fail(
                plan,
                p,
                format!("the new build was not ready within {}", spell(within)),
            );
        }
        p.sleep(LOOK);
        waited += LOOK;
    }
}

/// A wait as a person reads it: whole minutes, or seconds.
fn spell(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 && s.is_multiple_of(60) {
        format!("{} minutes", s / 60)
    } else {
        format!("{s} s")
    }
}

/// Put the old build back, say why, and start it again.
fn fail<P: Processes>(plan: &WatchPlan, p: &P, reason: String) -> Outcome {
    let last = plan
        .log
        .as_deref()
        .map(|l| last_lines(l, LAST_LINES))
        .unwrap_or_default();
    let reason = match restore(&plan.swapped) {
        Ok(()) => reason,
        Err(e) => format!("{reason}; the old build could not be put back: {e}"),
    };
    let failure = Failure {
        build: plan.installing.clone(),
        reason: reason.clone(),
        last_lines: last,
    };
    let _ = fs::create_dir_all(&plan.upgrade_dir);
    if let Err(e) = fs::write(
        plan.upgrade_dir.join(FAILED),
        serde_json::to_vec_pretty(&failure).expect("a failure always serializes"),
    ) {
        tracing::error!("the install's failure was not recorded: {e}");
    }
    let again = plan
        .launch
        .with_args(&[UPGRADE_FAILED.flag, &plan.installing]);
    if let Err(e) = p.spawn(&again) {
        tracing::error!("the old build did not start again: {e}");
    }
    Outcome::Restored(reason)
}

/// The plan file `--upgrade-watchdog` names, when `args` — the process's,
/// without its own name — ask for the watchdog.
pub fn asks_for_watchdog(args: &[OsString]) -> Option<PathBuf> {
    match args {
        [flag, _] if flag == UPGRADE_WATCHDOG.flag => {
            value_of(args, UPGRADE_WATCHDOG.flag).map(PathBuf::from)
        }
        _ => None,
    }
}

/// Write the plan where the watchdog will read it.
pub fn write_plan(plan: &WatchPlan) -> io::Result<PathBuf> {
    fs::create_dir_all(&plan.upgrade_dir)?;
    let path = plan.upgrade_dir.join(PLAN);
    fs::write(
        &path,
        serde_json::to_vec_pretty(plan).expect("a plan always serializes"),
    )?;
    Ok(path)
}

pub fn read_plan(path: &Path) -> io::Result<WatchPlan> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

    use tempfile::TempDir;

    use crate::upgrade::kept::{write_json, BuildJson};
    use crate::version::info::{BuildInfo, Opens};

    /// What the watchdog did, in order.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Did {
        Spawn(Vec<String>),
        Kill,
        KillPid(u32),
    }

    /// A scripted machine: the old pid lives for `old_lives` looks; the new
    /// child writes `ok` after `ready_after` looks, or exits after
    /// `exits_after`.
    struct Fake {
        old_lives: RefCell<u32>,
        ready_after: Option<u32>,
        exits_after: Option<u32>,
        looks: RefCell<u32>,
        ok: PathBuf,
        did: RefCell<Vec<Did>>,
    }

    impl Processes for Fake {
        type Child = ();
        fn alive(&self, _pid: u32) -> bool {
            let mut left = self.old_lives.borrow_mut();
            let alive = *left > 0;
            *left = left.saturating_sub(1);
            alive
        }
        fn kill_pid(&self, pid: u32) {
            self.did.borrow_mut().push(Did::KillPid(pid));
        }
        fn spawn(&self, launch: &LaunchSpec) -> io::Result<()> {
            self.did.borrow_mut().push(Did::Spawn(
                launch
                    .args
                    .iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect(),
            ));
            *self.looks.borrow_mut() = 0;
            Ok(())
        }
        fn exited(&self, _: &mut ()) -> bool {
            let looks = *self.looks.borrow();
            if self.ready_after == Some(looks) {
                fs::write(&self.ok, b"").unwrap();
            }
            self.exits_after.is_some_and(|n| looks >= n)
        }
        fn kill(&self, _: &mut ()) {
            self.did.borrow_mut().push(Did::Kill);
        }
        fn sleep(&self, _: Duration) {
            *self.looks.borrow_mut() += 1;
        }
    }

    /// A swapped machine: the new build at `bin/zend`, the old kept in
    /// `bin/builds/10-h10`.
    fn swapped(t: &TempDir) -> WatchPlan {
        let exe = t.path().join("bin/zend");
        let kept = t.path().join("bin/builds/10-h10");
        fs::create_dir_all(&kept).unwrap();
        fs::write(&exe, "new").unwrap();
        fs::write(kept.join("zend"), "old").unwrap();
        write_json(
            &kept,
            &BuildJson {
                info: BuildInfo {
                    kind: "zend".into(),
                    count: 10,
                    hash: "h10".into(),
                    local: String::new(),
                    platform: "p".into(),
                    opens: Opens::default(),
                },
                files: vec![],
                installed_ms: 1,
                retired_ms: 2,
            },
        )
        .unwrap();
        let log = t.path().join("zend.log");
        fs::write(
            &log,
            (1..=50).map(|i| format!("line {i}\n")).collect::<String>(),
        )
        .unwrap();
        WatchPlan {
            old_pid: 4242,
            launch: LaunchSpec {
                exe: exe.clone(),
                args: vec!["--port".into(), "8080".into()],
                cwd: t.path().to_path_buf(),
            },
            swapped: Swapped {
                exe,
                kept,
                from_kept: None,
            },
            installing: "12-h12".into(),
            upgrade_dir: t.path().join("data/upgrade"),
            log: Some(log),
            ready_within_ms: 1_000,
            keep_builds: 5,
            protect: vec![],
            leases: vec!["upgrade/zend".into()],
        }
    }

    fn fake(plan: &WatchPlan, ready_after: Option<u32>, exits_after: Option<u32>) -> Fake {
        Fake {
            old_lives: RefCell::new(3),
            ready_after,
            exits_after,
            looks: RefCell::new(0),
            ok: plan.upgrade_dir.join(OK),
            did: RefCell::new(Vec::new()),
        }
    }

    /// The new build starts only once the old daemon is gone, with the same
    /// arguments, and a ready one stands: nothing restored, nothing started
    /// again.
    #[test]
    fn a_ready_build_stands() {
        let t = TempDir::new().unwrap();
        let plan = swapped(&t);
        fs::create_dir_all(&plan.upgrade_dir).unwrap();
        let p = fake(&plan, Some(2), None);
        assert_eq!(watch(&plan, &p), Outcome::Ready);
        assert_eq!(
            *p.did.borrow(),
            [Did::Spawn(vec!["--port".into(), "8080".into()])]
        );
        assert_eq!(
            *p.old_lives.borrow(),
            0,
            "spawned after the old pid was gone"
        );
        assert_eq!(fs::read_to_string(&plan.swapped.exe).unwrap(), "new");
        assert!(plan.swapped.kept.exists());
    }

    /// A new build that exits is restored: the old executable and UI are
    /// back, the failure is written with the last log lines, and the old
    /// build starts again with `--upgrade-failed`.
    #[test]
    fn a_build_that_exits_is_restored() {
        let t = TempDir::new().unwrap();
        let plan = swapped(&t);
        let p = fake(&plan, None, Some(3));
        let reason = "the new build exited before it was ready".to_string();
        assert_eq!(watch(&plan, &p), Outcome::Restored(reason.clone()));
        assert_eq!(
            *p.did.borrow(),
            [
                Did::Spawn(vec!["--port".into(), "8080".into()]),
                Did::Spawn(vec![
                    "--port".into(),
                    "8080".into(),
                    "--upgrade-failed".into(),
                    "12-h12".into()
                ]),
            ]
        );
        assert_eq!(fs::read_to_string(&plan.swapped.exe).unwrap(), "old");
        assert!(!plan.swapped.kept.exists());
        let failure: Failure =
            serde_json::from_slice(&fs::read(plan.upgrade_dir.join(FAILED)).unwrap()).unwrap();
        assert_eq!((failure.build.as_str(), failure.reason), ("12-h12", reason));
        assert_eq!(failure.last_lines.len(), LAST_LINES);
        assert_eq!(failure.last_lines.last().unwrap(), "line 50");
    }

    /// A build not ready in time is stopped, then restored.
    #[test]
    fn a_build_not_ready_in_time_is_stopped_and_restored() {
        let t = TempDir::new().unwrap();
        let plan = swapped(&t);
        let p = fake(&plan, None, None);
        assert_eq!(
            watch(&plan, &p),
            Outcome::Restored("the new build was not ready within 1 s".into())
        );
        assert_eq!(p.did.borrow()[1], Did::Kill);
        assert_eq!(fs::read_to_string(&plan.swapped.exe).unwrap(), "old");
    }

    /// An old daemon that will not exit is stopped before the new one starts.
    #[test]
    fn an_old_daemon_that_hangs_is_stopped_first() {
        let t = TempDir::new().unwrap();
        let plan = swapped(&t);
        fs::create_dir_all(&plan.upgrade_dir).unwrap();
        let p = fake(&plan, Some(1), None);
        *p.old_lives.borrow_mut() = u32::MAX;
        assert_eq!(watch(&plan, &p), Outcome::Ready);
        assert_eq!(p.did.borrow()[0], Did::KillPid(4242));
    }

    #[test]
    fn the_design_window_reads_in_minutes() {
        assert_eq!(spell(READY_WITHIN), "15 minutes");
        assert_eq!(spell(Duration::from_secs(90)), "90 s");
    }

    /// The flag with its plan asks for the watchdog; anything else does not.
    #[test]
    fn the_watchdog_is_asked_for_by_its_flag_and_plan() {
        let args = |a: &[&str]| a.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(
            asks_for_watchdog(&args(&["--upgrade-watchdog", "/d/upgrade/watch.json"])),
            Some(PathBuf::from("/d/upgrade/watch.json"))
        );
        assert_eq!(asks_for_watchdog(&args(&["--upgrade-watchdog"])), None);
        assert_eq!(asks_for_watchdog(&args(&["--port", "1"])), None);
    }

    #[test]
    fn a_plan_round_trips_through_its_file() {
        let t = TempDir::new().unwrap();
        let plan = swapped(&t);
        let path = write_plan(&plan).unwrap();
        assert_eq!(path, plan.upgrade_dir.join(PLAN));
        assert_eq!(read_plan(&path).unwrap(), plan);
    }
}
