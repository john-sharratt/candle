//! Relaunch this process when the GPU context is poisoned, instead of
//! exiting for a supervisor that does not exist.
//!
//! `candle::gpu_poison` flags the context dead on a sticky CUDA fault, or on
//! an out-of-memory streak that has run unbroken for
//! `candle::gpu_poison::OOM_STICKY_AFTER` (60 s) with no successful device
//! operation in between. None of the daemons on this estate runs under a
//! service manager or a restart-on-exit wrapper
//! (`docs/deployment.md`: each is started directly, detached) — cf-ddns is
//! the one exception, via its own scheduled-task supervisor. So a watchdog
//! that only exited on poison would leave the daemon dead until a human
//! noticed and restarted it by hand — for a device stuck in an out-of-memory
//! retry storm, measured at 40+ minutes of serving nothing.
//!
//! On poison this module relaunches an identical process — from the
//! [`LaunchSpec`] captured at startup — before exiting: self-healing rather than
//! self-terminating. The substrate redo log is crash-safe, so the abrupt exit
//! loses nothing durable; `web::log_file::RESTART_MARKER_NAME` (dropped here,
//! consumed by `web::log_file::RotatingFileLog::new`) makes the relaunch append to
//! the existing log instead of rotating it out, so the fault and the recovery
//! read as one run.
//!
//! **Relaunching is capped, not unconditional.** A transient fault (the OOM
//! streak above) is exactly what one relaunch fixes — but nothing tells that
//! apart from a persistent one (a broken card, a bad driver, a repeatable
//! bug) that poisons every fresh process within moments of it starting. Left
//! unconditional, the watchdog would crash-loop forever on the persistent
//! case. [`relaunch_decision`] tracks consecutive *fast* poisonings — ones
//! that happen within [`FAST_POISON_WINDOW`] of the relaunched process's own
//! start — across relaunches via [`ATTEMPTS_FILE_NAME`], and after
//! [`MAX_FAST_RELAUNCHES`] of them gives up and stays down instead of
//! relaunching again. A poisoning after a long, healthy uptime resets the
//! count: it is treated as a fresh, unrelated occurrence, not a continuation.
//! So does any start that is not the watchdog's own relaunch — by hand, an
//! install, a clean restart: only a process this module started carries the
//! count of the one it replaced ([`carried_attempts`]).

use std::path::{Path, PathBuf};
use std::thread::Builder as ThreadBuilder;
use std::time::{Duration, Instant};

use candle::gpu_poison::{is_gpu_poisoned, root_fault};
use candle_conversation::relief_trace;
use cluster::launch::LaunchSpec;

#[cfg(test)]
use web::log_file::is_resuming;
use web::log_file::RESTART_MARKER_NAME;

/// Exit code a relaunch replaces — distinct from clean shutdown (0) and
/// Ctrl-C (130). Kept for anything external still watching for it (a log
/// scrape, a monitoring rule); nothing in this repo currently acts on it,
/// which is exactly the gap this module closes.
pub const GPU_POISON_EXIT_CODE: i32 = 75;

/// Exit code the watchdog uses when it gives up rather than relaunching —
/// distinct from [`GPU_POISON_EXIT_CODE`] specifically so a monitoring rule
/// can tell "restart me" from "a human needs to look at this machine" apart.
pub const GPU_POISON_GIVEUP_EXIT_CODE: i32 = 76;

/// A relaunched process poisoning again within this long of its own start is
/// evidence of a persistent fault (bad card, bad driver, a repeatable bug),
/// not the transient kind this module exists to recover from — it counts
/// toward [`MAX_FAST_RELAUNCHES`] instead of resetting the count. Comfortably
/// longer than the 60 s an out-of-memory streak needs to become sticky
/// (`candle::gpu_poison::OOM_STICKY_AFTER`), so a streak that resolves and
/// then recurs much later is still treated as a fresh occurrence.
const FAST_POISON_WINDOW: Duration = Duration::from_secs(300);

/// How many poisonings within an unbroken run of [`FAST_POISON_WINDOW`]-paced
/// relaunches are tolerated before the watchdog stops relaunching and stays
/// down. Without a cap, a genuinely broken card poisons every fresh process
/// within its first poll tick and the watchdog crash-loops forever — process
/// churn and log spam standing in for the "a human needs to look at this"
/// signal a real supervisor would give.
const MAX_FAST_RELAUNCHES: u32 = 5;

/// File beside the process's log recording how many *consecutive*
/// fast (within [`FAST_POISON_WINDOW`]) relaunches have happened — plain
/// decimal text, no format to version. Absent or unparseable reads as 0, the
/// same as a fresh burst. It is read only by a process the watchdog
/// relaunched ([`carried_attempts`]), and a long uptime before poisoning
/// ignores it ([`relaunch_decision`]), so it is never explicitly cleared, only
/// ever overwritten with a fresh count.
const ATTEMPTS_FILE_NAME: &str = ".self_heal_attempts";

/// What [`relaunch_decision`] says to do about a poisoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelaunchDecision {
    /// Relaunch, and persist this as the new consecutive-fast-relaunch count.
    Relaunch { attempts: u32 },
    /// Do not relaunch — [`MAX_FAST_RELAUNCHES`] consecutive fast poisonings
    /// means this is a persistent fault a relaunch will not fix.
    GiveUp,
}

/// Decide what to do about a poisoning, given how long *this* process had
/// been up when it happened and the consecutive-fast-relaunch count carried
/// over from [`ATTEMPTS_FILE_NAME`] (0 if the file was absent or this is the
/// first poisoning of a fresh burst).
///
/// A long uptime before poisoning means this occurrence is unrelated to
/// whatever came before, whatever `prior_attempts` says — the count resets
/// to 1 rather than accumulating. Only an unbroken run of *fast* poisonings
/// climbs toward [`MAX_FAST_RELAUNCHES`].
fn relaunch_decision(uptime: Duration, prior_attempts: u32) -> RelaunchDecision {
    let attempts = if uptime < FAST_POISON_WINDOW {
        prior_attempts.saturating_add(1)
    } else {
        1
    };
    if attempts > MAX_FAST_RELAUNCHES {
        RelaunchDecision::GiveUp
    } else {
        RelaunchDecision::Relaunch { attempts }
    }
}

/// Read the consecutive-fast-relaunch count from `state_dir`. Absent,
/// unreadable, or unparseable all read as 0 — the safe default (a fresh
/// burst), since refusing to relaunch over a corrupt counter file would be a
/// worse failure than miscounting one attempt.
fn read_attempts(state_dir: &Path) -> u32 {
    std::fs::read_to_string(state_dir.join(ATTEMPTS_FILE_NAME))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// The consecutive-fast-relaunch count this process starts from: the stored
/// one when it is the watchdog's relaunch (`resumed`, the restart marker it
/// found at start), and 0 for any other start. A count left by a crash loop
/// days ago is not this process's history: read regardless, it gave up on a
/// hand-started process's first poisoning without one relaunch.
fn carried_attempts(resumed: bool, state_dir: &Path) -> u32 {
    if resumed {
        read_attempts(state_dir)
    } else {
        0
    }
}

/// Persist the consecutive-fast-relaunch count. Best-effort, like
/// [`mark_resume`]: a failed write just means the next poisoning under-counts
/// from 0 rather than from where this run left off, which only delays
/// reaching [`MAX_FAST_RELAUNCHES`] — never prevents it, since a genuinely
/// crash-looping process keeps re-attempting the write on every poisoning.
fn write_attempts(state_dir: &Path, attempts: u32) {
    if let Err(e) = std::fs::create_dir_all(state_dir)
        .and_then(|()| std::fs::write(state_dir.join(ATTEMPTS_FILE_NAME), attempts.to_string()))
    {
        tracing::warn!("could not persist the self-heal attempt count: {e}");
    }
}

/// Drop the restart marker next to the log so the relaunched process appends
/// instead of rotating. Best-effort: a failure here means the relaunch's log
/// starts a new file (the fault is still in `.1`), so it is logged and
/// swallowed rather than aborting the relaunch over it.
fn mark_resume(state_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(state_dir)
        .and_then(|()| std::fs::write(state_dir.join(RESTART_MARKER_NAME), b""))
    {
        tracing::error!(
            "could not write the self-heal restart marker — the relaunched process's \
             log will start a new file instead of continuing this one: {e}"
        );
    }
}

/// Spawn the background thread that polls `candle::gpu_poison::is_gpu_poisoned`
/// and, on the transition, relaunches an identical process before exiting
/// this one. `state_dir` is where the log lives, and with it the restart
/// marker and the attempt count; `resumed` is whether this start is the
/// watchdog's own relaunch.
pub fn spawn_watchdog(spec: LaunchSpec, state_dir: PathBuf, resumed: bool) {
    let started = Instant::now();
    let carried = carried_attempts(resumed, &state_dir);
    ThreadBuilder::new()
        .name("gpu-poison-watchdog".into())
        .spawn(move || loop {
            // Poll tightly so the window in which other threads pile
            // identical downstream errors onto the dead context stays small.
            std::thread::sleep(Duration::from_millis(50));
            if !is_gpu_poisoned() {
                continue;
            }
            tracing::error!(
                root = %root_fault().unwrap_or_default(),
                "GPU context poisoned",
            );
            eprintln!("{}", relief_trace::dump());

            let decision = relaunch_decision(started.elapsed(), carried);
            let RelaunchDecision::Relaunch { attempts } = decision else {
                tracing::error!(
                    max = MAX_FAST_RELAUNCHES,
                    "self-heal: {MAX_FAST_RELAUNCHES} consecutive fast poisonings — this looks \
                     like a persistent fault a relaunch will not fix (bad card, bad driver, a \
                     repeatable bug), not the transient kind this watchdog recovers from. \
                     Staying down instead of crash-looping — this needs a human.",
                );
                std::thread::sleep(Duration::from_millis(80));
                std::process::exit(GPU_POISON_GIVEUP_EXIT_CODE);
            };
            write_attempts(&state_dir, attempts);
            tracing::error!(
                attempts,
                max = MAX_FAST_RELAUNCHES,
                "relaunching an identical process instead of exiting for a supervisor that \
                 does not exist (substrate redo log is durable, nothing lost)",
            );
            mark_resume(&state_dir);
            // Brief pause so the root log line reaches the file/console sinks
            // before the hard exit.
            std::thread::sleep(Duration::from_millis(80));
            match spec.command().spawn() {
                Ok(child) => tracing::error!(pid = child.id(), "relaunched — exiting"),
                Err(e) => {
                    // Nothing else can bring the daemon back. Exit anyway:
                    // staying up would just keep spewing the same poisoned
                    // error forever, which is the exact failure mode this
                    // module exists to end.
                    eprintln!("self-heal relaunch failed: {e} — exiting anyway");
                }
            }
            std::process::exit(GPU_POISON_EXIT_CODE);
        })
        .expect("spawn gpu-poison-watchdog");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fast poisoning (well inside the window) accumulates toward the cap.
    #[test]
    fn a_fast_poisoning_accumulates() {
        assert_eq!(
            relaunch_decision(Duration::from_secs(1), 0),
            RelaunchDecision::Relaunch { attempts: 1 }
        );
        assert_eq!(
            relaunch_decision(Duration::from_secs(1), 3),
            RelaunchDecision::Relaunch { attempts: 4 }
        );
    }

    /// A poisoning after a long, healthy uptime is a fresh occurrence —
    /// the count resets to 1 no matter how high it was before.
    #[test]
    fn a_slow_poisoning_resets_the_count() {
        assert_eq!(
            relaunch_decision(FAST_POISON_WINDOW, MAX_FAST_RELAUNCHES),
            RelaunchDecision::Relaunch { attempts: 1 }
        );
        assert_eq!(
            relaunch_decision(Duration::from_secs(3600), 99),
            RelaunchDecision::Relaunch { attempts: 1 }
        );
    }

    /// Exactly `MAX_FAST_RELAUNCHES` consecutive fast poisonings still
    /// relaunches — the cap is exceeded, not reached, that gives up.
    #[test]
    fn the_cap_itself_still_relaunches() {
        assert_eq!(
            relaunch_decision(Duration::from_secs(1), MAX_FAST_RELAUNCHES - 1),
            RelaunchDecision::Relaunch {
                attempts: MAX_FAST_RELAUNCHES
            }
        );
    }

    /// One fast poisoning past the cap gives up instead of relaunching again.
    #[test]
    fn past_the_cap_gives_up() {
        assert_eq!(
            relaunch_decision(Duration::from_secs(1), MAX_FAST_RELAUNCHES),
            RelaunchDecision::GiveUp
        );
        // Saturates rather than wrapping on an already-huge stored count.
        assert_eq!(
            relaunch_decision(Duration::from_secs(1), u32::MAX),
            RelaunchDecision::GiveUp
        );
    }

    /// The attempt count round-trips through the marker file, and an absent
    /// or corrupt file reads as 0 rather than erroring.
    #[test]
    fn attempts_round_trip_and_default_to_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("state");

        assert_eq!(read_attempts(&dir), 0, "no file yet");

        write_attempts(&dir, 3);
        assert_eq!(read_attempts(&dir), 3);

        write_attempts(&dir, 7);
        assert_eq!(read_attempts(&dir), 7, "overwritten, not accumulated");

        std::fs::write(dir.join(ATTEMPTS_FILE_NAME), b"not a number").unwrap();
        assert_eq!(read_attempts(&dir), 0, "unparseable reads as fresh");
    }

    /// **Only the watchdog's own relaunch carries the count.** A start by hand
    /// over a count of 5 left by an old crash loop begins a fresh burst — it
    /// had given up on its first poisoning, with no relaunch at all — while a
    /// relaunched process continues the burst it belongs to.
    #[test]
    fn only_a_relaunch_carries_the_count() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("state");
        write_attempts(&dir, MAX_FAST_RELAUNCHES);
        assert_eq!(carried_attempts(false, &dir), 0);
        assert_eq!(
            relaunch_decision(Duration::from_secs(120), carried_attempts(false, &dir)),
            RelaunchDecision::Relaunch { attempts: 1 }
        );
        assert_eq!(carried_attempts(true, &dir), MAX_FAST_RELAUNCHES);
        assert_eq!(
            relaunch_decision(Duration::from_secs(120), carried_attempts(true, &dir)),
            RelaunchDecision::GiveUp
        );
    }

    /// `mark_resume` creates the directory if needed and drops an empty
    /// marker file that `log_file::is_resuming` will find.
    #[test]
    fn mark_resume_creates_dir_and_marker() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("state");

        mark_resume(&dir);

        assert!(is_resuming(&dir));
    }
}
