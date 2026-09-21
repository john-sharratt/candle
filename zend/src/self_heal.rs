//! Relaunch this process when the GPU context is poisoned, instead of
//! exiting for a supervisor that does not exist.
//!
//! `candle::gpu_poison` flags the context dead on a sticky CUDA fault, or on
//! an out-of-memory streak that has run unbroken for
//! `candle::gpu_poison::OOM_STICKY_AFTER` (60 s) with no successful device
//! operation in between. Neither zend nor any of the daemons on this estate
//! runs under a service manager or a restart-on-exit wrapper
//! (`docs/deployment.md`: each is started directly, detached) — cf-ddns is
//! the one exception, via its own scheduled-task supervisor. So a watchdog
//! that only exits on poison leaves the daemon dead until a human notices and
//! restarts it by hand: measured in production, a device stuck in an
//! out-of-memory retry storm served nothing for 40+ minutes before this
//! existed.
//!
//! This module captures how the process was launched, and on poison,
//! relaunches an identical process before exiting — self-healing rather than
//! self-terminating. The substrate redo log is crash-safe, so the abrupt exit
//! loses nothing durable; `log_file::RESTART_MARKER_NAME` (dropped here,
//! consumed by `log_file::RotatingFileLog::new`) makes the relaunch append to
//! the existing log instead of truncating it, so the evidence for why this
//! happened survives.
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

use std::env;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::log_file::RESTART_MARKER_NAME;

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

/// File under the daemon's `.substrate` recording how many *consecutive*
/// fast (within [`FAST_POISON_WINDOW`]) relaunches have happened — plain
/// decimal text, no format to version. Absent or unparseable reads as 0, the
/// same as a fresh burst; [`relaunch_decision`] is what actually resets it
/// (a long uptime before poisoning ignores whatever this holds), so this file
/// is never explicitly cleared, only ever overwritten with a fresh count.
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

/// Read the consecutive-fast-relaunch count from `substrate_dir`. Absent,
/// unreadable, or unparseable all read as 0 — the safe default (a fresh
/// burst), since refusing to relaunch over a corrupt counter file would be a
/// worse failure than miscounting one attempt.
fn read_attempts(substrate_dir: &Path) -> u32 {
    std::fs::read_to_string(substrate_dir.join(ATTEMPTS_FILE_NAME))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Persist the consecutive-fast-relaunch count. Best-effort, like
/// [`mark_resume`]: a failed write just means the next poisoning under-counts
/// from 0 rather than from where this run left off, which only delays
/// reaching [`MAX_FAST_RELAUNCHES`] — never prevents it, since a genuinely
/// crash-looping process keeps re-attempting the write on every poisoning.
fn write_attempts(substrate_dir: &Path, attempts: u32) {
    if let Err(e) = std::fs::create_dir_all(substrate_dir)
        .and_then(|()| std::fs::write(substrate_dir.join(ATTEMPTS_FILE_NAME), attempts.to_string()))
    {
        tracing::warn!("could not persist the self-heal attempt count: {e}");
    }
}

/// How this process was launched — captured once at startup, before anything
/// can consume argv or change the working directory.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    exe: PathBuf,
    args: Vec<OsString>,
    cwd: PathBuf,
}

impl LaunchSpec {
    /// Capture the current process's executable path, arguments (excluding
    /// `argv[0]`, exactly as the shell passed them — not reconstructed from
    /// parsed CLI flags, so nothing about the original invocation is lost or
    /// normalised away), and working directory.
    pub fn capture() -> io::Result<Self> {
        Ok(Self {
            exe: env::current_exe()?,
            args: env::args_os().skip(1).collect(),
            cwd: env::current_dir()?,
        })
    }

    /// Build (without spawning) the command that relaunches an identical
    /// process. A fresh stdin: the child must not inherit a handle whose
    /// other end goes away when this process exits.
    fn relaunch_command(&self) -> Command {
        let mut cmd = Command::new(&self.exe);
        cmd.args(&self.args)
            .current_dir(&self.cwd)
            .stdin(Stdio::null());
        cmd
    }
}

/// Drop the restart marker next to the log so the relaunched process appends
/// instead of truncating. Best-effort: a failure here means the relaunch's
/// log starts fresh (loses the "why", not the daemon), so it is logged and
/// swallowed rather than aborting the relaunch over it.
fn mark_resume(substrate_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(substrate_dir)
        .and_then(|()| std::fs::write(substrate_dir.join(RESTART_MARKER_NAME), b""))
    {
        tracing::error!(
            "could not write the self-heal restart marker — the relaunched process's \
             log will start fresh instead of continuing this one: {e}"
        );
    }
}

/// Spawn the background thread that polls `candle::gpu_poison::is_gpu_poisoned`
/// and, on the transition, relaunches an identical process before exiting
/// this one. `substrate_dir` is the daemon's `.substrate` directory (where the
/// restart marker and the log both live).
pub fn spawn_watchdog(spec: LaunchSpec, substrate_dir: PathBuf) {
    let started = Instant::now();
    std::thread::Builder::new()
        .name("gpu-poison-watchdog".into())
        .spawn(move || loop {
            // Poll tightly so the window in which other threads pile
            // identical downstream errors onto the dead context stays small.
            std::thread::sleep(Duration::from_millis(50));
            if !candle::gpu_poison::is_gpu_poisoned() {
                continue;
            }
            tracing::error!(
                root = %candle::gpu_poison::root_fault().unwrap_or_default(),
                "GPU context poisoned",
            );
            eprintln!("{}", candle_conversation::relief_trace::dump());

            let decision = relaunch_decision(started.elapsed(), read_attempts(&substrate_dir));
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
            write_attempts(&substrate_dir, attempts);
            tracing::error!(
                attempts,
                max = MAX_FAST_RELAUNCHES,
                "relaunching an identical process instead of exiting for a supervisor that \
                 does not exist (substrate redo log is durable, nothing lost)",
            );
            mark_resume(&substrate_dir);
            // Brief pause so the root log line reaches the file/console sinks
            // before the hard exit.
            std::thread::sleep(Duration::from_millis(80));
            match spec.relaunch_command().spawn() {
                Ok(child) => tracing::error!(pid = child.id(), "relaunched — exiting"),
                Err(e) => {
                    // Nothing else can bring the daemon back. Exit anyway:
                    // staying up would just keep spewing the same poisoned
                    // error forever, which is the exact failure mode this
                    // module exists to end.
                    eprintln!("zend: self-heal relaunch failed: {e} — exiting anyway");
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
        let dir = std::env::temp_dir().join(format!("self-heal-attempts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(read_attempts(&dir), 0, "no file yet");

        write_attempts(&dir, 3);
        assert_eq!(read_attempts(&dir), 3);

        write_attempts(&dir, 7);
        assert_eq!(read_attempts(&dir), 7, "overwritten, not accumulated");

        std::fs::write(dir.join(ATTEMPTS_FILE_NAME), b"not a number").unwrap();
        assert_eq!(read_attempts(&dir), 0, "unparseable reads as fresh");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `capture` reflects the real process, and the command it builds carries
    /// the captured exe/args/cwd through untouched.
    #[test]
    fn capture_round_trips_into_the_relaunch_command() {
        let spec = LaunchSpec::capture().unwrap();
        assert_eq!(spec.exe, env::current_exe().unwrap());
        assert_eq!(spec.cwd, env::current_dir().unwrap());

        let cmd = spec.relaunch_command();
        assert_eq!(cmd.get_program(), spec.exe.as_os_str());
        let got_args: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        let want_args: Vec<&std::ffi::OsStr> = spec.args.iter().map(OsString::as_os_str).collect();
        assert_eq!(got_args, want_args);
        assert_eq!(cmd.get_current_dir(), Some(spec.cwd.as_path()));
    }

    /// `mark_resume` creates the directory if needed and drops an empty
    /// marker file that `log_file::is_resuming` will find.
    #[test]
    fn mark_resume_creates_dir_and_marker() {
        let dir = std::env::temp_dir().join(format!("self-heal-mark-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        mark_resume(&dir);

        assert!(crate::log_file::is_resuming(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
