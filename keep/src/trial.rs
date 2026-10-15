//! Start a trial executable and watch its stderr for the proof line.
//!
//! **A trial runs from the stable path** (`crate::place`): the stable file is
//! set aside and the trial's bytes installed in its place, so the process the
//! firewall allows, and the one a self-relaunch starts, is the one left
//! running. The staged file is only copied, and stays free for the next
//! deploy.
//!
//! **Its stderr goes to a file**, which this process reads for [`PROOF`]. A
//! file, not a pipe: the process keeps writing to it after the proof, and
//! after keep itself restarts, and nothing ever has to drain it. A pipe would
//! have to be read for the life of the process, and a write to one whose
//! reader had gone — keep restarted — would end the daemon.
//!
//! **A proven trial is the service from then on**: the set-aside file is let
//! go and the process is left running — a promotion never restarts the
//! service. A trial that exits without the line, or stays up without printing
//! it within the limit, is stopped, the set-aside file put back, and the
//! stable file started. The limit is the caller's; production passes ten
//! minutes.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use cluster::upgrade::switch::PROOF;

use crate::place::{discard_previous, install, restore, set_aside};

/// How often a trial's stderr file is read for new lines.
const POLL: Duration = Duration::from_millis(100);

/// What the trial's stderr did inside the limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// [`PROOF`] was read. The process may still be running, or it may have
    /// exited after the line.
    Proven,
    /// The limit passed with no proof line.
    TimedOut,
    /// The process exited without the proof line.
    Died,
}

/// How a switch ended, and the child left running.
pub struct Running {
    pub ended: Ended,
    pub child: Child,
}

/// Proven — the trial is the service — or rolled back to the stable file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ended {
    Proven,
    RolledBack(String),
}

/// Start `exe` from a stable path. Stdout and stderr are detached: this
/// process holds no handles on them, does not read or write them, and they
/// are not appended to the supervisor log.
pub fn spawn_detached(exe: &Path, args: &[OsString], cwd: &Path) -> io::Result<Child> {
    launch(exe, args, cwd, Stdio::null())
}

/// Start a trial, detached like a stable child, its stderr written to `log`.
fn spawn_trial(exe: &Path, args: &[OsString], cwd: &Path, log: &Path) -> io::Result<Child> {
    launch(exe, args, cwd, Stdio::from(File::create(log)?))
}

fn launch(exe: &Path, args: &[OsString], cwd: &Path, stderr: Stdio) -> io::Result<Child> {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        /// The new process does not inherit this process's console, so its
        /// standard streams are not the supervisor redirect.
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
    }
    cmd.spawn()
}

/// Read `log` — `child`'s stderr — until [`PROOF`], the limit, or the
/// process exiting. Lines read before that are logged.
pub fn watch_proof(child: &mut Child, log: &Path, limit: Duration) -> Verdict {
    let deadline = Instant::now() + limit;
    let mut at = 0u64;
    let mut partial = String::new();
    loop {
        // Read what is there before asking whether the process has exited, so
        // a line written just before the exit is still seen.
        let exited = matches!(child.try_wait(), Ok(Some(_)));
        if let Some(found) = read_new_lines(log, &mut at, &mut partial) {
            if found {
                return Verdict::Proven;
            }
        }
        if exited {
            return Verdict::Died;
        }
        if Instant::now() >= deadline {
            return Verdict::TimedOut;
        }
        thread::sleep(POLL);
    }
}

/// The complete lines `log` has gained since `at`, each logged; whether one
/// was [`PROOF`]. `None` when the file cannot be read. A line still being
/// written waits in `partial`.
fn read_new_lines(log: &Path, at: &mut u64, partial: &mut String) -> Option<bool> {
    let mut file = File::open(log).ok()?;
    file.seek(SeekFrom::Start(*at)).ok()?;
    let mut reader = BufReader::new(file);
    let mut proven = false;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return Some(proven);
        }
        *at += n as u64;
        partial.push_str(&line);
        if !partial.ends_with('\n') {
            continue;
        }
        let done = partial.trim_end_matches(['\r', '\n']).to_string();
        partial.clear();
        if !done.is_empty() {
            eprintln!("keep: trial: {done}");
        }
        if done == PROOF {
            proven = true;
        }
    }
}

/// Stop `child` and wait for it.
pub fn reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Install `trial` on the stable path and start it there, its stderr in `log`.
/// Proven: the process is left running as the service. Otherwise: it is
/// stopped, the previous stable file put back and started. The process the
/// stable file was started from has already been stopped.
pub fn switch_to(
    trial: &Path,
    stable: &Path,
    log: &Path,
    args: &[OsString],
    cwd: &Path,
    limit: Duration,
) -> io::Result<Running> {
    if let Err(e) = set_aside(stable) {
        return rolled_back(
            stable,
            args,
            cwd,
            format!("the stable file was not set aside: {e}"),
        );
    }
    if let Err(e) = install(trial, stable) {
        restore(stable)?;
        return rolled_back(
            stable,
            args,
            cwd,
            format!("the trial was not installed: {e}"),
        );
    }
    let mut child = match spawn_trial(stable, args, cwd, log) {
        Ok(child) => child,
        Err(e) => {
            restore(stable)?;
            return rolled_back(stable, args, cwd, format!("the trial did not start: {e}"));
        }
    };
    let why = match watch_proof(&mut child, log, limit) {
        Verdict::Proven => {
            // The trial is the service: what it replaced is not needed, and a
            // file left set aside would read as an unproven trial to the next
            // keep that starts.
            if let Err(e) = discard_previous(stable) {
                eprintln!("keep: the previous stable file was not removed: {e}");
            }
            return Ok(Running {
                ended: Ended::Proven,
                child,
            });
        }
        Verdict::TimedOut => "no proof line within the limit",
        Verdict::Died => "the trial exited without the proof line",
    };
    reap(&mut child);
    restore(stable)?;
    rolled_back(stable, args, cwd, why.to_string())
}

/// `stable` started again, the switch ended for `why`.
fn rolled_back(stable: &Path, args: &[OsString], cwd: &Path, why: String) -> io::Result<Running> {
    let child = spawn_ready(stable, args, cwd)?;
    Ok(Running {
        ended: Ended::RolledBack(why),
        child,
    })
}

/// Start `exe`, retrying while the previous image still has the file open.
/// The child is detached: no stdout or stderr handle, and no reader.
fn spawn_ready(exe: &Path, args: &[OsString], cwd: &Path) -> io::Result<Child> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match spawn_detached(exe, args, cwd) {
            Ok(child) => return Ok(child),
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(e);
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

#[cfg(test)]
mod tests;
