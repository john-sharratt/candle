//! Start a trial executable and read its stderr for the proof line.
//!
//! The limit is the caller's. Production passes ten minutes. A process that
//! exits without the line, or that stays up without printing it, is stopped
//! and the stable file is started with the same arguments.

use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use cluster::upgrade::switch::PROOF;

use crate::place::{promote, How};

/// What the trial's stderr did inside the limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// [`PROOF`] was read. The process may still be running, or it may have
    /// exited after the line.
    Proven,
    /// The limit passed with no proof line.
    TimedOut,
    /// The process closed stderr without the proof line.
    Died,
}

/// How a switch ended, and the child left running.
pub struct Running {
    pub ended: Ended,
    pub child: Child,
}

/// Proven, with the way the file was kept, or the stable file was started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ended {
    Proven(How),
    RolledBack(String),
}

/// Start `exe` with `args` in `cwd`. Stderr is a pipe. Stdout is the parent's.
pub fn spawn(exe: &Path, args: &[std::ffi::OsString], cwd: &Path) -> io::Result<Child> {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
    cmd.spawn()
}

/// Read `child`'s stderr until [`PROOF`], the limit, or the pipe closing.
///
/// The reader thread keeps the pipe after a proof line, so a process that
/// stays up does not block when stderr fills.
pub fn watch_proof(child: &mut Child, limit: Duration) -> Verdict {
    let Some(stderr) = child.stderr.take() else {
        return Verdict::Died;
    };
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tx
                        .send(line.trim_end_matches(['\r', '\n']).to_string())
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });
    wait_for_line(rx, limit)
}

fn wait_for_line(rx: Receiver<String>, limit: Duration) -> Verdict {
    let deadline = Instant::now() + limit;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Verdict::TimedOut;
        }
        match rx.recv_timeout(left) {
            Ok(line) if line == PROOF => return Verdict::Proven,
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => return Verdict::TimedOut,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Verdict::Died,
        }
    }
}

fn reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Start `trial`. On the proof line, make it the stable file, stop the trial,
/// and start `stable` with the same arguments. The process that stays up is
/// the one started from the stable path: a rename does not change the path a
/// running image was started from. Otherwise stop the trial and start
/// `stable` the same way. This start is not another trial.
pub fn switch_to(
    trial: &Path,
    stable: &Path,
    args: &[std::ffi::OsString],
    cwd: &Path,
    limit: Duration,
) -> io::Result<Running> {
    let mut child = match spawn(trial, args, cwd) {
        Ok(child) => child,
        Err(e) => {
            let child = spawn_ready(stable, args, cwd)?;
            return Ok(Running {
                ended: Ended::RolledBack(format!("the trial did not start: {e}")),
                child,
            });
        }
    };
    let verdict = watch_proof(&mut child, limit);
    if verdict == Verdict::Proven {
        match promote(trial, stable) {
            Ok(how) => {
                // The trial holds the file it was started from. After a rename
                // that file is `stable`, so it has to exit before the stable
                // path can be started.
                reap(&mut child);
                let child = spawn_ready(stable, args, cwd)?;
                return Ok(Running {
                    ended: Ended::Proven(how),
                    child,
                });
            }
            Err(e) => {
                reap(&mut child);
                let child = spawn_ready(stable, args, cwd)?;
                return Ok(Running {
                    ended: Ended::RolledBack(format!("the trial was not kept: {e}")),
                    child,
                });
            }
        }
    }
    let why = match verdict {
        Verdict::TimedOut => "no proof line within the limit".to_string(),
        Verdict::Died => "the trial exited without the proof line".to_string(),
        Verdict::Proven => unreachable!("the proof line returned above"),
    };
    reap(&mut child);
    let child = spawn_ready(stable, args, cwd)?;
    Ok(Running {
        ended: Ended::RolledBack(why),
        child,
    })
}

/// Start `exe`, retrying while the previous image still has the file open.
/// Stderr is drained so the process does not block when the pipe fills.
fn spawn_ready(exe: &Path, args: &[std::ffi::OsString], cwd: &Path) -> io::Result<Child> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match spawn(exe, args, cwd) {
            Ok(mut child) => {
                detach_stderr(&mut child);
                return Ok(child);
            }
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(e);
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn detach_stderr(child: &mut Child) {
    let Some(stderr) = child.stderr.take() else {
        return;
    };
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::OsString;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "keep-trial-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn copy_ping(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let ping = std::path::Path::new(r"C:\Windows\System32\ping.exe");
        let dest = dir.join(name);
        std::fs::copy(ping, &dest).unwrap();
        dest
    }

    #[cfg(windows)]
    #[test]
    fn the_proof_line_counts_when_the_process_then_exits() {
        // `echo` through `cmd /c` does not land on the stderr pipe this
        // process inherits. Write the line the same way a daemon does: one
        // exact line on stderr, then exit.
        let mut child = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "[Console]::Error.WriteLine('INFO serving')",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let verdict = watch_proof(&mut child, Duration::from_secs(5));
        assert_eq!(verdict, Verdict::Proven);
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = child.wait();
    }

    #[cfg(windows)]
    #[test]
    fn a_process_that_exits_without_the_line_is_not_waited_out() {
        let mut child = Command::new("cmd")
            .args(["/c", "exit 1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let verdict = watch_proof(&mut child, Duration::from_secs(10 * 60));
        assert_eq!(verdict, Verdict::Died);
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = child.wait();
    }

    #[cfg(windows)]
    #[test]
    fn no_proof_line_within_the_limit_is_a_timeout() {
        let ping = std::path::Path::new(r"C:\Windows\System32\ping.exe");
        if !ping.is_file() {
            return;
        }
        let mut child = Command::new(ping)
            .args(["-n", "30", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let verdict = watch_proof(&mut child, Duration::from_millis(300));
        assert_eq!(verdict, Verdict::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(started.elapsed() >= Duration::from_millis(300));
        reap(&mut child);
    }

    #[cfg(windows)]
    #[test]
    fn a_timed_out_trial_is_stopped_and_the_stable_file_is_started() {
        let ping = std::path::Path::new(r"C:\Windows\System32\ping.exe");
        if !ping.is_file() {
            return;
        }
        let dir = scratch();
        let trial = copy_ping(&dir, "trial.exe");
        let stable = copy_ping(&dir, "stable.exe");
        let args = [
            OsString::from("-n"),
            OsString::from("20"),
            OsString::from("127.0.0.1"),
        ];
        let started = Instant::now();
        let mut running =
            switch_to(&trial, &stable, &args, &dir, Duration::from_millis(300)).unwrap();
        assert!(started.elapsed() < Duration::from_secs(8));
        assert_eq!(
            running.ended,
            Ended::RolledBack("no proof line within the limit".into())
        );
        assert!(running.child.try_wait().unwrap().is_none());
        reap(&mut running.child);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn a_proven_trial_is_restarted_from_the_stable_path() {
        let dir = scratch();
        let built = compile_proof(&dir);
        let trial = dir.join("trial.exe");
        let stable = dir.join("stable.exe");
        std::fs::copy(&built, &trial).unwrap();
        std::fs::copy(&built, &stable).unwrap();
        let mut running = switch_to(&trial, &stable, &[], &dir, Duration::from_secs(15)).unwrap();
        let child = &mut running.child;
        let ended = running.ended.clone();
        let alive = child.try_wait().unwrap().is_none();
        let image = std::fs::canonicalize(image_of(child.id())).unwrap();
        let stable_path = std::fs::canonicalize(&stable).unwrap();
        reap(child);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(matches!(ended, Ended::Proven(_)));
        assert!(alive, "the stable process exited");
        assert_eq!(image, stable_path);
    }

    fn compile_proof(dir: &std::path::Path) -> std::path::PathBuf {
        let src = dir.join("proof.rs");
        std::fs::write(
            &src,
            "fn main() {\n    eprintln!(\"INFO serving\");\n    loop { std::thread::sleep(std::time::Duration::from_secs(60)); }\n}\n",
        )
        .unwrap();
        let exe = dir.join("proof.exe");
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let status = Command::new(rustc)
            .arg(&src)
            .arg("-o")
            .arg(&exe)
            .status()
            .unwrap();
        assert!(status.success(), "rustc did not build the proof executable");
        exe
    }

    fn image_of(pid: u32) -> std::path::PathBuf {
        use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
        let mut sys = System::new();
        let pid = Pid::from_u32(pid);
        for _ in 0..40 {
            sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                ProcessRefreshKind::new().with_exe(UpdateKind::Always),
            );
            if let Some(proc) = sys.process(pid) {
                if let Some(exe) = proc.exe() {
                    return exe.to_path_buf();
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!("process {pid} has no image path");
    }
}
