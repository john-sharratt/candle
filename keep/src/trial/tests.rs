use super::*;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::place::previous_of;

/// Where these tests put a trial's stderr.
const LOG: &str = "stderr.log";

fn scratch() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "keep-trial-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn ping() -> Option<PathBuf> {
    let ping = PathBuf::from(r"C:\Windows\System32\ping.exe");
    ping.is_file().then_some(ping)
}

/// A daemon in miniature: the proof line on stderr, then running on.
fn compile_proof(dir: &Path) -> PathBuf {
    compile(
        dir,
        "proof",
        "fn main() {\n    eprintln!(\"INFO serving\");\n    loop { std::thread::sleep(std::time::Duration::from_secs(60)); }\n}\n",
    )
}

/// The proof line on stderr, and then an exit.
fn compile_proof_then_exit(dir: &Path) -> PathBuf {
    compile(
        dir,
        "proof_exit",
        "fn main() {\n    eprintln!(\"INFO serving\");\n}\n",
    )
}

fn compile(dir: &Path, name: &str, body: &str) -> PathBuf {
    let src = dir.join(format!("{name}.rs"));
    fs::write(&src, body).unwrap();
    let exe = dir.join(format!("{name}.exe"));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let status = Command::new(rustc)
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .status()
        .unwrap();
    assert!(status.success(), "rustc did not build {name}");
    exe
}

/// The path process `pid` was started from.
fn image_of(pid: u32) -> PathBuf {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    for _ in 0..40 {
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            ProcessRefreshKind::new().with_exe(UpdateKind::Always),
        );
        if let Some(exe) = sys.process(pid).and_then(|p| p.exe()) {
            return exe.to_path_buf();
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("process {pid} has no image path");
}

#[cfg(windows)]
#[test]
fn the_proof_line_in_the_log_counts_when_the_process_then_exits() {
    let dir = scratch();
    let exe = compile_proof_then_exit(&dir);
    let log = dir.join(LOG);
    let mut child = spawn_trial(&exe, &[], &dir, &log).unwrap();
    let started = Instant::now();
    let verdict = watch_proof(&mut child, &log, Duration::from_secs(10));
    let elapsed = started.elapsed();
    reap(&mut child);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(verdict, Verdict::Proven);
    assert!(elapsed < Duration::from_secs(8));
}

#[cfg(windows)]
#[test]
fn a_process_that_exits_without_the_line_is_not_waited_out() {
    let dir = scratch();
    let log = dir.join(LOG);
    let mut child = spawn_trial(
        Path::new("cmd"),
        &[OsString::from("/c"), OsString::from("exit 1")],
        &dir,
        &log,
    )
    .unwrap();
    let started = Instant::now();
    let verdict = watch_proof(&mut child, &log, Duration::from_secs(10 * 60));
    let elapsed = started.elapsed();
    let _ = child.wait();
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(verdict, Verdict::Died);
    assert!(elapsed < Duration::from_secs(5));
}

#[cfg(windows)]
#[test]
fn no_proof_line_within_the_limit_is_a_timeout() {
    let Some(ping) = ping() else {
        return;
    };
    let dir = scratch();
    let log = dir.join(LOG);
    let mut child = spawn_trial(
        &ping,
        &[
            OsString::from("-n"),
            OsString::from("30"),
            OsString::from("127.0.0.1"),
        ],
        &dir,
        &log,
    )
    .unwrap();
    let started = Instant::now();
    let verdict = watch_proof(&mut child, &log, Duration::from_millis(300));
    let elapsed = started.elapsed();
    reap(&mut child);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(verdict, Verdict::TimedOut);
    assert!(elapsed >= Duration::from_millis(300) && elapsed < Duration::from_secs(5));
}

/// **A proven trial is the service: it goes on running, never restarted.**
/// It was started from the stable path — the path the firewall allows and a
/// self-relaunch starts — the stable file is its bytes, nothing is left set
/// aside, and the staged file is free for the next deploy.
#[cfg(windows)]
#[test]
fn a_proven_trial_keeps_running_from_the_stable_path() {
    let dir = scratch();
    let built = compile_proof(&dir);
    let staged = dir.join("stage").join("svc.exe");
    let stable = dir.join("stable").join("svc.exe");
    fs::create_dir_all(staged.parent().unwrap()).unwrap();
    fs::create_dir_all(stable.parent().unwrap()).unwrap();
    fs::copy(&built, &staged).unwrap();
    fs::write(&stable, b"previous").unwrap();
    let built_bytes = fs::read(&built).unwrap();
    let log = dir.join(LOG);

    let mut running =
        switch_to(&staged, &stable, &log, &[], &dir, Duration::from_secs(15)).unwrap();
    let ended = running.ended.clone();
    let alive = running.child.try_wait().unwrap().is_none();
    let detached = running.child.stdout.is_none() && running.child.stderr.is_none();
    let image = fs::canonicalize(image_of(running.child.id())).unwrap();
    let stable_path = fs::canonicalize(&stable).unwrap();
    let stable_bytes = fs::read(&stable).unwrap();
    let set_aside_left = previous_of(&stable).exists();
    let staged_free = fs::write(&staged, b"the next deploy").is_ok();
    reap(&mut running.child);
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(ended, Ended::Proven);
    assert!(alive, "the proven trial is still running");
    assert!(detached, "keep holds no stream of the running service");
    assert_eq!(image, stable_path, "started from the stable path");
    assert_eq!(
        stable_bytes, built_bytes,
        "the stable file is the trial's bytes"
    );
    assert!(!set_aside_left, "nothing is left set aside");
    assert!(staged_free, "the staged file is free for the next deploy");
}

/// **A trial that never proves itself is stopped, the previous stable file
/// put back and started**, detached, with nothing left set aside.
#[cfg(windows)]
#[test]
fn a_timed_out_trial_is_stopped_and_the_previous_stable_file_is_started() {
    let Some(ping) = ping() else {
        return;
    };
    let dir = scratch();
    let built = compile_proof(&dir);
    let staged = dir.join("stage").join("svc.exe");
    let stable = dir.join("stable").join("svc.exe");
    fs::create_dir_all(staged.parent().unwrap()).unwrap();
    fs::create_dir_all(stable.parent().unwrap()).unwrap();
    // The trial is a program that never prints the line; the stable file is
    // one that does, so its bytes are told apart.
    fs::copy(&ping, &staged).unwrap();
    fs::copy(&built, &stable).unwrap();
    let previous_bytes = fs::read(&built).unwrap();
    let args = [
        OsString::from("-n"),
        OsString::from("20"),
        OsString::from("127.0.0.1"),
    ];
    let log = dir.join(LOG);
    let mut running = switch_to(
        &staged,
        &stable,
        &log,
        &args,
        &dir,
        Duration::from_millis(300),
    )
    .unwrap();
    let ended = running.ended.clone();
    let alive = running.child.try_wait().unwrap().is_none();
    let detached = running.child.stdout.is_none() && running.child.stderr.is_none();
    let stable_bytes = fs::read(&stable).unwrap();
    let set_aside_left = previous_of(&stable).exists();
    reap(&mut running.child);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(
        ended,
        Ended::RolledBack("no proof line within the limit".into())
    );
    assert!(alive, "the previous stable file is running");
    assert!(detached);
    assert_eq!(
        stable_bytes, previous_bytes,
        "the previous stable file is back"
    );
    assert!(!set_aside_left);
}
