//! The old process's last act (§3.9 steps 3–4): once it has drained, flushed,
//! released its substrate and closed its sockets, it swaps the new build into
//! its executable's path, writes the plan `keep` reads if that build does not
//! become ready, and starts the new process. The new process waits until this
//! one has exited before it loads. This process does not stay up to watch it.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use super::decisions::build_name;
use super::runtime::{InstallJob, Setup};
use super::swap::{restore, swap, Running};
use super::watchdog::{write_plan, WatchPlan, PLAN, READY_WITHIN};
use crate::clock::now_ms;
use crate::launch::{LaunchSpec, AFTER_PID};

/// When the running build was installed: its executable's modification time.
fn installed_ms(launch: &LaunchSpec) -> u64 {
    fs::metadata(&launch.exe)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

/// The launch of the new build: the same command, plus `--after-pid` so it
/// waits until `pid` has exited.
pub fn successor(launch: &LaunchSpec, pid: u32) -> LaunchSpec {
    let pid = pid.to_string();
    launch.with_args(&[AFTER_PID.flag, &pid])
}

/// Swap `job` in for the build `setup` describes, launched as `launch`, write
/// the plan, and start the new process. `log` is the file the new process logs
/// to, whose last lines a failure reports. When the new process cannot be
/// started, the old build is put back and the plan is removed.
pub fn hand_off(
    launch: &LaunchSpec,
    setup: &Setup,
    job: &InstallJob,
    log: Option<PathBuf>,
) -> io::Result<()> {
    let running = Running {
        exe: &launch.exe,
        info: &setup.info,
        installed_ms: installed_ms(launch),
    };
    let swapped = swap(&running, &job.source, now_ms())?;
    let plan = WatchPlan {
        old_pid: std::process::id(),
        launch: launch.clone(),
        swapped: swapped.clone(),
        installing: build_name(&job.build),
        upgrade_dir: setup.upgrade_dir.clone(),
        log,
        ready_within_ms: READY_WITHIN.as_millis() as u64,
        keep_builds: setup.keep_builds,
        protect: job.protect.clone(),
        leases: job.leases.clone(),
    };
    if let Err(e) = write_plan(&plan) {
        restore(&swapped)?;
        return Err(e);
    }
    let started = successor(launch, plan.old_pid).command().spawn();
    if let Err(e) = started {
        tracing::error!(
            build = build_name(&job.build),
            "the new build did not start: putting the old build back: {e}"
        );
        restore(&swapped)?;
        let _ = fs::remove_file(setup.upgrade_dir.join(PLAN));
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::OsString;

    /// The new process is this executable, with the same arguments plus the
    /// predecessor's pid, in the same working directory.
    #[test]
    fn the_successor_is_the_new_executable_after_this_process() {
        let launch = LaunchSpec {
            exe: PathBuf::from("/opt/t/zend"),
            args: vec!["--port".into(), "8080".into()],
            cwd: PathBuf::from("/srv"),
        };
        let w = successor(&launch, 4242);
        assert_eq!(w.exe, PathBuf::from("/opt/t/zend"));
        assert_eq!(
            w.args,
            [
                OsString::from("--port"),
                OsString::from("8080"),
                OsString::from("--after-pid"),
                OsString::from("4242")
            ]
        );
        assert_eq!(w.cwd, PathBuf::from("/srv"));
    }
}
