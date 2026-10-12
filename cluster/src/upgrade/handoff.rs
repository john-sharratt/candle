//! The old process's last act (§3.9 steps 3–4): once it has drained, flushed,
//! released its substrate and closed its sockets, it swaps the new build into
//! its executable's and UI's paths, writes the watchdog's plan, and starts the
//! watchdog — itself, from its kept-build directory — then exits. The watchdog
//! waits for it to be gone before it starts the new build, so the two daemons
//! never run at once.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use super::decisions::build_name;
use super::runtime::{InstallJob, Setup};
use super::swap::{restore, swap, Running};
use super::watchdog::{write_plan, WatchPlan, READY_WITHIN};
use crate::clock::now_ms;
use crate::launch::{LaunchSpec, UPGRADE_WATCHDOG};

/// When the running build was installed: its executable's modification time.
fn installed_ms(launch: &LaunchSpec) -> u64 {
    fs::metadata(&launch.exe)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

/// The watchdog's launch: the old executable, now in `kept`, run with the plan.
pub fn watchdog_launch(launch: &LaunchSpec, kept: &Path, plan: &Path) -> io::Result<LaunchSpec> {
    let name = launch
        .exe
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the executable has no name"))?;
    Ok(LaunchSpec {
        exe: kept.join(name),
        args: vec![UPGRADE_WATCHDOG.flag.into(), plan.as_os_str().to_owned()],
        cwd: launch.cwd.clone(),
    })
}

/// Swap `job` in for the build `setup` describes, launched as `launch`, write
/// the plan, and start the watchdog. `log` is the file the new process logs
/// to, whose last lines a failure reports. On an error after the swap, the old
/// build is put back.
pub fn hand_off(
    launch: &LaunchSpec,
    setup: &Setup,
    job: &InstallJob,
    log: Option<PathBuf>,
) -> io::Result<()> {
    let running = Running {
        exe: &launch.exe,
        root: &launch.cwd,
        ui: &setup.ui,
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
    let started = write_plan(&plan).and_then(|path| {
        watchdog_launch(launch, &swapped.kept, &path)?
            .command()
            .spawn()
            .map(|_| ())
    });
    if let Err(e) = started {
        tracing::error!(
            build = build_name(&job.build),
            "the watchdog did not start: putting the old build back: {e}"
        );
        restore(&swapped)?;
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::OsString;

    /// The watchdog is the old executable from its kept directory, given only
    /// its plan, in the same working directory.
    #[test]
    fn the_watchdog_is_the_old_executable_with_its_plan() {
        let launch = LaunchSpec {
            exe: PathBuf::from("/opt/t/zend"),
            args: vec!["--port".into(), "8080".into()],
            cwd: PathBuf::from("/srv"),
        };
        let w = watchdog_launch(
            &launch,
            Path::new("/opt/t/builds/10-h10"),
            Path::new("/d/upgrade/watch.json"),
        )
        .unwrap();
        assert_eq!(w.exe, PathBuf::from("/opt/t/builds/10-h10/zend"));
        assert_eq!(
            w.args,
            [
                OsString::from("--upgrade-watchdog"),
                OsString::from("/d/upgrade/watch.json")
            ]
        );
        assert_eq!(w.cwd, PathBuf::from("/srv"));
    }
}
