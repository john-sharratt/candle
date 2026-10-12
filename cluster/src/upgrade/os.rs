//! The watchdog on the machine: [`Processes`] over the OS, and the entry every
//! binary calls first so `--upgrade-watchdog <plan>` runs the watchdog and
//! nothing else (§3.9).

use std::ffi::OsString;
use std::io;
use std::process::Child;
use std::thread::sleep;
use std::time::Duration;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

use super::watchdog::{asks_for_watchdog, read_plan, watch, Outcome, Processes};
use crate::launch::LaunchSpec;

/// The OS's processes.
pub struct OsProcesses;

impl Processes for OsProcesses {
    type Child = Child;

    fn alive(&self, pid: u32) -> bool {
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[Pid::from_u32(pid)]),
            ProcessRefreshKind::new(),
        ) > 0
    }

    fn kill_pid(&self, pid: u32) {
        let mut sys = System::new();
        let pid = Pid::from_u32(pid);
        sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), ProcessRefreshKind::new());
        if let Some(p) = sys.process(pid) {
            p.kill();
        }
    }

    fn spawn(&self, launch: &LaunchSpec) -> io::Result<Child> {
        launch.command().spawn()
    }

    fn exited(&self, child: &mut Child) -> bool {
        !matches!(child.try_wait(), Ok(None))
    }

    fn kill(&self, child: &mut Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    fn sleep(&self, d: Duration) {
        sleep(d);
    }
}

/// When this process was started as `--upgrade-watchdog <plan>`: watch the
/// install the plan describes, and exit. Every binary calls this first.
pub fn run_if_asked() {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let Some(path) = asks_for_watchdog(&args) else {
        return;
    };
    let plan = match read_plan(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "upgrade watchdog: the plan at {} is unreadable: {e}",
                path.display()
            );
            std::process::exit(1);
        }
    };
    match watch(&plan, &OsProcesses) {
        Outcome::Ready => {
            eprintln!("upgrade watchdog: {} is ready", plan.installing);
            std::process::exit(0)
        }
        Outcome::Restored(reason) => {
            eprintln!(
                "upgrade watchdog: {} failed — {reason}; the old build is restored",
                plan.installing
            );
            std::process::exit(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This process is alive and a pid nobody has is not.
    #[test]
    fn liveness_is_read_from_the_os() {
        let p = OsProcesses;
        assert!(p.alive(std::process::id()));
        assert!(!p.alive(u32::MAX - 7));
    }
}
