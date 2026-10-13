//! Process liveness for an upgrade (§3.9). The successor waits here until the
//! predecessor has exited, so the two never load a model at once. Recovery of
//! a build that does not become ready is `keep`'s, not a mode of this binary.

use std::io;
use std::process::Child;
use std::thread::sleep;
use std::time::Duration;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

use super::watchdog::{Processes, LOOK, OLD_EXIT_WITHIN};
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

/// Block until `pid` is gone. After [`OLD_EXIT_WITHIN`] it is killed, which is
/// as long as the successor waits before loading.
pub fn wait_until_gone(pid: u32) {
    let os = OsProcesses;
    let mut waited = Duration::ZERO;
    while os.alive(pid) {
        if waited >= OLD_EXIT_WITHIN {
            os.kill_pid(pid);
            break;
        }
        os.sleep(LOOK);
        waited += LOOK;
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
