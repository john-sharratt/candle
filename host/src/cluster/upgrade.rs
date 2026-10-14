//! A member's upgrades in the host (`docs/replicated_substrate.md` §3.9): what
//! they start from — this binary's build and its executable, its tenants' UI
//! compiled in — and how the process installs a build: it stops the way
//! `shutdown` does, drained, flushed and closed, and its last act stages the
//! new executable and asks `keep` to start it
//! ([`hand_off`](cluster::upgrade::handoff::hand_off)).

use std::path::Path;
use std::sync::{Arc, Mutex};

use cluster::launch::LaunchSpec;
use cluster::upgrade::runtime::{InstallJob, Installer, Setup};
use cluster::version::info::BuildInfo;

use crate::build_info::opens;
use crate::cli::HostArgs;
use crate::stopping::{Ending, Stopping};

/// Where a member keeps its downloads, plan and markers: beside its log.
pub const UPGRADE_DIR: &str = "upgrade";

/// The process side of an install: the install is asked of the stop, and run
/// once every tenant has stopped.
pub struct HostInstaller {
    stopping: Arc<Stopping>,
    job: Mutex<Option<InstallJob>>,
}

impl HostInstaller {
    pub fn new(stopping: Arc<Stopping>) -> Arc<HostInstaller> {
        Arc::new(HostInstaller {
            stopping,
            job: Mutex::new(None),
        })
    }

    /// The install asked for, once the process has stopped.
    pub fn take(&self) -> Option<InstallJob> {
        self.job.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

impl Installer for HostInstaller {
    fn install(&self, job: InstallJob) -> Result<(), String> {
        let mut held = self.job.lock().unwrap_or_else(|e| e.into_inner());
        if self.stopping.ending().is_some() {
            return Err("the process is already stopping".into());
        }
        *held = Some(job);
        self.stopping.ask(Ending::Install);
        Ok(())
    }
}

/// What a `kind` process launched as `launch` upgrades from: its executable,
/// its state beside `state_dir`, and its flags.
pub fn setup(kind: &str, launch: &LaunchSpec, state_dir: &Path, args: &HostArgs) -> Setup {
    Setup {
        info: BuildInfo::of(kind, opens()),
        exe: launch.exe.clone(),
        upgrade_dir: state_dir.join(UPGRADE_DIR),
        keep_builds: args.keep_builds,
        failed_from: args.upgrade_failed.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cluster::frame::BuildId;
    use cluster::upgrade::swap::Source;

    /// An install is asked of the stop, and handed over once; a second, or
    /// one while the process already stops, is refused.
    #[test]
    fn an_install_is_asked_of_the_stop() {
        let stopping = Arc::new(Stopping::new());
        let i = HostInstaller::new(stopping.clone());
        let job = InstallJob {
            build: BuildId {
                count: 12,
                hash: "h12".into(),
                local: String::new(),
            },
            source: Source::Kept("12-h12".into()),
            leases: vec!["upgrade/zend".into()],
            protect: vec![],
        };
        assert_eq!(i.install(job.clone()), Ok(()));
        assert_eq!(stopping.ending(), Some(Ending::Install));
        assert_eq!(
            i.install(job.clone()).unwrap_err(),
            "the process is already stopping"
        );
        assert_eq!(i.take(), Some(job));
        assert_eq!(i.take(), None);
    }
}
