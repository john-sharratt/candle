//! A member's upgrades in the host (`docs/replicated_substrate.md` §3.9): what
//! they start from — this binary's build, its executable and working directory,
//! the UI its tenants serve — and how the process installs a build: it stops
//! the way `shutdown` does, drained, flushed and closed, and its last act hands
//! off to the watchdog ([`hand_off`](cluster::upgrade::handoff::hand_off)).

use std::path::{Path, PathBuf};
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

/// `dir` as a name relative to `root`, `/` between its parts: how the UI
/// archive names it. `None` for a directory outside `root`, which an upgrade
/// cannot replace.
pub fn ui_name(dir: &Path, root: &Path) -> Option<String> {
    let dir = dir.canonicalize().ok()?;
    let root = root.canonicalize().ok()?;
    let rel = dir.strip_prefix(&root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// What a `kind` process launched as `launch` upgrades from: the UI `dirs`
/// its tenants serve, its state beside `state_dir`, and its flags.
pub fn setup(
    kind: &str,
    launch: &LaunchSpec,
    dirs: &[PathBuf],
    state_dir: &Path,
    args: &HostArgs,
) -> Setup {
    let ui = dirs
        .iter()
        .filter_map(|d| {
            let name = ui_name(d, &launch.cwd);
            if name.is_none() {
                tracing::warn!(
                    dir = %d.display(),
                    "a UI directory outside the working directory: an upgrade leaves it as it is"
                );
            }
            name
        })
        .collect();
    Setup {
        info: BuildInfo::of(kind, opens()),
        exe: launch.exe.clone(),
        root: launch.cwd.clone(),
        ui,
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
    use tempfile::TempDir;

    #[test]
    fn a_ui_directory_is_named_from_the_working_directory() {
        let t = TempDir::new().unwrap();
        std::fs::create_dir_all(t.path().join("web/content/common")).unwrap();
        let elsewhere = TempDir::new().unwrap();
        assert_eq!(
            ui_name(&t.path().join("web/content/common"), t.path()).as_deref(),
            Some("web/content/common")
        );
        assert_eq!(ui_name(elsewhere.path(), t.path()), None);
        assert_eq!(
            ui_name(t.path(), t.path()),
            None,
            "the root itself is no UI"
        );
    }

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
