//! What a process's upgrades start from: the build it runs, where its
//! executable and UI are, where it keeps its downloads, and the flags that
//! shape it (§3.9).

use std::path::PathBuf;

use crate::version::info::BuildInfo;

#[derive(Debug, Clone)]
pub struct Setup {
    /// The running build's `--build-info` answer.
    pub info: BuildInfo,
    /// The running executable.
    pub exe: PathBuf,
    /// The working directory, which the UI names are relative to.
    pub root: PathBuf,
    /// The UI directories it serves, by name: `zend/web`. Empty for an agent.
    pub ui: Vec<String>,
    /// `<data>/upgrade`: downloads, the watchdog's plan, `ok`, the failure.
    pub upgrade_dir: PathBuf,
    /// `--keep-builds`.
    pub keep_builds: usize,
    /// `--upgrade-failed`: the build an install restored this process from.
    pub failed_from: Option<String>,
}
