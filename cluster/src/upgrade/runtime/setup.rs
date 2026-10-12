//! What a process's upgrades start from: the build it runs, where its
//! executable is, where it keeps its downloads, and the flags that shape it
//! (§3.9). A build is its executable alone — the UI it serves is compiled in.

use std::path::PathBuf;

use crate::version::info::BuildInfo;

#[derive(Debug, Clone)]
pub struct Setup {
    /// The running build's `--build-info` answer.
    pub info: BuildInfo,
    /// The running executable.
    pub exe: PathBuf,
    /// `<data>/upgrade`: downloads, the watchdog's plan, `ok`, the failure.
    pub upgrade_dir: PathBuf,
    /// `--keep-builds`.
    pub keep_builds: usize,
    /// `--upgrade-failed`: the build an install restored this process from.
    pub failed_from: Option<String>,
}
