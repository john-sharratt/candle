//! What a process's upgrades start from: the build it runs, the apps it runs
//! of those the build opens, where its executable is, where it keeps its
//! downloads, and the flags that shape it (§3.9). A build is its executable
//! alone — the UI it serves is compiled in.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::version::info::BuildInfo;

#[derive(Debug, Clone)]
pub struct Setup {
    /// The running build's `--build-info` answer: every app the binary
    /// opens, run here or not — what this member serves its build as.
    pub info: BuildInfo,
    /// The apps this process runs. Only theirs are substrates here, so only
    /// they hold a build that replaces this one to their models.
    pub runs: Vec<String>,
    /// The running executable.
    pub exe: PathBuf,
    /// `<data>/upgrade`: downloads, the watchdog's plan, `ok`, the failure.
    pub upgrade_dir: PathBuf,
    /// `--keep-builds`.
    pub keep_builds: usize,
    /// `--upgrade-failed`: the build an install restored this process from.
    pub failed_from: Option<String>,
}

impl Setup {
    /// The model each app this process runs is locked to: what a build that
    /// replaces it must open.
    pub fn run_models(&self) -> BTreeMap<String, String> {
        self.info
            .opens
            .models
            .iter()
            .filter(|(app, _)| self.runs.contains(app))
            .map(|(app, model)| (app.clone(), model.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::version::info::Opens;

    /// A daemon the binary opens but this process leaves out holds nothing
    /// here, so it is not among the models a replacement must keep.
    #[test]
    fn only_the_apps_run_here_are_held_to_their_models() {
        let info = BuildInfo::of(
            "motherd",
            Opens {
                log_format: Some(3),
                record_tags: vec![1],
                models: [
                    (
                        "npcd".to_string(),
                        "Qwen36_35B_A3B_AntiLoop_StyleTune".to_string(),
                    ),
                    ("zend".to_string(), "Qwen38_FlashNext".to_string()),
                ]
                .into_iter()
                .collect(),
            },
        );
        let setup = Setup {
            info,
            runs: vec!["zend".into()],
            exe: PathBuf::new(),
            upgrade_dir: PathBuf::new(),
            keep_builds: 1,
            failed_from: None,
        };
        assert_eq!(
            setup.run_models(),
            [("zend".to_string(), "Qwen38_FlashNext".to_string())]
                .into_iter()
                .collect()
        );
    }
}
