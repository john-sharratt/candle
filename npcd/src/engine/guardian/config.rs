//! `<data>/guardian.yaml`: which modules a guardian is built from and how hard
//! it presses.
//!
//! ```yaml
//! enabled: true
//! scan_every_secs: 45     # how often each character is read
//! cooldown_secs: 180      # least time between two things done to one character
//! settle_secs: 60         # how long a nudge is given to work before it is judged
//! modules:
//!   - kind: drift
//!   - kind: looping
//!   - kind: stall
//!     no_progress_secs: 300
//!   - kind: step_tracker
//!     confirmations: 2
//!   - kind: journal       # asks when a stretch needs a journal entry
//! escalation: [nudge, restate, refresh, flag]
//! ```
//!
//! No file means no guardian. A file that names something this build does not
//! know is refused rather than half-applied.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::engine::guardian::ladder::Rung;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModuleSpec {
    Drift,
    Looping,
    Stall {
        no_progress_secs: u64,
    },
    StepTracker {
        #[serde(default = "default_confirmations")]
        confirmations: u32,
    },
    Journal,
}

fn default_confirmations() -> u32 {
    2
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardianConfig {
    #[serde(default)]
    pub enabled: bool,
    pub scan_every_secs: u64,
    pub cooldown_secs: u64,
    pub settle_secs: u64,
    pub modules: Vec<ModuleSpec>,
    pub escalation: Vec<Rung>,
}

/// Where the file lives under the data directory.
pub fn path(data: &Path) -> PathBuf {
    data.join("guardian.yaml")
}

impl GuardianConfig {
    pub fn parse(yaml: &str) -> anyhow::Result<Self> {
        serde_yaml::from_str(yaml).context("guardian.yaml")
    }

    /// The configuration under `data`, or `None` when there is no file.
    pub fn load(data: &Path) -> anyhow::Result<Option<Self>> {
        let file = path(data);
        if !file.exists() {
            return Ok(None);
        }
        let yaml = std::fs::read_to_string(&file)
            .with_context(|| format!("reading {}", file.display()))?;
        Self::parse(&yaml).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = "
enabled: true
scan_every_secs: 45
cooldown_secs: 180
settle_secs: 60
modules:
  - kind: drift
  - kind: looping
  - kind: stall
    no_progress_secs: 300
  - kind: step_tracker
  - kind: journal
escalation: [nudge, restate, refresh, flag]
";

    #[test]
    fn a_full_file_parses_into_its_modules_and_rungs() {
        let c = GuardianConfig::parse(FULL).unwrap();
        assert!(c.enabled);
        assert_eq!(c.scan_every_secs, 45);
        assert_eq!(
            c.modules,
            vec![
                ModuleSpec::Drift,
                ModuleSpec::Looping,
                ModuleSpec::Stall {
                    no_progress_secs: 300
                },
                ModuleSpec::StepTracker { confirmations: 2 },
                ModuleSpec::Journal,
            ]
        );
        assert_eq!(
            c.escalation,
            vec![Rung::Nudge, Rung::Restate, Rung::Refresh, Rung::Flag]
        );
    }

    #[test]
    fn an_unknown_module_is_refused() {
        let yaml = FULL.replace("kind: drift", "kind: telepathy");
        assert!(GuardianConfig::parse(&yaml).is_err());
    }

    #[test]
    fn an_unknown_field_is_refused() {
        let yaml = format!("{FULL}mood: grim\n");
        assert!(GuardianConfig::parse(&yaml).is_err());
    }

    #[test]
    fn a_missing_file_is_no_guardian() {
        let dir = std::env::temp_dir().join("guardian-config-absent");
        assert_eq!(GuardianConfig::load(&dir).unwrap(), None);
    }
}
