//! The agent's file (§11.1): the checkouts it holds, laid out as
//! `docs/zend_workspace_execution.md` §7.1 describes — one folder per
//! repository, each with an `origin` — how many jobs it runs at once, and
//! where their logs go.
//!
//! ```yaml
//! jobs_dir: D:/agent/jobs
//! slots: 4
//! checkouts:
//!   - name: candle
//!     dir: D:/agent/candle
//!     push: true
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One repository the agent holds.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckoutConfig {
    /// Its name in the workspaces that use it.
    pub name: String,
    pub dir: PathBuf,
    /// The agent holds the credentials to push to its origin.
    #[serde(default)]
    pub push: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub jobs_dir: PathBuf,
    /// Jobs run at once, one per checkout at most; half the cores when
    /// absent.
    #[serde(default)]
    pub slots: Option<u32>,
    pub checkouts: Vec<CheckoutConfig>,
}

impl AgentConfig {
    pub fn parse(text: &str) -> anyhow::Result<AgentConfig> {
        let cfg: AgentConfig = serde_yaml::from_str(text)?;
        let mut names: Vec<&str> = cfg.checkouts.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            anyhow::bail!("the checkout `{}` is named twice", w[0]);
        }
        Ok(cfg)
    }

    pub fn load(path: &Path) -> anyhow::Result<AgentConfig> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }

    /// Jobs run at once: as configured, or half of `cores`, at least one.
    pub fn slots(&self, cores: u16) -> u32 {
        self.slots.unwrap_or_else(|| u32::from(cores / 2)).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_names_its_checkouts_and_slots() {
        let cfg = AgentConfig::parse(
            "jobs_dir: /a/jobs\nslots: 3\ncheckouts:\n  - name: candle\n    dir: /a/candle\n    push: true\n  - name: site\n    dir: /a/site\n",
        )
        .unwrap();
        assert_eq!(
            cfg,
            AgentConfig {
                jobs_dir: "/a/jobs".into(),
                slots: Some(3),
                checkouts: vec![
                    CheckoutConfig {
                        name: "candle".into(),
                        dir: "/a/candle".into(),
                        push: true,
                    },
                    CheckoutConfig {
                        name: "site".into(),
                        dir: "/a/site".into(),
                        push: false,
                    },
                ],
            }
        );
        assert_eq!(cfg.slots(16), 3);
    }

    #[test]
    fn slots_default_to_half_the_cores_and_names_are_unique() {
        let cfg = AgentConfig::parse("jobs_dir: j\ncheckouts: []\n").unwrap();
        assert_eq!((cfg.slots(16), cfg.slots(1)), (8, 1));
        assert!(AgentConfig::parse(
            "jobs_dir: j\ncheckouts:\n  - {name: a, dir: x}\n  - {name: a, dir: y}\n"
        )
        .is_err());
        assert!(AgentConfig::parse("jobs_dir: j\ncheckouts: []\nshell: bash\n").is_err());
    }
}
