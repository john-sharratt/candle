//! The agent's file (§11.1): the checkouts it holds, laid out as
//! `docs/zend_workspace_execution.md` §7.1 describes — one folder per
//! repository, each with an `origin` — how many jobs it runs at once, and
//! where their logs go.
//!
//! ```yaml
//! jobs_dir: D:/agent/jobs
//! slots: 4
//! stage_dir: D:/prog/candle/keep/stage
//! checkouts:
//!   - name: candle
//!     dir: D:/agent/candle
//!     push: true
//! ```
//!
//! `stage_dir` is where a run that asks for it stages the file it built —
//! this machine's `keep/stage/`, which `keep` tries a build from
//! (`docs/claude_zend_gateway.md` §8.4). Without it, a run asking to stage is
//! refused.

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
    /// This checkout is the memory backbone's journal repository
    /// (`docs/memory_backbone.md` §6.1): the host serves the journals from
    /// it, beside the jobs it takes like any other checkout.
    #[serde(default)]
    pub journal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub jobs_dir: PathBuf,
    /// Jobs run at once, one per checkout at most; half the cores when
    /// absent.
    #[serde(default)]
    pub slots: Option<u32>,
    /// Where a run stages the file it built, for this machine's keep.
    #[serde(default)]
    pub stage_dir: Option<PathBuf>,
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
        let journals: Vec<&CheckoutConfig> = cfg.checkouts.iter().filter(|c| c.journal).collect();
        if journals.len() > 1 {
            anyhow::bail!(
                "`journal: true` is on {}: one checkout is the journal repository",
                journals
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(" and ")
            );
        }
        if let Some(j) = journals.iter().find(|c| !c.push) {
            anyhow::bail!(
                "the checkout `{}` is the journal repository and needs `push: true`: every write is pushed",
                j.name
            );
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
            "jobs_dir: /a/jobs\nslots: 3\nstage_dir: /a/stage\ncheckouts:\n  - name: candle\n    dir: /a/candle\n    push: true\n  - name: site\n    dir: /a/site\n",
        )
        .unwrap();
        assert_eq!(
            cfg,
            AgentConfig {
                jobs_dir: "/a/jobs".into(),
                slots: Some(3),
                stage_dir: Some("/a/stage".into()),
                checkouts: vec![
                    CheckoutConfig {
                        name: "candle".into(),
                        dir: "/a/candle".into(),
                        push: true,
                        journal: false,
                    },
                    CheckoutConfig {
                        name: "site".into(),
                        dir: "/a/site".into(),
                        push: false,
                        journal: false,
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

    /// **One checkout is the journal repository**, and it pushes.
    #[test]
    fn the_journal_checkout_is_one_and_pushes() {
        let cfg = AgentConfig::parse(
            "jobs_dir: j\ncheckouts:\n  - {name: journals, dir: x, push: true, journal: true}\n  - {name: candle, dir: y, push: true}\n",
        )
        .unwrap();
        assert_eq!(
            cfg.checkouts.iter().map(|c| c.journal).collect::<Vec<_>>(),
            [true, false]
        );
        let two = AgentConfig::parse(
            "jobs_dir: j\ncheckouts:\n  - {name: a, dir: x, push: true, journal: true}\n  - {name: b, dir: y, push: true, journal: true}\n",
        )
        .unwrap_err();
        assert!(two.to_string().contains("on a and b"), "{two}");
        let unpushed = AgentConfig::parse(
            "jobs_dir: j\ncheckouts:\n  - {name: journals, dir: x, journal: true}\n",
        )
        .unwrap_err();
        assert!(
            unpushed.to_string().contains("needs `push: true`"),
            "{unpushed}"
        );
    }

    /// **The machines' agent files parse**: blackworld's in-process agent,
    /// which serves the journals, and the laptop's standalone standby, which
    /// holds the same repositories under the same names on checkouts of its
    /// own.
    #[test]
    fn the_shipped_agent_files_parse() {
        let motherd = Path::new(env!("CARGO_MANIFEST_DIR")).join("../motherd");
        let names = |cfg: &AgentConfig| {
            cfg.checkouts
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
        };
        let blackworld = AgentConfig::load(&motherd.join("agent.yaml")).unwrap();
        let laptop = AgentConfig::load(&motherd.join("agent.laptop.yaml")).unwrap();
        assert_eq!(names(&laptop), names(&blackworld));
        assert!(laptop.checkouts.iter().all(|c| c.push && !c.journal));
        assert!(laptop
            .checkouts
            .iter()
            .all(|c| c.dir.starts_with("C:/Users/johna/prog/agent")));
        assert_eq!(laptop.slots(22), 2);
    }
}
