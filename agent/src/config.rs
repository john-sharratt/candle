//! The agent's file (§11.1): the checkouts it holds, laid out as
//! `docs/zend_workspace_execution.md` §7.1 describes — one folder per
//! repository, each with an `origin` — how long a restart waits for a deploy
//! build, and where the jobs' logs go. An agent runs one build at a time
//! (`docs/build_agent.md` §2).
//!
//! ```yaml
//! jobs_dir: D:/agent/jobs
//! build_hold_mins: 20       # optional; the default
//! stage_dir: D:/prog/candle/keep/stage
//! checkouts:
//!   - name: candle
//!     dir: D:/agent/candle
//!     push: true
//! sweep:                    # optional; these are the defaults
//!   low_free_gb: 150        # a volume is low under the larger of these two
//!   low_free_percent: 10
//!   worth_gb: 10            # a prune or a clean must bring back at least this
//! ```
//!
//! `build_hold_mins` is the longest a planned restart waits for the deploy
//! build that is running (`docs/build_agent.md` §3): longer than a normal
//! deploy build, short enough that one that hangs does not hold the restart
//! for long.
//!
//! `sweep` is when a run, as it takes its checkout, first takes back what the
//! checkouts' build outputs hold ([`crate::sweep`]): prune, then clean.
//!
//! What a checkout runs when the agent starts is not this file's to say: each
//! repository declares it in its own `.agent/setup.yaml` ([`crate::setup`]), so
//! every machine holding the repository runs the same setup.
//!
//! `stage_dir` is where a run that asks for it stages the file it built —
//! this machine's `keep/stage/`, which `keep` tries a build from
//! (`docs/claude_zend_gateway.md` §8.4). Without it, a run asking to stage is
//! refused.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

/// How long a restart waits for a running deploy build, when the file names
/// no `build_hold_mins`.
pub const BUILD_HOLD_MINS: u64 = 20;

fn build_hold_mins() -> u64 {
    BUILD_HOLD_MINS
}

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

/// When the agent sweeps its checkouts' build outputs ([`crate::sweep`]): when
/// the free space on a checkout's volume is below the larger of `low_free_gb`
/// and `low_free_percent` of the volume, and only where at least `worth_gb`
/// would come back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SweepConfig {
    pub low_free_gb: u64,
    pub low_free_percent: u8,
    pub worth_gb: u64,
}

impl Default for SweepConfig {
    fn default() -> Self {
        SweepConfig {
            low_free_gb: 150,
            low_free_percent: 10,
            worth_gb: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub jobs_dir: PathBuf,
    /// The longest a planned restart waits for a running deploy build, in
    /// minutes.
    #[serde(default = "build_hold_mins")]
    pub build_hold_mins: u64,
    /// Where a run stages the file it built, for this machine's keep.
    #[serde(default)]
    pub stage_dir: Option<PathBuf>,
    pub checkouts: Vec<CheckoutConfig>,
    /// When its checkouts' build outputs are swept.
    #[serde(default)]
    pub sweep: SweepConfig,
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

    /// The longest a planned restart waits for a running deploy build.
    pub fn build_hold(&self) -> Duration {
        Duration::from_secs(self.build_hold_mins * 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_names_its_checkouts_and_its_build_hold() {
        let cfg = AgentConfig::parse(
            "jobs_dir: /a/jobs\nbuild_hold_mins: 30\nstage_dir: /a/stage\ncheckouts:\n  - name: candle\n    dir: /a/candle\n    push: true\n  - name: site\n    dir: /a/site\n",
        )
        .unwrap();
        assert_eq!(
            cfg,
            AgentConfig {
                jobs_dir: "/a/jobs".into(),
                build_hold_mins: 30,
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
                sweep: SweepConfig::default(),
            }
        );
        assert_eq!(cfg.build_hold(), Duration::from_secs(30 * 60));
    }

    /// **The sweep's bars default, and each can be set alone.**
    #[test]
    fn the_sweep_defaults_and_each_bar_can_be_set() {
        let cfg = AgentConfig::parse("jobs_dir: j\ncheckouts: []\n").unwrap();
        assert_eq!(cfg.sweep, SweepConfig::default());
        let cfg =
            AgentConfig::parse("jobs_dir: j\ncheckouts: []\nsweep: {worth_gb: 25}\n").unwrap();
        assert_eq!(
            cfg.sweep,
            SweepConfig {
                worth_gb: 25,
                ..SweepConfig::default()
            }
        );
        assert!(AgentConfig::parse("jobs_dir: j\ncheckouts: []\nsweep: {often: 1}\n").is_err());
    }

    /// **The build hold defaults to 20 minutes, `slots` is no setting any
    /// more, and checkout names are unique.**
    #[test]
    fn the_build_hold_defaults_and_names_are_unique() {
        let cfg = AgentConfig::parse("jobs_dir: j\ncheckouts: []\n").unwrap();
        assert_eq!(cfg.build_hold(), Duration::from_secs(BUILD_HOLD_MINS * 60));
        assert!(AgentConfig::parse("jobs_dir: j\nslots: 4\ncheckouts: []\n").is_err());
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
        // Both take the default: a file naming `build_hold_mins` would stop a
        // build from before it, which refuses keys it does not know.
        assert_eq!(blackworld.build_hold_mins, BUILD_HOLD_MINS);
        assert_eq!(laptop.build_hold_mins, BUILD_HOLD_MINS);
    }
}
