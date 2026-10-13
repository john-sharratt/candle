//! zend's own settings — the `zend` binary's flags, and the `zend:` section of
//! `motherd.yaml`. One struct for both, so a setting means the same thing
//! whichever binary reads it. The process's settings (where it listens, whom it
//! believes, how much it logs) are the host's: `host::HostArgs`.
//!
//! [`ZendArgs::resolve`] turns them into a tenant, refusing a bad workspace, a
//! missing secrets file or a malformed `--ingest-dir` in a second, before
//! anything loads.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use candle_conversation::models::{Recent, StrataTokens};
use candle_conversation::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
use candle_conversation::persistence::SUBSTRATE_DIR;
use clap::Args;
use host::OneShot;
use serde::{Deserialize, Deserializer};

use crate::config::{DaemonConfig, SelfCheck};
use crate::secrets;
use crate::tenant::ZendTenant;
use crate::workspace;

/// zend's flags.
#[derive(Args, Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ZendArgs {
    /// The workspace folder: it holds `workspace.yaml`, which lists the
    /// repositories in scope (folders beside it), and the daemon's `substrate/`.
    /// Defaults to the current directory.
    #[arg(default_value = ".")]
    pub workspace: PathBuf,

    /// Override the workspace folder — where `workspace.yaml`, `substrate/` and
    /// an optional `projection.yaml` live. The daemon operates against this dir
    /// WITHOUT changing the terminal's cwd (paths are config-scoped, not a
    /// process `chdir`). Created if it doesn't exist. Takes precedence over the
    /// positional workspace. Use it to run a separate "mind" with its own
    /// substrate + tuned projection schema, e.g. `--working-dir ../mind`.
    #[arg(long)]
    pub working_dir: Option<PathBuf>,

    /// Take a projection sink OUT of service — a turn-sink **layer** (e.g.
    /// `repo_map`, `code_reading`) or a section **collection** (e.g. `response`,
    /// `mood`), by its schema name. Repeatable.
    ///
    /// The layer still exists in the schema, but it is inert: not ingested from
    /// the branches, **excluded from the provenance gather**, and not
    /// normalization-warmed. Its turns stay in the substrate untouched —
    /// nothing is deleted, and re-enabling restores them — but while disabled
    /// they cannot be selected into any projection. A disabled layer is also
    /// NOT cleaned up: its crashed-partial conversations are left as they are.
    #[arg(long = "disable-layer", value_name = "NAME")]
    pub disable_layer: Vec<String>,

    /// Tombstone EVERY conversation in a turn-sink layer, by its schema name
    /// (e.g. `repo_map`, `code_reading`), before the background ingest worker's
    /// first pass. Repeatable. Only the named layer's content is destroyed —
    /// the live dialogue, any other ingest layer, and uploads all survive; the
    /// worker's first pass rebuilds it from the branches. A layer named by
    /// `--disable-layer` is not wiped. Raw (ChatML) layers are not wipeable
    /// this way.
    #[arg(long = "wipe-layer", value_name = "NAME")]
    #[serde(skip)]
    pub wipe_layer: Vec<String>,

    /// Override the folder a derived ingest layer reads from, as
    /// `<layer>=<path>`. Repeatable (e.g. `--ingest-dir
    /// code_reading=candle/zend/src`). The path is relative to the workspace
    /// folder; for `repo_map` and `code_reading` it names a folder inside one
    /// repository. Scopes an ingest to a subtree so a rebuilt substrate stays
    /// small instead of absorbing every repository.
    #[arg(long = "ingest-dir", value_name = "LAYER=PATH")]
    pub ingest_dir: Vec<String>,

    /// Bound how deep the `repo_map` and `code_reading` layers read, in path
    /// components below each repository's root (or a layer's `--ingest-dir`
    /// folder) — `1` is the root's own files, `2` adds one folder down, and so
    /// on, like `find -maxdepth`. Content already ingested from below the bound
    /// is retired like anything else no branch holds within the layer's reach.
    /// Unbounded when omitted.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub max_depth: Option<u32>,

    /// Force a whole-store redo-log compaction once during load (after the
    /// substrate reload, before serving) instead of leaving reclaim to the
    /// incremental background maintenance pass.
    #[arg(long)]
    #[serde(skip)]
    pub compact_substrate: bool,

    /// Ask every stored conversation — the dialogue and the `repo_map` /
    /// `code_reading` layers — four yes/no questions about its own integrity
    /// during load, and tombstone each one that answers no to any of them. An
    /// ingest conversation tombstoned this way is read again by the ingest
    /// that follows; a dialogue is gone. Pair with `--dry-run` first.
    #[arg(long)]
    #[serde(skip)]
    pub self_check: bool,

    /// With `--self-check`: report every verdict and tombstone nothing.
    #[arg(long, requires = "self_check")]
    #[serde(skip)]
    pub dry_run: bool,

    /// DESTRUCTIVE: delete the working dir's `substrate` directory (redo-log
    /// segments, logs — the daemon's entire persistent memory) before loading,
    /// so this run starts from a blank substrate. Exists precisely so scripts
    /// and test harnesses never have to `rm -rf` a substrate path themselves.
    #[arg(long)]
    #[serde(skip)]
    pub wipe_substrate: bool,

    /// Run the QSA selection with this many positions in place of the
    /// checkpoint's own budget. A budget of at least the checkpoint's context
    /// makes the selection the identity — every cell is read — which is the
    /// dense control for a selection question. Refused at load for a model
    /// whose attention does not select, and for a budget the selection kernel
    /// cannot run.
    #[arg(long, value_name = "N")]
    pub qsa_selection_budget: Option<usize>,

    /// How a selecting attention divides the context
    /// (`docs/qsa_stratified_selection.md`): windows of this many positions,
    /// each spending the whole selection budget on its own blocks plus the
    /// system prompt's. `0` with `--qsa-recent 0` is one window over everything
    /// — the checkpoint's own selection.
    #[arg(long, value_name = "POSITIONS", default_value_t = StrataTokens::DEFAULT.window)]
    pub qsa_window: usize,

    /// The span of this many positions nearest the query that a stratified
    /// selection treats apart: ranked in every window, or attended whole —
    /// see `--qsa-recent-mode`.
    #[arg(long, value_name = "POSITIONS", default_value_t = StrataTokens::DEFAULT.recent)]
    pub qsa_recent: usize,

    /// How the `--qsa-recent` span enters the selection: `candidate` ranks it
    /// in every window beside that window's own blocks; `forced` attends it
    /// whole and ranks it nowhere.
    #[arg(long, value_name = "MODE", value_parser = parse_recent_mode, default_value = "candidate")]
    #[serde(deserialize_with = "recent_mode")]
    pub qsa_recent_mode: Recent,

    /// Let conversations launch background tree summaries — a summary of the
    /// last eight turns, of accumulated segments, and at each UTC day boundary.
    /// Off by default: each summary re-reads its window as a fresh prefill on
    /// the same scheduler, ahead of the live turn.
    #[arg(long)]
    pub summarize: bool,

    /// How many days the substrate keeps a deleted conversation's tombstone
    /// before a compaction may shrink it to a 24-byte entry in the deleted set
    /// (`docs/replicated_substrate.md` §6.8). Only when the bytes are reclaimed
    /// depends on it: the conversation stays deleted either way.
    #[arg(long, value_name = "DAYS", default_value_t = DEFAULT_TOMBSTONE_RETENTION_DAYS)]
    pub tombstone_retention: u64,

    /// The secrets file holding the API keys and tokens zend presents to
    /// third-party services (`tavily_api_key`, `github_token`). Defaults to
    /// `~/.zend/secrets.yaml`. A named file that does not exist fails the
    /// launch.
    #[arg(long, value_name = "PATH")]
    pub secrets: Option<PathBuf>,
}

impl Default for ZendArgs {
    fn default() -> Self {
        Self {
            workspace: PathBuf::from("."),
            working_dir: None,
            disable_layer: Vec::new(),
            wipe_layer: Vec::new(),
            ingest_dir: Vec::new(),
            max_depth: None,
            compact_substrate: false,
            self_check: false,
            dry_run: false,
            wipe_substrate: false,
            qsa_selection_budget: None,
            qsa_window: StrataTokens::DEFAULT.window,
            qsa_recent: StrataTokens::DEFAULT.recent,
            qsa_recent_mode: StrataTokens::DEFAULT.mode,
            summarize: false,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION_DAYS,
            secrets: None,
        }
    }
}

/// `--tombstone-retention`'s default, in days: the substrate's own.
const DEFAULT_TOMBSTONE_RETENTION_DAYS: u64 =
    DEFAULT_TOMBSTONE_RETENTION.as_secs() / SECONDS_PER_DAY;

const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

/// A `--qsa-recent-mode` value.
fn parse_recent_mode(mode: &str) -> Result<Recent, String> {
    match mode {
        "candidate" => Ok(Recent::Candidate),
        "forced" => Ok(Recent::Forced),
        other => Err(format!(
            "unknown recent mode {other:?}; expected candidate or forced"
        )),
    }
}

/// `qsa_recent_mode:` in YAML, spelled as the flag is.
fn recent_mode<'de, D: Deserializer<'de>>(d: D) -> Result<Recent, D::Error> {
    let s = String::deserialize(d)?;
    parse_recent_mode(&s).map_err(serde::de::Error::custom)
}

impl ZendArgs {
    /// The flags that act once, at the start they are given to: a wipe, a
    /// whole-store compaction, a self-check. A self-heal relaunch drops them
    /// (`host::OneShot`), and `motherd.yaml` cannot set them — a config file
    /// is read at every start, so one there would act at every start. Run the
    /// `zend` binary with the flag once instead.
    pub const ONE_SHOT: &'static [OneShot] = &[
        OneShot {
            flag: "--wipe-substrate",
            takes_value: false,
        },
        OneShot {
            flag: "--wipe-layer",
            takes_value: true,
        },
        OneShot {
            flag: "--compact-substrate",
            takes_value: false,
        },
        OneShot {
            flag: "--self-check",
            takes_value: false,
        },
        OneShot {
            flag: "--dry-run",
            takes_value: false,
        },
    ];

    /// The workspace folder these settings name: `--working-dir` when given,
    /// else the positional path.
    pub fn root(&self) -> PathBuf {
        self.working_dir
            .clone()
            .unwrap_or_else(|| self.workspace.clone())
    }

    /// The stratified selection the `--qsa-*` flags describe.
    pub fn qsa_strata(&self) -> StrataTokens {
        StrataTokens {
            window: self.qsa_window,
            recent: self.qsa_recent,
            mode: self.qsa_recent_mode,
        }
    }

    /// `--ingest-dir <layer>=<path>`, each checked against `root`: a typo
    /// would otherwise silently ingest nothing.
    fn ingest_dirs(&self, root: &Path) -> anyhow::Result<HashMap<String, String>> {
        let mut dirs = HashMap::new();
        for spec in &self.ingest_dir {
            let Some((layer, path)) = spec.split_once('=') else {
                anyhow::bail!("invalid --ingest-dir {spec:?}: expected <layer>=<path>");
            };
            let (layer, path) = (layer.trim(), path.trim());
            if layer.is_empty() || path.is_empty() {
                anyhow::bail!(
                    "invalid --ingest-dir {spec:?}: both <layer> and <path> are required"
                );
            }
            if !root.join(path).is_dir() {
                anyhow::bail!(
                    "invalid --ingest-dir {spec:?}: {} is not a directory",
                    root.join(path).display(),
                );
            }
            dirs.insert(layer.to_string(), path.to_string());
        }
        Ok(dirs)
    }

    /// zend, as a tenant: the workspace opened, the secrets read, the flags
    /// checked — and the substrate deleted first when `--wipe-substrate` asks.
    /// Runs before logging is up, so what it reports goes to the terminal.
    pub fn resolve(self) -> anyhow::Result<ZendTenant> {
        let root = self.root();
        // A fresh mind directory needs somewhere for `substrate` and
        // `projection.yaml` to land.
        fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        if self.wipe_substrate {
            wipe_substrate(&root)?;
        }
        let workspace = workspace::open(&root)?;
        let secrets = secrets::load(self.secrets.as_deref())?;
        let ingest_dirs = self.ingest_dirs(&root)?;
        let mut config = DaemonConfig::new(workspace);
        config.secrets = secrets;
        config.disabled_layers = self.disable_layer.iter().cloned().collect::<HashSet<_>>();
        config.wiped_layers = self.wipe_layer.iter().cloned().collect();
        config.ingest_dirs = ingest_dirs;
        config.max_depth = self.max_depth.map(|d| d as usize);
        config.compact_substrate = self.compact_substrate;
        config.self_check = match (self.self_check, self.dry_run) {
            (false, _) => SelfCheck::Off,
            (true, true) => SelfCheck::DryRun,
            (true, false) => SelfCheck::Tombstone,
        };
        config.qsa_selection_budget = self.qsa_selection_budget;
        config.qsa_strata = self.qsa_strata();
        config.summarize = self.summarize;
        config.tombstone_retention =
            Duration::from_secs(self.tombstone_retention.saturating_mul(SECONDS_PER_DAY));
        Ok(ZendTenant::new(config))
    }
}

/// Delete `workspace/substrate` — the daemon's entire persistent memory —
/// scoped strictly to the RESOLVED working dir's own `substrate`. The only
/// caller is the explicit `--wipe-substrate` flag; scripts and harnesses go
/// through it so no shell ever `rm -rf`s a substrate path itself. A missing
/// directory is a no-op (a fresh mind dir); a failure aborts the launch rather
/// than half-deleting.
fn wipe_substrate(workspace: &Path) -> anyhow::Result<()> {
    let substrate_dir = workspace.join(SUBSTRATE_DIR);
    if substrate_dir.exists() {
        eprintln!(
            "--wipe-substrate: deleting {} (persistent substrate)",
            substrate_dir.display()
        );
        if let Err(e) = fs::remove_dir_all(&substrate_dir) {
            anyhow::bail!(
                "--wipe-substrate: failed to delete {}: {e}",
                substrate_dir.display()
            );
        }
    } else {
        eprintln!(
            "--wipe-substrate: {} does not exist — nothing to delete",
            substrate_dir.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        zend: ZendArgs,
    }

    fn parse(args: &[&str]) -> ZendArgs {
        let mut all = vec!["zend"];
        all.extend_from_slice(args);
        Cli::try_parse_from(all).unwrap().zend
    }

    #[test]
    fn wipes_only_the_workspaces_own_substrate() {
        let root = tempfile::tempdir().unwrap();
        let ws = root.path().join("mind");
        fs::create_dir_all(ws.join(SUBSTRATE_DIR)).unwrap();
        fs::write(ws.join(SUBSTRATE_DIR).join("seg-0.log"), b"data").unwrap();
        // A sibling workspace's substrate must be untouched.
        let other = root.path().join("other");
        fs::create_dir_all(other.join(SUBSTRATE_DIR)).unwrap();
        fs::write(other.join(SUBSTRATE_DIR).join("seg-0.log"), b"keep").unwrap();

        wipe_substrate(&ws).unwrap();
        assert!(!ws.join(SUBSTRATE_DIR).exists(), "own substrate deleted");
        assert!(
            other.join(SUBSTRATE_DIR).join("seg-0.log").exists(),
            "sibling substrate untouched"
        );
        // Idempotent: a second wipe of the now-missing dir is a no-op.
        wipe_substrate(&ws).unwrap();
    }

    /// `--max-depth` takes a positive component count; absent means unbounded,
    /// and 0 is refused — a walk of nothing would freeze the whole layer.
    #[test]
    fn max_depth_parses_a_positive_count_and_refuses_zero() {
        assert_eq!(parse(&["--max-depth", "3"]).max_depth, Some(3));
        assert_eq!(parse(&[]).max_depth, None);
        assert!(Cli::try_parse_from(["zend", "--max-depth", "0"]).is_err());
    }

    /// With no `--qsa-*` flags zend runs the measured default — 128K windows
    /// and an 8K recent candidate; `0` and `0` is the checkpoint's own
    /// selection, and each flag replaces its own part.
    #[test]
    fn the_qsa_flags_default_to_the_measured_strata() {
        assert_eq!(
            parse(&[]).qsa_strata(),
            StrataTokens {
                window: 131_072,
                recent: 8192,
                mode: Recent::Candidate,
            }
        );
        assert_eq!(
            parse(&["--qsa-window", "0", "--qsa-recent", "0"]).qsa_strata(),
            StrataTokens::WHOLE
        );
        assert_eq!(
            parse(&["--qsa-window", "65536", "--qsa-recent-mode", "forced"]).qsa_strata(),
            StrataTokens {
                window: 65_536,
                recent: 8192,
                mode: Recent::Forced,
            }
        );
        assert!(Cli::try_parse_from(["zend", "--qsa-recent-mode", "sometimes"]).is_err());
    }

    /// The flags and `motherd.yaml`'s `zend:` section read the same settings,
    /// with the same defaults.
    #[test]
    fn flags_and_yaml_agree() {
        assert_eq!(parse(&[]), ZendArgs::default());
        assert_eq!(
            serde_yaml::from_str::<ZendArgs>("{}").unwrap(),
            ZendArgs::default()
        );
        let flags = parse(&[
            "--working-dir",
            "../mind",
            "--disable-layer",
            "repo_map",
            "--max-depth",
            "2",
            "--qsa-recent-mode",
            "forced",
            "--summarize",
        ]);
        let yaml: ZendArgs = serde_yaml::from_str(
            "working_dir: ../mind\ndisable_layer: [repo_map]\nmax_depth: 2\n\
             qsa_recent_mode: forced\nsummarize: true\n",
        )
        .unwrap();
        assert_eq!(flags, yaml);
        assert!(serde_yaml::from_str::<ZendArgs>("qsa_recent_mode: sometimes").is_err());
        assert!(serde_yaml::from_str::<ZendArgs>("model: Qwen35_0_8B_Q8").is_err());
    }

    /// Every one-shot flag is a real flag of the binary — a relaunch strips
    /// what the shell passed — and none can be set from `motherd.yaml`.
    #[test]
    fn the_one_shot_flags_are_flags_and_never_yaml() {
        for o in ZendArgs::ONE_SHOT {
            let mut args = vec!["zend", o.flag];
            if o.takes_value {
                args.push("repo_map");
            }
            if o.flag == "--dry-run" {
                args.push("--self-check");
            }
            assert!(Cli::try_parse_from(&args).is_ok(), "{}", o.flag);
            let key = o.flag.trim_start_matches("--").replace('-', "_");
            let yaml = format!("{key}: true");
            assert!(
                serde_yaml::from_str::<ZendArgs>(&yaml).is_err(),
                "{key} must not be settable from motherd.yaml"
            );
        }
    }

    /// A malformed or missing `--ingest-dir` refuses the launch, naming it.
    #[test]
    fn ingest_dirs_are_checked_against_the_workspace() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("candle").join("zend")).unwrap();
        let ok = parse(&["--ingest-dir", "repo_map=candle/zend"]);
        assert_eq!(
            ok.ingest_dirs(root.path()).unwrap(),
            HashMap::from([("repo_map".to_string(), "candle/zend".to_string())])
        );
        for bad in ["repo_map", "=x", "repo_map=nope"] {
            let a = parse(&["--ingest-dir", bad]);
            let err = a.ingest_dirs(root.path()).unwrap_err().to_string();
            assert!(err.starts_with("invalid --ingest-dir"), "{bad}: {err}");
        }
    }
}
