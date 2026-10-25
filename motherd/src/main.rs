//! `motherd` — zend and npcd in one process, on one model.
//!
//! The same startup path as `zend` and `npcd` — `host::run` — with both tenants,
//! equal on the scheduler (`docs/shared_model_host.md` §3.6): either daemon's
//! live conversation outranks both daemons' ingest, and their ingests run side
//! by side. One model loads, one engine runs, each daemon keeps its own
//! substrate, and one HTTP server serves both — the API on every name, each UI
//! on its own (`code.tokera.com` and a bare IP: zend's; `bot.tokera.com`:
//! npcd's).
//!
//! It reads `motherd.yaml` — the process's settings under `host:`, and each
//! daemon's under its name, in the same shape as that daemon's own flags:
//!
//! ```yaml
//! host:
//!   host: 192.168.0.5
//!   port: 8081
//!   tls_port: 8444
//! zend:
//!   working_dir: ..
//! npcd:
//!   mind_repo: battle-mind
//!   data: ../npcd-data
//! ```
//!
//! `apps: [npcd]` (or `[zend]`) runs one daemon alone — it alone serves, and
//! joins the cluster — for taking one app out of a member while the other
//! goes on elsewhere. The daemon left out still asks what it asks of the
//! engine, read from its own section (zend's from its workspace's projection
//! schema), so the engine is the one the two build together — its conversation
//! capacity included. The process's log, offline marker and upgrade record
//! stay beside zend's substrate either way.
//!
//! Paths resolve against the working directory — the repository root, which
//! every daemon runs from — so from `D:\prog\candle` that is zend's workspace
//! `D:\prog` and npcd's data folder `D:\prog\npcd-data`. `--config` defaults to
//! `motherd.yaml` in the working directory; the shipped example is
//! `motherd/motherd.yaml` (blackworld's), and `motherd/motherd.laptop.yaml` is
//! the laptop's.

use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use cluster::version::info::MOTHERD;
use host::build_info::first;
use host::{run_seated, HostArgs, HostConfig, Seat};
use npcd::tenant::NpcdArgs;
use serde::Deserialize;
use zend::cli::ZendArgs;

#[derive(Parser, Debug)]
#[command(name = "motherd", about = "zend and npcd in one process, on one model")]
struct Cli {
    /// The configuration file.
    #[arg(long, default_value = "motherd.yaml")]
    config: PathBuf,
    /// The build an install restored this process from (`host::HostArgs`):
    /// set by `keep` when it restores a failed install, never by the file.
    #[arg(long, value_name = "COUNT-HASH")]
    upgrade_failed: Option<String>,
    /// Wait until this pid has exited before loading. The predecessor sets it
    /// when it starts this process.
    #[arg(long, value_name = "PID")]
    after_pid: Option<u32>,
}

/// A daemon this process can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum App {
    Zend,
    Npcd,
}

/// `motherd.yaml`.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
struct Config {
    /// The daemons to run — `[npcd]` serves and joins the cluster as npcd
    /// alone. Both when absent. A daemon left out still has its section read
    /// for its asks of the engine, and zend's for where the process's state is.
    apps: Option<Vec<App>>,
    host: HostArgs,
    zend: ZendArgs,
    npcd: NpcdArgs,
}

impl Config {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let config: Self = serde_yaml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("in {}: {e}", path.display()))?;
        if config.apps.as_ref().is_some_and(Vec::is_empty) {
            anyhow::bail!(
                "in {}: `apps` names no daemon — leave it out to run both",
                path.display()
            );
        }
        Ok(config)
    }

    fn runs(&self, app: App) -> bool {
        self.apps.as_ref().is_none_or(|apps| apps.contains(&app))
    }
}

fn main() -> anyhow::Result<()> {
    first(MOTHERD);
    let cli = Cli::parse();
    let mut config = Config::load(&cli.config)?;
    config.host.upgrade_failed = cli.upgrade_failed;
    config.host.after_pid = cli.after_pid;
    // The process's state stays beside zend's substrate whichever daemons
    // run, so its log, offline marker and upgrade record do not move when
    // `apps` changes.
    let state_dir = config.zend.substrate_dir();
    // zend first: it builds the shared engine, and stops last. A daemon `apps`
    // leaves out keeps its seat: its asks of the engine still count, so the
    // engine is the one the two build together and every substrate they wrote
    // opens on it.
    let (runs_zend, runs_npcd) = (config.runs(App::Zend), config.runs(App::Npcd));
    let zend: Seat = if runs_zend {
        Seat::Runs(Box::new(config.zend.resolve()?))
    } else {
        Seat::Stands {
            name: "zend",
            asks: config.zend.engine_requests()?,
        }
    };
    let npcd: Seat = if runs_npcd {
        Seat::Runs(Box::new(config.npcd.resolve()?))
    } else {
        Seat::Stands {
            name: "npcd",
            asks: config.npcd.engine_requests(),
        }
    };
    // Nothing to drop on a relaunch: the one-shot flags cannot be set in
    // `motherd.yaml` (`ZendArgs::ONE_SHOT`, `NpcdArgs::ONE_SHOT`).
    run_seated(
        HostConfig {
            binary: "motherd",
            version: env!("CARGO_PKG_VERSION"),
            args: config.host,
            one_shot: &[],
            state_dir: Some(state_dir),
        },
        vec![zend, npcd],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use host::guard::Routes;
    use host::run::merge_routes;
    use web::auth::Roles;

    /// **The host's, zend's and npcd's real route tables merge** — the check
    /// `host::run` makes at start, run here on the tables themselves, so a
    /// path one daemon adds that the other already serves fails this test
    /// rather than motherd's start on `.5`.
    #[test]
    fn the_host_and_both_tenants_serve_disjoint_routes() {
        let roles = Roles::default();
        let host = host::endpoints::declared(roles.clone());
        let zend = zend::api::declared(roles.clone());
        let npcd = npcd::tenant::declared(&roles).unwrap();
        let merged = merge_routes(
            Routes::table(&host),
            vec![
                ("zend", Routes::table(&zend)),
                ("npcd", Routes::table(&npcd)),
            ],
        )
        .unwrap();
        assert_eq!(merged.declared.len(), host.len() + zend.len() + npcd.len());
    }

    /// The shipped example parses, and every section holds the same settings
    /// the daemons' own flags do: the process joins its gateway's cluster with
    /// an agent in process, and npcd's mind is the repository the agents hold.
    #[test]
    fn the_example_configuration_parses() {
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("motherd.yaml");
        let c = Config::load(&example).unwrap();
        assert_eq!(c.host.port, 8081);
        assert_eq!(c.host.tls_port, Some(8444));
        assert_eq!(
            c.host.clusters,
            ["https://gateway.tokera.local:9443 https://gateway.tokera.com:9443"]
        );
        assert_eq!(c.host.cluster_name.as_deref(), Some("tokera.com"));
        assert!(c.host.group_key.is_some() && c.host.agent.is_some());
        assert_eq!(c.npcd.mind_repo.as_deref(), Some("battle-mind"));
        assert!(c.npcd.data.is_some());
    }

    /// The laptop's file is blackworld's with its own address and agent: the
    /// same cluster and tenants, so a member started there joins the same way.
    #[test]
    fn the_laptop_configuration_parses() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let laptop = Config::load(&dir.join("motherd.laptop.yaml")).unwrap();
        let shipped = Config::load(&dir.join("motherd.yaml")).unwrap();
        assert_eq!(laptop.host.host, "192.168.0.237");
        assert_eq!(
            laptop.host.agent.as_deref(),
            Some(Path::new("motherd/agent.laptop.yaml"))
        );
        assert_eq!(laptop.host.clusters, shipped.host.clusters);
        assert_eq!(laptop.host.cluster_name, shipped.host.cluster_name);
        assert_eq!(laptop.zend, shipped.zend);
        assert_eq!(laptop.npcd, shipped.npcd);
    }

    /// A key no section knows is refused, naming the file — a misspelt
    /// setting must not run with the default.
    #[test]
    fn an_unknown_key_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("motherd.yaml");
        fs::write(&path, "zend:\n  model: Qwen35_0_8B_Q8\n").unwrap();
        let err = Config::load(&path).unwrap_err().to_string();
        assert!(err.contains("motherd.yaml"), "{err}");
        fs::write(&path, "npcd:\n  content: web/content/npcd\n").unwrap();
        assert!(Config::load(&path).is_err());
        // A one-shot wipe in a file read at every start would wipe at every
        // start, so the file cannot ask for one.
        fs::write(&path, "zend:\n  wipe_substrate: true\n").unwrap();
        assert!(Config::load(&path).is_err());
        fs::write(&path, "npcd:\n  forget_conversations: true\n").unwrap();
        assert!(Config::load(&path).is_err());
        fs::write(&path, "{}").unwrap();
        assert_eq!(Config::load(&path).unwrap(), Config::default());
    }

    /// `apps` picks the daemons: both when absent, one when named, and an
    /// empty or unknown list is refused rather than running nothing.
    #[test]
    fn apps_names_the_daemons_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("motherd.yaml");
        fs::write(&path, "{}").unwrap();
        let both = Config::load(&path).unwrap();
        assert!(both.runs(App::Zend) && both.runs(App::Npcd));
        fs::write(&path, "apps: [npcd]\n").unwrap();
        let npcd = Config::load(&path).unwrap();
        assert!(npcd.runs(App::Npcd) && !npcd.runs(App::Zend));
        fs::write(&path, "apps: []\n").unwrap();
        let err = Config::load(&path).unwrap_err().to_string();
        assert!(err.contains("`apps`"), "{err}");
        fs::write(&path, "apps: [zen]\n").unwrap();
        assert!(Config::load(&path).is_err());
    }

    /// The state directory is zend's substrate whether or not zend runs.
    #[test]
    fn the_state_stays_beside_zends_substrate() {
        let dir = tempfile::tempdir().unwrap();
        let zend = ZendArgs {
            working_dir: Some(dir.path().to_path_buf()),
            ..ZendArgs::default()
        };
        assert_eq!(
            zend.substrate_dir(),
            dir.path().canonicalize().unwrap().join("substrate")
        );
    }
}
