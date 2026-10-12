//! `tokerad` — zend and npcd in one process, on one model.
//!
//! The same startup path as `zend` and `npcd` — `host::run` — with both tenants,
//! zend ranked first (`docs/shared_model_host.md` §3.6): while zend has work,
//! npcd waits. One model loads, one engine runs, each daemon keeps its own
//! substrate, and one HTTP server serves both — the API on every name, each UI
//! on its own (`code.tokera.com` and a bare IP: zend's; `bot.tokera.com`:
//! npcd's).
//!
//! It reads `tokerad.yaml` — the process's settings under `host:`, and each
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
//!   mind: ../mind
//! ```
//!
//! Paths resolve against the working directory — the repository root, which
//! every daemon runs from — so from `D:\prog\candle` that is zend's workspace
//! `D:\prog` and npcd's mind `D:\prog\mind`. `--config` defaults to
//! `tokerad.yaml` in the working directory; the shipped example is
//! `tokerad/tokerad.yaml`.

use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use cluster::version::info::TOKERAD;
use host::build_info::first;
use host::{run, HostArgs, HostConfig, Tenant};
use npcd::tenant::NpcdArgs;
use serde::Deserialize;
use zend::cli::ZendArgs;

#[derive(Parser, Debug)]
#[command(name = "tokerad", about = "zend and npcd in one process, on one model")]
struct Cli {
    /// The configuration file.
    #[arg(long, default_value = "tokerad.yaml")]
    config: PathBuf,
    /// The build an install restored this process from (`host::HostArgs`):
    /// set by the upgrade's watchdog, never by the file.
    #[arg(long, value_name = "COUNT-HASH")]
    upgrade_failed: Option<String>,
}

/// `tokerad.yaml`.
#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct Config {
    host: HostArgs,
    zend: ZendArgs,
    npcd: NpcdArgs,
}

impl Config {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        serde_yaml::from_str(&text).map_err(|e| anyhow::anyhow!("in {}: {e}", path.display()))
    }
}

fn main() -> anyhow::Result<()> {
    first(TOKERAD);
    let cli = Cli::parse();
    let mut config = Config::load(&cli.config)?;
    config.host.upgrade_failed = cli.upgrade_failed;
    // zend first: it outranks npcd.
    let tenants: Vec<Box<dyn Tenant>> = vec![
        Box::new(config.zend.resolve()?),
        Box::new(config.npcd.resolve()?),
    ];
    // Nothing to drop on a relaunch: the one-shot flags cannot be set in
    // `tokerad.yaml` (`ZendArgs::ONE_SHOT`, `NpcdArgs::ONE_SHOT`).
    run(
        HostConfig {
            binary: "tokerad",
            version: env!("CARGO_PKG_VERSION"),
            args: config.host,
            one_shot: &[],
        },
        tenants,
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
    /// rather than tokerad's start on `.5`.
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
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("tokerad.yaml");
        let c = Config::load(&example).unwrap();
        assert_eq!(c.host.port, 8081);
        assert_eq!(c.host.tls_port, Some(8444));
        assert_eq!(c.host.clusters, ["https://192.168.0.5:9443"]);
        assert_eq!(c.host.cluster_name.as_deref(), Some("tokera.com"));
        assert!(c.host.group_key.is_some() && c.host.agent.is_some());
        assert_eq!(
            (c.npcd.mind_repo.as_deref(), c.npcd.mind.as_deref()),
            (Some("battle-mind"), None)
        );
        assert!(c.npcd.data.is_some());
    }

    /// A key no section knows is refused, naming the file — a misspelt
    /// setting must not run with the default.
    #[test]
    fn an_unknown_key_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokerad.yaml");
        fs::write(&path, "zend:\n  model: Qwen35_0_8B_Q8\n").unwrap();
        let err = Config::load(&path).unwrap_err().to_string();
        assert!(err.contains("tokerad.yaml"), "{err}");
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
}
