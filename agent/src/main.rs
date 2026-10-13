//! `tokera-agent --config agent.yaml --cluster https://… --group-key key`
//! (`docs/replicated_substrate.md` §11): join the agent group of every
//! cluster named and run the jobs members send, until Ctrl-C — or until it
//! installs a new build of itself (§3.9).

use std::path::PathBuf;
use std::sync::Arc;

use agent::checkouts::Checkouts;
use agent::config::AgentConfig;
use agent::install::{jobs_ended, AgentInstaller};
use agent::jobs::Jobs;
use agent::member::{join, node_in, Joining};
use clap::Parser;
use cluster::handshake::GroupKey;
use cluster::launch::LaunchSpec;
use cluster::machine::measure;
use cluster::upgrade::artifacts::RunProbe;
use cluster::upgrade::handoff::hand_off;
use cluster::upgrade::kept::KEEP_DEFAULT;
use cluster::upgrade::os::wait_until_gone;
use cluster::upgrade::runtime::{Setup, Upgrader};
use cluster::version::info::{answer_if_asked, BuildInfo, Opens, AGENT};
use cluster_dial::Target;
use tracing_subscriber::EnvFilter;
use zend_vfs::sandbox::programs::policy;
use zend_vfs::Priority;

/// Where an agent keeps its downloads, plan and markers: in its jobs folder.
const UPGRADE_DIR: &str = "upgrade";

#[derive(Parser, Debug)]
#[command(
    name = "tokera-agent",
    about = "Run build and git jobs for a tokera cluster"
)]
struct Args {
    /// The agent's file: its checkouts, slots and jobs folder.
    #[arg(long)]
    config: PathBuf,
    /// A gateway to join, `https://<host>[:port]`; repeat for each cluster.
    #[arg(long = "cluster", required = true)]
    clusters: Vec<String>,
    /// The name each cluster's certificate is checked against, when a
    /// `--cluster` names an address.
    #[arg(long)]
    cluster_name: Option<String>,
    /// The group key file every member and gateway shares.
    #[arg(long)]
    group_key: PathBuf,
    /// This machine's name on the status page.
    #[arg(long)]
    host: Option<String>,
    /// How many previous builds to keep beside the executable, for rollback.
    #[arg(long, default_value_t = KEEP_DEFAULT)]
    keep_builds: usize,
    /// The build an install restored this agent from, `count-hash`: never
    /// tried again. `keep` sets it when it restores this process.
    #[arg(long, value_name = "COUNT-HASH")]
    upgrade_failed: Option<String>,
    /// Wait until this pid has exited before taking jobs. The predecessor sets
    /// it when it starts the successor.
    #[arg(long, value_name = "PID")]
    after_pid: Option<u32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // An agent opens no substrate and runs no model.
    answer_if_asked(AGENT, Opens::default);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let launch = LaunchSpec::capture(&[])?;
    let args = Args::parse();
    if let Some(pid) = args.after_pid {
        wait_until_gone(pid);
    }
    let cfg = AgentConfig::load(&args.config)?;
    // The agent upgrades with its app's build (§3.9).
    let setup = Setup {
        info: BuildInfo::of(AGENT, Opens::default()),
        exe: launch.exe.clone(),
        upgrade_dir: cfg.jobs_dir.join(UPGRADE_DIR),
        keep_builds: args.keep_builds,
        failed_from: args.upgrade_failed.clone(),
    };
    let installer = AgentInstaller::new();
    let upgrader = Upgrader::new(setup.clone(), installer.clone(), Arc::new(RunProbe));
    upgrader.start();
    let targets = args
        .clusters
        .iter()
        .map(|u| Target::parse(u, args.cluster_name.as_deref()))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let checkouts = Checkouts::open(&cfg, &policy())?;
    let slots = cfg.slots(measure().cores);
    let jobs = Jobs::new(checkouts, slots, Priority::Normal);
    let node = node_in(&cfg.jobs_dir)?;
    let host = args
        .host
        .or_else(machine_name)
        .unwrap_or_else(|| format!("agent-{node:016x}"));
    tracing::info!(node = %format!("{node:016x}"), slots, "joining as an agent");
    let member = join(
        Joining {
            targets,
            key: GroupKey::load(&args.group_key)?,
            host,
            node,
        },
        Arc::clone(&jobs),
        &upgrader,
    )?;
    tokio::select! {
        stopped = tokio::signal::ctrl_c() => {
            stopped?;
            member.shutdown("the agent is stopping");
            Ok(())
        }
        job = installer.asked() => {
            member.shutdown("the agent is installing a new build");
            let left = jobs_ended(&jobs).await;
            if left > 0 {
                tracing::warn!(left, "jobs still running: they end with this process");
            }
            hand_off(&launch, &setup, &job, None)?;
            tracing::warn!("install handed to the watchdog — exiting");
            std::process::exit(0)
        }
    }
}

/// The machine's own name, where the platform gives one.
fn machine_name() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|h| !h.is_empty())
}
