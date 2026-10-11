//! `tokera-agent --config agent.yaml --cluster https://… --group-key key`
//! (`docs/replicated_substrate.md` §11): join the agent group of every
//! cluster named and run the jobs members send, until Ctrl-C.

use std::path::PathBuf;

use agent::checkouts::Checkouts;
use agent::config::AgentConfig;
use agent::jobs::Jobs;
use agent::member::{join, node_in, Joining};
use clap::Parser;
use cluster::handshake::GroupKey;
use cluster::machine::measure;
use cluster_dial::Target;
use tracing_subscriber::EnvFilter;
use zend_vfs::sandbox::programs::policy;
use zend_vfs::Priority;

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    let cfg = AgentConfig::load(&args.config)?;
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
        jobs,
    )?;
    tokio::signal::ctrl_c().await?;
    member.shutdown("the agent is stopping");
    Ok(())
}

/// The machine's own name, where the platform gives one.
fn machine_name() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|h| !h.is_empty())
}
