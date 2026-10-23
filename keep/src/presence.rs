//! `keep` joined to the gateway as the app `keep`, so the cluster page lists
//! it beside zend and npcd. The socket declares no routes and registers no
//! upgrader: the gateway stores none of its bytes, and nothing here is
//! certified or rolled out. The failsafe loop does not run on this thread.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use cluster::dispatch::RouteTable;
use cluster::frame::{Capacity, MemberState};
use cluster::gateway::KEEP;
use cluster::handshake::GroupKey;
use cluster::machine::measure;
use cluster::member::{AppMember, Identity};
use cluster::version::this_build;
use cluster_dial::{tls_connector, Dial, Target};
use serde::Deserialize;

/// How `keep.yaml` names the gateway this process lists itself on.
#[derive(Debug, Clone, Deserialize)]
pub struct ClusterJoin {
    pub url: String,
    pub name: String,
    pub group_key: PathBuf,
    pub host: String,
    /// The marker a host on this machine keeps its offline switch in
    /// (`cluster::offline`) — motherd's, beside its log. Named, keep follows it:
    /// offline, it dials no gateway elsewhere and drops the one it has.
    #[serde(default)]
    pub offline_marker: Option<PathBuf>,
}

/// Dial `cluster` on its own thread. A failure is printed; the failsafe keeps
/// running either way.
pub fn start(cluster: ClusterJoin, state: &Path) {
    let state = state.to_path_buf();
    let _ = std::thread::Builder::new()
        .name("keep-presence".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("keep: not listed on the cluster: {e}");
                    return;
                }
            };
            if let Err(e) = rt.block_on(join(cluster, &state)) {
                eprintln!("keep: not listed on the cluster: {e}");
            }
        });
}

async fn join(cluster: ClusterJoin, state: &Path) -> anyhow::Result<()> {
    let node = node_in(state)?;
    let key = GroupKey::load(&cluster.group_key).map_err(|e| anyhow::anyhow!("group key: {e}"))?;
    let target = Target::parse(&cluster.url, Some(&cluster.name))?;
    let member = AppMember::new(
        Identity {
            app: KEEP.to_string(),
            node,
            host: cluster.host,
            build: this_build(),
            model: None,
            repos: Vec::new(),
        },
        key,
        vec![target.url.clone()],
        RouteTable::new(Vec::new()).expect("no routes"),
        Box::new(capacity),
        None,
        Instant::now(),
    );
    if let Some(marker) = cluster.offline_marker.clone() {
        follow_offline(marker, member.clone());
    }
    let tls = tls_connector()?;
    let stopped = member
        .run(0, Dial::new(Arc::new(target), KEEP, tls), |conn| {
            conn.reset()
        })
        .await;
    eprintln!("keep: left the cluster: {stopped:?}");
    Ok(())
}

/// How often keep looks at the offline marker.
const OFFLINE_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

/// Follow the offline switch kept in `marker`: read it now, then every few
/// seconds; on the switch going on, drop `member`'s socket — it then dials
/// again and is refused, as the host's members are.
fn follow_offline(marker: PathBuf, member: Arc<AppMember>) {
    let switch = cluster::offline::global();
    let mut was = switch.init(marker);
    if was {
        eprintln!("keep: offline — not listed on the cluster until the host is online");
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(OFFLINE_EVERY);
        loop {
            tick.tick().await;
            let now = switch.refresh();
            if now && !was {
                eprintln!("keep: offline — leaving the cluster until the host is online");
                member.close_all("offline: this machine is told the network is down");
            } else if was && !now {
                eprintln!("keep: online — listed on the cluster again");
            }
            was = now;
        }
    });
}

fn capacity() -> Capacity {
    let machine = measure();
    Capacity {
        state: MemberState::Ready,
        idle: true,
        cores: machine.cores,
        load_milli: machine.load_milli,
        os: machine.os,
        mem_free_mib: machine.mem_free_mib,
        ..Capacity::default()
    }
}

/// The same node across restarts, drawn once into `state`.
fn node_in(dir: &Path) -> anyhow::Result<u64> {
    let path = dir.join("node");
    if let Ok(text) = fs::read_to_string(&path) {
        return u64::from_str_radix(text.trim(), 16)
            .map_err(|e| anyhow::anyhow!("{}: not a node: {e}", path.display()));
    }
    fs::create_dir_all(dir)?;
    let node = loop {
        let n: u64 = rand::random();
        if n != 0 {
            break n;
        }
    };
    fs::write(&path, format!("{node:016x}\n"))?;
    Ok(node)
}
