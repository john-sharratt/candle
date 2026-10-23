//! An agent joined to its clusters (§11.1): one socket per `--cluster` to
//! `/v1/cluster/agent`, with the same handshake and group key as any member,
//! no model and no substrate. Its `HELLO` advertises its checkouts; its
//! `CAPACITY`, every second, its job slots, the host's cores, load and free
//! memory, and the allow-listed toolchains on its `PATH` (§11.2) — and its
//! state, `disabled` while an admin has switched it off (§11.7). The streams
//! its gateways open are served the job API.
//!
//! An agent has no substrate to keep its node in, so the node is kept in a
//! `NODE` file beside its job logs: the same agent is the same member across
//! restarts, and its lineage claims stay its own.

use std::convert::Infallible;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::Router;
use cluster::dial::Connect;
use cluster::frame::{Capacity, MemberState};
use cluster::handshake::GroupKey;
use cluster::machine::measure;
use cluster::member::{AppMember, Identity};
use cluster::session::VirtualConn;
use cluster::upgrade::runtime::Upgrader;
use cluster::version::this_build;
use cluster_dial::{tls_connector, Dial, Target};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use rand::random;
use tokio::time::interval;
use tower::Service;
use zend_vfs::Priority;

use crate::jobs::Jobs;
use crate::routes::{self, OverSocket};

/// The app an agent joins.
pub const APP: &str = "agent";

/// The file an agent's node is kept in.
pub const NODE_FILE: &str = "NODE";

/// How long a lineage stays with the agent that last ran it, unserved.
pub const LINEAGE_IDLE: Duration = Duration::from_secs(10 * 60);

const RELEASE_EVERY: Duration = Duration::from_secs(30);

/// This agent's node: kept in `dir`'s [`NODE_FILE`], drawn at random the first
/// time.
pub fn node_in(dir: &Path) -> anyhow::Result<u64> {
    let path = dir.join(NODE_FILE);
    if let Ok(text) = fs::read_to_string(&path) {
        return u64::from_str_radix(text.trim(), 16)
            .map_err(|e| anyhow::anyhow!("{}: not a node: {e}", path.display()));
    }
    fs::create_dir_all(dir)?;
    let node = loop {
        let n: u64 = random();
        if n != 0 {
            break n;
        }
    };
    fs::write(&path, format!("{node:016x}\n"))?;
    Ok(node)
}

/// What `CAPACITY` reports for `jobs`, with `found` the toolchains on the
/// `PATH`.
pub fn capacity(jobs: &Jobs, found: &[String]) -> Capacity {
    let machine = measure();
    let (slots, running) = jobs.slots();
    Capacity {
        state: if jobs.enabled() {
            MemberState::Ready
        } else {
            MemberState::Disabled
        },
        power: u32::from(machine.cores),
        idle: running == 0,
        slots_free: slots.saturating_sub(running),
        cores: machine.cores,
        load_milli: machine.load_milli,
        os: machine.os,
        mem_free_mib: machine.mem_free_mib,
        jobs: running,
        toolchains: found.to_vec(),
        low_priority: jobs.priority() == Priority::Below,
        ..Capacity::default()
    }
}

/// Serve `conn`, a stream a gateway opened, with the job API.
pub fn serve(router: Router, conn: VirtualConn) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            async move {
                let mut req = req.map(Body::new);
                req.extensions_mut().insert(OverSocket);
                match router.call(req).await {
                    Ok(res) => Ok::<_, Infallible>(res),
                    Err(never) => match never {},
                }
            }
        });
        if let Err(e) = ConnBuilder::new(TokioExecutor::new())
            .serve_connection(TokioIo::new(conn), svc)
            .await
        {
            tracing::debug!(error = %e, "a gateway's job stream ended");
        }
    });
}

/// How an agent joins.
pub struct Joining {
    pub targets: Vec<Target>,
    pub key: GroupKey,
    /// This machine's name in `HELLO`.
    pub host: String,
    pub node: u64,
}

/// An agent's membership before its sockets dial: the member, and the job
/// API its streams are served.
pub struct Agent {
    pub member: Arc<AppMember>,
    router: Router,
}

impl Agent {
    /// The agent serving `jobs` as `node` on `host`, to the clusters at
    /// `urls`, proving `key`. A job's lineage is claimed for it as the job
    /// arrives; a claim unserved for [`LINEAGE_IDLE`] is let go.
    pub fn new(
        node: u64,
        host: String,
        key: GroupKey,
        urls: Vec<String>,
        jobs: Arc<Jobs>,
    ) -> Agent {
        let found = jobs.checkouts().toolchains().to_vec();
        let measured = Arc::clone(&jobs);
        let member = AppMember::new(
            Identity {
                app: APP.to_string(),
                node,
                host,
                build: this_build(),
                model: None,
                repos: jobs.checkouts().ads(),
            },
            key,
            urls,
            routes::table(),
            Box::new(move || capacity(&measured, &found)),
            None,
            Instant::now(),
        );
        let claimer = Arc::downgrade(&member);
        let switcher = Arc::downgrade(&member);
        let router = routes::router(
            jobs,
            Arc::new(move |lineage: &str| {
                if let Some(m) = claimer.upgrade() {
                    m.claim(lineage.as_bytes());
                    m.capacity_changed();
                }
            }),
            Arc::new(move || {
                if let Some(m) = switcher.upgrade() {
                    m.capacity_changed();
                }
            }),
        );
        let releaser = Arc::downgrade(&member);
        tokio::spawn(async move {
            let mut tick = interval(RELEASE_EVERY);
            loop {
                tick.tick().await;
                let Some(m) = releaser.upgrade() else { return };
                // An agent writes no substrate: its claims let go through 0.
                m.release_idle(LINEAGE_IDLE, 0);
            }
        });
        Agent { member, router }
    }

    /// Dial cluster `c` through `connector`, on the current runtime, serving
    /// what its gateway sends until the socket stops for good.
    pub fn dial<C: Connect + 'static>(&self, c: usize, connector: C) {
        let member = Arc::clone(&self.member);
        let router = self.router.clone();
        tokio::spawn(async move {
            let stopped = member
                .clone()
                .run(c, connector, move |conn| serve(router.clone(), conn))
                .await;
            tracing::warn!(cluster = c, "agent socket stopped: {stopped:?}");
        });
    }
}

/// Join every cluster in `joining` as an agent serving `jobs`, on the
/// current runtime, upgrading through `upgrader` (§3.9). The member is
/// returned at once; its sockets dial on.
pub fn join(
    joining: Joining,
    jobs: Arc<Jobs>,
    upgrader: &Arc<Upgrader>,
) -> anyhow::Result<Arc<AppMember>> {
    let urls = joining.targets.iter().map(|t| t.url.clone()).collect();
    let agent = Agent::new(joining.node, joining.host, joining.key, urls, jobs);
    // Before any socket dials: a member admitted below the floor is offered
    // the floor's build at once.
    upgrader.register(&agent.member);
    let tls = tls_connector()?;
    for (c, target) in joining.targets.into_iter().enumerate() {
        agent.dial(c, Dial::new(Arc::new(target), APP, tls.clone()));
    }
    Ok(agent.member)
}

#[cfg(test)]
mod tests {
    use zend_vfs::CommandPolicy;

    use super::*;
    use crate::checkouts::Checkouts;
    use crate::config::AgentConfig;

    /// **An agent whose jobs run below normal priority says so**, and a
    /// dedicated one does not.
    #[test]
    fn capacity_says_whether_the_agent_is_low_priority() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = AgentConfig {
            jobs_dir: dir.path().join("jobs"),
            slots: Some(2),
            checkouts: Vec::new(),
        };
        let open = || Checkouts::open(&cfg, &CommandPolicy::allowing(["cargo"])).unwrap();
        let beside = Jobs::new(open(), 2, Priority::Below);
        let dedicated = Jobs::new(open(), 2, Priority::Normal);
        assert!(capacity(&beside, &[]).low_priority);
        assert!(!capacity(&dedicated, &[]).low_priority);
        assert_eq!(capacity(&dedicated, &[]).slots_free, 2);
    }

    /// **An agent switched off reports itself `disabled`**, so no gateway
    /// routes it a job, and `ready` again once switched back on.
    #[test]
    fn capacity_reports_an_agent_switched_off() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = AgentConfig {
            jobs_dir: dir.path().join("jobs"),
            slots: Some(1),
            checkouts: Vec::new(),
        };
        let jobs = Jobs::new(
            Checkouts::open(&cfg, &CommandPolicy::allowing(["cargo"])).unwrap(),
            1,
            Priority::Normal,
        );
        assert_eq!(capacity(&jobs, &[]).state, MemberState::Ready);
        jobs.set_enabled(false).unwrap();
        assert_eq!(capacity(&jobs, &[]).state, MemberState::Disabled);
        jobs.set_enabled(true).unwrap();
        assert_eq!(capacity(&jobs, &[]).state, MemberState::Ready);
    }

    /// A node is drawn once and read back the same after.
    #[test]
    fn the_node_is_kept_beside_the_logs() {
        let dir = tempfile::tempdir().unwrap();
        let first = node_in(dir.path()).unwrap();
        assert_ne!(first, 0);
        assert_eq!(node_in(dir.path()).unwrap(), first);
        assert_eq!(
            fs::read_to_string(dir.path().join(NODE_FILE)).unwrap(),
            format!("{first:016x}\n")
        );
    }
}
