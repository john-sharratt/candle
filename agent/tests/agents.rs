//! Agents (`docs/replicated_substrate.md` §11; §14.1's agent-selection table
//! is the gateway's, in `cluster::gateway::agents`). A local bare repository
//! stands in for `origin`; the member's and the agent's checkouts are clones
//! of it.
//!
//! - A run on an agent lays a store's changes down on the agent's checkout,
//!   and what the command changed comes back as deltas the member's own store
//!   takes.
//! - A push of commits the member built is made by the agent, under its
//!   lease; an agent without push rights refuses it.
//! - Through an in-process gateway, a member's job reaches an agent over
//!   `group:agent`, a repository opened pushing through the agents pushes from
//!   there, and one opened fetching through them — a mirror with no way to
//!   origin — follows origin's branch. Cut off, a member reaches only an agent
//!   in its own process, and without one the job is `no_agent`.

use std::net::IpAddr;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent::checkouts::Checkouts;
use agent::client::{AgentError, Agents, At};
use agent::config::{AgentConfig, CheckoutConfig};
use agent::fetcher::AgentFetch;
use agent::jobs::{JobError, Jobs};
use agent::member::{Agent, APP};
use agent::pusher::AgentPush;
use agent::wire::{encode_bundle, FetchJob, JobView, PushJob, PushLine, RunJob, Submit};
use cluster::dial::Connect;
use cluster::dispatch::RouteTable;
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::{Choice, Hub, Need};
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Identity};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use tempfile::TempDir;
use tokio::runtime::Handle;
use tokio::task::spawn_blocking;
use tokio::time::sleep;
use web::cluster::{streams, Gateway};
use zend_vfs::{
    BranchName, CommandPolicy, FileDelta, GitSource, JobStatus, Lease, Oid, Priority, PushOutcome,
    PushSpec, RemoteName, Repo, Rev, VfsStore,
};

const WITHIN: Duration = Duration::from_secs(20);

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// The shell a test job runs, and its arguments to copy `from` to `to`.
fn copy(from: &str, to: &str) -> (String, Vec<String>) {
    if cfg!(windows) {
        ("cmd".into(), vec!["/C".into(), format!("copy {from} {to}")])
    } else {
        ("sh".into(), vec!["-c".into(), format!("cp {from} {to}")])
    }
}

/// `origin` with one commit on `main`, and a member's and an agent's clone.
struct Repos {
    _root: TempDir,
    origin: std::path::PathBuf,
    member: std::path::PathBuf,
    agent: std::path::PathBuf,
    jobs: std::path::PathBuf,
}

fn repos() -> Repos {
    let root = TempDir::new().unwrap();
    let origin = root.path().join("origin.git");
    std::fs::create_dir(&origin).unwrap();
    git(&origin, &["init", "-q", "--bare", "-b", "main"]);
    let seed = root.path().join("seed");
    std::fs::create_dir(&seed).unwrap();
    git(&seed, &["init", "-q", "-b", "main"]);
    std::fs::write(seed.join("in.txt"), b"hello\n").unwrap();
    git(&seed, &["add", "in.txt"]);
    git(&seed, &["commit", "-q", "-m", "first"]);
    git(&seed, &["push", "-q", origin.to_str().unwrap(), "main"]);
    let clone = |name: &str| {
        let dir = root.path().join(name);
        git(
            root.path(),
            &[
                "clone",
                "-q",
                origin.to_str().unwrap(),
                dir.to_str().unwrap(),
            ],
        );
        dir
    };
    let member = clone("member");
    let agent = clone("agent");
    Repos {
        jobs: root.path().join("jobs"),
        _root: root,
        origin,
        member,
        agent,
    }
}

fn jobs_over(r: &Repos, push: bool) -> Arc<Jobs> {
    let cfg = AgentConfig {
        jobs_dir: r.jobs.clone(),
        slots: Some(2),
        checkouts: vec![CheckoutConfig {
            name: "r".into(),
            dir: r.agent.clone(),
            push,
        }],
    };
    let shell = copy("a", "b").0;
    let checkouts = Checkouts::open(&cfg, &CommandPolicy::allowing([shell.as_str()])).unwrap();
    Jobs::new(checkouts, 2, Priority::Normal)
}

fn main() -> BranchName {
    BranchName::parse("main").unwrap()
}

/// The member's store over `main`, with `in.txt` changed.
fn changed_store(r: &Repos) -> VfsStore {
    let store = VfsStore::on_branch(GitSource::open(&r.member).unwrap(), Rev::Branch(main()));
    store.write("in.txt", "changed\n".into()).unwrap();
    store
}

fn run_job(store: &VfsStore) -> RunJob {
    let (program, args) = copy("in.txt", "out.txt");
    RunJob {
        repo: "r".into(),
        branch: "main".into(),
        overlay: store.snapshot(),
        program,
        args,
        timeout_secs: 60,
    }
}

async fn ended(jobs: &Arc<Jobs>, mut view: JobView) -> JobView {
    let deadline = Instant::now() + WITHIN;
    while !view.status.is_finished() {
        assert!(Instant::now() < deadline, "the job never ended: {view:?}");
        view = jobs.wait(&view.id, Duration::from_secs(5)).await.unwrap();
    }
    view
}

/// **A run lays the member's changes on the agent's checkout, and what the
/// command changed comes back as deltas** the member's own store takes. The
/// agent's checkout is put back as it was.
#[tokio::test]
async fn a_run_on_an_agent_comes_back_as_deltas() {
    let r = repos();
    let jobs = jobs_over(&r, false);
    let store = changed_store(&r);
    let started = jobs.submit(Submit::Run(run_job(&store))).await.unwrap();
    let view = ended(&jobs, started).await;
    assert_eq!(view.status, JobStatus::Exited { code: Some(0) }, "{view:?}");
    let outcome = view.outcome.expect("a run that ended has its outcome");
    let changed: Vec<_> = outcome.changed.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(changed, ["out.txt"]);
    for c in &outcome.changed {
        store
            .apply(&c.path, std::slice::from_ref(&c.delta))
            .unwrap();
    }
    assert_eq!(store.read("out.txt").unwrap().as_deref(), Some("changed\n"));
    assert_eq!(
        std::fs::read_to_string(r.agent.join("in.txt"))
            .unwrap()
            .replace("\r\n", "\n"),
        "hello\n",
        "the agent's checkout is its own again"
    );
    assert!(!r.agent.join("out.txt").exists());
    assert!(jobs.log(&view.id).unwrap().is_file());
    assert_eq!(jobs.slots(), (2, 0));
}

/// A commit the member built is pushed by the agent through a bundle, under
/// its lease; an agent without push rights refuses the push.
#[tokio::test]
async fn a_push_is_made_by_the_agent_with_its_credentials() {
    let r = repos();
    std::fs::write(r.member.join("in.txt"), b"two\n").unwrap();
    git(&r.member, &["commit", "-q", "-am", "second"]);
    let first = Oid::parse(git(&r.member, &["rev-parse", "origin/main"]).trim()).unwrap();
    let second = Oid::parse(git(&r.member, &["rev-parse", "HEAD"]).trim()).unwrap();
    let specs = [PushSpec::branch(
        second.clone(),
        main(),
        Lease::Expect(first),
    )];
    let origin = RemoteName::parse("origin").unwrap();
    let bundle = Repo::open(&r.member)
        .unwrap()
        .bundle_for_push(&origin, &specs)
        .unwrap();
    let job = PushJob {
        repo: "r".into(),
        bundle: bundle.as_deref().map(encode_bundle),
        specs: specs.iter().map(PushLine::of).collect(),
    };

    let refused = jobs_over(&r, false).submit(Submit::Push(job.clone())).await;
    assert_eq!(refused.unwrap_err(), JobError::NoPush("r".into()));

    let pushed = jobs_over(&r, true).submit(Submit::Push(job)).await.unwrap();
    assert_eq!(pushed.status, JobStatus::Exited { code: Some(0) });
    assert_eq!(pushed.pushed[0].outcome, PushOutcome::FastForward);
    assert_eq!(
        git(&r.origin, &["rev-parse", "main"]).trim(),
        second.to_string()
    );
}

fn key() -> GroupKey {
    GroupKey::from_bytes([3; 32])
}

/// A member's way to the in-process gateway, for `app`.
struct ToGateway {
    gateway: Arc<Gateway>,
    app: &'static str,
}

impl Connect for ToGateway {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        let (m, g) = pair(8 << 20);
        let gateway = self.gateway.clone();
        let app = self.app;
        let ip = IpAddr::from([10, 0, 0, 9]);
        tokio::spawn(async move {
            let hub = gateway.hub.clone();
            hub.serve(app, ip, g.tx, g.rx, move |s| {
                streams::serve(&gateway, ip, s)
            })
            .await;
        });
        Ok((m.tx, m.rx))
    }
}

/// A zend member with no tenant behind it: only its jobs go anywhere.
fn zend_member(node: u64) -> Arc<AppMember> {
    AppMember::new(
        Identity {
            app: "zend".into(),
            node,
            host: "m".into(),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            model: Some(ModelId {
                arch: "Qwen4Exp".into(),
                weights: [1; 32],
                tokenizer: [1; 32],
                rope: "yarn".into(),
                describe: "model".into(),
            }),
            repos: Vec::new(),
        },
        key(),
        vec!["https://gw".into()],
        RouteTable::new(Vec::new()).unwrap(),
        Box::new(|| Capacity {
            state: MemberState::Ready,
            ..Capacity::default()
        }),
        None,
        Instant::now(),
    )
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + WITHIN;
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(20)).await;
    }
}

/// **A zend tool runs on an agent**: a member's job goes up `group:agent`,
/// the gateway picks the agent holding the repository and claims its
/// lineage there, and the outcome and log come back. A repository opened
/// pushing through the agents pushes from the agent's checkout.
#[tokio::test]
async fn a_members_jobs_reach_an_agent_through_the_gateway() {
    let r = repos();
    let gateway = Gateway::with_hub(Hub::new(key(), "gw"));
    let jobs = jobs_over(&r, true);
    let agent = Agent::new(0xa1, "agent".into(), key(), vec!["https://gw".into()], jobs);
    agent.dial(
        0,
        ToGateway {
            gateway: gateway.clone(),
            app: APP,
        },
    );
    let member = zend_member(0x1);
    let run = member.clone();
    let to = ToGateway {
        gateway: gateway.clone(),
        app: "zend",
    };
    tokio::spawn(async move { run.run(0, to, |conn| conn.reset()).await });
    let hub = gateway.hub.clone();
    until("the agent ready and the member admitted", || {
        member.admitted() && hub.member(APP, 0xa1).is_some_and(|m| m.ready())
    })
    .await;

    let agents = Agents::new(member.clone(), None);
    let store = changed_store(&r);
    let ran = agents.run("conv-1", run_job(&store)).await.unwrap();
    assert_eq!(ran.at, At::Agent(0xa1));
    assert_eq!(ran.view.status, JobStatus::Exited { code: Some(0) });
    let changed = &ran.outcome.as_ref().unwrap().changed;
    assert!(
        matches!(&changed[0].delta.delta, FileDelta::Replace { content } if content == "changed\n"),
        "{changed:?}"
    );
    let need = Need {
        repo: "r".into(),
        toolchain: None,
        push: false,
    };
    until("the lineage claimed by the agent", || {
        hub.pick_agent(APP, Some("conv-1"), &need, false) == Choice::Run(0xa1)
    })
    .await;

    // A push from the member's checkout is made by the agent.
    std::fs::write(r.member.join("in.txt"), b"three\n").unwrap();
    git(&r.member, &["commit", "-q", "-am", "third"]);
    let tip = Oid::parse(git(&r.member, &["rev-parse", "HEAD"]).trim()).unwrap();
    let before = Oid::parse(git(&r.member, &["rev-parse", "origin/main"]).trim()).unwrap();
    let pusher = AgentPush::shared(agents.clone(), Handle::current(), "r");
    let dir = r.member.clone();
    let spec = PushSpec::branch(tip.clone(), main(), Lease::Expect(before));
    let pushed = spawn_blocking(move || {
        Repo::open(&dir)
            .unwrap()
            .pushing_through(pusher)
            .push(&RemoteName::parse("origin").unwrap(), &[spec])
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(pushed[0].outcome, PushOutcome::FastForward);
    assert_eq!(
        git(&r.origin, &["rev-parse", "main"]).trim(),
        tip.to_string()
    );

    // A mirror that cannot reach origin — its `origin` names the agent, not a
    // URL — follows origin through the agent (§10.3).
    let mirror = r.jobs.parent().unwrap().join("mirror");
    std::fs::create_dir(&mirror).unwrap();
    git(&mirror, &["init", "-q", "-b", "main"]);
    git(&mirror, &["remote", "add", "origin", "agent:r"]);
    let fetcher = AgentFetch::shared(agents.clone(), Handle::current(), "r");
    let pulled = spawn_blocking(move || {
        Repo::open(&mirror)
            .unwrap()
            .fetching_through(fetcher)
            .pull_branch(&main())
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(pulled.on_origin, Some(tip.clone()));
    assert_eq!(pulled.tip, Some(tip));
}

/// Cut off, a member's job goes to the agent in its own process, and with
/// none it is `no_agent` — never run on the member's own checkout.
#[tokio::test]
async fn a_cut_off_member_reaches_only_its_own_agent() {
    let r = repos();
    let member = zend_member(0x2);
    let store = changed_store(&r);
    let none = Agents::new(member.clone(), None)
        .run("conv-2", run_job(&store))
        .await;
    assert!(matches!(none, Err(AgentError::NoAgent(_))), "{none:?}");
    let own = Agents::new(member.clone(), Some(jobs_over(&r, false)))
        .run("conv-2", run_job(&store))
        .await
        .unwrap();
    assert_eq!(own.at, At::InProcess);
    assert_eq!(own.view.status, JobStatus::Exited { code: Some(0) });

    // A fetch goes the same way: to its own agent, else `no_agent`.
    let job = FetchJob {
        repo: "r".into(),
        branch: Some("main".into()),
        have: Vec::new(),
    };
    let none = Agents::new(member.clone(), None).fetch(job.clone()).await;
    assert!(matches!(none, Err(AgentError::NoAgent(_))), "{none:?}");
    let fetched = Agents::new(member, Some(jobs_over(&r, false)))
        .fetch(job)
        .await
        .unwrap();
    let origin_main = Oid::parse(git(&r.origin, &["rev-parse", "main"]).trim()).unwrap();
    assert_eq!(fetched.tips.get(&main()), Some(&origin_main));
    assert!(fetched.bundle.is_some(), "the member held nothing");
}
