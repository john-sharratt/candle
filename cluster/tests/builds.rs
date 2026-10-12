//! Builds across a gateway and members in one process
//! (`docs/replicated_substrate.md` §14.1 build floor, certification,
//! distribution, install, kept builds, forced rollout): a candidate certified
//! through the group, its build fetched by the older members and installed one
//! member at a time; a member below the floor admitted to upgrade; a failed
//! certification; and a downgrade swapped from every member's own kept build.
//!
//! Each member is a machine of its own: a temporary directory holding its
//! executable — a file whose contents are its `--build-info` answer, which is
//! all the upgrades ask of a binary — its kept builds and its downloads. The
//! install itself is recorded, not run.

mod support;

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cluster::dispatch::Mirror;
use cluster::frame::BuildId;
use cluster::gateway::upgrades::plan::Ask;
use cluster::gateway::upgrades::UpgradeConfig;
use cluster::gateway::Hub;
use cluster::replicate::{MemoryStore, Store};
use cluster::upgrade::artifacts::{binary, Probe};
use cluster::upgrade::decisions::{latest, Settled, Verdict};
use cluster::upgrade::kept::{write_json, BuildJson, FileSum};
use cluster::upgrade::runtime::{InstallJob, Installer, Setup, Upgrader};
use cluster::upgrade::sum::{file_sha256, hex};
use cluster::upgrade::swap::Source;
use cluster::version::info::{platform, BuildInfo, Opens, ZEND};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use support::{gateway, member_upgrading, until, Node, APP};

/// A binary whose answer to `--build-info` is its own contents.
struct FileProbe;

impl Probe for FileProbe {
    fn build_info(&self, exe: &Path) -> Option<BuildInfo> {
        BuildInfo::decode(&fs::read_to_string(exe).ok()?)
    }
}

/// Every install asked for, recorded.
#[derive(Default)]
struct Recorder {
    jobs: Mutex<Vec<InstallJob>>,
}

impl Installer for Recorder {
    fn install(&self, job: InstallJob) -> Result<(), String> {
        self.jobs.lock().unwrap().push(job);
        Ok(())
    }
}

impl Recorder {
    fn count(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }
}

fn info(count: u32) -> BuildInfo {
    BuildInfo {
        kind: ZEND.into(),
        count,
        hash: format!("h{count}"),
        local: String::new(),
        platform: platform(),
        opens: Opens::default(),
    }
}

fn b(count: u32) -> BuildId {
    info(count).build()
}

/// A machine running build `count`, keeping each of `kept`. Its directory
/// lives as long as the machine.
struct Machine {
    _dir: TempDir,
    up: Arc<Upgrader>,
    installs: Arc<Recorder>,
}

fn machine(count: u32, kept: &[u32]) -> Machine {
    let dir = TempDir::new().unwrap();
    let bin = dir.path().join("bin");
    fs::create_dir_all(&bin).unwrap();
    let exe = binary(&bin, ZEND);
    fs::write(&exe, info(count).encode()).unwrap();
    for k in kept {
        let at = bin.join("builds").join(format!("{k}-h{k}"));
        fs::create_dir_all(&at).unwrap();
        let name = exe.file_name().unwrap().to_string_lossy().into_owned();
        fs::write(at.join(&name), info(*k).encode()).unwrap();
        write_json(
            &at,
            &BuildJson {
                info: info(*k),
                files: vec![FileSum {
                    sha256: hex(&file_sha256(&at.join(&name)).unwrap()),
                    path: name,
                }],
                ui: vec![],
                ui_archive: String::new(),
                installed_ms: 1,
                retired_ms: 2,
            },
        )
        .unwrap();
    }
    let installs = Arc::new(Recorder::default());
    let up = Upgrader::new(
        Setup {
            info: info(count),
            exe,
            root: dir.path().to_path_buf(),
            ui: vec![],
            upgrade_dir: dir.path().join("data").join("upgrade"),
            keep_builds: 5,
            failed_from: None,
        },
        installs.clone(),
        Arc::new(FileProbe),
    );
    up.start_every(Duration::from_millis(50));
    Machine {
        _dir: dir,
        up,
        installs,
    }
}

fn join(g: &Arc<Hub>, node: u64, count: u32, m: &Machine) -> Node {
    member_upgrading(
        &[g],
        node,
        Arc::new(MemoryStore::new(node)),
        b(count),
        &m.up,
    )
}

/// The gateway's once-a-second judgement, as the web gateway runs it, faster.
fn ticking(g: &Arc<Hub>) -> JoinHandle<()> {
    let g = g.clone();
    tokio::spawn(async move {
        loop {
            g.upgrade_tick();
            sleep(Duration::from_millis(50)).await;
        }
    })
}

fn floor(g: &Hub) -> Option<u32> {
    g.status(0)
        .apps
        .into_iter()
        .find(|a| a.app == APP)
        .and_then(|a| a.floor)
        .map(|f| f.count)
}

fn state_of(g: &Hub, node: u64) -> String {
    let hex = format!("{node:016x}");
    g.status(0)
        .members
        .into_iter()
        .find(|m| m.app == APP && m.node == hex)
        .map_or_else(String::new, |m| m.state.to_string())
}

const WITHIN: Duration = Duration::from_secs(20);

/// **A pushed build certifies, reaches the older members, and installs one
/// member at a time** (§3.8, §3.9). Two members at build 10 set the floor; a
/// third at build 12 is the candidate. Its grace ends, the gateway sends its
/// verdict, and the candidate proposes `certify/zend/12-h12`, which every
/// member reads. The floor rises; each older member has already fetched the
/// candidate's binary while it certified, and installs it under
/// `upgrade/zend` — one, then none until that one is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_certified_build_reaches_the_others_and_installs_one_at_a_time() {
    let g = gateway("g1");
    g.set_upgrade_config(UpgradeConfig {
        grace: Duration::ZERO,
        canary_pct: 10,
    });
    let (m1, m2, m3) = (machine(10, &[]), machine(10, &[]), machine(12, &[]));
    let n1 = join(&g, 1, 10, &m1);
    let n2 = join(&g, 2, 10, &m2);
    until("the older members ready", WITHIN, || {
        n1.ready() && n2.ready()
    })
    .await;
    assert_eq!(floor(&g), Some(10));
    let n3 = join(&g, 3, 12, &m3);
    let tick = ticking(&g);
    until("the floor rises to 12", WITHIN, || floor(&g) == Some(12)).await;
    until("an older member installs", WITHIN, || {
        m1.installs.count() + m2.installs.count() == 1
    })
    .await;
    // The other waits: the lease is held through the installer's restart.
    sleep(Duration::from_millis(600)).await;
    assert_eq!(m1.installs.count() + m2.installs.count(), 1);
    let job = m1
        .installs
        .jobs
        .lock()
        .unwrap()
        .first()
        .cloned()
        .or_else(|| m2.installs.jobs.lock().unwrap().first().cloned())
        .unwrap();
    assert_eq!(job.build, b(12));
    assert_eq!(job.leases, ["upgrade/zend"]);
    let Source::Fetched { exe, .. } = &job.source else {
        panic!("a fetched build: {:?}", job.source);
    };
    assert_eq!(
        fs::read_to_string(exe).unwrap(),
        info(12).encode(),
        "the candidate's binary, verified"
    );
    // The group holds the verdict, by certification.
    let prefix = "certify/zend/12-h12/";
    let (_, verdict) = latest::<Verdict>(&n1.store.decisions(prefix), prefix).unwrap();
    assert_eq!(
        (verdict.result, verdict.by.as_str()),
        (Settled::Pass, "certification")
    );
    assert_eq!(m3.installs.count(), 0, "the candidate installs nothing");
    assert!(n3.ready() || state_of(&g, 3) == "ready");
    tick.abort();
}

/// **A member below the floor is admitted to upgrade** (§3.5): the floor's
/// build is held, so it is let in — upgrading, taking no traffic — offered
/// the build at once, fetches it and installs it with no lease.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_below_the_floor_upgrades_on_admission() {
    let g = gateway("g1");
    let (m1, m2) = (machine(12, &[]), machine(10, &[]));
    let n1 = join(&g, 1, 12, &m1);
    until("the floor's build is held", WITHIN, || {
        n1.ready() && !g.status(0).builds.is_empty()
    })
    .await;
    let n2 = join(&g, 2, 10, &m2);
    until("it is upgrading", WITHIN, || state_of(&g, 2) == "upgrading").await;
    until("it installs the floor's build", WITHIN, || {
        m2.installs.count() == 1
    })
    .await;
    let job = m2.installs.jobs.lock().unwrap()[0].clone();
    assert_eq!((job.build, job.leases.len()), (b(12), 0));
    assert!(!n2.ready(), "an upgrading member is never ready for work");
}

/// **A failed certification faults the candidate and raises nothing**
/// (§3.8): its error rate far above the baseline's, the gateway sends a
/// failing verdict at once; the group rejects the build, the candidate is in
/// fault mode, the floor stays, and no older member installs it. A gateway
/// restarted after it never takes the rejected build for its floor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_certification_rejects_the_build() {
    let g = gateway("g1");
    let (m1, m2, m3) = (machine(10, &[]), machine(10, &[]), machine(12, &[]));
    let n1 = join(&g, 1, 10, &m1);
    let n2 = join(&g, 2, 10, &m2);
    until("the older members ready", WITHIN, || {
        n1.ready() && n2.ready()
    })
    .await;
    let n3 = join(&g, 3, 12, &m3);
    until("the candidate certifies", WITHIN, || {
        g.status(0)
            .upgrades
            .iter()
            .any(|u| u.candidate.as_ref().is_some_and(|c| c.ready_ms.is_some()))
    })
    .await;
    for _ in 0..300 {
        g.certify_sample(APP, "GET /v1/x", &Mirror::None, 3, true, 10);
        g.certify_sample(APP, "GET /v1/x", &Mirror::None, 1, false, 10);
    }
    let tick = ticking(&g);
    until("the candidate is in fault mode", WITHIN, || {
        state_of(&g, 3) == "fault"
    })
    .await;
    until("the gateway knows the build rejected", WITHIN, || {
        g.status(0)
            .upgrades
            .iter()
            .any(|u| u.rejected.contains(&"12-h12".to_string()))
    })
    .await;
    sleep(Duration::from_millis(500)).await;
    assert_eq!(floor(&g), Some(10));
    assert_eq!(m1.installs.count() + m2.installs.count(), 0);
    let prefix = "certify/zend/12-h12/";
    let (_, verdict) = latest::<Verdict>(&n1.store.decisions(prefix), prefix).unwrap();
    assert_eq!(verdict.result, Settled::Fail);
    assert_eq!(
        verdict.reason,
        "GET /v1/x: its error rate is worse than the baseline's"
    );
    assert!(!n3.ready(), "a faulted candidate takes no work");
    tick.abort();
    // A gateway restarted after the verdict (§3.11): the rejected candidate,
    // back first, says so in its `HELLO` — it sets no floor and is no
    // candidate — and the next member back sets the floor.
    let (store, _) = n3.stop();
    let g2 = gateway("g2");
    let m3b = machine(12, &[]);
    let back = member_upgrading(&[&g2], 3, store, b(12), &m3b.up);
    until("the rejected member is back", WITHIN, || {
        !state_of(&g2, 3).is_empty()
    })
    .await;
    assert_eq!(floor(&g2), None, "a rejected build sets no floor");
    assert!(g2.status(0).upgrades.iter().all(|u| u.candidate.is_none()));
    let m4 = machine(10, &[]);
    let n4 = join(&g2, 4, 10, &m4);
    until("the next member sets the floor", WITHIN, || {
        floor(&g2) == Some(10)
    })
    .await;
    until("it serves", WITHIN, || n4.ready()).await;
    drop(back);
}

/// **A downgrade swaps from every member's own kept build** (§3.11): the plan
/// finds the build kept on each target and fetches nothing; once decided, the
/// floor falls at once, the builds above are rejected, and the members install
/// it one at a time from `builds/`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_swaps_from_the_kept_builds() {
    let g = gateway("g1");
    let ms = [machine(12, &[10]), machine(12, &[10]), machine(12, &[10])];
    let nodes: Vec<Node> = ms
        .iter()
        .enumerate()
        .map(|(i, m)| join(&g, i as u64 + 1, 12, m))
        .collect();
    until("every member ready and reporting", WITHIN, || {
        nodes.iter().all(Node::ready)
            && g.status(0)
                .members
                .iter()
                .filter(|m| m.coord.as_ref().is_some_and(|c| c.upgrade.is_some()))
                .count()
                == 3
    })
    .await;
    let tick = ticking(&g);
    let plan = g
        .plan_rollout(&Ask {
            app: APP.into(),
            build: "10-h10".into(),
            from: None,
            targets: None,
            parallel: 1,
            by: "ops@tokera.com".into(),
        })
        .unwrap();
    assert_eq!(plan.direction, "downgrade");
    assert!(plan
        .targets
        .iter()
        .all(|t| t.source == "kept" && t.bytes == 0));
    assert_eq!(plan.targets.len(), 3);
    assert_eq!(plan.rejects, ["12-h12"]);
    let n = ms[0].up.rollout(APP, &plan.decision).await.unwrap();
    assert_eq!(n, 1);
    until("the floor falls to 10", WITHIN, || floor(&g) == Some(10)).await;
    until("a member installs", WITHIN, || {
        ms.iter().map(|m| m.installs.count()).sum::<usize>() == 1
    })
    .await;
    sleep(Duration::from_millis(600)).await;
    assert_eq!(ms.iter().map(|m| m.installs.count()).sum::<usize>(), 1);
    let job = ms
        .iter()
        .find_map(|m| m.installs.jobs.lock().unwrap().first().cloned())
        .unwrap();
    assert_eq!(job.source, Source::Kept("10-h10".into()));
    assert_eq!(job.leases, ["upgrade/zend/0"]);
    assert_eq!(job.protect, [b(10)]);
    until("the gateway knows 12 rejected", WITHIN, || {
        g.status(0)
            .upgrades
            .iter()
            .any(|u| u.rejected.contains(&"12-h12".to_string()))
    })
    .await;
    tick.abort();
    drop(nodes);
}
