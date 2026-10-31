//! The tick against a store that does not answer: it finishes within its
//! store wait, says what it is blocked on, asks the held store nothing more,
//! and goes on once the store answers.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use applied_set::AppliedSet;
use tempfile::TempDir;
use tokio::time::sleep;

use super::{InstallJob, Installer, Setup, Upgrader};
use crate::consensus::votes::MemoryVotes;
use crate::dispatch::{Dispatch, Route, RouteTable};
use crate::frame::{Capacity, MemberState};
use crate::handshake::GroupKey;
use crate::member::{AppMember, Identity};
use crate::replicate::{Applied, Disk, Listing, MemoryStore, Replication, Store, Tombstones};
use crate::upgrade::artifacts::Probe;
use crate::version::info::{platform, BuildInfo, Opens, ZEND};

const APP: &str = "zend";

/// A [`MemoryStore`] whose census does not return while `held` is set — a
/// store whose own thread is busy with something else.
struct HeldStore {
    inner: MemoryStore,
    held: AtomicBool,
    censuses: AtomicUsize,
}

impl HeldStore {
    fn new() -> Arc<HeldStore> {
        Arc::new(HeldStore {
            inner: MemoryStore::new(1),
            held: AtomicBool::new(true),
            censuses: AtomicUsize::new(0),
        })
    }
}

impl Store for HeldStore {
    fn node(&self) -> u64 {
        self.inner.node()
    }
    fn own_through(&self) -> u64 {
        self.inner.own_through()
    }
    fn set_tap(&self, tap: Box<dyn Fn(u64) + Send + Sync>) {
        self.inner.set_tap(tap)
    }
    fn read(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        self.inner.read(origin, oseq)
    }
    fn applied(&self) -> AppliedSet {
        self.inner.applied()
    }
    fn apply(&self, records: &[(u64, u64, Vec<u8>)]) -> Applied {
        self.inner.apply(records)
    }
    fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        self.inner.list(origin, from, to)
    }
    fn gone(&self, origin: u64, gone: &[u64]) {
        self.inner.gone(origin, gone)
    }
    fn travels(&self, record_type: u8) -> bool {
        self.inner.travels(record_type)
    }
    fn ticket(&self) -> u64 {
        self.inner.ticket()
    }
    fn wait_written(&self, ticket: u64) {
        self.inner.wait_written(ticket)
    }
    fn detach(&self) {
        self.inner.detach()
    }
    fn detached(&self) -> bool {
        self.inner.detached()
    }
    fn rejoin(&self, gen: u64) {
        self.inner.rejoin(gen)
    }
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    fn record_decision(&self, key: &[u8], value: &[u8]) {
        self.inner.record_decision(key, value)
    }
    fn decision(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.decision(key)
    }
    fn max_members(&self) -> u32 {
        self.inner.max_members()
    }
    fn raise_max_members(&self, seen: u32) {
        self.inner.raise_max_members(seen)
    }
    fn set_max_members(&self, n: u32) {
        self.inner.set_max_members(n)
    }
    fn reload(&self) {
        self.inner.reload()
    }
    fn disk(&self) -> Disk {
        self.inner.disk()
    }
    fn test_write(&self) -> bool {
        self.inner.test_write()
    }
    fn drop_dead_segment(&self) -> bool {
        self.inner.drop_dead_segment()
    }
    fn deferred(&self) -> Vec<(String, u64, usize)> {
        self.inner.deferred()
    }
    fn census(&self) -> Vec<u8> {
        self.censuses.fetch_add(1, Ordering::SeqCst);
        while self.held.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        vec![7, 3]
    }
    fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        self.inner.decisions(prefix)
    }
    fn tombstones(&self) -> Tombstones {
        self.inner.tombstones()
    }
    fn mark_joined(&self) {
        self.inner.mark_joined()
    }
}

struct NoInstall;

impl Installer for NoInstall {
    fn install(&self, _job: InstallJob) -> Result<(), String> {
        Err("no installs here".into())
    }
}

struct NoProbe;

impl Probe for NoProbe {
    fn build_info(&self, _exe: &Path) -> Option<BuildInfo> {
        None
    }
}

fn info() -> BuildInfo {
    BuildInfo {
        kind: ZEND.into(),
        count: 10,
        hash: "h10".into(),
        local: String::new(),
        platform: platform(),
        opens: Opens::default(),
    }
}

/// An upgrader in `dir` with one app, replicating over `store`. The member
/// never dials: the tick needs only its replicator.
fn upgrader(dir: &TempDir, store: Arc<HeldStore>) -> (Arc<Upgrader>, Arc<AppMember>) {
    let up = Upgrader::new(
        Setup {
            info: info(),
            exe: dir.path().join("zend.exe"),
            upgrade_dir: dir.path().join("upgrade"),
            keep_builds: 2,
            failed_from: None,
        },
        Arc::new(NoInstall),
        Arc::new(NoProbe),
    );
    let replication = Replication::open(store, Box::new(MemoryVotes::new())).unwrap();
    let member = AppMember::new(
        Identity {
            app: APP.into(),
            node: 1,
            host: "m1".into(),
            card: 1,
            build: info().build(),
            model: None,
            repos: vec![],
            leaseholder: true,
        },
        GroupKey::from_bytes([5; 32]),
        vec!["https://gw".into()],
        RouteTable::new(vec![Route::new(
            &[],
            "/v1/conversations/:id",
            Dispatch::sticky_path("id"),
        )])
        .unwrap(),
        Box::new(|| Capacity {
            state: MemberState::Ready,
            ..Capacity::default()
        }),
        Some(replication),
        Instant::now(),
    );
    up.register(&member);
    (up, member)
}

/// **A store that does not answer its census holds the tick no longer than
/// the store wait.** The step names the store and the call; the next tick
/// asks it nothing new while the first read is still out; and once the store
/// answers, a tick takes its census and the step is no longer blocked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_held_store_blocks_the_tick_for_its_wait_and_no_longer() {
    let dir = TempDir::new().unwrap();
    let store = HeldStore::new();
    let (up, _member) = upgrader(&dir, store.clone());
    let wait = Duration::from_millis(200);

    let started = Instant::now();
    up.tick_within(wait).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the tick waited {:?} on a held store",
        started.elapsed()
    );
    let blocked = "blocked: zend's store did not answer census in 200ms";
    assert_eq!(up.view(APP).step, blocked);

    let again = Instant::now();
    up.tick_within(wait).await;
    assert!(
        again.elapsed() < wait,
        "a tick with a read already out waits for nothing"
    );
    assert_eq!(up.view(APP).step, blocked);
    assert_eq!(
        store.censuses.load(Ordering::SeqCst),
        1,
        "a held store is not asked again"
    );

    store.held.store(false, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        up.tick_within(wait).await;
        if up.view(APP).census == [3, 7] {
            break;
        }
        assert!(Instant::now() < deadline, "the answered census was taken");
        sleep(Duration::from_millis(20)).await;
    }
    assert!(
        !up.view(APP).step.starts_with("blocked"),
        "the step after the store answered: {}",
        up.view(APP).step
    );
    assert!(up.state().blocked.is_none());
}
