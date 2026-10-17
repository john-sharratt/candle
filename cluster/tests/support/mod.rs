//! Members replicating in one process: in-process gateways ([`Hub`]s) over
//! memory pipes, and members whose substrate is a [`MemoryStore`].

#![allow(dead_code)]

use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use applied_set::AppliedSet;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinHandle;
use tokio::time::sleep;

use cluster::consensus::votes::MemoryVotes;
use cluster::dial::Connect;
use cluster::dispatch::{Dispatch, Route, RouteTable};
use cluster::frame::Gone;
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Identity, Link};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::pipe::{Message, PipeRx};
use cluster::replicate::{Applied, Disk, Listing, MemoryStore, Replication, Store, Tombstones};
use cluster::session::VirtualConn;
use cluster::upgrade::runtime::Upgrader;

pub const APP: &str = "zend";

pub fn key() -> GroupKey {
    GroupKey::from_bytes([5; 32])
}

pub fn gateway(name: &str) -> Arc<Hub> {
    Hub::new(key(), name)
}

/// A member's receiving half that stops reading while `paused` — a member too
/// slow to take what its gateway sends.
pub struct PausableRx {
    inner: MemoryRx,
    paused: Arc<AtomicBool>,
}

impl PipeRx for PausableRx {
    async fn recv(&mut self) -> Option<Message> {
        while self.paused.load(Ordering::SeqCst) {
            sleep(Duration::from_millis(5)).await;
        }
        self.inner.recv().await
    }
}

/// Where a cluster's gateway is: empty while it is down, and replaced by a
/// fresh one when it restarts.
pub type HubSlot = Arc<Mutex<Option<Arc<Hub>>>>;

pub fn slot(hub: &Arc<Hub>) -> HubSlot {
    Arc::new(Mutex::new(Some(hub.clone())))
}

/// A connector to the gateway in `hub` that refuses while `open` is false — a
/// cluster this member cannot reach — or while the gateway is down.
pub struct ToHub {
    pub hub: HubSlot,
    pub open: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
}

impl Connect for ToHub {
    type Tx = MemoryTx;
    type Rx = PausableRx;
    async fn connect(&mut self) -> Result<(MemoryTx, PausableRx), String> {
        if !self.open.load(Ordering::SeqCst) {
            return Err("unreachable".into());
        }
        let Some(hub) = self.hub.lock().unwrap().clone() else {
            return Err("the gateway is down".into());
        };
        let (m, g) = pair(4 << 20);
        tokio::spawn(async move {
            hub.serve(APP, IpAddr::from([10, 0, 0, 9]), g.tx, g.rx, |_| {})
                .await;
        });
        Ok((
            m.tx,
            PausableRx {
                inner: m.rx,
                paused: self.paused.clone(),
            },
        ))
    }
}

pub struct Node {
    pub node: u64,
    pub app: Arc<AppMember>,
    pub store: Arc<MemoryStore>,
    pub votes: MemoryVotes,
    /// Per cluster: whether it can be reached.
    pub open: Vec<Arc<AtomicBool>>,
    /// Per cluster: whether the member has stopped reading its socket.
    pub paused: Vec<Arc<AtomicBool>>,
    tasks: Vec<JoinHandle<()>>,
}

impl Node {
    /// Stop the member's process: its sockets close and its tasks end. Its
    /// store and votes stay, for a restart.
    pub fn stop(self) -> (Arc<MemoryStore>, MemoryVotes) {
        self.app.shutdown("stopping");
        for t in self.tasks {
            t.abort();
        }
        (self.store, self.votes)
    }

    pub fn reach(&self, c: usize, open: bool) {
        self.open[c].store(open, Ordering::SeqCst);
    }

    pub fn pause(&self, c: usize, paused: bool) {
        self.paused[c].store(paused, Ordering::SeqCst);
    }

    pub fn ready(&self) -> bool {
        self.app.replicator().is_some_and(|r| r.caught_up()) && self.app.link() == Link::Joined
    }
}

/// Member `node` of [`APP`], on every gateway in `hubs`, over `store`.
pub fn member(
    hubs: &[&Arc<Hub>],
    node: u64,
    store: Arc<MemoryStore>,
    votes: MemoryVotes,
    open: bool,
) -> Node {
    let slots: Vec<HubSlot> = hubs.iter().map(|h| slot(h)).collect();
    build(&slots, node, store.clone(), store, votes, open, one(), None)
}

/// Member `node` on the gateways wherever `slots` say they are.
pub fn member_on(slots: &[HubSlot], node: u64, store: Arc<MemoryStore>) -> Node {
    build(
        slots,
        node,
        store.clone(),
        store,
        MemoryVotes::new(),
        true,
        one(),
        None,
    )
}

/// The build every member runs unless a test says otherwise.
pub fn one() -> BuildId {
    BuildId {
        count: 1,
        hash: "h1".into(),
        local: String::new(),
    }
}

/// Member `node` at `build_id`, upgrading through `upgrader`, which it is
/// registered with before it dials.
pub fn member_upgrading(
    hubs: &[&Arc<Hub>],
    node: u64,
    store: Arc<MemoryStore>,
    build_id: BuildId,
    upgrader: &Arc<Upgrader>,
) -> Node {
    let slots: Vec<HubSlot> = hubs.iter().map(|h| slot(h)).collect();
    build(
        &slots,
        node,
        store.clone(),
        store,
        MemoryVotes::new(),
        true,
        build_id,
        Some(upgrader),
    )
}

#[allow(clippy::too_many_arguments)]
fn build(
    hubs: &[HubSlot],
    node: u64,
    replicated: Arc<dyn Store>,
    store: Arc<MemoryStore>,
    votes: MemoryVotes,
    open: bool,
    build_id: BuildId,
    upgrader: Option<&Arc<Upgrader>>,
) -> Node {
    let replication = Replication::open(replicated, Box::new(votes.clone())).unwrap();
    let app = AppMember::new(
        Identity {
            app: APP.into(),
            node,
            host: format!("m{node}"),
            build: build_id,
            model: Some(ModelId {
                arch: "Qwen4Exp".into(),
                weights: [1; 32],
                tokenizer: [1; 32],
                rope: "yarn".into(),
                describe: "model".into(),
            }),
            repos: vec![],
        },
        key(),
        hubs.iter().map(|_| "https://gw".to_string()).collect(),
        RouteTable::new(vec![Route::new(
            &[],
            "/v1/conversations/:id",
            Dispatch::sticky_path("id"),
        )])
        .unwrap(),
        Box::new(move || Capacity {
            state: MemberState::Ready,
            power: 100,
            idle: true,
            slots_free: 4,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 1000,
            cores: 8,
            ..Capacity::default()
        }),
        Some(replication),
        Instant::now(),
    );
    if let Some(u) = upgrader {
        u.register(&app);
    }
    let mut opens = Vec::new();
    let mut pauses = Vec::new();
    let mut tasks = Vec::new();
    for (c, hub) in hubs.iter().enumerate() {
        let gate = Arc::new(AtomicBool::new(open));
        let paused = Arc::new(AtomicBool::new(false));
        opens.push(gate.clone());
        pauses.push(paused.clone());
        let connector = ToHub {
            hub: hub.clone(),
            open: gate,
            paused,
        };
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            app.run(c, connector, |conn| {
                tokio::spawn(echo(conn));
            })
            .await;
        }));
    }
    Node {
        node,
        app,
        store,
        votes,
        open: opens,
        paused: pauses,
        tasks,
    }
}

/// A member's answer to a request a gateway routed to it: the request's bytes
/// back, then the end of the stream.
async fn echo(mut conn: VirtualConn) {
    let mut request = Vec::new();
    if conn.read_to_end(&mut request).await.is_ok() {
        let _ = conn.write_all(&request).await;
        let _ = conn.shutdown().await;
    }
}

/// Wait for `ok`, failing after `within`.
pub async fn until(what: &str, within: Duration, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + within;
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

/// A [`MemoryStore`] that records every record it is asked to apply, in
/// arrival order and with the time it arrived: what the burst harnesses check
/// exactly-once, per-origin order and latency against.
pub struct Recording {
    pub inner: Arc<MemoryStore>,
    pub arrivals: Mutex<Vec<(u64, u64)>>,
    pub at: Mutex<Vec<Instant>>,
}

impl Recording {
    pub fn new(node: u64) -> Arc<Recording> {
        Arc::new(Recording {
            inner: Arc::new(MemoryStore::new(node)),
            arrivals: Mutex::new(Vec::new()),
            at: Mutex::new(Vec::new()),
        })
    }

    /// When `(origin, oseq)` first arrived.
    pub fn arrived(&self, origin: u64, oseq: u64) -> Option<Instant> {
        let arrivals = self.arrivals.lock().unwrap();
        let at = self.at.lock().unwrap();
        arrivals
            .iter()
            .position(|k| *k == (origin, oseq))
            .map(|i| at[i])
    }
}

impl Store for Recording {
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
        let now = Instant::now();
        self.arrivals
            .lock()
            .unwrap()
            .extend(records.iter().map(|(o, q, _)| (*o, *q)));
        self.at.lock().unwrap().extend(records.iter().map(|_| now));
        self.inner.apply(records)
    }
    fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        self.inner.list(origin, from, to)
    }
    fn gone(&self, origin: u64, gone: &[Gone]) {
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
        self.inner.census()
    }
    fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        self.inner.decisions(prefix)
    }
    fn tombstones(&self) -> Tombstones {
        self.inner.tombstones()
    }
}

/// Member `node` over a store of any kind; its [`Node::store`] is `inner`.
pub fn member_over(
    hubs: &[&Arc<Hub>],
    node: u64,
    store: Arc<dyn Store>,
    inner: Arc<MemoryStore>,
    votes: MemoryVotes,
) -> Node {
    let slots: Vec<HubSlot> = hubs.iter().map(|h| slot(h)).collect();
    build(&slots, node, store, inner, votes, true, one(), None)
}

/// Whether every store holds the same records.
pub fn converged(stores: &[&MemoryStore]) -> bool {
    let first = stores[0].snapshot();
    stores.iter().all(|s| s.snapshot() == first)
}

/// Each store's applied set, for the assertion that all ended equal.
pub fn applied(store: &MemoryStore) -> AppliedSet {
    store.applied()
}
