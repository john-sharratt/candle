//! Real substrates replicating through a gateway (`docs/replicated_substrate.md`
//! §6–§7): [`ConversationStore`] over CPU [`Conversation`]s, members on an
//! in-process gateway. What one member writes reaches the other live; a
//! member that joins later catches up; both end holding the same applied set
//! and the same values.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{Conversation, Substrate};
use cluster::dial::Connect;
use cluster::dispatch::{Dispatch, Route, RouteTable};
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Identity};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::replicate::{Store, Tombstones};
use host::cluster::store::{ConversationStore, VOTES};
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

fn key() -> GroupKey {
    GroupKey::from_bytes([4; 32])
}

struct ToHub(Arc<Hub>);

impl Connect for ToHub {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        let (m, g) = pair(4 << 20);
        let hub = self.0.clone();
        tokio::spawn(async move {
            hub.serve("zend", IpAddr::from([10, 0, 0, 2]), g.tx, g.rx, |_| {})
                .await;
        });
        Ok((m.tx, m.rx))
    }
}

struct Member {
    conv: Conversation,
    app: Arc<AppMember>,
    _dir: TempDir,
}

fn member(hub: &Arc<Hub>) -> Member {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let conv = Conversation::from_parts(substrate, sp);
    let store = ConversationStore::new(conv.clone(), 0, || {});
    let replication = store.replication().unwrap();
    let app = AppMember::new(
        Identity {
            app: "zend".into(),
            node: conv.node(),
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
            repos: vec![],
        },
        key(),
        vec!["https://gw".into()],
        RouteTable::new(vec![Route::new(&[], "/v1/models", Dispatch::Any)]).unwrap(),
        Box::new(|| Capacity {
            state: MemberState::Ready,
            power: 1,
            idle: true,
            slots_free: 1,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 0,
            cores: 1,
            load_milli: 0,
        }),
        Some(replication),
        Instant::now(),
    );
    let run = app.clone();
    let hub = hub.clone();
    tokio::spawn(async move { run.run(0, ToHub(hub), |conn| conn.reset()).await });
    Member {
        conv,
        app,
        _dir: dir,
    }
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

fn value(m: &Member, key: &str) -> Option<serde_json::Value> {
    m.conv
        .cluster_objects()
        .into_iter()
        .find(|(k, _, _)| k == key)
        .map(|(_, v, _)| v)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn substrates_replicate_live_and_by_catch_up() {
    let hub = Hub::new(key(), "gw");
    let a = member(&hub);
    let b = member(&hub);
    let caught_up = |m: &Member| m.app.replicator().is_some_and(|r| r.caught_up());
    until("both caught up", || caught_up(&a) && caught_up(&b)).await;
    assert!(a.conv.substrate_dir().join(VOTES).exists());

    for k in 0..5 {
        a.conv
            .put_cluster_object(&format!("cluster/a{k}"), json!(k))
            .unwrap();
        b.conv
            .put_cluster_object(&format!("cluster/b{k}"), json!(k))
            .unwrap();
    }
    until("live events applied both ways", || {
        value(&a, "cluster/b4") == Some(json!(4)) && value(&b, "cluster/a4") == Some(json!(4))
    })
    .await;

    // C joins later and catches up on both origins.
    let c = member(&hub);
    until("c caught up", || {
        caught_up(&c) && value(&c, "cluster/a4").is_some() && value(&c, "cluster/b4").is_some()
    })
    .await;
    until("one applied set", || {
        let (sa, sb, sc) = (
            a.conv.applied_set(),
            b.conv.applied_set(),
            c.conv.applied_set(),
        );
        sa == sb && sb == sc
    })
    .await;
}

/// What operating a member reads of a real substrate (§7.7, §12.1): its
/// volume measured with no write failed, a test write that succeeds, nothing
/// deferred and no tombstones on a fresh store — and the member reports it in
/// `COORD`, so the gateway's page carries its free space.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_real_store_reports_its_disk_and_state() {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let conv = Conversation::from_parts(substrate, sp);
    let store = ConversationStore::new(conv, 0, || {});
    let disk = store.disk();
    assert!(disk.total > 0 && disk.free <= disk.total, "{disk:?}");
    assert!(!disk.write_failed);
    assert!(store.test_write());
    assert!(store.deferred().is_empty());
    assert_eq!(store.tombstones(), Tombstones::default());
    assert!(
        !store.drop_dead_segment(),
        "a fresh log has no dead segment"
    );

    let hub = Hub::new(key(), "gw");
    let m = member(&hub);
    let node = format!("{:016x}", m.conv.node());
    until("the page has its disk", || {
        hub.status(0)
            .members
            .iter()
            .any(|r| r.node == node && r.coord.as_ref().is_some_and(|c| c.disk.total > 0))
    })
    .await;
}
