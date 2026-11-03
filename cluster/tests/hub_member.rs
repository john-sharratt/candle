//! A gateway's hub and members, in process over memory pipes (§3–§5): the
//! tables of §14.1 that need both ends running — close codes and admission,
//! modes, claims, both directions of stream, and a dropped socket failing its
//! streams.

use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};

use cluster::dial::{Connect, Stopped};
use cluster::dispatch::{Dispatch, Route, RouteTable};
use cluster::frame::{BuildId, Capacity, Epoch, Frame, Hello, MemberState, ModelId, Prove};
use cluster::gateway::{Hub, MemberStream, Pick};
use cluster::handshake::{member_proof, nonce, GroupKey, PROTO};
use cluster::machines::MachineTable;
use cluster::member::{AppMember, Check, Identity, Link};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::pipe::{Message, PipeRx, PipeTx};
use cluster::session::VirtualConn;

const IP: [u8; 4] = [10, 0, 0, 7];

fn key() -> GroupKey {
    GroupKey::from_bytes([3; 32])
}

fn model(weights: u8) -> ModelId {
    ModelId {
        arch: "Qwen4Exp".into(),
        weights: [weights; 32],
        tokenizer: [1; 32],
        rope: "yarn".into(),
        describe: format!("model {weights}"),
    }
}

fn build(count: u32) -> BuildId {
    BuildId {
        count,
        hash: format!("h{count}"),
        local: String::new(),
    }
}

fn routes() -> RouteTable {
    RouteTable::new(vec![
        Route::new(&[], "/v1/conversations/:id", Dispatch::sticky_path("id")),
        Route::new(&[], "/v1/models", Dispatch::Any),
    ])
    .unwrap()
}

/// A connector that hands the gateway end of a fresh pipe to `hub`.
struct ToHub {
    hub: Arc<Hub>,
    app: &'static str,
    up: mpsc::UnboundedSender<MemberStream>,
}

impl Connect for ToHub {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        let (m, g) = pair(1 << 20);
        let hub = self.hub.clone();
        let app = self.app;
        let up = self.up.clone();
        tokio::spawn(async move {
            hub.serve(app, IpAddr::from(IP), g.tx, g.rx, move |s| {
                let _ = up.send(s);
            })
            .await;
        });
        Ok((m.tx, m.rx))
    }
}

struct Member {
    app: Arc<AppMember>,
    ready: Arc<AtomicBool>,
    down: mpsc::UnboundedReceiver<VirtualConn>,
    stopped: tokio::task::JoinHandle<Stopped>,
}

fn member(
    hub: &Arc<Hub>,
    node: u64,
    build: BuildId,
    model: ModelId,
    up: &mpsc::UnboundedSender<MemberStream>,
) -> Member {
    let ready = Arc::new(AtomicBool::new(false));
    let r = ready.clone();
    let app = AppMember::new(
        Identity {
            app: "zend".into(),
            node,
            host: format!("m{node}"),
            card: node,
            build,
            model: Some(model),
            repos: vec![],
            leaseholder: true,
        },
        key(),
        vec!["https://gw".into()],
        routes(),
        Box::new(move || Capacity {
            state: if r.load(Ordering::SeqCst) {
                MemberState::Ready
            } else {
                MemberState::Loading
            },
            power: 100 + node as u32,
            idle: true,
            slots_free: 4,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 1000,
            cores: 8,
            ..Capacity::default()
        }),
        None,
        Instant::now(),
    );
    let (tx, down) = mpsc::unbounded_channel();
    let connector = ToHub {
        hub: hub.clone(),
        app: "zend",
        up: up.clone(),
    };
    let stopped = tokio::spawn(app.clone().run(0, connector, move |conn| {
        let _ = tx.send(conn);
    }));
    Member {
        app,
        ready,
        down,
        stopped,
    }
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(5)).await;
    }
}

/// Admission, the mode switch, routing by the machine table and claims, and
/// streams in both directions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn members_join_route_and_carry_streams() {
    let hub = Hub::new(key(), "gw-test");
    // B's machine ranks first; each is filled to one unit before any round.
    hub.set_machines(
        MachineTable::from_yaml(
            "m1:\n  conversations: {rank: 2, min: 1, max: 4}\n\
             m2:\n  conversations: {rank: 1, min: 1, max: 4}\n",
        )
        .unwrap(),
    );
    let (up_tx, mut up) = mpsc::unbounded_channel();
    let mut a = member(&hub, 1, build(10), model(1), &up_tx);
    // One build: whichever is admitted first sets the floor, and a lower
    // build behind it would be refused 4410.
    let b = member(&hub, 2, build(10), model(1), &up_tx);

    // Admitted and loading: joined, but the app is dumb until one is ready.
    until("both joined", || {
        a.app.link() == Link::Joined && b.app.link() == Link::Joined
    })
    .await;
    assert!(!hub.clustered("zend"));
    assert_eq!(
        hub.pick("zend", &Dispatch::Any, None, None, None).pick,
        Pick::Configured
    );

    a.ready.store(true, Ordering::SeqCst);
    a.app.capacity_changed();
    until("clustered", || hub.clustered("zend")).await;
    assert_eq!(hub.table("zend"), Some(routes()));
    // Only A is ready: every class goes to it.
    assert_eq!(
        hub.pick("zend", &Dispatch::Best, None, None, None).pick,
        Pick::Member(1)
    );
    b.ready.store(true, Ordering::SeqCst);
    b.app.capacity_changed();
    // B's machine ranks first.
    until("B ready", || {
        hub.pick("zend", &Dispatch::Best, None, None, None).pick == Pick::Member(2)
    })
    .await;
    // A request B is answering fills it to its `min`: the next goes to A, still
    // below its own, and to B again once B's answer is done.
    let held = hub.pick("zend", &Dispatch::Best, None, None, None);
    assert_eq!(held.pick, Pick::Member(2));
    assert!(held.load.is_some(), "a Best request is work for its member");
    assert_eq!(
        hub.pick("zend", &Dispatch::Best, None, None, None).pick,
        Pick::Member(1)
    );
    assert!(
        hub.pick("zend", &Dispatch::Any, None, None, None)
            .load
            .is_none(),
        "an Any request is not counted"
    );
    drop(held);
    assert_eq!(
        hub.pick("zend", &Dispatch::Best, None, None, None).pick,
        Pick::Member(2)
    );

    // A claims a conversation; the gateway routes it there, and B, told of
    // the claim, refuses to serve it.
    assert!(matches!(a.app.claim(b"c-1"), Check::Took(_)));
    until("the claim routed", || {
        hub.pick(
            "zend",
            &Dispatch::sticky_path("id"),
            None,
            Some("c-1"),
            None,
        )
        .pick
            == Pick::Member(1)
    })
    .await;
    until("B told", || b.app.claim(b"c-1") == Check::Held(1)).await;

    // Gateway → member: a stream reaches A and carries bytes both ways.
    let mut conn = hub.open("zend", 1, "zend").expect("A's socket");
    let mut served = a.down.recv().await.unwrap();
    conn.write_all(b"GET / HTTP/1.1\r\n\r\n").await.unwrap();
    let mut buf = [0u8; 18];
    served.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"GET / HTTP/1.1\r\n\r\n");
    served.write_all(b"ok").await.unwrap();
    served.shutdown().await.unwrap();
    let mut back = Vec::new();
    conn.read_to_end(&mut back).await.unwrap();
    assert_eq!(back, b"ok");

    // Member → gateway: a stream up arrives as A's, for the app.
    let mut upstream = a.app.open_up().expect("a ready socket");
    let MemberStream {
        app,
        node,
        target,
        conn: mut at_gateway,
    } = up.recv().await.unwrap();
    assert_eq!(
        (app.as_str(), node, target.as_str(), at_gateway.target()),
        ("zend", 1, "zend", "zend")
    );
    upstream.write_all(b"x").await.unwrap();
    let mut one = [0u8; 1];
    at_gateway.read_exact(&mut one).await.unwrap();
    assert_eq!(&one, b"x");

    // A stream to the agents' group arrives as A's, aimed at the group.
    let _jobs = a.app.open_group("agent").expect("a ready socket");
    let MemberStream { app, target, .. } = up.recv().await.unwrap();
    assert_eq!((app.as_str(), target.as_str()), ("zend", "group:agent"));

    // The log saw the admissions, the switch and the claim.
    let texts: Vec<String> = hub
        .log
        .read(Some("zend"), None, 100)
        .into_iter()
        .map(|e| e.text)
        .collect();
    assert!(
        texts.iter().any(|t| t.starts_with("admitted m1")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.starts_with("clustered")),
        "{texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("claimed by 0000000000000001")),
        "{texts:?}"
    );

    // A goes: B takes the conversation over, and the app stays clustered.
    a.app.close_all("test");
    a.stopped.abort();
    until("A left", || hub.member("zend", 1).is_none()).await;
    until("B takes over", || {
        matches!(b.app.claim(b"c-1"), Check::Took(_) | Check::Mine)
    })
    .await;
    assert!(hub.clustered("zend"));
}

/// The claims the gateway lists: key, holder, epoch.
fn listed(hub: &Hub) -> Vec<(String, String, u64)> {
    hub.status(10)
        .claims
        .into_iter()
        .map(|c| (c.key, c.node, c.epoch_ms))
        .collect()
}

/// A hub with A (node 1) and B (node 2) ready, A holding `c-1` at the epoch
/// returned, and B and the gateway told.
async fn a_holds_c1(
    hub: &Arc<Hub>,
    up_tx: &mpsc::UnboundedSender<MemberStream>,
) -> (Member, Member, u64) {
    hub.set_machines(
        MachineTable::from_yaml(
            "m1:\n  conversations: {rank: 1, min: 1, max: 4}\n\
             m2:\n  conversations: {rank: 2, min: 1, max: 4}\n",
        )
        .unwrap(),
    );
    let a = member(hub, 1, build(10), model(1), up_tx);
    let b = member(hub, 2, build(10), model(1), up_tx);
    for m in [&a, &b] {
        m.ready.store(true, Ordering::SeqCst);
    }
    until("both joined", || {
        a.app.link() == Link::Joined && b.app.link() == Link::Joined
    })
    .await;
    a.app.capacity_changed();
    b.app.capacity_changed();
    let Check::Took(epoch) = a.app.claim(b"c-1") else {
        panic!("A takes c-1");
    };
    until("B told", || b.app.holder_of(b"c-1") == Some(1)).await;
    until("the gateway told", || {
        listed(hub) == vec![("c-1".into(), "0000000000000001".into(), epoch.ms)]
    })
    .await;
    (a, b, epoch.ms)
}

/// **A member that restarts holding a claim releases it, and everyone agrees**
/// (§5.4). A holds `c-1` and B knows it; a new process for A's node is
/// admitted, replacing the old. Its `ADMIT` still names node 1 the holder — the
/// gateway keeps a claim until its holder lets it go — and the new process,
/// holding nothing, does: it takes nothing in and `UNCLAIM`s it. B and the
/// gateway hear the release, and the key, next asked for, is taken with a
/// `CLAIM` of the new process's own that all three agree on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restarted_member_releases_its_old_claims_and_everyone_agrees() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let (a, b, _) = a_holds_c1(&hub, &up_tx).await;

    // A restarts: a new process for node 1, holding nothing.
    let again = member(&hub, 1, build(10), model(1), &up_tx);
    again.ready.store(true, Ordering::SeqCst);
    until("restarted and admitted", || {
        again.app.link() == Link::Joined
    })
    .await;
    timeout(Duration::from_secs(5), a.stopped)
        .await
        .expect("the old session is replaced")
        .unwrap();
    assert_eq!(
        again.app.holder_of(b"c-1"),
        None,
        "it takes in no claim of its node it does not hold"
    );
    assert!(again.app.own_claims().is_empty());
    until("released, and B and the gateway told", || {
        b.app.holder_of(b"c-1").is_none() && listed(&hub).is_empty()
    })
    .await;

    // Asked for it, the restarted member takes it, and everyone agrees.
    let Check::Took(epoch) = again.app.claim(b"c-1") else {
        panic!("the restarted member takes c-1");
    };
    until("all agree", || {
        b.app.holder_of(b"c-1") == Some(1)
            && listed(&hub) == vec![("c-1".into(), "0000000000000001".into(), epoch.ms)]
    })
    .await;
    assert_eq!(b.app.claim(b"c-1"), Check::Held(1));
}

/// **A process that reconnects still holding its claims keeps them**: nobody
/// releases them on its behalf. The gateway drops every socket; A and B dial
/// again in the same processes. A's claim on `c-1` stands at its epoch at the
/// gateway and at B throughout — never released, so B never had it to take —
/// and A, still its holder, announces it again on its new session.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnect_keeps_its_claims() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let (a, b, epoch_ms) = a_holds_c1(&hub, &up_tx).await;
    let first = hub.member("zend", 1).unwrap().session;
    let held = vec![("c-1".to_string(), "0000000000000001".to_string(), epoch_ms)];

    hub.close_all("a blip");
    until("A back on a new session", || {
        hub.member("zend", 1).is_some_and(|m| m.session != first)
            && a.app.link() == Link::Joined
            && b.app.link() == Link::Joined
    })
    .await;
    assert_eq!(listed(&hub), held, "the gateway kept it");
    assert_eq!(
        a.app.own_claims(),
        vec![(
            b"c-1".to_vec(),
            Epoch {
                ms: epoch_ms,
                node: 1
            }
        )]
    );
    until("B sees its holder again", || {
        b.app.holder_of(b"c-1") == Some(1)
    })
    .await;
    assert_eq!(b.app.claim(b"c-1"), Check::Held(1));
    let released = hub
        .log
        .read(Some("zend"), None, 100)
        .into_iter()
        .any(|e| e.text.starts_with("released"));
    assert!(!released, "nobody let it go");
}

/// **A handshake that fails after its admission was decided changes nothing.**
/// A second connection for node 1 proves itself and is decided — and is gone
/// before its `ADMIT` can be sent. A stays admitted on its session, and its
/// claim stands at the gateway and at B.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_handshake_changes_nothing() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let (a, b, epoch_ms) = a_holds_c1(&hub, &up_tx).await;
    let first = hub.member("zend", 1).unwrap().session;

    let (m, g) = pair(1 << 20);
    let served = {
        let hub = hub.clone();
        tokio::spawn(async move {
            hub.serve("zend", IpAddr::from(IP), g.tx, g.rx, |_| {})
                .await;
        })
    };
    let (mut tx, mut rx) = (m.tx, m.rx);
    let hello = Hello {
        proto: PROTO,
        app: "zend".into(),
        node: 1,
        host: "m1".into(),
        card: 1,
        build: build(10),
        rejected: false,
        model: Some(model(1)),
        power: 101,
        repos: vec![],
        now_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        nonce: nonce(),
    };
    tx.send(Message::Frame(
        Frame::Hello(Box::new(hello.clone())).encode(),
    ))
    .await
    .unwrap();
    let Some(Message::Frame(bytes)) = rx.recv().await else {
        panic!("a CHALLENGE");
    };
    let Ok(Frame::Challenge(challenge)) = Frame::decode(&bytes) else {
        panic!("a CHALLENGE");
    };
    let proof = member_proof(&key(), "zend", 1, &challenge.nonce, &hello.nonce);
    // Gone before the ADMIT: the gateway decides it, and cannot send it.
    drop(rx);
    tx.send(Message::Frame(Frame::Prove(Prove { proof }).encode()))
        .await
        .unwrap();
    timeout(Duration::from_secs(5), served)
        .await
        .expect("the failed handshake ends")
        .unwrap();

    assert_eq!(hub.member("zend", 1).map(|m| m.session), Some(first));
    assert_eq!(a.app.link(), Link::Joined);
    assert_eq!(
        listed(&hub),
        vec![("c-1".into(), "0000000000000001".into(), epoch_ms)]
    );
    assert_eq!(b.app.holder_of(b"c-1"), Some(1));
    assert_eq!(a.app.own_claims().len(), 1);
}

/// A gateway shutting down closes its members `1001`; each counts the ended
/// attempt and dials again (§3.4). The close is logged, not kept: once the
/// member is admitted again its view describes the new socket, which nothing
/// has closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gateway_going_away_closes_1001_and_the_member_redials() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let a = member(&hub, 1, build(10), model(1), &up_tx);
    until("joined", || hub.member("zend", 1).is_some()).await;
    let first = hub.member("zend", 1).unwrap().session;
    let attempts = a.app.clusters()[0].attempts;
    hub.close_all("the gateway is shutting down");
    until("rejoined on a new session", || {
        hub.member("zend", 1).is_some_and(|m| m.session != first)
    })
    .await;
    until("admitted again", || a.app.clusters()[0].state == "admitted").await;
    let view = &a.app.clusters()[0];
    assert!(
        view.attempts > attempts,
        "the ended socket counts an attempt"
    );
    assert_eq!(view.last_close, None, "the new socket was not closed");
}

/// A dropped socket fails every stream riding it: the gateway's reads end in
/// an error, not a clean end (§5.6).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_socket_resets_its_streams() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let mut a = member(&hub, 1, build(10), model(1), &up_tx);
    until("joined", || a.app.link() == Link::Joined).await;
    let mut conn = hub.open("zend", 1, "zend").unwrap();
    let _served = a.down.recv().await.unwrap();
    a.app.close_all("going");
    a.stopped.abort();
    let mut buf = Vec::new();
    let read = timeout(Duration::from_secs(5), conn.read_to_end(&mut buf))
        .await
        .expect("the stream ends");
    assert!(read.is_err(), "a reset, not a clean end: {read:?}");
}

/// The refusals of §3.4 reach the member and the gateway's Refusals block: a
/// model mismatch 4412, a build below the floor 4410 — which stops the app —
/// and a second session of one node replacing the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refusals_are_coded_and_listed() {
    let hub = Hub::new(key(), "gw-test");
    let (up_tx, _up) = mpsc::unbounded_channel();
    let first = member(&hub, 1, build(20), model(1), &up_tx);
    until("first joined", || first.app.link() == Link::Joined).await;

    let other_model = member(&hub, 2, build(20), model(2), &up_tx);
    let low = member(&hub, 3, build(19), model(1), &up_tx);
    let stopped = timeout(Duration::from_secs(5), low.stopped)
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(&stopped, Stopped::App(c) if c.code == 4410),
        "{stopped:?}"
    );
    assert!(matches!(low.app.link(), Link::Stopped(c) if c.code == 4410));
    until("the model refusal listed", || {
        hub.status(10)
            .refusals
            .iter()
            .any(|r| r.code == 4412 && r.node == "0000000000000002")
    })
    .await;
    assert_ne!(other_model.app.link(), Link::Joined);

    // A second process claiming node 1 replaces the first, which is closed
    // 4409 and stops that socket.
    let again = member(&hub, 1, build(20), model(1), &up_tx);
    until("replaced", || again.app.link() == Link::Joined).await;
    let stopped = timeout(Duration::from_secs(5), first.stopped)
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(&stopped, Stopped::Socket(c) if c.code == 4409),
        "{stopped:?}"
    );
    let status = hub.status(10);
    let floor = status.apps.iter().find(|a| a.app == "zend").unwrap();
    assert_eq!(floor.floor.as_ref().map(|b| b.count), Some(20));
    assert_eq!(floor.model_lock.as_deref(), Some("model 1"));
}
