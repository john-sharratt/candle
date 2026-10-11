//! H5 (§14.2): handshake storms. The handshake half — valid, wrong key,
//! another protocol version, and a member that speaks out of turn — runs the
//! bare handshake; the admission half runs the same cases and a build below the
//! floor, another model and a duplicate node against a live gateway hub. Every
//! one ends in its exact code, the gateway sends nothing but `CHALLENGE` before
//! `ADMIT`, and the floor and lock stand as the first member set them. Budget
//! 2 s each.

use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cluster::clock::now_ms;
use cluster::close::Close;
use cluster::frame::{Admit, BuildId, Done, Frame, Hello, Kind, ModelId, RepoAd};
use cluster::gateway::Hub;
use cluster::handshake::{admit, join, GroupKey, HandshakeError, PROTO};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::pipe::{Message, PipeClosed, PipeRx, PipeTx};

const HANDSHAKES: usize = 500;
const BUDGET: Duration = Duration::from_secs(2);

struct Tap {
    inner: MemoryTx,
    kinds: Arc<Mutex<Vec<u8>>>,
}

impl PipeTx for Tap {
    async fn send(&mut self, message: Message) -> Result<(), PipeClosed> {
        if let Message::Frame(bytes) = &message {
            self.kinds.lock().unwrap().push(bytes[0]);
        }
        self.inner.send(message).await
    }

    fn unsent_bytes(&self) -> usize {
        self.inner.unsent_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Valid,
    WrongKey,
    OtherProtocol,
    OutOfTurn,
}

fn hello(node: u64, proto: u16) -> Hello {
    Hello {
        proto,
        app: "zend".into(),
        node,
        host: format!("m{node}"),
        build: BuildId {
            count: 100,
            hash: "abc1234".into(),
            local: String::new(),
        },
        model: None,
        power: 100,
        repos: vec![RepoAd {
            name: "candle".into(),
            origin: "git@h:candle.git".into(),
            push: false,
        }],
        now_ms: 0,
        nonce: [0; 32],
    }
}

fn admission(node: u64) -> Result<Admit, Close> {
    Ok(Admit {
        session: node,
        members: vec![],
        claims: vec![],
        floor_count: 100,
        floor_hash: "abc1234".into(),
        routes_wanted: true,
        own_applied_through: 0,
    })
}

/// One handshake: the code each side ends with (0 for an admission) and the
/// kinds the gateway sent.
async fn one(case: Case, node: u64) -> (u16, u16, Vec<u8>) {
    let gateway_key = GroupKey::from_bytes([1; 32]);
    let member_key = match case {
        Case::WrongKey => GroupKey::from_bytes([2; 32]),
        _ => gateway_key.clone(),
    };
    let (m, g) = pair(1 << 16);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let mut gtx = Tap {
        inner: g.tx,
        kinds: kinds.clone(),
    };
    let mut grx = g.rx;
    let (mut mtx, mut mrx) = (m.tx, m.rx);
    let member = async {
        match case {
            Case::OutOfTurn => {
                let done = Frame::Done(Done {
                    request: 1,
                    reply_to: node,
                });
                mtx.send(Message::Frame(done.encode())).await.unwrap();
                match mrx.recv().await {
                    Some(Message::Close(c)) => c.code,
                    other => panic!("{other:?}"),
                }
            }
            _ => {
                let proto = match case {
                    Case::OtherProtocol => PROTO + 1,
                    _ => PROTO,
                };
                match join(&mut mtx, &mut mrx, &member_key, hello(node, proto)).await {
                    Ok(_) => 0,
                    Err(HandshakeError::Closed(c)) | Err(HandshakeError::Refused(c)) => c.code,
                    Err(e) => panic!("{e}"),
                }
            }
        }
    };
    let gateway = async {
        match admit(
            &mut gtx,
            &mut grx,
            &gateway_key,
            |_| {},
            |h| admission(h.node),
        )
        .await
        {
            Ok(_) => 0,
            Err(HandshakeError::Closed(c)) | Err(HandshakeError::Refused(c)) => c.code,
            Err(e) => panic!("{e}"),
        }
    };
    let (mc, gc) = tokio::join!(member, gateway);
    let sent = kinds.lock().unwrap().clone();
    (mc, gc, sent)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h5_handshake_storm() {
    let started = Instant::now();
    let cases = [
        Case::Valid,
        Case::WrongKey,
        Case::OtherProtocol,
        Case::OutOfTurn,
    ];
    let mut runs = Vec::new();
    for i in 0..HANDSHAKES {
        let case = cases[i % cases.len()];
        runs.push(tokio::spawn(async move {
            (case, one(case, i as u64 + 1).await)
        }));
    }
    let mut admitted = 0;
    for r in runs {
        let (case, (member, gateway, sent)) = r.await.unwrap();
        let (want_member, want_gateway) = match case {
            Case::Valid => (0, 0),
            Case::WrongKey => (4401, 4401),
            Case::OtherProtocol => (4426, 4426),
            Case::OutOfTurn => (4400, 4400),
        };
        assert_eq!((member, gateway), (want_member, want_gateway), "{case:?}");
        // Nothing but the handshake crosses before ADMIT.
        let allowed = [Kind::Challenge.tag(), Kind::Admit.tag()];
        assert!(
            sent.iter().all(|k| allowed.contains(k)),
            "{case:?}: {sent:?}"
        );
        if member == 0 {
            admitted += 1;
            assert_eq!(sent, allowed);
        }
    }
    let elapsed = started.elapsed();
    println!("H5 (handshake half): {HANDSHAKES} handshakes, {admitted} admitted, in {elapsed:?}");
    assert_eq!(admitted, HANDSHAKES / 4);
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
}

// ── the admission half, against a live hub ──────────────────────────────────

const FLOOR: u32 = 100;

fn model(weights: u8) -> ModelId {
    ModelId {
        arch: "Qwen4Exp".into(),
        weights: [weights; 32],
        tokenizer: [1; 32],
        rope: "yarn".into(),
        describe: format!("model {weights}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    Valid,
    WrongKey,
    OtherProtocol,
    OutOfTurn,
    LowBuild,
    WrongModel,
    /// A second session of a node already connected: admitted, and the first
    /// closed 4409.
    Duplicate,
}

/// Connect to `hub` as the member at `ip`, its sends tapped: what the gateway
/// sent before `ADMIT` is in `kinds`.
fn dial(hub: &Arc<Hub>, ip: IpAddr) -> (MemoryTx, MemoryRx, Arc<Mutex<Vec<u8>>>) {
    let (m, g) = pair(1 << 16);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let gtx = Tap {
        inner: g.tx,
        kinds: kinds.clone(),
    };
    let hub = hub.clone();
    tokio::spawn(async move { hub.serve("zend", ip, gtx, g.rx, |_| {}).await });
    (m.tx, m.rx, kinds)
}

fn member_hello(node: u64, count: u32, weights: u8, proto: u16) -> Hello {
    let mut h = hello(node, proto);
    h.build.count = count;
    h.model = Some(model(weights));
    h.now_ms = now_ms();
    h
}

async fn joined(tx: &mut MemoryTx, rx: &mut MemoryRx, key: &GroupKey, h: Hello) -> u16 {
    match join(tx, rx, key, h).await {
        Ok(_) => 0,
        Err(HandshakeError::Closed(c)) | Err(HandshakeError::Refused(c)) => c.code,
        Err(e) => panic!("{e}"),
    }
}

/// What the gateway sent before `ADMIT`, or everything when there was none.
fn before_admit(kinds: &[u8]) -> Vec<u8> {
    kinds
        .iter()
        .take_while(|k| **k != Kind::Admit.tag())
        .copied()
        .collect()
}

async fn admission_case(hub: Arc<Hub>, case: Admission, i: u64) -> (Admission, Vec<u16>) {
    let key = GroupKey::from_bytes([1; 32]);
    let ip = IpAddr::from([10, (i >> 16) as u8, (i >> 8) as u8, i as u8]);
    let node = 1_000 + i;
    let (mut tx, mut rx, kinds) = dial(&hub, ip);
    let code = match case {
        Admission::Valid => {
            joined(&mut tx, &mut rx, &key, member_hello(node, FLOOR, 1, PROTO)).await
        }
        Admission::WrongKey => {
            let other = GroupKey::from_bytes([2; 32]);
            joined(
                &mut tx,
                &mut rx,
                &other,
                member_hello(node, FLOOR, 1, PROTO),
            )
            .await
        }
        Admission::OtherProtocol => {
            joined(
                &mut tx,
                &mut rx,
                &key,
                member_hello(node, FLOOR, 1, PROTO + 1),
            )
            .await
        }
        Admission::OutOfTurn => {
            let done = Frame::Done(Done {
                request: 1,
                reply_to: node,
            });
            tx.send(Message::Frame(done.encode())).await.unwrap();
            match rx.recv().await {
                Some(Message::Close(c)) => c.code,
                other => panic!("{other:?}"),
            }
        }
        Admission::LowBuild => {
            joined(
                &mut tx,
                &mut rx,
                &key,
                member_hello(node, FLOOR - 1, 1, PROTO),
            )
            .await
        }
        Admission::WrongModel => {
            joined(&mut tx, &mut rx, &key, member_hello(node, FLOOR, 2, PROTO)).await
        }
        Admission::Duplicate => {
            let first = joined(&mut tx, &mut rx, &key, member_hello(node, FLOOR, 1, PROTO)).await;
            let (mut tx2, mut rx2, _) = dial(&hub, ip);
            let second = joined(
                &mut tx2,
                &mut rx2,
                &key,
                member_hello(node, FLOOR, 1, PROTO),
            )
            .await;
            // The first session is closed 4409, once frames it was already
            // sent are read past.
            let replaced = loop {
                match rx.recv().await {
                    Some(Message::Close(c)) => break c.code,
                    Some(Message::Frame(_)) => continue,
                    None => panic!("the first session dropped without a close"),
                }
            };
            assert_eq!((first, second), (0, 0), "{case:?}");
            replaced
        }
    };
    let sent = kinds.lock().unwrap().clone();
    let early = before_admit(&sent);
    assert!(
        early.iter().all(|k| *k == Kind::Challenge.tag()),
        "{case:?}: {sent:?}"
    );
    // Valid members stay connected, holding the lock against the wrong model.
    if case == Admission::Valid {
        tokio::spawn(async move {
            let _keep = tx;
            while rx.recv().await.is_some() {}
        });
    }
    (case, vec![code])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h5_admission_storm() {
    let started = Instant::now();
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw-h5");
    // The first member sets the floor and the lock, and stays.
    let (mut tx, mut rx, _) = dial(&hub, IpAddr::from([10, 255, 0, 1]));
    let key = GroupKey::from_bytes([1; 32]);
    assert_eq!(
        joined(&mut tx, &mut rx, &key, member_hello(1, FLOOR, 1, PROTO)).await,
        0
    );
    let cases = [
        Admission::Valid,
        Admission::WrongKey,
        Admission::OtherProtocol,
        Admission::OutOfTurn,
        Admission::LowBuild,
        Admission::WrongModel,
        Admission::Duplicate,
    ];
    let mut runs = Vec::new();
    for i in 0..HANDSHAKES as u64 {
        let case = cases[i as usize % cases.len()];
        runs.push(tokio::spawn(admission_case(hub.clone(), case, i)));
    }
    for r in runs {
        let (case, codes) = r.await.unwrap();
        let want = match case {
            Admission::Valid => 0,
            Admission::WrongKey => 4401,
            Admission::OtherProtocol => 4426,
            Admission::OutOfTurn => 4400,
            Admission::LowBuild => 4410,
            Admission::WrongModel => 4412,
            Admission::Duplicate => 4409,
        };
        assert_eq!(codes, vec![want], "{case:?}");
    }
    let status = hub.status(0);
    let zend = status.apps.iter().find(|a| a.app == "zend").unwrap();
    assert_eq!(zend.floor.as_ref().map(|b| b.count), Some(FLOOR));
    assert_eq!(zend.model_lock.as_deref(), Some("model 1"));
    let refused: Vec<u16> = status.refusals.iter().map(|r| r.code).collect();
    for code in [4401, 4426, 4400, 4410, 4412] {
        assert!(refused.contains(&code), "{code} not listed in {refused:?}");
    }
    let elapsed = started.elapsed();
    println!("H5 (admission half): {HANDSHAKES} handshakes in {elapsed:?}");
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
    drop((tx, rx));
}
