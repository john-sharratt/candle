//! `LOST` (§7.2): a requester is told when its catch-up or fetch will get no
//! answer, rather than left to a silence that a reply still arriving over a
//! slow link also gives. Against a live gateway hub, with raw member sockets.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use applied_set::AppliedSet;
use cluster::clock::now_ms;
use cluster::frame::{BuildId, Catchup, Done, Fetch, Frame, Hello, Lost, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::{join, GroupKey, PROTO};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::pipe::{Message, PipeRx, PipeTx};
use tokio::time::timeout;

fn key() -> GroupKey {
    GroupKey::from_bytes([1; 32])
}

fn hello(node: u64) -> Hello {
    Hello {
        proto: PROTO,
        app: "zend".into(),
        node,
        host: format!("m{node}"),
        build: BuildId {
            count: 100,
            hash: "abc1234".into(),
            local: String::new(),
        },
        rejected: false,
        model: Some(ModelId {
            arch: "qwen3".into(),
            weights: [1; 32],
            tokenizer: [2; 32],
            rope: "base".into(),
            describe: "test".into(),
        }),
        power: 100,
        repos: vec![],
        now_ms: now_ms(),
        nonce: [0; 32],
    }
}

/// A member `node` joined to `hub` over a raw pipe.
async fn member(hub: &Arc<Hub>, node: u64) -> (MemoryTx, MemoryRx) {
    let (m, g) = pair(1 << 16);
    let hub = hub.clone();
    tokio::spawn(async move {
        hub.serve(
            "zend",
            IpAddr::from([10, 0, 0, node as u8]),
            g.tx,
            g.rx,
            |_| {},
        )
        .await
    });
    let (mut tx, mut rx) = (m.tx, m.rx);
    join(&mut tx, &mut rx, &key(), hello(node))
        .await
        .expect("admitted");
    (tx, rx)
}

async fn send(tx: &mut MemoryTx, frame: Frame) {
    tx.send(Message::Frame(frame.encode())).await.unwrap();
}

/// The first frame `rx` receives that `want` picks, within 5 s.
async fn first<T>(rx: &mut MemoryRx, mut want: impl FnMut(Frame) -> Option<T>) -> T {
    timeout(Duration::from_secs(5), async {
        loop {
            match rx.recv().await {
                Some(Message::Frame(bytes)) => {
                    if let Some(t) = want(Frame::decode(&bytes).unwrap()) {
                        return t;
                    }
                }
                other => panic!("the socket ended: {other:?}"),
            }
        }
    })
    .await
    .expect("in time")
}

/// The member answering a catch-up leaves before its `DONE`: the requester
/// is told the request is lost.
#[tokio::test]
async fn an_answerer_leaving_loses_its_open_catch_up() {
    let hub = Hub::new(key(), "gw-lost");
    let (mut a_tx, mut a_rx) = member(&hub, 1).await;
    let (b_tx, mut b_rx) = member(&hub, 2).await;

    send(
        &mut a_tx,
        Frame::Catchup(Catchup {
            request: 7,
            reply_to: 1,
            to: Some(2),
            since: AppliedSet::default(),
        }),
    )
    .await;
    let asked = first(&mut b_rx, |f| match f {
        Frame::Catchup(c) => Some(c.request),
        _ => None,
    })
    .await;
    assert_eq!(asked, 7, "the answerer was asked");

    drop((b_tx, b_rx));
    let lost = first(&mut a_rx, |f| match f {
        Frame::Lost(l) => Some(l),
        _ => None,
    })
    .await;
    assert_eq!(
        lost,
        Lost {
            request: 7,
            reply_to: 1
        }
    );
}

/// A fetch addressed to a member that is not here is lost at once.
#[tokio::test]
async fn a_fetch_from_a_member_not_here_is_lost_at_once() {
    let hub = Hub::new(key(), "gw-lost");
    let (mut a_tx, mut a_rx) = member(&hub, 1).await;
    send(
        &mut a_tx,
        Frame::Fetch(Fetch {
            request: 9,
            reply_to: 1,
            from_node: 42,
            origin: 42,
            oseqs: vec![1, 2],
        }),
    )
    .await;
    let lost = first(&mut a_rx, |f| match f {
        Frame::Lost(l) => Some(l),
        _ => None,
    })
    .await;
    assert_eq!(
        lost,
        Lost {
            request: 9,
            reply_to: 1
        }
    );
}

/// An answered request is not lost when its answerer leaves afterwards.
#[tokio::test]
async fn an_answered_request_is_not_lost_when_its_answerer_leaves() {
    let hub = Hub::new(key(), "gw-lost");
    let (mut a_tx, mut a_rx) = member(&hub, 1).await;
    let (mut b_tx, mut b_rx) = member(&hub, 2).await;
    send(
        &mut a_tx,
        Frame::Catchup(Catchup {
            request: 7,
            reply_to: 1,
            to: Some(2),
            since: AppliedSet::default(),
        }),
    )
    .await;
    first(&mut b_rx, |f| matches!(f, Frame::Catchup(_)).then_some(())).await;
    send(
        &mut b_tx,
        Frame::Done(Done {
            request: 7,
            reply_to: 1,
        }),
    )
    .await;
    first(&mut a_rx, |f| matches!(f, Frame::Done(_)).then_some(())).await;

    drop((b_tx, b_rx));
    // Nothing lost arrives: within a second only other traffic, if any.
    let lost = timeout(Duration::from_secs(1), async {
        loop {
            match a_rx.recv().await {
                Some(Message::Frame(bytes)) => {
                    if let Frame::Lost(l) = Frame::decode(&bytes).unwrap() {
                        return l;
                    }
                }
                _ => std::future::pending::<()>().await,
            }
        }
    })
    .await;
    assert!(lost.is_err(), "no LOST for an answered request: {lost:?}");
}
