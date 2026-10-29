//! The handshake end to end over an in-memory pipe.

use super::*;
use crate::close::CloseCode;
use crate::frame::{Admit, BuildId, Done, Frame, Hello, Prove};
use crate::pipe::memory::{pair, MemoryRx, MemoryTx};

fn key() -> GroupKey {
    GroupKey::from_bytes([7; 32])
}

/// A member's sending and receiving halves, then the gateway's.
fn ends() -> (MemoryTx, MemoryRx, MemoryTx, MemoryRx) {
    let (m, g) = pair(1 << 20);
    (m.tx, m.rx, g.tx, g.rx)
}

fn hello(node: u64) -> Hello {
    Hello {
        proto: PROTO,
        app: "zend".into(),
        node,
        host: ".5".into(),
        card: node,
        build: BuildId {
            count: 10,
            hash: "abc".into(),
            local: String::new(),
        },
        rejected: false,
        model: None,
        power: 1,
        repos: vec![],
        now_ms: 0,
        nonce: [0; 32],
    }
}

fn admission(session: u64) -> Admit {
    Admit {
        session,
        members: vec![],
        claims: vec![],
        floor_count: 10,
        floor_hash: "abc".into(),
        routes_wanted: true,
    }
}

async fn run(
    member_key: GroupKey,
    gateway_key: GroupKey,
    hello: Hello,
    decide: impl FnOnce(&Hello) -> Result<Admit, Close>,
) -> (
    Result<Admit, HandshakeError>,
    Result<(Hello, Admit), HandshakeError>,
) {
    let (mut mtx, mut mrx, mut gtx, mut grx) = ends();
    tokio::join!(
        join(&mut mtx, &mut mrx, &member_key, hello),
        admit(&mut gtx, &mut grx, &gateway_key, |_| {}, decide)
    )
}

/// The `HELLO` is shown as it arrives, before its proof is checked — so a
/// refused member can be named.
#[tokio::test]
async fn the_hello_is_seen_before_the_proof() {
    let (mut mtx, mut mrx, mut gtx, mut grx) = ends();
    let mut seen = None;
    let other = GroupKey::from_bytes([8; 32]);
    let k = key();
    let _ = tokio::join!(
        join(&mut mtx, &mut mrx, &other, hello(5)),
        admit(
            &mut gtx,
            &mut grx,
            &k,
            |h| seen = Some((h.node, h.host.clone())),
            |_| panic!("never admitted")
        )
    );
    assert_eq!(seen, Some((5, ".5".to_string())));
}

#[tokio::test]
async fn a_member_with_the_key_is_admitted() {
    let (m, g) = run(key(), key(), hello(5), |h| {
        assert_eq!(h.node, 5);
        Ok(admission(42))
    })
    .await;
    assert_eq!(m.unwrap().session, 42);
    let (h, a) = g.unwrap();
    assert_eq!((h.node, a.session), (5, 42));
}

#[tokio::test]
async fn a_member_with_another_key_is_closed_4401() {
    let (m, g) = run(GroupKey::from_bytes([8; 32]), key(), hello(5), |_| {
        panic!("a member that failed its proof is never considered for admission")
    })
    .await;
    // The member finds the gateway's proof wrong first, and closes it.
    let Err(HandshakeError::Refused(close)) = m else {
        panic!("{m:?}")
    };
    assert_eq!(close.code, 4401);
    let Err(HandshakeError::Closed(seen)) = g else {
        panic!("{g:?}")
    };
    assert_eq!(seen.code, 4401);
}

#[tokio::test]
async fn a_forged_member_proof_is_closed_4401_by_the_gateway() {
    let (mut mtx, mut mrx, mut gtx, mut grx) = ends();
    let member = async {
        // Says HELLO properly, then proves with garbage.
        send(&mut mtx, &Frame::Hello(Box::new(hello(5))))
            .await
            .unwrap();
        let Frame::Challenge(_) = next(&mut mtx, &mut mrx).await.unwrap() else {
            panic!()
        };
        send(&mut mtx, &Frame::Prove(Prove { proof: [0; 32] }))
            .await
            .unwrap();
        next(&mut mtx, &mut mrx).await
    };
    let k = key();
    let (m, g) = tokio::join!(
        member,
        admit(&mut gtx, &mut grx, &k, |_| {}, |_| panic!("never admitted"))
    );
    assert!(matches!(m, Err(HandshakeError::Closed(c)) if c.code == 4401));
    assert!(matches!(g, Err(HandshakeError::Refused(c)) if c.code == 4401));
}

#[tokio::test]
async fn another_protocol_version_is_closed_4426() {
    // A member a build behind the gateway: the case a cut-over meets.
    let older = PROTO - 1;
    let mut h = hello(5);
    h.proto = older;
    let (m, g) = run(key(), key(), h, |_| panic!("never admitted")).await;
    let Err(HandshakeError::Closed(close)) = m else {
        panic!("{m:?}")
    };
    assert_eq!(close.code, 4426);
    assert_eq!(
        close.reason,
        format!("protocol {older} is not supported; this gateway speaks {PROTO}")
    );
    assert!(matches!(g, Err(HandshakeError::Refused(c)) if c.code == 4426));
}

#[tokio::test]
async fn a_refused_admission_carries_its_code_and_reason() {
    let (m, _) = run(key(), key(), hello(5), |_| {
        Err(Close::new(
            CloseCode::Model,
            "locked to Flash, not AntiLoop",
        ))
    })
    .await;
    let Err(HandshakeError::Closed(close)) = m else {
        panic!("{m:?}")
    };
    assert_eq!(close.code, 4412);
    assert_eq!(close.reason, "locked to Flash, not AntiLoop");
}

/// Nothing but handshake frames crosses before `ADMIT`: a member that sends
/// anything else first, or instead of its proof, is closed 4400.
#[tokio::test]
async fn a_non_handshake_frame_before_admit_is_closed_4400() {
    let done = Frame::Done(Done {
        request: 1,
        reply_to: 2,
    });
    for after_hello in [false, true] {
        let (mut mtx, mut mrx, mut gtx, mut grx) = ends();
        let member = async {
            if after_hello {
                send(&mut mtx, &Frame::Hello(Box::new(hello(5))))
                    .await
                    .unwrap();
                next(&mut mtx, &mut mrx).await.unwrap();
            }
            send(&mut mtx, &done).await.unwrap();
            next(&mut mtx, &mut mrx).await
        };
        let k = key();
        let (m, g) = tokio::join!(
            member,
            admit(&mut gtx, &mut grx, &k, |_| {}, |_| panic!("never admitted"))
        );
        assert!(matches!(m, Err(HandshakeError::Closed(c)) if c.code == 4400));
        let Err(HandshakeError::Refused(close)) = g else {
            panic!("{g:?}")
        };
        let wanted = if after_hello { "PROVE" } else { "HELLO" };
        assert_eq!(
            close.reason,
            format!("expected {wanted}, got DONE during the handshake")
        );
    }
}

/// A gateway that sends anything but `CHALLENGE`, then anything but `ADMIT`,
/// is closed 4400 by the member.
#[tokio::test]
async fn a_gateway_out_of_order_is_closed_4400() {
    let (mut mtx, mut mrx, mut gtx, mut grx) = ends();
    let gateway = async {
        next(&mut gtx, &mut grx).await.unwrap();
        send(&mut gtx, &Frame::Ping([0; 8])).await.unwrap();
        next(&mut gtx, &mut grx).await
    };
    let k = key();
    let (m, g) = tokio::join!(join(&mut mtx, &mut mrx, &k, hello(5)), gateway);
    assert!(matches!(m, Err(HandshakeError::Refused(c)) if c.code == 4400));
    assert!(matches!(g, Err(HandshakeError::Closed(c)) if c.code == 4400));
}

#[tokio::test(start_paused = true)]
async fn a_silent_peer_is_closed_after_45_s() {
    let (mut mtx, mut mrx, _gtx, _grx) = ends();
    let started = tokio::time::Instant::now();
    let m = join(&mut mtx, &mut mrx, &key(), hello(5)).await;
    assert!(matches!(m, Err(HandshakeError::Refused(c)) if c.code == 4400));
    assert_eq!(started.elapsed(), HANDSHAKE_IDLE);
}
