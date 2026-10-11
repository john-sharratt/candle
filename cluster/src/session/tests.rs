//! Sessions end to end over an in-memory pipe.

use std::io::ErrorKind;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;

use super::*;
use crate::close::{Close, CloseCode};
use crate::frame::{BuildId, Frame, Hello, Open, Receipt};
use crate::pipe::memory::{pair, End, MemoryRx, MemoryTx};
use crate::pipe::Message;

fn sessions(cap: usize) -> (Session, Session) {
    let (g, m) = pair(cap);
    (
        start(Role::Gateway, g.tx, g.rx),
        start(Role::Member, m.tx, m.rx),
    )
}

/// One session, with a raw pipe end standing in for the peer.
fn against_raw(role: Role) -> (Session, MemoryTx, MemoryRx) {
    let (ours, End { tx, rx }) = pair(1 << 20);
    (start(role, ours.tx, ours.rx), tx, rx)
}

async fn accept(s: &mut Session) -> VirtualConn {
    loop {
        match s.incoming.recv().await.expect("session open") {
            Incoming::Stream(c) => return c,
            _ => continue,
        }
    }
}

#[tokio::test]
async fn a_stream_carries_bytes_both_ways_and_ends_with_fin() {
    let (gateway, mut member) = sessions(1 << 20);
    let mut out = gateway.handle.open("").unwrap();
    assert_eq!(out.id() % 2, 1, "the gateway opens odd streams");
    out.write_all(b"GET / HTTP/1.1\r\n\r\n").await.unwrap();
    out.shutdown().await.unwrap();

    let mut inbound = accept(&mut member).await;
    assert_eq!(inbound.id(), out.id());
    let mut request = Vec::new();
    inbound.read_to_end(&mut request).await.unwrap();
    assert_eq!(request, b"GET / HTTP/1.1\r\n\r\n");
    inbound.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await.unwrap();
    inbound.shutdown().await.unwrap();

    let mut response = Vec::new();
    out.read_to_end(&mut response).await.unwrap();
    assert_eq!(response, b"HTTP/1.1 200 OK\r\n\r\n");
}

#[tokio::test]
async fn a_member_opens_even_streams_with_a_target() {
    let (mut gateway, member) = sessions(1 << 20);
    let out = member.handle.open("agent").unwrap();
    assert_eq!(out.id(), 2);
    let inbound = accept(&mut gateway).await;
    assert_eq!((inbound.id(), inbound.target()), (2, "agent"));
}

/// A reader that does not read stalls its own writer at 1 MiB; reading returns
/// credit and the writer finishes. Another stream is not held up meanwhile.
#[tokio::test]
async fn a_slow_reader_stalls_only_its_own_stream() {
    let (gateway, mut member) = sessions(8 << 20);
    let mut slow = gateway.handle.open("").unwrap();
    let mut slow_in = accept(&mut member).await;
    let payload = vec![7u8; INITIAL_CREDIT + 300_000];
    let writing = tokio::spawn(async move {
        slow.write_all(&payload).await.unwrap();
        slow.shutdown().await.unwrap();
        slow
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!writing.is_finished(), "stalled at its credit");

    let mut quick = gateway.handle.open("").unwrap();
    let mut quick_in = accept(&mut member).await;
    quick.write_all(b"through").await.unwrap();
    quick.shutdown().await.unwrap();
    let mut got = Vec::new();
    quick_in.read_to_end(&mut got).await.unwrap();
    assert_eq!(got, b"through");

    let mut all = Vec::new();
    slow_in.read_to_end(&mut all).await.unwrap();
    assert_eq!(all.len(), INITIAL_CREDIT + 300_000);
    writing.await.unwrap();
}

#[tokio::test]
async fn dropping_a_connection_unfinished_resets_the_peer() {
    let (gateway, mut member) = sessions(1 << 20);
    let mut out = gateway.handle.open("").unwrap();
    out.write_all(b"half").await.unwrap();
    let mut inbound = accept(&mut member).await;
    drop(out);
    let mut buf = Vec::new();
    let err = inbound.read_to_end(&mut buf).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConnectionReset);
    assert_eq!(buf, b"half");
}

/// Events arrive whole and in the order sent, and each resolves to the frame
/// number of its last fragment.
#[tokio::test]
async fn events_arrive_whole_in_order_with_their_frame_numbers() {
    let (mut gateway, member) = sessions(4 << 20);
    let big: Vec<u8> = (0..(3 << 20) + 5).map(|i| i as u8).collect();
    let a = member.handle.event(9, 1, &big, None, false).unwrap();
    let b = member.handle.event(9, 2, b"small", None, false).unwrap();
    let c = member
        .handle
        .event(9, 3, b"fetched", Some(44), true)
        .unwrap();
    let mut got = Vec::new();
    while got.len() < 3 {
        if let Incoming::Event { catchup, whole } = gateway.incoming.recv().await.unwrap() {
            got.push((whole.oseq, whole.bytes.len(), catchup, whole.reply_to));
            if whole.oseq == 1 {
                assert_eq!(whole.bytes, big);
            }
        }
    }
    assert_eq!(
        got,
        vec![
            (1, big.len(), false, None),
            (2, 5, false, None),
            (3, 7, true, Some(44)),
        ]
    );
    assert_eq!(
        (a.await.unwrap(), b.await.unwrap(), c.await.unwrap()),
        (4, 5, 6)
    );
}

#[tokio::test]
async fn control_and_catchup_frames_reach_the_owner() {
    let (mut gateway, member) = sessions(1 << 20);
    member
        .handle
        .control(Frame::Receipt(Receipt { through: 3 }))
        .unwrap();
    let Incoming::Frame(Frame::Receipt(r)) = gateway.incoming.recv().await.unwrap() else {
        panic!()
    };
    assert_eq!(r.through, 3);
}

/// Pings keep an idle session alive past 45 s.
#[tokio::test(start_paused = true)]
async fn an_idle_session_stays_alive_on_its_pings() {
    let (gateway, mut member) = sessions(1 << 20);
    tokio::time::sleep(Duration::from_secs(120)).await;
    assert!(member.ended.try_recv().is_err());
    let out = gateway.handle.open("").unwrap();
    assert_eq!(accept(&mut member).await.id(), out.id());
}

/// A peer that sends nothing for 45 s is dead: the session closes 1001.
#[tokio::test(start_paused = true)]
async fn a_silent_peer_is_dead_after_45_s() {
    let (s, _tx, mut rx) = against_raw(Role::Gateway);
    let started = tokio::time::Instant::now();
    assert_eq!(s.ended.await.unwrap(), Ended::Dead);
    assert_eq!(started.elapsed(), DEAD_AFTER);
    // Its pings went out every 15 s while it waited, then the close.
    let mut seen = Vec::new();
    while let Some(m) = rx.recv().await {
        seen.push(m);
    }
    let Some((Message::Close(close), pings)) = seen.split_last() else {
        panic!("{seen:?}")
    };
    assert_eq!(close.code, 1001);
    assert!(pings.len() >= 2);
    for m in pings {
        let Message::Frame(bytes) = m else { panic!() };
        assert!(matches!(Frame::decode(bytes).unwrap(), Frame::Ping(_)));
    }
}

#[tokio::test]
async fn a_handshake_frame_after_admission_is_closed_4400() {
    let (s, mut tx, mut rx) = against_raw(Role::Gateway);
    let hello = Frame::Hello(Box::new(Hello {
        proto: 1,
        app: "zend".into(),
        node: 1,
        host: String::new(),
        build: BuildId {
            count: 1,
            hash: String::new(),
            local: String::new(),
        },
        model: None,
        power: 0,
        repos: vec![],
        now_ms: 0,
        nonce: [0; 32],
    }));
    tx.send(Message::Frame(hello.encode())).await.unwrap();
    let Ended::Refused(close) = s.ended.await.unwrap() else {
        panic!()
    };
    assert_eq!(close.code, CloseCode::Malformed.code());
    assert_eq!(close.reason, "HELLO after admission");
    assert_eq!(rx.recv().await, Some(Message::Close(close)));
}

#[tokio::test]
async fn garbage_and_a_stream_opened_out_of_turn_are_closed_4400() {
    let (s, mut tx, _rx) = against_raw(Role::Gateway);
    tx.send(Message::Frame(vec![200, 0, 0, 0, 0, 0]))
        .await
        .unwrap();
    assert!(matches!(s.ended.await.unwrap(), Ended::Refused(c) if c.code == 4400));

    // The member side may not open an odd stream: those are the gateway's.
    let (s, mut tx, _rx) = against_raw(Role::Gateway);
    let open = Frame::Open(Open {
        stream: 3,
        target: String::new(),
    });
    tx.send(Message::Frame(open.encode())).await.unwrap();
    let Ended::Refused(close) = s.ended.await.unwrap() else {
        panic!()
    };
    assert_eq!(close.reason, "stream 3 opened out of turn");
}

/// A session that ends resets every open stream.
#[tokio::test]
async fn a_session_that_ends_resets_its_streams() {
    let (gateway, mut member) = sessions(1 << 20);
    let mut out = gateway.handle.open("").unwrap();
    let _inbound = accept(&mut member).await;
    gateway
        .handle
        .close(Close::new(CloseCode::GoingAway, "shutting down"));
    let mut buf = [0u8; 4];
    let err = timeout(Duration::from_secs(5), out.read(&mut buf))
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConnectionReset);
    assert!(matches!(member.ended.await.unwrap(), Ended::Closed(c) if c.code == 1001));
}
