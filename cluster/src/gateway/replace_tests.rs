//! Replacing a node's session in one step (`docs/replicated_substrate.md`
//! §5.4): the admission that builds a session's `ADMIT` marks the session it
//! replaces under the same guard, what that session still reports of the
//! node's state lands nowhere, and a handshake that fails undoes it all.

use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::time::timeout;

use super::hub::{take_pending, undo_pending, Hub, Socket, Taken, PENDING_CAP};
use super::registry::MemberView;
use crate::close::{AfterClose, CloseCode};
use crate::delivery::ReceiptPacer;
use crate::frame::{
    BuildId, Capacity, Claim, Complete, Epoch, Frame, Hello, MemberState, ModelId, Prove, Unclaim,
};
use crate::handshake::{member_proof, nonce, GroupKey, PROTO};
use crate::pipe::memory::pair;
use crate::pipe::{Message, PipeRx, PipeTx};
use crate::session::{start, Role, SessionHandle};

fn hello(node: u64) -> Hello {
    Hello {
        proto: PROTO,
        app: "zend".into(),
        node,
        host: format!("m{node}"),
        card: node,
        build: BuildId {
            count: 10,
            hash: "h10".into(),
            local: String::new(),
        },
        rejected: false,
        model: Some(ModelId {
            arch: "Qwen4Exp".into(),
            weights: [1; 32],
            tokenizer: [1; 32],
            rope: "yarn".into(),
            describe: "model 1".into(),
        }),
        power: 100,
        repos: vec![],
        now_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        nonce: [0; 32],
    }
}

/// A session handle over a memory pipe nobody reads.
fn handle() -> SessionHandle {
    let (_member, gateway) = pair(1 << 16);
    start(Role::Gateway, gateway.tx, gateway.rx).handle
}

fn member(node: u64, session: u64) -> MemberView {
    MemberView {
        node,
        host: format!("m{node}"),
        card: node,
        app: "zend".into(),
        build: hello(node).build,
        weight: 100,
        session,
        repos: Vec::new(),
        capacity: None,
        routes: None,
        high_water: 0,
        applied: Vec::new(),
        lagging: false,
        syncing: false,
        silent: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
        closing: false,
    }
}

/// Node 1 admitted on a session whose socket stands, as `run` leaves it; its
/// replaced flag.
fn admitted_once(hub: &Hub) -> (u64, Arc<AtomicBool>) {
    let (_, first) = hub.decide("zend", &hello(1)).unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let mut s = hub.state();
    assert!(matches!(
        take_pending(&mut s, "zend", 1, &first),
        Taken::Ready(_)
    ));
    s.sockets.insert(
        ("zend".into(), 1),
        Socket {
            session: first.session,
            handle: handle(),
            replaced: Arc::clone(&flag),
            connected_ms: 0,
            clock_offset_ms: 0,
            coord: None,
            pacer: ReceiptPacer::new(Instant::now()),
            capacity_at: Instant::now(),
            skipped: 0,
            dropped: 0,
            pulses_dropped: 0,
        },
    );
    s.registry.join(member(1, first.session));
    (first.session, flag)
}

fn ready() -> Frame {
    Frame::Capacity(Capacity {
        state: MemberState::Ready,
        ..Capacity::default()
    })
}

/// **A replaced session's queued `CAPACITY { Ready }` does not mark the new,
/// still-loading session ready**, nor does its late `CLAIM` land: the session
/// is marked replaced under the guard the new `ADMIT` was built under, before
/// the new socket is in place. A one-time reply is still taken.
#[tokio::test]
async fn a_replaced_session_reports_nothing_of_its_node() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (_, old) = admitted_once(&hub);
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    assert!(
        old.load(Ordering::SeqCst),
        "marked as the new ADMIT is built"
    );
    hub.state().registry.join(member(1, second.session));

    let reply_on = handle();
    hub.on_frame("zend", 1, &old, ready(), &reply_on).unwrap();
    assert_eq!(
        hub.state().registry.member("zend", 1).unwrap().state(),
        MemberState::Loading,
        "the old Ready is not the new session's"
    );
    hub.on_frame("zend", 1, &old, claim(b"k", 5), &reply_on)
        .unwrap();
    assert_eq!(hub.state().registry.claims("zend").holder(b"k"), None);

    let current = AtomicBool::new(false);
    hub.on_frame("zend", 1, &current, ready(), &reply_on)
        .unwrap();
    assert_eq!(
        hub.state().registry.member("zend", 1).unwrap().state(),
        MemberState::Ready,
        "the current session's Ready is"
    );
}

/// **A handshake that fails after its admission undoes the replacement, and
/// loses nothing the standing session said meanwhile**: its `CAPACITY` and
/// `CLAIM`, held back while the replacement was pending, are applied under the
/// very guard that undoes it — before anything the session sends after — and
/// it reports again.
#[tokio::test]
async fn a_failed_handshake_undoes_the_replacement() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (first, old) = admitted_once(&hub);
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    assert!(old.load(Ordering::SeqCst));
    let reply_on = handle();
    hub.on_frame("zend", 1, &old, ready(), &reply_on).unwrap();
    hub.on_frame("zend", 1, &old, claim(b"k", 5), &reply_on)
        .unwrap();
    assert_eq!(hub.state().registry.claims("zend").holder(b"k"), None);

    let applied = {
        let mut s = hub.state();
        let applied = undo_pending(&mut s, "zend", 1, &second);
        assert_eq!(s.sockets[&("zend".to_string(), 1)].session, first);
        assert_eq!(
            s.registry.member("zend", 1).unwrap().state(),
            MemberState::Ready,
            "applied under the guard that undid it"
        );
        assert_eq!(
            s.registry.claims("zend").holder(b"k"),
            Some(1),
            "its claim from the window stands"
        );
        applied
    };
    assert_eq!(applied.len(), 2, "what it said while pending");
    assert!(
        !old.load(Ordering::SeqCst),
        "the standing session reports again"
    );
    hub.follow("zend", 1, applied);
}

/// **Held reports land in the order they arrived**: a key's `UNCLAIM`, then
/// its `COMPLETE`, both sent while a replacement was pending, are applied in
/// that order when the handshake fails — so the release's handover stands,
/// holding the standing member complete, rather than the release landing
/// after and starting it over with no member complete.
#[tokio::test]
async fn held_reports_land_in_the_order_they_arrived() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (_, old) = admitted_once(&hub);
    hub.state()
        .registry
        .claims("zend")
        .claim(b"k", Epoch { ms: 5, node: 1 });
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    let reply_on = handle();
    let release = Frame::Unclaim(Unclaim {
        key: b"k".to_vec(),
        epoch: Epoch { ms: 5, node: 1 },
        through: 30,
    });
    hub.on_frame("zend", 1, &old, release, &reply_on).unwrap();
    hub.on_frame(
        "zend",
        1,
        &old,
        Frame::Complete(Complete {
            keys: vec![b"k".to_vec()],
        }),
        &reply_on,
    )
    .unwrap();
    let mut s = hub.state();
    undo_pending(&mut s, "zend", 1, &second);
    assert_eq!(s.registry.claims("zend").holder(b"k"), None, "released");
    assert!(
        s.registry.holds_release("zend", b"k", 1),
        "the COMPLETE after it stands on the release"
    );
}

/// **A standing session that has left takes its held reports with it**: a
/// `CLAIM` it sent while a replacement was pending is not applied once the
/// handshake fails, for its node is no member here.
#[tokio::test]
async fn a_departed_sessions_held_reports_go_with_it() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (first, old) = admitted_once(&hub);
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    hub.on_frame("zend", 1, &old, claim(b"k", 5), &handle())
        .unwrap();
    let mut s = hub.state();
    s.sockets.remove(&("zend".to_string(), 1));
    assert!(s.registry.leave("zend", 1, first));
    assert!(undo_pending(&mut s, "zend", 1, &second).is_empty());
    assert_eq!(s.registry.claims("zend").holder(b"k"), None);
}

fn claim(key: &[u8], ms: u64) -> Frame {
    Frame::Claim(Claim {
        key: key.to_vec(),
        epoch: Epoch { ms, node: 1 },
    })
}

/// **Overlapping dials from one node converge**: while S1's admission is
/// pending, S2's `HELLO` is refused `4429` to dial again — it does not unseat
/// S1, so the two cannot unseat each other in turn. S1 is taken, or fails and
/// is undone; either way S2's next dial is admitted.
#[tokio::test]
async fn overlapping_dials_converge() {
    for s1_fails in [false, true] {
        let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
        let (_, old) = admitted_once(&hub);
        let (_, s1) = hub.decide("zend", &hello(1)).unwrap();
        let refused = hub.decide("zend", &hello(1)).unwrap_err();
        assert_eq!(refused.code, CloseCode::Retry.code());
        assert_eq!(refused.after(), AfterClose::Backoff, "it dials again");
        {
            let mut s = hub.state();
            if s1_fails {
                undo_pending(&mut s, "zend", 1, &s1);
                assert!(!old.load(Ordering::SeqCst));
            } else {
                assert!(matches!(
                    take_pending(&mut s, "zend", 1, &s1),
                    Taken::Ready(_)
                ));
            }
        }
        let (_, s2) = hub.decide("zend", &hello(1)).unwrap();
        assert!(s2.session > s1.session, "S2's next dial is admitted");
    }
}

/// Drive `hub`'s serve through a handshake for `hello` that its admission
/// decides — and then fails, the member gone before its `ADMIT`: the serve
/// task ends before `run` takes the admission's pending.
async fn decided_then_gone(hub: &Arc<Hub>, hello: Hello) {
    let (m, g) = pair(1 << 16);
    let served = {
        let hub = Arc::clone(hub);
        tokio::spawn(async move {
            hub.serve("zend", IpAddr::from([10, 0, 0, 7]), g.tx, g.rx, |_| {})
                .await;
        })
    };
    let (mut tx, mut rx) = (m.tx, m.rx);
    let hello = Hello {
        nonce: nonce(),
        ..hello
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
    let proof = member_proof(
        &GroupKey::from_bytes([1; 32]),
        "zend",
        hello.node,
        &challenge.nonce,
        &hello.nonce,
    );
    drop(rx);
    tx.send(Message::Frame(Frame::Prove(Prove { proof }).encode()))
        .await
        .unwrap();
    timeout(Duration::from_secs(5), served)
        .await
        .expect("the serve task ends")
        .unwrap();
}

/// **A serve task that ends before `run` takes its pending undoes it**,
/// through the guard armed when the admission was decided: the standing
/// session reports again, and the node's next `HELLO` is admitted rather than
/// refused for good.
#[tokio::test]
async fn a_serve_ending_before_run_undoes_its_admission() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (first, old) = admitted_once(&hub);
    decided_then_gone(&hub, hello(1)).await;
    assert!(!old.load(Ordering::SeqCst), "undone");
    assert_eq!(hub.state().sockets[&("zend".to_string(), 1)].session, first);
    assert!(
        hub.decide("zend", &hello(1)).is_ok(),
        "the next HELLO is admitted"
    );
}

/// **A redial that fails leaves a group of one its model lock**: the session
/// it would have replaced is the node's again, so a member of another model
/// is still refused `4412`.
#[tokio::test]
async fn a_failed_redial_keeps_the_model_lock() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    admitted_once(&hub);
    decided_then_gone(&hub, hello(1)).await;
    let mut other = hello(2);
    other.model = Some(ModelId {
        weights: [2; 32],
        describe: "model 2".into(),
        ..other.model.clone().unwrap()
    });
    let refused = hub.decide("zend", &other).unwrap_err();
    assert_eq!(refused.code, CloseCode::Model.code());
}

/// **A handshake during which more claim relays pile up than it can be
/// brought up to date with fails its admission**; what the standing session
/// reported meanwhile — however much, kept to what each kind last said — is
/// applied, and it reports again.
#[tokio::test]
async fn an_overflowing_handshake_fails_its_admission() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (_, old) = admitted_once(&hub);
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    let reply_on = handle();
    for _ in 0..=PENDING_CAP {
        hub.on_frame("zend", 1, &old, ready(), &reply_on).unwrap();
    }
    for ms in 0..=PENDING_CAP as u64 {
        hub.to_others(
            "zend",
            2,
            Frame::Claim(Claim {
                key: b"k2".to_vec(),
                epoch: Epoch { ms, node: 2 },
            }),
        );
    }
    let mut s = hub.state();
    let Taken::Overflowed(applied) = take_pending(&mut s, "zend", 1, &second) else {
        panic!("overflowed");
    };
    assert_eq!(applied.len(), 1, "the thousands of CAPACITYs, as the last");
    assert_eq!(
        s.registry.member("zend", 1).unwrap().state(),
        MemberState::Ready
    );
    assert!(
        !old.load(Ordering::SeqCst),
        "the standing session reports again"
    );
}
