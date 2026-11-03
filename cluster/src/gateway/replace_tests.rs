//! Replacing a node's session in one step (`docs/replicated_substrate.md`
//! §5.4): the admission that builds a session's `ADMIT` marks the session it
//! replaces under the same guard, what that session still reports of the
//! node's state lands nowhere, and a handshake that fails undoes it all.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::hub::{undo_pending, Hub, Socket};
use super::registry::MemberView;
use crate::delivery::ReceiptPacer;
use crate::frame::{BuildId, Capacity, Claim, Epoch, Frame, Hello, MemberState, ModelId};
use crate::handshake::{GroupKey, PROTO};
use crate::pipe::memory::pair;
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
    undo_pending(&mut s, "zend", 1, first.session);
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
    hub.on_frame(
        "zend",
        1,
        &old,
        Frame::Claim(Claim {
            key: b"k".to_vec(),
            epoch: Epoch { ms: 5, node: 1 },
        }),
        &reply_on,
    )
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

/// **A handshake that fails after its admission undoes the replacement**: the
/// node's standing session reports again, and nothing is kept for the session
/// that never arrived.
#[tokio::test]
async fn a_failed_handshake_undoes_the_replacement() {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    let (first, old) = admitted_once(&hub);
    let (_, second) = hub.decide("zend", &hello(1)).unwrap();
    assert!(old.load(Ordering::SeqCst));
    {
        let mut s = hub.state();
        undo_pending(&mut s, "zend", 1, second.session);
        assert_eq!(s.sockets[&("zend".to_string(), 1)].session, first);
    }
    assert!(
        !old.load(Ordering::SeqCst),
        "the standing session reports again"
    );
    hub.on_frame("zend", 1, &old, ready(), &handle()).unwrap();
    assert_eq!(
        hub.state().registry.member("zend", 1).unwrap().state(),
        MemberState::Ready
    );
}
