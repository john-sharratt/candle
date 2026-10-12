//! Gateways peering in one process, over in-memory pipes: two and three link
//! and agree on every site's orderer; crossed dials end with the lower node's
//! link on both ends; disagreeing secrets refuse the peering on both; a link
//! closed is redialled.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::{sleep, Instant};

use super::*;
use crate::dial::Connect;
use crate::pipe::memory::{pair, MemoryRx, MemoryTx};

fn key() -> GroupKey {
    GroupKey::from_bytes([5; 32])
}

fn identity(node: u64, session_key: &[u8]) -> PeerIdentity {
    PeerIdentity {
        node,
        build: BuildId {
            count: 12,
            hash: "h12".into(),
            local: String::new(),
        },
        session_key: digest(session_key),
        oauth_client: digest(b"client.apps.googleusercontent.com"),
    }
}

/// What a gateway's ACME would have been told.
#[derive(Default)]
struct Seen {
    linked: Mutex<Vec<(u64, bool)>>,
    unlinked: Mutex<Vec<u64>>,
    frames: Mutex<Vec<(u64, Frame)>>,
}

impl PeerHandler for Seen {
    fn linked(&self, peer: u64, secure: bool) {
        self.linked.lock().unwrap().push((peer, secure));
    }
    fn frame(&self, peer: u64, frame: Frame) {
        self.frames.lock().unwrap().push((peer, frame));
    }
    fn unlinked(&self, peer: u64) {
        self.unlinked.lock().unwrap().push(peer);
    }
}

struct Gw {
    peering: Arc<Peering>,
    seen: Arc<Seen>,
}

fn gateway(node: u64, session_key: &[u8]) -> Gw {
    let hub = Hub::new(key(), format!("g{node}"));
    let peering = Peering::new(key(), identity(node, session_key), hub);
    let seen = Arc::new(Seen::default());
    let handler: Arc<dyn PeerHandler> = seen.clone();
    peering.set_handler(Arc::downgrade(&handler));
    Gw { peering, seen }
}

/// A dial to `to`, taken by its accept as if on its TLS entrance.
struct ToPeer {
    to: Arc<Peering>,
    open: Arc<AtomicBool>,
}

impl Connect for ToPeer {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        if !self.open.load(Ordering::SeqCst) {
            return Err("unreachable".into());
        }
        let (d, a) = pair(1 << 20);
        let to = self.to.clone();
        tokio::spawn(async move { to.accept(a.tx, a.rx, true).await });
        Ok((d.tx, d.rx))
    }
}

fn dial(from: &Gw, to: &Gw) -> Arc<AtomicBool> {
    let open = Arc::new(AtomicBool::new(true));
    let peering = from.peering.clone();
    let connector = ToPeer {
        to: to.peering.clone(),
        open: open.clone(),
    };
    let url = format!("peer {:x}", to.peering.node());
    tokio::spawn(async move { peering.dial(&url, connector, true).await });
    open
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

fn linked(g: &Gw) -> BTreeSet<u64> {
    g.peering.linked().into_iter().map(|(n, _)| n).collect()
}

/// Two gateways link once one dials; both see the other, over a secure link,
/// and a frame broadcast by one reaches the other's handler.
#[tokio::test]
async fn two_gateways_link_and_carry_frames() {
    let (a, b) = (gateway(1, b"key"), gateway(2, b"key"));
    dial(&a, &b);
    until("both linked", || {
        linked(&a) == [2].into() && linked(&b) == [1].into()
    })
    .await;
    assert_eq!(*a.seen.linked.lock().unwrap(), [(2, true)]);
    assert_eq!(*b.seen.linked.lock().unwrap(), [(1, true)]);
    let sent = a.peering.broadcast(&Frame::AcmeOrdering {
        site: "tokera".into(),
    });
    assert_eq!(sent, [2]);
    until("the frame arrives", || {
        !b.seen.frames.lock().unwrap().is_empty()
    })
    .await;
    assert_eq!(
        b.seen.frames.lock().unwrap()[0],
        (
            1,
            Frame::AcmeOrdering {
                site: "tokera".into()
            }
        )
    );
    let view = b.peering.view();
    assert_eq!(view.links.len(), 1);
    assert_eq!(
        (
            view.links[0].node.as_str(),
            view.links[0].dialled_by.as_str(),
            view.links[0].build.as_str()
        ),
        ("0000000000000001", "0000000000000001", "12-h12")
    );
}

/// Three gateways in a full mesh name the same orderer for every site, and
/// each site has exactly one; alone, a gateway orders everything itself.
#[tokio::test]
async fn every_connected_gateway_names_the_same_orderer() {
    let gs = [gateway(1, b"k"), gateway(2, b"k"), gateway(3, b"k")];
    let sites = ["tokera", "zend", "npcd", "battlecities", "remote"];
    for g in &gs {
        for s in sites {
            assert_eq!(g.peering.orderer(s), g.peering.node(), "alone");
        }
    }
    dial(&gs[0], &gs[1]);
    dial(&gs[0], &gs[2]);
    dial(&gs[1], &gs[2]);
    until("a full mesh", || gs.iter().all(|g| linked(g).len() == 2)).await;
    for s in sites {
        let named: BTreeSet<u64> = gs.iter().map(|g| g.peering.orderer(s)).collect();
        assert_eq!(named.len(), 1, "{s}: {named:?}");
    }
}

/// Two gateways dialling each other at once end with one link, the one the
/// lower node dialled, on both ends.
#[tokio::test]
async fn crossed_dials_keep_the_lower_nodes_link() {
    let (a, b) = (gateway(3, b"k"), gateway(7, b"k"));
    dial(&a, &b);
    dial(&b, &a);
    until("one link each, dialled by 3", || {
        let (va, vb) = (a.peering.view(), b.peering.view());
        va.links.len() == 1
            && vb.links.len() == 1
            && va.links[0].dialled_by == "0000000000000003"
            && vb.links[0].dialled_by == "0000000000000003"
    })
    .await;
    // And it stays that way: the higher node's dial loop has stopped.
    sleep(Duration::from_millis(600)).await;
    assert_eq!(linked(&a), [7].into());
    assert_eq!(b.peering.view().links[0].dialled_by, "0000000000000003");
}

/// A gateway whose session key differs is refused on both ends, each saying
/// why; no link stands and the handlers hear nothing.
#[tokio::test]
async fn differing_secrets_refuse_the_peering_on_both_ends() {
    let (a, b) = (gateway(1, b"key one"), gateway(2, b"key two"));
    dial(&a, &b);
    until("both refused", || {
        !a.peering.view().refusals.is_empty() && !b.peering.view().refusals.is_empty()
    })
    .await;
    for g in [&a, &b] {
        assert!(g.peering.view().refusals[0]
            .reason
            .starts_with("the session keys differ"));
        assert!(g.seen.linked.lock().unwrap().is_empty());
        assert!(linked(g).is_empty());
    }
}

/// A link closed is gone from both ends, each handler told; the dialler
/// redials and the peers link again.
#[tokio::test]
async fn a_closed_link_is_redialled() {
    let (a, b) = (gateway(1, b"k"), gateway(2, b"k"));
    dial(&a, &b);
    until("linked", || {
        linked(&a) == [2].into() && linked(&b) == [1].into()
    })
    .await;
    b.peering.close_all("b is stopping");
    until("unlinked on both", || {
        a.seen.unlinked.lock().unwrap().contains(&2) && b.seen.unlinked.lock().unwrap().contains(&1)
    })
    .await;
    until("linked again", || {
        a.seen.linked.lock().unwrap().len() == 2 && linked(&b) == [1].into()
    })
    .await;
}
