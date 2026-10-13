//! Gateways peering with each other (§13.4): every gateway holds one link to
//! every other, so active-active gateways answering the same public names can
//! share what members cannot carry for them — ACME challenges, certificates,
//! the order ledger, the account — and agree who orders each site.
//!
//! A link is a session like a member's socket: the §3.3 handshake over the
//! group key with `app = gateway`, then `PEER_HELLO` each way, whose digests of
//! the session key and the OAuth client must agree, or the peering is refused
//! on both ends. Two dials that cross keep the lower node's ([`crossing`]).
//! What arrives on a link goes to the [`PeerHandler`] — the gateway's ACME — and
//! this module only carries it.
//!
//! | Module | |
//! |---|---|
//! | [`crossing`] | which of two crossed links is kept |
//! | [`link`] | one link: `PEER_HELLO`, registration, its frames |
//! | [`accept`] | a peer's dial, taken |
//! | [`dial`] | this gateway's dial loop to one peer |
//! | [`view`] | the Peers block's links, dials and refusals |

pub mod accept;
pub mod crossing;
pub mod dial;
pub mod link;
pub mod view;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use sha2::{Digest, Sha256};

use self::view::{DialView, LinkView, PeerRefusal, PeersView};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::consensus::rendezvous::{owner, Weighted};
use crate::frame::{BuildId, Frame, PeerHello};
use crate::gateway::Hub;
use crate::handshake::GroupKey;
use crate::session::SessionHandle;

/// The `app` a peer link's handshake names.
pub const GATEWAY_APP: &str = "gateway";

/// How many refusals the Peers block keeps.
const REFUSALS_KEPT: usize = 50;

/// What a gateway's ACME does with its peers' frames.
pub trait PeerHandler: Send + Sync {
    /// `peer` is linked; `secure` when the link runs over a verified TLS
    /// connection, the only kind certificate keys and the account cross.
    fn linked(&self, peer: u64, secure: bool);
    /// A frame from `peer`.
    fn frame(&self, peer: u64, frame: Frame);
    /// `peer`'s link is gone.
    fn unlinked(&self, peer: u64);
}

/// SHA-256 of `bytes`: how a shared secret is compared without crossing.
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Who this gateway is to its peers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerIdentity {
    pub node: u64,
    pub build: BuildId,
    /// [`digest`] of the session key's bytes.
    pub session_key: [u8; 32],
    /// [`digest`] of the OAuth client id.
    pub oauth_client: [u8; 32],
}

impl PeerIdentity {
    pub fn hello(&self) -> PeerHello {
        PeerHello {
            node: self.node,
            build: self.build.clone(),
            session_key: self.session_key,
            oauth_client: self.oauth_client,
        }
    }
}

struct Link {
    /// Tells this registration from a later one for the same peer.
    id: u64,
    handle: SessionHandle,
    dialler: u64,
    secure: bool,
    hello: PeerHello,
    since_ms: u64,
}

#[derive(Default)]
struct State {
    links: BTreeMap<u64, Link>,
    next_id: u64,
    dials: Vec<DialView>,
    refusals: Vec<PeerRefusal>,
}

pub struct Peering {
    key: GroupKey,
    me: PeerIdentity,
    hub: Arc<Hub>,
    handler: OnceLock<Weak<dyn PeerHandler>>,
    state: Mutex<State>,
}

fn hex(node: u64) -> String {
    format!("{node:016x}")
}

impl Peering {
    /// This gateway as `me`, logging to `hub`'s coordination log.
    pub fn new(key: GroupKey, me: PeerIdentity, hub: Arc<Hub>) -> Arc<Peering> {
        Arc::new(Peering {
            key,
            me,
            hub,
            handler: OnceLock::new(),
            state: Mutex::new(State::default()),
        })
    }

    /// Hand what arrives on links to `handler`.
    pub fn set_handler(&self, handler: Weak<dyn PeerHandler>) {
        let _ = self.handler.set(handler);
    }

    fn handler(&self) -> Option<Arc<dyn PeerHandler>> {
        self.handler.get()?.upgrade()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn node(&self) -> u64 {
        self.me.node
    }

    /// A peer's node id reached this gateway: the tie-break voter is the
    /// lowest id among the gateways that have been seen (§5.9).
    pub(crate) fn saw_peer(&self, node: u64) {
        self.hub.note_peer(node);
    }

    pub fn identity(&self) -> &PeerIdentity {
        &self.me
    }

    pub(crate) fn key(&self) -> &GroupKey {
        &self.key
    }

    /// A peering decision about `subject` — a link, a certificate order's
    /// step — for the coordination log (§12.3).
    pub fn note(&self, subject: u64, text: String) {
        self.hub.record(GATEWAY_APP, hex(subject), text);
    }

    fn refused(&self, node: Option<u64>, reason: String) {
        let mut s = self.state();
        s.refusals.push(PeerRefusal {
            at_ms: now_ms(),
            node: node.map(hex).unwrap_or_default(),
            reason: reason.clone(),
        });
        let excess = s.refusals.len().saturating_sub(REFUSALS_KEPT);
        s.refusals.drain(..excess);
        drop(s);
        self.note(
            node.unwrap_or_default(),
            format!("peering refused: {reason}"),
        );
    }

    /// The peers linked now, and whether each link is secure.
    pub fn linked(&self) -> Vec<(u64, bool)> {
        self.state()
            .links
            .iter()
            .map(|(n, l)| (*n, l.secure))
            .collect()
    }

    /// Whether `peer` is linked over a verified TLS connection.
    pub fn secure(&self, peer: u64) -> bool {
        self.state().links.get(&peer).is_some_and(|l| l.secure)
    }

    /// Send `frame` to `peer`; whether it was queued.
    pub fn send(&self, peer: u64, frame: Frame) -> bool {
        let handle = self.state().links.get(&peer).map(|l| l.handle.clone());
        handle.is_some_and(|h| h.control(frame).is_ok())
    }

    /// Send `frame` to every linked peer; the peers it was queued for.
    pub fn broadcast(&self, frame: &Frame) -> Vec<u64> {
        let handles: Vec<(u64, SessionHandle)> = self
            .state()
            .links
            .iter()
            .map(|(n, l)| (*n, l.handle.clone()))
            .collect();
        handles
            .into_iter()
            .filter(|(_, h)| h.control(frame.clone()).is_ok())
            .map(|(n, _)| n)
            .collect()
    }

    /// Close every link with `1001`: this gateway is stopping, and its peers
    /// order without it until it is back.
    pub fn close_all(&self, reason: &str) {
        for l in self.state().links.values() {
            l.handle.close(Close::new(CloseCode::GoingAway, reason));
        }
    }

    /// The gateway that orders `site`'s certificate: rendezvous, with equal
    /// weights, over this gateway and the peers linked to it. Every connected
    /// gateway names the same one, so exactly one orders.
    pub fn orderer(&self, site: &str) -> u64 {
        let mut nodes: Vec<Weighted> = vec![Weighted {
            node: self.me.node,
            weight: 1.0,
        }];
        nodes.extend(self.state().links.keys().map(|n| Weighted {
            node: *n,
            weight: 1.0,
        }));
        owner(site.as_bytes(), &nodes).expect("this gateway is always among them")
    }

    /// A dial to `url` begins: its row in the Peers block.
    pub(crate) fn dialling(&self, url: &str) -> usize {
        let mut s = self.state();
        s.dials.push(DialView {
            url: url.to_string(),
            state: "dialling",
            ..DialView::default()
        });
        s.dials.len() - 1
    }

    pub(crate) fn dial_state(&self, at: usize, state: &'static str, close: Option<String>) {
        let mut s = self.state();
        if let Some(d) = s.dials.get_mut(at) {
            if state == "retrying" {
                d.attempts += 1;
            }
            d.state = state;
            if close.is_some() {
                d.last_close = close;
            }
        }
    }

    pub fn view(&self) -> PeersView {
        let s = self.state();
        PeersView {
            node: hex(self.me.node),
            links: s
                .links
                .iter()
                .map(|(n, l)| LinkView {
                    node: hex(*n),
                    build: format!("{}-{}", l.hello.build.count, l.hello.build.hash),
                    secure: l.secure,
                    dialled_by: hex(l.dialler),
                    since_ms: l.since_ms,
                })
                .collect(),
            dials: s.dials.clone(),
            refusals: s.refusals.iter().rev().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests;
