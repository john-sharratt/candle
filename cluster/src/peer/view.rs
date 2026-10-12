//! The Peers block's links, dials and refusals (§12.2), as `status.json`
//! carries them.

use serde::Serialize;

/// One peer gateway this gateway is linked to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LinkView {
    pub node: String,
    pub build: String,
    /// Over a verified TLS connection: certificate keys may cross it.
    pub secure: bool,
    /// Which end dialled it.
    pub dialled_by: String,
    pub since_ms: u64,
}

/// One configured peer, as this gateway dials it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DialView {
    pub url: String,
    /// `dialling`, `linked`, `retrying` or `stopped`.
    pub state: &'static str,
    pub attempts: u64,
    pub last_close: Option<String>,
}

/// A peering refused: a failed handshake or disagreeing secrets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PeerRefusal {
    pub at_ms: u64,
    /// Empty when the peer named no node.
    pub node: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PeersView {
    /// This gateway's own node.
    pub node: String,
    pub links: Vec<LinkView>,
    pub dials: Vec<DialView>,
    pub refusals: Vec<PeerRefusal>,
}
