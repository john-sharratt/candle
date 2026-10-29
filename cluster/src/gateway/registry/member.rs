//! One admitted member, as the gateway sees it.

use crate::dispatch::RouteTable;
use crate::frame::{BuildId, Capacity, MemberInfo, MemberState, RepoAd};

/// One admitted member.
#[derive(Debug, Clone)]
pub struct MemberView {
    pub node: u64,
    pub host: String,
    /// The card its `HELLO` named (`docs/cooperative_admission.md` §4.2): the
    /// members of one process share one, and the gateway counts their work
    /// together. Zero names none — the member is a card of its own.
    pub card: u64,
    pub app: String,
    pub build: BuildId,
    /// Its power at admission, fixed for the session (§9.5).
    pub weight: u32,
    pub session: u64,
    /// The repositories its `HELLO` advertised (§10.1, §11.2).
    pub repos: Vec<RepoAd>,
    /// Its last `CAPACITY`; a member is loading until it sends one.
    pub capacity: Option<Capacity>,
    pub routes: Option<RouteTable>,
    /// Its own highest `oseq`, from its `ACK`s (§7.1).
    pub high_water: u64,
    /// Its last `ACK`: through which `oseq` it has applied each origin.
    pub applied: Vec<(u64, u64)>,
    /// Its ordered lane passed its cap: it is sent no live event and routed
    /// no request until it has drained and caught up (§4.4).
    pub lagging: bool,
    /// Admitted while the group already had a member serving: routed no
    /// request until its own `ACK`s cover what every other origin has written
    /// — the gateway's judgement, not the member's. Unlike lagging, it is sent
    /// live events throughout, so it catches up rather than chasing them.
    pub syncing: bool,
    /// Requests this gateway has handed it and it has not finished answering
    /// (§5.3): counted at the pick, so a burst sees its own earlier picks
    /// before the member's next `CAPACITY` reports them.
    pub in_flight: u32,
    /// Of those, the draws (§5.3): what a draw's placement spreads by, since a
    /// card draws one picture at a time whatever else it answers meanwhile.
    pub drawing: u32,
    /// Jobs handed to it, an agent, since its last `CAPACITY` (§11.3): a job
    /// is started and then read back, so it holds no request open to count,
    /// and the next `CAPACITY` reports it running. Until then the choice
    /// counts it, so a burst of jobs is spread instead of all reading the
    /// same free slot.
    pub handed: u32,
    /// Requests handed to it since its last `CAPACITY`: prompts its reported
    /// backlog does not hold yet. A request handed earlier is either queued,
    /// and in that backlog, or past its prefill and decoding, and in its
    /// decode step — so `finish_us` charges a nominal prompt for these alone.
    pub placed: u32,
    /// Its session has ended and the socket's cleanup has not yet run its
    /// leave: no stream can open to it, so it is routed nothing. A request
    /// picked for it in that window was answered `503` while another member
    /// stood ready.
    pub closing: bool,
}

/// What a member's request for a peer avoids ([`super::Registry::pick_from`]):
/// the member that sent it, and a canary while one certifies. Both `None` — the
/// default — for a client's request, placed as ever.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeerAsk {
    pub sender: Option<u64>,
    pub canary: Option<u64>,
}

/// How many of an origin's latest writes a member may still lack and count
/// as in sync: events in flight when its `ACK` was sent.
pub const SYNC_SLACK: u64 = 64;

impl MemberView {
    pub fn state(&self) -> MemberState {
        self.capacity
            .as_ref()
            .map_or(MemberState::Loading, |c| c.state)
    }

    pub fn ready(&self) -> bool {
        self.state() == MemberState::Ready
    }

    /// Whether it holds `origin`'s records through `through`: its own, or as
    /// its last `ACK` reported them applied.
    pub fn holds_through(&self, origin: u64, through: u64) -> bool {
        origin == self.node
            || self
                .applied
                .iter()
                .any(|(o, t)| *o == origin && *t >= through)
    }

    /// Ready, not lagging, not still syncing and not closing: a request may be
    /// routed to it.
    pub fn routable(&self) -> bool {
        self.ready() && !self.lagging && !self.syncing && !self.closing
    }

    /// Ready and not closing: a conversation may be placed on it (§4.4).
    /// Current enough is a conversation dimension, not a member one: a new
    /// conversation has no history to be behind on, and a released one goes
    /// only where its last holder's records are held ([`Self::holds_through`]).
    /// A member behind on everyone else's writes is still current for those.
    pub fn placeable(&self) -> bool {
        self.ready() && !self.closing
    }

    pub fn info(&self) -> MemberInfo {
        MemberInfo {
            node: self.node,
            host: self.host.clone(),
            app: self.app.clone(),
            build: self.build.clone(),
            weight: self.weight,
            state: self.state(),
            high_water: self.high_water,
        }
    }
}
