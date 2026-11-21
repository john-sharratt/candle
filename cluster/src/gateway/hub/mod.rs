//! The gateway's live cluster: every member socket from its handshake to its
//! close, and the answers the proxy gives about which member serves a request.
//!
//! `Hub::serve` runs one member socket through the handshake, admission, the
//! session and the frames the gateway keeps or relays. A live event goes to
//! every other member of the app unless that member is upgrading, lagging or in
//! fault mode. Catch-up traffic goes to the one member its reply names. A member
//! whose ordered lane holds more than `LAG_CAP` live events is lagging: it is
//! sent no further live events and no requests until its lane drains and its
//! acknowledgements cover every other member. An upgrade admits a candidate
//! above the floor, which takes its share of live traffic while it certifies.
//! Members a rollout leaves behind are closed with `4410`.
//!
//! - `certification`: upgrade verdicts, floors and rollouts.
//! - `constants`: timings, caps and the list of apps.
//! - `members`: which members are admitted, and where they run.
//! - `pending`: replacement sessions and the reports they hold back.
//! - `placement`: where a request goes, and the wait for a card with room.
//! - `plans`: the cast plan and its log lines.
//! - `sockets`: a member socket's admission, its session and its relays.

mod reports;

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::machines::MachineTable;
use crate::replicate::FETCH_WINDOW;
use reports::{apply_report, check_report, Applied, Arrived, Held, Report};
use std::net::IpAddr;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;
// A queued unit's wait is measured on the runtime's clock, as its hold is.
use tokio::time::Instant as RuntimeInstant;
use tokio::time::{interval, timeout, MissedTickBehavior};

use super::admission::{Admission, Admitted};
use super::agent_row::AgentRow;
use super::agents::{Choice, Need};
use super::load::{Load, Picked};
use super::log::{CoordLog, Entry};
use super::open_answers::OpenAnswers;
use super::paxos_log::paxos_step;
use super::planner::{Planned, Planner};
use super::queue::{Placing, Queue, RouteAt, Urgency, HOLD, RESERVE_FOR};
use super::registry::{
    is_errand, CardKey, MemberView, PeerAsk, Pick, Registry, HOSTING_APP, LATENCY_APP,
};
use super::routing::{standing_of, Routing, Standing};
use super::status::{
    mode_name, AppView, ClaimView, FaultRow, HubStatus, MemberRow, Queues, Refusal, SocketReport,
};
use super::traffic::Traffic;
use super::upgrades::held::HeldBuilds;
use super::upgrades::mirror::Compared;
use super::upgrades::plan::{plan, Ask, Member as PlanMember, Plan};
use super::upgrades::stats::Sample;
use super::upgrades::{Action, Below, UpgradeConfig, Upgrades};
use super::worlds_view::{disconnected, worlds};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::consensus::WallMillis;
use crate::coord::{CoordState, UpgradeView};
use crate::delivery::ReceiptPacer;
use crate::dispatch::{Dispatch, Mirror, RouteTable};
use crate::frame::{
    name, Admit, BuildFetch, BuildId, Capacity, Frame, Hello, Lost, MemberState, Members,
    Plan as PlanFrame,
};
use crate::handshake::{admit, GroupKey, HandshakeError, ProofLimiter, ADMIT_SEND, PROTO};
use crate::pipe::{Message, PipeRx, PipeTx};
use crate::session::{start, start_behind, Ended, Incoming, Role, SessionHandle, VirtualConn};

use super::tiebreak::TieBreak;
use crate::upgrade::decisions::build_name;
use crate::version::info::AGENT;

/// A stream a member opened towards the gateway (§5.1): served by the
/// gateway's own router as a request for `app`, or, when its target is
/// `group:<other>`, routed to that app's group.
pub struct MemberStream {
    pub app: String,
    pub node: u64,
    /// What it was opened towards: the member's own app, or `group:<app>`.
    pub target: String,
    pub conn: VirtualConn,
}

pub(super) struct Socket {
    pub session: u64,
    pub handle: SessionHandle,
    /// The address the socket came from: where the member's machine is
    /// reached by what is not carried over the socket itself — the game's
    /// datagrams (`docs/battlecities_app.md` §5).
    pub addr: IpAddr,
    /// Set, under the state's guard, once a newer session of the node is
    /// admitted: its session's loop reads it to drop what it still reports
    /// of its node's state ([`Report`]).
    pub replaced: Arc<AtomicBool>,
    pub connected_ms: u64,
    /// Its clock against this gateway's at admission, from its `HELLO`.
    pub clock_offset_ms: i64,
    /// Its last `COORD` (§12.1).
    pub coord: Option<CoordState>,
    pub pacer: ReceiptPacer,
    /// When its last `CAPACITY` arrived, or it was admitted.
    pub capacity_at: Instant,
    /// Live events not queued for it while it lagged, and catch-up frames
    /// dropped at its cap.
    pub skipped: u64,
    pub dropped: u64,
    /// `PULSE`s not queued for it under pressure (§9.1).
    pub pulses_dropped: u64,
}

/// A session admitted whose socket is not yet in place: its `ADMIT` was
/// built, and what has to bring that snapshot up to date waits here for it.
/// One per node: a newer `HELLO` of the node is refused while it stands
/// ([`Hub::decide`]), and only its own session's `run` or [`PendingGuard`]
/// removes it.
pub(super) struct Pending {
    session: u64,
    since: Instant,
    /// The claim relays and plans for the node since its `ADMIT`'s table was
    /// built, handed to the new socket under the guard that puts it in place.
    queue: Vec<Frame>,
    /// The state reports the node's standing session sent since — held back,
    /// not applied, for the session is marked replaced; applied after all if
    /// the handshake fails (`undo_pending`), dropped if it succeeds.
    held: Held,
    /// More than [`PENDING_CAP`] relays piled up: the admission fails.
    overflowed: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct Mode {
    clustered: bool,
    since_ms: u64,
}

#[derive(Default)]
pub(super) struct State {
    admission: Admission,
    pub registry: Registry,
    pub sockets: HashMap<(String, u64), Socket>,
    /// Sessions admitted whose socket is not in place yet, by app and node.
    pending: HashMap<(String, u64), Pending>,
    limiter: ProofLimiter,
    refusals: VecDeque<Refusal>,
    modes: HashMap<String, Mode>,
    last_close: HashMap<(String, u64), String>,
    /// Each member's last `COORD`, kept across its sockets: a member that
    /// reconnects is logged for what changed while it was away, not again for
    /// everything it reports.
    reported: HashMap<(String, u64), CoordState>,
    /// Admitting no sockets (§12.4 `drain-gateway`).
    pub draining: bool,
    /// Pending floors, candidates and their certification, rollouts (§3.5,
    /// §3.8–§3.11).
    upgrades: Upgrades,
    /// The builds members hold, from their `BUILD_OFFER`s and `BUILD_HAVE`s.
    held: HeldBuilds,
    /// Spreads `BUILD_FETCH`es naming no member across the holders.
    fetch_rr: usize,
    /// The catch-ups and fetches handed to an answerer and not yet `DONE`.
    pub open: OpenAnswers,
    /// The units waiting for a card with room (`docs/cooperative_admission.md`
    /// §4.3).
    queue: Queue,
    /// The plan for the movable work (§4.4–§4.5).
    planner: Planner,
    /// The cards line the plan last logged ([`log_plan`]).
    cards_logged: String,
}

pub struct Hub {
    key: GroupKey,
    /// This gateway's name in the coordination log.
    name: String,
    state: Mutex<State>,
    /// The tie-break acceptor (§5.9), once [`Self::enable_tiebreak`] opened it.
    tiebreak: Mutex<Option<TieBreak>>,
    pub log: CoordLog,
    pub traffic: Traffic,
}

/// An admission decided and not yet taken by its `run`, which owns undoing
/// it: dropped armed — its handshake failed, or the serve task was dropped or
/// panicked first — it undoes the replacement the admission began
/// ([`undo_pending`]), so the node's standing session reports again and its
/// next `HELLO` is admitted.
pub(super) struct PendingGuard {
    pub hub: Arc<Hub>,
    pub app: String,
    pub node: u64,
    pub admitted: Admitted,
    pub armed: bool,
}

mod certification;
mod constants;
mod members;
mod pending;
mod placement;
mod plans;
mod sockets;

pub use constants::*;
use members::*;
pub(super) use pending::*;
use plans::*;
