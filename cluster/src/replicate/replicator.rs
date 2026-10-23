//! One app's replication on a member (§6.3–§7.5): its own events going out,
//! other members' coming in, catch-ups asked and answered, and the rejoin of
//! a member that was cut off. The member's sessions hand it what arrives; it
//! sends on their handles.
//!
//! | Concern | Where |
//! |---|---|
//! | own events: the tap, round robin, a request's flush, receipts | [`super::send`] |
//! | applying what arrives, and the catch-ups this member asks for | [`super::receive`] |
//! | answering another member's `CATCHUP`, `FETCH` and `HAVE` | [`super::serve`] |
//! | Paxos, `max_members`, generations and the rejoin | [`super::consensus`] |
//! | the once-a-tick work: `ACK`, gaps, periodic catch-ups, cutting off | [`super::tick`] |

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use super::answering::Answering;
use super::catchup::{Progress, Purpose, Requester, FETCH_WINDOW};
use super::fault::Fault;
use super::lease::{LeaseEvent, Term, EVENTS_KEPT};
use super::pulse::PULSES_KEPT;
use super::quorum_governor::{Downgrade, Governor};
use super::store::Store;
use super::watch::GapWatch;
use crate::consensus::votes::VoteStore;
use crate::consensus::{Config, Millis, Paxos};
use crate::coord::{AppCoord, CommandLine};
use crate::delivery::Outbox;
use crate::fragment::Whole;
use crate::frame::{Catchup, Done, Frame, Headers, Members, Pulse};
use crate::member::AppMember;
use crate::session::SessionHandle;

/// Where the member stands in the group, beyond its sockets (§6.6, §6.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Writing under the group's generation; serving as its sockets allow.
    Joined,
    /// Holding what it wrote cut off, not yet placed by a generation: it
    /// serves its own listener locally and sends none of it.
    CutOff,
    /// Admitted again after writing cut off: catching up and taking a
    /// generation. Its listener answers `503`.
    Rejoining,
    /// Rebuilding its state from the log. Its listener answers `503`.
    Reloading,
}

/// Bytes of records arrived and not yet applied past which the next waits:
/// twice a catch-up's fetch window, so a full window and the batch being
/// applied beside it fit — the gateway sizes a member's catch-up lane the
/// same way (`gateway::hub::CATCHUP_CAP`).
///
/// **The queue is bounded so the socket is.** Unbounded, a member applying
/// slower than records arrived took them off the socket regardless and held
/// them in memory: WhiteWorld's zend, behind the cluster's layer ingest,
/// committed past twice its RAM and paged until its socket runtime went
/// seconds unpolled. Bounded, the member stops reading the session's
/// incoming queue, the session's reader stops reading the socket, and the
/// gateway holds what it cannot send — its lane, its lagging and catch-up,
/// the mechanism built for a member that cannot keep up.
pub const ARRIVALS_CAP: usize = 2 * FETCH_WINDOW as usize;

/// The unit [`ARRIVALS_CAP`] is counted in, so its permits fit a `u32`.
const ROOM_UNIT: usize = 1024;

/// [`ARRIVALS_CAP`] in [`ROOM_UNIT`]s.
const ROOM_PERMITS: usize = ARRIVALS_CAP / ROOM_UNIT;

/// The permits a record of `len` bytes holds while it waits to be applied.
/// One larger than the whole cap counts as the cap, so it waits for an empty
/// queue rather than for ever.
fn room_for(len: usize) -> u32 {
    len.div_ceil(ROOM_UNIT).clamp(1, ROOM_PERMITS) as u32
}

#[cfg(test)]
mod room_tests {
    use super::*;

    /// A record holds its size in units, at least one, and never more than
    /// the whole budget — so even an empty record counts, and one larger
    /// than the cap waits for an empty queue rather than for ever.
    #[test]
    fn a_record_holds_its_size_within_the_budget() {
        assert_eq!(room_for(0), 1);
        assert_eq!(room_for(1), 1);
        assert_eq!(room_for(ROOM_UNIT), 1);
        assert_eq!(room_for(ROOM_UNIT + 1), 2);
        assert_eq!(room_for(59 * 1024), 59);
        assert_eq!(room_for(ARRIVALS_CAP * 4), ROOM_PERMITS as u32);
        assert_eq!(ROOM_PERMITS, 131_072, "twice a 64 MiB fetch window, in KiB");
    }

    /// The budget holds: once it is spent, the next record waits until a
    /// queued one is applied and gives its share back.
    #[tokio::test]
    async fn a_full_budget_holds_the_next_record_until_room_is_given_back() {
        let room = Arc::new(Semaphore::new(ROOM_PERMITS));
        let full = room
            .clone()
            .acquire_many_owned(ROOM_PERMITS as u32)
            .await
            .unwrap();
        let next = room.clone().acquire_many_owned(room_for(4096));
        tokio::pin!(next);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut next)
                .await
                .is_err(),
            "the next record waits while the budget is spent"
        );
        drop(full);
        assert!(next.await.is_ok(), "applied records give their room back");
    }
}

/// Something that arrived for the requester's side, in arrival order: a
/// record, or a catch-up reply. They go through one queue so a `DONE` is
/// handled only after every record ahead of it is applied.
pub(super) enum Arrival {
    /// The `seq`-th record to arrive, counted from one, holding its bytes'
    /// share of [`ARRIVALS_CAP`] until it is applied.
    Event {
        c: usize,
        whole: Whole,
        seq: u64,
        room: OwnedSemaphorePermit,
    },
    Headers {
        c: usize,
        headers: Headers,
    },
    Done {
        c: usize,
        done: Done,
    },
}

/// A rejoin waiting on its generation's decision.
#[derive(Debug, Clone, Copy)]
pub(super) struct Rejoin {
    pub k: u64,
    pub deadline: Instant,
}

pub(super) struct State {
    pub outbox: Outbox,
    /// Own events tapped and not yet sent, and when: a request's flush takes
    /// them onto its own socket before the round robin does.
    pub fresh: BTreeMap<u64, Instant>,
    pub sockets: Vec<Option<SessionHandle>>,
    /// Per socket: its admission catch-up has finished.
    pub ready: Vec<bool>,
    /// An admission catch-up has finished since the process started: the
    /// member has served, and what it holds is current up to the moment a
    /// socket dropped. A socket admitted again serves at once — the gateway
    /// holds requests off it until its `ACK`s are in sync with the members
    /// serving — instead of reading as loading, which after a gateway restart
    /// took every member, and so the app, out of service for minutes.
    pub served: bool,
    /// Per socket: how many times its admission catch-up has gone unanswered
    /// since the socket was admitted (`tick::ADMISSION_QUIET_LIMIT`).
    pub admission_quiet: Vec<u32>,
    pub members: Vec<Option<Members>>,
    pub rr: usize,
    pub next_request: u64,
    pub catchups: BTreeMap<u64, Requester>,
    /// The catch-up answers this member is running for others, one per
    /// requester.
    pub answering: Answering,
    /// Each catch-up's last word from its answerer.
    pub heard: HashMap<u64, Instant>,
    /// Which catch-up each outstanding fetch belongs to.
    pub fetches: HashMap<u64, u64>,
    /// Records some catch-up of this member wants: another lists them as
    /// held, so two catch-ups never fetch one record.
    pub claimed: HashSet<(u64, u64)>,
    /// The first admission since the process started has classified the
    /// own events (§7.5).
    pub restarted: bool,
    /// `HAVE` has been offered since the process started.
    pub offered: bool,
    pub gaps: GapWatch,
    pub last_periodic: Instant,
    pub phase: Phase,
    pub rejoin: Option<Rejoin>,
    pub retry_rejoin: Option<Instant>,
    pub ack_dirty: bool,
    /// `max_members` and the greatest decided generation, as the substrate
    /// held them at the last read (once a second).
    pub max_members: u32,
    pub generation: u64,
    /// In fault mode for a full disk (§7.7).
    pub fault: Option<Fault>,
    /// Drained: taking no work, still connected (§12.4).
    pub drained: bool,
    /// The generation its last rejoin took.
    pub last_rejoin: Option<u64>,
    /// The last `COORD` sent, and when (§12.1).
    pub coord_sent: Option<(Vec<u8>, Instant)>,
    /// Whether the group has stayed short of `max_members`, and what is
    /// proposed for it (§5.9).
    pub governor: Governor,
    /// When a reply flagged as the gateway's tie-break vote last arrived.
    pub voter_seen: Option<Instant>,
    pub stats: Stats,
}

impl State {
    pub fn next_request(&mut self) -> u64 {
        self.next_request += 1;
        self.next_request
    }

    /// Drop catch-up `id`, letting go of every record it still wanted.
    pub fn drop_catchup(&mut self, id: u64) {
        if let Some(r) = self.catchups.remove(&id) {
            for k in r.outstanding_wants() {
                self.claimed.remove(&k);
            }
        }
        self.heard.remove(&id);
        self.fetches.retain(|_, owner| *owner != id);
    }

    /// The admitted sockets, in cluster order.
    pub fn admitted(&self) -> Vec<(usize, SessionHandle)> {
        self.sockets
            .iter()
            .enumerate()
            .filter_map(|(c, h)| h.clone().map(|h| (c, h)))
            .collect()
    }
}

/// Counters for the status page and the tests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub applied: u64,
    pub refused: u64,
    pub reloads: u64,
    pub catchups: u64,
    pub answered: u64,
    pub fetched_for_others: u64,
    pub resent: u64,
}

/// A catch-up in progress, for the loading page and `/v1/status` (§7.6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatchupView {
    pub cluster: usize,
    pub purpose: Purpose,
    #[serde(flatten)]
    pub progress: Progress,
}

/// What `/v1/status`'s `cluster` block shows of replication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplicationView {
    pub phase: Phase,
    pub generation: u64,
    pub unsent: usize,
    pub in_flight: usize,
    pub catchups: Vec<CatchupView>,
    /// Catch-up answers running for other members, one per requester.
    pub answering: usize,
    /// The counters, under their own key: flattened, `stats.catchups` (a
    /// count) overwrote the `catchups` list above.
    pub stats: Stats,
    pub quorum: QuorumView,
}

/// The group's size as this member counts it, and a downgrade under way
/// (§5.9).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuorumView {
    pub max_members: u32,
    /// The distinct ready members seen now, this one included.
    pub ready: u32,
    pub downgrade: Option<Downgrade>,
}

/// What a member replicates through: its substrate, and its acceptor opened
/// over its vote journal.
pub struct Replication {
    pub store: Arc<dyn Store>,
    pub paxos: Paxos<Box<dyn VoteStore>>,
}

impl Replication {
    /// `store`, with the acceptor's votes resumed from `votes`.
    pub fn open(store: Arc<dyn Store>, votes: Box<dyn VoteStore>) -> io::Result<Replication> {
        let node = store.node();
        Ok(Replication {
            paxos: Paxos::open(Config::new(node), votes, Millis(0), node)?,
            store,
        })
    }
}

/// See the module documentation.
pub struct Replicator {
    pub(super) store: Arc<dyn Store>,
    pub(super) node: u64,
    pub(super) s: Mutex<State>,
    /// Wakes the round-robin sender.
    pub(super) send_wake: Notify,
    /// Wakes a request's flush waiting on receipts.
    pub(super) receipts: Notify,
    /// Every arrival, in order. Unbounded in count: what bounds it is
    /// `room`, the bytes of records not yet applied.
    pub(super) arrivals: mpsc::UnboundedSender<Arrival>,
    arrivals_rx: Mutex<Option<mpsc::UnboundedReceiver<Arrival>>>,
    /// [`ARRIVALS_CAP`], in [`ROOM_UNIT`]s, less what queued records hold.
    room: Arc<Semaphore>,
    /// Records that have arrived, and through which of them every one has
    /// been applied: a request a gateway routes here is served only once
    /// every record that arrived before it is applied (§6.4).
    received: AtomicU64,
    pub(super) applied_upto: watch::Sender<u64>,
    /// The sender, receiver and ticker, ended by [`Self::shutdown`].
    tasks: Mutex<Vec<JoinHandle<()>>>,
    pub(super) paxos: Mutex<Paxos<Box<dyn VoteStore>>>,
    epoch: Instant,
    pub(super) member: Weak<AppMember>,
    /// The maintenance commands run lately, for `COORD` (§12.4).
    pub(super) commands: Mutex<VecDeque<CommandLine>>,
    /// What becomes of the leases this member asks for (§9.2).
    pub(super) leases: broadcast::Sender<LeaseEvent>,
    /// Each lease this member asked for: its term and the quorum counted, so
    /// one held under a smaller quorum is asked for again as the group grows.
    pub(super) terms: Mutex<BTreeMap<Vec<u8>, Term>>,
    /// The world pulses other members broadcast (§9.1).
    pub(super) pulses: broadcast::Sender<Pulse>,
    /// What the app adds to `COORD`: its worlds and NPC leases (§12.1).
    pub(super) app_coord: Mutex<Option<AppCoordFn>>,
}

/// How the app reports its worlds and NPC leases for `COORD`. Called once a
/// second off the ticker, so it reads state and never waits.
pub type AppCoordFn = Box<dyn Fn() -> AppCoord + Send + Sync>;

impl Replicator {
    /// `replication` for the member `member`, over `clusters` sockets.
    pub fn new(
        replication: Replication,
        clusters: usize,
        member: Weak<AppMember>,
    ) -> Arc<Replicator> {
        let Replication { store, mut paxos } = replication;
        // The quorum the substrate holds, from the first frame on (§5.9).
        let max_members = store.max_members();
        let generation = store.generation();
        paxos.set_quorum(max_members as usize / 2 + 1);
        paxos.allow_tiebreak(max_members == 2);
        let node = store.node();
        let epoch = Instant::now();
        let phase = if store.detached() {
            Phase::CutOff
        } else {
            Phase::Joined
        };
        let (arrivals, rx) = mpsc::unbounded_channel();
        let r = Arc::new(Replicator {
            node,
            s: Mutex::new(State {
                outbox: Outbox::new(),
                fresh: BTreeMap::new(),
                sockets: vec![None; clusters],
                ready: vec![false; clusters],
                served: false,
                admission_quiet: vec![0; clusters],
                members: vec![None; clusters],
                rr: 0,
                // A process's requests number from a random base: answers to a
                // previous process's requests, still in flight to this node,
                // never match one of this process's.
                next_request: (rand::random::<u32>() as u64) << 32,
                catchups: BTreeMap::new(),
                answering: Answering::default(),
                heard: HashMap::new(),
                fetches: HashMap::new(),
                claimed: HashSet::new(),
                restarted: false,
                offered: false,
                gaps: GapWatch::default(),
                last_periodic: epoch,
                phase,
                rejoin: None,
                retry_rejoin: None,
                ack_dirty: true,
                max_members,
                generation,
                fault: None,
                drained: false,
                last_rejoin: None,
                coord_sent: None,
                governor: Governor::default(),
                voter_seen: None,
                stats: Stats::default(),
            }),
            send_wake: Notify::new(),
            receipts: Notify::new(),
            arrivals,
            arrivals_rx: Mutex::new(Some(rx)),
            room: Arc::new(Semaphore::new(ROOM_PERMITS)),
            received: AtomicU64::new(0),
            applied_upto: watch::channel(0).0,
            tasks: Mutex::new(Vec::new()),
            paxos: Mutex::new(paxos),
            epoch,
            member,
            commands: Mutex::new(VecDeque::new()),
            leases: broadcast::channel(EVENTS_KEPT).0,
            terms: Mutex::new(BTreeMap::new()),
            pulses: broadcast::channel(PULSES_KEPT).0,
            app_coord: Mutex::new(None),
            store,
        });
        let tap = Arc::downgrade(&r);
        r.store.set_tap(Box::new(move |oseq| {
            if let Some(r) = tap.upgrade() {
                r.tapped(oseq);
            }
        }));
        r
    }

    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The consensus clock: milliseconds since this replicator started.
    pub(super) fn now(&self) -> Millis {
        Millis(self.epoch.elapsed().as_millis() as u64)
    }

    pub fn node(&self) -> u64 {
        self.node
    }

    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    /// Start the background work: the sender, the receiver and the ticker.
    /// Called once, on the runtime the sockets run on.
    pub fn start(self: &Arc<Self>) {
        let Some(rx) = self
            .arrivals_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
            return;
        };
        let mut tasks = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
        tasks.push(tokio::spawn(self.clone().sender()));
        tasks.push(tokio::spawn(self.clone().receiver(rx)));
        tasks.push(tokio::spawn(self.clone().ticker()));
    }

    /// Stop replicating: the sender, the receiver and the ticker end, so
    /// nothing more is applied to the store or sent from it. What arrived and
    /// was not applied is fetched again by the next process's catch-up.
    pub fn shutdown(&self) {
        for task in self
            .tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            task.abort();
        }
    }

    /// Socket `c` was admitted.
    pub fn admitted(self: &Arc<Self>, c: usize, handle: SessionHandle) {
        let first = {
            let mut s = self.state();
            s.sockets[c] = Some(handle);
            s.ready[c] = false;
            s.admission_quiet[c] = 0;
            s.ack_dirty = true;
            if s.phase == Phase::CutOff && s.retry_rejoin.is_none() {
                s.phase = Phase::Rejoining;
            }
            let first = !s.restarted;
            s.restarted = true;
            first
        };
        if first {
            // What the last process wrote is the resync's to offer, not the
            // outbox's to push (§7.5).
            let me = self.clone();
            tokio::spawn(async move {
                let store = me.store.clone();
                if let Ok(written) = tokio::task::spawn_blocking(move || store.own_through()).await
                {
                    me.state().outbox.restart(written);
                    me.send_wake.notify_one();
                }
            });
        }
        self.start_catchup(c, Purpose::Admission, None);
        self.send_wake.notify_one();
    }

    /// Socket `c` closed: what was in flight on it is owed again, and its
    /// catch-ups are abandoned.
    pub fn closed(&self, c: usize) {
        let mut s = self.state();
        s.sockets[c] = None;
        s.ready[c] = false;
        s.members[c] = None;
        s.outbox.dropped(c as u32);
        let gone: Vec<u64> = s
            .catchups
            .iter()
            .filter(|(_, r)| r.socket == c)
            .map(|(id, _)| *id)
            .collect();
        for id in gone {
            s.drop_catchup(id);
        }
        drop(s);
        self.send_wake.notify_one();
    }

    /// A record arrived on socket `c`: queued to be applied once its bytes
    /// fit in [`ARRIVALS_CAP`] beside what is already waiting. The wait is the
    /// back pressure — the caller stops reading its socket for as long.
    pub async fn event(&self, c: usize, whole: Whole) {
        let room = self
            .room
            .clone()
            .acquire_many_owned(room_for(whole.bytes.len()))
            .await
            .expect("the arrivals budget is never closed");
        let seq = self.received.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.arrivals.send(Arrival::Event {
            c,
            whole,
            seq,
            room,
        });
    }

    /// Resolves once every record that arrived before this call is applied:
    /// what a request a gateway routed here waits for before it is served.
    /// The gateway relayed those records ahead of the request, so the request
    /// sees what they wrote (§6.4).
    pub fn applied_so_far(&self) -> impl Future<Output = ()> + Send + 'static {
        let target = self.received.load(Ordering::SeqCst);
        let mut applied = self.applied_upto.subscribe();
        async move {
            let _ = applied.wait_for(|upto| *upto >= target).await;
        }
    }

    /// A substrate, consensus or membership frame arrived on socket `c`.
    pub fn frame(self: &Arc<Self>, c: usize, frame: Frame) {
        match frame {
            Frame::Receipt(r) => {
                self.state().outbox.receipt(c as u32, r.through);
                self.receipts.notify_waiters();
            }
            Frame::Members(m) => {
                let mut s = self.state();
                s.members[c] = Some(m);
                // An answer for a member no gateway lists any more has nobody
                // to read it.
                let listed: HashSet<u64> = s
                    .members
                    .iter()
                    .flatten()
                    .flat_map(|m| m.members.iter().map(|x| x.node))
                    .collect();
                s.answering.retain(|r| listed.contains(&r));
            }
            Frame::Headers(headers) => {
                let _ = self.arrivals.send(Arrival::Headers { c, headers });
            }
            Frame::Done(done) => {
                let _ = self.arrivals.send(Arrival::Done { c, done });
            }
            Frame::Catchup(asked) => self.answer_catchup(c, asked),
            Frame::Fetch(f) => self.answer_fetch(c, f),
            Frame::Have(h) => self.offered(c, h),
            Frame::Pulse(p) => self.pulsed(p),
            f @ (Frame::Prepare(_)
            | Frame::Promise(_)
            | Frame::Nack(_)
            | Frame::Propose(_)
            | Frame::Accepted(_)
            | Frame::Release(_)) => self.paxos_frame(&f),
            _ => {}
        }
    }

    /// Whether socket `c`'s `CAPACITY` reports ready (§7.2): the member is in
    /// the group, and either that socket's admission catch-up has finished or
    /// the member has already served since its process started (see
    /// `State::served`).
    pub fn socket_ready(&self, c: usize) -> bool {
        let s = self.state();
        s.phase == Phase::Joined
            && s.fault.is_none()
            && !s.drained
            && (s.served || s.ready.get(c).copied().unwrap_or(false))
    }

    /// Whether any socket's admission catch-up has finished: the `catchup`
    /// loading step is done (§7.6).
    pub fn caught_up(&self) -> bool {
        self.state().ready.iter().any(|r| *r)
    }

    pub fn phase(&self) -> Phase {
        self.state().phase
    }

    /// Every catch-up in progress.
    pub fn catchups(&self) -> Vec<CatchupView> {
        self.state()
            .catchups
            .values()
            .map(|r| CatchupView {
                cluster: r.socket,
                purpose: r.purpose,
                progress: r.progress().clone(),
            })
            .collect()
    }

    pub fn view(&self) -> ReplicationView {
        let ready = self.ready_members();
        let s = self.state();
        let (unsent, in_flight) = s.outbox.counts();
        ReplicationView {
            phase: s.phase,
            generation: s.generation,
            unsent,
            in_flight,
            catchups: s
                .catchups
                .values()
                .map(|r| CatchupView {
                    cluster: r.socket,
                    purpose: r.purpose,
                    progress: r.progress().clone(),
                })
                .collect(),
            answering: s.answering.len(),
            stats: s.stats.clone(),
            quorum: QuorumView {
                max_members: s.max_members,
                ready,
                downgrade: s.governor.view(Instant::now()),
            },
        }
    }

    /// Ask for what exists after what this member holds, on socket `c` —
    /// addressed `to` a member, or to whichever the gateway picks.
    pub(super) fn start_catchup(self: &Arc<Self>, c: usize, purpose: Purpose, to: Option<u64>) {
        let me = self.clone();
        tokio::spawn(async move {
            let store = me.store.clone();
            let Ok(since) = tokio::task::spawn_blocking(move || store.applied()).await else {
                return;
            };
            let (request, handle) = {
                let mut s = me.state();
                let Some(handle) = s.sockets[c].clone() else {
                    return;
                };
                let request = s.next_request();
                s.catchups.insert(
                    request,
                    Requester::new(request, c, purpose, me.node, since.clone()),
                );
                s.heard.insert(request, Instant::now());
                s.stats.catchups += 1;
                (request, handle)
            };
            let _ = handle.control(Frame::Catchup(Catchup {
                request,
                reply_to: me.node,
                to,
                since,
            }));
        });
    }

    /// The member this replicates for, while it lives.
    pub(super) fn member(&self) -> Option<Arc<AppMember>> {
        self.member.upgrade()
    }

    /// Install what the app adds to `COORD`.
    pub fn set_app_coord(&self, report: AppCoordFn) {
        *self.app_coord.lock().unwrap_or_else(|e| e.into_inner()) = Some(report);
    }

    /// What the app reports now; nothing before it installs its report.
    pub(super) fn app_coord(&self) -> AppCoord {
        self.app_coord
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|f| f())
            .unwrap_or_default()
    }
}
