//! The thread that owns a substrate's redo log (`docs/substrate_lock_free.md`
//! §9).
//!
//! A [`SubstratePersistence`] is the active segment's append cursor and every
//! index of where records are on disk; one handle exists per directory, and one
//! thread holds it. Nothing else touches it: every other thread hands the owner
//! work over lock-free queues and, when it needs an answer, waits for that work
//! alone — never for a lock another thread holds across I/O.
//!
//! Three queues, served in this order:
//!
//! 1. **Calls** — a closure run on the log with its result handed back
//!    ([`PersistenceOwner::call`]): a cold read past the durable prefix, a
//!    keyed write and its commit, a reload's recovery reads. Served between any
//!    two writes, so a call waits for at most one write job.
//! 2. **Writes** — the appends a writer queues and forgets ([`WriteJob`]),
//!    taken in submission order and made durable by one group commit per batch.
//!    Each is appended and then indexed in the substrate before the next job, so
//!    a record is indexed before its segment can seal (R9).
//! 3. **Background** — maintenance, one batch per job
//!    ([`PersistenceOwner::call_behind`]), run only while nothing else waits.
//!
//! A durable record is read without the owner at all ([`SealedReader`]); what
//! the owner publishes for that — the active segment and its committed end —
//! is published by the log itself, as atomics.
//!
//! **Flow control is the one wait left.** The write queue is bounded by events
//! and by bytes; a producer facing a full queue backs off until the owner
//! drains it, holding nothing while it waits, so nothing waits behind it.

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe, Location};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Receiver, Select, Sender};

use super::identity::Minter;
use super::log_facts::LogFacts;
use super::sealed_reader::SealedReader;
use super::writer::{self, WriteJob};
use super::written_snapshots::WrittenSnapshots;
use super::SubstratePersistence;
use crate::substrate::SubstrateLock;

/// Max pending write EVENTS before an enqueue backs off. Sized to absorb a full
/// maintenance batch's worth of seal metadata without back-pressuring producers
/// in the common case.
const MAX_PENDING_EVENTS: usize = 16_384;

/// Max pending write BYTES before a bulk enqueue backs off — the second cap, so
/// a burst of large payloads can't balloon host RAM even under the event cap.
/// One cap for every bulk job ([`WriteJob::is_bulk`]).
///
/// **Sized to hold a cohort's recurrent snapshots.** A recurrent model seals its
/// whole state with every turn — ~65 MB on the Qwen3.6-35B-A3B, fixed per
/// sequence whatever the turn's length — and a decode cohort finishes together,
/// so it seals together. At 256 MiB the cap held under four of them, and the
/// scheduler's loop thread, which is the enqueuer, waited on the disk for the
/// rest of an eight-turn cohort, ~50 ms a turn. 512 MiB holds that cohort.
const MAX_PENDING_BYTES: u64 = 512 * 1024 * 1024;

/// Upper bound on write jobs drained before a group commit, so a sustained
/// burst still reaches durability — and the background queue — regularly.
const MAX_BATCH: usize = 512;

/// How long [`PersistenceOwner::wait_written`] waits before giving up on an
/// owner that has stopped.
const WAIT_WRITTEN: Duration = Duration::from_secs(10);

/// The longest a waiting producer or durability waiter sleeps between looks.
const BACKOFF_MAX: Duration = Duration::from_millis(5);

/// A call that held the owner this long is named in the log. The owner is one
/// thread, so everything else on the substrate waited for it: a long call reads
/// from outside as a member gone quiet, and takes an hour to find by inference.
const SLOW_CALL: Duration = Duration::from_millis(250);

/// [`SLOW_CALL`] for background work, which runs only while nothing waits and
/// is sized to hold the owner longer: maintenance's liveness scan takes half a
/// second on a large store, every sixteen, and a warning that fires all day
/// names nothing.
const SLOW_BACKGROUND: Duration = Duration::from_secs(2);

thread_local! {
    /// Whether this thread is a persistence owner — whose calls would wait for
    /// themselves.
    static IS_OWNER: Cell<bool> = const { Cell::new(false) };
}

/// A closure run on the log by the owner. Its result travels through the
/// caller's own slot; `done` says it ran, or how it panicked.
struct Call {
    run: Box<dyn FnOnce(&mut SubstratePersistence) + Send + 'static>,
    done: Sender<thread::Result<()>>,
    /// Who asked, for [`SLOW_CALL`]'s warning.
    caller: &'static Location<'static>,
    /// How long it may hold the owner before that warning.
    slow: Duration,
}

/// What the write queue carries.
enum Write {
    Job(WriteJob, u64),
    /// A call that must see every write queued before it — a full rewrite of
    /// the log ([`PersistenceOwner::call_after_writes`]).
    Call(Call),
    /// Drain everything queued, commit, ack, and stop the thread.
    Shutdown(Sender<()>),
}

/// A raw pointer to the caller's result slot, sent to the owner. Sound because
/// the caller waits for `done` before the slot goes out of scope, and the owner
/// writes the slot only inside the run.
struct Slot<R>(*mut Option<R>);
// SAFETY: the slot is written by exactly one thread (the owner, inside the run)
// while the caller is blocked waiting on `done`; `R: Send` makes moving the
// value across sound.
unsafe impl<R: Send> Send for Slot<R> {}

/// The write queue's flow-control counters, shared by the handle and the
/// owner thread.
#[derive(Default)]
struct Flow {
    pending_events: AtomicUsize,
    pending_bytes: AtomicU64,
    queued: AtomicU64,
    written: AtomicU64,
}

impl Flow {
    fn release(&self, bytes: u64) {
        self.pending_events.fetch_sub(1, Ordering::AcqRel);
        self.pending_bytes.fetch_sub(bytes, Ordering::AcqRel);
    }

    fn wrote(&self, bytes: u64) {
        self.release(bytes);
        self.written.fetch_add(1, Ordering::AcqRel);
    }
}

/// The handle every thread holds to a substrate's redo log. See the module
/// documentation.
pub struct PersistenceOwner {
    calls: Sender<Call>,
    writes: Sender<Write>,
    background: Sender<Call>,
    /// The owner's thread, taken once by [`Self::shutdown`].
    thread: AtomicPtr<JoinHandle<()>>,
    flow: Arc<Flow>,
    written_snapshots: Arc<WrittenSnapshots>,
    dir: PathBuf,
    read_only: bool,
    sealed: SealedReader,
    minter: Minter,
    facts: LogFacts,
}

impl std::fmt::Debug for PersistenceOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PersistenceOwner").field(&self.dir).finish()
    }
}

impl PersistenceOwner {
    /// Hand `persistence` to a thread of its own. `substrate` is where the
    /// owner indexes what its write jobs append.
    pub fn spawn(persistence: SubstratePersistence, substrate: Arc<SubstrateLock>) -> Arc<Self> {
        let (calls, call_rx) = channel::unbounded::<Call>();
        let (writes, write_rx) = channel::unbounded::<Write>();
        let (background, background_rx) = channel::unbounded::<Call>();
        let flow = Arc::new(Flow::default());
        let written_snapshots = Arc::new(WrittenSnapshots::default());
        let dir = persistence.dir().to_path_buf();
        let read_only = persistence.is_read_only();
        let sealed = persistence.sealed_reader();
        let minter = persistence.minter();
        let facts = persistence.facts();
        let owner = Arc::new(Self {
            calls,
            writes,
            background,
            thread: AtomicPtr::new(std::ptr::null_mut()),
            flow: Arc::clone(&flow),
            written_snapshots: Arc::clone(&written_snapshots),
            dir,
            read_only,
            sealed,
            minter,
            facts,
        });
        let handle = thread::Builder::new()
            .name("substrate-persistence-owner".into())
            .spawn(move || {
                IS_OWNER.with(|o| o.set(true));
                let mut loop_ = OwnerLoop {
                    persistence,
                    substrate,
                    calls: call_rx,
                    writes: write_rx,
                    background: background_rx,
                    flow,
                    snapshots: written_snapshots,
                };
                loop_.run();
            })
            .expect("spawn the persistence owner thread");
        owner
            .thread
            .store(Box::into_raw(Box::new(handle)), Ordering::Release);
        owner
    }

    /// The directory the log is in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Whether the log was opened read-only: no write job is ever appended.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Reads durable records without the owner — see [`SealedReader`].
    pub fn sealed_reader(&self) -> SealedReader {
        self.sealed.clone()
    }

    /// What the log publishes for reading without a call — see [`LogFacts`].
    pub fn facts(&self) -> &LogFacts {
        &self.facts
    }

    /// The substrate's identity minter, shared with the log.
    pub fn minter(&self) -> Minter {
        self.minter.clone()
    }

    /// The recurrent snapshots the owner has appended — see
    /// [`WrittenSnapshots`].
    pub fn written_snapshots(&self) -> &WrittenSnapshots {
        &self.written_snapshots
    }

    /// Run `f` on the log and return what it returns, waiting for it alone:
    /// it is served before any queued write it was not submitted behind, and
    /// before all background work.
    #[track_caller]
    pub fn call<R: Send>(&self, f: impl FnOnce(&mut SubstratePersistence) -> R + Send) -> R {
        self.submit(&self.calls, SLOW_CALL, f)
    }

    /// [`Self::call`] as background work: served only when no call and no
    /// write is waiting — maintenance, one batch at a time.
    #[track_caller]
    pub fn call_behind<R: Send>(&self, f: impl FnOnce(&mut SubstratePersistence) -> R + Send) -> R {
        self.submit(&self.background, SLOW_BACKGROUND, f)
    }

    /// [`Self::call`] in the write queue's order: served after every write
    /// queued before it. For work that must see every record whose in-memory
    /// change came first — a full rewrite of the log.
    #[track_caller]
    pub fn call_after_writes<R: Send>(
        &self,
        f: impl FnOnce(&mut SubstratePersistence) -> R + Send,
    ) -> R {
        self.submit_with(SLOW_CALL, f, |call| {
            self.writes
                .send(Write::Call(call))
                .map_err(|channel::SendError(w)| match w {
                    Write::Call(call) => call,
                    _ => unreachable!("the write sent back is the call"),
                })
        })
    }

    #[track_caller]
    fn submit<R: Send>(
        &self,
        queue: &Sender<Call>,
        slow: Duration,
        f: impl FnOnce(&mut SubstratePersistence) -> R + Send,
    ) -> R {
        self.submit_with(slow, f, |call| {
            queue.send(call).map_err(|channel::SendError(c)| c)
        })
    }

    #[track_caller]
    fn submit_with<R: Send>(
        &self,
        slow: Duration,
        f: impl FnOnce(&mut SubstratePersistence) -> R + Send,
        send: impl FnOnce(Call) -> Result<(), Call>,
    ) -> R {
        let caller = Location::caller();
        assert!(
            !IS_OWNER.with(Cell::get),
            "a persistence call from the owner's own thread would wait for itself"
        );
        let mut out: Option<R> = None;
        let slot = Slot(&mut out as *mut Option<R>);
        let run: Box<dyn FnOnce(&mut SubstratePersistence) + Send + '_> = Box::new(move |p| {
            let slot = slot;
            let r = f(p);
            // SAFETY: see `Slot` — the caller is blocked on `done` below.
            unsafe { *slot.0 = Some(r) };
        });
        // SAFETY: the closure borrows from this frame, and this frame does not
        // return until the owner has run and dropped it: `done` is sent after
        // the run returns (or unwinds), and a call the owner never takes is
        // dropped — with the closure — before `done`'s sender drops, which is
        // what ends the wait.
        let run: Box<dyn FnOnce(&mut SubstratePersistence) + Send + 'static> =
            unsafe { std::mem::transmute(run) };
        let (done, ran) = channel::bounded(1);
        let call = Call {
            run,
            done,
            caller,
            slow,
        };
        if let Err(call) = send(call) {
            drop(call);
            panic!("the persistence owner has stopped");
        }
        match ran.recv() {
            Ok(Ok(())) => out.expect("the owner ran the call"),
            Ok(Err(panic)) => panic::resume_unwind(panic),
            Err(_) => panic!("the persistence owner stopped before running the call"),
        }
    }

    /// Queue a durable write and return. Backs off while the queue is at
    /// either cap. Over a read-only log the job is dropped here.
    pub(crate) fn enqueue(&self, job: WriteJob) {
        if self.read_only {
            return;
        }
        let bytes = job.byte_size();
        self.reserve(bytes, job.is_bulk());
        // `queued` counts before the send so a ticket taken after this call
        // returns covers the job; a send that fails (the owner is gone,
        // post-shutdown) counts it written, since nothing will ever write it.
        self.flow.queued.fetch_add(1, Ordering::SeqCst);
        if self.writes.send(Write::Job(job, bytes)).is_err() {
            // The in-memory change stands, and on next boot the record is
            // simply absent.
            self.flow.wrote(bytes);
        }
    }

    /// Reserve room for one write job, backing off under either cap. A single
    /// in-flight job always makes progress, however large: an oversized
    /// payload must still drain, not deadlock.
    fn reserve(&self, bytes: u64, bulk: bool) {
        let flow = &self.flow;
        let mut sleep = Duration::from_micros(50);
        loop {
            let events = flow.pending_events.load(Ordering::Acquire);
            let held = flow.pending_bytes.load(Ordering::Acquire);
            let full = events >= MAX_PENDING_EVENTS
                || (bulk && events > 0 && held + bytes > MAX_PENDING_BYTES);
            if !full
                && flow
                    .pending_events
                    .compare_exchange(events, events + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                flow.pending_bytes.fetch_add(bytes, Ordering::AcqRel);
                return;
            }
            if full {
                thread::sleep(sleep);
                sleep = (sleep * 2).min(BACKOFF_MAX);
            }
        }
    }

    /// A position in the write queue: every job enqueued before this call is
    /// at or below it.
    pub fn ticket(&self) -> u64 {
        self.flow.queued.load(Ordering::SeqCst)
    }

    /// Wait until every write at or below `ticket` has been appended — or the
    /// owner has stopped, which gives up after [`WAIT_WRITTEN`]. Over a
    /// read-only log nothing is queued, so it returns at once.
    pub fn wait_written(&self, ticket: u64) {
        let deadline = Instant::now() + WAIT_WRITTEN;
        let mut sleep = Duration::from_micros(50);
        loop {
            let written = self.flow.written.load(Ordering::Acquire);
            if written >= ticket {
                return;
            }
            if Instant::now() >= deadline {
                tracing::warn!(
                    target: "candle_conversation::persistence::owner",
                    ticket,
                    written,
                    "gave up waiting for the persistence owner"
                );
                return;
            }
            thread::sleep(sleep);
            sleep = (sleep * 2).min(BACKOFF_MAX);
        }
    }

    /// Drain all pending writes, commit, and stop the thread. Idempotent.
    pub fn shutdown(&self) {
        let handle = self.thread.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if handle.is_null() {
            return;
        }
        // SAFETY: the pointer came from `Box::into_raw` in `spawn`, and the swap
        // above hands it to exactly one caller.
        let handle = unsafe { Box::from_raw(handle) };
        let (ack, acked) = channel::bounded::<()>(1);
        let sent = self.writes.send(Write::Shutdown(ack)).is_ok();
        // On the owner's own thread — the last handle dropped inside a call —
        // the owner stops after this job returns; waiting for it here would
        // wait for itself.
        if IS_OWNER.with(Cell::get) {
            return;
        }
        if sent {
            let _ = acked.recv();
        }
        let _ = handle.join();
    }
}

impl Drop for PersistenceOwner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The owner thread's state: the log, and the queues it serves.
struct OwnerLoop {
    persistence: SubstratePersistence,
    substrate: Arc<SubstrateLock>,
    calls: Receiver<Call>,
    writes: Receiver<Write>,
    background: Receiver<Call>,
    flow: Arc<Flow>,
    snapshots: Arc<WrittenSnapshots>,
}

impl OwnerLoop {
    fn run(&mut self) {
        loop {
            self.serve_calls();
            match self.serve_writes() {
                Served::Stop(ack) => {
                    self.stop(ack);
                    return;
                }
                Served::Wrote => continue,
                Served::Nothing => {}
            }
            if let Ok(call) = self.background.try_recv() {
                self.run_call(call);
                continue;
            }
            if !self.wait_for_work() {
                // Every handle is gone without a shutdown: flush the tail.
                self.commit();
                return;
            }
        }
    }

    fn serve_calls(&mut self) {
        while let Ok(call) = self.calls.try_recv() {
            self.run_call(call);
        }
    }

    fn run_call(&mut self, call: Call) {
        let Call {
            run,
            done,
            caller,
            slow,
        } = call;
        let p = &mut self.persistence;
        let started = Instant::now();
        let result = panic::catch_unwind(AssertUnwindSafe(move || run(p)));
        let held = started.elapsed();
        let _ = done.send(result);
        if held >= slow {
            tracing::warn!(
                target: "candle_conversation::persistence::owner",
                dir = %self.persistence.dir().display(),
                held_ms = held.as_millis() as u64,
                caller = %caller,
                "a call held the persistence owner: everything on this substrate waited for it"
            );
        }
    }

    /// Up to [`MAX_BATCH`] write jobs, a call served between any two, then one
    /// group commit.
    fn serve_writes(&mut self) -> Served {
        let mut wrote = 0usize;
        let mut stop = None;
        while wrote < MAX_BATCH {
            match self.writes.try_recv() {
                Ok(Write::Job(job, bytes)) => {
                    self.write(job, bytes);
                    wrote += 1;
                    self.serve_calls();
                }
                Ok(Write::Call(call)) => self.run_call(call),
                Ok(Write::Shutdown(ack)) => {
                    stop = Some(ack);
                    break;
                }
                Err(_) => break,
            }
        }
        if wrote > 0 {
            self.commit();
        }
        match (stop, wrote) {
            (Some(ack), _) => Served::Stop(ack),
            (None, 0) => Served::Nothing,
            (None, _) => Served::Wrote,
        }
    }

    fn write(&mut self, job: WriteJob, bytes: u64) {
        writer::process(job, &self.substrate, &mut self.persistence, &self.snapshots);
        self.flow.wrote(bytes);
    }

    fn commit(&mut self) {
        if let Err(e) = self.persistence.commit() {
            tracing::error!(
                target: "candle_conversation::persistence::owner",
                "persistence commit failed: {e}"
            );
        }
    }

    /// Block until any queue has work. `false` once every sender is gone.
    fn wait_for_work(&mut self) -> bool {
        let mut select = Select::new();
        let calls = select.recv(&self.calls);
        let writes = select.recv(&self.writes);
        let background = select.recv(&self.background);
        let ready = select.select();
        let index = ready.index();
        if index == calls {
            match ready.recv(&self.calls) {
                Ok(call) => self.run_call(call),
                Err(_) => return false,
            }
        } else if index == writes {
            match ready.recv(&self.writes) {
                Ok(Write::Job(job, bytes)) => {
                    self.write(job, bytes);
                    return self.commit_after_burst();
                }
                Ok(Write::Call(call)) => self.run_call(call),
                Ok(Write::Shutdown(ack)) => {
                    self.stop(ack);
                    return false;
                }
                Err(_) => return false,
            }
        } else if index == background {
            match ready.recv(&self.background) {
                Ok(call) => self.run_call(call),
                Err(_) => return false,
            }
        }
        true
    }

    /// A lone write woke the owner: take the burst behind it, then commit.
    /// `false` when a shutdown came in behind the burst — acknowledged here, so
    /// the owner stops.
    fn commit_after_burst(&mut self) -> bool {
        match self.serve_writes() {
            Served::Stop(ack) => {
                self.stop(ack);
                false
            }
            Served::Wrote => true,
            Served::Nothing => {
                self.commit();
                true
            }
        }
    }

    /// Stop: serve every call already queued — each is a caller blocked on its
    /// answer, and a queue that picked the shutdown first would otherwise drop
    /// it unanswered — then commit the tail and acknowledge the shutdown.
    fn stop(&mut self, ack: Sender<()>) {
        self.serve_calls();
        self.commit();
        let _ = ack.send(());
    }
}

enum Served {
    Stop(Sender<()>),
    Wrote,
    Nothing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::record::RecordType;
    use crate::substrate::Substrate;

    fn owner() -> (tempfile::TempDir, Arc<PersistenceOwner>, Arc<SubstrateLock>) {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let owner = PersistenceOwner::spawn(p, Arc::clone(&substrate));
        (dir, owner, substrate)
    }

    /// **A call returns its closure's value, and borrows from its caller.**
    #[test]
    fn a_call_returns_and_borrows() {
        let (_dir, owner, _) = owner();
        let mut seen = Vec::new();
        let n = owner.call(|p| {
            seen.push(p.segment_count());
            41 + 1
        });
        assert_eq!(n, 42);
        assert_eq!(seen.len(), 1);
    }

    /// **What is published and an audit's walk need no owner.** The owner is
    /// held for the whole test by a call that waits for them to finish: on the
    /// queue, either would wait for the other for ever. This is what stopped a
    /// member's upgrades and replication for the length of an audit.
    #[test]
    fn status_and_an_audit_walk_go_on_while_the_owner_is_held() {
        let (_dir, owner, substrate) = owner();
        let plan = {
            let view = substrate.read();
            owner.call_behind(|p| p.liveness_audit_plan(&view))
        };
        let (entered, held) = channel::bounded(1);
        let (release, finish) = channel::bounded::<()>(1);
        let holder = {
            let owner = Arc::clone(&owner);
            thread::spawn(move || {
                owner.call(move |_| {
                    entered.send(()).unwrap();
                    finish.recv().unwrap();
                })
            })
        };
        held.recv().unwrap();
        let facts = owner.facts().clone();
        let worker = thread::spawn(move || {
            assert!(!facts.out_of_space());
            let _ = facts.held_record_tags();
            assert_eq!(facts.dead_ratio(), None);
            plan.walk().unwrap().gone
        });
        let gone = worker.join().unwrap();
        assert!(gone.is_empty());
        release.send(()).unwrap();
        holder.join().unwrap();
    }

    /// **A call's panic reaches its caller, and the owner keeps serving.**
    #[test]
    fn a_panicking_call_unwinds_into_its_caller() {
        let (_dir, owner, _) = owner();
        let caught = panic::catch_unwind(AssertUnwindSafe(|| {
            owner.call(|_| -> u32 { panic!("inside the owner") })
        }));
        assert!(caught.is_err());
        assert_eq!(owner.call(|_| 7u32), 7, "the owner is still serving");
    }

    /// **Writes land in order, and a waiter sees them.** A burst of writes is
    /// appended and indexed, and `wait_written` returns once they are.
    #[test]
    fn queued_writes_are_written_and_waited_for() {
        let (_dir, owner, substrate) = owner();
        for i in 0..64u64 {
            owner.enqueue(WriteJob::ConvMeta {
                record: RecordType::DebugId,
                payload: format!("debug-{i}").into_bytes(),
                identity: RecordIdentity::NONE,
            });
        }
        let ticket = owner.ticket();
        assert_eq!(ticket, 64);
        owner.wait_written(ticket);
        assert!(owner.flow.written.load(Ordering::Acquire) >= 64);
        drop(substrate);
    }

    /// **Calls from many threads are each served once.**
    #[test]
    fn concurrent_calls_are_each_served() {
        let (_dir, owner, _) = owner();
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let owner = Arc::clone(&owner);
                thread::spawn(move || {
                    (0..100)
                        .map(|i| owner.call(move |_| t * 1000 + i))
                        .sum::<u64>()
                })
            })
            .collect();
        let total: u64 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        let expected: u64 = (0..8u64)
            .map(|t| (0..100u64).map(|i| t * 1000 + i).sum::<u64>())
            .sum();
        assert_eq!(total, expected);
    }

    /// **Background work waits for calls, not the other way round.** A call
    /// submitted while background jobs are queued is answered before they
    /// have all run.
    #[test]
    fn a_call_goes_ahead_of_queued_background_work() {
        let (_dir, owner, _) = owner();
        let ran = Arc::new(AtomicUsize::new(0));
        let behind: Vec<_> = (0..50)
            .map(|_| {
                let owner = Arc::clone(&owner);
                let ran = Arc::clone(&ran);
                thread::spawn(move || {
                    owner.call_behind(move |_| {
                        thread::sleep(Duration::from_millis(2));
                        ran.fetch_add(1, Ordering::SeqCst);
                    })
                })
            })
            .collect();
        thread::sleep(Duration::from_millis(10));
        owner.call(|_| ());
        let at_call = ran.load(Ordering::SeqCst);
        for b in behind {
            b.join().unwrap();
        }
        assert!(at_call < 50, "the call waited for every background job");
    }

    /// **Shutdown drains, and is idempotent.**
    #[test]
    fn shutdown_drains_and_is_idempotent() {
        let (_dir, owner, _) = owner();
        owner.enqueue(WriteJob::ConvMeta {
            record: RecordType::DebugId,
            payload: b"last".to_vec(),
            identity: RecordIdentity::NONE,
        });
        owner.shutdown();
        owner.shutdown();
        assert_eq!(owner.flow.written.load(Ordering::Acquire), 1);
    }

    /// **A shutdown queued behind a burst stops the owner.** The owner is idle
    /// when the burst arrives, so its first write wakes it and the rest —
    /// shutdown last — are taken behind that one. The owner acknowledged that
    /// shutdown and went back to waiting, and `shutdown` waited on the join
    /// for ever.
    #[test]
    fn a_shutdown_behind_a_burst_stops_the_owner() {
        let (_dir, owner, _) = owner();
        // Let the owner reach its idle wait.
        thread::sleep(Duration::from_millis(50));
        for i in 0..64u64 {
            owner.enqueue(WriteJob::ConvMeta {
                record: RecordType::DebugId,
                payload: format!("burst-{i}").into_bytes(),
                identity: RecordIdentity::NONE,
            });
        }
        let (stopped, is_stopped) = channel::bounded(1);
        let stopping = Arc::clone(&owner);
        thread::spawn(move || {
            stopping.shutdown();
            let _ = stopped.send(());
        });
        is_stopped
            .recv_timeout(Duration::from_secs(10))
            .expect("the owner stopped");
        assert_eq!(owner.flow.written.load(Ordering::Acquire), 64);
    }

    /// **A call queued when the shutdown is taken is still answered.** With a
    /// call and a shutdown both waiting, the owner's wait picks either; when it
    /// picks the shutdown, the call is served before the owner stops rather
    /// than dropped unanswered — which panicked its caller. Repeated, so both
    /// orders are taken.
    #[test]
    fn a_call_queued_beside_a_shutdown_is_answered() {
        for _ in 0..64 {
            let dir = tempfile::tempdir().unwrap();
            let mut substrate = Substrate::new();
            let persistence =
                SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
            let (call_tx, calls) = channel::unbounded::<Call>();
            let (write_tx, writes) = channel::unbounded::<Write>();
            let (_background_tx, background) = channel::unbounded::<Call>();
            let (done, answered) = channel::bounded(1);
            call_tx
                .send(Call {
                    run: Box::new(|_| {}),
                    done,
                    caller: Location::caller(),
                    slow: SLOW_CALL,
                })
                .unwrap();
            let (ack, _acked) = channel::bounded(1);
            write_tx.send(Write::Shutdown(ack)).unwrap();
            let mut owner = OwnerLoop {
                persistence,
                substrate: Arc::new(SubstrateLock::new(substrate)),
                calls,
                writes,
                background,
                flow: Arc::new(Flow::default()),
                snapshots: Arc::new(WrittenSnapshots::default()),
            };
            owner.wait_for_work();
            assert!(
                matches!(answered.try_recv(), Ok(Ok(()))),
                "the queued call went unanswered"
            );
        }
    }
}
