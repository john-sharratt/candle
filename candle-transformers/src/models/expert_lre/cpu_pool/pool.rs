//! The pool's threads: pinned workers that run one job at a time, with the
//! calling thread taking part (`docs/cpu_expert_compute.md` §5.6).
//!
//! A job is a closure every participant runs with its own index; the layer
//! schedule ([`super::layer`]) hands out its tasks inside that closure through
//! atomics, so the pool itself knows nothing of experts.
//!
//! **Warm through inference, parked outside it.** A worker that parked costs
//! tens of microseconds to wake on Windows — the last of a decode layer's
//! participants joined 50–100 µs in. So the pool has a warm window
//! ([`Warmth::warm`]), which the forward extends at every invocation by a
//! cool-down: inside it the workers on performance cores wait for their next
//! job in `umwait` on the job word (C0.1, light, and woken by the store at no
//! latency cost — measured, `handoff`) and never park; the first call after
//! an idle spell unparks them, ahead of the doorbell that follows. Outside it
//! — once inference has been quiet for the cool-down — every worker waits only
//! `spin` after its last job, then parks.
//!
//! **A job takes only the participants it needs** ([`Pool::run_on`]): the
//! caller and the first workers, in pool order — performance cores first. The
//! rest are neither handed the job nor woken by it. The slower cores are kept
//! out of inference altogether: the layer schedule runs on the first
//! [`Pool::fast`] participants (`layer::participants_for`), whatever its expert
//! count. A worker waiting is still a busy core to the package: on the laptop
//! fourteen of them held the CPU at ~75% of its base clock, so only the
//! performance cores stay warm.

use std::any::Any;
use std::cell::UnsafeCell;
use std::io;
use std::mem::transmute;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use tracing::warn;

use super::backoff::{Backoff, INTER_LAYER_SPINS, LOW_POWER_CYCLES};
use super::pin::{pin_current_thread, raise_priority};

/// A value alone on its cache line, so a worker waiting on one word never
/// shares the line with another's stores.
#[repr(align(64))]
#[derive(Default)]
struct Line<T>(T);

/// The job being run: a borrowed closure, valid only between the epoch that
/// published it and the moment `run` sees every worker finish.
type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    /// Bumped once per job; workers run the job when it moves past theirs.
    epoch: Line<AtomicU64>,
    /// Workers still running the current job.
    running: Line<AtomicUsize>,
    /// A shared run's claims ([`Pool::run_shared`]): the run's tag in the high
    /// word, [`CLOSED`] once the caller stops taking claimants, and the workers
    /// that claimed it and have not yet left in the rest.
    claims: Line<AtomicU64>,
    stop: AtomicBool,
    /// Written only by `run_on` before it bumps `epoch`; read by workers after
    /// they observe the bump.
    job: UnsafeCell<Option<Job>>,
    /// Parked workers' indices and handles, so `run_on` can wake the ones a
    /// job takes.
    parked: Mutex<Vec<(usize, Thread)>>,
    waits: PoolWaits,
    /// Workers on performance cores: `1..=fast`.
    fast: usize,
    /// The warm window's end, in ns after `origin`.
    warm_until: AtomicU64,
    origin: Instant,
    /// The first panic a worker's run of the current job raised, for `run_on`
    /// to raise again in the caller once every participant is done.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Shared {
    fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos() as u64
    }

    /// Whether worker `index` waits through the warm window rather than
    /// parking: a performance core's, inside it.
    fn warm_for(&self, index: usize) -> bool {
        index <= self.fast && self.now_ns() < self.warm_until.load(Ordering::Relaxed)
    }
}

// SAFETY: `job` is written by the one thread in `run` while no worker reads it
// (all have finished the previous job), and read by workers only after the
// release/acquire on `epoch` that follows the write.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// The pool: `workers` pinned threads, plus the caller of [`Pool::run`].
pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    /// Serialises `run`: one job at a time.
    gate: Mutex<()>,
}

/// How the pool's threads wait (`backoff`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolWaits {
    /// Outside the warm window, how long a worker waits for its next job
    /// before it parks.
    pub spin: Duration,
    /// A worker's hot spins before it waits for its next job in `umwait` — a
    /// job is most of a millisecond away between layers.
    pub worker_hot: u32,
    /// The caller's hot spins waiting for its participants' last tasks before
    /// it waits in `umwait`; `u32::MAX` spins throughout.
    pub caller_hot: u32,
}

impl PoolWaits {
    /// The waits the expert cache runs with, parking `spin` after a job
    /// outside the warm window. The caller spins throughout: its participants
    /// finish within microseconds of it, and a caller in `umwait` there
    /// measured the next layer's last participant joining 40–60 µs in rather
    /// than ~5 (Flash-Next decode, the laptop).
    pub fn new(spin: Duration) -> Self {
        Self {
            spin,
            worker_hot: INTER_LAYER_SPINS,
            caller_hot: u32::MAX,
        }
    }
}

/// The pool's warm window, held by whoever knows when inference runs — the
/// forward, through the service ([`super::service::CpuService::wake`]).
#[derive(Clone)]
pub struct Warmth(Arc<Shared>);

impl Warmth {
    /// Keep the performance-core workers waiting warm for `hold` from now,
    /// and wake the ones that parked: the next job's participants are awake
    /// when it lands.
    pub fn warm(&self, hold: Duration) {
        let s = &*self.0;
        let until = s.now_ns() + hold.as_nanos() as u64;
        if s.warm_until.fetch_max(until, Ordering::Relaxed) >= s.now_ns() {
            // Already warm: the workers are awake.
            return;
        }
        s.parked
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|(index, _)| *index <= s.fast)
            .for_each(|(_, t)| t.unpark());
    }

    /// Whether the window is open.
    pub fn is_warm(&self) -> bool {
        self.0.now_ns() < self.0.warm_until.load(Ordering::Relaxed)
    }
}

impl Pool {
    /// One worker per CPU set in `cpu_sets`, each pinned to it, the first
    /// `fast` of them on performance cores, waiting as `waits` says.
    pub fn new(cpu_sets: &[u32], fast: usize, waits: PoolWaits) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            epoch: Line(AtomicU64::new(0)),
            running: Line(AtomicUsize::new(0)),
            claims: Line(AtomicU64::new(0)),
            stop: AtomicBool::new(false),
            job: UnsafeCell::new(None),
            parked: Mutex::new(Vec::new()),
            waits,
            fast: fast.min(cpu_sets.len()),
            warm_until: AtomicU64::new(0),
            origin: Instant::now(),
            panic: Mutex::new(None),
        });
        let mut threads = Vec::with_capacity(cpu_sets.len());
        for (i, &cpu_set) in cpu_sets.iter().enumerate() {
            let shared = shared.clone();
            let handle = thread::Builder::new()
                .name(format!("cpu-expert-{i}"))
                .spawn(move || {
                    if let Err(e) = pin_current_thread(cpu_set) {
                        warn!("cpu expert worker {i}: not pinned to cpu set {cpu_set}: {e}");
                    }
                    if let Err(e) = raise_priority() {
                        warn!("cpu expert worker {i}: priority not raised: {e}");
                    }
                    worker(&shared, i + 1);
                })?;
            threads.push(handle);
        }
        Ok(Self {
            shared,
            threads,
            gate: Mutex::new(()),
        })
    }

    /// Participants in a job: the workers and the caller.
    pub fn participants(&self) -> usize {
        self.threads.len() + 1
    }

    /// Participants on performance cores: the caller and the fast workers.
    pub fn fast(&self) -> usize {
        self.shared.fast + 1
    }

    /// The warm window's handle.
    pub fn warmth(&self) -> Warmth {
        Warmth(self.shared.clone())
    }

    /// Runs `job(i)` on every participant — the caller is `0`, the workers
    /// `1..=workers` — and returns when all have returned.
    pub fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        self.run_on(self.participants(), job);
    }

    /// Runs `job(i)` on the first `participants` (at least the caller, at
    /// most [`Self::participants`]) — the caller is `0`, workers `1..` in
    /// pool order — and returns when all have returned. The workers past them
    /// are not woken.
    ///
    /// **A panic in any participant is raised here**, in the caller, once
    /// every participant has returned — never before, since the job is
    /// borrowed until then — the caller's own first. A worker that panicked
    /// goes on to wait for the next job.
    pub fn run_on(&self, participants: usize, job: &(dyn Fn(usize) + Sync)) {
        let _one_at_a_time = self.gate.lock().unwrap_or_else(|p| p.into_inner());
        let s = &*self.shared;
        let workers = participants.clamp(1, self.participants()) - 1;
        // SAFETY: every worker finished the previous job (`running` reached 0
        // before the previous `run_on` returned), so none reads `job` now. The
        // lifetime is erased; `run_on` does not return until every worker it
        // hands the job to is done with it.
        unsafe {
            *s.job.get() = Some(transmute::<*const (dyn Fn(usize) + Sync + '_), Job>(job));
        }
        s.running.0.store(workers, Ordering::Relaxed);
        trace::start(workers + 1);
        bump(&s.epoch.0, workers);
        if workers > 0 {
            s.parked
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .retain(|(index, t)| {
                    let taken = *index <= workers;
                    if taken {
                        t.unpark();
                    }
                    !taken
                });
        }
        trace::joined(0);
        let caller = catch_unwind(AssertUnwindSafe(|| job(0)));
        trace::left(0);
        let mut backoff = Backoff::new(s.waits.caller_hot);
        loop {
            let running = s.running.0.load(Ordering::Acquire);
            if running == 0 {
                break;
            }
            // SAFETY: the pool's own counter, a 64-bit word on the 64-bit
            // targets this runs on.
            unsafe {
                backoff.wait_u64(
                    s.running.0.as_ptr() as *const u64,
                    running as u64,
                    LOW_POWER_CYCLES,
                )
            };
        }
        let worker = s.panic.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Err(payload) = caller {
            resume_unwind(payload);
        }
        if let Some(payload) = worker {
            resume_unwind(payload);
        }
    }

    /// Runs `job` on the caller and on whichever of the first `participants`
    /// workers arrive while it runs, and returns once the caller's `job(0)` has
    /// returned and every worker that joined has left — **not** once every
    /// worker has.
    ///
    /// For a job whose participants share its work through atomics and take no
    /// part of it by index, so the caller returning means the work is done: a
    /// layer of experts (`layer::run_layer`), where a participant that arrives
    /// after the last task finds nothing and leaves. [`Self::run_on`] waits for
    /// every worker it hands a job to, and a worker that does not wake holds
    /// the whole layer: on the RTX 3090 (2026-10-18) worker 3 of seven never
    /// joined a run its six peers finished in 165 µs, and the device gave the
    /// layer up after a second with the pool still waiting on it. Windows CPU
    /// sets are soft affinity, and a pinned worker can be kept off its core.
    ///
    /// **A worker claims a shared run before it touches the job**, against the
    /// run's tag, and the caller closes the claims once its share is done. A
    /// worker arriving after that — or with another run's tag — sees the claim
    /// refused and never reads the job, which is borrowed only until this
    /// returns. Panics are raised as in [`Self::run_on`].
    pub fn run_shared(&self, participants: usize, job: &(dyn Fn(usize) + Sync)) {
        let _one_at_a_time = self.gate.lock().unwrap_or_else(|p| p.into_inner());
        let s = &*self.shared;
        let workers = participants.clamp(1, self.participants()) - 1;
        // SAFETY: as in `run_on` — no worker holds the previous job: every
        // claimant of the last shared run left before it returned, and a
        // worker refused a claim never read it.
        unsafe {
            *s.job.get() = Some(transmute::<*const (dyn Fn(usize) + Sync + '_), Job>(job));
        }
        let tag = run_tag(next_count(&s.epoch.0));
        s.claims.0.store(tag << 32, Ordering::Relaxed);
        trace::start(workers + 1);
        bump_shared(&s.epoch.0, workers);
        if workers > 0 {
            s.parked
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .retain(|(index, t)| {
                    let taken = *index <= workers;
                    if taken {
                        t.unpark();
                    }
                    !taken
                });
        }
        trace::joined(0);
        let caller = catch_unwind(AssertUnwindSafe(|| job(0)));
        trace::left(0);
        // No claimant after this one: the work is the caller's to have finished.
        s.claims.0.fetch_or(CLOSED, Ordering::AcqRel);
        let mut backoff = Backoff::new(s.waits.caller_hot);
        loop {
            let c = s.claims.0.load(Ordering::Acquire);
            if c & CLAIM_COUNT == 0 {
                break;
            }
            // SAFETY: the pool's own claims word, a live aligned `u64`.
            unsafe { backoff.wait_u64(s.claims.0.as_ptr() as *const u64, c, LOW_POWER_CYCLES) };
        }
        let worker = s.panic.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Err(payload) = caller {
            resume_unwind(payload);
        }
        if let Some(payload) = worker {
            resume_unwind(payload);
        }
    }
}

/// A shared run's claims word: set once the caller takes no more claimants.
const CLOSED: u64 = 1 << 31;
/// The claims word's count of workers in the run.
const CLAIM_COUNT: u64 = CLOSED - 1;

/// The tag a run published at epoch count `count` carries in its claims word.
fn run_tag(count: u64) -> u64 {
    count & 0xffff_ffff
}

/// The count the next [`bump`] of `epoch` publishes.
fn next_count(epoch: &AtomicU64) -> u64 {
    (epoch.load(Ordering::Relaxed) >> ACTIVE_BITS) + 1
}

/// Join the shared run published at `epoch`: `true` once claimed, and the
/// worker must leave it with [`leave_shared`]; `false` when the caller has
/// closed it or another run has replaced it — the job is not this worker's to
/// read.
fn claim_shared(s: &Shared, epoch: u64) -> bool {
    let tag = run_tag(epoch >> ACTIVE_BITS);
    let mut c = s.claims.0.load(Ordering::Acquire);
    loop {
        if c >> 32 != tag || c & CLOSED != 0 {
            return false;
        }
        match s
            .claims
            .0
            .compare_exchange_weak(c, c + 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => return true,
            Err(now) => c = now,
        }
    }
}

fn leave_shared(s: &Shared) {
    s.claims.0.fetch_sub(1, Ordering::Release);
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        bump(&self.shared.epoch.0, self.threads.len());
        for t in &self.threads {
            t.thread().unpark();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn worker(s: &Shared, index: usize) {
    // The epoch the pool was built at, not the one current when this thread
    // first runs: a job published before the thread was scheduled must still
    // be seen as new.
    let mut seen = 0;
    loop {
        seen = wait_for_epoch(s, seen, index);
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        #[cfg(test)]
        tests::hold_here(s, index);
        // A shared run is joined, not owed: one already closed, or replaced by
        // the next, is not this worker's to read, and the caller is not
        // waiting for it.
        let shared = seen & SHARED_RUN != 0;
        if shared && !claim_shared(s, seen) {
            continue;
        }
        // SAFETY: published before the epoch bump this thread acquired; valid
        // until `running` reaches 0, which this thread's decrement below gates
        // — or, for a shared run, until the claims count it joined falls to 0,
        // which its `leave_shared` below gates.
        let job = unsafe { (*s.job.get()).expect("epoch moved without a job") };
        // SAFETY: as above. A panic is kept for the caller and the count still
        // falls, so `run_on` neither waits for this worker forever nor returns
        // while the job is in use.
        trace::joined(index);
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(index) })) {
            s.panic
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get_or_insert(payload);
        }
        trace::left(index);
        if shared {
            leave_shared(s);
        } else {
            s.running.0.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Waits until the epoch moves past `seen` to a job that takes worker
/// `index` (or to the stop): hot spins, then `umwait` on the job word
/// (`backoff`) through the warm window if it is a performance core's, else for
/// `s.spin`, then parking. A job that does not take it moves `seen` on and
/// leaves the spin's clock running, so a worker left out of a run of jobs
/// parks. Returns the new epoch.
fn wait_for_epoch(s: &Shared, mut seen: u64, index: usize) -> u64 {
    let t0 = Instant::now();
    let mut backoff = Backoff::new(s.waits.worker_hot);
    loop {
        let now = s.epoch.0.load(Ordering::Acquire);
        if now != seen {
            if takes(now, index) {
                return now;
            }
            seen = now;
        }
        if t0.elapsed() < s.waits.spin || s.warm_for(index) {
            // SAFETY: the pool's own epoch word, which outlives its workers.
            unsafe { backoff.wait_u64(s.epoch.0.as_ptr(), seen, LOW_POWER_CYCLES) };
            continue;
        }
        let me = thread::current();
        s.parked
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((index, me.clone()));
        // Re-check after registering, so neither a bump nor a warm window
        // opened between the checks above and the push is slept through.
        let now = s.epoch.0.load(Ordering::Acquire);
        let job = now != seen && takes(now, index);
        if job || s.warm_for(index) {
            s.parked
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .retain(|(i, _)| *i != index);
            if job {
                return now;
            }
            continue;
        }
        thread::park_timeout(Duration::from_millis(50));
        s.parked
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|(i, _)| *i != index);
    }
}

/// **Where the run in flight is, per participant** — process-wide, as the
/// service's own state is (`service::state_line`), for a device that gave up
/// waiting on the pool to name. A layer the pool normally serves in under two
/// milliseconds once stood in its compute for over a second, while the device
/// rang twenty layers behind it, and nothing said whether a worker never
/// woke for the job or woke and ran slowly. Each participant stamps when it
/// joined the run and when it left; the line reads them against the run's
/// start. Relaxed stores, a handful per layer.
mod trace {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    /// Participants traced; a pool wider than this traces its first ones.
    pub(super) const SLOTS: usize = 64;

    static START_NS: AtomicU64 = AtomicU64::new(0);
    static PARTICIPANTS: AtomicUsize = AtomicUsize::new(0);
    static JOINED_NS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static LEFT_NS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

    /// Nanoseconds since the first trace, never zero once read.
    fn now_ns() -> u64 {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
    }

    pub(super) fn start(participants: usize) {
        PARTICIPANTS.store(participants, Ordering::Relaxed);
        START_NS.store(now_ns(), Ordering::Release);
    }

    pub(super) fn joined(index: usize) {
        if let Some(j) = JOINED_NS.get(index) {
            j.store(now_ns(), Ordering::Relaxed);
        }
    }

    pub(super) fn left(index: usize) {
        if let Some(l) = LEFT_NS.get(index) {
            l.store(now_ns(), Ordering::Relaxed);
        }
    }

    /// The run in flight, or the last one: its age, and each participant as
    /// `index:+joined/+left` µs from its start — `-` for a stamp older than
    /// the run, which is a participant not yet there or not yet done.
    pub(crate) fn line() -> String {
        let start = START_NS.load(Ordering::Acquire);
        if start == 0 {
            return "no pool run yet".to_string();
        }
        let n = PARTICIPANTS.load(Ordering::Relaxed).min(SLOTS);
        let at = |t: u64| {
            if t >= start {
                format!("+{}", (t - start) / 1_000)
            } else {
                "-".to_string()
            }
        };
        let parts: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "{i}:{}/{}",
                    at(JOINED_NS[i].load(Ordering::Relaxed)),
                    at(LEFT_NS[i].load(Ordering::Relaxed))
                )
            })
            .collect();
        format!(
            "pool run started {} µs ago, participants joined/left µs [{}]",
            (now_ns().saturating_sub(start)) / 1_000,
            parts.join(" ")
        )
    }
}

/// The pool's run in flight, per participant — see `trace`.
pub(crate) fn trace_line() -> String {
    trace::line()
}

/// Bits of the epoch word holding the workers its job takes and its mode; the
/// count above them. One word, so a worker reads a job's epoch, its reach and
/// whether it is a shared run together — read apart, a worker could pair one
/// run's epoch with the next run's mode.
const ACTIVE_BITS: u32 = 16;
/// Set in the epoch word of a shared run ([`Pool::run_shared`]).
const SHARED_RUN: u64 = 1 << (ACTIVE_BITS - 1);
/// The epoch word's workers field.
const WORKERS: u64 = SHARED_RUN - 1;

/// Publish the next job, taking workers `1..=workers` (the stop takes all).
fn bump(epoch: &AtomicU64, workers: usize) {
    publish(epoch, workers, 0);
}

/// Publish the next job as a shared run, taking workers `1..=workers`.
fn bump_shared(epoch: &AtomicU64, workers: usize) {
    publish(epoch, workers, SHARED_RUN);
}

fn publish(epoch: &AtomicU64, workers: usize, mode: u64) {
    let count = next_count(epoch);
    let workers = (workers as u64).min(WORKERS);
    // Only the gate's holder (or the drop) bumps, so load-then-store races
    // with no other writer.
    epoch.store((count << ACTIVE_BITS) | mode | workers, Ordering::Release);
}

/// Whether the job an epoch word publishes takes worker `index`.
fn takes(epoch: u64, index: usize) -> bool {
    index as u64 <= epoch & WORKERS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};
    use std::sync::atomic::AtomicU32;

    /// The worker a test keeps off its next job (0: none), standing in for one
    /// the OS has kept off its core — of the pool whose shared state is at
    /// [`HOLD_POOL`], so no other test's pool is held.
    static HOLD: AtomicUsize = AtomicUsize::new(0);
    static HOLD_POOL: AtomicUsize = AtomicUsize::new(0);

    /// Where a worker waits while [`HOLD`] names it: after it saw the epoch,
    /// before it claims or touches the job — the late worker's window.
    pub(super) fn hold_here(s: &Shared, index: usize) {
        let me = s as *const Shared as usize;
        while HOLD_POOL.load(Ordering::Acquire) == me && HOLD.load(Ordering::Acquire) == index {
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn hold(pool: &Pool, index: usize) {
        HOLD.store(index, Ordering::Release);
        HOLD_POOL.store(&*pool.shared as *const Shared as usize, Ordering::Release);
    }

    fn release() {
        HOLD_POOL.store(0, Ordering::Release);
        HOLD.store(0, Ordering::Release);
    }

    /// **A worker that does not arrive does not hold a shared run.** The other
    /// participants take the work, the run returns, and the late worker, once
    /// it does arrive, finds the run closed and never touches the job — it is
    /// borrowed only until the run returns. The next run takes it as ever.
    #[test]
    fn a_late_worker_neither_holds_nor_touches_a_shared_run() {
        let pool = small_pool();
        let workers = pool.participants() - 1;
        let late = 2;
        let first = AtomicU32::new(0);
        let late_ran_first = AtomicBool::new(false);
        hold(&pool, late);
        let t0 = Instant::now();
        pool.run_shared(pool.participants(), &|i| {
            first.fetch_add(1, Ordering::Relaxed);
            if i == late {
                late_ran_first.store(true, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(20));
        });
        let took = t0.elapsed();
        assert!(
            took < Duration::from_millis(500),
            "the run waited {took:?} for a worker that never arrived"
        );
        // The caller always runs it; the punctual workers join if they make it
        // before the caller leaves — not guaranteed beside other tests' pools.
        let ran = first.load(Ordering::Relaxed);
        assert!(
            (1..=workers as u32).contains(&ran),
            "{ran} ran: the caller, and at most the punctual workers"
        );
        // The late worker arrives after the run has closed.
        release();
        thread::sleep(Duration::from_millis(20));
        assert!(
            !late_ran_first.load(Ordering::Relaxed),
            "it touched a closed run's job"
        );
        // The late worker takes part in runs again: an owed run waits for it.
        let second = AtomicU32::new(0);
        pool.run_on(pool.participants(), &|_| {
            second.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(
            second.load(Ordering::Relaxed),
            pool.participants() as u32,
            "the next run took every participant again"
        );
    }

    /// Shared and owed runs alternate on one pool: a worker reads a run's mode
    /// from the epoch it saw, so neither kind is taken for the other.
    #[test]
    fn shared_and_owed_runs_alternate() {
        let pool = small_pool();
        for round in 0..200 {
            let n = AtomicU32::new(0);
            if round % 2 == 0 {
                pool.run_shared(pool.participants(), &|_| {
                    n.fetch_add(1, Ordering::Relaxed);
                });
                assert!(n.load(Ordering::Relaxed) >= 1, "round {round}");
            } else {
                pool.run_on(pool.participants(), &|_| {
                    n.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(
                    n.load(Ordering::Relaxed),
                    pool.participants() as u32,
                    "round {round}: an owed run is run by every participant"
                );
            }
        }
    }

    fn small_pool() -> Pool {
        let cores = performance_cores(&physical_cores().unwrap());
        let sets: Vec<u32> = cores.iter().skip(1).take(3).map(|c| c.primary()).collect();
        Pool::new(&sets, sets.len(), PoolWaits::new(Duration::from_millis(1))).unwrap()
    }

    fn parked(pool: &Pool) -> Vec<usize> {
        let mut v: Vec<usize> = pool
            .shared
            .parked
            .lock()
            .unwrap()
            .iter()
            .map(|(i, _)| *i)
            .collect();
        v.sort_unstable();
        v
    }

    /// Inside the warm window the fast workers wait instead of parking — the
    /// ones that had parked are woken by opening it — and a slower one parks
    /// as ever; once the window has cooled down, every worker parks, and the
    /// next job still runs on all of them.
    #[test]
    fn the_warm_window_keeps_the_fast_workers_up_then_cools() {
        let cores = performance_cores(&physical_cores().unwrap());
        let sets: Vec<u32> = cores.iter().skip(1).take(3).map(|c| c.primary()).collect();
        // Workers 1 and 2 fast, worker 3 not.
        let pool = Pool::new(&sets, 2, PoolWaits::new(Duration::from_millis(1))).unwrap();
        let settle = |f: &dyn Fn(&[usize]) -> bool, what: &str| {
            let t0 = Instant::now();
            while !f(&parked(&pool)) {
                assert!(
                    t0.elapsed() < Duration::from_secs(2),
                    "{what}: {:?}",
                    parked(&pool)
                );
                thread::sleep(Duration::from_millis(1));
            }
        };
        settle(&|p| p == [1, 2, 3], "idle: all parked");
        let warmth = pool.warmth();
        warmth.warm(Duration::from_millis(150));
        assert!(warmth.is_warm());
        settle(&|p| p == [3], "warm: only the slower worker parked");
        thread::sleep(Duration::from_millis(40));
        assert_eq!(parked(&pool), [3], "still warm 40 ms in");
        settle(&|p| p == [1, 2, 3], "cooled: all parked");
        assert!(!warmth.is_warm());
        let count = AtomicU32::new(0);
        pool.run(&|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), pool.participants() as u32);
        assert_eq!(pool.fast(), 3);
    }

    /// A panic in a worker, or in the caller, is raised in the caller once
    /// every participant has returned — with its own message, the caller's
    /// first when both panic — and the pool runs the next job on every
    /// participant as before.
    #[test]
    fn a_panicking_job_is_raised_in_the_caller_and_the_pool_runs_on() {
        let pool = small_pool();
        let n = pool.participants();
        let finished = AtomicU32::new(0);
        let message = |r: thread::Result<()>| -> String {
            let p = r.expect_err("the panic reaches the caller");
            p.downcast_ref::<&str>()
                .map(|s| s.to_string())
                .unwrap_or_else(|| p.downcast_ref::<String>().cloned().unwrap_or_default())
        };
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|i| {
                if i == 2 {
                    panic!("worker two");
                }
                thread::sleep(Duration::from_millis(5));
                finished.fetch_add(1, Ordering::Relaxed);
            })
        }));
        assert_eq!(message(r), "worker two");
        assert_eq!(
            finished.load(Ordering::Relaxed),
            n as u32 - 1,
            "the others ran to the end before the caller saw it"
        );
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|i| {
                if i <= 1 {
                    panic!("participant {i}");
                }
            })
        }));
        assert_eq!(message(r), "participant 0", "the caller's own first");
        let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        pool.run(&|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// Every participant runs every job exactly once, with its own index.
    #[test]
    fn every_participant_runs_each_job_once() {
        let pool = small_pool();
        let n = pool.participants();
        for _ in 0..100 {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            pool.run(&|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    /// A job run on `n` participants reaches the caller and the first `n - 1`
    /// workers, once each, and no other — through jobs of every width back to
    /// back, so a worker left out of one is taken by the next that reaches it.
    #[test]
    fn a_job_reaches_only_its_participants() {
        let pool = small_pool();
        let all = pool.participants();
        for round in 0..50 {
            for n in [1, 2, all, 3, 1, all] {
                let n = n.min(all);
                let hits: Vec<AtomicU32> = (0..all).map(|_| AtomicU32::new(0)).collect();
                pool.run_on(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                let got: Vec<u32> = hits.iter().map(|h| h.load(Ordering::Relaxed)).collect();
                let want: Vec<u32> = (0..all).map(|i| u32::from(i < n)).collect();
                assert_eq!(got, want, "round {round}, {n} participants");
            }
        }
    }

    /// Workers a run of narrow jobs leaves out park past their spin (1 ms),
    /// and a wide job after 20 ms still wakes them.
    #[test]
    fn workers_left_out_park_and_wake() {
        let pool = small_pool();
        let count = AtomicU32::new(0);
        for _ in 0..20 {
            pool.run_on(1, &|_| {
                count.fetch_add(1, Ordering::Relaxed);
            });
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(count.load(Ordering::Relaxed), 20);
        thread::sleep(Duration::from_millis(20));
        let n = pool.participants();
        pool.run(&|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 20 + n as u32);
    }

    /// The epoch word carries a job's reach with its count.
    #[test]
    fn the_epoch_word_carries_its_reach() {
        let e = AtomicU64::new(0);
        bump(&e, 3);
        let w = e.load(Ordering::Relaxed);
        assert_eq!(w, (1 << ACTIVE_BITS) | 3);
        assert!(takes(w, 3) && !takes(w, 4));
        bump(&e, 0);
        let w2 = e.load(Ordering::Relaxed);
        assert_eq!(w2, 2 << ACTIVE_BITS);
        assert!(!takes(w2, 1));
    }

    /// Jobs after the workers have parked still run: the spin is 1 ms, the gap
    /// 20 ms.
    #[test]
    fn parked_workers_wake() {
        let pool = small_pool();
        let count = AtomicU32::new(0);
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(20));
            pool.run(&|_| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(
            count.load(Ordering::Relaxed),
            3 * pool.participants() as u32
        );
    }
}
