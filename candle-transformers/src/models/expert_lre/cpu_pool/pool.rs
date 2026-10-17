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
        let caller = catch_unwind(AssertUnwindSafe(|| job(0)));
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
        // SAFETY: published before the epoch bump this thread acquired; valid
        // until `running` reaches 0, which this thread's decrement below gates.
        let job = unsafe { (*s.job.get()).expect("epoch moved without a job") };
        // SAFETY: as above. A panic is kept for the caller and the count still
        // falls, so `run_on` neither waits for this worker forever nor returns
        // while the job is in use.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(index) })) {
            s.panic
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get_or_insert(payload);
        }
        s.running.0.fetch_sub(1, Ordering::Release);
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

/// Bits of the epoch word holding the workers its job takes; the count above
/// them. One word, so a worker reads a job's epoch and its reach together.
const ACTIVE_BITS: u32 = 16;

/// Publish the next job, taking workers `1..=workers` (the stop takes all).
fn bump(epoch: &AtomicU64, workers: usize) {
    let count = (epoch.load(Ordering::Relaxed) >> ACTIVE_BITS) + 1;
    let workers = (workers as u64).min((1 << ACTIVE_BITS) - 1);
    // Only the gate's holder (or the drop) bumps, so load-then-store races
    // with no other writer.
    epoch.store((count << ACTIVE_BITS) | workers, Ordering::Release);
}

/// Whether the job an epoch word publishes takes worker `index`.
fn takes(epoch: u64, index: usize) -> bool {
    index as u64 <= epoch & ((1 << ACTIVE_BITS) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};
    use std::sync::atomic::AtomicU32;

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
