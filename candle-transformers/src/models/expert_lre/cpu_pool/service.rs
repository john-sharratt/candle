//! The pool's service thread: the host side of the CPU class
//! (`docs/cpu_expert_compute.md` §5.6, §5.9, §5.13).
//!
//! One thread, pinned, waits on the doorbell; each ring is one layer's CPU
//! work. It reads the work slot and the row ring, builds each CPU expert in
//! place from the host addresses bucketize published (pad slot images, which
//! the host and the device address alike), runs the layer on the pool
//! ([`super::layer`]) with itself as the pool's caller, writes the result
//! rows, and publishes `done`. A cold expert — snapshot entries 0 — is run
//! from the warm tier's pageable copy at once if it has one, else once the
//! stager publishes it into the pad (§5.4, [`super::cold`]), after the experts
//! already in memory. It makes no CUDA call.
//!
//! **Waking.** Inside a forward a layer arrives every few hundred
//! microseconds. The forward calls [`CpuService::wake`] ahead of every
//! invocation, which holds the thread and the pool's performance-core workers
//! warm for a cool-down (`backoff::COOL_DOWN`): after each layer the thread
//! spins a few times, then waits on the doorbell's line in `umwait`, which a
//! device write wakes at no latency cost (measured, `handoff`). Once the
//! cool-down passes it waits `spin` more, then parks, and the next `wake`
//! unparks it ahead of the doorbell.
//!
//! **Liveness.** The thread sets the mapped `alive` word on start and a drop
//! guard clears it on any exit, panic included: a device item waiting for this
//! pool's results sees the word fall and claims the fault word at once,
//! instead of spinning to its limit.

use std::hint::spin_loop;
use std::ptr::copy_nonoverlapping;
use std::slice;
use std::sync::atomic::{fence, AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use candle::{Error, Result};
use tracing::{info, warn};

use super::backoff::{Backoff, COOL_DOWN, INTER_LAYER_SPINS, LOW_POWER_CYCLES};
use super::bandwidth::{read_rate, touched_buffer};
use super::calibrate::calibrate;
use super::cold::{ColdSource, Published};
use super::kernel::act::{ActTiles, TILE};
use super::kernel::expert::Expert;
use super::kernel::ko::{Ko, KoMatrix};
use super::kernel::simd::Isa;
use super::layer::{run_layer, ExpertWork, LayerWork};
use super::online::{weight, LateCause, Observation, Online};
use super::params::{shared, BUCKETS};
use super::pin::{pin_current_thread, raise_priority};
use super::pool::{Pool, PoolWaits, Warmth};
use super::rings::{q8a1024_bytes, CpuRings, WorkExpert};
use super::stats::{record as record_phases, record_device_wait, Served};

/// Layers every pool in the process has served: what a gate or the admin
/// view reads to know the CPU class actually ran.
static SERVED_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Assignments — rows — those layers held.
static SERVED_ROWS: AtomicU64 = AtomicU64::new(0);

/// Layers every pool in the process has served.
pub fn served_total() -> u64 {
    SERVED_TOTAL.load(Ordering::Relaxed)
}

/// Experts those layers held.
static SERVED_EXPERTS: AtomicU64 = AtomicU64::new(0);
/// Doorbells whose work slot did not yet name their invocation when first
/// read, and had to be read again.
static STALE_SLOTS: AtomicU64 = AtomicU64::new(0);

/// How long a doorbell's work slot may take to name its invocation before
/// the layer is refused: far past any write's flight, far inside the device's
/// spin limit.
const SLOT_WAIT: Duration = Duration::from_millis(5);

/// Doorbells every pool in the process found its work slot not yet theirs.
pub fn stale_slots() -> u64 {
    STALE_SLOTS.load(Ordering::Relaxed)
}

/// Assignments every pool in the process has computed.
pub fn served_rows() -> u64 {
    SERVED_ROWS.load(Ordering::Relaxed)
}

/// Experts every pool in the process has computed, a layer's each counted
/// once.
pub fn served_experts() -> u64 {
    SERVED_EXPERTS.load(Ordering::Relaxed)
}

/// Nanoseconds those layers took, doorbell seen to `done` published.
static SERVED_NS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds every pool in the process has spent serving layers.
pub fn served_ns() -> u64 {
    SERVED_NS.load(Ordering::Relaxed)
}

/// Where the service is, for a device that gave up waiting on it to name —
/// process-wide, like the counters above, so every pool in the process writes
/// it: the phase in the top byte, the last doorbell it took in the low word.
static STATE: AtomicU64 = AtomicU64::new(0);
/// The last layer the service refused or gave up — the reason in the top
/// byte, its doorbell in the low word — and how many it has.
static LAST_REFUSAL: AtomicU64 = AtomicU64::new(0);
static REFUSALS: AtomicU64 = AtomicU64::new(0);

const PHASES: [&str; 8] = [
    "not started",
    "calibrating",
    "waiting for a doorbell",
    "reading the work slot",
    "computing",
    "serving cold experts",
    "publishing",
    "stopped",
];
const PHASE_CALIBRATING: u64 = 1;
const PHASE_WAITING: u64 = 2;
const PHASE_PARSING: u64 = 3;
const PHASE_COMPUTING: u64 = 4;
const PHASE_COLD: u64 = 5;
const PHASE_PUBLISHING: u64 = 6;
const PHASE_STOPPED: u64 = 7;

const REASONS: [&str; 7] = [
    "none",
    "the work slot never named its invocation",
    "work for a row outside the table (the service stopped)",
    "a ring row named an input row past the inputs",
    "cold experts with no live table",
    "a cold expert was given up or published to device memory",
    "a pinned expert named memory outside the pinned ranges",
];
const REFUSED_STALE: u64 = 1;
const REFUSED_ROW: u64 = 2;
const REFUSED_INPUT: u64 = 3;
const REFUSED_NO_TABLE: u64 = 4;
const REFUSED_COLD: u64 = 5;
const REFUSED_PINNED: u64 = 6;

fn set_phase(phase: u64, seq: u64) {
    STATE.store((phase << 56) | (seq & 0xffff_ffff), Ordering::Relaxed);
}

fn note_refusal(reason: u64, seq: u64) {
    LAST_REFUSAL.store((reason << 56) | (seq & 0xffff_ffff), Ordering::Relaxed);
    REFUSALS.fetch_add(1, Ordering::Relaxed);
}

/// The service's state in words: its phase and last doorbell, and its last
/// refusal. What a device that gave up on the pool reports beside its error.
pub fn state_line() -> String {
    let s = STATE.load(Ordering::Relaxed);
    let r = LAST_REFUSAL.load(Ordering::Relaxed);
    fn name(t: &[&'static str], v: u64) -> &'static str {
        t.get((v >> 56) as usize).copied().unwrap_or("?")
    }
    format!(
        "service {} (last doorbell {}); {} layers refused, the last {} ({}); {} served",
        name(&PHASES, s),
        s & 0xffff_ffff,
        REFUSALS.load(Ordering::Relaxed),
        r & 0xffff_ffff,
        name(&REASONS, r),
        SERVED_TOTAL.load(Ordering::Relaxed),
    )
}

/// The buffer the pool's DRAM read rate is measured over at load: far past
/// any last-level cache, small enough not to matter to the host's RAM.
const CALIBRATION_BYTES: usize = 512 << 20;
/// Each window of that measurement.
const CALIBRATION_WINDOW: Duration = Duration::from_millis(200);

/// One MoE row's expert geometry, as the pool reads a slot image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowShape {
    pub hidden: usize,
    pub ff: usize,
    pub ko: Ko,
    /// Bytes of each projection in the image: gate, up, down.
    pub bytes: [usize; 3],
}

/// Where the service thread and its pool run, as CPU sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolCores {
    /// The service thread's: the participant every job starts on.
    pub caller: u32,
    /// The workers', in pool order — performance cores first.
    pub workers: Vec<u32>,
    /// Of `workers`, the first this many are on performance cores, as
    /// `caller` is.
    pub fast_workers: usize,
}

/// The running service.
pub struct CpuService {
    rings: Arc<CpuRings>,
    sum_by_amax: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    woken: Arc<AtomicBool>,
    served: Arc<AtomicU64>,
    cold: Arc<OnceLock<ColdSource>>,
    warmth: Warmth,
    thread: Option<JoinHandle<()>>,
}

impl CpuService {
    /// Starts the thread and its pool on `cores`, serving `rings` for rows
    /// shaped by `shapes` (indexed by MoE row). Outside the warm window
    /// ([`Self::wake`]), `spin` is how long it waits on the doorbell after a
    /// layer before it parks, `worker_spin` how long a worker waits for its
    /// next job.
    ///
    /// When the process's split is enabled (`params::shared`), the thread first
    /// calibrates on its pool — the pool's DRAM read rate, then the kernel at
    /// every count bucket — and publishes the measured costs; `link_bps` is the
    /// link's measured rate. It does so once the first forward wakes it
    /// ([`Self::wake`]): the model is still loading when the service starts,
    /// and a measurement taken beside the loader is of the loader. Until then
    /// the split stays off and the engine runs all-GPU.
    pub fn start(
        rings: Arc<CpuRings>,
        shapes: Vec<RowShape>,
        cores: PoolCores,
        spin: Duration,
        worker_spin: Duration,
        link_bps: f64,
    ) -> Result<Self> {
        let PoolCores {
            caller,
            workers,
            fast_workers,
        } = cores;
        let sum_by_amax = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let woken = Arc::new(AtomicBool::new(false));
        let served = Arc::new(AtomicU64::new(0));
        let cold = Arc::new(OnceLock::new());
        let first_wake = woken.clone();
        let pool =
            Pool::new(&workers, fast_workers, PoolWaits::new(worker_spin)).map_err(Error::wrap)?;
        let warmth = pool.warmth();
        let ctx = Ctx {
            rings: rings.clone(),
            shapes,
            sum_by_amax: sum_by_amax.clone(),
            stop: stop.clone(),
            served: served.clone(),
            cold: cold.clone(),
            spin,
            warmth: warmth.clone(),
        };
        let thread = thread::Builder::new()
            .name("cpu-expert-service".into())
            .spawn(move || {
                if let Err(e) = pin_current_thread(caller) {
                    warn!("cpu expert service: not pinned to cpu set {caller}: {e}");
                }
                if let Err(e) = raise_priority() {
                    warn!("cpu expert service: priority not raised: {e}");
                }
                // Alive before calibration publishes the costs that turn the
                // split on: the first layer handed over may follow at once.
                ctx.rings.set_alive(true);
                let _guard = AliveGuard(&ctx.rings);
                let isa = Isa::detect();
                let params = shared();
                let enabled = params.lock().map(|p| p.enabled).unwrap_or(false);
                // The first forward: the load is over.
                while !first_wake.load(Ordering::Acquire) {
                    if ctx.stop.load(Ordering::Acquire) {
                        return;
                    }
                    thread::park_timeout(Duration::from_millis(100));
                }
                if enabled {
                    set_phase(PHASE_CALIBRATING, 0);
                    if let Some(shape) = ctx.shapes.first() {
                        // The cores inference runs on: the performance cores'.
                        let mut sets = vec![caller];
                        sets.extend_from_slice(&workers[..pool.fast() - 1]);
                        match read_rate(
                            touched_buffer(CALIBRATION_BYTES),
                            &sets,
                            CALIBRATION_WINDOW,
                            2,
                        ) {
                            Ok(dram) => {
                                let m = calibrate(&pool, isa, shape, dram.bytes_per_s, link_bps);
                                info!(
                                    copy_us = m.copy_ns / 1e3,
                                    cpu_1_us = m.cpu_n_ns[0] / 1e3,
                                    cpu_32_us = m.cpu_n_ns[BUCKETS.len() - 1] / 1e3,
                                    dram_gbps = dram.gb_per_s(),
                                    workers = workers.len(),
                                    ?isa,
                                    "cpu experts: calibrated"
                                );
                                if let Ok(mut p) = params.lock() {
                                    p.measured = Some(m);
                                }
                            }
                            Err(e) => warn!("cpu expert service: DRAM rate not measured: {e}"),
                        }
                    }
                }
                serve(&ctx, &pool, isa);
            })
            .map_err(Error::wrap)?;
        Ok(Self {
            rings,
            sum_by_amax,
            stop,
            woken,
            served,
            cold,
            warmth,
            thread: Some(thread),
        })
    }

    pub fn rings(&self) -> &Arc<CpuRings> {
        &self.rings
    }

    /// Where the pool waits on the cold experts bucketize hands it (§5.4):
    /// the dispatch's live table, abort and fault words. Set once, before the
    /// first forward; a layer with cold experts handed over before it is
    /// refused.
    pub(crate) fn attach_cold(&self, src: ColdSource) {
        if self.cold.set(src).is_err() {
            warn!("cpu expert service: cold source attached twice — the first stands");
        }
    }

    /// The MoE input operand's sum convention (`SumScale`): whether its stored
    /// Σ field is Σx/amax. Set by the forward thread as it records a layer.
    pub fn set_sum_by_amax(&self, by_amax: bool) {
        self.sum_by_amax.store(by_amax, Ordering::Relaxed);
    }

    /// Inference is running: the forward thread calls this ahead of every
    /// invocation's launches. It holds the pool warm for [`COOL_DOWN`] from
    /// now — the service thread and the performance-core workers wait for the
    /// next layer in `umwait` instead of parking — and wakes whichever had
    /// parked, so they are up before the doorbell. The first call also
    /// releases the calibration.
    pub fn wake(&self) {
        if !self.woken.load(Ordering::Relaxed) {
            self.woken.store(true, Ordering::Release);
        }
        let was_warm = self.warmth.is_warm();
        self.warmth.warm(COOL_DOWN);
        if !was_warm {
            if let Some(t) = &self.thread {
                t.thread().unpark();
            }
        }
    }

    /// Layers served so far.
    pub fn served(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }
}

impl Drop for CpuService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

struct Ctx {
    rings: Arc<CpuRings>,
    shapes: Vec<RowShape>,
    sum_by_amax: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    served: Arc<AtomicU64>,
    cold: Arc<OnceLock<ColdSource>>,
    spin: Duration,
    warmth: Warmth,
}

/// Clears the liveness word however the thread exits.
struct AliveGuard<'a>(&'a CpuRings);

impl Drop for AliveGuard<'_> {
    fn drop(&mut self) {
        self.0.set_alive(false);
        let seq = STATE.load(Ordering::Relaxed) & 0xffff_ffff;
        set_phase(PHASE_STOPPED, seq);
    }
}

fn serve(ctx: &Ctx, pool: &Pool, isa: Isa) {
    let rings = &*ctx.rings;
    // The doorbell starts at 0 and carries tickets, which start at 1: a layer
    // rung before this thread got here is still served.
    let mut last = 0;
    let params = shared();
    let mut online = Online::default();
    let mut last_sample: Option<Instant> = None;
    let mut last_cold_wait = 0u64;
    loop {
        set_phase(PHASE_WAITING, last);
        let Some(seq) = wait_doorbell(ctx, last) else {
            return;
        };
        // The work slot and the rows were written before the doorbell; none of
        // the reads below may be ordered ahead of seeing it.
        fence(Ordering::Acquire);
        last = seq;
        set_phase(PHASE_PARSING, seq);
        let t0 = Instant::now();
        // The previous served layer's: its down launch finished before this
        // doorbell's export ran.
        let device_wait_ns = rings.take_wait();
        // The slot names its invocation as bucketize's summary word does — the
        // ticket's low word, never 0. A slot naming another is not this
        // layer's work: read again while it may still be landing, and refuse
        // the layer if it never does, so the device faults on it rather than
        // taking another layer's results as this one's.
        let want = (seq as u32).max(1);
        let mut h = rings.header();
        if h.seq != want {
            STALE_SLOTS.fetch_add(1, Ordering::Relaxed);
            let t = Instant::now();
            while h.seq != want && t.elapsed() < SLOT_WAIT {
                spin_loop();
                fence(Ordering::Acquire);
                h = rings.header();
            }
            if h.seq != want {
                note_refusal(REFUSED_STALE, seq);
                warn!(
                    doorbell = seq,
                    slot = h.seq,
                    "cpu expert service: the work slot never named this layer's invocation — \
                     the layer is refused"
                );
                continue;
            }
        }
        let Some(&shape) = ctx.shapes.get(h.row as usize) else {
            note_refusal(REFUSED_ROW, seq);
            warn!(
                "cpu expert service: work for row {} outside the table",
                h.row
            );
            return;
        };
        // `n_rows` ring rows — one per assignment, each with its own result
        // row — over `n_in` input rows, each distinct token once.
        let n_rows = (h.n_rows as usize).min(rings.cap_rows);
        let (n_in, in_row) = rings.input_map(n_rows);
        if in_row.iter().any(|&r| r >= n_in) {
            note_refusal(REFUSED_INPUT, seq);
            warn!(
                row = h.row,
                n_in,
                "cpu expert service: a ring row names an input row past the inputs — the layer \
                 is refused"
            );
            continue;
        }
        let tiles = shape.hidden / TILE;
        // SAFETY: the row ring holds `cap_rows` rows; the export kernel wrote
        // the first `n_in` before it rang the doorbell this thread read.
        let ring = unsafe {
            slice::from_raw_parts(rings.rows.host() as *const u8, q8a1024_bytes(n_in, tiles))
        };
        let rows: Vec<usize> = (0..n_in).collect();
        let act =
            ActTiles::from_q8a1024(ring, &rows, tiles, ctx.sum_by_amax.load(Ordering::Relaxed));
        let entries: Vec<_> = (0..h.n_experts as usize).map(|j| rings.expert(j)).collect();
        let counts: Vec<u32> = entries.iter().map(|w| w.count).collect();
        // A cold expert's snapshot entries are 0 (§5.4): its addresses come
        // from the live table once the stager publishes it.
        let (ready, cold): (Vec<&WorkExpert>, Vec<&WorkExpert>) = entries
            .iter()
            .partition(|w| w.gate != 0 && w.up != 0 && w.down != 0);
        let n_cold = cold.len();
        let t_parsed = Instant::now();
        let mut compute = Duration::ZERO;
        let mut run = |landed: &[(&WorkExpert, [u64; 3])]| {
            let t = Instant::now();
            let experts = landed
                .iter()
                .map(|&(w, addrs)| expert_work(shape, w, addrs, &in_row))
                .collect();
            let work = LayerWork {
                act: &act,
                experts,
                rows: n_rows,
            };
            let out = run_layer(pool, isa, &work);
            compute += t.elapsed();
            write_results(rings, shape.hidden, landed, &out);
        };
        let ready: Vec<_> = ready
            .into_iter()
            .map(|w| (w, [w.gate, w.up, w.down]))
            .collect();
        // Every projection of a pinned expert must be host memory: the pool
        // reads it in place, and a device address is an access violation, not
        // an error. Bucketize classes an expert pinned only when all three are
        // (`snapshot_expert`); this refuses the layer if one ever is not.
        if let Some(src) = ctx.cold.get() {
            if let Some((w, a)) = ready
                .iter()
                .find(|(_, a)| a.iter().any(|&x| !src.host_readable(x)))
            {
                note_refusal(REFUSED_PINNED, seq);
                warn!(
                    row = h.row,
                    expert = w.expert,
                    gate = format_args!("{:#x}", a[0]),
                    up = format_args!("{:#x}", a[1]),
                    down = format_args!("{:#x}", a[2]),
                    "cpu expert service: a pinned expert names memory outside the pinned ranges — \
                     the layer is refused"
                );
                continue;
            }
        }
        set_phase(PHASE_COMPUTING, seq);
        if !ready.is_empty() {
            run(&ready);
        }
        let mut cold_wait = Duration::ZERO;
        if n_cold > 0 {
            set_phase(PHASE_COLD, seq);
        }
        let served = n_cold == 0
            || match ctx.cold.get() {
                Some(src) => {
                    let ok = serve_cold(src, h.row as usize, cold, &mut run, &mut cold_wait);
                    if !ok {
                        note_refusal(REFUSED_COLD, seq);
                    }
                    ok
                }
                None => {
                    note_refusal(REFUSED_NO_TABLE, seq);
                    warn!(
                        row = h.row,
                        "cpu expert service: handed cold experts with no live table to wait \
                         on — the layer is refused"
                    );
                    false
                }
            };
        let t_computed = Instant::now();
        set_phase(PHASE_PUBLISHING, seq);
        if served {
            rings.publish_done(seq);
        }
        let busy = t0.elapsed();
        record_device_wait(device_wait_ns);
        record_phases(Served {
            rows: n_rows as u32,
            experts: counts.len() as u32,
            cold_experts: n_cold as u32,
            parse: t_parsed - t0,
            compute,
            cold_wait,
            publish: t0 + busy - t_computed,
        });
        // The pool's own time and the device's wait on it (read for the layer
        // before this one), less the time spent waiting on pack reads — the
        // drive's, which the split's costs price on neither side.
        let device_wait_ns = device_wait_ns.saturating_sub(last_cold_wait);
        last_cold_wait = cold_wait.as_nanos() as u64;
        if !served {
            continue;
        }
        let busy = busy.saturating_sub(cold_wait);
        ctx.served.fetch_add(1, Ordering::Relaxed);
        SERVED_TOTAL.fetch_add(1, Ordering::Relaxed);
        SERVED_ROWS.fetch_add(n_rows as u64, Ordering::Relaxed);
        SERVED_EXPERTS.fetch_add(counts.len() as u64, Ordering::Relaxed);
        SERVED_NS.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
        // The online loop, behind `done`: nothing on the device waits on it.
        {
            let mut p = params.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(predicted_ns) = p.predicted_pool_ns(&counts) {
                let w = last_sample.map_or(0.0, |at| {
                    weight(
                        t0.duration_since(at),
                        Duration::from_secs_f32(p.knobs.online_tau_ms.max(0.0) / 1e3),
                    )
                });
                let obs = Observation {
                    busy_ns: busy.as_nanos() as f32,
                    predicted_ns,
                    device_wait_ns,
                };
                match online.observe(&mut p, obs, w) {
                    Some(LateCause::SlowPool) => warn!(
                        row = h.row,
                        busy_us = obs.busy_ns / 1e3,
                        predicted_us = predicted_ns / 1e3,
                        wait_us = device_wait_ns / 1000,
                        "cpu experts: the device keeps waiting on the pool — the pool is \
                         running slower than its measured costs"
                    ),
                    Some(LateCause::Handoff) => warn!(
                        row = h.row,
                        busy_us = obs.busy_ns / 1e3,
                        wait_us = device_wait_ns / 1000,
                        "cpu experts: the device keeps waiting on the pool — the pool keeps \
                         to its costs, so the time is in the handoff"
                    ),
                    None => {}
                }
            }
        }
        last_sample = Some(t0);
    }
}

/// One CPU expert's work from its three projections' host addresses: its ring
/// rows' input rows (`in_row`, ring row → input row) in, a result row per ring
/// row out.
fn expert_work<'a>(
    shape: RowShape,
    w: &WorkExpert,
    addrs: [u64; 3],
    in_row: &[usize],
) -> ExpertWork<'a> {
    // SAFETY: host addresses of a slot image's projections. A pad slot's —
    // the snapshot's for a pinned expert, the live table's for a cold one the
    // stager published (`cold`) — stay put until the device has finished this
    // invocation (the reclaim rule), and that waits on `done`. A pageable
    // copy's (`cold::PagedImages`) never move, and the images hold the warm
    // tier they lie in for as long as the pool can read them.
    let view =
        |addr: u64, len: usize| unsafe { slice::from_raw_parts(addr as usize as *const u8, len) };
    let start = w.ring_start as usize;
    let count = w.count as usize;
    ExpertWork {
        expert: Expert {
            gate: KoMatrix::new(
                view(addrs[0], shape.bytes[0]),
                shape.ff,
                shape.hidden,
                shape.ko,
            ),
            up: KoMatrix::new(
                view(addrs[1], shape.bytes[1]),
                shape.ff,
                shape.hidden,
                shape.ko,
            ),
            down: KoMatrix::new(
                view(addrs[2], shape.bytes[2]),
                shape.hidden,
                shape.ff,
                shape.ko,
            ),
        },
        tokens: in_row[start..start + count].to_vec(),
        weights: vec![1.0; count],
        outputs: (start..start + count).collect(),
    }
}

/// `landed`'s rows of `out` — `run_layer`'s output over the layer's whole row
/// ring, the rows of experts it did not run left 0 — into the result ring.
/// Each ring row is one assignment's, so the experts' rows are disjoint.
fn write_results(rings: &CpuRings, hidden: usize, landed: &[(&WorkExpert, [u64; 3])], out: &[f32]) {
    let dst = rings.results.host() as *mut f32;
    for (w, _) in landed {
        let at = w.ring_start as usize * hidden;
        let len = w.count as usize * hidden;
        // SAFETY: the result ring holds `cap_rows × hidden` floats, and the
        // rows are inside `n_rows ≤ cap_rows`; the device reads them only
        // after `done`.
        unsafe { copy_nonoverlapping(out.as_ptr().add(at), dst.add(at), len) };
    }
}

/// Runs `cold`'s experts on `run` as the stager publishes each, the ones that
/// have landed together; time with nothing landed accrues to `waited`. False
/// when a wait gave up (`cold::ColdSource::must_give_up`) or an expert was
/// published where the pool cannot read it: the fault word is claimed, and the
/// layer gets no `done`.
fn serve_cold(
    src: &ColdSource,
    row: usize,
    cold: Vec<&WorkExpert>,
    run: &mut impl FnMut(&[(&WorkExpert, [u64; 3])]),
    waited: &mut Duration,
) -> bool {
    let t0 = Instant::now();
    let mut pending = cold;
    let mut idle_since: Option<Instant> = None;
    let mut backoff = Backoff::new(INTER_LAYER_SPINS);
    while !pending.is_empty() {
        let mut landed = Vec::new();
        let mut still = Vec::new();
        for w in pending {
            match src.published(row, w.expert as usize) {
                Published::Host(addrs) => landed.push((w, addrs)),
                Published::Pending => still.push(w),
                Published::Unreadable => {
                    warn!(
                        row,
                        expert = w.expert,
                        "cpu expert service: a cold expert was published to device memory — \
                         given up"
                    );
                    src.give_up(row, w.expert as usize, t0.elapsed());
                    return false;
                }
            }
        }
        pending = still;
        if !landed.is_empty() {
            if let Some(at) = idle_since.take() {
                *waited += at.elapsed();
            }
            run(&landed);
            backoff.reset();
            continue;
        }
        let at = *idle_since.get_or_insert_with(Instant::now);
        if src.must_give_up(t0.elapsed()) {
            *waited += at.elapsed();
            src.give_up(row, pending[0].expert as usize, t0.elapsed());
            return false;
        }
        src.idle(&mut backoff, row, pending[0].expert as usize);
    }
    true
}

/// Waits for the doorbell to move past `last`: hot spins, then `umwait` on its
/// line (`backoff`) through the warm window, or for `ctx.spin` outside it,
/// then parked (the forward thread's next `wake` unparks it), re-checking
/// every millisecond. `None` once the service is told to stop.
fn wait_doorbell(ctx: &Ctx, last: u64) -> Option<u64> {
    let rings = &*ctx.rings;
    let t0 = Instant::now();
    let mut backoff = Backoff::new(INTER_LAYER_SPINS);
    loop {
        if ctx.stop.load(Ordering::Acquire) {
            return None;
        }
        let now = rings.doorbell();
        if now != last {
            return Some(now);
        }
        if t0.elapsed() < ctx.spin || ctx.warmth.is_warm() {
            // SAFETY: the mapped doorbell, which the rings own for the
            // service's life.
            unsafe { backoff.wait_u64(rings.doorbell_ptr(), last, LOW_POWER_CYCLES) };
        } else {
            thread::park_timeout(Duration::from_millis(1));
        }
    }
}
