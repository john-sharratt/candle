//! Where a served layer's time goes, by phase — process-wide counters the
//! service adds to after each layer's `done`, cheap enough to keep on.
//!
//! | Phase | From → to |
//! |---|---|
//! | `parse` | doorbell seen → the experts and activation tiles built |
//! | `compute` | the layer's experts on the pool (`run_layer`), every run summed |
//! | `cold_wait` | time with every remaining expert cold and none yet published (§5.4) |
//! | `publish` | the last results copied into the mapped ring → `done` stored |
//! | `device_wait` | the device's own measure of how long its CPU items waited |
//!
//! Inside `compute`, each `run_layer` on the pool adds its participants'
//! split ([`record_run`]):
//!
//! | Measure | What |
//! |---|---|
//! | `join` | the last participant's entry into the job, after the caller's start |
//! | `work` | time participants spent in matvecs, SwiGLU and reduce, summed |
//! | `in_job` | time participants spent inside the job, summed |
//!
//! `work` over `in_job` is the pool's occupancy; `in_job − work` is waiting on
//! one another.
//!
//! Layers are counted in two classes, because they cost differently: a
//! decode-sized layer (at most [`SMALL_ROWS`] rows), and a prompt-sized one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Rows a decode-sized layer has at most.
pub const SMALL_ROWS: u32 = 16;

/// One class's totals.
struct Class {
    layers: AtomicU64,
    rows: AtomicU64,
    experts: AtomicU64,
    cold_experts: AtomicU64,
    parse_ns: AtomicU64,
    compute_ns: AtomicU64,
    cold_wait_ns: AtomicU64,
    publish_ns: AtomicU64,
    device_wait_ns: AtomicU64,
    runs: AtomicU64,
    participants: AtomicU64,
    join_ns: AtomicU64,
    work_ns: AtomicU64,
    in_job_ns: AtomicU64,
}

impl Class {
    const fn new() -> Self {
        Self {
            layers: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            experts: AtomicU64::new(0),
            cold_experts: AtomicU64::new(0),
            parse_ns: AtomicU64::new(0),
            compute_ns: AtomicU64::new(0),
            cold_wait_ns: AtomicU64::new(0),
            publish_ns: AtomicU64::new(0),
            device_wait_ns: AtomicU64::new(0),
            runs: AtomicU64::new(0),
            participants: AtomicU64::new(0),
            join_ns: AtomicU64::new(0),
            work_ns: AtomicU64::new(0),
            in_job_ns: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> ClassStats {
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        ClassStats {
            layers: l(&self.layers),
            rows: l(&self.rows),
            experts: l(&self.experts),
            cold_experts: l(&self.cold_experts),
            parse_ns: l(&self.parse_ns),
            compute_ns: l(&self.compute_ns),
            cold_wait_ns: l(&self.cold_wait_ns),
            publish_ns: l(&self.publish_ns),
            device_wait_ns: l(&self.device_wait_ns),
            runs: l(&self.runs),
            participants: l(&self.participants),
            join_ns: l(&self.join_ns),
            work_ns: l(&self.work_ns),
            in_job_ns: l(&self.in_job_ns),
        }
    }
}

static SMALL: Class = Class::new();
static LARGE: Class = Class::new();
/// The device's wait is read at the NEXT doorbell, for the layer before it;
/// this is that layer's class.
static LAST_SMALL: AtomicU64 = AtomicU64::new(1);

/// One class's totals, read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassStats {
    pub layers: u64,
    pub rows: u64,
    pub experts: u64,
    /// Of `experts`, those cold when the layer was handed over.
    pub cold_experts: u64,
    pub parse_ns: u64,
    pub compute_ns: u64,
    pub cold_wait_ns: u64,
    pub publish_ns: u64,
    pub device_wait_ns: u64,
    /// `run_layer`s on the pool, and their participants, summed.
    pub runs: u64,
    pub participants: u64,
    pub join_ns: u64,
    pub work_ns: u64,
    pub in_job_ns: u64,
}

impl ClassStats {
    /// What accrued since `before`.
    pub fn since(&self, before: &Self) -> Self {
        Self {
            runs: self.runs - before.runs,
            participants: self.participants - before.participants,
            join_ns: self.join_ns - before.join_ns,
            work_ns: self.work_ns - before.work_ns,
            in_job_ns: self.in_job_ns - before.in_job_ns,
            layers: self.layers - before.layers,
            rows: self.rows - before.rows,
            experts: self.experts - before.experts,
            cold_experts: self.cold_experts - before.cold_experts,
            parse_ns: self.parse_ns - before.parse_ns,
            compute_ns: self.compute_ns - before.compute_ns,
            cold_wait_ns: self.cold_wait_ns - before.cold_wait_ns,
            publish_ns: self.publish_ns - before.publish_ns,
            device_wait_ns: self.device_wait_ns - before.device_wait_ns,
        }
    }

    /// Per layer, µs: parse, compute, cold wait, publish, the device's wait.
    pub fn per_layer_us(&self) -> [f64; 5] {
        let n = self.layers.max(1) as f64 * 1e3;
        [
            self.parse_ns as f64 / n,
            self.compute_ns as f64 / n,
            self.cold_wait_ns as f64 / n,
            self.publish_ns as f64 / n,
            self.device_wait_ns as f64 / n,
        ]
    }

    /// Per pool run: participants, the last one's join (µs), work and time in
    /// the job per participant (µs).
    pub fn per_run(&self) -> [f64; 4] {
        let r = self.runs.max(1) as f64;
        let p = self.participants.max(1) as f64 * 1e3;
        [
            self.participants as f64 / r,
            self.join_ns as f64 / (r * 1e3),
            self.work_ns as f64 / p,
            self.in_job_ns as f64 / p,
        ]
    }
}

/// Both classes: decode-sized, then prompt-sized.
pub fn stats() -> [ClassStats; 2] {
    [SMALL.snapshot(), LARGE.snapshot()]
}

/// One served layer: its size and where its time went.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Served {
    pub(crate) rows: u32,
    pub(crate) experts: u32,
    pub(crate) cold_experts: u32,
    pub(crate) parse: Duration,
    pub(crate) compute: Duration,
    pub(crate) cold_wait: Duration,
    pub(crate) publish: Duration,
}

/// Count one served layer.
pub(crate) fn record(s: Served) {
    let small = s.rows <= SMALL_ROWS;
    let c = if small { &SMALL } else { &LARGE };
    let ns = |d: Duration| d.as_nanos() as u64;
    c.layers.fetch_add(1, Ordering::Relaxed);
    c.rows.fetch_add(u64::from(s.rows), Ordering::Relaxed);
    c.experts.fetch_add(u64::from(s.experts), Ordering::Relaxed);
    c.cold_experts
        .fetch_add(u64::from(s.cold_experts), Ordering::Relaxed);
    c.parse_ns.fetch_add(ns(s.parse), Ordering::Relaxed);
    c.compute_ns.fetch_add(ns(s.compute), Ordering::Relaxed);
    c.cold_wait_ns.fetch_add(ns(s.cold_wait), Ordering::Relaxed);
    c.publish_ns.fetch_add(ns(s.publish), Ordering::Relaxed);
    LAST_SMALL.store(u64::from(small), Ordering::Relaxed);
}

/// One `run_layer` on the pool: `rows` result rows, over `participants`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Run {
    pub(crate) rows: usize,
    pub(crate) participants: usize,
    pub(crate) join: Duration,
    pub(crate) work: Duration,
    pub(crate) in_job: Duration,
}

/// Count one pool run.
pub(crate) fn record_run(r: Run) {
    let c = if r.rows <= SMALL_ROWS as usize {
        &SMALL
    } else {
        &LARGE
    };
    let ns = |d: Duration| d.as_nanos() as u64;
    c.runs.fetch_add(1, Ordering::Relaxed);
    c.participants
        .fetch_add(r.participants as u64, Ordering::Relaxed);
    c.join_ns.fetch_add(ns(r.join), Ordering::Relaxed);
    c.work_ns.fetch_add(ns(r.work), Ordering::Relaxed);
    c.in_job_ns.fetch_add(ns(r.in_job), Ordering::Relaxed);
}

/// The device's wait on the layer served before this doorbell.
pub(crate) fn record_device_wait(ns: u64) {
    let c = if LAST_SMALL.load(Ordering::Relaxed) == 1 {
        &SMALL
    } else {
        &LARGE
    };
    c.device_wait_ns.fetch_add(ns, Ordering::Relaxed);
}
