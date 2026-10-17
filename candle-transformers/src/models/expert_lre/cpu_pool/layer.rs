//! One MoE layer's CPU experts on the pool — the schedule of
//! `docs/cpu_expert_compute.md` §5.6.
//!
//! Dataflow per expert, with no pool-wide barrier:
//!
//! 1. **Gate + up.** Tasks of a few FF rows ([`tasks_for`], [`task_range`]),
//!    claimed from one counter across all experts. The first task of an
//!    expert to start gathers
//!    and permutes its tokens' activation rows (once, shared by its tasks).
//! 2. **SwiGLU.** Each gate/up task applies SwiGLU to its own rows as soon as
//!    it has them; the task that finishes an expert's last rows quantizes the
//!    down projection's input, which needs whole 128-row tiles.
//! 3. **Down.** That expert's down tasks become claimable at once, and are
//!    preferred over gate/up work so finished experts drain first. Each writes
//!    the expert's unweighted rows into its own scratch.
//! 4. **Reduce.** Each hidden row range counts down the experts still to write
//!    it; the task that takes it to zero sums `weight × rows` over the token's
//!    experts into the output. Every output element has exactly one writer.
//!
//! The output is `LayerWork::rows` rows of `hidden`, each the routing-weighted
//! sum of the expert rows that name it: one per token sums the token's
//! experts; one per assignment — what the service asks for, since the device
//! weights and sums the rows in its scatter — keeps each expert's apart.

use std::ops::Range;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::backoff::{Backoff, INTRA_LAYER_SPINS};
use super::kernel::act::{ActTiles, TILE};
use super::kernel::expert::{swiglu, Expert};
use super::kernel::ko::Permuted;
use super::kernel::simd::{matvec_into, Isa};
use super::pool::Pool;
use super::spin_once::SpinOnce;
use super::stats::{record_run, Run};

/// Tasks per participant a phase is cut into: enough that a slow core's last
/// task is a small share of the phase, few enough that claiming stays cheap.
const TASKS_PER_PARTICIPANT: usize = 3;

/// Empty polls a participant waits through before it starts yielding: the
/// backoff's hot spins, then the rest in `tpause` — ~100 µs in all.
const IDLE_SPINS: u32 = 256;

/// One empty poll's wait past the hot spins: `tpause` for ~0.5 µs of TSC —
/// short against a task of tens of microseconds, and in C0.1 rather than a
/// `pause` loop.
const IDLE_PAUSE_CYCLES: u64 = 1_500;

/// The participants a layer takes: every performance core (`Pool::fast`),
/// whatever its expert count, and never a slower one. The matvec is
/// compute-bound on a core — Q3 one token unpacks at ~16 GB/s, against ~22
/// that one core can read — so every performance core adds to a layer's rate
/// until together they meet DRAM's: on the laptop one DRAM-cold one-token
/// expert takes ~60 µs on five, ~62 on four (`kernel_profile`). How many
/// experts a layer gets is the split's: at most two a performance core
/// (`attach::EXPERTS_PER_FAST_CORE`, `CpuSplit::max_experts`), the price
/// deciding under that. The slower cores wait unwoken and park
/// (`pool`): on the laptop, fourteen threads spinning between decode layers
/// held the CPU at ~75% of base clock, where five computed the layer as fast.
pub fn participants_for(fast: usize) -> usize {
    fast.max(1)
}

/// Tasks one projection of `groups` row groups (8 rows each) is cut into
/// when `experts` experts' projections are shared by `participants`
/// threads: [`TASKS_PER_PARTICIPANT`] per participant across the experts, at
/// least one, at most a group each.
pub fn tasks_for(groups: usize, experts: usize, participants: usize) -> usize {
    (participants.max(1) * TASKS_PER_PARTICIPANT)
        .div_ceil(experts.max(1))
        .clamp(1, groups.max(1))
}

/// Rows of task `i` of `tasks` over `groups` row groups: the groups split as
/// evenly as whole groups allow, so a single expert's tasks — the one cold
/// miss the pool runs alone, on the critical path — finish together rather
/// than leaving a last round with idle participants.
pub fn task_range(i: usize, tasks: usize, groups: usize) -> Range<usize> {
    (i * groups / tasks) * 8..((i + 1) * groups / tasks) * 8
}

/// One CPU expert's share of a layer.
pub struct ExpertWork<'a> {
    pub expert: Expert<'a>,
    /// Indices into the layer's CPU tokens (`LayerWork::act`'s rows), distinct.
    pub tokens: Vec<usize>,
    /// The routing weight of each of `tokens`.
    pub weights: Vec<f32>,
    /// The result row each of `tokens` adds its weighted rows into: the
    /// token's own, to sum its experts, or one of its own — the service's,
    /// one per assignment, since the device weights and sums them.
    pub outputs: Vec<usize>,
}

/// One layer's CPU work: the CPU tokens' gate/up input, the experts, and the
/// result rows they write.
pub struct LayerWork<'a> {
    pub act: &'a ActTiles,
    pub experts: Vec<ExpertWork<'a>>,
    pub rows: usize,
}

/// A buffer workers write disjoint ranges of.
///
/// Every access goes through `base`, a raw pointer taken once when the buffer
/// is made: a writer's slice borrows its own range and nothing else, so two
/// workers' slices never overlap — a `&mut` to the whole `Vec` per access
/// would, whatever ranges they then touched. The `Vec` is not touched again
/// until [`Self::into_inner`] takes it back.
struct Disjoint<T> {
    buf: Vec<T>,
    base: *mut T,
}

// SAFETY: every writer owns a range no other writer touches, and readers read
// only ranges whose writers finished before an acquire on the counter that
// published them; `base` points into `buf`'s heap allocation, which moving
// `buf` does not move.
unsafe impl<T: Send> Sync for Disjoint<T> {}

impl<T: Copy + Default> Disjoint<T> {
    fn new(len: usize) -> Self {
        let mut buf = vec![T::default(); len];
        let base = buf.as_mut_ptr();
        Self { buf, base }
    }

    /// # Safety
    /// No other thread may access `start..start + len` while the slice lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        assert!(
            start + len <= self.buf.len(),
            "disjoint buffer: {start}+{len} past {}",
            self.buf.len()
        );
        slice::from_raw_parts_mut(self.base.add(start), len)
    }

    /// The buffer's base, for a writer that addresses its range by stride.
    fn ptr(&self) -> *mut T {
        self.base
    }

    fn into_inner(self) -> Vec<T> {
        self.buf
    }
}

/// Per-expert state.
struct ExpertState {
    /// The expert's tokens' activation tiles, gathered, and permuted.
    /// Built by the first task to reach the expert; the others spin the
    /// microseconds it takes (`spin_once`).
    input: SpinOnce<(ActTiles, Permuted)>,
    /// `tokens × ff`, gate then up.
    gate: Disjoint<f32>,
    up: Disjoint<f32>,
    /// `tokens × ff`: SwiGLU of the two, written by the task that computed
    /// each row range.
    swiglu: Disjoint<f32>,
    /// Gate/up tasks of this expert still to finish.
    gate_left: AtomicUsize,
    /// The down projection's input, set by the last gate/up task.
    down_input: SpinOnce<(ActTiles, Permuted)>,
    /// Next down task to claim.
    down_next: AtomicUsize,
    /// `tokens × hidden`, unweighted.
    out: Disjoint<f32>,
}

/// Runs `work` on `pool` with the matvec on `isa`, and returns `work.rows ×
/// hidden`: each result row the weighted sum of the expert rows its
/// `outputs` name, 0 where none do.
pub fn run_layer(pool: &Pool, isa: Isa, work: &LayerWork) -> Vec<f32> {
    let participants = participants_for(pool.fast());
    run_layer_on(pool, isa, work, participants)
}

/// [`run_layer`] on the first `participants` of the pool.
pub fn run_layer_on(pool: &Pool, isa: Isa, work: &LayerWork, participants: usize) -> Vec<f32> {
    let n_out = work.rows;
    let Some(first) = work.experts.first() else {
        return Vec::new();
    };
    let (hidden, ff) = (first.expert.hidden(), first.expert.ff());
    let k = work.experts.len();
    let participants = participants.clamp(1, pool.participants());
    let (gate_groups, down_groups) = (ff / 8, hidden / 8);
    let gate_tasks = tasks_for(gate_groups, k, participants);
    let down_tasks = tasks_for(down_groups, k, participants);

    let states: Vec<ExpertState> = work
        .experts
        .iter()
        .map(|e| {
            let t = e.tokens.len();
            ExpertState {
                input: SpinOnce::new(),
                gate: Disjoint::new(t * ff),
                up: Disjoint::new(t * ff),
                swiglu: Disjoint::new(t * ff),
                gate_left: AtomicUsize::new(gate_tasks),
                down_input: SpinOnce::new(),
                down_next: AtomicUsize::new(0),
                out: Disjoint::new(t * hidden),
            }
        })
        .collect();
    let gate_next = AtomicUsize::new(0);
    let total_gate = gate_tasks * work.experts.len();
    // Per down range: experts still to write it.
    let range_left: Vec<AtomicUsize> = (0..down_tasks)
        .map(|_| AtomicUsize::new(work.experts.len()))
        .collect();
    let ranges_done = AtomicUsize::new(0);
    let result: Disjoint<f32> = Disjoint::new(n_out * hidden);
    let join_ns = AtomicU64::new(0);
    let work_ns = AtomicU64::new(0);
    let in_job_ns = AtomicU64::new(0);
    // Set by a task that panicked: the work it held will never finish, so the
    // other participants stop waiting for it, and the pool raises the panic in
    // the caller once all have returned.
    let failed = AtomicBool::new(false);
    let t0 = Instant::now();

    let body = || {
        let entered = Instant::now();
        join_ns.fetch_max(
            entered.duration_since(t0).as_nanos() as u64,
            Ordering::Relaxed,
        );
        let mut busy = Duration::ZERO;
        let mut idle = 0u32;
        let mut backoff = Backoff::new(INTRA_LAYER_SPINS);
        loop {
            if failed.load(Ordering::Acquire) {
                return;
            }
            if ranges_done.load(Ordering::Acquire) == down_tasks {
                work_ns.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
                in_job_ns.fetch_add(entered.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return;
            }
            let t = Instant::now();
            // Down work first: experts whose input is ready.
            let mut did = false;
            for (e, st) in states.iter().enumerate() {
                let Some((ha, hp)) = st.down_input.get() else {
                    continue;
                };
                let d = st.down_next.fetch_add(1, Ordering::Relaxed);
                if d >= down_tasks {
                    continue;
                }
                let rows = task_range(d, down_tasks, down_groups);
                // SAFETY: rows `rows` of every token of expert `e` belong to this
                // task alone; the buffer is `tokens × hidden`.
                unsafe {
                    matvec_into(
                        isa,
                        &work.experts[e].expert.down,
                        ha,
                        hp,
                        rows.clone(),
                        st.out.ptr().add(rows.start),
                        hidden,
                    );
                }
                if range_left[d].fetch_sub(1, Ordering::AcqRel) == 1 {
                    reduce(work, &states, &result, rows, hidden);
                    ranges_done.fetch_add(1, Ordering::Release);
                }
                did = true;
                break;
            }
            if did {
                busy += t.elapsed();
                idle = 0;
                backoff.reset();
                continue;
            }
            // Then gate/up work.
            let g = gate_next.fetch_add(1, Ordering::Relaxed);
            if g < total_gate {
                let (e, c) = (g / gate_tasks, g % gate_tasks);
                let rows = task_range(c, gate_tasks, gate_groups);
                gate_up_task(isa, work, &states, e, rows, ff);
                busy += t.elapsed();
                idle = 0;
                backoff.reset();
                continue;
            }
            // Nothing to claim: another participant holds the work this one waits
            // for. Spin a few times, then wait at low power, then yield, so a
            // holder the OS preempted gets its core back instead of the layer
            // waiting out a scheduling quantum.
            idle += 1;
            if idle < IDLE_SPINS {
                backoff.pause(IDLE_PAUSE_CYCLES);
            } else {
                thread::yield_now();
            }
        }
    };
    let job = |_: usize| {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(body)) {
            failed.store(true, Ordering::Release);
            resume_unwind(payload);
        }
    };
    pool.run_on(participants, &job);
    record_run(Run {
        rows: n_out,
        participants,
        join: Duration::from_nanos(join_ns.into_inner()),
        work: Duration::from_nanos(work_ns.into_inner()),
        in_job: Duration::from_nanos(in_job_ns.into_inner()),
    });
    result.into_inner()
}

/// Gate/up rows `rows` of expert `e`; the task that finishes the expert's last
/// rows also prepares its down input.
fn gate_up_task(
    isa: Isa,
    work: &LayerWork,
    states: &[ExpertState],
    e: usize,
    rows: Range<usize>,
    ff: usize,
) {
    let ew = &work.experts[e];
    let st = &states[e];
    let (act, pa) = st.input.get_or_init(|| {
        let a = gather(work.act, &ew.tokens);
        let p = Permuted::new(&a, ew.expert.gate.ko);
        (a, p)
    });
    let t = ew.tokens.len();
    // SAFETY: rows `rows` of every token of expert `e` belong to this task
    // alone; both buffers are `tokens × ff`.
    unsafe {
        matvec_into(
            isa,
            &ew.expert.gate,
            act,
            pa,
            rows.clone(),
            st.gate.ptr().add(rows.start),
            ff,
        );
        matvec_into(
            isa,
            &ew.expert.up,
            act,
            pa,
            rows.clone(),
            st.up.ptr().add(rows.start),
            ff,
        );
    }
    for tok in 0..t {
        let at = tok * ff + rows.start;
        // SAFETY: as the matvecs above — these rows of this token are this
        // task's alone, in all three buffers.
        unsafe {
            swiglu(
                st.gate.slice(at, rows.len()),
                st.up.slice(at, rows.len()),
                st.swiglu.slice(at, rows.len()),
            );
        }
    }
    if st.gate_left.fetch_sub(1, Ordering::AcqRel) == 1 {
        // SAFETY: every gate/up task of `e` has finished (the AcqRel above).
        let h = unsafe { st.swiglu.slice(0, t * ff) };
        let ha = ActTiles::quantize(h, t, ff);
        let hp = Permuted::new(&ha, ew.expert.down.ko);
        // The only setter: the one task that took the count to zero.
        let _ = st.down_input.set((ha, hp));
    }
}

/// Rows `rows` of the result, for every result row: Σ, in expert order, of
/// weight × the expert rows its `outputs` name.
fn reduce(
    work: &LayerWork,
    states: &[ExpertState],
    result: &Disjoint<f32>,
    rows: Range<usize>,
    hidden: usize,
) {
    for out in 0..work.rows {
        // SAFETY: this range of the result has exactly one writer: the task
        // that took the range's count to zero.
        let dst = unsafe { result.slice(out * hidden + rows.start, rows.len()) };
        dst.iter_mut().for_each(|v| *v = 0.0);
    }
    for (ew, st) in work.experts.iter().zip(states) {
        for (slot, (&out, &w)) in ew.outputs.iter().zip(&ew.weights).enumerate() {
            // SAFETY: every expert finished writing `rows` before the count
            // reached zero; the result range is this task's alone.
            let (src, dst) = unsafe {
                (
                    st.out.slice(slot * hidden + rows.start, rows.len()),
                    result.slice(out * hidden + rows.start, rows.len()),
                )
            };
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d += w * s;
            }
        }
    }
}

/// The rows `tokens` of `act`, in that order.
fn gather(act: &ActTiles, tokens: &[usize]) -> ActTiles {
    let tiles = act.tiles;
    let mut out = ActTiles {
        tokens: tokens.len(),
        tiles,
        qs: Vec::with_capacity(tokens.len() * tiles * TILE),
        scale: Vec::with_capacity(tokens.len() * tiles),
        sum: Vec::with_capacity(tokens.len() * tiles),
    };
    for &t in tokens {
        out.qs
            .extend_from_slice(&act.qs[t * tiles * TILE..(t + 1) * tiles * TILE]);
        out.scale
            .extend_from_slice(&act.scale[t * tiles..(t + 1) * tiles]);
        out.sum
            .extend_from_slice(&act.sum[t * tiles..(t + 1) * tiles]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::kernel::expert::forward;
    use crate::models::expert_lre::cpu_pool::kernel::ko::Ko;
    use crate::models::expert_lre::cpu_pool::pool::PoolWaits;
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};
    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::GgmlDType;
    use std::sync::mpsc;
    use std::time::Duration;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn slot(hidden: usize, ff: usize, seed: &mut u64) -> (Vec<u8>, (usize, usize, usize)) {
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(seed) * 0.1).collect() };
        let (g, u, d) = (
            quantize_ko(&w(ff * hidden), ff, hidden, GgmlDType::Q3_KO),
            quantize_ko(&w(ff * hidden), ff, hidden, GgmlDType::Q3_KO),
            quantize_ko(&w(hidden * ff), hidden, ff, GgmlDType::Q3_KO),
        );
        let offsets = (0, g.len(), g.len() + u.len());
        ([g, u, d].concat(), offsets)
    }

    /// A layer whose task panics — an expert built for a hidden width the
    /// layer's activations do not have, which the matvec refuses — fails at
    /// once with that panic, rather than leaving the other participants
    /// waiting on work that will never finish; the same pool then runs a good
    /// layer.
    #[test]
    fn a_task_that_panics_fails_the_layer_rather_than_hanging_it() {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (hidden, ff) = (2560, 640);
            let mut seed = 5;
            let good = slot(hidden, ff, &mut seed);
            let bad = slot(256, ff, &mut seed);
            let x: Vec<f32> = (0..hidden).map(|_| lcg(&mut seed)).collect();
            let act = ActTiles::quantize(&x, 1, hidden);
            let pool = test_pool();
            let isa = Isa::detect();
            fn one(
                slot: &(Vec<u8>, (usize, usize, usize)),
                width: usize,
                ff: usize,
            ) -> ExpertWork<'_> {
                ExpertWork {
                    expert: Expert::from_slot(&slot.0, slot.1, width, ff, Ko::Q3),
                    tokens: vec![0],
                    weights: vec![1.0],
                    outputs: vec![0],
                }
            }
            let failing = LayerWork {
                act: &act,
                experts: vec![one(&good, hidden, ff), one(&bad, 256, ff)],
                rows: 1,
            };
            let failed =
                catch_unwind(AssertUnwindSafe(|| run_layer(&pool, isa, &failing))).is_err();
            let next = LayerWork {
                act: &act,
                experts: vec![one(&good, hidden, ff)],
                rows: 1,
            };
            let ran = run_layer(&pool, isa, &next).len() == hidden;
            let _ = tx.send((failed, ran));
        });
        let (failed, ran) = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("the layer hung on its panicked task");
        assert!(failed, "the panic reaches the caller");
        assert!(ran, "the pool runs the next layer");
    }

    /// Every performance core — the laptop's five: the service thread and
    /// four workers — and at least the caller.
    #[test]
    fn a_layer_takes_every_performance_core() {
        assert_eq!(participants_for(5), 5);
        assert_eq!(participants_for(7), 7);
        assert_eq!(participants_for(1), 1);
        assert_eq!(participants_for(0), 1);
    }

    /// Three tasks per participant across the experts, at least one, at most
    /// a group each.
    #[test]
    fn tasks_are_three_per_participant_across_the_experts() {
        assert_eq!(tasks_for(80, 1, 5), 15);
        assert_eq!(tasks_for(320, 1, 5), 15);
        assert_eq!(tasks_for(80, 2, 5), 8);
        assert_eq!(tasks_for(80, 5, 5), 3);
        assert_eq!(tasks_for(80, 100, 16), 1);
        assert_eq!(tasks_for(8, 1, 64), 8);
        assert_eq!(tasks_for(80, 0, 0), 3);
    }

    /// The ranges tile the projection in whole chunks, in order, each within
    /// one group of every other — one expert's 80 gate groups over 15 tasks
    /// are ten 5-group and five 6-group ranges.
    #[test]
    fn task_ranges_split_the_groups_evenly() {
        for (groups, tasks) in [(80usize, 15usize), (320, 15), (80, 8), (5, 5), (7, 3)] {
            let ranges: Vec<_> = (0..tasks).map(|i| task_range(i, tasks, groups)).collect();
            assert_eq!(ranges[0].start, 0);
            assert_eq!(ranges[tasks - 1].end, groups * 8);
            for w in ranges.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
            let lens: Vec<usize> = ranges.iter().map(|r| r.len() / 8).collect();
            let (lo, hi) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
            assert!(hi - lo <= 1, "{groups}/{tasks}: {lens:?}");
            assert!(ranges.iter().all(|r| r.start % 8 == 0 && !r.is_empty()));
        }
        let lens: Vec<usize> = (0..15).map(|i| task_range(i, 15, 80).len() / 8).collect();
        assert_eq!(lens.iter().filter(|&&l| l == 5).count(), 10);
        assert_eq!(lens.iter().filter(|&&l| l == 6).count(), 5);
    }

    /// The pooled layer equals, byte for byte, each expert run alone by
    /// `forward` and summed per token in expert order — at Flash-Next's
    /// geometry, with tokens shared between experts and one expert's tokens a
    /// non-contiguous subset.
    #[test]
    fn pooled_layer_is_the_per_expert_sum() {
        let (hidden, ff, n_tok) = (2560, 640, 4);
        let mut seed = 1;
        let slots: Vec<_> = (0..3).map(|_| slot(hidden, ff, &mut seed)).collect();
        let x: Vec<f32> = (0..n_tok * hidden).map(|_| lcg(&mut seed)).collect();
        let act = ActTiles::quantize(&x, n_tok, hidden);
        let assign: [(&[usize], &[f32]); 3] = [
            (&[0, 1, 2, 3], &[0.5, 0.25, 0.125, 0.0625]),
            (&[1, 3], &[0.3, 0.7]),
            (&[2], &[1.0]),
        ];
        let experts: Vec<ExpertWork> = slots
            .iter()
            .zip(assign)
            .map(|((bytes, off), (toks, ws))| ExpertWork {
                expert: Expert::from_slot(bytes, *off, hidden, ff, Ko::Q3),
                tokens: toks.to_vec(),
                weights: ws.to_vec(),
                outputs: toks.to_vec(),
            })
            .collect();
        let isa = Isa::detect();
        let pool = test_pool();
        let work = LayerWork {
            act: &act,
            experts,
            rows: n_tok,
        };
        let got = run_layer(&pool, isa, &work);

        let mut expect = vec![0f32; n_tok * hidden];
        for ew in &work.experts {
            let sub = gather(&act, &ew.tokens);
            let mut y = vec![0f32; ew.tokens.len() * hidden];
            forward(isa, &ew.expert, &sub, &mut y);
            for (slot, (&tok, &w)) in ew.tokens.iter().zip(&ew.weights).enumerate() {
                for r in 0..hidden {
                    expect[tok * hidden + r] += w * y[slot * hidden + r];
                }
            }
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&expect));
    }

    fn test_pool() -> Pool {
        let cores = performance_cores(&physical_cores().unwrap());
        let sets: Vec<u32> = cores.iter().skip(1).take(3).map(|c| c.primary()).collect();
        Pool::new(&sets, sets.len(), PoolWaits::new(Duration::from_millis(1))).unwrap()
    }

    /// One result row per assignment, as the service asks: experts sharing a
    /// token read its one input row and each write their own result row,
    /// unweighted at weight 1 — byte for byte each expert's `forward` over
    /// that token, in the row its `outputs` names.
    #[test]
    fn per_assignment_rows_share_their_tokens_input() {
        let (hidden, ff) = (2560, 640);
        let mut seed = 7;
        let slots: Vec<_> = (0..3).map(|_| slot(hidden, ff, &mut seed)).collect();
        // Two tokens; assignments: e0 ← t0, e1 ← t0, e1 ← t1, e2 ← t1.
        let x: Vec<f32> = (0..2 * hidden).map(|_| lcg(&mut seed)).collect();
        let act = ActTiles::quantize(&x, 2, hidden);
        let assign: [(&[usize], &[usize]); 3] = [(&[0], &[0]), (&[0, 1], &[1, 2]), (&[1], &[3])];
        let experts: Vec<ExpertWork> = slots
            .iter()
            .zip(assign)
            .map(|((bytes, off), (toks, outs))| ExpertWork {
                expert: Expert::from_slot(bytes, *off, hidden, ff, Ko::Q3),
                tokens: toks.to_vec(),
                weights: vec![1.0; toks.len()],
                outputs: outs.to_vec(),
            })
            .collect();
        let isa = Isa::detect();
        let pool = test_pool();
        let work = LayerWork {
            act: &act,
            experts,
            rows: 4,
        };
        let got = run_layer(&pool, isa, &work);
        assert_eq!(got.len(), 4 * hidden);
        for ew in &work.experts {
            let sub = gather(&act, &ew.tokens);
            let mut y = vec![0f32; ew.tokens.len() * hidden];
            forward(isa, &ew.expert, &sub, &mut y);
            for (slot, &out) in ew.outputs.iter().enumerate() {
                let g = &got[out * hidden..(out + 1) * hidden];
                let w = &y[slot * hidden..(slot + 1) * hidden];
                assert!(
                    g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "result row {out}"
                );
            }
        }
    }
}
