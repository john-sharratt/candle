//! The stager — the thread that makes cold experts readable by the device.
//!
//! A cold expert (live-table entry 0: on the NVMe pack only, or in a pageable
//! warm slot) is computed by its layer's GEMM workers as soon as it is in the
//! pad, a pinned tier this thread owns (`pad`). Per routed row, in the order the
//! forward thread enqueued them:
//!
//! 1. wait for the row's routing-summary word in the mapped ring (the same word
//!    the pipeline thread polls);
//! 2. read its **cold set** — the experts bucketize found cold (summary bit 30);
//! 3. give each a pad slot — free, else a victim (Rule R′ below) — and hand the
//!    read to a reader thread: a positioned direct read of the pack record, or a
//!    `memcpy` from its pageable warm slot;
//! 4. publish each expert as its read lands, through the residency lock (up,
//!    down, fence, gate — `live_table`), which releases the workers waiting on
//!    it.
//!
//! **No CUDA call on this path, anywhere.** The reads, the copies and the
//! publishing stores need no driver, so a thread blocked in the driver — a lazy
//! kernel load behind a waiting launch — can delay nothing a waiting worker
//! depends on (`docs/moe_live_dispatch_design.md` §0.1, §0.6).
//!
//! **Rule R′ — the demand window.** While a cold expert of ticket `T` is
//! unpublished, the GPU is inside `T`'s gate launch: every invocation before `T`
//! has completed and none after it has started. So any pad slot not holding
//! one of `T`'s routed experts may be evicted and overwritten at once, even if
//! that empties its entry — the next reader of it is a later bucketize. The
//! pad holds at least one layer of slots, so a row's cold set always fits. A
//! forward that faulted breaks the premise — its workers gave `T`'s expert up
//! and the GPU moved on while the read is still out — so every eviction stores
//! the entry and then checks the row's retire key (`Stager::evict`): under R′
//! it always passes, and past a fault it keeps a slot a later invocation may
//! have snapshotted.
//!
//! **Speculative staging** runs beside the demand row, so the drive keeps
//! reading the layers ahead while a layer's workers wait on its own cold set.
//! Its requests come from two places: the pipeline thread's predictions for
//! the rows within two (`blend`'s list, router votes, Markov and recent
//! routing ranked together), and, for a wide (prefill) row, the stager's own
//! lookahead — when it begins the row, the next rows' highest-scored experts
//! in its pad score table, which is what recent passes routed there. A decode
//! row leaves the coming rows to the pipeline's predictions (`begin_row`). A
//! lookahead read may only evict a slot scoring below the candidate. A
//! speculative read never delays a demand read: the readers take demand jobs
//! first (`read_queue`), at most [`SPEC_IN_FLIGHT`] speculative reads are out
//! at once so three readers in four are always free for demand, and a speculative
//! read the wave catches up with before it starts is promoted to demand. Its
//! victims are any held slot but the demand row's routed experts, a pinned
//! one, or a slot staged ahead and not yet reached (`pad`'s fresh slots) —
//! VRAM-backed ones first, then the coldest — under the ordinary reclaim rule
//! (`reclaim`): an evicted slot is reused once every invocation that could
//! have snapshotted its address is done. A prediction for a row the stager has
//! already begun is dropped: that row's demand covers it. As a speculative
//! read lands, the stager reports it to the pipeline thread (`landed_ahead`),
//! which lists the expert from the pad for the device's read-ahead.

use super::cache::pinned_layer_count;
use super::cpu_pool::backoff::{Backoff, COOL_DOWN, INTER_LAYER_SPINS, LOW_POWER_CYCLES};
use super::dispatch::{AbortWord, SummaryRing, SPIN_LIMIT_NS};
use super::pack::ExpertPack;
use super::pad::{Fresh, Pad, PadBook, PadSlot};
use super::read_latency::ReadSource;
use super::read_queue::{Priority, ReadQueue};
use super::reclaim::ReclaimClock;
use super::residency::Residency;
use super::resident::{plan, Bands, HostBacked, HostScores};
use super::types::PipelineStats;
use super::warm_tier::WarmTier;
use candle::direct_io::AlignedScratch;
use candle::Result;
use candle_kernels::simple::moe_bucketize::{SUMMARY_COUNT_MASK, SUMMARY_CPU_BIT};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ptr::read_volatile;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// Reader threads, one pack file handle each: the queue depth the drive sees.
/// One whole-expert read already saturates the dev-box drive (~3 GB/s at
/// 819 KB, §0.10.2); eight covers smaller records and faster drives.
pub(crate) const NVME_QD: usize = 8;

/// Speculative reads out at once: a quarter of the readers, so a demand read
/// shares the drive with at most two of them. Half the readers measured the
/// same decode all-GPU and ~2% slower with the CPU class taking cold experts
/// (Flash-Next, RTX 4090 Laptop, 2026-10-17: mean pack read 1.15 → 0.90 ms),
/// whose pool waits on every demand read; three quarters ~4% slower than
/// half: the extra speculative reads lengthened the demand reads queued
/// behind them. One measured worse the other way: the mean pack read did not
/// fall further, and the reads that land ahead are pack misses that never
/// happen — without them decode's cold misses rose ~15–25%.
const SPEC_IN_FLIGHT: usize = NVME_QD / 4;

/// Adjacent pack records a reader serves in one request (≈ 15 MiB at the
/// Q3_KO rung's 1.9 MiB stride): a wide prefill's cold set, queued in expert
/// order, reads its neighbouring experts together, paying the drive's
/// per-request cost once. (Measured at first while the pack's reads were
/// serialised by a cached view of the file — `direct_io::open_direct` — where
/// it was worth 12%; with the reads overlapped the drive is full at eight
/// single-record requests, and runs only trim the request count.)
const RUN_RECORDS: usize = 8;

/// The stager's low-power wait for a landing past its hot spins: `tpause` for
/// ~2 µs — a landing comes over the channel, which no line watches, so the
/// wait is short against a ~0.9 ms read.
const LANDING_PAUSE_CYCLES: u64 = 6_000;

/// Rows ahead of the one just begun that the stager's lookahead stages for.
const LOOKAHEAD_ROWS: usize = 2;

/// Experts per row the lookahead queues — about one wide decode row's cold
/// set, so a row's predictable misses can be on the drive a layer early.
const LOOKAHEAD_PER_ROW: usize = 32;

/// The pad score a row's expert needs before the lookahead stages it: a
/// decode routing credits 1.0 and a pass decays it by 0.85, so this admits an
/// expert decode routed there within the last four passes.
const LOOKAHEAD_MIN_SCORE: f32 = 0.5;

/// The summary bits the stager reads. The count is the field below the CPU
/// class's bit (`moe_bucketize_cpu.cuh`).
const SUMMARY_COUNT: u32 = SUMMARY_COUNT_MASK;
const SUMMARY_CPU: u32 = 1 << SUMMARY_CPU_BIT;
const SUMMARY_PINNED: u32 = 1 << 29;
const SUMMARY_COLD: u32 = 1 << 30;
const SUMMARY_DECODE: u32 = 1 << 31;

/// What the stager is told.
pub(crate) enum StagerMsg {
    /// A routed row: its summary is in ring slot `slot` once word
    /// `summary_word` is.
    Routed {
        row: usize,
        slot: usize,
        summary_word: u32,
        ticket: u64,
    },
    /// The predictor expects `row` to route to `experts`: stage the cold ones
    /// ahead of it.
    Stage { row: usize, experts: Vec<usize> },
    /// A reader finished pad slot `slot`. `opens_request` on the first record
    /// of each request a reader made, so the requests can be counted.
    Landed {
        slot: usize,
        result: Result<()>,
        read_ns: u64,
        source: ReadSource,
        opens_request: bool,
    },
}

/// Where a reader's bytes come from.
enum Source {
    Pack {
        row: usize,
        expert: usize,
    },
    /// A pageable warm slot's record.
    Warm(usize),
}

struct ReadJob {
    slot: usize,
    source: Source,
}

/// A queued speculative read: `(row, expert)`, the score a victim must sit
/// below — infinite for the pipeline's predictions (its blended list for the
/// rows within two), the candidate's own pad score for the stager's
/// lookahead — and which of the two asked for it.
#[derive(Clone, Copy)]
struct SpecRead {
    row: usize,
    expert: usize,
    floor: f32,
    predicted: bool,
}

/// One routed row with cold experts still unpublished.
struct Demand {
    row: usize,
    /// The invocation `T` whose cold set this is.
    ticket: u64,
    /// Experts this row routed to — Rule R′'s protected set.
    routed: Vec<bool>,
    /// Cold experts waiting for a slot.
    unassigned: VecDeque<usize>,
    /// Cold experts whose read is in flight.
    pending: HashSet<usize>,
}

/// Everything the stager owns or shares.
pub(crate) struct StagerCtx {
    pub(crate) pack: Arc<ExpertPack>,
    pub(crate) warm: Arc<WarmTier>,
    pub(crate) pad: Arc<Pad>,
    pub(crate) residency: Arc<Mutex<Residency>>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) ring: Arc<SummaryRing>,
    pub(crate) abort: Arc<AbortWord>,
    /// The ticket of the last summary this thread has read — the forward
    /// thread's run-ahead bound on the ring.
    pub(crate) consumed: Arc<AtomicU64>,
    /// Where a speculative read that landed is reported, as `(row, expert)`:
    /// the pipeline thread lists it from the pad for read-ahead.
    pub(crate) landed_ahead: mpsc::Sender<(usize, usize)>,
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    pub(crate) rows: usize,
    pub(crate) n_experts: usize,
    /// The startup fill's residents, `(pad slot, row, expert)`: the pad's
    /// resident region as the fill left it.
    pub(crate) residents: Vec<(usize, usize, usize)>,
    /// How many pad slots may be resident: the resident region's size.
    pub(crate) resident_capacity: usize,
    /// Which experts hold a resident slot, for the VRAM eviction's cost
    /// classes; the stager keeps it current as it relabels.
    pub(crate) host_backed: Arc<HostBacked>,
}

struct Stager {
    ctx: StagerCtx,
    book: PadBook,
    rx: mpsc::Receiver<StagerMsg>,
    jobs: Arc<ReadQueue<ReadJob>>,
    routed: VecDeque<(usize, usize, u32, u64)>,
    demand: Option<Demand>,
    spec: VecDeque<SpecRead>,
    loading: HashSet<(usize, usize)>,
    in_flight: usize,
    /// Speculative reads out, by expert, with the pad slot each is landing in —
    /// at most [`SPEC_IN_FLIGHT`]. One the demand row routes is promoted and
    /// leaves this map.
    spec_loads: HashMap<(usize, usize), usize>,
    /// Speculative reads issued at the pipeline's request rather than the
    /// lookahead's, until their row settles — to judge the two apart.
    predicted_ahead: HashSet<(usize, usize)>,
    /// The speculative queue's head found no slot it may take: wait for the
    /// next message (a landing, a routed row, a prediction) before looking
    /// again, instead of spinning on the same answer.
    spec_stalled: bool,
    /// The row last begun was wide ([`wide_row`]): a prefill's, whose next row
    /// is read ahead whole and whose speculative reads may use every reader.
    wide: bool,
    last_row: Option<usize>,
    disconnected: bool,
    /// Pad pins held for the CPU pool's experts, `(key, row, expert)` in key
    /// order: the next invocation may copy each from its pad slot into VRAM
    /// (the carry, `moe_bucketize_live.cuh`), so it is not evicted until
    /// every invocation up to `key`, that one, is done.
    carry_pins: VecDeque<(u64, usize, usize)>,
    /// How the loop waits when a pass made no progress (`cpu_pool::backoff`).
    backoff: Backoff,
    /// When the stager last heard of inference — a routed row, a prediction,
    /// a landing. Inside the cool-down after it, a pass that made no progress
    /// waits in low power; past it, asleep on the channel.
    active: Instant,
    /// Decode's host accesses, the resident set's policy input (`resident`).
    host: HostScores,
    /// How many residents each layer may hold, around the startup fill's.
    bands: Bands,
}

/// The share of a layer's experts a row must route to count as wide
/// (`1 / WIDE_ROW_SHARE`).
const WIDE_ROW_SHARE: usize = 4;

/// Whether a row routing `routed` of a layer's `n_experts` is a wide one — a
/// prefill wave's, whose successor's voted set is staged ahead and whose
/// speculative reads may use every reader.
///
/// A decode row routes a few experts per token; a prefill row of a few hundred
/// tokens or more routes about half its layer. The second's staging keeps the
/// drive busy through the layer; the first keeps three readers in four free
/// for its demand.
fn wide_row(routed: usize, n_experts: usize) -> bool {
    n_experts > 0 && routed * WIDE_ROW_SHARE >= n_experts
}

/// Whether an evicted pad slot of `row`, whose entry is already emptied, may be
/// overwritten now: every invocation up to the row's retire `key` is done
/// (`reclaimable`), or `key` is the demand window's own invocation `T` of the
/// row (`demand = (row, T)`), whose snapshot holds only the experts it routed —
/// which the caller never evicts.
fn overwritable(reclaimable: bool, demand: Option<(usize, u64)>, row: usize, key: u64) -> bool {
    reclaimable || demand == Some((row, key))
}

/// Whether the GPU has begun an invocation after the demand's own `ticket`:
/// tickets are global and increase with every invocation, and the next one's
/// bucketize starts only once every launch of `ticket` has completed — which,
/// with a cold expert of `ticket` unpublished, only a fault allows.
fn demand_left_behind(latest_started: u64, ticket: u64) -> bool {
    latest_started > ticket
}

/// Marks the stager dead, raises the abort word and closes the readers' queue
/// when dropped — including on unwind: a cold expert it was staging will never
/// be published, so every worker waiting on one must give it up and fail its
/// forward.
struct DeadGuard {
    dead: Arc<AtomicBool>,
    abort: Arc<AbortWord>,
    jobs: Arc<ReadQueue<ReadJob>>,
}

impl Drop for DeadGuard {
    fn drop(&mut self) {
        self.abort.raise();
        self.dead.store(true, Ordering::Release);
        self.jobs.close();
    }
}

/// Spawn the stager and its readers. Returns the stager's sender.
pub(crate) fn spawn_stager(
    ctx: StagerCtx,
    dead: Arc<AtomicBool>,
) -> Result<mpsc::Sender<StagerMsg>> {
    let (tx, rx) = mpsc::channel::<StagerMsg>();
    let jobs = Arc::new(ReadQueue::<ReadJob>::new());
    for handle in 0..NVME_QD {
        let jobs = jobs.clone();
        let done = tx.clone();
        let pack = ctx.pack.clone();
        let warm = ctx.warm.clone();
        let pad = ctx.pad.clone();
        std::thread::Builder::new()
            .name(format!("expert-reader-{handle}"))
            .spawn(move || reader(handle, &jobs, &done, &pack, &warm, &pad))
            .map_err(|e| candle::Error::Msg(format!("expert stager: reader spawn: {e}")))?;
    }
    let book = PadBook::new(ctx.pad.num_slots(), ctx.rows, ctx.n_experts, &ctx.residents);
    let host = HostScores::new(ctx.rows, ctx.n_experts);
    let mut startup_counts = vec![0usize; ctx.rows];
    for &(_, row, _) in &ctx.residents {
        startup_counts[row] += 1;
    }
    let bands = Bands::around(&startup_counts);
    let mut s = Stager {
        ctx,
        book,
        rx,
        jobs: jobs.clone(),
        routed: VecDeque::new(),
        demand: None,
        spec: VecDeque::new(),
        loading: HashSet::new(),
        in_flight: 0,
        spec_loads: HashMap::new(),
        predicted_ahead: HashSet::new(),
        spec_stalled: false,
        wide: false,
        last_row: None,
        disconnected: false,
        carry_pins: VecDeque::new(),
        backoff: Backoff::new(INTER_LAYER_SPINS),
        active: Instant::now(),
        host,
        bands,
    };
    std::thread::Builder::new()
        .name("expert-stager".into())
        .spawn(move || {
            let _guard = DeadGuard {
                dead,
                abort: s.ctx.abort.clone(),
                jobs,
            };
            if let Err(e) = s.run() {
                tracing::error!(
                    target: "candle_transformers::expert_lre",
                    "expert stager: a cold expert could not be staged, aborting: {e}"
                );
            }
        })
        .map_err(|e| candle::Error::Msg(format!("expert stager: spawn: {e}")))?;
    Ok(tx)
}

impl Source {
    /// Whether `next`'s record follows this one's in the pack: the same row,
    /// the next expert — so one read serves both.
    fn continued_by(&self, next: &Source) -> bool {
        matches!(
            (self, next),
            (Source::Pack { row, expert }, Source::Pack { row: r, expert: e })
                if r == row && *e == expert + 1
        )
    }
}

/// A reader: one pack handle, demand jobs first, and a queued run of adjacent
/// pack records — up to [`RUN_RECORDS`] — in one request.
fn reader(
    handle: usize,
    jobs: &ReadQueue<ReadJob>,
    done: &mpsc::Sender<StagerMsg>,
    pack: &ExpertPack,
    warm: &WarmTier,
    pad: &Pad,
) {
    let stride = pad.stride();
    let mut bounce = AlignedScratch::new();
    while let Some(run) = jobs.pop_run(RUN_RECORDS, |a, b| a.source.continued_by(&b.source)) {
        let t = Instant::now();
        let landed: Vec<(usize, Result<()>, ReadSource)> = match (run.len(), &run[0].source) {
            (1, source) => {
                let job = &run[0];
                // SAFETY: the stager hands slot `job.slot` to exactly one reader
                // and neither publishes nor reuses it until this job reports back.
                let dest = unsafe { pad.slot_mut(job.slot) };
                let (result, from) = match *source {
                    Source::Pack { row, expert } => (
                        pack.read_run_with_handle(handle, row, expert, dest),
                        ReadSource::Pack,
                    ),
                    Source::Warm(slot) => {
                        dest.copy_from_slice(warm.slot_ref(slot, stride));
                        (Ok(()), ReadSource::Paged)
                    }
                };
                vec![(job.slot, result, from)]
            }
            (n, &Source::Pack { row, expert }) => {
                let read = bounce
                    .ensure(n * stride)
                    .map_err(|e| candle::Error::Msg(format!("expert reader: bounce buffer: {e}")))
                    .and_then(|()| {
                        pack.read_run_with_handle(
                            handle,
                            row,
                            expert,
                            bounce.as_mut_slice(n * stride),
                        )
                    });
                let read = read.map_err(|e| e.to_string());
                run.iter()
                    .enumerate()
                    .map(|(i, job)| {
                        let result = match &read {
                            Ok(()) => {
                                // SAFETY: as above — the stager handed this
                                // reader each slot of the run.
                                let dest = unsafe { pad.slot_mut(job.slot) };
                                dest.copy_from_slice(
                                    &bounce.as_slice(n * stride)[i * stride..(i + 1) * stride],
                                );
                                Ok(())
                            }
                            Err(e) => Err(candle::Error::Msg(e.clone())),
                        };
                        (job.slot, result, ReadSource::Pack)
                    })
                    .collect()
            }
            (_, Source::Warm(_)) => unreachable!("a run extends only over pack records"),
        };
        // Each record's share of the request, so the read-latency figures and
        // the miss-cost model stay per record.
        let read_ns = t.elapsed().as_nanos() as u64 / landed.len() as u64;
        for (i, (slot, result, source)) in landed.into_iter().enumerate() {
            let msg = StagerMsg::Landed {
                slot,
                result,
                read_ns,
                source,
                opens_request: i == 0,
            };
            if done.send(msg).is_err() {
                return;
            }
        }
    }
}

impl Stager {
    fn run(&mut self) -> Result<()> {
        loop {
            if self.ctx.abort.is_raised() {
                return Ok(());
            }
            loop {
                match self.rx.try_recv() {
                    Ok(m) => self.handle(m)?,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.disconnected = true;
                        break;
                    }
                }
            }
            let mut progressed = false;
            if self.demand.is_none() {
                if let Some(&(row, slot, word, ticket)) = self.routed.front() {
                    if self.ctx.ring.ready(slot, word) {
                        self.routed.pop_front();
                        self.begin_row(row, slot, ticket)?;
                        progressed = true;
                    }
                }
            }
            if self.demand.is_some() {
                progressed |= self.assign()?;
            }
            progressed |= self.serve_spec()?;
            let idle = self.routed.is_empty() && self.demand.is_none() && self.in_flight == 0;
            if idle && (self.spec.is_empty() || self.spec_stalled) {
                if self.disconnected {
                    return Ok(());
                }
                match self.rx.recv() {
                    Ok(m) => self.handle(m)?,
                    Err(_) => self.disconnected = true,
                }
            } else if progressed {
                self.backoff.reset();
            } else {
                self.wait_step()?;
            }
        }
    }

    /// One wait after a pass that made no progress, the way the pool's threads
    /// wait (`cpu_pool::backoff`): hot spins first, then low power —
    /// - with the next routed row's summary still to come, `umwait` on its
    ///   sequence word, which the device's bucketize stores: the read it
    ///   starts is on the layer's critical path, so the store wakes the wait
    ///   at once;
    /// - otherwise — reads in flight, whose landings come over the channel —
    ///   `tpause` for ~2 µs, then the channel again;
    /// - and once inference has been quiet for the cool-down, asleep on the
    ///   channel, a millisecond at a time.
    fn wait_step(&mut self) -> Result<()> {
        if self.demand.is_none() {
            if let Some(&(_, slot, word, _)) = self.routed.front() {
                let p = self.ctx.ring.word_ptr(slot);
                // SAFETY: the mapped ring outlives the stager; the device
                // writes the word.
                let seen = unsafe { read_volatile(p) };
                if seen != word {
                    // SAFETY: as above.
                    unsafe { self.backoff.wait_u32(p, seen, LOW_POWER_CYCLES) };
                }
                return Ok(());
            }
        }
        if self.active.elapsed() < COOL_DOWN {
            self.backoff.pause(LANDING_PAUSE_CYCLES);
            return Ok(());
        }
        match self.rx.recv_timeout(Duration::from_millis(1)) {
            Ok(m) => self.handle(m),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.disconnected = true;
                Ok(())
            }
        }
    }

    fn handle(&mut self, m: StagerMsg) -> Result<()> {
        // Any message can change what the speculative queue may take, and
        // every one is inference running.
        self.spec_stalled = false;
        self.active = Instant::now();
        match m {
            StagerMsg::Routed {
                row,
                slot,
                summary_word,
                ticket,
            } => self.routed.push_back((row, slot, summary_word, ticket)),
            // The pipeline's prediction for a row within two: it may evict
            // any slot a speculative read may.
            StagerMsg::Stage { row, experts } => {
                for e in experts {
                    self.queue_spec(row, e, f32::INFINITY, true);
                }
            }
            StagerMsg::Landed {
                slot,
                result,
                read_ns,
                source,
                opens_request,
            } => {
                result?;
                self.in_flight -= 1;
                let (row, expert) = self.book.landed(slot);
                self.loading.remove(&(row, expert));
                // Still speculative — not promoted to a demand read by its row
                // beginning — so its row is ahead and may list it.
                let ahead = self.spec_loads.remove(&(row, expert)).is_some();
                let addr = self.ctx.pad.slot_addr(slot);
                self.residency()?.set_pad(row, expert, Some((slot, addr)));
                // A closed channel means the pipeline thread is gone; its guard
                // has raised the abort word.
                if ahead {
                    let _ = self.ctx.landed_ahead.send((row, expert));
                }
                let slow = self.ctx.stats.lock().is_ok_and(|mut s| {
                    s.staged_bytes += self.ctx.pad.stride();
                    s.stage_read_ns += read_ns;
                    if opens_request && source == ReadSource::Pack {
                        s.pack_requests += 1;
                    }
                    match source {
                        ReadSource::Pack => s.pack_reads.record(read_ns),
                        ReadSource::Paged => s.paged_reads.record(read_ns),
                    }
                });
                if slow {
                    tracing::warn!(
                        target: "candle_transformers::expert_lre",
                        "expert stager: row {row} expert {expert} took {:.1} ms to read from its {} \
                         — cold workers give an expert up at {} ms",
                        read_ns as f64 / 1e6,
                        source.name(),
                        SPIN_LIMIT_NS / 1_000_000
                    );
                }
                let done = self.demand.as_mut().is_some_and(|d| {
                    if d.row == row {
                        d.pending.remove(&expert);
                    }
                    d.pending.is_empty() && d.unassigned.is_empty()
                });
                if done {
                    self.demand = None;
                }
            }
        }
        Ok(())
    }

    /// Re-choose the resident set at a pass boundary (`resident::plan`): a
    /// relabel each, no bytes moved — a promoted slot already holds its
    /// expert, a demoted one keeps its own until churn evicts it.
    fn rebalance(&mut self) -> Result<()> {
        let (mut churn, mut residents) = (Vec::new(), Vec::new());
        for (slot, row, expert, resident) in self.book.held() {
            let s = (slot, row, self.host.score(row, expert));
            if resident {
                residents.push(s);
            } else {
                churn.push(s);
            }
        }
        let relabels = plan(churn, residents, self.ctx.resident_capacity, &self.bands);
        if relabels.is_empty() {
            return Ok(());
        }
        // A demoted expert stays host-backed if the warm tier also holds a
        // pageable copy of it: its reload needs no pack read either way.
        let residency = self.ctx.residency.clone();
        let places = residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert stager: residency poisoned".into()))?;
        let label = |book: &mut PadBook, slot: usize, resident: bool, backed: &HostBacked| {
            book.set_resident(slot, resident);
            if let PadSlot::Held(row, expert) = book.state(slot) {
                backed.set(
                    row,
                    expert,
                    resident || places.place(row, expert).warm.is_some(),
                );
            }
        };
        let (mut promoted, mut demoted) = (0usize, 0usize);
        for r in &relabels {
            if let Some(d) = r.demote {
                label(&mut self.book, d, false, &self.ctx.host_backed);
                demoted += 1;
            }
            label(&mut self.book, r.promote, true, &self.ctx.host_backed);
            promoted += 1;
        }
        drop(places);
        let mut rows = vec![0usize; self.ctx.rows];
        for (_, row, _, resident) in self.book.held() {
            if resident {
                rows[row] += 1;
            }
        }
        if let Ok(mut s) = self.ctx.stats.lock() {
            s.resident_promotions += promoted;
            s.resident_demotions += demoted;
            s.resident_rows = rows;
        }
        Ok(())
    }

    fn residency(&self) -> Result<std::sync::MutexGuard<'_, Residency>> {
        self.ctx
            .residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert stager: residency poisoned".into()))
    }

    /// The row's summary is in: score it, and stage its cold set.
    fn begin_row(&mut self, row: usize, slot: usize, ticket: u64) -> Result<()> {
        self.ctx.clock.observe(ticket);
        // SAFETY: `ready` returned for this word, and the forward thread does
        // not rewrite the slot until `consumed` passes this ticket.
        let summary: Vec<u32> = unsafe { self.ctx.ring.read(slot) }.to_vec();
        self.ctx.consumed.store(ticket, Ordering::Release);
        if let Some(last) = self.last_row {
            if row <= last {
                self.book.decay(0.85);
            }
            // A pass boundary — the one place the resident set changes — is a
            // row before the last. The same row again is not one: the draft
            // head's repeated steps are one pass to decode's host scores, which
            // would otherwise age, and relabel, once per draft step.
            if row < last && self.host.end_pass() {
                self.rebalance()?;
            }
        }
        self.last_row = Some(row);
        let mut routed = vec![false; self.ctx.n_experts];
        let mut cold = Vec::new();
        let mut pooled = Vec::new();
        for (e, &w) in summary.iter().enumerate() {
            if w & SUMMARY_COUNT == 0 {
                continue;
            }
            routed[e] = true;
            let decode = w & SUMMARY_DECODE != 0;
            self.book.credit(row, e, decode);
            // Decode reaching the host for it: pinned or cold, not VRAM.
            if decode && w & (SUMMARY_COLD | SUMMARY_PINNED) != 0 {
                self.host.credit(row, e);
            }
            if w & SUMMARY_COLD != 0 {
                cold.push(e);
            }
            if w & SUMMARY_CPU != 0 {
                pooled.push(e);
            }
        }
        self.pin_carries(row, ticket, &pooled)?;
        // The wave has reached this row: what was staged ahead for it is no
        // longer ahead, and a prediction for it not yet read is its demand's.
        let hit: Vec<usize> = self
            .book
            .settle_row(row)
            .into_iter()
            .filter(|&e| routed[e])
            .collect();
        let predicted_hits = hit
            .iter()
            .filter(|&&e| self.predicted_ahead.contains(&(row, e)))
            .count();
        self.predicted_ahead.retain(|&(r, _)| r != row);
        self.spec.retain(|s| s.row != row);
        // By what the row routes, not by what is still cold: staged ahead, a
        // prefill row's cold set shrinks below the mark at ×2, which took its
        // speculative reads back to decode's cap and left ~3,300 of its
        // successor's stagings unstarted a config (Flash-Next, 16 GB laptop).
        self.wide = wide_row(routed.iter().filter(|&&r| r).count(), self.ctx.n_experts);
        // A decode row's coming rows are predicted from the router look-ahead
        // (the pipeline's `Stage`): queuing the pad's own scores beside them
        // put 30–80 unrouted reads a step on the drive ahead of the predicted
        // ones, and without them Flash-Next decode ran 11% faster all-GPU
        // (RTX 4090 Laptop, 2026-10-17: 11.5 → 12.8 t/s).
        if self.wide {
            self.look_ahead(row);
        }
        if let Ok(mut s) = self.ctx.stats.lock() {
            s.staged_ahead_routed += hit.len();
            s.staged_predicted_routed += predicted_hits;
        }
        // A speculative read already in flight for one of them counts as its
        // demand read — promoted ahead of the speculative queue if no reader
        // has taken it yet; anything published since bucketize read 0 needs
        // nothing.
        let mut d = Demand {
            row,
            ticket,
            routed,
            unassigned: VecDeque::new(),
            pending: HashSet::new(),
        };
        let mut promoted = Vec::new();
        {
            let mut r = self
                .ctx
                .residency
                .lock()
                .map_err(|_| candle::Error::Msg("expert stager: residency poisoned".into()))?;
            for e in cold {
                if self.loading.contains(&(row, e)) {
                    promoted.extend(self.spec_loads.remove(&(row, e)));
                    d.pending.insert(e);
                    continue;
                }
                // A lazy victim with a cold fallback reads cold only once a
                // claim has zeroed its entries on the device, and the pipeline
                // thread may not have collected that claim yet: book the
                // eviction here, or the entry would still read VRAM and this
                // cold wait would go unanswered.
                if r.place(row, e).offered_cold {
                    r.device_evicted(row, e);
                }
                if r.place(row, e).entry() == 0 {
                    d.unassigned.push_back(e);
                }
            }
        }
        for slot in promoted {
            self.jobs.promote(|j| j.slot == slot);
        }
        if let Ok(mut s) = self.ctx.stats.lock() {
            s.staged_cold += d.unassigned.len();
        }
        if !d.unassigned.is_empty() || !d.pending.is_empty() {
            self.demand = Some(d);
        }
        Ok(())
    }

    /// Pin the pad slots of `pooled` — the experts the CPU pool serves for
    /// invocation `ticket` of `row` — until the next invocation is done: it may
    /// copy them into VRAM (the carry), and under the demand rule a pad slot
    /// its own row does not route could otherwise be overwritten while it
    /// reads. Every one is pinned, whatever the split's `promote` says, so no
    /// copy ever runs unpinned; a pin with no copy costs only the slot's
    /// eviction for one invocation. First releases the pins whose invocation
    /// is done.
    ///
    /// A cold expert is pinned here, at `begin_row`, before its read has
    /// landed: the pin count is kept per `(row, expert)`, not per slot, so it
    /// holds whichever pad slot the read lands in from the moment it does.
    fn pin_carries(&mut self, row: usize, ticket: u64, pooled: &[usize]) -> Result<()> {
        let clock = &self.ctx.clock;
        let done = self
            .carry_pins
            .iter()
            .take_while(|&&(key, _, _)| clock.reclaimable(key))
            .count();
        if done == 0 && pooled.is_empty() {
            return Ok(());
        }
        let residency = self.ctx.residency.clone();
        let mut r = residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert stager: residency poisoned".into()))?;
        for (_, pr, pe) in self.carry_pins.drain(..done) {
            r.pin_pad(pr, pe, -1);
        }
        for &e in pooled {
            r.pin_pad(row, e, 1);
            self.carry_pins.push_back((ticket + 1, row, e));
        }
        Ok(())
    }

    /// Queue a speculative read of `(row, expert)` unless one is queued; it may
    /// evict only a slot scoring below `floor`. A pipeline prediction for a
    /// read the lookahead already queued takes it over — its floor and its
    /// source — so it is neither dropped as not worth a slot nor counted as
    /// the lookahead's.
    fn queue_spec(&mut self, row: usize, expert: usize, floor: f32, predicted: bool) {
        if let Some(s) = self
            .spec
            .iter_mut()
            .find(|s| s.row == row && s.expert == expert)
        {
            if predicted {
                s.floor = s.floor.max(floor);
                s.predicted = true;
            }
        } else {
            self.spec.push_back(SpecRead {
                row,
                expert,
                floor,
                predicted,
            });
        }
    }

    /// The rows after `row` the wave reaches next — the next pass's leading
    /// rows past the last one — each with its highest-scored experts queued
    /// for a speculative read: what the recent passes routed there. A queued
    /// expert already readable by the device is dropped when its turn comes
    /// (`serve_spec`). Each may evict only a lower-scored slot, so the
    /// lookahead never trades a copy the pad's own history rates higher.
    /// The permanently resident prefix never misses, so its rows are skipped.
    fn look_ahead(&mut self, row: usize) {
        let rows = self.ctx.rows;
        let pinned = pinned_layer_count(rows);
        for hop in 1..=LOOKAHEAD_ROWS {
            let target = (row + hop) % rows;
            if target < pinned {
                continue;
            }
            for (e, score) in self
                .book
                .top_scored(target, LOOKAHEAD_MIN_SCORE, LOOKAHEAD_PER_ROW)
            {
                self.queue_spec(target, e, score, false);
            }
        }
    }

    /// Give the demand row's unassigned cold experts slots and reads.
    fn assign(&mut self) -> Result<bool> {
        let Some(mut d) = self.demand.take() else {
            return Ok(false);
        };
        let out = self.assign_into(&mut d);
        if !(d.unassigned.is_empty() && d.pending.is_empty()) {
            self.demand = Some(d);
        }
        out
    }

    fn assign_into(&mut self, d: &mut Demand) -> Result<bool> {
        // The GPU has begun an invocation after `T` — which it does with one of
        // `T`'s cold experts unpublished only when `T`'s forward faulted and its
        // workers gave the expert up. Nothing waits on the rest of the cold
        // set, so it is abandoned (the next forward stages what it routes), and
        // Rule R′, which assumes the GPU is inside `T`, no longer holds.
        if demand_left_behind(self.ctx.clock.latest_started(), d.ticket) {
            let abandoned = !d.unassigned.is_empty();
            d.unassigned.clear();
            return Ok(abandoned);
        }
        // Queued together once the loop ends, in expert order, so the readers
        // find the row's adjacent records side by side and read them as runs.
        let mut batch = Vec::new();
        let assigned = self.assign_slots(d, &mut batch);
        if !batch.is_empty() && !self.jobs.push_all(batch, Priority::Demand) {
            candle::bail!("expert stager: readers gone");
        }
        assigned
    }

    /// The slot-assigning loop of [`Self::assign_into`], each started read's
    /// job added to `batch`.
    fn assign_slots(&mut self, d: &mut Demand, batch: &mut Vec<ReadJob>) -> Result<bool> {
        let mut progressed = false;
        while let Some(&e) = d.unassigned.front() {
            let slot = match self.book.take_free() {
                Some(s) => s,
                None => {
                    // Rule R′: anything but this row's routed experts, and
                    // never a slot a read-ahead listing pins. A slot staged
                    // ahead goes only when nothing else can. Chosen and evicted
                    // under one residency lock, so no pin lands in between.
                    let mut r = self.residency()?;
                    let victim = self
                        .book
                        .victims(1, Fresh::Last, |vr, ve| {
                            let p = r.place(vr, ve);
                            (!(vr == d.row && d.routed[ve]) && p.pins == 0)
                                .then_some(p.vram.is_some())
                        })
                        .first()
                        .copied();
                    // Every slot is loading or pinned: wait for a landing.
                    let Some(v) = victim else { break };
                    if !self.evict(&mut r, v, Some((d.row, d.ticket))) {
                        // The GPU has left `T` (its forward faulted): the
                        // victim's row began an invocation that may read it.
                        // The next pass abandons the demand (above).
                        break;
                    }
                    drop(r);
                    if let Ok(mut s) = self.ctx.stats.lock() {
                        s.pad_evictions += 1;
                    }
                    v
                }
            };
            d.unassigned.pop_front();
            batch.push(self.start(slot, d.row, e, Priority::Demand)?);
            d.pending.insert(e);
            progressed = true;
        }
        Ok(progressed)
    }

    /// Empty held pad slot `v`'s entry so the slot can take another read, and
    /// say whether it may: the entry is stored first and the row's retire key
    /// read after (`reclaim`), so an invocation of the row that begins in
    /// between either read the new entry or is named by the key. A slot some
    /// begun invocation may have snapshotted is not overwritten — its entry is
    /// restored, which is always safe (the slot still holds the expert), and
    /// `false` returned. `demand` is the demand window `(row, T)`: under Rule R′
    /// the GPU is inside `T`, whose snapshot holds only its routed experts, so a
    /// slot of `T`'s own row that `T` did not route is free as long as `T` is
    /// the row's latest invocation — which a forward that faulted and moved on
    /// is not.
    fn evict(&self, r: &mut Residency, v: usize, demand: Option<(usize, u64)>) -> bool {
        let PadSlot::Held(vr, ve) = self.book.state(v) else {
            unreachable!("a pad victim is a held slot")
        };
        r.set_pad(vr, ve, None);
        let clock = &self.ctx.clock;
        let key = clock.retire_key(vr);
        if overwritable(clock.reclaimable(key), demand, vr, key) {
            return true;
        }
        r.set_pad(vr, ve, Some((v, self.ctx.pad.slot_addr(v))));
        false
    }

    /// Start `(row, expert)`'s read into `slot`.
    fn issue(&mut self, slot: usize, row: usize, expert: usize, priority: Priority) -> Result<()> {
        let job = self.start(slot, row, expert, priority)?;
        if !self.jobs.push(job, priority) {
            candle::bail!("expert stager: readers gone");
        }
        Ok(())
    }

    /// Book `(row, expert)`'s read into `slot` as started, and return the job
    /// for the readers — for [`Self::issue`], or for a batch queued at once.
    fn start(
        &mut self,
        slot: usize,
        row: usize,
        expert: usize,
        priority: Priority,
    ) -> Result<ReadJob> {
        let source = self.source(row, expert)?;
        let ahead = priority == Priority::Speculative;
        self.book.start_load(slot, row, expert, ahead);
        self.loading.insert((row, expert));
        if ahead {
            self.spec_loads.insert((row, expert), slot);
        }
        self.in_flight += 1;
        Ok(ReadJob { slot, source })
    }

    /// Where `(row, expert)`'s record comes from: its pageable warm slot, or
    /// the pack.
    fn source(&self, row: usize, expert: usize) -> Result<Source> {
        Ok(match self.residency()?.place(row, expert).warm {
            Some(ws) => {
                if let Ok(mut s) = self.ctx.stats.lock() {
                    s.staged_paged += 1;
                }
                Source::Warm(ws)
            }
            None => Source::Pack { row, expert },
        })
    }

    /// Speculative reads for the predicted cold experts, oldest prediction
    /// first, while fewer than [`SPEC_IN_FLIGHT`] are out. A slot is a free
    /// one, else a victim: not one of the demand row's routed experts, not
    /// pinned, not staged ahead itself, and reclaimable now — no invocation that
    /// could have snapshotted its address is still to finish — VRAM-backed
    /// first, then the coldest. When there is none the queue waits for the next
    /// message rather than being dropped: a landing or a routed row is what
    /// frees one.
    fn serve_spec(&mut self) -> Result<bool> {
        if self.spec_stalled {
            return Ok(false);
        }
        let mut progressed = false;
        // A wide row's read-ahead is the next layer whole, and the readers
        // still take demand jobs first, so it may use every reader; decode's
        // predictions keep three in four free for demand.
        let cap = if self.wide { NVME_QD } else { SPEC_IN_FLIGHT };
        while self.spec_loads.len() < cap {
            let Some(&SpecRead {
                row,
                expert: e,
                floor,
                predicted,
            }) = self.spec.front()
            else {
                break;
            };
            if self.loading.contains(&(row, e)) || self.residency()?.place(row, e).entry() != 0 {
                self.spec.pop_front();
                continue;
            }
            let slot = match self.book.take_free() {
                Some(s) => s,
                None => {
                    // Chosen and evicted under one residency lock, so no
                    // read-ahead pin lands in between.
                    let mut r = self.residency()?;
                    let victim = {
                        let clock = &self.ctx.clock;
                        let demand = self.demand.as_ref();
                        let book = &self.book;
                        book.victims(1, Fresh::Spare, |vr, ve| {
                            let p = r.place(vr, ve);
                            let routed = demand.is_some_and(|d| d.row == vr && d.routed[ve]);
                            (!routed
                                && p.pins == 0
                                && (p.vram.is_some() || book.score(vr, ve) < floor)
                                && clock.reclaimable(clock.begun(vr)))
                            .then_some(p.vram.is_some())
                        })
                        .first()
                        .copied()
                    };
                    let Some(v) = victim else {
                        drop(r);
                        if floor.is_finite() {
                            // Everything evictable rates at least as high:
                            // this lookahead is not worth a slot.
                            self.spec.pop_front();
                            continue;
                        }
                        if let Ok(mut s) = self.ctx.stats.lock() {
                            s.spec_stalls_no_slot += 1;
                        }
                        self.spec_stalled = true;
                        break;
                    };
                    if !self.evict(&mut r, v, None) {
                        // Its row began an invocation between the choice and
                        // the store: look again once something has moved.
                        drop(r);
                        if let Ok(mut s) = self.ctx.stats.lock() {
                            s.spec_stalls_in_use += 1;
                        }
                        self.spec_stalled = true;
                        break;
                    }
                    drop(r);
                    if let Ok(mut s) = self.ctx.stats.lock() {
                        s.pad_evictions += 1;
                    }
                    v
                }
            };
            self.spec.pop_front();
            self.issue(slot, row, e, Priority::Speculative)?;
            if predicted {
                self.predicted_ahead.insert((row, e));
            }
            if let Ok(mut s) = self.ctx.stats.lock() {
                s.staged_speculative += 1;
                s.staged_predicted += usize::from(predicted);
            }
            progressed = true;
        }
        Ok(progressed)
    }
}

#[cfg(test)]
mod tests {
    use super::{demand_left_behind, overwritable, wide_row};

    /// A quarter of the layer routed is wide; less is not, and an empty layer
    /// never is.
    #[test]
    fn a_row_is_wide_from_a_quarter_of_its_layer_routed() {
        assert!(wide_row(128, 512));
        assert!(wide_row(512, 512));
        assert!(!wide_row(127, 512));
        assert!(!wide_row(10, 512));
        assert!(!wide_row(0, 0));
    }

    /// The demand stands while the GPU is inside its invocation or behind it
    /// (the stager can be ahead of the device), and is left behind only once a
    /// later invocation has begun.
    #[test]
    fn a_demand_is_left_behind_only_by_a_later_invocation() {
        assert!(!demand_left_behind(0, 40));
        assert!(!demand_left_behind(39, 40));
        assert!(!demand_left_behind(40, 40));
        assert!(demand_left_behind(41, 40));
    }

    /// Rule R′ frees the demand row's own unrouted slots while `T` is the
    /// row's latest invocation, and no longer once a later one has begun (a
    /// faulted forward moved on); any other slot needs its retire key done.
    #[test]
    fn a_slot_is_overwritable_once_its_readers_are_done_or_inside_the_demand_window() {
        // Another row, every reader done.
        assert!(overwritable(true, Some((7, 40)), 3, 39));
        // Another row with a reader still running.
        assert!(!overwritable(false, Some((7, 40)), 3, 41));
        // The demand row, `T` itself the latest invocation begun on it.
        assert!(overwritable(false, Some((7, 40)), 7, 40));
        // The demand row after the GPU left `T`: a later invocation began.
        assert!(!overwritable(false, Some((7, 40)), 7, 45));
        // Outside a demand window only the clock decides.
        assert!(!overwritable(false, None, 7, 40));
        assert!(overwritable(true, None, 7, 40));
    }
}
