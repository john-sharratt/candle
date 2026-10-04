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
//! pad holds at least one layer of slots, so a row's cold set always fits.
//!
//! **Speculative staging** (requests from the pipeline thread's predictor) runs
//! only when no demand row is outstanding, into free slots or slots whose
//! expert is also in VRAM — an eviction that changes no entry — and under the
//! ordinary reclaim rule (`reclaim`): such a slot is reused once every
//! invocation that could have snapshotted its address is done.

use super::dispatch::{AbortWord, SummaryRing};
use super::pack::ExpertPack;
use super::pad::{Pad, PadBook, PadSlot};
use super::reclaim::{ReclaimClock, RetireList};
use super::residency::Residency;
use super::types::PipelineStats;
use super::warm_tier::WarmTier;
use candle::Result;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// Reader threads, one pack file handle each: the queue depth the drive sees.
/// One whole-expert read already saturates the dev-box drive (~3 GB/s at
/// 819 KB, §0.10.2); eight covers smaller records and faster drives.
pub(crate) const NVME_QD: usize = 8;

/// The summary bits the stager reads.
const SUMMARY_COUNT: u32 = 0x1fff_ffff;
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
    /// A reader finished pad slot `slot`.
    Landed {
        slot: usize,
        result: Result<()>,
        read_ns: u64,
    },
}

/// Where a reader's bytes come from.
enum Source {
    Pack { row: usize, expert: usize },
    /// A pageable warm slot's record.
    Warm(usize),
}

struct ReadJob {
    slot: usize,
    source: Source,
}

/// One routed row with cold experts still unpublished.
struct Demand {
    row: usize,
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
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    pub(crate) rows: usize,
    pub(crate) n_experts: usize,
}

struct Stager {
    ctx: StagerCtx,
    book: PadBook,
    rx: mpsc::Receiver<StagerMsg>,
    jobs: Option<mpsc::Sender<ReadJob>>,
    routed: VecDeque<(usize, usize, u32, u64)>,
    demand: Option<Demand>,
    spec: VecDeque<(usize, usize)>,
    loading: HashSet<(usize, usize)>,
    in_flight: usize,
    retired: RetireList<usize>,
    last_row: Option<usize>,
    disconnected: bool,
}

/// Marks the stager dead and raises the abort word when dropped — including on
/// unwind: a cold expert it was staging will never be published, so every
/// worker waiting on one must trap.
struct DeadGuard {
    dead: Arc<AtomicBool>,
    abort: Arc<AbortWord>,
}

impl Drop for DeadGuard {
    fn drop(&mut self) {
        self.abort.raise();
        self.dead.store(true, Ordering::Release);
    }
}

/// Spawn the stager and its readers. Returns the stager's sender.
pub(crate) fn spawn_stager(
    ctx: StagerCtx,
    dead: Arc<AtomicBool>,
) -> Result<mpsc::Sender<StagerMsg>> {
    let (tx, rx) = mpsc::channel::<StagerMsg>();
    let (jobs_tx, jobs_rx) = mpsc::channel::<ReadJob>();
    let jobs_rx = Arc::new(Mutex::new(jobs_rx));
    for handle in 0..NVME_QD {
        let jobs_rx = jobs_rx.clone();
        let done = tx.clone();
        let pack = ctx.pack.clone();
        let warm = ctx.warm.clone();
        let pad = ctx.pad.clone();
        std::thread::Builder::new()
            .name(format!("expert-reader-{handle}"))
            .spawn(move || reader(handle, &jobs_rx, &done, &pack, &warm, &pad))
            .map_err(|e| candle::Error::Msg(format!("expert stager: reader spawn: {e}")))?;
    }
    let book = PadBook::new(ctx.pad.num_slots(), ctx.rows, ctx.n_experts);
    let mut s = Stager {
        ctx,
        book,
        rx,
        jobs: Some(jobs_tx),
        routed: VecDeque::new(),
        demand: None,
        spec: VecDeque::new(),
        loading: HashSet::new(),
        in_flight: 0,
        retired: RetireList::new(),
        last_row: None,
        disconnected: false,
    };
    std::thread::Builder::new()
        .name("expert-stager".into())
        .spawn(move || {
            let _guard = DeadGuard {
                dead,
                abort: s.ctx.abort.clone(),
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

/// A reader: one pack handle, one job at a time.
fn reader(
    handle: usize,
    jobs: &Mutex<mpsc::Receiver<ReadJob>>,
    done: &mpsc::Sender<StagerMsg>,
    pack: &ExpertPack,
    warm: &WarmTier,
    pad: &Pad,
) {
    loop {
        let job = {
            let Ok(rx) = jobs.lock() else { return };
            match rx.recv() {
                Ok(j) => j,
                Err(_) => return,
            }
        };
        let t = Instant::now();
        // SAFETY: the stager hands slot `job.slot` to exactly one reader and
        // neither publishes nor reuses it until this job reports back.
        let dest = unsafe { pad.slot_mut(job.slot) };
        let result = match job.source {
            Source::Pack { row, expert } => pack.read_into_with_handle(handle, row, expert, dest),
            Source::Warm(slot) => {
                dest.copy_from_slice(warm.slot_ref(slot, pad.stride()));
                Ok(())
            }
        };
        let msg = StagerMsg::Landed {
            slot: job.slot,
            result,
            read_ns: t.elapsed().as_nanos() as u64,
        };
        if done.send(msg).is_err() {
            return;
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
            } else {
                progressed |= self.serve_spec()?;
            }
            let idle = self.routed.is_empty() && self.demand.is_none() && self.in_flight == 0;
            if idle && self.spec.is_empty() {
                if self.disconnected {
                    return Ok(());
                }
                match self.rx.recv() {
                    Ok(m) => self.handle(m)?,
                    Err(_) => self.disconnected = true,
                }
            } else if !progressed {
                std::thread::yield_now();
            }
        }
    }

    fn handle(&mut self, m: StagerMsg) -> Result<()> {
        match m {
            StagerMsg::Routed {
                row,
                slot,
                summary_word,
                ticket,
            } => self.routed.push_back((row, slot, summary_word, ticket)),
            StagerMsg::Stage { row, experts } => {
                for e in experts {
                    if !self.spec.contains(&(row, e)) {
                        self.spec.push_back((row, e));
                    }
                }
            }
            StagerMsg::Landed {
                slot,
                result,
                read_ns,
            } => {
                result?;
                self.in_flight -= 1;
                let (row, expert) = self.book.landed(slot);
                self.loading.remove(&(row, expert));
                let addr = self.ctx.pad.slot_addr(slot);
                self.residency()?.set_pad(row, expert, Some((slot, addr)));
                if let Ok(mut s) = self.ctx.stats.lock() {
                    s.staged_bytes += self.ctx.pad.stride();
                    s.stage_read_ns += read_ns;
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
        if self.last_row.is_some_and(|l| row <= l) {
            self.book.decay(0.85);
        }
        self.last_row = Some(row);
        let mut routed = vec![false; self.ctx.n_experts];
        let mut cold = Vec::new();
        for (e, &w) in summary.iter().enumerate() {
            if w & SUMMARY_COUNT == 0 {
                continue;
            }
            routed[e] = true;
            self.book.credit(row, e, w & SUMMARY_DECODE != 0);
            if w & SUMMARY_COLD != 0 {
                cold.push(e);
            }
        }
        // A speculative read already in flight for one of them counts as its
        // demand read; anything published since bucketize read 0 needs nothing.
        let mut d = Demand {
            row,
            routed,
            unassigned: VecDeque::new(),
            pending: HashSet::new(),
        };
        {
            let r = self.residency()?;
            for e in cold {
                if self.loading.contains(&(row, e)) {
                    d.pending.insert(e);
                } else if r.place(row, e).entry() == 0 {
                    d.unassigned.push_back(e);
                }
            }
        }
        if let Ok(mut s) = self.ctx.stats.lock() {
            s.staged_cold += d.unassigned.len();
        }
        if !d.unassigned.is_empty() || !d.pending.is_empty() {
            self.demand = Some(d);
        }
        Ok(())
    }

    /// Give the demand row's unassigned cold experts slots and reads.
    fn assign(&mut self) -> Result<bool> {
        for s in self.retired.drain(&self.ctx.clock) {
            self.book.release(s);
        }
        let Some(mut d) = self.demand.take() else {
            return Ok(false);
        };
        let out = self.assign_into(&mut d);
        self.demand = Some(d);
        out
    }

    fn assign_into(&mut self, d: &mut Demand) -> Result<bool> {
        let mut progressed = false;
        while let Some(&e) = d.unassigned.front() {
            let slot = match self.book.take_free() {
                Some(s) => s,
                None => {
                    // Rule R′: anything but this row's routed experts, and
                    // never a slot a promotion copy is reading.
                    let victim = {
                        let r = self.residency()?;
                        self.book
                            .victims(1, |vr, ve| {
                                let p = r.place(vr, ve);
                                (!(vr == d.row && d.routed[ve]) && p.pins == 0)
                                    .then_some(p.vram.is_some())
                            })
                            .first()
                            .copied()
                    };
                    // Every slot is loading or pinned: wait for a landing.
                    let Some(v) = victim else { break };
                    let PadSlot::Held(vr, ve) = self.book.state(v) else {
                        unreachable!("a pad victim is a held slot")
                    };
                    self.residency()?.set_pad(vr, ve, None);
                    if let Ok(mut s) = self.ctx.stats.lock() {
                        s.pad_evictions += 1;
                    }
                    v
                }
            };
            d.unassigned.pop_front();
            self.issue(slot, d.row, e)?;
            d.pending.insert(e);
            progressed = true;
        }
        Ok(progressed)
    }

    /// Start `(row, expert)`'s read into `slot`.
    fn issue(&mut self, slot: usize, row: usize, expert: usize) -> Result<()> {
        let source = self.source(row, expert)?;
        self.book.start_load(slot, row, expert);
        self.loading.insert((row, expert));
        self.in_flight += 1;
        self.jobs
            .as_ref()
            .and_then(|j| j.send(ReadJob { slot, source }).ok())
            .ok_or_else(|| candle::Error::Msg("expert stager: readers gone".into()))
    }

    /// Where `(row, expert)`'s record comes from: its pageable warm slot, or
    /// the pack.
    fn source(&self, row: usize, expert: usize) -> Result<Source> {
        Ok(match self.residency()?.place(row, expert).warm {
            Some((ws, None)) => {
                if let Ok(mut s) = self.ctx.stats.lock() {
                    s.staged_paged += 1;
                }
                Source::Warm(ws)
            }
            _ => Source::Pack { row, expert },
        })
    }

    /// One speculative read, if a slot can be had without changing an entry.
    fn serve_spec(&mut self) -> Result<bool> {
        if self.in_flight >= NVME_QD {
            return Ok(false);
        }
        for s in self.retired.drain(&self.ctx.clock) {
            self.book.release(s);
        }
        while let Some((row, e)) = self.spec.pop_front() {
            if self.loading.contains(&(row, e)) {
                continue;
            }
            let wanted = {
                let r = self.residency()?;
                r.place(row, e).entry() == 0
            };
            if !wanted {
                continue;
            }
            let slot = match self.book.take_free() {
                Some(s) => s,
                None => {
                    let victim = {
                        let r = self.residency()?;
                        self.book
                            .victims(1, |vr, ve| {
                                let p = r.place(vr, ve);
                                (p.vram.is_some() && p.pins == 0).then_some(true)
                            })
                            .first()
                            .copied()
                    };
                    let Some(v) = victim else {
                        // Nothing evictable without changing an entry.
                        self.spec.clear();
                        return Ok(false);
                    };
                    let PadSlot::Held(vr, ve) = self.book.state(v) else {
                        unreachable!("a pad victim is a held slot")
                    };
                    self.residency()?.set_pad(vr, ve, None);
                    let key = self.ctx.clock.retire_key(vr);
                    if !self.ctx.clock.reclaimable(key) {
                        // Its old readers may still run: hold it, and look again.
                        self.book.vacate(v);
                        self.retired.push(key, v);
                        self.spec.push_front((row, e));
                        continue;
                    }
                    v
                }
            };
            self.issue(slot, row, e)?;
            if let Ok(mut s) = self.ctx.stats.lock() {
                s.staged_speculative += 1;
            }
            return Ok(true);
        }
        Ok(false)
    }
}
