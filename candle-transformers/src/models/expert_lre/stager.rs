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
//! Its requests come from two places: the pipeline thread's predictor
//! (confident layer-to-layer transitions, a few experts per layer), and the
//! stager's own lookahead — when it begins a row, the next rows' highest-scored
//! experts in its pad score table, which is what recent passes routed there. A
//! lookahead read may only evict a slot scoring below the candidate. A
//! speculative read never delays a demand read: the readers take demand jobs
//! first (`read_queue`), at most [`SPEC_IN_FLIGHT`] speculative reads are out
//! at once so half the readers are always free for demand, and a speculative
//! read the wave catches up with before it starts is promoted to demand. Its
//! victims are any held slot but the demand row's routed experts, a pinned
//! one, or a slot staged ahead and not yet reached (`pad`'s fresh slots) —
//! VRAM-backed ones first, then the coldest — under the ordinary reclaim rule
//! (`reclaim`): an evicted slot is reused once every invocation that could
//! have snapshotted its address is done. A prediction for a row the stager has
//! already begun is dropped: that row's demand covers it. As a speculative
//! read lands, the stager reports it to the pipeline thread (`landed_ahead`),
//! which lists the expert from the pad for the device's read-ahead.

use super::dispatch::{AbortWord, SummaryRing, SPIN_LIMIT_NS};
use super::pack::ExpertPack;
use super::pad::{Fresh, Pad, PadBook, PadSlot};
use super::read_latency::ReadSource;
use super::read_queue::{Priority, ReadQueue};
use super::reclaim::ReclaimClock;
use super::residency::Residency;
use super::types::PipelineStats;
use super::warm_tier::WarmTier;
use candle::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// Reader threads, one pack file handle each: the queue depth the drive sees.
/// One whole-expert read already saturates the dev-box drive (~3 GB/s at
/// 819 KB, §0.10.2); eight covers smaller records and faster drives.
pub(crate) const NVME_QD: usize = 8;

/// Speculative reads out at once: half the readers, so a demand read never
/// waits for more than the reads already in flight on the other half. Three
/// quarters measured ~4% slower on Flash-Next decode (RTX 4090 Laptop): the
/// extra speculative reads lengthened the demand reads queued behind them.
const SPEC_IN_FLIGHT: usize = NVME_QD / 2;

/// Rows ahead of the one just begun that the stager's lookahead stages for.
const LOOKAHEAD_ROWS: usize = 2;

/// Experts per row the lookahead queues — about one wide decode row's cold
/// set, so a row's predictable misses can be on the drive a layer early.
const LOOKAHEAD_PER_ROW: usize = 32;

/// The pad score a row's expert needs before the lookahead stages it: a
/// decode routing credits 1.0 and a pass decays it by 0.85, so this admits an
/// expert decode routed there within the last four passes.
const LOOKAHEAD_MIN_SCORE: f32 = 0.5;

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
        source: ReadSource,
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

/// A queued speculative read: `(row, expert)`, and the score a victim must sit
/// below — infinite for the pipeline predictor's confident transitions, the
/// candidate's own pad score for the stager's lookahead.
#[derive(Clone, Copy)]
struct SpecRead {
    row: usize,
    expert: usize,
    floor: f32,
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
    /// The speculative queue's head found no slot it may take: wait for the
    /// next message (a landing, a routed row, a prediction) before looking
    /// again, instead of spinning on the same answer.
    spec_stalled: bool,
    last_row: Option<usize>,
    disconnected: bool,
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
    let book = PadBook::new(ctx.pad.num_slots(), ctx.rows, ctx.n_experts);
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
        spec_stalled: false,
        last_row: None,
        disconnected: false,
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

/// A reader: one pack handle, one job at a time, demand jobs first.
fn reader(
    handle: usize,
    jobs: &ReadQueue<ReadJob>,
    done: &mpsc::Sender<StagerMsg>,
    pack: &ExpertPack,
    warm: &WarmTier,
    pad: &Pad,
) {
    while let Some(job) = jobs.pop() {
        let t = Instant::now();
        // SAFETY: the stager hands slot `job.slot` to exactly one reader and
        // neither publishes nor reuses it until this job reports back.
        let dest = unsafe { pad.slot_mut(job.slot) };
        let (result, source) = match job.source {
            Source::Pack { row, expert } => (
                pack.read_into_with_handle(handle, row, expert, dest),
                ReadSource::Pack,
            ),
            Source::Warm(slot) => {
                dest.copy_from_slice(warm.slot_ref(slot, pad.stride()));
                (Ok(()), ReadSource::Paged)
            }
        };
        let msg = StagerMsg::Landed {
            slot: job.slot,
            result,
            read_ns: t.elapsed().as_nanos() as u64,
            source,
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
            } else if !progressed {
                std::thread::yield_now();
            }
        }
    }

    fn handle(&mut self, m: StagerMsg) -> Result<()> {
        // Any message can change what the speculative queue may take.
        self.spec_stalled = false;
        match m {
            StagerMsg::Routed {
                row,
                slot,
                summary_word,
                ticket,
            } => self.routed.push_back((row, slot, summary_word, ticket)),
            // The pipeline's predictor is confident in a transition: its
            // prediction may evict any slot a speculative read may.
            StagerMsg::Stage { row, experts } => {
                for e in experts {
                    self.queue_spec(row, e, f32::INFINITY);
                }
            }
            StagerMsg::Landed {
                slot,
                result,
                read_ns,
                source,
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
        // The wave has reached this row: what was staged ahead for it is no
        // longer ahead, and a prediction for it not yet read is its demand's.
        let hits = self
            .book
            .settle_row(row)
            .into_iter()
            .filter(|&e| routed[e])
            .count();
        self.spec.retain(|s| s.row != row);
        self.look_ahead(row);
        if let Ok(mut s) = self.ctx.stats.lock() {
            s.staged_ahead_routed += hits;
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

    /// Queue a speculative read of `(row, expert)` unless one is queued; it may
    /// evict only a slot scoring below `floor`.
    fn queue_spec(&mut self, row: usize, expert: usize, floor: f32) {
        if !self.spec.iter().any(|s| s.row == row && s.expert == expert) {
            self.spec.push_back(SpecRead { row, expert, floor });
        }
    }

    /// The rows after `row` the wave reaches next — the next pass's leading
    /// rows past the last one — each with its highest-scored experts queued
    /// for a speculative read: what the recent passes routed there. A queued
    /// expert already readable by the device is dropped when its turn comes
    /// (`serve_spec`). Each may evict only a lower-scored slot, so the
    /// lookahead never trades a copy the pad's own history rates higher.
    /// Rows the pack holds no record for (the permanently resident prefix)
    /// are skipped.
    fn look_ahead(&mut self, row: usize) {
        let rows = self.ctx.rows;
        let pinned = self.ctx.pack.pinned_layers();
        for hop in 1..=LOOKAHEAD_ROWS {
            let target = (row + hop) % rows;
            if target < pinned {
                continue;
            }
            for (e, score) in self
                .book
                .top_scored(target, LOOKAHEAD_MIN_SCORE, LOOKAHEAD_PER_ROW)
            {
                self.queue_spec(target, e, score);
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
            self.issue(slot, d.row, e, Priority::Demand)?;
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
        let source = self.source(row, expert)?;
        let ahead = priority == Priority::Speculative;
        self.book.start_load(slot, row, expert, ahead);
        self.loading.insert((row, expert));
        if ahead {
            self.spec_loads.insert((row, expert), slot);
        }
        self.in_flight += 1;
        if !self.jobs.push(ReadJob { slot, source }, priority) {
            candle::bail!("expert stager: readers gone");
        }
        Ok(())
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
        while self.spec_loads.len() < SPEC_IN_FLIGHT {
            let Some(&SpecRead {
                row,
                expert: e,
                floor,
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
                        self.spec_stalled = true;
                        break;
                    };
                    if !self.evict(&mut r, v, None) {
                        // Its row began an invocation between the choice and
                        // the store: look again once something has moved.
                        drop(r);
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
            if let Ok(mut s) = self.ctx.stats.lock() {
                s.staged_speculative += 1;
            }
            progressed = true;
        }
        Ok(progressed)
    }
}

#[cfg(test)]
mod tests {
    use super::{demand_left_behind, overwritable};

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
