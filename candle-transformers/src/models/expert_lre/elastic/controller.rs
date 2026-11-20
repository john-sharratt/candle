//! The elastic warm tier's controller thread (`docs/elastic_warm_tier.md`
//! §4–§6): once a tick it reads this process's hard faults and the machine's
//! free RAM, asks the policy, and carries the answer out on the pageable part
//! — a grow allocates a chunk, fills it from the pack and hands its experts to
//! the stager; a shrink asks the stager to drop the last chunk's experts and
//! frees it once the stager says nothing can reach it.
//!
//! The pinned part is never touched. Off the critical path throughout: the
//! readings are two system calls, a grow is a heap allocation and positioned
//! pack reads, a free is a heap free. The one device-wide call is the
//! synchronise that settles the reclaim clock while inference is idle — taken
//! only when no invocation began over a whole tick, and only while no segment
//! records.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle::cuda_backend::graph::try_without_recording;
use candle::cuda_backend::CudaDevice;
use candle::vram::available_physical_ram;
use candle::vram::process_paging::own_paging;

use super::super::pack::ExpertPack;
use super::super::page_pressure::press;
use super::super::reclaim::ReclaimClock;
use super::super::stager::StagerMsg;
use super::super::types::PipelineStats;
use super::super::warm_tier::WarmTier;
use super::policy::{Decision, Limits, Policy, ShrinkReason, Signals, HEADROOM};

/// The controller's period.
const TICK: Duration = Duration::from_secs(1);

/// Ticks between telemetry lines when nothing changes.
const REPORT_TICKS: u64 = 60;

const GIB: f64 = (1u64 << 30) as f64;

/// The most chunks the pageable part may hold: one per `per` experts of the
/// draw past the pinned part, and no more than the host partition's expert
/// budget left after the pinned part (`docs/elastic_warm_tier.md` §4.4).
pub(crate) fn max_chunks(
    order: usize,
    pinned_slots: usize,
    per: usize,
    chunk_bytes: u64,
    expert_budget: u64,
    pinned_bytes: u64,
) -> usize {
    let by_experts = order.saturating_sub(pinned_slots) / per.max(1);
    let by_budget = expert_budget.saturating_sub(pinned_bytes) / chunk_bytes.max(1);
    by_experts.min(by_budget as usize)
}

/// Whether a press left the room a chunk needs: the headroom and the chunk
/// both free.
pub(crate) fn press_made_room(available: u64, chunk_bytes: u64) -> bool {
    available >= HEADROOM + chunk_bytes
}

/// The experts pageable chunk `chunk` holds, and its first warm slot: the
/// draw `order` past the pinned part, `per` at a time.
pub(crate) fn chunk_members(
    order: &[(usize, usize)],
    pinned_slots: usize,
    per: usize,
    chunk: usize,
) -> (usize, &[(usize, usize)]) {
    let first = pinned_slots + chunk * per;
    let end = (first + per).min(order.len());
    (first, &order[first.min(end)..end])
}

/// What the controller is given.
pub(crate) struct ControllerCtx {
    pub(crate) warm: Arc<WarmTier>,
    pub(crate) pack: Arc<ExpertPack>,
    /// Every evictable expert in draw order: warm slot `i` holds entry `i`.
    pub(crate) order: Vec<(usize, usize)>,
    pub(crate) stager: mpsc::Sender<StagerMsg>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) device: CudaDevice,
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    /// [`max_chunks`].
    pub(crate) max_chunks: usize,
}

/// The running controller: stopped and joined when dropped, before the stager
/// it talks to is.
pub(crate) struct Elastic {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Elastic {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Start the controller.
pub(crate) fn spawn_controller(ctx: ControllerCtx) -> candle::Result<Elastic> {
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let stop = stop.clone();
        std::thread::Builder::new()
            .name("warm-tier-elastic".into())
            .spawn(move || Controller::new(ctx).run(&stop))
            .map_err(|e| candle::Error::Msg(format!("elastic warm tier: spawn: {e}")))?
    };
    Ok(Elastic {
        stop,
        thread: Some(thread),
    })
}

/// A shrink in progress: the last chunk draining, and how many more after it.
struct Draining {
    remaining: usize,
    reason: ShrinkReason,
    done: mpsc::Receiver<()>,
    /// The latest invocation begun when the controller last looked: unchanged
    /// over a tick, inference is idle.
    latest_seen: u64,
    started: Instant,
}

/// What the last change was, for the telemetry line.
#[derive(Clone, Copy)]
enum Change {
    None,
    Grow,
    Refused,
    Shrink(ShrinkReason),
}

impl Change {
    fn name(self) -> &'static str {
        match self {
            Change::None => "none",
            Change::Grow => "grow",
            Change::Refused => "refused",
            Change::Shrink(ShrinkReason::Severe) => "shrink(severe)",
            Change::Shrink(ShrinkReason::Build) => "shrink(build)",
            Change::Shrink(ShrinkReason::Short) => "shrink(short)",
            Change::Shrink(ShrinkReason::Pressure) => "shrink(pressure)",
        }
    }
}

struct Controller {
    ctx: ControllerCtx,
    policy: Policy,
    draining: Option<Draining>,
    pressed: bool,
    available: u64,
    ticks: u64,
}

impl Controller {
    fn new(ctx: ControllerCtx) -> Self {
        Self {
            ctx,
            policy: Policy::new(),
            draining: None,
            pressed: false,
            available: 0,
            ticks: 0,
        }
    }

    fn run(mut self, stop: &AtomicBool) {
        let mut next = Instant::now() + TICK;
        while !stop.load(Ordering::Acquire) {
            let wait = next.saturating_duration_since(Instant::now());
            match &self.draining {
                Some(d) => match d.done.recv_timeout(wait) {
                    Ok(()) => {
                        if !self.chunk_drained() {
                            return;
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                },
                None => std::thread::sleep(wait),
            }
            if Instant::now() < next {
                continue;
            }
            // From now, not from the missed deadline: a machine back from sleep
            // ticks once, not once per second it slept.
            next = Instant::now() + TICK;
            if !self.tick() {
                return;
            }
        }
    }

    fn paged_chunks(&self) -> usize {
        self.ctx.warm.paged().chunks()
    }

    /// One tick. `false` once the stager is gone.
    fn tick(&mut self) -> bool {
        self.ticks += 1;
        if self.draining.is_some() && !self.nudge() {
            return false;
        }
        let (Some(paging), Some(available)) = (own_paging(), available_physical_ram()) else {
            return true;
        };
        self.available = available;
        let draining = self.draining.as_ref().map_or(0, |d| d.remaining + 1);
        let limits = Limits {
            held: self.paged_chunks().saturating_sub(draining),
            max: self.ctx.max_chunks,
            segment_bytes: self.ctx.warm.paged().chunk_bytes(),
        };
        let decision = self.policy.tick(
            Signals {
                hard_faults: paging.hard_faults,
                available,
            },
            limits,
        );
        let change = match decision {
            Decision::Hold => Change::None,
            Decision::Grow { press } if self.draining.is_none() => {
                match self.grow(press, limits.held) {
                    Some(c) => c,
                    None => return false,
                }
            }
            // A grow while a shrink drains waits for the next quiet tick.
            Decision::Grow { .. } => Change::None,
            Decision::Shrink { segments, reason } => {
                match &mut self.draining {
                    Some(d) => d.remaining += segments,
                    None => {
                        if !self.begin_shrink(segments - 1, reason) {
                            return false;
                        }
                    }
                }
                Change::Shrink(reason)
            }
        };
        if !matches!(change, Change::None) || self.ticks.is_multiple_of(REPORT_TICKS) {
            self.report(change);
        }
        true
    }

    /// Take one chunk: allocate it, fill it from the pack, then hand its
    /// experts to the stager, which publishes them. `press` first, when free
    /// RAM alone cannot pay for it. `None` once the stager is gone.
    fn grow(&mut self, press_first: bool, held: usize) -> Option<Change> {
        let paged = self.ctx.warm.paged();
        if press_first {
            self.pressed = press(paged.chunk_bytes() as usize);
            // The press is a probe: the RAM was idle pages only if it is free
            // now. If not, something is using it — the grow would only take
            // the machine's headroom, and is refused for a back-off.
            let after = available_physical_ram().unwrap_or(0);
            if !press_made_room(after, paged.chunk_bytes()) {
                self.policy.refused(held);
                return Some(Change::Refused);
            }
        }
        let chunk = paged.chunks();
        let pinned = self.ctx.warm.pinned_slots();
        let (first, experts) = chunk_members(&self.ctx.order, pinned, paged.per_chunk(), chunk);
        if experts.len() < paged.per_chunk() || !paged.grow() {
            self.policy.refused(held);
            return Some(Change::Refused);
        }
        let started = Instant::now();
        for (i, &(row, expert)) in experts.iter().enumerate() {
            // SAFETY: the chunk was appended above and nothing names its slots
            // until the stager publishes them; this thread is the only writer.
            let dest = unsafe { paged.slot_mut(first - pinned + i) };
            if let Err(e) = self.ctx.pack.read_into(row, expert, dest) {
                tracing::warn!(
                    target: "candle_transformers::expert_lre",
                    "elastic warm tier: a grow's pack read failed, the chunk is given back: {e}"
                );
                paged.pop();
                self.policy.refused(held);
                return Some(Change::Refused);
            }
        }
        let experts = experts.to_vec();
        if self
            .ctx
            .stager
            .send(StagerMsg::PagedGrown { first, experts })
            .is_err()
        {
            return None;
        }
        tracing::info!(
            target: "candle_transformers::expert_lre",
            chunk,
            fill_ms = started.elapsed().as_millis() as u64,
            "warm tier: elastic chunk filled"
        );
        self.gauges(|s| s.warm_grown += 1);
        Some(Change::Grow)
    }

    /// Ask the stager to drain the last chunk, `then` more after it. `false`
    /// once it is gone.
    fn begin_shrink(&mut self, then: usize, reason: ShrinkReason) -> bool {
        let chunks = self.paged_chunks();
        if chunks == 0 {
            self.draining = None;
            return true;
        }
        let first = self.ctx.warm.pinned_slots() + (chunks - 1) * self.ctx.warm.paged().per_chunk();
        let (tx, rx) = mpsc::channel();
        if self
            .ctx
            .stager
            .send(StagerMsg::PagedShrink { first, done: tx })
            .is_err()
        {
            return false;
        }
        let started = self
            .draining
            .as_ref()
            .map_or_else(Instant::now, |d| d.started);
        self.draining = Some(Draining {
            remaining: then,
            reason,
            done: rx,
            latest_seen: self.ctx.clock.latest_started(),
            started,
        });
        true
    }

    /// A drain still waiting a tick on: if no invocation began over it,
    /// inference is idle and nothing later will prove the last invocation's
    /// readers done — wait out everything begun, settle the clock, and have
    /// the stager look again. `false` once the stager is gone.
    fn nudge(&mut self) -> bool {
        let Some(d) = self.draining.as_mut() else {
            return true;
        };
        let latest = self.ctx.clock.latest_started();
        if latest == d.latest_seen {
            let ctx = self.ctx.device.cuda_context();
            let synced = try_without_recording(|q| q.synchronize_context(ctx).is_ok());
            if synced == Some(true) {
                self.ctx.clock.settled(latest);
            }
        }
        d.latest_seen = latest;
        self.ctx.stager.send(StagerMsg::Drain).is_ok()
    }

    /// The stager has drained the last chunk: free it, then drain the next
    /// one owed. `false` once the stager is gone.
    fn chunk_drained(&mut self) -> bool {
        let Some(d) = self.draining.take() else {
            return true;
        };
        self.ctx.warm.paged().pop();
        self.gauges(|s| s.warm_shrunk += 1);
        if d.remaining > 0 && self.paged_chunks() > 0 {
            self.draining = Some(d);
            let (then, reason) = self
                .draining
                .as_ref()
                .map_or((0, ShrinkReason::Pressure), |d| (d.remaining - 1, d.reason));
            return self.begin_shrink(then, reason);
        }
        tracing::info!(
            target: "candle_transformers::expert_lre",
            chunks = self.paged_chunks(),
            reason = d.reason.name(),
            drain_ms = d.started.elapsed().as_millis() as u64,
            "warm tier: elastic chunks freed"
        );
        true
    }

    fn gauges(&self, f: impl FnOnce(&mut PipelineStats)) {
        if let Ok(mut s) = self.ctx.stats.lock() {
            f(&mut s);
            s.warm_paged_chunks = self.paged_chunks();
            s.warm_paged_bytes = self.ctx.warm.paged().held_bytes();
            s.warm_paged_slots = self.ctx.warm.paged_slots();
            s.warm_slots = self.ctx.warm.num_slots();
        }
    }

    fn report(&mut self, change: Change) {
        self.gauges(|_| {});
        let (fast, slow) = self.policy.faults();
        let ceiling = self.policy.ceiling().map_or(-1, |c| c.segments as i64);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            chunks = self.paged_chunks(),
            paged_gib = self.ctx.warm.paged().held_bytes() as f64 / GIB,
            change = change.name(),
            faults_fast = fast,
            faults_slow = slow,
            available_gib = self.available as f64 / GIB,
            ceiling,
            backoff_s = self.policy.backoff(),
            pressed = self.pressed,
            draining = self.draining.as_ref().map_or(0, |d| d.remaining + 1),
            "warm tier: elastic"
        );
        self.pressed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: u64 = 1 << 30;

    /// The bound is the lesser of the draw's chunks past the pinned part and
    /// the budget the pinned part leaves.
    #[test]
    fn the_bound_is_the_lesser_of_the_draw_and_the_budget() {
        assert_eq!(
            max_chunks(24_064, 4_296, 537, CHUNK, 100 * CHUNK, 8 * CHUNK),
            36
        );
        assert_eq!(
            max_chunks(24_064, 4_296, 537, CHUNK, 30 * CHUNK, 8 * CHUNK),
            22
        );
        assert_eq!(
            max_chunks(24_064, 4_296, 537, CHUNK, 8 * CHUNK, 9 * CHUNK),
            0
        );
        assert_eq!(
            max_chunks(100, 100, 537, CHUNK, 100 * CHUNK, 0),
            0,
            "all pinned"
        );
    }

    /// A press made room only if the headroom and a whole chunk are free
    /// after it.
    #[test]
    fn a_press_made_room_only_with_the_headroom_and_a_chunk_free() {
        assert!(press_made_room(HEADROOM + CHUNK, CHUNK));
        assert!(!press_made_room(HEADROOM + CHUNK - 1, CHUNK));
        assert!(!press_made_room(HEADROOM, CHUNK));
    }

    /// Chunk `c` holds the draw's next `per` experts past the pinned part, at
    /// the warm slots that match their draw index; past the draw there is
    /// none.
    #[test]
    fn a_chunk_holds_the_draws_next_experts() {
        let order: Vec<(usize, usize)> = (0..10).map(|i| (i / 4, i % 4)).collect();
        assert_eq!(chunk_members(&order, 3, 2, 0), (3, &order[3..5]));
        assert_eq!(chunk_members(&order, 3, 2, 2), (7, &order[7..9]));
        assert_eq!(
            chunk_members(&order, 3, 2, 3),
            (9, &order[9..10]),
            "short of a chunk"
        );
        assert_eq!(chunk_members(&order, 3, 2, 4).1.len(), 0);
    }
}
