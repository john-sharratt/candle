//! The elastic warm tier's controller thread (`docs/elastic_warm_tier.md`
//! §4–§6): once a tick it reads this process's hard faults and the machine's
//! free RAM, asks the policy, and carries the answer out — a grow allocates a
//! segment, adds its range and hands its slots to the stager; a shrink asks the
//! stager to drain the last segments and frees them once it answers.
//!
//! Off the critical path throughout: the readings are two system calls, a grow
//! is a `cuMemAllocHost` (no stream, no capture), and a shrink's frees wait for
//! the stager, which evicts only slots no invocation can still read. The one
//! device-wide call is the synchronise that settles the reclaim clock while
//! inference is idle — taken only when no invocation began over a whole tick,
//! and only while no segment records.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle::cuda_backend::graph::try_without_recording;
use candle::cuda_backend::CudaDevice;
use candle::vram::process_paging::own_paging;
use candle::vram::{available_physical_ram, gpu_addressable_room};

use super::super::page_pressure::press;
use super::super::reclaim::ReclaimClock;
use super::super::stager::StagerMsg;
use super::super::types::PipelineStats;
use super::super::warm_tier::{driver_lock_margin_free, ADDRESSABLE_MARGIN};
use super::policy::{Decision, Limits, Policy, ShrinkReason, Signals};
use super::region::ResidentRegion;

/// The controller's period.
const TICK: Duration = Duration::from_secs(1);

/// Ticks between telemetry lines when nothing changes.
const REPORT_TICKS: u64 = 60;

const GIB: f64 = (1u64 << 30) as f64;

/// The most segments the tier may hold: one per `per` evictable experts, and
/// no more than the host partition's expert budget left after the pageable
/// part (`docs/elastic_warm_tier.md` §4.4).
pub(crate) fn max_segments(
    evictable: usize,
    per: usize,
    segment_bytes: u64,
    expert_budget: u64,
    paged_bytes: u64,
) -> usize {
    let by_experts = evictable / per.max(1);
    let by_budget = expert_budget.saturating_sub(paged_bytes) / segment_bytes.max(1);
    by_experts.min(by_budget as usize)
}

/// Whether the device can still address one more segment of `segment_bytes`
/// and keep [`ADDRESSABLE_MARGIN`] of `room` unspent. `None` is a driver model
/// with no such budget: the driver's own refusal is the bound.
pub(crate) fn room_for_segment(room: Option<u64>, segment_bytes: u64) -> bool {
    room.is_none_or(|room| room.saturating_sub(ADDRESSABLE_MARGIN) >= segment_bytes)
}

/// What the controller is given.
pub(crate) struct ControllerCtx {
    pub(crate) region: Arc<ResidentRegion>,
    pub(crate) stager: mpsc::Sender<StagerMsg>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) device: CudaDevice,
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    /// [`max_segments`].
    pub(crate) max_segments: usize,
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

/// A shrink the stager is draining.
struct Draining {
    segments: usize,
    reason: ShrinkReason,
    done: mpsc::Receiver<usize>,
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
            Change::Shrink(ShrinkReason::Pressure) => "shrink(pressure)",
        }
    }
}

struct Controller {
    ctx: ControllerCtx,
    policy: Policy,
    draining: Option<Draining>,
    /// Segments a shrink decided while another drained, asked for next.
    owed: Option<(usize, ShrinkReason)>,
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
            owed: None,
            pressed: false,
            available: 0,
            ticks: 0,
        }
    }

    fn run(mut self, stop: &AtomicBool) {
        // `cuMemAllocHost` and the synchronise need the context current here.
        if let Err(e) = self.ctx.device.cuda_context().bind_to_thread() {
            tracing::error!(
                target: "candle_transformers::expert_lre",
                "elastic warm tier: the context would not bind, the tier stays at its size: {e}"
            );
            return;
        }
        let mut next = Instant::now() + TICK;
        while !stop.load(Ordering::Acquire) {
            let wait = next.saturating_duration_since(Instant::now());
            match &self.draining {
                Some(d) => match d.done.recv_timeout(wait) {
                    Ok(gone) => {
                        self.finish_shrink(gone);
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
        let draining = self.draining.as_ref().map_or(0, |d| d.segments);
        let limits = Limits {
            held: self.ctx.region.segments().saturating_sub(draining),
            max: self.ctx.max_segments,
            segment_bytes: self.ctx.region.segment_bytes(),
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
            Decision::Grow { press } if self.draining.is_none() => self.grow(press, limits.held),
            // A grow while a shrink drains waits for the next quiet tick.
            Decision::Grow { .. } => Change::None,
            Decision::Shrink { segments, reason } => {
                if self.draining.is_some() {
                    let owed = self.owed.map_or(0, |o| o.0);
                    self.owed = Some((owed + segments, reason));
                } else if !self.begin_shrink(segments, reason) {
                    return false;
                }
                Change::Shrink(reason)
            }
        };
        if !matches!(change, Change::None) || self.ticks.is_multiple_of(REPORT_TICKS) {
            self.report(change);
        }
        true
    }

    /// Take one segment. `press` first, when free RAM alone cannot pay for it.
    fn grow(&mut self, press_first: bool, held: usize) -> Change {
        let seg = self.ctx.region.segment_bytes();
        if press_first {
            self.pressed = press(seg as usize);
        }
        let room = gpu_addressable_room(&self.ctx.device).ok().flatten();
        let grew = room_for_segment(room, seg) && self.ctx.region.grow();
        // A segment the driver granted but that leaves it no lock margin is
        // given straight back, as the startup sizing does.
        if grew && !driver_lock_margin_free() {
            self.ctx.region.pop();
        } else if grew {
            let slots = self.ctx.region.per_segment();
            if self.ctx.stager.send(StagerMsg::Grown { slots }).is_err() {
                return Change::None;
            }
            self.gauges(|s| s.warm_grown += 1);
            return Change::Grow;
        }
        self.policy.refused(held);
        Change::Refused
    }

    /// Ask the stager to drain the last `segments`. `false` once it is gone.
    fn begin_shrink(&mut self, segments: usize, reason: ShrinkReason) -> bool {
        let segments = segments.min(self.ctx.region.segments());
        if segments == 0 {
            return true;
        }
        let (tx, rx) = mpsc::channel();
        let slots = segments * self.ctx.region.per_segment();
        if self
            .ctx
            .stager
            .send(StagerMsg::Shrink { slots, done: tx })
            .is_err()
        {
            return false;
        }
        self.draining = Some(Draining {
            segments,
            reason,
            done: rx,
            latest_seen: self.ctx.clock.latest_started(),
            started: Instant::now(),
        });
        true
    }

    /// A drain still waiting a tick on: if no invocation began over it,
    /// inference is idle and nothing later will prove the last row's slots
    /// free — wait out everything begun, settle the clock, and have the stager
    /// look again. `false` once the stager is gone.
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

    /// The stager has drained `gone` slots: free their segments, then ask for
    /// any shrink decided meanwhile.
    fn finish_shrink(&mut self, gone: usize) {
        let Some(d) = self.draining.take() else {
            return;
        };
        let segments = gone / self.ctx.region.per_segment();
        for _ in 0..segments {
            self.ctx.region.pop();
        }
        self.gauges(|s| s.warm_shrunk += segments);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            segments,
            reason = d.reason.name(),
            drain_ms = d.started.elapsed().as_millis() as u64,
            "warm tier: elastic segments freed"
        );
        if let Some((owed, reason)) = self.owed.take() {
            // A stager gone with the cache leaves nothing to drain; the next
            // tick finds it so.
            let _ = self.begin_shrink(owed, reason);
        }
    }

    fn gauges(&self, f: impl FnOnce(&mut PipelineStats)) {
        if let Ok(mut s) = self.ctx.stats.lock() {
            f(&mut s);
            s.warm_segments = self.ctx.region.segments();
            s.warm_pinned_bytes = self.ctx.region.held_bytes();
        }
    }

    fn report(&mut self, change: Change) {
        self.gauges(|_| {});
        let (fast, slow) = self.policy.faults();
        let ceiling = self.policy.ceiling().map_or(-1, |c| c.segments as i64);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            segments = self.ctx.region.segments(),
            gib = self.ctx.region.held_bytes() as f64 / GIB,
            change = change.name(),
            faults_fast = fast,
            faults_slow = slow,
            available_gib = self.available as f64 / GIB,
            ceiling,
            backoff_s = self.policy.backoff(),
            pressed = self.pressed,
            draining = self.draining.as_ref().map_or(0, |d| d.segments),
            "warm tier: elastic"
        );
        self.pressed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEG: u64 = 1 << 30;

    /// The tier's bound is the lesser of the evictable experts' segments and
    /// the budget left after the pageable part.
    #[test]
    fn the_bound_is_the_lesser_of_the_experts_and_the_budget() {
        assert_eq!(max_segments(24_064, 537, SEG, 100 * SEG, 0), 44);
        assert_eq!(max_segments(24_064, 537, SEG, 20 * SEG, 4 * SEG), 16);
        assert_eq!(max_segments(24_064, 537, SEG, 3 * SEG, 4 * SEG), 0);
        assert_eq!(max_segments(100, 100, SEG, 100 * SEG, 0), 1);
    }

    /// A segment fits while the addressable room keeps its margin past it;
    /// no budget is no bound.
    #[test]
    fn a_segment_needs_its_bytes_past_the_addressable_margin() {
        assert!(room_for_segment(None, SEG));
        assert!(room_for_segment(Some(ADDRESSABLE_MARGIN + SEG), SEG));
        assert!(!room_for_segment(Some(ADDRESSABLE_MARGIN + SEG - 1), SEG));
        assert!(!room_for_segment(Some(0), SEG));
    }
}
