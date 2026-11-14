//! The gap quantize in the batched gate, as the scheduler runs it
//! (`docs/kv_quantize_in_the_gap.md` §3, §7): a catch-up and a quantize in
//! the gap after every forward, inside the timed windows, and what they did
//! and cost, summed over a config.
//!
//! **What the gap costs, by window.** Each call's host time is charged to the
//! clock running when it was made, and each gap's device time — from its first
//! upload to its completion word, on the stream the forwards share, so the
//! forward behind it waits it out — to the clock running when it was launched.
//! The settle after the decode runs outside both clocks and is charged to
//! neither. A window's rate with its gap time taken out is the forward's own
//! rate, comparable with a run that sealed outside the clocks. Host and device
//! time can overlap — the host builds the next forward while a gap's kernels
//! run — so the sum is an upper bound on what the gap added.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use candle::{Device, Result};
use candle_nn::kv_cache::{GapCatchUp, GapKind, GapQuantize, QuantizeHost};

use crate::models::batched_inference::BatchedInferenceSession;

/// Gaps a settle runs at most before giving up on the cycle going quiet.
const SETTLE_GAPS: usize = 64;

/// The clock a gap's cost is charged to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Prefill,
    Decode,
    /// Outside every clock: the settle before the figures are read.
    Outside,
}

/// One window's gap cost.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowCost {
    /// Host time in catch-ups.
    pub catch_up: Duration,
    /// Host time in quantizes.
    pub quantize: Duration,
    /// The quantizes' host time by phase.
    pub phases: QuantizeHost,
    /// Gaps' time on the stream.
    pub device: Duration,
}

impl WindowCost {
    /// Everything the gap added to the window, at most.
    pub fn total(&self) -> Duration {
        self.catch_up + self.quantize + self.device
    }
}

/// What the gaps of one config did and cost.
#[derive(Debug)]
pub struct GapTally {
    gaps: usize,
    jobs: usize,
    retried: usize,
    installed: usize,
    deferred: usize,
    stale: usize,
    moved: usize,
    outstanding: usize,
    short_slots: usize,
    refused: Option<String>,
    errored: bool,
    window: Window,
    /// The window the outstanding gap was launched in: its device time goes
    /// there when it is caught up.
    launched_in: Option<Window>,
    prefill: WindowCost,
    decode: WindowCost,
}

impl Default for GapTally {
    fn default() -> Self {
        Self {
            gaps: 0,
            jobs: 0,
            retried: 0,
            installed: 0,
            deferred: 0,
            stale: 0,
            moved: 0,
            outstanding: 0,
            short_slots: 0,
            refused: None,
            errored: false,
            window: Window::Prefill,
            launched_in: None,
            prefill: WindowCost::default(),
            decode: WindowCost::default(),
        }
    }
}

impl GapTally {
    /// Charge what follows to `window`.
    pub fn set_window(&mut self, window: Window) {
        self.window = window;
    }

    fn cost(&mut self, window: Window) -> Option<&mut WindowCost> {
        match window {
            Window::Prefill => Some(&mut self.prefill),
            Window::Decode => Some(&mut self.decode),
            Window::Outside => None,
        }
    }

    /// A window's gap cost.
    pub fn window_cost(&self, window: Window) -> WindowCost {
        match window {
            Window::Prefill => self.prefill,
            Window::Decode => self.decode,
            Window::Outside => WindowCost::default(),
        }
    }

    /// The head of a gap: put the outstanding gap's blocks in place.
    pub fn catch_up(&mut self, session: &mut BatchedInferenceSession) -> Result<GapCatchUp> {
        let t = Instant::now();
        let c = session.gap_catch_up()?;
        let host = t.elapsed();
        if let Some(cost) = self.cost(self.window) {
            cost.catch_up += host;
        }
        if c.caught_up {
            self.installed += c.installed;
            self.deferred += c.deferred;
            self.stale += c.stale;
            self.moved += c.moved;
            self.errored |= c.errored;
            if let Some(w) = self.launched_in.take() {
                if let Some(cost) = self.cost(w) {
                    cost.device += Duration::from_secs_f64(c.device_ms / 1e3);
                }
            }
        }
        Ok(c)
    }

    /// The tail of a gap: quantize what the forwards completed.
    pub fn quantize(
        &mut self,
        session: &mut BatchedInferenceSession,
        kind: GapKind,
    ) -> Result<GapQuantize> {
        let t = Instant::now();
        let q = session.gap_quantize(kind)?;
        let host = t.elapsed();
        if let Some(cost) = self.cost(self.window) {
            cost.quantize += host;
            cost.phases.setup += q.host.setup;
            cost.phases.table += q.host.table;
            cost.phases.lease += q.host.lease;
            cost.phases.launch += q.host.launch;
        }
        if q.outstanding {
            self.outstanding += 1;
        } else if q.jobs > 0 {
            self.launched_in = Some(self.window);
            self.gaps += 1;
            self.jobs += q.jobs;
            self.retried += q.retried;
            self.short_slots += q.short.iter().sum::<usize>();
            if self.refused.is_none() {
                self.refused = q
                    .refusals
                    .first()
                    .map(|r| format!("class {} short {}: {}", r.class, r.short, r.why));
            }
        }
        Ok(q)
    }

    /// Run gaps until the cycle is quiet — none outstanding and nothing left
    /// to take — waiting for each on `device`. Outside every clock: what it
    /// leaves is what the compression figures read. A gap launched inside the
    /// decode clock and caught up here keeps its device time in the decode.
    pub fn settle(&mut self, session: &mut BatchedInferenceSession, device: &Device) -> Result<()> {
        self.set_window(Window::Outside);
        for _ in 0..SETTLE_GAPS {
            device.synchronize()?;
            self.catch_up(session)?;
            let q = self.quantize(session, GapKind::Decode)?;
            if q.jobs == 0 && !session.gap_outstanding() {
                return Ok(());
            }
        }
        candle::bail!("the gap cycle did not go quiet in {SETTLE_GAPS} gaps")
    }

    /// The config's gap line.
    pub fn line(&self, label: &str) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mut s = format!(
            "[{label}] gap: {} gaps, {} jobs ({} retried) | installed {}, deferred {}, stale {}, \
             moved {} | {} skipped outstanding | lease short {} slots",
            self.gaps,
            self.jobs,
            self.retried,
            self.installed,
            self.deferred,
            self.stale,
            self.moved,
            self.outstanding,
            self.short_slots
        );
        for (name, c) in [("prefill", self.prefill), ("decode", self.decode)] {
            let _ = write!(
                s,
                " | {name} {:.1} ms (catch-up {:.1}, quantize {:.1} [setup {:.1}, table {:.1}, \
                 lease {:.1}, launch {:.1}], device {:.1})",
                ms(c.total()),
                ms(c.catch_up),
                ms(c.quantize),
                ms(c.phases.setup),
                ms(c.phases.table),
                ms(c.phases.lease),
                ms(c.phases.launch),
                ms(c.device)
            );
        }
        if let Some(r) = &self.refused {
            let _ = write!(s, " (first refusal: {r})");
        }
        if self.errored {
            s.push_str(" | the claim met a code its tables do not cover");
        }
        s
    }
}

/// `tokens` over `elapsed` with `gap` taken out of it: the window's rate
/// without the gap's work.
pub fn rate_ex_gap(tokens: usize, elapsed: Duration, gap: Duration) -> f64 {
    let own = elapsed.saturating_sub(gap).as_secs_f64();
    if tokens == 0 || own <= 0.0 {
        0.0
    } else {
        tokens as f64 / own
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_ex_gap_takes_the_gap_out_of_the_window() {
        let r = rate_ex_gap(1000, Duration::from_secs(2), Duration::from_secs(1));
        assert_eq!(r, 1000.0);
        assert_eq!(
            rate_ex_gap(1000, Duration::from_secs(2), Duration::ZERO),
            500.0
        );
        assert_eq!(rate_ex_gap(0, Duration::from_secs(2), Duration::ZERO), 0.0);
        // A gap as long as the window leaves no measurable rate.
        assert_eq!(
            rate_ex_gap(10, Duration::from_secs(1), Duration::from_secs(1)),
            0.0
        );
    }

    #[test]
    fn device_time_goes_to_the_window_the_gap_was_launched_in() {
        let mut t = GapTally::default();
        t.set_window(Window::Prefill);
        t.launched_in = Some(t.window);
        t.set_window(Window::Decode);
        // The catch-up's bookkeeping, without a session.
        if let Some(w) = t.launched_in.take() {
            if let Some(cost) = t.cost(w) {
                cost.device += Duration::from_millis(5);
            }
        }
        assert_eq!(
            t.window_cost(Window::Prefill).device,
            Duration::from_millis(5)
        );
        assert_eq!(t.window_cost(Window::Decode).device, Duration::ZERO);
        assert_eq!(t.window_cost(Window::Outside).total(), Duration::ZERO);
    }
}
