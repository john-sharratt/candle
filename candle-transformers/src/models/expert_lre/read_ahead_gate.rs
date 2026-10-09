//! Whether reading ahead pays for itself.
//!
//! Reading ahead costs device time on every decode invocation, whether or not
//! anything is copied: the look-ahead routers' stacked projections and their
//! votes (`votes`), and bucketize's read-ahead walk, whose serial reads of the
//! ring's mapped words are one link round trip each. Measured on
//! Qwen3.8-Flash-Next (RTX PRO 5000, Strata's single-session workload) that was
//! ~41 µs per MoE invocation — ~17 µs of router launches, ~14 µs of votes and
//! ~10 µs of walk — and, at about three forwards of 48 MoE layers per MTP step,
//! decode 131 → 103 t/s.
//!
//! What it buys back is link time: an expert read ahead is one the workers do
//! not copy while the layer waits. So the pipeline thread keeps the link time
//! per decode invocation — the copies the routing left it (its misses, and the
//! read-ahead claims, which are misses moved earlier, so the figure does not
//! change with the gate's own state) priced at what each costs here: a pinned
//! copy at the slot image over the link's measured rate (`link_rate`), a cold
//! expert at the stager's measured pack read. The forward reads ahead only
//! while that time is about twice the look-ahead's own cost — the break-even at
//! the half of predicted experts the look-ahead gets right.
//!
//! The two cards it was measured on sit far apart. On the RTX PRO 5000 the
//! model's working set is resident: decode copies 0.08 experts an invocation at
//! 4K and ~0.5 just after a 128K prompt, each a ~26 µs pinned copy — ~2–13 µs
//! of link time an invocation, which the look-ahead cost three times over. The
//! RTX 4090 Laptop GPU streams its experts from NVMe, and a cold expert is a
//! pack read of milliseconds.
//!
//! The figure is a moving average over [`RATE_WINDOW`] invocations, about one
//! and a third passes of a 48-layer model, with hysteresis between [`ON_US`] and
//! [`OFF_US`] so a figure near the line does not flip the forward's shape from
//! one pass to the next.

use std::sync::atomic::{AtomicBool, Ordering};

/// Invocations the link time averages over.
pub(crate) const RATE_WINDOW: f64 = 64.0;

/// Link time per decode invocation, µs, at or above which reading ahead
/// switches on: twice its measured ~41 µs cost.
pub(crate) const ON_US: f64 = 80.0;

/// Link time per decode invocation, µs, below which reading ahead switches off.
pub(crate) const OFF_US: f64 = 40.0;

/// One decode invocation's link time, µs: `pinned` copies of `image_bytes`
/// over a link of `link_rate` bytes/s, and `cold` reads of `cold_secs` each.
pub(crate) fn link_us(
    pinned: usize,
    cold: usize,
    image_bytes: usize,
    link_rate: f64,
    cold_secs: f64,
) -> f64 {
    let copy_secs = if link_rate > 0.0 {
        image_bytes as f64 / link_rate
    } else {
        0.0
    };
    1e6 * (pinned as f64 * copy_secs + cold as f64 * cold_secs)
}

/// The link time and the decision it drives, owned by the pipeline thread.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinkTime {
    us: f64,
    on: bool,
}

impl LinkTime {
    /// A cache that can miss starts reading ahead, at the figure that switches
    /// it on, and holds there until its own copies say otherwise; one that
    /// holds every expert never misses, so never reads ahead.
    pub(crate) fn new(all_resident: bool) -> Self {
        if all_resident {
            Self { us: 0.0, on: false }
        } else {
            Self {
                us: ON_US,
                on: true,
            }
        }
    }

    /// Count one decode invocation's link time, µs, and answer whether
    /// reading ahead pays.
    pub(crate) fn observe(&mut self, us: f64) -> bool {
        self.us += (us - self.us) / RATE_WINDOW;
        self.on = if self.on {
            self.us >= OFF_US
        } else {
            self.us >= ON_US
        };
        self.on
    }

    /// Link time per decode invocation, µs, as averaged.
    pub(crate) fn us(&self) -> f64 {
        self.us
    }

    pub(crate) fn on(&self) -> bool {
        self.on
    }
}

/// The pipeline thread's decision, read by the forward thread at each MoE
/// invocation it records.
#[derive(Debug)]
pub(crate) struct ReadAheadGate {
    on: AtomicBool,
}

impl ReadAheadGate {
    pub(crate) fn new(time: &LinkTime) -> Self {
        Self {
            on: AtomicBool::new(time.on()),
        }
    }

    /// Whether the invocation being recorded reads ahead: runs the look-ahead
    /// routers and their votes, and gives bucketize the read-ahead buffers.
    pub(crate) fn pays(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    pub(crate) fn set(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flash-Next's slot image, 1,384,448 bytes, over a 52 GB/s link is a
    /// 26.6 µs copy; a cold expert at a 2.5 ms pack read is 2,500 µs.
    #[test]
    fn link_time_prices_each_copy_where_it_comes_from() {
        let pinned = link_us(1, 0, 1_384_448, 52e9, 2.5e-3);
        assert_eq!((pinned * 10.0).round(), 266.0);
        assert_eq!(link_us(2, 1, 1_384_448, 52e9, 2.5e-3).round(), 2553.0);
        assert_eq!(link_us(0, 0, 1_384_448, 0.0, 0.0), 0.0);
    }

    /// The RTX PRO 5000 at 4K copies about one expert in twelve invocations,
    /// ~2.2 µs an invocation: within two passes of 48 layers the gate is off,
    /// and it stays off.
    #[test]
    fn a_resident_working_set_switches_reading_ahead_off() {
        let mut time = LinkTime::new(false);
        assert!(time.on());
        let decisions: Vec<bool> = (0..96)
            .map(|i| time.observe(if i % 12 == 0 { 26.6 } else { 0.0 }))
            .collect();
        let off_at = decisions.iter().position(|&on| !on);
        assert_eq!(off_at, Some(45));
        assert!(decisions[45..].iter().all(|&on| !on));
    }

    /// After a 128K prompt the same card copies ~0.5 experts an invocation —
    /// ~13 µs, a sixth of the line: an off gate stays off.
    #[test]
    fn a_churned_zone_on_a_fast_link_stays_off() {
        let mut time = LinkTime { us: 0.0, on: false };
        assert!((0..500).all(|i| !time.observe(if i % 2 == 0 { 26.6 } else { 0.0 })));
    }

    /// A streaming card waits on cold experts: one pack read in ten
    /// invocations is 250 µs an invocation, and the gate never switches off.
    #[test]
    fn a_streaming_working_set_keeps_reading_ahead_on() {
        let mut time = LinkTime::new(false);
        assert!((0..500).all(|i| time.observe(if i % 10 == 0 { 2_500.0 } else { 0.0 })));
        assert!(time.us() > ON_US);
    }

    /// Between the lines the decision holds: from off, 60 µs stays off; from
    /// on, it stays on.
    #[test]
    fn a_figure_between_the_lines_keeps_the_last_decision() {
        let mut from_off = LinkTime {
            us: 60.0,
            on: false,
        };
        let mut from_on = LinkTime { us: 60.0, on: true };
        for _ in 0..640 {
            from_off.observe(60.0);
            from_on.observe(60.0);
        }
        assert!(!from_off.on());
        assert!(from_on.on());
    }

    /// A burst of cold reads on an otherwise resident cache switches the gate
    /// on, and it switches off again once the burst has passed.
    #[test]
    fn a_burst_switches_on_and_decays_off() {
        let mut time = LinkTime::new(true);
        assert!(!time.on());
        assert!(!time.observe(0.0));
        // 39.1, 77.5, then 115.4 µs: on at the third.
        let on_at = (0..64).position(|_| time.observe(2_500.0));
        assert_eq!(on_at, Some(2));
        // 115.4 µs decays past 40 at the sixty-eighth idle invocation.
        let off_at = (0..1000).position(|_| !time.observe(0.0));
        assert_eq!(off_at, Some(67));
        assert!(!time.on());
    }

    #[test]
    fn the_gate_publishes_the_decision() {
        let time = LinkTime::new(false);
        let gate = ReadAheadGate::new(&time);
        assert!(gate.pays());
        gate.set(false);
        assert!(!gate.pays());
        assert!(!ReadAheadGate::new(&LinkTime::new(true)).pays());
    }
}
