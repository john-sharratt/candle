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
//! **Decided once a pass, from passes.** Each pass's link time per decode
//! invocation is folded into a moving average over [`PASS_WINDOW`] passes, with
//! hysteresis between [`ON_US`] and [`OFF_US`], so a figure near the line does
//! not flip the forward's shape from one pass to the next. Averaged over 64
//! invocations — a pass and a third — and decided at each, it did: the link
//! time arrives in bursts, a dip's misses lifting it past the on line within a
//! few invocations and a run of resident layers decaying it under the off line
//! within one forward. On Qwen3.8-Flash-Next (BlackWorld, 2026-10-27, build
//! 3897) the gate changed 2,145 times in an hour, several times a second, and
//! read ahead about half the time — through the recoveries after a prefill's
//! concession too, where it is what refills the zone.
//!
//! **Fillable ground holds it on.** The break-even above prices a read-ahead
//! claim against the look-ahead's cost on the assumption that it might evict.
//! A claim into an empty slot evicts nothing, so where the zone holds empty
//! slots for a layer's window and as many experts decode routed recently that
//! it does not hold — the ground a concession hands back, grown into — reading
//! ahead refills them at the link's rate ahead of the misses that would bring
//! them back one by one. The pipeline thread decides that (`fillable`); it
//! clears once the zone has filled.

use std::sync::atomic::{AtomicBool, Ordering};

/// Passes the link time averages over.
pub(crate) const PASS_WINDOW: f64 = 4.0;

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
    /// The pass being counted: its decode invocations' link time, µs, summed,
    /// and how many there were.
    pass_us: f64,
    pass_invocations: usize,
}

impl LinkTime {
    /// A cache that can miss starts reading ahead, at the figure that switches
    /// it on, and holds there until its own copies say otherwise; one that
    /// holds every expert never misses, so never reads ahead.
    pub(crate) fn new(all_resident: bool) -> Self {
        let (us, on) = if all_resident {
            (0.0, false)
        } else {
            (ON_US, true)
        };
        Self {
            us,
            on,
            pass_us: 0.0,
            pass_invocations: 0,
        }
    }

    /// Count one decode invocation's link time, µs, into the pass.
    pub(crate) fn observe(&mut self, us: f64) {
        self.pass_us += us;
        self.pass_invocations += 1;
    }

    /// Close the pass: fold its link time per decode invocation into the
    /// average, and answer whether the next pass reads ahead — while the
    /// average repays the look-ahead, or while there is `fillable` ground.
    /// A pass with no decode invocation says nothing about the link, and the
    /// average stands.
    pub(crate) fn end_pass(&mut self, fillable: bool) -> bool {
        if self.pass_invocations > 0 {
            let mean = self.pass_us / self.pass_invocations as f64;
            self.us += (mean - self.us) / PASS_WINDOW;
        }
        self.pass_us = 0.0;
        self.pass_invocations = 0;
        let pays = if self.on {
            self.us >= OFF_US
        } else {
            self.us >= ON_US
        };
        self.on = pays || fillable;
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

    /// Decode invocations a pass of a 48-layer model makes.
    const LAYERS: usize = 48;

    /// A gate standing at `us` with decision `on`.
    fn at(us: f64, on: bool) -> LinkTime {
        LinkTime {
            us,
            on,
            pass_us: 0.0,
            pass_invocations: 0,
        }
    }

    /// One pass of `LAYERS` decode invocations, the `i`th costing `us(i)`,
    /// closed with `fillable` ground or none: the decision for the next pass.
    fn pass(time: &mut LinkTime, us: impl Fn(usize) -> f64, fillable: bool) -> bool {
        for i in 0..LAYERS {
            time.observe(us(i));
        }
        time.end_pass(fillable)
    }

    /// The RTX PRO 5000 at 4K copies about one expert in twelve invocations,
    /// ~2.2 µs an invocation: the average falls 60.5, 46.0, 35.0 over three
    /// passes, the gate is off at the third, and it stays off.
    #[test]
    fn a_resident_working_set_switches_reading_ahead_off() {
        let mut time = LinkTime::new(false);
        assert!(time.on());
        let decisions: Vec<bool> = (0..8)
            .map(|_| pass(&mut time, |i| if i % 12 == 0 { 26.6 } else { 0.0 }, false))
            .collect();
        assert_eq!(decisions.iter().position(|&on| !on), Some(2));
        assert!(decisions[2..].iter().all(|&on| !on));
    }

    /// After a 128K prompt the same card copies ~0.5 experts an invocation —
    /// ~13 µs, a sixth of the line: an off gate stays off.
    #[test]
    fn a_churned_zone_on_a_fast_link_stays_off() {
        let mut time = at(0.0, false);
        assert!((0..100).all(|_| !pass(&mut time, |i| if i % 2 == 0 { 26.6 } else { 0.0 }, false)));
    }

    /// A streaming card waits on cold experts: a pack read in every tenth
    /// invocation is ~260 µs an invocation, and the gate never switches off.
    #[test]
    fn a_streaming_working_set_keeps_reading_ahead_on() {
        let mut time = LinkTime::new(false);
        assert!((0..100).all(|_| pass(
            &mut time,
            |i| if i % 10 == 0 { 2_500.0 } else { 0.0 },
            false
        )));
        assert!(time.us() > ON_US);
    }

    /// Between the lines the decision holds: from off, 60 µs stays off; from
    /// on, it stays on.
    #[test]
    fn a_figure_between_the_lines_keeps_the_last_decision() {
        let mut from_off = at(60.0, false);
        let mut from_on = at(60.0, true);
        for _ in 0..100 {
            pass(&mut from_off, |_| 60.0, false);
            pass(&mut from_on, |_| 60.0, false);
        }
        assert!(!from_off.on());
        assert!(from_on.on());
    }

    /// **Bursts inside a pass do not move the decision.** A dip's misses land
    /// a few layers at a time: seven invocations of 400 µs and forty-one
    /// resident ones, 58 µs a pass, between the lines. Decided per invocation
    /// over 64 of them, that pattern crossed both lines inside every pass;
    /// decided per pass, an on gate stays on and never changes.
    #[test]
    fn bursts_inside_a_pass_never_flip_the_gate() {
        let mut time = at(OFF_US + 20.0, true);
        let decisions: Vec<bool> = (0..200)
            .map(|_| pass(&mut time, |i| if i < 7 { 400.0 } else { 0.0 }, false))
            .collect();
        assert!(decisions.iter().all(|&on| on));
    }

    /// A burst of cold reads on an otherwise resident cache switches the gate
    /// on at the pass that carried it — 625 µs averaged in — and it switches
    /// off again ten idle passes later, at 35 µs.
    #[test]
    fn a_burst_switches_on_and_decays_off() {
        let mut time = LinkTime::new(true);
        assert!(!time.on());
        assert!(!pass(&mut time, |_| 0.0, false));
        assert!(pass(&mut time, |_| 2_500.0, false));
        let off_at = (0..100).position(|_| !pass(&mut time, |_| 0.0, false));
        assert_eq!(off_at, Some(9));
        assert!(!time.on());
    }

    /// **Fillable ground holds the gate on, and lets it go once filled.** A
    /// concession handed back leaves empty slots and recently routed experts
    /// to fill them with: below either line, the gate reads ahead while that
    /// stands and stops the pass after it does not.
    #[test]
    fn fillable_ground_reads_ahead_until_the_zone_has_filled() {
        let mut time = at(0.0, false);
        assert!(pass(&mut time, |_| 0.0, true), "empty ground opens it");
        assert!(pass(&mut time, |_| 0.0, true), "and holds it open");
        assert!(!pass(&mut time, |_| 0.0, false), "filled, the link decides");
    }

    /// A pass with no decode invocation — a prompt's — says nothing about the
    /// link: the average and the decision stand.
    #[test]
    fn a_pass_without_decode_leaves_the_average() {
        let mut time = at(60.0, true);
        assert!(time.end_pass(false));
        assert_eq!(time.us(), 60.0);
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
