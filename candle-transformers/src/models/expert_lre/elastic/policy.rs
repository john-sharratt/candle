//! The elastic warm tier's controller (`docs/elastic_warm_tier.md` §4): from
//! this process's hard faults and the machine's free RAM, once a tick, whether
//! the pageable part grows by a segment (a chunk), shrinks, or holds.
//!
//! Pure: every reading is passed in and time is counted in ticks, so the whole
//! law pins down in tests against signal sequences.

use std::collections::VecDeque;

const GIB: u64 = 1024 * 1024 * 1024;

/// Fast fault average's weight on the newest tick: reacts within ~2 s.
const FAST_ALPHA: f64 = 0.5;
/// Slow fault average's weight: decides quiet over ~10 s.
const SLOW_ALPHA: f64 = 0.1;

/// Faults a second, fast average, past which the tier sheds a quarter at once.
pub(crate) const SEVERE_FAULTS: f64 = 2_000.0;
/// Free RAM below which the tier sheds a quarter at once.
pub(crate) const FLOOR: u64 = GIB;
/// Faults a second, fast average, that count as pressure.
pub(crate) const PRESSURE_FAULTS: f64 = 400.0;
/// Ticks running at pressure before a one-segment shrink.
pub(crate) const PRESSURE_TICKS: u32 = 3;
/// Faults a second, slow average, under which the process is quiet.
pub(crate) const QUIET_FAULTS: f64 = 25.0;
/// Quiet ticks running before a grow.
pub(crate) const QUIET_TICKS: u32 = 15;
/// Quiet ticks running before a grow may press other processes' pages out.
pub(crate) const PRESS_QUIET_TICKS: u32 = 60;
/// Free RAM a grow leaves, and that a build shrink restores.
pub(crate) const HEADROOM: u64 = 3 * GIB;
/// A fall in free RAM this large over [`SLOPE_TICKS`] is a build starting.
pub(crate) const SLOPE_BYTES: u64 = 2 * GIB;
pub(crate) const SLOPE_TICKS: usize = 3;
/// Ticks between shrinks: the pages need a moment to reach the free list.
pub(crate) const SHRINK_DWELL: u64 = 2;
/// Ticks between grows, and from a shrink to the next grow.
pub(crate) const GROW_DWELL: u64 = 5;
/// Ticks running with free RAM under [`HEADROOM`] before the tier gives back
/// what restores it: a build that ramps too slowly for the slope rule still
/// takes the machine's headroom, and gets it back within seconds.
pub(crate) const SHORT_TICKS: u32 = 5;
/// Ticks between pressed grows.
pub(crate) const PRESS_DWELL: u64 = 30;
/// Ticks after a shrink before a grow may press: a shrink says something on
/// the machine is using its RAM, and pressing writes through a segment's worth
/// of memory that trims whoever is — a build's live pages as readily as an idle
/// app's (WhiteWorld, 2026-11-21: a press 70 s after a shrink took free RAM
/// back to 1.86 GiB under a 14 GiB build).
pub(crate) const PRESS_AFTER_SHRINK: u64 = 10 * 60;
/// The back-off a ceiling holds growth for, at its least and its most.
pub(crate) const BACKOFF_MIN: u64 = 60;
pub(crate) const BACKOFF_MAX: u64 = 30 * 60;
/// Ticks without pressure after which the back-off halves: twice the longest
/// back-off, so the quiet a tier spends waiting one out never relaxes it.
pub(crate) const RELAX_AFTER: u64 = 2 * BACKOFF_MAX;

/// One tick's readings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Signals {
    /// This process's hard faults since it started.
    pub(crate) hard_faults: u64,
    /// Free physical RAM.
    pub(crate) available: u64,
}

/// What the tier holds and may hold, in segments.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) held: usize,
    /// The most the tier may hold: every evictable expert, the host budget and
    /// the driver's addressable room, whichever is least.
    pub(crate) max: usize,
    pub(crate) segment_bytes: u64,
}

/// Why a shrink was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShrinkReason {
    /// Faults past [`SEVERE_FAULTS`], or free RAM under [`FLOOR`].
    Severe,
    /// Free RAM falling fast towards the headroom.
    Build,
    /// Free RAM under the headroom for [`SHORT_TICKS`] running.
    Short,
    /// Faults past [`PRESSURE_FAULTS`] for [`PRESSURE_TICKS`] running.
    Pressure,
}

impl ShrinkReason {
    pub(crate) fn name(self) -> &'static str {
        match self {
            ShrinkReason::Severe => "severe",
            ShrinkReason::Build => "build",
            ShrinkReason::Short => "short",
            ShrinkReason::Pressure => "pressure",
        }
    }
}

/// One tick's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    Hold,
    /// Take one segment; `press` first when free RAM alone cannot pay for it.
    Grow {
        press: bool,
    },
    Shrink {
        segments: usize,
        reason: ShrinkReason,
    },
}

/// Growth held at `segments` until tick `until`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ceiling {
    pub(crate) segments: usize,
    pub(crate) until: u64,
}

/// The controller's state between ticks.
#[derive(Debug, Clone)]
pub(crate) struct Policy {
    tick: u64,
    last_faults: Option<u64>,
    fast: f64,
    slow: f64,
    pressure_run: u32,
    quiet_run: u32,
    /// Ticks running with free RAM under the headroom.
    short_run: u32,
    /// Free RAM over the last [`SLOPE_TICKS`] ticks, oldest first.
    available: VecDeque<u64>,
    last_shrink: Option<u64>,
    last_grow: Option<u64>,
    last_press: Option<u64>,
    ceiling: Option<Ceiling>,
    backoff: u64,
    /// When a ceiling last lifted by a probe.
    last_lift: Option<u64>,
    /// When pressure last set a ceiling, or the back-off last relaxed.
    last_pressure: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy {
    pub(crate) fn new() -> Self {
        Self {
            tick: 0,
            last_faults: None,
            fast: 0.0,
            slow: 0.0,
            pressure_run: 0,
            quiet_run: 0,
            short_run: 0,
            available: VecDeque::with_capacity(SLOPE_TICKS + 1),
            last_shrink: None,
            last_grow: None,
            last_press: None,
            ceiling: None,
            backoff: BACKOFF_MIN,
            last_lift: None,
            last_pressure: 0,
        }
    }

    /// The fault averages, fast and slow, in faults a tick.
    pub(crate) fn faults(&self) -> (f64, f64) {
        (self.fast, self.slow)
    }

    pub(crate) fn ceiling(&self) -> Option<Ceiling> {
        self.ceiling
    }

    pub(crate) fn backoff(&self) -> u64 {
        self.backoff
    }

    /// A grow was refused — the allocator said no, or a press left no room
    /// for it: hold growth where it is for a back-off.
    pub(crate) fn refused(&mut self, held: usize) {
        self.ceiling = Some(Ceiling {
            segments: held,
            until: self.tick + self.backoff,
        });
    }

    /// One tick: take the readings, answer what to do.
    pub(crate) fn tick(&mut self, s: Signals, l: Limits) -> Decision {
        self.tick += 1;
        let rate = self
            .last_faults
            .map_or(0.0, |prev| s.hard_faults.saturating_sub(prev) as f64);
        self.last_faults = Some(s.hard_faults);
        self.fast = FAST_ALPHA * rate + (1.0 - FAST_ALPHA) * self.fast;
        self.slow = SLOW_ALPHA * rate + (1.0 - SLOW_ALPHA) * self.slow;
        self.pressure_run = if self.fast > PRESSURE_FAULTS {
            self.pressure_run + 1
        } else {
            0
        };
        self.quiet_run = if self.slow < QUIET_FAULTS {
            self.quiet_run + 1
        } else {
            0
        };
        self.short_run = if s.available < HEADROOM {
            self.short_run + 1
        } else {
            0
        };
        let fell = self
            .available
            .front()
            .is_some_and(|&then| then.saturating_sub(s.available) >= SLOPE_BYTES);
        self.available.push_back(s.available);
        while self.available.len() > SLOPE_TICKS {
            self.available.pop_front();
        }
        self.relax();
        self.lift(l);

        if let Some(d) = self.shrink(s, l, fell) {
            return d;
        }
        self.grow(s, l)
    }

    fn since(&self, at: Option<u64>) -> u64 {
        at.map_or(u64::MAX, |t| self.tick - t)
    }

    /// Half the back-off after a long quiet stretch.
    fn relax(&mut self) {
        if self.tick - self.last_pressure >= RELAX_AFTER && self.backoff > BACKOFF_MIN {
            self.backoff = (self.backoff / 2).max(BACKOFF_MIN);
            self.last_pressure = self.tick;
        }
    }

    /// An expired ceiling lifts one segment, a probe; one at the tier's bound
    /// goes.
    fn lift(&mut self, l: Limits) {
        if let Some(c) = self.ceiling {
            if self.tick >= c.until {
                self.ceiling = (c.segments + 1 < l.max).then_some(Ceiling {
                    segments: c.segments + 1,
                    until: self.tick + self.backoff,
                });
                self.last_lift = Some(self.tick);
            }
        }
    }

    fn shrink(&mut self, s: Signals, l: Limits, fell: bool) -> Option<Decision> {
        if l.held == 0 || self.since(self.last_shrink) < SHRINK_DWELL {
            return None;
        }
        let seg = l.segment_bytes.max(1);
        let (segments, reason) = if self.fast > SEVERE_FAULTS || s.available < FLOOR {
            (l.held.div_ceil(4), ShrinkReason::Severe)
        } else if fell && s.available < HEADROOM + seg {
            let short = HEADROOM.saturating_sub(s.available).div_ceil(seg) as usize;
            (short.max(1), ShrinkReason::Build)
        } else if self.short_run >= SHORT_TICKS {
            let short = HEADROOM.saturating_sub(s.available).div_ceil(seg) as usize;
            (short.max(1), ShrinkReason::Short)
        } else if self.pressure_run >= PRESSURE_TICKS {
            (1, ShrinkReason::Pressure)
        } else {
            return None;
        };
        let segments = segments.clamp(1, l.held);
        // Pressure recurring soon after a probe lifted the ceiling: the edge is
        // where it was, so the next probe waits twice as long.
        if self
            .last_lift
            .is_some_and(|t| self.tick - t <= 2 * self.backoff)
        {
            self.backoff = (self.backoff * 2).min(BACKOFF_MAX);
        }
        // The ceiling is the size the shrink leaves, not the one it left: the
        // machine has just said it wants those segments, and growing back into
        // them while it still does is the churn (WhiteWorld, 2026-11-21: shed
        // 18 → 13 and regrown to 17 within 70 s under a held 14 GiB build).
        // Each probe past it is one segment per back-off.
        self.ceiling = Some(Ceiling {
            segments: l.held - segments,
            until: self.tick + self.backoff,
        });
        self.last_lift = None;
        self.last_pressure = self.tick;
        self.last_shrink = Some(self.tick);
        self.pressure_run = 0;
        self.short_run = 0;
        Some(Decision::Shrink { segments, reason })
    }

    fn grow(&mut self, s: Signals, l: Limits) -> Decision {
        let capped = self.ceiling.is_some_and(|c| l.held >= c.segments);
        if self.quiet_run < QUIET_TICKS
            || l.held >= l.max
            || capped
            || self.since(self.last_grow) < GROW_DWELL
            || self.since(self.last_shrink) < GROW_DWELL
        {
            return Decision::Hold;
        }
        if s.available >= HEADROOM + l.segment_bytes {
            self.last_grow = Some(self.tick);
            return Decision::Grow { press: false };
        }
        // Pressing is for RAM held by idle pages, never short of the headroom
        // (that is the machine's), and never soon after a shrink said something
        // is using it. The controller grows only if the press made the room.
        if s.available >= HEADROOM
            && self.quiet_run >= PRESS_QUIET_TICKS
            && self.since(self.last_shrink) >= PRESS_AFTER_SHRINK
            && self.since(self.last_press) >= PRESS_DWELL
        {
            self.last_grow = Some(self.tick);
            self.last_press = Some(self.tick);
            return Decision::Grow { press: true };
        }
        Decision::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEG: u64 = GIB;

    fn limits(held: usize, max: usize) -> Limits {
        Limits {
            held,
            max,
            segment_bytes: SEG,
        }
    }

    /// Run `ticks` with no new faults and `available` free, applying each
    /// decision to `held`. Returns the decisions.
    fn run(
        p: &mut Policy,
        held: &mut usize,
        max: usize,
        faults: &mut u64,
        per_tick: u64,
        available: u64,
        ticks: usize,
    ) -> Vec<Decision> {
        (0..ticks)
            .map(|_| {
                *faults += per_tick;
                let d = p.tick(
                    Signals {
                        hard_faults: *faults,
                        available,
                    },
                    limits(*held, max),
                );
                match d {
                    Decision::Grow { .. } => *held += 1,
                    Decision::Shrink { segments, .. } => *held -= segments,
                    Decision::Hold => {}
                }
                d
            })
            .collect()
    }

    fn grows(ds: &[Decision]) -> usize {
        ds.iter()
            .filter(|d| matches!(d, Decision::Grow { .. }))
            .count()
    }

    /// **A quiet machine with free RAM grows one segment per dwell**, after the
    /// quiet run, up to its bound and no further.
    #[test]
    fn a_quiet_machine_grows_one_segment_per_dwell_to_its_bound() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (2usize, 0u64);
        let ds = run(&mut p, &mut held, 6, &mut faults, 0, 40 * GIB, 14);
        assert_eq!(grows(&ds), 0, "not before {QUIET_TICKS} quiet ticks");
        let ds = run(&mut p, &mut held, 6, &mut faults, 0, 40 * GIB, 1);
        assert_eq!(ds, vec![Decision::Grow { press: false }]);
        let ds = run(&mut p, &mut held, 6, &mut faults, 0, 40 * GIB, 4);
        assert_eq!(grows(&ds), 0, "one per {GROW_DWELL} ticks");
        run(&mut p, &mut held, 6, &mut faults, 0, 40 * GIB, 100);
        assert_eq!(held, 6, "and no further than its bound");
    }

    /// **Growth leaves the headroom**: free RAM short of it holds the tier,
    /// until a long quiet run allows one pressed grow per press dwell.
    #[test]
    fn growth_leaves_the_headroom_and_presses_only_after_a_long_quiet() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (2usize, 0u64);
        let tight = HEADROOM + SEG / 2;
        let ds = run(
            &mut p,
            &mut held,
            10,
            &mut faults,
            0,
            tight,
            PRESS_QUIET_TICKS as usize - 1,
        );
        assert_eq!(grows(&ds), 0);
        let ds = run(&mut p, &mut held, 10, &mut faults, 0, tight, 1);
        assert_eq!(ds, vec![Decision::Grow { press: true }]);
        let ds = run(
            &mut p,
            &mut held,
            10,
            &mut faults,
            0,
            tight,
            PRESS_DWELL as usize - 1,
        );
        assert_eq!(grows(&ds), 0, "one pressed grow per {PRESS_DWELL} ticks");
    }

    /// **Severe faults shed a quarter at once**, and free RAM under the floor
    /// does the same with no fault at all.
    #[test]
    fn severe_faults_or_an_empty_free_list_shed_a_quarter() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (12usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 1);
        let ds = run(&mut p, &mut held, 20, &mut faults, 10_000, 40 * GIB, 1);
        assert_eq!(
            ds,
            vec![Decision::Shrink {
                segments: 3,
                reason: ShrinkReason::Severe
            }]
        );
        let mut p = Policy::new();
        let mut held = 10usize;
        let ds = run(&mut p, &mut held, 20, &mut faults, 0, GIB / 2, 1);
        assert_eq!(
            ds,
            vec![Decision::Shrink {
                segments: 3,
                reason: ShrinkReason::Severe
            }]
        );
    }

    /// **A build — free RAM falling fast — is answered before any fault**,
    /// with as many segments as put the headroom back.
    #[test]
    fn a_build_is_answered_before_any_fault_by_the_headroom_it_took() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (10usize, 0u64);
        for avail in [20 * GIB, 14 * GIB, 8 * GIB] {
            let ds = run(&mut p, &mut held, 20, &mut faults, 0, avail, 1);
            assert_eq!(ds, vec![Decision::Hold], "{avail}: above the headroom");
        }
        let ds = run(&mut p, &mut held, 20, &mut faults, 0, GIB + GIB / 2, 1);
        assert_eq!(
            ds,
            vec![Decision::Shrink {
                segments: 2,
                reason: ShrinkReason::Build
            }]
        );
    }

    /// **Pressure takes three ticks running and one segment**; another
    /// shrink waits out the dwell.
    #[test]
    fn pressure_sheds_one_segment_after_three_ticks() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (8usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 1);
        let ds = run(&mut p, &mut held, 20, &mut faults, 900, 40 * GIB, 4);
        let shrinks: Vec<_> = ds
            .iter()
            .filter(|d| matches!(d, Decision::Shrink { .. }))
            .collect();
        assert_eq!(
            shrinks,
            vec![&Decision::Shrink {
                segments: 1,
                reason: ShrinkReason::Pressure
            }]
        );
        assert_eq!(held, 7);
    }

    /// **A build that ramps too slowly for the slope still gets the headroom
    /// back within seconds**: free RAM under it for [`SHORT_TICKS`] running
    /// sheds what restores it, with this process quiet — its own faults are
    /// not the only sign the machine needs its RAM.
    #[test]
    fn free_ram_held_short_of_the_headroom_is_given_back() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (18usize, 0u64);
        let ds = run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            GIB + GIB / 4,
            SHORT_TICKS as usize,
        );
        assert!(ds[..SHORT_TICKS as usize - 1]
            .iter()
            .all(|d| *d == Decision::Hold));
        assert_eq!(
            ds[SHORT_TICKS as usize - 1],
            Decision::Shrink {
                segments: 2,
                reason: ShrinkReason::Short
            }
        );
        assert_eq!(held, 16);
    }

    /// **No growth back into what a shrink gave up**: the ceiling is the size
    /// the shrink left, held through the back-off however free RAM looks — a
    /// build that took the headroom and is still running leaves it free once
    /// the tier has shrunk.
    #[test]
    fn a_shrink_is_not_grown_back_while_its_back_off_holds() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (18usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 0, GIB / 2, 1);
        assert_eq!(held, 13, "severe: a quarter");
        assert_eq!(p.ceiling().map(|c| c.segments), Some(13));
        let ds = run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            5 * GIB,
            BACKOFF_MIN as usize - 1,
        );
        assert_eq!(grows(&ds), 0, "{ds:?}");
        assert_eq!(held, 13);
    }

    /// **Pressing waits long after a shrink**, and never presses into the
    /// headroom: free RAM just past it, with this process quiet, presses only
    /// once [`PRESS_AFTER_SHRINK`] has passed since the last shrink.
    #[test]
    fn pressing_waits_long_after_a_shrink() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (8usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 900, 40 * GIB, 4);
        assert_eq!(held, 7, "a pressure shrink");
        let tight = HEADROOM + SEG / 2;
        let ds = run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            tight,
            PRESS_AFTER_SHRINK as usize - 10,
        );
        assert!(
            ds.iter()
                .all(|d| !matches!(d, Decision::Grow { press: true })),
            "no press within {PRESS_AFTER_SHRINK} ticks of a shrink"
        );
        let ds = run(&mut p, &mut held, 20, &mut faults, 0, tight, 60);
        assert!(ds.contains(&Decision::Grow { press: true }), "{ds:?}");
        let mut p = Policy::new();
        let mut held = 8usize;
        let ds = run(&mut p, &mut held, 20, &mut faults, 0, HEADROOM - 1, 4);
        assert!(
            ds.iter().all(|d| !matches!(d, Decision::Grow { .. })),
            "never a press short of the headroom"
        );
    }

    /// **A shrink sets a ceiling at the size it leaves**, held for the
    /// back-off; then the ceiling lifts one segment, a probe.
    #[test]
    fn a_ceiling_holds_growth_for_the_back_off_then_lifts_one() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (8usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 1);
        run(&mut p, &mut held, 20, &mut faults, 900, 40 * GIB, 3);
        assert_eq!(held, 7);
        assert_eq!(p.ceiling().map(|c| c.segments), Some(7));
        let ds = run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            40 * GIB,
            BACKOFF_MIN as usize - 1,
        );
        assert_eq!(grows(&ds), 0, "held at the ceiling through the back-off");
        let ds = run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 10);
        assert_eq!(grows(&ds), 1, "the lift allows one segment, then holds");
        assert_eq!(held, 8);
    }

    /// **Pressure right after a probe doubles the back-off**, to its cap; a
    /// long quiet stretch halves it.
    #[test]
    fn pressure_after_a_probe_doubles_the_back_off_and_quiet_halves_it() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (8usize, 0u64);
        run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 1);
        let mut backoffs = Vec::new();
        for _ in 0..8 {
            // Pressure until it sheds, then quiet until the probe regrows.
            while !matches!(
                run(&mut p, &mut held, 20, &mut faults, 900, 40 * GIB, 1)[0],
                Decision::Shrink { .. }
            ) {}
            backoffs.push(p.backoff());
            while !matches!(
                run(&mut p, &mut held, 20, &mut faults, 0, 40 * GIB, 1)[0],
                Decision::Grow { .. }
            ) {}
        }
        assert_eq!(backoffs[0], BACKOFF_MIN, "the first meeting with the edge");
        assert!(
            backoffs
                .windows(2)
                .all(|w| w[1] == (w[0] * 2).min(BACKOFF_MAX)),
            "{backoffs:?}"
        );
        assert_eq!(*backoffs.last().unwrap(), BACKOFF_MAX);
        run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            40 * GIB,
            RELAX_AFTER as usize + 1,
        );
        assert_eq!(p.backoff(), BACKOFF_MAX / 2);
    }

    /// **The edge is met rarely**: a machine that pages past one size settles
    /// one segment under it, and over a day probes past it only a handful of
    /// times.
    #[test]
    fn a_steady_edge_is_probed_rarely() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (2usize, 0u64);
        let edge = 9usize;
        let mut shrinks = 0;
        for _ in 0..24 * 3600 {
            let per_tick = if held > edge { 900 } else { 0 };
            if matches!(
                run(&mut p, &mut held, 30, &mut faults, per_tick, 40 * GIB, 1)[0],
                Decision::Shrink { .. }
            ) {
                shrinks += 1;
            }
            assert!(
                held <= edge + 1,
                "never more than one segment past the edge"
            );
        }
        assert!(held == edge || held == edge + 1, "{held}");
        assert!(shrinks <= 60, "{shrinks} shrinks in a day");
    }

    /// A refused grow holds the tier where it is for a back-off.
    #[test]
    fn a_refused_grow_holds_for_a_back_off() {
        let mut p = Policy::new();
        let (mut held, mut faults) = (4usize, 0u64);
        run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            40 * GIB,
            QUIET_TICKS as usize,
        );
        assert_eq!(held, 5);
        p.refused(held);
        let ds = run(
            &mut p,
            &mut held,
            20,
            &mut faults,
            0,
            40 * GIB,
            BACKOFF_MIN as usize - 1,
        );
        assert_eq!(grows(&ds), 0);
    }
}
