//! Where the scheduler thread's wall clock went, by the shape of each step.
//!
//! The airtime split says what the creep is *offered*: with `R` near zero and
//! no debt, a co-batched step hands the creep most of the device. What it is
//! *given* also depends on whether a cohort was ready to ride, whether
//! admission let work in, and how long the loop spent outside any forward.
//! On BlackWorld (d919bd410, 2026-10-20) heavy-read's creep held the device
//! about 13% of the wall with the split offering it two thirds, and nothing in
//! the rates line could say where the rest went. This census counts every step
//! by what it carried, and the time no forward ran.

use std::time::{Duration, Instant};

/// What one step of the scheduler loop carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StepShape {
    /// Decode with the creep or a section chunk riding it.
    Cobatched,
    /// Decode with only glue riding: no prefill rows advanced.
    GlueOnly,
    /// Decode alone because the creep owed decode its airtime.
    DecodeOwed,
    /// Decode alone with prefill admitted and not owed: no cohort rode.
    DecodeHeld,
    /// Decode alone with prefill waiting for admission and none admitted.
    DecodeQueued,
    /// Decode alone with no prefill work anywhere.
    DecodeAlone,
    /// The creep with no decode beside it.
    CreepOnly,
    /// A decode-less step its tier was refused: it waits for ground.
    Refused,
}

const SHAPES: [StepShape; 8] = [
    StepShape::Cobatched,
    StepShape::GlueOnly,
    StepShape::DecodeOwed,
    StepShape::DecodeHeld,
    StepShape::DecodeQueued,
    StepShape::DecodeAlone,
    StepShape::CreepOnly,
    StepShape::Refused,
];

impl StepShape {
    fn label(self) -> &'static str {
        match self {
            StepShape::Cobatched => "cob",
            StepShape::GlueOnly => "glue",
            StepShape::DecodeOwed => "owed",
            StepShape::DecodeHeld => "held",
            StepShape::DecodeQueued => "queued",
            StepShape::DecodeAlone => "alone",
            StepShape::CreepOnly => "creep",
            StepShape::Refused => "refused",
        }
    }

    fn index(self) -> usize {
        SHAPES
            .iter()
            .position(|&s| s == self)
            .expect("every shape is listed")
    }

    /// The shape of a decode step, from what it carried and what was waiting.
    pub(super) fn of_decode(
        cobatched: bool,
        carried: bool,
        owed: bool,
        admitted: bool,
        queued: bool,
    ) -> Self {
        match (cobatched, carried) {
            (true, true) => StepShape::Cobatched,
            (true, false) => StepShape::GlueOnly,
            _ if owed && admitted => StepShape::DecodeOwed,
            _ if admitted => StepShape::DecodeHeld,
            _ if queued => StepShape::DecodeQueued,
            _ => StepShape::DecodeAlone,
        }
    }
}

/// What the scheduler thread was doing between steps.
///
/// **The wall no step covers, by what covered it.** On BlackWorld (41bd79ae3,
/// 2026-10-20) heavy-read stepped 19–26 s of every minute with work queued,
/// and the rest sat inside the decode quantum by the phase timeline — host
/// work around each step, which nothing named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Activity {
    /// The loop's drain of new submissions: projection, elevation, gap-fill.
    Drain,
    /// Promoting queued prefills and finished ones: the admission passes.
    Promote,
    /// Mid-quantum admission: its drain and its admission pass.
    MidAdmit,
    /// Continuous re-projection between steps.
    Reproject,
    /// Cleaning up finished turns: their seals.
    Seal,
    /// Publishing the learned rates: the placed step and the traded width.
    Rates,
    /// The gallery keep trade and its eviction pass.
    Gallery,
    /// Tool-call stencil injection.
    Stencil,
    /// Giving idle recurrent stores back.
    Park,
    /// The between-quanta housekeeping block.
    House,
    /// Blocked on the request channel with nothing to run.
    Idle,
    /// The decode quantum's relief check and what it relieved.
    Relief,
    /// Withdrawing prefills their callers let go of.
    Withdraw,
    /// Telling waiting prompts what they wait behind.
    Waits,
}

const ACTIVITIES: [Activity; 14] = [
    Activity::Drain,
    Activity::Promote,
    Activity::MidAdmit,
    Activity::Reproject,
    Activity::Seal,
    Activity::Rates,
    Activity::Gallery,
    Activity::Stencil,
    Activity::Park,
    Activity::House,
    Activity::Idle,
    Activity::Relief,
    Activity::Withdraw,
    Activity::Waits,
];

impl Activity {
    fn label(self) -> &'static str {
        match self {
            Activity::Drain => "drain",
            Activity::Promote => "promote",
            Activity::MidAdmit => "midadmit",
            Activity::Reproject => "reproject",
            Activity::Seal => "seal",
            Activity::Rates => "rates",
            Activity::Gallery => "gallery",
            Activity::Stencil => "stencil",
            Activity::Park => "park",
            Activity::House => "house",
            Activity::Idle => "idle",
            Activity::Relief => "relief",
            Activity::Withdraw => "withdraw",
            Activity::Waits => "waits",
        }
    }

    fn index(self) -> usize {
        ACTIVITIES
            .iter()
            .position(|&a| a == self)
            .expect("every activity is listed")
    }
}

/// Steps and their wall time by shape, and the activities between them, since
/// the census last reported.
#[derive(Debug)]
pub(super) struct StepCensus {
    since: Instant,
    steps: [u64; SHAPES.len()],
    wall: [Duration; SHAPES.len()],
    spent: [Duration; ACTIVITIES.len()],
}

impl Default for StepCensus {
    fn default() -> Self {
        Self::starting(Instant::now())
    }
}

impl StepCensus {
    fn starting(since: Instant) -> Self {
        Self {
            since,
            steps: [0; SHAPES.len()],
            wall: [Duration::ZERO; SHAPES.len()],
            spent: [Duration::ZERO; ACTIVITIES.len()],
        }
    }

    pub(super) fn record(&mut self, shape: StepShape, took: Duration) {
        let i = shape.index();
        self.steps[i] += 1;
        self.wall[i] += took;
    }

    /// File `took` of wall the thread spent on `activity` between steps.
    pub(super) fn spend(&mut self, activity: Activity, took: Duration) {
        self.spent[activity.index()] += took;
    }

    /// The census since it last reported, as `shape=steps/ms` for every shape
    /// that ran, `outside` for the wall no step covered, and what of it each
    /// activity took with `other` the rest; then start again.
    pub(super) fn take(&mut self, now: Instant) -> String {
        let elapsed = now.saturating_duration_since(self.since);
        let stepped: Duration = self.wall.iter().sum();
        let mut parts: Vec<String> = SHAPES
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.steps[i] > 0)
            .map(|(i, s)| {
                format!(
                    "{}={}/{}ms",
                    s.label(),
                    self.steps[i],
                    self.wall[i].as_millis()
                )
            })
            .collect();
        let outside = elapsed.saturating_sub(stepped);
        parts.push(format!(
            "outside={}ms of {}ms",
            outside.as_millis(),
            elapsed.as_millis()
        ));
        let spent: Duration = self.spent.iter().sum();
        let mut around: Vec<String> = ACTIVITIES
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.spent[i].is_zero())
            .map(|(i, a)| format!("{}={}ms", a.label(), self.spent[i].as_millis()))
            .collect();
        around.push(format!(
            "other={}ms",
            outside.saturating_sub(spent).as_millis()
        ));
        parts.push(format!("around[{}]", around.join(" ")));
        *self = Self::starting(now);
        parts.join(" ")
    }
}

/// The recent step time at each decode width the scheduler has stepped, an
/// average per width.
///
/// **A step at a width bounds every narrower one from above.** One more
/// decode never makes a step faster, so a step measured at `w` decodes — alone
/// or carrying a creep, which only adds — is the most a step at `w` or fewer
/// can take. What the rates publish for a width is held to the fastest fresh
/// measurement at that width or wider: on BlackWorld (136060fab, 2026-10-20)
/// its step sixteen decodes wider was published at 492 ms where heavy-decode
/// had stepped 20–24 decodes in 35–45 ms, and shift's first decode phase went
/// 28 of 48 to WhiteWorld for it.
#[derive(Debug, Default)]
pub(super) struct WidthSteps {
    by: std::collections::BTreeMap<usize, (Duration, Instant)>,
}

impl WidthSteps {
    /// The newest step's weight in each width's average.
    const NEWEST: f64 = 0.25;

    /// Fold one step of `decodes` decodes that took `took`, at `now`.
    pub(super) fn record(&mut self, decodes: usize, took: Duration, now: Instant) {
        if decodes == 0 {
            return;
        }
        let avg = match self.by.get(&decodes) {
            Some(&(a, _)) => a.mul_f64(1.0 - Self::NEWEST) + took.mul_f64(Self::NEWEST),
            None => took,
        };
        self.by.insert(decodes, (avg, now));
    }

    /// The most a step of `decodes` decodes takes, by the fastest width at or
    /// above it measured within `fresh` of `now` — `None` with none.
    pub(super) fn bound(&self, decodes: usize, now: Instant, fresh: Duration) -> Option<Duration> {
        self.by
            .range(decodes.max(1)..)
            .filter(|(_, &(_, at))| now.saturating_duration_since(at) <= fresh)
            .map(|(_, &(avg, _))| avg)
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decode_step_is_filed_by_what_held_the_creep_off_it() {
        let of = StepShape::of_decode;
        assert_eq!(of(true, true, false, true, false), StepShape::Cobatched);
        assert_eq!(of(true, false, true, true, false), StepShape::GlueOnly);
        assert_eq!(of(false, false, true, true, true), StepShape::DecodeOwed);
        assert_eq!(of(false, false, false, true, true), StepShape::DecodeHeld);
        assert_eq!(
            of(false, false, false, false, true),
            StepShape::DecodeQueued
        );
        assert_eq!(of(false, false, true, false, false), StepShape::DecodeAlone);
    }

    #[test]
    fn a_step_at_a_width_bounds_every_narrower_one() {
        let t0 = Instant::now();
        let fresh = Duration::from_secs(60);
        let mut w = WidthSteps::default();
        w.record(24, Duration::from_millis(40), t0);
        w.record(8, Duration::from_millis(30), t0);
        // Sixteen wider than eight: bounded by the 24-decode step.
        assert_eq!(w.bound(24, t0, fresh), Some(Duration::from_millis(40)));
        assert_eq!(w.bound(9, t0, fresh), Some(Duration::from_millis(40)));
        // At eight, the fastest of eight and wider.
        assert_eq!(w.bound(8, t0, fresh), Some(Duration::from_millis(30)));
        // Wider than any measured: nothing to say.
        assert_eq!(w.bound(40, t0, fresh), None);
        // A stale width says nothing either.
        let later = t0 + Duration::from_secs(120);
        w.record(8, Duration::from_millis(30), later);
        assert_eq!(w.bound(9, later, fresh), None);
        assert_eq!(w.bound(8, later, fresh), Some(Duration::from_millis(30)));
    }

    #[test]
    fn the_census_reports_each_shape_and_what_covered_the_rest() {
        let t0 = Instant::now();
        let mut c = StepCensus::starting(t0);
        c.record(StepShape::Cobatched, Duration::from_millis(300));
        c.record(StepShape::Cobatched, Duration::from_millis(200));
        c.record(StepShape::DecodeHeld, Duration::from_millis(100));
        c.spend(Activity::Rates, Duration::from_millis(250));
        c.spend(Activity::Seal, Duration::from_millis(50));
        let line = c.take(t0 + Duration::from_secs(1));
        assert_eq!(
            line,
            "cob=2/500ms held=1/100ms outside=400ms of 1000ms \
             around[seal=50ms rates=250ms other=100ms]"
        );
        // And it starts again from the report.
        let line = c.take(t0 + Duration::from_secs(2));
        assert_eq!(line, "outside=1000ms of 1000ms around[other=1000ms]");
    }
}
