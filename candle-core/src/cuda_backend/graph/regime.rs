//! Which executables the library keeps: learned, trialled, then locked.
//!
//! A graph launch replaces a segment's launches with one, but keeping a shape
//! has costs of its own: it is recorded every wave (that is how its arguments
//! are refreshed), the record gate holds other threads' quiet calls while it
//! records, and its executable is driver memory outside the reservation. A
//! shape is worth those only if it recurs and its graph is faster than its
//! launches issued one by one. So the library finds out, shape by shape, and
//! then stops changing:
//!
//! 1. **Learn.** Segments record, and the library admits what it folds, into
//!    its fixed budget. Nothing is evicted to make room: the learn ends at the
//!    first admission that does not fit, or once [`LEARN_IDLE_WAVES`] pass
//!    with no admission at all — the traffic has shown its shapes.
//! 2. **Trial**, for [`TRIAL_WAVES`] waves. Every shape admitted in the learn
//!    alternates graphed and eager each time a segment predicts it, and each
//!    run is timed on the device from the segment's start to its end — host
//!    lateness included, which is what a graph removes. A shape hit in fewer
//!    than one trial wave in [`MIN_HIT_DIVISOR`], or whose graphed median is not
//!    at least [`MIN_GAIN`] faster than its eager one, is **banned**: dropped,
//!    and never admitted again. The rest are proven and never trialled again.
//! 3. A trial that proved anything returns to the learn, which fills the
//!    ground the bans freed. One that proved nothing — no candidate left
//!    worth keeping — **locks** the library: nothing is admitted, evicted or
//!    timed again, a held shape records and replays, and every other runs
//!    eagerly.
//!
//! Every phase that ends leaves a summary ([`Transition`]) — a learn's length,
//! what it admitted and why it stopped; a trial's verdicts and why each ban
//! fell — which the hub logs at the moment of the transition, with what the
//! library then holds.
//!
//! Pure bookkeeping over keys, wave numbers and durations: every rule is tested
//! without a device.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// A segment's `(topology, shape)`: what an executable is kept for.
pub(crate) type Key = (u64, u64);

/// Waves a trial lasts. Short on purpose: a decode wave is a token or so, and
/// reaching the lock should cost a few seconds of decode, not minutes. A shape
/// hit every wave is timed 32 times in each mode, ample for a median.
pub(crate) const TRIAL_WAVES: u64 = 64;

/// Waves without an admission that end a learn — about a second of decode.
pub(crate) const LEARN_IDLE_WAVES: u64 = 32;

/// A candidate is hit in at least one trial wave in this many, or banned.
pub(crate) const MIN_HIT_DIVISOR: u64 = 8;

/// Timed runs of each mode a candidate needs for a verdict on its speed — the
/// fewest hits a candidate may have, split between the modes.
pub(crate) const MIN_SAMPLES: usize = 4;

/// How much faster a graph must run than its eager launches to be kept: its
/// median at most this fraction of the eager median. A tie is not kept — a
/// kept shape pays for its recording every wave.
pub(crate) const MIN_GAIN: f64 = 0.95;

/// How a segment runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Recorded and launched as a graph.
    Graph,
    /// Issued as launched, unrecorded.
    Eager,
}

/// Where the library is in its lifecycle — reported on the stats line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Learn,
    Trial,
    Lock,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Learn => "learn",
            Phase::Trial => "trial",
            Phase::Lock => "lock",
        }
    }
}

/// One candidate's trial so far.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Tally {
    hits: u64,
    /// Runs dispatched graphed so far — what the next run alternates against.
    /// Not the timings: those arrive a wave late, once their events have
    /// fired, so a shape hit twice in one wave would see the same counts both
    /// times and run graphed both times.
    dispatched_graph: u64,
    graph: Vec<Duration>,
    eager: Vec<Duration>,
}

/// What a trial made of one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Judged {
    /// Kept; its graphed median over its eager one.
    Proven(f64),
    /// Hit in fewer than one trial wave in [`MIN_HIT_DIVISOR`].
    Rare,
    /// Hit enough, but its timings did not arrive in time to judge — reported
    /// apart from [`Self::Rare`] because it says the trial failed to measure,
    /// not that the shape failed to recur.
    Untimed,
    /// Hit enough, not faster enough; the same ratio.
    Slower(f64),
}

impl Tally {
    /// What this candidate earned over a trial of `waves`.
    fn judge(&self, waves: u64) -> Judged {
        if self.hits * MIN_HIT_DIVISOR < waves {
            return Judged::Rare;
        }
        if self.graph.len() < MIN_SAMPLES || self.eager.len() < MIN_SAMPLES {
            return Judged::Untimed;
        }
        let eager = median(&self.eager).as_secs_f64();
        let ratio = if eager > 0.0 {
            median(&self.graph).as_secs_f64() / eager
        } else {
            1.0
        };
        if ratio <= MIN_GAIN {
            Judged::Proven(ratio)
        } else {
            Judged::Slower(ratio)
        }
    }
}

/// Why a learn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LearnEnd {
    /// An admission did not fit the budget.
    Full,
    /// [`LEARN_IDLE_WAVES`] passed with no admission.
    Quiet,
}

/// A phase that has just ended, summarised for the log.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Transition {
    Learned {
        waves: u64,
        admitted: usize,
        ended: LearnEnd,
        next: Phase,
    },
    Trialled {
        waves: u64,
        /// What the trial judged: every shape the library held unproven.
        judged: usize,
        proven: usize,
        /// The median, over the shapes it proved, of graphed time over eager.
        proven_ratio: Option<f64>,
        rare: usize,
        untimed: usize,
        slower: usize,
        next: Phase,
    },
}

/// See the module documentation.
#[derive(Debug, Default)]
pub(crate) struct Regime {
    phase: Phase,
    /// The wave the learn last admitted in, or began — `None` until the learn
    /// sees its first wave. The device's wave count does not start with the
    /// learn, so a clock started at zero would find a learn that began at wave
    /// 50 idle already, and lock it before it had recorded anything.
    last_admit: Option<u64>,
    /// The wave the current phase began at, once it has seen one.
    began: Option<u64>,
    /// The wave the trial ends at.
    trial_until: u64,
    /// Shapes the current learn has admitted — whether it has anything to
    /// trial, and what its summary reports.
    admitted: usize,
    /// What the trial judges: every shape the library held, unproven, when it
    /// began ([`Self::begin_trial`]) — not the keys the learn admitted. A learn
    /// folds a one-off shape over another executable, which then holds the
    /// one-off's shape; judged by its admitted key, that executable was never
    /// hit, and its real shape was never judged at all. On `.5` (2026-10-17)
    /// every one of 1,065 admitted keys was banned as rarely hit and 681
    /// executables were left held, untrialled.
    candidates: HashMap<Key, Tally>,
    proven: HashSet<Key>,
    banned: HashSet<Key>,
    /// Phases that have ended and not yet been logged ([`Self::take_transitions`]).
    transitions: Vec<Transition>,
}

impl Regime {
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// Whether the library admits new executables.
    pub(crate) fn admits(&self) -> bool {
        self.phase == Phase::Learn
    }

    /// Whether `key` may never be kept.
    pub(crate) fn banned(&self, key: Key) -> bool {
        self.banned.contains(&key)
    }

    /// Whether `key` was proven: a later learn must not reshape its executable
    /// into another shape.
    pub(crate) fn proven(&self, key: Key) -> bool {
        self.proven.contains(&key)
    }

    /// Shapes banned, and proven, so far.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.banned.len(), self.proven.len())
    }

    /// The learn admitted a shape in wave `wave`.
    pub(crate) fn admitted(&mut self, wave: u64) {
        if self.admits() {
            self.admitted += 1;
            self.last_admit = Some(wave);
        }
    }

    /// A trial has begun: judge every shape in `held` — what the library holds
    /// now — that is not already proven. Returns how many. With none to judge,
    /// there is nothing a trial could prove, and the library locks.
    pub(crate) fn begin_trial(&mut self, held: impl IntoIterator<Item = Key>) -> usize {
        if self.phase != Phase::Trial {
            return 0;
        }
        self.candidates = held
            .into_iter()
            .filter(|k| !self.proven.contains(k) && !self.banned.contains(k))
            .map(|k| (k, Tally::default()))
            .collect();
        if self.candidates.is_empty() {
            self.phase = Phase::Lock;
        }
        self.candidates.len()
    }

    /// An admission in wave `wave` did not fit the budget: the learn is over.
    pub(crate) fn full(&mut self, wave: u64) {
        if self.admits() {
            self.end_learn(wave, LearnEnd::Full);
        }
    }

    /// The phases that have ended since the last call, oldest first.
    pub(crate) fn take_transitions(&mut self) -> Vec<Transition> {
        std::mem::take(&mut self.transitions)
    }

    fn end_learn(&mut self, wave: u64, ended: LearnEnd) {
        let next = if self.admitted == 0 {
            Phase::Lock
        } else {
            self.trial_until = wave + TRIAL_WAVES;
            Phase::Trial
        };
        self.transitions.push(Transition::Learned {
            waves: wave.saturating_sub(self.began.unwrap_or(wave)),
            admitted: self.admitted,
            ended,
            next,
        });
        self.admitted = 0;
        self.candidates.clear();
        self.phase = next;
        self.began = Some(wave);
    }

    /// Wave `wave` begins: a learn that has gone quiet ends, and a trial that
    /// has run its course reaches its verdict. Returns the shapes it banned,
    /// for the library to drop.
    pub(crate) fn wave_begins(&mut self, wave: u64) -> Vec<Key> {
        match self.phase {
            Phase::Learn => {
                self.began.get_or_insert(wave);
                let since = *self.last_admit.get_or_insert(wave);
                if wave.saturating_sub(since) >= LEARN_IDLE_WAVES {
                    self.end_learn(wave, LearnEnd::Quiet);
                }
                Vec::new()
            }
            Phase::Trial if wave >= self.trial_until => self.verdict(wave),
            Phase::Trial | Phase::Lock => Vec::new(),
        }
    }

    fn verdict(&mut self, wave: u64) -> Vec<Key> {
        let mut banned = Vec::new();
        let mut ratios = Vec::new();
        let judged = self.candidates.len();
        let (mut rare, mut untimed, mut slower) = (0, 0, 0);
        for (key, tally) in self.candidates.drain() {
            match tally.judge(TRIAL_WAVES) {
                Judged::Proven(r) => {
                    self.proven.insert(key);
                    ratios.push(r);
                    continue;
                }
                Judged::Rare => rare += 1,
                Judged::Untimed => untimed += 1,
                Judged::Slower(_) => slower += 1,
            }
            self.banned.insert(key);
            banned.push(key);
        }
        let next = if ratios.is_empty() {
            Phase::Lock
        } else {
            self.last_admit = Some(wave);
            Phase::Learn
        };
        ratios.sort_by(f64::total_cmp);
        self.transitions.push(Transition::Trialled {
            waves: wave.saturating_sub(self.began.unwrap_or(wave)),
            judged,
            proven: ratios.len(),
            proven_ratio: ratios.get(ratios.len() / 2).copied(),
            rare,
            untimed,
            slower,
            next,
        });
        self.phase = next;
        self.began = Some(wave);
        banned
    }

    /// A segment predicted to be `key` is about to run: how. A candidate on
    /// trial alternates, starting with whichever mode it has fewer timings of,
    /// and the prediction counts as its hit; everything else runs as a graph.
    pub(crate) fn mode_for(&mut self, key: Key) -> Mode {
        if self.phase != Phase::Trial {
            return Mode::Graph;
        }
        let Some(t) = self.candidates.get_mut(&key) else {
            return Mode::Graph;
        };
        // Alternate by what has been dispatched: graphed on odd hits, eager on
        // even ones, whenever the timings arrive.
        t.hits += 1;
        if t.dispatched_graph * 2 < t.hits {
            t.dispatched_graph += 1;
            Mode::Graph
        } else {
            Mode::Eager
        }
    }

    /// Whether a segment of `key` is timed: it is a candidate on trial.
    pub(crate) fn timed(&self, key: Key) -> bool {
        self.phase == Phase::Trial && self.candidates.contains_key(&key)
    }

    /// A timed run of `key` in `mode` took `took` on the device.
    pub(crate) fn sample(&mut self, key: Key, mode: Mode, took: Duration) {
        if self.phase != Phase::Trial {
            return;
        }
        if let Some(t) = self.candidates.get_mut(&key) {
            match mode {
                Mode::Graph => t.graph.push(took),
                Mode::Eager => t.eager.push(took),
            }
        }
    }
}

/// The middle of `d` (the upper of the two middles when even); zero if empty.
fn median(d: &[Duration]) -> Duration {
    if d.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = d.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us(n: u64) -> Duration {
        Duration::from_micros(n)
    }

    const A: Key = (1, 10);
    const B: Key = (2, 20);

    /// A learn that admitted one shape for each of `held`, ended full at
    /// `wave`, and handed its trial what the library holds — `held`.
    fn learned(r: &mut Regime, wave: u64, held: &[Key]) -> usize {
        for _ in held {
            r.admitted(wave);
        }
        r.full(wave);
        r.begin_trial(held.iter().copied())
    }

    /// Run a whole trial from wave `start`: every listed key is hit every wave
    /// and its runs take `graph` or `eager`. Returns the verdict's bans.
    fn trial(r: &mut Regime, start: u64, keys: &[(Key, Duration, Duration)]) -> Vec<Key> {
        for wave in start..start + TRIAL_WAVES {
            assert!(r.wave_begins(wave).is_empty(), "no verdict before the end");
            for &(key, graph, eager) in keys {
                let mode = r.mode_for(key);
                assert!(r.timed(key));
                r.sample(key, mode, if mode == Mode::Graph { graph } else { eager });
            }
        }
        r.wave_begins(start + TRIAL_WAVES)
    }

    /// **The learn admits until the budget refuses**, and the trial then
    /// judges what the library holds.
    #[test]
    fn a_full_budget_ends_the_learn_and_starts_a_trial() {
        let mut r = Regime::default();
        assert_eq!(r.phase(), Phase::Learn);
        assert!(r.admits());
        r.admitted(3);
        r.full(4);
        assert_eq!(r.phase(), Phase::Trial);
        assert!(!r.admits(), "a trial admits nothing");
        assert_eq!(r.begin_trial([A]), 1);
        assert!(r.timed(A));
        assert!(!r.timed(B), "only what the library holds is on trial");
        assert_eq!(r.mode_for(B), Mode::Graph);
    }

    /// **The trial judges what is held, not what was admitted**: an
    /// executable a one-off reshaped holds the one-off's shape, and that is
    /// the shape judged. Proven and banned shapes are not judged again, and a
    /// trial with nothing left to judge locks at once.
    #[test]
    fn the_trial_judges_the_shapes_the_library_holds() {
        let mut r = Regime::default();
        r.admitted(1);
        r.full(2);
        let reshaped: Key = (1, 99);
        assert_eq!(r.begin_trial([reshaped]), 1);
        assert!(
            r.timed(reshaped) && !r.timed(A),
            "the held shape, not the admitted one"
        );

        let mut r = Regime::default();
        learned(&mut r, 2, &[A, B]);
        trial(&mut r, 2, &[(A, us(50), us(100)), (B, us(100), us(100))]);
        r.admitted(200);
        r.full(201);
        assert_eq!(
            r.begin_trial([A, B]),
            0,
            "A proven, B banned: nothing to judge"
        );
        assert_eq!(r.phase(), Phase::Lock);
    }

    /// A learn whose traffic has shown every shape ends on its own.
    #[test]
    fn a_quiet_learn_ends_after_its_idle_waves() {
        let mut r = Regime::default();
        r.admitted(10);
        r.wave_begins(10 + LEARN_IDLE_WAVES - 1);
        assert_eq!(r.phase(), Phase::Learn);
        r.wave_begins(10 + LEARN_IDLE_WAVES);
        assert_eq!(r.phase(), Phase::Trial);
    }

    /// **The idle clock starts at the learn's first wave**, not at wave zero:
    /// a learn beginning at wave 5,000 is not idle on arrival.
    #[test]
    fn a_learn_begun_late_is_not_idle_on_arrival() {
        let mut r = Regime::default();
        r.wave_begins(5_000);
        assert_eq!(r.phase(), Phase::Learn);
        r.wave_begins(5_000 + LEARN_IDLE_WAVES - 1);
        assert_eq!(r.phase(), Phase::Learn);
        r.wave_begins(5_000 + LEARN_IDLE_WAVES);
        assert_eq!(r.phase(), Phase::Lock, "idle with nothing admitted");
    }

    /// A learn that admitted nothing has nothing to trial: it locks.
    #[test]
    fn a_learn_that_admitted_nothing_locks() {
        let mut r = Regime::default();
        r.full(1);
        assert_eq!(r.phase(), Phase::Lock);
    }

    /// **A candidate alternates graphed and eager**, one run each, and counts
    /// a hit each time it is predicted.
    #[test]
    fn a_candidate_alternates_its_modes() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        let mut modes = Vec::new();
        for _ in 0..4 {
            let m = r.mode_for(A);
            r.sample(A, m, us(10));
            modes.push(m);
        }
        assert_eq!(modes, [Mode::Graph, Mode::Eager, Mode::Graph, Mode::Eager]);
    }

    /// **Alternation follows what was dispatched, not what was timed**: a
    /// shape hit twice before any timing arrives runs once each way.
    #[test]
    fn a_candidate_alternates_before_its_timings_arrive() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        let modes: Vec<Mode> = (0..4).map(|_| r.mode_for(A)).collect();
        assert_eq!(modes, [Mode::Graph, Mode::Eager, Mode::Graph, Mode::Eager]);
    }

    /// **Every phase that ends is summarised**: a learn by how long it ran,
    /// what it admitted and why it stopped; a trial by what it proved, at what
    /// median ratio, and what it banned and why.
    #[test]
    fn each_phase_that_ends_leaves_its_summary() {
        let mut r = Regime::default();
        r.wave_begins(100);
        r.admitted(110);
        r.admitted(120);
        r.full(130);
        assert_eq!(
            r.take_transitions(),
            vec![Transition::Learned {
                waves: 30,
                admitted: 2,
                ended: LearnEnd::Full,
                next: Phase::Trial,
            }]
        );
        const C: Key = (3, 30);
        const D: Key = (4, 40);
        r.begin_trial([A, B, C, D]);
        // C is hit every wave but never timed; D is never hit.
        for wave in 130..130 + TRIAL_WAVES {
            r.wave_begins(wave);
            for (key, graph, eager) in [(A, us(50), us(100)), (B, us(100), us(90))] {
                let mode = r.mode_for(key);
                r.sample(key, mode, if mode == Mode::Graph { graph } else { eager });
            }
            r.mode_for(C);
        }
        r.wave_begins(130 + TRIAL_WAVES);
        assert_eq!(
            r.take_transitions(),
            vec![Transition::Trialled {
                waves: TRIAL_WAVES,
                judged: 4,
                proven: 1,
                proven_ratio: Some(0.5),
                rare: 1,
                untimed: 1,
                slower: 1,
                next: Phase::Learn,
            }]
        );
        assert!(r.take_transitions().is_empty(), "each is taken once");
        let back = 130 + TRIAL_WAVES;
        r.wave_begins(back + LEARN_IDLE_WAVES);
        assert_eq!(
            r.take_transitions(),
            vec![Transition::Learned {
                waves: LEARN_IDLE_WAVES,
                admitted: 0,
                ended: LearnEnd::Quiet,
                next: Phase::Lock,
            }]
        );
    }

    /// **The verdict keeps a fast, busy graph and bans a slow one**; a trial
    /// that kept something returns to the learn.
    #[test]
    fn the_verdict_keeps_what_is_faster_and_bans_what_is_not() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A, B]);
        let bans = trial(&mut r, 2, &[(A, us(80), us(100)), (B, us(100), us(100))]);
        assert_eq!(bans, vec![B], "a tie is not kept");
        assert!(r.banned(B) && !r.banned(A));
        assert_eq!(r.counts(), (1, 1));
        assert_eq!(r.phase(), Phase::Learn, "A was proven: learn again");
        assert!(!r.timed(A), "a proven shape is never trialled again");
    }

    /// A graph faster by less than `MIN_GAIN` is not worth what it costs.
    #[test]
    fn a_graph_barely_faster_is_banned() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        assert_eq!(trial(&mut r, 2, &[(A, us(97), us(100))]), vec![A]);
    }

    /// **A rarely hit candidate is banned whatever its speed**: fewer than one
    /// trial wave in `MIN_HIT_DIVISOR`.
    #[test]
    fn a_rarely_hit_candidate_is_banned() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        let hits_needed = TRIAL_WAVES / MIN_HIT_DIVISOR;
        for i in 0..hits_needed - 1 {
            let m = r.mode_for(A);
            r.sample(A, m, if m == Mode::Graph { us(10) } else { us(100) });
            assert!(r.wave_begins(2 + i).is_empty());
        }
        assert_eq!(r.wave_begins(2 + TRIAL_WAVES), vec![A]);
    }

    /// **A trial that proves nothing locks the library for good**: nothing is
    /// admitted, timed or banned again.
    #[test]
    fn a_trial_that_proves_nothing_locks() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        trial(&mut r, 2, &[(A, us(100), us(90))]);
        assert_eq!(r.phase(), Phase::Lock);
        r.admitted(600);
        r.full(601);
        assert_eq!(r.begin_trial([B]), 0, "a locked library begins no trial");
        assert!(!r.admits() && !r.timed(B));
        assert!(r.wave_begins(1 << 40).is_empty());
        assert_eq!(r.phase(), Phase::Lock, "locked for good");
    }

    /// **Bans free ground the next learn fills**: the second trial judges only
    /// the new admissions, and one that proves nothing locks.
    #[test]
    fn learn_and_trial_repeat_until_no_candidate_is_worth_keeping() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        trial(&mut r, 2, &[(A, us(50), us(100))]);
        assert_eq!(r.phase(), Phase::Learn);
        assert!(r.proven(A));
        let next = 2 + TRIAL_WAVES + 1;
        assert_eq!(learned(&mut r, next + 1, &[A, B]), 1, "A is proven already");
        assert!(r.timed(B) && !r.timed(A));
        assert_eq!(trial(&mut r, next + 1, &[(B, us(100), us(100))]), vec![B]);
        assert_eq!(r.phase(), Phase::Lock);
        assert_eq!(r.counts(), (1, 1));
    }

    /// The median ignores a few stalled runs.
    #[test]
    fn a_few_stalled_runs_do_not_decide_the_median() {
        let mut d = vec![us(20); 16];
        d[..3].fill(us(5_000_000));
        assert_eq!(median(&d), us(20));
        assert_eq!(median(&[]), Duration::ZERO);
    }
}
