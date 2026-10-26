//! Which executables the library keeps: learned, trialled, then locked.
//!
//! A graph launch replaces a segment's launches with one, but keeping a
//! segment's topology has costs of its own: it is recorded every wave (that is
//! how its arguments are refreshed), the record gate holds other threads' quiet
//! calls while it records, and its executables are driver memory outside the
//! reservation. A topology is worth those only if it recurs and its graph is
//! faster than its launches issued one by one. So the library finds out,
//! topology by topology ([`Key`]), and then stops changing:
//!
//! 1. **Learn.** Segments record, and the library admits what it folds, into
//!    its fixed budget. Nothing is evicted to make room: the learn ends at the
//!    first admission that does not fit, or once [`LEARN_IDLE_WAVES`] pass
//!    with no admission at all — the traffic has shown its topologies.
//! 2. **Trial**, for at least [`TRIAL_WAVES`] waves and until every candidate
//!    has [`MIN_SAMPLES`] timed in each mode, or [`TRIAL_MAX_WAVES`]. Every
//!    topology the library holds unproven alternates graphed and eager each
//!    time a segment predicts it, and each run is timed on the device from the
//!    segment's start to its end — host lateness included, which is what a
//!    graph removes. One whose graphed median is not at least [`MIN_GAIN`]
//!    faster than its eager one is **banned**: dropped, and never admitted
//!    again. One hit fewer than [`MIN_HITS`] times, or never timed, is
//!    **benched** until the lock reopens. The rest are proven and never
//!    trialled again.
//! 3. A trial that proved anything returns to the learn, which fills the
//!    ground the bans freed. One that proved nothing — no candidate left
//!    worth keeping — **locks** the library: nothing is admitted or timed, a
//!    held shape records and replays, a shape it does not hold runs once, and
//!    every other runs eagerly.
//! 4. A lock with room **reopens** when traffic keeps bringing topologies it
//!    does not hold — [`REOPEN_SHAPES`] of them met [`REOPEN_SIGHTINGS`] times each,
//!    after it has stood [`MIN_LOCK_WAVES`], with the library under half its
//!    budget. A learn's traffic is whatever ran while it learned, and one that
//!    learns at startup sees startup's; reopened, it learns the traffic that
//!    came since, keeping everything already proven and banned.
//!
//! Every phase that ends leaves a summary ([`Transition`]) — a learn's length,
//! what it admitted and why it stopped; a trial's verdicts and why each ban
//! fell — which the hub logs at the moment of the transition, with what the
//! library then holds.
//!
//! Pure bookkeeping over keys, wave numbers and durations: every rule is tested
//! without a device.

use std::collections::{HashMap, HashSet};
use std::mem::take;
use std::time::Duration;

/// A segment's topology: what the library judges and keeps.
///
/// **Not its shape.** Judged by `(topology, shape)`, each combination of
/// sessions and verify widths was a candidate of its own, hit too rarely to
/// prove. A held topology's shapes fold in place; any other runs once (`slot`).
pub(crate) type Key = u64;

/// Waves a trial lasts at least. A trial whose every candidate has its
/// [`MIN_SAMPLES`] in both modes by then ends here: a topology hit every wave
/// is timed 32 times in each mode, ample for a median.
pub(crate) const TRIAL_WAVES: u64 = 64;

/// Waves a trial lasts at most, waiting for its candidates' samples.
///
/// **A trial waits for samples, not for a share of all waves.** A decode step
/// opens several waves of its own kinds (`ordinals`), and its segments come at
/// a few widths, so a shape every step uses is in a small fraction of the
/// waves: on `.5` (2026-10-17), with timing working, the most-hit shape of a
/// 64-wave trial had 7 hits, the rest 3 or fewer, and the trial banned 174 of
/// 176 as rare while each recurred steadily. Measured by the samples it needs
/// instead, a trial lasts until every candidate is judged or this cap, and a
/// topology hit fewer than [`MIN_HITS`] times by then is rare by any measure
/// worth a graph.
///
/// **The cap is sized by a serving daemon's mix, not a benchmark's.** A
/// benchmark's topologies recur every wave and its trial ends at
/// [`TRIAL_WAVES`]; the cap only binds where conversations, ingest and their
/// drafts share the waves. At 512 (~45 s of that mix on `.5`, 2026-10-26) a
/// trial of 119 candidates proved 13 and benched 71 as rarely hit, and 5% of
/// segments ran as graphs; at 4,096, 36–56% did.
pub(crate) const TRIAL_MAX_WAVES: u64 = 4096;

/// Waves without an admission that end a learn — about a second of decode.
pub(crate) const LEARN_IDLE_WAVES: u64 = 32;

/// Timed runs of each mode a candidate needs for a verdict on its speed.
pub(crate) const MIN_SAMPLES: usize = 4;

/// Hits a candidate needs over its trial, or it is banned as rare: enough to
/// time [`MIN_SAMPLES`] in each mode.
pub(crate) const MIN_HITS: u64 = 2 * MIN_SAMPLES as u64;

/// How much faster a graph must run than its eager launches to be kept: its
/// median at most this fraction of the eager median. A tie is not kept — a
/// kept topology pays for its recording every wave.
pub(crate) const MIN_GAIN: f64 = 0.95;

/// Times a lock must meet a topology it does not hold before that topology
/// counts as recurring — a prompt's own length is met once.
///
/// **Doubled for every bench a topology has already served.** A reopen takes
/// back what the trials benched, so a topology that keeps recurring at a rate
/// too low to judge was otherwise benched, reopened for and trialled again on
/// every cycle: on `.5` (2026-10-17) one run logged 49 reopens, each a learn
/// and a full trial with its eager halves. A topology benched `n`
/// times counts as recurring only once met `REOPEN_SIGHTINGS << n` times
/// ([`Regime::sightings_needed`]), and only then does a reopen take it back.
pub(crate) const REOPEN_SIGHTINGS: u32 = 3;

/// Benches past which [`REOPEN_SIGHTINGS`] stops doubling — so the threshold
/// stays reachable (3 << 6 = 192 sightings) and the shift cannot overflow.
pub(crate) const MAX_BENCH_DOUBLINGS: u32 = 6;

/// Recurring topologies the lock does not hold that reopen it.
pub(crate) const REOPEN_SHAPES: usize = 4;

/// Waves a lock stands before anything may reopen it.
pub(crate) const MIN_LOCK_WAVES: u64 = 256;

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
    /// fired, so a topology hit twice in one wave would see the same counts
    /// both times and run graphed both times.
    dispatched_graph: u64,
    graph: Vec<Duration>,
    eager: Vec<Duration>,
}

/// What a trial made of one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Judged {
    /// Kept; its graphed median over its eager one.
    Proven(f64),
    /// Hit fewer than [`MIN_HITS`] times.
    Rare,
    /// Hit enough, but its timings did not arrive in time to judge — reported
    /// apart from [`Self::Rare`] because it says the trial failed to measure,
    /// not that the topology failed to recur.
    Untimed,
    /// Hit enough, not faster enough; the same ratio.
    Slower(f64),
}

impl Tally {
    /// Whether its speed can be judged: [`MIN_SAMPLES`] timed in each mode.
    fn sampled(&self) -> bool {
        self.graph.len() >= MIN_SAMPLES && self.eager.len() >= MIN_SAMPLES
    }

    /// What this candidate earned over its trial.
    fn judge(&self) -> Judged {
        if self.hits < MIN_HITS {
            return Judged::Rare;
        }
        if !self.sampled() {
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

/// A trial's segment starts, by what each was predicted to be.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Starts {
    /// A topology on trial: a hit.
    pub(crate) candidate: u64,
    /// A topology known but not on trial — proven, banned, benched, or not
    /// held.
    pub(crate) other: u64,
    /// An ordinal no wave of its kind had recorded.
    pub(crate) unpredicted: u64,
    /// Recorded segments that were a topology on trial and had been predicted
    /// as another — hits the prediction lost.
    pub(crate) missed: u64,
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
        /// What the trial judged: every topology the library held unproven.
        judged: usize,
        proven: usize,
        /// The median, over the topologies it proved, of graphed time over
        /// eager.
        proven_ratio: Option<f64>,
        rare: usize,
        untimed: usize,
        slower: usize,
        next: Phase,
        /// What the trial's segment starts were predicted to be.
        starts: Starts,
        /// The judged topologies' hits, fewest first, as `(hits, graphed
        /// samples, eager samples)` — what the verdicts were drawn from.
        tallies: Vec<(u64, usize, usize)>,
    },
    /// A lock gave way to a learn: traffic kept bringing topologies it does
    /// not hold, and its budget had room for them.
    Reopened {
        waves: u64,
        /// Topologies met unheld at least their own sightings
        /// ([`Regime::sightings_needed`]).
        recurring: usize,
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
    /// Executables the current learn has admitted — whether it has anything to
    /// trial, and what its summary reports.
    admitted: usize,
    /// What the trial judges: every topology the library held, unproven, when
    /// it began ([`Self::begin_trial`]).
    candidates: HashMap<Key, Tally>,
    /// The current trial's segment starts ([`Self::start`]).
    starts: Starts,
    proven: HashSet<Key>,
    /// Topologies a trial measured and found not faster: never kept again.
    banned: HashSet<Key>,
    /// Topologies a trial could not judge — too rarely hit, or never timed —
    /// dropped and not admitted again until the lock reopens.
    ///
    /// **Benched, not banned.** Whether a segment recurs is a property of the
    /// traffic a trial happened to see — a trial during an ingest prefill
    /// misses every segment a conversation's decode uses — while whether its
    /// graph is faster is a property of the segment. Banned for good, every
    /// shape a trial's traffic missed ran eagerly for the life of the process:
    /// on `.5` (2026-10-17) 894 bans held a lock to graphing one segment in
    /// eight. Benched, it waits out the current cycle, so the learn that
    /// follows a trial does not admit it straight back, and the lock's
    /// rechecks count it like any topology it does not hold — traffic that
    /// keeps bringing it reopens the learn, which may keep it.
    benched: HashSet<Key>,
    /// How many times each topology has been benched, ever — what scales the
    /// sightings it needs to reopen the lock ([`REOPEN_SIGHTINGS`]).
    benches: HashMap<Key, u32>,
    /// Topologies the lock has met and does not hold, by how often — what
    /// reopens it ([`Self::reopen_due`]).
    unheld: HashMap<Key, u32>,
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

    /// Whether `key` may not be kept now: banned, or benched until the lock
    /// reopens.
    pub(crate) fn banned(&self, key: Key) -> bool {
        self.banned.contains(&key) || self.benched.contains(&key)
    }

    /// Whether `key` was proven: kept, and never trialled again — and its
    /// executables never reshaped for a one-off (`slot`).
    pub(crate) fn proven(&self, key: Key) -> bool {
        self.proven.contains(&key)
    }

    /// Topologies banned, and proven, so far.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.banned.len(), self.proven.len())
    }

    /// The learn admitted an executable in wave `wave`.
    pub(crate) fn admitted(&mut self, wave: u64) {
        if self.admits() {
            self.admitted += 1;
            self.last_admit = Some(wave);
        }
    }

    /// A trial has begun: judge every topology in `held` — what the library
    /// holds now — that is not already proven, banned or benched. Returns how
    /// many. With none to judge,
    /// there is nothing a trial could prove, and the library locks.
    pub(crate) fn begin_trial(&mut self, held: impl IntoIterator<Item = Key>) -> usize {
        if self.phase != Phase::Trial {
            return 0;
        }
        self.candidates = held
            .into_iter()
            .filter(|&k| !self.proven.contains(&k) && !self.banned(k))
            .map(|k| (k, Tally::default()))
            .collect();
        self.starts = Starts::default();
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

    /// The lock met `key`, a topology at a shape it does not hold, on a segment
    /// it recorded to see what the wave puts there. A banned topology does not
    /// count: its graph was judged and lost. A benched one does: it was never
    /// judged.
    pub(crate) fn missed(&mut self, key: Key) {
        if self.phase == Phase::Lock && !self.banned.contains(&key) {
            *self.unheld.entry(key).or_default() += 1;
        }
    }

    /// Whether the lock should give way to a learn at wave `wave`: it has stood
    /// [`MIN_LOCK_WAVES`], and traffic keeps bringing [`REOPEN_SHAPES`]
    /// topologies or more that it does not hold. Whether the budget has room
    /// for them is the library's to say.
    ///
    /// A lock trusts the traffic its learn saw, and a learn begun at startup
    /// sees startup's: on `.5` (2026-10-17) one went quiet in 185 waves of
    /// background ingest and locked at 185 MiB of a 2 GiB budget, to run every
    /// segment the conversations brought later as issued, for good.
    pub(crate) fn reopen_due(&self, wave: u64) -> bool {
        self.phase == Phase::Lock
            && wave.saturating_sub(self.began.unwrap_or(wave)) >= MIN_LOCK_WAVES
            && self.recurring() >= REOPEN_SHAPES
    }

    /// The sightings `key` needs to count as recurring: [`REOPEN_SIGHTINGS`],
    /// doubled for each bench it has served.
    pub(crate) fn sightings_needed(&self, key: Key) -> u32 {
        let benches = self.benches.get(&key).copied().unwrap_or(0);
        REOPEN_SIGHTINGS << benches.min(MAX_BENCH_DOUBLINGS)
    }

    fn recurs(&self, key: Key, sightings: u32) -> bool {
        sightings >= self.sightings_needed(key)
    }

    fn recurring(&self) -> usize {
        self.unheld
            .iter()
            .filter(|&(&k, &n)| self.recurs(k, n))
            .count()
    }

    /// Reopen the lock at wave `wave`: a learn begins, keeping what the trials
    /// proved and banned, and taking back each benched topology that met its
    /// own sightings ([`Self::sightings_needed`]); the rest stay benched.
    pub(crate) fn reopen(&mut self, wave: u64) {
        if self.phase != Phase::Lock {
            return;
        }
        self.transitions.push(Transition::Reopened {
            waves: wave.saturating_sub(self.began.unwrap_or(wave)),
            recurring: self.recurring(),
        });
        let back: Vec<Key> = self
            .benched
            .iter()
            .copied()
            .filter(|&k| self.unheld.get(&k).is_some_and(|&n| self.recurs(k, n)))
            .collect();
        for k in back {
            self.benched.remove(&k);
        }
        self.unheld.clear();
        self.admitted = 0;
        self.last_admit = Some(wave);
        self.phase = Phase::Learn;
        self.began = Some(wave);
    }

    fn end_learn(&mut self, wave: u64, ended: LearnEnd) {
        let next = if self.admitted == 0 {
            Phase::Lock
        } else {
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
    /// has run its course reaches its verdict. Returns the topologies it did
    /// not prove — banned or benched — for the library to drop.
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
            Phase::Trial if self.trial_over(wave) => self.verdict(wave),
            Phase::Trial | Phase::Lock => Vec::new(),
        }
    }

    /// Whether the trial reaches its verdict at wave `wave`: it has run
    /// [`TRIAL_WAVES`] and every candidate is [sampled](Tally::sampled), or it
    /// has run [`TRIAL_MAX_WAVES`].
    fn trial_over(&self, wave: u64) -> bool {
        let ran = wave.saturating_sub(self.began.unwrap_or(wave));
        ran >= TRIAL_MAX_WAVES
            || (ran >= TRIAL_WAVES && self.candidates.values().all(Tally::sampled))
    }

    /// Bench `key` until a reopen takes it back, counting the bench.
    fn bench(&mut self, key: Key) {
        self.benched.insert(key);
        *self.benches.entry(key).or_default() += 1;
    }

    fn verdict(&mut self, wave: u64) -> Vec<Key> {
        let mut dropped = Vec::new();
        let mut ratios = Vec::new();
        let judged = self.candidates.len();
        let (mut rare, mut untimed, mut slower) = (0, 0, 0);
        let mut tallies: Vec<(u64, usize, usize)> = self
            .candidates
            .values()
            .map(|t| (t.hits, t.graph.len(), t.eager.len()))
            .collect();
        tallies.sort_unstable();
        for (key, tally) in take(&mut self.candidates) {
            match tally.judge() {
                Judged::Proven(r) => {
                    self.proven.insert(key);
                    ratios.push(r);
                    continue;
                }
                Judged::Rare => {
                    rare += 1;
                    self.bench(key);
                }
                Judged::Untimed => {
                    untimed += 1;
                    self.bench(key);
                }
                Judged::Slower(_) => {
                    slower += 1;
                    self.banned.insert(key);
                }
            }
            dropped.push(key);
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
            starts: take(&mut self.starts),
            tallies,
        });
        self.phase = next;
        self.began = Some(wave);
        dropped
    }

    /// Whether a segment predicted to be topology `predicted` runs as issued on
    /// the prediction alone, unrecorded — `held` saying whether the library
    /// holds the predicted shape of it:
    ///
    /// - a banned or benched topology, always: its graph was judged and lost,
    ///   or waits out the cycle — recording it would buy one launch of an
    ///   executable nothing keeps, at every segment it fills;
    /// - in the learn, never otherwise: everything records, to be admitted;
    /// - in the lock, a shape the library does not hold, or an ordinal never
    ///   recorded: recording it would buy one launch of an executable nothing
    ///   keeps, instantiated for that launch alone;
    /// - **in the trial, never otherwise.** Unrecorded, an ordinal's
    ///   prediction never changes: one predicted as a shape the library does
    ///   not hold — a width it did not admit, a one-off — ran as issued until
    ///   its periodic recheck, which outlasts the trial, so the held topologies
    ///   that ordinal really runs collected no hits and were benched as rare.
    ///   Recorded, it is replayed once if it is not held and its prediction
    ///   follows what the wave put there.
    pub(crate) fn runs_unkept(&self, predicted: Option<Key>, held: bool) -> bool {
        match (self.phase, predicted) {
            (_, Some(k)) if self.banned(k) => true,
            (Phase::Learn | Phase::Trial, _) => false,
            (Phase::Lock, None) => true,
            (Phase::Lock, Some(_)) => !held,
        }
    }

    /// A segment predicted to be `predicted` is starting: counted into the
    /// trial's [`Starts`], whatever mode it then runs in.
    pub(crate) fn start(&mut self, predicted: Option<Key>) {
        if self.phase != Phase::Trial {
            return;
        }
        match predicted {
            None => self.starts.unpredicted += 1,
            Some(k) if self.candidates.contains_key(&k) => self.starts.candidate += 1,
            Some(_) => self.starts.other += 1,
        }
    }

    /// A segment predicted as `predicted` recorded as `actual`: a topology on
    /// trial predicted as another counts as a hit lost ([`Starts::missed`]).
    pub(crate) fn recorded(&mut self, predicted: Option<Key>, actual: Key) {
        if self.phase == Phase::Trial
            && predicted != Some(actual)
            && self.candidates.contains_key(&actual)
        {
            self.starts.missed += 1;
        }
    }

    /// A segment predicted to be `key` is about to run: how. A candidate on
    /// trial alternates by what it has been dispatched — graphed on its odd
    /// hits, eager on its even ones — and the prediction counts as its hit;
    /// everything else runs as a graph.
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

    const A: Key = 1;
    const B: Key = 2;

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

    /// **The trial judges the topologies the library holds**, whatever the
    /// learn counted as admissions. Proven and banned topologies are not judged
    /// again, and a trial with nothing left to judge locks at once.
    #[test]
    fn the_trial_judges_the_topologies_the_library_holds() {
        let mut r = Regime::default();
        r.admitted(1);
        r.full(2);
        let held: Key = 9;
        assert_eq!(r.begin_trial([held]), 1);
        assert!(
            r.timed(held) && !r.timed(A),
            "what is held, not what was counted"
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

    /// **Only a banned shape, or the lock, runs a segment unrecorded on its
    /// prediction.** A trial records an ordinal predicted as a shape the
    /// library does not hold, so its prediction follows the wave; the lock
    /// runs it as issued; the learn records everything but a banned shape.
    #[test]
    fn a_trial_records_what_it_does_not_hold() {
        let mut r = Regime::default();
        assert!(!r.runs_unkept(None, false), "the learn records the unseen");
        assert!(!r.runs_unkept(Some(A), false));
        learned(&mut r, 2, &[A, B]);
        trial(&mut r, 2, &[(A, us(50), us(100)), (B, us(100), us(100))]);
        assert!(r.banned(B));
        r.admitted(200);
        r.full(201);
        let unheld: Key = 3;
        assert_eq!(r.begin_trial([A, unheld]), 1);
        assert_eq!(r.phase(), Phase::Trial);
        assert!(!r.runs_unkept(None, false), "a trial records the unseen");
        assert!(
            !r.runs_unkept(Some(4), false),
            "a trial records an ordinal predicted as a shape it does not hold"
        );
        assert!(!r.runs_unkept(Some(A), true));
        assert!(r.runs_unkept(Some(B), true), "banned runs as issued");

        let mut locked = Regime::default();
        learned(&mut locked, 2, &[A]);
        trial(&mut locked, 2, &[(A, us(100), us(100))]);
        assert_eq!(locked.phase(), Phase::Lock);
        assert!(locked.runs_unkept(None, false));
        assert!(locked.runs_unkept(Some(4), false));
        assert!(locked.runs_unkept(Some(A), true), "A was banned as slower");
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
        const C: Key = 3;
        const D: Key = 4;
        r.begin_trial([A, B, C, D]);
        // C is hit every wave but never timed; D is never hit. Neither is
        // sampled, so the trial runs to its cap; the traffic stops at
        // `TRIAL_WAVES`.
        for wave in 130..130 + TRIAL_WAVES {
            r.wave_begins(wave);
            for (key, graph, eager) in [(A, us(50), us(100)), (B, us(100), us(90))] {
                let mode = r.mode_for(key);
                r.sample(key, mode, if mode == Mode::Graph { graph } else { eager });
            }
            r.mode_for(C);
            for start in [Some(A), Some(B), Some(C), None, Some(9)] {
                r.start(start);
            }
            r.recorded(Some(A), A);
            r.recorded(Some(B), A);
            r.recorded(None, 9);
        }
        for wave in 130 + TRIAL_WAVES..130 + TRIAL_MAX_WAVES {
            r.wave_begins(wave);
        }
        assert!(r.take_transitions().is_empty(), "held open to the cap");
        r.wave_begins(130 + TRIAL_MAX_WAVES);
        assert_eq!(
            r.take_transitions(),
            vec![Transition::Trialled {
                waves: TRIAL_MAX_WAVES,
                judged: 4,
                proven: 1,
                proven_ratio: Some(0.5),
                rare: 1,
                untimed: 1,
                slower: 1,
                next: Phase::Learn,
                starts: Starts {
                    candidate: 3 * TRIAL_WAVES,
                    other: TRIAL_WAVES,
                    unpredicted: TRIAL_WAVES,
                    missed: TRIAL_WAVES,
                },
                tallies: vec![
                    (0, 0, 0),
                    (TRIAL_WAVES, 0, 0),
                    (TRIAL_WAVES, 32, 32),
                    (TRIAL_WAVES, 32, 32),
                ],
            }]
        );
        assert!(r.take_transitions().is_empty(), "each is taken once");
        let back = 130 + TRIAL_MAX_WAVES;
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

    /// **A rarely hit candidate is banned whatever its speed**: fewer than
    /// `MIN_HITS` hits by the time the trial runs out. Unsampled, it holds the
    /// trial open until `TRIAL_MAX_WAVES`.
    #[test]
    fn a_rarely_hit_candidate_is_banned() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        for i in 0..MIN_HITS - 1 {
            let m = r.mode_for(A);
            r.sample(A, m, if m == Mode::Graph { us(10) } else { us(100) });
            assert!(r.wave_begins(2 + i).is_empty());
        }
        assert!(
            r.wave_begins(2 + TRIAL_WAVES).is_empty(),
            "unsampled, it holds the trial open"
        );
        assert_eq!(r.wave_begins(2 + TRIAL_MAX_WAVES), vec![A], "dropped");
        assert_eq!(r.phase(), Phase::Lock);
        assert!(r.banned(A), "benched: not kept this cycle");
        assert_eq!(r.counts(), (0, 0), "but not banned for good");
        assert!(r.runs_unkept(Some(A), false));
    }

    /// A trial over `held` from wave `start` in which only `B` is hit — every
    /// wave, at a tie, so it is banned — and every other candidate is benched as
    /// rare. Returns the wave the verdict landed at.
    fn bench_all_but_b(r: &mut Regime, start: u64, held: &[Key]) -> u64 {
        learned(r, start, held);
        for wave in start..start + TRIAL_MAX_WAVES {
            r.wave_begins(wave);
            if r.timed(B) {
                let m = r.mode_for(B);
                r.sample(B, m, us(100));
            }
        }
        r.wave_begins(start + TRIAL_MAX_WAVES);
        start + TRIAL_MAX_WAVES
    }

    /// **A benched topology counts toward a reopen once met its own
    /// sightings, and the reopen takes it back**: the lock meets it like any
    /// topology it does not hold, and the learn it reopens to may admit it
    /// again. A banned topology stays banned.
    #[test]
    fn a_reopen_takes_back_what_a_trial_benched() {
        let mut r = Regime::default();
        let locked_at = bench_all_but_b(&mut r, 2, &[A, B]);
        assert_eq!(r.phase(), Phase::Lock);
        assert!(r.banned(A) && r.banned(B));
        assert_eq!(
            r.sightings_needed(A),
            REOPEN_SIGHTINGS << 1,
            "benched once: doubled"
        );
        let others: Vec<Key> = (10..10 + REOPEN_SHAPES as u64 - 1).collect();
        for _ in 0..r.sightings_needed(A) {
            r.missed(A);
            r.missed(B);
            for &k in &others {
                r.missed(k);
            }
        }
        assert!(
            r.reopen_due(locked_at + MIN_LOCK_WAVES),
            "A counts as recurring, B does not"
        );
        r.reopen(locked_at + MIN_LOCK_WAVES);
        assert!(!r.banned(A), "the reopen takes A back");
        assert!(r.banned(B), "B stays banned");
    }

    /// **Each bench doubles what a topology must be met to come back**, and a
    /// reopen it did not earn leaves it benched. Without that, a topology too
    /// rare to judge was benched, reopened for and trialled again every cycle.
    #[test]
    fn a_topology_benched_again_needs_twice_the_sightings() {
        let mut r = Regime::default();
        const C: Key = 3;
        let locked_at = bench_all_but_b(&mut r, 2, &[A, B, C]);
        assert_eq!(r.sightings_needed(A), REOPEN_SIGHTINGS << 1);
        // Reopen on others' sightings; A met fewer than it needs stays benched.
        let others: Vec<Key> = (10..10 + REOPEN_SHAPES as u64).collect();
        for _ in 0..REOPEN_SIGHTINGS {
            for &k in &others {
                r.missed(k);
            }
            r.missed(A);
        }
        assert!(r.reopen_due(locked_at + MIN_LOCK_WAVES));
        r.reopen(locked_at + MIN_LOCK_WAVES);
        assert!(r.banned(A), "A met 3 of the 6 it needs: still benched");
        assert!(r.banned(C), "C never met: still benched");
        assert!(r.admits());
        // A benched topology is never admitted, so it is never judged again
        // until a reopen takes it back.
        r.admitted(locked_at + MIN_LOCK_WAVES + 1);
        r.full(locked_at + MIN_LOCK_WAVES + 1);
        assert_eq!(r.begin_trial([A, C]), 0, "benched: not judged");
        assert_eq!(
            REOPEN_SIGHTINGS << MAX_BENCH_DOUBLINGS,
            192,
            "the cap keeps the threshold reachable"
        );
    }

    /// **A trial waits for its candidates' samples**: a shape hit once every
    /// sixteen waves has too few samples at `TRIAL_WAVES`, and the trial runs on
    /// until it has them — then proves it, well before the cap.
    #[test]
    fn a_trial_waits_for_a_sparse_shape() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A]);
        let mut ended = None;
        for wave in 2..2 + TRIAL_MAX_WAVES {
            if !r.wave_begins(wave).is_empty() || r.phase() != Phase::Trial {
                ended = Some(wave);
                break;
            }
            if (wave - 2) % 16 == 0 {
                let m = r.mode_for(A);
                r.sample(A, m, if m == Mode::Graph { us(50) } else { us(100) });
            }
        }
        let ended = ended.expect("the trial ended before its cap");
        assert!(ended > 2 + TRIAL_WAVES, "it outlasted the minimum: {ended}");
        assert!(r.proven(A), "sparse but steady, and faster");
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

    /// **A lock reopens once traffic keeps bringing shapes it does not hold**:
    /// not before it has stood `MIN_LOCK_WAVES`, not for shapes met once, not
    /// for banned ones — and the learn it reopens to keeps what was proven.
    #[test]
    fn a_lock_reopens_for_recurring_shapes_it_does_not_hold() {
        let mut r = Regime::default();
        learned(&mut r, 2, &[A, B]);
        trial(&mut r, 2, &[(A, us(50), us(100)), (B, us(100), us(100))]);
        r.wave_begins(2 + TRIAL_WAVES + LEARN_IDLE_WAVES);
        assert_eq!(r.phase(), Phase::Lock);
        let locked_at = 2 + TRIAL_WAVES + LEARN_IDLE_WAVES;
        r.take_transitions();

        let new: Vec<Key> = (10..10 + REOPEN_SHAPES as u64).collect();
        // Each met once, and the banned B met often: not recurring.
        for &k in &new {
            r.missed(k);
        }
        for _ in 0..10 {
            r.missed(B);
        }
        assert!(!r.reopen_due(locked_at + MIN_LOCK_WAVES));
        // Met enough times — but the lock has not stood long enough yet.
        for _ in 1..REOPEN_SIGHTINGS {
            for &k in &new {
                r.missed(k);
            }
        }
        assert!(!r.reopen_due(locked_at + MIN_LOCK_WAVES - 1));
        assert!(r.reopen_due(locked_at + MIN_LOCK_WAVES));

        r.reopen(locked_at + MIN_LOCK_WAVES);
        assert_eq!(
            r.take_transitions(),
            vec![Transition::Reopened {
                waves: MIN_LOCK_WAVES,
                recurring: REOPEN_SHAPES,
            }]
        );
        assert!(r.admits(), "learning again");
        assert!(r.proven(A) && r.banned(B), "the verdicts stand");
        assert!(!r.reopen_due(1 << 40), "a learn does not reopen");
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
