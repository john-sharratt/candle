//! Grouped phases on a streaming card (`docs/continuous_fair_waves.md` §3.2).
//!
//! A card that holds a small share of a MoE model's experts copies every
//! non-resident expert a forward routes, once per layer, however many rows ride
//! it. Prompts that arrive one at a time and creep beside the decodes pay that
//! copy for a few dozen rows at a time: on WhiteWorld (RTX 3090, Flash-Next,
//! 2026-10-30) 23% of the experts were resident, character turns appended
//! 54–98 rows each, expert copies were 50–60% of every forward, and decode ran
//! at ~10 t/s beside them. A layer's distinct experts saturate near 233 rows
//! (10 of 512 routed per row), so one sweep of a few turns together copies a
//! third of what the same turns copy apart.
//!
//! So below [`ENTER_RESIDENCY`] the engine alternates. While a decode runs it
//! is in [`Phase::Decode`]: prompts are held, and the decodes step without new
//! rows beside them. Once [`PREFILL_ENTER_ROWS`] of prompt rows wait, or the
//! phase has held one for [`MAX_WAIT`], it is in [`Phase::Prefill`]: admission runs
//! as it always does, the held prompts form one cohort, and the phase lasts
//! until that cohort has swept ([`PREFILL_EXIT_ROWS`]) or a sweep's measured
//! time has passed. Running slots keep running in either phase; only what is
//! admitted changes. Nothing below `Urgent` is exempt, and nothing `Urgent` is
//! held or counted.
//!
//! **Every phase ends.** A decode phase ends on its prompts' age, a prefill
//! phase on its slice; neither waits for the other kind of work to stop
//! arriving. The pause `docs/continuous_fair_waves.md` §3.1 records — a lower
//! band held out while any higher band had work, which never switched back
//! beside characters that tick every few seconds — had no such bound.
//!
//! Pure: a reading in, a phase out. What a reading counts is
//! `super::grouping_ground`'s.

use std::time::{Duration, Instant};

/// Resident share of the experts below which the engine groups.
///
/// Where expert copies stop being a small part of a forward on the cards this
/// engine runs on: above it the creep's copies per row stay within a few times
/// its compute, and alternating would only cost the prompts their wait.
pub(super) const ENTER_RESIDENCY: f64 = 0.60;

/// Resident share above which a grouping engine stops. Above
/// [`ENTER_RESIDENCY`] by enough that a zone moving with the K/V beside it — a
/// prefill phase buys its tier from the experts — does not flip the mode.
pub(super) const LEAVE_RESIDENCY: f64 = 0.70;

/// Prompt rows waiting that start a prefill phase: past the width at which a
/// layer routes 99% of its experts (`512 × (1 − (1 − 10/512)^233)`), so the
/// group pays each copy once for as many rows as a copy can serve.
pub(super) const PREFILL_ENTER_ROWS: usize = 256;

/// Rows still to sweep below which a prefill phase has done its work — two
/// chunks, the tail of a cohort rather than a cohort.
pub(super) const PREFILL_EXIT_ROWS: usize = 64;

/// The longest a prompt waits for its group: a decode phase holding a prompt
/// this long gives way to a prefill phase however few rows wait.
pub(super) const MAX_WAIT: Duration = Duration::from_secs(4);

/// The longest a prefill phase lasts when no sweep has been measured, and the
/// most one lasts when one has.
pub(super) const MAX_PREFILL_SLICE: Duration = Duration::from_secs(6);

/// The least a decode phase lasts before rows alone may end it, so a prefill
/// phase that ran out its slice with a burst still waiting does not come
/// straight back. A prompt that has waited [`MAX_WAIT`] still ends it.
pub(super) const MIN_DECODE_DWELL: Duration = Duration::from_secs(1);

/// Time constant of the residency's average, in seconds: long enough that one
/// cohort's tier purchase is a dip and not a reading.
const RESIDENCY_TAU_SECS: f64 = 10.0;

/// What the engine admits now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    /// Not grouping: the card holds enough of the experts, or reports none.
    Off,
    /// Grouping, with nothing decoding to hold prompts for: admission as ever.
    Mixed,
    /// Prompts below `Urgent` are held while decodes step.
    Decode,
    /// Admission as ever, so the prompts held form one cohort.
    Prefill,
}

impl Phase {
    /// The phase's name, as the logs spell it.
    pub(super) fn name(self) -> &'static str {
        match self {
            Phase::Off => "off",
            Phase::Mixed => "mixed",
            Phase::Decode => "decode",
            Phase::Prefill => "prefill",
        }
    }
}

/// What the engine stands at, counted below `Urgent`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Reading {
    /// Resident share of the model's experts; `None` for a card that reports
    /// no expert cache.
    pub(super) resident: Option<f64>,
    /// Prompt rows waiting for admission: queued turns, pending submissions
    /// and sections.
    pub(super) waiting_rows: usize,
    /// How long the oldest of them has waited; zero with none waiting.
    pub(super) oldest_wait: Duration,
    /// Rows admitted and still to sweep.
    pub(super) sweeping_rows: usize,
    /// Decodes running.
    pub(super) decodes: usize,
    /// How long a cohort's sweep has measured to take, when one has.
    pub(super) sweep: Option<Duration>,
}

/// A change of phase, for the line that reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Transition {
    pub(super) from: Phase,
    pub(super) to: Phase,
    pub(super) why: &'static str,
    /// How long the engine stood in `from`.
    pub(super) lasted: Duration,
    /// The averaged residency the change was read at.
    pub(super) resident: Option<f64>,
}

/// See the module documentation.
#[derive(Debug)]
pub(super) struct Grouping {
    phase: Phase,
    since: Option<Instant>,
    resident: Option<f64>,
    sampled: Option<Instant>,
}

impl Default for Grouping {
    fn default() -> Self {
        Self {
            phase: Phase::Off,
            since: None,
            resident: None,
            sampled: None,
        }
    }
}

impl Grouping {
    /// A grouping standing in `phase` since now, at `resident`.
    #[cfg(test)]
    pub(super) fn standing(phase: Phase, resident: f64) -> Self {
        let now = Instant::now();
        Self {
            phase,
            since: Some(now),
            resident: Some(resident),
            sampled: Some(now),
        }
    }

    /// The phase the engine is in.
    pub(super) fn phase(&self) -> Phase {
        self.phase
    }

    /// The averaged residency, once one has been read.
    pub(super) fn resident(&self) -> Option<f64> {
        self.resident
    }

    /// Fold `r` in at `now` and move to the phase it calls for, answering the
    /// change when there is one.
    pub(super) fn step(&mut self, now: Instant, r: &Reading) -> Option<Transition> {
        self.average(now, r.resident);
        let lasted = self
            .since
            .map_or(Duration::ZERO, |s| now.saturating_duration_since(s));
        let next = match (self.phase, self.resident) {
            (Phase::Off, None) => None,
            (_, None) => Some((Phase::Off, "the card reports no expert residency")),
            (Phase::Off, Some(res)) => {
                (res < ENTER_RESIDENCY).then_some((Phase::Mixed, "residency fell below the line"))
            }
            (_, Some(res)) if res > LEAVE_RESIDENCY => {
                Some((Phase::Off, "residency rose above the line"))
            }
            (Phase::Mixed, _) => (r.decodes > 0).then_some((Phase::Decode, "a decode is running")),
            (Phase::Decode, _) => decode_ends(r, lasted),
            (Phase::Prefill, _) => prefill_ends(r, lasted),
        };
        let (to, why) = next?;
        let from = self.phase;
        self.phase = to;
        self.since = Some(now);
        Some(Transition {
            from,
            to,
            why,
            lasted,
            resident: self.resident,
        })
    }

    /// Fold one residency reading into the average, weighted by the time since
    /// the last.
    fn average(&mut self, now: Instant, resident: Option<f64>) {
        let Some(read) = resident.filter(|v| v.is_finite()) else {
            self.resident = None;
            self.sampled = None;
            return;
        };
        let read = read.clamp(0.0, 1.0);
        self.resident = Some(match (self.resident, self.sampled) {
            (Some(was), Some(then)) => {
                let dt = now.saturating_duration_since(then).as_secs_f64();
                let w = 1.0 - (-dt / RESIDENCY_TAU_SECS).exp();
                was + w * (read - was)
            }
            _ => read,
        });
        self.sampled = Some(now);
    }
}

/// Where a decode phase that has lasted `lasted` goes on `r`, if anywhere.
fn decode_ends(r: &Reading, lasted: Duration) -> Option<(Phase, &'static str)> {
    if r.decodes == 0 {
        return Some((Phase::Mixed, "no decode is running"));
    }
    if r.waiting_rows == 0 {
        return None;
    }
    // Held by this phase, not queued in all: a prompt the rate model kept out
    // before the phase began has not waited on the phase, and counted at its
    // whole age it would end every decode phase on its first step.
    if r.oldest_wait.min(lasted) >= MAX_WAIT {
        return Some((Phase::Prefill, "a prompt waited its limit"));
    }
    (r.waiting_rows >= PREFILL_ENTER_ROWS && lasted >= MIN_DECODE_DWELL)
        .then_some((Phase::Prefill, "a group's rows are waiting"))
}

/// Where a prefill phase that has lasted `lasted` goes on `r`, if anywhere.
fn prefill_ends(r: &Reading, lasted: Duration) -> Option<(Phase, &'static str)> {
    let after = if r.decodes > 0 {
        Phase::Decode
    } else {
        Phase::Mixed
    };
    if r.sweeping_rows + r.waiting_rows < PREFILL_EXIT_ROWS {
        return Some((after, "the group has swept"));
    }
    (lasted >= prefill_slice(r.sweep)).then_some((after, "the prefill slice ended"))
}

/// How long a prefill phase lasts: a sweep's measured time, within
/// [`MAX_PREFILL_SLICE`].
fn prefill_slice(sweep: Option<Duration>) -> Duration {
    sweep.map_or(MAX_PREFILL_SLICE, |s| s.min(MAX_PREFILL_SLICE))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(t0: Instant, secs: f64) -> Instant {
        t0 + Duration::from_secs_f64(secs)
    }

    /// WhiteWorld's card: a fifth of the experts resident, one decode running.
    fn streaming() -> Reading {
        Reading {
            resident: Some(0.23),
            decodes: 1,
            ..Reading::default()
        }
    }

    /// Steps `g` once and answers the phase it is in after.
    fn phase_after(g: &mut Grouping, now: Instant, r: Reading) -> Phase {
        g.step(now, &r);
        g.phase()
    }

    /// A card holding most of the experts never groups; one holding few does,
    /// and holds prompts as soon as a decode runs.
    #[test]
    fn residency_decides_whether_the_card_groups() {
        let t0 = Instant::now();
        let mut held = Grouping::default();
        let full = Reading {
            resident: Some(1.0),
            decodes: 3,
            ..Reading::default()
        };
        assert_eq!(phase_after(&mut held, t0, full), Phase::Off);
        let mut small = Grouping::default();
        assert_eq!(phase_after(&mut small, t0, streaming()), Phase::Mixed);
        assert_eq!(
            phase_after(&mut small, at(t0, 0.1), streaming()),
            Phase::Decode
        );
    }

    /// A card that reports no expert cache is never grouped, and one that
    /// stops reporting leaves grouping.
    #[test]
    fn no_residency_reading_is_no_grouping() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        let dense = Reading {
            resident: None,
            decodes: 4,
            waiting_rows: 1_000,
            ..Reading::default()
        };
        assert_eq!(g.step(t0, &dense), None);
        assert_eq!(g.phase(), Phase::Off);
        g.step(t0, &streaming());
        g.step(t0, &streaming());
        assert_eq!(g.phase(), Phase::Decode);
        let t = g.step(at(t0, 1.0), &dense).expect("it leaves");
        assert_eq!((t.from, t.to), (Phase::Decode, Phase::Off));
    }

    /// The residency is averaged in time, so a dip as a cohort buys its tier
    /// does not end grouping, and the line is crossed back only well above it.
    #[test]
    fn the_residency_line_has_hysteresis_and_an_average() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        let read = |r: f64| Reading {
            resident: Some(r),
            ..Reading::default()
        };
        assert_eq!(phase_after(&mut g, t0, read(0.55)), Phase::Mixed);
        // Between the lines: still grouping.
        assert_eq!(phase_after(&mut g, at(t0, 60.0), read(0.65)), Phase::Mixed);
        // A one-second spike to everything resident moves the average a tenth
        // of the way, from where sixty seconds took it toward 0.65.
        assert_eq!(phase_after(&mut g, at(t0, 61.0), read(1.0)), Phase::Mixed);
        let was = 0.55 + (0.65 - 0.55) * (1.0 - (-6.0f64).exp());
        let expected = was + (1.0 - was) * (1.0 - (-0.1f64).exp());
        let avg = g.resident().unwrap();
        assert!(
            (avg - expected).abs() < 1e-12,
            "average {avg} against {expected}"
        );
        assert!(avg < LEAVE_RESIDENCY);
        // Held there long enough, it leaves.
        assert_eq!(phase_after(&mut g, at(t0, 121.0), read(1.0)), Phase::Off);
        // And comes back only below the lower line.
        assert_eq!(phase_after(&mut g, at(t0, 300.0), read(0.65)), Phase::Off);
        assert_eq!(phase_after(&mut g, at(t0, 400.0), read(0.50)), Phase::Mixed);
    }

    /// A decode phase holds prompts until a group's rows wait — after its
    /// dwell — and then gives way to a prefill phase.
    #[test]
    fn a_decode_phase_ends_on_a_groups_rows() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        g.step(t0, &streaming());
        g.step(t0, &streaming());
        assert_eq!(g.phase(), Phase::Decode);
        let rows = |n: usize, waited: f64| Reading {
            waiting_rows: n,
            oldest_wait: Duration::from_secs_f64(waited),
            ..streaming()
        };
        assert_eq!(
            phase_after(&mut g, at(t0, 0.5), rows(300, 0.5)),
            Phase::Decode,
            "inside the dwell"
        );
        assert_eq!(
            phase_after(&mut g, at(t0, 0.9), rows(255, 0.9)),
            Phase::Decode,
            "one row short"
        );
        let t = g
            .step(at(t0, 1.2), &rows(256, 1.2))
            .expect("the group's rows");
        assert_eq!(
            (t.from, t.to, t.why),
            (Phase::Decode, Phase::Prefill, "a group's rows are waiting")
        );
        assert_eq!(t.lasted, Duration::from_secs_f64(1.2));
    }

    /// However few rows wait, a decode phase holds a prompt no longer than
    /// [`MAX_WAIT`] — counted from when the phase began holding it, so a
    /// prompt that had queued for other reasons before the phase does not end
    /// it on its first step.
    #[test]
    fn a_prompt_waits_no_longer_than_its_limit() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        g.step(t0, &streaming());
        g.step(t0, &streaming());
        let one = |waited: f64| Reading {
            waiting_rows: 60,
            oldest_wait: Duration::from_secs_f64(waited),
            ..streaming()
        };
        assert_eq!(
            phase_after(&mut g, at(t0, 0.5), one(30.0)),
            Phase::Decode,
            "queued long before the phase, held by it half a second"
        );
        assert_eq!(
            phase_after(&mut g, at(t0, 4.0), one(33.5)),
            Phase::Prefill,
            "held by the phase its limit"
        );
        let mut h = Grouping::default();
        h.step(t0, &streaming());
        h.step(t0, &streaming());
        assert_eq!(phase_after(&mut h, at(t0, 3.9), one(3.9)), Phase::Decode);
        assert_eq!(phase_after(&mut h, at(t0, 4.0), one(4.0)), Phase::Prefill);
    }

    /// With nothing waiting a decode phase simply continues, however long it
    /// has run: there is no prompt to hold, so there is nothing to bound.
    #[test]
    fn a_decode_phase_with_nothing_waiting_continues() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        g.step(t0, &streaming());
        g.step(t0, &streaming());
        assert_eq!(
            phase_after(&mut g, at(t0, 600.0), streaming()),
            Phase::Decode
        );
    }

    /// A prefill phase ends when its group has swept, or when a sweep's
    /// measured time has passed — capped at [`MAX_PREFILL_SLICE`], and that cap
    /// with none measured — and goes back to decode while one runs.
    #[test]
    fn a_prefill_phase_ends_swept_or_on_its_slice() {
        let t0 = Instant::now();
        // Into a decode phase a limit before t0, and out of it at t0.
        let held_from = t0 - MAX_WAIT;
        let into_prefill = |g: &mut Grouping| {
            g.step(held_from, &streaming());
            g.step(held_from, &streaming());
            g.step(
                t0,
                &Reading {
                    waiting_rows: 60,
                    oldest_wait: MAX_WAIT,
                    ..streaming()
                },
            );
            assert_eq!(g.phase(), Phase::Prefill);
        };
        let sweeping = |rows: usize, sweep: Option<f64>| Reading {
            sweeping_rows: rows,
            sweep: sweep.map(Duration::from_secs_f64),
            ..streaming()
        };
        let mut swept = Grouping::default();
        into_prefill(&mut swept);
        assert_eq!(
            phase_after(&mut swept, at(t0, 0.5), sweeping(64, None)),
            Phase::Prefill
        );
        let t = swept.step(at(t0, 0.6), &sweeping(63, None)).expect("swept");
        assert_eq!((t.to, t.why), (Phase::Decode, "the group has swept"));

        let mut measured = Grouping::default();
        into_prefill(&mut measured);
        assert_eq!(
            phase_after(&mut measured, at(t0, 2.9), sweeping(500, Some(3.0))),
            Phase::Prefill
        );
        let t = measured
            .step(at(t0, 3.0), &sweeping(500, Some(3.0)))
            .expect("slice");
        assert_eq!((t.to, t.why), (Phase::Decode, "the prefill slice ended"));

        let mut long = Grouping::default();
        into_prefill(&mut long);
        assert_eq!(
            phase_after(&mut long, at(t0, 5.9), sweeping(500, Some(120.0))),
            Phase::Prefill
        );
        assert_eq!(
            phase_after(&mut long, at(t0, 6.0), sweeping(500, Some(120.0))),
            Phase::Decode
        );

        let mut unmeasured = Grouping::default();
        into_prefill(&mut unmeasured);
        assert_eq!(
            phase_after(&mut unmeasured, at(t0, 6.0), sweeping(500, None)),
            Phase::Decode
        );
    }

    /// With no decode left a prefill phase ends to mixed, and a decode phase
    /// whose decodes finish goes to mixed too.
    #[test]
    fn with_no_decode_the_engine_is_mixed() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        g.step(t0, &streaming());
        g.step(t0, &streaming());
        let idle = Reading {
            resident: Some(0.23),
            ..Reading::default()
        };
        let t = g.step(at(t0, 1.0), &idle).expect("decodes finished");
        assert_eq!((t.from, t.to), (Phase::Decode, Phase::Mixed));
        let mut p = Grouping::default();
        p.step(t0, &streaming());
        p.step(t0, &streaming());
        p.step(
            t0 + MAX_WAIT,
            &Reading {
                waiting_rows: 60,
                oldest_wait: MAX_WAIT,
                ..streaming()
            },
        );
        assert_eq!(p.phase(), Phase::Prefill);
        assert_eq!(phase_after(&mut p, at(t0, 4.5), idle), Phase::Mixed);
    }

    /// **Every phase ends.** Driven by a load that never stops arriving —
    /// prompts every step, decodes always running — the engine alternates,
    /// and no phase outlasts its bound.
    #[test]
    fn every_phase_ends_under_a_load_that_never_stops() {
        let t0 = Instant::now();
        let mut g = Grouping::default();
        let mut since_wait = 0.0f64;
        let mut longest = [Duration::ZERO; 2];
        for k in 0..2_000 {
            let now = at(t0, 0.05 * k as f64);
            let r = Reading {
                waiting_rows: 60,
                oldest_wait: Duration::from_secs_f64(since_wait),
                sweeping_rows: 120,
                ..streaming()
            };
            if let Some(t) = g.step(now, &r) {
                let i = match t.from {
                    Phase::Decode => 0,
                    Phase::Prefill => 1,
                    _ => continue,
                };
                longest[i] = longest[i].max(t.lasted);
            }
            // Held prompts age in a decode phase; a prefill phase admits them.
            since_wait = match g.phase() {
                Phase::Decode => since_wait + 0.05,
                _ => 0.0,
            };
        }
        assert!(
            longest[0] <= MAX_WAIT + Duration::from_millis(50),
            "decode {:?}",
            longest[0]
        );
        assert!(
            longest[1] <= MAX_PREFILL_SLICE + Duration::from_millis(50),
            "prefill {:?}",
            longest[1]
        );
        assert!(
            longest[0] > Duration::ZERO && longest[1] > Duration::ZERO,
            "it alternated"
        );
    }
}
