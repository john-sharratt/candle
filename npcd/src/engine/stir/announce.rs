//! The public address system — the building talking about itself out loud.
//!
//! # The one fixture that reads everything
//!
//! Every other fixture reads the one or two [`Cond`] flags it happens to care
//! about. This one reads the lot: it looks at what is true of the building,
//! picks the worst of it, and pages somebody about it. That makes it a genuine
//! interconnect rather than another list — an announcement about a coolant leak
//! only exists because the coolant loop has been leaking, and nothing here
//! mentions the coolant loop.
//!
//! It is also the second thing in the vault that can put out [`Cond::Alarmed`],
//! and therefore the second way the rat gets frightened. **The full chain:** the
//! supply strains, a breaker goes, the lights fail, the loop starts weeping,
//! two serious faults now stand at once, the PA raises an alarm, and something
//! bolts across the floor and puts a rack of plates over. Six fixtures, one
//! incident, no fixture naming another.
//!
//! # Why it repeats itself deliberately
//!
//! A PA that never says the same thing twice is a PA nobody would recognise.
//! `said_about` stops it paging about the same fault back to back, but it will
//! come round to it again if the fault is still there — which is exactly how a
//! character learns that a fault is *standing*.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

/// What the system will page about, worst first. Reading in order is the whole
/// of its judgement.
const PRESSING: &[Cond] = &[
    Cond::Alarmed,
    Cond::Leaking,
    Cond::Unstable,
    Cond::Dark,
    Cond::Vermin,
    Cond::Damp,
    Cond::Cold,
];

pub struct Announcements {
    due: Due,
    rng: Rng,
    /// The last fault it paged about, so it works down the list rather than
    /// saying the same thing every time.
    said_about: Option<Cond>,
    /// Faults standing at once. Two is what makes it raise an alarm rather than
    /// page a name.
    alarming: bool,
    /// When it last raised one, and how bad things were when it did.
    ///
    /// **An alarm is for the change, not for the state.** Without these two, a
    /// section that has gone dark and stayed dark gets an alarm every ten
    /// minutes for the rest of the day — which is both wrong about how a real
    /// base behaves and the fastest way to teach a character to ignore the one
    /// signal here that is supposed to mean something.
    last_alarm: Option<Duration>,
    alarm_level: usize,
}

impl Announcements {
    pub fn new(seed: u64) -> Announcements {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(200), Duration::from_secs(900));
        Announcements {
            due: Due::at(first),
            rng,
            said_about: None,
            alarming: false,
            last_alarm: None,
            alarm_level: 0,
        }
    }

    /// The worst thing standing that it did not just talk about.
    fn topic(&self, w: &Watch) -> Option<Cond> {
        let standing: Vec<Cond> = PRESSING.iter().copied().filter(|c| w.is(*c)).collect();
        standing
            .iter()
            .find(|c| Some(**c) != self.said_about)
            .or_else(|| standing.first())
            .copied()
    }

    fn about(&mut self, c: Cond) -> (&'static str, Salience) {
        let lines: &[&str] = match c {
            Cond::Leaking => &[
                "The address system asks for somebody from maintenance at the coolant gallery, twice.",
                "A voice over the address system reads out a coolant loop reference and asks for it \
                 to be isolated.",
            ],
            Cond::Unstable => &[
                "The address system warns that the supply is being transferred and asks for \
                 non-essential load to be shed.",
                "A recorded voice over the address system advises that power conditioning is \
                 offline until further notice.",
            ],
            Cond::Dark => &[
                "The address system advises that lighting is on reduced service in this section.",
                "A voice over the address system asks anybody working in a dark section to say so.",
            ],
            Cond::Vermin => &[
                "The address system reminds everybody that the containment log is to be signed \
                 hourly, in a tone that suggests it has not been.",
                "A voice over the address system asks for any biological contamination to be \
                 reported to the duty officer.",
            ],
            Cond::Damp | Cond::Cold => &[
                "The address system asks for an environmental reading from this section, and gets \
                 no reply anybody can hear.",
                "A voice over the address system reads out an environmental deviation and moves on \
                 to the next item.",
            ],
            Cond::Alarmed => &[
                "The alarm tone repeats over the address system, and a voice underneath it asks \
                 for the section to be cleared.",
            ],
            _ => &["The address system carries a short announcement that is not for this section."],
        };
        let line = self.rng.pick(lines).copied().unwrap_or(
            "The address system carries a short announcement that is not for this section.",
        );
        let salience = match c {
            Cond::Alarmed => Salience::URGENT,
            Cond::Leaking | Cond::Unstable => Salience::NORMAL,
            _ => Salience::IDLE,
        };
        (line, salience)
    }

    /// What it says when the building is behaving. Routine, and quiet.
    fn routine(&mut self, w: &Watch) -> Stirring {
        // The hour is real, so a shift call lands when a shift would.
        let on_the_hour = w.secs_today() % 3600 < 180;
        let line = match on_the_hour {
            true => *self
                .rng
                .pick(&[
                    "The address system calls the shift change and reads out three names.",
                    "The address system reads the hour and the outside temperature, and stops.",
                ])
                .unwrap_or(&"The address system calls the shift change and reads out three names."),
            false => *self
                .rng
                .pick(&[
                    "The address system clicks on, carries a few seconds of somebody's open \
                     microphone, and clicks off.",
                    "A test tone goes out over the address system and is not followed by anything.",
                    "The address system pages a name that nobody in this section answers to.",
                    "The address system carries a bulletin from another section, too quiet to \
                     follow.",
                ])
                .unwrap_or(&"A test tone goes out over the address system."),
        };
        Stirring::new("announce", line, Salience::IDLE)
    }
}

impl Fixture for Announcements {
    fn id(&self) -> &'static str {
        "announce"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        if self.alarming {
            out.push(Cond::Alarmed);
            out.push(Cond::Loud);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(180), Duration::from_secs(900)),
        );

        // An alarm that is up gets stood down; nothing here alarms for ever.
        if self.alarming {
            self.alarming = false;
            return Some(Stirring::new(
                "announce",
                "The alarm tone stops, and the address system says the section is clear.",
                Salience::NORMAL,
            ));
        }

        // Two serious faults standing at once is what an alarm is for.
        let serious = [Cond::Leaking, Cond::Unstable, Cond::Dark]
            .iter()
            .filter(|c| w.is(**c))
            .count();
        // A fault clearing arms the alarm again for the next one.
        self.alarm_level = self.alarm_level.min(serious);
        // Otherwise it takes something genuinely worse than last time, or an
        // hour of nobody dealing with it, before the tone goes out again.
        let worse = serious > self.alarm_level;
        let stale = match self.last_alarm {
            None => true,
            Some(t) => w.since_start.saturating_sub(t) > Duration::from_secs(3600),
        };
        if serious >= 2 && (worse || stale) {
            self.alarming = true;
            self.said_about = Some(Cond::Alarmed);
            self.last_alarm = Some(w.since_start);
            self.alarm_level = serious;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(60), Duration::from_secs(240)),
            );
            return Some(
                Stirring::new(
                    "announce",
                    "The alarm goes off over the address system, and a voice under it starts \
                     reading out a section reference.",
                    Salience::URGENT,
                )
                .tagged(&[Cond::Alarmed, Cond::Loud]),
            );
        }

        match self.topic(w) {
            Some(c) => {
                self.said_about = Some(c);
                let (line, salience) = self.about(c);
                Some(Stirring::new("announce", line, salience))
            }
            None => {
                self.said_about = None;
                Some(self.routine(w))
            }
        }
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // Something loud enough to stop a person is something the system tends
        // to have an opinion about, shortly afterwards.
        if what.salience.preempts() && what.from != "announce" {
            self.due.hold(
                w,
                self.rng
                    .between(Duration::from_secs(15), Duration::from_secs(60)),
            );
        }
    }
}
