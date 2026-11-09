//! The lights' two threads (§9.3). The **planner** keeps two bars of the
//! lights planned ahead on the beat clock ([`super::timeline`]) and plans
//! them again whenever what they were made from changes: the clock jumps, a
//! cue is armed, a person asks for a scene, the decks change hands, the
//! energy moves, a bulb dies or comes back. The **sender** only plays the
//! timeline — each send at its instant, through the rate gate — and asks the
//! bulbs whether they are there, timing each answer for that bulb's lead.
//! Neither calls a model or waits on the network: a stall in the planner is
//! covered by the plan already made, and a light that is late is a light that
//! is late, not a beat.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use effector_bridge::proto::wiz::{read, GET_PILOT};
use serde::Serialize;

use super::cue::Cues;
use super::director::{Ask, Director, Input};
use super::fixtures::Fixtures;
use super::gate::{Gate, BURST, RATE};
use super::health::{Change, Health};
use super::lead::Leads;
use super::timeline::{now_only, plan, Basis, Planned, HORIZON_BEATS};
use super::wire::Wire;
use crate::beat::{reading, BeatClock, Read};
use crate::deck::Snapshot;
use crate::events::{Events, Kind};
use crate::hands::Holder;
use crate::live::Live;

/// How often the planner looks.
pub const PLAN_TICK: Duration = Duration::from_millis(10);
/// How often the sender looks: a send goes within this of its instant.
pub const SEND_TICK: Duration = Duration::from_millis(1);
/// The energy, in steps of which a change is a new plan.
const ENERGY_STEP: f32 = 0.05;
/// How far behind now a topped-up timeline keeps what it held: more than the
/// sender's tick, so a send it has not reached yet is not lost.
const KEEP_BEHIND: Duration = Duration::from_millis(50);

/// What the lights are doing, for the tools and the screens.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct View {
    /// How many bulbs the map names.
    pub bulbs: usize,
    /// Whether ZenJ is driving them: rekordbox is open.
    pub driving: bool,
    /// The look now, in a word.
    pub scene: String,
    /// Scenes a person held, by group.
    pub held: Vec<String>,
    pub house: bool,
    /// The bulbs not answering, by id.
    pub dead: Vec<String>,
    /// How far ahead the lights are planned, in beats.
    pub ahead: f64,
}

/// The timeline the sender plays: replaced whole by the planner.
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    /// Bumped with every plan made again from scratch, so the sender knows to
    /// play it from now rather than after what it last sent.
    pub made: u64,
    pub sends: Vec<Planned>,
}

/// What the two threads share.
#[derive(Clone)]
pub struct Shared {
    pub timeline: Arc<ArcSwap<Timeline>>,
    pub leads: Arc<Mutex<Leads>>,
    pub dead: Arc<ArcSwap<Vec<usize>>>,
    /// Bumped by the sender when a bulb dies or comes back.
    pub health_changed: Arc<AtomicU64>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Everything the lights hold.
pub struct Lights<W: Wire> {
    pub fixtures: Fixtures,
    pub wire: W,
    pub snapshot: Arc<ArcSwap<Snapshot>>,
    pub live: Arc<ArcSwap<Live>>,
    pub cues: Receiver<Cues>,
    pub asks: Receiver<Ask>,
    pub view: Arc<ArcSwap<View>>,
    pub events: Arc<Events>,
    /// Whether rekordbox runs: read here directly, not through the live
    /// view, so the bulbs are let go the moment it closes.
    pub rekordbox: Arc<AtomicBool>,
    /// The bulb's own lag past the network, in milliseconds.
    pub lead_ms: f64,
}

/// What a plan was made from, so a change of it is a new plan.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Made {
    rekordbox: bool,
    holder: Holder,
    energy_step: i32,
    health: u64,
}

/// The planner's state between ticks.
pub struct Planner {
    pub director: Director,
    pub clock: BeatClock,
    made: Option<Made>,
    /// The beat planned through.
    through: Option<f64>,
    generation: u64,
}

impl Planner {
    pub fn new() -> Planner {
        Planner {
            director: Director::new(),
            clock: BeatClock::new(),
            made: None,
            through: None,
            generation: 0,
        }
    }

    /// One planning tick at `now`: the timeline topped up or made again.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        now: Instant,
        snap: &Snapshot,
        live: &Live,
        rekordbox: bool,
        fixtures: &Fixtures,
        shared: &Shared,
        asked: bool,
    ) -> View {
        let jumped = self.clock.read(now, reading(snap)) == Read::Jump;
        let dead = shared.dead.load_full();
        let energy = 0.7 * live.target_energy + 0.3 * live.landing;
        let made = Made {
            rekordbox,
            holder: live.holder,
            energy_step: (energy / ENERGY_STEP).round() as i32,
            health: shared.health_changed.load(Ordering::Relaxed),
        };
        let input = Input {
            rekordbox,
            holder: live.holder,
            energy,
            beat: self.clock.beat_at(now),
            to_midnight: live.to_midnight_s as f64,
            bulbs: fixtures.len(),
            dead: dead.len(),
        };
        let leads = lock(&shared.leads).clone();
        let lead = |i: usize| leads.of(i);
        let groups = |g: Option<&str>| fixtures.in_group(g);
        let basis = Basis {
            director: &self.director,
            now_input: input,
            clock: &self.clock,
            now,
            lead: &lead,
            groups: &groups,
        };
        let afresh = jumped || asked || self.made != Some(made);
        let now_beat = self.clock.beat_at(now);
        match now_beat {
            None => {
                // No beat to plan on: what shows now, whenever it changes.
                if afresh || self.through.is_some() {
                    self.generation += 1;
                    shared.timeline.store(Arc::new(Timeline {
                        made: self.generation,
                        sends: now_only(&basis),
                    }));
                }
                self.through = None;
            }
            Some(b) => {
                let ends = b + HORIZON_BEATS;
                match (afresh, self.through) {
                    (false, Some(through)) if through >= b + HORIZON_BEATS / 2.0 => {}
                    (false, Some(through)) => {
                        // Topped up: the next stretch after what is planned.
                        let more = plan(&basis, through + f64::EPSILON, ends);
                        // What the sender may not have reached yet is kept;
                        // it never plays a send twice.
                        let old = shared.timeline.load_full();
                        let keep = now.checked_sub(KEEP_BEHIND).unwrap_or(now);
                        let mut sends: Vec<Planned> =
                            old.sends.iter().copied().filter(|s| s.at >= keep).collect();
                        sends.extend(more);
                        shared.timeline.store(Arc::new(Timeline {
                            made: old.made,
                            sends,
                        }));
                        self.through = Some(ends);
                    }
                    _ => {
                        self.generation += 1;
                        shared.timeline.store(Arc::new(Timeline {
                            made: self.generation,
                            sends: plan(&basis, b, ends),
                        }));
                        self.through = Some(ends);
                    }
                }
            }
        }
        self.made = Some(made);
        let look = self.director.look(&input, &groups);
        View {
            bulbs: fixtures.len(),
            driving: look.is_some(),
            scene: look.map(|l| l.name).unwrap_or_default(),
            held: self.director.held(),
            house: self.director.house(),
            dead: dead.iter().map(|&i| fixtures.bulbs[i].id.clone()).collect(),
            ahead: match (self.through, now_beat) {
                (Some(t), Some(b)) => (t - b).max(0.0),
                _ => 0.0,
            },
        }
    }
}

impl Default for Planner {
    fn default() -> Planner {
        Planner::new()
    }
}

/// The sender's state between ticks.
pub struct Sender {
    pub gate: Gate,
    pub health: Health,
    /// The timeline it plays, and the instant it has played it through —
    /// `None` for a timeline not yet begun, played from its start.
    made: u64,
    played: Option<Instant>,
}

impl Sender {
    pub fn new(bulbs: usize, now: Instant) -> Sender {
        Sender {
            gate: Gate::new(bulbs, RATE, BURST, now),
            health: Health::new(bulbs),
            made: 0,
            played: None,
        }
    }

    /// One sending tick at `now`, while the lights are ZenJ's.
    pub fn tick<W: Wire>(
        &mut self,
        now: Instant,
        wire: &mut W,
        fixtures: &Fixtures,
        shared: &Shared,
        events: &Events,
    ) {
        let timeline = shared.timeline.load_full();
        if timeline.made != self.made {
            // A plan made again: what the old one sent stands, and the new
            // one is played from its start, which is the planner's now.
            self.made = timeline.made;
            self.played = None;
        }
        let played = self.played;
        for s in timeline
            .sends
            .iter()
            .filter(|s| s.at <= now && played.is_none_or(|p| s.at > p))
        {
            if self.health.is_dead(s.bulb) {
                continue;
            }
            match s.key {
                true => {
                    if let Some(p) = self.gate.key(s.bulb, s.pilot, now) {
                        wire.send(fixtures.bulbs[s.bulb].ip, p.message().as_bytes());
                    }
                }
                false => self.gate.want(s.bulb, s.pilot),
            }
        }
        self.played = Some(now);
        for (i, p) in self.gate.due(now) {
            if !wire.send(fixtures.bulbs[i].ip, p.message().as_bytes()) {
                self.gate.forget(i);
            }
        }
        // The bulbs, asked and timed.
        let (asked, died) = self.health.ask(now);
        if let Some(i) = asked {
            wire.send(fixtures.bulbs[i].ip, GET_PILOT.as_bytes());
        }
        let mut changed = false;
        if let Some(Change::Died(i)) = died {
            events.push(
                Kind::Fault,
                format!("light {} stopped answering", fixtures.bulbs[i].id),
            );
            changed = true;
        }
        while let Some((from, bytes)) = wire.recv() {
            let Some(i) = fixtures.bulbs.iter().position(|b| b.ip == from) else {
                continue;
            };
            if read(&bytes).is_err() {
                continue;
            }
            let (rtt, change) = self.health.answered(i, now);
            if let Some(rtt) = rtt {
                lock(&shared.leads).measured(i, rtt);
            }
            if let Some(Change::Back(_)) = change {
                self.gate.forget(i);
                events.push(
                    Kind::Lights,
                    format!("light {} is back", fixtures.bulbs[i].id),
                );
                changed = true;
            }
        }
        if changed {
            shared.dead.store(Arc::new(self.health.dead()));
            shared.health_changed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl<W: Wire + Send + 'static> Lights<W> {
    /// Run both threads until `stop`.
    pub fn run(self, stop: Arc<AtomicBool>) {
        let n = self.fixtures.len();
        let shared = Shared {
            timeline: Arc::new(ArcSwap::from_pointee(Timeline::default())),
            leads: Arc::new(Mutex::new(Leads::new(
                n,
                Duration::from_secs_f64(self.lead_ms.max(0.0) / 1000.0),
            ))),
            dead: Arc::new(ArcSwap::from_pointee(Vec::new())),
            health_changed: Arc::new(AtomicU64::new(0)),
        };
        let Lights {
            fixtures,
            mut wire,
            snapshot,
            live,
            cues,
            asks,
            view,
            events,
            rekordbox,
            ..
        } = self;
        {
            let (fixtures, shared, rekordbox, stop) = (
                fixtures.clone(),
                shared.clone(),
                rekordbox.clone(),
                stop.clone(),
            );
            let spawned = thread::Builder::new()
                .name("zenj-lights-send".into())
                .spawn(move || {
                    let mut s = Sender::new(fixtures.len(), Instant::now());
                    let mut next = Instant::now();
                    while !stop.load(Ordering::Relaxed) {
                        if rekordbox.load(Ordering::Relaxed) {
                            s.tick(Instant::now(), &mut wire, &fixtures, &shared, &events);
                        }
                        next += SEND_TICK;
                        let now = Instant::now();
                        match next > now {
                            true => thread::sleep(next - now),
                            false => next = now,
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::error!("lights: the sender did not start: {e}");
                return;
            }
        }
        let mut p = Planner::new();
        let mut next = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            let mut asked = false;
            while let Ok(a) = asks.try_recv() {
                p.director.ask(a);
                asked = true;
            }
            while let Ok(c) = cues.try_recv() {
                p.director.cues(match c {
                    Cues::Armed(list) => list,
                    Cues::Clear => Vec::new(),
                });
                asked = true;
            }
            let v = p.tick(
                Instant::now(),
                &snapshot.load_full(),
                &live.load_full(),
                rekordbox.load(Ordering::Relaxed),
                &fixtures,
                &shared,
                asked,
            );
            if **view.load() != v {
                view.store(Arc::new(v));
            }
            next += PLAN_TICK;
            let now = Instant::now();
            match next > now {
                true => thread::sleep(next - now),
                false => next = now,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::*;
    use crate::deck::state::Deck;
    use crate::lights::cue::Cue;
    use crate::lights::palette::HIT;
    use crate::lights::wire::Recorded;

    const MAP: &str = "\
bulbs:
  - {id: t1, ip: 192.168.50.11, mac: m1, pos: 1, aim: floor}
  - {id: t2, ip: 192.168.50.12, mac: m2, pos: 2, aim: wall}
";

    fn fixtures() -> Fixtures {
        Fixtures::parse("lights.yaml", MAP).unwrap()
    }

    fn playing(beat: f64) -> Snapshot {
        Snapshot {
            decks: vec![Deck {
                number: 1,
                track: None,
                bpm: 120.0,
                position_ms: 0.0,
                beat: Some(beat),
                playing: true,
            }],
            master: Some(1),
            blind: None,
        }
    }

    fn live() -> Live {
        Live {
            target_energy: 1.0,
            landing: 1.0,
            to_midnight_s: 3600,
            ..Live::default()
        }
    }

    fn shared() -> Shared {
        Shared {
            timeline: Arc::new(ArcSwap::from_pointee(Timeline::default())),
            leads: Arc::new(Mutex::new(Leads::new(2, Duration::from_millis(30)))),
            dead: Arc::new(ArcSwap::from_pointee(Vec::new())),
            health_changed: Arc::new(AtomicU64::new(0)),
        }
    }

    fn count(sent: &[(Ipv4Addr, String)], what: &str) -> usize {
        sent.iter().filter(|(_, m)| m.contains(what)).count()
    }

    /// **The planner keeps two bars planned, and only tops them up while
    /// nothing changes**; a jump of the clock plans afresh.
    #[test]
    fn the_planner_keeps_two_bars_ahead() {
        let t = Instant::now();
        let (f, sh) = (fixtures(), shared());
        let mut p = Planner::new();
        let v = p.tick(t, &playing(64.0), &live(), true, &f, &sh, false);
        assert_eq!(v.ahead, HORIZON_BEATS);
        assert_eq!(v.scene, "pulse");
        let first = sh.timeline.load_full();
        assert!(!first.sends.is_empty());
        // A beat on, in step: nothing to do yet.
        let one = Duration::from_millis(500);
        p.tick(t + one, &playing(65.0), &live(), true, &f, &sh, false);
        assert_eq!(sh.timeline.load_full().made, first.made, "same plan");
        // Five beats on, past half the horizon: topped up, not made again.
        let five = Duration::from_millis(2500);
        let v = p.tick(t + five, &playing(69.0), &live(), true, &f, &sh, false);
        let topped = sh.timeline.load_full();
        assert_eq!(topped.made, first.made);
        assert_eq!(v.ahead, HORIZON_BEATS);
        // A seek: made again.
        p.tick(t + five, &playing(200.0), &live(), true, &f, &sh, false);
        assert_ne!(sh.timeline.load_full().made, first.made);
    }

    /// **The sender plays each send at its instant**, the hit of an armed cue
    /// to both bulbs at once — planned bars before its beat.
    #[test]
    fn the_sender_plays_the_plan_at_its_instants() {
        let t = Instant::now();
        let (f, sh) = (fixtures(), shared());
        let mut p = Planner::new();
        p.director.cues(vec![(66.0, Cue::Hit)]);
        p.tick(t, &playing(64.0), &live(), true, &f, &sh, true);
        let mut s = Sender::new(2, t);
        let mut w = Recorded::default();
        let events = Events::new();
        let hit_at = p.clock.time_of(66.0).unwrap() - Duration::from_millis(30);
        s.tick(hit_at - Duration::from_millis(1), &mut w, &f, &sh, &events);
        let hit = HIT.message();
        assert_eq!(count(&w.sent, &hit), 0, "not before its instant");
        s.tick(hit_at, &mut w, &f, &sh, &events);
        assert_eq!(count(&w.sent, &hit), 2, "both bulbs, at once");
        assert_eq!(count(&w.sent, "getPilot"), 1, "and one bulb asked");
    }

    /// **A bulb's answer leads it by its round trip.**
    #[test]
    fn an_answer_times_its_bulb() {
        let t = Instant::now();
        let (f, sh) = (fixtures(), shared());
        let mut s = Sender::new(2, t);
        let mut w = Recorded::default();
        let events = Events::new();
        s.tick(t, &mut w, &f, &sh, &events);
        w.answers.push_back((
            "192.168.50.11".parse().unwrap(),
            br#"{"result":{"state":true}}"#.to_vec(),
        ));
        s.tick(t + Duration::from_millis(40), &mut w, &f, &sh, &events);
        assert_eq!(
            lock(&sh.leads).of(0),
            Duration::from_millis(50),
            "30 of its own, 20 of the network"
        );
    }

    /// **With rekordbox closed nothing is planned**, and the view says the
    /// lights are not ZenJ's.
    #[test]
    fn with_rekordbox_closed_nothing_is_planned() {
        let t = Instant::now();
        let (f, sh) = (fixtures(), shared());
        let mut p = Planner::new();
        let v = p.tick(t, &playing(64.0), &live(), false, &f, &sh, false);
        assert!(!v.driving);
        assert!(sh.timeline.load_full().sends.is_empty());
    }
}
