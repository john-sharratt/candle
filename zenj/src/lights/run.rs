//! The lights thread (§9.3): every [`TICK`], what the director says the room
//! looks like is offered to the rate gate, and what the gate lets through is
//! sent; one bulb a beat-fraction is asked whether it is there. It reads the
//! deck reader's snapshot and the live view, takes the conductor's cues and
//! the tools' asks from queues, and never calls a model or waits on the
//! network: a light that is late is a light that is late, not a beat.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use effector_bridge::proto::wiz::{read, GET_PILOT};
use serde::Serialize;

use super::cue::Cue;
use super::director::{Ask, Director, Input};
use super::fixtures::Fixtures;
use super::gate::{Gate, BURST, RATE};
use super::health::{Change, Health};
use super::wire::Wire;
use crate::deck::Snapshot;
use crate::events::{Events, Kind};
use crate::live::Live;

/// How often the thread looks.
pub const TICK: Duration = Duration::from_millis(10);

/// What the lights are doing, for the tools and the screens.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct View {
    /// How many bulbs the map names.
    pub bulbs: usize,
    /// Whether ZenJ is driving them: rekordbox is open.
    pub driving: bool,
    /// The look, in a word.
    pub scene: String,
    /// Scenes a person held, by group.
    pub held: Vec<String>,
    pub house: bool,
    /// The bulbs not answering, by id.
    pub dead: Vec<String>,
}

/// Everything the thread holds.
pub struct Lights<W: Wire> {
    pub fixtures: Fixtures,
    pub wire: W,
    pub snapshot: Arc<ArcSwap<Snapshot>>,
    pub live: Arc<ArcSwap<Live>>,
    pub cues: Receiver<Cue>,
    pub asks: Receiver<Ask>,
    pub view: Arc<ArcSwap<View>>,
    pub events: Arc<Events>,
    /// Whether rekordbox runs: read here directly, not through the live
    /// view, so the bulbs are let go the moment it closes.
    pub rekordbox: Arc<AtomicBool>,
    /// How late a bulb shows what it is sent, in milliseconds: what the
    /// lights lead the beat by.
    pub lead_ms: f64,
}

/// What the director is given, from the snapshot and the live view.
pub fn input(
    rekordbox: bool,
    snap: &Snapshot,
    live: &Live,
    lead_ms: f64,
    bulbs: usize,
    dead: usize,
) -> Input {
    let air = snap.on_air();
    let beat = air.and_then(|d| d.beat.map(|b| b + lead_ms / 1000.0 * d.bpm as f64 / 60.0));
    Input {
        rekordbox,
        holder: live.holder,
        energy: 0.7 * live.target_energy + 0.3 * live.landing,
        beat,
        to_midnight: live.to_midnight_s as f64,
        bulbs,
        dead,
    }
}

impl<W: Wire> Lights<W> {
    /// One tick at `now`.
    pub fn tick(&mut self, d: &mut Director, gate: &mut Gate, health: &mut Health, now: Instant) {
        while let Ok(a) = self.asks.try_recv() {
            d.ask(a);
        }
        let snap = self.snapshot.load_full();
        let live = self.live.load_full();
        let beat = snap.on_air().and_then(|x| x.beat).unwrap_or(0.0);
        while let Ok(c) = self.cues.try_recv() {
            d.cue(c, beat);
        }
        let inp = input(
            self.rekordbox.load(Ordering::Relaxed),
            &snap,
            &live,
            self.lead_ms,
            self.fixtures.len(),
            health.dead().len(),
        );
        let fixtures = &self.fixtures;
        let look = d.look(&inp, &|g| fixtures.in_group(g));
        if let Some(look) = &look {
            // The bulbs answer while they are driven.
            let (asked, died) = health.ask(now);
            if let Some(i) = asked {
                self.wire.send(fixtures.bulbs[i].ip, GET_PILOT.as_bytes());
            }
            if let Some(Change::Died(i)) = died {
                self.events.push(
                    Kind::Fault,
                    format!("light {} stopped answering", fixtures.bulbs[i].id),
                );
            }
            while let Some((from, bytes)) = self.wire.recv() {
                let Some(i) = fixtures.bulbs.iter().position(|b| b.ip == from) else {
                    continue;
                };
                if read(&bytes).is_ok() {
                    if let Some(Change::Back(_)) = health.answered(i) {
                        gate.forget(i);
                        self.events.push(
                            Kind::Lights,
                            format!("light {} is back", fixtures.bulbs[i].id),
                        );
                    }
                }
            }
            for &(i, p) in &look.bulbs {
                if health.is_dead(i) {
                    continue;
                }
                match look.key {
                    true => {
                        if let Some(p) = gate.key(i, p, now) {
                            self.wire.send(fixtures.bulbs[i].ip, p.message().as_bytes());
                        }
                    }
                    false => gate.want(i, p),
                }
            }
            for (i, p) in gate.due(now) {
                if !self.wire.send(fixtures.bulbs[i].ip, p.message().as_bytes()) {
                    gate.forget(i);
                }
            }
        }
        let view = View {
            bulbs: fixtures.len(),
            driving: look.is_some(),
            scene: look.map(|l| l.name).unwrap_or_default(),
            held: d.held(),
            house: d.house(),
            dead: health
                .dead()
                .into_iter()
                .map(|i| fixtures.bulbs[i].id.clone())
                .collect(),
        };
        if **self.view.load() != view {
            self.view.store(Arc::new(view));
        }
    }

    /// Run until `stop`.
    pub fn run(mut self, stop: Arc<AtomicBool>) {
        let start = Instant::now();
        let mut d = Director::new();
        let mut gate = Gate::new(self.fixtures.len(), RATE, BURST, start);
        let mut health = Health::new(self.fixtures.len());
        let mut next = start;
        while !stop.load(Ordering::Relaxed) {
            self.tick(&mut d, &mut gate, &mut health, Instant::now());
            next += TICK;
            let now = Instant::now();
            if next > now {
                thread::sleep(next - now);
            } else {
                next = now;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;
    use std::sync::mpsc;

    use super::*;
    use crate::deck::state::Deck;
    use crate::lights::wire::Recorded;

    const MAP: &str = "\
bulbs:
  - {id: t1, ip: 192.168.50.11, mac: m1, pos: 1, aim: floor}
  - {id: t2, ip: 192.168.50.12, mac: m2, pos: 2, aim: wall}
";

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

    fn parts(t: Instant) -> (Director, Gate, Health) {
        (
            Director::new(),
            Gate::new(2, RATE, BURST, t),
            Health::new(2),
        )
    }

    fn count(sent: &[(Ipv4Addr, String)], what: &str) -> usize {
        sent.iter().filter(|(_, m)| m.contains(what)).count()
    }

    struct Rig {
        lights: Lights<Recorded>,
        cue: mpsc::Sender<Cue>,
        ask: mpsc::Sender<Ask>,
    }

    fn rig(rekordbox: bool) -> Rig {
        let (cue, cues) = mpsc::channel();
        let (ask, asks) = mpsc::channel();
        let live = Live {
            target_energy: 1.0,
            landing: 1.0,
            to_midnight_s: 3600,
            ..Live::default()
        };
        Rig {
            lights: Lights {
                fixtures: Fixtures::parse("lights.yaml", MAP).unwrap(),
                wire: Recorded::default(),
                snapshot: Arc::new(ArcSwap::from_pointee(playing(64.0))),
                live: Arc::new(ArcSwap::from_pointee(live)),
                cues,
                asks,
                view: Arc::new(ArcSwap::from_pointee(View::default())),
                events: Arc::new(Events::new()),
                rekordbox: Arc::new(AtomicBool::new(rekordbox)),
                lead_ms: 0.0,
            },
            cue,
            ask,
        }
    }

    /// **The beat is led by the bulbs' latency**, in the playing deck's beats.
    #[test]
    fn the_beat_is_led_by_the_bulbs_latency() {
        let i = input(true, &playing(10.0), &Live::default(), 500.0, 2, 0);
        assert_eq!(i.beat, Some(11.0), "half a second at 120 BPM is a beat");
    }

    /// **While rekordbox is closed nothing goes to a bulb**, and the view
    /// says the lights are not ZenJ's.
    #[test]
    fn while_rekordbox_is_closed_nothing_is_sent() {
        let mut r = rig(false);
        let t = Instant::now();
        let (mut d, mut g, mut h) = parts(t);
        r.lights.tick(&mut d, &mut g, &mut h, t);
        assert!(r.lights.wire.sent.is_empty());
        assert!(!r.lights.view.load().driving);
    }

    /// **Open, every bulb is sent its state once, and one is asked whether
    /// it is there**; a hit cue goes to every bulb at once.
    #[test]
    fn open_every_bulb_is_sent_its_state_and_a_hit_goes_at_once() {
        let mut r = rig(true);
        let t = Instant::now();
        let (mut d, mut g, mut h) = parts(t);
        r.lights.tick(&mut d, &mut g, &mut h, t);
        let sent = std::mem::take(&mut r.lights.wire.sent);
        assert_eq!(count(&sent, "getPilot"), 1);
        assert_eq!(count(&sent, "setPilot"), 2);
        assert_eq!(r.lights.view.load().scene, "pulse");
        r.lights.tick(&mut d, &mut g, &mut h, t);
        let again = count(&r.lights.wire.sent, "setPilot");
        assert_eq!(again, 0, "the same state is not sent again");
        r.cue.send(Cue::Hit).unwrap();
        r.lights.wire.sent.clear();
        r.lights.tick(&mut d, &mut g, &mut h, t);
        let hits = r
            .lights
            .wire
            .sent
            .iter()
            .filter(|(_, m)| m.contains(r#""r":255,"g":255,"b":255,"dimming":100"#))
            .count();
        assert_eq!(hits, 2, "both bulbs, at once");
    }

    /// **The house lights asked for go to every bulb**, and the view says so.
    #[test]
    fn the_house_lights_go_to_every_bulb() {
        let mut r = rig(true);
        let t = Instant::now();
        let (mut d, mut g, mut h) = parts(t);
        r.ask.send(Ask::House(true)).unwrap();
        r.lights.tick(&mut d, &mut g, &mut h, t);
        let warm = r
            .lights
            .wire
            .sent
            .iter()
            .filter(|(_, m)| m.contains(r#""temp":2700,"dimming":80"#))
            .count();
        assert_eq!(warm, 2);
        assert!(r.lights.view.load().house);
    }
}
