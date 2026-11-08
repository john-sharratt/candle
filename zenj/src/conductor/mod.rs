//! The conductor (§2, §6): the per-beat loop. It runs the armed plan through
//! the virtual controller, loads tracks and confirms them through the deck
//! reader, fires samples on the beat, hands the mixer back without a jump,
//! and every bar asks the watchdog whether Automix must take over. It never
//! calls a model and never waits on the network: it reads the reader's latest
//! snapshot, the hands' touches and the brain's commands, each from a queue.
//!
//! | Module | |
//! |---|---|
//! | [`runner`] | one plan against the beat |
//! | [`loader`] | a load, confirmed |
//! | [`watchdog`] | when Automix must take over |

pub mod loader;
pub mod runner;
pub mod watchdog;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use rtrb::Consumer;
use serde::{Deserialize, Serialize};

use self::loader::{Load, Step};
use self::runner::{Runner, Send};
use self::watchdog::{fault, Watched};
use crate::control::{Button, Control};
use crate::deck::Snapshot;
use crate::events::{Events, Kind};
use crate::hands::{Change, Hands, Holder};
use crate::midi::{Map, Message, Out, Touch};
use crate::music::{Grid, TrackId};
use crate::plan::{Move, Plan, Step as PlanStep, Style};

/// How often the conductor looks.
pub const TICK: Duration = Duration::from_millis(2);
/// How long the hand-back walk takes, in beats.
pub const WALK_BEATS: f64 = 4.0;

/// When a sample fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quant {
    Now,
    Beat,
    Bar,
}

/// What the brain asks of the conductor.
#[derive(Debug, Clone)]
pub enum Command {
    /// Run `plan` from the playing deck's next steps.
    Arm(Plan),
    /// Load `track` to `deck` and confirm it.
    Load {
        deck: u8,
        track: TrackId,
    },
    /// The grid and length of the track on each deck, for the beat and the
    /// watchdog.
    Tracks(Vec<(u8, Grid, u64)>),
    /// Give the decks to a person, to ZenJ beside them, or to ZenJ.
    Hand(Holder),
    /// Fire sampler slot `slot`.
    Sample {
        slot: u8,
        when: Quant,
    },
    /// Automix on or off.
    Automix(bool),
}

/// What the conductor says of itself, for the brain, the tools and the
/// screens.
#[derive(Debug, Clone, Default, Serialize)]
pub struct State {
    pub holder: Holder,
    /// The armed plan's pair and style, while one runs.
    pub armed: Option<String>,
    pub loading: Option<String>,
    /// The deck each confirmed load put its track on.
    pub loaded: Vec<(u8, TrackId)>,
    pub automix: bool,
    pub fault: Option<String>,
    /// Whether the virtual controller's port is open.
    pub hands_open: bool,
}

/// Everything the conductor's thread holds.
pub struct Conductor {
    pub map: Map,
    pub out: Option<Out>,
    pub touches: Consumer<Touch>,
    pub commands: Receiver<Command>,
    pub snapshot: Arc<ArcSwap<Snapshot>>,
    pub state: Arc<ArcSwap<State>>,
    pub events: Arc<Events>,
    pub hands: Hands,
    /// Beats sends lead the beat by, for the MIDI path's latency.
    pub lead_beats: f64,
}

struct Pending {
    slot: u8,
    when: Quant,
    from: f64,
}

impl Conductor {
    fn send(&mut self, sends: &[Send]) {
        let mut msgs: Vec<Message> = Vec::new();
        for s in sends {
            match *s {
                Send::Set(k, v) => msgs.extend(self.map.set(k, v).unwrap_or_default()),
                Send::Press(b) => {
                    msgs.extend(self.map.press(Control::Button(b)).unwrap_or_default())
                }
            }
        }
        if msgs.is_empty() {
            return;
        }
        if let Some(out) = &mut self.out {
            if let Err(e) = out.send(&msgs) {
                self.events
                    .push(Kind::Fault, format!("the virtual controller: {e}"));
                self.out = None;
            }
        }
    }

    fn publish(&self, f: impl FnOnce(&mut State)) {
        let mut s = (**self.state.load()).clone();
        f(&mut s);
        self.state.store(Arc::new(s));
    }

    /// Run until `stop`.
    pub fn run(mut self, stop: Arc<AtomicBool>) {
        let mut runner: Option<Runner> = None;
        let mut load: Option<Load> = None;
        let mut tracks: Vec<(u8, Grid, u64)> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();
        let mut walk: Option<Runner> = None;
        let mut last_bar: Option<i64> = None;
        let open = self.out.is_some();
        self.publish(|s| {
            s.hands_open = open;
            s.holder = Holder::Human;
        });
        let mut next = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            let snap = self.snapshot.load_full();
            let on_air = snap.on_air().cloned();
            let beat = on_air.as_ref().and_then(|d| d.beat);

            // The hands first: a touch outranks every plan.
            while let Ok(t) = self.touches.pop() {
                match self.hands.touch(&t) {
                    Some(Change::TookBack(k)) => {
                        self.events.push(
                            Kind::Hands,
                            format!("{} taken back by hand", Control::Knob(k)),
                        );
                    }
                    Some(Change::HandedBack) => {
                        runner = None;
                        walk = self.walk_back(beat);
                        self.events
                            .push(Kind::Hands, "the decks are a person's again");
                        self.publish(|s| {
                            s.holder = Holder::Human;
                            s.armed = None;
                        });
                    }
                    Some(Change::Asked(m)) => {
                        self.hands.set_holder(m);
                        self.events
                            .push(Kind::Hands, format!("{} asked for on the pads", m.name()));
                        self.publish(|s| s.holder = m);
                    }
                    None => {}
                }
            }

            loop {
                match self.commands.try_recv() {
                    Ok(Command::Arm(plan)) => {
                        let now = on_air.as_ref().and_then(|d| d.beat).unwrap_or(0.0);
                        let line = format!("{} → {} ({})", plan.from, plan.to, plan.style.name());
                        runner = Some(Runner::new(plan, now));
                        self.publish(|s| s.armed = Some(line));
                    }
                    Ok(Command::Load { deck, track }) => {
                        let line = format!("{track} to deck {deck}");
                        load = Some(Load::new(deck, track));
                        self.publish(|s| s.loading = Some(line));
                    }
                    Ok(Command::Tracks(t)) => tracks = t,
                    Ok(Command::Hand(m)) => {
                        if m == Holder::Human && self.hands.holder() != Holder::Human {
                            walk = self.walk_back(beat);
                            runner = None;
                        } else {
                            self.hands.set_holder(m);
                        }
                        self.publish(|s| {
                            s.holder = m;
                            if m == Holder::Human {
                                s.armed = None;
                            }
                        });
                    }
                    Ok(Command::Sample { slot, when }) => pending.push(Pending {
                        slot,
                        when,
                        from: beat.unwrap_or(0.0),
                    }),
                    Ok(Command::Automix(on)) => {
                        self.send(&[Send::Press(Button::Automix)]);
                        self.publish(|s| s.automix = on);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }

            // The walk back runs as ZenJ still, then the person has it.
            if let (Some(w), Some(b)) = (walk.as_mut(), beat) {
                let sends = w.tick(b, self.lead_beats, &snap, &mut self.hands);
                self.send(&sends);
                if w.done() {
                    walk = None;
                    self.hands.set_holder(Holder::Human);
                }
            } else if walk.is_some() && beat.is_none() {
                walk = None;
                self.hands.set_holder(Holder::Human);
            }

            if let (Some(r), Some(b)) = (runner.as_mut(), beat) {
                let sends = r.tick(b, self.lead_beats, &snap, &mut self.hands);
                self.send(&sends);
                if r.done() {
                    let line = format!("{} → {} done", r.plan.from, r.plan.to);
                    self.events.push(Kind::Transition, line);
                    runner = None;
                    self.publish(|s| s.armed = None);
                }
            }

            if let Some(l) = load.as_mut() {
                match l.step(Instant::now(), &snap) {
                    Step::Press(buttons) => {
                        let sends: Vec<Send> = buttons.into_iter().map(Send::Press).collect();
                        self.send(&sends);
                    }
                    Step::Waiting => {}
                    Step::Loaded => {
                        let (deck, track) = (l.deck, l.track.clone());
                        self.events
                            .push(Kind::Next, format!("{track} loaded on deck {deck}"));
                        self.publish(|s| {
                            s.loading = None;
                            s.loaded.retain(|(d, _)| *d != deck);
                            s.loaded.push((deck, track));
                        });
                        load = None;
                    }
                    Step::Failed(why) => {
                        self.events
                            .push(Kind::Fault, format!("a load failed: {why}"));
                        self.publish(|s| {
                            s.loading = None;
                            s.fault = Some(why);
                        });
                        load = None;
                    }
                }
            }

            if let Some(b) = beat {
                pending.retain(|p| {
                    let due = match p.when {
                        Quant::Now => true,
                        Quant::Beat => b.floor() > p.from.floor(),
                        Quant::Bar => (b / 4.0).floor() > (p.from / 4.0).floor(),
                    };
                    if due {
                        let press = self.map.press(Control::Button(Button::Sampler(p.slot)));
                        if let Some(o) = self.out.as_mut() {
                            let _ = o.send(&press.unwrap_or_default());
                        }
                    }
                    !due
                });
                let bar = (b / 4.0).floor() as i64;
                if last_bar != Some(bar) {
                    last_bar = Some(bar);
                    self.bar(&snap, &tracks, runner.is_some());
                }
            } else if !pending.is_empty() && self.hands.holder() != Holder::Human {
                for p in pending.drain(..) {
                    if let Some(o) = self.out.as_mut() {
                        let _ = o.send(
                            &self
                                .map
                                .press(Control::Button(Button::Sampler(p.slot)))
                                .unwrap_or_default(),
                        );
                    }
                }
            }

            next += TICK;
            let now = Instant::now();
            if next > now {
                thread::sleep(next - now);
            } else {
                next = now;
            }
        }
    }

    /// The walk of what ZenJ moved back to the hardware, from `beat`; the
    /// mixer stays ZenJ's until it is done (`run`). With nothing to walk, or
    /// no beat to walk on, the person has it at once.
    fn walk_back(&mut self, beat: Option<f64>) -> Option<Runner> {
        let walk = beat.and_then(|at| {
            let steps = self.hands.walk_back(at, WALK_BEATS);
            (!steps.is_empty()).then(|| Runner::new(walk_plan(steps, at), at - 0.001))
        });
        if walk.is_none() {
            self.hands.set_holder(Holder::Human);
        }
        walk
    }

    /// The watchdog, once a bar, while ZenJ mixes.
    fn bar(&mut self, snap: &Snapshot, tracks: &[(u8, Grid, u64)], armed: bool) {
        if self.hands.holder() != Holder::Zenj {
            return;
        }
        let state = self.state.load_full();
        let on_air = snap
            .on_air()
            .and_then(|d| tracks.iter().find(|t| t.0 == d.number));
        let next_loaded = snap.decks.iter().filter(|d| !d.playing).any(|d| {
            d.track.is_some()
                && state
                    .loaded
                    .iter()
                    .any(|(n, t)| *n == d.number && d.track.as_ref() == Some(t))
        });
        let why = fault(&Watched {
            snapshot: snap,
            on_air: on_air.map(|t| (t.1, t.2)),
            next_loaded,
            armed,
        });
        if let Some(why) = why {
            if !state.automix {
                self.send(&[Send::Press(Button::Automix)]);
                self.events.push(Kind::Fault, format!("Automix on: {why}"));
                self.publish(|s| {
                    s.automix = true;
                    s.fault = Some(why);
                });
            }
        }
    }
}

/// A plan of only the walk back's moves.
fn walk_plan(steps: Vec<(f64, Move)>, at: f64) -> Plan {
    Plan {
        from: TrackId("walk".into()),
        to: TrackId("hand".into()),
        from_deck: 0,
        to_deck: 0,
        style: Style::Blend,
        start: at,
        end: at + WALK_BEATS,
        steps: steps
            .into_iter()
            .map(|(at, act)| PlanStep { at, act })
            .collect(),
        why: "the walk back to the hardware".into(),
    }
}
