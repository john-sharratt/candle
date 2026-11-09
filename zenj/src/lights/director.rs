//! What the room looks like at a beat (§9.4–§9.7): pure — the night's state
//! and one instant in, every bulb's state out — so the lights can be planned
//! ahead and planned again, and each rule is tested without a socket.
//!
//! In order, the first that holds decides:
//!
//! 1. **rekordbox closed**: the lights are the house's (`docs/house_modes.md`),
//!    and nothing is drawn.
//! 2. **The house lights**, asked for by a person.
//! 3. **Most bulbs dead**: the warm house scene.
//! 4. **Midnight**: the last ten seconds dim toward dark, the instant itself is
//!    the hit, then a minute of gold.
//! 5. **A cue** of the armed plan: a blackout for its bar, the hit on its beat,
//!    a wash across a blend.
//! 6. **A scene a person held**, on its group.
//! 7. **The scene of the phrase**: Zen's, by energy and phrase, while ZenJ or
//!    its assist has the decks; a plain energy-following one while a person
//!    mixes — nothing Zen imagined is imposed on someone mixing.

use effector_bridge::proto::wiz::Pilot;

use super::cue::Cue;
use super::palette::{self, Palette, HIT, HOUSE};
use super::scene::{draw, for_phrase, Scene};
use crate::hands::Holder;

/// The beats of a phrase, on which the scene changes, as the screens' do.
pub const PHRASE: f64 = 32.0;
/// The beats a blackout cue holds: its bar.
pub const BLACKOUT_BEATS: f64 = 4.0;
/// The beats a hit is: what a planner samples it at.
pub const HIT_BEATS: f64 = 0.5;
/// The seconds before midnight the lights dim through.
pub const COUNTDOWN_S: f64 = 10.0;
/// The seconds midnight's hit is.
pub const MIDNIGHT_HIT_S: f64 = 0.25;
/// The seconds of gold after it.
pub const AFTER_S: f64 = 60.0;
/// The new year's colours.
pub const GOLD: Palette = Palette {
    a: (255, 170, 0),
    b: (255, 255, 255),
};

/// The night at one instant, as the director reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    pub rekordbox: bool,
    pub holder: Holder,
    /// The night's energy, `0..=1`.
    pub energy: f32,
    /// The playing deck's beat at the instant.
    pub beat: Option<f64>,
    /// Seconds from the instant to midnight, negative after it.
    pub to_midnight: f64,
    pub bulbs: usize,
    pub dead: usize,
}

/// What is asked of the director from outside the set.
#[derive(Debug, Clone, PartialEq)]
pub enum Ask {
    /// Hold `scene` on `group` (every bulb with none) until it is let go.
    Hold { scene: Scene, group: Option<String> },
    /// Let every held scene go: back to what the set wants.
    Release,
    /// The house lights on, or off again.
    House(bool),
}

/// What the room looks like at the instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Look {
    /// Every bulb's state, by index.
    pub bulbs: Vec<(usize, Pilot)>,
    /// A hit: sent at once whatever the budget, and again.
    pub key: bool,
    /// The look, in a word, for the tools and the screens.
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct Director {
    holds: Vec<(Option<String>, Scene)>,
    house: bool,
    /// The armed plan's cues, each with the beat it falls on.
    cues: Vec<(f64, Cue)>,
}

impl Director {
    pub fn new() -> Director {
        Director::default()
    }

    pub fn ask(&mut self, a: Ask) {
        match a {
            Ask::Hold { scene, group } => {
                if group.is_none() {
                    self.holds.clear();
                }
                self.holds.retain(|(g, _)| *g != group);
                self.holds.push((group, scene));
            }
            Ask::Release => self.holds.clear(),
            Ask::House(on) => self.house = on,
        }
    }

    /// The armed plan's cues, each at its beat — none once the plan is gone.
    pub fn cues(&mut self, cues: Vec<(f64, Cue)>) {
        self.cues = cues;
    }

    /// The beats the cues fall on: moments a planner samples.
    pub fn cue_beats(&self) -> Vec<f64> {
        self.cues.iter().map(|(b, _)| *b).collect()
    }

    /// The cue in force at `beat`: the latest begun whose window holds it.
    fn cue_at(&self, beat: f64) -> Option<(Cue, f64)> {
        self.cues
            .iter()
            .filter(|(from, c)| {
                let since = beat - from;
                let len = match c {
                    Cue::Blackout => BLACKOUT_BEATS,
                    Cue::Hit => HIT_BEATS,
                    Cue::Fade { beats } => *beats,
                };
                (0.0..len).contains(&since)
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(from, c)| (*c, *from))
    }

    pub fn house(&self) -> bool {
        self.house
    }

    /// The scenes held, by group, in words.
    pub fn held(&self) -> Vec<String> {
        self.holds
            .iter()
            .map(|(g, s)| format!("{} on {}", s.name(), g.as_deref().unwrap_or("every bulb")))
            .collect()
    }

    /// The look at `inp`'s instant, `groups(name)` naming a group's bulbs —
    /// `None` while the lights are the house's.
    pub fn look(&self, inp: &Input, groups: &dyn Fn(Option<&str>) -> Vec<usize>) -> Option<Look> {
        if !inp.rekordbox {
            return None;
        }
        let all = groups(None);
        let whole = |s: Scene, p: Palette, level: u8, b: f64| draw(s, p, level, b, &all);
        let beat = inp.beat.unwrap_or(0.0);
        let p = palette::of(inp.energy);
        let level = palette::level(inp.energy);
        let every = |pilot: Pilot| all.iter().map(|&i| (i, pilot)).collect::<Vec<_>>();
        if self.house {
            return Some(plain(every(HOUSE), "house lights"));
        }
        if inp.bulbs > 0 && inp.dead * 2 > inp.bulbs {
            return Some(plain(every(HOUSE), "house: most bulbs are not answering"));
        }
        let t = inp.to_midnight;
        if t > 0.0 && t <= COUNTDOWN_S {
            let d = (10.0 + 90.0 * (t / COUNTDOWN_S)).round() as u8;
            return Some(plain(whole(Scene::Wash, GOLD, d, beat), "the countdown"));
        }
        if t <= 0.0 && t > -MIDNIGHT_HIT_S {
            return Some(keyed(every(HIT), "midnight"));
        }
        if t <= 0.0 && t > -AFTER_S {
            return Some(plain(
                whole(Scene::Pulse, GOLD, 100, beat),
                "happy new year",
            ));
        }
        if let Some((c, _)) = inp.beat.and_then(|b| self.cue_at(b)) {
            return Some(match c {
                Cue::Blackout => plain(every(Pilot::Off), "blackout"),
                Cue::Hit => keyed(every(HIT), "hit"),
                Cue::Fade { .. } => plain(
                    whole(Scene::Wash, p, level, beat),
                    "a wash across the blend",
                ),
            });
        }
        let scene = match inp.holder {
            Holder::Human if inp.energy < 0.5 => Scene::Wash,
            Holder::Human => Scene::Pulse,
            _ => for_phrase(inp.energy, (beat / PHRASE).floor().max(0.0) as u64),
        };
        let mut bulbs = whole(scene, p, level, beat);
        let mut name = scene.name().to_string();
        for (group, held) in &self.holds {
            for (i, pilot) in draw(*held, p, level, beat, &groups(group.as_deref())) {
                if let Some(slot) = bulbs.iter_mut().find(|(j, _)| *j == i) {
                    slot.1 = pilot;
                }
            }
            name = format!("{name}, {} held", held.name());
        }
        Some(plain(bulbs, &name))
    }
}

fn plain(bulbs: Vec<(usize, Pilot)>, name: &str) -> Look {
    Look {
        bulbs,
        key: false,
        name: name.to_string(),
    }
}

fn keyed(bulbs: Vec<(usize, Pilot)>, name: &str) -> Look {
    Look {
        bulbs,
        key: true,
        name: name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp() -> Input {
        Input {
            rekordbox: true,
            holder: Holder::Zenj,
            energy: 0.9,
            beat: Some(64.0),
            to_midnight: 3600.0,
            bulbs: 4,
            dead: 0,
        }
    }

    fn at_beat(b: f64) -> Input {
        Input {
            beat: Some(b),
            ..inp()
        }
    }

    fn to_midnight(s: f64) -> Input {
        Input {
            to_midnight: s,
            ..inp()
        }
    }

    fn groups(g: Option<&str>) -> Vec<usize> {
        match g {
            None => vec![0, 1, 2, 3],
            Some("wall") => vec![2, 3],
            Some(_) => vec![],
        }
    }

    fn look(d: &Director, i: &Input) -> Look {
        d.look(i, &groups).unwrap()
    }

    /// **With rekordbox closed the lights are the house's**: nothing drawn.
    #[test]
    fn with_rekordbox_closed_nothing_is_drawn() {
        let closed = Input {
            rekordbox: false,
            ..inp()
        };
        assert!(Director::new().look(&closed, &groups).is_none());
    }

    /// **The phrase's scene follows the energy while ZenJ mixes**, and a
    /// plain one while a person does.
    #[test]
    fn the_phrase_scene_follows_who_mixes() {
        let d = Director::new();
        assert_eq!(look(&d, &inp()).name, "pulse");
        let middle = Input {
            energy: 0.6,
            beat: Some(33.0),
            ..inp()
        };
        assert_eq!(look(&d, &middle).name, "chase", "phrase 1 of the middle");
        let person = Input {
            holder: Holder::Human,
            ..middle
        };
        assert_eq!(look(&d, &person).name, "pulse");
        let person_low = Input {
            energy: 0.3,
            ..person
        };
        assert_eq!(look(&d, &person_low).name, "wash");
    }

    /// **A cue is a window of beats, the same however often it is asked**: a
    /// blackout for its bar, a hit on its beat, keyed.
    #[test]
    fn a_cue_is_a_window_of_beats() {
        let mut d = Director::new();
        // A cut's cues: its bar dark, the hit on the drop.
        d.cues(vec![(60.0, Cue::Blackout), (64.0, Cue::Hit)]);
        assert_eq!(d.cue_beats(), [60.0, 64.0]);
        assert_eq!(look(&d, &at_beat(59.5)).name, "pulse", "not yet");
        let b = look(&d, &at_beat(61.0));
        assert_eq!(b.name, "blackout");
        assert!(b.bulbs.iter().all(|(_, p)| *p == Pilot::Off));
        assert_eq!(look(&d, &at_beat(61.0)), b, "asked again, the same");
        let h = look(&d, &at_beat(64.0));
        assert!(h.key && h.bulbs.iter().all(|(_, p)| *p == HIT));
        assert_eq!(
            look(&d, &at_beat(64.5)).name,
            "pulse",
            "a hit is half a beat"
        );
        d.cues(vec![(64.0, Cue::Fade { beats: 32.0 })]);
        assert_eq!(look(&d, &at_beat(80.0)).name, "a wash across the blend");
        assert_eq!(look(&d, &at_beat(96.0)).name, "pulse");
        d.cues(Vec::new());
        assert_eq!(look(&d, &at_beat(80.0)).name, "pulse", "the plan is gone");
    }

    /// **Midnight**: the countdown dims, the instant is a keyed hit, then gold.
    #[test]
    fn midnight_dims_strikes_and_turns_gold() {
        let d = Director::new();
        let c = look(&d, &to_midnight(5.0));
        assert_eq!(c.name, "the countdown");
        assert_eq!(
            c.bulbs[0].1,
            Pilot::Colour {
                r: 255,
                g: 170,
                b: 0,
                dimming: 55
            }
        );
        let strike = look(&d, &to_midnight(0.0));
        assert!(strike.key && strike.name == "midnight");
        assert_eq!(look(&d, &to_midnight(-1.0)).name, "happy new year");
        assert_eq!(look(&d, &to_midnight(-61.0)).name, "pulse");
    }

    /// **A held scene sits on its group, over the phrase's**, until let go;
    /// the house lights and most bulbs dead are the house scene.
    #[test]
    fn a_hold_the_house_and_the_dead() {
        let mut d = Director::new();
        d.ask(Ask::Hold {
            scene: Scene::Blackout,
            group: Some("wall".into()),
        });
        let l = look(&d, &inp());
        assert_eq!(l.name, "pulse, blackout held");
        assert_eq!(l.bulbs[2].1, Pilot::Off);
        assert_ne!(l.bulbs[0].1, Pilot::Off);
        assert_eq!(d.held(), ["blackout on wall"]);
        d.ask(Ask::Release);
        assert_eq!(look(&d, &inp()).name, "pulse");
        d.ask(Ask::House(true));
        assert!(look(&d, &inp()).bulbs.iter().all(|(_, p)| *p == HOUSE));
        d.ask(Ask::House(false));
        let dead = Input { dead: 3, ..inp() };
        assert_eq!(look(&d, &dead).name, "house: most bulbs are not answering");
    }
}
