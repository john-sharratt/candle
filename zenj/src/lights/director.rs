//! What the room should look like now (§9.4–§9.7): pure — the night's state
//! in, every bulb's state out — so each rule is tested without a socket.
//!
//! In order, the first that holds decides:
//!
//! 1. **rekordbox closed**: the lights are the house's (`docs/house_modes.md`),
//!    and nothing is drawn.
//! 2. **The house lights**, asked for by a person.
//! 3. **Most bulbs dead**: the warm house scene, and the screens say so.
//! 4. **Midnight**: the last ten seconds dim toward dark, the drop is the hit,
//!    sent twice, then a minute of gold.
//! 5. **A cue** of the armed plan: a blackout for its bar, a hit, a wash
//!    across a long blend.
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
/// The seconds before midnight the lights dim through.
pub const COUNTDOWN_S: f64 = 10.0;
/// The seconds of gold after it.
pub const AFTER_S: f64 = 60.0;
/// The new year's colours.
pub const GOLD: Palette = Palette {
    a: (255, 170, 0),
    b: (255, 255, 255),
};

/// What the director is given each tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    pub rekordbox: bool,
    pub holder: Holder,
    /// The night's energy now, `0..=1`.
    pub energy: f32,
    /// The playing deck's beat, led by the bulbs' latency.
    pub beat: Option<f64>,
    /// Seconds to midnight, negative after it.
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

/// What the room looks like this tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Look {
    /// Every bulb's state, by index.
    pub bulbs: Vec<(usize, Pilot)>,
    /// The ones sent at once, twice: a hit.
    pub key: bool,
    /// The scene it is, in a word, for the tools and the screens.
    pub name: String,
}

#[derive(Debug, Default)]
pub struct Director {
    holds: Vec<(Option<String>, Scene)>,
    house: bool,
    /// The cue in force, and the beat it began on.
    cue: Option<(Cue, f64)>,
    /// The beat of the last hit sent, so a hit is sent once.
    hit_at: Option<f64>,
    /// Whether midnight's hit has gone.
    struck: bool,
}

impl Director {
    pub fn new() -> Director {
        Director::default()
    }

    pub fn ask(&mut self, a: Ask) {
        match a {
            Ask::Hold { scene, group } => {
                self.holds.retain(|(g, _)| *g != group);
                if group.is_none() {
                    self.holds.clear();
                }
                self.holds.push((group, scene));
            }
            Ask::Release => self.holds.clear(),
            Ask::House(on) => self.house = on,
        }
    }

    /// A cue of the plan, at `beat`.
    pub fn cue(&mut self, c: Cue, beat: f64) {
        self.cue = Some((c, beat));
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

    /// The look for `inp`, `groups(name)` naming a group's bulbs — `None`
    /// while the lights are the house's.
    pub fn look(
        &mut self,
        inp: &Input,
        groups: &dyn Fn(Option<&str>) -> Vec<usize>,
    ) -> Option<Look> {
        if !inp.rekordbox {
            self.struck = false;
            return None;
        }
        let all = groups(None);
        let whole =
            |scene: Scene, p: Palette, level: u8, beat: f64| draw(scene, p, level, beat, &all);
        let beat = inp.beat.unwrap_or(0.0);
        let p = palette::of(inp.energy);
        let level = palette::level(inp.energy);
        if self.house {
            return Some(plain(
                all.iter().map(|&i| (i, HOUSE)).collect(),
                "house lights",
            ));
        }
        if inp.bulbs > 0 && inp.dead * 2 > inp.bulbs {
            return Some(plain(
                all.iter().map(|&i| (i, HOUSE)).collect(),
                "house: most bulbs are not answering",
            ));
        }
        // Midnight.
        let t = inp.to_midnight;
        if t > 0.0 && t <= COUNTDOWN_S {
            let d = (10.0 + 90.0 * (t / COUNTDOWN_S)).round() as u8;
            return Some(plain(whole(Scene::Wash, GOLD, d, beat), "the countdown"));
        }
        if t <= 0.0 && t > -AFTER_S {
            if !self.struck {
                self.struck = true;
                return Some(Look {
                    bulbs: all.iter().map(|&i| (i, HIT)).collect(),
                    key: true,
                    name: "midnight".into(),
                });
            }
            return Some(plain(
                whole(Scene::Pulse, GOLD, 100, beat),
                "happy new year",
            ));
        }
        // A cue of the plan.
        if let Some((c, from)) = self.cue {
            match c {
                Cue::Blackout if beat < from + 4.0 => {
                    return Some(plain(whole(Scene::Blackout, p, level, beat), "blackout"));
                }
                Cue::Hit if self.hit_at != Some(from) => {
                    self.hit_at = Some(from);
                    return Some(Look {
                        bulbs: whole(Scene::Hit, p, level, beat),
                        key: true,
                        name: "hit".into(),
                    });
                }
                Cue::Fade { beats } if beat < from + beats => {
                    return Some(plain(
                        whole(Scene::Wash, p, level, beat),
                        "a wash across the blend",
                    ));
                }
                _ => self.cue = None,
            }
        }
        // The phrase's scene, then any held on top of it.
        let scene = match inp.holder {
            Holder::Human if inp.energy < 0.5 => Scene::Wash,
            Holder::Human => Scene::Pulse,
            _ => for_phrase(inp.energy, (beat / PHRASE).floor().max(0.0) as u64),
        };
        let mut bulbs = whole(scene, p, level, beat);
        let mut name = scene.name().to_string();
        for (group, held) in &self.holds {
            let members = groups(group.as_deref());
            for (i, pilot) in draw(*held, p, level, beat, &members) {
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

    fn look(d: &mut Director, i: &Input) -> Look {
        d.look(i, &groups).unwrap()
    }

    /// **With rekordbox closed the lights are the house's**: nothing drawn.
    #[test]
    fn with_rekordbox_closed_nothing_is_drawn() {
        let mut d = Director::new();
        let closed = Input {
            rekordbox: false,
            ..inp()
        };
        assert!(d.look(&closed, &groups).is_none());
    }

    /// **The phrase's scene follows the energy while ZenJ mixes**, and a
    /// plain one while a person does.
    #[test]
    fn the_phrase_scene_follows_who_mixes() {
        let mut d = Director::new();
        assert_eq!(look(&mut d, &inp()).name, "pulse");
        let quiet = Input {
            energy: 0.6,
            beat: Some(33.0),
            ..inp()
        };
        assert_eq!(look(&mut d, &quiet).name, "chase", "phrase 1 of the middle");
        let person = Input {
            holder: Holder::Human,
            ..quiet
        };
        assert_eq!(look(&mut d, &person).name, "pulse");
        let person_low = Input {
            energy: 0.3,
            ..person
        };
        assert_eq!(look(&mut d, &person_low).name, "wash");
    }

    /// **A blackout holds its bar, then the hit is sent once, twice over.**
    #[test]
    fn a_blackout_then_a_hit() {
        let mut d = Director::new();
        d.cue(Cue::Blackout, 60.0);
        let b = look(&mut d, &at_beat(61.0));
        assert_eq!(b.name, "blackout");
        assert!(b.bulbs.iter().all(|(_, p)| *p == Pilot::Off));
        d.cue(Cue::Hit, 64.0);
        let h = look(&mut d, &inp());
        assert!(h.key && h.bulbs.iter().all(|(_, p)| *p == HIT));
        assert_eq!(look(&mut d, &inp()).name, "pulse", "once, then the phrase");
    }

    /// **A fade is a wash for its beats.**
    #[test]
    fn a_fade_is_a_wash_for_its_beats() {
        let mut d = Director::new();
        d.cue(Cue::Fade { beats: 32.0 }, 64.0);
        let blend = look(&mut d, &at_beat(80.0));
        assert_eq!(blend.name, "a wash across the blend");
        assert_eq!(look(&mut d, &at_beat(96.0)).name, "pulse");
    }

    /// **Midnight**: the countdown dims, the drop is a hit once, then gold.
    #[test]
    fn midnight_dims_strikes_and_turns_gold() {
        let mut d = Director::new();
        let c = look(&mut d, &to_midnight(5.0));
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
        let strike = look(&mut d, &to_midnight(-0.1));
        assert!(strike.key && strike.name == "midnight");
        assert_eq!(look(&mut d, &to_midnight(-1.0)).name, "happy new year");
        assert_eq!(look(&mut d, &to_midnight(-61.0)).name, "pulse");
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
        let l = look(&mut d, &inp());
        assert_eq!(l.name, "pulse, blackout held");
        assert_eq!(l.bulbs[2].1, Pilot::Off);
        assert_ne!(l.bulbs[0].1, Pilot::Off);
        assert_eq!(d.held(), ["blackout on wall"]);
        d.ask(Ask::Release);
        assert_eq!(look(&mut d, &inp()).name, "pulse");
        d.ask(Ask::House(true));
        assert!(look(&mut d, &inp()).bulbs.iter().all(|(_, p)| *p == HOUSE));
        d.ask(Ask::House(false));
        let dead = look(&mut d, &Input { dead: 3, ..inp() });
        assert_eq!(dead.name, "house: most bulbs are not answering");
    }
}
