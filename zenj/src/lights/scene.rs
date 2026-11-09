//! The scenes (§9.4): a small named vocabulary, each a rule from the beat,
//! the palette and the brightness to what every bulb shows. A scene is drawn
//! whole each time it is asked; the rate gate sends only what changed.

use effector_bridge::proto::wiz::{Pilot, DIM_FLOOR};

use super::palette::{Palette, HIT, HOUSE};

/// What the room looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    /// Every bulb one colour.
    Wash,
    /// The bulbs in two colours, alternating along the track.
    Split,
    /// One bright bulb walking the track, a step a beat.
    Chase,
    /// Bright on the beat, dim off it.
    Pulse,
    /// Every bulb full white.
    Hit,
    /// Every bulb dark.
    Blackout,
    /// Warm white, for the hand-back and the end of the night.
    House,
}

impl Scene {
    pub const ALL: [Scene; 7] = [
        Scene::Wash,
        Scene::Split,
        Scene::Chase,
        Scene::Pulse,
        Scene::Hit,
        Scene::Blackout,
        Scene::House,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Scene::Wash => "wash",
            Scene::Split => "split",
            Scene::Chase => "chase",
            Scene::Pulse => "pulse",
            Scene::Hit => "hit",
            Scene::Blackout => "blackout",
            Scene::House => "house",
        }
    }

    pub fn parse(s: &str) -> Option<Scene> {
        Scene::ALL.into_iter().find(|x| x.name() == s.trim())
    }

    /// Whether it moves with the beat, and so is drawn again on each one.
    pub fn moves(self) -> bool {
        matches!(self, Scene::Chase | Scene::Pulse)
    }
}

/// The scene for phrase `phrase` of the set at `energy`: a wash while it is
/// quiet, a split or a chase in turn through the middle, a pulse at the peak.
pub fn for_phrase(energy: f32, phrase: u64) -> Scene {
    match energy {
        e if e < 0.4 => Scene::Wash,
        e if e < 0.75 => match phrase % 2 {
            0 => Scene::Split,
            _ => Scene::Chase,
        },
        _ => Scene::Pulse,
    }
}

fn colour((r, g, b): (u8, u8, u8), dimming: u8) -> Pilot {
    Pilot::Colour { r, g, b, dimming }
}

/// What each of `members` — bulbs by index, in a chase's order — shows in
/// `scene` at `beat`, drawn with `palette` at `level`.
pub fn draw(
    scene: Scene,
    palette: Palette,
    level: u8,
    beat: f64,
    members: &[usize],
) -> Vec<(usize, Pilot)> {
    let low = (level / 3).max(DIM_FLOOR);
    let n = members.len().max(1);
    let step = beat.max(0.0).floor() as usize % n;
    let on_beat = beat.max(0.0).fract() < 0.5;
    members
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let p = match scene {
                Scene::Wash => colour(palette.a, level),
                Scene::Split if k % 2 == 0 => colour(palette.a, level),
                Scene::Split => colour(palette.b, level),
                Scene::Chase if k == step => colour(palette.a, 100),
                Scene::Chase => colour(palette.b, low),
                Scene::Pulse if on_beat => colour(palette.a, 100),
                Scene::Pulse => colour(palette.a, low),
                Scene::Hit => HIT,
                Scene::Blackout => Pilot::Off,
                Scene::House => HOUSE,
            };
            (i, p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Palette = Palette {
        a: (255, 0, 0),
        b: (0, 0, 255),
    };

    fn red(d: u8) -> Pilot {
        colour((255, 0, 0), d)
    }

    fn blue(d: u8) -> Pilot {
        colour((0, 0, 255), d)
    }

    #[test]
    fn every_scene_reads_back_by_its_name() {
        for s in Scene::ALL {
            assert_eq!(Scene::parse(s.name()), Some(s));
        }
        assert_eq!(Scene::parse("strobe"), None);
    }

    /// **Energy chooses the phrase's scene**, the middle alternating.
    #[test]
    fn energy_chooses_the_phrases_scene() {
        assert_eq!(for_phrase(0.2, 0), Scene::Wash);
        assert_eq!(for_phrase(0.6, 4), Scene::Split);
        assert_eq!(for_phrase(0.6, 5), Scene::Chase);
        assert_eq!(for_phrase(0.9, 5), Scene::Pulse);
    }

    /// **Each scene draws every member exactly**: a chase walks a step a beat,
    /// a pulse is bright on the beat and dim off it.
    #[test]
    fn each_scene_draws_every_member() {
        let m = [3, 4, 5];
        assert_eq!(
            draw(Scene::Split, P, 80, 0.0, &m),
            vec![(3, red(80)), (4, blue(80)), (5, red(80))]
        );
        assert_eq!(
            draw(Scene::Chase, P, 60, 4.2, &m),
            vec![(3, blue(20)), (4, red(100)), (5, blue(20))],
            "beat 4 of three bulbs: the second"
        );
        assert_eq!(draw(Scene::Pulse, P, 30, 7.1, &m)[0], (3, red(100)));
        assert_eq!(
            draw(Scene::Pulse, P, 30, 7.6, &m)[0],
            (3, red(DIM_FLOOR)),
            "off the beat, no dimmer than a bulb goes"
        );
        assert!(draw(Scene::Blackout, P, 90, 0.0, &m)
            .iter()
            .all(|(_, p)| *p == Pilot::Off));
        assert!(draw(Scene::House, P, 90, 0.0, &m)
            .iter()
            .all(|(_, p)| *p == HOUSE));
        assert!(draw(Scene::Wash, P, 90, 0.0, &[]).is_empty());
    }
}
