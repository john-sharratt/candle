//! The colours and brightness the night's energy asks for (§9.4): deep and
//! cool while it warms up, hot and bright at its peak.

use effector_bridge::proto::wiz::Pilot;

/// Two colours a scene draws with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    pub a: (u8, u8, u8),
    pub b: (u8, u8, u8),
}

/// The palettes, from the lowest energy to the highest.
const BY_ENERGY: [Palette; 5] = [
    Palette {
        a: (40, 0, 160),
        b: (0, 40, 120),
    },
    Palette {
        a: (120, 0, 200),
        b: (0, 120, 200),
    },
    Palette {
        a: (255, 0, 140),
        b: (120, 0, 255),
    },
    Palette {
        a: (255, 80, 0),
        b: (255, 0, 100),
    },
    Palette {
        a: (255, 0, 0),
        b: (255, 200, 120),
    },
];

/// The palette for `energy`, `0..=1`.
pub fn of(energy: f32) -> Palette {
    let i = (energy.clamp(0.0, 1.0) * (BY_ENERGY.len() - 1) as f32).round() as usize;
    BY_ENERGY[i]
}

/// How bright the room is at `energy`: 30% at its lowest, full at its peak.
pub fn level(energy: f32) -> u8 {
    (30.0 + 70.0 * energy.clamp(0.0, 1.0)).round() as u8
}

/// The house lights: warm white, bright enough to find a coat by.
pub const HOUSE: Pilot = Pilot::White {
    kelvin: 2700,
    dimming: 80,
};

/// A hit: every bulb full white.
pub const HIT: Pilot = Pilot::Colour {
    r: 255,
    g: 255,
    b: 255,
    dimming: 100,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_picks_the_palette_and_the_level() {
        assert_eq!(of(0.0), BY_ENERGY[0]);
        assert_eq!(of(1.0), BY_ENERGY[4]);
        assert_eq!(of(0.55), BY_ENERGY[2]);
        assert_eq!(of(7.0), BY_ENERGY[4], "held to the peak");
        assert_eq!(level(0.0), 30);
        assert_eq!(level(1.0), 100);
        assert_eq!(level(0.5), 65);
    }
}
