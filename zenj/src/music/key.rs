//! Musical keys on the Camelot wheel: what rekordbox and TIDAL call a key,
//! read into one form, and how far apart two keys are for a mix.
//!
//! The wheel numbers the twelve keys of each mode around the circle of fifths:
//! a step round it is a fifth, `A` is minor and `B` major, and a key's
//! relative minor or major shares its number. Two keys mix cleanly when they
//! are the same, one step apart in the same mode, or relatives.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A key as the wheel names it: `8A` is A minor, `8B` C major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Camelot {
    /// 1 to 12.
    pub number: u8,
    pub minor: bool,
}

/// How two keys sit for a mix, best first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyFit {
    Same,
    /// One step round the wheel in the same mode.
    Adjacent,
    /// The relative major or minor.
    Relative,
    /// Two steps round: an energy lift, audible but often wanted.
    Lift,
    Clash,
}

impl KeyFit {
    /// The fit as a score in `0..=1`.
    pub fn score(self) -> f32 {
        match self {
            KeyFit::Same => 1.0,
            KeyFit::Adjacent => 0.9,
            KeyFit::Relative => 0.85,
            KeyFit::Lift => 0.5,
            KeyFit::Clash => 0.0,
        }
    }

    /// Whether a blend of the two keys is clean.
    pub fn mixes(self) -> bool {
        self <= KeyFit::Relative
    }
}

/// The pitch class of a note name — `C`, `C#`, `Db`, `CSharp`, `Bb` — with C
/// as 0; `None` for anything else.
fn pitch_class(note: &str) -> Option<u8> {
    let mut chars = note.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let base: i8 = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let rest: String = chars.collect();
    let shift: i8 = match rest.to_ascii_lowercase().as_str() {
        "" => 0,
        "#" | "♯" | "sharp" => 1,
        "b" | "♭" | "flat" => -1,
        _ => return None,
    };
    Some((base + shift).rem_euclid(12) as u8)
}

impl Camelot {
    /// The key whose tonic has pitch class `pc` (C = 0) in the given mode.
    pub fn of(pc: u8, minor: bool) -> Camelot {
        // A fifth up (+7 semitones) is a step round the wheel, and 7 is its own
        // inverse mod 12, so the step count from the mode's anchor — A minor at
        // 8A, C major at 8B — is (pc − anchor) × 7.
        let anchor = if minor { 9 } else { 0 };
        let steps = ((pc as i32 - anchor) * 7).rem_euclid(12);
        Camelot {
            number: ((steps + 7) % 12 + 1) as u8,
            minor,
        }
    }

    /// A key as rekordbox writes it: `Am`, `F#m`, `Db`, `Bbm`; or as the
    /// wheel does: `8A`, `11B`.
    pub fn parse(text: &str) -> Option<Camelot> {
        let t = text.trim();
        if let Some(c) = Self::parse_wheel(t) {
            return Some(c);
        }
        let (note, minor) = match t.strip_suffix('m') {
            Some(n) if !n.is_empty() => (n, true),
            _ => match t.strip_suffix("min") {
                Some(n) => (n, true),
                None => (t.strip_suffix("maj").unwrap_or(t), false),
            },
        };
        Some(Camelot::of(pitch_class(note)?, minor))
    }

    /// A key as TIDAL gives it: a note (`A`, `CSharp`, `Eb`) and a scale
    /// (`MAJOR`, `MINOR`).
    pub fn of_tidal(key: &str, scale: &str) -> Option<Camelot> {
        let minor = match scale.to_ascii_uppercase().as_str() {
            "MINOR" => true,
            "MAJOR" => false,
            _ => return None,
        };
        Some(Camelot::of(pitch_class(key)?, minor))
    }

    fn parse_wheel(t: &str) -> Option<Camelot> {
        let (digits, mode) = t.split_at(t.len().checked_sub(1)?);
        let minor = match mode {
            "A" | "a" => true,
            "B" | "b" => false,
            _ => return None,
        };
        let number: u8 = digits.parse().ok()?;
        (1..=12)
            .contains(&number)
            .then_some(Camelot { number, minor })
    }

    /// Steps round the wheel from `self` to `other`, ignoring mode: 0 to 6.
    fn steps(self, other: Camelot) -> u8 {
        let d = (self.number as i8 - other.number as i8).rem_euclid(12) as u8;
        d.min(12 - d)
    }

    /// How `next` sits after `self`.
    pub fn fit(self, next: Camelot) -> KeyFit {
        let steps = self.steps(next);
        match (steps, self.minor == next.minor) {
            (0, true) => KeyFit::Same,
            (1, true) => KeyFit::Adjacent,
            (0, false) => KeyFit::Relative,
            (2, true) => KeyFit::Lift,
            _ => KeyFit::Clash,
        }
    }
}

impl fmt::Display for Camelot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.number, if self.minor { 'A' } else { 'B' })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(t: &str) -> Camelot {
        Camelot::parse(t).unwrap_or_else(|| panic!("{t}"))
    }

    /// **Every key lands on its place on the wheel**, in both modes, however
    /// it is spelled.
    #[test]
    fn every_key_lands_on_the_wheel() {
        let minors = [
            ("Abm", "1A"),
            ("Ebm", "2A"),
            ("Bbm", "3A"),
            ("Fm", "4A"),
            ("Cm", "5A"),
            ("Gm", "6A"),
            ("Dm", "7A"),
            ("Am", "8A"),
            ("Em", "9A"),
            ("Bm", "10A"),
            ("F#m", "11A"),
            ("Dbm", "12A"),
        ];
        let majors = [
            ("B", "1B"),
            ("F#", "2B"),
            ("Db", "3B"),
            ("Ab", "4B"),
            ("Eb", "5B"),
            ("Bb", "6B"),
            ("F", "7B"),
            ("C", "8B"),
            ("G", "9B"),
            ("D", "10B"),
            ("A", "11B"),
            ("E", "12B"),
        ];
        for (name, wheel) in minors.iter().chain(majors.iter()) {
            assert_eq!(c(name).to_string(), *wheel, "{name}");
            assert_eq!(c(wheel), c(name), "{wheel} reads back");
        }
        assert_eq!(c("G#m"), c("Abm"), "enharmonic");
        assert_eq!(c("C#"), c("Db"));
    }

    #[test]
    fn tidal_names_a_key_by_note_and_scale() {
        assert_eq!(Camelot::of_tidal("CSharp", "MINOR"), Some(c("12A")));
        assert_eq!(Camelot::of_tidal("A", "MAJOR"), Some(c("11B")));
        assert_eq!(Camelot::of_tidal("Eb", "MINOR"), Some(c("2A")));
        assert_eq!(Camelot::of_tidal("UNKNOWN", "MINOR"), None);
        assert_eq!(Camelot::of_tidal("A", "DORIAN"), None);
    }

    #[test]
    fn anything_else_is_no_key() {
        for t in ["", "H", "13A", "0B", "Xm", "m", "C##"] {
            assert_eq!(Camelot::parse(t), None, "{t}");
        }
    }

    /// **The fit is the wheel's**: same, a step either way (12 wraps to 1),
    /// the relative, a two-step lift, and everything else a clash.
    #[test]
    fn the_fit_is_read_off_the_wheel() {
        assert_eq!(c("8A").fit(c("8A")), KeyFit::Same);
        assert_eq!(c("8A").fit(c("9A")), KeyFit::Adjacent);
        assert_eq!(c("8A").fit(c("7A")), KeyFit::Adjacent);
        assert_eq!(c("12B").fit(c("1B")), KeyFit::Adjacent, "the wheel wraps");
        assert_eq!(c("8A").fit(c("8B")), KeyFit::Relative);
        assert_eq!(c("8A").fit(c("10A")), KeyFit::Lift);
        assert_eq!(c("8A").fit(c("9B")), KeyFit::Clash);
        assert_eq!(c("8A").fit(c("2A")), KeyFit::Clash);
        assert!(KeyFit::Relative.mixes() && !KeyFit::Lift.mixes());
    }
}
