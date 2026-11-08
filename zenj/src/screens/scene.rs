//! Which scene each screen shows (§8): a MilkDrop preset from the library,
//! changed on a phrase of the track on air — every 32 beats — so a change
//! lands with the music. Each screen starts at its own place in the library,
//! so no two show the same scene. The beat is read early by the screen's own
//! lag, so the change lands when the room sees it.

use std::fs;
use std::path::{Path, PathBuf};

/// Beats a phrase lasts.
pub const PHRASE: f64 = 32.0;

/// The presets in `dir`, by name: every `.milk` file, at any depth.
pub fn presets(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("milk")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// One screen's place in the library and in the music.
#[derive(Debug, Clone)]
pub struct Scenes {
    count: usize,
    at: usize,
    /// The phrase the screen last changed on.
    phrase: Option<i64>,
}

impl Scenes {
    /// A screen, the `screen`th of `screens`, over `count` presets.
    pub fn new(count: usize, screen: usize, screens: usize) -> Scenes {
        Scenes {
            count,
            at: if screens == 0 { 0 } else { count * screen / screens.max(1) },
            phrase: None,
        }
    }

    pub fn current(&self) -> Option<usize> {
        (self.count > 0).then_some(self.at % self.count.max(1))
    }

    /// The preset to change to at `beat` — read `lead_ms` early at `bpm` —
    /// when a new phrase has begun; a preset that failed to load is passed
    /// over by calling again with `skip`.
    pub fn at_beat(&mut self, beat: f64, bpm: f32, lead_ms: f64) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let seen = beat + lead_ms * bpm as f64 / 60_000.0;
        let phrase = (seen / PHRASE).floor() as i64;
        match self.phrase {
            None => {
                self.phrase = Some(phrase);
                None
            }
            Some(p) if phrase != p => {
                self.phrase = Some(phrase);
                self.skip()
            }
            Some(_) => None,
        }
    }

    /// The next preset along.
    pub fn skip(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.at = (self.at + 1) % self.count;
        Some(self.at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screens_start_apart_and_change_on_each_phrase() {
        assert_eq!(Scenes::new(100, 0, 4).current(), Some(0));
        assert_eq!(Scenes::new(100, 3, 4).current(), Some(75));
        assert_eq!(Scenes::new(0, 1, 4).current(), None);
        let mut s = Scenes::new(10, 0, 1);
        assert_eq!(s.at_beat(5.0, 120.0, 0.0), None, "the first look only learns the phrase");
        assert_eq!(s.at_beat(31.0, 120.0, 0.0), None);
        assert_eq!(s.at_beat(32.0, 120.0, 0.0), Some(1));
        assert_eq!(s.at_beat(40.0, 120.0, 0.0), None);
        // 500 ms early at 120 BPM is a beat early: 63 is seen as 64.
        assert_eq!(s.at_beat(63.0, 120.0, 500.0), Some(2));
        assert_eq!(s.skip(), Some(3));
    }

    #[test]
    fn the_library_is_every_milk_file_by_name() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("pack")).unwrap();
        fs::write(dir.path().join("b.milk"), "").unwrap();
        fs::write(dir.path().join("pack").join("a.MILK"), "").unwrap();
        fs::write(dir.path().join("readme.txt"), "").unwrap();
        let names: Vec<String> = presets(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["b.milk", "a.MILK"], "sorted by path: b.milk sorts before pack/");
    }
}
