//! The decks as ZenJ knows them (§4.2): each sample read from rekordbox
//! checked before it is believed, played-ness from a position that moves,
//! and the beat from the track's grid. A reader that cannot vouch for what
//! it read says it is blind rather than guess, and a blind reader drops ZenJ
//! to Automix (§7.4).

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::music::{Grid, TrackId};

/// rekordbox's sample position runs at this rate whatever the track's.
pub const SAMPLE_RATE: f64 = 44_100.0;
/// A deck plays while its position has moved this recently.
pub const PLAYING_HOLD: Duration = Duration::from_millis(300);
/// The tempos a loaded deck may report.
pub const BPM_RANGE: std::ops::RangeInclusive<f32> = 40.0..=250.0;

/// One deck as read, before it is checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Raw {
    pub bpm: f32,
    pub samples: i64,
    /// The loaded track's analysis file, empty with none loaded.
    pub anlz: String,
}

/// One deck as believed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Deck {
    pub number: u8,
    pub track: Option<TrackId>,
    /// The tempo it plays at now.
    pub bpm: f32,
    pub position_ms: f64,
    /// Its beat on the track's grid, when the track is known.
    pub beat: Option<f64>,
    pub playing: bool,
}

/// All decks at one instant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Snapshot {
    pub decks: Vec<Deck>,
    /// The deck rekordbox follows as master, 1-based.
    pub master: Option<u8>,
    /// Why the reader cannot vouch for this sample, if it cannot.
    pub blind: Option<String>,
}

impl Snapshot {
    pub fn blind(why: impl Into<String>) -> Snapshot {
        Snapshot {
            decks: Vec::new(),
            master: None,
            blind: Some(why.into()),
        }
    }

    pub fn deck(&self, number: u8) -> Option<&Deck> {
        self.decks.iter().find(|d| d.number == number)
    }

    /// The deck on air: the master while it plays, else the one playing.
    pub fn on_air(&self) -> Option<&Deck> {
        let master = self.master.and_then(|m| self.deck(m)).filter(|d| d.playing);
        master.or_else(|| self.decks.iter().find(|d| d.playing))
    }
}

/// An analysis file's path as rekordbox's library and its memory both name
/// it: from `PIONEER/`, forward slashes, lower case — so one file has one
/// name whichever wrote it.
pub fn anlz_key(path: &str) -> Option<String> {
    let p = path.replace('\\', "/").to_ascii_lowercase();
    let at = p.find("pioneer/")?;
    Some(p[at..].to_string())
}

/// What the reader needs of the library: the track an analysis file is, and
/// its grid.
pub trait Tracks {
    fn by_anlz(&self, key: &str) -> Option<(TrackId, Grid)>;
}

/// One deck's history between samples: where it last moved.
#[derive(Debug, Default, Clone)]
struct Seen {
    samples: Option<i64>,
    moved: Option<Instant>,
}

/// The check every sample passes (§4.2), and what the reader keeps between
/// samples to judge playing.
#[derive(Debug, Default)]
pub struct Judge {
    seen: Vec<Seen>,
}

impl Judge {
    pub fn new() -> Judge {
        Judge::default()
    }

    /// The snapshot `raws` and `master` make at `now`, or a blind one
    /// naming the first deck whose sample cannot be believed.
    pub fn judge(
        &mut self,
        raws: &[Raw],
        master: u8,
        now: Instant,
        tracks: &dyn Tracks,
    ) -> Snapshot {
        if self.seen.len() < raws.len() {
            self.seen.resize(raws.len(), Seen::default());
        }
        let mut decks = Vec::with_capacity(raws.len());
        for (i, raw) in raws.iter().enumerate() {
            let number = i as u8 + 1;
            let loaded = !raw.anlz.is_empty();
            let track = if loaded {
                let Some(key) = anlz_key(&raw.anlz) else {
                    return Snapshot::blind(format!(
                        "deck {number}: `{}` is no analysis file",
                        raw.anlz
                    ));
                };
                match tracks.by_anlz(&key) {
                    Some(t) => Some(t),
                    None => {
                        return Snapshot::blind(format!(
                            "deck {number}: {key} is in no library track"
                        ))
                    }
                }
            } else {
                None
            };
            if loaded && !BPM_RANGE.contains(&raw.bpm) {
                return Snapshot::blind(format!("deck {number}: a tempo of {}", raw.bpm));
            }
            // A cue before the first beat sits a little before zero; a minute
            // before it is no position.
            if loaded && raw.samples < -(SAMPLE_RATE as i64) * 60 {
                return Snapshot::blind(format!("deck {number}: a position of {}", raw.samples));
            }
            let seen = &mut self.seen[i];
            if seen.samples.is_some_and(|s| s != raw.samples) {
                seen.moved = Some(now);
            }
            seen.samples = Some(raw.samples);
            let playing = loaded && seen.moved.is_some_and(|m| now.duration_since(m) < PLAYING_HOLD);
            let position_ms = raw.samples as f64 / SAMPLE_RATE * 1000.0;
            decks.push(Deck {
                number,
                beat: track.as_ref().map(|(_, g)| g.beat_at(position_ms)),
                track: track.map(|(id, _)| id),
                bpm: raw.bpm,
                position_ms,
                playing,
            });
        }
        // rekordbox counts its master from 0, and past its decks before one
        // is chosen.
        let master = ((master as usize) < decks.len()).then_some(master + 1);
        Snapshot {
            decks,
            master,
            blind: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct Lib(HashMap<String, (TrackId, Grid)>);

    impl Tracks for Lib {
        fn by_anlz(&self, key: &str) -> Option<(TrackId, Grid)> {
            self.0.get(key).cloned()
        }
    }

    fn lib() -> Lib {
        Lib(HashMap::from([(
            "pioneer/usbanlz/a1b/2c3d/anlz0000.dat".to_string(),
            (
                TrackId("t1".into()),
                Grid {
                    bpm: 120.0,
                    first_beat_ms: 100.0,
                },
            ),
        )]))
    }

    fn raw(bpm: f32, samples: i64, anlz: &str) -> Raw {
        Raw {
            bpm,
            samples,
            anlz: anlz.into(),
        }
    }

    const PATH: &str = "C:\\Users\\j\\AppData\\Roaming\\Pioneer\\rekordbox\\share\\PIONEER\\USBANLZ\\a1b\\2c3d\\ANLZ0000.DAT";

    #[test]
    fn an_analysis_path_has_one_name_whoever_wrote_it() {
        assert_eq!(
            anlz_key(PATH).as_deref(),
            Some("pioneer/usbanlz/a1b/2c3d/anlz0000.dat")
        );
        assert_eq!(
            anlz_key("/PIONEER/USBANLZ/a1b/2c3d/ANLZ0000.DAT"),
            anlz_key(PATH)
        );
        assert_eq!(anlz_key("C:/music/song.mp3"), None);
    }

    /// **A deck plays while its position moves**, and holds playing for
    /// 300 ms after; its beat is its position on the track's grid.
    #[test]
    fn a_deck_plays_while_its_position_moves() {
        let mut j = Judge::new();
        let t0 = Instant::now();
        let lib = lib();
        let s = j.judge(&[raw(120.0, 44_100, PATH), raw(0.0, 0, "")], 0, t0, &lib);
        assert_eq!(s.blind, None);
        assert_eq!(s.master, Some(1));
        let d = s.deck(1).unwrap();
        assert_eq!(d.track, Some(TrackId("t1".into())));
        assert_eq!(d.position_ms, 1000.0);
        assert_eq!(d.beat, Some(1.8), "(1000 − 100) ms at 500 ms a beat");
        assert!(!d.playing, "one sample is no movement");
        assert_eq!(s.deck(2).unwrap().track, None);

        let t1 = t0 + Duration::from_millis(5);
        let s = j.judge(&[raw(120.0, 44_320, PATH), raw(0.0, 0, "")], 0, t1, &lib);
        assert!(s.deck(1).unwrap().playing);
        assert_eq!(s.on_air().map(|d| d.number), Some(1));
        let held = j.judge(&[raw(120.0, 44_320, PATH), raw(0.0, 0, "")], 0, t1 + Duration::from_millis(200), &lib);
        assert!(held.deck(1).unwrap().playing, "inside the hold");
        let stopped = j.judge(&[raw(120.0, 44_320, PATH), raw(0.0, 0, "")], 0, t1 + Duration::from_millis(400), &lib);
        assert!(!stopped.deck(1).unwrap().playing);
        assert_eq!(stopped.on_air(), None);
    }

    /// **What cannot be believed blinds the reader**, naming the deck: an
    /// analysis file in no library track, a tempo out of range, a position
    /// a minute before the track.
    #[test]
    fn what_cannot_be_believed_blinds_the_reader() {
        let lib = lib();
        let mut j = Judge::new();
        let now = Instant::now();
        let other = "/PIONEER/USBANLZ/fff/0000/ANLZ0000.DAT";
        assert_eq!(
            j.judge(&[raw(120.0, 0, other)], 0, now, &lib).blind.as_deref(),
            Some("deck 1: pioneer/usbanlz/fff/0000/anlz0000.dat is in no library track")
        );
        assert_eq!(
            j.judge(&[raw(900.0, 0, PATH)], 0, now, &lib).blind.as_deref(),
            Some("deck 1: a tempo of 900")
        );
        assert_eq!(
            j.judge(&[raw(120.0, -44_100 * 61, PATH)], 0, now, &lib).blind.as_deref(),
            Some("deck 1: a position of -2690100")
        );
        assert_eq!(
            j.judge(&[raw(0.0, 0, "garbage")], 0, now, &lib).blind.as_deref(),
            Some("deck 1: `garbage` is no analysis file")
        );
        assert_eq!(
            j.judge(&[raw(0.0, 0, "")], 7, now, &lib).master,
            None,
            "no master chosen yet"
        );
    }
}
