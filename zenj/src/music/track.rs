//! A track as the brain knows it (§4.3): who it is, how it is built — tempo,
//! key, beat grid, phrases, cues — how much energy it carries, and whose it
//! is. A profile is computed ahead of time and read at decision time; nothing
//! here is measured live.

use serde::{Deserialize, Serialize};

use super::key::Camelot;

/// A track's id in the library: its ISRC when it has one, which every service
/// agrees on, else its TIDAL id as `tidal:<id>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackId(pub String);

impl TrackId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TrackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a stretch of a track is, as rekordbox's phrase analysis names it,
/// folded across its three moods into what a mix needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhraseKind {
    Intro,
    /// A verse, or a build towards the chorus.
    Up,
    /// A breakdown.
    Down,
    Bridge,
    Chorus,
    Outro,
}

impl PhraseKind {
    /// How much a phrase of this kind carries, `0..=1`, for the energy curve
    /// over a track.
    pub fn weight(self) -> f32 {
        match self {
            PhraseKind::Intro | PhraseKind::Outro => 0.4,
            PhraseKind::Down => 0.35,
            PhraseKind::Bridge => 0.55,
            PhraseKind::Up => 0.7,
            PhraseKind::Chorus => 1.0,
        }
    }
}

/// One phrase: from `start` up to `end`, in beats from the track's first
/// beat (beat 0 is the grid's first downbeat).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Phrase {
    pub kind: PhraseKind,
    pub start: u32,
    pub end: u32,
}

impl Phrase {
    pub fn beats(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }
}

/// One hot cue rekordbox holds for the track: pad `pad` (1–8) at `beat`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HotCue {
    pub pad: u8,
    pub beat: f64,
}

/// The beat grid, as a constant tempo from a first downbeat: what rekordbox's
/// analysis gives a quantised track, and what a mix plans against.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub bpm: f32,
    /// Where beat 0 falls, in milliseconds from the start of the audio.
    pub first_beat_ms: f64,
}

impl Grid {
    pub fn ms_per_beat(&self) -> f64 {
        60_000.0 / self.bpm as f64
    }

    /// The beat at `ms` into the audio; negative before the first.
    pub fn beat_at(&self, ms: f64) -> f64 {
        (ms - self.first_beat_ms) / self.ms_per_beat()
    }

    /// The time of beat `beat`, in milliseconds into the audio.
    pub fn ms_at(&self, beat: f64) -> f64 {
        self.first_beat_ms + beat * self.ms_per_beat()
    }
}

/// How much a person likes a track: their plays and likes folded into
/// `0..=1` (`library::taste`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Liked {
    pub person: String,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub id: TrackId,
    pub title: String,
    pub artists: Vec<String>,
    #[serde(default)]
    pub isrc: Option<String>,
    #[serde(default)]
    pub tidal_id: Option<String>,
    #[serde(default)]
    pub spotify_id: Option<String>,
    pub duration_ms: u64,
    pub grid: Grid,
    #[serde(default)]
    pub key: Option<Camelot>,
    /// The whole track's energy, `0..=1`.
    pub energy: f32,
    #[serde(default)]
    pub phrases: Vec<Phrase>,
    #[serde(default)]
    pub cues: Vec<HotCue>,
    #[serde(default)]
    pub year: Option<u16>,
    /// What Zen knows of it: `singalong`, `classic`, `slow-burner`, a scene.
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub liked: Vec<Liked>,
    /// The crate it is filed in (`library::crates`).
    #[serde(default)]
    pub crate_name: Option<String>,
    /// Its analysis file in rekordbox (`deck::anlz_key`), once rekordbox has
    /// analysed it: how a deck names the track it holds.
    #[serde(default)]
    pub anlz: Option<String>,
}

impl Profile {
    /// The track's length in beats.
    pub fn beats(&self) -> u32 {
        self.grid.beat_at(self.duration_ms as f64).max(0.0).floor() as u32
    }

    /// How much `person` likes it, 0 when they never played or liked it.
    pub fn liked_by(&self, person: &str) -> f32 {
        self.liked
            .iter()
            .find(|l| l.person == person)
            .map_or(0.0, |l| l.weight)
    }

    /// The first phrase of `kind` starting at or after `beat`.
    pub fn phrase_from(&self, kind: PhraseKind, beat: u32) -> Option<&Phrase> {
        self.phrases
            .iter()
            .find(|p| p.kind == kind && p.start >= beat)
    }

    /// The phrase boundaries — every phrase's start, and the last one's end
    /// — or, for a track with no phrases, every 32nd beat.
    pub fn boundaries(&self) -> Vec<u32> {
        if self.phrases.is_empty() {
            return (0..=self.beats()).step_by(32).collect();
        }
        let mut b: Vec<u32> = self.phrases.iter().map(|p| p.start).collect();
        if let Some(last) = self.phrases.last() {
            b.push(last.end);
        }
        b.dedup();
        b
    }

    /// The hot cue nearest after `beat`.
    pub fn cue_at_or_after(&self, beat: f64) -> Option<&HotCue> {
        self.cues
            .iter()
            .filter(|c| c.beat >= beat)
            .min_by(|a, b| a.beat.total_cmp(&b.beat))
    }

    /// Its artists as one line.
    pub fn artist_line(&self) -> String {
        self.artists.join(", ")
    }
}

#[cfg(test)]
pub mod fixture {
    use super::*;

    /// A track of `beats` beats at `bpm`, starting on the audio's first
    /// millisecond, with the phrases given as `(kind, start, end)`.
    pub fn track(
        id: &str,
        bpm: f32,
        key: &str,
        energy: f32,
        beats: u32,
        phrases: &[(PhraseKind, u32, u32)],
    ) -> Profile {
        let grid = Grid {
            bpm,
            first_beat_ms: 0.0,
        };
        Profile {
            id: TrackId(id.into()),
            title: format!("{id} title"),
            artists: vec![format!("{id} artist")],
            isrc: None,
            tidal_id: Some(format!("{id}-tidal")),
            spotify_id: None,
            duration_ms: grid.ms_at(beats as f64).round() as u64,
            grid,
            key: Camelot::parse(key),
            energy,
            phrases: phrases
                .iter()
                .map(|&(kind, start, end)| Phrase { kind, start, end })
                .collect(),
            cues: Vec::new(),
            year: None,
            tags: Vec::new(),
            liked: Vec::new(),
            crate_name: None,
            anlz: None,
        }
    }

    /// The shape of a club track: 32 beats of intro, up, chorus, down,
    /// chorus, 32 of outro — 256 beats.
    pub fn club(id: &str, bpm: f32, key: &str, energy: f32) -> Profile {
        use PhraseKind::*;
        track(
            id,
            bpm,
            key,
            energy,
            256,
            &[
                (Intro, 0, 32),
                (Up, 32, 64),
                (Chorus, 64, 128),
                (Down, 128, 160),
                (Chorus, 160, 224),
                (Outro, 224, 256),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    #[test]
    fn the_grid_maps_beats_and_time_both_ways() {
        let g = Grid {
            bpm: 120.0,
            first_beat_ms: 250.0,
        };
        assert_eq!(g.ms_per_beat(), 500.0);
        assert_eq!(g.beat_at(250.0), 0.0);
        assert_eq!(g.beat_at(1250.0), 2.0);
        assert_eq!(g.beat_at(0.0), -0.5);
        assert_eq!(g.ms_at(4.0), 2250.0);
    }

    #[test]
    fn a_track_knows_its_length_phrases_and_boundaries() {
        let t = club("a", 128.0, "8A", 0.7);
        assert_eq!(t.beats(), 256);
        assert_eq!(t.phrase_from(PhraseKind::Chorus, 65).unwrap().start, 160);
        assert_eq!(t.phrase_from(PhraseKind::Outro, 0).unwrap().beats(), 32);
        assert_eq!(t.boundaries(), vec![0, 32, 64, 128, 160, 224, 256]);
        let bare = track("b", 120.0, "8A", 0.5, 100, &[]);
        assert_eq!(bare.boundaries(), vec![0, 32, 64, 96]);
    }

    #[test]
    fn a_profile_reads_back_from_its_json() {
        let mut t = club("a", 128.0, "8A", 0.7);
        t.liked.push(Liked {
            person: "yen".into(),
            weight: 0.8,
        });
        let back: Profile = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.liked_by("yen"), 0.8);
        assert_eq!(back.liked_by("john"), 0.0);
    }
}
