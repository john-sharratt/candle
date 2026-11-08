//! The library ZenJ plays from: every matched track's profile, kept as one
//! JSON file in its data folder (`library.json`) and read whole at start. It
//! is built ahead of the night (`library::sync`) and never computed live.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::matching::normal;
use crate::deck::{anlz_key, Tracks};
use crate::music::{Grid, Profile, TrackId};

/// A track that could not be matched, kept so it is listed, not lost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unmatched {
    pub title: String,
    pub artist: String,
    pub why: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Library {
    pub tracks: Vec<Profile>,
    #[serde(default)]
    pub unmatched: Vec<Unmatched>,
    /// Each crate's TIDAL playlist id, by crate name.
    #[serde(default)]
    pub playlists: HashMap<String, String>,
    #[serde(skip)]
    by_id: HashMap<TrackId, usize>,
    #[serde(skip)]
    by_anlz: HashMap<String, usize>,
}

impl Library {
    pub fn new(tracks: Vec<Profile>) -> Library {
        let mut l = Library {
            tracks,
            ..Library::default()
        };
        l.index();
        l
    }

    fn index(&mut self) {
        self.by_id = self
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.clone(), i))
            .collect();
        self.by_anlz = self
            .tracks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| Some((anlz_key(t.anlz.as_deref()?)?, i)))
            .collect();
    }

    pub fn load(path: &Path) -> Result<Library, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut l: Library =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        l.index();
        Ok(l)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let tmp = path.with_extension("tmp");
        let text = serde_json::to_string_pretty(self).expect("a library serializes");
        fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn get(&self, id: &TrackId) -> Option<&Profile> {
        self.by_id.get(id).map(|&i| &self.tracks[i])
    }

    pub fn get_mut(&mut self, id: &TrackId) -> Option<&mut Profile> {
        let i = *self.by_id.get(id)?;
        Some(&mut self.tracks[i])
    }

    /// Add or replace `t`.
    pub fn put(&mut self, t: Profile) {
        match self.by_id.get(&t.id) {
            Some(&i) => self.tracks[i] = t,
            None => self.tracks.push(t),
        }
        self.index();
    }

    /// The tracks whose artist and title hold every word of `query`, best
    /// first: those whose title is the query's rest.
    pub fn search(&self, query: &str) -> Vec<&Profile> {
        let words: Vec<String> = normal(query).split(' ').filter(|w| !w.is_empty()).map(String::from).collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<(&Profile, usize)> = self
            .tracks
            .iter()
            .filter_map(|t| {
                let hay = normal(&format!("{} {}", t.artist_line(), t.title));
                let hay_words: Vec<&str> = hay.split(' ').collect();
                words
                    .iter()
                    .all(|w| hay_words.contains(&w.as_str()))
                    .then(|| (t, hay_words.len() - words.len()))
            })
            .collect();
        hits.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.id.cmp(&b.0.id)));
        hits.into_iter().map(|(t, _)| t).collect()
    }

    /// The tracks ready to mix: analysed, so their grid and phrases are
    /// rekordbox's.
    pub fn playable(&self) -> Vec<Profile> {
        self.tracks.iter().filter(|t| t.anlz.is_some()).cloned().collect()
    }
}

impl Tracks for Library {
    fn by_anlz(&self, key: &str) -> Option<(TrackId, Grid)> {
        let t = &self.tracks[*self.by_anlz.get(key)?];
        Some((t.id.clone(), t.grid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;

    fn lib() -> Library {
        let mut a = club("a", 128.0, "8A", 0.6);
        a.title = "Dancing Queen".into();
        a.artists = vec!["ABBA".into()];
        a.anlz = Some("/PIONEER/USBANLZ/x/ANLZ0000.DAT".into());
        let mut b = club("b", 128.0, "8A", 0.6);
        b.title = "Dancing Queen (Live)".into();
        b.artists = vec!["ABBA".into()];
        let mut c = club("c", 128.0, "8A", 0.6);
        c.title = "Waterloo".into();
        c.artists = vec!["ABBA".into()];
        Library::new(vec![a, b, c])
    }

    #[test]
    fn a_library_finds_tracks_by_id_words_and_analysis() {
        let l = lib();
        assert_eq!(l.get(&TrackId("c".into())).unwrap().title, "Waterloo");
        let hits: Vec<&str> = l.search("abba dancing queen").iter().map(|t| t.id.as_str()).collect();
        assert_eq!(hits, ["a", "b"]);
        assert!(l.search("queen waterloo").is_empty());
        let (id, grid) = l.by_anlz("pioneer/usbanlz/x/anlz0000.dat").unwrap();
        assert_eq!((id.as_str(), grid.bpm), ("a", 128.0));
        assert_eq!(l.playable().len(), 1);
    }

    #[test]
    fn a_library_is_kept_and_read_back_with_its_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library.json");
        let mut l = lib();
        l.unmatched.push(Unmatched {
            title: "x".into(),
            artist: "y".into(),
            why: "z".into(),
        });
        l.save(&path).unwrap();
        let back = Library::load(&path).unwrap();
        assert_eq!(back.tracks, l.tracks);
        assert_eq!(back.unmatched, l.unmatched);
        assert!(back.by_anlz("pioneer/usbanlz/x/anlz0000.dat").is_some());
        let mut back = back;
        let mut a = back.get(&TrackId("a".into())).unwrap().clone();
        a.energy = 0.9;
        back.put(a);
        assert_eq!(back.tracks.len(), 3);
        assert_eq!(back.get(&TrackId("a".into())).unwrap().energy, 0.9);
    }
}
