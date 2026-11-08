//! Building the library ahead of the night (§4.1): each person's taste from
//! their export and their Spotify, the tracks they like most found on TIDAL,
//! filed into crates there — and, once rekordbox has analysed them, each
//! track's grid, phrases, cues and key taken from its analysis.
//!
//! `zenj library sync` runs [`build`] and [`file_crates`]; `zenj library
//! analyse` runs [`analyse`] after rekordbox has analysed the crates.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use super::crates::{crates_of, ALL};
use super::energy::estimate;
use super::matching::{by_isrc, by_search, query, Wanted};
use super::store::{Library, Unmatched};
use super::taste::weight;
use crate::music::{Camelot, Grid, Liked, Profile, TrackId};
use crate::rekordbox::analysis::file;
use crate::rekordbox::{analyse as read_analysis, Content};
use crate::spotify::api::Artist;
use crate::spotify::export::{read_dir, Heard, Taste};
use crate::spotify::Spotify;
use crate::tidal::Tidal;

/// One person whose taste the library is built from.
pub struct Person {
    pub name: String,
    /// Their unpacked privacy export, when they have one.
    pub export: Option<PathBuf>,
    pub spotify: Option<Spotify>,
}

/// The tempo a profile carries before rekordbox has analysed its track,
/// when TIDAL gives none.
const UNANALYSED_BPM: f32 = 120.0;

/// One person's taste: their export, then their liked tracks and their
/// playlists from the API.
async fn taste_of(p: &Person) -> Result<Taste, String> {
    let mut t = match &p.export {
        Some(dir) => read_dir(dir)?,
        None => Taste::new(),
    };
    if let Some(s) = &p.spotify {
        for saved in s.liked().await? {
            if let Some(id) = saved.track.id.clone() {
                let h = t.entry(id).or_default();
                h.liked = true;
                fill(h, &saved.track.name, &saved.track.artists);
            }
        }
        for pl in s.playlists().await? {
            for item in s.items(&pl.id).await? {
                let Some(track) = item.track else { continue };
                let Some(id) = track.id.clone() else { continue };
                let h = t.entry(id).or_default();
                if !h.playlists.contains(&pl.name) {
                    h.playlists.push(pl.name.clone());
                }
                fill(h, &track.name, &track.artists);
            }
        }
    }
    Ok(t)
}

fn fill(h: &mut Heard, title: &str, artists: &[Artist]) {
    if h.title.is_empty() {
        h.title = title.to_string();
        h.artist = artists.first().map(|a| a.name.clone()).unwrap_or_default();
    }
}

/// The ids each person likes most, at most `limit` each, with every
/// person's weight for each.
pub fn candidates(
    tastes: &[(String, Taste)],
    limit: usize,
) -> BTreeMap<String, Vec<(String, f32)>> {
    let mut out: BTreeMap<String, Vec<(String, f32)>> = BTreeMap::new();
    for (_, taste) in tastes {
        let mut w: Vec<(&String, f32)> = taste
            .iter()
            .map(|(id, h)| (id, weight(h)))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        w.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        for (id, _) in w.into_iter().take(limit) {
            out.entry(id.clone()).or_default();
        }
    }
    for (id, likes) in out.iter_mut() {
        for (person, taste) in tastes {
            if let Some(h) = taste.get(id) {
                let w = weight(h);
                if w > 0.0 {
                    likes.push((person.clone(), w));
                }
            }
        }
    }
    out
}

/// The library built from `people`'s taste, matched on `tidal`.
pub async fn build(people: &[Person], tidal: &Tidal, limit: usize) -> Result<Library, String> {
    let mut tastes = Vec::new();
    for p in people {
        let t = taste_of(p).await?;
        tracing::info!(person = %p.name, tracks = t.len(), "library: taste read");
        tastes.push((p.name.clone(), t));
    }
    let wanted = candidates(&tastes, limit);
    let api = people
        .iter()
        .find_map(|p| p.spotify.as_ref())
        .ok_or("no one is signed in to Spotify: the tracks' ISRCs come from its API")?;
    let ids: Vec<String> = wanted.keys().cloned().collect();
    let tracks = api.tracks(&ids).await?;
    let names: Vec<String> = people.iter().map(|p| p.name.clone()).collect();
    let mut lib = Library::default();
    for t in tracks {
        let Some(sid) = t.id.clone() else { continue };
        let w = Wanted {
            title: t.name.clone(),
            artists: t.artists.iter().map(|a| a.name.clone()).collect(),
            isrc: t.external_ids.isrc.clone(),
            duration_ms: t.duration_ms,
        };
        let found = match &w.isrc {
            Some(isrc) => by_isrc(&w, &tidal.by_isrc(isrc).await?),
            None => None,
        };
        let found = match found {
            Some(f) => Ok(f),
            None => by_search(&w, &tidal.search(&query(&w)).await?),
        };
        let found = match found {
            Ok(f) => f,
            Err(why) => {
                lib.unmatched.push(Unmatched {
                    title: w.title.clone(),
                    artist: w.artists.join(", "),
                    why,
                });
                continue;
            }
        };
        let bpm = found.bpm.unwrap_or(UNANALYSED_BPM);
        let mut p = Profile {
            id: TrackId(
                w.isrc
                    .clone()
                    .unwrap_or_else(|| format!("tidal:{}", found.id)),
            ),
            title: w.title.clone(),
            artists: w.artists.clone(),
            isrc: w.isrc.clone(),
            tidal_id: Some(found.id.clone()),
            spotify_id: Some(sid.clone()),
            duration_ms: found.duration_ms.max(w.duration_ms),
            grid: Grid {
                bpm,
                first_beat_ms: 0.0,
            },
            key: found.key,
            energy: estimate(bpm, None, &[]),
            phrases: Vec::new(),
            cues: Vec::new(),
            year: t.year(),
            tags: Vec::new(),
            liked: wanted
                .get(&sid)
                .into_iter()
                .flatten()
                .map(|(person, w)| Liked {
                    person: person.clone(),
                    weight: *w,
                })
                .collect(),
            crate_name: None,
            anlz: None,
        };
        p.crate_name = crates_of(&p, &names).first().map(|c| c.to_string());
        lib.put(p);
    }
    Ok(lib)
}

/// Each crate's playlist on TIDAL made if it is missing, and every track
/// that belongs in it added.
pub async fn file_crates(
    lib: &mut Library,
    tidal: &Tidal,
    people: &[String],
) -> Result<(), String> {
    for name in ALL {
        let id = match lib.playlists.get(name) {
            Some(id) => id.clone(),
            None => {
                let id = tidal
                    .create_playlist(name, "Filed by ZenJ for New Year's Eve")
                    .await?;
                lib.playlists.insert(name.to_string(), id.clone());
                id
            }
        };
        let have: Vec<String> = tidal
            .items(&id)
            .await?
            .into_iter()
            .map(|i| i.track)
            .collect();
        let missing: Vec<String> = lib
            .tracks
            .iter()
            .filter(|t| crates_of(t, people).contains(&name))
            .filter_map(|t| t.tidal_id.clone())
            .filter(|t| !have.contains(t))
            .collect();
        tidal.add(&id, &missing, None).await?;
        tracing::info!(
            crate_name = name,
            added = missing.len(),
            "library: crate filed"
        );
    }
    Ok(())
}

/// What an analysis pass did.
#[derive(Debug, Default, PartialEq)]
pub struct Analysed {
    pub analysed: usize,
    /// Tracks rekordbox lists none of, or has not analysed.
    pub waiting: Vec<String>,
    pub failed: Vec<String>,
}

/// Each library track's analysis from rekordbox's `contents` and the
/// analysis files under `share`, matched by ISRC, then by TIDAL id.
pub fn analyse(
    lib: &mut Library,
    contents: &[Content],
    share: &Path,
    people: &[String],
) -> Analysed {
    let by_isrc: HashMap<&str, &Content> = contents
        .iter()
        .filter_map(|c| Some((c.isrc.as_deref()?, c)))
        .collect();
    let by_tidal: HashMap<String, &Content> = contents
        .iter()
        .filter_map(|c| Some((c.tidal_id()?, c)))
        .collect();
    let mut out = Analysed::default();
    let ids: Vec<TrackId> = lib.tracks.iter().map(|t| t.id.clone()).collect();
    for id in ids {
        let Some(t) = lib.get(&id).cloned() else {
            continue;
        };
        let content = t
            .isrc
            .as_deref()
            .and_then(|i| by_isrc.get(i))
            .or_else(|| t.tidal_id.as_ref().and_then(|i| by_tidal.get(i)));
        let Some(c) = content else {
            out.waiting
                .push(format!("{} — {}", t.artist_line(), t.title));
            continue;
        };
        let Some(anlz) = &c.anlz else {
            out.waiting
                .push(format!("{} — {}", t.artist_line(), t.title));
            continue;
        };
        match read_analysis(&file(share, anlz)) {
            Ok(a) => {
                let mut p = t.clone();
                p.grid = a.grid;
                p.phrases = a.phrases;
                p.cues = a.cues;
                if let Some(k) = c.key.as_deref().and_then(Camelot::parse) {
                    p.key = Some(k);
                }
                if p.year.is_none() {
                    p.year = c.year;
                }
                p.energy = estimate(p.grid.bpm, a.mood, &p.phrases);
                p.anlz = Some(anlz.clone());
                p.crate_name = crates_of(&p, people).first().map(|c| c.to_string());
                lib.put(p);
                out.analysed += 1;
            }
            Err(e) => out.failed.push(e),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::deck::Tracks;
    use crate::music::track::fixture::club;
    use crate::rekordbox::anlz::build as anlz_build;

    fn heard(listens: u32, liked: bool) -> Heard {
        Heard {
            title: "t".into(),
            artist: "a".into(),
            listens,
            liked,
            ..Heard::default()
        }
    }

    /// **Each person's favourites are candidates, and each candidate carries
    /// every person's weight** — a track only Yen chose still says John likes
    /// it a little.
    #[test]
    fn candidates_are_each_persons_favourites_with_everyones_weight() {
        let john: Taste = [("A".into(), heard(40, true)), ("B".into(), heard(2, false))].into();
        let yen: Taste = [("C".into(), heard(30, true)), ("B".into(), heard(1, false))].into();
        let c = candidates(&[("john".into(), john), ("yen".into(), yen)], 1);
        assert_eq!(c.keys().collect::<Vec<_>>(), ["A", "C"]);
        assert_eq!(c["A"].len(), 1);
        let c = candidates(
            &[
                ("john".into(), [("B".to_string(), heard(2, false))].into()),
                ("yen".into(), [("B".to_string(), heard(1, false))].into()),
            ],
            5,
        );
        assert_eq!(
            c["B"].iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
            ["john", "yen"]
        );
    }

    /// **rekordbox's analysis lands on the track it is of**, by ISRC: its
    /// grid, phrases, cues and key, and its energy from its mood; a track
    /// rekordbox has not analysed waits.
    #[test]
    fn rekordboxs_analysis_lands_on_its_track() {
        let dir = tempfile::tempdir().unwrap();
        let share = dir.path();
        let rel = "/PIONEER/USBANLZ/a1b/2c3d/ANLZ0000.DAT";
        let dat = file(share, rel);
        fs::create_dir_all(dat.parent().unwrap()).unwrap();
        fs::write(
            &dat,
            anlz_build::file(&[anlz_build::grid(&[(1, 12400, 40), (2, 12400, 524)])]),
        )
        .unwrap();
        fs::write(
            dat.with_extension("EXT"),
            anlz_build::file(&[anlz_build::phrases(
                1,
                129,
                &[(1, 1), (33, 5), (97, 6)],
                true,
            )]),
        )
        .unwrap();
        let mut t = club("ISRC1", 120.0, "8A", 0.5);
        t.isrc = Some("ISRC1".into());
        let mut w = club("ISRC2", 120.0, "8A", 0.5);
        w.isrc = Some("ISRC2".into());
        let mut lib = Library::new(vec![t, w]);
        let content = Content {
            id: "1".into(),
            title: "x".into(),
            artist: "y".into(),
            bpm: 124.0,
            key: Some("Dm".into()),
            length_s: 200,
            isrc: Some("ISRC1".into()),
            folder_path: String::new(),
            anlz: Some(rel.into()),
            year: Some(2010),
        };
        let people = vec!["john".to_string()];
        let r = analyse(&mut lib, &[content], share, &people);
        assert_eq!(r.analysed, 1);
        assert_eq!(r.waiting, vec!["ISRC2 artist — ISRC2 title".to_string()]);
        let p = lib.get(&TrackId("ISRC1".into())).unwrap();
        assert_eq!(
            p.grid,
            Grid {
                bpm: 124.0,
                first_beat_ms: 40.0
            }
        );
        assert_eq!(p.key, Camelot::parse("7A"));
        assert_eq!(p.phrases.len(), 3);
        assert_eq!(p.anlz.as_deref(), Some(rel));
        assert_eq!(p.year, Some(2010));
        assert!(p.energy > 0.5, "a high mood: {}", p.energy);
        assert!(lib
            .by_anlz("pioneer/usbanlz/a1b/2c3d/anlz0000.dat")
            .is_some());
    }
}
