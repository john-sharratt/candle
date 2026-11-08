//! The shortlist (§5.2): every candidate scored against this moment, in
//! milliseconds and with no model, so a hard rule can never pick a clash.
//! Out first — played tonight, an artist heard too recently, a tempo too far,
//! a key that clashes — then the rest scored on tempo and key fit, distance
//! from the curve's target, taste and balance, requests, and likeness to what
//! is landing now. The top entries go to Zen.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::budget::Budget;
use super::transition::tempo_gap;
use crate::music::{Profile, TrackId};

/// The widest tempo gap a candidate may have, allowing half and double time.
pub const TEMPO_MAX: f32 = 0.06;
/// How many recent tracks an artist must be absent from.
pub const ARTIST_GAP: usize = 6;
/// How many go to Zen.
pub const SHORTLIST: usize = 8;

const W_TEMPO: f32 = 0.15;
const W_KEY: f32 = 0.15;
const W_ENERGY: f32 = 0.30;
const W_TASTE: f32 = 0.20;
const W_REQUEST: f32 = 0.35;
const W_OWED: f32 = 0.30;
const W_LIKE_NOW: f32 = 0.10;
/// The bonus for a track every person likes.
const BOTH: f32 = 0.08;

/// What the shortlist is scored against.
pub struct Moment<'a> {
    /// The track the candidate follows.
    pub after: Option<&'a Profile>,
    /// The curve's target energy when the candidate will play.
    pub target: f32,
    pub played: &'a HashSet<TrackId>,
    /// The artists of the last tracks, newest first.
    pub recent_artists: &'a [String],
    /// Requested tracks and their urgency, `0..=1`.
    pub requests: &'a HashMap<TrackId, f32>,
    pub budget: &'a Budget,
    /// How well the room takes what is playing now, `0..=1`
    /// (`room::Room::landing`).
    pub landing: f32,
}

/// One candidate's score, with its parts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scored {
    pub id: TrackId,
    pub score: f32,
    pub tempo: f32,
    pub key: f32,
    pub energy: f32,
    pub taste: f32,
    pub request: f32,
    pub owed: f32,
    pub like_now: f32,
}

/// Why a candidate is out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Out {
    Played,
    Artist,
    Tempo,
    Key,
}

/// The hard rules: why `c` cannot follow now, if it cannot.
pub fn out(c: &Profile, m: &Moment<'_>) -> Option<Out> {
    if m.played.contains(&c.id) || m.after.is_some_and(|a| a.id == c.id) {
        return Some(Out::Played);
    }
    let recent = &m.recent_artists[..m.recent_artists.len().min(ARTIST_GAP)];
    if c.artists
        .iter()
        .any(|a| recent.iter().any(|r| r.eq_ignore_ascii_case(a)))
    {
        return Some(Out::Artist);
    }
    let after = m.after?;
    if tempo_gap(after.grid.bpm, c.grid.bpm) > TEMPO_MAX {
        return Some(Out::Tempo);
    }
    if let (Some(x), Some(y)) = (after.key, c.key) {
        if !x.fit(y).mixes() {
            return Some(Out::Key);
        }
    }
    None
}

/// Likeness to the playing track, `0..=1`: tags in common, the same crate,
/// years close.
fn likeness(a: &Profile, b: &Profile) -> f32 {
    let tags = if a.tags.is_empty() || b.tags.is_empty() {
        0.0
    } else {
        let shared = a.tags.iter().filter(|t| b.tags.contains(t)).count();
        shared as f32 / a.tags.len().max(b.tags.len()) as f32
    };
    let same_crate = (a.crate_name.is_some() && a.crate_name == b.crate_name) as u8 as f32;
    let years = match (a.year, b.year) {
        (Some(x), Some(y)) => (1.0 - (x.abs_diff(y) as f32 / 10.0)).max(0.0),
        _ => 0.0,
    };
    (tags + same_crate + years) / 3.0
}

/// `c`'s score now; `None` when a hard rule puts it out.
pub fn score(c: &Profile, m: &Moment<'_>) -> Option<Scored> {
    if out(c, m).is_some() {
        return None;
    }
    let (tempo, key, like_now) = match m.after {
        Some(a) => (
            1.0 - tempo_gap(a.grid.bpm, c.grid.bpm) / TEMPO_MAX,
            match (a.key, c.key) {
                (Some(x), Some(y)) => x.fit(y).score(),
                _ => 0.5,
            },
            likeness(a, c) * m.landing,
        ),
        None => (1.0, 1.0, 0.0),
    };
    let energy = 1.0 - (c.energy - m.target).abs();
    let people: Vec<&String> = m.budget.people().collect();
    let liked: Vec<f32> = people.iter().map(|p| c.liked_by(p)).collect();
    let mut taste = liked.iter().copied().fold(0.0, f32::max);
    if !liked.is_empty() && liked.iter().all(|w| *w > 0.0) {
        taste += BOTH;
    }
    let owed = people
        .iter()
        .zip(&liked)
        .map(|(p, w)| m.budget.owed(p) * w)
        .fold(0.0, f32::max);
    let request = m
        .requests
        .get(&c.id)
        .map_or(0.0, |u| 0.5 + 0.5 * u + m.budget.requests_owed());
    let score = W_TEMPO * tempo
        + W_KEY * key
        + W_ENERGY * energy
        + W_TASTE * taste
        + W_REQUEST * request
        + W_OWED * owed
        + W_LIKE_NOW * like_now;
    Some(Scored {
        id: c.id.clone(),
        score,
        tempo,
        key,
        energy,
        taste,
        request,
        owed,
        like_now,
    })
}

/// The best `n` of `candidates` now, best first; ties go to the id, so one
/// moment has one shortlist.
pub fn shortlist(candidates: &[Profile], m: &Moment<'_>, n: usize) -> Vec<Scored> {
    let mut scored: Vec<Scored> = candidates.iter().filter_map(|c| score(c, m)).collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    scored.truncate(n);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;
    use crate::music::Liked;

    fn by(mut t: Profile, who: &[(&str, f32)]) -> Profile {
        t.liked = who
            .iter()
            .map(|(p, w)| Liked {
                person: p.to_string(),
                weight: *w,
            })
            .collect();
        t
    }

    fn budget() -> Budget {
        Budget::new(
            [("john".to_string(), 0.4), ("yen".to_string(), 0.4)].into(),
            0.2,
        )
    }

    struct World {
        played: HashSet<TrackId>,
        recent: Vec<String>,
        requests: HashMap<TrackId, f32>,
        budget: Budget,
    }

    impl World {
        fn new() -> World {
            World {
                played: HashSet::new(),
                recent: Vec::new(),
                requests: HashMap::new(),
                budget: budget(),
            }
        }

        fn moment<'a>(&'a self, after: &'a Profile, target: f32) -> Moment<'a> {
            Moment {
                after: Some(after),
                target,
                played: &self.played,
                recent_artists: &self.recent,
                requests: &self.requests,
                budget: &self.budget,
                landing: 0.5,
            }
        }
    }

    /// **The hard rules put a clash out whatever else it has**: played
    /// tonight, an artist heard within six tracks, a tempo gap past 6%, a
    /// clashing key — even a request.
    #[test]
    fn hard_rules_put_clashes_out() {
        let now = club("now", 128.0, "8A", 0.6);
        let mut w = World::new();
        let played = club("played", 128.0, "8A", 0.6);
        w.played.insert(played.id.clone());
        let mut artist = club("artist", 128.0, "8A", 0.6);
        artist.artists = vec!["Daft Punk".into()];
        w.recent = vec!["x".into(), "daft punk".into()];
        let fast = club("fast", 140.0, "8A", 0.6);
        let double = club("double", 64.5, "8A", 0.6);
        let clash = club("clash", 128.0, "2B", 0.6);
        w.requests.insert(clash.id.clone(), 1.0);
        let m = w.moment(&now, 0.6);
        assert_eq!(out(&now, &m), Some(Out::Played), "the playing track");
        assert_eq!(out(&played, &m), Some(Out::Played));
        assert_eq!(out(&artist, &m), Some(Out::Artist));
        assert_eq!(out(&fast, &m), Some(Out::Tempo));
        assert_eq!(out(&double, &m), None, "half time is in");
        assert_eq!(out(&clash, &m), Some(Out::Key), "a request too");
    }

    /// **The score is its parts, exactly**: a candidate on tempo, key and
    /// target, liked by both.
    #[test]
    fn a_score_is_the_sum_of_its_weighted_parts() {
        let now = club("now", 128.0, "8A", 0.6);
        let c = by(club("c", 128.0, "8A", 0.7), &[("john", 0.5), ("yen", 0.25)]);
        let w = World::new();
        let s = score(&c, &w.moment(&now, 0.6)).unwrap();
        assert_eq!((s.tempo, s.key), (1.0, 1.0));
        assert!((s.energy - 0.9).abs() < 1e-6);
        assert!((s.taste - 0.58).abs() < 1e-6, "the most liked, and both");
        assert_eq!((s.request, s.owed), (0.0, 0.0));
        let want = W_TEMPO + W_KEY + W_ENERGY * 0.9 + W_TASTE * 0.58 + W_LIKE_NOW * s.like_now;
        assert!((s.score - want).abs() < 1e-6);
    }

    /// **The shortlist leans to whoever is owed and to requests**, and is
    /// ordered best first, ties by id.
    #[test]
    fn the_shortlist_leans_to_the_owed_and_the_requested() {
        let now = club("now", 128.0, "8A", 0.6);
        let johns = by(club("j", 128.0, "8A", 0.6), &[("john", 1.0)]);
        let yens = by(club("y", 128.0, "8A", 0.6), &[("yen", 1.0)]);
        let plain = club("p", 128.0, "8A", 0.6);
        let twin = club("q", 128.0, "8A", 0.6);
        let mut w = World::new();
        for i in 0..3 {
            w.budget.count(
                &by(club(&format!("h{i}"), 128.0, "8A", 0.6), &[("john", 1.0)]),
                false,
            );
        }
        let all = [johns.clone(), yens.clone(), plain.clone(), twin.clone()];
        let ids = |l: Vec<Scored>| l.into_iter().map(|s| s.id.0).collect::<Vec<_>>();
        assert_eq!(
            ids(shortlist(&all, &w.moment(&now, 0.6), 8)),
            ["y", "j", "p", "q"]
        );
        w.requests.insert(plain.id.clone(), 0.5);
        assert_eq!(ids(shortlist(&all, &w.moment(&now, 0.6), 2)), ["p", "y"]);
    }
}
