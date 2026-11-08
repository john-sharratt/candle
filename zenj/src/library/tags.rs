//! What Zen knows of each song (§4.3): a singalong, a classic, an anthem, a
//! slow burner, its scene — asked once per track ahead of the night, in
//! batches, and kept in its profile, so at decision time it is a lookup and
//! never a model call. The prompt and the reading are pure; `zenj library
//! tag` sends them.

use std::collections::BTreeMap;
use std::fmt::Write;

use serde::Deserialize;

use crate::music::Profile;

/// Tracks asked about at once.
pub const BATCH: usize = 20;

/// The words a tag may be: what the crates and the shortlist read.
pub const VOCABULARY: [&str; 10] = [
    "singalong",
    "classic",
    "anthem",
    "slow-burner",
    "party-starter",
    "throwback",
    "chill",
    "romantic",
    "dance-floor",
    "cult",
];

pub const SYSTEM: &str = "You know popular music well. For each numbered track, say which of these \
describe it at a house party: singalong, classic, anthem, slow-burner, party-starter, throwback, \
chill, romantic, dance-floor, cult — none, one or several — and the year it came out if you know it. \
Answer with JSON only: {\"1\": {\"tags\": [...], \"year\": 1999}, \"2\": {...}}. Leave out a track \
you do not know.";

pub fn prompt(tracks: &[&Profile]) -> String {
    let mut p = String::new();
    for (i, t) in tracks.iter().enumerate() {
        let _ = writeln!(p, "{}. {} — {}", i + 1, t.artist_line(), t.title);
    }
    p
}

#[derive(Deserialize)]
struct Known {
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    year: Option<u16>,
}

/// Zen's answer for `count` tracks: each track's index (0-based), its tags
/// from the vocabulary, and its year when one between 1900 and 2027 is
/// given. Anything else is left out.
pub fn read(text: &str, count: usize) -> BTreeMap<usize, (Vec<String>, Option<u16>)> {
    let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) else {
        return BTreeMap::new();
    };
    let Ok(map) = serde_json::from_str::<BTreeMap<String, Known>>(&text[start..=end]) else {
        return BTreeMap::new();
    };
    map.into_iter()
        .filter_map(|(k, v)| {
            let i = k.trim().parse::<usize>().ok()?.checked_sub(1)?;
            (i < count).then(|| {
                let mut tags: Vec<String> = v
                    .tags
                    .into_iter()
                    .map(|t| t.trim().to_ascii_lowercase())
                    .filter(|t| VOCABULARY.contains(&t.as_str()))
                    .collect();
                tags.dedup();
                (i, (tags, v.year.filter(|y| (1900..=2027).contains(y))))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;

    #[test]
    fn the_prompt_numbers_each_track() {
        let a = club("a", 128.0, "8A", 0.5);
        let b = club("b", 128.0, "8A", 0.5);
        assert_eq!(
            prompt(&[&a, &b]),
            "1. a artist — a title\n2. b artist — b title\n"
        );
    }

    /// **Only the vocabulary and a plausible year are kept**, for tracks
    /// that were asked about.
    #[test]
    fn an_answer_keeps_the_vocabulary_and_a_plausible_year() {
        let got = read(
            r#"Here: {"1": {"tags": ["Singalong", "anthem", "banger"], "year": 1976},
                      "2": {"tags": [], "year": 3000}, "9": {"tags": ["classic"]}, "x": {}}"#,
            2,
        );
        assert_eq!(
            got[&0],
            (
                vec!["singalong".to_string(), "anthem".to_string()],
                Some(1976)
            )
        );
        assert_eq!(got[&1], (vec![], None));
        assert_eq!(got.len(), 2);
        assert!(read("no idea", 2).is_empty());
    }
}
