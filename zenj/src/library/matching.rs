//! A Spotify track found on TIDAL (§4.1, step 2): by its ISRC, accepted only
//! when the artist, title and length agree; otherwise by a search under the
//! same check, accepted only when one result agrees. What cannot be matched
//! is listed, never guessed.

use crate::tidal::Track;

/// The longest two lengths of one recording may differ by.
pub const LENGTH_SLACK_MS: u64 = 3_000;

/// What is looked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Wanted {
    pub title: String,
    pub artists: Vec<String>,
    pub isrc: Option<String>,
    pub duration_ms: u64,
}

/// A title or name reduced to what two catalogues agree on: lower case,
/// no bracketed parts (`(feat. …)`, `[Remastered]`), nothing after ` - `
/// (`- Radio Edit`), letters and digits only, single spaces.
pub fn normal(s: &str) -> String {
    let s = s.to_lowercase();
    let s = s.split(" - ").next().unwrap_or("");
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = (depth - 1).max(0),
            _ if depth > 0 => {}
            c if c.is_alphanumeric() => out.push(c),
            '&' => out.push_str(" and "),
            _ => out.push(' '),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `t` is `w`: some artist in common, the titles one within the
/// other, and the lengths within the slack when both are known.
pub fn agrees(w: &Wanted, t: &Track) -> bool {
    let (a, b) = (normal(&w.title), normal(&t.title));
    let titles = !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a));
    let artists = w.artists.iter().map(|x| normal(x)).any(|x| {
        t.artists.iter().map(|y| normal(y)).any(|y| !x.is_empty() && (x == y || x.contains(&y) || y.contains(&x)))
    });
    let lengths = w.duration_ms == 0
        || t.duration_ms == 0
        || w.duration_ms.abs_diff(t.duration_ms) <= LENGTH_SLACK_MS;
    titles && artists && lengths
}

/// The match among `by_isrc`, the tracks TIDAL holds under the wanted ISRC:
/// those that agree are one recording, released more than once, and the one
/// nearest in length is taken (the lower id on a tie).
pub fn by_isrc(w: &Wanted, by_isrc: &[Track]) -> Option<Track> {
    by_isrc
        .iter()
        .filter(|t| agrees(w, t))
        .min_by(|x, y| {
            let dx = x.duration_ms.abs_diff(w.duration_ms);
            let dy = y.duration_ms.abs_diff(w.duration_ms);
            dx.cmp(&dy).then_with(|| x.id.cmp(&y.id))
        })
        .cloned()
}

/// The match among a search's `results`: only when exactly one agrees.
pub fn by_search(w: &Wanted, results: &[Track]) -> Result<Track, String> {
    let agreeing: Vec<&Track> = results.iter().filter(|t| agrees(w, t)).collect();
    match agreeing.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(format!(
            "no TIDAL track agrees with {} — {}",
            w.artists.join(", "),
            w.title
        )),
        many => Err(format!(
            "{} TIDAL tracks agree with {} — {}: not guessed",
            many.len(),
            w.artists.join(", "),
            w.title
        )),
    }
}

/// The search ZenJ runs for `w`.
pub fn query(w: &Wanted) -> String {
    format!("{} {}", w.artists.first().map(String::as_str).unwrap_or(""), normal(&w.title))
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wanted(title: &str, artist: &str, ms: u64) -> Wanted {
        Wanted {
            title: title.into(),
            artists: vec![artist.into()],
            isrc: None,
            duration_ms: ms,
        }
    }

    fn tidal(id: &str, title: &str, artist: &str, ms: u64) -> Track {
        Track {
            id: id.into(),
            title: title.into(),
            artists: vec![artist.into()],
            isrc: None,
            duration_ms: ms,
            bpm: None,
            key: None,
        }
    }

    #[test]
    fn titles_and_names_reduce_to_what_catalogues_agree_on() {
        assert_eq!(normal("Don't Stop Me Now - Remastered 2011"), "don t stop me now");
        assert_eq!(normal("Titanium (feat. Sia)"), "titanium");
        assert_eq!(normal("One More Time [Radio Edit]"), "one more time");
        assert_eq!(normal("Simon & Garfunkel"), "simon and garfunkel");
        assert_eq!(normal("Beyoncé"), "beyoncé");
    }

    /// **An ISRC's tracks are one recording**: of those that agree, the
    /// nearest in length is taken; one that does not agree never is.
    #[test]
    fn an_isrc_takes_the_nearest_agreeing_release() {
        let w = wanted("Titanium (feat. Sia)", "David Guetta", 245_000);
        let found = by_isrc(
            &w,
            &[
                tidal("3", "Titanium", "David Guetta", 250_000),
                tidal("2", "Titanium (feat. Sia)", "David Guetta", 245_500),
                tidal("1", "Karaoke Titanium", "Karaoke Stars", 245_000),
            ],
        );
        assert_eq!(found.map(|t| t.id), Some("2".into()));
        assert_eq!(by_isrc(&w, &[tidal("9", "Titanium", "David Guetta", 400_000)]), None);
    }

    /// **A search is accepted only when exactly one result agrees.**
    #[test]
    fn a_search_is_accepted_only_when_one_agrees() {
        let w = wanted("Dancing Queen", "ABBA", 231_000);
        assert_eq!(
            by_search(&w, &[tidal("1", "Dancing Queen", "ABBA", 230_000), tidal("2", "Dancing Queen", "Cover Band", 231_000)])
                .unwrap()
                .id,
            "1"
        );
        assert_eq!(
            by_search(&w, &[]).unwrap_err(),
            "no TIDAL track agrees with ABBA — Dancing Queen"
        );
        assert_eq!(
            by_search(&w, &[tidal("1", "Dancing Queen", "ABBA", 230_000), tidal("2", "Dancing Queen - Live", "ABBA", 232_000)])
                .unwrap_err(),
            "2 TIDAL tracks agree with ABBA — Dancing Queen: not guessed"
        );
        assert_eq!(query(&w), "ABBA dancing queen");
    }
}
