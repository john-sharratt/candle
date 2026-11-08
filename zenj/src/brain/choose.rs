//! Zen's choice (§5.2): the shortlist, put to Zen with what only it has —
//! the conversation, the moment, a request's note — and its answer read
//! back. The prompt and the reading are pure, so both are tested to the
//! character; the call itself is `zen::Zen`'s, under a deadline, and when it
//! does not answer in time the shortlist's top entry stands.

use std::fmt::Write;

use serde::Deserialize;

use super::shortlist::Scored;
use crate::music::{Profile, TrackId};

/// One shortlisted track as Zen reads it.
pub struct Offer<'a> {
    pub track: &'a Profile,
    pub scored: &'a Scored,
    /// Who asked for it, and their note, when it is a request.
    pub request: Option<(&'a str, &'a str)>,
}

/// What the prompt says of the moment.
pub struct Scene<'a> {
    /// `22:41`.
    pub clock: &'a str,
    pub to_midnight_min: i64,
    pub playing: Option<&'a Profile>,
    pub target: f32,
    /// The conversation's last lines, oldest first: `who: text`.
    pub said: &'a [String],
    /// Whose music is owed, and how much.
    pub owed: &'a [(String, f32)],
}

/// The system prompt: who Zen is in this job, and the one form of answer.
pub const SYSTEM: &str = "You are Zen, DJing a New Year's Eve house party for John and Yen. \
You choose the next track from a shortlist that already fits the mix. Weigh what the room \
said, the moment and the night's arc, who has not heard one of theirs in a while, and any \
request's note. Answer with JSON only, on one line: {\"pick\": <the number of a shortlisted \
track>, \"why\": \"<one short line for the screens>\"}. If someone asked for a specific song \
that is not listed, answer {\"search\": \"<artist> - <title>\", \"why\": \"...\"} instead.";

fn track_line(t: &Profile) -> String {
    let key = t.key.map(|k| k.to_string()).unwrap_or_else(|| "?".into());
    let year = t.year.map(|y| format!(", {y}")).unwrap_or_default();
    let tags = if t.tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", t.tags.join(", "))
    };
    format!(
        "{} — {} ({:.0} BPM, {key}, energy {:.2}{year}){tags}",
        t.artist_line(),
        t.title,
        t.grid.bpm,
        t.energy
    )
}

/// The user prompt for `scene` and its `offers`.
pub fn prompt(scene: &Scene<'_>, offers: &[Offer<'_>]) -> String {
    let mut p = String::new();
    let _ = writeln!(
        p,
        "It is {}, {} minutes to midnight. Target energy {:.2}.",
        scene.clock, scene.to_midnight_min, scene.target
    );
    if let Some(t) = scene.playing {
        let _ = writeln!(p, "Playing: {}", track_line(t));
    }
    if !scene.owed.is_empty() {
        let owed: Vec<String> = scene
            .owed
            .iter()
            .map(|(who, by)| format!("{who} {by:.2}"))
            .collect();
        let _ = writeln!(p, "Owed a track: {}", owed.join(", "));
    }
    if !scene.said.is_empty() {
        let _ = writeln!(p, "The room said:");
        for line in scene.said {
            let _ = writeln!(p, "  {line}");
        }
    }
    let _ = writeln!(p, "Shortlist:");
    for (i, o) in offers.iter().enumerate() {
        let liked: Vec<String> = o
            .track
            .liked
            .iter()
            .filter(|l| l.weight > 0.0)
            .map(|l| l.person.clone())
            .collect();
        let liked = if liked.is_empty() {
            String::new()
        } else {
            format!(", liked by {}", liked.join(" and "))
        };
        let req = match o.request {
            Some((who, "")) => format!(", requested by {who}"),
            Some((who, note)) => format!(", requested by {who}: \"{note}\""),
            None => String::new(),
        };
        let _ = writeln!(
            p,
            "{}. {}{liked}{req} — score {:.2}",
            i + 1,
            track_line(o.track),
            o.scored.score
        );
    }
    p
}

/// What Zen chose.
#[derive(Debug, Clone, PartialEq)]
pub enum Choice {
    Pick { id: TrackId, why: String },
    Search { query: String, why: String },
}

#[derive(Deserialize)]
struct Said {
    #[serde(default)]
    pick: Option<usize>,
    #[serde(default)]
    search: Option<String>,
    #[serde(default)]
    why: String,
}

/// Zen's answer `text`, read against the `offers` it was given; `None` when
/// it named nothing that can be played — the caller takes the top entry.
pub fn read(text: &str, offers: &[Offer<'_>]) -> Option<Choice> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let said: Said = serde_json::from_str(text.get(start..=end)?).ok()?;
    let why = said.why.trim().to_string();
    if let Some(n) = said.pick {
        let o = offers.get(n.checked_sub(1)?)?;
        return Some(Choice::Pick {
            id: o.track.id.clone(),
            why,
        });
    }
    let query = said.search?.trim().to_string();
    (!query.is_empty()).then_some(Choice::Search { query, why })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;
    use crate::music::Liked;

    fn scored(id: &str, score: f32) -> Scored {
        Scored {
            id: TrackId(id.into()),
            score,
            tempo: 1.0,
            key: 1.0,
            energy: 1.0,
            taste: 0.0,
            request: 0.0,
            owed: 0.0,
            like_now: 0.0,
        }
    }

    /// **The prompt is exact**: the moment, what is owed, what was said, and
    /// each offer with whose it is and any request's note.
    #[test]
    fn the_prompt_says_the_moment_and_the_offers() {
        let now = club("now", 128.0, "8A", 0.6);
        let mut a = club("a", 127.0, "9A", 0.7);
        a.liked = vec![
            Liked {
                person: "john".into(),
                weight: 1.0,
            },
            Liked {
                person: "yen".into(),
                weight: 0.3,
            },
        ];
        a.year = Some(1999);
        a.tags = vec!["singalong".into()];
        let b = club("b", 128.0, "8B", 0.65);
        let (sa, sb) = (scored("a", 0.91), scored("b", 0.8));
        let offers = [
            Offer {
                track: &a,
                scored: &sa,
                request: None,
            },
            Offer {
                track: &b,
                scored: &sb,
                request: Some(("yen", "for mum")),
            },
        ];
        let said = ["yen: more energy please".to_string()];
        let owed = [("yen".to_string(), 0.25)];
        let p = prompt(
            &Scene {
                clock: "22:41",
                to_midnight_min: 79,
                playing: Some(&now),
                target: 0.72,
                said: &said,
                owed: &owed,
            },
            &offers,
        );
        assert_eq!(
            p,
            "It is 22:41, 79 minutes to midnight. Target energy 0.72.\n\
             Playing: now artist — now title (128 BPM, 8A, energy 0.60)\n\
             Owed a track: yen 0.25\n\
             The room said:\n  yen: more energy please\n\
             Shortlist:\n\
             1. a artist — a title (127 BPM, 9A, energy 0.70, 1999) [singalong], liked by john and yen — score 0.91\n\
             2. b artist — b title (128 BPM, 8B, energy 0.65), requested by yen: \"for mum\" — score 0.80\n"
        );
    }

    #[test]
    fn an_answer_is_read_to_a_pick_or_a_search() {
        let a = club("a", 127.0, "9A", 0.7);
        let b = club("b", 128.0, "8B", 0.65);
        let (sa, sb) = (scored("a", 0.9), scored("b", 0.8));
        let offers = [
            Offer {
                track: &a,
                scored: &sa,
                request: None,
            },
            Offer {
                track: &b,
                scored: &sb,
                request: None,
            },
        ];
        assert_eq!(
            read(r#"{"pick": 2, "why": " Yen's mum is here "}"#, &offers),
            Some(Choice::Pick {
                id: TrackId("b".into()),
                why: "Yen's mum is here".into()
            })
        );
        assert_eq!(
            read(
                "Sure! {\"search\": \"ABBA - Dancing Queen\", \"why\": \"asked\"} hope that helps",
                &offers
            ),
            Some(Choice::Search {
                query: "ABBA - Dancing Queen".into(),
                why: "asked".into()
            })
        );
        for nothing in [
            r#"{"pick": 3}"#,
            r#"{"pick": 0}"#,
            "no json",
            r#"{"search": " "}"#,
            "{bad",
        ] {
            assert_eq!(read(nothing, &offers), None, "{nothing}");
        }
    }
}
