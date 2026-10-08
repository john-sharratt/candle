//! What the model is shown about a target, appended under headings after the
//! generator's prompt.
//!
//! **The documents themselves, not a summary of them.** A mission is only as
//! specific as what its writer read: shown a character's name and asked for
//! "the next event", a model invents one that contradicts the eras; shown the
//! eras and the events already written, it can name the year and the hole. Each
//! document is cut to a word budget so the whole stays inside one prompt.
//!
//! The material comes **before** the generator's instruction in the prompt, so
//! the last thing the model reads is what it is asked to do with it — shown the
//! instruction first, then a character's anchor written in the second person, a
//! model went on writing in the anchor's voice to the character rather than to
//! the Maker.

use super::corpus::{Corpus, Life};
use super::target::{Kind, Subject, Target};

/// Words of a character's anchor shown.
const ANCHOR_WORDS: usize = 220;
/// Words of a life story shown.
const STORY_WORDS: usize = 450;
/// Words of the latest written event shown.
const EVENT_WORDS: usize = 350;
/// Words of each document in a pair.
const PAIR_WORDS: usize = 900;
/// Words of an era a gap is found in.
const ERA_WORDS: usize = 900;
/// Words of each era's opening shown in a timeline.
const TIMELINE_WORDS: usize = 40;

/// The material for `target`, as the text appended to the prompt. `None` when
/// the target names something the corpus no longer holds.
pub fn render(kind: Kind, target: &Target, corpus: &Corpus) -> Option<String> {
    match (&target.subject, kind) {
        (Subject::Life { who }, Kind::LifeEvent) => life(corpus.life(who)?, corpus),
        (Subject::Pair { a, b }, Kind::Contradiction) => pair(corpus, a, b),
        (Subject::Era { path }, Kind::Gap) => gap(corpus, path),
        (Subject::Written { path }, Kind::Review) => review(corpus, path),
        _ => None,
    }
}

/// Words of a document under review shown — all of any document a mission
/// asks for.
const REVIEW_WORDS: usize = 1400;

/// A written document and what it answers to: for a life event, whose life it
/// is, the events around it and how they are written, and the era it falls in;
/// for a story, the world and its eras.
fn review(corpus: &Corpus, path: &str) -> Option<String> {
    let text = corpus.text(path)?;
    let mut s = format!(
        "## The document under review (`{path}`)\n\n{}\n",
        cut(&strip_calls(&text), REVIEW_WORDS)
    );
    if let Some(who) = path
        .strip_prefix("layers/life/")
        .and_then(|rest| rest.split('/').next())
    {
        let l = corpus.life(who)?;
        s.push_str(&format!(
            "\n## Whose life it is\n\n{} (`{}`)\n\n{}\n",
            l.name,
            l.who,
            cut(&l.anchor, ANCHOR_WORDS)
        ));
        s.push_str("\n## The other events of this life\n\n");
        for e in l.events.iter().filter(|e| e.path != path) {
            s.push_str(&format!("- {} — {} (`{}`)\n", e.date, e.title, e.path));
        }
        let year: Option<u32> = path
            .rsplit('/')
            .next()
            .and_then(|f| f.get(..4))
            .and_then(|y| y.parse().ok());
        // How this life's other events are written: the voice the document
        // must keep.
        if let Some(other) = year.and_then(|y| {
            l.events
                .iter()
                .filter(|e| e.path != path)
                .min_by_key(|e| e.year().map_or(u32::MAX, |ey| ey.abs_diff(y)))
        }) {
            if let Some(t) = corpus.text(&other.path) {
                s.push_str(&format!(
                    "\n## Another event of this life, as written (`{}`)\n\n{}\n",
                    other.path,
                    cut(&strip_calls(&t), EVENT_WORDS)
                ));
            }
        }
        if let Some(era) = year.and_then(|y| corpus.era_of(y)) {
            s.push_str(&format!(
                "\n## The era it falls in (`{}`)\n\n{}\n",
                era.path,
                cut(&era.text, ERA_WORDS)
            ));
        }
    }
    if let Some(world) = &corpus.setting {
        s.push_str(&format!("\n## The world\n\n{}\n", cut(world, 120)));
    }
    s.push_str(&timeline(corpus));
    Some(s)
}

fn life(l: &Life, corpus: &Corpus) -> Option<String> {
    let mut s = format!("## Whose life\n\n{} (`{}`)\n\n", l.name, l.who);
    if !l.anchor.is_empty() {
        s.push_str(&cut(&l.anchor, ANCHOR_WORDS));
        s.push_str("\n\n");
    }
    s.push_str("## What is known of their life\n\n");
    match &l.story {
        Some(story) => s.push_str(&cut(story, STORY_WORDS)),
        None => s.push_str("Nothing beyond the events below has been written."),
    }
    s.push_str("\n\n## The events already written, in order\n\n");
    if l.events.is_empty() {
        s.push_str("None. This life has no events written yet.\n");
    }
    for e in &l.events {
        s.push_str(&format!("- {} — {} (`{}`)\n", e.date, e.title, e.path));
    }
    if let Some(latest) = l.events.last() {
        if let Some(text) = corpus.text(&latest.path) {
            s.push_str(&format!(
                "\n## The latest event, as written (`{}`)\n\n{}\n",
                latest.path,
                cut(&strip_calls(&text), EVENT_WORDS)
            ));
        }
    }
    s.push_str(&timeline(corpus));
    s.push_str(&format!(
        "\n## Where the next event belongs\n\n{}\n",
        belongs(l, corpus)
    ));
    s.push_str(
        "\n## How an event is dated\n\n`YYYY` for a year of the life, `YYYY-MM` for a month, \
         `YYYY-MM-DD` for a single day that became a memory — a date no event above already \
         covers, inside the eras of the world.\n",
    );
    Some(s)
}

/// Where in a life its next event has to fall — the rule [`super::answer`]
/// holds the answer to, said the same way here.
pub fn belongs(l: &Life, corpus: &Corpus) -> String {
    match (l.longest_stretch(), l.events.as_slice()) {
        (Some((from, to)), _) => format!(
            "Between {from} and {to}: the longest stretch of this life with nothing written in \
             it. The event falls in those years."
        ),
        (None, [only]) => {
            // **Said as years, not as a distance.** Told only "not within three
            // years of 3086", a model writing a Mech whose one event is a year
            // before the present chose 3087 three times running.
            let first = corpus.eras.iter().find_map(|e| e.year);
            let now = corpus.present();
            let year = only.year();
            let ranges: Vec<String> = [
                year.zip(first)
                    .map(|(y, f)| (f, y.saturating_sub(ONLY_EVENT_YEARS + 1))),
                year.zip(now).map(|(y, n)| (y + ONLY_EVENT_YEARS + 1, n)),
            ]
            .into_iter()
            .flatten()
            .filter(|(a, b)| a <= b)
            .map(|(a, b)| format!("{a} to {b}"))
            .collect();
            format!(
                "In {}: away from the one event written ({}). A life is filled from its largest \
                 silences, not from the days around what is already written.",
                ranges.join(" or "),
                only.date
            )
        }
        (None, []) => "Anywhere in this life — nothing of it is written yet. Begin with the event \
                       that most made them who they are."
            .to_string(),
        (None, _) => "Inside a year already written, as a month or a single day within it — \
                      every stretch between the written years is covered."
            .to_string(),
    }
}

/// How far from a life's only written event its next must be, in years.
pub const ONLY_EVENT_YEARS: u32 = 3;

fn pair(corpus: &Corpus, a: &str, b: &str) -> Option<String> {
    let (ta, tb) = (corpus.text(a)?, corpus.text(b)?);
    Some(format!(
        "## Document A (`{a}`)\n\n{}\n\n## Document B (`{b}`)\n\n{}\n{}",
        cut(&strip_calls(&ta), PAIR_WORDS),
        cut(&strip_calls(&tb), PAIR_WORDS),
        timeline(corpus)
    ))
}

fn gap(corpus: &Corpus, path: &str) -> Option<String> {
    let at = corpus.eras.iter().position(|e| e.path == path)?;
    let era = &corpus.eras[at];
    let mut s = format!(
        "## The era (`{}`)\n\n{}\n",
        era.path,
        cut(&era.text, ERA_WORDS)
    );
    s.push_str(&timeline(corpus));
    s.push_str("\n## Stories already told\n\n");
    if corpus.stories.is_empty() {
        s.push_str("None yet.\n");
    }
    for story in &corpus.stories {
        s.push_str(&format!("- {} (`{}`)\n", story.title, story.path));
    }
    Some(s)
}

/// The world's eras in order, one line each — the dates everything else is
/// checked against.
fn timeline(corpus: &Corpus) -> String {
    let mut s = String::from("\n## The eras of the world, in order\n\n");
    for e in &corpus.eras {
        let year = e
            .year
            .map(|y| format!("{y} CE"))
            .unwrap_or_else(|| "undated".into());
        s.push_str(&format!(
            "- {year} — {} (`{}`): {}\n",
            e.title,
            e.path,
            cut(&opening(&e.text), TIMELINE_WORDS)
        ));
    }
    s
}

/// An era's first paragraph of prose, past its heading and date line.
fn opening(text: &str) -> String {
    text.split("\n\n")
        .map(str::trim)
        .find(|p| !p.is_empty() && !p.starts_with('#') && !p.starts_with("**Era"))
        .unwrap_or_default()
        .to_string()
}

/// A document without its `<tool_call>` blocks — the beliefs and relationships
/// a life document forms are machinery, not what happened.
pub(crate) fn strip_calls(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("<tool_call>") {
        out.push_str(&rest[..at]);
        rest = match rest[at..].find("</tool_call>") {
            Some(end) => &rest[at + end + "</tool_call>".len()..],
            None => "",
        };
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// `text` cut to about `words` words, marked when cut.
pub(crate) fn cut(text: &str, words: usize) -> String {
    let all: Vec<&str> = text.split_whitespace().collect();
    if all.len() <= words {
        return text.trim().to_string();
    }
    // Cut on the word, keeping the original line breaks up to it.
    let mut seen = 0;
    let mut end = 0;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() && i > 0 && !text[..i].ends_with(char::is_whitespace) {
            seen += 1;
            if seen == words {
                end = i;
                break;
            }
        }
    }
    format!("{} …", text[..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::engine::mission_gen::target::next;

    #[test]
    fn a_life_shows_who_what_is_written_the_eras_and_where_the_event_goes() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        let m = render(Kind::LifeEvent, &t, &c).unwrap();
        assert!(m.starts_with("## Whose life\n\nKeeper (`keeper`)\n\nYou keep the towers."));
        assert!(m.contains(
            "- 2487-03-08 — The Second the Sky Went Out (`layers/life/keeper/2487-03-08 The \
             Second the Sky Went Out.md`)\n"
        ));
        assert!(m.contains("## The latest event, as written (`layers/life/keeper/2786 The Charge.md`)\n\nWe were given the plan."));
        assert!(m.contains(
            "- 2607 CE — The Retreat (`layers/eras/the-retreat.md`): Everyone went underground.\n"
        ));
        assert!(m.contains("## How an event is dated"));
    }

    #[test]
    fn a_pair_shows_both_documents_and_a_gap_shows_the_era_and_the_stories() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::Contradiction, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Contradiction, &t, &c).unwrap();
        assert!(m.contains("## Document A (`layers/eras/the-fall.md`)"));
        assert!(m.contains("## Document B (`layers/eras/the-retreat.md`)"));
        let t = next(Kind::Gap, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Gap, &t, &c).unwrap();
        assert!(m.contains("- The Charge (`layers/stories/the-charge.md`)"));
        // The wrong pairing of kind and subject renders nothing.
        assert_eq!(render(Kind::LifeEvent, &t, &c), None);
    }

    /// **A life event under review is shown with whose it is, a sibling event
    /// for its voice, and its era.**
    #[test]
    fn a_review_shows_the_document_and_what_it_answers_to() {
        let dir = mind();
        let mut c = Corpus::read(dir.path(), "test");
        c.reviewable = vec!["layers/life/keeper/2786 The Charge.md".into()];
        let t = next(Kind::Review, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Review, &t, &c).unwrap();
        assert!(m.starts_with(
            "## The document under review (`layers/life/keeper/2786 The Charge.md`)\n\nWe were \
             given the plan."
        ));
        assert!(m.contains("## Whose life it is\n\nKeeper (`keeper`)"));
        assert!(m.contains(
            "## Another event of this life, as written (`layers/life/keeper/2487-03-08 The \
             Second the Sky Went Out.md`)"
        ));
        assert!(m.contains("## The era it falls in (`layers/eras/the-retreat.md`)"));
        assert!(m.contains("## The world\n\nA world of towers"));
    }

    /// **A life with one event is told where else to look, in years.** The
    /// fixture's world runs 2487 to 2787.
    #[test]
    fn a_life_with_one_event_is_told_the_years_away_from_it() {
        let dir = mind();
        std::fs::create_dir_all(dir.path().join("layers/life/kaelor")).unwrap();
        std::fs::write(
            dir.path().join("layers/life/kaelor/2785 The Last Stand.md"),
            "x",
        )
        .unwrap();
        let c = Corpus::read(dir.path(), "test");
        assert_eq!(
            belongs(c.life("kaelor").unwrap(), &c),
            "In 2487 to 2781: away from the one event written (2785). A life is filled from its \
             largest silences, not from the days around what is already written."
        );
        std::fs::rename(
            dir.path().join("layers/life/kaelor/2785 The Last Stand.md"),
            dir.path().join("layers/life/kaelor/2600 Midway.md"),
        )
        .unwrap();
        let c = Corpus::read(dir.path(), "test");
        assert!(
            belongs(c.life("kaelor").unwrap(), &c).starts_with("In 2487 to 2596 or 2604 to 2787:")
        );
    }

    #[test]
    fn tool_calls_are_not_part_of_what_happened() {
        assert_eq!(
            strip_calls("We finished.\n\n<tool_call>\n{\"name\":\"x\"}\n</tool_call>\nAfter."),
            "We finished.\n\n\nAfter."
        );
    }

    #[test]
    fn a_long_text_is_cut_on_a_word_and_marked() {
        assert_eq!(cut("one two three", 5), "one two three");
        assert_eq!(cut("one two three four", 2), "one two …");
        assert_eq!(cut("one\ntwo three", 2), "one\ntwo …");
    }
}
