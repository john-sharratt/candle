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

use npc_map::text::list;

use super::answer::taken_names;
use super::corpus::{Corpus, Era, Life};
use super::glossary;
use super::target::{Kind, Subject, Target};

/// Words of a character's anchor shown — to the generator, to the table's
/// reading, and to the Maker who writes the event, all the same.
pub(crate) const ANCHOR_WORDS: usize = 220;
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

/// The material for `target`, as the text shown before the prompt: the
/// sections its generator's `context` names (`config::PROPOSAL_CONTEXT`), in
/// the order the kind of work sets them. `None` when the target names
/// something the corpus no longer holds.
pub fn render(kind: Kind, target: &Target, corpus: &Corpus, context: &[String]) -> Option<String> {
    let shown = |name: &str| context.iter().any(|c| c == name);
    match (&target.subject, kind) {
        (Subject::Life { who }, Kind::LifeEvent) => Some(life(corpus.life(who)?, corpus, &shown)),
        (Subject::Pair { a, b }, Kind::Contradiction) => pair(corpus, a, b, &shown),
        (Subject::Era { path }, Kind::Gap) => gap(corpus, path, &shown),
        _ => None,
    }
}

/// Words of a draft shown to the table's reading — all of any document an
/// operation asks for.
const DRAFT_WORDS: usize = 1600;

/// A draft and what it answers to, for the table's reading: for a life event,
/// whose life it is, the events around it and how they are written, and the
/// era it falls in; for a story, the era it tells (`tells`, the operation's
/// own), the world and its eras. `None` when the draft is not on the record.
/// The draft itself is always shown; every other section only when the
/// reading step's `context` names it (`config::READING_CONTEXT`).
///
/// **A story is read in the era it tells.** Shown only the world as it is now
/// and a list of eras, the table failed a story of the Stabilisation for "a
/// significant discrepancy" with the present year, 3087 — the era the story was
/// written to was the one thing its reading never had in front of it.
pub fn draft(
    corpus: &Corpus,
    path: &str,
    tells: Option<&str>,
    context: &[String],
) -> Option<String> {
    let shown = |name: &str| context.iter().any(|c| c == name);
    let text = corpus.text(path)?;
    let mut s = format!(
        "## The draft (`{path}`)\n\n{}\n",
        cut(&strip_calls(&text), DRAFT_WORDS)
    );
    // What the draft answers to, for the world's terms it names.
    let mut around: Vec<String> = corpus.setting.iter().cloned().collect();
    if shown("when-set") {
        if let Some(when) = when_set(corpus, path, tells) {
            s.push_str(&format!("\n## When it is set\n\n{when}\n"));
        }
    }
    if let Some((era, words)) = tells.and_then(|e| Some((e, corpus.text(e)?))) {
        if shown("era-it-tells") {
            s.push_str(&format!(
                "\n## The era it tells (`{era}`) — the time it is set in\n\n{}\n",
                cut(&strip_calls(&words), ERA_WORDS)
            ));
        }
        around.push(words);
    }
    if let Some(who) = path
        .strip_prefix("layers/life/")
        .and_then(|rest| rest.split('/').next())
    {
        let l = corpus.life(who)?;
        if shown("whose-life") {
            s.push_str(&format!(
                "\n## Whose life it is\n\n{} (`{}`)\n\n{}\n",
                l.name,
                l.who,
                cut(&l.anchor, ANCHOR_WORDS)
            ));
        }
        around.push(l.anchor.clone());
        if shown("other-events") {
            s.push_str("\n## The other events of this life\n\n");
            for e in l.events.iter().filter(|e| e.path != path) {
                s.push_str(&format!("- {} — {} (`{}`)\n", e.date, e.title, e.path));
            }
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
            if let Some(t) = corpus.text(&other.path).filter(|_| shown("voice-example")) {
                s.push_str(&format!(
                    "\n## Another event of this life, as written (`{}`)\n\n{}\n",
                    other.path,
                    cut(&strip_calls(&t), EVENT_WORDS)
                ));
            }
        }
        if let Some(era) = year.and_then(|y| corpus.era_of(y)) {
            if shown("era-it-falls-in") {
                s.push_str(&format!(
                    "\n## The era it falls in (`{}`)\n\n{}\n",
                    era.path,
                    cut(&era.text, ERA_WORDS)
                ));
            }
            around.push(era.text.clone());
        }
    }
    if shown("worlds-words") {
        let around: Vec<&str> = around.iter().map(String::as_str).collect();
        s.push_str(&glossary::render(&glossary::named(
            &corpus.terms,
            &text,
            &around,
        )));
    }
    // The setting is the world as it is now, said so: read as the world of
    // the draft's own time, it makes a story set a century earlier look wrong
    // for not saying "three centuries after the war".
    if let Some(world) = corpus.setting.as_ref().filter(|_| shown("world-now")) {
        let now = corpus
            .present()
            .map(|y| format!(" (in {y})"))
            .unwrap_or_default();
        s.push_str(&format!(
            "\n## The world as it is now{now} — not as it was at the draft's own time\n\n{}\n",
            cut(world, 120)
        ));
    }
    if shown("timeline") {
        s.push_str(&timeline(corpus));
    }
    Some(s)
}

fn life(l: &Life, corpus: &Corpus, shown: &dyn Fn(&str) -> bool) -> String {
    let mut s = String::new();
    if shown("whose-life") {
        s.push_str(&format!("## Whose life\n\n{} (`{}`)\n\n", l.name, l.who));
        if !l.anchor.is_empty() {
            s.push_str(&cut(&l.anchor, ANCHOR_WORDS));
            s.push_str("\n\n");
        }
    }
    if shown("life-story") {
        s.push_str("## What is known of their life\n\n");
        match &l.story {
            Some(story) => s.push_str(&cut(story, STORY_WORDS)),
            None => s.push_str("Nothing beyond the events below has been written."),
        }
        s.push_str("\n\n");
    }
    if shown("events-written") {
        s.push_str("## The events already written, in order\n\n");
        if l.events.is_empty() {
            s.push_str("None. This life has no events written yet.\n");
        }
        for e in &l.events {
            s.push_str(&format!("- {} — {} (`{}`)\n", e.date, e.title, e.path));
        }
    }
    if let Some(latest) = l.events.last().filter(|_| shown("latest-event")) {
        if let Some(text) = corpus.text(&latest.path) {
            s.push_str(&format!(
                "\n## The latest event, as written (`{}`)\n\n{}\n",
                latest.path,
                cut(&strip_calls(&text), EVENT_WORDS)
            ));
        }
    }
    if shown("timeline") {
        s.push_str(&timeline(corpus));
    }
    if shown("where-it-belongs") {
        s.push_str(&format!(
            "\n## Where the next event belongs\n\n{}\n",
            belongs(l, corpus)
        ));
    }
    if shown("how-dated") {
        s.push_str(
            "\n## How an event is dated\n\n`YYYY` for a year of the life, `YYYY-MM` for a month, \
             `YYYY-MM-DD` for a single day that became a memory — a date no event above already \
             covers, inside the eras of the world.\n",
        );
    }
    s.trim_start().to_string()
}

/// Where in a life its next event has to fall — the rule [`super::answer`]
/// holds the answer to, said the same way here.
pub fn belongs(l: &Life, corpus: &Corpus) -> String {
    match (l.longest_stretch(), l.events.as_slice()) {
        // **The stretch is shown as the eras that fill it.** Given only its two
        // ends, "between 2776 and 3084", a model three times running chose 3085
        // and 3086 — the end nearest the present and the life's latest event,
        // and outside it. The eras inside are somewhere to stand in the middle.
        (Some((from, to)), _) => {
            let within: Vec<String> = corpus
                .eras
                .iter()
                .filter_map(|e| {
                    let (start, end) = corpus.span(e)?;
                    let end = end.unwrap_or(u32::MAX);
                    (start <= to && end >= from)
                        .then(|| format!("{} ({})", e.title, span_text(corpus, e)))
                })
                .collect();
            let eras = match within.is_empty() {
                true => String::new(),
                false => format!(" Those years are {}.", list(&within)),
            };
            // **The written years at either end are named as written.** Told
            // "between 2776 and 3041", the model chose 3042 — the year of the
            // event that closes the stretch — three lives running.
            format!(
                "Between {from} and {to}: the longest stretch of this life with nothing written \
                 in it. The event falls in those years, not before {from} and not after {to} — \
                 {} and {} already have their events.{eras}",
                from - 1,
                to + 1
            )
        }
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

fn pair(corpus: &Corpus, a: &str, b: &str, shown: &dyn Fn(&str) -> bool) -> Option<String> {
    let (ta, tb) = (corpus.text(a)?, corpus.text(b)?);
    let mut s = String::new();
    if shown("document-a") {
        s.push_str(&format!(
            "## Document A (`{a}`)\n\n{}\n\n",
            cut(&strip_calls(&ta), PAIR_WORDS)
        ));
    }
    if shown("document-b") {
        s.push_str(&format!(
            "## Document B (`{b}`)\n\n{}\n",
            cut(&strip_calls(&tb), PAIR_WORDS)
        ));
    }
    if shown("timeline") {
        s.push_str(&timeline(corpus));
    }
    Some(s.trim_start().to_string())
}

fn gap(corpus: &Corpus, path: &str, shown: &dyn Fn(&str) -> bool) -> Option<String> {
    let at = corpus.eras.iter().position(|e| e.path == path)?;
    let era = &corpus.eras[at];
    let mut s = String::new();
    if shown("era") {
        s.push_str(&format!(
            "## The era (`{}`)\n\n{}\n",
            era.path,
            cut(&era.text, ERA_WORDS)
        ));
    }
    if shown("timeline") {
        s.push_str(&timeline(corpus));
    }
    if shown("stories-told") {
        s.push_str("\n## Stories already told\n\n");
        if corpus.stories.is_empty() {
            s.push_str("None yet.\n");
        }
        for story in &corpus.stories {
            s.push_str(&format!("- {} (`{}`)\n", story.title, story.path));
        }
    }
    let taken = taken_names(corpus);
    if shown("names-taken") && !taken.is_empty() {
        s.push_str(&format!(
            "\n## Names the stories have already given their people\n\n{}. The people a new story \
             invents are new people, with names of their own — none of these.\n",
            list(&taken)
        ));
    }
    Some(s.trim_start().to_string())
}

/// The world's eras in order, one line each — the dates everything else is
/// checked against.
/// The years an era covers, as the table reads them: "2758–2785 CE", "3087 CE
/// to now", or "undated".
///
/// **Where an era ends is given, not left to be worked out.** Shown only the
/// year each opened, the table placed 3085 after an era that ran to 3086, put
/// the Salvation two centuries late, and failed drafts for both.
fn span_text(corpus: &Corpus, era: &Era) -> String {
    match corpus.span(era) {
        Some((from, Some(to))) => format!("{from}–{to} CE"),
        Some((from, None)) => format!("{from} CE to now"),
        None => "undated".into(),
    }
}

/// When the draft at `path` is set, with the arithmetic done: its date, the
/// era that holds it and that era's years, and how long before the world's
/// present. A story is set in the era it `tells`. `None` when neither is known.
fn when_set(corpus: &Corpus, path: &str, tells: Option<&str>) -> Option<String> {
    let present = corpus.present();
    let before = |year: u32| match present {
        Some(now) if now == year + 1 => format!(", a year before the world's present ({now})"),
        Some(now) if now > year => {
            format!(", {} years before the world's present ({now})", now - year)
        }
        Some(now) if now == year => format!(", the world's present ({now})"),
        _ => String::new(),
    };
    if path.starts_with("layers/life/") {
        let file = path.rsplit('/').next()?;
        let date = file.split(' ').next()?;
        let year: u32 = date.get(..4)?.parse().ok()?;
        let era = corpus.era_of(year)?;
        return Some(format!(
            "It is set on {date}: in {} ({}){}.",
            era.title,
            span_text(corpus, era),
            before(year)
        ));
    }
    let era = corpus
        .eras
        .iter()
        .find(|e| Some(e.path.as_str()) == tells)?;
    Some(format!(
        "It is set in {} ({}), the era it tells{}.",
        era.title,
        span_text(corpus, era),
        era.year.map(before).unwrap_or_default()
    ))
}

fn timeline(corpus: &Corpus) -> String {
    let mut s = String::from("\n## The eras of the world, in order\n\n");
    for e in &corpus.eras {
        let year = span_text(corpus, e);
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
    use crate::engine::mission_gen::config::{PROPOSAL_CONTEXT, READING_CONTEXT};
    use crate::engine::mission_gen::corpus::tests::mind;
    use crate::engine::mission_gen::target::next;

    #[test]
    fn a_life_shows_who_what_is_written_the_eras_and_where_the_event_goes() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        let m = render(Kind::LifeEvent, &t, &c, &proposal()).unwrap();
        assert!(m.starts_with("## Whose life\n\nKeeper (`keeper`)\n\nYou keep the towers."));
        assert!(m.contains(
            "- 2487-03-08 — The Second the Sky Went Out (`layers/life/keeper/2487-03-08 The \
             Second the Sky Went Out.md`)\n"
        ));
        assert!(m.contains("## The latest event, as written (`layers/life/keeper/2786 The Charge.md`)\n\nWe were given the plan."));
        assert!(m.contains(
            "- 2607–2786 CE — The Retreat (`layers/eras/the-retreat.md`): Everyone went \
             underground.\n"
        ));
        assert!(m.contains("## How an event is dated"));
    }

    /// Every section a proposal may be shown.
    fn proposal() -> Vec<String> {
        PROPOSAL_CONTEXT.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_pair_shows_both_documents_and_a_gap_shows_the_era_and_the_stories() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::Contradiction, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Contradiction, &t, &c, &proposal()).unwrap();
        assert!(m.starts_with("## Document A (`layers/eras/the-fall.md`)"));
        assert!(m.contains("## Document B (`layers/eras/the-retreat.md`)"));
        let t = next(Kind::Gap, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Gap, &t, &c, &proposal()).unwrap();
        assert!(m.contains("- The Charge (`layers/stories/the-charge.md`)"));
        // The wrong pairing of kind and subject renders nothing.
        assert_eq!(render(Kind::LifeEvent, &t, &c, &proposal()), None);
    }

    /// **A proposal is shown only the sections its generator lists.**
    #[test]
    fn a_proposal_shows_only_the_sections_its_generator_lists() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        let only = ["where-it-belongs".to_string(), "how-dated".to_string()];
        let m = render(Kind::LifeEvent, &t, &c, &only).unwrap();
        assert!(m.starts_with("## Where the next event belongs"), "{m}");
        assert!(m.contains("## How an event is dated"));
        for gone in [
            "## Whose life",
            "## What is known",
            "## The events already",
            "## The eras",
        ] {
            assert!(!m.contains(gone), "{gone} was not listed");
        }
        let t = next(Kind::Gap, &c, &|_, _| false, 0).unwrap();
        let m = render(Kind::Gap, &t, &c, &["era".to_string()]).unwrap();
        assert!(m.starts_with("## The era ("));
        assert!(!m.contains("## Stories already told"));
    }

    /// **A drafted life event is shown with whose it is, a sibling event for
    /// its voice, and its era.**
    #[test]
    fn a_draft_shows_the_document_and_what_it_answers_to() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let m = draft(&c, "layers/life/keeper/2786 The Charge.md", None, &every()).unwrap();
        assert!(m.starts_with(
            "## The draft (`layers/life/keeper/2786 The Charge.md`)\n\nWe were given the plan."
        ));
        assert!(m.contains("## Whose life it is\n\nKeeper (`keeper`)"));
        assert!(m.contains(
            "## Another event of this life, as written (`layers/life/keeper/2487-03-08 The \
             Second the Sky Went Out.md`)"
        ));
        assert!(m.contains("## The era it falls in (`layers/eras/the-retreat.md`)"));
        assert!(m.contains(
            "## When it is set\n\nIt is set on 2786: in The Retreat (2607–2786 CE), a year before \
             the world's present (2787).\n"
        ));
        // Every era with the years it covers; the last runs to now.
        assert!(m.contains("- 2487–2606 CE — The Fall (`layers/eras/the-fall.md`)"));
        assert!(m.contains("- 2787 CE to now — The Salvation (`layers/eras/the-salvation.md`)"));
        assert!(m.contains(
            "## The world as it is now (in 2787) — not as it was at the draft's own time\n\nA \
             world of towers"
        ));
    }

    /// **A story is read beside the era it tells**, ahead of the world as it
    /// is now; one with no era of its own is read without.
    #[test]
    fn a_story_draft_is_read_in_the_era_it_tells() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let story = "layers/stories/the-charge.md";
        let m = draft(&c, story, Some("layers/eras/the-retreat.md"), &every()).unwrap();
        let era = m
            .find("## The era it tells (`layers/eras/the-retreat.md`) — the time it is set in\n\n")
            .expect("the era it tells");
        let now = m.find("## The world as it is now").expect("the world now");
        assert!(era < now, "the story's own time comes first");
        assert!(m.contains(
            "## When it is set\n\nIt is set in The Retreat (2607–2786 CE), the era it tells, 180 \
             years before the world's present (2787).\n"
        ));
        assert!(!draft(&c, story, None, &every())
            .unwrap()
            .contains("## The era it tells"));
        assert!(
            !draft(&c, story, Some("layers/eras/no-such-era.md"), &every())
                .unwrap()
                .contains("## The era it tells"),
            "an era not on the record is not shown"
        );
    }

    /// Every section a reading may be shown.
    fn every() -> Vec<String> {
        READING_CONTEXT.iter().map(|s| s.to_string()).collect()
    }

    /// **A reading is shown the draft and only the sections its step lists.**
    #[test]
    fn a_draft_shows_only_the_sections_its_step_lists() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        let path = "layers/life/keeper/2786 The Charge.md";
        let m = draft(&c, path, None, &["whose-life".to_string()]).unwrap();
        assert!(m.starts_with(
            "## The draft (`layers/life/keeper/2786 The Charge.md`)\n\nWe were given the plan."
        ));
        assert!(m.contains("## Whose life it is\n\nKeeper (`keeper`)"));
        for gone in [
            "## When it is set",
            "## The other events of this life",
            "## Another event of this life",
            "## The era it falls in",
            "## The world as it is now",
            "The Fall (`layers/eras/the-fall.md`)",
        ] {
            assert!(!m.contains(gone), "{gone} was not listed");
        }
    }

    /// **A stretch is given its bounds and the eras that fill it.**
    #[test]
    fn a_stretch_names_the_eras_inside_it() {
        let dir = mind();
        let c = Corpus::read(dir.path(), "test");
        assert_eq!(
            belongs(c.life("keeper").unwrap(), &c),
            "Between 2488 and 2785: the longest stretch of this life with nothing written in it. \
             The event falls in those years, not before 2488 and not after 2785 — 2487 and 2786 \
             already have their events. Those years are \
             The Fall (2487–2606 CE) and The Retreat (2607–2786 CE)."
        );
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
