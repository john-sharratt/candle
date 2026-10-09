//! What a Maker reads before it writes: the record around the work, up to the
//! year the work is set in.
//!
//! **Write from what came before.** A Maker drafting one day of a life used to
//! read the era and the single written event nearest that day — which for a
//! day in 2950 was an event in 3086, a hundred and thirty-six years later — and
//! nothing that led up to it. With so little of the record in front of it, it
//! wrote from what was: the room it stood in, and the colleagues talking in it.
//!
//! The rule is one rule for any document an operation writes, whatever kind it
//! is and whatever world it is in. It knows four things and nothing else about
//! the work: the year it is set in, the folder it is written into, the dates of
//! the documents already written, and the names its brief uses. From those it
//! picks, in this order and up to [`MOST`]:
//!
//! 1. **The era** the year falls in — and the era before it, when the year is
//!    within [`ERA_EDGE`] years of the era's opening, so a work written just
//!    after a change of era knows what it changed from.
//! 2. **What came just before it beside it** — the latest [`BEFORE_HERE`]
//!    documents in its own folder dated before it: the entries a life holds
//!    before the day being written, the stories told before this one.
//! 3. **The world around it** — up to [`AROUND`] documents elsewhere, dated in
//!    the [`AROUND_YEARS`] before it, that name somebody or somewhere its brief
//!    names.
//!
//! **Nothing after the year.** The Maker works in that year at a time machine,
//! which keeps the future out of what it recalls; what it is told to read keeps
//! the future out the same way. A later entry the work must not contradict is
//! the canon check's to hold it to, not the drafter's to write from.
//!
//! Read in the order they happened, eras first.

use super::corpus::Corpus;
use crate::engine::chronology::dated;

/// How close to an era's opening a year is for the era before it to be read
/// too.
pub const ERA_EDGE: u32 = 10;
/// How many of the documents just before it, in its own folder, are read.
pub const BEFORE_HERE: usize = 2;
/// How far back the world around it is looked for, in years.
pub const AROUND_YEARS: u32 = 30;
/// How many documents from around it, elsewhere, are read.
pub const AROUND: usize = 2;
/// The most documents read before writing.
pub const MOST: usize = 5;

/// One document to read before writing, and what it is read for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub path: String,
    pub why: String,
}

/// A written document with a date.
struct Dated<'a> {
    path: &'a str,
    /// `YYYY`, `YYYY-MM` or `YYYY-MM-DD` — comparable as text.
    date: String,
    year: u32,
}

/// What to read before writing `writes`, set in `year`, from `brief`.
pub fn before_writing(corpus: &Corpus, writes: &str, year: u32, brief: &str) -> Vec<Read> {
    let mut eras: Vec<Read> = Vec::new();
    if let Some(era) = corpus.era_of(year) {
        let opened = era.year.unwrap_or(year);
        if year - opened <= ERA_EDGE {
            if let Some(before) = opened.checked_sub(1).and_then(|y| corpus.era_of(y)) {
                eras.push(Read {
                    path: before.path.clone(),
                    why: format!(
                        "the era before it — {year} is {} years into {}, and this is what it \
                         followed",
                        year - opened,
                        era.title
                    ),
                });
            }
        }
        eras.push(Read {
            path: era.path.clone(),
            why: format!("the era {year} falls in, {}", era.title),
        });
    }

    let own = date_of_name(writes).unwrap_or_else(|| year.to_string());
    let folder = folder_of(writes);
    let written = written(corpus);
    let before: Vec<&Dated> = written
        .iter()
        .filter(|d| d.path != writes && d.date < own)
        .collect();

    // What came just before it, beside it: the latest first, then put back in
    // the order they happened.
    let mut here: Vec<&Dated> = before
        .iter()
        .copied()
        .filter(|d| folder_of(d.path) == folder)
        .collect();
    here.sort_by(|a, b| b.date.cmp(&a.date));
    here.truncate(BEFORE_HERE);

    // The world around it: elsewhere, recent, and about somebody or somewhere
    // the brief names.
    let names = names_in(brief);
    let mut around: Vec<(usize, &Dated, Vec<String>)> = before
        .iter()
        .copied()
        .filter(|d| folder_of(d.path) != folder && year - d.year.min(year) <= AROUND_YEARS)
        .filter_map(|d| {
            let text = corpus.text(d.path)?;
            let shared: Vec<String> = names
                .iter()
                .filter(|n| has_word(&text, n) || has_word(d.path, n))
                .cloned()
                .collect();
            (!shared.is_empty()).then_some((shared.len(), d, shared))
        })
        .collect();
    around.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.date.cmp(&a.1.date)));
    around.truncate(AROUND);

    let mut rest: Vec<(&str, Read)> = here
        .into_iter()
        .map(|d| {
            (
                d.date.as_str(),
                Read {
                    path: d.path.to_string(),
                    why: format!("written just before it ({}), beside it in {folder}", d.date),
                },
            )
        })
        .chain(around.into_iter().map(|(_, d, shared)| {
            (
                d.date.as_str(),
                Read {
                    path: d.path.to_string(),
                    why: format!(
                        "from around then ({}), and it names {}",
                        d.date,
                        shared.join(", ")
                    ),
                },
            )
        }))
        .collect();
    rest.sort_by(|a, b| a.0.cmp(b.0));

    eras.into_iter()
        .chain(rest.into_iter().map(|(_, r)| r))
        .take(MOST)
        .collect()
}

/// The brief's account of what to read and why, to follow the brief.
pub fn said(reads: &[Read]) -> String {
    if reads.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = reads
        .iter()
        .map(|r| format!("- `{}` — {}", r.path, r.why))
        .collect();
    format!(
        "\n\nBefore you write, read what the record already holds up to then, in the order it \
         happened, and write on from it:\n{}",
        lines.join("\n")
    )
}

/// Every written document in the corpus with a date: by the date its file name
/// opens with, or else the year its text is set in.
fn written(corpus: &Corpus) -> Vec<Dated<'_>> {
    let events = corpus
        .lives
        .iter()
        .flat_map(|l| l.events.iter())
        .filter_map(|e| {
            Some(Dated {
                path: &e.path,
                year: e.year()?,
                date: e.date.clone(),
            })
        });
    let stories = corpus.stories.iter().filter_map(|s| {
        let date = date_of_name(&s.path).or_else(|| dated(&s.text).map(|y| y.to_string()))?;
        Some(Dated {
            path: &s.path,
            year: date.get(..4)?.parse().ok()?,
            date,
        })
    });
    events.chain(stories).collect()
}

/// The date a file name opens with — `2950-06-12 The Unlocked Door.md` →
/// `2950-06-12` — when it opens with a year.
fn date_of_name(path: &str) -> Option<String> {
    let name = path.rsplit('/').next()?;
    let date: String = name
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    let date = date.trim_end_matches('-');
    (date.len() >= 4 && date[..4].chars().all(|c| c.is_ascii_digit())).then(|| date.to_string())
}

/// The folder a mind path is in.
fn folder_of(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

/// The names a text uses: capitalised words of four letters or more that do not
/// open a sentence — a name is capitalised wherever it stands, any other word
/// only at the start.
fn names_in(text: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut opens = true;
    for raw in text.split_whitespace() {
        let word: String = raw.chars().filter(|c| c.is_alphanumeric()).collect();
        let capital = word.chars().next().is_some_and(char::is_uppercase);
        if capital && !opens && word.chars().count() >= 4 && !names.contains(&word) {
            names.push(word);
        }
        opens = raw.ends_with(['.', '!', '?', ':', '"']) || raw.starts_with('"');
    }
    names
}

/// Whether `word` stands in `text` as a whole word.
fn has_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|w| w == word)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mind with three eras, one life with entries either side of the day to
    /// be written, a story set before it that names the same place, and one
    /// that names nothing the brief does.
    fn mind() -> (tempfile::TempDir, Corpus) {
        let dir = tempfile::tempdir().unwrap();
        let w = |path: &str, text: &str| {
            let p = dir.path().join(path);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        w(
            "layers/eras/the-old.md",
            "# The Old\n\n**Era 0 · 2900 CE**\n\nBefore.\n",
        );
        w(
            "layers/eras/the-new.md",
            "# The New\n\n**Era 1 · 2945 CE**\n\nAfter.\n",
        );
        w(
            "layers/eras/the-next.md",
            "# The Next\n\n**Era 2 · 3000 CE**\n\nLater.\n",
        );
        w(
            "personalities/ana.yaml",
            "id: ana\nname: Ana\nanchor: You hold the gate.\n",
        );
        w(
            "personalities/bel.yaml",
            "id: bel\nname: Bel\nanchor: You keep the ledger.\n",
        );
        w(
            "layers/life/ana/2910 First.md",
            "You come to Marrow Gate.\n",
        );
        w(
            "layers/life/ana/2930-04 Second.md",
            "You hold Marrow Gate alone.\n",
        );
        w(
            "layers/life/ana/2940 Third.md",
            "You leave the gate to Bel.\n",
        );
        w("layers/life/ana/2990 Later.md", "You are old.\n");
        w(
            "layers/life/bel/2935 Ledger.md",
            "You count the stores at Marrow Gate.\n",
        );
        w(
            "layers/life/bel/2890 Too Early.md",
            "You are young at Marrow Gate.\n",
        );
        w(
            "layers/stories/the-quiet-shift.md",
            "# The Quiet Shift\n\nThe year is 2938. Nobody named came.\n",
        );
        let c = Corpus::read(dir.path(), "test");
        (dir, c)
    }

    const BRIEF: &str = "Ana stands at the door of Marrow Gate and waits for Bel.";

    fn paths(reads: &[Read]) -> Vec<&str> {
        reads.iter().map(|r| r.path.as_str()).collect()
    }

    /// **Read what came before, oldest first, and nothing after.** The era it
    /// falls in and the one before it (it is two years in), the two entries of
    /// its own life just before it, and from elsewhere the entry that names the
    /// same place — not the later entry, not one outside the years around it,
    /// and not a story that names nothing the brief does.
    #[test]
    fn a_work_reads_what_came_just_before_it_and_nothing_after() {
        let (_dir, c) = mind();
        let reads = before_writing(&c, "layers/life/ana/2947 Fourth.md", 2947, BRIEF);
        assert_eq!(
            paths(&reads),
            [
                "layers/eras/the-old.md",
                "layers/eras/the-new.md",
                "layers/life/ana/2930-04 Second.md",
                "layers/life/bel/2935 Ledger.md",
                "layers/life/ana/2940 Third.md",
            ]
        );
        assert_eq!(reads[1].why, "the era 2947 falls in, The New");
        assert_eq!(
            reads[2].why,
            "written just before it (2930-04), beside it in layers/life/ana"
        );
        assert_eq!(
            reads[3].why,
            "from around then (2935), and it names Marrow, Gate"
        );
    }

    /// Deep into an era, the era before it is not read.
    #[test]
    fn deep_into_an_era_only_its_own_era_is_read() {
        let (_dir, c) = mind();
        let reads = before_writing(&c, "layers/life/ana/2960 Fifth.md", 2960, "Nothing named.");
        assert_eq!(
            paths(&reads),
            [
                "layers/eras/the-new.md",
                "layers/life/ana/2930-04 Second.md",
                "layers/life/ana/2940 Third.md",
            ]
        );
    }

    /// A document with no date in its name is dated by its text, and a work
    /// with none is dated by its year — the same rule for a story as for a life.
    #[test]
    fn a_story_reads_the_stories_told_before_it() {
        let (_dir, c) = mind();
        let reads = before_writing(&c, "layers/stories/a-new-story.md", 2945, "Nothing named.");
        assert_eq!(
            paths(&reads),
            [
                "layers/eras/the-old.md",
                "layers/eras/the-new.md",
                "layers/stories/the-quiet-shift.md",
            ]
        );
    }

    #[test]
    fn names_are_capitalised_words_that_do_not_open_a_sentence() {
        assert_eq!(
            names_in("When Ana came. Then she saw Marrow Gate, and Bel was there."),
            ["Marrow", "Gate"]
        );
    }

    #[test]
    fn the_brief_says_what_each_read_is_for() {
        let s = said(&[Read {
            path: "layers/eras/the-new.md".into(),
            why: "the era 2947 falls in, The New".into(),
        }]);
        assert!(s.ends_with("- `layers/eras/the-new.md` — the era 2947 falls in, The New"));
    }
}
