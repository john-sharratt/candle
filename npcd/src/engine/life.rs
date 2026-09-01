//! A personality's life, as dated documents that make their own beliefs.
//!
//! # The inversion this implements
//!
//! A belief used to be a record somebody typed — a statement, a confidence, a
//! threshold, sitting on the character with no origin. Here nobody types a
//! belief. You author a **life**, and the beliefs are what that life produced.
//!
//! That is not a presentation change. A belief with a history can be argued
//! with: you can point at the day it formed and at what happened that day. A
//! belief that is simply asserted has nothing behind it, which is exactly the
//! shape that makes a character feel authored rather than lived.
//!
//! # A life document
//!
//! One file per episode, named for the date it covers and what it was called:
//!
//! ```text
//! layers/life/cindy-tan/1998 Nanyang.md                      ← a year
//! layers/life/cindy-tan/1998-09 First Term.md                ← a month
//! layers/life/cindy-tan/1998-09-14 First Week at Nanyang.md  ← a day
//! ```
//!
//! **Date first, so lexical order is chronological order.** Sorting is then
//! free, and a mis-dated file sorts visibly wrong instead of quietly landing in
//! the wrong decade. Both halves become conversation metadata at ingest, so the
//! substrate knows when each episode happened and what it was called.
//!
//! The body is ChatML prose in the character's own second person — the same
//! register the semantic layers use — and it may contain `<tool_call>` blocks.
//!
//! # Three precisions, and why lexical order still does all the work
//!
//! A life is written at three grains: every year, every month inside it, and
//! the days that became memories. That is not three kinds of file — it is one
//! kind, dated to the resolution it covers, and [`Precision`] is read back off
//! the name rather than declared inside it.
//!
//! The reason the grains can share a directory is that a plain lexical sort
//! orders them correctly *and* hierarchically:
//!
//! ```text
//! 1998            <  1998-09  <  1998-09-14  <  1998-10  <  1999
//! ```
//!
//! A shorter string sorts before any string it prefixes, so a year's overview
//! precedes its own months and each month precedes its own days — the ordering
//! the reader wants, from the ordering that was already free.
//!
//! # Why every month is written, and only some days
//!
//! The coarse strata are **continuous**: every year of a life, every month of
//! every year. Nothing is skipped for being uneventful, because a person has a
//! sense of every season they lived through, and an engine whose whole claim is
//! unbounded context has no reason to make a character's memory sparse to save
//! room. Days are the exception, and they are selective on purpose: a day gets
//! its own document when it became a *memory*, which is what makes it worth
//! recalling separately from the month that contains it.
//!
//! So the two grains do different work. Years and months are the gist — the
//! texture of a period, cheap to hold standing. Days are episodic — many, and
//! retrieved when something makes them relevant.
//!
//! # Absolute dates, not T-days
//!
//! An earlier sketch had days-since-birth. Absolute dates are simpler and they
//! are the only thing that works across a shared world: two characters born
//! forty years apart have to be able to read the same world document and agree
//! about when it happened. Arithmetic against a birth date is available to
//! anything that wants it; a T-day baked into the file is not.
//!
//! # The tool calls are the point
//!
//! A life document does not *describe* that she came to distrust Hess. It
//! contains the call that forms that belief, at the point in her life where it
//! formed:
//!
//! ```text
//! <tool_call>
//! {"name":"form_belief","arguments":{"statement":"Hess burned the east granary","confidence":0.9}}
//! </tool_call>
//! ```
//!
//! Prefilling the document simulates that call, and the belief lands in the
//! substrate as a tagged record. The mechanism that creates a belief during
//! authoring is the mechanism that creates one at run time — one path, not two,
//! which is what keeps the authored character and the lived character the same
//! kind of thing.
//!
//! # What does not move here
//!
//! **Semantic memory stays where it is, undated.** `layers/memory/<who>/` holds
//! what a character knows and is like — "she is analytical", "she pulled
//! all-nighters in graduate school". Those are traits and standing knowledge,
//! not episodes, and there is no honest date to give them. Forcing them into a
//! dated life would mean inventing an episode for every disposition, which is
//! writing fiction to satisfy a schema.
//!
//! Beliefs are different and that is why they move: a belief is a *conclusion*,
//! and a conclusion without an origin is the thing this module exists to end.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// How much of the calendar one document covers.
///
/// Ordered coarse-to-fine, and the derived `Ord` follows that order — which is
/// also the order the documents sort in, so a comparison on `Precision` and a
/// comparison on the dates themselves never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// `YYYY` — a year of the character's life.
    Year,
    /// `YYYY-MM` — a month inside one.
    Month,
    /// `YYYY-MM-DD` — a day that became a memory.
    Day,
}

impl Precision {
    /// How many characters of date this precision writes.
    pub fn width(self) -> usize {
        match self {
            Precision::Year => 4,
            Precision::Month => 7,
            Precision::Day => 10,
        }
    }

    /// Finest first. [`parse_name`] must try them in this order: `1998-09-14`
    /// also matches the `YYYY` shape at its first four characters, so a
    /// coarse-first walk would read every day as a year with a numeric title.
    pub const FINEST_FIRST: &'static [Precision] =
        &[Precision::Day, Precision::Month, Precision::Year];

    /// The stratum this precision names in a tag and in the console.
    pub fn label(self) -> &'static str {
        match self {
            Precision::Year => "year",
            Precision::Month => "month",
            Precision::Day => "day",
        }
    }
}

/// One episode in a character's life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Episode {
    /// `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, from the filename. Kept as a string
    /// rather than a date type: it is metadata to carry and sort by, every
    /// operation this module performs on it is lexical, and a date type would
    /// have to invent a month and a day for the coarser grains — which is
    /// precisely the false precision the three shapes exist to avoid.
    pub date: String,
    /// The episode's title, from the filename with the date and extension
    /// stripped. Becomes the conversation's title.
    pub title: String,
    /// Which grain this document is written at, read back off its name.
    pub precision: Precision,
    /// Whose life this is.
    pub who: String,
    pub path: PathBuf,
}

impl Episode {
    /// The year this episode falls in, whatever its grain — the first four
    /// characters, which every precision shares.
    pub fn year(&self) -> &str {
        &self.date[..4]
    }
}

/// Why a file in a life directory is not an episode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotAnEpisode {
    /// No `YYYY-MM-DD ` prefix.
    ///
    /// Reported rather than skipped. A life document that silently does not load
    /// is a hole in a character's history that nothing announces, and the author
    /// will find out from behaviour rather than from a log line.
    Undated(PathBuf),
    /// Dated, but nothing after the date.
    Untitled(PathBuf),
}

impl NotAnEpisode {
    pub fn message(&self) -> String {
        match self {
            NotAnEpisode::Undated(p) => format!(
                "{}: no date — a life document is named `YYYY Title.md`, \
                 `YYYY-MM Title.md` or `YYYY-MM-DD Title.md`",
                p.display()
            ),
            NotAnEpisode::Untitled(p) => {
                format!(
                    "{}: dated but untitled — the date needs a title after it",
                    p.display()
                )
            }
        }
    }
}

/// A filename read back into its parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Named {
    pub date: String,
    pub title: String,
    pub precision: Precision,
}

/// Does `s` carry an exact date of this precision at its head?
///
/// Shape only — `NNNN`, `NNNN-NN`, `NNNN-NN-NN`. Nothing here checks that a
/// month is 1..=12, and deliberately: a `2001-13` is a real authoring mistake
/// worth reporting *by name* further up, not a file to pretend is undated.
fn shaped(s: &str, p: Precision) -> bool {
    let b = s.as_bytes();
    b.len() >= p.width()
        && b[..p.width()].iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
}

/// Parse a life document's name into its date, its title and its grain.
///
/// Strict, and strict in a specific direction: it accepts exactly the three
/// shapes and refuses everything else rather than salvaging what it can. A
/// loose parse would file `19980914 No Dashes.md` under something plausible,
/// and the author would never learn the name was misread — they would find out
/// from a character that does not remember something.
///
/// **Finest precision first.** `1998-09-14` also matches `YYYY` at its first
/// four characters, so a coarse-first walk would read every day document as the
/// year 1998 with the title `09-14 …`.
pub fn parse_name(file: &Path) -> Result<Named, NotAnEpisode> {
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let undated = || NotAnEpisode::Undated(file.to_path_buf());

    for &p in Precision::FINEST_FIRST {
        if !shaped(stem, p) {
            continue;
        }
        let rest = &stem[p.width()..];

        // **A hyphen followed by a digit means the date itself is malformed,
        // not that the title starts with one.** `1998-9-14 Unpadded` matches
        // `YYYY` and would otherwise become the year 1998 titled `9-14
        // Unpadded` — a wrong date that reads as a right one. Only checked
        // below `Day`, because at the finest grain there is no longer form the
        // author could have been reaching for, so `1998-09-14-3rd-attempt` is
        // an ordinary slug.
        if p < Precision::Day {
            let mut c = rest.chars();
            if c.next() == Some('-') && c.next().is_some_and(|d| d.is_ascii_digit()) {
                return Err(undated());
            }
        }

        // A space, hyphen or underscore may separate the date from the title,
        // so `2024-03-01 The Fire` and `2024-03-01-the-fire` read the same. A
        // date that runs straight into a non-separator is not a date at all.
        if !rest.is_empty() && !rest.starts_with([' ', '-', '_']) {
            continue;
        }
        // The title is verbatim past the separators — an author's
        // capitalisation and punctuation are theirs.
        let title = rest.trim_start_matches([' ', '-', '_']).trim();
        if title.is_empty() {
            return Err(NotAnEpisode::Untitled(file.to_path_buf()));
        }
        return Ok(Named {
            date: stem[..p.width()].to_string(),
            title: title.to_string(),
            precision: p,
        });
    }
    Err(undated())
}

/// Every episode in one character's life directory, **in chronological order**.
///
/// Returns what could not be read alongside what could: a life with a hole in it
/// is still a life, and the author needs to be told which document fell out
/// rather than left to infer it from a character that does not remember
/// something.
pub fn episodes(dir: &Path, who: &str) -> (Vec<Episode>, Vec<NotAnEpisode>) {
    let mut found = Vec::new();
    let mut rejected = Vec::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return (found, rejected);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() || !crate::engine::watcher::is_ingestible(&path) {
            continue;
        }
        match parse_name(&path) {
            Ok(n) => found.push(Episode {
                date: n.date,
                title: n.title,
                precision: n.precision,
                who: who.to_string(),
                path,
            }),
            Err(e) => rejected.push(e),
        }
    }
    // Chronological *and* hierarchical, from one lexical sort. A shorter date
    // sorts before every date it prefixes, so `1998` precedes `1998-09`, which
    // precedes `1998-09-14` — each stratum's overview ahead of the finer
    // documents inside it. That property is the whole reason the date leads the
    // filename. The title breaks ties so two episodes at the same date and
    // grain keep a stable order across runs.
    found.sort_by(|a, b| a.date.cmp(&b.date).then_with(|| a.title.cmp(&b.title)));
    (found, rejected)
}

/// Which characters have a life to run.
///
/// `layers/life/<personality>/`. Absent is ordinary — most characters have no
/// authored life yet, and that is a character with no beliefs rather than an
/// error.
pub fn lives(mind: &Path, characters: &[String]) -> Vec<(String, PathBuf)> {
    let root = mind.join("layers").join("life");
    let mut out: Vec<(String, PathBuf)> = characters
        .iter()
        .filter_map(|who| {
            let dir = root.join(who);
            dir.is_dir().then(|| (who.clone(), dir))
        })
        .collect();
    out.sort();
    out
}

/// The address an episode is known by in the substrate.
///
/// `life/cindy-tan/1998-09-14 First Week at Nanyang` — the date is inside the
/// address, so a gather that matches on it can reach an episode by when it
/// happened as well as by what it was about.
pub fn address(ep: &Episode) -> String {
    format!("life/{}/{} {}", ep.who, ep.date, ep.title)
}

/// The gather-scope tags an episode's turns carry.
pub fn tags(ep: &Episode) -> Vec<String> {
    vec![
        "life".to_string(),
        format!("life:{}", ep.who),
        // The year, so a projection can scope to a period of a character's life
        // without parsing addresses.
        format!("life:{}:{}", ep.who, ep.year()),
        // The grain, so a gather can ask for the gist without dragging in every
        // day inside it — years and months are few and cheap to hold standing,
        // days are many and wanted only when something makes them relevant.
        format!("life:grain:{}", ep.precision.label()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("npcd-life-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn named(s: &str) -> Named {
        parse_name(&PathBuf::from("/m").join(s)).unwrap()
    }

    #[test]
    fn a_dated_title_parses_into_its_parts() {
        let n = named("1998-09-14 First Week at Nanyang.md");
        assert_eq!(n.date, "1998-09-14");
        assert_eq!(n.title, "First Week at Nanyang");
        assert_eq!(n.precision, Precision::Day);
    }

    /// **The three grains, each read back off its own name.** Nothing inside a
    /// document declares which it is.
    #[test]
    fn each_precision_is_recognised_by_its_shape() {
        assert_eq!(named("1998 Nanyang.md").precision, Precision::Year);
        assert_eq!(named("1998-09 First Term.md").precision, Precision::Month);
        assert_eq!(named("1998-09-14 First Week.md").precision, Precision::Day);
        // And each keeps only its own date, not a padded one — a month document
        // must not claim to be about the first of the month.
        assert_eq!(named("1998 Nanyang.md").date, "1998");
        assert_eq!(named("1998-09 First Term.md").date, "1998-09");
    }

    /// **A day must not read as a year with a numeric title.** `1998-09-14`
    /// matches the `YYYY` shape at its first four characters, so the walk has
    /// to go finest-first. This is the test that pins that order.
    #[test]
    fn a_finer_date_wins_over_the_coarser_shape_inside_it() {
        assert_eq!(named("1998-09-14 A Day.md").date, "1998-09-14");
        assert_eq!(named("1998-09 A Month.md").date, "1998-09");
        assert_eq!(
            Precision::FINEST_FIRST,
            &[Precision::Day, Precision::Month, Precision::Year],
            "the parse order is load-bearing, not incidental"
        );
    }

    /// Either separator, so an author who prefers slugs is not fought with.
    #[test]
    fn a_hyphen_separator_reads_the_same_as_a_space() {
        let n = named("2001-04-02-the-granary-fire.md");
        assert_eq!(n.date, "2001-04-02");
        assert_eq!(n.title, "the-granary-fire");
        // And at the coarser grains, where the same hyphen could have been the
        // start of a longer date.
        assert_eq!(named("1998-nanyang.md").title, "nanyang");
        assert_eq!(named("1998-09-first-term.md").date, "1998-09");
    }

    /// **A loose parse is worse than a refusal.** The author would never learn
    /// the name was misread — they would find out from a character that does
    /// not remember something.
    #[test]
    fn an_undated_file_is_refused_rather_than_guessed_at() {
        for name in [
            "about_courage.md",
            "24-01-01 Short Year.md",
            "19980914 No Dashes.md",
            "1998x A Letter.md",
        ] {
            let p = PathBuf::from("/m").join(name);
            assert!(
                matches!(parse_name(&p), Err(NotAnEpisode::Undated(_))),
                "{name} was accepted as dated"
            );
        }
    }

    /// **A malformed date must not degrade into a coarser one.** `1998-9-14` is
    /// an unpadded day, and reading it as the year 1998 titled `9-14 Unpadded`
    /// produces a wrong date that looks like a right one.
    #[test]
    fn an_unpadded_date_is_refused_rather_than_read_as_a_coarser_grain() {
        for name in [
            "1998-9-14 Unpadded.md",
            "1998-09-1 Unpadded Day.md",
            "1998-9 Unpadded Month.md",
        ] {
            let p = PathBuf::from("/m").join(name);
            assert!(
                matches!(parse_name(&p), Err(NotAnEpisode::Undated(_))),
                "{name} degraded to a coarser date instead of being refused"
            );
        }
        // But at the finest grain there is no longer form to have meant, so a
        // title that genuinely starts with a digit is ordinary.
        assert_eq!(named("1998-09-14-3rd-attempt.md").title, "3rd-attempt");
    }

    #[test]
    fn a_date_with_no_title_is_refused_at_every_grain() {
        for name in ["1998.md", "1998-09.md", "1998-09-14.md", "1998-09-14 .md"] {
            let p = PathBuf::from("/m").join(name);
            assert!(
                matches!(parse_name(&p), Err(NotAnEpisode::Untitled(_))),
                "{name} was accepted without a title"
            );
        }
    }

    /// **The property the naming standard exists for.** Lexical order is
    /// chronological order, so ordering is free and a mis-dated file sorts
    /// visibly wrong rather than landing quietly in the wrong decade.
    #[test]
    fn episodes_come_back_in_chronological_order() {
        let dir = tmp("order");
        for name in [
            "2003-01-09 Later.md",
            "1998-09-14 Earliest.md",
            "2001-12-31 Middle.md",
        ] {
            std::fs::write(dir.join(name), "# x").unwrap();
        }
        let (eps, rejected) = episodes(&dir, "cindy-tan");
        assert!(rejected.is_empty());
        let titles: Vec<&str> = eps.iter().map(|e| e.title.as_str()).collect();
        assert_eq!(titles, vec!["Earliest", "Middle", "Later"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two episodes on one day keep a stable order, so a run is reproducible.
    #[test]
    fn a_shared_date_is_broken_by_title() {
        let dir = tmp("sameday");
        std::fs::write(dir.join("1998-09-14 Second.md"), "x").unwrap();
        std::fs::write(dir.join("1998-09-14 First.md"), "x").unwrap();
        let (eps, _) = episodes(&dir, "who");
        assert_eq!(
            eps.iter().map(|e| e.title.as_str()).collect::<Vec<_>>(),
            vec!["First", "Second"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A life with a hole in it is still a life.** The undated document is
    /// reported and the rest still load — silently skipping it would leave the
    /// author to discover the gap from a character that does not remember
    /// something.
    #[test]
    fn an_undated_document_is_reported_and_the_rest_still_load() {
        let dir = tmp("hole");
        std::fs::write(dir.join("1998-09-14 Good.md"), "x").unwrap();
        std::fs::write(dir.join("about_courage.md"), "x").unwrap();
        let (eps, rejected) = episodes(&dir, "who");
        assert_eq!(eps.len(), 1);
        assert_eq!(rejected.len(), 1);
        assert!(rejected[0].message().contains("about_courage"));
        assert!(rejected[0].message().contains("YYYY-MM-DD"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_character_with_no_life_directory_is_ordinary() {
        let mind = tmp("nolife");
        std::fs::create_dir_all(mind.join("layers/life/zen")).unwrap();
        let found = lives(&mind, &["zen".into(), "keeper".into()]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "zen");
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// The date is inside the address, so a gather can reach an episode by when
    /// it happened and not only by what it was about.
    #[test]
    fn an_address_carries_the_date() {
        let ep = Episode {
            date: "1998-09-14".into(),
            title: "First Week at Nanyang".into(),
            precision: Precision::Day,
            who: "cindy-tan".into(),
            path: PathBuf::new(),
        };
        assert_eq!(
            address(&ep),
            "life/cindy-tan/1998-09-14 First Week at Nanyang"
        );
        assert!(tags(&ep).contains(&"life:cindy-tan".to_string()));
        assert!(tags(&ep).contains(&"life:cindy-tan:1998".to_string()));
        assert!(tags(&ep).contains(&"life:grain:day".to_string()));
    }

    /// **The property the three grains rest on.** One lexical sort orders the
    /// strata hierarchically: a year's overview lands ahead of its own months,
    /// and each month ahead of its own days, because a shorter string sorts
    /// before every string it prefixes. Nothing computes this — it is free.
    #[test]
    fn the_grains_interleave_into_a_hierarchy_under_one_sort() {
        let dir = tmp("grains");
        for name in [
            "1999 The Year After.md",
            "1998-09-14 A Day.md",
            "1998 Nanyang.md",
            "1998-10 October.md",
            "1998-09 First Term.md",
            "1998-09-30 Another Day.md",
        ] {
            std::fs::write(dir.join(name), "x").unwrap();
        }
        let (eps, rejected) = episodes(&dir, "cindy-tan");
        assert!(rejected.is_empty(), "{rejected:?}");
        assert_eq!(
            eps.iter().map(|e| e.date.as_str()).collect::<Vec<_>>(),
            vec![
                "1998",
                "1998-09",
                "1998-09-14",
                "1998-09-30",
                "1998-10",
                "1999"
            ]
        );
        // Every grain reports the year it belongs to, whatever its width.
        assert!(eps.iter().take(5).all(|e| e.year() == "1998"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A year and a month both survive the round trip through a real directory,
    /// with their grain intact — the parse is not only exercised on strings.
    #[test]
    fn coarse_documents_load_from_a_directory_with_their_grain() {
        let dir = tmp("coarse");
        std::fs::write(dir.join("1998 Nanyang.md"), "x").unwrap();
        std::fs::write(dir.join("1998-09 First Term.md"), "x").unwrap();
        let (eps, _) = episodes(&dir, "who");
        assert_eq!(
            eps.iter().map(|e| e.precision).collect::<Vec<_>>(),
            vec![Precision::Year, Precision::Month]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
