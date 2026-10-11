//! A plan node written out as the life document the ingest reads.
//!
//! # This is the only handoff
//!
//! The generator's entire contact with the running engine is a `.md` file in
//! `layers/life/<who>/`. Once it is in the mind folder — published, over a mind
//! under git — the watcher notices it, [`crate::engine::ingest`] puts it on a
//! conversation, and [`crate::engine::authoring`] executes the `<tool_call>`
//! blocks inside it. Nothing here writes to the substrate's memory, and
//! nothing here is privileged over a document somebody typed.
//!
//! That is worth defending rather than treating as an implementation detail. It
//! means a generated life can be read, corrected in a text editor, diffed, and
//! deleted — and that a bug in this module produces a bad file rather than a
//! bad character.
//!
//! # The filename is the contract
//!
//! [`crate::engine::life::parse_name`] reads a document's date and grain back
//! off its name, so a name this module writes and cannot read back is a hole in
//! a character's history that only shows up as a warning at load. Every write
//! goes through [`file_name`], and a test asserts the round trip for all three
//! grains and for titles hostile enough to break it.

use std::path::Path;

use serde::Serialize;

use super::consequence::render_all;
use super::plan::{is_safe_id, stored, NodeId, Plan};
use crate::engine::life::{parse_name, Precision};
use crate::sim::shelf::Shelf;

/// Characters a filename may not carry.
///
/// The Windows set, applied everywhere: a mind directory is routinely edited on
/// one platform and served from another, and a file that only exists on Linux
/// is a character whose history is missing on the machine that runs the game.
const FORBIDDEN: &[char] = &['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

/// Names Windows refuses whatever the extension.
const RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Longest title kept. A title is a document name and a conversation title, not
/// a paragraph; a model that returns one gets it cut rather than producing a
/// path the filesystem refuses.
const MAX_TITLE: usize = 90;

/// What a title becomes on disk.
///
/// Lossy on purpose, and in a direction that keeps the parse working: control
/// characters and path syntax go, runs of whitespace collapse, and the leading
/// characters that would be read as part of the date are stripped. What is left
/// is the author's — capitalisation and punctuation are theirs.
pub fn safe_title(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut space = false;
    for c in raw.chars() {
        if c.is_control() || FORBIDDEN.contains(&c) {
            continue;
        }
        if c.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        if space {
            out.push(' ');
            space = false;
        }
        out.push(c);
    }
    // Leading separators would be eaten by the parser as part of the date's
    // separator run, so a title of "- Nanyang" would come back as "Nanyang" and
    // the name would not round trip.
    let mut s = out.trim_matches([' ', '-', '_', '.']).to_string();
    if s.chars().count() > MAX_TITLE {
        s = s
            .chars()
            .take(MAX_TITLE)
            .collect::<String>()
            .trim_end()
            .to_string();
    }
    if s.is_empty() || RESERVED.iter().any(|r| r.eq_ignore_ascii_case(&s)) {
        // Something rather than nothing: an untitled document is refused by the
        // parser, and a character silently missing an episode is the failure
        // this whole path exists to avoid.
        s = format!("Untitled {s}").trim_end().to_string();
    }
    s
}

/// The filename a node's document is written under.
///
/// `None` for the story, which is undated and therefore not a life document —
/// it belongs in semantic memory, not in the dated history.
pub fn file_name(id: NodeId, title: &str) -> Option<String> {
    let key = id.date_key()?;
    let title = safe_title(title);
    Some(format!("{key} {}.md", strip_leading_key(&title, &key)))
}

/// Drop a leading repetition of the date the filename already begins with.
///
/// Asked for a year, a model titles it *"2751 Year of the First Battle"* — which
/// is a good title and becomes `2751 2751 Year of the First Battle.md` once the
/// key is prepended. Nine of one character's forty-three documents came out that
/// way; the rest had titled themselves without the year, which is the same model
/// making a free choice either way.
///
/// Only the node's **own** key is stripped, and only from the front. A year that
/// is genuinely part of the title — a document about 2751 filed under 2760 — is
/// not this, and survives.
fn strip_leading_key(title: &str, key: &str) -> String {
    let rest = match title.strip_prefix(key) {
        Some(r) => r.trim_start_matches([' ', '-', '_', '.', ':']),
        None => return title.to_string(),
    };
    // A title that was *only* the date has nothing left to say, and a filename
    // of just the key repeated is worse than the title it came from.
    if rest.is_empty() {
        return title.to_string();
    }
    rest.to_string()
}

/// The body of a life document: the prose, then the calls it produces.
///
/// The calls are appended rather than interleaved. An author writing by hand
/// puts a call at the moment in the episode where the belief formed, and that
/// reads well — but a generator cannot know where in prose it did not write
/// that moment falls, and guessing would attach the belief to the wrong
/// paragraph.
pub fn body(prose: &str, consequences: &str) -> String {
    let prose = prose.trim();
    if consequences.is_empty() {
        return format!("{prose}\n");
    }
    format!("{prose}\n\n{consequences}\n")
}

/// Where a character's life documents live in the mind.
///
/// `None` for an id that must not become a path — `who` reaches here from a
/// URL, and an unchecked one would let a request choose the folder this
/// module writes into and removes `.md` files from. See
/// [`super::plan::is_safe_id`].
pub fn life_dir(who: &str) -> Option<String> {
    is_safe_id(who).then(|| format!("layers/life/{who}"))
}

/// What one keeping of a plan did to its dated documents.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Written {
    pub wrote: usize,
    pub unchanged: usize,
    /// Documents that no longer correspond to any generated node.
    ///
    /// **Removed, not left.** A regeneration that produces fewer days would
    /// otherwise leave the character remembering an episode that has been
    /// written out of their history — and nothing anywhere would say so.
    pub removed: Vec<String>,
}

/// Keep `plan` on `shelf` as one commit: every generated node's document in
/// the character's life folder, the story in semantic memory once it is
/// generated, the removal of the dated documents the plan no longer accounts
/// for, and the plan itself. On an author's shelf that is their draft, to be
/// published with the rest of what they wrote (`docs/replicated_substrate.md`
/// §10.3).
///
/// Idempotent: a document whose text already matches is left alone, so a
/// re-run does not churn the ledger the ingest diffs against.
pub fn keep(shelf: &dyn Shelf, plan: &Plan) -> anyhow::Result<Written> {
    let who = &plan.seed.who;
    let unsafe_id = || anyhow::anyhow!("`{who}` is not a character id a path is built from");
    let dir = life_dir(who).ok_or_else(unsafe_id)?;

    let mut report = Written::default();
    let mut changes: Vec<(String, Option<String>)> = Vec::new();
    let mut kept: Vec<String> = Vec::new();

    for (id, prose, consequences) in nodes(plan) {
        let Some(name) = file_name(id, &prose.0) else {
            continue;
        };
        let text = body(&prose.1, &consequences);
        let path = format!("{dir}/{name}");
        kept.push(name);
        if shelf.read(&path).is_some_and(|old| old == text) {
            report.unchanged += 1;
            continue;
        }
        changes.push((path, Some(text)));
        report.wrote += 1;
    }

    // Anything dated in the folder that the plan no longer accounts for.
    // Undated files are left alone: they are not ours, and an author may keep
    // notes beside a life.
    for name in shelf.list(&dir) {
        if name.ends_with('/') || !name.ends_with(".md") {
            continue;
        }
        if parse_name(Path::new(&name)).is_err() || kept.contains(&name) {
            continue;
        }
        changes.push((format!("{dir}/{name}"), None));
        report.removed.push(name);
    }
    report.removed.sort();

    // Only a story that was actually generated is kept: a redo that cleared it
    // and failed to refill it must not write the empty node over the good
    // document that is there.
    if plan.story.content.is_generated() {
        let path = story_path(who).ok_or_else(unsafe_id)?;
        let text = format!("{}\n", plan.story.content.text.trim());
        if shelf.read(&path).as_deref() != Some(text.as_str()) {
            changes.push((path, Some(text)));
        }
    }

    let (path, text) = stored(plan)?;
    if shelf.read(&path).as_deref() != Some(text.as_str()) {
        changes.push((path, Some(text)));
    }
    if !changes.is_empty() {
        shelf.commit(&changes).map_err(anyhow::Error::msg)?;
    }
    Ok(report)
}

/// Every node of a plan that has something to write, as
/// `(id, (title, prose), consequences)`.
#[allow(clippy::type_complexity)]
fn nodes(plan: &Plan) -> Vec<(NodeId, (String, String), String)> {
    let mut out = Vec::new();
    for y in &plan.years {
        if y.content.is_generated() {
            out.push((
                NodeId::Year { year: y.year },
                (y.content.title.clone(), y.content.text.clone()),
                String::new(),
            ));
        }
        for m in &y.months {
            if m.content.is_generated() {
                out.push((
                    NodeId::Month {
                        year: y.year,
                        month: m.month,
                    },
                    (m.content.title.clone(), m.content.text.clone()),
                    String::new(),
                ));
            }
            for d in &m.days {
                if d.content.is_generated() {
                    out.push((
                        NodeId::Day {
                            year: y.year,
                            month: m.month,
                            day: d.day,
                        },
                        (d.content.title.clone(), d.content.text.clone()),
                        render_all(&d.consequences),
                    ));
                }
            }
        }
    }
    out
}

/// Where the story goes in the mind: semantic memory, not the dated history.
///
/// [`crate::engine::life`] is explicit that `layers/memory/<who>/` holds what a
/// character *is like* — traits and standing knowledge, with no honest date to
/// give them. A whole-life arc is exactly that, and dating it would mean
/// inventing a day on which a character acquired their own disposition.
pub fn story_path(who: &str) -> Option<String> {
    is_safe_id(who).then(|| format!("layers/memory/{who}/life-story.md"))
}

/// The grain a node's document is written at, for the console.
pub fn precision(id: NodeId) -> Option<Precision> {
    match id {
        NodeId::Story => None,
        NodeId::Year { .. } => Some(Precision::Year),
        NodeId::Month { .. } => Some(Precision::Month),
        NodeId::Day { .. } => Some(Precision::Day),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use super::*;
    use crate::drafts::disk::Disk;
    use crate::lifegen::consequence::Consequence;
    use crate::lifegen::plan::Plan;
    use crate::lifegen::seed::{check, Cadence, Seed};

    /// A model that titles its year with the year gets one date in the filename,
    /// not two. Nine of one character's forty-three came out as
    /// `2751 2751 Year of the First Battle.md`.
    #[test]
    fn a_title_that_repeats_its_own_date_does_not_get_it_twice() {
        let y = NodeId::Year { year: 2751 };
        assert_eq!(
            file_name(y, "2751 Year of the First Battle").unwrap(),
            "2751 Year of the First Battle.md"
        );
        // Separators between the date and the title go with it.
        assert_eq!(
            file_name(y, "2751 - The Year of Hardened Resolve").unwrap(),
            "2751 The Year of Hardened Resolve.md"
        );
        // A title that never mentioned the year is untouched.
        assert_eq!(
            file_name(y, "Year of Injured Pride").unwrap(),
            "2751 Year of Injured Pride.md"
        );
    }

    /// A date that is genuinely part of the title, rather than a repetition of
    /// this node's own, is the author's and stays.
    #[test]
    fn a_different_year_in_a_title_is_not_stripped() {
        let y = NodeId::Year { year: 2760 };
        assert_eq!(
            file_name(y, "2751 Remembered").unwrap(),
            "2760 2751 Remembered.md"
        );
    }

    /// A title that was only the date leaves nothing to name the document by, so
    /// it keeps what it had rather than becoming a bare key.
    #[test]
    fn a_title_that_is_only_its_date_keeps_it() {
        let y = NodeId::Year { year: 2751 };
        assert_eq!(file_name(y, "2751").unwrap(), "2751 2751.md");
    }

    /// A shelf over a folder that counts the commits made to it.
    struct Counting {
        disk: Disk,
        commits: Mutex<Vec<usize>>,
    }

    impl Shelf for Counting {
        fn read(&self, key: &str) -> Option<String> {
            self.disk.read(key)
        }
        fn list(&self, dir: &str) -> BTreeSet<String> {
            self.disk.list(dir)
        }
        fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String> {
            self.commits.lock().unwrap().push(changes.len());
            self.disk.commit(changes)
        }
    }

    fn counting(dir: &tempfile::TempDir) -> Counting {
        Counting {
            disk: Disk::new(dir.path()),
            commits: Mutex::new(Vec::new()),
        }
    }

    fn plan() -> Plan {
        let seed = Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: "1998-09-14".into(),
            through: "1999-02-01".into(),
            place: "Nanyang".into(),
            role: "a clerk".into(),
            cadence: Cadence::Even,
            facts: Vec::new(),
            world: Vec::new(),
            cast: Vec::new(),
            eras: Vec::new(),
        };
        Plan::new(&check(&seed).unwrap())
    }

    /// **The property every write depends on.** A name this module writes and
    /// the parser cannot read back is a hole in a history that surfaces only as
    /// a warning at load.
    #[test]
    fn every_written_name_parses_back_to_the_same_date_and_grain() {
        let cases = [
            (
                NodeId::Year { year: 1998 },
                "Nanyang",
                "1998",
                Precision::Year,
            ),
            (
                NodeId::Month {
                    year: 1998,
                    month: 9,
                },
                "First Term",
                "1998-09",
                Precision::Month,
            ),
            (
                NodeId::Day {
                    year: 1998,
                    month: 9,
                    day: 4,
                },
                "First Week at Nanyang",
                "1998-09-04",
                Precision::Day,
            ),
        ];
        for (id, title, date, grain) in cases {
            let name = file_name(id, title).unwrap();
            let parsed = parse_name(Path::new(&name)).unwrap();
            assert_eq!(parsed.date, date, "{name}");
            assert_eq!(parsed.title, title, "{name}");
            assert_eq!(parsed.precision, grain, "{name}");
        }
    }

    /// Titles hostile enough to break the name, and the round trip surviving
    /// all of them.
    #[test]
    fn a_hostile_title_is_made_safe_and_still_round_trips() {
        for raw in [
            "A/B: the \"fire\"?",
            "  leading and trailing  ",
            "- starts with a separator",
            "line\nbreak\ttab",
            "NUL",
            "",
            "...",
        ] {
            let id = NodeId::Day {
                year: 2001,
                month: 4,
                day: 2,
            };
            let name = file_name(id, raw).unwrap();
            let parsed = parse_name(Path::new(&name))
                .unwrap_or_else(|e| panic!("{raw:?} produced {name:?}: {}", e.message()));
            assert_eq!(parsed.date, "2001-04-02", "{raw:?} → {name:?}");
            assert!(!parsed.title.is_empty());
            assert!(
                !parsed
                    .title
                    .chars()
                    .any(|c| FORBIDDEN.contains(&c) || c.is_control()),
                "{name:?} kept a forbidden character"
            );
        }
    }

    /// A leading separator would be eaten as part of the date's separator run,
    /// so the title must not start with one.
    #[test]
    fn a_title_starting_with_a_separator_keeps_its_words() {
        assert_eq!(
            safe_title("- starts with a separator"),
            "starts with a separator"
        );
        assert_eq!(safe_title("_x_"), "x");
    }

    #[test]
    fn an_overlong_title_is_cut_rather_than_refused() {
        let long = "word ".repeat(60);
        let t = safe_title(&long);
        assert!(t.chars().count() <= MAX_TITLE);
        assert!(!t.ends_with(' '));
    }

    /// The story is undated, so it has no life-document name at all.
    #[test]
    fn the_story_is_not_a_life_document() {
        assert_eq!(file_name(NodeId::Story, "An Arc"), None);
        assert_eq!(precision(NodeId::Story), None);
    }

    #[test]
    fn a_body_with_no_consequences_is_just_the_prose() {
        assert_eq!(body("  You came back.  ", ""), "You came back.\n");
    }

    #[test]
    fn a_body_carries_its_calls_after_the_prose() {
        let c = Consequence {
            tool: "form_belief".into(),
            args: match serde_json::json!({"statement": "Hess burned it"}) {
                serde_json::Value::Object(m) => m,
                _ => unreachable!(),
            },
        };
        let text = body("You came back.", &render_all(std::slice::from_ref(&c)));
        assert_eq!(
            text,
            "You came back.\n\n<tool_call>\n{\"name\":\"form_belief\",\
             \"arguments\":{\"statement\":\"Hess burned it\"}}\n</tool_call>\n"
        );
        // And it survives the parser the ingest uses.
        let parsed = crate::engine::authoring::parse(&text);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.prose, "You came back.");
    }

    /// **Every generated node, and the plan itself, is kept in one commit**,
    /// what was not generated is not written, and keeping it again changes
    /// nothing.
    #[test]
    fn keeping_writes_every_generated_node_and_the_plan_in_one_commit() {
        let mind = tempfile::tempdir().unwrap();
        let shelf = counting(&mind);
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1998 })
            .unwrap()
            .generated("Nanyang".into(), "You were born.".into());
        p.content_mut(NodeId::Month {
            year: 1998,
            month: 9,
        })
        .unwrap()
        .generated("September".into(), "It rained.".into());

        let r = keep(&shelf, &p).unwrap();
        assert_eq!((r.wrote, r.unchanged), (2, 0));
        assert_eq!(
            *shelf.commits.lock().unwrap(),
            [3],
            "two documents and the plan"
        );
        let dir = mind.path().join("layers/life/cindy-tan");
        assert_eq!(
            std::fs::read_to_string(dir.join("1998 Nanyang.md")).unwrap(),
            "You were born.\n"
        );
        assert!(dir.join("1998-09 September.md").is_file());
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            2,
            "ungenerated months are not written"
        );
        assert_eq!(
            std::fs::read_to_string(mind.path().join(".lifegen/cindy-tan.json")).unwrap(),
            serde_json::to_string_pretty(&p).unwrap()
        );

        let r2 = keep(&shelf, &p).unwrap();
        assert_eq!((r2.wrote, r2.unchanged), (0, 2));
        assert_eq!(*shelf.commits.lock().unwrap(), [3], "nothing churned");
    }

    /// **A regeneration that produces fewer days must not leave the character
    /// remembering an episode written out of their history.**
    #[test]
    fn a_document_the_plan_no_longer_accounts_for_is_removed() {
        let mind = tempfile::tempdir().unwrap();
        let shelf = Disk::new(mind.path());
        let mut p = plan();
        p.ensure_day(1998, 9, 14)
            .unwrap()
            .content
            .generated("A Day".into(), "It happened.".into());
        keep(&shelf, &p).unwrap();
        let day = mind
            .path()
            .join("layers/life/cindy-tan/1998-09-14 A Day.md");
        assert!(day.is_file());

        // The day is written out of the plan.
        p.month_mut(1998, 9).unwrap().days.clear();
        let r = keep(&shelf, &p).unwrap();
        assert_eq!(r.removed, vec!["1998-09-14 A Day.md".to_string()]);
        assert!(!day.exists());
    }

    /// An author's own notes beside a life are not the generator's to delete.
    #[test]
    fn an_undated_file_in_the_life_folder_is_left_alone() {
        let mind = tempfile::tempdir().unwrap();
        let dir = mind.path().join("layers/life/cindy-tan");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "mine").unwrap();
        let r = keep(&Disk::new(mind.path()), &plan()).unwrap();
        assert!(r.removed.is_empty());
        assert!(dir.join("notes.md").is_file());
    }

    /// **The story is semantic memory, not dated history.** Dating it would
    /// mean inventing a day on which a character acquired their own
    /// disposition. Only a generated story is kept.
    #[test]
    fn the_story_is_kept_in_semantic_memory() {
        let mind = tempfile::tempdir().unwrap();
        let shelf = Disk::new(mind.path());
        let mut p = plan();
        keep(&shelf, &p).unwrap();
        let story = mind.path().join("layers/memory/cindy-tan/life-story.md");
        assert!(!story.exists(), "an ungenerated story is not written");

        p.story
            .content
            .generated("An Arc".into(), "  She was always careful.  ".into());
        keep(&shelf, &p).unwrap();
        assert_eq!(
            std::fs::read_to_string(&story).unwrap(),
            "She was always careful.\n"
        );
        assert_eq!(
            story_path("cindy-tan").unwrap(),
            "layers/memory/cindy-tan/life-story.md"
        );
        assert_eq!(story_path("../x"), None);
        assert_eq!(life_dir("cindy-tan").unwrap(), "layers/life/cindy-tan");
    }
}
