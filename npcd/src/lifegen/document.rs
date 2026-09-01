//! A plan node written out as the life document the ingest reads.
//!
//! # This is the only handoff
//!
//! The generator's entire contact with the running engine is a `.md` file in
//! `layers/life/<who>/`. The watcher notices it, [`crate::engine::ingest`] puts
//! it on a conversation, and [`crate::engine::authoring`] executes the
//! `<tool_call>` blocks inside it. Nothing here writes to the substrate, and
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

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::consequence::render_all;
use super::plan::{is_safe_id, NodeId, Plan};
use crate::engine::life::{parse_name, Precision};

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
    Some(format!("{} {}.md", id.date_key()?, safe_title(title)))
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

/// Where a character's life documents live.
///
/// `None` for an id that must not become a path — `who` reaches here from a
/// URL, and an unchecked one would let a request choose the directory this
/// module creates, writes into, and deletes `.md` files from. See
/// [`super::plan::is_safe_id`].
pub fn life_dir(mind: &Path, who: &str) -> Option<PathBuf> {
    is_safe_id(who).then(|| mind.join("layers").join("life").join(who))
}

/// What one write-out did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Written {
    pub wrote: usize,
    pub unchanged: usize,
    /// Documents on disk that no longer correspond to any generated node.
    ///
    /// **Removed, not left.** A regeneration that produces fewer days would
    /// otherwise leave the character remembering an episode that has been
    /// written out of their history — and nothing anywhere would say so.
    pub removed: Vec<String>,
}

/// Write every generated node of a plan into the character's life directory,
/// and remove the documents that no longer belong to one.
///
/// Idempotent: a document whose bytes already match is left alone, so a
/// re-run does not churn the ledger the ingest diffs against.
pub fn sync(mind: &Path, plan: &Plan) -> anyhow::Result<Written> {
    let who = &plan.seed.who;
    let dir = life_dir(mind, who)
        .ok_or_else(|| anyhow::anyhow!("`{who}` is not a character id a path is built from"))?;
    std::fs::create_dir_all(&dir)?;

    let mut report = Written::default();
    let mut keep: Vec<String> = Vec::new();

    for (id, prose, consequences) in nodes(plan) {
        let Some(name) = file_name(id, &prose.0) else {
            continue;
        };
        let text = body(&prose.1, &consequences);
        keep.push(name.clone());
        let path = dir.join(&name);
        if std::fs::read_to_string(&path).is_ok_and(|old| old == text) {
            report.unchanged += 1;
            continue;
        }
        let tmp = path.with_extension("md.tmp");
        std::fs::write(&tmp, &text)?;
        std::fs::rename(&tmp, &path)?;
        report.wrote += 1;
    }

    // Anything dated in this directory that the plan no longer accounts for.
    // Undated files are left alone: they are not ours, and an author may keep
    // notes beside a life.
    for entry in std::fs::read_dir(&dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() || path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if parse_name(&path).is_err() || keep.iter().any(|k| k == name) {
            continue;
        }
        std::fs::remove_file(&path)?;
        report.removed.push(name.to_string());
    }
    report.removed.sort();
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

/// Where the story goes: semantic memory, not the dated history.
///
/// [`crate::engine::life`] is explicit that `layers/memory/<who>/` holds what a
/// character *is like* — traits and standing knowledge, with no honest date to
/// give them. A whole-life arc is exactly that, and dating it would mean
/// inventing a day on which a character acquired their own disposition.
pub fn story_path(mind: &Path, who: &str) -> Option<PathBuf> {
    is_safe_id(who).then(|| {
        mind.join("layers")
            .join("memory")
            .join(who)
            .join("life-story.md")
    })
}

/// Write the story into semantic memory.
pub fn write_story(mind: &Path, who: &str, text: &str) -> anyhow::Result<bool> {
    let path = story_path(mind, who)
        .ok_or_else(|| anyhow::anyhow!("`{who}` is not a character id a path is built from"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let text = format!("{}\n", text.trim());
    if std::fs::read_to_string(&path).is_ok_and(|old| old == text) {
        return Ok(false);
    }
    let tmp = path.with_extension("md.tmp");
    std::fs::write(&tmp, &text)?;
    std::fs::rename(&tmp, &path)?;
    Ok(true)
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
    use super::*;
    use crate::lifegen::consequence::Consequence;
    use crate::lifegen::plan::Plan;
    use crate::lifegen::seed::{check, Cadence, Seed};

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("npcd-doc-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
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
            "You came back.\n\n<tool_call>\n{\"arguments\":{\"statement\":\"Hess burned it\"},\
             \"name\":\"form_belief\"}\n</tool_call>\n"
        );
        // And it survives the parser the ingest uses.
        let parsed = crate::engine::authoring::parse(&text);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.prose, "You came back.");
    }

    #[test]
    fn syncing_writes_every_generated_node_and_skips_the_rest() {
        let mind = tmp("sync");
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

        let r = sync(&mind, &p).unwrap();
        assert_eq!((r.wrote, r.unchanged), (2, 0));
        let dir = life_dir(&mind, "cindy-tan").unwrap();
        assert!(dir.join("1998 Nanyang.md").is_file());
        assert!(dir.join("1998-09 September.md").is_file());
        assert!(
            !dir.join("1998-10 .md").exists(),
            "ungenerated months are not written"
        );

        // Idempotent: nothing churns on a second pass.
        let r2 = sync(&mind, &p).unwrap();
        assert_eq!((r2.wrote, r2.unchanged), (0, 2));
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// **A regeneration that produces fewer days must not leave the character
    /// remembering an episode written out of their history.**
    #[test]
    fn a_document_the_plan_no_longer_accounts_for_is_removed() {
        let mind = tmp("orphan");
        let mut p = plan();
        p.ensure_day(1998, 9, 14)
            .unwrap()
            .content
            .generated("A Day".into(), "It happened.".into());
        sync(&mind, &p).unwrap();
        let dir = life_dir(&mind, "cindy-tan").unwrap();
        assert!(dir.join("1998-09-14 A Day.md").is_file());

        // The day is written out of the plan.
        p.month_mut(1998, 9).unwrap().days.clear();
        let r = sync(&mind, &p).unwrap();
        assert_eq!(r.removed, vec!["1998-09-14 A Day.md".to_string()]);
        assert!(!dir.join("1998-09-14 A Day.md").exists());
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// An author's own notes beside a life are not the generator's to delete.
    #[test]
    fn an_undated_file_in_the_life_directory_is_left_alone() {
        let mind = tmp("notes");
        let dir = life_dir(&mind, "cindy-tan").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "mine").unwrap();
        let r = sync(&mind, &plan()).unwrap();
        assert!(r.removed.is_empty());
        assert!(dir.join("notes.md").is_file());
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// **The story is semantic memory, not dated history.** Dating it would
    /// mean inventing a day on which a character acquired their own
    /// disposition.
    #[test]
    fn the_story_is_written_into_semantic_memory() {
        let mind = tmp("story");
        assert!(write_story(&mind, "cindy-tan", "She was always careful.").unwrap());
        let p = story_path(&mind, "cindy-tan").unwrap();
        assert!(p.starts_with(mind.join("layers").join("memory").join("cindy-tan")));
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "She was always careful.\n"
        );
        // Idempotent, so an unchanged story does not churn the ingest ledger.
        assert!(!write_story(&mind, "cindy-tan", "She was always careful.").unwrap());
        let _ = std::fs::remove_dir_all(&mind);
    }
}
