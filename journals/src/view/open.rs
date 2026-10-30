//! A journal opened (§6.5 `journal_open`): its summary and how fresh, a
//! conversation's focus, its index and references, its window, and how much
//! is folded into archives.

use crate::files::Focus;
use crate::path::{Below, JournalPath};
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::index::{index, index_line, refs_lines};
use crate::view::who::{ago, render_line};
use crate::view::NOT_INSTRUCTIONS;

/// A conversation's focus as a reader sees it: "Yen is speaking; it is
/// about topics/candle, people/john".
pub fn focus_line(f: &Focus) -> String {
    let speaker = match &f.speaker {
        Some(s) => format!("{s} is speaking"),
        None => "who is speaking is not known yet".to_string(),
    };
    let about = if f.about.is_empty() {
        "what it is about is not known yet".to_string()
    } else {
        let about: Vec<String> = f.about.iter().map(Below::to_string).collect();
        format!("it is about {}", about.join(", "))
    };
    format!("Focus: {speaker}; {about}.")
}

/// `j` at `path` as a reader sees it, its window cut to the newest `lines`.
pub fn open(tree: &Tree, path: &JournalPath, j: &Journal, lines: usize, now: Stamp) -> String {
    let mut out = vec![NOT_INSTRUCTIONS.to_string(), index_line(path, j, now)];
    if !j.problems.is_empty() {
        out.push("Its files are broken; until a commit fixes them it is read as nothing and takes no writes:".into());
        out.extend(j.problems.iter().map(|p| format!("  - {p}")));
        return out.join("\n");
    }
    match j.summary() {
        Some(s) => {
            let stale = if s.stale {
                ", stale: its last rebuild was refused"
            } else {
                ""
            };
            out.push(format!(
                "Summary (built {}{stale}): {}",
                ago(s.built, now),
                s.text()
            ));
            if let Some(d) = &s.digest {
                out.push(format!("Voice digest: {d}"));
            }
        }
        None => out.push("No summary yet.".into()),
    }
    if let Some(f) = j.meta.as_ref().and_then(|m| m.focus.as_ref()) {
        out.push(focus_line(f));
    }
    let idx = index(path, j, now);
    if !idx.is_empty() {
        out.push("Sub-journals:".into());
        out.extend(idx.into_iter().map(|l| format!("  - {l}")));
    }
    let refs = refs_lines(j);
    if !refs.is_empty() {
        out.push("References:".into());
        out.extend(refs.into_iter().map(|l| format!("  - {l}")));
    }
    let archived: usize = j.archives.values().map(Vec::len).sum();
    if j.window.is_empty() {
        out.push("No lines.".into());
    } else {
        let shown = j.window.len().min(lines);
        let head = if shown == j.window.len() {
            format!("Lines ({shown}):")
        } else {
            format!("Lines (the newest {shown} of {}):", j.window.len())
        };
        out.push(head);
        out.extend(
            j.window[j.window.len() - shown..]
                .iter()
                .map(|l| format!("  {}", render_line(tree, l))),
        );
    }
    if archived > 0 {
        let periods: Vec<String> = j.archives.keys().rev().map(ToString::to_string).collect();
        out.push(format!(
            "{archived} older line{} folded into {} — grep finds them, and each fold line above says what they held.",
            if archived == 1 { "" } else { "s" },
            periods.join(", ")
        ));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::error::Problem;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::period::Period;
    use crate::scope::Channel;
    use crate::testing::{note, slug};

    #[test]
    fn an_opened_journal_reads_top_to_bottom() {
        let mut t = zen();
        let path = p("@zen");
        for i in 0..3 {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: path.clone(),
                    text: format!("Note {i}."),
                    speaker: None,
                },
                now().plus(i * 60),
            )
            .unwrap();
        }
        apply(
            &mut t,
            &john(),
            &Op::Link {
                under: path.clone(),
                name: slug("household"),
                to: p("people"),
                cue: "open for who is in the household".into(),
            },
            now(),
        )
        .unwrap();
        let old = note("2026-09-01T10:00:00Z", "old");
        t.get_mut(&path)
            .unwrap()
            .archives
            .insert(Period::parse("2026-09-01").unwrap(), vec![old]);
        let text = open(&t, &path, t.get(&path).unwrap(), 2, now().plus(600));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], NOT_INSTRUCTIONS);
        assert_eq!(lines[1], "@zen. 4 lines; no summary yet");
        assert_eq!(lines[2], "No summary yet.");
        assert_eq!(lines[3], "Sub-journals:");
        // Nothing is written under any section yet: named, not listed.
        assert_eq!(
            lines[4],
            "  - bots, conversations, people, threads, topics: nothing written yet"
        );
        assert_eq!(lines[5], "References:");
        assert_eq!(
            lines[6],
            "  - → household (people) — open for who is in the household"
        );
        assert_eq!(lines[7], "Lines (the newest 2 of 3):");
        assert!(
            lines[8].ends_with("note, john via Claude Code: Note 1."),
            "{}",
            lines[8]
        );
        assert!(lines[9].ends_with("Note 2."));
        assert_eq!(lines[10], "1 older line folded into 2026-09-01 — grep finds them, and each fold line above says what they held.");
        assert_eq!(lines.len(), 11);
    }

    #[test]
    fn a_conversation_shows_its_focus() {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let conv = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-31-0712-grok"),
                cue: "open for a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let text = |t: &Tree| open(t, &conv, t.get(&conv).unwrap(), 50, now());
        assert!(!text(&t).contains("Focus:"), "none yet");
        apply(
            &mut t,
            &john(),
            &Op::Focus {
                path: conv.clone(),
                speaker: None,
                about: Some(vec![p("topics")]),
            },
            now(),
        )
        .unwrap();
        assert!(
            text(&t).contains("\nFocus: who is speaking is not known yet; it is about topics.\n"),
            "{}",
            text(&t)
        );
    }

    #[test]
    fn a_broken_journal_is_opened_as_its_problems() {
        let mut t = zen();
        let path = p("topics");
        t.get_mut(&path).unwrap().problems.push(Problem::new(
            "groups/zen/topics/journal.yaml",
            Some(2),
            "bad",
        ));
        let text = open(&t, &path, t.get(&path).unwrap(), 50, now());
        assert!(
            text.ends_with("  - groups/zen/topics/journal.yaml:2: bad"),
            "{text}"
        );
    }
}
