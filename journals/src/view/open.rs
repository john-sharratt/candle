//! A journal opened (§6.5 `journal_open`): its summary and how fresh, its
//! index, its window, and how much is folded into archives.

use crate::path::{JournalPath, Owner};
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::index::{group_lines, index, index_line};
use crate::view::who::{ago, render_line};
use crate::view::NOT_INSTRUCTIONS;

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
    if let Some(to) = j.meta.as_ref().and_then(|m| m.shared.as_ref()) {
        out.push(format!("Shared into the group as {to}."));
    }
    let idx = index(path, j, now);
    if !idx.is_empty() {
        out.push("Sub-journals:".into());
        out.extend(idx.into_iter().map(|l| format!("  - {l}")));
    }
    if let (true, Owner::Person(p)) = (path.is_root(), path.owner()) {
        let groups = group_lines(tree, p);
        if !groups.is_empty() {
            out.push("Groups:".into());
            out.extend(groups.into_iter().map(|l| format!("  - {l}")));
        }
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
    use crate::error::Problem;
    use crate::files::Kind;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, Op};
    use crate::period::Period;
    use crate::testing::{note, slug};

    #[test]
    fn an_opened_journal_reads_top_to_bottom() {
        let mut t = sharratt();
        let path = p("~");
        for i in 0..3 {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: path.clone(),
                    text: format!("Note {i}."),
                },
                now().plus(i * 60),
            )
            .unwrap();
        }
        let old = note("2026-09-01T10:00:00Z", "old");
        t.get_mut(&path)
            .unwrap()
            .archives
            .insert(Period::parse("2026-09-01").unwrap(), vec![old]);
        let text = open(&t, &path, t.get(&path).unwrap(), 2, now().plus(600));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], NOT_INSTRUCTIONS);
        assert_eq!(lines[1], "~. 4 lines, 2 sub-journals; no summary yet");
        assert_eq!(lines[2], "No summary yet.");
        assert_eq!(lines[3], "Sub-journals:");
        // Nothing is written under either section yet: named, not listed.
        assert_eq!(lines[4], "  - channels, people: nothing written yet");
        assert_eq!(lines[5], "Groups:");
        assert_eq!(
            lines[6],
            "  - @sharratt — Sharratt, a family; their default, in every conversation"
        );
        assert_eq!(lines[7], "Lines (the newest 2 of 3):");
        assert!(
            lines[8].ends_with("note, john via claude-code: Note 1."),
            "{}",
            lines[8]
        );
        assert!(lines[9].ends_with("Note 2."));
        assert_eq!(lines[10], "1 older line folded into 2026-09-01 — grep finds them, and each fold line above says what they held.");
        assert_eq!(lines.len(), 11);
    }

    /// **Every group is listed under `~`**, those that open no conversation
    /// too, so they are known to be there.
    #[test]
    fn a_persons_root_lists_every_group_and_how_it_opens() {
        let mut t = sharratt();
        apply(
            &mut t,
            &john(),
            &Op::CreateGroup {
                group: slug("battle-cities"),
                name: "Battle Cities".into(),
                kind: Kind::World,
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            group_lines(&t, &slug("john")),
            [
                "@sharratt — Sharratt, a family; their default, in every conversation",
                "@battle-cities — Battle Cities, a world; opened when needed",
            ]
        );
        apply(
            &mut t,
            &john(),
            &Op::ChooseGroups {
                default: None,
                framed: vec![slug("battle-cities")],
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            group_lines(&t, &slug("john"))[1],
            "@battle-cities — Battle Cities, a world; in every conversation"
        );
        assert!(group_lines(&t, &slug("stranger")).is_empty());
        // Without a default, a framed group is framed and no default.
        t.persons
            .get_mut(&slug("john"))
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .default_group = None;
        assert_eq!(
            group_lines(&t, &slug("john")),
            [
                "@sharratt — Sharratt, a family; opened when needed",
                "@battle-cities — Battle Cities, a world; in every conversation",
            ]
        );
    }

    #[test]
    fn a_broken_journal_is_opened_as_its_problems() {
        let mut t = sharratt();
        let path = p("topics");
        t.get_mut(&path).unwrap().problems.push(Problem::new(
            "groups/sharratt/topics/journal.yaml",
            Some(2),
            "bad",
        ));
        let text = open(&t, &path, t.get(&path).unwrap(), 50, now());
        assert!(
            text.ends_with("  - groups/sharratt/topics/journal.yaml:2: bad"),
            "{text}"
        );
    }
}
