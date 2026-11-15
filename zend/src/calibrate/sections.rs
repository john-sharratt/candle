//! A unit's calibration prompts asked as one message, each a part under its
//! own heading, and the answer split back into those parts.
//!
//! **One turn, not one per prompt.** Asked in turn in one draft, a later prompt
//! often found the reading gone from what it could see — "I don't have the
//! contents of `npcd/src/reports.rs` in this conversation" — and wrote nothing,
//! or a single question, or a tool call (BlackWorld, 2026-11-15: about half the
//! drafts kept only their first prompt's questions). Asked together, every part
//! is written with the reading in front of it, as the single prompt was, and
//! the parts see each other, so each can ask nothing another asked.

/// How the answer is to be laid out, after the parts.
const LAYOUT: &str = "Answer every part above, in this order, each under its own heading \
     exactly as written (`## name`), with one question per line and nothing else.";

/// What a priming-chain link's draft is told before the layout: its questions
/// are about its own file. `<<path>>` is the unit's own path.
///
/// **Only a chain link sees other files.** A link — a repository's README,
/// ARCHITECTURE, AGENTS or CLAUDE — is read behind every link before it
/// (`branch_ingest::prime`), and its draft, forked from its reading, sees them
/// all: asked about `yenbot/AGENTS.md`, a draft wrote ten of its fifteen
/// questions about `battle-cities/AGENTS.md` — Bevy, `i64` world coordinates,
/// the `sim` crate, none of them in the file (BlackWorld, 2026-11-15). Every
/// other unit is sealed behind the system prompt alone and is asked without
/// it. The chain cannot be taken out of the link's draft: the link's K/V was
/// computed behind it. A link's calibration marker digests this text
/// ([`super::spec::CalibrateSpec::marker`]), so a change to it calibrates the
/// links again and no other unit.
pub const OWN_FILE: &str = "Every question is about `<<path>>` alone, as you just read it. \
     Other files appear earlier in this conversation; ask nothing about them.";

/// The message asking every `(name, prompt)` part, each under a `## name`
/// heading, then — for a priming-chain link, the unit at `link` — [`OWN_FILE`],
/// then the layout.
pub fn compose(parts: &[(&str, String)], link: Option<&str>) -> String {
    let mut out = String::new();
    for (name, prompt) in parts {
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(prompt.trim_end());
        out.push_str("\n\n");
    }
    if let Some(path) = link {
        out.push_str(&OWN_FILE.replace("<<path>>", path));
        out.push(' ');
    }
    out.push_str(LAYOUT);
    out
}

/// `answer` split into the text under each of `names`' headings, in `names`'
/// order — empty for a part it did not answer — or `None` when it carries none
/// of the headings, so the caller can read it as one list.
///
/// A heading is a line naming a part, however it is marked: `## uses`,
/// `# Uses:`, `**uses**`. Text before the first heading, and under a heading
/// that names no part, belongs to no part.
pub fn split<'a>(answer: &'a str, names: &[&str]) -> Option<Vec<&'a str>> {
    let mut marks: Vec<(usize, usize, usize)> = Vec::new(); // (part, body start, line start)
    let mut at = 0usize;
    for line in answer.split_inclusive('\n') {
        if let Some(part) = heading(line, names) {
            marks.push((part, at + line.len(), at));
        }
        at += line.len();
    }
    if marks.is_empty() {
        return None;
    }
    let mut out = vec![""; names.len()];
    for (i, &(part, start, _)) in marks.iter().enumerate() {
        let end = marks.get(i + 1).map_or(answer.len(), |&(_, _, next)| next);
        if out[part].is_empty() {
            out[part] = &answer[start..end];
        }
    }
    Some(out)
}

/// Which of `names` the line is a heading for, if any.
fn heading(line: &str, names: &[&str]) -> Option<usize> {
    let s = line.trim();
    if !(s.starts_with('#') || s.starts_with("**")) {
        return None;
    }
    let s = s
        .trim_start_matches('#')
        .trim()
        .trim_matches('*')
        .trim()
        .trim_end_matches(':')
        .trim_matches('*')
        .trim();
    names.iter().position(|n| n.eq_ignore_ascii_case(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 3] = ["definitions", "uses", "purpose"];

    #[test]
    fn every_part_is_asked_under_its_heading_then_the_layout() {
        let parts = [
            ("definitions", "Write 4 about `a.rs`.\n".to_string()),
            ("uses", "Now write 3 more.".to_string()),
        ];
        assert_eq!(
            compose(&parts, None),
            "## definitions\nWrite 4 about `a.rs`.\n\n## uses\nNow write 3 more.\n\n\
             Answer every part above, in this order, each under its own heading exactly as \
             written (`## name`), with one question per line and nothing else."
        );
        // A priming-chain link is told its questions are about its own file.
        assert_eq!(
            compose(&parts[..1], Some("AGENTS.md")),
            "## definitions\nWrite 4 about `a.rs`.\n\n\
             Every question is about `AGENTS.md` alone, as you just read it. Other files \
             appear earlier in this conversation; ask nothing about them. Answer every part \
             above, in this order, each under its own heading exactly as written (`## name`), \
             with one question per line and nothing else."
        );
    }

    /// **Each heading's lines are its part's**, however the heading is
    /// marked, and a part left out is empty.
    #[test]
    fn an_answer_splits_by_its_headings_however_they_are_marked() {
        let answer = "Here they are:\n## definitions\nWhat is `A`?\nWhere is `B`?\n\
                      **Uses:**\nWhat calls `A`?\n";
        assert_eq!(
            split(answer, &NAMES),
            Some(vec![
                "What is `A`?\nWhere is `B`?\n",
                "What calls `A`?\n",
                ""
            ])
        );
        let answer = "# Definitions\nWhat is `A`?\n# purpose\nWhy?\n";
        assert_eq!(
            split(answer, &NAMES),
            Some(vec!["What is `A`?\n", "", "Why?\n"])
        );
    }

    /// An answer with no heading is no split: the caller reads it whole.
    #[test]
    fn an_answer_without_headings_is_no_split() {
        assert_eq!(split("What is `A`?\nWhat calls it?\n", &NAMES), None);
        assert_eq!(
            split("## Notes\nWhat is `A`?\n", &NAMES),
            None,
            "a heading naming no part"
        );
    }
}
