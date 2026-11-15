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

/// What follows the parts: whose file every question is about, then how the
/// answer is laid out. `<<path>>` is the unit's own path.
///
/// **The unit's file alone.** A draft forks from the unit's reading, and the
/// reading carries the files read before it in its lineage: asked about
/// `yenbot/AGENTS.md`, a draft wrote ten of its fifteen questions about
/// `battle-cities/AGENTS.md` — Bevy, `i64` world coordinates, the `sim` crate,
/// none of them in the file (BlackWorld, 2026-11-15). Each such question is an
/// exemplar voting for the wrong file. The calibration marker digests this
/// text ([`super::spec::CalibrateSpec::marker`]), so a change to it calibrates
/// every unit again.
pub const CLOSING: &str = "Every question is about `<<path>>` alone, as you just read it. Other \
     files may appear earlier in this conversation; ask nothing about them. Answer every part \
     above, in this order, each under its own heading exactly as written (`## name`), with one \
     question per line and nothing else.";

/// The message asking every `(name, prompt)` part, each under a `## name`
/// heading, then [`CLOSING`] for the unit at `path`.
pub fn compose(parts: &[(&str, String)], path: &str) -> String {
    let mut out = String::new();
    for (name, prompt) in parts {
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(prompt.trim_end());
        out.push_str("\n\n");
    }
    out.push_str(&CLOSING.replace("<<path>>", path));
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
            compose(&parts, "src/a.rs"),
            "## definitions\nWrite 4 about `a.rs`.\n\n## uses\nNow write 3 more.\n\n\
             Every question is about `src/a.rs` alone, as you just read it. Other files may \
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
