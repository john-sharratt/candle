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

use candle_conversation::stencil::{
    BuildError, FreeTextLimits, StencilTreeBuilder, Terminator, TreeSpec,
};

/// How the answer is to be laid out, after the parts.
const LAYOUT: &str = "Answer every part above, in this order, each under its own heading \
     exactly as written (`## name`), with one question per line and nothing else.";

/// The message asking every `(name, prompt)` part, each under a `## name`
/// heading, then the layout. The draft's knowledge base holds its file alone
/// (§12 D11), so nothing else is there to be asked about.
pub fn compose(parts: &[(&str, String)]) -> String {
    let mut out = String::new();
    for (name, prompt) in parts {
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(prompt.trim_end());
        out.push_str("\n\n");
    }
    out.push_str(LAYOUT);
    out
}

/// Decode tokens a part may run to per question it asks for, before the
/// grammar moves it on: a developer's question runs to ~40 tokens.
const TOKENS_PER_QUESTION: u32 = 80;

/// The grammar a draft's answer is decoded under: each `(name, questions)`
/// part's heading written by the grammar, in order, and the model's own text
/// under it until it starts the next heading.
///
/// **Every part is reached.** Asked in free text, a draft that wrote nothing
/// under an optional part ended its answer there: the `code` profile's
/// `messages` comes before `purpose`, and 110 code units in a day left
/// `purpose` empty, one in five (BlackWorld, 2026-11-15). Under the grammar the
/// answer cannot end before the last part — a part's text ends only where the
/// model starts a heading (`\n##`) or at its budget, and the grammar then
/// writes the next part's heading itself — and no heading is missing,
/// misspelled or out of order. The questions are the model's own; only the
/// headings are fixed. The grammar is not what the unit is asked, so the
/// calibration marker does not digest it ([`super::spec::CalibrateSpec::marker`]).
pub fn grammar(parts: &[(&str, usize)]) -> Result<TreeSpec, BuildError> {
    let mut b = StencilTreeBuilder::new("calibration_draft").root("h0");
    for (i, &(name, questions)) in parts.iter().enumerate() {
        let heading = if i == 0 {
            format!("## {name}\n")
        } else {
            // `\n##` was the model's, closing the part before.
            format!(" {name}\n")
        };
        b = b.static_node(format!("h{i}"), heading, format!("b{i}"));
        let budget = (questions as u32).max(1) * TOKENS_PER_QUESTION;
        let last = i + 1 == parts.len();
        b = if last {
            b.free_text(
                format!("b{i}"),
                Terminator::Never,
                true,
                FreeTextLimits::think_flow(budget),
                "end",
            )
        } else {
            b.free_text(
                format!("b{i}"),
                Terminator::Until {
                    marker: "\n##",
                    tails: &[],
                },
                false,
                FreeTextLimits::think_flow(budget),
                format!("h{}", i + 1),
            )
        };
    }
    b.end("end").build()
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
    use candle_conversation::stencil::{compile, simulate, Oracle, TestVocab, Vocab};
    use std::sync::Arc;

    /// Decode `script` under the grammar for `parts` on a byte vocabulary:
    /// the text that entered the answer.
    fn decode(parts: &[(&str, usize)], script: &str) -> String {
        let vocab = TestVocab::new();
        let tree = Arc::new(compile(&grammar(parts).unwrap(), &vocab).unwrap());
        let mut tokens: Vec<u32> = script.bytes().map(u32::from).collect();
        tokens.push(256);
        let run = simulate(tree, &vocab, Oracle::Scripted(tokens), 10_000).unwrap();
        String::from_utf8_lossy(&vocab.decode(&run.tokens)).into_owned()
    }

    /// **The grammar writes every heading, in order, and the model writes the
    /// questions**: a part ends where the model starts a heading, and the next
    /// part's heading is the grammar's.
    #[test]
    fn the_grammar_writes_each_heading_and_the_model_each_part() {
        let answer = decode(
            &[("definitions", 2), ("uses", 1), ("purpose", 1)],
            "What is `a`?\nWhere is `b` defined?\n##What calls `a`?\n##What is the file for?\n",
        );
        assert_eq!(
            answer,
            "## definitions\nWhat is `a`?\nWhere is `b` defined?\n## uses\nWhat calls `a`?\n## \
             purpose\nWhat is the file for?\n"
        );
        assert_eq!(
            split(&answer, &["definitions", "uses", "purpose"]),
            Some(vec![
                "What is `a`?\nWhere is `b` defined?\n",
                "What calls `a`?\n",
                "What is the file for?\n"
            ])
        );
    }

    /// **An empty part does not end the answer**: the model cannot end its turn
    /// before the last part, so a part it leaves empty is followed by the next.
    #[test]
    fn an_empty_part_is_followed_by_the_next() {
        let answer = decode(
            &[("messages", 1), ("purpose", 1)],
            "\n##What is the file for?\n",
        );
        assert_eq!(answer, "## messages\n\n## purpose\nWhat is the file for?\n");
    }

    const NAMES: [&str; 3] = ["definitions", "uses", "purpose"];

    #[test]
    fn every_part_is_asked_under_its_heading_then_the_layout() {
        let parts = [
            ("definitions", "Write 4 about `a.rs`.\n".to_string()),
            ("uses", "Now write 3 more.".to_string()),
        ];
        assert_eq!(
            compose(&parts),
            "## definitions\nWrite 4 about `a.rs`.\n\n## uses\nNow write 3 more.\n\n\
             Answer every part above, in this order, each under its own heading exactly as \
             written (`## name`), with one question per line and nothing else."
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
