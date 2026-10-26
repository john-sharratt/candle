//! What the writer is asked to write about a stretch of its own history.
//!
//! The question carries the stretch as text, because a claim cites a turn by
//! number and the live history is not numbered. Each turn the writer may rely on
//! carries its number; a turn it may not rely on is shown without one, marked as
//! background, so it is context and cannot be cited. When the journal already
//! holds an entry, the same question first asks whether the stretch is a step
//! of its own, naming the latest entry, and offers `journal_skip` for one that
//! is not.

use crate::entry::Entry;
use crate::span::Span;
use crate::style::Style;

/// The turns of a stretch, as the writer reads them.
pub fn stretch(span: &Span, style: &Style) -> String {
    let mut out = format!("{}\n", (style.stretch)(span.from_ms, span.to_ms));
    for t in &span.turns {
        let text = t.text.trim();
        let at = (style.clock)(t.at_ms);
        match t.evidence {
            Some(_) => out.push_str(&format!("\n#{} {at} {text}", t.id)),
            None => out.push_str(&format!("\n   {at} (background) {text}")),
        }
    }
    out
}

/// The writing question: the stretch with its turn numbers, then what is asked.
///
/// With a `latest` entry to weigh the stretch against, the skip is asked
/// first, and against that entry by name: whether the stretch is a step of its
/// own or more of what the latest entry is about is the decision, and a writer
/// shown the step it would be filed under compares against it rather than
/// against the whole journal. The write follows for a stretch that is a step.
///
/// When the call offers `per` — `above`, the writer has journals above it —
/// the style's `per_question` follows the write.
///
/// After a refused attempt `refused` holds that attempt as it was written and why
/// it was refused, because each question is put afresh and the writer would
/// otherwise not know what to change.
pub fn writing(
    span: &Span,
    style: &Style,
    latest: Option<&Entry>,
    above: bool,
    refused: Option<(&str, &str)>,
) -> String {
    let mut out = stretch(span, style);
    if let Some(latest) = latest {
        out.push_str(&format!(
            "\n\n{} {} {} — {}\n\n{}",
            style.latest,
            style.entry,
            latest.id,
            latest.summary.trim(),
            style.skip_question
        ));
    }
    out.push_str("\n\n");
    out.push_str(style.write_question);
    if above {
        out.push(' ');
        out.push_str(style.per_question);
    }
    if let Some((attempt, why)) = refused {
        out.push_str(&format!(
            "\n\nYou tried this:\n{}\n\nIt was refused: {why}\nWrite it again with that put right.",
            attempt.trim()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::entry;
    use crate::entry::Kind;
    use crate::span::testing::turn;
    use crate::style::testing::PLAIN;

    fn span() -> Span {
        Span::of(vec![
            turn(1, 10, Some(Kind::Heard), "User: the build is red"),
            turn(2, 20, Some(Kind::Did), " You: ran cargo test "),
            turn(3, 30, None, "A standing instruction."),
        ])
        .unwrap()
    }

    #[test]
    fn the_stretch_opens_with_when_it_was_and_numbers_what_may_be_cited() {
        assert_eq!(
            stretch(&span(), &PLAIN),
            "From 10 to 30:\n\
             \n#1 t10 User: the build is red\
             \n#2 t20 You: ran cargo test\
             \n   t30 (background) A standing instruction."
        );
    }

    #[test]
    fn the_writing_question_is_the_stretch_then_the_call() {
        let asked = writing(&span(), &PLAIN, None, false, None);
        assert_eq!(
            asked,
            format!(
                "{}\n\nWrite it up with journal_write.",
                stretch(&span(), &PLAIN)
            )
        );
    }

    /// `per` is described only to a writer the call offers it to.
    #[test]
    fn per_is_asked_for_only_when_there_are_journals_above() {
        assert!(writing(&span(), &PLAIN, None, true, None)
            .ends_with("Write it up with journal_write. Name an entry above in per."));
        assert!(!writing(&span(), &PLAIN, None, false, None).contains("per."));
    }

    /// With an entry to weigh the stretch against, the step test comes first
    /// and names the entry a skip would file the stretch under.
    #[test]
    fn the_step_test_comes_first_and_names_the_latest_entry() {
        let latest = Entry {
            summary: "Looked up IN_PROMPT. ".into(),
            ..entry(4)
        };
        let asked = writing(&span(), &PLAIN, Some(&latest), false, None);
        assert_eq!(
            asked,
            format!(
                "{}\n\nLatest: Entry 4 — Looked up IN_PROMPT.\n\nOr skip it with journal_skip.\
                 \n\nWrite it up with journal_write.",
                stretch(&span(), &PLAIN)
            )
        );
    }

    #[test]
    fn a_refused_attempt_comes_back_with_its_reason() {
        let asked = writing(
            &span(),
            &PLAIN,
            None,
            false,
            Some((" {\"name\": \"journal_write\"} ", "Claim 1 cites turn 3.")),
        );
        assert!(
            asked.ends_with(
                "Write it up with journal_write.\n\nYou tried this:\n{\"name\": \"journal_write\"}\
                 \n\nIt was refused: Claim 1 cites turn 3.\nWrite it again with that put right."
            ),
            "{asked}"
        );
    }
}
