//! What the writer is asked to write about a stretch of its own history.
//!
//! The question carries the stretch as text, because a claim cites a turn by
//! number and the live history is not numbered. Each turn the writer may rely on
//! carries its number; a turn it may not rely on is shown without one, marked as
//! background, so it is context and cannot be cited. When the journal already
//! holds an entry, the same question offers `journal_skip` for a stretch that
//! adds nothing to it.

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

/// The writing question: the stretch with its turn numbers, then what is asked
/// — with the skip offered when `may_skip`.
///
/// After a refused attempt `refused` holds that attempt as it was written and why
/// it was refused, because each question is put afresh and the writer would
/// otherwise not know what to change.
pub fn writing(
    span: &Span,
    style: &Style,
    may_skip: bool,
    refused: Option<(&str, &str)>,
) -> String {
    let mut out = format!("{}\n\n{}", stretch(span, style), style.write_question);
    if may_skip {
        out.push(' ');
        out.push_str(style.skip_question);
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
        let asked = writing(&span(), &PLAIN, false, None);
        assert_eq!(
            asked,
            format!(
                "{}\n\nWrite it up with journal_write.",
                stretch(&span(), &PLAIN)
            )
        );
    }

    #[test]
    fn the_skip_is_offered_after_the_write_when_it_may_be_taken() {
        let asked = writing(&span(), &PLAIN, true, None);
        assert_eq!(
            asked,
            format!(
                "{}\n\nWrite it up with journal_write. Or skip it with journal_skip.",
                stretch(&span(), &PLAIN)
            )
        );
    }

    #[test]
    fn a_refused_attempt_comes_back_with_its_reason() {
        let asked = writing(
            &span(),
            &PLAIN,
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
