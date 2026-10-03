//! What the character is asked to write about a stretch of its own life.
//!
//! The question is put to the character's live conversation and carries the
//! stretch as text, because a claim cites a turn by number and the live history is
//! not numbered. Each turn the character may rely on carries its number; a turn it
//! may not rely on (scenery, standing instructions, its own reflection) is shown
//! without one, so it is context and cannot be cited. Whether a stretch is worth
//! an entry is the guardian's question, not asked here.

use crate::clock::{stamp, WorldTime};
use crate::engine::journal::state::Span;
use crate::engine::window::Speaker;

/// What the writing is asked of the character once the stretch is on the table.
///
/// The call is not shown anywhere else, so the question says what each part of it
/// is.
pub const WRITE_QUESTION: &str = "Write it up with journal_write. `claims` is what happened, \
one thing each: its `text` in one short sentence, and `cite`, the number of the turn it rests \
on (`also_cite` for a second). Something you worked out and did not see or hear still cites \
the turn that led you to it. If a claim is a fact that can be looked up — what a gauge read, \
whose turn it is, where something is — add `typed`, with its `subject`, `attribute` and \
`value`. `intend` is what you mean to do next, as short lines, at most three. `open` is what \
is still unsettled: its `text`, and `relates` — `new` for something you were not carrying, \
`restates N` to say an open item again, `resolves N` when it has been settled. Keep the whole \
entry brief, and put nothing in that did not happen: a conclusion you only worked out is \
written as one, and something that may have changed since you looked is written as it was \
when you looked.";

/// The clock part of a stamp: `14:20` out of `14 Jun 2187, 14:20`.
fn clock(ms: u64) -> String {
    WorldTime::of(ms).clock()
}

/// The turns of a stretch, as the character reads them.
pub fn stretch(span: &Span) -> String {
    let mut out = format!(
        "What happened to you, from {} to {}:\n",
        stamp(span.from_ms),
        stamp(span.to_ms)
    );
    for t in &span.turns {
        let who = match t.speaker {
            Speaker::Npc => "You: ",
            Speaker::World => "",
        };
        let text = t.text.trim();
        if t.origin.citable() {
            out.push_str(&format!("\n#{} {} {who}{text}", t.id, clock(t.at_ms)));
        } else {
            out.push_str(&format!("\n   {} (background) {who}{text}", clock(t.at_ms)));
        }
    }
    out
}

/// The writing question: the stretch with its turn numbers, then what is asked.
///
/// After a refused attempt `refused` holds that attempt as it was written and why
/// it was refused, because each question is put afresh and the character would
/// otherwise not know what to change.
pub fn writing(span: &Span, refused: Option<(&str, &str)>) -> String {
    let mut out = format!("{}\n\n{WRITE_QUESTION}", stretch(span));
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
    use crate::engine::event::{Event, EventKind, Salience};
    use crate::engine::journal::state::JournalState;
    use crate::clock::DAY_MS;
    use crate::engine::window::Window;

    const MINUTE_MS: u64 = 60_000;

    /// Turns: 1 speech, 2 own `tell`, 3 ambient scenery, 4 own `reflect`.
    fn span() -> Span {
        let mut w = Window::with_default_cap();
        let at = 2 * DAY_MS + 14 * 60 * MINUTE_MS + 20 * MINUTE_MS;
        w.push_event(&Event::new(
            0,
            at,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Pax".into(),
                text: "Fabricator 5 is lost.".into(),
                to: Default::default(),
            },
        ));
        w.push_npc("tell — to Pax: on my way", at + MINUTE_MS);
        w.push_event(&Event::new(
            0,
            at + 2 * MINUTE_MS,
            Salience::IDLE,
            EventKind::Description {
                text: "A light shifts.".into(),
            },
        ));
        w.push_npc("reflect — I feel uneasy", at + 3 * MINUTE_MS);
        JournalState::new(0).begin(&w).unwrap()
    }

    #[test]
    fn a_turn_it_may_rely_on_is_numbered() {
        let text = stretch(&span());
        assert!(text.contains("#1 14:20 "), "{text}");
        assert!(
            text.contains("#2 14:21 You: tell — to Pax: on my way"),
            "{text}"
        );
    }

    #[test]
    fn scenery_and_reflection_are_shown_without_a_number() {
        let text = stretch(&span());
        assert!(text.contains("(background)"), "{text}");
        assert!(!text.contains("#3"), "{text}");
        assert!(!text.contains("#4"), "{text}");
        assert!(text.contains("A light shifts."), "{text}");
    }

    #[test]
    fn the_stretch_says_when_it_was() {
        let text = stretch(&span());
        assert!(
            text.starts_with("What happened to you, from 3 Jan 1970, 14:20 to 3 Jan 1970, 14:23:"),
            "{text}"
        );
    }

    #[test]
    fn the_writing_question_carries_the_numbered_stretch_then_the_call() {
        let asked = writing(&span(), None);
        assert!(asked.contains("#1 14:20 "), "{asked}");
        assert!(asked.ends_with(WRITE_QUESTION), "{asked}");
    }

    #[test]
    fn a_refused_attempt_comes_back_with_its_reason() {
        let asked = writing(
            &span(),
            Some(("{\"name\": \"journal_write\"}", "Claim 1 cites turn 3.")),
        );
        assert!(
            asked.contains("You tried this:\n{\"name\": \"journal_write\"}"),
            "{asked}"
        );
        assert!(
            asked.contains("It was refused: Claim 1 cites turn 3."),
            "{asked}"
        );
        assert!(asked.ends_with("put right."), "{asked}");
    }
}
