//! Whether what a character has just lived needs a new journal entry.
//!
//! The question is put to the character's live conversation, which already holds
//! the stretch as it lived it, so it names the stretch by when it was and says
//! nothing more. The journal the character reads is in its system prompt, so
//! "what your journal already says" is something it can answer from. A yes sends
//! the stretch to be written up; a no closes it, so the next question is about
//! what has happened since. A journal with no entry has no section in the
//! character's prompt, so nothing is asked: the stretch is written up.

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{NpcView, Question, Verdict};
use crate::clock::stamp;
use crate::engine::journal::state::Waiting;

pub const YES: &str = "yes";
pub const NO: &str = "no";

pub struct Journal;

/// The question for a stretch of `waiting.turns` turns.
pub fn gate_question(waiting: &Waiting) -> String {
    format!(
        "Your last {} turns, from {} to {}, are not in your journal yet. Given what happened in \
         them and what your journal already says, does your journal need a new entry? Say why in \
         a sentence.",
        waiting.turns,
        stamp(waiting.from_ms),
        stamp(waiting.to_ms),
    )
}

impl Module for Journal {
    fn name(&self) -> &'static str {
        "journal"
    }

    fn question(&self, view: &NpcView) -> Option<Question> {
        let waiting = view.journal.as_ref().filter(|w| !w.empty)?;
        Some(Question {
            text: gate_question(waiting),
            choices: vec![YES.to_string(), NO.to_string()],
        })
    }

    fn judge(&self, view: &NpcView, answer: Option<&str>) -> Verdict {
        if view.journal.as_ref().is_some_and(|w| w.empty) {
            return Verdict::Journal { worth: true };
        }
        match answer.map(str::trim) {
            Some(a) if a.eq_ignore_ascii_case(YES) => Verdict::Journal { worth: true },
            Some(a) if a.eq_ignore_ascii_case(NO) => Verdict::Journal { worth: false },
            _ => Verdict::Healthy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::view;
    use crate::clock::DAY_MS;

    const MINUTE_MS: u64 = 60_000;

    fn waiting() -> Waiting {
        let from_ms = 2 * DAY_MS + 14 * 60 * MINUTE_MS + 20 * MINUTE_MS;
        Waiting {
            turns: 4,
            from_ms,
            to_ms: from_ms + 3 * MINUTE_MS,
            empty: false,
        }
    }

    fn with_stretch() -> NpcView {
        let mut v = view(None);
        v.journal = Some(waiting());
        v
    }

    fn with_stretch_and_no_entries() -> NpcView {
        let mut v = view(None);
        v.journal = Some(Waiting {
            empty: true,
            ..waiting()
        });
        v
    }

    #[test]
    fn an_empty_journal_is_not_asked_about_and_is_written_up() {
        let v = with_stretch_and_no_entries();
        assert_eq!(Journal.question(&v), None);
        assert_eq!(Journal.judge(&v, None), Verdict::Journal { worth: true });
    }

    #[test]
    fn it_asks_nothing_while_the_journal_covers_everything() {
        assert_eq!(Journal.question(&view(None)), None);
    }

    #[test]
    fn the_question_names_the_stretch_and_asks_for_a_reason() {
        let q = Journal.question(&with_stretch()).unwrap();
        assert!(
            q.text
                .starts_with("Your last 4 turns, from 3 Jan 1970, 14:20 to 3 Jan 1970, 14:23,"),
            "{}",
            q.text
        );
        assert!(
            q.text.contains("what your journal already says"),
            "{}",
            q.text
        );
        assert!(q.text.ends_with("Say why in a sentence."), "{}", q.text);
        assert_eq!(q.choices, vec![YES, NO]);
    }

    #[test]
    fn the_question_does_not_repeat_the_turns() {
        let q = gate_question(&waiting());
        assert!(!q.contains("journal_write"), "{q}");
    }

    #[test]
    fn a_yes_or_a_no_is_a_verdict_either_way() {
        let v = with_stretch();
        assert_eq!(
            Journal.judge(&v, Some("Yes")),
            Verdict::Journal { worth: true }
        );
        assert_eq!(
            Journal.judge(&v, Some(" no ")),
            Verdict::Journal { worth: false }
        );
    }

    #[test]
    fn no_answer_or_an_odd_one_is_not_a_verdict() {
        let v = with_stretch();
        assert_eq!(Journal.judge(&v, None), Verdict::Healthy);
        assert_eq!(Journal.judge(&v, Some("maybe")), Verdict::Healthy);
    }
}
