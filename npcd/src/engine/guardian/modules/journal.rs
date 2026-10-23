//! Starting a journal draft when a stretch of a character's life is due one.
//!
//! Nothing is asked here. Whether the stretch is worth an entry is decided in
//! the draft itself, by the same answer that would write it: once the journal
//! holds an entry, the draft offers `journal_skip` beside `journal_write`, so
//! the character weighs the stretch with its turns numbered in front of it
//! rather than from a separate question first.

use crate::engine::guardian::module::Module;
use crate::engine::guardian::view::{NpcView, Question, Verdict};

pub struct Journal;

impl Module for Journal {
    fn name(&self) -> &'static str {
        "journal"
    }

    fn question(&self, _: &NpcView) -> Option<Question> {
        None
    }

    fn judge(&self, view: &NpcView, _: Option<&str>) -> Verdict {
        match view.journal {
            Some(_) => Verdict::Journal,
            None => Verdict::Healthy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::guardian::modules::fixtures::view;
    use crate::engine::journal::state::Waiting;

    fn with_stretch() -> NpcView {
        let mut v = view(None);
        v.journal = Some(Waiting {
            turns: 4,
            from_ms: 0,
            to_ms: 180_000,
        });
        v
    }

    #[test]
    fn a_stretch_due_a_draft_starts_one_without_a_question() {
        let v = with_stretch();
        assert_eq!(Journal.question(&v), None);
        assert_eq!(Journal.judge(&v, None), Verdict::Journal);
    }

    #[test]
    fn nothing_starts_while_the_journal_covers_everything() {
        let v = view(None);
        assert_eq!(Journal.question(&v), None);
        assert_eq!(Journal.judge(&v, None), Verdict::Healthy);
    }
}
