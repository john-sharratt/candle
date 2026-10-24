//! One fold, start to finish: which entries are due, the question, the keep.
//!
//! [`run`] is the fold's counterpart of [`crate::workflow::run`], written
//! against the same [`Desk`]. It folds at most one period: the caller runs it
//! again while it keeps folding, so a journal that slept through a month
//! catches up a period at a time, lowest level first. It shares the draft's
//! single flight ([`Journal::begin_fold`](crate::journal::Journal::begin_fold)),
//! and like a draft it takes the state's lock only to read it and to record
//! how it ended, never across a question.
//!
//! The keep is lossy on purpose. The entries folded leave the journal and their
//! text goes with them; the folded entry keeps its line in the journal and its
//! digest behind it, filed as the text its tools read.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::entry::Entry;
use crate::fold::{self, Fold};
use crate::journal::Holds;
use crate::style::Style;
use crate::transcript::{Filed, Part};
use crate::workflow::{Abandon, Desk, MAX_ROUNDS};

/// How a fold ended.
#[derive(Debug, PartialEq)]
pub enum Folded {
    /// No period is over, or a draft or fold is already running.
    Nothing,
    /// The entries `fold` named were folded into `entry`.
    Kept { fold: Fold, entry: Entry },
    /// The fold did not finish; the same entries are due again next time.
    Abandoned(Abandon),
}

fn hold<S: Holds>(state: &Mutex<S>) -> MutexGuard<'_, S> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The digest of `entry`, filed as the one text the tools read behind it.
fn digest_filed(entry: &Entry) -> Filed {
    Filed {
        entry: entry.id,
        parts: vec![Part {
            id: entry.to_turn,
            at_ms: entry.to_ms,
            text: entry.digest.clone(),
        }],
    }
}

/// Fold the earliest period that is over at `now_ms`, if there is one and no
/// draft or fold is running.
pub async fn run<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    now_ms: u64,
    style: &Style,
) -> Folded {
    let (fold, entries) = {
        let mut held = hold(state);
        let entries: Vec<Entry> = held.journal().entries().cloned().collect();
        let Some(fold) = fold::next(&entries, now_ms, style) else {
            return Folded::Nothing;
        };
        if !held.journal_mut().begin_fold() {
            return Folded::Nothing;
        }
        (fold, entries)
    };
    let outcome = fold_up(desk, state, &fold, &entries, style).await;
    if matches!(outcome, Folded::Abandoned(_)) {
        hold(state).journal_mut().abandon();
    }
    outcome
}

async fn fold_up<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    fold: &Fold,
    entries: &[Entry],
    style: &Style,
) -> Folded {
    let specs = [fold::spec()];
    let asked = fold::question(fold, entries, style);
    let mut refused: Option<String> = None;
    for _ in 0..MAX_ROUNDS {
        let question = match &refused {
            None => asked.clone(),
            Some(back) => format!("{asked}\n\n{back}"),
        };
        let answer = match desk.write(&specs, &question).await {
            Ok(answer) => answer,
            Err(e) => return Folded::Abandoned(Abandon::Asking(e.to_string())),
        };
        match fold::parse(&answer) {
            Ok((summary, digest)) => {
                let made = fold::entry(fold, entries, summary, digest);
                return keep(desk, state, fold, made, style).await;
            }
            Err(reason) => {
                refused = Some(format!(
                    "You tried this:\n{answer}\n\nIt was refused: {reason}"
                ));
            }
        }
    }
    Folded::Abandoned(Abandon::Refused(refused.unwrap_or_default()))
}

/// Keep the folded entry: ids first, then the substrate — its digest filed,
/// the folded entries' text ceased, the record — then memory.
async fn keep<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    fold: &Fold,
    made: Entry,
    style: &Style,
) -> Folded {
    let (staged, prompt) = {
        let mut held = hold(state);
        let staged = held.journal_mut().reserve(made);
        let prompt = held.journal().prompt_after_fold(fold, &staged, style);
        (staged, prompt)
    };
    let filed = digest_filed(&staged);
    if let Err(e) = desk.keep(&prompt, Some(&filed), &fold.ids).await {
        return Folded::Abandoned(Abandon::Keeping(e.to_string()));
    }
    hold(state).journal_mut().kept_fold(fold, staged.clone());
    Folded::Kept {
        fold: fold.clone(),
        entry: staged,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use serde_json::Value;

    use super::*;
    use crate::entry::testing::entry_to;
    use crate::journal::Journal;
    use crate::period::Period;
    use crate::section::JournalPrompt;
    use crate::span::testing::turn;
    use crate::style::testing::PLAIN;

    const FOLDED: &str = r#"{"name": "journal_fold", "arguments": {
        "summary": "Fixed the scan race.", "digest": "Read scan.rs, found the race, fixed it."}}"#;

    #[derive(Default)]
    struct Script {
        answers: VecDeque<String>,
        asked: Vec<String>,
        kept: Vec<(JournalPrompt, Option<Filed>, Vec<u64>)>,
        refuse_keeping: bool,
    }

    impl Desk for Script {
        async fn write(&mut self, specs: &[Value], question: &str) -> anyhow::Result<String> {
            assert_eq!(specs, [fold::spec()]);
            self.asked.push(question.to_string());
            self.answers
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("the script ran out"))
        }

        async fn keep(
            &mut self,
            prompt: &JournalPrompt,
            filed: Option<&Filed>,
            ceased: &[u64],
        ) -> anyhow::Result<()> {
            if self.refuse_keeping {
                anyhow::bail!("the substrate said no");
            }
            self.kept
                .push((prompt.clone(), filed.cloned(), ceased.to_vec()));
            Ok(())
        }
    }

    fn answering(answers: &[&str]) -> Script {
        Script {
            answers: answers.iter().map(|a| a.to_string()).collect(),
            ..Script::default()
        }
    }

    /// Steps 1 and 2 of day 1, step 3 of day 2 (PLAIN's day is 1000 ms).
    fn two_days() -> Mutex<Journal> {
        let mut j = Journal::new();
        for (to_turn, from_ms, to_ms) in [(3, 1_100, 1_200), (6, 1_500, 1_900), (9, 2_100, 2_200)] {
            j.kept(Entry {
                from_ms,
                to_ms,
                ..entry_to(to_turn)
            });
        }
        Mutex::new(j)
    }

    #[tokio::test]
    async fn a_finished_day_folds_into_one_entry_with_its_digest_filed_and_its_steps_text_ceased() {
        let state = two_days();
        let mut desk = answering(&[FOLDED]);
        let outcome = run(&mut desk, &state, 2_500, &PLAIN).await;
        let Folded::Kept { fold, entry } = outcome else {
            panic!("expected a fold, got {outcome:?}");
        };
        assert_eq!(fold.ids, [1, 2]);
        assert_eq!((entry.id, entry.period), (4, Period::Day));
        assert_eq!(entry.summary, "Fixed the scan race.");
        let (prompt, filed, ceased) = &desk.kept[0];
        assert_eq!(ceased, &[1, 2]);
        assert_eq!(filed.as_ref(), Some(&digest_filed(&entry)));
        assert_eq!(
            filed.as_ref().unwrap().parts[0].text,
            "Read scan.rs, found the race, fixed it."
        );
        let j = hold(&state);
        assert_eq!(&j.prompt(&PLAIN).sections, &prompt.sections);
        let held: Vec<u64> = j.entries().map(|e| e.id).collect();
        assert_eq!(held, [4, 3]);
        assert!(!j.pending());
        assert!(
            desk.asked[0].starts_with("Day 1 is over."),
            "{}",
            desk.asked[0]
        );
        assert!(!desk.asked[0].contains("Entry 3 "), "day 2 is not folded");
    }

    #[tokio::test]
    async fn nothing_is_asked_while_no_period_is_over() {
        let state = two_days();
        let mut desk = answering(&[]);
        assert_eq!(run(&mut desk, &state, 1_999, &PLAIN).await, Folded::Nothing);
        assert!(desk.asked.is_empty());
        assert!(!hold(&state).pending());
    }

    #[tokio::test]
    async fn a_fold_waits_for_a_draft_in_flight() {
        let state = two_days();
        hold(&state).begin(vec![turn(10, 2_300, None, "x")], 10);
        let mut desk = answering(&[FOLDED]);
        assert_eq!(run(&mut desk, &state, 2_500, &PLAIN).await, Folded::Nothing);
        assert!(desk.asked.is_empty());
    }

    #[tokio::test]
    async fn a_refused_answer_is_sent_back_and_may_be_corrected() {
        let state = two_days();
        let bare = r#"{"name": "journal_fold", "arguments": {"summary": "A day."}}"#;
        let mut desk = answering(&[bare, FOLDED]);
        let outcome = run(&mut desk, &state, 2_500, &PLAIN).await;
        assert!(matches!(outcome, Folded::Kept { .. }), "{outcome:?}");
        assert!(desk.asked[1].ends_with("It was refused: journal_fold needs a `digest`."));
    }

    #[tokio::test]
    async fn a_fold_that_cannot_be_kept_leaves_the_journal_as_it_was() {
        let state = two_days();
        let mut desk = answering(&[FOLDED]);
        desk.refuse_keeping = true;
        let outcome = run(&mut desk, &state, 2_500, &PLAIN).await;
        assert!(
            matches!(outcome, Folded::Abandoned(Abandon::Keeping(_))),
            "{outcome:?}"
        );
        let j = hold(&state);
        assert_eq!(j.entries().count(), 3);
        assert!(!j.pending());
    }

    #[tokio::test]
    async fn answers_never_accepted_give_the_fold_up() {
        let state = two_days();
        let mut desk = answering(&["no", "no", "no", "no"]);
        let outcome = run(&mut desk, &state, 2_500, &PLAIN).await;
        assert!(
            matches!(outcome, Folded::Abandoned(Abandon::Refused(_))),
            "{outcome:?}"
        );
        assert_eq!(desk.asked.len(), MAX_ROUNDS);
        assert!(!hold(&state).pending());
    }
}
