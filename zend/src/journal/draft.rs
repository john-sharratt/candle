//! One draft of a session's journal.
//!
//! The write, the skip, the check and the keep are `journal_core`'s
//! ([`journal_core::workflow::run`]): one answer either writes the entry or —
//! once the journal holds an entry to weigh the stretch against — skips it with
//! a reason. Whichever way a draft ends, the journal is told and the draft is
//! recorded, so the API shows every draft's outcome and timings.

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use journal_core::journal::Holds;
use journal_core::record::DraftRecord;
use journal_core::span::Span;
use journal_core::verify::World;
use journal_core::workflow::{self, Desk};

use crate::journal::state::SessionJournal;
use crate::journal::style::CODING;

fn hold(state: &Mutex<SessionJournal>) -> MutexGuard<'_, SessionJournal> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Run one draft over `span`, which the caller took with
/// [`SessionJournal::begin`], and return its record — which the journal holds
/// too.
pub async fn draft<D: Desk>(
    desk: &mut D,
    state: &Mutex<SessionJournal>,
    span: &Span,
    world: &(dyn World + Sync),
) -> DraftRecord {
    let started = Instant::now();
    let (outcome, timing) = workflow::run(desk, state, span, world, &CODING).await;
    let record = DraftRecord::of(span, started.elapsed(), &outcome).timed(timing);
    hold(state).journal_mut().record(record.clone());
    record
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use journal_core::entry::{Entry, Kind};
    use journal_core::record::Ending;
    use journal_core::section::{restore, JournalPrompt};
    use journal_core::span::SpanTurn;
    use journal_core::tools::{SKIP, WRITE};
    use journal_core::transcript::Filed;
    use serde_json::Value;

    use super::*;
    use crate::journal::origin::{reply_part, user_part};
    use crate::journal::style::{SKIP_QUESTION, WRITE_QUESTION};

    struct Silent;

    impl World for Silent {
        fn read(&self, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    /// A desk that answers from a script and records what it was asked.
    #[derive(Default)]
    struct Script {
        writes: VecDeque<anyhow::Result<String>>,
        asked: Vec<String>,
        offered: Vec<Vec<String>>,
        kept: Vec<JournalPrompt>,
    }

    impl Desk for Script {
        async fn write(&mut self, specs: &[Value], question: &str) -> anyhow::Result<String> {
            self.asked.push(question.to_string());
            self.offered.push(
                specs
                    .iter()
                    .map(|s| s["name"].as_str().unwrap_or_default().to_string())
                    .collect(),
            );
            self.writes
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("the script ran out")))
        }

        async fn keep(
            &mut self,
            prompt: &JournalPrompt,
            _filed: Option<&Filed>,
        ) -> anyhow::Result<()> {
            self.kept.push(prompt.clone());
            Ok(())
        }
    }

    fn answering(writes: &[&str]) -> Script {
        Script {
            writes: writes.iter().map(|w| Ok(w.to_string())).collect(),
            ..Script::default()
        }
    }

    const SKIPPED: &str =
        r#"{"name": "journal_skip", "arguments": {"reason": "Already written."}}"#;
    const WRITTEN: &str = r#"{"name": "journal_write", "arguments": {
        "claims": [{"text": "The user asked for the scan fix.", "cite": "1"},
                   {"text": "scan.rs fails its parity test.", "cite": "3"}],
        "summary": "Took on the scan fix; its parity test fails.",
        "open": [{"text": "Does the fix hold on the 3090?", "relates": "new"}]}}"#;

    /// Turn 0: the user's question, the reply's call; turn 1: the result, the
    /// reply.
    fn stretch() -> Vec<SpanTurn> {
        let part = |id, kind, text: &str| SpanTurn {
            id,
            at_ms: id * 1_000,
            evidence: Some(kind),
            text: text.to_string(),
            full: text.to_string(),
        };
        vec![
            part(user_part(0), Kind::Heard, "User: fix the scan"),
            part(
                reply_part(0),
                Kind::Did,
                "You: <tool_call>cargo test</tool_call>",
            ),
            part(user_part(1), Kind::Observed, "Tool results: parity FAILED"),
            part(reply_part(1), Kind::Did, "You: The parity test fails."),
        ]
    }

    fn begun(journal: SessionJournal) -> (Mutex<SessionJournal>, Span) {
        let mut journal = journal;
        let span = journal.begin(stretch(), 1).unwrap();
        (Mutex::new(journal), span)
    }

    /// A journal that already holds one entry, covering nothing of
    /// [`stretch`].
    fn written() -> SessionJournal {
        let mut j = SessionJournal::new();
        j.journal_mut().kept(Entry {
            id: 0,
            from_turn: 0,
            to_turn: 0,
            from_ms: 0,
            to_ms: 0,
            summary: "Started on the scan.".into(),
            claims: vec![],
            intend: vec!["Look at the scan".into()],
            opened: vec![],
            resolved: vec![],
        });
        j
    }

    #[tokio::test]
    async fn an_empty_journal_is_asked_for_the_entry_alone() {
        let (state, span) = begun(SessionJournal::new());
        let mut desk = answering(&[WRITTEN]);
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Wrote);
        assert_eq!(desk.offered, vec![vec![WRITE.to_string()]]);
        assert!(desk.asked[0].ends_with(WRITE_QUESTION), "{}", desk.asked[0]);
        let j = state.lock().unwrap();
        let kept = j.journal().entries().next().unwrap();
        assert_eq!(kept.claims[0].kind, Kind::Heard);
        assert_eq!(kept.claims[1].kind, Kind::Observed);
        assert_eq!(j.journal().covered_to(), reply_part(1));
        assert_eq!(j.journal().drafts().count(), 1);
        assert_eq!(desk.kept.len(), 1);
    }

    #[tokio::test]
    async fn a_journal_with_an_entry_is_offered_the_skip_in_the_same_answer() {
        let (state, span) = begun(written());
        let mut desk = answering(&[WRITTEN]);
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Wrote);
        assert_eq!(record.entry, Some(2));
        assert_eq!(desk.asked.len(), 1, "one answer, no separate gate");
        assert_eq!(desk.offered[0], vec![WRITE.to_string(), SKIP.to_string()]);
        assert!(
            desk.asked[0].ends_with(&format!(
                "Your latest entry: Entry 1 — Started on the scan.\n\n{SKIP_QUESTION}\n\n\
                 {WRITE_QUESTION}"
            )),
            "{}",
            desk.asked[0]
        );
        assert!(desk.asked[0].contains("#3 "), "{}", desk.asked[0]);
    }

    #[tokio::test]
    async fn a_skip_covers_the_stretch_and_keeps_the_reason() {
        let (state, span) = begun(written());
        let mut desk = answering(&[SKIPPED]);
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Nothing);
        assert_eq!(record.detail.as_deref(), Some("Already written."));
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), reply_part(1));
        assert!(!j.journal().pending());
        assert_eq!(j.journal().drafts().last().unwrap(), &record);
        let recorded = restore(desk.kept[0].record()).unwrap();
        assert_eq!(
            SessionJournal::restore(recorded).first_uncovered_turn(),
            2,
            "a restart does not draft the skipped stretch again"
        );
    }

    #[tokio::test]
    async fn a_draft_that_cannot_be_asked_is_abandoned_and_covers_nothing() {
        let (state, span) = begun(written());
        let mut desk = Script::default();
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Abandoned);
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), 0, "nothing covered");
        assert!(!j.journal().pending());
    }

    #[tokio::test]
    async fn a_write_refused_four_times_is_abandoned_and_recorded() {
        let (state, span) = begun(SessionJournal::new());
        let bad =
            r#"{"name": "journal_write", "arguments": {"claims": [{"text": "x", "cite": "9"}]}}"#;
        let mut desk = answering(&[bad; 4]);
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Abandoned);
        assert_eq!(desk.asked.len(), 4);
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), 0);
        assert_eq!(j.journal().drafts().last().unwrap(), &record);
    }
}
