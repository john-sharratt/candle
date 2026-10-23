//! One draft of a session's journal: the gate, then the write.
//!
//! The write, the check and the keep are `journal_core`'s
//! ([`journal_core::workflow::run`]). What is the session's is the gate in
//! front of it ([`super::gate`]): a journal that already holds an entry is asked
//! whether the stretch adds anything, and a `no` covers the stretch without a
//! write. Whichever way a draft ends, the journal is told and the draft is
//! recorded, so the API shows every draft's outcome and timings.

use std::future::Future;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use journal_core::journal::Holds;
use journal_core::record::DraftRecord;
use journal_core::span::Span;
use journal_core::verify::World;
use journal_core::workflow::{self, Abandon, Desk, Outcome, Timing};

use crate::journal::gate;
use crate::journal::state::SessionJournal;
use crate::journal::style::CODING;

/// Where the gate's question is put.
pub trait Gatekeeper: Send {
    /// Ask whether the stretch is worth an entry, held to one `journal_gate`
    /// call, and return the decode as it came back.
    fn gate(&mut self, question: &str) -> impl Future<Output = anyhow::Result<String>> + Send;
}

fn hold(state: &Mutex<SessionJournal>) -> MutexGuard<'_, SessionJournal> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn millis(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The gate's answer: `Ok(None)` to go on and write, `Ok(Some(why))` for a no,
/// `Err` when it could not be had.
async fn ask_gate<D: Gatekeeper>(desk: &mut D, span: &Span) -> Result<Option<String>, String> {
    let decoded = desk
        .gate(&gate::question(span, &CODING))
        .await
        .map_err(|e| e.to_string())?;
    let verdict = gate::read(&decoded)?;
    Ok(match verdict.worth {
        true => None,
        false if verdict.reason.is_empty() => {
            Some("the session judged the stretch not worth an entry".to_string())
        }
        false => Some(verdict.reason),
    })
}

/// Run one draft over `span`, which the caller took with
/// [`SessionJournal::begin`], and return its record — which the journal holds
/// too.
pub async fn draft<D: Desk + Gatekeeper>(
    desk: &mut D,
    state: &Mutex<SessionJournal>,
    span: &Span,
    world: &(dyn World + Sync),
) -> DraftRecord {
    let started = Instant::now();
    let asks = hold(state).journal().entries().next().is_some();
    let mut gate_ms = 0;
    if asks {
        let asked = Instant::now();
        let answer = ask_gate(desk, span).await;
        gate_ms = millis(asked);
        let ended = match answer {
            Ok(None) => None,
            Ok(Some(why)) => {
                hold(state).journal_mut().nothing_to_write(span);
                Some(Outcome::Nothing { why })
            }
            Err(e) => {
                hold(state).journal_mut().abandon();
                Some(Outcome::Abandoned(Abandon::Asking(e)))
            }
        };
        if let Some(outcome) = ended {
            return recorded(state, span, started, &outcome, gate_ms, 0);
        }
    }
    let (outcome, timing) = workflow::run(desk, state, span, world, &CODING).await;
    recorded(state, span, started, &outcome, gate_ms, timing.write_ms)
}

fn recorded(
    state: &Mutex<SessionJournal>,
    span: &Span,
    started: Instant,
    outcome: &Outcome,
    gate_ms: u64,
    write_ms: u64,
) -> DraftRecord {
    let record =
        DraftRecord::of(span, started.elapsed(), outcome).timed(Timing { gate_ms, write_ms });
    hold(state).journal_mut().record(record.clone());
    record
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use journal_core::entry::{Entry, Kind};
    use journal_core::record::Ending;
    use journal_core::section::JournalPrompt;
    use journal_core::span::SpanTurn;
    use serde_json::Value;

    use super::*;
    use crate::journal::origin::{reply_part, user_part};

    struct Silent;

    impl World for Silent {
        fn read(&self, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    /// A desk that answers from a script and records what it was asked.
    #[derive(Default)]
    struct Script {
        gates: VecDeque<anyhow::Result<String>>,
        writes: VecDeque<anyhow::Result<String>>,
        gate_asked: Vec<String>,
        write_asked: Vec<String>,
        kept: Vec<JournalPrompt>,
    }

    impl Desk for Script {
        async fn write(&mut self, _: &Value, question: &str) -> anyhow::Result<String> {
            self.write_asked.push(question.to_string());
            self.writes
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("the script ran out")))
        }

        async fn keep(&mut self, prompt: &JournalPrompt) -> anyhow::Result<()> {
            self.kept.push(prompt.clone());
            Ok(())
        }
    }

    impl Gatekeeper for Script {
        async fn gate(&mut self, question: &str) -> anyhow::Result<String> {
            self.gate_asked.push(question.to_string());
            self.gates
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("the script ran out")))
        }
    }

    const YES: &str =
        r#"{"name": "journal_gate", "arguments": {"answer": "yes", "reason": "A fix landed."}}"#;
    const NO: &str =
        r#"{"name": "journal_gate", "arguments": {"answer": "no", "reason": "Already written."}}"#;
    const WRITE: &str = r#"{"name": "journal_write", "arguments": {
        "claims": [{"text": "The user asked for the scan fix.", "cite": "1"},
                   {"text": "scan.rs fails its parity test.", "cite": "3"}],
        "open": [{"text": "Does the fix hold on the 3090?", "relates": "new"}]}}"#;

    /// Turn 0: the user's question, the reply's call; turn 1: the result, the
    /// reply.
    fn stretch() -> Vec<SpanTurn> {
        let part = |id, kind, text: &str| SpanTurn {
            id,
            at_ms: id * 1_000,
            evidence: Some(kind),
            text: text.to_string(),
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
            claims: vec![],
            intend: vec!["Look at the scan".into()],
            opened: vec![],
            resolved: vec![],
        });
        j
    }

    #[tokio::test]
    async fn an_empty_journal_writes_without_asking_the_gate() {
        let (state, span) = begun(SessionJournal::new());
        let mut desk = Script {
            writes: VecDeque::from([Ok(WRITE.to_string())]),
            ..Script::default()
        };
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Wrote);
        assert!(desk.gate_asked.is_empty(), "nothing to weigh against");
        assert_eq!(record.gate_ms, 0);
        let j = state.lock().unwrap();
        let kept = j.journal().entries().next().unwrap();
        assert_eq!(kept.claims[0].kind, Kind::Heard);
        assert_eq!(kept.claims[1].kind, Kind::Observed);
        assert_eq!(j.journal().covered_to(), reply_part(1));
        assert_eq!(j.journal().drafts().count(), 1);
        assert_eq!(desk.kept.len(), 1);
    }

    #[tokio::test]
    async fn a_yes_goes_on_to_the_write() {
        let (state, span) = begun(written());
        let mut desk = Script {
            gates: VecDeque::from([Ok(YES.to_string())]),
            writes: VecDeque::from([Ok(WRITE.to_string())]),
            ..Script::default()
        };
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Wrote);
        assert_eq!(record.entry, Some(2));
        assert_eq!(desk.gate_asked.len(), 1);
        assert_eq!(desk.write_asked.len(), 1);
        assert!(
            desk.write_asked[0].contains("#3 "),
            "{}",
            desk.write_asked[0]
        );
    }

    #[tokio::test]
    async fn a_no_covers_the_stretch_and_keeps_the_reason() {
        let (state, span) = begun(written());
        let mut desk = Script {
            gates: VecDeque::from([Ok(NO.to_string())]),
            ..Script::default()
        };
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Nothing);
        assert_eq!(record.detail.as_deref(), Some("Already written."));
        assert!(desk.write_asked.is_empty());
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), reply_part(1));
        assert!(!j.journal().pending());
    }

    #[tokio::test]
    async fn a_gate_that_cannot_be_read_abandons_the_draft() {
        let (state, span) = begun(written());
        let mut desk = Script {
            gates: VecDeque::from([Ok("I am not sure.".to_string())]),
            ..Script::default()
        };
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Abandoned);
        assert!(
            record
                .detail
                .as_deref()
                .unwrap()
                .contains("no journal_gate call"),
            "{record:?}"
        );
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), 0, "nothing covered");
        assert!(!j.journal().pending());
    }

    #[tokio::test]
    async fn a_gate_that_cannot_be_asked_abandons_the_draft() {
        let (state, span) = begun(written());
        let mut desk = Script::default();
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Abandoned);
        assert!(!state.lock().unwrap().journal().pending());
    }

    #[tokio::test]
    async fn a_write_refused_four_times_is_abandoned_and_recorded() {
        let (state, span) = begun(SessionJournal::new());
        let bad =
            r#"{"name": "journal_write", "arguments": {"claims": [{"text": "x", "cite": "9"}]}}"#;
        let mut desk = Script {
            writes: (0..4).map(|_| Ok(bad.to_string())).collect(),
            ..Script::default()
        };
        let record = draft(&mut desk, &state, &span, &Silent).await;
        assert_eq!(record.result, Ending::Abandoned);
        assert_eq!(desk.write_asked.len(), 4);
        let j = state.lock().unwrap();
        assert_eq!(j.journal().covered_to(), 0);
        assert_eq!(j.journal().drafts().last().unwrap(), &record);
    }
}
