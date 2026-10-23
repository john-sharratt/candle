//! One draft, start to finish: the write, the check, the keep.
//!
//! [`run`] is everything that happens between "this stretch is worth an entry"
//! and "the journal has, or has not, a new entry", and it is deterministic: the
//! model is asked for the entry, every answer is read by [`tools::parse`] and
//! checked by [`verify`], and the same answers always draw the same outcome. It
//! is written against [`Desk`] — where the question is put and an entry is kept
//! — so the whole workflow runs in tests against a script, and each product's
//! engine-backed desk is only the plumbing underneath. Whether a stretch is worth
//! an entry is not decided here: the caller asks, and starts a draft on a yes.
//!
//! # The shape of a draft
//!
//! 1. A stretch with nothing a claim could rest on is not written about at all.
//! 2. The write, at most [`MAX_ROUNDS`] answers. The grammar holds each answer to
//!    one `journal_write` call, so the only ways a round fails are a call the
//!    check refuses and a decode that does not parse. Either is sent back with
//!    the refused answer and the reason, and a draft that passes is kept. Keeping
//!    ends the draft; the model is never asked whether it is finished.
//!
//! # What the state sees
//!
//! A draft takes the state's lock only to read it and to record how it ended,
//! never across a question to the model, so the writer's main line is never
//! waiting on a draft.

use std::future::Future;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use serde_json::Value;

use crate::ask;
use crate::entry::{Entry, Item};
use crate::journal::Holds;
use crate::repeats;
use crate::section::JournalPrompt;
use crate::span::Span;
use crate::style::Style;
use crate::tools;
use crate::verify::{verify, Note, Verified, World};

/// Answers the write may take before the draft is given up.
pub const MAX_ROUNDS: usize = 4;

/// Where a draft's question is put and its entry is kept.
pub trait Desk: Send {
    /// Ask for the entry, held to one call of `spec` (the flat `{name, params}`
    /// tool form), and return the decode as it came back.
    fn write(
        &mut self,
        spec: &Value,
        question: &str,
    ) -> impl Future<Output = anyhow::Result<String>> + Send;

    /// Keep the journal as it will read once the new entry is in it.
    fn keep(&mut self, prompt: &JournalPrompt) -> impl Future<Output = anyhow::Result<()>> + Send;
}

/// Why a draft ended without an entry or a verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Abandon {
    /// The model could not be asked.
    Asking(String),
    /// What came back was refused until the draft ran out of answers.
    Refused(String),
    /// The entry passed and could not be kept.
    Keeping(String),
}

/// How a draft ended.
#[derive(Debug)]
pub enum Outcome {
    /// An entry was kept, after this many answers to the write.
    Wrote {
        entry: Entry,
        notes: Vec<Note>,
        rounds: usize,
    },
    /// There was nothing worth writing, and this is why. The workflow itself
    /// reaches this for a stretch nothing can be cited from and for an entry
    /// that only repeats the journal; a gate's "no" is recorded by its caller.
    Nothing { why: String },
    /// The draft did not finish; the same turns are covered again next time.
    Abandoned(Abandon),
}

/// Where a draft's time went.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    /// Asking whether the stretch was worth an entry. The caller asks, so the
    /// workflow leaves this at zero and the caller fills it in.
    pub gate_ms: u64,
    /// Every answer to the write, refused ones included.
    pub write_ms: u64,
}

fn hold<S: Holds>(state: &Mutex<S>) -> MutexGuard<'_, S> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn millis(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Run one draft over `span`, which the caller took with
/// [`crate::journal::Journal::begin`]. Whichever way it ends, the state has been
/// told.
pub async fn run<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    span: &Span,
    world: &(dyn World + Sync),
    style: &Style,
) -> (Outcome, Timing) {
    let mut timing = Timing::default();
    let outcome = draft(desk, state, span, world, style, &mut timing).await;
    if matches!(outcome, Outcome::Abandoned(_)) {
        hold(state).journal_mut().abandon();
    }
    (outcome, timing)
}

async fn draft<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    span: &Span,
    world: &(dyn World + Sync),
    style: &Style,
    timing: &mut Timing,
) -> Outcome {
    let open = hold(state).journal().open().to_vec();
    let Some(spec) = tools::write_spec(&span.citable_ids(), &open) else {
        hold(state).journal_mut().nothing_to_write(span);
        return Outcome::Nothing {
            why: "nothing in the stretch was something a claim could rest on".to_string(),
        };
    };

    let started = Instant::now();
    let outcome = write(desk, state, span, world, style, &spec, &open).await;
    timing.write_ms = millis(started);
    outcome
}

/// Ask for the entry until one passes the check or the rounds run out.
async fn write<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    span: &Span,
    world: &(dyn World + Sync),
    style: &Style,
    spec: &Value,
    open: &[Item],
) -> Outcome {
    let mut refused: Option<(String, String)> = None;
    for round in 1..=MAX_ROUNDS {
        let question = ask::writing(
            span,
            style,
            refused
                .as_ref()
                .map(|(answer, reason)| (answer.as_str(), reason.as_str())),
        );
        let answer = match desk.write(spec, &question).await {
            Ok(answer) => answer,
            Err(e) => return Outcome::Abandoned(Abandon::Asking(e.to_string())),
        };
        let reason = match tools::parse(&answer) {
            Err(reason) => reason,
            Ok(written) => match verify(written, span, open, world) {
                Ok(verified) => {
                    // What the journal already says is not kept again; an
                    // entry of nothing else covers its stretch as a gate's
                    // "no" would.
                    let held: Vec<Entry> = hold(state).journal().entries().cloned().collect();
                    let (entry, dropped) = repeats::drop_repeats(verified.entry, &held, open);
                    if !repeats::says_anything(&entry) {
                        hold(state).journal_mut().nothing_to_write(span);
                        return Outcome::Nothing {
                            why: format!(
                                "everything it would have said ({dropped} line(s)) is already in \
                                 the journal"
                            ),
                        };
                    }
                    let verified = Verified {
                        entry,
                        notes: verified.notes,
                    };
                    return keep(desk, state, verified, style, round).await;
                }
                Err(refusal) => refusal.reason(),
            },
        };
        refused = Some((answer, reason));
    }
    let (_, reason) = refused.unwrap_or_default();
    Outcome::Abandoned(Abandon::Refused(reason))
}

/// Keep a verified entry: ids first, then the substrate, then memory — so a
/// write that fails leaves the state as it was.
async fn keep<D: Desk, S: Holds>(
    desk: &mut D,
    state: &Mutex<S>,
    verified: Verified,
    style: &Style,
    rounds: usize,
) -> Outcome {
    let Verified { entry, notes } = verified;
    let (staged, prompt) = {
        let held = hold(state);
        let staged = held.journal().stage(entry);
        let prompt = held.journal().prompt_after(&staged, style);
        (staged, prompt)
    };
    if let Err(e) = desk.keep(&prompt).await {
        return Outcome::Abandoned(Abandon::Keeping(e.to_string()));
    }
    let entry = hold(state).journal_mut().kept(staged);
    Outcome::Wrote {
        entry,
        notes,
        rounds,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::entry::Kind;
    use crate::journal::Journal;
    use crate::span::testing::turn;
    use crate::span::SpanTurn;
    use crate::style::testing::PLAIN;
    use crate::verify::testing::Fake;

    const WRITE: &str = r#"{"name": "journal_write", "arguments": {
        "claims": [{"text": "I told Pax hello.", "cite": "1"}],
        "open": [{"text": "Pax owes me an answer", "relates": "new"}]}}"#;
    const WRITE_SCENERY: &str = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "A light shifted.", "cite": "3"}]}}"#;

    /// A desk that answers from a script and records what it was asked.
    struct Script {
        writes: VecDeque<anyhow::Result<String>>,
        write_asked: Vec<String>,
        specs: Vec<Value>,
        kept: Vec<JournalPrompt>,
        refuse_keeping: bool,
    }

    impl Script {
        fn new(writes: &[&str]) -> Self {
            Self {
                writes: writes.iter().map(|a| Ok(a.to_string())).collect(),
                write_asked: Vec::new(),
                specs: Vec::new(),
                kept: Vec::new(),
                refuse_keeping: false,
            }
        }
    }

    impl Desk for Script {
        async fn write(&mut self, spec: &Value, question: &str) -> anyhow::Result<String> {
            self.write_asked.push(question.to_string());
            self.specs.push(spec.clone());
            self.writes
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("the script ran out")))
        }

        async fn keep(&mut self, prompt: &JournalPrompt) -> anyhow::Result<()> {
            if self.refuse_keeping {
                anyhow::bail!("the substrate said no");
            }
            self.kept.push(prompt.clone());
            Ok(())
        }
    }

    /// Turns 1 and 2 are the writer's own acts; turn 3 is scenery.
    fn turns() -> Vec<SpanTurn> {
        vec![
            turn(1, 1_000, Some(Kind::Did), "You: tell — to Pax: hello"),
            turn(
                2,
                2_000,
                Some(Kind::Did),
                "You: tell — to Pax: are you there",
            ),
            turn(3, 3_000, None, "A light shifts."),
        ]
    }

    fn started(turns: Vec<SpanTurn>) -> (Mutex<Journal>, Span) {
        let mut j = Journal::new();
        let newest = turns.last().map_or(0, |t| t.id);
        let span = j.begin(turns, newest).unwrap();
        (Mutex::new(j), span)
    }

    async fn drafted(desk: &mut Script, turns: Vec<SpanTurn>) -> (Outcome, Mutex<Journal>) {
        let (state, span) = started(turns);
        let (outcome, _) = run(desk, &state, &span, &Fake::default(), &PLAIN).await;
        (outcome, state)
    }

    #[tokio::test]
    async fn a_good_write_keeps_the_entry() {
        let mut desk = Script::new(&[WRITE]);
        let (outcome, state) = drafted(&mut desk, turns()).await;
        let Outcome::Wrote { entry, rounds, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert_eq!(entry.id, 1);
        assert_eq!(rounds, 1);
        assert_eq!(entry.opened.len(), 1);
        assert_eq!(entry.opened[0].id, 1);
        assert_eq!(entry.claims[0].kind, Kind::Did);
        let state = hold(&state);
        assert_eq!(state.covered_to(), 3);
        assert!(!state.pending());
        assert_eq!(state.open().len(), 1);
        assert_eq!(desk.kept.len(), 1);
        assert_eq!(desk.kept[0], state.prompt(&PLAIN));
        assert_eq!(desk.kept[0].sections[0].text, entry.render(&PLAIN));
    }

    #[tokio::test]
    async fn the_write_carries_the_stretch_and_a_spec_of_only_its_citable_turns() {
        let (state, span) = started(turns());
        let mut desk = Script::new(&[WRITE]);
        run(&mut desk, &state, &span, &Fake::default(), &PLAIN).await;
        let first = &desk.write_asked[0];
        assert_eq!(first, &ask::writing(&span, &PLAIN, None));
        assert!(first.contains("#1 ") && first.contains("#2 "), "{first}");
        assert!(first.contains("(background)"), "{first}");
        assert_eq!(desk.specs[0], tools::write_spec(&[1, 2], &[]).unwrap());
    }

    #[tokio::test]
    async fn a_stretch_nothing_can_be_cited_from_is_not_written_about() {
        let mut desk = Script::new(&[]);
        let (outcome, state) =
            drafted(&mut desk, vec![turn(1, 1, None, "The conduit hums.")]).await;
        assert!(matches!(outcome, Outcome::Nothing { .. }), "{outcome:?}");
        assert!(desk.write_asked.is_empty());
        assert_eq!(hold(&state).covered_to(), 1);
        assert!(!hold(&state).pending());
    }

    #[tokio::test]
    async fn a_refused_draft_comes_back_with_itself_and_the_reason() {
        let mut desk = Script::new(&[WRITE_SCENERY, WRITE]);
        let (outcome, _) = drafted(&mut desk, turns()).await;
        let Outcome::Wrote { rounds, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert_eq!(rounds, 2);
        let sent_back = &desk.write_asked[1];
        assert!(sent_back.contains("A light shifted."), "{sent_back}");
        assert!(
            sent_back.contains("It was refused: Claim 1 cites turn 3"),
            "{sent_back}"
        );
    }

    #[tokio::test]
    async fn an_answer_that_does_not_parse_is_sent_back_too() {
        let mut desk = Script::new(&["hmm", WRITE]);
        let (outcome, _) = drafted(&mut desk, turns()).await;
        assert!(
            matches!(outcome, Outcome::Wrote { rounds: 2, .. }),
            "{outcome:?}"
        );
        assert!(desk.write_asked[1].contains("You tried this:\nhmm"));
    }

    #[tokio::test]
    async fn answers_that_are_never_accepted_end_the_draft_after_four_rounds() {
        let mut desk = Script::new(&[WRITE_SCENERY, WRITE_SCENERY, "hmm", WRITE_SCENERY, WRITE]);
        let (outcome, state) = drafted(&mut desk, turns()).await;
        let Outcome::Abandoned(Abandon::Refused(why)) = outcome else {
            panic!("expected the draft to be given up, got {outcome:?}");
        };
        assert!(why.contains("turn 3"), "{why}");
        assert_eq!(desk.write_asked.len(), MAX_ROUNDS);
        assert_eq!(hold(&state).covered_to(), 0);
        assert!(!hold(&state).pending());
        assert!(desk.kept.is_empty());
    }

    #[tokio::test]
    async fn a_write_that_cannot_be_asked_abandons_the_draft() {
        let mut desk = Script::new(&[]);
        let (outcome, state) = drafted(&mut desk, turns()).await;
        assert!(
            matches!(outcome, Outcome::Abandoned(Abandon::Asking(_))),
            "{outcome:?}"
        );
        assert!(!hold(&state).pending());
        assert_eq!(hold(&state).covered_to(), 0);
    }

    #[tokio::test]
    async fn an_entry_that_cannot_be_kept_leaves_the_state_as_it_was() {
        let mut desk = Script::new(&[WRITE]);
        desk.refuse_keeping = true;
        let (outcome, state) = drafted(&mut desk, turns()).await;
        let Outcome::Abandoned(Abandon::Keeping(why)) = outcome else {
            panic!("expected keeping to fail, got {outcome:?}");
        };
        assert!(why.contains("substrate"), "{why}");
        let state = hold(&state);
        assert_eq!(state.covered_to(), 0);
        assert_eq!(state.entries().count(), 0);
        assert!(state.open().is_empty());
        assert!(!state.pending());
    }

    #[tokio::test]
    async fn an_entry_of_nothing_but_repeats_is_not_kept_and_covers_its_stretch() {
        let (state, first) = started(turns());
        run(
            &mut Script::new(&[WRITE]),
            &state,
            &first,
            &Fake::default(),
            &PLAIN,
        )
        .await;

        let second = hold(&state)
            .begin(
                vec![turn(4, 4_000, Some(Kind::Did), "You: tell — to Pax: hello")],
                4,
            )
            .unwrap();
        let again = r#"{"name": "journal_write", "arguments": {
            "claims": [{"text": "I told Pax hello.", "cite": "4"}]}}"#;
        let mut desk = Script::new(&[again]);
        let (outcome, _) = run(&mut desk, &state, &second, &Fake::default(), &PLAIN).await;
        let Outcome::Nothing { why } = outcome else {
            panic!("expected nothing new, got {outcome:?}");
        };
        assert!(why.contains("already in the journal"), "{why}");
        assert!(desk.kept.is_empty());
        let state = hold(&state);
        assert_eq!(state.covered_to(), 4);
        assert_eq!(state.written(), 1);
    }

    #[tokio::test]
    async fn the_next_entry_takes_the_next_ids() {
        let (state, first) = started(turns());
        run(
            &mut Script::new(&[WRITE]),
            &state,
            &first,
            &Fake::default(),
            &PLAIN,
        )
        .await;

        let second = hold(&state)
            .begin(
                vec![turn(
                    4,
                    4_000,
                    Some(Kind::Did),
                    "You: tell — to Pax: hello again",
                )],
                4,
            )
            .unwrap();
        assert_eq!(second.from_turn, 4);
        let again = WRITE
            .replace("\"1\"", "\"4\"")
            .replace("I told Pax hello.", "I greeted Pax a second time.")
            .replace("new", "restates 1");
        let mut desk = Script::new(&[again.as_str()]);
        let (outcome, _) = run(&mut desk, &state, &second, &Fake::default(), &PLAIN).await;
        let Outcome::Wrote { entry, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert_eq!(entry.id, 2);
        let state = hold(&state);
        assert_eq!(state.open().len(), 1);
        assert_eq!(state.open()[0].id, 1);
        assert_eq!(state.covered_to(), 4);
    }

    /// A caller's state that keeps its own bookkeeping beside the journal runs
    /// the same workflow.
    #[tokio::test]
    async fn the_workflow_runs_against_any_holder() {
        struct Beside {
            journal: Journal,
            drafted: u32,
        }
        impl Holds for Beside {
            fn journal(&self) -> &Journal {
                &self.journal
            }
            fn journal_mut(&mut self) -> &mut Journal {
                self.drafted += 1;
                &mut self.journal
            }
        }
        let mut journal = Journal::new();
        let span = journal.begin(turns(), 3).unwrap();
        let state = Mutex::new(Beside {
            journal,
            drafted: 0,
        });
        let (outcome, _) = run(
            &mut Script::new(&[WRITE]),
            &state,
            &span,
            &Fake::default(),
            &PLAIN,
        )
        .await;
        assert!(matches!(outcome, Outcome::Wrote { .. }), "{outcome:?}");
        let held = hold(&state);
        assert_eq!(held.journal.written(), 1);
        assert!(held.drafted > 0);
    }

    /// A typed claim the world reads differently is kept as the world reads it.
    #[tokio::test]
    async fn a_claim_the_world_contradicts_is_kept_corrected() {
        let typed = r#"{"name": "journal_write", "arguments": {"claims": [{
            "text": "The build is red.", "cite": "1",
            "typed": {"subject": "build", "attribute": "status", "value": "red"}}]}}"#;
        let (state, span) = started(turns());
        let world = Fake::default().with("build", "status", "green");
        let (outcome, _) = run(&mut Script::new(&[typed]), &state, &span, &world, &PLAIN).await;
        let Outcome::Wrote { entry, notes, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert!(entry.claims[0].corrected);
        assert_eq!(notes[0].verdict.name(), "corrected");
    }
}
