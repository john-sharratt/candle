//! One draft, start to finish: the write, the check, the keep.
//!
//! [`run`] is everything that happens between "the character said its journal
//! needs an entry" and "the journal has, or has not, a new entry", and it is
//! deterministic: the model is asked for the entry, every answer is read by
//! [`tools::parse`] and checked by [`verify`], and the same answers always draw
//! the same outcome. It is written against [`Desk`] — where the question is put
//! and an entry is kept — so the whole workflow runs here against a script, and
//! the real engine-backed desk is only the plumbing underneath. Whether a stretch
//! is worth an entry is not decided here: the guardian's journal module asks the
//! character, and a draft starts only on a yes.
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
//! never across a question to the model, so the main line is never waiting on a
//! draft.

use std::future::Future;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use candle_conversation::stencil::ToolSpec;

use crate::engine::journal::ask;
use crate::engine::journal::entry::{Entry, Item};
use crate::engine::journal::section::JournalPrompt;
use crate::engine::journal::state::{JournalState, Span};
use crate::engine::journal::tools;
use crate::engine::journal::verify::{verify, Note, Verified, World};

/// Answers the write may take before the draft is given up.
pub const MAX_ROUNDS: usize = 4;

/// Where a draft's question is put and its entry is kept.
pub trait Desk: Send {
    /// Ask for the entry, held to one call of `spec`, and return the decode as it
    /// came back.
    fn write(
        &mut self,
        spec: &ToolSpec,
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
    /// reaches this only for a stretch nothing can be cited from; the guardian's
    /// "no" is recorded by the runtime.
    Nothing { why: String },
    /// The draft did not finish; the same turns are covered again next time.
    Abandoned(Abandon),
}

/// Where a draft's time went.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    /// Asking whether the stretch was worth an entry. The guardian asks, so the
    /// workflow leaves this at zero and the runtime fills it in.
    pub gate_ms: u64,
    /// Every answer to the write, refused ones included.
    pub write_ms: u64,
}

fn hold(state: &Mutex<JournalState>) -> MutexGuard<'_, JournalState> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn millis(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Run one draft over `span`, which the caller took with
/// [`JournalState::begin`]. Whichever way it ends, the state has been told.
pub async fn run<D: Desk>(
    desk: &mut D,
    state: &Mutex<JournalState>,
    span: &Span,
    world: &(dyn World + Sync),
) -> (Outcome, Timing) {
    let mut timing = Timing::default();
    let outcome = draft(desk, state, span, world, &mut timing).await;
    if matches!(outcome, Outcome::Abandoned(_)) {
        hold(state).abandon();
    }
    (outcome, timing)
}

async fn draft<D: Desk>(
    desk: &mut D,
    state: &Mutex<JournalState>,
    span: &Span,
    world: &(dyn World + Sync),
    timing: &mut Timing,
) -> Outcome {
    let open = hold(state).open().to_vec();
    let Some(spec) = tools::write_spec(&span.citable_ids(), &open) else {
        hold(state).nothing_to_write(span);
        return Outcome::Nothing {
            why: "nothing in the stretch was something a claim could rest on".to_string(),
        };
    };

    let started = Instant::now();
    let outcome = write(desk, state, span, world, &spec, &open).await;
    timing.write_ms = millis(started);
    outcome
}

/// Ask for the entry until one passes the check or the rounds run out.
async fn write<D: Desk>(
    desk: &mut D,
    state: &Mutex<JournalState>,
    span: &Span,
    world: &(dyn World + Sync),
    spec: &ToolSpec,
    open: &[Item],
) -> Outcome {
    let mut refused: Option<(String, String)> = None;
    for round in 1..=MAX_ROUNDS {
        let question = ask::writing(
            span,
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
                Ok(verified) => return keep(desk, state, verified, round).await,
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
async fn keep<D: Desk>(
    desk: &mut D,
    state: &Mutex<JournalState>,
    verified: Verified,
    rounds: usize,
) -> Outcome {
    let Verified { entry, notes } = verified;
    let (staged, prompt) = {
        let held = hold(state);
        let staged = held.stage(entry);
        let prompt = held.prompt_after(&staged);
        (staged, prompt)
    };
    if let Err(e) = desk.keep(&prompt).await {
        return Outcome::Abandoned(Abandon::Keeping(e.to_string()));
    }
    let entry = hold(state).kept(staged);
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
    use crate::engine::event::{Event, EventKind, Salience};
    use crate::engine::window::Window;

    const WRITE: &str = r#"{"name": "journal_write", "arguments": {
        "claims": [{"text": "I told Pax hello.", "cite": "1"}],
        "open": [{"text": "Pax owes me an answer", "relates": "new"}]}}"#;
    const WRITE_SCENERY: &str = r#"{"name": "journal_write", "arguments": {"claims": [{"text": "A light shifted.", "cite": "3"}]}}"#;

    struct Quiet;

    impl World for Quiet {
        fn read(&self, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    /// A desk that answers from a script and records what it was asked.
    struct Script {
        writes: VecDeque<anyhow::Result<String>>,
        write_asked: Vec<String>,
        kept: Vec<JournalPrompt>,
        refuse_keeping: bool,
    }

    impl Script {
        fn new(writes: &[&str]) -> Self {
            Self {
                writes: writes.iter().map(|a| Ok(a.to_string())).collect(),
                write_asked: Vec::new(),
                kept: Vec::new(),
                refuse_keeping: false,
            }
        }
    }

    impl Desk for Script {
        async fn write(&mut self, _: &ToolSpec, question: &str) -> anyhow::Result<String> {
            self.write_asked.push(question.to_string());
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

    /// Turns 1 and 2 are the character's own acts; turn 3 is scenery.
    fn window() -> Window {
        let mut w = Window::with_default_cap();
        w.push_npc("tell — to Pax: hello", 1_000);
        w.push_npc("tell — to Pax: are you there", 2_000);
        w.push_event(&Event::new(
            0,
            3_000,
            Salience::IDLE,
            EventKind::Description {
                text: "A light shifts.".into(),
            },
        ));
        w
    }

    fn started(w: &Window) -> (Mutex<JournalState>, Span) {
        let mut state = JournalState::new(0);
        let span = state.begin(w).unwrap();
        (Mutex::new(state), span)
    }

    async fn drafted(desk: &mut Script, w: &Window) -> (Outcome, Mutex<JournalState>) {
        let (state, span) = started(w);
        let (outcome, _) = run(desk, &state, &span, &Quiet).await;
        (outcome, state)
    }

    #[tokio::test]
    async fn a_good_write_keeps_the_entry() {
        let mut desk = Script::new(&[WRITE]);
        let (outcome, state) = drafted(&mut desk, &window()).await;
        let Outcome::Wrote { entry, rounds, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert_eq!(entry.id, 1);
        assert_eq!(rounds, 1);
        assert_eq!(entry.opened.len(), 1);
        assert_eq!(entry.opened[0].id, 1);
        let state = hold(&state);
        assert_eq!(state.covered_to(), 3);
        assert!(!state.pending());
        assert_eq!(state.open().len(), 1);
        assert_eq!(desk.kept.len(), 1);
        assert_eq!(desk.kept[0], state.prompt());
        assert_eq!(desk.kept[0].sections[0].text, entry.render());
    }

    #[tokio::test]
    async fn the_write_carries_the_stretch() {
        let w = window();
        let (state, span) = started(&w);
        let mut desk = Script::new(&[WRITE]);
        run(&mut desk, &state, &span, &Quiet).await;
        let first = &desk.write_asked[0];
        assert!(first.contains("#1 ") && first.contains("#2 "), "{first}");
        assert!(first.contains("(background)"), "{first}");
        assert_eq!(first, &ask::writing(&span, None));
    }

    #[tokio::test]
    async fn a_stretch_nothing_can_be_cited_from_is_not_written_about() {
        let mut w = Window::with_default_cap();
        w.push_event(&Event::new(
            0,
            1,
            Salience::IDLE,
            EventKind::Description {
                text: "The conduit hums.".into(),
            },
        ));
        let mut desk = Script::new(&[]);
        let (outcome, state) = drafted(&mut desk, &w).await;
        assert!(matches!(outcome, Outcome::Nothing { .. }), "{outcome:?}");
        assert!(desk.write_asked.is_empty());
        assert_eq!(hold(&state).covered_to(), 1);
    }

    #[tokio::test]
    async fn a_refused_draft_comes_back_with_itself_and_the_reason() {
        let mut desk = Script::new(&[WRITE_SCENERY, WRITE]);
        let (outcome, _) = drafted(&mut desk, &window()).await;
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
    async fn answers_that_are_never_accepted_end_the_draft_after_four_rounds() {
        let mut desk = Script::new(&[WRITE_SCENERY, WRITE_SCENERY, "hmm", WRITE_SCENERY, WRITE]);
        let (outcome, state) = drafted(&mut desk, &window()).await;
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
        let (outcome, state) = drafted(&mut desk, &window()).await;
        assert!(
            matches!(outcome, Outcome::Abandoned(Abandon::Asking(_))),
            "{outcome:?}"
        );
        assert!(!hold(&state).pending());
    }

    #[tokio::test]
    async fn an_entry_that_cannot_be_kept_leaves_the_state_as_it_was() {
        let mut desk = Script::new(&[WRITE]);
        desk.refuse_keeping = true;
        let (outcome, state) = drafted(&mut desk, &window()).await;
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
    async fn the_next_entry_takes_the_next_ids() {
        let mut w = window();
        let mut state = JournalState::new(0);
        let first = state.begin(&w).unwrap();
        let state = Mutex::new(state);
        run(&mut Script::new(&[WRITE]), &state, &first, &Quiet).await;

        w.push_npc("tell — to Pax: hello again", 4_000);
        let second = hold(&state).begin(&w).unwrap();
        assert_eq!(second.from_turn, 4);
        let again = WRITE.replace("\"1\"", "\"4\"").replace("new", "restates 1");
        let mut desk = Script::new(&[again.as_str()]);
        let (outcome, _) = run(&mut desk, &state, &second, &Quiet).await;
        let Outcome::Wrote { entry, .. } = outcome else {
            panic!("expected an entry, got {outcome:?}");
        };
        assert_eq!(entry.id, 2);
        let state = hold(&state);
        assert_eq!(state.open().len(), 1);
        assert_eq!(state.open()[0].id, 1);
        assert_eq!(state.covered_to(), 4);
    }
}
