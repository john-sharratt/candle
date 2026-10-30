//! The memory's writer (`docs/memory_backbone.md` §5.3): what keeps every
//! journal's summary current, folds each full window's closed periods, and
//! closes conversations gone quiet — on the resident model, at the `Low`
//! band, beneath every turn a person waits on.
//!
//! What is due, the question each job puts and the operation its answer
//! becomes are the `journals` crate's (`journals::brief`); this file asks the
//! model and writes what comes back.
//!
//! **Each question is asked in a conversation opened for it alone.** One
//! person's journal is never in the question about another's: the
//! conversation is new, its group closed to every other turn in the reserved
//! `Prose` group ([`Builder::for_closed_reserved`]), and retired the moment
//! its answer is read. Its KV and its frame never go to disk; the sealed
//! turn's tokens and the timeline's records do, as every sealed turn's do —
//! the asker's own journal text, kept in the asker's own substrate — until
//! compaction drops the tombstoned timeline. The answer is held to the one
//! call the job offers, so a summary cites only what it was shown.
//!
//! A summary whose answer is refused leaves the old one, marked stale; any
//! refused job waits for what its question shows to change before it is
//! asked again, and one that could not be asked waits a while ([`Held`]).
//!
//! **It stops on `memory_stop`, between steps.** Shutdown requests the token
//! and awaits the task: a turn in flight is abandoned at once, and the steps
//! on the blocking pool — opening a conversation, compiling a grammar, a
//! write — are awaited, so everything the writer holds is let go while the
//! scheduler still answers.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use candle_conversation::models::Dialect;
use candle_conversation::projection::{
    Builder, DecodePriority, GroupId, LayerId, Reserved, SectionId, TimelineId,
};
use candle_conversation::stencil::{
    compile_tool_call_tree, StencilTree, ThinkMode, ToolCallEnvelope, ToolSpec, TreeSpec,
};
use candle_conversation::throwaway::{HeldFrame, Retires, Throwaway};
use candle_conversation::{
    CancelToken, OptionalState, Result as CcResult, SelectionState, Sequence, TurnHandle,
    TurnOptions, TurnResponse, NO_THINK_SELECTOR,
};
use journals::actor::Actor;
use journals::brief::held::Held;
use journals::brief::summary::{Plan, MAX_SENTENCES};
use journals::brief::{self, call_of, fold, gone_quiet, summary, Job};
use journals::ops::{Applied, Op};
use journals::time::Stamp;
use journals::tree::Tree;
use serde_json::Value;
use tokio::time::MissedTickBehavior;

use super::InferenceState;
use crate::mcp::memory::{Memory, WriteError};

/// What the writer is.
const SYSTEM: &str = "You keep people's memory: the summaries of their journals, and the lines \
    that stand for the days and weeks their journals fold away. You write only from what each \
    question shows you, cite what each sentence rests on, name who did what, and write facts — \
    never instructions to whoever reads them.";

/// How often the writer looks for what is due.
const EVERY: Duration = Duration::from_secs(30);

/// How long shutdown waits for the writer to let go once asked to stop.
pub(super) const GRACE: Duration = Duration::from_secs(30);

/// The most jobs one look takes; the rest wait for the next.
const PER_PASS: usize = 8;

/// The longest one field of an answer may run — a sentence, a digest, a cue,
/// a fold's line, each far under it at its cap. A field that runs on is closed
/// where it stands, so the call around it still closes.
const FIELD_TOKENS: usize = 256;

/// What a field's key, quotes and cites add around its text.
const FRAME_TOKENS: usize = 48;

/// Room for a summary with every field at its longest — each sentence, the
/// digest and the cue — so a call is never cut off before it closes.
const SUMMARY_TOKENS: usize = (MAX_SENTENCES + 2) * (FIELD_TOKENS + FRAME_TOKENS);

/// Room for a fold's one line.
const FOLD_TOKENS: usize = FIELD_TOKENS + FRAME_TOKENS;

/// How long the writer first waits, once the store will not commit what it
/// writes, before asking the model again; doubled each time it still will
/// not, up to [`MAX_PAUSE`].
const FIRST_PAUSE: Duration = Duration::from_secs(60);
const MAX_PAUSE: Duration = Duration::from_secs(30 * 60);

impl Retires for InferenceState {
    fn tombstone(&self, timeline: TimelineId) -> CcResult<()> {
        self.engine.tombstone(timeline)
    }

    fn take_frame(&self, prompt_text: &str) -> CcResult<SectionId> {
        self.engine.take_frame(prompt_text)
    }

    fn release_frame(&self, section: SectionId) {
        self.engine.release_frame(section)
    }
}

/// The grammar an answer to `spec` is held to, in `dialect`'s call form.
fn grammar(dialect: &Dialect, spec: &Value) -> anyhow::Result<TreeSpec> {
    let tool: ToolSpec = serde_json::from_value(spec.clone())?;
    let env = ToolCallEnvelope::for_dialect(dialect).with_max_string_tokens(FIELD_TOKENS as u32);
    compile_tool_call_tree(&[tool], &env)
        .map_err(|e| anyhow::anyhow!("the answer's grammar would not build: {e:#}"))
}

/// [`grammar`] for `spec`, compiled for this engine. A spec that builds no
/// grammar is unusable as its journal stands; an engine that will not compile
/// one now is passing.
///
/// The compile runs off the async workers, like every other engine call here
/// that can take a while.
async fn compiled(
    state: &Arc<InferenceState>,
    spec: &Value,
) -> Result<Arc<StencilTree>, Ungrammatical> {
    let tree = grammar(&state.refresh_config.dialect, spec)
        .map_err(|e| Ungrammatical::Spec(format!("{e:#}")))?;
    let compiling = Arc::clone(state);
    let compiled = tokio::task::spawn_blocking(move || compiling.engine.compile_stencil(&tree))
        .await
        .map_err(|e| Ungrammatical::Engine(format!("the grammar's compile did not finish: {e}")))?
        .map_err(|e| {
            Ungrammatical::Engine(format!("the answer's grammar would not compile: {e}"))
        })?;
    Ok(Arc::new(compiled))
}

/// `op` written for `actor`, off the async workers: a write waits for the
/// store, which the sync thread holds through each push and fetch.
async fn write(memory: &Arc<Memory>, actor: Actor, op: Op) -> Result<Applied, WriteError> {
    let memory = Arc::clone(memory);
    tokio::task::spawn_blocking(move || memory.write(&actor, &op))
        .await
        .unwrap_or_else(|e| {
            Err(WriteError::Unavailable(format!(
                "the write did not finish: {e}"
            )))
        })
}

/// Why a job's grammar is not there.
enum Ungrammatical {
    /// Its spec builds none: unusable as its journal stands.
    Spec(String),
    /// The engine would not compile it now: passing.
    Engine(String),
}

impl From<Ungrammatical> for Outcome {
    fn from(u: Ungrammatical) -> Outcome {
        match u {
            Ungrammatical::Spec(why) => Outcome::Unusable(why),
            Ungrammatical::Engine(why) => Outcome::Passing(why),
        }
    }
}

/// One question's conversation and what it holds, let go in the order the
/// fields are declared on every path out — an error, a cancelled future, an
/// answer: the slot, then the timeline it wrote, then the frame it read.
struct Asking {
    seq: Sequence,
    _retired: Throwaway<Arc<InferenceState>>,
    _frame: HeldFrame<Arc<InferenceState>>,
}

/// A conversation for one question: the writer's system prompt, then nothing
/// — its group admits no other turn. Its KV and frame never go to disk.
fn open(state: &Arc<InferenceState>) -> anyhow::Result<Asking> {
    let engine = &state.engine;
    // The frame seals the prompt's own text, which ingest wraps in the
    // dialect's markers; the conversation is opened on the wrapped form —
    // as `ConversationEngine::new_conversation` does.
    let formatted = state.refresh_config.dialect.format_system_prompt(SYSTEM);
    let frame = HeldFrame::take(Arc::clone(state), SYSTEM)?;
    let builder =
        Builder::for_closed_reserved(SYSTEM, Reserved::Prose, frame.id(), DecodePriority::Low);
    let mut config = state.refresh_config.clone();
    config.context_window_turns = 0;
    let seq = engine.new_conversation_with_projection(
        &formatted,
        builder,
        LayerId::reserved(Reserved::Prose),
        GroupId::reserved(Reserved::Prose),
        config,
    )?;
    engine.mark_timeline_transient(seq.timeline_id());
    Ok(Asking {
        _retired: Throwaway::new(Arc::clone(state), seq.timeline_id(), "memory writer"),
        seq,
        _frame: frame,
    })
}

/// `question` put to the writer, its answer held to `grammar`: the call as
/// written.
async fn ask(
    state: &Arc<InferenceState>,
    question: &str,
    grammar: Arc<StencilTree>,
    max_tokens: usize,
) -> anyhow::Result<String> {
    // Opening waits on the scheduler for a slot, synchronously; off the
    // async workers.
    let opening = Arc::clone(state);
    let mut asking = tokio::task::spawn_blocking(move || open(&opening))
        .await
        .map_err(|e| anyhow::anyhow!("the writer's conversation did not open: {e}"))??;
    let seq = &mut asking.seq;
    let mut selection = SelectionState::new();
    selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    let mut sampling = seq.default_sampling();
    sampling.temperature = 0.0;
    sampling.apply_think_mode(ThinkMode::Off, &state.tokenizer, max_tokens);
    let opts = TurnOptions {
        max_tokens: Some(max_tokens),
        selection,
        sampling: Some(sampling),
        turn_grammar: Some(grammar),
        ..Default::default()
    };
    let turn = match seq.submit_turn_with_options(question, opts) {
        Ok(handle) => match handle.wait_cancellable_async(&state.memory_stop).await {
            Ok(resp) => Ok((handle, resp)),
            Err(e) => Err(anyhow::anyhow!("the writer ended without an answer: {e}")),
        },
        Err(e) => Err(anyhow::anyhow!("the writer's question was not taken: {e}")),
    };
    // Sealing, abandoning and letting the conversation go each wait on the
    // scheduler: off the async workers, as the opening was.
    tokio::task::spawn_blocking(move || end(asking, turn))
        .await
        .map_err(|e| anyhow::anyhow!("the writer's turn did not end: {e}"))?
}

/// A question's turn ended: sealed and its answer read, or abandoned — and
/// its conversation let go, slot then timeline then frame.
fn end(
    mut asking: Asking,
    turn: anyhow::Result<(TurnHandle, TurnResponse)>,
) -> anyhow::Result<String> {
    match turn {
        Ok((handle, resp)) => {
            if let Err(e) = asking.seq.finish_turn(handle, &resp) {
                tracing::warn!("journals: the writer's turn would not seal: {e}");
            }
            Ok(resp.text)
        }
        Err(e) => {
            asking.seq.abort_turn();
            Err(e)
        }
    }
}

/// What a write's result leaves the writer to do.
#[derive(Debug, PartialEq, Eq)]
enum Settled {
    /// Written: the commit's subject.
    Written(String),
    /// Built over what has moved since: asked again next pass.
    Again(String),
    /// Refused as made: held until what its question shows moves.
    Refused(String),
    /// A file it touches is broken: asked again in a while.
    Later(String),
    /// The store will not commit anything.
    StoreFailing(String),
}

fn settle(result: Result<Applied, WriteError>) -> Settled {
    match result {
        Ok(applied) => Settled::Written(applied.subject),
        Err(e) if e.is_conflict() => Settled::Again(e.to_string()),
        Err(e) if e.is_refusal() => Settled::Refused(e.to_string()),
        Err(e) if e.is_broken() => Settled::Later(e.to_string()),
        Err(e) => Settled::StoreFailing(e.to_string()),
    }
}

/// What one job came to.
enum Outcome {
    /// Nothing to do: the job is no longer wanted as the tree stands.
    Moot,
    /// The operation the answer — or the job itself — makes.
    Write(Actor, Op),
    /// Nothing usable can come of it as its journal stands: a grammar that
    /// would not build, an answer the job cannot read.
    Unusable(String),
    /// The model could not be asked now; asked again in a while.
    Passing(String),
}

/// Where a job left the writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Done {
    /// On to the next job.
    Next,
    /// The store would not commit what was written: asking the model more
    /// spends turns nothing can keep.
    StoreFailing,
}

/// What the writer keeps between passes.
struct Writer {
    memory: Arc<Memory>,
    held: Held,
    /// The fold's grammar, the same for every fold: compiled once.
    fold: Option<Arc<StencilTree>>,
}

impl Writer {
    async fn fold_grammar(
        &mut self,
        state: &Arc<InferenceState>,
    ) -> Result<Arc<StencilTree>, Ungrammatical> {
        if let Some(g) = &self.fold {
            return Ok(Arc::clone(g));
        }
        let g = compiled(state, &fold::spec()).await?;
        self.fold = Some(Arc::clone(&g));
        Ok(g)
    }

    /// What `job` comes to, asked over `tree`.
    async fn answer(&mut self, state: &Arc<InferenceState>, tree: &Tree, job: &Job) -> Outcome {
        match job {
            Job::Close {
                path,
                person,
                client,
            } if tree.get(path).is_some_and(|j| gone_quiet(j, Stamp::now())) => Outcome::Write(
                Actor::Server {
                    person: person.clone(),
                    client: *client,
                },
                Op::Close { path: path.clone() },
            ),
            // Spoken in again since the pass was planned.
            Job::Close { .. } => Outcome::Moot,
            Job::Fold { path, plan } => {
                let Some(fold::Ask { question, over }) = fold::ask(tree, path, plan) else {
                    return Outcome::Moot;
                };
                let g = match self.fold_grammar(state).await {
                    Ok(g) => g,
                    Err(u) => return u.into(),
                };
                match ask(state, &question, g, FOLD_TOKENS).await {
                    Ok(text) => match call_of(&text, fold::NAME)
                        .and_then(|a| fold::op(path, plan, &over, &a))
                    {
                        Ok(op) => Outcome::Write(Actor::Zend, op),
                        Err(why) => Outcome::Unusable(why),
                    },
                    Err(e) => Outcome::Passing(format!("{e:#}")),
                }
            }
            Job::Summary { path } => match summary::plan(tree, path) {
                None => Outcome::Moot,
                Some(Plan::Clear(op)) => Outcome::Write(Actor::Zend, op),
                Some(Plan::Ask {
                    question,
                    spec,
                    over,
                }) => {
                    let g = match compiled(state, &spec).await {
                        Ok(g) => g,
                        Err(u) => return u.into(),
                    };
                    match ask(state, &question, g, SUMMARY_TOKENS).await {
                        Ok(text) => match call_of(&text, summary::NAME)
                            .and_then(|a| summary::op(path, &over, &a))
                        {
                            Ok(op) => Outcome::Write(Actor::Zend, op),
                            Err(why) => Outcome::Unusable(why),
                        },
                        Err(e) => Outcome::Passing(format!("{e:#}")),
                    }
                }
            },
        }
    }

    /// One job, done: its answer written; or, refused, the job held back
    /// until what its question shows changes — and a summary's old text
    /// marked stale; or, not asked, the job deferred a while.
    async fn work(&mut self, state: &Arc<InferenceState>, job: Job) -> Done {
        let Ok(tree) = self.memory.tree() else {
            return Done::Next;
        };
        let what = match &job {
            Job::Close { path, .. } => format!("close of {path}"),
            Job::Fold { path, plan } => format!("fold of {} in {path}", plan.period),
            Job::Summary { path } => format!("summary of {path}"),
        };
        let unusable = match self.answer(state, &tree, &job).await {
            Outcome::Moot => return Done::Next,
            Outcome::Passing(why) => {
                tracing::warn!("journals: no {what} for now: {why}");
                self.held.deferred(&job, Stamp::now());
                return Done::Next;
            }
            Outcome::Unusable(why) => why,
            Outcome::Write(actor, op) => match settle(write(&self.memory, actor, op).await) {
                Settled::Written(subject) => {
                    tracing::info!("journals: {subject}");
                    return Done::Next;
                }
                Settled::Refused(why) => why,
                Settled::Again(why) => {
                    tracing::info!("journals: the {what} is asked again next pass: {why}");
                    return Done::Next;
                }
                Settled::Later(why) => {
                    tracing::warn!("journals: the {what} waits for a mended file: {why}");
                    self.held.deferred(&job, Stamp::now());
                    return Done::Next;
                }
                Settled::StoreFailing(why) => {
                    tracing::warn!("journals: the {what} was not written: {why}");
                    return Done::StoreFailing;
                }
            },
        };
        tracing::warn!("journals: the {what} was refused: {unusable}");
        self.held.refused(&tree, &job);
        if let Job::Summary { path } = &job {
            if tree.get(path).is_some_and(|j| j.summary().is_some()) {
                let stale = Op::MarkStale { path: path.clone() };
                if let Err(e) = write(&self.memory, Actor::Zend, stale).await {
                    tracing::debug!("journals: {path} not marked stale: {e}");
                }
            }
        }
        Done::Next
    }
}

/// The pause after the store failed again, having paused `last` (if at all):
/// [`FIRST_PAUSE`], then doubled, to [`MAX_PAUSE`].
fn next_pause(last: Option<Duration>) -> Duration {
    last.map_or(FIRST_PAUSE, |d| (d * 2).min(MAX_PAUSE))
}

/// Look for what is due every [`EVERY`] and do it, until `stop` is requested,
/// the engine is gone, or the memory cannot be served. The engine is held only
/// for a job at a time. A pass longer than [`EVERY`] is followed by a full
/// [`EVERY`], never by the ticks it missed.
pub(super) async fn run(state: Weak<InferenceState>, memory: Arc<Memory>, stop: CancelToken) {
    let mut every = tokio::time::interval(EVERY);
    every.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut writer = Writer {
        memory,
        held: Held::default(),
        fold: None,
    };
    // While the store will not commit: until when the model is not asked, and
    // how long the last such pause was.
    let mut paused: Option<(Instant, Duration)> = None;
    loop {
        tokio::select! {
            _ = every.tick() => {}
            _ = stop.wait() => return,
        }
        if !writer.memory.serves() {
            tracing::warn!("journals: the memory cannot be served; its writer stops");
            return;
        }
        if paused.is_some_and(|(until, _)| Instant::now() < until) {
            continue;
        }
        let Ok(tree) = writer.memory.tree() else {
            continue;
        };
        let now = Stamp::now();
        let jobs = writer.held.sift(&tree, brief::due(&tree, now), now);
        drop(tree);
        let mut failing = false;
        for job in jobs.into_iter().take(PER_PASS) {
            if stop.is_cancelled() {
                return;
            }
            let Some(state) = state.upgrade() else { return };
            if writer.work(&state, job).await == Done::StoreFailing {
                failing = true;
                break;
            }
        }
        paused = match (failing, paused) {
            (false, _) => None,
            (true, last) => {
                let pause = next_pause(last.map(|(_, d)| d));
                tracing::warn!(
                    secs = pause.as_secs(),
                    "journals: the store is not committing; the writer waits before asking again"
                );
                Some((Instant::now() + pause, pause))
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use journals::error::{Error, Problem};
    use journals::ops::apply;
    use journals::path::{JournalPath, Owner};
    use journals::scope::Client;
    use journals::slug::Slug;
    use journals::store::StoreError;
    use journals::tree::load::load;

    use super::*;

    /// **Every call the writer offers builds as a grammar** in a checkpoint's
    /// own call form: a summary's, with its guided cites, and a fold's.
    #[test]
    fn the_writers_calls_build_as_grammars() {
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([(
            "persons/john/person.yaml".to_string(),
            b"person: john\nname: \"John\"\naccounts: []\ngroups: []\n".to_vec(),
        )]);
        let mut tree = load(&files);
        let john = Slug::parse("john").unwrap();
        let root = JournalPath::root(Owner::Person(john.clone()));
        apply(
            &mut tree,
            &Actor::Person {
                person: john,
                client: Client::Grok,
                admin: false,
            },
            &Op::Note {
                path: root.clone(),
                text: "Prefers tea.".into(),
            },
            Stamp::now(),
        )
        .unwrap();
        let spec = summary::spec(&tree, &root).unwrap();
        for dialect in [Dialect::qwen35(), Dialect::chat_ml()] {
            grammar(&dialect, &spec).unwrap();
            grammar(&dialect, &fold::spec()).unwrap();
        }
        assert!(grammar(&Dialect::qwen35(), &serde_json::json!({"params": 1})).is_err());
    }

    /// **A write's result is routed by its kind**: a conflict is asked again,
    /// a refusal held, a broken file deferred, anything else is the store
    /// failing — and nothing is routed by its wording.
    #[test]
    fn a_writes_result_is_settled_by_its_kind() {
        let journal = |e: Error| Err(WriteError::Store(StoreError::Journal(e)));
        assert_eq!(
            settle(Ok(Applied {
                subject: "Summary of ~".into(),
                ..Applied::default()
            })),
            Settled::Written("Summary of ~".into())
        );
        assert_eq!(
            settle(journal(Error::conflict("moved"))),
            Settled::Again("moved".into())
        );
        for refusal in [
            Error::refused("over its cap"),
            Error::invalid("not one line"),
            Error::forbidden("not yours"),
            Error::not_found("gone"),
        ] {
            assert!(matches!(settle(journal(refusal)), Settled::Refused(_)));
        }
        assert!(matches!(
            settle(journal(Error::Broken(Problem::new("a.yaml", None, "bad")))),
            Settled::Later(_)
        ));
        assert!(matches!(
            settle(Err(WriteError::Store(StoreError::Repository(
                "lock held".into()
            )))),
            Settled::StoreFailing(_)
        ));
        assert!(matches!(
            settle(Err(WriteError::Unavailable("still being read".into()))),
            Settled::StoreFailing(_)
        ));
    }

    #[test]
    fn a_failing_store_is_waited_on_longer_each_time_to_a_ceiling() {
        assert_eq!(next_pause(None), Duration::from_secs(60));
        assert_eq!(
            next_pause(Some(Duration::from_secs(60))),
            Duration::from_secs(120)
        );
        assert_eq!(
            next_pause(Some(Duration::from_secs(20 * 60))),
            Duration::from_secs(30 * 60)
        );
        assert_eq!(next_pause(Some(MAX_PAUSE)), MAX_PAUSE);
    }
}
