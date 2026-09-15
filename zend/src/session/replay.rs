//! Replay a recorded conversation to one of its turns, then decode that turn
//! live — the harness for reproducing where a conversation went wrong.
//!
//! The recorded turns before the resume point go back in through the real turn
//! path, DECODED again with their recorded ids forced in place of sampling
//! ([`TurnOptions::recorded_turn`]): the same prefill, the same one-token steps
//! and page cuts, the same reasoning boundary and mid-decode reprojections, and
//! exactly the ids the conversation sealed. That rebuilds the K/V, recurrent and
//! index state the way the conversation built it — the substrate keeps only a
//! conversation's latest recurrent snapshot, so the state before an earlier turn
//! cannot be read back. Prefilling the recorded text instead seals none of the
//! pages a decode cuts, so every later projection shows those turns' reasoning
//! where the conversation had windowed it out.
//! The resume turn's user half is then decoded under the chat's own projection,
//! sampling, think steering and tool stencil ([`turn_projection`],
//! [`turn_sampling`], [`turn_triggers`]), once per run. **Every run replays the
//! history onto a timeline of its own** and decodes there, and that timeline is
//! tombstoned whole before the next run starts, so no run sees another's turns.
//! A timeline per run is what keeps the runs independent: a run seals its
//! answer — tool round included — onto the timeline it decodes on, so a second
//! run sharing it would project the first run's turn as history, and retiring
//! the timeline whole removes everything the run added in one step. On a
//! read-only substrate (`DaemonConfig::read_only_substrate`) none of it reaches
//! disk.
//!
//! A replay is not bit-identical to the recorded run: the resume turn's live
//! seed was drawn from the clock and never recorded, and each replayed turn's
//! projections are rebuilt from a substrate holding this run's turns rather
//! than the recording's. A failure is therefore measured as a rate over seeds,
//! with argmax as the deterministic anchor.
//!
//! The replay also sets the turns it replayed beside the recorded ones
//! ([`TurnComparison`]): the ids each sealed and the widths of its index pages.
//! A turn that differs there is not the context the recorded decode read, so a
//! replay that passes where the recording failed points at that turn.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use candle_conversation::{
    ProjectionEvent, RecoveredMessage, Role as TurnRole, SamplingConfig, SealedTurn,
    SelectionState, TurnEvent, TurnOptions,
};

use super::{
    think_mode_from_selection, timeline_for, turn_projection, turn_sampling, turn_triggers,
    ZendSession,
};
use crate::api::chat::tool_round_selection;
use crate::tools::tool_round_text;
use crate::types::ToolMode;

/// Where a turn projects the tool-call demonstration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Demonstration {
    /// On every turn, tool rounds included — how conversations recorded before
    /// tool rounds dropped it ran.
    EveryTurn,
    /// On a reply's first step only — the chat as it runs now
    /// (`tool_round_selection`).
    FirstStepOnly,
}

/// How one run samples its decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplaySampling {
    /// Greedy — the deterministic anchor.
    Argmax,
    /// The chat's own sampling, with this seed.
    Seed(u64),
}

/// What to replay, and how many times.
pub struct ReplaySpec {
    /// The recorded conversation.
    pub conv_id: String,
    /// The turn to decode, counted from 0 over the conversation's turns; every
    /// turn before it is replayed.
    pub resume_turn: usize,
    /// The decode limit, in tokens.
    pub max_tokens: usize,
    /// The reply's dials, as the chat builds them (`dial_selection` and
    /// `apply_tools_dial`).
    pub selection: SelectionState,
    pub tools_mode: ToolMode,
    pub demonstration: Demonstration,
    /// One decode of the resume turn per entry.
    pub runs: Vec<ReplaySampling>,
}

/// One recorded turn: its user half and the assistant's reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTurn {
    pub user: String,
    pub assistant: String,
}

/// One run: the history it rebuilt, and its decode of the resume turn.
#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    /// The turns before the resume point as this run sealed them, beside the
    /// recorded ones.
    pub turns: Vec<TurnComparison>,
    /// Wall time to replay them.
    pub replay_secs: f64,
    pub sampling: ReplaySampling,
    pub text: String,
    pub tokens: usize,
    pub prefill_ms: f64,
    pub tokens_per_second: f64,
    /// Projection points the decode emitted — the opening one and each
    /// reprojection.
    pub projections: usize,
    /// The last of them: the context the decode finished against, turn by
    /// turn. `None` when the decode emitted none.
    pub last_projection: Option<ProjectionEvent>,
    /// Wall time of the resume turn's decode.
    pub decode_secs: f64,
}

/// A whole replay: every run, in the order of `ReplaySpec::runs`.
#[derive(Debug, Clone)]
pub struct ReplayReport {
    pub outcomes: Vec<ReplayOutcome>,
}

/// One recorded turn beside its replay: the same turn, sealed twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnComparison {
    /// The turn, counted from 0 over the conversation's turns.
    pub turn: usize,
    pub recorded: SealedTurn,
    /// `None` when the replay sealed no such turn.
    pub replayed: Option<SealedTurn>,
    /// Where the replayed ids first part from the recorded ones; `None` when
    /// they are the same ids.
    pub difference: Option<TokenDifference>,
}

/// The first id at which a replayed turn parts from the recorded one, with the
/// text either side of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDifference {
    /// Its position in the turn's ids.
    pub at: usize,
    /// The recorded ids from [`CONTEXT_BEFORE`] before `at` to
    /// [`CONTEXT_AFTER`] after it, decoded.
    pub recorded: String,
    /// The replayed ids over the same positions, decoded.
    pub replayed: String,
}

impl TurnComparison {
    /// The same ids under the same page widths — a projection borrowing either
    /// turn hands the model the same context.
    pub fn matches(&self) -> bool {
        self.difference.is_none()
            && self
                .replayed
                .as_ref()
                .is_some_and(|t| t.pages == self.recorded.pages)
    }
}

/// Ids a [`TokenDifference`] shows before the first difference.
pub const CONTEXT_BEFORE: usize = 8;
/// Ids a [`TokenDifference`] shows from the first difference on.
pub const CONTEXT_AFTER: usize = 24;

/// The first position at which `a` and `b` differ, a length difference counting
/// as one at the shorter's end; `None` when they are equal.
fn first_difference(a: &[u32], b: &[u32]) -> Option<usize> {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(at) => Some(at),
        None if a.len() == b.len() => None,
        None => Some(a.len().min(b.len())),
    }
}

/// `ids` around position `at`, decoded.
fn context(ids: &[u32], at: usize, decode: &dyn Fn(&[u32]) -> String) -> String {
    let from = at.saturating_sub(CONTEXT_BEFORE).min(ids.len());
    let to = (at + CONTEXT_AFTER).min(ids.len());
    decode(&ids[from..to])
}

/// The recorded turns before `resume_turn`, each beside the replay's turn at the
/// same position, with `decode` rendering the ids around any difference.
pub fn compare_turns(
    recorded: &[SealedTurn],
    replayed: &[SealedTurn],
    resume_turn: usize,
    decode: &dyn Fn(&[u32]) -> String,
) -> Vec<TurnComparison> {
    recorded
        .iter()
        .take(resume_turn)
        .enumerate()
        .map(|(turn, r)| {
            let replayed = replayed.get(turn).cloned();
            let replayed_ids = replayed.as_ref().map_or(&[][..], |t| &t.token_ids[..]);
            let difference =
                first_difference(&r.token_ids, replayed_ids).map(|at| TokenDifference {
                    at,
                    recorded: context(&r.token_ids, at, decode),
                    replayed: context(replayed_ids, at, decode),
                });
            TurnComparison {
                turn,
                recorded: r.clone(),
                replayed,
                difference,
            }
        })
        .collect()
}

/// A recovered history as turns: each a user bubble followed by its assistant
/// bubble.
pub fn recorded_turns(history: &[RecoveredMessage]) -> anyhow::Result<Vec<RecordedTurn>> {
    let mut turns = Vec::new();
    let mut bubbles = history.iter();
    while let Some(user) = bubbles.next() {
        if !matches!(user.role, TurnRole::User) {
            bail!(
                "turn {} opens with a {:?} bubble, not the user's",
                turns.len(),
                user.role
            );
        }
        let Some(assistant) = bubbles.next() else {
            bail!("turn {} has no assistant half", turns.len());
        };
        if !matches!(assistant.role, TurnRole::Assistant) {
            bail!(
                "turn {} answers with a {:?} bubble, not the assistant's",
                turns.len(),
                assistant.role
            );
        }
        turns.push(RecordedTurn {
            user: user.text.clone(),
            assistant: assistant.text.clone(),
        });
    }
    Ok(turns)
}

/// The selection a turn projects with: the reply's own, less the demonstration
/// on a tool round when the chat drops it there.
pub fn turn_selection(
    reply: &SelectionState,
    user: &str,
    demonstration: Demonstration,
) -> SelectionState {
    let tool_round = user.starts_with("<tool_response>");
    match demonstration {
        Demonstration::FirstStepOnly if tool_round => tool_round_selection(reply),
        _ => reply.clone(),
    }
}

impl ZendSession {
    /// Wait until the model has loaded — what a submit waits for before its
    /// first turn.
    pub async fn wait_ready(&self) {
        let mut ready = self.ready_tx.subscribe();
        while !*ready.borrow_and_update() {
            if ready.changed().await.is_err() {
                return;
            }
        }
    }

    /// The recorded conversation `conv_id`, as turns.
    pub fn recorded_conversation(&self, conv_id: &str) -> anyhow::Result<Vec<RecordedTurn>> {
        let history = self
            .conversation_history(conv_id)
            .context("the model is not loaded")?;
        recorded_turns(&history)
    }

    /// Run `spec`: once per entry of `spec.runs`, replay the turns before the
    /// resume point onto a timeline of that run's own and decode the resume turn
    /// there. `on_history` is called with the run, its replayed turns beside the
    /// recorded ones and the wall time the replay took, before that run's
    /// decode; `on_run` with each run as it finishes. Blocking — call it off the
    /// async runtime.
    pub fn replay(
        &self,
        spec: &ReplaySpec,
        on_history: &mut dyn FnMut(usize, &[TurnComparison], f64),
        on_run: &mut dyn FnMut(usize, &ReplayOutcome),
    ) -> anyhow::Result<ReplayReport> {
        let state = self
            .inference
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
            .context("the model is not loaded")?;
        let turns = self.recorded_conversation(&spec.conv_id)?;
        let Some(resume) = turns.get(spec.resume_turn) else {
            bail!(
                "{} has {} turns; there is no turn {} to resume at",
                spec.conv_id,
                turns.len(),
                spec.resume_turn
            );
        };
        let recorded = state
            .base_conv
            .lock()
            .unwrap()
            .sealed_turns(timeline_for(&spec.conv_id));
        let decode = |ids: &[u32]| {
            state
                .tokenizer
                .decode(ids, false)
                .unwrap_or_else(|e| format!("<undecodable: {e}>"))
        };

        let mut outcomes = Vec::with_capacity(spec.runs.len());
        for (run, sampling) in spec.runs.iter().enumerate() {
            // A timeline per run, retired whole when the run ends — see the
            // module notes for why a shared one lets a run read the last one's.
            let replay_id = format!("replay/{}/{}/{run}", spec.conv_id, spec.resume_turn);
            let timeline = timeline_for(&replay_id);
            let mut conv = state
                .base_conv
                .lock()
                .unwrap()
                .fork_resuming(timeline)
                .with_context(|| format!("forking run {run}'s conversation"))?;
            conv.set_projection(turn_projection(&state, &replay_id, None, spec.tools_mode));

            // The history, each turn decoded again under the triggers and
            // sampling config the chat gave it, its recorded ids forced.
            let replay_start = Instant::now();
            for (index, turn) in turns[..spec.resume_turn].iter().enumerate() {
                let Some(sealed) = recorded.get(index) else {
                    bail!("turn {index} has no sealed record to replay");
                };
                let selection = turn_selection(&spec.selection, &turn.user, spec.demonstration);
                let options = TurnOptions {
                    sampling: Some(turn_sampling(
                        &conv,
                        Some(SamplingConfig::argmax()),
                        &selection,
                        &state.think_closer_phrase,
                        0,
                    )),
                    triggers: turn_triggers(&state, think_mode_from_selection(&selection)),
                    selection,
                    recorded_turn: Some(sealed.token_ids.clone()),
                    ..Default::default()
                };
                let handle = conv
                    .submit_turn_with_options(tool_round_text(&turn.user), options)
                    .with_context(|| format!("run {run}: replaying turn {index}"))?;
                let response = handle
                    .wait()
                    .with_context(|| format!("run {run}: replaying turn {index}"))?;
                conv.finish_turn(handle, &response)
                    .with_context(|| format!("run {run}: sealing replayed turn {index}"))?;
            }
            let replay_secs = replay_start.elapsed().as_secs_f64();
            let history = compare_turns(
                &recorded,
                &conv.sealed_turns(timeline),
                spec.resume_turn,
                &decode,
            );
            on_history(run, &history, replay_secs);

            let selection = turn_selection(&spec.selection, &resume.user, spec.demonstration);
            let think_mode = think_mode_from_selection(&selection);
            let (explicit, seed) = match sampling {
                ReplaySampling::Argmax => (Some(SamplingConfig::argmax()), 0),
                ReplaySampling::Seed(seed) => (None, *seed),
            };
            let options = TurnOptions {
                max_tokens: Some(spec.max_tokens),
                sampling: Some(turn_sampling(
                    &conv,
                    explicit,
                    &selection,
                    &state.think_closer_phrase,
                    seed,
                )),
                triggers: turn_triggers(&state, think_mode),
                selection,
                ..Default::default()
            };
            let decode_start = Instant::now();
            let handle = conv
                .submit_turn_with_options(tool_round_text(&resume.user), options)
                .context("submitting the resume turn")?;
            let mut projections = 0;
            let mut last_projection = None;
            let mut finished = None;
            for event in handle.stream() {
                match event {
                    TurnEvent::Projection(p) => {
                        projections += 1;
                        last_projection = Some(p);
                    }
                    TurnEvent::Done(response) => finished = Some(response),
                    TurnEvent::Error(e) => bail!("decoding the resume turn: {e}"),
                    _ => {}
                }
            }
            let response =
                finished.context("the scheduler closed the resume turn without finishing it")?;
            let decode_secs = decode_start.elapsed().as_secs_f64();
            conv.finish_turn(handle, &response)
                .with_context(|| format!("sealing run {run}"))?;
            state
                .engine
                .lock()
                .unwrap()
                .tombstone_timeline(timeline)
                .with_context(|| format!("retiring run {run}'s timeline"))?;
            let outcome = ReplayOutcome {
                turns: history,
                replay_secs,
                sampling: *sampling,
                text: response.text.clone(),
                tokens: response.stats.tokens_generated,
                prefill_ms: response.stats.prefill_ms,
                tokens_per_second: response.stats.tokens_per_second,
                projections,
                last_projection,
                decode_secs,
            };
            on_run(run, &outcome);
            outcomes.push(outcome);
        }
        Ok(ReplayReport { outcomes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::{OptionalState, SealedPages};

    fn sealed(index: u32, ids: &[u32], pages: SealedPages) -> SealedTurn {
        SealedTurn {
            index,
            token_ids: ids.to_vec(),
            pages,
        }
    }

    fn ids(ids: &[u32]) -> String {
        format!("{ids:?}")
    }

    #[test]
    fn ids_part_at_their_first_difference() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2]), Some(2));
        assert_eq!(first_difference(&[], &[7]), Some(0));
    }

    /// Pages cut differently over the same ids are a different context: the
    /// projection pushes each page at its own width.
    #[test]
    fn turns_pair_in_order_up_to_the_resume_point() {
        let widths = |w: &[usize]| SealedPages::Widths(w.to_vec());
        let recorded = [
            sealed(0, &[1, 2], widths(&[2])),
            sealed(1, &[3, 4], widths(&[2])),
            sealed(2, &[5], widths(&[1])),
        ];
        let replayed = [
            sealed(0, &[1, 2], widths(&[2])),
            sealed(1, &[3, 4], widths(&[1, 1])),
        ];
        let turns = compare_turns(&recorded, &replayed, 2, &ids);
        assert_eq!(turns.len(), 2);
        assert!(turns[0].matches());
        assert_eq!(turns[1].difference, None);
        assert!(!turns[1].matches());
    }

    #[test]
    fn a_turn_the_replay_never_sealed_does_not_match() {
        let recorded = [sealed(0, &[1], SealedPages::Absent)];
        let turns = compare_turns(&recorded, &[], 1, &ids);
        assert_eq!(
            turns[0].difference,
            Some(TokenDifference {
                at: 0,
                recorded: "[1]".to_string(),
                replayed: "[]".to_string(),
            })
        );
        assert!(!turns[0].matches());
    }

    /// The context runs from 8 ids before the difference to 24 from it,
    /// clipped to the turn.
    #[test]
    fn a_difference_carries_the_ids_either_side_of_it() {
        let recorded: Vec<u32> = (0..40).collect();
        let mut replayed = recorded.clone();
        replayed[20] = 99;
        let turns = compare_turns(
            &[sealed(0, &recorded, SealedPages::Absent)],
            &[sealed(0, &replayed, SealedPages::Absent)],
            1,
            &ids,
        );
        let mut changed: Vec<u32> = (12..40).collect();
        changed[8] = 99;
        assert_eq!(
            turns[0].difference,
            Some(TokenDifference {
                at: 20,
                recorded: format!("{:?}", (12..40).collect::<Vec<u32>>()),
                replayed: format!("{changed:?}"),
            })
        );
    }

    fn bubble(role: TurnRole, text: &str) -> RecoveredMessage {
        RecoveredMessage {
            role,
            text: text.to_string(),
            no_think: false,
            thinking: None,
            tool_tokens: Vec::new(),
        }
    }

    #[test]
    fn a_history_pairs_into_turns_in_order() {
        let history = [
            bubble(TurnRole::User, "Read the paper"),
            bubble(TurnRole::Assistant, "<tool_call>{}</tool_call>"),
            bubble(TurnRole::User, "<tool_response>x</tool_response>\n"),
            bubble(TurnRole::Assistant, "Done."),
        ];
        assert_eq!(
            recorded_turns(&history).unwrap(),
            vec![
                RecordedTurn {
                    user: "Read the paper".to_string(),
                    assistant: "<tool_call>{}</tool_call>".to_string(),
                },
                RecordedTurn {
                    user: "<tool_response>x</tool_response>\n".to_string(),
                    assistant: "Done.".to_string(),
                },
            ]
        );
    }

    #[test]
    fn a_history_that_does_not_pair_is_refused() {
        let unanswered = [bubble(TurnRole::User, "hi")];
        assert_eq!(
            recorded_turns(&unanswered).unwrap_err().to_string(),
            "turn 0 has no assistant half"
        );
        let out_of_order = [
            bubble(TurnRole::Assistant, "hello"),
            bubble(TurnRole::User, "hi"),
        ];
        assert_eq!(
            recorded_turns(&out_of_order).unwrap_err().to_string(),
            "turn 0 opens with a Assistant bubble, not the user's"
        );
    }

    /// The demonstration leaves a tool round only when the chat drops it there;
    /// a reply's first step keeps it either way.
    #[test]
    fn a_turn_selects_the_demonstration_as_the_chat_does() {
        let mut reply = SelectionState::new();
        reply.set_optional("tool_call_example", OptionalState::Present);
        let demo = |sel: SelectionState| sel.get("tool_call_example").map(str::to_string);
        let round = "<tool_response>x</tool_response>\n";
        assert_eq!(
            demo(turn_selection(&reply, round, Demonstration::FirstStepOnly)),
            Some("absent".to_string())
        );
        assert_eq!(
            demo(turn_selection(&reply, round, Demonstration::EveryTurn)),
            Some("present".to_string())
        );
        assert_eq!(
            demo(turn_selection(
                &reply,
                "Read the paper",
                Demonstration::FirstStepOnly
            )),
            Some("present".to_string())
        );
    }
}
