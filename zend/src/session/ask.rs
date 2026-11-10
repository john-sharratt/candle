//! Ask the loaded model one question against everything the workspace's
//! substrate holds — the harness for the dependency battery of
//! `docs/unbounded_agents.md` §9.12.
//!
//! Each question is a fresh conversation on a timeline of its own: the base
//! conversation is forked onto it, the chat's own projection is set with tools
//! off, so the answer can come only from what the projection retrieves, and
//! the question is decoded once, greedily. The timeline is tombstoned whole
//! when the answer is in, so no question reads another's turn; on a read-only
//! substrate (`DaemonConfig::read_only_substrate`) none of it reaches disk.
//!
//! [`AskSpec::reproject`] is the battery's ablation: with it off the question
//! decodes on the working set its opening projection chose, never re-selected
//! during the decode (`Sequence::set_reprojection(0, [])`) — the one-shot
//! retrieval the battery compares the full system against.

use std::time::Instant;

use anyhow::{bail, Context};
use candle_conversation::{
    ProjectionEvent, SamplingConfig, SelectionState, TurnEvent, TurnOptions,
};

use super::{
    think_mode_from_selection, timeline_for, turn_projection, turn_sampling, turn_triggers,
    ZendSession,
};
use crate::types::ToolMode;

/// One question to put to the model.
#[derive(Debug, Clone)]
pub struct AskSpec {
    /// Names the question's timeline; distinct per question and per arm.
    pub id: String,
    /// The user turn.
    pub question: String,
    /// The decode limit, in tokens.
    pub max_tokens: usize,
    /// The reply's dials, as the chat builds them (`dial_selection`, with
    /// `apply_tools_dial` at [`ToolMode::None`]).
    pub selection: SelectionState,
    /// Re-select the working set during the decode, as the chat does.
    pub reproject: bool,
}

/// What one question produced.
#[derive(Debug, Clone)]
pub struct AskOutcome {
    /// The whole reply, reasoning included.
    pub text: String,
    pub tokens: usize,
    /// Projection points the decode emitted — the opening one and each
    /// reprojection.
    pub projections: usize,
    /// The context the decode finished against, turn by turn.
    pub last_projection: Option<ProjectionEvent>,
    pub decode_secs: f64,
}

impl ZendSession {
    /// Decode `spec`'s question on a fresh timeline, tools off, greedily.
    pub fn ask(&self, spec: &AskSpec) -> anyhow::Result<AskOutcome> {
        let state = self
            .inference
            .load_full()
            .context("the model is not loaded")?;
        let conv_id = format!("ask/{}", spec.id);
        let timeline = timeline_for(&conv_id);
        let mut conv = state
            .base_conv
            .fork_resuming(timeline)
            .with_context(|| format!("forking {conv_id}"))?;
        conv.set_projection(turn_projection(&state, &conv_id, None, ToolMode::None));
        if !spec.reproject {
            conv.set_reprojection(0, Vec::new());
        }
        let think_mode = think_mode_from_selection(&spec.selection);
        let options = TurnOptions {
            max_tokens: Some(spec.max_tokens),
            sampling: Some(turn_sampling(
                &conv,
                Some(SamplingConfig::argmax()),
                &spec.selection,
                &state.think_closer_phrase,
                0,
            )),
            triggers: turn_triggers(&state, think_mode, ToolMode::None),
            selection: spec.selection.clone(),
            ..Default::default()
        };
        let start = Instant::now();
        let handle = conv
            .submit_turn_with_options(spec.question.as_str(), options)
            .with_context(|| format!("submitting {conv_id}"))?;
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
                TurnEvent::Error(e) => bail!("decoding {conv_id}: {e}"),
                _ => {}
            }
        }
        let response = finished
            .with_context(|| format!("the scheduler closed {conv_id} without finishing it"))?;
        let decode_secs = start.elapsed().as_secs_f64();
        conv.finish_turn(handle, &response)
            .with_context(|| format!("sealing {conv_id}"))?;
        state
            .engine
            .tombstone_timeline(timeline)
            .with_context(|| format!("retiring {conv_id}"))?;
        Ok(AskOutcome {
            text: response.text.clone(),
            tokens: response.stats.tokens_generated,
            projections,
            last_projection,
            decode_secs,
        })
    }
}
