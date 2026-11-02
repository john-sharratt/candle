//! The capped diagnostic decode: `n` tokens from token `token` of a recorded
//! turn, with everything about that decode written down.
//!
//! Each run rebuilds the conversation on a timeline of its own, as
//! [`ZendSession::replay`] does, but **under its recorded projections**
//! ([`ProjectionScript`]) rather than choices made again over a substrate that
//! has moved on. The target turn then decodes its recorded ids up to `token`
//! and samples from there, alone on the card, while the capture
//! (`decode_diag`) records every step's logits and every attention layer's
//! selection, mapped onto the pieces of the slot. It ends on a live
//! reprojection, whose cost and whose choice are both in the report.
//!
//! Repeating a run is the determinism check: two runs of one control and seed
//! are the same decode exactly when every step's logit digest agrees. The
//! controls ask the question a repeat cannot: the same decode with every cell
//! attended, with every piece placed afresh, or without given turns.
//!
//! A run writes turns to its own timeline and tombstones that timeline when it
//! ends, on every path out.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{bail, Context};
use candle_conversation::decode_diag::{self, Arm, Capture, Control, StepRecord};
use candle_conversation::projection::{ProjectionScript, TimelineId, TurnIndex};
use candle_conversation::{SamplingConfig, SelectionState, TurnEvent, TurnOptions};
use serde::Serialize;

use super::replay::{compare_turns, replay_history, turn_selection, Demonstration, ReplaySampling};
use super::{
    think_mode_from_selection, timeline_for, turn_projection, turn_sampling, turn_triggers,
    Sequence, ZendSession,
};
use crate::tools::tool_round_text;
use crate::types::ToolMode;

/// The most tokens one run may decode free: each records every attention
/// layer, and the card serves nothing else meanwhile.
pub const MAX_TOKENS: usize = 256;

/// The most runs one call may ask for.
pub const MAX_RUNS: usize = 6;

/// One decode of the target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagRun {
    pub control: Control,
    pub sampling: ReplaySampling,
    /// Turns taken out of the target's projections, by `(timeline, index)`.
    pub without: Vec<(u64, u32)>,
}

/// What to decode.
#[derive(Debug, Clone)]
pub struct DiagSpec {
    pub conv_id: String,
    /// The target turn, counted from 0 over the conversation's turns.
    pub turn: usize,
    /// The ids of the target's reply forced before sampling starts.
    pub token: usize,
    /// Tokens to decode from there.
    pub n: usize,
    /// The turn's dials, as the chat builds them.
    pub selection: SelectionState,
    pub tools_mode: ToolMode,
    /// The identity the conversation runs as.
    pub identity: Option<String>,
    pub runs: Vec<DiagRun>,
}

/// One projection point as the run reached it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PointReached {
    pub start_token: u32,
    pub materialized_tokens: u32,
    pub turns: usize,
    pub seconds: f64,
}

/// One piece of the context the decode finished on.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContextPiece {
    pub label: String,
    pub start: usize,
    pub end: usize,
    /// What the piece holds, when it is a turn or a section.
    pub text: Option<String>,
}

/// One layer of the recurrent memory.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecurrentLayer {
    pub layer: u32,
    pub state_bytes: usize,
    pub tail_bytes: usize,
    pub digest: u64,
}

/// One finished run.
#[derive(Debug, Clone, Serialize)]
pub struct DiagOutcome {
    pub control: Control,
    pub seed: Option<u64>,
    pub without: Vec<(u64, u32)>,
    pub replay_secs: f64,
    pub decode_secs: f64,
    pub prefill_ms: f64,
    pub tokens_per_second: f64,
    /// The recorded projection points of the target before the free decode.
    pub points_recorded: usize,
    /// Those a recorded turn could not be placed for.
    pub dropped_turns: usize,
    pub points_reached: Vec<PointReached>,
    /// Earlier turns whose replayed ids or pages differ from the recording.
    pub history_differs: Vec<usize>,
    /// The text sampled from `token` on.
    pub text: String,
    pub context: Vec<ContextPiece>,
    pub recurrent: Vec<RecurrentLayer>,
    pub capture: Capture,
}

/// The first step at which two runs parted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Divergence {
    pub run_a: usize,
    pub run_b: usize,
    /// `None` when every step agreed.
    pub step: Option<u64>,
}

/// A whole call.
#[derive(Debug, Clone, Serialize)]
pub struct DiagReport {
    pub conv_id: String,
    pub turn: usize,
    pub token: usize,
    pub n: usize,
    pub runs: Vec<DiagOutcome>,
    /// Each pair of runs of one control, seed and turn set.
    pub determinism: Vec<Divergence>,
}

/// The first step whose digest differs, a shorter run counting as different
/// where it ends.
pub fn first_divergence(a: &[StepRecord], b: &[StepRecord]) -> Option<u64> {
    match a
        .iter()
        .zip(b)
        .find(|(x, y)| x.step != y.step || x.logits_hash != y.logits_hash)
    {
        Some((x, _)) => Some(x.step),
        None if a.len() == b.len() => None,
        None => a.get(b.len()).or(b.get(a.len())).map(|s| s.step),
    }
}

/// FNV-1a over `bytes`.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The turn a piece's label names.
fn piece_turn(label: &str) -> Option<(u64, u32)> {
    let (timeline, index) = label.strip_prefix("turn:")?.split_once(':')?;
    Some((timeline.parse().ok()?, index.parse().ok()?))
}

/// Disarms on every way out of a run.
struct Armed;

impl Drop for Armed {
    fn drop(&mut self) {
        decode_diag::disarm();
    }
}

/// Retires a run's timeline on every way out of it.
struct Retire<'a> {
    engine: &'a candle_conversation::ConversationEngine,
    timeline: TimelineId,
}

impl Drop for Retire<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.engine.tombstone_timeline(self.timeline) {
            tracing::warn!(timeline = self.timeline.raw(), "diag: retiring failed: {e}");
        }
    }
}

impl ZendSession {
    /// Run `spec`. Blocking, and for as long as a target turn decodes the card
    /// serves nothing else — call it off the async runtime, on purpose.
    pub fn diag_decode(&self, spec: &DiagSpec) -> anyhow::Result<DiagReport> {
        if spec.n == 0 || spec.n > MAX_TOKENS {
            bail!("n must be 1..={MAX_TOKENS}, not {}", spec.n);
        }
        if spec.runs.is_empty() || spec.runs.len() > MAX_RUNS {
            bail!("runs must be 1..={MAX_RUNS}, not {}", spec.runs.len());
        }
        let mut runs = Vec::with_capacity(spec.runs.len());
        for (i, run) in spec.runs.iter().enumerate() {
            runs.push(
                self.diag_run(spec, i, run)
                    .with_context(|| format!("run {i}"))?,
            );
        }
        let mut determinism = Vec::new();
        for a in 0..runs.len() {
            for b in a + 1..runs.len() {
                if spec.runs[a] == spec.runs[b] {
                    determinism.push(Divergence {
                        run_a: a,
                        run_b: b,
                        step: first_divergence(&runs[a].capture.steps, &runs[b].capture.steps),
                    });
                }
            }
        }
        Ok(DiagReport {
            conv_id: spec.conv_id.clone(),
            turn: spec.turn,
            token: spec.token,
            n: spec.n,
            runs,
            determinism,
        })
    }

    fn diag_run(
        &self,
        spec: &DiagSpec,
        run_index: usize,
        run: &DiagRun,
    ) -> anyhow::Result<DiagOutcome> {
        let state = self
            .inference
            .load_full()
            .context("the model is not loaded")?;
        let turns = self.recorded_conversation(&spec.conv_id)?;
        let Some(target) = turns.get(spec.turn) else {
            bail!("{} has {} turns", spec.conv_id, turns.len());
        };
        let recorded_timeline = timeline_for(&spec.conv_id);
        let recorded = state.base_conv.sealed_turns(recorded_timeline);
        let events = state
            .base_conv
            .recovered_projection_events(recorded_timeline);
        if events.len() < turns.len() {
            bail!(
                "{} holds projections for {} of {} turns",
                spec.conv_id,
                events.len(),
                turns.len()
            );
        }
        let Some(target_sealed) = recorded.get(spec.turn) else {
            bail!("turn {} has no sealed record", spec.turn);
        };

        let diag_id = format!("diag/{}/{}/{run_index}", spec.conv_id, spec.turn);
        let timeline = timeline_for(&diag_id);
        let mut conv = state
            .base_conv
            .fork_resuming(timeline)
            .context("forking the run's conversation")?;
        let _retire = Retire {
            engine: &state.engine,
            timeline,
        };
        let builder = turn_projection(&state, &diag_id, spec.identity.as_deref(), spec.tools_mode);

        // Where each recorded turn of this conversation stands on the run's
        // timeline: a turn's index is minted as it seals.
        let recorded_raw = recorded_timeline.raw();
        let script_of = |events, conv: &Sequence| {
            let own: HashMap<u32, TurnIndex> = recorded
                .iter()
                .zip(conv.sealed_turns(timeline))
                .map(|(r, t)| (r.index, TurnIndex(t.index)))
                .collect();
            ProjectionScript::from_events(events, builder.schema(), &|tl, index| {
                if tl == recorded_raw {
                    own.get(&index).map(|&i| (timeline, i))
                } else {
                    TimelineId::from_raw(tl).map(|t| (t, TurnIndex(index)))
                }
            })
        };

        let replay_start = Instant::now();
        replay_history(
            &state,
            &mut conv,
            &turns[..spec.turn],
            &recorded,
            &spec.selection,
            spec.tools_mode,
            Demonstration::FirstStepOnly,
            &mut |index, conv| {
                let script = script_of(&events[index], conv);
                conv.set_projection(builder.with_script(script).into());
            },
        )?;
        let replay_secs = replay_start.elapsed().as_secs_f64();
        let history_differs = compare_turns(
            &recorded,
            &conv.sealed_turns(timeline),
            spec.turn,
            &|ids: &[u32]| format!("{ids:?}"),
        )
        .iter()
        .filter(|t| !t.matches())
        .map(|t| t.turn)
        .collect();

        // The target: its recorded points up to the end, then one live.
        let without: HashSet<(u64, u32)> = run.without.iter().copied().collect();
        let end = spec.token + spec.n;
        let script = script_of(&events[spec.turn], &conv)
            .ending_live_at(end.saturating_sub(1).max(1))
            .without(&without);
        let points_recorded = script.len() - 1;
        let dropped_turns = script.dropped_turns;
        conv.set_projection(builder.with_script(script).into());

        let selection = turn_selection(&spec.selection, &target.user, Demonstration::FirstStepOnly);
        let think_mode = think_mode_from_selection(&selection);
        let (explicit, seed) = match run.sampling {
            ReplaySampling::Argmax => (Some(SamplingConfig::argmax()), None),
            ReplaySampling::Seed(seed) => (None, Some(seed)),
        };
        let options = TurnOptions {
            max_tokens: Some(end),
            sampling: Some(turn_sampling(
                &conv,
                explicit,
                &selection,
                &state.think_closer_phrase,
                seed.unwrap_or(0),
            )),
            triggers: turn_triggers(&state, think_mode, spec.tools_mode),
            selection,
            recorded_turn: Some(target_sealed.token_ids.clone()),
            forced_tokens: Some(spec.token),
            ..Default::default()
        };

        if !decode_diag::arm(Arm {
            control: run.control,
            timeline: timeline.raw(),
            from: spec.token,
        }) {
            bail!("another diagnostic decode holds the capture");
        }
        let armed = Armed;
        let decode_start = Instant::now();
        let handle = conv
            .submit_turn_with_options(tool_round_text(&target.user), options)
            .context("submitting the target turn")?;
        let mut points_reached = Vec::new();
        let mut finished = None;
        for event in handle.stream() {
            match event {
                TurnEvent::Projection(p) => points_reached.push(PointReached {
                    start_token: p.start_token,
                    materialized_tokens: p.materialized_tokens,
                    turns: p.selection.turns.len(),
                    seconds: p.seconds,
                }),
                TurnEvent::Done(response) => finished = Some(response),
                TurnEvent::Error(e) => bail!("decoding the target: {e}"),
                _ => {}
            }
        }
        let response = finished.context("the scheduler closed the target without finishing it")?;
        let decode_secs = decode_start.elapsed().as_secs_f64();
        let capture = decode_diag::disarm();
        drop(armed);

        let recurrent = conv
            .memory()
            .context("reading the recurrent memory")?
            .unwrap_or_default()
            .iter()
            .map(|l| RecurrentLayer {
                layer: l.layer_index,
                state_bytes: l.state.len(),
                tail_bytes: l.conv_tail.len(),
                digest: fnv(&l.state) ^ fnv(&l.conv_tail).rotate_left(1),
            })
            .collect();
        conv.finish_turn(handle, &response)
            .context("sealing the target")?;

        let context = capture
            .pieces
            .iter()
            .map(|p| ContextPiece {
                label: p.label.clone(),
                start: p.start,
                end: p.end,
                text: piece_turn(&p.label).and_then(|(tl, index)| {
                    self.resolve_turn_full_text(TimelineId::from_raw(tl)?, index)
                }),
            })
            .collect();
        Ok(DiagOutcome {
            control: run.control,
            seed,
            without: run.without.clone(),
            replay_secs,
            decode_secs,
            prefill_ms: response.stats.prefill_ms,
            tokens_per_second: response.stats.tokens_per_second,
            points_recorded,
            dropped_turns,
            points_reached,
            history_differs,
            text: response.text.clone(),
            context,
            recurrent,
            capture,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(step: u64, logits_hash: u64) -> StepRecord {
        StepRecord {
            step,
            rows: 1,
            sampled: 0,
            top: Vec::new(),
            logits_hash,
        }
    }

    #[test]
    fn runs_part_at_their_first_different_digest() {
        let a = [step(5, 1), step(6, 2), step(7, 3)];
        assert_eq!(first_divergence(&a, &a), None);
        assert_eq!(
            first_divergence(&a, &[step(5, 1), step(6, 9), step(7, 3)]),
            Some(6)
        );
    }

    #[test]
    fn a_run_that_stopped_early_parts_where_it_stopped() {
        let a = [step(5, 1), step(6, 2)];
        assert_eq!(first_divergence(&a, &a[..1]), Some(6));
        assert_eq!(first_divergence(&a[..1], &a), Some(6));
    }

    #[test]
    fn a_piece_names_its_turn() {
        assert_eq!(piece_turn("turn:12:34"), Some((12, 34)));
        assert_eq!(piece_turn("section:3"), None);
        assert_eq!(piece_turn("glue"), None);
    }

    /// FNV-1a's published value for `a`.
    #[test]
    fn the_digest_is_fnv() {
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
