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
//! A run writes turns to a timeline of its own — named by the call and the
//! run, so no two runs ever share one — and tombstones it when it ends, on
//! every path out. A timeline that already held turns would resume them and
//! their recurrent snapshot in place of the run's own replay.
//!
//! A run may change the target before it decodes: text spliced into its user
//! half ([`splice`]), and the recurrent rows of one of its blocks captured for
//! a later run of the call, or merged in from an earlier run's capture
//! (`docs/recurrent_reply_calibration.md` §3: rows run through any state).

mod splice;

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use candle_conversation::decode_diag::{self, Arm, Capture, Control, StepRecord};
use candle_conversation::placed_span::PlacedText;
use candle_conversation::projection::{recorded_dials, ProjectionScript, TimelineId, TurnIndex};
use candle_conversation::recurrent_span::{ArmKind, ReplyRows, TextSpan};
use candle_conversation::{SamplingConfig, SelectionState, TurnEvent, TurnOptions};
use serde::{Deserialize, Serialize};

pub use splice::Splice;
use splice::{splice_grid, splice_text};

use super::replay::{
    compare_turns, replay_history, turn_selection, Demonstration, ReplaySampling, TurnComparison,
};
use super::{
    think_mode_from_selection, timeline_for, turn_projection, turn_sampling, turn_triggers,
    Sequence, ZendSession,
};
use crate::tools::{tool_round_blocks, tool_round_text};
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
    /// Turns the recurrent memory is taken over before the target, by
    /// `(timeline, index)`: the control for K/V the attention layers hold and
    /// the recurrent layers never saw.
    pub catch_up: Vec<(u64, u32)>,
    /// The target turn's user half in place of the recorded one. Its recorded
    /// ids then belong to another prompt, so the reply decodes free from its
    /// first token.
    pub user: Option<String>,
    /// Text put into the target's user half before it is submitted, the
    /// recorded reply kept.
    pub splices: Vec<Splice>,
    /// A block of the submitted half whose recurrent rows are kept, under a
    /// name, for the call's later runs. It is the half's last armed span.
    pub capture: Option<NamedBlock>,
    /// Blocks of the submitted half that take an earlier run's capture: its
    /// rows run through the block's entering state in place of the block's.
    pub merge: Vec<NamedBlock>,
    /// Blocks of the submitted half whose bodies are placed from a sealed
    /// turn holding the same text, their edges prefilled
    /// (`docs/inline_read_injection.md`).
    pub placed: Vec<PlacedBlock>,
}

/// A `<tool_response>` block of the submitted half, by its position in the
/// round, whose body is placed from timeline `source` with `edge` grid steps of
/// each end prefilled.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PlacedBlock {
    pub block: usize,
    pub source: u64,
    pub edge: usize,
}

/// The runs `blocks` names in `half`, the user half a run submits.
fn placed_runs(half: &str, blocks: &[PlacedBlock]) -> anyhow::Result<Vec<PlacedText>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let round = tool_round_blocks(half).context("the submitted user half is not a tool round")?;
    blocks
        .iter()
        .map(|p| {
            let block = round.get(p.block).with_context(|| {
                format!(
                    "the round holds {} blocks, so no block {}",
                    round.len(),
                    p.block
                )
            })?;
            Ok(PlacedText {
                bytes: block.body.clone(),
                source: TimelineId::from_raw(p.source)
                    .with_context(|| format!("{} names no timeline", p.source))?,
                edge: p.edge,
            })
        })
        .collect()
}

/// A `<tool_response>` block of the submitted user half, by its position in
/// the round (from 0), and the capture it names.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct NamedBlock {
    pub name: String,
    pub block: usize,
}

/// An earlier turn the replay did not seal as recorded.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistoryDiffers {
    pub turn: usize,
    /// Where its ids first part from the recording, with the text either side
    /// — `None` when only its page cuts differ, which leaves its recurrent
    /// state as recorded.
    pub ids_at: Option<usize>,
    pub recorded: Option<String>,
    pub replayed: Option<String>,
    /// The replay sealed no such turn.
    pub missing: bool,
}

impl HistoryDiffers {
    pub(super) fn of(c: &TurnComparison) -> Self {
        Self {
            turn: c.turn,
            ids_at: c.difference.as_ref().map(|d| d.at),
            recorded: c.difference.as_ref().map(|d| d.recorded.clone()),
            replayed: c.difference.as_ref().map(|d| d.replayed.clone()),
            missing: c.replayed.is_none(),
        }
    }
}

/// A span the run armed, as the report names it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpanArmed {
    pub bytes: Range<usize>,
    /// `capture:<name>` or `merge:<name>`.
    pub kind: String,
}

/// What to decode.
#[derive(Debug, Clone)]
pub struct DiagSpec {
    /// The call's name, which its runs' timelines are named by.
    pub run: String,
    pub conv_id: String,
    /// The target turn, counted from 0 over the conversation's turns.
    pub turn: usize,
    /// The ids of the target's reply forced before sampling starts.
    pub token: usize,
    /// Text forced after those ids, the opening of a sentence for the model to
    /// finish: whether it finishes it from the context is a test of reading
    /// that does not rest on what the model says it can see.
    pub prompt: String,
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
    pub catch_up: Vec<(u64, u32)>,
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
    pub history_differs: Vec<HistoryDiffers>,
    /// The spans of the submitted half the run armed.
    pub spans: Vec<SpanArmed>,
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

/// The spans `run` arms on `half`, the user half it submits, in the half's
/// order: its merges, then its capture.
/// The capture is the half's last span, since an arm replaces the one before
/// it and a capture is handed back as the turn ends.
fn armed_spans(
    run: &DiagRun,
    half: &str,
    captures: &HashMap<String, Arc<ReplyRows>>,
) -> anyhow::Result<(Vec<TextSpan>, Vec<SpanArmed>)> {
    let block = |b: usize| -> anyhow::Result<Range<usize>> {
        let blocks =
            tool_round_blocks(half).context("the submitted user half is not a tool round")?;
        blocks
            .get(b)
            .map(|x| x.block.clone())
            .with_context(|| format!("the round holds {} blocks, so no block {b}", blocks.len()))
    };
    let mut armed: Vec<(TextSpan, String)> = Vec::new();
    for m in &run.merge {
        let rows = captures
            .get(&m.name)
            .with_context(|| format!("no earlier run of the call captured {}", m.name))?;
        armed.push((
            TextSpan {
                bytes: block(m.block)?,
                kind: ArmKind::Replay(Arc::clone(rows)),
            },
            format!("merge:{}", m.name),
        ));
    }
    armed.sort_by_key(|(s, _)| s.bytes.start);
    if let Some(c) = &run.capture {
        let bytes = block(c.block)?;
        if armed.iter().any(|(s, _)| s.bytes.end > bytes.start) {
            bail!(
                "the capture of block {} is not the half's last armed span",
                c.block
            );
        }
        armed.push((
            TextSpan {
                bytes,
                kind: ArmKind::Capture,
            },
            format!("capture:{}", c.name),
        ));
    }
    if armed
        .windows(2)
        .any(|w| w[0].0.bytes.end > w[1].0.bytes.start)
    {
        bail!("two of the run's armed spans overlap");
    }
    Ok(armed
        .into_iter()
        .map(|(s, kind)| {
            let report = SpanArmed {
                bytes: s.bytes.clone(),
                kind,
            };
            (s, report)
        })
        .unzip())
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
        let mut captures = HashMap::new();
        for (i, run) in spec.runs.iter().enumerate() {
            runs.push(
                self.diag_run(spec, i, run, &mut captures)
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
        captures: &mut HashMap<String, Arc<ReplyRows>>,
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
        let Some(target_opening) = events[spec.turn].first() else {
            bail!("turn {} has no recorded projection", spec.turn);
        };

        let diag_id = format!(
            "diag/{}/{}/{}/{run_index}",
            spec.conv_id, spec.turn, spec.run
        );
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
        // The turn's dials are what its recording ran under, whatever the caller
        // set: they steer the think block as well as the projection.
        let mut reply = spec.selection.clone();
        for (selector, option) in recorded_dials(target_opening) {
            reply.select(selector, option);
        }

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
            &reply,
            spec.tools_mode,
            Demonstration::FirstStepOnly,
            &mut |index, conv| {
                let script = script_of(&events[index], conv);
                conv.set_projection(builder.with_script(script).into());
            },
            &mut |_, _, _, _| Ok(()),
        )?;
        let replay_secs = replay_start.elapsed().as_secs_f64();
        let replayed = conv.sealed_turns(timeline);
        if replayed.len() != spec.turn {
            bail!(
                "the run's timeline holds {} turns after replaying the {} before the target",
                replayed.len(),
                spec.turn
            );
        }
        let decode = |ids: &[u32]| state.tokenizer.decode(ids, false).unwrap_or_default();
        let history_differs = compare_turns(&recorded, &replayed, spec.turn, &decode)
            .iter()
            .filter(|t| !t.matches())
            .map(HistoryDiffers::of)
            .collect();

        for &(raw, index) in &run.catch_up {
            let tl = TimelineId::from_raw(raw).context("a catch-up turn's timeline")?;
            let Some(turn) = conv.sealed_turns(tl).into_iter().find(|t| t.index == index) else {
                bail!("no turn {index} on timeline {raw} to catch up over");
            };
            conv.catch_memory_up_over(turn.token_ids)
                .context("catching the memory up")?;
        }

        // The target: its recorded points up to the end, then one live.
        let without: HashSet<(u64, u32)> = run.without.iter().copied().collect();
        let suffix: Vec<u32> = state
            .tokenizer
            .encode(spec.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing the prompt: {e}"))?
            .get_ids()
            .to_vec();
        let from = spec.token + suffix.len();
        let end = from + spec.n;
        let live_at = end.saturating_sub(1).max(1);
        let script = script_of(&events[spec.turn], &conv)
            .ending_live_at(live_at)
            .without(&without);
        let points_recorded = script.len() - 1;
        let dropped_turns = script.dropped_turns;
        conv.set_projection(builder.with_script(script).into());

        let recorded_half = tool_round_text(run.user.as_deref().unwrap_or(&target.user));
        let half = splice_text(&recorded_half, &run.splices)?;
        // The recording replays onto the spliced prefill once it is spliced
        // alike; a run with its own user half has no recording to keep.
        let recorded_turn = match &run.user {
            Some(_) => None,
            None if run.splices.is_empty() => Some(target_sealed.token_ids.clone()),
            None => Some(
                splice_grid(
                    &target_sealed.token_ids,
                    &recorded_half,
                    &run.splices,
                    |t| Ok(Vec::from(conv.encode_text(t)?)),
                )
                .context("splicing the recorded turn")?,
            ),
        };
        let half_text = half.text();
        let (recurrent_spans, spans) = armed_spans(run, &half_text, captures)?;
        let placed = placed_runs(&half_text, &run.placed)?;
        tracing::info!(run = run_index, spans = ?spans, "diag: the target's armed spans");
        let selection = turn_selection(&reply, &target.user, Demonstration::FirstStepOnly);
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
            recorded_turn,
            forced_tokens: Some(spec.token),
            forced_suffix: suffix,
            recurrent_spans,
            placed,
            ..Default::default()
        };
        if run.user.is_some() && from > 0 {
            bail!("a run with its own user half decodes free: it takes no token or prompt");
        }

        if !decode_diag::arm(Arm {
            control: run.control,
            timeline: timeline.raw(),
            from,
            live_at,
        }) {
            bail!("another diagnostic decode holds the capture");
        }
        let armed = Armed;
        let decode_start = Instant::now();
        let handle = conv
            .submit_turn_with_options(half, options)
            .context("submitting the target turn")?;
        let mut points_reached = Vec::new();
        let mut finished = None;
        let mut taken = None;
        for event in handle.stream() {
            match event {
                TurnEvent::Projection(p) => points_reached.push(PointReached {
                    start_token: p.start_token,
                    materialized_tokens: p.materialized_tokens,
                    turns: p.selection.turns.len(),
                    seconds: p.seconds,
                }),
                TurnEvent::RecurrentCapture(rows) => taken = Some(rows),
                TurnEvent::Done(response) => finished = Some(response),
                TurnEvent::Error(e) => bail!("decoding the target: {e}"),
                _ => {}
            }
        }
        let response = finished.context("the scheduler closed the target without finishing it")?;
        if let Some(c) = &run.capture {
            let rows =
                taken.with_context(|| format!("the capture of {} came back empty", c.name))?;
            captures.insert(c.name.clone(), rows);
        }
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
            catch_up: run.catch_up.clone(),
            replay_secs,
            decode_secs,
            prefill_ms: response.stats.prefill_ms,
            tokens_per_second: response.stats.tokens_per_second,
            points_recorded,
            dropped_turns,
            points_reached,
            history_differs,
            spans,
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

    const ROUND: &str = "<tool_response>a</tool_response>\n<tool_response>bb</tool_response>";

    fn run_of(capture: Option<NamedBlock>, merge: Vec<NamedBlock>) -> DiagRun {
        DiagRun {
            control: Control::Sparse,
            sampling: ReplaySampling::Argmax,
            without: Vec::new(),
            catch_up: Vec::new(),
            user: None,
            splices: Vec::new(),
            capture,
            merge,
            placed: Vec::new(),
        }
    }

    fn named(name: &str, block: usize) -> NamedBlock {
        NamedBlock {
            name: name.to_string(),
            block,
        }
    }

    fn captured(name: &str) -> HashMap<String, Arc<ReplyRows>> {
        let rows = ReplyRows {
            layers: vec![0],
            tokens: 1,
            conv_dim: 1,
            n_v_heads: 1,
            data: vec![Vec::new()],
        };
        HashMap::from([(name.to_string(), Arc::new(rows))])
    }

    fn reported(spans: &[SpanArmed]) -> Vec<(Range<usize>, &str)> {
        spans
            .iter()
            .map(|s| (s.bytes.clone(), s.kind.as_str()))
            .collect()
    }

    /// A merge takes its block's bytes and the named rows; the capture comes
    /// last.
    #[test]
    fn a_run_arms_its_merges_then_its_capture_by_block() {
        let run = run_of(Some(named("page", 1)), vec![named("read", 0)]);
        let (spans, report) = armed_spans(&run, ROUND, &captured("read")).unwrap();
        assert_eq!(
            reported(&report),
            [(0..32, "merge:read"), (33..66, "capture:page")]
        );
        assert!(matches!(spans[0].kind, ArmKind::Replay(_)));
        assert!(matches!(spans[1].kind, ArmKind::Capture));
    }

    #[test]
    fn a_capture_before_another_span_is_refused() {
        let run = run_of(Some(named("page", 0)), vec![named("read", 1)]);
        let err = armed_spans(&run, ROUND, &captured("read")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the capture of block 0 is not the half's last armed span"
        );
    }

    #[test]
    fn a_merge_of_nothing_captured_or_of_no_block_is_refused() {
        let unknown = run_of(None, vec![named("read", 0)]);
        assert_eq!(
            armed_spans(&unknown, ROUND, &HashMap::new())
                .unwrap_err()
                .to_string(),
            "no earlier run of the call captured read"
        );
        let far = run_of(None, vec![named("read", 5)]);
        assert_eq!(
            armed_spans(&far, ROUND, &captured("read"))
                .unwrap_err()
                .to_string(),
            "the round holds 2 blocks, so no block 5"
        );
    }

    /// A placed block is its body, between the tags; a block the round does not
    /// hold is refused.
    #[test]
    fn a_placed_block_is_its_body() {
        let placed = |block| PlacedBlock {
            block,
            source: 5,
            edge: 2,
        };
        let runs = placed_runs(ROUND, &[placed(1)]).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(&ROUND[runs[0].bytes.clone()], "bb");
        assert_eq!(runs[0].source.raw(), 5);
        assert_eq!(runs[0].edge, 2);
        assert_eq!(
            placed_runs(ROUND, &[placed(5)]).unwrap_err().to_string(),
            "the round holds 2 blocks, so no block 5"
        );
        assert!(placed_runs("plain", &[]).unwrap().is_empty());
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
