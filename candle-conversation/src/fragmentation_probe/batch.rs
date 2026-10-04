//! One batch of the forward gate's `StoryRewrite`, driven through the real engine and
//! timed the way the gate times its own rows.
//!
//! # What "the way the gate times it" means
//!
//! The gate prefills every session's system prompt first, outside any clock, then
//! starts the prefill clock, prefills every user turn, and stops it. Decode is a
//! second window from the first generated token's step to the last. Its rows are
//! therefore `user-turn tokens / prefill window` and `tokens after each session's
//! first / decode window`.
//!
//! Through the engine the same two windows are read off the token streams:
//!
//! - every conversation is opened — system prompt prefilled — before the clock starts;
//! - every turn is submitted at once, and the **prefill window** runs from that
//!   instant to the moment the *last* session receives its first token, because the
//!   gate's window also ends only when every session's prompt is in;
//! - the **decode window** runs from there to the moment the last token arrives, and
//!   counts only the tokens that **arrive inside it**. The gate stops its decode clock
//!   when its last token is produced and seals outside both windows, so a turn's seal
//!   is not decode here either;
//! - the **completion window** runs from the same start to the last `Done` — the
//!   decode plus whatever the engine does between a turn's last token and its reply,
//!   the seal above all. The gate has no such window; it is reported so that cost
//!   stays in view rather than being folded into decode or dropped.
//!
//! Anything the engine does between submit and the first token — admission, the turn
//! assembler, the wave scheduling — lands in the prefill window, and anything between
//! decode steps lands in the decode window. That is deliberate: those costs are what
//! separate the engine's rows from the gate's, and a clock that left them out would
//! report the gate's numbers back.
//!
//! **Why only the tokens inside the window.** The gate's windows are disjoint by
//! construction — every prompt is in before any session decodes. The engine's need not
//! be: a session admitted early decodes while a later one is still prefilling, and its
//! tokens arrive before the decode window opens. Counting them against that window
//! credits it with work done on the prefill window's time — measured, 72 tokens charged
//! to the last session's 0.14 s tail, reading 528 t/s when the batch had run one session
//! at a time. Counted by arrival, the two figures converge on the gate's exactly when
//! the engine runs the batch the way the gate does, and that is the property that makes
//! this row a baseline to optimise against.

use std::time::{Duration, Instant};

use candle_transformers::models::batch_test::fixtures;
use candle_transformers::models::batch_test::story_normalize::normalize_story;

use super::profile::StoryGate;
use crate::models::ModelBuilder;
use crate::{
    ConversationEngine, OptionalState, SamplingConfig, SelectionState, SequenceConfig, TurnOptions,
};
use crate::{FinishReason, Sequence, TurnEvent, TurnHandle, TurnResponse, NO_THINK_SELECTOR};

/// One session's token arrivals, as offsets from the moment every turn was submitted.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionClock {
    /// When each generated token arrived, in order. The first marks the session's
    /// prefill as done.
    pub token_times: Vec<Duration>,
    /// When the session's turn completed.
    pub done: Duration,
    /// The formatted user turn the engine prefilled, in tokens.
    pub prefill_tokens: usize,
    /// Tokens the session's context held when its turn ended: the projected
    /// context, the turn's prompt and everything it generated.
    pub context_tokens: usize,
}

impl SessionClock {
    /// When the session's prefill finished: its first token, or — for a turn that
    /// streamed none — its completion.
    pub fn first_token(&self) -> Duration {
        self.token_times.first().copied().unwrap_or(self.done)
    }
}

/// A batch's two windows and what each processed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchTiming {
    pub prefill_tokens: usize,
    pub prefill_s: f64,
    pub decode_tokens: usize,
    pub decode_s: f64,
    /// From the last first token to the last `Done`: the decode window plus the
    /// engine's work between a turn's last token and its reply.
    pub complete_s: f64,
    /// Every token the batch streamed, first tokens included — so the decode
    /// count can be read against what the turns actually produced.
    pub streamed_tokens: usize,
    /// The tokens live across the batch when its turns ended — every session's
    /// whole context, as the gate's peak counts system prompt, user turn and
    /// generation.
    pub peak_tokens: usize,
}

impl BatchTiming {
    /// The gate's windows over a batch's per-session clocks.
    ///
    /// The prefill window closes on the **latest** first token and the decode window
    /// on the **latest** token, because each of the gate's windows closes only when
    /// its whole cohort is through. A decode token is one that arrived after the
    /// prefill window closed — see the module doc for why arrival, not count. Each
    /// session's first token comes off the prefill's logits and arrives at or before
    /// that moment, so — as in the gate — it is never a decode token.
    pub fn from_sessions(sessions: &[SessionClock]) -> Self {
        let last_first = sessions
            .iter()
            .map(SessionClock::first_token)
            .max()
            .unwrap_or_default();
        let last_token = sessions
            .iter()
            .flat_map(|s| s.token_times.iter().copied())
            .max()
            .unwrap_or_default();
        let last_done = sessions.iter().map(|s| s.done).max().unwrap_or_default();
        let decode_tokens = sessions
            .iter()
            .flat_map(|s| s.token_times.iter())
            .filter(|&&t| t > last_first)
            .count();
        Self {
            prefill_tokens: sessions.iter().map(|s| s.prefill_tokens).sum(),
            prefill_s: last_first.as_secs_f64(),
            decode_tokens,
            decode_s: last_token.saturating_sub(last_first).as_secs_f64(),
            complete_s: last_done.saturating_sub(last_first).as_secs_f64(),
            streamed_tokens: sessions.iter().map(|s| s.token_times.len()).sum(),
            peak_tokens: sessions.iter().map(|s| s.context_tokens).sum(),
        }
    }

    pub fn prefill_tps(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill_s.max(1e-9)
    }

    pub fn decode_tps(&self) -> f64 {
        self.decode_tokens as f64 / self.decode_s.max(1e-9)
    }
}

/// A batch's replies, judged, and its timing.
pub struct StoryBatch {
    /// The conversations, still open, so the caller decides how long their KV lives.
    pub conversations: Vec<Sequence>,
    pub timing: BatchTiming,
    /// How long each `submit_turn_with_options` call blocked its caller, in session
    /// order. A submit that waits on the scheduler holds back every turn queued behind
    /// it in the same caller, so this is part of what the prefill window pays.
    pub submit_blocked: Vec<Duration>,
    pub story_pass: usize,
    /// One line per session that failed its [`StoryGate`].
    pub story_fail: Vec<String>,
    /// Turns that errored rather than replied.
    pub errors: Vec<String>,
}

/// The conversation's own sampling, made deterministic.
///
/// Derived from the model's configured sampling rather than built from scratch, so
/// every other lever — the penalties, the banned tokens, the segment behaviour —
/// stays as the model ships it and only the randomness goes. `temperature = 0` is
/// argmax; `top_k = 1` and `top_p = 1` remove the two ways a nucleus could still
/// widen the choice.
fn greedy_sampling(config: SequenceConfig) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        segment_temp_boost: 0.0,
        top_k: 1,
        top_p: 1.0,
        ..config.sampling
    }
}

/// The composer's thinking dial, off.
///
/// `Present` on [`NO_THINK_SELECTOR`] is what the turn assembler reads to emit the
/// dialect's suppression — `/no_think` as live glue for the families that carry the
/// switch in the user turn, a pre-closed block for the rest. The gate suppresses
/// thinking the same way, so its reply begins with the rewrite.
pub(super) fn no_think() -> SelectionState {
    let mut s = SelectionState::default();
    s.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
    s
}

/// The reply with a leading reasoning block removed.
///
/// A suppressed turn still carries the dialect's *framing* — Qwen3 opens a turn
/// with a pre-closed `<think> </think>` — and that is engine structure, not model
/// output: the gate drives the session directly and never sees it. Stripping it is
/// what makes the two comparable; leaving it in fails every session on a prefix the
/// model was never asked to produce.
fn strip_think(reply: &str) -> &str {
    let t = reply.trim_start();
    match t.find("</think>") {
        Some(end) if t.starts_with("<think>") => &t[end + "</think>".len()..],
        _ => reply,
    }
}

/// Judge one reply against its expected rewrite by `gate`; `Err` carries the line
/// to report.
///
/// `StoryGate::Verbatim` is the forward gate's own rule: `normalize_story`, then a
/// common-prefix comparison with a 5-char tolerance, read from the same module so
/// "correct" means here what it means there. `StoryGate::OwnName` asks only that the
/// session names its own protagonist — see [`StoryGate`].
fn judge(i: usize, name: &str, reply: &str, expected: &str, gate: StoryGate) -> Result<(), String> {
    let got = normalize_story(strip_think(reply).trim());
    let want = normalize_story(expected.trim());
    if gate == StoryGate::OwnName {
        if got.contains(name) {
            return Ok(());
        }
        let show = got.chars().take(120).collect::<String>();
        return Err(format!(
            "session {i}: its own protagonist {name:?} is absent from the reply\n      got:  {show:?}",
        ));
    }
    let g: Vec<char> = got.chars().collect();
    let w: Vec<char> = want.chars().collect();
    let common = g.iter().zip(w.iter()).take_while(|(a, b)| a == b).count();
    let min_len = g.len().min(w.len());
    const TOLERANCE: usize = 5;
    // An empty or near-empty decode is a failure, not a vacuous pass: with `min_len`
    // at zero the tolerance would make `required` zero and anything would match.
    let required = min_len.saturating_sub(TOLERANCE);
    if min_len >= 16 && common >= required {
        return Ok(());
    }
    let show = min_len.min(common + 24);
    Err(format!(
        "session {i} (name {name}): matched {common}/{min_len} chars\n      got:  {:?}\n      want: {:?}",
        g[..show.min(g.len())].iter().collect::<String>(),
        w[..show.min(w.len())].iter().collect::<String>(),
    ))
}

/// The line to report when a turn's stream differs from the tokens it generated,
/// naming the first position where they part.
///
/// Every generated token is streamed but one: the end-of-sequence a turn that
/// `stopped` closes on, which is the model ending the turn rather than reply
/// content (`stats::streams_committed`). So a stopped turn may generate exactly
/// one token past its stream, at the end.
fn unstreamed(i: usize, streamed: &[u32], generated: &[u32], stopped: bool) -> Option<String> {
    let closed_by_eos =
        stopped && generated.len() == streamed.len() + 1 && generated.starts_with(streamed);
    if streamed == generated || closed_by_eos {
        return None;
    }
    let at = streamed
        .iter()
        .zip(generated)
        .take_while(|(a, b)| a == b)
        .count();
    Some(format!(
        "session {i}: streamed {} tokens but generated {} — they part at position {at} \
         (streamed {:?}, generated {:?})",
        streamed.len(),
        generated.len(),
        streamed.get(at),
        generated.get(at),
    ))
}

/// One turn as its reader saw it: the reply, every streamed token with its arrival,
/// and when the turn ended.
struct ClockedTurn {
    resp: TurnResponse,
    streamed: Vec<u32>,
    token_times: Vec<Duration>,
    done: Duration,
}

/// Read one turn's events until it settles, stamping every token's arrival and the
/// turn's end against `t0`.
fn clock_turn(handle: &TurnHandle, t0: Instant) -> Result<ClockedTurn, String> {
    let mut streamed = Vec::new();
    let mut token_times = Vec::new();
    for event in handle.stream() {
        match event {
            TurnEvent::Token(id) => {
                streamed.push(id);
                token_times.push(t0.elapsed());
            }
            TurnEvent::Done(resp) => {
                return Ok(ClockedTurn {
                    resp,
                    streamed,
                    token_times,
                    done: t0.elapsed(),
                })
            }
            TurnEvent::Error(e) => return Err(format!("decode: {e}")),
            _ => {}
        }
    }
    Err("decode: the scheduler dropped the turn before it completed".to_string())
}

/// Run `width` sessions of the gate's `StoryRewrite` through `engine`, each decoding
/// `max_tokens` tokens, and time them as the gate times its rows.
///
/// Sessions are named and prompted exactly as the gate names and prompts them — the
/// gate's own system prompt with the name substituted, the gate's story, the gate's
/// session index into its name list — so the rows compare text for text. Each turn is
/// sealed after the clocks stop, the way a live turn is, so the batch's KV is recorded
/// rather than abandoned; the conversations are returned open.
pub fn run_story_batch(
    engine: &ConversationEngine,
    builder: &ModelBuilder,
    width: usize,
    max_tokens: usize,
    gate: StoryGate,
) -> crate::Result<StoryBatch> {
    let story = fixtures::story_prompt();
    let names = fixtures::session_names();
    let config = builder.conversation_config();

    // Opened first, off the clock: opening a conversation prefills its system prompt,
    // which the gate also does before its prefill clock starts.
    let mut conversations: Vec<Sequence> = Vec::with_capacity(width);
    let mut prompts: Vec<String> = Vec::with_capacity(width);
    let mut expected: Vec<String> = Vec::with_capacity(width);
    for i in 0..width {
        let name = &names[i % names.len()];
        // **The gate's own system prompt, with the name substituted the way the gate
        // substitutes it.** The rewrite is entirely a function of this text; under
        // the daemon's own system prompt the model answers helpfully instead, which
        // matches no prefix of the expected rewrite.
        let system = fixtures::system_prompt().replace("{INSERT_NAME}", name);
        conversations.push(engine.new_conversation(&system, config.clone())?);
        prompts.push(fixtures::story_rewrite_prompt(&story, name));
        expected.push(fixtures::story_rewrite_expected(&story, name));
    }

    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(width);
    let mut submit_blocked = Vec::with_capacity(width);
    for (c, prompt) in conversations.iter_mut().zip(prompts.iter()) {
        let opts = TurnOptions {
            max_tokens: Some(max_tokens),
            selection: no_think(),
            // **Greedy, because the check is a verbatim reproduction.** With any
            // temperature a session can diverge honestly and read in the report as
            // the corruption the check exists to find.
            sampling: Some(greedy_sampling(config.clone())),
            ..Default::default()
        };
        let t_submit = Instant::now();
        handles.push(c.submit_turn_with_options(prompt.as_str(), opts)?);
        submit_blocked.push(t_submit.elapsed());
    }

    // One reader per turn, so each stamp is taken the moment its event arrives
    // rather than when a sequential wait happens to reach it.
    let clocked: Vec<Result<ClockedTurn, String>> = std::thread::scope(|s| {
        let readers: Vec<_> = handles
            .iter()
            .map(|h| s.spawn(move || clock_turn(h, t0)))
            .collect();
        readers
            .into_iter()
            .map(|r| {
                r.join()
                    .unwrap_or_else(|_| Err("decode: the reader panicked".to_string()))
            })
            .collect()
    });

    let mut clocks = Vec::with_capacity(width);
    let mut story_pass = 0usize;
    let mut story_fail = Vec::new();
    let mut errors = Vec::new();
    for (i, ((c, h), result)) in conversations
        .iter_mut()
        .zip(handles)
        .zip(clocked)
        .enumerate()
    {
        match result {
            Ok(ClockedTurn {
                resp,
                streamed,
                token_times,
                done,
            }) => {
                // Every token the turn generated reaches its stream, but the EOS
                // that closes it: a client reads the reply off the stream, and
                // the clocks count it there.
                let stopped = resp.stats.finish == FinishReason::Stop;
                if let Some(line) = unstreamed(i, &streamed, resp.token_ids.as_slice(), stopped) {
                    errors.push(line);
                }
                clocks.push(SessionClock {
                    token_times,
                    done,
                    // The turn's own prompt, as the gate counts its user turn:
                    // the system prompt was prefilled off the clock on both sides.
                    prefill_tokens: resp.stats.turn_prefill_tokens,
                    context_tokens: resp.stats.context_tokens,
                });
                let name = &names[i % names.len()];
                match judge(i, name, &resp.text, &expected[i], gate) {
                    Ok(()) => story_pass += 1,
                    Err(line) => story_fail.push(line),
                }
                if let Err(e) = c.finish_turn(h, &resp) {
                    errors.push(format!("finish_turn: {e}"));
                }
            }
            Err(e) => errors.push(e),
        }
    }

    Ok(StoryBatch {
        conversations,
        timing: BatchTiming::from_sessions(&clocks),
        submit_blocked,
        story_pass,
        story_fail,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(token_ms: &[u64], done_ms: u64, prefill: usize) -> SessionClock {
        SessionClock {
            token_times: token_ms.iter().map(|&t| Duration::from_millis(t)).collect(),
            done: Duration::from_millis(done_ms),
            prefill_tokens: prefill,
            // A fresh conversation: its context is its prompt and what it said.
            context_tokens: prefill + token_ms.len(),
        }
    }

    /// **The gate's shape: every prompt in, then every session decodes.** Prefill
    /// closes on the last first token (300 ms), decode on the last token (590 ms),
    /// completion on the last `Done` (600 ms), and each session's first token is
    /// prefill's, not decode's.
    #[test]
    fn a_batch_run_together_reads_as_the_gates_two_windows() {
        let t = BatchTiming::from_sessions(&[
            clock(&[290, 400, 500], 520, 1000),
            clock(&[300, 410, 590], 600, 1000),
            clock(&[295, 420], 450, 1040),
        ]);
        assert_eq!(
            t,
            BatchTiming {
                prefill_tokens: 3040,
                prefill_s: 0.3,
                decode_tokens: 5,
                decode_s: 0.29,
                complete_s: 0.3,
                streamed_tokens: 8,
                peak_tokens: 3048,
            }
        );
        assert_eq!(t.prefill_tps(), 3040.0 / 0.3);
        assert_eq!(t.decode_tps(), 5.0 / 0.29);
    }

    /// **A turn's seal is not decode.** The last token lands at 400 ms and its
    /// `Done` at 900 ms: decode closes on the token, completion on the `Done`.
    #[test]
    fn the_seal_after_the_last_token_is_completion_not_decode() {
        let t = BatchTiming::from_sessions(&[clock(&[100, 250, 400], 900, 500)]);
        assert_eq!(t.decode_s, 0.3);
        assert_eq!(t.complete_s, 0.8);
        assert_eq!(t.decode_tokens, 2);
    }

    /// **A batch run one session at a time is not credited with decode it did on the
    /// prefill window's time.** The first two sessions finish before the third's
    /// prompt is in; only the third's two later tokens fall in the decode window.
    #[test]
    fn tokens_that_arrive_before_the_window_opens_are_not_decode_tokens() {
        let t = BatchTiming::from_sessions(&[
            clock(&[100, 120, 140], 150, 600),
            clock(&[300, 320, 340], 350, 600),
            clock(&[500, 520, 540], 550, 600),
        ]);
        assert_eq!(t.prefill_s, 0.5);
        assert_eq!(t.decode_tokens, 2);
        assert_eq!(t.decode_s, 0.04);
        assert_eq!(t.complete_s, 0.05);
    }

    /// A turn that streamed nothing finished its prefill when it ended, and a batch of
    /// them has an empty decode window rather than a negative one.
    #[test]
    fn a_turn_that_streams_nothing_decodes_nothing() {
        let t = BatchTiming::from_sessions(&[clock(&[250], 250, 800), clock(&[], 240, 800)]);
        assert_eq!(t.prefill_s, 0.25);
        assert_eq!(t.decode_tokens, 0);
        assert_eq!(t.decode_s, 0.0);
        assert_eq!(t.decode_tps(), 0.0);
    }

    /// An empty batch is zero throughout, not a division by zero.
    #[test]
    fn an_empty_batch_is_zero() {
        let t = BatchTiming::from_sessions(&[]);
        assert_eq!(
            t,
            BatchTiming {
                prefill_tokens: 0,
                prefill_s: 0.0,
                decode_tokens: 0,
                decode_s: 0.0,
                complete_s: 0.0,
                streamed_tokens: 0,
                peak_tokens: 0,
            }
        );
        assert_eq!(t.prefill_tps(), 0.0);
    }

    #[test]
    fn a_stream_that_carries_every_generated_token_reports_nothing() {
        assert_eq!(unstreamed(0, &[5, 6, 7], &[5, 6, 7], false), None);
    }

    /// A turn that stopped closes on an EOS it does not stream; that one trailing
    /// token is the only difference allowed, and only on a stop.
    #[test]
    fn the_eos_a_stopped_turn_closes_on_is_the_one_token_not_streamed() {
        assert_eq!(unstreamed(0, &[5, 6], &[5, 6, 2], true), None);
        assert!(
            unstreamed(0, &[5, 6], &[5, 6, 2], false).is_some(),
            "a length-capped turn streams its last token"
        );
        assert!(
            unstreamed(0, &[5], &[5, 6, 2], true).is_some(),
            "only the closing token may be missing"
        );
        assert!(
            unstreamed(0, &[5, 9], &[5, 6, 2], true).is_some(),
            "and the rest must match"
        );
    }

    #[test]
    fn a_generated_token_missing_from_the_stream_is_named_by_position() {
        assert_eq!(
            unstreamed(3, &[5, 6], &[5, 6, 7], false).as_deref(),
            Some(
                "session 3: streamed 2 tokens but generated 3 — they part at position 2 \
                 (streamed None, generated Some(7))"
            )
        );
        assert_eq!(
            unstreamed(1, &[5, 9, 7], &[5, 6, 7], false).as_deref(),
            Some(
                "session 1: streamed 3 tokens but generated 3 — they part at position 1 \
                 (streamed Some(9), generated Some(6))"
            )
        );
    }

    const WANT: &str = "The quick brown fox jumps over the lazy dog";

    #[test]
    fn a_verbatim_prefix_within_tolerance_passes() {
        let got = "The quick brown fox jumps";
        assert_eq!(judge(0, "x", got, WANT, StoryGate::Verbatim), Ok(()));
    }

    #[test]
    fn a_wrong_word_fails_the_verbatim_gate() {
        let got = "The quick brown cat jumps over";
        assert!(judge(0, "x", got, WANT, StoryGate::Verbatim).is_err());
    }

    /// A pre-closed think block is engine framing, not reply.
    #[test]
    fn the_framing_think_block_is_not_part_of_the_reply() {
        let framed = "<think>\n\n</think>\n\nOnce upon";
        assert_eq!(strip_think(framed), "\n\nOnce upon");
        assert_eq!(strip_think("Once upon"), "Once upon");
    }
}
