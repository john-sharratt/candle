//! A tool round that repeats the previous round and learns nothing new.
//!
//! A model that has just made a call can copy it on the next round instead of
//! taking the step its own reasoning names. Measured: asked to create
//! `scratch/Temperature.ts`, a turn read the missing file, got `not_found`, and
//! then read it again — forty times — with every reasoning block in between
//! saying "the file doesn't exist yet, so I'll create it with the write tool".
//! Each identical result was more of the pattern it was copying.
//!
//! The guard breaks the pattern where it forms. A call identical to one made
//! earlier in the same turn, whose result is also identical, carried no
//! information, so its result is replaced with a `repeated_call` notice telling
//! the model to act on what it already has. Polling a session (`ssh_session_poll`,
//! `tcp_session_recv`) repeats a call legitimately, and its result changes —
//! so it passes.
//!
//! **Earlier in the turn, not just the round before.** A cycle repeats just as
//! surely as a stutter does, and comparing only against the previous round
//! cannot see one: measured, a turn alternated `git_log` and `git_show` with
//! the same arguments seven times over, every round different from the one
//! before it and every call identical to one two rounds back — 89k tokens of
//! results the conversation already held, and a streak that never started.
//!
//! After [`STOP_AFTER`] such rounds in a row the loop closes:
//! that round's notices say the tools are finished, it is submitted like any
//! other, and the turn that answers it is the last: `<tool_call>` is banned
//! from its sampling, and a call it writes some other way is not run. The
//! conversation keeps a response for every call it ran, and the model gets to
//! say what it has rather than the turn ending mid-call.
//!
//! **That answer is bounded** ([`closing_answer`]). A model that loops on
//! tools does not always stop when it cannot call one: measured twice, the
//! closing turn wrote "Let me revert the commit:" — reached for the banned
//! call — and wrote it again, for 4,700 and 5,800 tokens, until the turn's
//! own limit. What it has to say is a summary of results it already holds.

use candle_conversation::SamplingConfig;
use serde_json::{json, Value};

use crate::tools::{ToolCall, ToolResult};

/// Consecutive rounds made entirely of repeats after which the loop closes —
/// with the first call, three identical rounds in all.
pub const STOP_AFTER: usize = 2;

/// Answer tokens past the think block after which the closing answer ends at
/// the next sentence.
pub const CLOSING_ANSWER_GRACEFUL: i32 = 768;
/// Answer tokens past the think block at which the closing answer ends
/// regardless.
pub const CLOSING_ANSWER_FORCED: i32 = 1024;

/// `sampling` for the turn that answers the closing round: `<tool_call>`
/// (`open`, when resolved) banned, and the answer ended at the first sentence
/// [`CLOSING_ANSWER_GRACEFUL`] tokens past the point the think block is forced
/// shut, and [`CLOSING_ANSWER_FORCED`] past it at the latest — or sooner,
/// where the turn already had a tighter bound. The EOS limits count the whole
/// turn, reasoning included, so the answer's room sits above the thinking
/// cap, as the turn's own budget has it ([`crate::think_budget`]): measured
/// from the start instead, a closing turn that reasoned for 768 tokens would
/// end inside its block with no answer at all.
pub fn closing_answer(sampling: &mut SamplingConfig, open: i32) {
    if open >= 0 {
        sampling.banned_tokens.push(open);
    }
    let thinking = sampling.force_segment_close_after.max(0);
    let bounded = |set: i32, cap: i32| if set > 0 { set.min(cap) } else { cap };
    let graceful = bounded(
        sampling.graceful_eos_after,
        thinking + CLOSING_ANSWER_GRACEFUL,
    );
    sampling.graceful_eos_after = graceful;
    sampling.forced_eos_after =
        bounded(sampling.forced_eos_after, thinking + CLOSING_ANSWER_FORCED);
    // The EOS pressure ramps to the graceful point, as it does on any turn.
    sampling.eos_ramp_len = bounded(sampling.eos_ramp_len, graceful);
}

/// What the loop does after a screened round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Submit the round's results and carry on.
    Continue,
    /// The round was the [`STOP_AFTER`]th repeat in a row. Submit its results
    /// (closing notices) and make the next turn the last: no call it makes
    /// runs.
    Close,
}

/// Every call this turn's tool loop has run, with the result it got.
#[derive(Debug, Default)]
pub struct RepeatGuard {
    seen: Vec<(ToolCall, Value)>,
    streak: usize,
}

impl RepeatGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Screen the round just run: each result whose call and response both
    /// match one made earlier in the turn becomes a `repeated_call` notice.
    ///
    /// A round is compared with the rounds before it, never with itself, so
    /// two identical calls batched into one round both run.
    pub fn screen(&mut self, results: &mut [ToolResult]) -> Verdict {
        let repeated: Vec<bool> = results
            .iter()
            .map(|result| {
                self.seen
                    .iter()
                    .any(|(call, response)| *call == result.call && *response == result.response)
            })
            .collect();
        for (result, repeated) in results.iter().zip(&repeated) {
            if !repeated {
                self.seen
                    .push((result.call.clone(), result.response.clone()));
            }
        }
        let all_repeats = !results.is_empty() && repeated.iter().all(|&r| r);
        self.streak = if all_repeats { self.streak + 1 } else { 0 };
        let closing = self.streak >= STOP_AFTER;
        for (result, repeated) in results.iter_mut().zip(repeated) {
            if closing {
                result.response = closing_notice(&result.call);
            } else if repeated {
                result.response = notice(&result.call);
            }
        }
        if closing {
            Verdict::Close
        } else {
            Verdict::Continue
        }
    }
}

/// The result every call gets in the round that closes the loop.
fn closing_notice(call: &ToolCall) -> Value {
    json!({
        "error": "repeated_call",
        "detail": format!(
            "This {} call repeats earlier rounds exactly, with the same result each time, \
             so the tools are finished for this request: no further tool call will run. \
             Answer the user now from what the earlier results showed, and say plainly \
             what could not be done.",
            call.name
        ),
    })
}

/// The result a repeated call gets in place of the same answer again.
fn notice(call: &ToolCall) -> Value {
    json!({
        "error": "repeated_call",
        "detail": format!(
            "This {} call is exactly one made earlier in this turn, and its result would \
             be exactly the one shown there. Repeating it cannot change anything. Act on \
             that result instead: take the next step your reasoning names — if a file you \
             need does not exist, create it with `write` — or answer the user.",
            call.name
        ),
    })
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::ThinkMode;

    use super::*;
    use crate::think_budget::steer;

    fn result(name: &str, args: Value, response: Value) -> ToolResult {
        ToolResult {
            call: ToolCall {
                name: name.to_string(),
                arguments: args,
            },
            response,
        }
    }

    /// **The closing answer bans the call and is bounded above the think
    /// block** — on a turn the balanced dial programmed, the answer keeps its
    /// room past the 3072-token thinking cap — and a tighter bound the turn
    /// already had stands.
    #[test]
    fn the_closing_answer_is_banned_from_calling_and_bounded() {
        let mut balanced = SamplingConfig::default();
        steer(&mut balanced, ThinkMode::Balanced, 7168, &[]);
        closing_answer(&mut balanced, 42);
        assert!(balanced.banned_tokens.contains(&42));
        assert_eq!(
            (balanced.graceful_eos_after, balanced.forced_eos_after),
            (3072 + CLOSING_ANSWER_GRACEFUL, 3072 + CLOSING_ANSWER_FORCED)
        );
        assert_eq!(balanced.eos_ramp_len, balanced.graceful_eos_after);

        let mut unsteered = SamplingConfig::default();
        closing_answer(&mut unsteered, -1);
        assert!(!unsteered.banned_tokens.contains(&-1));
        assert_eq!(
            (unsteered.graceful_eos_after, unsteered.forced_eos_after),
            (CLOSING_ANSWER_GRACEFUL, CLOSING_ANSWER_FORCED)
        );

        let mut tight = SamplingConfig {
            graceful_eos_after: 100,
            forced_eos_after: 5000,
            ..SamplingConfig::default()
        };
        closing_answer(&mut tight, -1);
        assert_eq!(
            (tight.graceful_eos_after, tight.forced_eos_after),
            (100, CLOSING_ANSWER_FORCED)
        );
    }

    fn missing() -> ToolResult {
        result(
            "file_read",
            json!({"path": "scratch/T.ts"}),
            json!({"error": "not_found", "detail": "file not found: scratch/T.ts"}),
        )
    }

    /// **The same call with the same answer is a repeat; the loop closes after
    /// [`STOP_AFTER`] in a row**, and the closing round says no call will run.
    #[test]
    fn an_identical_round_is_answered_with_a_notice_and_closes_the_loop() {
        let mut guard = RepeatGuard::new();
        let mut first = vec![missing()];
        assert_eq!(guard.screen(&mut first), Verdict::Continue);
        assert_eq!(
            first[0].response["error"], "not_found",
            "a first call is itself"
        );
        for round in 1..=STOP_AFTER {
            let mut again = vec![missing()];
            let verdict = guard.screen(&mut again);
            assert_eq!(again[0].response["error"], "repeated_call", "round {round}");
            let detail = again[0].response["detail"].as_str().unwrap();
            if round == STOP_AFTER {
                assert_eq!(verdict, Verdict::Close, "round {round}");
                assert!(detail.contains("no further tool call will run"), "{detail}");
            } else {
                assert_eq!(verdict, Verdict::Continue, "round {round}");
                assert!(detail.contains("create it with `write`"), "{detail}");
            }
        }
    }

    /// **A poll whose answer changes is not a repeat.**
    #[test]
    fn a_repeated_call_with_a_new_result_passes() {
        let mut guard = RepeatGuard::new();
        let poll = |out: &str| {
            result(
                "ssh_session_poll",
                json!({"session_id": "s", "process_id": "p"}),
                json!({"stdout": out, "running": true}),
            )
        };
        for out in ["a", "ab", "abc", "abcd"] {
            let mut round = vec![poll(out)];
            assert_eq!(guard.screen(&mut round), Verdict::Continue);
            assert_eq!(round[0].response["stdout"], out);
        }
    }

    /// **A cycle is a repeat.** Two calls taking turns differ from the round
    /// before every time, which is how a turn alternated `git_log` and
    /// `git_show` seven times over with the guard never firing. Each call is
    /// the same as one two rounds back, so from the third round on each is a
    /// notice, and the loop closes like any other run of repeats.
    #[test]
    fn two_calls_taking_turns_are_caught_and_close_the_loop() {
        let mut guard = RepeatGuard::new();
        let log = || {
            result(
                "git_log",
                json!({"repo": "candle", "page": 0}),
                json!({"count": 2000, "commits": []}),
            )
        };
        let show = || {
            result(
                "git_show",
                json!({"repo": "candle", "what": "changes", "page": 0}),
                json!({"subject": "s", "changes": []}),
            )
        };
        let mut first = vec![log()];
        assert_eq!(guard.screen(&mut first), Verdict::Continue);
        let mut second = vec![show()];
        assert_eq!(guard.screen(&mut second), Verdict::Continue);
        assert_eq!(second[0].response["subject"], "s", "a new call is itself");

        let mut third = vec![log()];
        assert_eq!(guard.screen(&mut third), Verdict::Continue);
        assert_eq!(third[0].response["error"], "repeated_call", "{:?}", third);
        let detail = third[0].response["detail"].as_str().unwrap();
        assert!(detail.contains("earlier in this turn"), "{detail}");

        let mut fourth = vec![show()];
        assert_eq!(guard.screen(&mut fourth), Verdict::Close);
        let detail = fourth[0].response["detail"].as_str().unwrap();
        assert!(detail.contains("no further tool call will run"), "{detail}");
    }

    /// Two identical calls batched into ONE round are not measured against
    /// each other — only against the rounds before — so both run.
    #[test]
    fn identical_calls_within_one_round_both_run() {
        let mut guard = RepeatGuard::new();
        let mut round = vec![missing(), missing()];
        assert_eq!(guard.screen(&mut round), Verdict::Continue);
        assert_eq!(round[0].response["error"], "not_found");
        assert_eq!(round[1].response["error"], "not_found");
    }

    /// A round that repeats one call but also does something new keeps the
    /// loop going, and progress resets the streak.
    #[test]
    fn progress_resets_the_streak() {
        let mut guard = RepeatGuard::new();
        let write = result(
            "write",
            json!({"path": "scratch/T.ts", "content": "x"}),
            json!({"path": "scratch/T.ts", "created": true}),
        );
        guard.screen(&mut [missing()]);
        let mut mixed = vec![missing(), write];
        assert_eq!(guard.screen(&mut mixed), Verdict::Continue);
        assert_eq!(mixed[0].response["error"], "repeated_call");
        assert_eq!(mixed[1].response["created"], true);
        assert_eq!(guard.streak, 0);
    }
}
