//! A tool round that repeats the previous round and learns nothing new.
//!
//! A model that has just made a call can copy it on the next round instead of
//! taking the step its own reasoning names. Measured: asked to create
//! `scratch/Temperature.ts`, a turn read the missing file, got `not_found`, and
//! then read it again — forty times — with every reasoning block in between
//! saying "the file doesn't exist yet, so I'll create it with the write tool".
//! Each identical result was more of the pattern it was copying.
//!
//! The guard breaks the pattern where it forms. A call identical to one the
//! previous round made, whose result is also identical, carried no
//! information, so its result is replaced with a `repeated_call` notice telling
//! the model to act on what it already has. Polling a session (`ssh_session_poll`,
//! `tcp_session_recv`) repeats a call legitimately, and its result changes —
//! so it passes. After [`STOP_AFTER`] such rounds in a row the loop closes:
//! that round's notices say the tools are finished, it is submitted like any
//! other, and the turn that answers it is the last: `<tool_call>` is banned
//! from its sampling, and a call it writes some other way is not run. The
//! conversation keeps a response for every call it ran, and the model gets to
//! say what it has rather than the turn ending mid-call.

use serde_json::{json, Value};

use crate::tools::{ToolCall, ToolResult};

/// Consecutive rounds made entirely of repeats after which the loop closes —
/// with the first call, three identical rounds in all.
pub const STOP_AFTER: usize = 2;

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

/// The previous round's calls and results, for one turn's tool loop.
#[derive(Debug, Default)]
pub struct RepeatGuard {
    last: Vec<(ToolCall, Value)>,
    streak: usize,
}

impl RepeatGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Screen the round just run: each result whose call and response both
    /// match one from the previous round becomes a `repeated_call` notice.
    pub fn screen(&mut self, results: &mut [ToolResult]) -> Verdict {
        let previous = std::mem::replace(
            &mut self.last,
            results
                .iter()
                .map(|r| (r.call.clone(), r.response.clone()))
                .collect(),
        );
        let repeated: Vec<bool> = results
            .iter()
            .map(|result| {
                previous
                    .iter()
                    .any(|(call, response)| *call == result.call && *response == result.response)
            })
            .collect();
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
            "This {} call repeats the previous rounds exactly, with the same result each \
             time, so the tools are finished for this request: no further tool call will \
             run. Answer the user now from what the earlier results showed, and say \
             plainly what could not be done.",
            call.name
        ),
    })
}

/// The result a repeated call gets in place of the same answer again.
fn notice(call: &ToolCall) -> Value {
    json!({
        "error": "repeated_call",
        "detail": format!(
            "This {} call is exactly the one made in the previous round, and its result \
             would be exactly the one shown there. Repeating it cannot change anything. \
             Act on that result instead: take the next step your reasoning names — if a \
             file you need does not exist, create it with `write` — or answer the user.",
            call.name
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(name: &str, args: Value, response: Value) -> ToolResult {
        ToolResult {
            call: ToolCall {
                name: name.to_string(),
                arguments: args,
            },
            response,
        }
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
