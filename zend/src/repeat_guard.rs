//! A tool round that repeats the previous round and learns nothing new.
//!
//! A model that has just made a call can copy it on the next round instead of
//! taking the step its own reasoning names. Measured: asked to create
//! `scratch/Temperature.ts`, a turn read the missing file, got `not_found`, and
//! then read it again — forty times — with every reasoning block in between
//! saying "the file doesn't exist yet, so I'll create it with the write tool".
//! Each identical result was more of the pattern it was copying.
//!
//! The guard breaks the pattern where it forms. A call made for the
//! [`NOTICE_FROM`]th time in the same turn, with the same arguments and the same
//! result each time, carries no information, so its result is replaced with a
//! `repeated_call` notice telling the model to act on what it already has.
//! Polling a session (`ssh_session_poll`, `tcp_session_recv`) repeats a call
//! legitimately, and its result changes — so it passes.
//!
//! **A few repeats are allowed.** The notice used to answer the second
//! occurrence, and a model that re-reads to check its own work — or simply
//! re-lists a folder it has lost track of — was told it could change nothing.
//! Two occurrences pass as they are.
//!
//! **Then it warns, and only then refuses.** From [`WARN_FROM`] a repeat still
//! gets its real result, with a `warning` saying how many it has left, so a
//! model has room to change course on what it sees. Refused outright on the
//! fourth, a worker that had finished half a job lost every tool on the fifth
//! round of a mistake it never heard about.
//!
//! **One refused call leaves the others open.** A refusal stops that call; only
//! rounds made of nothing but refusals close the loop, and each says so first,
//! so finished work can be saved. Closing on the second, a worker went out with
//! seven finished files it never got to commit (2026-11-01).
//!
//! **The same failure is a pattern even when the call differs.** A model trying
//! variations of a call that cannot work repeats nothing exactly, so nothing
//! above sees it: thirteen commands in one turn were refused for one reason.
//! From [`FAILURES_WARN_FROM`] in a row of one tool's same error, its result
//! carries a `warning`. Nothing is refused for it: a different argument may be
//! the one that works.
//!
//! **A change starts the count again.** A call that mutates — writes a file,
//! changes a repository, runs a program or code ([`mutates`]) — is never a
//! repeat, and it clears every count: after it, reading the same file again
//! may well read something new. Clearing everything rather than only what the
//! change touched keeps the rule exact without knowing what each command
//! touches, and costs at most three more reads of something unchanged.
//!
//! **A paged result says how to get the next page.** The commonest repeat is a
//! model that wants the next page and sends the same call again without
//! `page`; a notice that only said "act on that result" sent it back to the
//! page it already had.
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

use std::collections::HashMap;

use candle_conversation::SamplingConfig;
use serde_json::{json, Value};
use zend_tools::{registry, Capability};

use crate::tools::{ToolCall, ToolResult};

/// The occurrence of an identical call, with an identical result, from which
/// its result carries a warning.
pub const WARN_FROM: usize = 3;

/// The occurrence of an identical call, with an identical result, that is
/// answered with a notice instead: those before it run as themselves.
pub const NOTICE_FROM: usize = 6;

/// Consecutive rounds made entirely of repeats after which the loop closes.
pub const STOP_AFTER: usize = 3;

/// The failure in a row of one tool's one error from which its result carries
/// a warning.
pub const FAILURES_WARN_FROM: usize = 3;

/// The file tools that change the session's files. They write the session
/// overlay rather than the disk, so they declare no capability, and nothing
/// else about them says they change anything.
const OVERLAY_WRITERS: [&str; 6] = [
    "write",
    "file_edit",
    "file_delete",
    "file_move",
    "file_image",
    "journal_move",
];

/// Whether `call` changes anything: writes the session's files or the disk (a
/// repository, a database), or runs a program or code, which can change
/// anything. Such a call is never a repeat, and it clears the guard's counts.
pub fn mutates(call: &ToolCall) -> bool {
    registry::find(&call.name).is_some_and(|tool| {
        OVERLAY_WRITERS.contains(&tool.name)
            || tool.requires.iter().any(|c| {
                matches!(
                    c,
                    Capability::DiskWrite | Capability::Exec | Capability::Sandbox
                )
            })
    })
}

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

/// One call the turn has made, with the result it got, and how many times.
#[derive(Debug)]
struct Seen {
    call: ToolCall,
    response: Value,
    count: usize,
}

/// Every call this turn's tool loop has run since its last mutation, with the
/// result it got and how often.
#[derive(Debug, Default)]
pub struct RepeatGuard {
    /// Each tool's run of one error: the error and how many in a row.
    failing: HashMap<String, (String, usize)>,
    seen: Vec<Seen>,
    streak: usize,
}

impl RepeatGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Screen the round just run: each result whose call and response both
    /// match one made [`NOTICE_FROM`]` − 1` times already since the last
    /// mutation becomes a `repeated_call` notice. A mutating call is never one,
    /// and any in the round clears the counts after it is screened.
    ///
    /// A mutating call its tool refused changed nothing, so it is screened
    /// like a read: a turn sent the same ambiguous `file_edit` nine rounds
    /// running, each answered "occurs 9 times", and the guard never fired
    /// because every one counted as a mutation and cleared the counts.
    ///
    /// A round is compared with the rounds before it, never with itself, so
    /// identical calls batched into one round all run — and all count.
    pub fn screen(&mut self, results: &mut [ToolResult]) -> Verdict {
        let mutating: Vec<bool> = results
            .iter()
            .map(|r| mutates(&r.call) && r.response.get("error").is_none() && !changed_nothing(r))
            .collect();
        // Which occurrence of itself each result is; a mutation is none.
        let occurrences: Vec<usize> = results
            .iter()
            .zip(&mutating)
            .map(|(result, &mutating)| match mutating {
                true => 0,
                false => self.find(result).map_or(1, |seen| seen.count + 1),
            })
            .collect();
        let repeated: Vec<bool> = occurrences.iter().map(|&n| n >= NOTICE_FROM).collect();
        let failures: Vec<usize> = results.iter().map(|r| self.note_failure(r)).collect();
        if mutating.iter().any(|&m| m) {
            self.seen.clear();
        } else {
            for result in results.iter() {
                let answer = answer_of(result);
                match self
                    .seen
                    .iter_mut()
                    .find(|s| s.call == result.call && s.response == answer)
                {
                    Some(seen) => seen.count += 1,
                    None => self.seen.push(Seen {
                        call: result.call.clone(),
                        response: answer,
                        count: 1,
                    }),
                }
            }
        }
        let all_repeats = !results.is_empty() && repeated.iter().all(|&r| r);
        self.streak = if all_repeats { self.streak + 1 } else { 0 };
        let closing = self.streak >= STOP_AFTER;
        let rounds_left = STOP_AFTER.saturating_sub(self.streak);
        for (i, result) in results.iter_mut().enumerate() {
            if closing {
                result.response = closing_notice(&result.call);
            } else if repeated[i] {
                result.response = notice(&result.call, &result.response, occurrences[i]);
                if self.streak > 0 {
                    warn_of(&mut result.response, closing_warning(rounds_left));
                }
            } else if occurrences[i] >= WARN_FROM {
                let left = NOTICE_FROM - occurrences[i];
                let warning = repeat_warning(&result.call, occurrences[i], left);
                warn_of(&mut result.response, warning);
            } else if failures[i] >= FAILURES_WARN_FROM {
                let warning = failure_warning(&result.call, failures[i], &result.response);
                warn_of(&mut result.response, warning);
            }
        }
        if closing {
            Verdict::Close
        } else {
            Verdict::Continue
        }
    }

    /// Count `result` in its tool's run of one error: how many in a row it now
    /// is, 0 when it did not fail. Any other result of the tool ends the run.
    fn note_failure(&mut self, result: &ToolResult) -> usize {
        let Some(error) = result.response.get("error").and_then(Value::as_str) else {
            self.failing.remove(&result.call.name);
            return 0;
        };
        let run = self
            .failing
            .entry(result.call.name.clone())
            .or_insert_with(|| (error.to_string(), 0));
        if run.0 != error {
            *run = (error.to_string(), 0);
        }
        run.1 += 1;
        run.1
    }

    /// The earlier occurrences of `result`'s call with `result`'s response.
    fn find(&self, result: &ToolResult) -> Option<&Seen> {
        let answer = answer_of(result);
        self.seen
            .iter()
            .find(|s| s.call == result.call && s.response == answer)
    }
}

/// Whether `result` is a program that failed and left no file changed: it
/// mutated nothing, so it is screened like a read. Counted as a mutation for
/// starting a program, the same `node -e` failed sixteen times in one turn
/// with the guard clearing its counts each time.
fn changed_nothing(result: &ToolResult) -> bool {
    let response = &result.response;
    result.call.name == "run_command"
        && response.get("exit_code").and_then(Value::as_i64) != Some(0)
        && response
            .get("changed")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
}

/// `result`'s response as repeats are compared: a run's job id is new every
/// time, and says nothing about what the run came to.
fn answer_of(result: &ToolResult) -> Value {
    let mut answer = result.response.clone();
    if result.call.name == "run_command" {
        if let Some(fields) = answer.as_object_mut() {
            fields.remove("job");
        }
    }
    answer
}

/// The page a paged result names as the next one, if more remain.
fn next_page(response: &Value) -> Option<u64> {
    response.get("paging")?.get("next_page")?.as_u64()
}

/// Tell the turn answering `results` that it is the last of its request:
/// the round cap's closing turn. Every result stays what it was — the calls
/// ran — and the last one carries a `notice` saying no further call runs.
/// Without it the closing turn planned another round of greps, could not
/// write the call, and ended on "Grep checks to confirm nothing stale
/// remains:" with nothing said to the user (measured 2026-10-24).
pub fn cap_notice(results: &mut [ToolResult]) {
    let Some(last) = results.last_mut() else {
        return;
    };
    let notice = Value::String(
        "This was the last tool round this request allows: no further tool call will run. \
         Answer the user now — say what was done, and what remains undone."
            .to_string(),
    );
    match &mut last.response {
        Value::Object(map) => {
            map.insert("notice".to_string(), notice);
        }
        other => {
            let result = other.take();
            *other = json!({"result": result, "notice": notice});
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

/// Put `warning` on `response`, which stays what it was.
fn warn_of(response: &mut Value, warning: String) {
    match response {
        Value::Object(map) => {
            map.insert("warning".to_string(), Value::String(warning));
        }
        other => {
            let result = other.take();
            *other = json!({"result": result, "warning": warning});
        }
    }
}

/// The warning on the `occurrence`th identical call, with `left` still to run.
fn repeat_warning(call: &ToolCall, occurrence: usize, left: usize) -> String {
    format!(
        "This {} call has now run {occurrence} times in this turn with the same arguments and \
         the same result. It will run {left} more times, and then it is refused. Repeating it \
         will not change the result: do something different.",
        call.name
    )
}

/// The warning on the `failures`th failure in a row with the same error.
fn failure_warning(call: &ToolCall, failures: usize, response: &Value) -> String {
    let error = response
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "The last {failures} {} calls all failed with `{error}`. Changing the arguments has not \
         helped, so the approach is the problem: read the error, and use a different tool or \
         a different way to reach the goal.",
        call.name
    )
}

/// What a refused round says, with `rounds` more like it before the tools close.
fn closing_warning(rounds: usize) -> String {
    format!(
        "Every call in this round was refused. After {rounds} more rounds like it the tools \
         close for this request, and nothing further can be done. Save what you have finished \
         now, committing it if the task asks for that, then do something different."
    )
}

/// The result a repeated call gets in place of the same answer again —
/// `response`, its `occurrence`th. When that answer is one page of several, the
/// notice names the next, which is what a model repeating a paged call almost
/// always wanted.
fn notice(call: &ToolCall, response: &Value, occurrence: usize) -> Value {
    let detail = match next_page(response) {
        Some(next) => format!(
            "This {name} call is exactly one already made {times} times in this turn, and \
             its result would be exactly the one shown there: the same page. More pages \
             remain — to read the next one, call {name} again with the same arguments \
             and `page: {next}`.",
            name = call.name,
            times = occurrence - 1,
        ),
        None => format!(
            "This {} call is exactly one already made {} times in this turn, and its result \
             would be exactly the one shown there, so it is refused; your other tools still \
             work. Act on that result instead: take the next step your reasoning names — if a \
             file you need does not exist, create it with `write` — or answer the user.",
            call.name,
            occurrence - 1,
        ),
    };
    json!({ "error": "repeated_call", "detail": detail })
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

    /// **The round cap's last results say no further call runs**, on the last
    /// result only, and every result keeps what its call returned — a value
    /// that is no object is kept under `result`.
    #[test]
    fn the_cap_notice_rides_the_last_result_and_keeps_every_answer() {
        let mut results = vec![
            result("file_read", json!({}), json!({"text": "a"})),
            result("file_edit", json!({}), json!({"replacements": 1})),
        ];
        cap_notice(&mut results);
        assert_eq!(results[0].response, json!({"text": "a"}));
        assert_eq!(results[1].response["replacements"], 1);
        assert!(results[1].response["notice"]
            .as_str()
            .unwrap()
            .starts_with("This was the last tool round"));

        let mut bare = vec![result("calculator", json!({}), json!(42))];
        cap_notice(&mut bare);
        assert_eq!(bare[0].response["result"], 42);
        assert!(bare[0].response["notice"].is_string());

        cap_notice(&mut []);
    }

    /// **The closing answer bans the call and is bounded above the think
    /// block** — on a turn the balanced dial programmed, the answer keeps its
    /// room past the 3584-token thinking cap — and a tighter bound the turn
    /// already had stands.
    #[test]
    fn the_closing_answer_is_banned_from_calling_and_bounded() {
        let mut balanced = SamplingConfig::default();
        steer(&mut balanced, ThinkMode::Balanced, 7168, &[]);
        closing_answer(&mut balanced, 42);
        assert!(balanced.banned_tokens.contains(&42));
        assert_eq!(
            (balanced.graceful_eos_after, balanced.forced_eos_after),
            (3584 + CLOSING_ANSWER_GRACEFUL, 3584 + CLOSING_ANSWER_FORCED)
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

    /// Run `round` through the guard `times` times, each keeping its real
    /// result, a warning aside.
    fn passes(guard: &mut RepeatGuard, round: impl Fn() -> Vec<ToolResult>, times: usize) {
        for n in 1..=times {
            let mut r = round();
            let expected: Vec<Value> = r.iter().map(|x| x.response.clone()).collect();
            assert_eq!(guard.screen(&mut r), Verdict::Continue, "occurrence {n}");
            let mut got: Vec<Value> = r.iter().map(|x| x.response.clone()).collect();
            for response in &mut got {
                if let Some(fields) = response.as_object_mut() {
                    fields.remove("warning");
                }
            }
            assert_eq!(got, expected, "occurrence {n} keeps its real result");
        }
    }

    /// **A repeat is warned before it is refused**: the real result comes back
    /// every time, the warning counting down, until the refusal.
    #[test]
    fn a_repeat_is_warned_and_runs_before_it_is_refused() {
        let mut guard = RepeatGuard::new();
        let read = || {
            vec![result(
                "file_read",
                json!({"path": "a"}),
                json!({"text": "same"}),
            )]
        };
        for occurrence in 1..NOTICE_FROM {
            let mut round = read();
            assert_eq!(guard.screen(&mut round), Verdict::Continue);
            assert_eq!(round[0].response["text"], "same", "{occurrence} ran");
            let warning = round[0].response.get("warning").and_then(Value::as_str);
            if occurrence < WARN_FROM {
                assert_eq!(warning, None, "occurrence {occurrence}");
            } else {
                let left = format!("run {} more times", NOTICE_FROM - occurrence);
                assert!(warning.unwrap().contains(&left), "{warning:?}");
            }
        }
        let mut refused = read();
        assert_eq!(guard.screen(&mut refused), Verdict::Continue);
        assert_eq!(refused[0].response["error"], "repeated_call");
        assert!(refused[0].response.get("text").is_none());
    }

    /// **One refused call leaves the rest of the round alone**, and a round of
    /// nothing but refusals says to save the work before the tools close.
    #[test]
    fn a_refusal_leaves_other_tools_and_warns_before_they_close() {
        let mut guard = RepeatGuard::new();
        passes(&mut guard, || vec![missing()], NOTICE_FROM - 1);
        let other = || result("file_list", json!({"path": "b"}), json!({"entries": ["x"]}));
        let mut mixed = vec![missing(), other()];
        assert_eq!(guard.screen(&mut mixed), Verdict::Continue);
        assert_eq!(mixed[0].response["error"], "repeated_call");
        assert_eq!(mixed[1].response["entries"][0], "x");

        for rounds in [2, 1] {
            let mut refused = vec![missing()];
            assert_eq!(guard.screen(&mut refused), Verdict::Continue);
            let warning = refused[0].response["warning"].as_str().unwrap();
            assert!(
                warning.contains(&format!("After {rounds} more")),
                "{warning}"
            );
            assert!(warning.contains("Save what you have finished"), "{warning}");
        }
        let mut last = vec![missing()];
        assert_eq!(guard.screen(&mut last), Verdict::Close);
    }

    /// **One error over and over is a pattern, whatever the arguments**: it is
    /// warned, refused for nothing, and a success or another error starts over.
    #[test]
    fn the_same_failure_with_different_arguments_is_warned() {
        let mut guard = RepeatGuard::new();
        let refused = |n: u32| {
            vec![result(
                "run_command",
                json!({"program": "node", "args": [format!("D:/a/{n}.js")]}),
                json!({"error": "refused", "detail": "an absolute path"}),
            )]
        };
        for n in 1..FAILURES_WARN_FROM as u32 {
            let mut round = refused(n);
            guard.screen(&mut round);
            assert!(round[0].response.get("warning").is_none(), "failure {n}");
        }
        for n in 0..20 {
            let mut round = refused(100 + n);
            assert_eq!(guard.screen(&mut round), Verdict::Continue);
            assert_eq!(round[0].response["error"], "refused");
            let warning = round[0].response["warning"].as_str().unwrap();
            assert!(warning.contains("failed with `refused`"), "{warning}");
        }
        let mut worked = vec![result(
            "run_command",
            json!({"program": "node", "args": ["a.js"]}),
            json!({"exit_code": 0, "changed": []}),
        )];
        guard.screen(&mut worked);
        let mut again = refused(999);
        guard.screen(&mut again);
        assert!(
            again[0].response.get("warning").is_none(),
            "it started over"
        );
    }

    /// **Repeats are warned, then refused, and the loop closes after
    /// [`STOP_AFTER`] such rounds in a row**, the closing round saying no call
    /// will run.
    #[test]
    fn a_repeated_call_is_a_notice_and_the_loop_then_closes() {
        let mut guard = RepeatGuard::new();
        passes(&mut guard, || vec![missing()], NOTICE_FROM - 1);
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
                assert!(
                    detail.contains(&format!("already made {} times", NOTICE_FROM + round - 2)),
                    "{detail}"
                );
            }
        }
    }

    /// **A mutation starts every count again**, and is never a repeat itself:
    /// a file read three times, then written, can be read three times more.
    #[test]
    fn a_mutation_clears_every_count() {
        let mut guard = RepeatGuard::new();
        let write = || {
            result(
                "write",
                json!({"repo": "candle", "path": "scratch/T.ts", "content": "x"}),
                json!({"path": "scratch/T.ts", "created": true}),
            )
        };
        passes(&mut guard, || vec![missing()], NOTICE_FROM - 1);
        // Written the same way more times than any read may be: never a repeat.
        passes(&mut guard, || vec![write()], NOTICE_FROM + 1);
        passes(&mut guard, || vec![missing()], NOTICE_FROM - 1);
        let mut fourth = vec![missing()];
        guard.screen(&mut fourth);
        assert_eq!(fourth[0].response["error"], "repeated_call");
    }

    /// **A refused mutation is screened like a read**: the same ambiguous
    /// `file_edit`, refused the same way each round, is a notice on its fourth
    /// occurrence, and the loop closes after [`STOP_AFTER`] such rounds.
    #[test]
    fn a_refused_edit_repeated_is_a_notice_and_closes_the_loop() {
        let mut guard = RepeatGuard::new();
        let ambiguous = || {
            vec![result(
                "file_edit",
                json!({"repo": "candle", "path": "docs/a.md",
                       "at": "Qm7x", "do": "replace", "old": "W_warm_max", "text": "B_out"}),
                json!({"error": "ambiguous", "detail": "`old` occurs 9 times in Qm7x"}),
            )]
        };
        passes(&mut guard, ambiguous, NOTICE_FROM - 1);
        for _ in 1..STOP_AFTER {
            let mut refused = ambiguous();
            assert_eq!(guard.screen(&mut refused), Verdict::Continue);
            assert_eq!(refused[0].response["error"], "repeated_call");
        }
        let mut last = ambiguous();
        assert_eq!(guard.screen(&mut last), Verdict::Close);
    }

    /// **A program failing the same way is a repeat**, whatever job id each
    /// run was given; one that changed a file or succeeded is not.
    #[test]
    fn a_program_failing_the_same_way_is_a_repeat() {
        let run = |job: u32, exit: i32, changed: Value| {
            result(
                "run_command",
                json!({"repo": "r", "program": "node", "args": ["-e"]}),
                json!({"job": format!("j{job}"), "exit_code": exit, "changed": changed}),
            )
        };
        let mut guard = RepeatGuard::new();
        for job in 0..NOTICE_FROM as u32 - 1 {
            let mut round = [run(job, 9, json!([]))];
            assert_eq!(guard.screen(&mut round), Verdict::Continue);
            assert!(round[0].response.get("job").is_some(), "not yet a repeat");
        }
        let mut round = [run(90, 9, json!([]))];
        assert_eq!(guard.screen(&mut round), Verdict::Continue);
        assert!(round[0].response.get("job").is_none(), "{:?}", round[0]);

        let mut succeeded = RepeatGuard::new();
        let mut wrote = RepeatGuard::new();
        for job in 0..2 * NOTICE_FROM as u32 {
            let mut ok = [run(job, 0, json!([]))];
            assert_eq!(succeeded.screen(&mut ok), Verdict::Continue);
            assert!(ok[0].response.get("job").is_some());
            let mut made = [run(job, 1, json!([{"path": "a", "change": "written"}]))];
            assert_eq!(wrote.screen(&mut made), Verdict::Continue);
            assert!(made[0].response.get("job").is_some());
        }
    }

    /// **A repeat of a paged call names the next page** — the step a model
    /// re-sending a listing without `page` was after.
    #[test]
    fn a_repeated_paged_call_is_told_the_next_page() {
        let mut guard = RepeatGuard::new();
        let listing = || {
            vec![result(
                "file_list",
                json!({"repo": "candle", "path": "zend/src"}),
                json!({"entries": [], "paging": {"page": 1, "pages": 2, "next_page": 2}}),
            )]
        };
        passes(&mut guard, listing, NOTICE_FROM - 1);
        let mut again = listing();
        guard.screen(&mut again);
        let detail = again[0].response["detail"].as_str().unwrap();
        assert!(detail.contains("`page: 2`"), "{detail}");
        assert!(!detail.contains("create it with `write`"), "{detail}");
    }

    /// Writing, changing a repository and running code or a program mutate;
    /// reading does not.
    #[test]
    fn what_mutates() {
        let call = |name: &str| ToolCall {
            name: name.to_string(),
            arguments: json!({}),
        };
        for name in [
            "write",
            "file_edit",
            "file_delete",
            "git_commit",
            "run_command",
            "code_run",
        ] {
            assert!(mutates(&call(name)), "{name}");
        }
        for name in [
            "file_read",
            "file_list",
            "file_grep",
            "git_log",
            "git_show",
            "web_fetch",
        ] {
            assert!(!mutates(&call(name)), "{name}");
        }
        // A writer renamed in the registry would silently stop clearing counts.
        for name in OVERLAY_WRITERS {
            assert!(
                registry::find(name).is_some(),
                "{name} is a registered tool"
            );
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
    /// `git_show` seven times over with the guard never firing. Counted over
    /// the whole turn, each reaches its fourth occurrence, is a notice, and the
    /// loop closes like any other run of repeats.
    #[test]
    fn two_calls_taking_turns_are_caught_and_close_the_loop() {
        let mut guard = RepeatGuard::new();
        let log = || {
            result(
                "git_log",
                json!({"repo": "candle", "page": 1}),
                json!({"count": 2000, "commits": []}),
            )
        };
        let show = || {
            result(
                "git_show",
                json!({"repo": "candle", "what": "changes", "page": 1}),
                json!({"subject": "s", "changes": []}),
            )
        };
        for _ in 1..NOTICE_FROM {
            passes(&mut guard, || vec![log()], 1);
            passes(&mut guard, || vec![show()], 1);
        }

        let mut log_again = vec![log()];
        assert_eq!(guard.screen(&mut log_again), Verdict::Continue);
        assert_eq!(
            log_again[0].response["error"], "repeated_call",
            "{log_again:?}"
        );
        let detail = log_again[0].response["detail"].as_str().unwrap();
        assert!(detail.contains("in this turn"), "{detail}");

        let mut show_again = vec![show()];
        assert_eq!(guard.screen(&mut show_again), Verdict::Continue);
        let mut log_last = vec![log()];
        assert_eq!(guard.screen(&mut log_last), Verdict::Close);
        let detail = log_last[0].response["detail"].as_str().unwrap();
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
        let listing = |path: &str| {
            result(
                "file_list",
                json!({"repo": "candle", "path": path}),
                json!({"entries": []}),
            )
        };
        passes(&mut guard, || vec![missing()], NOTICE_FROM - 1);
        let mut mixed = vec![missing(), listing("docs")];
        assert_eq!(guard.screen(&mut mixed), Verdict::Continue);
        assert_eq!(mixed[0].response["error"], "repeated_call");
        assert_eq!(mixed[1].response["entries"], json!([]));
        assert_eq!(guard.streak, 0);
    }
}
