//! What a running worker is doing, counted from its turn's stream — never
//! summarised by a model — and said in one sentence: how long it has run, its
//! rounds, its thinking, its tool calls by name, and what it is doing now
//! (`docs/claude_zend_gateway.md` §3.3).
//!
//! The counts live on the worker's record, so whichever member a waiting call
//! lands on reads the same sentence.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::session::StreamItem;

/// What a worker is doing at the moment.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Doing {
    #[default]
    Starting,
    /// Prefilling its input: the task, or the last round's tool results.
    Reading,
    Thinking,
    /// Running a batch of tool calls, by name.
    Calling(Vec<String>),
    /// Writing past its think block: a tool call, or its answer.
    Answering,
    /// Its turn is over and its journal entry is being written.
    Journaling,
}

/// A worker's activity so far.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Activity {
    /// Rounds finished: each tool round is one, and so is the answer.
    pub rounds: u32,
    /// Thinking tokens over the think blocks that have closed.
    pub thinking: u64,
    /// The open think block's tokens so far.
    pub thinking_now: u64,
    /// Tool calls made, by tool name.
    pub tools: BTreeMap<String, u32>,
    pub now: Doing,
    /// How the program a tool call is waiting on stands, while it runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    /// When the batch of tool calls now running began, in epoch milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_ms: Option<u64>,
    /// Milliseconds spent in tool calls, the rest of the turn being the model's.
    #[serde(default)]
    pub tool_ms: u64,
    /// Failed calls by `<tool> <how>`: the error's code, or `exit <code>`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub failed: BTreeMap<String, u32>,
    /// When a call first changed a file, in epoch milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_change_ms: Option<u64>,
    /// When the first commit landed, in epoch milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_commit_ms: Option<u64>,
}

/// The tools that change a conversation's files.
const CHANGES_FILES: [&str; 5] = [
    "write",
    "file_edit",
    "file_delete",
    "file_move",
    "file_image",
];

/// How `result` failed, if it did: its error's code, or its program's exit.
fn failure(result: &Value) -> Option<String> {
    if let Some(error) = result.get("error").and_then(Value::as_str) {
        return Some(error.to_string());
    }
    match result.get("exit_code").and_then(Value::as_i64) {
        Some(0) | None => None,
        Some(code) => Some(format!("exit {code}")),
    }
}

/// Whether `tool`'s `result`, which did not fail, changed a file: `file_image`
/// asked for no copies only reads.
fn changed_a_file(tool: &str, result: &Value) -> bool {
    CHANGES_FILES.contains(&tool)
        && result
            .get("written")
            .and_then(Value::as_array)
            .is_none_or(|written| !written.is_empty())
}

impl Activity {
    /// [`Self::observe`], also timing the tool calls at `now_ms` and keeping
    /// how each came out: what a task's turn report reads (`tools::turn_report`).
    pub fn observe_at(&mut self, item: &StreamItem, now_ms: u64) -> bool {
        if let StreamItem::Tool(status) = item {
            match status.phase {
                "running" => self.batch_ms = Some(now_ms),
                "done" => {
                    let began = self.batch_ms.take().unwrap_or(now_ms);
                    self.tool_ms += now_ms.saturating_sub(began);
                    for (tool, result) in status.tools.iter().zip(&status.results) {
                        self.note_result(tool, result, now_ms);
                    }
                }
                _ => {}
            }
        }
        self.observe(item)
    }

    fn note_result(&mut self, tool: &str, result: &Value, now_ms: u64) {
        if let Some(how) = failure(result) {
            *self.failed.entry(format!("{tool} {how}")).or_default() += 1;
            return;
        }
        if changed_a_file(tool, result) {
            self.first_change_ms.get_or_insert(now_ms);
        }
        if tool == "git_commit" {
            self.first_commit_ms.get_or_insert(now_ms);
        }
    }

    /// Fold one item of the turn's stream in. Whether it is a step worth
    /// telling at once — a round ended, or a batch of tool calls began — rather
    /// than on the next beat.
    pub fn observe(&mut self, item: &StreamItem) -> bool {
        match item {
            StreamItem::Prefill { .. } => {
                self.now = Doing::Reading;
                false
            }
            StreamItem::Think {
                tokens,
                done: false,
            } => {
                self.thinking_now = *tokens as u64;
                self.now = Doing::Thinking;
                false
            }
            StreamItem::Think { tokens, done: true } => {
                self.thinking += *tokens as u64;
                self.thinking_now = 0;
                self.now = Doing::Answering;
                false
            }
            StreamItem::Tool(status) if status.phase == "running" => {
                for name in &status.tools {
                    *self.tools.entry(name.clone()).or_default() += 1;
                }
                self.now = Doing::Calling(status.tools.clone());
                self.run = None;
                true
            }
            StreamItem::Tool(status) if status.phase == "progress" => {
                self.run = status.tools.first().cloned();
                true
            }
            StreamItem::Tool(_) => {
                self.now = Doing::Reading;
                self.run = None;
                false
            }
            StreamItem::TurnEnd { .. } => {
                self.rounds += 1;
                true
            }
            _ => false,
        }
    }

    /// The thinking tokens spent so far, the open block's included.
    pub fn thinking_total(&self) -> u64 {
        self.thinking + self.thinking_now
    }

    /// One sentence on it, `elapsed_ms` into the task: `running 4m10s, 6
    /// rounds: 1,934 thinking tokens, 12 tool calls (web_fetch 5, web_search
    /// 4, file_grep 3); now thinking`.
    pub fn sentence(&self, elapsed_ms: u64) -> String {
        let calls: u32 = self.tools.values().sum();
        let tools = if calls == 0 {
            "no tool calls yet".to_string()
        } else {
            let mut by_count: Vec<(&String, &u32)> = self.tools.iter().collect();
            by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let list: Vec<String> = by_count.iter().map(|(n, c)| format!("{n} {c}")).collect();
            format!(
                "{} tool call{} ({})",
                count(u64::from(calls)),
                plural(u64::from(calls)),
                list.join(", ")
            )
        };
        let run = self
            .run
            .as_ref()
            .map_or_else(String::new, |run| format!(" ({run})"));
        format!(
            "running {}, {} round{}: {} thinking tokens, {tools}; now {}{run}",
            duration(elapsed_ms),
            self.rounds,
            plural(u64::from(self.rounds)),
            count(self.thinking_total()),
            doing(&self.now)
        )
    }
}

fn doing(now: &Doing) -> String {
    match now {
        Doing::Starting => "starting".into(),
        Doing::Reading => "reading its input".into(),
        Doing::Thinking => "thinking".into(),
        Doing::Calling(tools) => format!("calling {}", tools.join(", ")),
        Doing::Answering => "writing past its thinking".into(),
        Doing::Journaling => "writing its journal entry".into(),
    }
}

fn plural(n: u64) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// `n` with thousands separated: `1,934`.
fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `ms` as `45s`, `4m10s` or `1h02m`.
pub fn duration(ms: u64) -> String {
    let s = ms / 1000;
    match (s / 3600, (s / 60) % 60, s % 60) {
        (0, 0, sec) => format!("{sec}s"),
        (0, min, sec) => format!("{min}m{sec:02}s"),
        (h, min, _) => format!("{h}h{min:02}m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::ToolStatusOut;

    fn tools(phase: &'static str, names: &[&str]) -> StreamItem {
        StreamItem::Tool(ToolStatusOut {
            phase,
            tools: names.iter().map(|n| n.to_string()).collect(),
            results: Vec::new(),
            tokens: Vec::new(),
        })
    }

    #[test]
    fn a_turns_stream_is_counted_into_one_sentence() {
        let mut a = Activity::default();
        assert!(!a.observe(&StreamItem::Prefill {
            done: 10,
            total: 20
        }));
        assert!(!a.observe(&StreamItem::Think {
            tokens: 400,
            done: false
        }));
        assert_eq!(a.now, Doing::Thinking);
        assert!(!a.observe(&StreamItem::Think {
            tokens: 512,
            done: true
        }));
        assert!(a.observe(&tools("running", &["web_fetch", "web_search"])));
        assert!(!a.observe(&tools("done", &["web_fetch", "web_search"])));
        assert!(a.observe(&tools("running", &["web_fetch"])));
        assert!(!a.observe(&StreamItem::Think {
            tokens: 1422,
            done: false
        }));
        assert_eq!(
            a.sentence(250_000),
            "running 4m10s, 0 rounds: 1,934 thinking tokens, 3 tool calls (web_fetch 2, \
             web_search 1); now thinking"
        );
    }

    /// **Tool calls are timed and how each came out is kept**: an error by its
    /// code, a program by its exit, and the first change and commit by when.
    /// A `file_image` that only read changed nothing.
    #[test]
    fn how_each_call_came_out_is_kept() {
        let done = |names: &[&str], results: Vec<Value>| {
            StreamItem::Tool(ToolStatusOut {
                phase: "done",
                tools: names.iter().map(|n| n.to_string()).collect(),
                results,
                tokens: Vec::new(),
            })
        };
        let mut a = Activity::default();
        a.observe_at(&tools("running", &["file_image", "run_command"]), 1_000);
        a.observe_at(
            &done(
                &["file_image", "run_command"],
                vec![
                    serde_json::json!({"written": []}),
                    serde_json::json!({"error": "no_agent"}),
                ],
            ),
            3_500,
        );
        assert_eq!(a.tool_ms, 2_500);
        assert_eq!(a.first_change_ms, None);
        a.observe_at(&tools("running", &["run_command", "file_edit"]), 4_000);
        a.observe_at(
            &done(
                &["run_command", "file_edit"],
                vec![
                    serde_json::json!({"exit_code": 101}),
                    serde_json::json!({"written": 1}),
                ],
            ),
            4_100,
        );
        a.observe_at(&tools("running", &["git_commit"]), 5_000);
        a.observe_at(
            &done(&["git_commit"], vec![serde_json::json!({"id": "abc"})]),
            5_200,
        );
        assert_eq!(a.tool_ms, 2_800);
        assert_eq!(a.first_change_ms, Some(4_100));
        assert_eq!(a.first_commit_ms, Some(5_200));
        assert_eq!(a.failed["run_command no_agent"], 1);
        assert_eq!(a.failed["run_command exit 101"], 1);
        assert_eq!(a.tools["run_command"], 2);
    }

    /// **A long tool call says how its program stands**, and stops saying so
    /// when its results are back.
    #[test]
    fn a_running_programs_progress_joins_the_sentence() {
        let mut a = Activity::default();
        a.observe(&tools("running", &["run_command"]));
        assert!(a.observe(&tools("progress", &["1,243 lines, last: test foo ... ok"])));
        assert_eq!(a.tools["run_command"], 1, "progress is no tool call");
        assert_eq!(a.now, Doing::Calling(vec!["run_command".into()]));
        assert_eq!(
            a.sentence(10_000),
            "running 10s, 0 rounds: 0 thinking tokens, 1 tool call (run_command 1); now calling \
             run_command (1,243 lines, last: test foo ... ok)"
        );
        a.observe(&tools("done", &["run_command"]));
        assert_eq!(a.run, None);
    }

    #[test]
    fn a_fresh_worker_says_it_is_starting() {
        assert_eq!(
            Activity::default().sentence(3_000),
            "running 3s, 0 rounds: 0 thinking tokens, no tool calls yet; now starting"
        );
        let mut one = Activity {
            rounds: 1,
            ..Activity::default()
        };
        one.now = Doing::Journaling;
        assert_eq!(
            one.sentence(3_725_000),
            "running 1h02m, 1 round: 0 thinking tokens, no tool calls yet; now writing its \
             journal entry"
        );
    }

    #[test]
    fn counts_are_grouped_by_thousands() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_000), "1,000");
        assert_eq!(count(1_234_567), "1,234,567");
    }
}
