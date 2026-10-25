//! What a running worker is doing, counted from its turn's stream — never
//! summarised by a model — and said in one sentence: how long it has run, its
//! rounds, its thinking, its tool calls by name, and what it is doing now
//! (`docs/claude_zend_gateway.md` §3.3).
//!
//! The counts live on the worker's record, so whichever member a waiting call
//! lands on reads the same sentence.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
}

impl Activity {
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
                true
            }
            StreamItem::Tool(_) => {
                self.now = Doing::Reading;
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
        format!(
            "running {}, {} round{}: {} thinking tokens, {tools}; now {}",
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
