//! Whether a `code_reading` chain read the whole file (`docs/zend_working_set.md`
//! §4.5).
//!
//! A chain can close without reading every page — cut at the round cap, or a
//! page skipped — and serving it as "the full contents" would promise lines
//! the model never gets. The chain's own record decides: each call turn's
//! `file_read` pages, paired in order with the `<tool_response>` blocks of the
//! turn after it, count when the response carries no `error`. The chain is
//! complete when those pages cover `0 .. ceil(lines / PAGE_LINES)`.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use serde_json::Value;
use zend_vfs::vfs::PAGE_LINES;

use crate::tool_round::{plan, Step};

const FILE_READ: &str = "file_read";
const OPEN: &str = "<tool_response>";
const CLOSE: &str = "</tool_response>";

/// Completeness per chain, computed once: a finished chain never changes.
#[derive(Default)]
pub struct Coverage {
    complete: Mutex<HashMap<TimelineId, bool>>,
}

impl Coverage {
    /// Whether the chain on `timeline` read every page of its `lines`-line file.
    pub fn is_complete(
        &self,
        engine: &ConversationEngine,
        timeline: TimelineId,
        lines: usize,
    ) -> bool {
        if let Some(&known) = self.complete.lock().unwrap().get(&timeline) {
            return known;
        }
        let complete = covers(&read_pages(&engine.turn_texts(timeline)), lines);
        self.complete.lock().unwrap().insert(timeline, complete);
        complete
    }
}

/// Whether `pages` covers every page of a `lines`-line file.
pub fn covers(pages: &BTreeSet<u32>, lines: usize) -> bool {
    let needed = lines.div_ceil(PAGE_LINES as usize) as u32;
    (0..needed).all(|p| pages.contains(&p))
}

/// The `file_read` pages a chain served without an error. `turns` are the
/// chain's `(user, assistant)` texts, oldest first: a call turn's calls are
/// answered by the next turn's user half, block for block.
pub fn read_pages(turns: &[(String, String)]) -> BTreeSet<u32> {
    let mut pages = BTreeSet::new();
    for pair in turns.windows(2) {
        let calls = plan(&pair[0].1);
        let responses = response_bodies(&pair[1].0);
        for (step, body) in calls.iter().zip(responses) {
            let Step::Run(call) = step else {
                continue;
            };
            if call.name != FILE_READ || is_error(body) {
                continue;
            }
            if let Some(page) = call.arguments.get("page").and_then(Value::as_u64) {
                pages.insert(page as u32);
            }
        }
    }
    pages
}

/// Each `<tool_response>` block's body in `user`, in order, without the
/// `[n/total name]` label a multi-call round puts on its first line.
fn response_bodies(user: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = user;
    while let Some(open) = rest.find(OPEN) {
        let after = &rest[open + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else {
            break;
        };
        out.push(strip_label(&after[..end]));
        rest = &after[end + CLOSE.len()..];
    }
    out
}

/// `body` without a leading `[n/total name]\n` label.
fn strip_label(body: &str) -> &str {
    let Some((first, rest)) = body.split_once('\n') else {
        return body;
    };
    let is_label = first
        .strip_prefix('[')
        .and_then(|l| l.strip_suffix(']'))
        .and_then(|l| l.split_once(' '))
        .and_then(|(count, _name)| count.split_once('/'))
        .is_some_and(|(n, total)| {
            !n.is_empty()
                && !total.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit())
                && total.bytes().all(|b| b.is_ascii_digit())
        });
    if is_label {
        rest
    } else {
        body
    }
}

/// Whether a response body is a failed call's error envelope.
fn is_error(body: &str) -> bool {
    serde_json::from_str::<Value>(body)
        .ok()
        .is_some_and(|v| v.get("error").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(page: u32) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"file_read\", \"arguments\": \
             {{\"repo\": \"candle\", \"path\": \"a.rs\", \"page\": {page}}}}}\n</tool_call>"
        )
    }

    fn response(body: &str) -> String {
        format!("<tool_response>{body}</tool_response>\n")
    }

    const PAGE: &str =
        "\n```rust file=candle/a.rs page=1/2 lines=350\n…\n```\nend of candle/a.rs page 1/2\n";

    /// Two call turns answered in turn, then the summary: both pages count.
    #[test]
    fn a_chain_that_read_every_page_is_complete() {
        let turns = vec![
            ("Read a.rs".to_string(), call(0)),
            (response(PAGE), call(1)),
            (response(PAGE), "It defines A.".to_string()),
        ];
        let pages = read_pages(&turns);
        assert_eq!(pages, BTreeSet::from([0, 1]));
        assert!(covers(&pages, 350));
    }

    /// A chain missing a page, or whose page came back an error, is not.
    #[test]
    fn a_missing_page_or_an_error_is_not_complete() {
        let skipped = vec![
            ("Read a.rs".to_string(), call(0)),
            (response(PAGE), "It defines A.".to_string()),
        ];
        assert!(!covers(&read_pages(&skipped), 350));

        let failed = vec![
            ("Read a.rs".to_string(), call(0)),
            (response(PAGE), call(1)),
            (
                response(r#"{"error":"not_found","detail":"x"}"#),
                "It defines A.".to_string(),
            ),
        ];
        assert_eq!(read_pages(&failed), BTreeSet::from([0]));
        assert!(!covers(&read_pages(&failed), 350));
    }

    /// A multi-call round's labels are stripped before the error test, and the
    /// responses pair with the calls in order.
    #[test]
    fn labelled_responses_pair_with_calls_in_order() {
        let calls = format!("{}\n{}", call(0), call(1));
        let user = format!(
            "{}{}",
            response(&format!("[1/2 file_read]\n{PAGE}")),
            response("[2/2 file_read]\n{\"error\":\"bad_page\"}"),
        );
        let turns = vec![("Read a.rs".to_string(), calls), (user, "…".to_string())];
        assert_eq!(read_pages(&turns), BTreeSet::from([0]));
    }

    /// Page counts are exact at the boundary: 200 lines is one page, 201 two,
    /// and an empty file needs none.
    #[test]
    fn the_page_count_is_exact_at_the_boundary() {
        let first = BTreeSet::from([0]);
        assert!(covers(&first, 200));
        assert!(!covers(&first, 201));
        assert!(covers(&BTreeSet::new(), 0));
    }
}
