//! Whether a `code_reading` chain read the whole file (`docs/zend_working_set.md`
//! §4.5).
//!
//! A chain can close without reading every line — cut at the round cap, or a
//! range skipped — and placing it as "the full contents" would promise lines
//! the model never gets. The chain's own record decides: each call turn's
//! `file_read` calls, paired in order with the `<tool_response>` blocks of the
//! turn after it, count for the lines each response says it held — its closing
//! line, `end of <file> lines A-B of N` — when it carries no `error`. What a
//! response held, not what its call asked for: a read asked for more than one
//! read holds stops short and says so. The chain is complete when those ranges
//! cover `1 ..= lines`, the count its ingest recorded.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, TurnText};
use serde_json::Value;

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
    /// Whether the chain on `timeline` read every line of its `lines`-line
    /// file.
    pub fn is_complete(
        &self,
        engine: &ConversationEngine,
        timeline: TimelineId,
        lines: usize,
    ) -> bool {
        if let Some(&known) = self.complete.lock().unwrap().get(&timeline) {
            return known;
        }
        let complete = covers(&read_lines(&engine.turn_texts(timeline)), lines);
        self.complete.lock().unwrap().insert(timeline, complete);
        complete
    }
}

/// Whether `read` (ranges of lines from 1, inclusive) covers every line of a
/// `lines`-line file.
pub fn covers(read: &[(usize, usize)], lines: usize) -> bool {
    let mut ranges = read.to_vec();
    ranges.sort_unstable();
    let mut next = 1;
    for (from, to) in ranges {
        if from > next {
            break;
        }
        next = next.max(to + 1);
    }
    next > lines
}

/// The ranges of lines the `file_read`s of a chain held, as their responses
/// say, for each served without an error. `turns` are the chain's `(user,
/// assistant)` texts, oldest first: a call turn's calls are answered by the
/// next turn's user half, block for block.
pub fn read_lines(turns: &[(String, String)]) -> Vec<(usize, usize)> {
    let mut read = Vec::new();
    for pair in turns.windows(2) {
        // A stored turn keeps its text, not the split of it its ids carried, so
        // it is read with every tag as markup; a quoted tag can only add a
        // refused step here, and a refused step reads no line.
        let calls = plan(&TurnText::markup(pair[0].1.as_str()));
        let responses = response_bodies(&pair[1].0);
        for (step, body) in calls.iter().zip(responses) {
            let Step::Run(call) = step else {
                continue;
            };
            if call.name != FILE_READ || is_error(body) {
                continue;
            }
            read.extend(held(body));
        }
    }
    read
}

/// The lines a read's response says it held: its closing line,
/// `end of <file> lines A-B of N — …`.
fn held(body: &str) -> Option<(usize, usize)> {
    let close = body
        .lines()
        .rev()
        .find(|l| l.starts_with("end of ") && !l.starts_with("end of part "))?;
    let (said, _) = close.split_once(" — ")?;
    let at = said.rfind(" lines ")?;
    let (range, _) = said[at + " lines ".len()..].split_once(" of ")?;
    let number = |s: &str| s.parse::<usize>().ok();
    match range.split_once('-') {
        Some((a, b)) => Some((number(a)?, number(b)?)),
        None => number(range).map(|n| (n, n)),
    }
}

/// Each `<tool_response>` block's body in `user`, in order, without the
/// `[n/total name]` label a multi-call round puts on its first line.
pub fn response_bodies(user: &str) -> Vec<&str> {
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

    fn call(lines: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"file_read\", \"arguments\": \
             {{\"repo\": \"candle\", \"path\": \"a.rs\", \"lines\": \"{lines}\"}}}}\n</tool_call>"
        )
    }

    fn response(body: &str) -> String {
        format!("<tool_response>{body}</tool_response>\n")
    }

    const FIRST: &str = "```rust file=candle/a.rs part=Qm7x lines=1-3 scope=\"fn a\"\n +1 …\n```\n\
                         end of candle/a.rs lines 1-198 of 350 — next: lines=199-350\n";
    const LAST: &str = "```rust file=candle/a.rs part=b4_K lines=199-350 scope=\"fn b\"\n +1 …\n```\n\
                        end of candle/a.rs lines 199-350 of 350 — the end of the file\n";

    /// Two call turns answered in turn, then the summary: both ranges count.
    #[test]
    fn a_chain_that_read_every_line_is_complete() {
        let turns = vec![
            ("Read a.rs".to_string(), call("1-198")),
            (response(FIRST), call("199-350")),
            (response(LAST), "It defines A.".to_string()),
        ];
        let read = read_lines(&turns);
        assert_eq!(read, vec![(1, 198), (199, 350)]);
        assert!(covers(&read, 350));
    }

    /// A chain missing a range, or whose range came back an error, is not.
    #[test]
    fn a_missing_range_or_an_error_is_not_complete() {
        let skipped = vec![
            ("Read a.rs".to_string(), call("1-198")),
            (response(FIRST), "It defines A.".to_string()),
        ];
        assert!(!covers(&read_lines(&skipped), 350));

        let failed = vec![
            ("Read a.rs".to_string(), call("1-198")),
            (response(FIRST), call("199-350")),
            (
                response(r#"{"error":"not_found","detail":"x"}"#),
                "It defines A.".to_string(),
            ),
        ];
        assert_eq!(read_lines(&failed), vec![(1, 198)]);
        assert!(!covers(&read_lines(&failed), 350));
    }

    /// **What a response held counts, not what its call asked for**: a read of
    /// the whole file that held its first 198 lines covers those alone.
    #[test]
    fn a_read_counts_for_what_it_held() {
        let turns = vec![
            ("Read a.rs".to_string(), call("1-350")),
            (response(FIRST), "It defines A.".to_string()),
        ];
        assert_eq!(read_lines(&turns), vec![(1, 198)]);
        assert!(!covers(&read_lines(&turns), 350));
    }

    /// A multi-call round's labels are stripped before the error test, and the
    /// responses pair with the calls in order.
    #[test]
    fn labelled_responses_pair_with_calls_in_order() {
        let calls = format!("{}\n{}", call("1-198"), call("199-350"));
        let user = format!(
            "{}{}",
            response(&format!("[1/2 file_read]\n{FIRST}")),
            response("[2/2 file_read]\n{\"error\":\"invalid_arguments\"}"),
        );
        let turns = vec![("Read a.rs".to_string(), calls), (user, "…".to_string())];
        assert_eq!(read_lines(&turns), vec![(1, 198)]);
    }

    /// Coverage is exact: every line from 1 to the last, in any order, with
    /// overlaps allowed and no gap; a file with no lines needs none.
    #[test]
    fn coverage_is_exact() {
        assert!(covers(&[(5, 9), (1, 4)], 9));
        assert!(covers(&[(1, 6), (4, 9)], 9));
        assert!(!covers(&[(1, 4), (6, 9)], 9));
        assert!(!covers(&[(2, 9)], 9));
        assert!(!covers(&[(1, 8)], 9));
        assert!(covers(&[], 0));
    }

    #[test]
    fn a_one_line_read_holds_its_line() {
        assert_eq!(
            held("line 4 is part=Qm7x +2\n…\nend of candle/a.rs lines 4 of 9 — next: lines=5-9\n"),
            Some((4, 4))
        );
        assert_eq!(held("end of part Qm7x of candle/a.rs — lines 3-5 of 9\n"), None);
    }
}
