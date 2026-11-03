//! A log read where it is written, so only the answer crosses the cluster: an
//! agent reads a job's log, keep a trial's. A test run's log is megabytes, and
//! a caller reading it to find one failure would otherwise carry all of it
//! into its context (`docs/build_deploy_tools.md` §3).
//!
//! Every line is numbered from 1 across the whole log, so a line a search
//! found is the line a later read of [`LogAsk::Lines`] returns. Each answer
//! holds at most [`MAX_LINES`] lines of at most [`LINE_CHARS`] characters.
//!
//! | Ask | Answers |
//! |---|---|
//! | [`LogAsk::Summary`] | cargo's `test result:` lines, the failures and errors, counts, and the last lines |
//! | [`LogAsk::Tail`] | the last `lines` |
//! | [`LogAsk::Lines`] | `count` lines from line `from` |
//! | [`LogAsk::Grep`] | the lines holding `pattern`, ignoring case, with `context` lines around each |

use std::collections::VecDeque;
use std::io::{self, BufRead};

use serde::{Deserialize, Serialize};

/// The most lines one answer holds.
pub const MAX_LINES: u32 = 200;

/// The most characters of a line an answer holds; the rest is cut, marked.
pub const LINE_CHARS: usize = 400;

/// The most context lines either side of a hit.
pub const MAX_CONTEXT: u32 = 5;

/// The last lines a summary holds.
const SUMMARY_TAIL: usize = 15;

/// The most of each kind of line a summary holds.
const SUMMARY_EACH: usize = 30;

/// What a reader asks of a log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum LogAsk {
    Summary,
    Tail {
        lines: u32,
    },
    Lines {
        /// The first line, counting from 1.
        from: u64,
        count: u32,
    },
    Grep {
        pattern: String,
        #[serde(default)]
        context: u32,
        /// Search from this line on, counting from 1: a search that hit its
        /// cap reads on from the line its answer names.
        #[serde(default = "first_line")]
        from: u64,
    },
}

fn first_line() -> u64 {
    1
}

/// One line of the log and its number, counting from 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Line {
    pub n: u64,
    pub text: String,
}

/// What a summary found across the whole log.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    /// cargo's `test result:` lines, one per test binary.
    pub results: Vec<Line>,
    /// Lines naming a failure: a panic, a `FAILED` test, a failing test's
    /// output header.
    pub failures: Vec<Line>,
    /// Compiler and tool errors: lines starting `error`.
    pub errors: Vec<Line>,
    /// How many lines of each there were — more than the lists hold when a
    /// list is full.
    pub failure_count: u64,
    pub error_count: u64,
    pub warning_count: u64,
    /// The last lines of the log.
    pub last: Vec<Line>,
}

/// The answer to a [`LogAsk`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogRead {
    /// Lines in the whole log when it was read.
    pub total: u64,
    /// Bytes in the log when it was read.
    pub size: u64,
    /// What wrote the log had ended before it was read, so this is all of it.
    pub ended: bool,
    /// The lines asked for: a tail, a range, or a search's hits with their
    /// context, in order.
    pub lines: Vec<Line>,
    /// A search's hits past the ones answered.
    #[serde(default)]
    pub more: u64,
    /// The line to read on from, when the answer stopped short of what was
    /// asked: a range or a search cut at [`MAX_LINES`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Summary>,
}

/// `text` cut to [`LINE_CHARS`] characters, marked when it was cut.
fn cut(text: &str) -> String {
    let text = text.trim_end_matches(['\r', '\n']);
    match text.char_indices().nth(LINE_CHARS) {
        Some((at, _)) => format!(
            "{} …[{} more chars]",
            &text[..at],
            text[at..].chars().count()
        ),
        None => text.to_string(),
    }
}

/// Every line of `log`, numbered from 1, decoded leniently: a command's
/// output need not be UTF-8.
fn each_line(mut log: impl BufRead, mut on: impl FnMut(u64, &str)) -> io::Result<u64> {
    let mut buf = Vec::new();
    let mut n = 0;
    loop {
        buf.clear();
        if log.read_until(b'\n', &mut buf)? == 0 {
            return Ok(n);
        }
        n += 1;
        on(n, &String::from_utf8_lossy(&buf));
    }
}

/// A line cargo prints for a failing test, a panic, or the list of failures.
fn is_failure(line: &str) -> bool {
    let t = line.trim();
    t.contains("panicked at")
        || (t.starts_with("test ") && t.ends_with("FAILED"))
        || (t.starts_with("---- ") && t.ends_with(" ----"))
        || t.starts_with("error: test failed")
}

/// A compiler or tool error line.
fn is_error(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("error[") || t.starts_with("error:")
}

fn is_warning(line: &str) -> bool {
    line.trim_start().starts_with("warning:")
}

/// Answer `ask` of `log`. `size` and `ended` are the caller's to fill.
pub fn read(log: impl BufRead, ask: &LogAsk) -> io::Result<LogRead> {
    match ask {
        LogAsk::Summary => summary(log),
        LogAsk::Tail { lines } => {
            let keep = (*lines).clamp(1, MAX_LINES) as usize;
            let mut last = VecDeque::with_capacity(keep);
            let total = each_line(log, |n, text| {
                if last.len() == keep {
                    last.pop_front();
                }
                last.push_back(Line { n, text: cut(text) });
            })?;
            Ok(LogRead {
                total,
                lines: last.into(),
                ..LogRead::default()
            })
        }
        LogAsk::Lines { from, count } => {
            let from = (*from).max(1);
            let count = u64::from((*count).clamp(1, MAX_LINES));
            let to = from + count;
            let mut lines = Vec::new();
            let total = each_line(log, |n, text| {
                if (from..to).contains(&n) {
                    lines.push(Line { n, text: cut(text) });
                }
            })?;
            Ok(LogRead {
                total,
                lines,
                next: (to <= total).then_some(to),
                ..LogRead::default()
            })
        }
        LogAsk::Grep {
            pattern,
            context,
            from,
        } => grep(log, pattern, (*context).min(MAX_CONTEXT), (*from).max(1)),
    }
}

fn summary(log: impl BufRead) -> io::Result<LogRead> {
    let mut s = Summary::default();
    let mut last = VecDeque::with_capacity(SUMMARY_TAIL);
    let total = each_line(log, |n, text| {
        let line = || Line { n, text: cut(text) };
        if text.trim_start().starts_with("test result:") && s.results.len() < SUMMARY_EACH {
            s.results.push(line());
        }
        if is_failure(text) {
            s.failure_count += 1;
            if s.failures.len() < SUMMARY_EACH {
                s.failures.push(line());
            }
        }
        if is_error(text) {
            s.error_count += 1;
            if s.errors.len() < SUMMARY_EACH {
                s.errors.push(line());
            }
        }
        if is_warning(text) {
            s.warning_count += 1;
        }
        if !text.trim().is_empty() {
            if last.len() == SUMMARY_TAIL {
                last.pop_front();
            }
            last.push_back(line());
        }
    })?;
    s.last = last.into();
    Ok(LogRead {
        total,
        summary: Some(s),
        ..LogRead::default()
    })
}

fn grep(log: impl BufRead, pattern: &str, context: u32, from: u64) -> io::Result<LogRead> {
    let needle = pattern.to_lowercase();
    let context = u64::from(context);
    let mut lines: Vec<Line> = Vec::new();
    // The lines before the one being read, for a hit's leading context.
    let mut before: VecDeque<Line> = VecDeque::with_capacity(context as usize);
    // Trailing context still owed after the last hit.
    let mut owed = 0u64;
    let mut more = 0u64;
    let mut next = None;
    let full = |lines: &Vec<Line>| lines.len() >= MAX_LINES as usize;
    let total = each_line(log, |n, text| {
        if n < from {
            return;
        }
        let hit = !needle.is_empty() && text.to_lowercase().contains(&needle);
        if hit && (full(&lines) || next.is_some()) {
            more += 1;
            next.get_or_insert(n);
            return;
        }
        let line = Line { n, text: cut(text) };
        if hit {
            lines.extend(before.drain(..));
            lines.push(line);
            owed = context;
        } else if owed > 0 && !full(&lines) {
            owed -= 1;
            lines.push(line);
        } else if context > 0 {
            if before.len() == context as usize {
                before.pop_front();
            }
            before.push_back(line);
        }
    })?;
    Ok(LogRead {
        total,
        lines,
        more,
        next,
        ..LogRead::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(log: &str, a: LogAsk) -> LogRead {
        read(log.as_bytes(), &a).unwrap()
    }

    fn texts(r: &LogRead) -> Vec<(u64, &str)> {
        r.lines.iter().map(|l| (l.n, l.text.as_str())).collect()
    }

    /// **A tail is the last lines, numbered across the whole log**, a CRLF
    /// log's line ends dropped.
    #[test]
    fn a_tail_is_the_last_lines_numbered_in_the_log() {
        let r = ask("a\r\nb\r\nc\r\n", LogAsk::Tail { lines: 2 });
        assert_eq!(texts(&r), [(2, "b"), (3, "c")]);
        assert_eq!(r.total, 3);
        let none = ask("", LogAsk::Tail { lines: 5 });
        assert_eq!((none.total, none.lines.len()), (0, 0));
    }

    /// **A range answers its lines and where to read on**, none past the end.
    #[test]
    fn a_range_answers_its_lines_and_where_to_read_on() {
        let log = "1\n2\n3\n4\n5\n";
        let r = ask(log, LogAsk::Lines { from: 2, count: 2 });
        assert_eq!(texts(&r), [(2, "2"), (3, "3")]);
        assert_eq!(r.next, Some(4));
        let end = ask(log, LogAsk::Lines { from: 4, count: 9 });
        assert_eq!(texts(&end), [(4, "4"), (5, "5")]);
        assert_eq!(end.next, None);
        let zero = ask(log, LogAsk::Lines { from: 0, count: 1 });
        assert_eq!(texts(&zero), [(1, "1")], "line 0 reads as line 1");
    }

    /// **A search ignores case, numbers its hits in the log and carries its
    /// context**, overlapping context written once.
    #[test]
    fn a_search_numbers_its_hits_with_their_context() {
        let log = "a\nb\nERROR one\nc\nd\ne\nerror two\nf\n";
        let r = ask(
            log,
            LogAsk::Grep {
                pattern: "error".into(),
                context: 1,
                from: 1,
            },
        );
        assert_eq!(
            texts(&r),
            [
                (2, "b"),
                (3, "ERROR one"),
                (4, "c"),
                (6, "e"),
                (7, "error two"),
                (8, "f")
            ]
        );
        assert_eq!((r.more, r.next, r.total), (0, None, 8));
        let later = ask(
            log,
            LogAsk::Grep {
                pattern: "error".into(),
                context: 0,
                from: 4,
            },
        );
        assert_eq!(texts(&later), [(7, "error two")]);
    }

    /// **A search past its cap counts what it left and says where to go on**:
    /// the line of the first hit it did not answer.
    #[test]
    fn a_search_past_its_cap_says_where_to_go_on() {
        let log: String = (1..=MAX_LINES + 3).map(|i| format!("hit {i}\n")).collect();
        let r = ask(
            &log,
            LogAsk::Grep {
                pattern: "HIT".into(),
                context: 0,
                from: 1,
            },
        );
        assert_eq!(r.lines.len(), MAX_LINES as usize);
        assert_eq!(r.more, 3);
        assert_eq!(r.next, Some(u64::from(MAX_LINES) + 1));
    }

    /// **A summary finds cargo's results, failures and errors across the whole
    /// log**, counts the warnings, and ends with the last lines.
    #[test]
    fn a_summary_finds_results_failures_and_errors() {
        let log = "   Compiling agent v0.1.0\n\
                   warning: unused import: `Foo`\n\
                   error[E0425]: cannot find value `x` in this scope\n\
                   running 2 tests\n\
                   test a::works ... ok\n\
                   test a::breaks ... FAILED\n\
                   \n\
                   failures:\n\
                   \n\
                   ---- a::breaks stdout ----\n\
                   thread 'a::breaks' panicked at src/a.rs:9:5:\n\
                   test result: FAILED. 1 passed; 1 failed; 0 ignored\n\
                   error: test failed, to rerun pass `-p agent --lib`\n";
        let r = ask(log, LogAsk::Summary);
        let s = r.summary.unwrap();
        let ns = |v: &[Line]| v.iter().map(|l| l.n).collect::<Vec<_>>();
        assert_eq!(ns(&s.results), [12]);
        assert_eq!(ns(&s.failures), [6, 10, 11, 13]);
        assert_eq!(ns(&s.errors), [3, 13]);
        assert_eq!(
            (s.failure_count, s.error_count, s.warning_count),
            (4, 2, 1)
        );
        assert_eq!(s.last.last().map(|l| l.n), Some(13));
        assert_eq!(r.total, 13);
    }

    /// **A long line is cut, saying how much was left**, and bytes that are not
    /// UTF-8 read as replacement characters rather than failing the read.
    #[test]
    fn a_long_line_is_cut_and_bad_bytes_read() {
        let long = "x".repeat(LINE_CHARS + 7);
        let r = ask(&long, LogAsk::Tail { lines: 1 });
        assert_eq!(
            r.lines[0].text,
            format!("{} …[7 more chars]", "x".repeat(LINE_CHARS))
        );
        let bad = read(&b"ok\n\xff\xfe\n"[..], &LogAsk::Tail { lines: 1 }).unwrap();
        assert_eq!(bad.lines[0].text, "\u{fffd}\u{fffd}");
    }

    /// **An ask is its JSON**, tagged by mode, with a search's defaults.
    #[test]
    fn an_ask_is_its_json() {
        let a: LogAsk = serde_json::from_str(r#"{"mode":"grep","pattern":"FAILED"}"#).unwrap();
        assert_eq!(
            a,
            LogAsk::Grep {
                pattern: "FAILED".into(),
                context: 0,
                from: 1
            }
        );
        assert_eq!(
            serde_json::to_string(&LogAsk::Tail { lines: 3 }).unwrap(),
            r#"{"mode":"tail","lines":3}"#
        );
        assert_eq!(
            serde_json::to_string(&LogAsk::Summary).unwrap(),
            r#"{"mode":"summary"}"#
        );
    }
}
