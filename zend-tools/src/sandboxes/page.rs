//! A job's log, a page at a time.
//!
//! A page is [`PAGE_LINES`] lines, as `file_read`'s is, and a page past the
//! end clamps to the last one rather than coming back empty — an empty page
//! reads as "no output", which is the wrong answer. Terminal colour codes are
//! stripped, since a program that thinks it prints to a terminal colours its
//! output and the codes are noise to a reader; a line longer than
//! [`MAX_LINE_CHARS`] is cut, saying how much was left off, so one minified
//! line cannot fill a page.

use std::io;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

/// Lines in one page of a job's output.
pub const PAGE_LINES: usize = 200;

/// Characters of one line shown before it is cut.
pub const MAX_LINE_CHARS: usize = 500;

/// An ANSI escape sequence: a colour, a cursor move, an erase.
static ANSI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").expect("the pattern is valid"));

/// One page of a job's output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogPage {
    /// This page, 0-based.
    pub page: usize,
    /// How many pages the output has; at least one.
    pub pages: usize,
    /// The first line on this page, 1-based; 0 when there is no output.
    pub first_line: usize,
    /// The last line on this page.
    pub last_line: usize,
    /// Lines of output in all.
    pub total_lines: usize,
    pub text: String,
}

/// Page `page` of the log at `path`.
pub fn read(path: &Path, page: usize) -> io::Result<LogPage> {
    let bytes = std::fs::read(path)?;
    Ok(of(&String::from_utf8_lossy(&bytes), page))
}

/// Page `page` of `log`.
pub fn of(log: &str, page: usize) -> LogPage {
    let plain = ANSI.replace_all(log, "");
    let lines: Vec<&str> = plain.lines().collect();
    let total_lines = lines.len();
    let pages = total_lines.div_ceil(PAGE_LINES).max(1);
    let page = page.min(pages - 1);
    let start = page * PAGE_LINES;
    let end = (start + PAGE_LINES).min(total_lines);
    let text = lines[start..end]
        .iter()
        .map(|line| clip(line))
        .collect::<Vec<_>>()
        .join("\n");
    LogPage {
        page,
        pages,
        first_line: if total_lines == 0 { 0 } else { start + 1 },
        last_line: end,
        total_lines,
        text,
    }
}

/// `line`, cut at [`MAX_LINE_CHARS`] characters with a note of the rest.
fn clip(line: &str) -> String {
    let count = line.chars().count();
    if count <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{kept}… [{} more characters]", count - MAX_LINE_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}\n")).collect()
    }

    /// **A page is 200 lines, and says where it sits in the whole.**
    #[test]
    fn a_page_is_its_lines_and_where_they_sit() {
        let log = numbered(450);
        let first = of(&log, 0);
        assert_eq!(
            (
                first.page,
                first.pages,
                first.first_line,
                first.last_line,
                first.total_lines
            ),
            (0, 3, 1, 200, 450)
        );
        assert!(first.text.starts_with("line 1\nline 2\n"));
        assert!(first.text.ends_with("line 200"));
        let last = of(&log, 2);
        assert_eq!((last.first_line, last.last_line), (401, 450));
        assert_eq!(last.text.lines().count(), 50);
    }

    /// A page past the end is the last page, never an empty one.
    #[test]
    fn a_page_past_the_end_is_the_last() {
        let page = of(&numbered(250), 9);
        assert_eq!((page.page, page.first_line, page.last_line), (1, 201, 250));
    }

    /// No output is one empty page.
    #[test]
    fn no_output_is_one_empty_page() {
        assert_eq!(
            of("", 0),
            LogPage {
                page: 0,
                pages: 1,
                first_line: 0,
                last_line: 0,
                total_lines: 0,
                text: String::new(),
            }
        );
    }

    /// Colour codes go; CRLF endings read as lines; a long line is cut.
    #[test]
    fn colour_goes_and_a_long_line_is_cut() {
        let page = of("\x1b[32m✔ passed\x1b[0m\r\nplain\r\n", 0);
        assert_eq!(page.text, "✔ passed\nplain");
        let long = "x".repeat(MAX_LINE_CHARS + 7);
        assert_eq!(
            of(&long, 0).text,
            format!("{}… [7 more characters]", "x".repeat(MAX_LINE_CHARS))
        );
    }
}
