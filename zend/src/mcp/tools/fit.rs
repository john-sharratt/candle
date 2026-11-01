//! Keeping an answer to what a caller's context can take.
//!
//! The caller is a model with a token budget, and every answer lands in it
//! whole. Paging keeps most of them short, but a page counts lines or turns, and
//! a tool result is stored as one line of JSON, so "200 lines" can be a whole
//! file. [`fit`] is the bound on every answer, whichever tool made it;
//! [`clip_lines`] and [`around`] are the line bound where a tool reads stored
//! turns.

use super::Answer;

/// The most characters of one answer: about 8k tokens.
pub const MAX_ANSWER: usize = 32_000;

/// The most characters of one line of a stored turn.
pub const MAX_LINE: usize = 2_000;

/// How much of a long line [`around`] shows on each side of what it matched.
const AROUND: usize = 200;

/// `text`'s first `n` characters.
fn head(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// `answer`, cut to [`MAX_ANSWER`] saying how much it held and how to read on.
///
/// **A cut answer marks nothing delivered.** A wait's marks say its journal
/// entries reached the caller, and no entry is given twice, so an entry past the
/// cut would be lost for good. Unmarked, the next wait gives them again.
pub fn fit(answer: Answer) -> Answer {
    let total = answer.text.chars().count();
    if total <= MAX_ANSWER {
        return answer;
    }
    let kept = head(&answer.text, MAX_ANSWER);
    Answer {
        text: format!(
            "{kept}\n\n[Cut: {MAX_ANSWER} of {total} characters shown. Ask for less with this \
             tool's `offset`, `limit`, `from` or `grep`. A worker's whole answer reads on with \
             `journal_read` and `answer_from`, and its entries one at a time by their address.]"
        ),
        is_error: answer.is_error,
        marks: Vec::new(),
    }
}

/// `text` with every line cut to [`MAX_LINE`], saying how much each lost.
pub fn clip_lines(text: &str) -> String {
    text.split('\n')
        .map(|line| {
            let total = line.chars().count();
            if total <= MAX_LINE {
                line.to_string()
            } else {
                format!(
                    "{} … [{} more characters]",
                    head(line, MAX_LINE),
                    total - MAX_LINE
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `line` as a search shows it: whole when short, otherwise [`AROUND`]
/// characters either side of the first place it holds `pattern`, ignoring case.
pub fn around(line: &str, pattern: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= 2 * AROUND {
        return line.to_string();
    }
    let lower: Vec<char> = chars.iter().flat_map(|c| c.to_lowercase()).collect();
    let needle: Vec<char> = pattern.chars().flat_map(|c| c.to_lowercase()).collect();
    let found = (!needle.is_empty())
        .then(|| lower.windows(needle.len()).position(|w| w == needle))
        .flatten()
        .unwrap_or(0)
        .min(chars.len());
    let from = found.saturating_sub(AROUND);
    let to = (found + needle.len() + AROUND).min(chars.len());
    format!(
        "{}{}{}",
        if from > 0 { "… " } else { "" },
        chars[from..to].iter().collect::<String>(),
        if to < chars.len() { " …" } else { "" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::reported::Mark;

    fn mark() -> Mark {
        Mark {
            thread: "t".into(),
            name: "w".into(),
            entry: 3,
        }
    }

    /// An answer in bounds passes untouched, its marks with it.
    #[test]
    fn a_short_answer_is_left_alone() {
        let answer = Answer::ok("x".repeat(MAX_ANSWER)).marking(vec![mark()]);
        assert_eq!(fit(answer.clone()), answer);
    }

    /// **A cut answer says what it holds, and hands over no entry**: marked,
    /// the entries past the cut would never be given again.
    #[test]
    fn a_cut_answer_marks_nothing_delivered() {
        let answer = fit(Answer::ok("é".repeat(MAX_ANSWER + 5)).marking(vec![mark()]));
        assert!(answer.marks.is_empty());
        assert!(answer.text.starts_with(&"é".repeat(MAX_ANSWER)));
        assert!(answer.text.contains("Cut: 32000 of 32005 characters shown"));
        assert!(!answer.is_error);
    }

    /// A long line loses its tail and says how much; a short one and the line
    /// breaks stay as they were.
    #[test]
    fn a_long_line_is_clipped_and_says_so() {
        let text = format!("short\n{}\nlast", "y".repeat(MAX_LINE + 7));
        let clipped = clip_lines(&text);
        let lines: Vec<&str> = clipped.split('\n').collect();
        assert_eq!(lines[0], "short");
        assert_eq!(
            lines[1],
            format!("{} … [7 more characters]", "y".repeat(MAX_LINE))
        );
        assert_eq!(lines[2], "last");
    }

    /// A hit deep in a long line shows the match with its neighbours, not the
    /// start of the line; a short line and a miss stay readable.
    #[test]
    fn a_search_shows_the_match_in_a_long_line() {
        assert_eq!(around("short", "x"), "short");
        let line = format!("{}Needle{}", "a".repeat(1000), "b".repeat(1000));
        assert_eq!(
            around(&line, "needle"),
            format!("… {}Needle{} …", "a".repeat(AROUND), "b".repeat(AROUND))
        );
        let miss = around(&"c".repeat(1000), "zz");
        assert_eq!(miss, format!("{} …", "c".repeat(2 + AROUND)));
    }
}
