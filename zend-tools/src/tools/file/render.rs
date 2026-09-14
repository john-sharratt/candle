//! Rendering for file excerpts — the shared `<tool_response>` body format.
//!
//! One format, used from both ends of the system:
//!
//! * The `code_reading` ingest prefills tens of thousands of scope turns whose
//!   tool response is an excerpt in exactly this shape (`zend`'s
//!   `code_read::header::render_tool_response` delegates here).
//! * The live [`file_read`](super::read) tool returns the same shape, so a
//!   response the model reads at runtime is byte-identical to the ones it was
//!   conditioned on.
//!
//! ````text
//! src/auth/handler.rs (lines 47-93):
//!
//! ```rust
//!     47  impl AuthHandler {
//!     48      pub fn validate_token(&self, token: &str) -> Result<Claims> {
//!     93  }
//! ```
//! ````
//!
//! `cat -n` numbering, right-aligned to the widest line number, two spaces, then
//! the source verbatim. The header names the file and the absolute line range —
//! absolute so a follow-up read can ask for the next range directly.

/// Header + language-tagged fence + `cat -n` numbered body, as one string. The
/// caller frames it in `<tool_response>` tags.
///
/// `total_lines` lets the header say `(lines 1-200 of 1135)` when the excerpt is
/// a slice of something longer — the continuation signal, carried in the text
/// the model already reads rather than in a side-channel field it has to
/// correlate. When the excerpt ends at the last line the header is the plain
/// `(lines a-b)` form, matching the ingest corpus exactly.
///
/// `fence_tag` is the markdown language tag (`rust`, `python`, …); empty renders
/// a bare fence.
pub fn numbered_excerpt(
    path: &str,
    start_line: u32,
    end_line: u32,
    total_lines: u32,
    fence_tag: &str,
    body: &str,
) -> String {
    let width = digit_width(end_line);
    let mut numbered = String::with_capacity(body.len() + 8);
    // Emit exactly the lines the range names. Bounding by the count rather than
    // sniffing for a trailing newline keeps a legitimately blank LAST line —
    // `"a\n"` for range 1-2 is two lines, the second empty — while still dropping
    // the phantom element `split('\n')` leaves after a terminating newline.
    let expected = if total_lines == 0 {
        0
    } else {
        (end_line.saturating_sub(start_line) + 1) as usize
    };
    for (idx, line) in body.split('\n').take(expected).enumerate() {
        let line_no = start_line + idx as u32;
        numbered.push_str(&format!("{line_no:width$}  {line}\n", width = width));
    }

    let range = if total_lines == 0 {
        // An empty file has no range to state; "lines 1-0" reads as a bug.
        "empty".to_string()
    } else if end_line >= total_lines {
        format!("lines {start_line}-{end_line}")
    } else {
        format!("lines {start_line}-{end_line} of {total_lines}")
    };
    let fence_open = if fence_tag.is_empty() {
        String::from("```\n")
    } else {
        format!("```{fence_tag}\n")
    };
    format!("\n{path} ({range}):\n\n{fence_open}{numbered}```\n")
}

/// Markdown fence tag for a path's extension. Mirrors `zend`'s
/// `repo_scan::Language::fence_tag`; the two agree by a test in `zend`, which is
/// the only crate that can see both.
pub fn fence_tag_for_path(path: &str) -> &'static str {
    let ext = path
        .rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.'))
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "go" => "go",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "hh" => "cpp",
        "java" => "java",
        "rb" | "rake" | "ru" | "gemspec" => "ruby",
        "php" | "phtml" => "php",
        "sh" | "bash" | "zsh" => "bash",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" | "less" => "css",
        "md" | "markdown" | "mdx" => "markdown",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "json" | "json5" | "jsonc" => "json",
        _ => "",
    }
}

/// Decimal digits in `n` — the `cat -n` column width.
fn digit_width(n: u32) -> usize {
    let mut w = 1;
    let mut v = n;
    while v >= 10 {
        v /= 10;
        w += 1;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_right_aligned_and_fenced() {
        let out = numbered_excerpt("a.rs", 8, 10, 10, "rust", "one\ntwo\nthree\n");
        assert_eq!(
            out,
            "\na.rs (lines 8-10):\n\n```rust\n 8  one\n 9  two\n10  three\n```\n",
        );
    }

    /// A truncated excerpt says so in the header, so the model can ask for the
    /// next range without being told the page size.
    #[test]
    fn a_partial_excerpt_reports_the_total() {
        let out = numbered_excerpt("a.rs", 1, 2, 900, "rust", "one\ntwo\n");
        assert!(out.starts_with("\na.rs (lines 1-2 of 900):\n"), "{out}");
    }

    #[test]
    fn a_trailing_newline_does_not_invent_a_line() {
        let out = numbered_excerpt("a.txt", 1, 1, 1, "", "only\n");
        assert_eq!(out, "\na.txt (lines 1-1):\n\n```\n1  only\n```\n");
    }

    #[test]
    fn an_empty_file_says_so_instead_of_an_impossible_range() {
        let out = numbered_excerpt("a.rs", 1, 0, 0, "rust", "");
        assert_eq!(out, "\na.rs (empty):\n\n```rust\n```\n");
    }

    /// A range whose last line is legitimately blank keeps it — the count, not a
    /// trailing-newline heuristic, decides how many lines an excerpt has.
    #[test]
    fn a_blank_last_line_inside_the_range_is_kept() {
        let out = numbered_excerpt("a.rs", 1, 2, 2, "rust", "a\n");
        assert_eq!(out, "\na.rs (lines 1-2):\n\n```rust\n1  a\n2  \n```\n");
    }

    /// A body whose last line has no trailing newline keeps that line.
    #[test]
    fn a_body_without_a_trailing_newline_keeps_its_last_line() {
        let out = numbered_excerpt("a.rs", 1, 2, 2, "rust", "one\ntwo");
        assert_eq!(out, "\na.rs (lines 1-2):\n\n```rust\n1  one\n2  two\n```\n");
    }

    #[test]
    fn an_unknown_extension_renders_a_bare_fence() {
        assert_eq!(fence_tag_for_path("x/y.bin"), "");
        assert_eq!(fence_tag_for_path("Makefile"), "");
        assert_eq!(fence_tag_for_path("a/b/c.rs"), "rust");
        assert_eq!(fence_tag_for_path("A/B/C.RS"), "rust", "case-insensitive");
    }

    #[test]
    fn digit_width_matches_decimal_length() {
        for (n, w) in [(1, 1), (9, 1), (10, 2), (99, 2), (100, 3), (1234, 4)] {
            assert_eq!(digit_width(n), w, "n={n}");
        }
    }
}
