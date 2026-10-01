//! Rendering for file excerpts — the shared `<tool_response>` body format.
//!
//! One format, used from both ends of the system:
//!
//! * The live [`file_read`](super::read) tool returns this shape.
//! * The `code_reading` ingest drives that same tool call for every page of
//!   every file it reads, so the pages prefilled into its conversations are
//!   byte-identical to the ones a live read returns.
//!
//! ````text
//! ```rust file=server/src/auth/handler.rs page=0/3 lines=620
//!   1  impl AuthHandler {
//!   2      pub fn validate_token(&self, token: &str) -> Result<Claims> {
//! 200      // ...
//! ```
//! end of server/src/auth/handler.rs page 0/3
//! ````
//!
//! `cat -n` numbering, right-aligned to the widest line number, two spaces, then
//! the source verbatim. There is no header line: the opening fence itself names
//! the file as a workspace path (`<repo>/<path>` — a path alone is ambiguous
//! across a workspace of several repositories), which page this is (0-based,
//! matching `file_read`'s own request parameter), the page count, and the file's
//! length. The line numbers give the range. The closing line names the file and
//! page again, so the page's content is bracketed by its identity at both ends —
//! content at the bottom of a long page sits next to its name, not 200 lines away
//! from it, which is what keeps look-alike pages from different files apart.

/// The attribute every page of a file opens with, `file=<repo>/<path>` — the
/// file's anchor. A reply that points the model at pages it already holds names
/// them by exactly this string, so the two can be matched byte for byte.
pub fn file_anchor(repo: &str, path: &str) -> String {
    format!("file={repo}/{path}")
}

/// Fence + `cat -n` numbered body + closing line, as one string. The caller
/// frames it in `<tool_response>` tags.
///
/// `total_pages` and `total_lines` are always known by the time this is
/// called — [`zend_vfs::VfsStore::read_page`] streams the whole file to compute
/// them — so the fence always states them.
///
/// `fence_tag` is the markdown language tag (`rust`, `python`, …); empty renders
/// `text`, so the fence's first word is always a language and never the
/// `file=` attribute.
#[allow(clippy::too_many_arguments)]
pub fn numbered_excerpt(
    repo: &str,
    path: &str,
    page: u32,
    total_pages: u32,
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

    let file = format!("{repo}/{path}");
    let anchor = file_anchor(repo, path);
    let lang = if fence_tag.is_empty() {
        "text"
    } else {
        fence_tag
    };
    if total_lines == 0 {
        // An empty file has no page to state; "page=0/0" reads as a bug the same
        // way "lines 1-0" used to.
        return format!("\n```{lang} {anchor} lines=0\n```\nend of {file}\n");
    }
    format!(
        "\n```{lang} {anchor} page={page}/{total_pages} lines={total_lines}\n\
         {numbered}```\nend of {file} page {page}/{total_pages}\n"
    )
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
        let out = numbered_excerpt("r", "a.rs", 0, 1, 8, 10, 10, "rust", "one\ntwo\nthree\n");
        assert_eq!(
            out,
            "\n```rust file=r/a.rs page=0/1 lines=10\n 8  one\n 9  two\n10  three\n```\nend of r/a.rs page 0/1\n",
        );
    }

    /// The anchor is the fence's own `file=` attribute, so a reply naming it
    /// matches the page exactly.
    #[test]
    fn the_anchor_is_what_the_fence_opens_with() {
        assert_eq!(file_anchor("r", "src/a.rs"), "file=r/src/a.rs");
        let out = numbered_excerpt("r", "src/a.rs", 0, 1, 1, 1, 1, "rust", "x\n");
        assert!(out.starts_with("\n```rust file=r/src/a.rs page="), "{out}");
    }

    /// A page short of the last one says so in the fence, so the model knows
    /// to ask for `page + 1` without being told the stride.
    #[test]
    fn a_partial_page_reports_the_total() {
        let out = numbered_excerpt("r", "a.rs", 0, 3, 1, 2, 900, "rust", "one\ntwo\n");
        assert_eq!(
            out,
            "\n```rust file=r/a.rs page=0/3 lines=900\n1  one\n2  two\n```\nend of r/a.rs page 0/3\n"
        );
    }

    /// A file with no language tag still opens on a language, so `file=` is
    /// never read as the fence's info string.
    #[test]
    fn a_trailing_newline_does_not_invent_a_line() {
        let out = numbered_excerpt("r", "a.txt", 0, 1, 1, 1, 1, "", "only\n");
        assert_eq!(
            out,
            "\n```text file=r/a.txt page=0/1 lines=1\n1  only\n```\nend of r/a.txt page 0/1\n"
        );
    }

    #[test]
    fn an_empty_file_says_so_instead_of_an_impossible_range() {
        let out = numbered_excerpt("r", "a.rs", 0, 0, 1, 0, 0, "rust", "");
        assert_eq!(out, "\n```rust file=r/a.rs lines=0\n```\nend of r/a.rs\n");
    }

    /// A range whose last line is legitimately blank keeps it — the count, not a
    /// trailing-newline heuristic, decides how many lines an excerpt has.
    #[test]
    fn a_blank_last_line_inside_the_range_is_kept() {
        let out = numbered_excerpt("r", "a.rs", 0, 1, 1, 2, 2, "rust", "a\n");
        assert_eq!(
            out,
            "\n```rust file=r/a.rs page=0/1 lines=2\n1  a\n2  \n```\nend of r/a.rs page 0/1\n"
        );
    }

    /// A body whose last line has no trailing newline keeps that line.
    #[test]
    fn a_body_without_a_trailing_newline_keeps_its_last_line() {
        let out = numbered_excerpt("r", "a.rs", 0, 1, 1, 2, 2, "rust", "one\ntwo");
        assert_eq!(
            out,
            "\n```rust file=r/a.rs page=0/1 lines=2\n1  one\n2  two\n```\nend of r/a.rs page 0/1\n"
        );
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
