//! Line-splitting and file-header detection shared with [`crate::repo_scan::anchor`].
//!
//! Two utilities survive here from the old per-scope carve pipeline (see
//! `code_reading`'s module doc for why that pipeline is gone): `repo_map`'s
//! anchor excerpt still needs to bound line width and find where a file's
//! leading comment block ends.

/// Maximum characters a single carved turn should carry.
const MAX_LINE_CHARS: usize = 160;

/// Soft target: once a split piece reaches this width, break at the next safe
/// point (outside a string, after a delimiter) rather than running to the hard cap
/// — keeps pieces averaging near this instead of always maxing out.
const SOFT_LINE_CHARS: usize = 100;

/// Minimum number of comment lines a file's leading comment block must span for
/// [`file_header_end`] to report it as a real header (see that function's doc).
const MIN_FILE_HEADER_LINES: u32 = 2;

/// Rewrite `source` so no line exceeds [`MAX_LINE_CHARS`], inserting `\n` at safe
/// division points so a minified / one-line file becomes a normal multi-line
/// document.
///
/// A line already within the cap is untouched; a file with no over-long line is
/// returned unchanged. Split points are chosen **outside string literals**
/// (single/double/backtick, with `\` escapes) so we never cut a token in half if
/// we can help it — preferring a delimiter (`;,)}]>` or whitespace) once a piece
/// passes [`SOFT_LINE_CHARS`], and hard-clipping at [`MAX_LINE_CHARS`] when a
/// line (a giant string blob) offers no safe break. Operates on chars, so a
/// split never lands mid-UTF-8.
pub fn split_long_lines(source: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(source);
    if text
        .split('\n')
        .all(|l| l.chars().count() <= MAX_LINE_CHARS)
    {
        return source.to_vec();
    }
    let mut out = String::with_capacity(text.len() + text.len() / 8 + 16);
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        split_one_line(line, &mut out);
    }
    out.into_bytes()
}

fn split_one_line(line: &str, out: &mut String) {
    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= MAX_LINE_CHARS {
        out.push_str(line);
        return;
    }
    let mut piece_start = 0usize;
    let mut in_string: Option<char> = None;
    let mut escaped = false;
    for i in 0..chars.len() {
        let c = chars[i];
        // Track string state so a break is only taken outside a literal.
        match in_string {
            Some(q) => {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == q {
                    in_string = None;
                }
            }
            None => {
                if c == '"' || c == '\'' || c == '`' {
                    in_string = Some(c);
                }
            }
        }
        let piece_len = i + 1 - piece_start;
        let safe = in_string.is_none()
            && piece_len >= SOFT_LINE_CHARS
            && matches!(
                c,
                ';' | ',' | ')' | '}' | ']' | '>' | ' ' | '\t' | '&' | '|'
            );
        // Break AFTER char `i` (forward only) so string state never desyncs; a piece
        // with no safe break is hard-clipped at the cap.
        if safe || piece_len >= MAX_LINE_CHARS {
            out.extend(chars[piece_start..=i].iter());
            out.push('\n');
            piece_start = i + 1;
        }
    }
    if piece_start < chars.len() {
        out.extend(chars[piece_start..].iter());
    }
}

/// Line- and block-comment syntax for a language, used only to detect a leading
/// file-header comment block. `.0` = line-comment prefixes; `.1` = `(open, close)`
/// for a block comment. Empty / `None` where the language has no such form (and
/// where a header split makes no sense — Markdown is all prose, JSON has no
/// comments), which disables the split for that language.
fn comment_syntax(
    language: crate::repo_scan::Language,
) -> (
    &'static [&'static str],
    Option<(&'static str, &'static str)>,
) {
    use crate::repo_scan::Language;
    match language {
        Language::Rust
        | Language::TypeScript
        | Language::JavaScript
        | Language::Go
        | Language::C
        | Language::Cpp
        | Language::Java => (&["//"], Some(("/*", "*/"))),
        Language::Php => (&["//", "#"], Some(("/*", "*/"))),
        Language::Python | Language::Ruby | Language::Bash | Language::Yaml | Language::Toml => {
            (&["#"], None)
        }
        Language::Css => (&[], Some(("/*", "*/"))),
        Language::Html => (&[], Some(("<!--", "-->"))),
        // Markdown / JSON / PlainText / unknown: no header split.
        _ => (&[], None),
    }
}

/// Last line (1-indexed, inclusive) of the leading comment block at the very top
/// of `source`, or `0` when there is nothing to split off. Returns `0` unless ALL
/// of these hold:
///   * the file opens with a comment section spanning at least
///     [`MIN_FILE_HEADER_LINES`] comment lines (line comments, or a `/* … */`
///     block; blank lines interleaved with the comments are permitted and belong
///     to the block but do not extend its end), and
///   * a real code line follows the block — a file that is nothing but comments
///     has no "header vs. body" split to make.
///
/// The block ends at its LAST comment line (trailing blanks are excluded).
pub(crate) fn file_header_end(source: &[u8], language: crate::repo_scan::Language) -> u32 {
    let (line_prefixes, block) = comment_syntax(language);
    if line_prefixes.is_empty() && block.is_none() {
        return 0;
    }
    let block_close = block.map_or("", |(_, c)| c);
    let text = String::from_utf8_lossy(source);

    let mut last_comment: u32 = 0;
    let mut comment_lines: u32 = 0;
    let mut in_block = false;
    let mut code_follows = false;

    for (i, raw) in text.split('\n').enumerate() {
        let lineno = (i + 1) as u32;
        let t = raw.trim();
        if in_block {
            last_comment = lineno;
            comment_lines += 1;
            if t.contains(block_close) {
                in_block = false;
            }
            continue;
        }
        if t.is_empty() {
            // A blank line inside the leading region is tolerated (doc headers
            // often separate paragraphs) but does not extend the block's end.
            continue;
        }
        if line_prefixes.iter().any(|p| t.starts_with(p)) {
            last_comment = lineno;
            comment_lines += 1;
            continue;
        }
        if let Some((open, close)) = block {
            if t.starts_with(open) {
                in_block = true;
                last_comment = lineno;
                comment_lines += 1;
                // A single-line block comment (`/* … */`) closes on its own line.
                if t.contains(close) {
                    in_block = false;
                }
                continue;
            }
        }
        // First real code line — the header block ends before it.
        code_follows = true;
        break;
    }

    if !code_follows || comment_lines < MIN_FILE_HEADER_LINES {
        return 0;
    }
    last_comment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::Language;

    // ── Long-line splitting failsafe ──────────────────────────────────────────

    #[test]
    fn split_long_lines_noop_on_normal_file() {
        let src = b"fn a() {}\nfn b() {}\n";
        assert_eq!(split_long_lines(src), src.to_vec());
    }

    #[test]
    fn split_long_lines_breaks_a_minified_line_under_the_cap() {
        // One long line of statements — every emitted line must fit the cap.
        let one = format!("{}\n", "let x=1; ".repeat(60)); // ~540 chars, one line
        let out = split_long_lines(one.as_bytes());
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains('\n'));
        for line in text.split('\n') {
            assert!(
                line.chars().count() <= MAX_LINE_CHARS,
                "line exceeds cap ({}): {line:?}",
                line.chars().count()
            );
        }
    }

    #[test]
    fn split_long_lines_does_not_break_inside_a_string() {
        // A break must never land inside the quoted run. Build a line where the only
        // delimiters sit inside a string, forcing the splitter to either break
        // outside it or hard-clip — never mid-quote at a `;` that is inside quotes.
        let s = format!("const A = \"{}\"; const B = 1;\n", "a; b; c; ".repeat(30));
        let out = split_long_lines(s.as_bytes());
        let text = String::from_utf8(out).unwrap();
        // Every piece still under the cap.
        for line in text.split('\n') {
            assert!(line.chars().count() <= MAX_LINE_CHARS);
        }
        // The quoted payload's characters are all preserved (content not lost).
        let stripped: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let orig: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            stripped, orig,
            "splitting must preserve every non-whitespace char"
        );
    }

    #[test]
    fn split_long_lines_hard_clips_an_unbreakable_string() {
        // A single giant string literal with no safe break inside — must still be
        // clipped so no line exceeds the cap.
        let s = format!("x=\"{}\"\n", "a".repeat(1000));
        let out = split_long_lines(s.as_bytes());
        let text = String::from_utf8(out).unwrap();
        for line in text.split('\n') {
            assert!(
                line.chars().count() <= MAX_LINE_CHARS,
                "unbreakable line not clipped: {} chars",
                line.chars().count()
            );
        }
    }

    // ── File-header split ─────────────────────────────────────────────────────

    #[test]
    fn file_header_end_detects_line_comment_block() {
        assert_eq!(file_header_end(b"// a\n// b\ncode\n", Language::Rust), 2);
        assert_eq!(
            file_header_end(
                b"//! doc 1\n//! doc 2\n\nuse x;\nfn a(){}\n",
                Language::Rust
            ),
            2
        );
    }

    #[test]
    fn file_header_end_detects_block_comment() {
        // Multi-line `/* … */` header: ends on the closing-delimiter line.
        assert_eq!(
            file_header_end(
                b"/*\n * Copyright.\n * MIT.\n */\nfn a(){}\n",
                Language::Rust
            ),
            4
        );
    }

    #[test]
    fn file_header_end_zero_without_code_after() {
        // A file that is nothing but comments has no header/body split.
        assert_eq!(
            file_header_end(b"//! doc 1\n//! doc 2\n", Language::Rust),
            0
        );
    }

    #[test]
    fn file_header_end_zero_below_floor() {
        // A lone one-line comment is not a section.
        assert_eq!(file_header_end(b"// lone\nfn a(){}\n", Language::Rust), 0);
    }

    #[test]
    fn file_header_end_zero_when_file_opens_with_code() {
        assert_eq!(
            file_header_end(b"fn a(){}\n// mid-file\n", Language::Rust),
            0
        );
    }

    #[test]
    fn file_header_end_zero_for_languages_without_the_form() {
        // Markdown / JSON carry no code-comment header form.
        assert_eq!(file_header_end(b"# Title\n\nbody\n", Language::Markdown), 0);
        assert_eq!(file_header_end(b"{\n  \"a\": 1\n}\n", Language::Json), 0);
    }
}
