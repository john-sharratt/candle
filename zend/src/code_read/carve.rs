//! Language dispatch + tier-fallback policy.
//!
//! Order of attempt:
//!   1. Language-specific tree-sitter parser (Rust / Python / TS / JS / Go).
//!   2. Header-based fallback (Markdown headers, YAML/TOML/JSON
//!      top-level keys).
//!   3. Fixed-window fallback (100-line chunks).
//!
//! A parser is allowed to return `Some(vec![])` — that means "I parsed
//! it but it contains nothing carve-worthy"; we fall through to the
//! header fallback so we still emit something for the file.

use crate::code_read::parsers::{
    bash, c, cpp, css, fallback, go, html, java, javascript, markdown, php, python, ruby, rust,
    structured_config, typescript,
};
use crate::code_read::types::{
    ChunkKind, Scope, MAX_LINE_CHARS, MAX_SCOPE_CHARS, MAX_SCOPE_LINES, MIN_FILE_HEADER_LINES,
    MIN_SCOPE_LINES, SOFT_LINE_CHARS,
};
use crate::repo_scan::Language;

/// Carve `source` for a file with the given `language` and `is_tsx`
/// (used to pick the TSX dialect of the TS grammar).  Never panics;
/// always returns at least one scope unless the source is empty.
///
/// **Content preservation invariant**: every line of `source` is
/// covered by at least one returned scope.  When the language-aware
/// tier-1 parser only partially succeeds (or skips top-level items
/// like module-level statements that aren't function/type
/// definitions), [`fill_gaps`] adds `Fallback` scopes for every
/// uncovered run of lines so no source content is ever dropped from
/// the prefill pass.  This is load-bearing — a file's content can't
/// be retrieved from the repo_map / code_reading layers if it never
/// reached either of them.
pub fn carve(source: &[u8], language: Language, is_tsx: bool) -> Vec<Scope> {
    if source.is_empty() {
        return Vec::new();
    }
    let total_lines = count_lines(source);
    let raw = carve_raw(source, language, is_tsx);
    if raw.is_empty() {
        // `carve_raw` only returns empty for empty input (fallback always covers a
        // non-empty file), but guard so `refine`/`fill_gaps` never see nothing.
        return Vec::new();
    }
    // fill_gaps → split the file-header comment block off as its own first turn →
    // refine (drop blank runs + merge small scopes forward) → cap.
    // Fill-gaps is a no-op on the already-contiguous fallback output; the header
    // split is a no-op unless the file opens with a multi-line comment section.
    let filled = fill_gaps(raw, total_lines);
    let header_end = file_header_end(source, language);
    let staged = if header_end > 0 {
        split_file_header(filled, header_end)
    } else {
        filled
    };
    cap_scope_size(refine(staged, source), source)
}

/// The RAW tier-1 / tier-2 / fallback scopes, BEFORE gap-fill, refine, and capping.
///
/// Exposed for parser unit tests that verify per-scope EXTRACTION: the dispatcher's
/// [`refine`] pass merges small scopes forward, which on a small test fixture would
/// collapse every item into one chunk and hide the parser's per-item boundaries.
/// Production carving always goes through [`carve`].
pub fn carve_raw(source: &[u8], language: Language, is_tsx: bool) -> Vec<Scope> {
    if source.is_empty() {
        return Vec::new();
    }
    let tier1 = match language {
        Language::Rust => rust::carve(source),
        Language::Python => python::carve(source),
        Language::TypeScript => typescript::carve(source, is_tsx),
        Language::JavaScript => javascript::carve(source),
        Language::Go => go::carve(source),
        Language::C => c::carve(source),
        Language::Cpp => cpp::carve(source),
        Language::Java => java::carve(source),
        Language::Ruby => ruby::carve(source),
        Language::Php => php::carve(source),
        Language::Bash => bash::carve(source),
        Language::Html => html::carve(source),
        Language::Css => css::carve(source),
        _ => None,
    };
    if let Some(scopes) = tier1 {
        if !scopes.is_empty() {
            return scopes;
        }
    }
    let tier2 = match language {
        Language::Markdown => Some(markdown::carve(source)),
        Language::Yaml | Language::Toml | Language::Json => Some(structured_config::carve(source)),
        _ => None,
    };
    if let Some(scopes) = tier2 {
        if !scopes.is_empty() {
            return scopes;
        }
    }
    fallback::carve(source)
}

/// Cumulative character count by 1-indexed line: `prefix[l]` is the total number
/// of characters (Unicode scalar values, `\n` counted as one) in lines `1..=l`.
/// `prefix[0] == 0`, so the char span of an inclusive line range `[a, b]` is
/// `prefix[b] - prefix[a - 1]`. Used by [`refine`] and [`cap_scope_size`] to
/// enforce [`MAX_SCOPE_CHARS`] without re-slicing the source per query.
fn line_char_prefix(source: &[u8]) -> Vec<u32> {
    let text = String::from_utf8_lossy(source);
    let mut prefix = vec![0u32];
    let mut acc = 0u32;
    for line in text.split('\n') {
        acc = acc.saturating_add(line.chars().count() as u32 + 1);
        prefix.push(acc);
    }
    prefix
}

/// Character span of the inclusive 1-indexed line range `[start, end]` from a
/// [`line_char_prefix`] table, clamped so an out-of-range line never panics.
fn span_chars(prefix: &[u32], start: u32, end: u32) -> u32 {
    let last = prefix.last().copied().unwrap_or(0);
    let hi = prefix.get(end as usize).copied().unwrap_or(last);
    let lo = prefix
        .get(start.saturating_sub(1) as usize)
        .copied()
        .unwrap_or(0);
    hi.saturating_sub(lo)
}

/// Split any scope exceeding [`MAX_SCOPE_LINES`] lines OR [`MAX_SCOPE_CHARS`]
/// characters into `[part N]` windows at line boundaries, preserving order and
/// coverage.
///
/// The cap is enforced HERE, over the final scope list, rather than in each
/// parser. Tier-1 splits oversize nodes on child boundaries (a semantic split,
/// better labels — see `tree_sitter_util::emit_split`), but that only bounds
/// nodes that *have* children to split on. Every other producer was unbounded:
/// [`fill_gaps`] emits one scope per uncovered run however long, the Markdown and
/// YAML/TOML/JSON header carves emit one scope per section/top-level key, and a
/// childless tree-sitter leaf can exceed the cap on its own. A single oversize
/// scope becomes one enormous turn, which is what drives a turn past the
/// per-token signature limit and dilutes its provenance across the whole span.
///
/// By this point [`refine`] has already ended multi-member scopes on whole
/// sub-scope boundaries within both caps, so anything reaching here over-budget is
/// a SINGLE sub-scope too big on its own (a giant function, a generated table, a
/// minified blob) — the unavoidable case. It is split at line boundaries (never
/// mid-line: `split_long_lines` already bounded every line to [`MAX_LINE_CHARS`],
/// so one line always fits [`MAX_SCOPE_CHARS`] and each part carries ≥ 1 line).
fn cap_scope_size(scopes: Vec<Scope>, source: &[u8]) -> Vec<Scope> {
    let prefix = line_char_prefix(source);
    let over_cap = |s: &Scope| {
        s.end_line.saturating_sub(s.start_line) + 1 > MAX_SCOPE_LINES
            || span_chars(&prefix, s.start_line, s.end_line) > MAX_SCOPE_CHARS
    };
    if !scopes.iter().any(over_cap) {
        return scopes; // common case: nothing to split, no reallocation
    }
    let mut out = Vec::with_capacity(scopes.len() + 8);
    for scope in scopes {
        if !over_cap(&scope) {
            out.push(scope);
            continue;
        }
        let mut part = 1usize;
        let mut chunk_start = scope.start_line;
        while chunk_start <= scope.end_line {
            // Grow the window line by line until the next line would break either
            // cap; always take at least the first line so the walk terminates.
            let mut chunk_end = chunk_start;
            while chunk_end < scope.end_line {
                let next = chunk_end + 1;
                let lines = next - chunk_start + 1;
                if lines > MAX_SCOPE_LINES
                    || span_chars(&prefix, chunk_start, next) > MAX_SCOPE_CHARS
                {
                    break;
                }
                chunk_end = next;
            }
            let mut labelled = scope.path.clone();
            if let Some(last) = labelled.last_mut() {
                *last = format!("{last} [part {part}]");
            }
            out.push(Scope {
                path: labelled,
                kind: scope.kind,
                start_line: chunk_start,
                end_line: chunk_end,
            });
            chunk_start = chunk_end + 1;
            part += 1;
        }
    }
    out
}

/// Rewrite `source` so no line exceeds [`MAX_LINE_CHARS`], inserting `\n` at safe
/// division points so a minified / one-line file becomes a normal multi-line
/// document the line-based carve can chunk sensibly.
///
/// This runs at READ time, before carving, so the carved scopes AND the source the
/// turn shows the model are the same split text. A line already within the cap is
/// untouched; a file with no over-long line is returned unchanged. Split points are
/// chosen **outside string literals** (single/double/backtick, with `\` escapes) so
/// we never cut a token in half if we can help it — preferring a delimiter
/// (`;,)}]>` or whitespace) once a piece passes [`SOFT_LINE_CHARS`], and
/// hard-clipping at [`MAX_LINE_CHARS`] when a line (a giant string blob) offers no
/// safe break. Operates on chars, so a split never lands mid-UTF-8.
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

/// Count `source`'s logical line count — matches the semantics used
/// by the per-scope `start_line` / `end_line` values (1-indexed
/// inclusive ranges over lines split on `\n`).
fn count_lines(source: &[u8]) -> u32 {
    if source.is_empty() {
        return 0;
    }
    let nl_count = source.iter().filter(|&&b| b == b'\n').count() as u32;
    // Source without a trailing newline still has one final line for
    // its un-terminated tail.
    if source.last() == Some(&b'\n') {
        nl_count
    } else {
        nl_count + 1
    }
}

/// Insert `Fallback` scopes covering every line in `[1, total_lines]`
/// that isn't already covered by some scope in `scopes`.  Returns the
/// merged list sorted by `start_line` ascending.
///
/// "Uncovered" is computed by sweeping the input scopes' line ranges
/// in sorted order and emitting a fill scope for each gap.  Gaps are
/// merged greedily so a 50-line stretch of uncovered code produces a
/// single fill scope rather than fifty single-line scopes.  Empty
/// input is treated as "nothing covered" — the whole file becomes one
/// fill scope.
pub fn fill_gaps(mut scopes: Vec<Scope>, total_lines: u32) -> Vec<Scope> {
    if total_lines == 0 {
        return scopes;
    }
    // Sort by start_line; ties broken by end_line so smaller scopes
    // come first.  Sort is stable so input order is preserved within
    // equal ranges (useful when the parser deliberately emits two
    // scopes with the same range — e.g. a typedef-as-struct that
    // covers both an enclosing typedef declarator and the inner
    // struct body).
    scopes.sort_by_key(|s| (s.start_line, s.end_line));

    let mut filled: Vec<Scope> = Vec::with_capacity(scopes.len() + 4);
    let mut cursor: u32 = 1;
    let mut gap_idx: usize = 1;

    for scope in &scopes {
        if scope.start_line > cursor {
            // Uncovered run from `cursor` through `scope.start_line - 1`.
            filled.push(make_fill(cursor, scope.start_line - 1, gap_idx));
            gap_idx += 1;
        }
        filled.push(scope.clone());
        if scope.end_line >= cursor {
            cursor = scope.end_line + 1;
        }
    }
    if cursor <= total_lines {
        filled.push(make_fill(cursor, total_lines, gap_idx));
    }
    filled
}

fn make_fill(start: u32, end: u32, idx: usize) -> Scope {
    Scope {
        path: vec![format!("uncovered {idx}")],
        kind: ChunkKind::Fallback,
        start_line: start,
        end_line: end,
    }
}

/// Line- and block-comment syntax for a language, used only to detect a leading
/// file-header comment block. `.0` = line-comment prefixes; `.1` = `(open, close)`
/// for a block comment. Empty / `None` where the language has no such form (and
/// where a header split makes no sense — Markdown is all prose, JSON has no
/// comments), which disables the split for that language.
fn comment_syntax(
    language: Language,
) -> (
    &'static [&'static str],
    Option<(&'static str, &'static str)>,
) {
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
/// The block ends at its LAST comment line (trailing blanks are excluded). This is
/// deliberately language-aware but line-based: it runs uniformly across every tier
/// (tree-sitter, header, fallback) in one place, so the first turn of any file is
/// the file's own overview rather than its first function — including when the
/// tree-sitter walk already attached the header comments to that function via
/// `start_with_leading_comments`, which [`split_file_header`] then peels back off.
pub(crate) fn file_header_end(source: &[u8], language: Language) -> u32 {
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

/// Split the leading comment block (`[1, header_end]`) off as its own
/// [`ChunkKind::FileHeader`] first scope, trimming it out of whatever scope(s)
/// currently cover those lines. `scopes` must be the gap-filled, `start_line`-sorted
/// list (contiguous from line 1), so the result stays fully covered: the header
/// scope owns `[1, header_end]`, scopes wholly inside it are dropped, and the scope
/// straddling the boundary has its start pulled down to `header_end + 1`.
fn split_file_header(scopes: Vec<Scope>, header_end: u32) -> Vec<Scope> {
    let mut out = Vec::with_capacity(scopes.len() + 1);
    out.push(Scope {
        path: vec!["file header".to_string()],
        kind: ChunkKind::FileHeader,
        start_line: 1,
        end_line: header_end,
    });
    for s in scopes {
        if s.end_line <= header_end {
            // Wholly inside the header block — its lines are covered by the header.
            continue;
        }
        if s.start_line <= header_end {
            out.push(Scope {
                start_line: header_end + 1,
                ..s
            });
        } else {
            out.push(s);
        }
    }
    out
}

/// `map[l]` is true iff 1-indexed line `l` is empty or whitespace-only. Index 0 is
/// an unused placeholder so the map indexes by line number directly.
fn blank_line_map(source: &[u8]) -> Vec<bool> {
    let text = String::from_utf8_lossy(source);
    let mut map = vec![false];
    for line in text.split('\n') {
        map.push(line.trim().is_empty());
    }
    map
}

/// Post-carve granularity pass over a fully gap-filled, sorted scope list:
///
/// 1. **Drop pure-blank runs** — a scope whose every line is whitespace carries no
///    content and makes a useless standalone turn, so it is skipped entirely (the
///    coverage invariant is intentionally relaxed for blank lines only).
/// 2. **Merge small scopes forward** — a scope shorter than [`MIN_SCOPE_LINES`]
///    absorbs the following scopes (real items and the small blank/gap runs between
///    them) until it reaches that width, never past [`MAX_SCOPE_LINES`] lines or
///    [`MAX_SCOPE_CHARS`] characters. This groups single `const`s, one-line
///    `type`s, and tiny helper `fn`s so the split points stay real functions and
///    types — small items are only ever absorbed, never split — and every emitted
///    scope is a meaty, well-provenanced chunk.
///
/// Both upper bounds are enforced against WHOLE sub-scopes: a merge stops at the
/// last sub-scope that still fits, clipping the next section into the following
/// turn rather than cutting it in half. So a scope never ends mid-function because
/// a char budget ran out partway through it — only a single sub-scope that is
/// itself over-budget is ever split (by [`cap_scope_size`], at line boundaries).
fn refine(scopes: Vec<Scope>, source: &[u8]) -> Vec<Scope> {
    if scopes.is_empty() {
        return scopes;
    }
    let blank = blank_line_map(source);
    let is_blank_only = |s: &Scope| {
        (s.start_line..=s.end_line).all(|l| blank.get(l as usize).copied().unwrap_or(false))
    };
    let prefix = line_char_prefix(source);

    let mut out: Vec<Scope> = Vec::with_capacity(scopes.len());
    let mut i = 0;
    while i < scopes.len() {
        // A standalone blank run is dropped outright (no turn for whitespace).
        if is_blank_only(&scopes[i]) {
            i += 1;
            continue;
        }
        // The file-header comment block is always its own turn: never merged
        // forward into the first function (that is the whole point of splitting
        // it) and never below-MIN-merged away, however short it is.
        if scopes[i].kind == ChunkKind::FileHeader {
            out.push(scopes[i].clone());
            i += 1;
            continue;
        }
        // Accumulate from this content scope forward until it is wide enough.
        let start = scopes[i].start_line;
        let mut end = scopes[i].end_line;
        let mut members: Vec<&Scope> = vec![&scopes[i]];
        i += 1;
        while end.saturating_sub(start) + 1 < MIN_SCOPE_LINES && i < scopes.len() {
            let nxt = &scopes[i];
            // `max`: a nested/overlapping scope (an inner item emitted after its
            // encloser) can END BEFORE the running span — never let the span shrink
            // or lines after the inner scope are dropped from coverage.
            let new_end = end.max(nxt.end_line);
            // Never merge past either hard cap — end the scope on this whole
            // sub-scope boundary and let the next one start the following turn. An
            // over-cap SINGLE sub-scope is split later by `cap_scope_size`.
            if new_end.saturating_sub(start) + 1 > MAX_SCOPE_LINES
                || span_chars(&prefix, start, new_end) > MAX_SCOPE_CHARS
            {
                break;
            }
            end = new_end;
            // Blank runs between items are absorbed into the span but never named.
            if !is_blank_only(nxt) {
                members.push(nxt);
            }
            i += 1;
        }
        out.push(merge_members(start, end, &members));
    }
    out
}

/// Build the merged scope for a run of absorbed content scopes. The span is
/// `[start, end]`; the label keeps the first member's nesting and, when more than
/// one item merged, names the range as `first … last` so the chunk is still legible.
fn merge_members(start: u32, end: u32, members: &[&Scope]) -> Scope {
    let first = members[0];
    let path = if members.len() == 1 {
        first.path.clone()
    } else {
        // Name every grouped item on the merged chunk's leaf so a query can still
        // surface it by name; truncate a long run so the label stays readable.
        let leaves: Vec<String> = members
            .iter()
            .filter_map(|s| s.path.last().cloned())
            .collect();
        let combined = if leaves.len() <= 4 {
            leaves.join(", ")
        } else {
            format!("{}, … (+{} more)", leaves[..3].join(", "), leaves.len() - 3)
        };
        let mut p = first.path.clone();
        if let Some(leaf) = p.last_mut() {
            *leaf = combined;
        }
        p
    };
    Scope {
        path,
        kind: first.kind,
        start_line: start,
        end_line: end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_dispatches_to_tree_sitter() {
        let src = b"fn alpha() {}\n";
        let scopes = carve(src, Language::Rust, false);
        assert!(!scopes.is_empty());
        assert!(scopes
            .iter()
            .any(|s| s.path.last().is_some_and(|n| n == "fn alpha")));
    }

    #[test]
    fn markdown_dispatches_to_header_carver() {
        let src = b"## hello\n\nworld\n";
        let scopes = carve(src, Language::Markdown, false);
        assert!(!scopes.is_empty());
        assert!(scopes.iter().any(|s| s.path[0].contains("hello")));
    }

    #[test]
    fn toml_dispatches_to_structured_carver() {
        let src = b"[package]\nname = \"foo\"\n";
        let scopes = carve(src, Language::Toml, false);
        assert!(!scopes.is_empty());
        assert!(scopes.iter().any(|s| s.path[0] == "package"));
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(carve(b"", Language::Rust, false).is_empty());
    }

    #[test]
    fn fallback_for_text_files() {
        let src = b"just plain text\n";
        let scopes = carve(src, Language::PlainText, false);
        assert!(!scopes.is_empty());
    }

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

    // ── Content-preservation invariant ───────────────────────────────────────

    /// The all-important guarantee: every line of source survives the
    /// carve pass, sitting in at least one returned scope.
    fn every_line_covered(scopes: &[Scope], total: u32) -> bool {
        let mut cursor = 1u32;
        let mut sorted = scopes.to_vec();
        sorted.sort_by_key(|s| (s.start_line, s.end_line));
        for s in &sorted {
            if s.start_line > cursor {
                return false;
            }
            if s.end_line >= cursor {
                cursor = s.end_line + 1;
            }
        }
        cursor > total
    }

    #[test]
    fn fill_gaps_covers_leading_uncovered_lines() {
        // Scope at lines 5-10 leaves 1-4 uncovered.
        let scope = Scope {
            path: vec!["fn x".into()],
            kind: ChunkKind::Function,
            start_line: 5,
            end_line: 10,
        };
        let filled = fill_gaps(vec![scope], 10);
        assert!(every_line_covered(&filled, 10));
        assert!(filled.iter().any(|s| s.start_line == 1 && s.end_line == 4));
    }

    #[test]
    fn fill_gaps_covers_trailing_uncovered_lines() {
        let scope = Scope {
            path: vec!["fn x".into()],
            kind: ChunkKind::Function,
            start_line: 1,
            end_line: 5,
        };
        let filled = fill_gaps(vec![scope], 10);
        assert!(every_line_covered(&filled, 10));
        assert!(filled.iter().any(|s| s.start_line == 6 && s.end_line == 10));
    }

    #[test]
    fn fill_gaps_covers_middle_gaps() {
        let scopes = vec![
            Scope {
                path: vec!["fn a".into()],
                kind: ChunkKind::Function,
                start_line: 1,
                end_line: 5,
            },
            Scope {
                path: vec!["fn b".into()],
                kind: ChunkKind::Function,
                start_line: 15,
                end_line: 20,
            },
        ];
        let filled = fill_gaps(scopes, 20);
        assert!(every_line_covered(&filled, 20));
        assert!(filled.iter().any(|s| s.start_line == 6 && s.end_line == 14));
    }

    #[test]
    fn fill_gaps_no_op_when_already_fully_covered() {
        let scopes = vec![Scope {
            path: vec!["fn a".into()],
            kind: ChunkKind::Function,
            start_line: 1,
            end_line: 10,
        }];
        let filled = fill_gaps(scopes.clone(), 10);
        assert_eq!(filled, scopes);
    }

    #[test]
    fn fill_gaps_handles_overlapping_scopes() {
        // Tree-sitter sometimes emits nested scopes that overlap —
        // gap-fill should treat the outer one as covering the inner
        // range without inserting a spurious fill.
        let scopes = vec![
            Scope {
                path: vec!["outer".into()],
                kind: ChunkKind::TypeDefinition,
                start_line: 1,
                end_line: 20,
            },
            Scope {
                path: vec!["inner".into()],
                kind: ChunkKind::Function,
                start_line: 5,
                end_line: 10,
            },
        ];
        let filled = fill_gaps(scopes, 20);
        assert!(every_line_covered(&filled, 20));
        // No gaps inserted between or after the overlapping pair.
        assert_eq!(
            filled
                .iter()
                .filter(|s| s.path[0].starts_with("uncovered"))
                .count(),
            0
        );
    }

    #[test]
    fn carve_rust_preserves_module_level_use_statements() {
        // Without gap-fill the `use` declarations at lines 1-3 would
        // be dropped because tree-sitter-rust isn't registered for
        // them.  Gap-fill ensures every line including the imports
        // shows up in some scope.
        let src = br#"use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;

fn alpha() { let _ = (HashMap::<String, u32>::new(), Arc::new(0), Path::new(".")); }
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(
            every_line_covered(&scopes, total),
            "module-level use statements must survive carve: {:?}",
            scopes
                .iter()
                .map(|s| (s.start_line, s.end_line))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn carve_rust_preserves_top_level_let_in_invalid_position() {
        // Malformed Rust (top-level `let`) — tree-sitter still parses
        // most of the file but the `let` ends up in an error node.
        // Gap-fill keeps the invalid region in the prefill so the
        // model can still see what's there.
        let src = br#"let oops = 1;

fn good() { let _ = 0; }
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(
            every_line_covered(&scopes, total),
            "invalid top-level region must survive carve: {:?}",
            scopes
        );
    }

    #[test]
    fn carve_python_preserves_module_imports_and_globals() {
        let src = br#"import os
import sys
from pathlib import Path

X = 42
Y = "hello"

def alpha():
    return X
"#;
        let scopes = carve(src, Language::Python, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_typescript_preserves_module_imports_and_exports() {
        let src = br#"import { Foo } from './foo';
import * as utils from './utils';

export const MAX = 100;

export function bar(): number { return MAX; }
"#;
        let scopes = carve(src, Language::TypeScript, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_go_preserves_package_and_import_block() {
        let src = br#"package main

import (
    "fmt"
    "os"
)

func main() {
    fmt.Println("hi")
    os.Exit(0)
}
"#;
        let scopes = carve(src, Language::Go, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_completely_invalid_file_without_loss() {
        // Garbled bytes — no parser succeeds, the carver falls
        // through to the fixed-window fallback which covers the whole
        // file by construction.
        let src = b"$$$ this is not valid in any language $$$\n@@@@@@@\n!!!!\n";
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_truncated_rust_fn() {
        // Function with no closing brace — tree-sitter recovers as
        // much as it can; the trailing-incomplete region is still
        // covered via gap-fill.
        let src = br#"fn alpha() {
    let x = 1;
    let y = 2;
"#;
        let scopes = carve(src, Language::Rust, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    #[test]
    fn carve_handles_invalid_json_without_loss() {
        let src = b"{ \"key\":\n  \"unterminated\n";
        let scopes = carve(src, Language::Json, false);
        let total = count_lines(src);
        assert!(every_line_covered(&scopes, total));
    }

    /// No tier may emit a scope longer than `MAX_SCOPE_LINES`, and capping must
    /// preserve full line coverage. The gap-fill path was the unbounded one: a
    /// file tier-1 parses successfully but that leaves a huge unclaimed region
    /// (a generated table, a long data literal) yielded ONE scope spanning the
    /// whole run, which becomes one enormous turn.
    #[test]
    fn no_tier_emits_a_scope_over_the_line_cap() {
        let big = (MAX_SCOPE_LINES as usize) * 3 + 17;
        let assert_capped = |scopes: &[Scope], total: u32, what: &str| {
            assert!(
                every_line_covered(scopes, total),
                "{what}: coverage preserved"
            );
            for s in scopes {
                let lines = s.end_line - s.start_line + 1;
                assert!(
                    lines <= MAX_SCOPE_LINES,
                    "{what}: scope {:?} spans {lines} lines (cap {MAX_SCOPE_LINES})",
                    s.path
                );
            }
        };

        // Gap-fill: a tiny fn tier-1 recognises, then a long uncovered tail.
        let mut rust = String::from("fn a() {}\n");
        for i in 0..big {
            rust.push_str(&format!("const K{i}: u32 = {i};\n"));
        }
        let scopes = carve(rust.as_bytes(), Language::Rust, false);
        assert_capped(&scopes, count_lines(rust.as_bytes()), "rust gap-fill");

        // Tier-2 markdown: one section with a very long body.
        let mut md = String::from("# only section\n");
        for i in 0..big {
            md.push_str(&format!("line {i}\n"));
        }
        let scopes = carve(md.as_bytes(), Language::Markdown, false);
        assert_capped(&scopes, count_lines(md.as_bytes()), "markdown section");

        // Tier-2 structured config: one enormous top-level key.
        let mut yaml = String::from("root:\n");
        for i in 0..big {
            yaml.push_str(&format!("  k{i}: {i}\n"));
        }
        let scopes = carve(yaml.as_bytes(), Language::Yaml, false);
        assert_capped(&scopes, count_lines(yaml.as_bytes()), "yaml top-level key");
    }

    // ── File-header split (first turn = file overview, not first function) ────

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

    #[test]
    fn carve_splits_rust_module_doc_into_its_own_first_turn() {
        let src = b"//! Module does X.\n//! It also does Y.\n\nuse std::fmt;\n\nfn alpha() { let _ = 1; }\n";
        let scopes = carve(src, Language::Rust, false);
        // Turn 1 is the file header, covering exactly the doc block.
        assert_eq!(scopes[0].kind, ChunkKind::FileHeader);
        assert_eq!(scopes[0].start_line, 1);
        assert_eq!(scopes[0].end_line, 2);
        // The first function lives in a LATER scope — never inside the header.
        assert!(scopes
            .iter()
            .skip(1)
            .any(|s| s.path.last().is_some_and(|p| p.contains("alpha"))));
        assert!(scopes
            .iter()
            .all(|s| s.start_line >= 3 || s.kind == ChunkKind::FileHeader));
        assert!(every_line_covered(&scopes, count_lines(src)));
    }

    #[test]
    fn carve_peels_header_back_off_a_function_it_was_attached_to() {
        // Comments sit DIRECTLY above the fn (no blank), so the tree-sitter walk's
        // `start_with_leading_comments` folds them into the fn's scope. The header
        // split must still peel them back out into turn 1.
        let src = b"//! File doc line 1.\n//! File doc line 2.\nfn alpha() { let _ = 1; }\n";
        let scopes = carve(src, Language::Rust, false);
        assert_eq!(scopes[0].kind, ChunkKind::FileHeader);
        assert_eq!(scopes[0].end_line, 2);
        assert!(scopes
            .iter()
            .any(|s| s.start_line == 3 && s.path.last().is_some_and(|p| p.contains("alpha"))));
        assert!(every_line_covered(&scopes, count_lines(src)));
    }

    #[test]
    fn carve_splits_python_hash_header() {
        let src =
            b"# Tool for X.\n# Handles Y.\n\nimport os\n\ndef run():\n    return os.getpid()\n";
        let scopes = carve(src, Language::Python, false);
        assert_eq!(scopes[0].kind, ChunkKind::FileHeader);
        assert_eq!(scopes[0].end_line, 2);
        assert!(every_line_covered(&scopes, count_lines(src)));
    }

    #[test]
    fn carve_does_not_split_a_single_line_comment() {
        let src = b"// lone comment\nfn a() { let _ = 0; }\n";
        let scopes = carve(src, Language::Rust, false);
        assert!(scopes.iter().all(|s| s.kind != ChunkKind::FileHeader));
        assert!(every_line_covered(&scopes, count_lines(src)));
    }

    // ── Char-budget clean breaks ─────────────────────────────────────────────

    #[test]
    fn refine_breaks_on_char_budget_at_member_boundary() {
        // Five 10-line members of maximally dense (160-char) lines. Reaching the
        // 50-line MIN would cost 50 × 161 = 8050 chars — over MAX_SCOPE_CHARS — so
        // the merge must stop on a whole-member edge before then, never mid-member.
        let dense = "x".repeat(160);
        let mut src = String::new();
        for _ in 0..50 {
            src.push_str(&dense);
            src.push('\n');
        }
        let members: Vec<Scope> = (0..5)
            .map(|m| Scope {
                path: vec![format!("fn m{m}")],
                kind: ChunkKind::Function,
                start_line: m * 10 + 1,
                end_line: m * 10 + 10,
            })
            .collect();
        let refined = refine(members, src.as_bytes());
        let prefix = line_char_prefix(src.as_bytes());
        assert!(
            refined.len() >= 2,
            "char budget should force at least one clean break, got {refined:?}"
        );
        for s in &refined {
            assert!(
                span_chars(&prefix, s.start_line, s.end_line) <= MAX_SCOPE_CHARS,
                "scope {:?} exceeds char budget",
                s.path
            );
            // Every boundary lands on a 10-line member edge — nothing cut in half.
            assert_eq!(
                (s.start_line - 1) % 10,
                0,
                "starts mid-member: {}",
                s.start_line
            );
            assert_eq!(s.end_line % 10, 0, "ends mid-member: {}", s.end_line);
        }
    }

    #[test]
    fn cap_scope_size_splits_a_single_dense_scope_at_line_boundaries() {
        // One 60-line function of 160-char lines: 60 × 161 = 9660 chars > the char
        // budget but only 60 lines (< the line cap). The unavoidable single-item
        // case — split by chars, at line boundaries, coverage preserved.
        let dense = "y".repeat(160);
        let mut src = String::new();
        for _ in 0..60 {
            src.push_str(&dense);
            src.push('\n');
        }
        let scope = Scope {
            path: vec!["fn big".into()],
            kind: ChunkKind::Function,
            start_line: 1,
            end_line: 60,
        };
        let out = cap_scope_size(vec![scope], src.as_bytes());
        assert!(out.len() >= 2, "dense single scope should split by chars");
        let prefix = line_char_prefix(src.as_bytes());
        for p in &out {
            assert!(
                span_chars(&prefix, p.start_line, p.end_line) <= MAX_SCOPE_CHARS,
                "part {:?} exceeds char budget",
                p.path
            );
        }
        // Contiguous, gapless coverage of the whole original span.
        assert_eq!(out.first().unwrap().start_line, 1);
        assert_eq!(out.last().unwrap().end_line, 60);
        for w in out.windows(2) {
            assert_eq!(w[0].end_line + 1, w[1].start_line, "parts not contiguous");
        }
    }

    #[test]
    fn cap_scope_size_leaves_ordinary_code_on_the_line_cap() {
        // Sparse ~20-char lines: the char budget never binds, so a long run still
        // splits on the 150-line cap exactly as before (no behavior change).
        let mut src = String::new();
        for i in 0..(MAX_SCOPE_LINES as usize * 2) {
            src.push_str(&format!("const K{i}: u32 = {i};\n"));
        }
        let scope = Scope {
            path: vec!["consts".into()],
            kind: ChunkKind::Constants,
            start_line: 1,
            end_line: MAX_SCOPE_LINES * 2,
        };
        let out = cap_scope_size(vec![scope], src.as_bytes());
        for p in &out {
            assert!(p.end_line - p.start_line < MAX_SCOPE_LINES);
        }
        assert_eq!(
            out.first().unwrap().end_line,
            MAX_SCOPE_LINES,
            "first window is a full 150 lines"
        );
    }

    #[test]
    fn carve_header_survives_refine_even_when_short() {
        // A 2-line header is well under MIN_SCOPE_LINES; it must NOT be merged
        // forward into the following body by the refine pass.
        let src = b"//! Tiny.\n//! Header.\nfn a() { let _ = 0; }\nfn b() { let _ = 1; }\n";
        let scopes = carve(src, Language::Rust, false);
        assert_eq!(scopes[0].kind, ChunkKind::FileHeader);
        assert_eq!(scopes[0].end_line, 2);
        // The header did not absorb the functions below it.
        assert!(scopes.len() >= 2);
    }
}
