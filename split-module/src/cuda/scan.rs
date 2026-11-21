//! A CUDA or C++ source cut into its top-level units: the declarations and
//! definitions a split can move, the preprocessor lines it cannot, and the
//! containers it can open.
//!
//! The scan is lexical. It skips comments, string and character literals and
//! preprocessor lines (with their `\` continuations), counts braces in code,
//! and ends a unit at a `;` or a closing `}` that brings it back to the level
//! it began at, or at the end of a preprocessor line — an `#if` running on to
//! its `#endif`. A unit owns the comments over it and the rest of its last
//! line. Nothing is parsed: the only promise is that a unit's bytes are whole —
//! no brace, comment, literal or `#if` begins in one unit and ends in another —
//! which is what moving units into an included file needs.

use std::ops::Range;

/// What a unit is, for what a split may do with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A declaration or definition: it moves.
    Code,
    /// `#include` or `#pragma once`: it stays in the file it was written in,
    /// so that every include path keeps the meaning it had.
    Kept,
    /// Any other directive (`#define`, `#undef`, `#pragma`): it moves.
    Directive,
    /// An `#if`, `#ifdef` or `#ifndef` through its `#endif`: it moves whole.
    Cond,
}

/// A unit as scanned: what it is, where it stops, its name, and the body of a
/// block that can be opened.
type Scanned = (Kind, usize, String, Option<Range<usize>>);

/// One top-level unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub kind: Kind,
    /// From its first comment line through the end of its last line.
    pub span: Range<usize>,
    /// What the map calls it: `fn quantize_q8_1`, `struct block_q4_0`,
    /// `#define WARP_SIZE`, `#if __CUDA_ARCH__ >= 800`.
    pub name: String,
    /// The first line of the comment over it.
    pub doc: String,
    /// For a block that can be opened — `extern "C" { }`, `namespace n { }`,
    /// an `#if` with no `#else` — where its body is: after the opening line
    /// or brace, before the closing `}` or `#endif` line.
    pub body: Option<Range<usize>>,
}

/// Every unit of `src[range]`, in order, with the whitespace between them left
/// out. `range` must start and end at the same brace and `#if` level.
pub fn units(src: &str, range: Range<usize>) -> Result<Vec<Unit>, String> {
    let b = src.as_bytes();
    let mut out: Vec<Unit> = Vec::new();
    let mut i = range.start;
    let end = range.end;
    loop {
        // Blank lines are gaps; comments belong to the unit after them.
        let mut lead: Option<usize> = None;
        let mut doc = String::new();
        loop {
            while i < end && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n') {
                i += 1;
            }
            if i + 1 < end && b[i] == b'/' && b[i + 1] == b'/' {
                let s = line_start(b, i);
                lead.get_or_insert(s);
                let e = line_end(b, i, end);
                if doc.is_empty() {
                    doc = comment_text(&src[i..e]);
                }
                i = e;
                continue;
            }
            if i + 1 < end && b[i] == b'/' && b[i + 1] == b'*' {
                let s = line_start(b, i);
                lead.get_or_insert(s);
                let e = src[i + 2..end]
                    .find("*/")
                    .map(|p| i + 2 + p + 2)
                    .ok_or("an unterminated block comment")?;
                if doc.is_empty() {
                    doc = comment_text(&src[i..e]);
                }
                i = e;
                continue;
            }
            break;
        }
        if i >= end {
            // Comments with nothing after them belong to the last unit.
            if let (Some(_), Some(last)) = (lead, out.last_mut()) {
                last.span.end = trimmed_end(b, end);
            } else if let Some(s) = lead {
                out.push(Unit {
                    kind: Kind::Directive,
                    span: s..trimmed_end(b, end),
                    name: "// comment".into(),
                    doc,
                    body: None,
                });
            }
            break;
        }
        // From the start of its line, or of its comment, but never before the
        // end of the unit before it: two units can share a line.
        let floor = out.last().map_or(range.start, |u| u.span.end);
        let start = lead.unwrap_or_else(|| line_start(b, i)).max(floor);
        let (kind, stop, name, body) = if is_directive_start(b, i) {
            directive_unit(src, i, end)?
        } else {
            code_unit(src, i, end)?
        };
        let stop = rest_of_line(b, stop, end);
        out.push(Unit {
            kind,
            span: start..stop,
            name,
            doc,
            body,
        });
        i = stop;
    }
    Ok(out)
}

/// The start of the line `at` is on.
fn line_start(b: &[u8], at: usize) -> usize {
    b[..at]
        .iter()
        .rposition(|&c| c == b'\n')
        .map_or(0, |p| p + 1)
}

/// Just past the newline ending the line `at` is on, or `end`.
fn line_end(b: &[u8], at: usize, end: usize) -> usize {
    b[at..end]
        .iter()
        .position(|&c| c == b'\n')
        .map_or(end, |p| at + p + 1)
}

/// `end` moved back over trailing whitespace, then on through its newline.
fn trimmed_end(b: &[u8], end: usize) -> usize {
    let mut e = end;
    while e > 0 && matches!(b[e - 1], b' ' | b'\t' | b'\r' | b'\n') {
        e -= 1;
    }
    line_end(b, e, end.max(e))
}

/// After `stop`: the rest of its line when it holds only spaces or a `//`
/// comment, through the newline.
fn rest_of_line(b: &[u8], stop: usize, end: usize) -> usize {
    if stop > 0 && b[stop - 1] == b'\n' {
        return stop;
    }
    let mut j = stop;
    while j < end && matches!(b[j], b' ' | b'\t' | b'\r') {
        j += 1;
    }
    if j < end && b[j] == b'\n' {
        return j + 1;
    }
    if j + 1 < end && b[j] == b'/' && b[j + 1] == b'/' {
        return line_end(b, j, end);
    }
    if j >= end {
        return end;
    }
    stop
}

/// A comment's first line of text, its markers taken off.
fn comment_text(c: &str) -> String {
    c.lines()
        .map(|l| {
            l.trim()
                .trim_start_matches("/**")
                .trim_start_matches("/*")
                .trim_start_matches("///")
                .trim_start_matches("//")
                .trim_start_matches('*')
                .trim_end_matches("*/")
                .trim()
        })
        .find(|l| !l.is_empty() && !l.chars().all(|c| "─━═-=*#~_".contains(c)))
        .unwrap_or("")
        .chars()
        .take(100)
        .collect()
}

/// Whether `i` is a `#` that opens a preprocessor line.
fn is_directive_start(b: &[u8], i: usize) -> bool {
    b[i] == b'#'
        && b[line_start(b, i)..i]
            .iter()
            .all(|c| matches!(c, b' ' | b'\t'))
}

/// The end of the preprocessor line starting at `i`, its continuations and
/// any comment on it included, and its directive word.
fn directive_line(src: &str, i: usize, end: usize) -> (usize, String) {
    let b = src.as_bytes();
    let mut j = i + 1;
    while j < end && matches!(b[j], b' ' | b'\t') {
        j += 1;
    }
    let w0 = j;
    while j < end && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
        j += 1;
    }
    let word = src[w0..j].to_string();
    // To the end of the line, past `\` continuations and block comments.
    while j < end {
        match b[j] {
            b'\\' if j + 1 < end && (b[j + 1] == b'\n' || b[j + 1] == b'\r') => {
                j += if b[j + 1] == b'\r' { 3 } else { 2 };
            }
            b'/' if j + 1 < end && b[j + 1] == b'*' => {
                j = src[j + 2..end].find("*/").map_or(end, |p| j + 2 + p + 2);
            }
            b'\n' => return (j + 1, word),
            _ => j += 1,
        }
    }
    (end, word)
}

/// A unit that starts with a directive.
fn directive_unit(src: &str, i: usize, end: usize) -> Result<Scanned, String> {
    let (line_stop, word) = directive_line(src, i, end);
    let text = src[i..line_stop].trim_end().to_string();
    let first: String = text.lines().next().unwrap_or("").chars().take(80).collect();
    match word.as_str() {
        "if" | "ifdef" | "ifndef" => {
            // On to the matching `#endif`, through code and directives alike.
            let mut depth = 1usize;
            let mut j = line_stop;
            let mut has_else = false;
            let mut endif_at = None;
            while j < end {
                let (next, is_dir) = skip_to_directive_or_code(src, j, end);
                if !is_dir {
                    j = next;
                    continue;
                }
                let (stop, w) = directive_line(src, next, end);
                match w.as_str() {
                    "if" | "ifdef" | "ifndef" => depth += 1,
                    "else" | "elif" | "elifdef" | "elifndef" if depth == 1 => has_else = true,
                    "endif" => {
                        depth -= 1;
                        if depth == 0 {
                            endif_at = Some((line_start(src.as_bytes(), next), stop));
                            break;
                        }
                    }
                    _ => {}
                }
                j = stop;
            }
            let (endif_start, stop) = endif_at.ok_or_else(|| format!("`{first}` has no #endif"))?;
            let body = (!has_else).then_some(line_stop..endif_start);
            Ok((Kind::Cond, stop, first, body))
        }
        "include" => Ok((Kind::Kept, line_stop, first, None)),
        "pragma" if text.contains("once") => Ok((Kind::Kept, line_stop, first, None)),
        "define" | "undef" => {
            let name: String = text
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            Ok((Kind::Directive, line_stop, format!("#{word} {name}"), None))
        }
        _ => Ok((Kind::Directive, line_stop, first, None)),
    }
}

/// From `j`, past whitespace, comments, literals and code to the next
/// directive: `(where it starts, true)`, or `(end, false)` when there is none.
fn skip_to_directive_or_code(src: &str, mut j: usize, end: usize) -> (usize, bool) {
    let b = src.as_bytes();
    while j < end {
        if is_directive_start(b, j) {
            return (j, true);
        }
        j = skip_token(src, j, end);
    }
    (end, false)
}

/// Past one lexical piece at `j`: a comment, a literal, or one byte.
fn skip_token(src: &str, j: usize, end: usize) -> usize {
    let b = src.as_bytes();
    match b[j] {
        b'/' if j + 1 < end && b[j + 1] == b'/' => line_end(b, j, end),
        b'/' if j + 1 < end && b[j + 1] == b'*' => {
            src[j + 2..end].find("*/").map_or(end, |p| j + 2 + p + 2)
        }
        b'"' | b'\'' => {
            let q = b[j];
            let mut k = j + 1;
            while k < end && b[k] != q && b[k] != b'\n' {
                if b[k] == b'\\' {
                    k += 1;
                }
                k += 1;
            }
            (k + 1).min(end)
        }
        _ => j + 1,
    }
}

/// A unit of code from `i`: to a `;` or a closing `}` back at its own level,
/// the `;` after such a `}` included.
fn code_unit(src: &str, i: usize, end: usize) -> Result<Scanned, String> {
    let b = src.as_bytes();
    let mut depth = 0usize;
    let mut parens = 0usize;
    let mut j = i;
    let mut open_at: Option<usize> = None;
    while j < end {
        if is_directive_start(b, j) {
            j = directive_line(src, j, end).0;
            continue;
        }
        match b[j] {
            b'(' => {
                parens += 1;
                j += 1;
            }
            b')' => {
                parens = parens.saturating_sub(1);
                j += 1;
            }
            // A macro invoked at the top — `CAST_OP(float, half, …)` — ends
            // with its line, having no `;` of its own.
            b'\n' if depth == 0 && parens == 0 && is_macro_call(&src[i..j]) => {
                let name = code_name(&src[i..j]);
                return Ok((Kind::Code, j + 1, name, None));
            }
            b'{' => {
                if depth == 0 && open_at.is_none() {
                    open_at = Some(j);
                }
                depth += 1;
                j += 1;
            }
            b'}' => {
                if depth == 0 {
                    return Err(format!("a `}}` closes nothing at byte {j}"));
                }
                depth -= 1;
                j += 1;
                // A brace inside parentheses — `f(std::make_index_sequence<256>{})`
                // — closes an argument, not the unit.
                if depth == 0 && parens == 0 {
                    let close = j - 1;
                    // `struct S { … };` and `T t[] = { … };` end at the `;`.
                    let mut k = j;
                    while k < end && matches!(b[k], b' ' | b'\t' | b'\r' | b'\n') {
                        k += 1;
                    }
                    let stop = if k < end && b[k] == b';' { k + 1 } else { j };
                    let head = &src[i..open_at.unwrap_or(close)];
                    let name = code_name(head);
                    let opens = is_container_head(head);
                    let body = opens.then(|| open_at.map(|o| o + 1..close)).flatten();
                    return Ok((Kind::Code, stop, name, body));
                }
            }
            b';' if depth == 0 && parens == 0 => {
                let name = code_name(&src[i..j]);
                return Ok((Kind::Code, j + 1, name, None));
            }
            _ => j = skip_token(src, j, end),
        }
    }
    Err(format!("code from byte {i} never ends"))
}

/// Whether `text` is a whole macro invocation: a name followed at once by a
/// parenthesised argument list that ends it, as `UNARY_OP(f32, abs, fabsf(x))`.
fn is_macro_call(text: &str) -> bool {
    let t = strip_comments(text);
    let t = t.trim();
    let name_len = t
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(t.len());
    name_len > 0 && t[name_len..].starts_with('(') && t.ends_with(')')
}

/// Whether a block's head opens a container a split may enter.
fn is_container_head(head: &str) -> bool {
    let h = head.split_whitespace().collect::<Vec<_>>().join(" ");
    h == "extern \"C\"" || h.starts_with("namespace ") || h == "namespace"
}

/// A unit's name, from the text before its body.
fn code_name(head: &str) -> String {
    let h = strip_comments(head);
    let h = h.split_whitespace().collect::<Vec<_>>().join(" ");
    if is_container_head(&h) {
        return h;
    }
    let ident = |s: &str| -> String {
        s.chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    };
    for kw in ["struct ", "class ", "union ", "enum class ", "enum "] {
        if let Some(p) = h.find(kw) {
            let before_paren = h.find('(').is_none_or(|q| p < q);
            if before_paren {
                let n = ident(h[p + kw.len()..].trim_start());
                if !n.is_empty() {
                    return format!("{}{n}", kw.trim_end().to_string() + " ");
                }
            }
        }
    }
    let upto = h.find('(').unwrap_or(h.find('=').unwrap_or(h.len()));
    let last = h[..upto]
        .trim_end()
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|w| !w.is_empty())
        .unwrap_or("")
        .to_string();
    if h[..upto.min(h.len())].contains('(') || h.contains('(') {
        format!("fn {last}")
    } else {
        last
    }
}

/// `text` without its comments.
fn strip_comments(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut j = 0;
    while j < b.len() {
        if j + 1 < b.len() && b[j] == b'/' && (b[j + 1] == b'/' || b[j + 1] == b'*') {
            j = skip_token(text, j, b.len());
            out.push(' ');
        } else {
            out.push(b[j] as char);
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(src: &str) -> Vec<(Kind, String, String)> {
        units(src, 0..src.len())
            .unwrap()
            .into_iter()
            .map(|u| (u.kind, u.name, src[u.span].to_string()))
            .collect()
    }

    /// **Each unit is whole and owns the comment over it**: a function to its
    /// closing brace, a struct through its `;`, a multi-line `#define`, an
    /// `#if` to its `#endif`; an `#include` is kept.
    #[test]
    fn units_are_whole_and_own_their_comments() {
        let src = "#pragma once\n#include \"a.cuh\"\n\n// Block size.\n#define BS \\\n  32\n\n\
                   /// A block.\nstruct block_q { int x; /* } */ };\n\n\
                   #if __CUDA_ARCH__ >= 800\nint f() { return 1; }\n#else\nint f() { return 0; }\n#endif\n\n\
                   template <int N>\n__global__ void k(const char* s = \"}\") {\n  if (N) { }\n}\n";
        let got = names(src);
        assert_eq!(
            got.iter()
                .map(|(k, n, _)| (k.clone(), n.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (Kind::Kept, "#pragma once"),
                (Kind::Kept, "#include \"a.cuh\""),
                (Kind::Directive, "#define BS"),
                (Kind::Code, "struct block_q"),
                (Kind::Cond, "#if __CUDA_ARCH__ >= 800"),
                (Kind::Code, "fn k"),
            ]
        );
        assert_eq!(got[2].2, "// Block size.\n#define BS \\\n  32\n");
        assert_eq!(
            got[3].2,
            "/// A block.\nstruct block_q { int x; /* } */ };\n"
        );
        assert!(got[5].2.ends_with("  if (N) { }\n}\n"));
    }

    /// **A brace inside a call's parentheses does not end the unit**, and two
    /// units on one line do not overlap.
    #[test]
    fn a_braced_argument_stays_in_its_unit() {
        let src = "constexpr T t = make(std::make_index_sequence<256>{});\nint a; int b;\n";
        let u = units(src, 0..src.len()).unwrap();
        let spans: Vec<&str> = u.iter().map(|u| &src[u.span.clone()]).collect();
        assert_eq!(
            spans,
            vec![
                "constexpr T t = make(std::make_index_sequence<256>{});\n",
                "int a;",
                " int b;\n",
            ]
        );
    }

    /// **A macro invoked at the top ends with its line**, however its
    /// arguments are spread; a prototype split over lines does not.
    #[test]
    fn a_top_level_macro_call_is_a_unit() {
        let src = "CAST_OP(float, half,\n        cast_f32_f16)\nUNARY(abs, fabsf(x))\nvoid f(int a,\n       int b);\n";
        let got: Vec<String> = names(src).into_iter().map(|(_, _, t)| t).collect();
        assert_eq!(
            got,
            vec![
                "CAST_OP(float, half,\n        cast_f32_f16)\n",
                "UNARY(abs, fabsf(x))\n",
                "void f(int a,\n       int b);\n",
            ]
        );
    }

    /// **A container's body is known**: `extern "C" { }` and a namespace open,
    /// and an `#if` with no `#else`; one with an `#else` does not.
    #[test]
    fn containers_know_their_bodies() {
        let src = "extern \"C\" {\nvoid a();\nvoid b();\n}\n#ifdef X\nvoid c();\n#endif\nnamespace n {\nint d;\n}\n";
        let u = units(src, 0..src.len()).unwrap();
        let bodies: Vec<Option<&str>> = u.iter().map(|u| u.body.clone().map(|r| &src[r])).collect();
        assert_eq!(
            bodies,
            vec![
                Some("\nvoid a();\nvoid b();\n"),
                Some("void c();\n"),
                Some("\nint d;\n"),
            ]
        );
    }
}
