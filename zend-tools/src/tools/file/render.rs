//! Rendering for file pages — the shared `<tool_response>` body format.
//!
//! One format, used from both ends of the system:
//!
//! * The live [`file_read`](super::read) tool returns this shape.
//! * The `code_reading` ingest drives that same tool call for every page of
//!   every file it reads, so the pages prefilled into its conversations are
//!   byte-identical to the ones a live read returns.
//!
//! ````text
//! ```rust file=server/src/auth/handler.rs page=1 pages=4 lines=620
//! Hk2a| file header
//! //! Token validation.
//! r7Lm| impl AuthHandler > fn validate_token
//!     pub fn validate_token(&self, token: &str) -> Result<Claims> {
//!         …
//!     }
//! ```
//! end of server/src/auth/handler.rs page 1 of 4 — next: page=2
//! ````
//!
//! Each part of the file — a function, a type, a block of imports, a
//! paragraph — is shown under its anchor (`docs/npcd_anchored_edits.md` §5,
//! [`doc_units::render`]), and `file_edit` names it by that anchor. A page
//! holds whole parts, at most [`PAGE_LINES`] lines of them. There is no header
//! line: the opening fence itself names the file as a workspace path
//! (`<repo>/<path>` — a path alone is ambiguous across a workspace of several
//! repositories), which page this is, how many pages there are, and the
//! file's length. The closing line names the file and page again, so the
//! page's content is bracketed by its identity at both ends.
//!
//! **Pages count from 1, and the closing line says what comes next** — the page
//! to ask for, or that this was the last. Zero-based, the last of two pages read
//! `page 1/2`, which says "page one of two" to anyone reading it: a model that
//! had read a file whole reasoned its way to a missing page.

use doc_units::render::{pages, units};
use doc_units::Doc;

pub use doc_units::render::PAGE_LINES;

/// The attribute every page of a file opens with, `file=<repo>/<path>` — the
/// file's identity. A reply that points the model at pages it already holds
/// names them by exactly this string, so the two can be matched byte for byte.
pub fn file_anchor(repo: &str, path: &str) -> String {
    format!("file={repo}/{path}")
}

/// How many pages `content`, at `path`, reads as.
pub fn page_count(path: &str, content: &str) -> u32 {
    pages(&Doc::new(path, content), PAGE_LINES).len() as u32
}

/// Which page (from 1) holds unit `unit` of `doc`.
pub fn page_of(doc: &Doc, unit: usize) -> u32 {
    pages(doc, PAGE_LINES)
        .iter()
        .position(|p| p.contains(&unit))
        .map_or(1, |i| i as u32 + 1)
}

/// Page `page` (from 1, clamped into range) of `doc`, as fence + anchored
/// parts + closing line. The caller frames it in `<tool_response>` tags.
///
/// `fence_tag` is the markdown language tag (`rust`, `python`, …); empty renders
/// `text`, so the fence's first word is always a language and never the
/// `file=` attribute.
pub fn anchored_page(repo: &str, path: &str, doc: &Doc, page: u32, fence_tag: &str) -> String {
    let file = format!("{repo}/{path}");
    let anchor = file_anchor(repo, path);
    let lang = if fence_tag.is_empty() {
        "text"
    } else {
        fence_tag
    };
    let total_lines = doc.line_count();
    if doc.units().is_empty() {
        // Nothing to show and no page to state; "page=1 pages=0" reads as a
        // bug the same way "lines 1-0" used to.
        return format!("\n```{lang} {anchor} lines={total_lines}\n```\nend of {file}\n");
    }
    let all = pages(doc, PAGE_LINES);
    let total = all.len() as u32;
    let page = page.clamp(1, total);
    let body = units(doc, all[page as usize - 1].clone());
    let after = if page < total {
        format!("— next: page={}", page + 1)
    } else {
        "— the last page".to_string()
    };
    format!(
        "\n```{lang} {anchor} page={page} pages={total} lines={total_lines}\n\
         {body}```\nend of {file} page {page} of {total} {after}\n"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn anchors(doc: &Doc) -> Vec<String> {
        doc.anchors()
    }

    #[test]
    fn a_page_shows_each_part_under_its_anchor() {
        let doc = Doc::new("a.rs", "use x;\n\nfn go() {\n    1\n}\n");
        let a = anchors(&doc);
        assert_eq!(
            anchored_page("r", "a.rs", &doc, 1, "rust"),
            format!(
                "\n```rust file=r/a.rs page=1 pages=1 lines=5\n{}| use x;\n{}| fn go\nfn go() {{\n    \
                 1\n}}\n```\nend of r/a.rs page 1 of 1 — the last page\n",
                a[0], a[1]
            )
        );
    }

    /// The identity is the fence's own `file=` attribute, so a reply naming
    /// it matches the page exactly.
    #[test]
    fn the_identity_is_what_the_fence_opens_with() {
        assert_eq!(file_anchor("r", "src/a.rs"), "file=r/src/a.rs");
        let doc = Doc::new("src/a.rs", "x\n");
        let out = anchored_page("r", "src/a.rs", &doc, 1, "rust");
        assert!(out.starts_with("\n```rust file=r/src/a.rs page="), "{out}");
    }

    /// Pages hold whole parts, and one short of the last names the next.
    #[test]
    fn pages_hold_whole_parts_and_name_the_next() {
        let text: String = (0..150).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        let doc = Doc::new("a.md", &text);
        assert_eq!(page_count("a.md", &text), 2);
        let first = anchored_page("r", "a.md", &doc, 1, "markdown");
        assert!(first.starts_with("\n```markdown file=r/a.md page=1 pages=2 lines=450\n"));
        assert!(first.ends_with("```\nend of r/a.md page 1 of 2 — next: page=2\n"));
        // 200 lines of two-line paragraphs: exactly the first hundred.
        assert!(first.contains("| p99 a\np99 b\n```"), "{first}");
        let last = anchored_page("r", "a.md", &doc, 9, "markdown");
        assert!(last.contains("page=2 pages=2"));
        assert!(last.ends_with("end of r/a.md page 2 of 2 — the last page\n"));
        assert_eq!(page_of(&doc, 0), 1);
        assert_eq!(page_of(&doc, 120), 2);
    }

    #[test]
    fn an_empty_file_says_so_instead_of_an_impossible_range() {
        let doc = Doc::new("a.rs", "");
        assert_eq!(
            anchored_page("r", "a.rs", &doc, 1, "rust"),
            "\n```rust file=r/a.rs lines=0\n```\nend of r/a.rs\n"
        );
        let blank = Doc::new("a.rs", "\n\n");
        assert_eq!(
            anchored_page("r", "a.rs", &blank, 1, ""),
            "\n```text file=r/a.rs lines=2\n```\nend of r/a.rs\n"
        );
    }

    #[test]
    fn an_unknown_extension_renders_a_bare_fence() {
        assert_eq!(fence_tag_for_path("x/y.bin"), "");
        assert_eq!(fence_tag_for_path("Makefile"), "");
        assert_eq!(fence_tag_for_path("a/b/c.rs"), "rust");
        assert_eq!(fence_tag_for_path("A/B/C.RS"), "rust", "case-insensitive");
    }
}
