//! Markdown carver — splits at `##` (and deeper) section headers.

use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Vec<Scope> {
    let text = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut headers: Vec<(usize, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("## ") || trimmed.starts_with("### ") || trimmed.starts_with("# ") {
            let title = trimmed.trim_start_matches('#').trim().to_string();
            headers.push((i, title));
        }
    }

    let mut scopes = Vec::new();
    if headers.is_empty() {
        scopes.push(Scope {
            path: vec!["document".to_string()],
            kind: ChunkKind::HeaderSection,
            start_line: 1,
            end_line: lines.len() as u32,
        });
        return scopes;
    }

    if headers[0].0 > 0 {
        scopes.push(Scope {
            path: vec!["preamble".to_string()],
            kind: ChunkKind::HeaderSection,
            start_line: 1,
            end_line: headers[0].0 as u32,
        });
    }

    for (i, (start, title)) in headers.iter().enumerate() {
        let end = if i + 1 < headers.len() {
            headers[i + 1].0
        } else {
            lines.len()
        };
        scopes.push(Scope {
            path: vec![format!("## {title}")],
            kind: ChunkKind::HeaderSection,
            start_line: (*start as u32) + 1,
            end_line: end as u32,
        });
    }

    scopes
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Markdown, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Markdown, false)
    }

    // ── Header parsing ───────────────────────────────────────────────────────

    #[test]
    fn splits_on_h2_headers() {
        verify(
            "# title\n\nintro\n\n## first\n\ncontent\n\n## second\n\nmore\n",
            &[("first", "## first"), ("second", "## second")],
        );
    }

    #[test]
    fn splits_on_h1_headers() {
        verify(
            "# top\n\nintro\n\n# second\n\nmore\n",
            &[("top", "# top"), ("second", "# second")],
        );
    }

    #[test]
    fn splits_on_h3_headers() {
        verify("### A\nx\n### B\ny\n", &[("A", "### A"), ("B", "### B")]);
    }

    #[test]
    fn handles_headers_with_inline_formatting() {
        verify(
            "## `code` and **bold**\nx\n",
            &[("code", "## `code` and **bold**")],
        );
    }

    // ── Preamble ─────────────────────────────────────────────────────────────

    #[test]
    fn preamble_chunk_covers_content_before_first_header() {
        let src = "intro line 1\nintro line 2\n\n## first\n\nbody\n";
        let scopes = verify(src, &[("preamble", "intro line 1"), ("first", "## first")]);
        let preamble = find_scope(&scopes, "preamble").unwrap();
        assert_eq!(preamble.start_line, 1);
        assert_eq!(preamble.end_line, 3);
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn section_scope_lines_match_section() {
        let src = "## hello\n\nworld\n\n## bye\n";
        let scopes = verify(src, &[("hello", "## hello")]);
        let hello = find_scope(&scopes, "hello").unwrap();
        assert_eq!(hello.start_line, 1);
        assert_eq!(hello.end_line, 4);
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn no_headers_emits_single_document_scope() {
        let scopes = verify_cov("plain text\nwith two lines\n");
        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].path[0], "document");
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert!(verify_cov("").is_empty());
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "## one\r\n\r\nx\r\n\r\n## two\r\n",
            &[("one", "## one"), ("two", "## two")],
        );
    }

    #[test]
    fn preserves_invalid_utf8() {
        let src = &[0xff, 0xfe, 0xfd, b'\n'];
        let scopes = crate::code_read::carve::carve(src, Language::Markdown, false);
        assert!(scopes.is_empty());
    }
}
