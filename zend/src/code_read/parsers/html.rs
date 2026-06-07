//! HTML scope carver — splits on top-level structural landmark
//! elements (`<head>`, `<body>`, `<header>`, `<main>`, `<footer>`,
//! `<section>`, `<nav>`, `<article>`, `<aside>`).  HTML is
//! structurally shallow at the document level so coarse landmarks fit
//! the retrieval shape better than per-element splitting.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, slice_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_html::language()).ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label: |_, _| None,
    };
    rules
        .kind_to_chunk
        .insert("element", ChunkKind::HeaderSection);
    let raw = carve_tree(&mut parser, source, &rules)?;
    // Post-filter: only keep scopes whose label survived the
    // landmark check.
    Some(
        raw.into_iter()
            .filter(|s| {
                s.path
                    .last()
                    .map(|p| !p.starts_with("element"))
                    .unwrap_or(false)
            })
            .collect(),
    )
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "element" {
        return None;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "start_tag" && child.kind() != "self_closing_tag" {
            continue;
        }
        let mut tc = child.walk();
        for tag_child in child.children(&mut tc) {
            if tag_child.kind() == "tag_name" {
                let tag = slice_text(&tag_child, source);
                let landmark = matches!(
                    tag.as_str(),
                    "head"
                        | "body"
                        | "header"
                        | "main"
                        | "footer"
                        | "section"
                        | "nav"
                        | "article"
                        | "aside"
                );
                if !landmark {
                    return None;
                }
                return Some(format!("<{tag}>"));
            }
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Html, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Html, false)
    }

    #[test]
    fn extracts_head_body_landmarks() {
        verify(
            "<!DOCTYPE html>\n<html>\n<head>\n  <title>Test</title>\n</head>\n<body>\n  <p>hello</p>\n</body>\n</html>\n",
            &[("<head>", "<head>"), ("<body>", "<body>")],
        );
    }

    #[test]
    fn extracts_main_header_footer_landmarks() {
        verify(
            "<html><body>\n<header>top</header>\n<main>content</main>\n<footer>end</footer>\n</body></html>\n",
            &[("<header>", "<header>"), ("<main>", "<main>"), ("<footer>", "<footer>")],
        );
    }

    #[test]
    fn extracts_nav_article_aside_landmarks() {
        verify(
            "<html><body>\n<nav>links</nav>\n<article>post</article>\n<aside>side</aside>\n</body></html>\n",
            &[("<nav>", "<nav>"), ("<article>", "<article>"), ("<aside>", "<aside>")],
        );
    }

    #[test]
    fn extracts_multiple_sections() {
        let scopes = verify_cov(
            "<html><body>\n<section>one</section>\n<section>two</section>\n</body></html>\n",
        );
        let count = scopes
            .iter()
            .filter(|s| s.qualified_path() == "<section>")
            .count();
        assert!(count >= 2, "expected ≥2 <section> scopes, got {count}");
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn landmark_scope_lines_match_element() {
        let src = "<html>\n<body>\n<main>\n  hi\n</main>\n</body>\n</html>\n";
        let scopes = verify(src, &[("<main>", "<main>")]);
        let main = find_scope(&scopes, "<main>").unwrap();
        assert_eq!(main.start_line, 3);
        assert_eq!(main.end_line, 5);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_landmark_free_fragment() {
        verify_cov("<p>hello</p><div>x</div>");
    }

    #[test]
    fn preserves_self_closing_landmark() {
        verify_cov("<html><body><nav /></body></html>");
    }

    #[test]
    fn preserves_uppercase_tags() {
        verify_cov("<HTML><BODY><MAIN>x</MAIN></BODY></HTML>");
    }

    #[test]
    fn preserves_unbalanced_tags() {
        verify_cov("<html><body><header>top<body></html>");
    }

    #[test]
    fn preserves_malformed_html() {
        verify_cov("<<<>>><html >><<x>");
    }

    #[test]
    fn preserves_doctype_only() {
        verify_cov("<!DOCTYPE html>\n");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "<html>\r\n<body>\r\n<main>hi</main>\r\n</body>\r\n</html>\r\n",
            &[("<main>", "<main>")],
        );
    }
}
