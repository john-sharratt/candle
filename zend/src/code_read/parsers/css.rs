//! CSS scope carver — splits on top-level rule_sets and at-rules
//! (`@media`, `@keyframes`, etc).  Selectors become the scope label.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, slice_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_css::language()).ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label: |_, _| None,
    };
    rules
        .kind_to_chunk
        .insert("rule_set", ChunkKind::HeaderSection);
    rules
        .kind_to_chunk
        .insert("media_statement", ChunkKind::HeaderSection);
    rules
        .kind_to_chunk
        .insert("keyframes_statement", ChunkKind::HeaderSection);
    rules
        .kind_to_chunk
        .insert("supports_statement", ChunkKind::HeaderSection);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "rule_set" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "selectors" {
                    let text = slice_text(&child, source);
                    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    return Some(one_line);
                }
            }
            None
        }
        "media_statement" => Some("@media".to_string()),
        "keyframes_statement" => Some("@keyframes".to_string()),
        "supports_statement" => Some("@supports".to_string()),
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Css, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Css, false)
    }

    #[test]
    fn extracts_simple_rule_set() {
        verify(
            ".container { display: flex; }\nbody { margin: 0; }\n",
            &[(".container", ".container"), ("body", "body")],
        );
    }

    #[test]
    fn extracts_media_query() {
        verify(
            "@media (max-width: 600px) { body { font-size: 14px; } }\n",
            &[("@media", "@media")],
        );
    }

    #[test]
    fn extracts_keyframes() {
        verify(
            "@keyframes fade { from { opacity: 0; } to { opacity: 1; } }\n",
            &[("@keyframes", "@keyframes")],
        );
    }

    #[test]
    fn extracts_supports_query() {
        verify(
            "@supports (display: grid) { .grid { display: grid; } }\n",
            &[("@supports", "@supports")],
        );
    }

    #[test]
    fn extracts_multi_selector_rules() {
        verify(
            "h1, h2, h3 { font-weight: bold; }\n",
            &[("h1, h2, h3", "h1, h2, h3")],
        );
    }

    #[test]
    fn extracts_id_and_attribute_selectors() {
        verify(
            "#main { color: red; }\n[data-x] { color: blue; }\n",
            &[("#main", "#main"), ("[data-x]", "[data-x]")],
        );
    }

    #[test]
    fn extracts_pseudo_class_selector() {
        verify(
            "a:hover { text-decoration: underline; }\n",
            &[("a:hover", "a:hover")],
        );
    }

    #[test]
    fn extracts_pseudo_element_selector() {
        verify(
            ".btn::before { content: '<'; }\n",
            &[(".btn::before", ".btn::before")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn rule_scope_lines_match_rule() {
        let src = ".container {\n    display: flex;\n    gap: 8px;\n}\n";
        let scopes = verify(src, &[(".container", ".container")]);
        let container = find_scope(&scopes, ".container").unwrap();
        assert_eq!(container.start_line, 1);
        assert_eq!(container.end_line, 4);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_nested_media_query() {
        verify_cov("@media print { @media (color) { body { color: black; } } }\n");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            ".a { color: red; }\r\n.b { color: blue; }\r\n",
            &[(".a", ".a"), (".b", ".b")],
        );
    }

    #[test]
    fn preserves_unterminated_rule() {
        verify_cov(".broken { color: red\n");
    }

    #[test]
    fn preserves_only_comments() {
        verify_cov("/* header */\n/* more */\n");
    }
}
