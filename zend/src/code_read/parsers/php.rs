//! PHP scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_php::LANGUAGE_PHP.into())
        .ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("function_definition", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("method_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("interface_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("trait_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("enum_declaration", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_definition" => field_text(node, "name", source).map(|n| format!("function {n}")),
        "method_declaration" => field_text(node, "name", source).map(|n| format!("{n}()")),
        "class_declaration" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "interface_declaration" => {
            field_text(node, "name", source).map(|n| format!("interface {n}"))
        }
        "trait_declaration" => field_text(node, "name", source).map(|n| format!("trait {n}")),
        "enum_declaration" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        _ => None,
    }
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "namespace_definition" => {
            field_text(node, "name", source).map(|n| format!("namespace {n}"))
        }
        "class_declaration" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "interface_declaration" => {
            field_text(node, "name", source).map(|n| format!("interface {n}"))
        }
        "trait_declaration" => field_text(node, "name", source).map(|n| format!("trait {n}")),
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
        verify_carve(src, Language::Php, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Php, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_top_level_function() {
        verify(
            "<?php\nfunction greet($name) {\n    return \"hi $name\";\n}\n",
            &[("function greet", "function greet")],
        );
    }

    #[test]
    fn extracts_class_with_methods() {
        verify(
            "<?php\nclass User {\n    private $name;\n    public function __construct($n) { $this->name = $n; }\n    public function name() { return $this->name; }\n}\n",
            &[
                ("class User", "class User"),
                ("class User > __construct()", "__construct"),
                ("class User > name()", "name"),
            ],
        );
    }

    #[test]
    fn extracts_interface_and_trait() {
        verify(
            "<?php\ninterface Greeter { public function greet(): string; }\ntrait Named { public function name(): string { return 'x'; } }\n",
            &[
                ("interface Greeter", "interface Greeter"),
                ("trait Named", "trait Named"),
            ],
        );
    }

    #[test]
    fn extracts_enum() {
        verify(
            "<?php\nenum Status: string {\n    case OK = 'ok';\n    case FAIL = 'fail';\n}\n",
            &[("enum Status", "enum Status")],
        );
    }

    #[test]
    fn extracts_namespaced_class() {
        verify(
            "<?php\nnamespace App\\Models;\n\nclass User {\n    public function name() { return 'x'; }\n}\n",
            &[
                ("class User", "class User"),
                ("class User > name()", "name"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_extends_and_implements() {
        verify(
            "<?php\ninterface Logger { public function log(string $m): void; }\nabstract class Base { abstract public function name(): string; }\n\nclass App extends Base implements Logger {\n    public function name(): string { return 'app'; }\n    public function log(string $m): void { echo $m; }\n}\n",
            &[
                ("interface Logger", "interface Logger"),
                ("class Base", "abstract class Base"),
                ("class App", "class App extends Base implements Logger"),
            ],
        );
    }

    #[test]
    fn extracts_static_method() {
        verify(
            "<?php\nclass Math {\n    public static function square(int $x): int { return $x * $x; }\n}\n",
            &[("class Math > square()", "public static function square")],
        );
    }

    #[test]
    fn extracts_method_with_visibility_modifiers() {
        verify(
            "<?php\nclass X {\n    private function hidden(): void {}\n    protected function family(): void {}\n    public function open(): void {}\n}\n",
            &[
                ("class X > hidden()", "private function hidden"),
                ("class X > family()", "protected function family"),
                ("class X > open()", "public function open"),
            ],
        );
    }

    #[test]
    fn extracts_trait_with_methods() {
        verify(
            "<?php\ntrait Loggable {\n    private string $tag = '';\n    public function tag(): string { return $this->tag; }\n    public function setTag(string $t): void { $this->tag = $t; }\n}\n",
            &[
                ("trait Loggable", "trait Loggable"),
                ("trait Loggable > tag()", "public function tag"),
                ("trait Loggable > setTag()", "public function setTag"),
            ],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn method_scope_lines_match_method() {
        let src = "<?php\nclass A {\n    public function m(): void {\n        return;\n    }\n}\n";
        let scopes = verify(src, &[("class A > m()", "public function m")]);
        let m = find_scope(&scopes, "class A > m()").unwrap();
        assert_eq!(m.start_line, 3);
        assert_eq!(m.end_line, 5);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_short_open_tag() {
        verify("<?php class A {}\n", &[("class A", "class A")]);
    }

    #[test]
    fn preserves_arrow_function_in_method_body() {
        verify_cov(
            "<?php\nclass C {\n    public function map(array $items): array {\n        return array_map(fn($x) => $x * 2, $items);\n    }\n}\n",
        );
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "<?php\r\nfunction alpha(): int { return 1; }\r\n",
            &[("function alpha", "function alpha")],
        );
    }

    #[test]
    fn preserves_truncated_class() {
        verify_cov("<?php\nclass Broken {\n    public function m()");
    }

    #[test]
    fn preserves_namespace_only_file() {
        verify_cov("<?php\nnamespace App;\n");
    }
}
