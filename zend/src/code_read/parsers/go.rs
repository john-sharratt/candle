//! Go scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, slice_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_go::LANGUAGE.into()).ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("function_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("method_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("type_declaration", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_declaration" => field_text(node, "name", source).map(|n| format!("func {n}")),
        "method_declaration" => {
            // method_declaration has fields `receiver` and `name`.
            let name = field_text(node, "name", source)?;
            let recv = field_text(node, "receiver", source).unwrap_or_default();
            let recv = recv
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim()
                .to_string();
            Some(format!("func ({recv}) {name}"))
        }
        "type_declaration" => {
            // `type_declaration` wraps one or more `type_spec` children; use
            // the first one's name for the scope label.
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "type_spec" {
                    if let Some(name) = field_text(&child, "name", source) {
                        return Some(format!("type {name}"));
                    }
                }
            }
            Some(
                slice_text(node, source)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            )
        }
        _ => None,
    }
}

fn enclosing_label(_node: &Node, _source: &[u8]) -> Option<String> {
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Go, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Go, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_function_and_method() {
        verify(
            "package main\n\nfunc Hello() string { return \"hi\" }\n\ntype Handler struct{}\n\nfunc (h *Handler) Serve() {}\n",
            &[
                ("func Hello", "func Hello"),
                ("type Handler", "type Handler"),
                ("Serve", "Serve"),
            ],
        );
    }

    #[test]
    fn extracts_init_function() {
        verify(
            "package main\n\nfunc init() {\n    setupGlobals()\n}\n",
            &[("func init", "func init")],
        );
    }

    #[test]
    fn extracts_method_with_value_receiver() {
        let scopes = verify(
            "package main\n\ntype Counter int\n\nfunc (c Counter) Get() int { return int(c) }\n",
            &[("type Counter", "type Counter")],
        );
        assert!(scopes
            .iter()
            .any(|s| s.qualified_path().contains("Get") && s.qualified_path().contains("Counter")));
    }

    #[test]
    fn extracts_method_with_pointer_receiver() {
        let scopes = verify_cov(
            "package main\n\ntype Server struct{}\n\nfunc (s *Server) Start() {}\nfunc (s *Server) Stop() {}\n",
        );
        let names: Vec<String> = scopes.iter().map(|s| s.qualified_path()).collect();
        assert!(names.iter().any(|n| n.contains("Start")));
        assert!(names.iter().any(|n| n.contains("Stop")));
    }

    #[test]
    fn extracts_blank_receiver_method() {
        let scopes = verify_cov("package main\n\ntype T struct{}\n\nfunc (_ T) Method() {}\n");
        assert!(scopes.iter().any(|s| s.qualified_path().contains("Method")));
    }

    #[test]
    fn extracts_function_returning_multiple_values() {
        verify(
            "package x\n\nfunc Divide(a, b int) (int, error) {\n    if b == 0 { return 0, fmt.Errorf(\"zero\") }\n    return a / b, nil\n}\n",
            &[("func Divide", "func Divide")],
        );
    }

    #[test]
    fn extracts_named_return_function() {
        verify(
            "package x\n\nfunc Compute() (result int, err error) {\n    result = 42\n    return\n}\n",
            &[("func Compute", "func Compute")],
        );
    }

    #[test]
    fn extracts_variadic_function() {
        verify(
            "package x\n\nfunc Sum(nums ...int) int {\n    s := 0\n    for _, n := range nums { s += n }\n    return s\n}\n",
            &[("func Sum", "func Sum")],
        );
    }

    // ── Types ────────────────────────────────────────────────────────────────

    #[test]
    fn extracts_interface_type() {
        verify(
            "package x\n\ntype Greeter interface {\n    Greet(name string) string\n}\n",
            &[("type Greeter", "type Greeter")],
        );
    }

    #[test]
    fn extracts_struct_with_tags() {
        verify(
            "package x\n\ntype User struct {\n    Name string `json:\"name\"`\n    Age  int    `json:\"age\"`\n}\n",
            &[("type User", "type User")],
        );
    }

    #[test]
    fn extracts_struct_with_embedded_type() {
        verify(
            "package x\n\nimport \"io\"\n\ntype LoggingReader struct {\n    io.Reader\n    name string\n}\n",
            &[("type LoggingReader", "type LoggingReader")],
        );
    }

    #[test]
    fn extracts_multiple_type_specs_in_block() {
        let scopes =
            verify_cov("package x\n\ntype (\n    Foo string\n    Bar int\n    Baz []byte\n)\n");
        assert!(scopes.iter().any(|s| s.qualified_path().contains("type")));
    }

    #[test]
    fn extracts_constructor_pattern() {
        verify(
            "package db\n\ntype Connection struct{ url string }\n\nfunc NewConnection(url string) *Connection {\n    return &Connection{url: url}\n}\n",
            &[
                ("type Connection", "type Connection"),
                ("func NewConnection", "func NewConnection"),
            ],
        );
    }

    // ── Generics (Go 1.18+) ──────────────────────────────────────────────────

    #[test]
    fn extracts_generic_function() {
        verify(
            "package x\n\nfunc Identity[T any](x T) T { return x }\n",
            &[("func Identity", "func Identity[T any]")],
        );
    }

    #[test]
    fn extracts_generic_type() {
        verify(
            "package x\n\ntype Stack[T any] struct {\n    items []T\n}\n\nfunc (s *Stack[T]) Push(v T) {\n    s.items = append(s.items, v)\n}\n",
            &[("type Stack", "type Stack[T any]")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn function_scope_lines_match_signature_through_close_brace() {
        let src = "package main\n\nfunc Hello() {\n    return\n}\n";
        let scopes = verify(src, &[("func Hello", "func Hello")]);
        let hello = find_scope(&scopes, "func Hello").unwrap();
        assert_eq!(hello.start_line, 3);
        assert_eq!(hello.end_line, 5);
    }

    #[test]
    fn type_scope_lines_match_type_declaration() {
        let src = "package x\n\ntype Foo struct {\n    a int\n    b string\n}\n";
        let scopes = verify(src, &[("type Foo", "type Foo")]);
        let foo = find_scope(&scopes, "type Foo").unwrap();
        assert_eq!(foo.start_line, 3);
        assert_eq!(foo.end_line, 6);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_around_broken_middle_fn() {
        verify_cov("package main\n\nfunc Alpha() string { return \"a\" }\n\nfunc Broken(\n{\n}\n\nfunc Gamma() string { return \"g\" }\n");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "package main\r\n\r\nfunc Alpha() {}\r\nfunc Beta() {}\r\n",
            &[("func Alpha", "func Alpha"), ("func Beta", "func Beta")],
        );
    }

    #[test]
    fn preserves_truncated_function() {
        verify_cov("package x\nfunc broken(\n");
    }

    #[test]
    fn preserves_package_and_import_block() {
        verify_cov(
            "package main\n\nimport (\n    \"fmt\"\n    \"os\"\n)\n\nfunc main() {\n    fmt.Println(\"hi\")\n    os.Exit(0)\n}\n",
        );
    }

    #[test]
    fn preserves_package_only_file() {
        verify_cov("package main\n");
    }

    #[test]
    fn preserves_import_only_file() {
        verify_cov("package main\n\nimport (\n    \"fmt\"\n    \"os\"\n)\n");
    }
}
