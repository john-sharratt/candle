//! C scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, slice_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_c::language()).ok()?;
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
        .insert("struct_specifier", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("union_specifier", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("enum_specifier", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("type_definition", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_definition" => {
            let decl = node.child_by_field_name("declarator")?;
            let name = innermost_identifier(&decl, source)?;
            Some(format!("{name}()"))
        }
        "struct_specifier" => field_text(node, "name", source).map(|n| format!("struct {n}")),
        "union_specifier" => field_text(node, "name", source).map(|n| format!("union {n}")),
        "enum_specifier" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        "type_definition" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(id) = innermost_identifier(&child, source) {
                    return Some(format!("typedef {id}"));
                }
            }
            None
        }
        _ => None,
    }
}

fn innermost_identifier(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() == "identifier"
        || node.kind() == "type_identifier"
        || node.kind() == "field_identifier"
    {
        return Some(slice_text(node, source));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(id) = innermost_identifier(&child, source) {
            return Some(id);
        }
    }
    None
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
        verify_carve(src, Language::C, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::C, false)
    }

    // ── Functions ────────────────────────────────────────────────────────────

    #[test]
    fn extracts_simple_function() {
        verify(
            "int alpha(void) { return 1; }\nvoid beta(int x) { return; }\n",
            &[("alpha()", "alpha"), ("beta()", "beta")],
        );
    }

    #[test]
    fn extracts_static_function() {
        verify(
            "static int counter(void) { return 0; }\n",
            &[("counter()", "static int counter")],
        );
    }

    #[test]
    fn extracts_inline_function() {
        verify(
            "static inline int fast(int x) { return x + 1; }\ninline int hot(int x) { return x * 2; }\n",
            &[("fast()", "static inline int fast"), ("hot()", "inline int hot")],
        );
    }

    #[test]
    fn extracts_function_with_pointer_return() {
        verify(
            "char *get_name(int id) { return \"x\"; }\n",
            &[("get_name()", "get_name")],
        );
    }

    #[test]
    fn extracts_function_with_void_pointer_arg() {
        verify(
            "void *alloc(size_t n) { return 0; }\n",
            &[("alloc()", "alloc")],
        );
    }

    #[test]
    fn extracts_function_with_function_pointer_param() {
        verify(
            "int caller(int (*cb)(int)) { return cb(0); }\n",
            &[("caller()", "int caller")],
        );
    }

    // ── Types ────────────────────────────────────────────────────────────────

    #[test]
    fn extracts_struct_and_union() {
        verify(
            "struct Point { int x; int y; };\nunion Value { int i; float f; };\n",
            &[("struct Point", "struct Point"), ("union Value", "union Value")],
        );
    }

    #[test]
    fn extracts_enum() {
        verify(
            "enum Color { RED, GREEN, BLUE };\n",
            &[("enum Color", "enum Color")],
        );
    }

    #[test]
    fn extracts_typedef_struct() {
        verify(
            "typedef struct Node Node;\ntypedef int (*callback)(int);\n",
            &[("typedef Node", "typedef")],
        );
    }

    #[test]
    fn extracts_struct_with_function_pointer_field() {
        verify(
            "struct Ops {\n    int (*read)(void *ctx);\n    int (*write)(void *ctx, int x);\n};\n",
            &[("struct Ops", "struct Ops")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn function_scope_lines_match_signature_through_close_brace() {
        let src = "// header\nint hello(void) {\n    return 1;\n}\n";
        let scopes = verify(src, &[("hello()", "hello")]);
        let hello = find_scope(&scopes, "hello()").unwrap();
        assert_eq!(hello.start_line, 2);
        assert_eq!(hello.end_line, 4);
    }

    #[test]
    fn struct_scope_lines_match_struct_through_semicolon() {
        let src = "struct Foo {\n    int x;\n    int y;\n};\n";
        let scopes = verify(src, &[("struct Foo", "struct Foo")]);
        let foo = find_scope(&scopes, "struct Foo").unwrap();
        assert_eq!(foo.start_line, 1);
        assert!(foo.end_line >= 4, "struct should span through closing brace");
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_preprocessor_noise() {
        verify(
            "#include <stdio.h>\n#define MAX 10\n\nint main(int argc, char **argv) {\n    return 0;\n}\n",
            &[("main()", "main")],
        );
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "int alpha(void) { return 1; }\r\nint beta(void) { return 2; }\r\n",
            &[("alpha()", "alpha"), ("beta()", "beta")],
        );
    }

    #[test]
    fn preserves_truncated_function() {
        verify_cov("int alpha(int x) {\n    return\n");
    }

    #[test]
    fn preserves_only_preprocessor_directives() {
        verify_cov("#include <stdio.h>\n#define X 1\n");
    }

    #[test]
    fn preserves_anonymous_enum() {
        // No name on the enum — gap-fill must still cover it.
        verify_cov("enum { A = 1, B = 2, C = 3 };\n");
    }
}
