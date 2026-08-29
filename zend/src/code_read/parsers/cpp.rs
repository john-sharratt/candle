//! C++ scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, slice_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
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
        .insert("struct_specifier", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("class_specifier", ChunkKind::TypeDefinition);
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
        "class_specifier" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "union_specifier" => field_text(node, "name", source).map(|n| format!("union {n}")),
        "enum_specifier" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        "type_definition" => Some("typedef".to_string()),
        _ => None,
    }
}

fn innermost_identifier(node: &Node, source: &[u8]) -> Option<String> {
    let kind = node.kind();
    if kind == "identifier" || kind == "field_identifier" || kind == "type_identifier" {
        return Some(slice_text(node, source));
    }
    if kind == "qualified_identifier" || kind == "operator_name" || kind == "destructor_name" {
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

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "namespace_definition" => {
            field_text(node, "name", source).map(|n| format!("namespace {n}"))
        }
        "class_specifier" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "struct_specifier" => field_text(node, "name", source).map(|n| format!("struct {n}")),
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
        verify_carve(src, Language::Cpp, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Cpp, false)
    }

    // ── Classes + methods ───────────────────────────────────────────────────

    #[test]
    fn extracts_class_with_methods() {
        verify(
            "class Widget {\npublic:\n    int alpha() { return 1; }\n    int beta() { return 2; }\n};\n",
            &[
                ("class Widget", "class Widget"),
                ("class Widget > alpha()", "alpha"),
                ("class Widget > beta()", "beta"),
            ],
        );
    }

    #[test]
    fn extracts_struct_with_method() {
        verify(
            "struct S {\n    void doit() {}\n};\n",
            &[("struct S", "struct S"), ("struct S > doit()", "doit")],
        );
    }

    #[test]
    fn extracts_class_with_constructor_and_destructor() {
        let scopes = verify(
            "class C {\npublic:\n    C() {}\n    ~C() {}\n};\n",
            &[("class C", "class C")],
        );
        let ctor_destructor: Vec<&String> = scopes
            .iter()
            .map(|s| &s.path[s.path.len() - 1])
            .filter(|p| p.starts_with('C') || p.starts_with("~C"))
            .collect();
        assert!(
            !ctor_destructor.is_empty(),
            "expected ctor or dtor scope: {:?}",
            scopes
                .iter()
                .map(|s| s.qualified_path())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn extracts_class_with_inheritance() {
        verify(
            "class Base { public: virtual void run() {} };\nclass Derived : public Base { public: void run() override {} };\n",
            &[
                ("class Base", "class Base"),
                ("class Derived", "class Derived"),
            ],
        );
    }

    #[test]
    fn extracts_pure_virtual_function() {
        verify(
            "class Shape {\npublic:\n    virtual double area() const = 0;\n};\n",
            &[("class Shape", "class Shape")],
        );
    }

    #[test]
    fn extracts_enum_class() {
        verify(
            "enum class Color : int { Red, Green, Blue };\n",
            &[("Color", "Color")],
        );
    }

    // ── Namespaces ───────────────────────────────────────────────────────────

    #[test]
    fn extracts_namespace_with_function() {
        verify(
            "namespace math {\n    int square(int x) { return x * x; }\n}\n",
            &[("namespace math > square()", "square")],
        );
    }

    #[test]
    fn extracts_nested_namespaces() {
        verify(
            "namespace outer {\n    namespace inner {\n        void deep() {}\n    }\n}\n",
            &[("namespace outer > namespace inner > deep()", "deep")],
        );
    }

    // ── Templates ────────────────────────────────────────────────────────────

    #[test]
    fn extracts_template_function() {
        verify(
            "template<typename T>\nT identity(T x) { return x; }\n",
            &[("identity()", "T identity")],
        );
    }

    #[test]
    fn extracts_template_class() {
        verify(
            "template<typename T>\nclass Stack {\n    T items[100];\npublic:\n    void push(T x) { items[0] = x; }\n};\n",
            &[
                ("class Stack", "class Stack"),
                ("class Stack > push()", "push"),
            ],
        );
    }

    // ── Operator overloads + out-of-class definitions ────────────────────────

    #[test]
    fn extracts_operator_overload() {
        let scopes = verify(
            "class V {\npublic:\n    V operator+(const V& other) const { return *this; }\n};\n",
            &[("class V", "class V")],
        );
        assert!(scopes
            .iter()
            .any(|s| s.qualified_path().contains("operator")));
    }

    #[test]
    fn extracts_method_defined_outside_class() {
        verify(
            "struct Point {\n    int x, y;\n    int sum() const;\n};\n\nint Point::sum() const { return x + y; }\n",
            &[("struct Point", "struct Point"), ("sum()", "Point::sum")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn class_scope_lines_match_class_through_close_brace() {
        let src = "// header\nclass Foo {\n    int x;\n    void run() {}\n};\n";
        let scopes = verify(src, &[("class Foo", "class Foo")]);
        let foo = find_scope(&scopes, "class Foo").unwrap();
        // Leading `// header` comment now attached — scope starts at line 1.
        assert_eq!(foo.start_line, 1);
        assert!(foo.end_line >= 5);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "class A { public: void m() {} };\r\nclass B { public: void n() {} };\r\n",
            &[("class A", "class A"), ("class B", "class B")],
        );
    }

    #[test]
    fn preserves_truncated_class_body() {
        verify_cov("class Broken {\n    int x;\n");
    }

    #[test]
    fn preserves_only_preprocessor_and_using() {
        verify_cov("#include <vector>\nusing std::vector;\n");
    }

    #[test]
    fn preserves_lambda_inside_function() {
        verify_cov(
            "void caller() {\n    auto fn = [](int x) { return x + 1; };\n    (void)fn;\n}\n",
        );
    }
}
