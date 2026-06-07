//! Java scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_java::language()).ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("method_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("constructor_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("interface_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("enum_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("record_declaration", ChunkKind::TypeDefinition);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "method_declaration" => field_text(node, "name", source).map(|n| format!("{n}()")),
        "constructor_declaration" => field_text(node, "name", source).map(|n| format!("{n}()")),
        "class_declaration" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "interface_declaration" => {
            field_text(node, "name", source).map(|n| format!("interface {n}"))
        }
        "enum_declaration" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        "record_declaration" => field_text(node, "name", source).map(|n| format!("record {n}")),
        _ => None,
    }
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "class_declaration" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "interface_declaration" => {
            field_text(node, "name", source).map(|n| format!("interface {n}"))
        }
        "enum_declaration" => field_text(node, "name", source).map(|n| format!("enum {n}")),
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
        verify_carve(src, Language::Java, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Java, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_class_with_methods() {
        verify(
            "public class Demo {\n    public int alpha() { return 1; }\n    private void beta(String s) { }\n}\n",
            &[
                ("class Demo", "public class Demo"),
                ("class Demo > alpha()", "public int alpha"),
                ("class Demo > beta()", "private void beta"),
            ],
        );
    }

    #[test]
    fn extracts_interface_with_default_method() {
        verify(
            "public interface Greeter {\n    String greet(String name);\n    default String hello() { return greet(\"world\"); }\n}\n",
            &[
                ("interface Greeter", "public interface Greeter"),
                ("interface Greeter > hello()", "default String hello"),
            ],
        );
    }

    #[test]
    fn extracts_enum_with_methods() {
        verify(
            "public enum Status {\n    OK, FAIL;\n    public boolean isOk() { return this == OK; }\n}\n",
            &[
                ("enum Status", "public enum Status"),
                ("enum Status > isOk()", "isOk"),
            ],
        );
    }

    #[test]
    fn extracts_constructor() {
        verify(
            "class Box {\n    private int value;\n    public Box(int v) { this.value = v; }\n}\n",
            &[
                ("class Box", "class Box"),
                ("class Box > Box()", "public Box"),
            ],
        );
    }

    #[test]
    fn extracts_record() {
        verify(
            "public record Point(int x, int y) {\n    public double distanceFromOrigin() { return Math.hypot(x, y); }\n}\n",
            &[("record Point", "public record Point")],
        );
    }

    #[test]
    fn extracts_generic_method() {
        verify(
            "class Util {\n    public <T> T identity(T x) { return x; }\n}\n",
            &[("class Util > identity()", "public <T> T identity")],
        );
    }

    // ── Class variants ───────────────────────────────────────────────────────

    #[test]
    fn extracts_class_inheritance() {
        verify(
            "class Animal { public String name() { return \"x\"; } }\nclass Dog extends Animal { public String bark() { return \"woof\"; } }\n",
            &[
                ("class Animal", "class Animal"),
                ("class Dog", "class Dog extends Animal"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_interface_implements() {
        verify(
            "interface Greeter { String greet(); }\nclass Hello implements Greeter {\n    public String greet() { return \"hi\"; }\n}\n",
            &[
                ("interface Greeter", "interface Greeter"),
                ("class Hello", "class Hello implements Greeter"),
                ("class Hello > greet()", "public String greet"),
            ],
        );
    }

    #[test]
    fn extracts_nested_class() {
        verify(
            "class Outer {\n    static class Inner {\n        public int answer() { return 42; }\n    }\n}\n",
            &[
                ("class Outer > class Inner", "static class Inner"),
                ("class Inner > answer()", "public int answer"),
            ],
        );
    }

    #[test]
    fn extracts_static_method() {
        verify(
            "public class Util {\n    public static int square(int x) { return x * x; }\n}\n",
            &[("class Util > square()", "public static int square")],
        );
    }

    #[test]
    fn extracts_annotated_method() {
        verify(
            "public class Service {\n    @Override\n    @SuppressWarnings(\"unchecked\")\n    public String toString() { return \"x\"; }\n}\n",
            &[("class Service > toString()", "@Override")],
        );
    }

    #[test]
    fn extracts_method_with_throws_clause() {
        verify(
            "public class IO {\n    public void read() throws java.io.IOException { }\n}\n",
            &[("class IO > read()", "public void read")],
        );
    }

    #[test]
    fn extracts_abstract_method() {
        verify(
            "public abstract class Shape {\n    public abstract double area();\n}\n",
            &[
                ("class Shape", "public abstract class Shape"),
                ("class Shape > area()", "public abstract double area"),
            ],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn method_scope_lines_match_method_lines() {
        let src = "class Foo {\n    void run() {\n        return;\n    }\n}\n";
        let scopes = verify(src, &[("class Foo > run()", "void run")]);
        let run = find_scope(&scopes, "class Foo > run()").unwrap();
        assert_eq!(run.start_line, 2);
        assert_eq!(run.end_line, 4);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "class Foo {\r\n    void m() {}\r\n}\r\n",
            &[("class Foo", "class Foo")],
        );
    }

    #[test]
    fn preserves_truncated_class() {
        verify_cov("class Broken {\n    void m()");
    }

    #[test]
    fn preserves_package_only_file() {
        verify_cov("package com.example.app;\n");
    }
}
