//! TypeScript / TSX scope carver.

use std::collections::HashMap;

use tree_sitter::{Language, Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

/// Carve TypeScript source.  `is_tsx` selects the TSX dialect grammar
/// when the file extension is `.tsx`; everything else uses the plain
/// TypeScript grammar.
pub fn carve(source: &[u8], is_tsx: bool) -> Option<Vec<Scope>> {
    let language: Language = if is_tsx {
        tree_sitter_typescript::language_tsx()
    } else {
        tree_sitter_typescript::language_typescript()
    };
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
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
        .insert("method_definition", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("interface_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("type_alias_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("enum_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("abstract_class_declaration", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("generator_function_declaration", ChunkKind::Function);
    // Arrow-bound and function-expression-bound `const`s.
    rules
        .kind_to_chunk
        .insert("lexical_declaration", ChunkKind::Function);
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_declaration" => field_text(node, "name", source).map(|n| format!("function {n}")),
        "generator_function_declaration" => {
            field_text(node, "name", source).map(|n| format!("function* {n}"))
        }
        "method_definition" => field_text(node, "name", source),
        "class_declaration" => field_text(node, "name", source).map(|n| format!("class {n}")),
        "abstract_class_declaration" => {
            field_text(node, "name", source).map(|n| format!("class {n}"))
        }
        "interface_declaration" => field_text(node, "name", source).map(|n| format!("interface {n}")),
        "type_alias_declaration" => field_text(node, "name", source).map(|n| format!("type {n}")),
        "enum_declaration" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        "lexical_declaration" => arrow_binding_label(node, source),
        _ => None,
    }
}

fn arrow_binding_label(node: &Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let name = field_text(&child, "name", source)?;
        let value = child.child_by_field_name("value")?;
        match value.kind() {
            "arrow_function" | "function_expression" => return Some(format!("const {name}")),
            _ => {}
        }
    }
    None
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "class_declaration" | "abstract_class_declaration" => {
            field_text(node, "name", source).map(|n| format!("class {n}"))
        }
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
        verify_carve(src, Language::TypeScript, false, expectations)
    }

    fn verify_tsx(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::TypeScript, true, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::TypeScript, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_interface_and_class() {
        verify(
            "interface Foo { bar(): void }\nclass Baz {\n  qux(): void {}\n}\n",
            &[
                ("interface Foo", "interface Foo"),
                ("class Baz > qux", "qux"),
            ],
        );
    }

    #[test]
    fn extracts_type_alias() {
        verify(
            "type UserId = string;\nfunction lookup(id: UserId): void {}\n",
            &[("type UserId", "type UserId"), ("function lookup", "function lookup")],
        );
    }

    #[test]
    fn extracts_enum() {
        verify(
            "enum Color { Red, Green, Blue }\n",
            &[("enum Color", "enum Color")],
        );
    }

    // ── Function variants ────────────────────────────────────────────────────

    #[test]
    fn extracts_generic_function() {
        verify(
            "function identity<T>(x: T): T { return x; }\n",
            &[("function identity", "function identity<T>")],
        );
    }

    #[test]
    fn extracts_async_function() {
        verify(
            "async function load(): Promise<string> { return ''; }\n",
            &[("function load", "async function load")],
        );
    }

    #[test]
    fn extracts_generic_with_constraint() {
        verify(
            "function pick<T extends { id: string }>(items: T[]): T { return items[0]; }\n",
            &[("function pick", "function pick<T extends { id: string }>")],
        );
    }

    #[test]
    fn extracts_overloaded_function() {
        let src = "function fmt(x: number): string;\nfunction fmt(x: Date): string;\nfunction fmt(x: any): string {\n    return String(x);\n}\n";
        let scopes = verify(src, &[("function fmt", "function fmt")]);
        let count = scopes
            .iter()
            .filter(|s| s.qualified_path() == "function fmt")
            .count();
        assert!(count >= 1, "expected ≥1 fmt scope, got {count}");
    }

    #[test]
    fn extracts_arrow_const() {
        verify(
            "const square = (x: number): number => x * x;\n",
            &[("const square", "const square")],
        );
    }

    #[test]
    fn extracts_arrow_assigned_to_typed_const() {
        verify(
            "const square: (x: number) => number = (x) => x * x;\n",
            &[("const square", "const square")],
        );
    }

    #[test]
    fn extracts_default_export_function() {
        verify(
            "export default function defaultFn() { return 1; }\n",
            &[("function defaultFn", "export default function defaultFn")],
        );
    }

    #[test]
    fn extracts_exported_declarations() {
        verify(
            "export function alpha(): number { return 1; }\nexport class Beta { }\nexport interface Gamma { x: number; }\n",
            &[
                ("function alpha", "export function alpha"),
                ("class Beta", "export class Beta"),
                ("interface Gamma", "export interface Gamma"),
            ],
        );
    }

    // ── Class variants ───────────────────────────────────────────────────────

    #[test]
    fn extracts_generic_class() {
        verify(
            "class Container<T> {\n  constructor(public value: T) {}\n  get(): T { return this.value; }\n}\n",
            &[
                ("class Container", "class Container<T>"),
                ("class Container > get", "get()"),
            ],
        );
    }

    #[test]
    fn extracts_abstract_class_and_method() {
        verify(
            "abstract class Animal {\n    abstract speak(): string;\n    name(): string { return 'x'; }\n}\n",
            &[
                ("class Animal", "abstract class Animal"),
                ("class Animal > name", "name()"),
            ],
        );
    }

    #[test]
    fn extracts_interface_with_extends() {
        verify(
            "interface Base { id: string; }\ninterface User extends Base { name: string; }\n",
            &[
                ("interface Base", "interface Base"),
                ("interface User", "interface User extends Base"),
            ],
        );
    }

    #[test]
    fn extracts_type_alias_with_union() {
        verify(
            "type Result = { ok: true } | { ok: false; error: string };\n",
            &[("type Result", "type Result")],
        );
    }

    #[test]
    fn extracts_class_with_decorators() {
        // Tree-sitter wraps the class declaration in the leading
        // decorator line — the class scope starts at the decorator,
        // not the `class` keyword.  That's correct behaviour: the
        // decorator IS part of the declaration in TypeScript syntax.
        verify(
            "@Injectable()\nclass UserService {\n    @log\n    findOne(id: string) { return null; }\n}\n",
            &[
                ("class UserService", "@Injectable()"),
                ("class UserService > findOne", "findOne"),
            ],
        );
    }

    #[test]
    fn extracts_namespace_with_nested_function() {
        verify(
            "namespace Math {\n    export function square(x: number): number { return x * x; }\n}\n",
            &[("function square", "export function square")],
        );
    }

    #[test]
    fn extracts_empty_class_body() {
        verify("class Empty {}\n", &[("class Empty", "class Empty")]);
    }

    // ── TSX ──────────────────────────────────────────────────────────────────

    #[test]
    fn extracts_tsx_component() {
        verify_tsx(
            "function Button(props: { label: string }) {\n  return <button>{props.label}</button>;\n}\n",
            &[("function Button", "function Button")],
        );
    }

    #[test]
    fn extracts_tsx_component_with_hooks() {
        verify_tsx(
            "import { useState } from 'react';\n\nfunction Counter() {\n    const [n, setN] = useState(0);\n    return <button onClick={() => setN(n + 1)}>{n}</button>;\n}\n",
            &[("function Counter", "function Counter")],
        );
    }

    #[test]
    fn extracts_arrow_tsx_component() {
        verify_tsx(
            "const Greeting = ({ name }: { name: string }) => <p>Hi {name}</p>;\n",
            &[("const Greeting", "const Greeting")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn function_scope_spans_signature_through_close_brace() {
        let src = "function alpha(): number {\n  return 1;\n}\n";
        let scopes = verify(src, &[("function alpha", "function alpha")]);
        let alpha = find_scope(&scopes, "function alpha").unwrap();
        assert_eq!(alpha.start_line, 1);
        assert_eq!(alpha.end_line, 3);
    }

    #[test]
    fn interface_scope_is_interface_only() {
        let src = "interface Foo {\n  bar: number;\n  baz(): void;\n}\n";
        let scopes = verify(src, &[("interface Foo", "interface Foo")]);
        let foo = find_scope(&scopes, "interface Foo").unwrap();
        assert_eq!(foo.start_line, 1);
        assert_eq!(foo.end_line, 4);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_around_broken_middle_fn() {
        verify_cov("function alpha(): number { return 1; }\n\nfunction beta(x:): number { return 2; }\n\nfunction gamma(): number { return 3; }\n");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "function alpha() {}\r\nfunction beta() {}\r\n",
            &[("function alpha", "function alpha"), ("function beta", "function beta")],
        );
    }

    #[test]
    fn preserves_truncated_function() {
        verify_cov("function broken(\n");
    }

    #[test]
    fn preserves_file_with_only_imports() {
        verify_cov("import { a } from './a';\nimport { b } from './b';\n");
    }
}
