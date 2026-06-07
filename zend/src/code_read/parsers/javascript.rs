//! JavaScript scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_javascript::language())
        .ok()?;
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
        .insert("generator_function_declaration", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("method_definition", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("class_declaration", ChunkKind::TypeDefinition);
    // Top-level `const foo = (…) => {}` bindings surface as
    // lexical_declaration nodes — emit them as Functions so the
    // arrow form gets a scope even though tree-sitter calls it a
    // variable.  identifier_for filters to keep only arrow-bound
    // declarations.
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
        "lexical_declaration" => arrow_binding_label(node, source),
        _ => None,
    }
}

/// Return a label for a `const x = (...) => {…}` / `let x = function(){…}`
/// binding; `None` for plain value bindings so the carver doesn't
/// emit a turn for every top-level constant.
fn arrow_binding_label(node: &Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let name = field_text(&child, "name", source)?;
        let value = child.child_by_field_name("value")?;
        match value.kind() {
            "arrow_function" => return Some(format!("const {name}")),
            "function_expression" => return Some(format!("const {name}")),
            _ => {}
        }
    }
    None
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() == "class_declaration" {
        field_text(node, "name", source).map(|n| format!("class {n}"))
    } else {
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::Scope;
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::JavaScript, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::JavaScript, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_function_and_class_method() {
        verify(
            "function alpha() {}\nclass Foo {\n  bar() {}\n}\n",
            &[("function alpha", "function alpha"), ("class Foo > bar", "bar")],
        );
    }

    #[test]
    fn extracts_arrow_function_const_binding() {
        verify(
            "const add = (a, b) => a + b;\nconst greet = (name) => { return `hi`; };\n",
            &[("const add", "const add"), ("const greet", "const greet")],
        );
    }

    #[test]
    fn extracts_function_expression_const_binding() {
        verify(
            "const fn1 = function () { return 1; };\nconst fn2 = function named() { return 2; };\n",
            &[("const fn1", "const fn1"), ("const fn2", "const fn2")],
        );
    }

    #[test]
    fn extracts_async_function() {
        verify(
            "async function fetchUser() { return await get(); }\n",
            &[("function fetchUser", "async function fetchUser")],
        );
    }

    #[test]
    fn extracts_generator_function() {
        verify(
            "function* counter() { yield 1; yield 2; }\n",
            &[("function* counter", "function* counter")],
        );
    }

    #[test]
    fn extracts_named_function_expression_binding() {
        verify(
            "const handler = function namedInner(e) { return e; };\n",
            &[("const handler", "const handler")],
        );
    }

    #[test]
    fn extracts_async_arrow_binding() {
        verify(
            "const fetchUser = async (id) => { return await db.get(id); };\n",
            &[("const fetchUser", "const fetchUser")],
        );
    }

    // ── Classes ──────────────────────────────────────────────────────────────

    #[test]
    fn extracts_class_with_constructor_and_static() {
        verify(
            "class Box {\n  constructor(x) { this.x = x; }\n  static make() { return new Box(0); }\n  get value() { return this.x; }\n}\n",
            &[
                ("class Box > constructor", "constructor"),
                ("class Box > make", "static make"),
                ("class Box > value", "get value"),
            ],
        );
    }

    #[test]
    fn extracts_class_extension() {
        verify(
            "class Animal { speak() { return 'sound'; } }\nclass Dog extends Animal { speak() { return 'woof'; } }\n",
            &[
                ("class Animal > speak", "speak"),
                ("class Dog > speak", "speak"),
            ],
        );
    }

    #[test]
    fn extracts_class_with_private_field_method() {
        verify(
            "class Counter {\n  #count = 0;\n  inc() { this.#count++; }\n  get value() { return this.#count; }\n}\n",
            &[
                ("class Counter > inc", "inc"),
                ("class Counter > value", "get value"),
            ],
        );
    }

    #[test]
    fn extracts_static_class_method() {
        verify(
            "class Util {\n  static log(msg) { console.log(msg); }\n}\n",
            &[("class Util > log", "static log")],
        );
    }

    #[test]
    fn extracts_class_with_static_block() {
        verify(
            "class App {\n  static instance;\n  static {\n    App.instance = new App();\n  }\n}\n",
            &[("class App", "class App")],
        );
    }

    // ── Boundary correctness ─────────────────────────────────────────────────

    #[test]
    fn function_scope_lines_match_declaration_through_close_brace() {
        let src = "// header\nfunction alpha() {\n  return 1;\n}\n";
        let scopes = verify(src, &[("function alpha", "function alpha")]);
        let alpha = find_scope(&scopes, "function alpha").unwrap();
        assert_eq!(alpha.start_line, 2);
        assert_eq!(alpha.end_line, 4);
    }

    #[test]
    fn method_scope_lines_match_method_only() {
        let src = "class Foo {\n  bar() {\n    return 1;\n  }\n}\n";
        let scopes = verify(src, &[("class Foo > bar", "bar")]);
        let bar = find_scope(&scopes, "class Foo > bar").unwrap();
        assert_eq!(bar.start_line, 2);
        assert_eq!(bar.end_line, 4);
    }

    #[test]
    fn arrow_const_scope_is_full_lexical_declaration() {
        let src = "const add = (a, b) => {\n  return a + b;\n};\n";
        let scopes = verify(src, &[("const add", "const add")]);
        let add = find_scope(&scopes, "const add").unwrap();
        assert_eq!(add.start_line, 1);
        assert_eq!(add.end_line, 3);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────

    #[test]
    fn preserves_around_broken_middle_fn() {
        verify_cov("function alpha() { return 1; }\n\nfunction beta() { return  // bad }\n\nfunction gamma() { return 3; }\n");
    }

    #[test]
    fn preserves_crlf_endings() {
        verify(
            "function alpha() {}\r\nfunction beta() {}\r\n",
            &[("function alpha", "function alpha"), ("function beta", "function beta")],
        );
    }

    #[test]
    fn preserves_empty_class() {
        verify("class Empty {}\n", &[("class Empty", "class Empty")]);
    }

    #[test]
    fn preserves_file_with_only_imports() {
        verify_cov("import { foo } from './foo.js';\nimport { bar } from './bar.js';\n");
    }

    #[test]
    fn preserves_truncated_function() {
        verify_cov("function broken(\n");
    }

    #[test]
    fn preserves_object_method_shorthand_export() {
        // Method-shorthand inside an object literal isn't a
        // first-class scope; the binding still gets prefilled.
        verify_cov("export const api = {\n    fetch(id) { return id; },\n    save(x) { return x; },\n};\n");
    }

    #[test]
    fn preserves_value_constants_without_function() {
        // Plain value bindings — not arrow / function expressions.
        // Don't crash; coverage still holds.
        verify_cov("const MAX = 100;\nconst NAME = 'x';\n");
    }

    #[test]
    fn preserves_iife_pattern() {
        verify_cov("const result = (function compute() { return 1; })();\n");
    }
}
