//! Rust scope carver.

use std::collections::HashMap;

use tree_sitter::{Node, Parser};

use super::tree_sitter_util::{carve_tree, field_text, LanguageRules};
use crate::code_read::types::{ChunkKind, Scope};

pub fn carve(source: &[u8]) -> Option<Vec<Scope>> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::language()).ok()?;
    let mut rules = LanguageRules {
        kind_to_chunk: HashMap::new(),
        identifier_for,
        enclosing_label,
    };
    rules
        .kind_to_chunk
        .insert("function_item", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("function_signature_item", ChunkKind::Function);
    rules
        .kind_to_chunk
        .insert("struct_item", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("union_item", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("enum_item", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("trait_item", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("type_item", ChunkKind::TypeDefinition);
    rules
        .kind_to_chunk
        .insert("const_item", ChunkKind::Constants);
    rules
        .kind_to_chunk
        .insert("static_item", ChunkKind::Constants);
    rules
        .kind_to_chunk
        .insert("macro_definition", ChunkKind::TopLevel);
    // Note: we deliberately do NOT register `impl_item` as a top-level
    // scope — its inner `function_item`s are emitted individually, each
    // with `impl Foo` in their scope path via the enclosing_label
    // callback.  Registering it here would produce a giant outer chunk
    // and force the splitter to fire on every impl block.
    carve_tree(&mut parser, source, &rules)
}

fn identifier_for(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_item" => field_text(node, "name", source).map(|n| format!("fn {n}")),
        "function_signature_item" => field_text(node, "name", source).map(|n| format!("fn {n}")),
        "struct_item" => field_text(node, "name", source).map(|n| format!("struct {n}")),
        "union_item" => field_text(node, "name", source).map(|n| format!("union {n}")),
        "enum_item" => field_text(node, "name", source).map(|n| format!("enum {n}")),
        "trait_item" => field_text(node, "name", source).map(|n| format!("trait {n}")),
        "type_item" => field_text(node, "name", source).map(|n| format!("type {n}")),
        "const_item" => field_text(node, "name", source).map(|n| format!("const {n}")),
        "static_item" => field_text(node, "name", source).map(|n| format!("static {n}")),
        "macro_definition" => field_text(node, "name", source).map(|n| format!("macro_rules! {n}")),
        _ => None,
    }
}

fn enclosing_label(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "mod_item" => field_text(node, "name", source).map(|n| format!("mod {n}")),
        "impl_item" => {
            let type_ = field_text(node, "type", source)?;
            match field_text(node, "trait", source) {
                Some(tr) => Some(format!("impl {tr} for {type_}")),
                None => Some(format!("impl {type_}")),
            }
        }
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::code_read::test_util::{find_scope, verify_carve, verify_coverage_only};
    use crate::code_read::types::{Scope, MAX_SCOPE_LINES};
    use crate::repo_scan::Language;

    fn verify(src: &str, expectations: &[(&str, &str)]) -> Vec<Scope> {
        verify_carve(src, Language::Rust, false, expectations)
    }

    fn verify_cov(src: &str) -> Vec<Scope> {
        verify_coverage_only(src, Language::Rust, false)
    }

    // ── Top-level items ──────────────────────────────────────────────────────

    #[test]
    fn extracts_top_level_fn() {
        verify(
            "fn alpha() {}\nfn beta() {}\n",
            &[("fn alpha", "fn alpha"), ("fn beta", "fn beta")],
        );
    }

    #[test]
    fn extracts_fns_in_named_module() {
        verify(
            "mod cache {\n    fn helper() {}\n}\n",
            &[("mod cache > fn helper", "fn helper")],
        );
    }

    #[test]
    fn extracts_impl_blocks_with_method_scopes() {
        verify(
            "struct Foo;\nimpl Foo {\n    fn bar(&self) {}\n    fn baz(&self) {}\n}\n",
            &[
                ("struct Foo", "struct Foo"),
                ("impl Foo > fn bar", "fn bar"),
                ("impl Foo > fn baz", "fn baz"),
            ],
        );
    }

    #[test]
    fn extracts_trait_impl_with_for_keyword() {
        verify(
            "struct Foo;\ntrait Bar { fn baz(&self); }\nimpl Bar for Foo {\n    fn baz(&self) {}\n}\n",
            &[
                ("struct Foo", "struct Foo"),
                ("trait Bar", "trait Bar"),
                ("impl Bar for Foo > fn baz", "fn baz"),
            ],
        );
    }

    #[test]
    fn extracts_struct_trait_and_enum_definitions() {
        verify(
            "struct A;\nenum B { X, Y }\ntrait C { fn d(&self); }\n",
            &[
                ("struct A", "struct A"),
                ("enum B", "enum B"),
                ("trait C", "trait C"),
            ],
        );
    }

    // ── Function variants ────────────────────────────────────────────────────

    #[test]
    fn extracts_async_fn() {
        verify(
            "async fn fetch(url: &str) -> String { String::new() }\n",
            &[("fn fetch", "async fn fetch")],
        );
    }

    #[test]
    fn extracts_unsafe_fn() {
        verify(
            "unsafe fn dangerous() {}\n",
            &[("fn dangerous", "unsafe fn dangerous")],
        );
    }

    #[test]
    fn extracts_async_unsafe_fn() {
        verify(
            "async unsafe fn dq() {}\n",
            &[("fn dq", "async unsafe fn dq")],
        );
    }

    #[test]
    fn extracts_generic_fn_with_where_clause() {
        verify(
            "fn identity<T>(x: T) -> T where T: Clone { x.clone() }\n",
            &[("fn identity", "fn identity<T>")],
        );
    }

    #[test]
    fn extracts_const_fn() {
        verify(
            "pub const fn squared(x: u32) -> u32 { x * x }\n",
            &[("fn squared", "pub const fn squared")],
        );
    }

    #[test]
    fn extracts_pub_visibility_levels() {
        verify(
            "pub fn alpha() {}\npub(crate) fn beta() {}\npub(super) fn gamma() {}\nfn delta() {}\n",
            &[
                ("fn alpha", "pub fn alpha"),
                ("fn beta", "pub(crate) fn beta"),
                ("fn gamma", "pub(super) fn gamma"),
                ("fn delta", "fn delta"),
            ],
        );
    }

    #[test]
    fn extracts_fn_with_lifetime_params() {
        verify(
            "fn borrows<'a>(x: &'a str) -> &'a str { x }\n",
            &[("fn borrows", "fn borrows<'a>")],
        );
    }

    #[test]
    fn extracts_fn_with_const_generic() {
        verify(
            "fn array<const N: usize>(_x: [u8; N]) {}\n",
            &[("fn array", "fn array<const N: usize>")],
        );
    }

    #[test]
    fn extracts_attribute_decorated_fn() {
        verify(
            "#[inline(always)]\n#[must_use]\npub fn fast() -> u32 { 42 }\n",
            &[("fn fast", "pub fn fast")],
        );
    }

    #[test]
    fn extracts_doc_decorated_fn() {
        verify(
            "/// Compute the answer.\n///\n/// Always returns 42.\npub fn answer() -> u32 { 42 }\n",
            &[("fn answer", "pub fn answer")],
        );
    }

    #[test]
    fn extracts_test_decorated_fn() {
        verify(
            "#[test]\nfn check_one() {\n    assert_eq!(1, 1);\n}\n",
            &[("fn check_one", "fn check_one")],
        );
    }

    // ── Type variants ────────────────────────────────────────────────────────

    #[test]
    fn extracts_tuple_struct() {
        verify(
            "pub struct Point(pub f32, pub f32);\n",
            &[("struct Point", "pub struct Point")],
        );
    }

    #[test]
    fn extracts_unit_struct() {
        verify(
            "struct Sentinel;\n",
            &[("struct Sentinel", "struct Sentinel")],
        );
    }

    #[test]
    fn extracts_struct_with_lifetime_and_generics() {
        verify(
            "pub struct View<'a, T: Clone>(pub &'a T);\n",
            &[("struct View", "pub struct View<'a, T: Clone>")],
        );
    }

    #[test]
    fn extracts_enum_with_data_variants() {
        verify(
            "enum Event {\n    Tick(u32),\n    Message { text: String },\n    Stop,\n}\n",
            &[("enum Event", "enum Event")],
        );
    }

    #[test]
    fn extracts_union_item() {
        verify(
            "union MyUnion { f: u32, i: i32 }\n",
            &[("union MyUnion", "union MyUnion")],
        );
    }

    #[test]
    fn extracts_type_alias_with_generics() {
        verify(
            "pub type Pair<A, B> = (A, B);\n",
            &[("type Pair", "pub type Pair<A, B>")],
        );
    }

    // ── Constants / statics / macros ─────────────────────────────────────────

    #[test]
    fn extracts_module_level_constants() {
        verify(
            "pub const MAX: u32 = 100;\nconst MIN: u32 = 0;\n",
            &[("const MAX", "pub const MAX"), ("const MIN", "const MIN")],
        );
    }

    #[test]
    fn extracts_const_with_complex_type() {
        verify(
            "pub const TABLE: [(u8, u8); 3] = [(1, 1), (2, 4), (3, 9)];\n",
            &[("const TABLE", "pub const TABLE")],
        );
    }

    #[test]
    fn extracts_static_items() {
        verify(
            "pub static GLOBAL: &str = \"hi\";\n",
            &[("static GLOBAL", "pub static GLOBAL")],
        );
    }

    #[test]
    fn extracts_static_with_mut_keyword() {
        verify(
            "pub static mut COUNTER: u32 = 0;\n",
            &[("static COUNTER", "pub static mut COUNTER")],
        );
    }

    #[test]
    fn extracts_macro_definition() {
        verify(
            "macro_rules! my_vec {\n    () => { Vec::new() };\n}\n",
            &[("macro_rules! my_vec", "macro_rules! my_vec")],
        );
    }

    #[test]
    fn extracts_macro_with_multiple_arms() {
        verify(
            "macro_rules! pick {\n    ($x:expr) => { $x };\n    ($x:expr, $y:expr) => { ($x, $y) };\n}\n",
            &[("macro_rules! pick", "macro_rules! pick")],
        );
    }

    // ── Nesting ──────────────────────────────────────────────────────────────

    #[test]
    fn extracts_nested_modules() {
        verify(
            "mod outer {\n    mod inner {\n        fn deep() {}\n    }\n}\n",
            &[("mod outer > mod inner > fn deep", "fn deep")],
        );
    }

    #[test]
    fn extracts_fn_inside_cfg_test_module() {
        verify(
            "#[cfg(test)]\nmod tests {\n    fn check_one() { assert_eq!(1, 1); }\n    fn check_two() { assert_eq!(2, 2); }\n}\n",
            &[
                ("mod tests > fn check_one", "fn check_one"),
                ("mod tests > fn check_two", "fn check_two"),
            ],
        );
    }

    #[test]
    fn extracts_impl_with_associated_const() {
        verify(
            "struct Foo;\nimpl Foo {\n    pub const MAX: u32 = 100;\n    pub fn new() -> Self { Foo }\n}\n",
            &[
                ("struct Foo", "struct Foo"),
                ("impl Foo > fn new", "pub fn new"),
                ("impl Foo > const MAX", "pub const MAX"),
            ],
        );
    }

    // ── Boundary correctness — exact line numbers ────────────────────────────

    #[test]
    fn fn_scope_starts_at_signature_line_and_ends_at_close_brace() {
        let src = "// preamble\n\nfn alpha() {\n    let x = 1;\n    let y = 2;\n}\n";
        let scopes = verify(src, &[("fn alpha", "fn alpha")]);
        let alpha = find_scope(&scopes, "fn alpha").unwrap();
        assert_eq!(alpha.start_line, 3, "start_line of fn alpha");
        assert_eq!(alpha.end_line, 6, "end_line of fn alpha (close brace)");
    }

    #[test]
    fn impl_method_scope_is_method_lines_only() {
        let src = "struct Foo(u32);\nimpl Foo {\n    fn bar(&self) {\n        self.0;\n    }\n}\n";
        let scopes = verify(src, &[("impl Foo > fn bar", "fn bar")]);
        let bar = find_scope(&scopes, "impl Foo > fn bar").unwrap();
        assert_eq!(bar.start_line, 3);
        assert_eq!(bar.end_line, 5);
    }

    #[test]
    fn struct_scope_starts_at_struct_keyword() {
        let src = "// header\nstruct Foo {\n    x: u32,\n    y: u32,\n}\n";
        let scopes = verify(src, &[("struct Foo", "struct Foo")]);
        let foo = find_scope(&scopes, "struct Foo").unwrap();
        assert_eq!(foo.start_line, 2);
        assert_eq!(foo.end_line, 5);
    }

    #[test]
    fn multiple_fns_have_disjoint_line_ranges() {
        let src = "fn alpha() {\n    1\n}\nfn beta() {\n    2\n}\n";
        let scopes = verify(src, &[("fn alpha", "fn alpha"), ("fn beta", "fn beta")]);
        let alpha = find_scope(&scopes, "fn alpha").unwrap();
        let beta = find_scope(&scopes, "fn beta").unwrap();
        assert_eq!(alpha.start_line, 1);
        assert_eq!(alpha.end_line, 3);
        assert_eq!(beta.start_line, 4);
        assert_eq!(beta.end_line, 6);
        assert!(alpha.end_line < beta.start_line);
    }

    // ── Partial / malformed file resilience ──────────────────────────────────
    //
    // The dispatcher must guarantee every line is reachable from some
    // scope even when tree-sitter fails on parts of the file.  These
    // are coverage-only tests because we can't predict which subset
    // of names will recover.

    #[test]
    fn preserves_around_broken_middle_fn() {
        verify_cov("fn alpha() { 1 }\n\nfn beta() {\n    let x = 1;\n\nfn gamma() { 3 }\n");
    }

    #[test]
    fn preserves_after_garbage_preamble() {
        verify_cov("$$$ not rust at all $$$\n\nfn good_one() { 1 }\n");
    }

    #[test]
    fn preserves_truncated_fn() {
        verify_cov("fn truncated() {\n    let x = 1;\n");
    }

    #[test]
    fn preserves_crlf_line_endings() {
        verify(
            "fn alpha() {}\r\nfn beta() {}\r\n",
            &[("fn alpha", "fn alpha"), ("fn beta", "fn beta")],
        );
    }

    #[test]
    fn preserves_module_level_use_statements() {
        verify_cov(
            "use std::collections::HashMap;\nuse std::sync::Arc;\n\nfn alpha() { let _ = (HashMap::<String, u32>::new(), Arc::new(0)); }\n",
        );
    }

    #[test]
    fn preserves_top_level_invalid_let() {
        verify_cov("let oops = 1;\n\nfn good() { let _ = 0; }\n");
    }

    #[test]
    fn preserves_doc_comments_only() {
        verify_cov("//! Module docs.\n//! More docs.\n");
    }

    #[test]
    fn preserves_attribute_only_file() {
        verify_cov("#![allow(dead_code)]\n#![feature(never_type)]\n");
    }

    #[test]
    fn preserves_only_comments() {
        verify_cov("// A\n// B\n/* multi\n   line */\n");
    }

    // ── Oversize split ───────────────────────────────────────────────────────

    #[test]
    fn splits_oversize_function_into_parts_under_max_lines() {
        let mut src = String::from("fn big() {\n");
        for _ in 0..200 {
            src.push_str("    let _ = 1;\n");
        }
        src.push_str("}\n");
        let scopes = verify_cov(&src);
        let parts: Vec<&Scope> = scopes
            .iter()
            .filter(|s| s.path[0].contains("fn big"))
            .collect();
        assert!(
            parts.len() >= 2,
            "oversize fn should split into ≥2 parts, got {}",
            parts.len()
        );
        for p in &parts {
            assert!(
                p.end_line - p.start_line < MAX_SCOPE_LINES,
                "part too large: {p:?}",
            );
        }
    }
}
