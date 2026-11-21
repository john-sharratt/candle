//! What changes in an item's text when it moves one module deeper.
//!
//! A moved item lives in a child of the module it was written in, so:
//!
//! - a path through `super` reaches one module less far: `super::x` becomes
//!   `super::super::x`, and `pub(super)` becomes `pub(in super::super)`;
//! - its siblings and its old module reach it from outside its own module
//!   now: an item, an inherent method or a struct field with no visibility of
//!   its own becomes `pub(super)` — visible in the old module and every child
//!   of it, which is where its users were.
//!
//! Inside a `mod { }` body paths are relative to that module, which moves with
//! it, so nothing in one changes. An item that stays in `mod.rs` changes
//! nothing either — none of these edits applies to its bytes.

use std::ops::Range;

use tree_sitter::{Node, Parser};

/// One change to the source: replace `range` with `with`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub range: Range<usize>,
    pub with: String,
}

/// Item kinds that take a visibility.
const VISIBLE_ITEMS: &[&str] = &[
    "function_item",
    "struct_item",
    "enum_item",
    "union_item",
    "trait_item",
    "type_item",
    "const_item",
    "static_item",
];

/// Inherent-impl member kinds that take a visibility.
const VISIBLE_MEMBERS: &[&str] = &["function_item", "const_item", "type_item"];

/// Every edit moving any top-level item of `source` one module deeper would
/// make, in source order. Kept parts (uses, docs, macros) get none: they do
/// not move.
///
/// `module_file` names the file a file-less `mod` declared inside an inline
/// module is in, from its path below the split's folder without an extension
/// (`tests/ledger`): `tests/ledger.rs`, or `tests/ledger/mod.rs`.
pub fn edits(source: &str, module_file: &dyn Fn(&str) -> String) -> Result<Vec<Edit>, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| format!("the Rust grammar does not load: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or("the file does not parse")?;
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut cursor = tree.root_node().walk();
    for node in tree.root_node().children(&mut cursor) {
        if !node.is_named() || stays(&node) {
            continue;
        }
        widen_item(&node, bytes, "pub(super)", &mut out);
        // An inline module moved whole is reached at its old path, through
        // the re-export of its file.
        if node.kind() == "mod_item" {
            widen(&node, bytes, "pub(super)", &mut out);
        }
        deepen(&node, bytes, &mut Vec::new(), module_file, &mut out);
    }
    out.sort_by_key(|e| (e.range.start, e.range.end));
    Ok(out)
}

/// Whether a root child stays in `mod.rs` — see `layout::classify`.
fn stays(node: &Node) -> bool {
    match node.kind() {
        "use_declaration"
        | "extern_crate_declaration"
        | "inner_attribute_item"
        | "macro_definition"
        | "line_comment"
        | "block_comment"
        | "attribute_item" => true,
        "mod_item" => node.child_by_field_name("body").is_none(),
        _ => false,
    }
}

/// `vis` before an item with no visibility of its own, before each of its
/// fields with none if it is a struct or union, and before each of its
/// members with none if it is an inherent impl.
fn widen_item(node: &Node, bytes: &[u8], vis: &str, out: &mut Vec<Edit>) {
    if VISIBLE_ITEMS.contains(&node.kind()) {
        widen(node, bytes, vis, out);
    }
    if matches!(node.kind(), "struct_item" | "union_item") {
        widen_fields(node, bytes, vis, out);
    }
    if node.kind() == "impl_item" && node.child_by_field_name("trait").is_none() {
        if let Some(list) = node.child_by_field_name("body") {
            let mut c = list.walk();
            for m in list.children(&mut c) {
                if VISIBLE_MEMBERS.contains(&m.kind()) {
                    widen(&m, bytes, vis, out);
                }
            }
        }
    }
}

/// `vis` before `node` when it has no visibility of its own. A `const _` — a
/// compile-time assertion — takes none.
fn widen(node: &Node, bytes: &[u8], vis: &str, out: &mut Vec<Edit>) {
    let text = node.utf8_text(bytes).unwrap_or("");
    if node.kind() == "const_item"
        && text
            .strip_prefix("const")
            .and_then(|rest| rest.trim_start().strip_prefix('_'))
            .is_some_and(|rest| rest.trim_start().starts_with(':'))
    {
        return;
    }
    let mut c = node.walk();
    let has = node
        .children(&mut c)
        .any(|ch| ch.kind() == "visibility_modifier");
    if !has {
        let at = node.start_byte();
        out.push(Edit {
            range: at..at,
            with: format!("{vis} "),
        });
    }
}

/// `vis` before each of a struct's fields that has no visibility.
fn widen_fields(node: &Node, bytes: &[u8], vis: &str, out: &mut Vec<Edit>) {
    let Some(list) = node.child_by_field_name("body") else {
        return;
    };
    let mut c = list.walk();
    match list.kind() {
        "field_declaration_list" => {
            for f in list.children(&mut c) {
                if f.kind() == "field_declaration" {
                    widen(&f, bytes, vis, out);
                }
            }
        }
        "ordered_field_declaration_list" => {
            // `( [attrs] [vis] Type, ... )`: a type with no visibility just
            // before it gets one.
            let kids: Vec<Node> = list.children(&mut c).collect();
            for (i, k) in kids.iter().enumerate() {
                let is_type = k.is_named()
                    && !matches!(
                        k.kind(),
                        "visibility_modifier" | "attribute_item" | "line_comment" | "block_comment"
                    );
                let after_vis = i > 0 && kids[i - 1].kind() == "visibility_modifier";
                if is_type && !after_vis {
                    let at = k.start_byte();
                    out.push(Edit {
                        range: at..at,
                        with: format!("{vis} "),
                    });
                }
            }
        }
        _ => {}
    }
}

/// Every `super` path and relative visibility in `node` that leaves the file's
/// module, one module deeper. `inside` names the `mod { }` bodies of the
/// moved item that enclose `node`, outermost first: a path climbing `k`
/// modules from there leaves the file's module when `k` is more than their
/// number, and the file's module now lies one further down. A path that stays
/// inside the moved item reaches what it did.
///
/// A file-less `mod x;` inside them is looked for under the new file's name,
/// so it is given a `#[path]` back to its file: from the folder of
/// `c.rs`'s `m` (`<split>/c/m/`) up to the split's folder and down to `m/x.rs`.
fn deepen(
    node: &Node,
    bytes: &[u8],
    inside: &mut Vec<String>,
    module_file: &dyn Fn(&str) -> String,
    out: &mut Vec<Edit>,
) {
    let depth = inside.len();
    match node.kind() {
        "mod_item" if depth > 0 && node.child_by_field_name("body").is_none() => {
            let has_path = node
                .prev_named_sibling()
                .filter(|p| p.kind() == "attribute_item")
                .and_then(|p| p.utf8_text(bytes).ok())
                .is_some_and(|t| t.trim_start_matches("#[").trim_start().starts_with("path"));
            let name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(bytes).ok());
            if let (false, Some(name)) = (has_path, name) {
                let below = format!("{}/{name}", inside.join("/"));
                let at = node.start_byte();
                out.push(Edit {
                    range: at..at,
                    with: format!(
                        "#[path = \"{}{}\"]\n{}",
                        "../".repeat(depth + 1),
                        module_file(&below),
                        indent_at(bytes, at)
                    ),
                });
            }
            return;
        }
        "visibility_modifier" => {
            let text = node.utf8_text(bytes).unwrap_or("");
            if let Some(with) = deeper_visibility(text, depth) {
                out.push(Edit {
                    range: node.start_byte()..node.end_byte(),
                    with,
                });
            }
            return;
        }
        "super" => {
            // Only the first `super` of a chain is edited, once for the chain.
            let start = node.start_byte();
            if !follows_path_separator(bytes, start) && super_chain(bytes, start) > depth {
                out.push(Edit {
                    range: start..node.end_byte(),
                    with: "super::super".to_string(),
                });
            }
            return;
        }
        _ => {}
    }
    let entered = node.kind() == "mod_item";
    if entered {
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(bytes).ok())
            .unwrap_or("");
        inside.push(name.to_string());
    }
    let mut c = node.walk();
    for ch in node.children(&mut c) {
        deepen(&ch, bytes, inside, module_file, out);
    }
    if entered {
        inside.pop();
    }
}

/// The indentation of the line `at` is on, up to `at`.
fn indent_at(bytes: &[u8], at: usize) -> String {
    let start = bytes[..at]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |nl| nl + 1);
    String::from_utf8_lossy(&bytes[start..at])
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

/// Whether the source before `at`, past whitespace, is `::` — so the `super`
/// at `at` continues a chain rather than starting one.
fn follows_path_separator(bytes: &[u8], at: usize) -> bool {
    let before = String::from_utf8_lossy(&bytes[..at]);
    before.trim_end().ends_with("::")
}

/// How many `super`s the path starting at `at` climbs: `super::super::x` is 2.
fn super_chain(bytes: &[u8], at: usize) -> usize {
    let text = String::from_utf8_lossy(&bytes[at..]);
    let mut rest = text.as_ref();
    let mut k = 0;
    while let Some(after) = rest.strip_prefix("super") {
        if after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            break;
        }
        k += 1;
        match after.trim_start().strip_prefix("::") {
            Some(next) => rest = next.trim_start(),
            None => break,
        }
    }
    k
}

/// A visibility written relative to its module, one module deeper when it
/// reaches the file's module or past it from `depth` modules inside the item
/// — the file's module is now the child, one further down; `None` for one
/// that does not (`pub`, `pub(crate)`, a reach within the item).
fn deeper_visibility(text: &str, depth: usize) -> Option<String> {
    let inner: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let (k, path) = match inner.as_str() {
        "pub(self)" => (0, "self".to_string()),
        "pub(super)" => (1, "super".to_string()),
        _ => {
            let path = inner
                .strip_prefix("pub(in")
                .and_then(|p| p.strip_suffix(')'))?;
            let k = path.split("::").take_while(|s| *s == "super").count();
            (k, path.to_string())
        }
    };
    if k < depth || (k == 0 && path != "self") {
        return None;
    }
    Some(match path.as_str() {
        "self" => "pub(super)".to_string(),
        _ => format!("pub(in super::{path})"),
    })
}

/// The edits that let the shares of a `mod { }` split across files see each
/// other: each item of its body with no visibility of its own is seen as far
/// as `seen` — the module every share is under. A test module's helpers,
/// written beside the tests that call them, are called from every share. A
/// `#[test]` function is called by the harness alone, and is left as it is.
pub fn share_edits(source: &str, seen: &str) -> Result<Vec<Edit>, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| format!("the Rust grammar does not load: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or("the file does not parse")?;
    let mut out = Vec::new();
    let mut cursor = tree.root_node().walk();
    for node in tree.root_node().children(&mut cursor) {
        if node.kind() != "mod_item" {
            continue;
        }
        let Some(list) = node.child_by_field_name("body") else {
            continue;
        };
        let mut c = list.walk();
        // Whether an attribute since the last item marks the next one a test.
        let mut a_test = false;
        for m in list.children(&mut c) {
            if m.kind() == "attribute_item" {
                let text = m.utf8_text(source.as_bytes()).unwrap_or("");
                a_test |= text.contains("test]") || text.contains("::test");
                continue;
            }
            if std::mem::take(&mut a_test) && m.kind() == "function_item" {
                continue;
            }
            widen_item(&m, source.as_bytes(), seen, &mut out);
            // A module declared in one share is reached from the others.
            if m.kind() == "mod_item" {
                widen(&m, source.as_bytes(), seen, &mut out);
            }
        }
    }
    out.sort_by_key(|e| (e.range.start, e.range.end));
    Ok(out)
}

/// `source[range]` with every edit inside it applied. An edit straddling the
/// range's edge is a bug in the caller, and refused.
pub fn apply(source: &str, range: Range<usize>, edits: &[Edit]) -> Result<String, String> {
    let mut out = String::with_capacity(range.len() + 64);
    let mut at = range.start;
    let inside = |e: &&Edit| {
        e.range.start < range.end
            && (e.range.end > range.start || (e.range.is_empty() && e.range.start >= range.start))
    };
    for e in edits.iter().filter(inside) {
        if e.range.start < range.start || e.range.end > range.end {
            return Err(format!(
                "an edit at {:?} straddles the moved range {range:?}",
                e.range
            ));
        }
        out.push_str(&source[at..e.range.start]);
        out.push_str(&e.with);
        at = e.range.end;
    }
    // An insertion exactly at the range's end belongs to the next part.
    out.push_str(&source[at..range.end]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each module's file as `<path>.rs`, as when none has a folder.
    fn flat(below: &str) -> String {
        format!("{below}.rs")
    }

    fn moved(src: &str) -> String {
        let e = edits(src, &flat).unwrap();
        apply(src, 0..src.len(), &e).unwrap()
    }

    /// **A file-less `mod` inside a moved module keeps its file**: it is given
    /// the `#[path]` from the new file's folder back to where its file is,
    /// and one that already has a `#[path]` is left.
    #[test]
    fn a_nested_file_module_keeps_its_file() {
        let src = "mod tests {\n    mod ledger;\n    #[path = \"x.rs\"]\n    mod pinned;\n    mod inner {\n        mod deep;\n    }\n}\n";
        assert_eq!(
            moved(src),
            "pub(super) mod tests {\n    #[path = \"../../tests/ledger.rs\"]\n    mod ledger;\n    #[path = \"x.rs\"]\n    mod pinned;\n    \
             mod inner {\n        #[path = \"../../../tests/inner/deep.rs\"]\n        mod deep;\n    }\n}\n"
        );
        let folders = |below: &str| format!("{below}/mod.rs");
        let e = edits("mod tests {\n    mod ledger;\n}\n", &folders).unwrap();
        assert_eq!(
            e.iter()
                .find(|e| e.with.starts_with("#[path"))
                .map(|e| e.with.as_str()),
            Some("#[path = \"../../tests/ledger/mod.rs\"]\n    ")
        );
    }

    /// **A private item, inherent method and field become `pub(super)`**;
    /// what already says who sees it is left, and a trait impl's methods
    /// take no visibility.
    #[test]
    fn what_had_no_visibility_becomes_pub_super() {
        let src = "struct A {\n    x: u8,\n    pub y: u8,\n}\nstruct T(u8, pub u16);\n\
                   impl A {\n    fn f(&self) {}\n    pub fn g(&self) {}\n}\n\
                   impl Display for A {\n    fn fmt(&self) {}\n}\nconst K: u8 = 1;\n\
                   const _: () = assert!(true);\nconst _K: u8 = 2;\n";
        assert_eq!(
            moved(src),
            "pub(super) struct A {\n    pub(super) x: u8,\n    pub y: u8,\n}\n\
             pub(super) struct T(pub(super) u8, pub u16);\n\
             impl A {\n    pub(super) fn f(&self) {}\n    pub fn g(&self) {}\n}\n\
             impl Display for A {\n    fn fmt(&self) {}\n}\npub(super) const K: u8 = 1;\n\
             const _: () = assert!(true);\npub(super) const _K: u8 = 2;\n"
        );
    }

    /// **Relative paths that leave the file reach one module further**, in
    /// code and in visibilities, a chain gaining one `super` and not one per
    /// link. Inside a `mod { }`, a path within the file is left; one climbing
    /// out of it, and a visibility reaching the file's module — now the child
    /// — are deepened; one reaching only within the item is left.
    #[test]
    fn relative_paths_go_one_module_deeper() {
        let src = "pub(super) fn a() -> super::X {\n    super::b()\n}\n\
                   pub(in super::m) fn c() -> super :: super::Y {}\n\
                   mod tests {\n    use super::*;\n    use super::super::lock;\n    \
                   pub(super) fn d() {}\n    pub(in super::super::q) fn e() { super::super::f(); }\n    \
                   pub(self) fn g() {}\n}\n";
        assert_eq!(
            moved(src),
            "pub(in super::super) fn a() -> super::super::X {\n    super::super::b()\n}\n\
             pub(in super::super::m) fn c() -> super::super :: super::Y {}\n\
             pub(super) mod tests {\n    use super::*;\n    use super::super::super::lock;\n    \
             pub(in super::super) fn d() {}\n    pub(in super::super::super::q) fn e() { super::super::super::f(); }\n    \
             pub(self) fn g() {}\n}\n"
        );
    }

    /// **A split module's private items are seen by every share**; what has
    /// a visibility, and items outside a `mod { }`, are left.
    #[test]
    fn a_split_modules_items_are_seen_by_every_share() {
        let src = "fn top() {}\nmod tests {\n    use super::*;\n    const N: u8 = 1;\n    pub fn h() {}\n    fn t() {}\n    \
                   #[test]\n    fn a_test() {}\n    #[tokio::test]\n    async fn b_test() {}\n    \
                   struct Rig {\n        cfg: u8,\n    }\n    impl Rig {\n        fn new() -> Self { todo() }\n    }\n    \
                   pub(super) fn fixture() {}\n}\n";
        let e = share_edits(src, "pub(in super::super)").unwrap();
        assert_eq!(
            apply(src, 0..src.len(), &e).unwrap(),
            "fn top() {}\nmod tests {\n    use super::*;\n    pub(in super::super) const N: u8 = 1;\n    \
             pub fn h() {}\n    pub(in super::super) fn t() {}\n    \
             #[test]\n    fn a_test() {}\n    #[tokio::test]\n    async fn b_test() {}\n    \
             pub(in super::super) struct Rig {\n        pub(in super::super) cfg: u8,\n    }\n    \
             impl Rig {\n        pub(in super::super) fn new() -> Self { todo() }\n    }\n    \
             pub(super) fn fixture() {}\n}\n"
        );
    }

    /// A kept part — a `use`, a macro — is never edited.
    #[test]
    fn kept_parts_are_not_edited() {
        let src = "use super::x;\nmacro_rules! m { () => { super::y() } }\n";
        assert_eq!(edits(src, &flat).unwrap(), vec![]);
    }

    /// Only the edits inside a range apply to it; one across its edge is
    /// refused.
    #[test]
    fn edits_apply_inside_their_range_only() {
        let e = vec![
            Edit {
                range: 2..2,
                with: "X".into(),
            },
            Edit {
                range: 6..6,
                with: "Y".into(),
            },
        ];
        assert_eq!(apply("abcdefgh", 0..4, &e).unwrap(), "abXcd");
        assert_eq!(apply("abcdefgh", 4..8, &e).unwrap(), "efYgh");
        // An insertion at a range's start is the range's.
        assert_eq!(apply("abcdefgh", 2..4, &e).unwrap(), "Xcd");
        let straddle = vec![Edit {
            range: 3..5,
            with: "Z".into(),
        }];
        assert!(apply("abcdefgh", 0..4, &straddle).is_err());
    }
}
