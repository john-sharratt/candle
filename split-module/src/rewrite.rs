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
pub fn edits(source: &str) -> Result<Vec<Edit>, String> {
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
        if VISIBLE_ITEMS.contains(&node.kind()) {
            widen(&node, bytes, &mut out);
        }
        if matches!(node.kind(), "struct_item" | "union_item") {
            widen_fields(&node, bytes, &mut out);
        }
        if node.kind() == "impl_item" && node.child_by_field_name("trait").is_none() {
            if let Some(list) = node.child_by_field_name("body") {
                let mut c = list.walk();
                for m in list.children(&mut c) {
                    if VISIBLE_MEMBERS.contains(&m.kind()) {
                        widen(&m, bytes, &mut out);
                    }
                }
            }
        }
        if node.kind() != "mod_item" {
            deepen(&node, bytes, &mut out);
        }
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

/// `pub(super) ` before `node` when it has no visibility of its own. A
/// `const _` — a compile-time assertion — takes none.
fn widen(node: &Node, bytes: &[u8], out: &mut Vec<Edit>) {
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
            with: "pub(super) ".to_string(),
        });
    }
}

/// `pub(super) ` before each of a struct's fields that has no visibility.
fn widen_fields(node: &Node, bytes: &[u8], out: &mut Vec<Edit>) {
    let Some(list) = node.child_by_field_name("body") else {
        return;
    };
    let mut c = list.walk();
    match list.kind() {
        "field_declaration_list" => {
            for f in list.children(&mut c) {
                if f.kind() == "field_declaration" {
                    widen(&f, bytes, out);
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
                        with: "pub(super) ".to_string(),
                    });
                }
            }
        }
        _ => {}
    }
}

/// Every `super` path and relative visibility in `node`, one module deeper —
/// but nothing inside a nested `mod { }`, whose paths move with it.
fn deepen(node: &Node, bytes: &[u8], out: &mut Vec<Edit>) {
    match node.kind() {
        "mod_item" => return,
        "visibility_modifier" => {
            let text = node.utf8_text(bytes).unwrap_or("");
            if let Some(with) = deeper_visibility(text) {
                out.push(Edit {
                    range: node.start_byte()..node.end_byte(),
                    with,
                });
            }
            return;
        }
        "super" => {
            out.push(Edit {
                range: node.start_byte()..node.end_byte(),
                with: "super::super".to_string(),
            });
            return;
        }
        _ => {}
    }
    let mut c = node.walk();
    for ch in node.children(&mut c) {
        deepen(&ch, bytes, out);
    }
}

/// A visibility written relative to its module, one module deeper; `None`
/// for one that is not relative (`pub`, `pub(crate)`).
fn deeper_visibility(text: &str) -> Option<String> {
    let inner: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    match inner.as_str() {
        "pub(self)" => Some("pub(super)".to_string()),
        "pub(super)" => Some("pub(in super::super)".to_string()),
        _ => inner
            .strip_prefix("pub(insuper")
            .map(|rest| format!("pub(in super::super{})", rest.trim_end_matches(')'))),
    }
}

/// The edits that let the shares of a `mod { }` split across files see each
/// other: each item of its body with no visibility of its own is seen as far
/// as `seen` — the module every share is under. A test module's helpers,
/// written beside the tests that call them, are called from every share.
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
        for m in list.children(&mut c) {
            if VISIBLE_ITEMS.contains(&m.kind()) {
                let mut cc = m.walk();
                if !m
                    .children(&mut cc)
                    .any(|ch| ch.kind() == "visibility_modifier")
                {
                    let at = m.start_byte();
                    out.push(Edit {
                        range: at..at,
                        with: format!("{seen} "),
                    });
                }
            }
        }
    }
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

    fn moved(src: &str) -> String {
        let e = edits(src).unwrap();
        apply(src, 0..src.len(), &e).unwrap()
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

    /// **Relative paths reach one module further**, in code and in
    /// visibilities; a nested `mod { }` keeps its own.
    #[test]
    fn relative_paths_go_one_module_deeper() {
        let src = "pub(super) fn a() -> super::X {\n    super::b()\n}\n\
                   pub(in super::m) fn c() {}\n\
                   mod tests {\n    use super::*;\n    pub(super) fn d() {}\n}\n";
        assert_eq!(
            moved(src),
            "pub(in super::super) fn a() -> super::super::X {\n    super::super::b()\n}\n\
             pub(in super::super::m) fn c() {}\n\
             mod tests {\n    use super::*;\n    pub(super) fn d() {}\n}\n"
        );
    }

    /// **A split module's private items are seen by every share**; what has
    /// a visibility, and items outside a `mod { }`, are left.
    #[test]
    fn a_split_modules_items_are_seen_by_every_share() {
        let src = "fn top() {}\nmod tests {\n    use super::*;\n    const N: u8 = 1;\n    pub fn h() {}\n    fn t() {}\n}\n";
        let e = share_edits(src, "pub(in super::super)").unwrap();
        assert_eq!(
            apply(src, 0..src.len(), &e).unwrap(),
            "fn top() {}\nmod tests {\n    use super::*;\n    pub(in super::super) const N: u8 = 1;\n    \
             pub fn h() {}\n    pub(in super::super) fn t() {}\n}\n"
        );
    }

    /// A kept part — a `use`, a macro — is never edited.
    #[test]
    fn kept_parts_are_not_edited() {
        let src = "use super::x;\nmacro_rules! m { () => { super::y() } }\n";
        assert_eq!(edits(src).unwrap(), vec![]);
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
