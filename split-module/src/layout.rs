//! A Rust file as the parts a split moves: what stays in `mod.rs`, and the
//! items — some of them containers whose members can go to different files.
//!
//! Every byte of the file belongs to exactly one part, or is whitespace
//! between parts. A comment or an outer attribute belongs to the item after
//! it (a section comment introduces what follows); one with nothing after it
//! belongs to the part before it.

use std::ops::Range;

use tree_sitter::{Node, Parser};

/// Where a top-level part of the file goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stay {
    /// Stays in `mod.rs` where it was: module docs, inner attributes, `use`
    /// declarations, `extern crate`, file-less `mod x;` declarations.
    Header,
    /// Stays in `mod.rs` ahead of every child module: a `macro_rules!`
    /// definition, which is visible only to code after it.
    Macro,
}

/// A part that stays in `mod.rs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kept {
    pub stay: Stay,
    pub span: Range<usize>,
}

/// One member of a container's body — a method of an `impl`, an item of a
/// `mod { }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    /// The member with its leading comments and attributes.
    pub span: Range<usize>,
    /// A `use` or an inner attribute inside a `mod { }` body: it is about
    /// the module, so every part of the body the split writes carries it.
    pub is_use: bool,
}

/// The parts of a container: its header up to and including `{`, its
/// members, and its closing `}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    /// From the item's start through its `{`.
    pub head: Range<usize>,
    pub members: Vec<Member>,
    /// The closing `}` to the item's end.
    pub tail: Range<usize>,
}

/// A top-level item a split may move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// What the item is and is called: `fn new`, `impl Scheduler`,
    /// `mod tests`.
    pub name: String,
    /// The item with its leading comments and attributes.
    pub span: Range<usize>,
    /// An `impl` or a `mod { }` with members.
    pub body: Option<Body>,
    /// For any `mod { }`: what lies between its braces. A plan can hoist the
    /// module into a file of its own, which then holds exactly this.
    pub inline: Option<Range<usize>>,
}

/// The whole file, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub kept: Vec<Kept>,
    pub items: Vec<Item>,
}

/// How a root child is placed.
enum Part {
    Kept(Stay),
    Item,
    /// A comment or an outer attribute: it joins the next part.
    Leading,
}

/// Lay `source` out, or say why it cannot be split.
pub fn layout(source: &str) -> Result<Layout, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| format!("the Rust grammar does not load: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or("the file does not parse")?;
    let root = tree.root_node();
    if root.has_error() {
        return Err("the file has syntax errors; split only a file that builds".into());
    }
    let bytes = source.as_bytes();
    let mut kept = Vec::new();
    let mut items = Vec::new();
    // Where the comments and attributes waiting for their item begin.
    let mut leading: Option<usize> = None;
    // The last part placed: `true` for an item.
    let mut last_was_item = false;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor).filter(|n| n.is_named()) {
        let part = classify(&node, bytes);
        if let Part::Leading = part {
            leading.get_or_insert(node.start_byte());
            continue;
        }
        let start = leading.take().unwrap_or(node.start_byte());
        let span = start..node.end_byte();
        match part {
            Part::Kept(stay) => {
                kept.push(Kept { stay, span });
                last_was_item = false;
            }
            Part::Item => {
                let inline = (node.kind() == "mod_item")
                    .then(|| node.child_by_field_name("body"))
                    .flatten()
                    .map(|list| list.start_byte() + 1..list.end_byte() - 1);
                items.push(Item {
                    name: item_name(&node, bytes),
                    body: body_of(&node, bytes, start)?,
                    inline,
                    span,
                });
                last_was_item = true;
            }
            Part::Leading => unreachable!("handled above"),
        }
    }
    // Comments with nothing after them belong to the part before.
    if let Some(start) = leading {
        let end = trimmed_end(source, source.len());
        if last_was_item {
            let item = items.last_mut().expect("an item was placed");
            item.span.end = end;
            if let Some(body) = item.body.as_mut() {
                body.tail.end = end;
            }
        } else if let Some(k) = kept.last_mut() {
            k.span.end = end;
        } else {
            kept.push(Kept {
                stay: Stay::Header,
                span: start..end,
            });
        }
    }
    Ok(Layout { kept, items })
}

/// What a root child is.
fn classify(node: &Node, bytes: &[u8]) -> Part {
    match node.kind() {
        "line_comment" | "block_comment" => {
            let text = node.utf8_text(bytes).unwrap_or("");
            if text.starts_with("//!") || text.starts_with("/*!") {
                Part::Kept(Stay::Header)
            } else {
                Part::Leading
            }
        }
        "attribute_item" => Part::Leading,
        "inner_attribute_item" | "use_declaration" | "extern_crate_declaration" => {
            Part::Kept(Stay::Header)
        }
        "mod_item" if node.child_by_field_name("body").is_none() => Part::Kept(Stay::Header),
        "macro_definition" => Part::Kept(Stay::Macro),
        _ => Part::Item,
    }
}

/// An item's name as the map shows it.
fn item_name(node: &Node, bytes: &[u8]) -> String {
    let field = |f: &str| {
        node.child_by_field_name(f)
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(str::to_string)
    };
    let named = |word: &str| match field("name") {
        Some(n) => format!("{word} {n}"),
        None => word.to_string(),
    };
    match node.kind() {
        "function_item" | "function_signature_item" => named("fn"),
        "struct_item" => named("struct"),
        "enum_item" => named("enum"),
        "union_item" => named("union"),
        "trait_item" => named("trait"),
        "type_item" => named("type"),
        "const_item" => named("const"),
        "static_item" => named("static"),
        "mod_item" => named("mod"),
        "impl_item" => {
            let ty = field("type").unwrap_or_default();
            match field("trait") {
                Some(tr) => format!("impl {tr} for {ty}"),
                None => format!("impl {ty}"),
            }
        }
        _ => {
            let text = node.utf8_text(bytes).unwrap_or("");
            let first = text.lines().next().unwrap_or("").trim();
            first.chars().take(60).collect()
        }
    }
}

/// The members of an inherent `impl` or a `#[cfg(test)] mod { }`, or `None`
/// for any other item.
///
/// Only those two can be spread over several files. A trait impl is one
/// block by the language's rules, and an ordinary `mod { }` spread over files
/// becomes one module per file, whose glob re-exports then cancel out: a path
/// through it (`tables::X`) names nothing. A test module's pieces are wired to
/// each other by the shares instead.
fn body_of(node: &Node, bytes: &[u8], start: usize) -> Result<Option<Body>, String> {
    let divisible = match node.kind() {
        "impl_item" => node.child_by_field_name("trait").is_none(),
        "mod_item" => is_cfg_test(&bytes[start..node.start_byte()]),
        _ => false,
    };
    if !divisible {
        return Ok(None);
    }
    let Some(list) = node.child_by_field_name("body") else {
        return Ok(None);
    };
    let open = list.start_byte() + 1;
    let close = list.end_byte() - 1;
    let mut members = Vec::new();
    let mut leading: Option<usize> = None;
    let mut cursor = list.walk();
    for child in list.children(&mut cursor).filter(|n| n.is_named()) {
        if matches!(
            child.kind(),
            "line_comment" | "block_comment" | "attribute_item"
        ) {
            leading.get_or_insert(child.start_byte());
            continue;
        }
        let from = leading.take().unwrap_or(child.start_byte());
        members.push(Member {
            name: item_name(&child, bytes),
            span: line_start(bytes, from)..child.end_byte(),
            is_use: matches!(child.kind(), "use_declaration" | "inner_attribute_item"),
        });
    }
    if let Some(from) = leading {
        // Comments at the end of the body belong to its last member.
        match members.last_mut() {
            Some(m) => m.span.end = trimmed_end_bytes(bytes, close),
            None => {
                let _ = from;
                return Ok(None);
            }
        }
    }
    if members.is_empty() {
        return Ok(None);
    }
    Ok(Some(Body {
        head: start..open,
        members,
        tail: close..node.end_byte(),
    }))
}

/// Whether the attributes before an item gate it on `test`: `#[cfg(test)]`,
/// or a `cfg` whose predicate names `test` as a word outside its quoted
/// values, as `#[cfg(all(test, feature = "cuda"))]` does.
pub fn is_cfg_test(leading: &[u8]) -> bool {
    let text = String::from_utf8_lossy(leading);
    text.lines()
        .map(str::trim)
        .filter_map(|l| l.strip_prefix("#[cfg("))
        .any(|predicate| {
            let mut quoted = false;
            let bare: String = predicate
                .chars()
                .map(|c| {
                    if c == '"' {
                        quoted = !quoted;
                    }
                    if quoted || !(c.is_ascii_alphanumeric() || c == '_') {
                        ' '
                    } else {
                        c
                    }
                })
                .collect();
            bare.split_whitespace().any(|word| word == "test")
        })
}

/// `at`, moved back over the indentation before it to the start of its line
/// — so a member keeps its own indentation when it moves.
fn line_start(bytes: &[u8], at: usize) -> usize {
    let mut s = at;
    while s > 0 && (bytes[s - 1] == b' ' || bytes[s - 1] == b'\t') {
        s -= 1;
    }
    s
}

/// The end of `source[..end]` with trailing whitespace taken off.
fn trimmed_end(source: &str, end: usize) -> usize {
    source[..end].trim_end().len()
}

fn trimmed_end_bytes(bytes: &[u8], end: usize) -> usize {
    let mut e = end;
    while e > 0 && bytes[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text<'a>(src: &'a str, r: &Range<usize>) -> &'a str {
        &src[r.clone()]
    }

    /// **What stays and what moves.** Module docs, `use` lines and a
    /// file-less `mod` stay; a macro stays, ahead of the children; every
    /// other item moves, with the comments and attributes above it.
    #[test]
    fn header_macros_and_items_are_told_apart() {
        let src = "//! Docs.\n\nuse std::fmt;\nmod other;\n\nmacro_rules! m { () => {} }\n\n\
                   // ── a section ──\n\n/// A thing.\n#[derive(Debug)]\npub struct A;\n\nfn b() {}\n";
        let l = layout(src).unwrap();
        let kept: Vec<(Stay, &str)> = l
            .kept
            .iter()
            .map(|k| (k.stay, text(src, &k.span)))
            .collect();
        assert_eq!(
            kept,
            vec![
                (Stay::Header, "//! Docs.\n"),
                (Stay::Header, "use std::fmt;"),
                (Stay::Header, "mod other;"),
                (Stay::Macro, "macro_rules! m { () => {} }"),
            ]
        );
        let items: Vec<(&str, &str)> = l
            .items
            .iter()
            .map(|i| (i.name.as_str(), text(src, &i.span)))
            .collect();
        assert_eq!(
            items,
            vec![
                (
                    "struct A",
                    "// ── a section ──\n\n/// A thing.\n#[derive(Debug)]\npub struct A;"
                ),
                ("fn b", "fn b() {}"),
            ]
        );
    }

    /// **An `impl` opens into its methods**, each with its docs and its own
    /// indentation; its head runs through `{` and its tail is the `}`.
    #[test]
    fn an_impl_opens_into_its_members() {
        let src = "impl A {\n    /// One.\n    fn one(&self) {}\n\n    #[inline]\n    pub fn two(&self) {}\n}\n";
        let l = layout(src).unwrap();
        let body = l.items[0].body.as_ref().unwrap();
        assert_eq!(l.items[0].name, "impl A");
        assert_eq!(text(src, &body.head), "impl A {");
        assert_eq!(text(src, &body.tail), "}");
        let members: Vec<(&str, &str)> = body
            .members
            .iter()
            .map(|m| (m.name.as_str(), text(src, &m.span)))
            .collect();
        assert_eq!(
            members,
            vec![
                ("fn one", "    /// One.\n    fn one(&self) {}"),
                ("fn two", "    #[inline]\n    pub fn two(&self) {}"),
            ]
        );
    }

    /// A test module's `use` lines are marked, so every part of it carries
    /// them; a trait impl is named for its trait.
    #[test]
    fn a_mod_bodys_uses_are_marked() {
        let src = "#[cfg(test)]\nmod tests {\n    #![allow(dead_code)]\n    use super::*;\n\n    #[test]\n    fn t() {}\n}\n\n\
                   impl Display for A {\n    fn fmt(&self) {}\n}\n";
        let l = layout(src).unwrap();
        let body = l.items[0].body.as_ref().unwrap();
        assert_eq!(text(src, &body.head), "#[cfg(test)]\nmod tests {");
        assert_eq!(
            body.members.iter().map(|m| m.is_use).collect::<Vec<_>>(),
            vec![true, true, false]
        );
        assert_eq!(l.items[1].name, "impl Display for A");
    }

    /// **A trait impl and an ordinary `mod { }` stay whole**: neither opens
    /// into members, so no plan can spread one over several files.
    #[test]
    fn trait_impls_and_ordinary_mods_are_indivisible() {
        let src = "impl Display for A {\n    fn fmt(&self) {}\n}\n\n\
                   pub mod tables {\n    pub const X: u8 = 1;\n}\n\n\
                   #[cfg(test)]\nmod tests {\n    fn t() {}\n}\n\n\
                   #[cfg(all(test, feature = \"cuda\"))]\nmod gpu_tests {\n    fn t() {}\n}\n\n\
                   #[cfg(feature = \"test\")]\nmod featured {\n    fn t() {}\n}\n";
        let l = layout(src).unwrap();
        assert_eq!(
            l.items
                .iter()
                .map(|i| (i.name.as_str(), i.body.is_some()))
                .collect::<Vec<_>>(),
            vec![
                ("impl Display for A", false),
                ("mod tables", false),
                ("mod tests", true),
                ("mod gpu_tests", true),
                ("mod featured", false),
            ]
        );
    }

    /// A file that does not build is refused.
    #[test]
    fn a_broken_file_is_refused() {
        assert!(layout("fn a( {\n").is_err());
    }
}
