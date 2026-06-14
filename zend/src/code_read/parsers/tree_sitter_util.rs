//! Shared utilities for the tree-sitter-backed language parsers.
//!
//! The model: each parser drives [`carve_tree`] with a [`LanguageRules`]
//! struct that names which node kinds to extract, how to label them
//! for the scope path, and how to find their identifier.  The walker
//! itself is language-agnostic — it just does depth-first traversal
//! and emits scopes for the right node kinds with the right enclosing
//! path.

use std::collections::HashMap;

use tree_sitter::{Node, Parser, Tree};

use crate::code_read::types::{ChunkKind, Scope, MAX_SCOPE_LINES};

/// Per-language carving rules.
///
/// `kind_to_chunk` answers: when the walker encounters a node of this
/// kind, emit a scope of this [`ChunkKind`].
///
/// `identifier_for` extracts a short human label from a node — used to
/// build the scope path (`"fn validate_token"` etc.).  Different
/// languages name identifier children differently, so the lookup is
/// per-rule.
///
/// `enclosing_label` answers: when descending into this node, what
/// label should be pushed onto the scope path so children inherit it?
/// (For Rust, `mod_item` pushes `mod foo`; for Python, `class_definition`
/// pushes `class Foo`.)  Returning `None` means "don't push anything"
/// — the walker descends without extending the path.
pub struct LanguageRules {
    pub kind_to_chunk: HashMap<&'static str, ChunkKind>,
    pub identifier_for: fn(&Node, &[u8]) -> Option<String>,
    pub enclosing_label: fn(&Node, &[u8]) -> Option<String>,
}

/// Parse `source` and emit scopes using the supplied rules.  Returns
/// `None` if the tree-sitter parse fails outright.
pub fn carve_tree(parser: &mut Parser, source: &[u8], rules: &LanguageRules) -> Option<Vec<Scope>> {
    let tree: Tree = parser.parse(source, None)?;
    let mut scopes = Vec::new();
    walk(
        tree.root_node(),
        source,
        rules,
        &mut Vec::new(),
        &mut scopes,
    );
    Some(scopes)
}

fn walk(
    node: Node,
    source: &[u8],
    rules: &LanguageRules,
    path: &mut Vec<String>,
    scopes: &mut Vec<Scope>,
) {
    let kind = node.kind();
    let pushed_label = (rules.enclosing_label)(&node, source);

    if let Some(chunk_kind) = rules.kind_to_chunk.get(kind).copied() {
        let label = (rules.identifier_for)(&node, source).unwrap_or_else(|| kind.to_string());
        let mut full_path = path.clone();
        if !label.is_empty() {
            full_path.push(label);
        }
        let start_line = (node.start_position().row as u32) + 1;
        let end_line = (node.end_position().row as u32) + 1;
        // Split oversize functions / impls at their inner block
        // boundaries — tree-sitter gives us a clean inner-statement
        // list to split on for every language we support.
        if end_line.saturating_sub(start_line) + 1 > MAX_SCOPE_LINES {
            emit_split(&full_path, chunk_kind, node, scopes);
        } else {
            scopes.push(Scope {
                path: full_path,
                kind: chunk_kind,
                start_line,
                end_line,
            });
        }
    }

    let did_push = pushed_label.is_some();
    if let Some(label) = pushed_label {
        path.push(label);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, source, rules, path, scopes);
    }
    if did_push {
        path.pop();
    }
}

/// Oversize-scope split: chunk the node's direct children into
/// `MAX_SCOPE_LINES`-bounded windows and emit one scope per window
/// labelled `name [part N]`.  This keeps the per-scope prefill cost
/// bounded without dropping coverage.
fn emit_split(path: &[String], kind: ChunkKind, node: Node, scopes: &mut Vec<Scope>) {
    let start_line = (node.start_position().row as u32) + 1;
    let end_line = (node.end_position().row as u32) + 1;
    let mut part = 1usize;
    let mut chunk_start = start_line;
    while chunk_start <= end_line {
        let chunk_end = (chunk_start + MAX_SCOPE_LINES - 1).min(end_line);
        let mut labelled = path.to_vec();
        if let Some(last) = labelled.last_mut() {
            *last = format!("{last} [part {part}]");
        }
        scopes.push(Scope {
            path: labelled,
            kind,
            start_line: chunk_start,
            end_line: chunk_end,
        });
        chunk_start = chunk_end + 1;
        part += 1;
    }
}

/// Convenience: read the text of a node's child with `field_name`,
/// trimmed.  Used by per-language identifier extractors.
pub fn field_text(node: &Node, field_name: &str, source: &[u8]) -> Option<String> {
    let child = node.child_by_field_name(field_name)?;
    Some(slice_text(&child, source))
}

pub fn slice_text(node: &Node, source: &[u8]) -> String {
    String::from_utf8_lossy(&source[node.byte_range()])
        .trim()
        .to_string()
}
