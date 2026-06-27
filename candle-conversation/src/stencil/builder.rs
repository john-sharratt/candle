//! Front-end A — a programmatic builder.
//!
//! Nodes are declared with string ids and reference their successors by id, so
//! forward references are natural.  `build` resolves to a [`TreeSpec`];
//! `compile` then produces the tree.

use super::error::BuildError;
use super::spec::{LabeledNode, LabeledTree, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;
use super::vocab::TokenId;

/// A fluent builder for a stencil grammar.
pub struct StencilTreeBuilder {
    label: String,
    root: Option<String>,
    nodes: Vec<(String, LabeledNode)>,
}

impl StencilTreeBuilder {
    pub fn new(label: impl Into<String>) -> Self {
        StencilTreeBuilder {
            label: label.into(),
            root: None,
            nodes: Vec::new(),
        }
    }

    /// Set the root node id.
    pub fn root(mut self, id: impl Into<String>) -> Self {
        self.root = Some(id.into());
        self
    }

    /// A static run `id`: emit `text`, then go to `next`.
    pub fn static_node(
        mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        next: impl Into<String>,
    ) -> Self {
        self.nodes.push((
            id.into(),
            LabeledNode::Static {
                text: text.into(),
                next: next.into(),
            },
        ));
        self
    }

    /// A branch `id` with `(arm literal, next id)` pairs.
    pub fn branch(mut self, id: impl Into<String>, arms: &[(&str, &str)]) -> Self {
        self.nodes.push((
            id.into(),
            LabeledNode::Branch {
                arms: arms
                    .iter()
                    .map(|(a, n)| (a.to_string(), n.to_string()))
                    .collect(),
            },
        ));
        self
    }

    /// A free-text span `id` (no close token).
    pub fn free_text(
        self,
        id: impl Into<String>,
        term: Terminator,
        eos_ends: bool,
        limits: FreeTextLimits,
        next: impl Into<String>,
    ) -> Self {
        self.free_text_token_closed(id, term, eos_ends, limits, None, false, next)
    }

    /// A free-text span `id` that may also close on a sampled `close_token`.
    /// When `suppress_close` is set, the close token is dropped and the
    /// successor prefills in its place (the steering continuation).
    #[allow(clippy::too_many_arguments)]
    pub fn free_text_token_closed(
        mut self,
        id: impl Into<String>,
        term: Terminator,
        eos_ends: bool,
        limits: FreeTextLimits,
        close_token: Option<TokenId>,
        suppress_close: bool,
        next: impl Into<String>,
    ) -> Self {
        self.nodes.push((
            id.into(),
            LabeledNode::FreeText {
                term,
                eos_ends,
                limits,
                close_token,
                suppress_close,
                next: next.into(),
            },
        ));
        self
    }

    /// A JSON-string free-text span `id` with default limits.
    pub fn free_string(self, id: impl Into<String>, next: impl Into<String>) -> Self {
        self.free_text(
            id,
            Terminator::JsonString,
            false,
            FreeTextLimits::json_string(),
            next,
        )
    }

    /// A terminal node `id`.
    pub fn end(mut self, id: impl Into<String>) -> Self {
        self.nodes.push((id.into(), LabeledNode::End));
        self
    }

    /// Resolve to a [`TreeSpec`] (still untokenized).
    pub fn build(self) -> Result<TreeSpec, BuildError> {
        let root = self
            .root
            .ok_or_else(|| BuildError::UnknownRoot(String::from("<unset>")))?;
        LabeledTree {
            label: self.label,
            root,
            nodes: self.nodes,
        }
        .resolve()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::session::{StencilAction, StencilSession};
    use crate::stencil::tree::StencilNode;
    use crate::stencil::vocab::TestVocab;

    #[test]
    fn builds_a_branching_tree() {
        let spec = StencilTreeBuilder::new("t")
            .root("open")
            .static_node("open", "\"", "name")
            .branch("name", &[("ab", "close"), ("cd", "close")])
            .static_node("close", "\"", "done")
            .end("done")
            .build()
            .unwrap();
        let tree = compile(&spec, &TestVocab::new()).unwrap();
        // root Static "\"" -> Branch -> (per-arm) Static "\"" -> End.
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => assert_eq!(tokens, &[b'"' as u32]),
            _ => panic!(),
        }
    }

    #[test]
    fn unset_root_errors() {
        let e = StencilTreeBuilder::new("t").end("x").build().unwrap_err();
        assert!(matches!(e, BuildError::UnknownRoot(_)));
    }

    #[test]
    fn unknown_ref_errors() {
        let e = StencilTreeBuilder::new("t")
            .root("a")
            .static_node("a", "x", "missing")
            .build()
            .unwrap_err();
        assert!(matches!(e, BuildError::UnknownNodeId(_)));
    }

    #[test]
    fn duplicate_id_errors() {
        let e = StencilTreeBuilder::new("t")
            .root("a")
            .end("a")
            .end("a")
            .build()
            .unwrap_err();
        assert!(matches!(e, BuildError::DuplicateNodeId(_)));
    }

    #[test]
    fn free_string_helper_round_trips() {
        let spec = StencilTreeBuilder::new("t")
            .root("v")
            .free_string("v", "done")
            .end("done")
            .build()
            .unwrap();
        let tree = compile(&spec, &TestVocab::new()).unwrap();
        let mut sess = StencilSession::new(Arc::new(tree));
        assert!(matches!(
            sess.next_action(),
            StencilAction::FreeDecode { .. }
        ));
    }
}
