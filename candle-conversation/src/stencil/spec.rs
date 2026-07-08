//! The string-space, untokenized intermediate every front-end produces.
//!
//! `TreeSpec` is what the builder, the JSON tool compiler, and the YAML loader
//! all emit; [`compile`](super::compile::compile) is the single backend that
//! tokenizes, folds single-arm branches, fuses adjacent statics, and verifies
//! invariants.  Keeping the front-ends in string space guarantees they produce
//! identical trees for identical logical input.

use std::collections::HashMap;

use super::error::BuildError;
use super::terminator::Terminator;
use super::tree::FreeTextLimits;
use super::vocab::TokenId;

/// Index of a node within a [`TreeSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpecId(pub usize);

/// One string-space node.
#[derive(Debug, Clone)]
pub enum NodeSpec {
    /// A literal string and its single successor.
    Static { text: String, next: SpecId },
    /// A choice: `(arm literal, successor)` pairs.
    Branch { arms: Vec<(String, SpecId)> },
    /// A free-text span.
    FreeText {
        term: Terminator,
        eos_ends: bool,
        limits: FreeTextLimits,
        /// A close *token* that ends the span (already resolved to an id by the
        /// front-end; the compiler copies it verbatim, never tokenizing it in
        /// context).  `None` ⇒ the span closes only via `term`/EOS/limit.
        close_token: Option<TokenId>,
        /// When `true` and the span closed via `close_token`, drop that token and
        /// let the successor prefill the continuation.  See [`FreeTextSpan`].
        ///
        /// [`FreeTextSpan`]: super::tree::FreeTextSpan
        suppress_close: bool,
        next: SpecId,
    },
    /// Terminal.
    End,
}

/// A complete string-space tree.
#[derive(Debug, Clone)]
pub struct TreeSpec {
    pub nodes: Vec<NodeSpec>,
    pub root: SpecId,
    pub label: String,
    /// Text emitted to gracefully terminate the invocation if an out-of-grammar
    /// token is decoded (the failsafe bail set).  Empty ⇒ just exit.
    pub bail: String,
}

impl TreeSpec {
    pub fn new(label: impl Into<String>) -> Self {
        TreeSpec {
            nodes: Vec::new(),
            root: SpecId(0),
            label: label.into(),
            bail: String::new(),
        }
    }

    /// Push a node and return its id.
    pub fn push(&mut self, node: NodeSpec) -> SpecId {
        let id = SpecId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: SpecId) -> &NodeSpec {
        &self.nodes[id.0]
    }
}

// ── Label-based intermediate (shared by the builder and YAML front-ends) ─────

/// A node whose successors are referenced by string id, before resolution.
#[derive(Debug, Clone)]
pub enum LabeledNode {
    Static {
        text: String,
        next: String,
    },
    Branch {
        arms: Vec<(String, String)>,
    },
    FreeText {
        term: Terminator,
        eos_ends: bool,
        limits: FreeTextLimits,
        close_token: Option<TokenId>,
        suppress_close: bool,
        next: String,
    },
    End,
}

/// A tree authored with string node ids — what the builder and the YAML loader
/// produce.  [`Self::resolve`] turns it into a [`TreeSpec`] (still untokenized),
/// validating that every referenced id exists and is unique.
#[derive(Debug, Clone)]
pub struct LabeledTree {
    pub label: String,
    pub root: String,
    pub nodes: Vec<(String, LabeledNode)>,
}

impl LabeledTree {
    pub fn resolve(self) -> Result<TreeSpec, BuildError> {
        // Build id -> index, rejecting duplicates.
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, (id, _)) in self.nodes.iter().enumerate() {
            if idx.insert(id.as_str(), i).is_some() {
                return Err(BuildError::DuplicateNodeId(id.clone()));
            }
        }
        let resolve_ref = |id: &str| -> Result<SpecId, BuildError> {
            idx.get(id)
                .map(|&i| SpecId(i))
                .ok_or_else(|| BuildError::UnknownNodeId(id.to_string()))
        };

        let mut spec = TreeSpec::new(self.label.clone());
        for (_, node) in &self.nodes {
            let n = match node {
                LabeledNode::Static { text, next } => NodeSpec::Static {
                    text: text.clone(),
                    next: resolve_ref(next)?,
                },
                LabeledNode::Branch { arms } => {
                    let mut resolved = Vec::with_capacity(arms.len());
                    for (a, n) in arms {
                        resolved.push((a.clone(), resolve_ref(n)?));
                    }
                    NodeSpec::Branch { arms: resolved }
                }
                LabeledNode::FreeText {
                    term,
                    eos_ends,
                    limits,
                    close_token,
                    suppress_close,
                    next,
                } => NodeSpec::FreeText {
                    term: *term,
                    eos_ends: *eos_ends,
                    limits: *limits,
                    close_token: *close_token,
                    suppress_close: *suppress_close,
                    next: resolve_ref(next)?,
                },
                LabeledNode::End => NodeSpec::End,
            };
            spec.push(n);
        }
        spec.root = idx
            .get(self.root.as_str())
            .map(|&i| SpecId(i))
            .ok_or_else(|| BuildError::UnknownRoot(self.root.clone()))?;
        Ok(spec)
    }
}
