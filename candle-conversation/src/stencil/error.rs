//! Construction and walk errors.

use thiserror::Error;

/// Raised while compiling a [`TreeSpec`](crate::stencil::TreeSpec) into a
/// [`StencilTree`](crate::stencil::StencilTree).  Every failure is loud — the
/// compiler never produces a silently-wrong tree.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BuildError {
    #[error("spec root {0} is out of range")]
    BadRoot(usize),
    #[error("node {from} references successor {to} which is out of range")]
    BadRef { from: usize, to: usize },
    #[error("the spec has a cycle through node {0} (stencil trees must be acyclic)")]
    Cycle(usize),
    #[error("a path does not reach an End node (node {0})")]
    NoEnd(usize),
    #[error("branch at node {0} has no arms")]
    EmptyBranch(usize),
    #[error("free-text span at node {0} has forced_after == 0 (a hard limit is required)")]
    NoHardLimit(usize),
    #[error("branch arm {arm:?} tokenizes empty (no tokens)")]
    EmptyArm { arm: String },
    #[error("branch arm {long:?} has arm {short:?} as a token-level prefix — ambiguous")]
    AmbiguousArms { short: String, long: String },
    #[error(
        "tokenization boundary merge at node {node}: segment {segment:?} merges with its \
         left context (pulled back {pullback} prefix token(s)); the grammar's boundaries \
         must be merge-stable"
    )]
    BoundaryMerge {
        node: usize,
        segment: String,
        pullback: usize,
    },
    #[error("yaml parse error: {0}")]
    Yaml(String),
    // The id errors below are shared by both the builder and YAML front-ends, so
    // their messages carry no front-end prefix.
    #[error("node id {0:?} referenced but not defined")]
    UnknownNodeId(String),
    #[error("duplicate node id {0:?}")]
    DuplicateNodeId(String),
    #[error("no node has id {0:?} (root)")]
    UnknownRoot(String),
    #[error("tool schema error: {0}")]
    ToolSchema(String),
}

/// Raised while walking a tree at decode time.  An out-of-grammar token is NOT
/// an error — it triggers the bail failsafe (see the session) — so the only
/// walk error is calling `observe` with no decode pending.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WalkError {
    #[error("observe called with no active decode pending")]
    NotDecoding,
}
