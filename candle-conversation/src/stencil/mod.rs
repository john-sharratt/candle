//! Stencil tree — schema-guided constrained decoding.
//!
//! A *stencil tree* is a compiled grammar that alternates fixed **static** runs
//! (prefilled atomically — guaranteed-correct speculative decoding), **branch**
//! choice points (the sampler masked to a token frontier), and **free-text**
//! spans (the model decodes a value, watched by an escape- and nesting-aware
//! terminator) until a terminal.  Its primary use is forcing a tool call to a
//! catalog's exact names and JSON schema.
//!
//! This module is **standalone**: it builds, walks, and validates trees and
//! masks logits, but does not touch the scheduler or sampler.  The design is in
//! `docs/stencil_tree.md`.
//!
//! # Construction
//!
//! Three front-ends all produce a string-space [`TreeSpec`] that the single
//! [`compile`] backend tokenizes, folds, fuses, and verifies:
//!
//! - [`StencilTreeBuilder`] — programmatic.
//! - [`compile_tool_call_tree`] — from a JSON tool catalog.
//! - [`TreeSpec::from_yaml`] — a declarative YAML node spec.
//!
//! # Walking
//!
//! [`StencilSession`] walks a compiled [`StencilTree`]: [`next_action`] says what
//! to do (prefill / masked decode / free decode / exit) and [`observe`] consumes
//! a decoded token.  [`simulate`] drives a session with no model for testing.
//!
//! [`next_action`]: StencilSession::next_action
//! [`observe`]: StencilSession::observe

mod builder;
mod compile;
mod driver;
mod error;
mod mask;
mod session;
mod sim;
mod spec;
mod terminator;
mod think;
mod tool_call;
mod tree;
mod trie;
mod trigger;
mod vocab;
mod yaml;

#[cfg(test)]
mod tests;

pub use builder::StencilTreeBuilder;
pub use compile::compile;
pub use driver::{Healed, PathStats, StencilDriver, StepMask};
pub use error::{BuildError, WalkError};
pub use mask::{ban, boost, AllowedSet};
pub use session::{Observe, StencilAction, StencilSession};
pub use sim::{lowest_arm_policy, simulate, Oracle, SimError, SimRun};
pub use spec::{LabeledNode, LabeledTree, NodeSpec, SpecId, TreeSpec};
pub use terminator::{Feed, Terminator, TerminatorState};
pub use think::{compile_think_tree, ThinkMode, ThinkSteerEnvelope};
pub use tool_call::{
    compile_tool_call_tree, parse_tools, Param, ParamType, ToolCallEnvelope, ToolSpec,
    TOOL_CALL_TREE_LABEL,
};
pub use tree::{FreeTextLimits, FreeTextSpan, NodeId, StencilNode, StencilTree};
pub use trie::{Step, TokenTrie, TrieNodeId};
pub use trigger::TriggerRegistry;
pub use vocab::{HfVocab, TestVocab, TokenId, Vocab};
