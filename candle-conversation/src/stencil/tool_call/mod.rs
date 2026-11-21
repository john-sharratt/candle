//! Compiles a tool catalog into the stencil tree that holds a model's tool calls
//! to the catalog's names and schemas.
//!
//! The tree guarantees that a name is one of the catalog's, that required
//! parameters are present in order, that enum values are exact, and that the
//! structure closes. A value is free only where the grammar cannot begin it.
//! Strings are bounded spans, integers are written digit by digit, and a turn's
//! calls are capped by a bound that every level can stop at. The call's syntax
//! comes from the checkpoint's dialect, so the shape the grammar holds a model to
//! is the one its weights were trained on.
//!
//! - `builder`: the argument grammar for one tool's parameters.
//! - `call_tree`: the single-call tree and the multi-call action loop.
//! - `envelope`: the dialect's strings that wrap a call.
//! - `param`: a tool's parameters and the schema facts that bound them.
//! - `tests_envelope`: the envelope and the grammar's shapes.
//! - `tests_schema`: schema reading into parameter trees.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use candle_transformers::models::dialect::{CallStyle, Dialect};

use super::element_close::{Follows, FUNCTION_TAG, PARAM_CLOSE, PARAM_TAG};
use super::error::BuildError;
use super::field_order::Owed;
use super::spec::{ChosenArms, NodeSpec, SpecId, TreeSpec};
use super::terminator::Terminator;
use super::tree::{FreeTextLimits, MAX_STRING_VALUE_TOKENS};

/// The value one arm of a branch writes, when it writes one.
type ArmValue = Option<Arc<str>>;

/// A branch arm's text and successor, with the value it writes.
type ValuedArm = (String, SpecId, ArmValue);

/// A parameter's value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
}

/// What a field owes a choice among an object's optional fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pick {
    #[default]
    None,
    /// `anyOf` of `required`: this or another member.
    AtLeastOne,
    /// `oneOf` of `required`: this or another member, and no second.
    ExactlyOne,
}

/// How far a `$ref` chain is followed before the schema is treated as opaque.
/// A schema that refers to itself — a tree node whose children are the same
/// node — is legal JSON Schema and has no finite parameter tree, so the
/// recursion has to stop somewhere rather than hang the builder.
const MAX_REF_DEPTH: usize = 8;

/// The tokens a string value's span spends on its delimiters rather than its
/// value: the opening quote the model writes, and its own close.
const VALUE_DELIMITER_TOKENS: u32 = 2;

/// The tool-call stencil tree's label. The scheduler keys tool-call-specific
/// behavior off the active stencil's tree label (repetition-penalty
/// suppression, the in-call reprojection freeze at first-token promotion), so
/// the label is a shared constant rather than a string literal in each place.
pub const TOOL_CALL_TREE_LABEL: &str = "tool_call";

/// How many tool calls one assistant turn may make.
///
/// **A turn that can only call once pays a full round-trip per call.** Reading
/// four files it already knows it wants costs four reasoning blocks, four
/// prefills of a growing context, four reprojections and four belief scans — and
/// measured on a codebase tour, fifteen such rounds were the bulk of the wall
/// clock while the calls themselves ran in milliseconds. Batching what the model
/// already knows it needs collapses those round-trips into one.
///
/// Twelve covers the common fan-out — list a directory, then read the files it
/// names — and a batch of known edits: at six, a job of 27 writes and 28 deletes
/// took a reply for every six, each a prefill of the whole conversation
/// (2026-10-31). It still stops short of one turn committing to a long
/// speculative run whose later calls are all chosen before any result. The
/// ceiling is a grammar bound, not a target: a turn making one call remains
/// perfectly ordinary, because the loop's other arm is always the turn
/// terminator.
pub const MAX_TOOL_CALLS_PER_TURN: usize = 12;

/// The tree label for a standalone invoke-body sub-stencil (effector design §11).
pub const INVOKE_BODY_TREE_LABEL: &str = "invoke_body";

/// How many elements a guided array admits. The grammar is unrolled one level
/// per element — the compiler rejects cycles, and a cycle would hide the count
/// the bound needs — so the last level offers only the close.
pub const MAX_ARRAY_ELEMENTS: usize = 64;

/// How many digits an integer value may have — enough for any `i64`
/// (`9223372036854775807` is nineteen). Unrolled one level per digit, as an
/// array is per element, for the same reason: the compiler rejects cycles.
pub const MAX_INTEGER_DIGITS: usize = 19;

mod builder;
mod call_tree;
mod envelope;
mod param;
#[cfg(test)]
mod tests_envelope;
#[cfg(test)]
mod tests_schema;

use builder::*;
pub use call_tree::*;
pub use envelope::*;
pub use param::*;
