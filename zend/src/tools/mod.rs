//! Turns a model's tool calls into dispatched calls and their results, for the
//! daemon's conversations.
//!
//! The registered tools are placed in the system prompt as one section each, and
//! every turn's response is scanned for calls outside its reasoning. The calls
//! run through the registry in their context, and their results are wrapped and
//! submitted as the next turn's user message, so the loop continues until a turn
//! makes no call or the iteration cap is reached. Tool calls and their results
//! are filtered from the streamed output, so the client sees only the answer.
//!
//! - `call`: finds the calls in a response, outside its reasoning.
//! - `catalog`: the tool catalog in the prompt and the specs the grammar compiles.
//! - `dispatch`: runs a round of calls and formats their results.
//! - `tool_host`: the per-mode contexts and their conversation-bound file stores.
//! - `tests_calls`: tests for call extraction.
//! - `tests_dispatch`: tests for dispatch scope and resumed rounds.
//! - `tests_results`: tests for the mode catalogs and result text.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io;
use std::iter;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use candle_conversation::models::Dialect;
use candle_conversation::projection::{Builder as ProjectionBuilder, SectionId};
use candle_conversation::stencil::{function_blocks_to_json, ToolSpec};
use candle_conversation::think_strip::strip_think_blocks_keep_layout;
use candle_conversation::TurnText;
use host::calibration::Frame;
use serde::Deserialize;
use serde_json::Value;

use agent::jobs::Jobs;
use host::cluster::Join;
use tokio::runtime::Handle;
use zend_tools::executor::Executor;
use zend_tools::sandboxes::Sandboxes;
use zend_tools::state::Secrets;
use zend_tools::tools::journal::JournalShelf;
use zend_tools::{alias_pins, registry, replay, OpsRunner, Replay, ToolContext};
use zend_vfs::sandbox::programs;
use zend_vfs::{GitError, RepoFiles, Workspace, JOBS_DIR};

use crate::access;
use crate::agent_exec::AgentExecutor;
use crate::mcp::ownership::owner_of;
use crate::memory_paths;
use crate::substrate_git::agents as agents_for;
use crate::tool_availability::Facts;
use crate::tool_constraints::grammar_schema;
use crate::tool_guidance;
use crate::tool_round::{self, Step};
use crate::types::ToolMode;
use crate::workspace::UPLOADS_REPO;

// ── Public helper bundle ─────────────────────────────────────────────────────

/// Tool execution context for the daemon: one [`ToolContext`] per tools mode,
/// each carrying the session's operations runner once [`Self::running_ops`]
/// gives it one. Cloned cheaply (Arc-shared stores).
///
/// One host serves the whole daemon, but the `file_*` overlay does not: each
/// conversation has its own file stores ([`Self::conversation_files`]), and a
/// round runs in a context bound to them ([`Self::context_for`]), so what one
/// conversation writes, edits or deletes is never what another one reads. The
/// lower layer is each of the workspace's repositories, read-only.
///
/// It holds one context per tools mode. They share every other store and differ
/// only in their [`Grants`](zend_tools::Grants) ([`access::grants`]) — so what
/// a round may do is fixed by the context it is handed, not by which tools its
/// prompt offered. The workspace's command sandboxes are shared by all of
/// them; only a context granted `Exec` can reach them.
#[derive(Clone)]
pub struct ToolHost {
    /// Indexed by [`ToolMode::level`]. Each mode's own file stores are never
    /// handed to a round — a round gets its conversation's.
    contexts: [Arc<ToolContext>; ToolMode::ALL.len()],
    /// The branches of each repository a conversation owned through `/mcp`
    /// may never push to (`docs/claude_zend_gateway.md` §11).
    default_branches: Arc<BTreeMap<String, BTreeSet<String>>>,
}

/// The names a default branch goes by. Closed beside each repository's own
/// base branch, since the remote's default is not read here and a task's
/// branch never needs either name.
const CONVENTIONAL_DEFAULTS: [&str; 2] = ["main", "master"];

mod call;
mod catalog;
mod dispatch;
#[cfg(test)]
mod tests_calls;
#[cfg(test)]
mod tests_dispatch;
#[cfg(test)]
mod tests_results;
mod tool_host;

pub use call::*;
pub use catalog::*;
pub use dispatch::*;

// ── Tests ─────────────────────────────────────────────────────────────────────
