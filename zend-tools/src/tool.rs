//! Core trait and supporting types that every tool implements.
//!
//! # The three-type pattern
//!
//! Every tool defines exactly three public types:
//!
//! - **`Request`** — derives `Deserialize + JsonSchema + Validate`.  Schemars
//!   auto-generates the JSON Schema the LLM sees; the validator crate enforces
//!   field constraints before `run` is ever called.
//! - **`Response`** — derives `Serialize`.  Returned verbatim as the
//!   `<tool_response>` payload on success.
//! - **`Error`** — implements [`ToolError`].  Returned as
//!   `{"error": "<code>", "detail": "..."}` on failure.  Error codes are stable
//!   across releases; the LLM may key off them for retry logic.
//!
//! # Confirmation flow
//!
//! Tools with remote side-effects implement [`Tool::confirmation`] to return a
//! [`ConfirmationDetails`] before `run` is called.  The orchestrator renders a
//! user-facing prompt, waits for approval, and only proceeds on Allow.  Deny or
//! 60-second timeout returns `{"error": "denied_by_user"}` to the LLM without
//! calling `run`.

use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use validator::Validate;

use crate::context::ToolContext;

/// Per-tool error trait. Each tool defines its own error enum and implements
/// this trait to surface a stable `code` and `detail` to the LLM.
///
/// Error responses are emitted to the LLM as:
/// `{"error": "<code>", "detail": "<detail>"}`
pub trait ToolError: std::error::Error + Send + Sync + 'static {
    /// Stable, machine-readable error code, e.g. `"url_blocked"`.
    /// Must not change across releases — the LLM may key off it.
    fn code(&self) -> &'static str;

    /// Human-readable detail string. Defaults to `Display`.
    fn detail(&self) -> String {
        self.to_string()
    }
}

/// Authored content returned to the orchestrator when a tool requires
/// user confirmation. The orchestrator renders this as an SSE
/// `confirmation_required` frame; the LLM never sees it.
#[derive(Debug, Clone)]
pub struct ConfirmationDetails {
    /// One-line summary of what the tool is about to do.
    pub summary: String,
    /// Structured fields rendered as a key/value list in the prompt
    /// (e.g. `("host", "bastion.prod.example.com")`).
    pub fields: Vec<(&'static str, String)>,
}

impl ConfirmationDetails {
    pub fn new(summary: impl Into<String>) -> Self {
        Self { summary: summary.into(), fields: Vec::new() }
    }

    pub fn with_field(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.fields.push((name, value.into()));
        self
    }
}

/// Input to a subagent invocation.
pub struct SubagentRequest {
    pub instruction: String,
    pub tools: Option<Vec<String>>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub max_turns: u32,
}

/// Output from a subagent invocation.
pub struct SubagentResponse {
    pub result: String,
    pub turns: u32,
    pub tool_calls_made: u32,
}

/// Injected by the caller (e.g. `zend` daemon) to run nested agent loops.
/// `zend-tools` defines the interface; the daemon implements it.
pub trait SubagentRunner: Send + Sync + 'static {
    fn run(&self, req: SubagentRequest) -> Result<SubagentResponse, String>;
}

/// A tool implementation.
///
/// Tools are zero-state types — all state lives in [`ToolContext`].
/// `run` is synchronous; if a tool needs async I/O the orchestrator wraps
/// the dispatch call in `tokio::task::spawn_blocking`.
pub trait Tool: 'static {
    /// Tool name used in `<tool_call>{"name": "..."}` blocks.
    const NAME: &'static str;

    /// Trigger-rich description (the "description tier") shown to the LLM.
    /// See `docs/tool-system.md` for authoring guidance.
    const DESCRIPTION: &'static str;

    /// Request parameters. Must derive
    /// `Deserialize`, `JsonSchema`, and `Validate`.
    type Request: DeserializeOwned + JsonSchema + Validate;

    /// Successful response payload. Must derive `Serialize`.
    type Response: Serialize;

    /// Tool-specific error type.
    type Error: ToolError;

    /// Execute the tool with the validated request and the shared context.
    fn run(ctx: &ToolContext, req: Self::Request) -> Result<Self::Response, Self::Error>;

    /// Optional confirmation prompt details. `None` means no confirmation.
    /// Default impl: never confirm.
    fn confirmation(_req: &Self::Request) -> Option<ConfirmationDetails> {
        None
    }
}
