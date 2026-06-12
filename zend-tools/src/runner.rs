//! Public dispatch entry points used by the `zend` orchestrator.
//!
//! The orchestrator calls these two functions and nothing else from this crate:
//!
//! - [`run`] — look up the tool, parse + validate args, execute, return the JSON
//!   response.  Always returns a value; never panics.
//! - [`confirmation`] — return the confirmation prompt for a tool call, or `None`
//!   if no confirmation is needed.  The orchestrator calls this *before* `run` to
//!   decide whether to pause for user approval.
//!
//! Both functions are synchronous.  If a tool needs async I/O (network, subprocess),
//! the orchestrator wraps the call in `tokio::task::spawn_blocking`.

use serde_json::{json, Value};

use crate::context::ToolContext;
use crate::registry;
use crate::tool::ConfirmationDetails;

/// Execute a tool by name and return the JSON the orchestrator should
/// place inside the `<tool_response>...</tool_response>` block.
///
/// The returned value is **always** valid JSON the LLM can act on:
/// - success → the tool's typed response
/// - tool error → `{"error": "<code>", "detail": "..."}`
/// - bad arguments → `{"error": "invalid_arguments", "detail": "..."}`
/// - unknown tool → `{"error": "unknown_tool", "detail": "..."}`
pub fn run(tool_name: &str, tool_call_id: &str, args: &Value, ctx: &ToolContext) -> Value {
    let span = tracing::info_span!("tool", name = tool_name, call_id = tool_call_id,);
    let _enter = span.enter();

    let Some(tool) = registry::find(tool_name) else {
        tracing::warn!("unknown tool {tool_name:?}");
        return json!({
            "error": "unknown_tool",
            "detail": format!("no tool registered with name {tool_name:?}"),
        });
    };

    tracing::debug!("dispatch start");
    let result = (tool.run)(ctx, args);
    tracing::debug!(success = !is_error_response(&result), "dispatch complete");
    result
}

/// Return confirmation details for a tool call, or `None` if no confirmation
/// is required (or the args are malformed — in which case `run` will surface
/// the error to the LLM).
pub fn confirmation(tool_name: &str, args: &Value) -> Option<ConfirmationDetails> {
    let tool = registry::find(tool_name)?;
    (tool.confirmation)(args)
}

/// Heuristic: is this response value an error envelope?
/// Used for tracing only; the orchestrator emits the value verbatim.
fn is_error_response(v: &Value) -> bool {
    v.get("error").is_some()
}
