//! Shared test-harness utilities for tool integration tests.
//!
//! Each per-tool test file invokes the dispatcher exactly the way the
//! orchestrator does at runtime — JSON in, JSON out.  Tests use [`invoke`] for
//! stateless calls (a fresh `ToolContext` is created per call) or
//! [`invoke_with_ctx`] for lifecycle tests that need state to persist across
//! multiple tool calls (open → use → close patterns).
//!
//! [`expect_success`] asserts the response has no `"error"` field.
//! [`expect_error`] asserts the response has `"error": "<code>"`.
//!
//! Tool *definitions* (name, description, `parameters` schema, examples) live in
//! the bundled `zend/src/prompts/tools/*.yaml` now; this harness exercises only
//! execution (dispatch, results, confirmation).

#![allow(dead_code)] // Helpers are pulled into per-tool test binaries selectively.

use serde_json::{json, Value};
use zend_tools::ToolContext;

/// Invoke a tool by name with default `ToolContext`.
pub fn invoke(tool: &str, args: Value) -> Value {
    let ctx = ToolContext::new();
    invoke_with_ctx(tool, args, &ctx)
}

/// Invoke a tool by name, threading a caller-supplied context.
/// Use this when a test needs to pre-seed VFS, credentials, sessions, etc.
pub fn invoke_with_ctx(tool: &str, args: Value, ctx: &ToolContext) -> Value {
    zend_tools::run(tool, "test_call_id", &args, ctx)
}

/// Confirmation details (if any) for a tool call. Mirrors the runtime call.
pub fn confirmation(tool: &str, args: Value) -> Option<zend_tools::ConfirmationDetails> {
    zend_tools::confirmation(tool, &args)
}

// ── Assertions ────────────────────────────────────────────────────────────────

/// Assert that the response is an error envelope with the given code.
/// Returns the `detail` string for further assertions.
#[track_caller]
pub fn expect_error(response: &Value, expected_code: &str) -> String {
    let code = response
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!("expected error response with code {expected_code:?}, got: {response}",)
        });
    assert_eq!(
        code, expected_code,
        "expected error code {expected_code:?}, got {code:?}: {response}",
    );
    response
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Assert that the response is *not* an error envelope. Returns it.
#[track_caller]
pub fn expect_success(response: Value) -> Value {
    if response.get("error").is_some() {
        panic!("expected success, got error: {response}");
    }
    response
}

// ── Smoke test for the harness itself ─────────────────────────────────────────

/// A canonical "unknown tool" response, useful as a sanity check.
pub fn unknown_tool_response(name: &str) -> Value {
    json!({
        "error": "unknown_tool",
        "detail": format!("no tool registered with name {name:?}"),
    })
}
