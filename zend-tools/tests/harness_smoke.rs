//! Smoke test verifying the harness wiring: dispatch path, error envelope
//! shape, and unknown-tool handling. Per-tool tests live in their own files.

mod harness;

use serde_json::json;

#[test]
fn unknown_tool_returns_invalid_arguments_envelope() {
    let response = harness::invoke("definitely_not_a_real_tool", json!({}));
    let detail = harness::expect_error(&response, "unknown_tool");
    assert!(
        detail.contains("definitely_not_a_real_tool"),
        "detail should mention the offending tool name, got {detail:?}",
    );
}

#[test]
fn unknown_tool_confirmation_returns_none() {
    assert!(harness::confirmation("definitely_not_a_real_tool", json!({})).is_none());
}
