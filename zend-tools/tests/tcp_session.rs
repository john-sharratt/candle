mod harness;

use serde_json::json;
use zend_tools::ToolContext;

#[test]
fn tcp_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("tcp_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn tcp_session_open_connection_refused() {
    let resp = harness::invoke(
        "tcp_session_open",
        json!({"host": "127.0.0.1", "port": 1, "timeout_ms": 200}),
    );
    assert!(resp.get("error").is_some());
}

#[test]
fn tcp_session_close_nonexistent() {
    let resp = harness::expect_success(harness::invoke(
        "tcp_session_close",
        json!({"session_id": "nonexistent"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn tcp_session_send_not_found() {
    let resp = harness::invoke(
        "tcp_session_send",
        json!({"session_id": "sess_missing", "data": "hello"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn tcp_session_recv_not_found() {
    let resp = harness::invoke(
        "tcp_session_recv",
        json!({"session_id": "sess_missing", "recv_amt": 64}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn tcp_session_list_returns_sessions_field() {
    let resp = harness::expect_success(harness::invoke("tcp_session_list", json!({})));
    assert!(resp["sessions"].is_array());
}

#[test]
fn tcp_session_close_returns_session_id() {
    let resp = harness::expect_success(harness::invoke(
        "tcp_session_close",
        json!({"session_id": "sess_tcp_abc"}),
    ));
    assert_eq!(resp["session_id"], "sess_tcp_abc");
}

#[test]
fn tcp_session_list_empty_after_context_create() {
    let ctx = ToolContext::new();
    let list = harness::expect_success(harness::invoke_with_ctx("tcp_session_list", json!({}), &ctx));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn tcp_session_recv_missing_recv_mode() {
    // Neither recv_amt nor recv_wait → missing_recv_mode
    let resp = harness::invoke(
        "tcp_session_recv",
        json!({"session_id": "sess_missing"}),
    );
    harness::expect_error(&resp, "missing_recv_mode");
}

#[test]
fn tcp_session_recv_conflicting_recv_modes() {
    // Both recv_amt AND recv_wait → conflicting_recv_modes
    let resp = harness::invoke(
        "tcp_session_recv",
        json!({"session_id": "sess_missing", "recv_amt": 64, "recv_wait": 1.0}),
    );
    harness::expect_error(&resp, "conflicting_recv_modes");
}
