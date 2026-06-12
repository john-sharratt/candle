mod harness;

use serde_json::json;
use zend_tools::ToolContext;

#[test]
fn tls_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("tls_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn tls_session_close_nonexistent() {
    let resp = harness::expect_success(harness::invoke(
        "tls_session_close",
        json!({"session_id": "sess_nonexistent"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn tls_session_send_not_found() {
    let resp = harness::invoke(
        "tls_session_send",
        json!({"session_id": "sess_missing", "data": "hello"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn tls_session_recv_not_found() {
    let resp = harness::invoke("tls_session_recv", json!({"session_id": "sess_missing"}));
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn tls_session_open_connection_refused() {
    // Port 1 on localhost is almost certainly not a TLS server
    let resp = harness::invoke(
        "tls_session_open",
        json!({"host": "127.0.0.1", "port": 1, "accept_invalid_certs": true, "timeout_ms": 200}),
    );
    assert!(resp.get("error").is_some());
}

#[test]
fn tls_session_list_returns_sessions_field() {
    let resp = harness::expect_success(harness::invoke("tls_session_list", json!({})));
    assert!(resp["sessions"].is_array());
}

#[test]
fn tls_session_empty_in_fresh_context() {
    let ctx = ToolContext::new();
    let list = harness::expect_success(harness::invoke_with_ctx(
        "tls_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 0);
}
