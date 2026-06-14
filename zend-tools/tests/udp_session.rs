mod harness;

use serde_json::json;
use zend_tools::ToolContext;

#[test]
fn udp_session_open_close() {
    let ctx = ToolContext::new();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_open",
        json!({}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();
    assert!(!sid.is_empty());

    let close = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_close",
        json!({"session_id": sid}),
        &ctx,
    ));
    assert_eq!(close["closed"], true);
}

#[test]
fn udp_session_list_after_open() {
    let ctx = ToolContext::new();
    harness::invoke_with_ctx("udp_session_open", json!({}), &ctx);
    let list = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);
}

#[test]
fn udp_session_list_empty_initially() {
    let ctx = ToolContext::new();
    let list = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn udp_session_close_nonexistent() {
    let resp = harness::expect_success(harness::invoke(
        "udp_session_close",
        json!({"session_id": "sess_not_real"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn udp_session_open_multiple_independent() {
    let ctx = ToolContext::new();
    let s1 = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_open",
        json!({}),
        &ctx,
    ));
    let s2 = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_open",
        json!({}),
        &ctx,
    ));
    assert_ne!(s1["session_id"], s2["session_id"]);
    let list = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 2);
}

#[test]
fn udp_session_session_id_starts_with_sess() {
    let ctx = ToolContext::new();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "udp_session_open",
        json!({}),
        &ctx,
    ));
    assert!(open["session_id"].as_str().unwrap().starts_with("sess_"));
}

#[test]
fn udp_send_not_found() {
    let resp = harness::invoke(
        "udp_session_send",
        json!({"session_id": "sess_ghost", "data": "aabb", "host": "127.0.0.1", "port": 9999}),
    );
    harness::expect_error(&resp, "session_not_found");
}
