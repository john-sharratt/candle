mod harness;

use serde_json::json;
use zend_tools::ToolContext;

#[test]
fn http_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("http_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn http_session_request_no_session() {
    let resp = harness::invoke(
        "http_request",
        json!({"session_id": "sess_bogus", "path": "https://example.com"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn http_session_open_returns_session_id() {
    let ctx = ToolContext::new();
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "http_session_open",
        json!({"base_url": "https://example.com"}),
        &ctx,
    ));
    let sid = resp["session_id"].as_str().unwrap();
    assert!(sid.starts_with("sess_"));
}

#[test]
fn http_session_open_list_close() {
    let ctx = ToolContext::new();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "http_session_open",
        json!({"base_url": "https://api.example.com"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    let list = harness::expect_success(harness::invoke_with_ctx(
        "http_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);

    let closed = harness::expect_success(harness::invoke_with_ctx(
        "http_session_close",
        json!({"session_id": sid}),
        &ctx,
    ));
    assert_eq!(closed["closed"], true);

    let list2 = harness::expect_success(harness::invoke_with_ctx(
        "http_session_list",
        json!({}),
        &ctx,
    ));
    assert_eq!(list2["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn http_session_close_nonexistent() {
    let resp = harness::expect_success(harness::invoke(
        "http_session_close",
        json!({"session_id": "sess_nonexistent"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn http_session_open_credential_not_found() {
    let resp = harness::invoke(
        "http_session_open",
        json!({"credential_name": "no_such_cred"}),
    );
    harness::expect_error(&resp, "credential_not_found");
}

#[test]
fn http_session_open_no_base_url() {
    let ctx = ToolContext::new();
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "http_session_open",
        json!({}),
        &ctx,
    ));
    assert!(resp["base_url"].is_null());
}
