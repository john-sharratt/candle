mod harness;

use serde_json::json;

#[test]
fn ssh_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("ssh_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn ssh_session_close_missing() {
    let resp = harness::expect_success(harness::invoke(
        "ssh_session_close",
        json!({"session_id": "nonexistent"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn ssh_session_open_credential_not_found() {
    let resp = harness::invoke(
        "ssh_session_open",
        json!({"credential_name": "bogus_cred", "host": "localhost"}),
    );
    harness::expect_error(&resp, "credential_not_found");
}

#[test]
fn ssh_session_close_returns_session_id() {
    let resp = harness::expect_success(harness::invoke(
        "ssh_session_close",
        json!({"session_id": "sess_abc"}),
    ));
    assert_eq!(resp["session_id"], "sess_abc");
    assert_eq!(resp["closed"], false);
}

#[test]
fn ssh_session_poll_process_not_found() {
    let resp = harness::invoke(
        "ssh_session_poll",
        json!({"process_id": "proc_nonexistent"}),
    );
    harness::expect_error(&resp, "process_not_found");
}

#[test]
fn ssh_session_exec_session_not_found() {
    let resp = harness::invoke(
        "ssh_session_exec",
        json!({"session_id": "sess_missing", "command": "ls"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn ssh_session_list_returns_sessions_field() {
    let resp = harness::expect_success(harness::invoke("ssh_session_list", json!({})));
    assert!(resp["sessions"].is_array());
}
