mod harness;

use serde_json::json;

#[test]
fn telnet_session_list_empty() {
    let resp = harness::expect_success(harness::invoke("telnet_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn telnet_session_close_nonexistent() {
    let resp = harness::expect_success(harness::invoke(
        "telnet_session_close",
        json!({"session_id": "x"}),
    ));
    assert_eq!(resp["closed"], false);
}

#[test]
fn telnet_session_list_returns_sessions_field() {
    let resp = harness::expect_success(harness::invoke("telnet_session_list", json!({})));
    assert!(resp["sessions"].is_array());
}

#[test]
fn telnet_session_close_returns_session_id() {
    let resp = harness::expect_success(harness::invoke(
        "telnet_session_close",
        json!({"session_id": "sess_telnet_xyz"}),
    ));
    assert_eq!(resp["session_id"], "sess_telnet_xyz");
    assert_eq!(resp["closed"], false);
}

#[test]
fn telnet_session_send_not_found() {
    let resp = harness::invoke(
        "telnet_session_send",
        json!({"session_id": "no_session", "send": "hello\r\n"}),
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn telnet_open_connection_refused() {
    // Port 1 is almost certainly not a telnet server
    let resp = harness::invoke(
        "telnet_session_open",
        json!({"host": "127.0.0.1", "port": 1, "timeout_sec": 1}),
    );
    assert!(resp.get("error").is_some());
}
