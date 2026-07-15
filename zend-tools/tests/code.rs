mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

// ── code_run (one-shot) ─────────────────────────────────────────────────────────

#[test]
fn code_run_console_log() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "javascript", "code": "console.log('hello from js')"}),
    ));
    assert!(r["stdout"].as_str().unwrap().contains("hello from js"));
    assert_eq!(r["exit_code"], 0);
}

#[test]
fn code_run_returns_final_value() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "js", "code": "2 + 40"}),
    ));
    assert_eq!(r["result"], "42");
    assert_eq!(r["exit_code"], 0);
}

#[test]
fn code_run_logs_objects_as_json() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "javascript", "code": "console.log({a: 1, b: [2, 3]})"}),
    ));
    assert!(
        r["stdout"]
            .as_str()
            .unwrap()
            .contains("{\"a\":1,\"b\":[2,3]}"),
        "stdout: {}",
        r["stdout"]
    );
}

#[test]
fn code_run_throw_sets_exit_code_and_stderr() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "javascript", "code": "throw new Error('boom')"}),
    ));
    assert_eq!(r["exit_code"], 1);
    assert!(
        r["stderr"].as_str().unwrap().contains("boom"),
        "stderr: {}",
        r["stderr"]
    );
}

#[test]
fn code_run_console_error_to_stderr() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "javascript", "code": "console.error('bad thing'); 1"}),
    ));
    assert!(r["stderr"].as_str().unwrap().contains("bad thing"));
    assert_eq!(r["exit_code"], 0); // console.error is not a throw
}

#[test]
fn code_run_exposes_stdin_global() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({"language": "javascript", "stdin": "payload", "code": "console.log(stdin)"}),
    ));
    assert!(r["stdout"].as_str().unwrap().contains("payload"));
}

#[test]
fn code_run_exposes_env_global() {
    let r = harness::expect_success(harness::invoke(
        "code_run",
        json!({
            "language": "javascript",
            "env": {"FOO": "bar"},
            "code": "console.log(env.FOO)"
        }),
    ));
    assert!(r["stdout"].as_str().unwrap().contains("bar"));
}

#[test]
fn code_run_rejects_non_javascript() {
    for lang in ["python", "python3", "bash", "sh", "cobol"] {
        let resp = harness::invoke("code_run", json!({"language": lang, "code": "print(1)"}));
        harness::expect_error(&resp, "interpreter_not_found");
    }
}

// ── code_session_* (persistent state) ───────────────────────────────────────────

#[test]
fn code_session_open_close() {
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "javascript"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();
    assert!(!sid.is_empty());
    assert_eq!(open["language"], "javascript");

    let close = harness::expect_success(harness::invoke_with_ctx(
        "code_session_close",
        json!({"session_id": sid}),
        &ctx,
    ));
    assert_eq!(close["closed"], true);
}

#[test]
fn code_session_state_persists_across_execs() {
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "javascript"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    // Define a variable and a function in one call...
    harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "let counter = 41; function inc() { counter += 1; return counter; }"}),
        &ctx,
    ));

    // ...and use them in the next.
    let exec = harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "console.log(inc())"}),
        &ctx,
    ));
    assert_eq!(exec["ok"], true);
    assert!(
        exec["stdout"].as_str().unwrap().contains("42"),
        "stdout: {}",
        exec["stdout"]
    );

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_throwing_exec_does_not_poison_state() {
    let ctx = ctx();
    let sid = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "javascript"}),
        &ctx,
    ))["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "const kept = 7;"}),
        &ctx,
    ));

    // A throwing snippet must not join the replay history. Note the response is
    // a *successful* tool call reporting a JS fault (`ok: false` + `error`), not
    // a tool-error envelope — so we read it directly rather than via
    // `expect_success` (which keys on the `error` field).
    let bad = harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "throw new Error('nope')"}),
        &ctx,
    );
    assert_eq!(bad["ok"], false);
    assert!(bad["error"].as_str().unwrap().contains("nope"));

    // Prior state survives; the bad snippet left no trace.
    let good = harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "console.log(kept)"}),
        &ctx,
    ));
    assert_eq!(good["ok"], true);
    assert!(good["stdout"].as_str().unwrap().contains("7"));

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_list_shows_open_sessions() {
    let ctx = ctx();
    let sid = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "js"}),
        &ctx,
    ))["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let list = harness::expect_success(harness::invoke_with_ctx(
        "code_session_list",
        json!({}),
        &ctx,
    ));
    let sessions = list["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["language"], "javascript");

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_open_rejects_non_javascript() {
    let ctx = ctx();
    let resp = harness::invoke_with_ctx("code_session_open", json!({"language": "python"}), &ctx);
    harness::expect_error(&resp, "interpreter_not_found");
}

#[test]
fn code_session_exec_not_found() {
    let ctx = ctx();
    let resp = harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": "sess_nonexistent", "code": "1 + 1"}),
        &ctx,
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn code_session_list_empty_initial() {
    let resp = harness::expect_success(harness::invoke("code_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}
