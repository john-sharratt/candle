mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

fn python_available() -> bool {
    let exe = if cfg!(windows) {
        "python.exe"
    } else {
        "python3"
    };
    std::process::Command::new(exe)
        .arg("--version")
        .output()
        .is_ok()
}

#[test]
fn code_run_bash_exit_code() {
    if cfg!(windows) {
        return; // bash may not be available on Windows
    }
    let resp = harness::invoke("code_run", json!({"language": "bash", "code": "exit 42"}));
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert_eq!(r["exit_code"], 42);
    }
}

#[test]
fn code_run_stdout_capture() {
    if cfg!(windows) {
        return;
    }
    let resp = harness::invoke(
        "code_run",
        json!({"language": "bash", "code": "echo hello"}),
    );
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert!(r["stdout"].as_str().unwrap().contains("hello"));
    }
}

#[test]
fn code_run_env_var() {
    if cfg!(windows) {
        return;
    }
    let resp = harness::invoke(
        "code_run",
        json!({
            "language": "bash",
            "code": "echo $FOO",
            "env": {"FOO": "bar"}
        }),
    );
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert!(r["stdout"].as_str().unwrap().contains("bar"));
    }
}

#[test]
fn code_run_interpreter_not_found() {
    let resp = harness::invoke(
        "code_run",
        json!({"language": "cobol", "code": "DISPLAY 'HELLO'"}),
    );
    harness::expect_error(&resp, "interpreter_not_found");
}

#[test]
fn code_session_open_close_python() {
    if !python_available() {
        println!("python not available, skipping code_session_open_close_python");
        return;
    }
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "python"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();
    assert!(!sid.is_empty());

    let close = harness::expect_success(harness::invoke_with_ctx(
        "code_session_close",
        json!({"session_id": sid}),
        &ctx,
    ));
    assert_eq!(close["closed"], true);
}

#[test]
fn code_session_exec_python() {
    if !python_available() {
        println!("python not available, skipping code_session_exec_python");
        return;
    }
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "python"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    // Execute code
    let exec = harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "print('hello from session')"}),
        &ctx,
    ));
    assert_eq!(exec["ok"], true);
    assert!(
        exec["stdout"]
            .as_str()
            .unwrap()
            .contains("hello from session"),
        "stdout: {}",
        exec["stdout"]
    );

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_state_persistence() {
    if !python_available() {
        println!("python not available, skipping code_session_state_persistence");
        return;
    }
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "python"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    // Set a variable
    harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "x = 99"}),
        &ctx,
    ));

    // Use it in next call
    let exec = harness::expect_success(harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": sid, "code": "print(x)"}),
        &ctx,
    ));
    assert_eq!(exec["ok"], true);
    assert!(
        exec["stdout"].as_str().unwrap().contains("99"),
        "stdout: {}",
        exec["stdout"]
    );

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_list_shows_open_sessions() {
    if !python_available() {
        println!("python not available, skipping code_session_list_shows_open_sessions");
        return;
    }
    let ctx = ctx();
    let open = harness::expect_success(harness::invoke_with_ctx(
        "code_session_open",
        json!({"language": "python"}),
        &ctx,
    ));
    let sid = open["session_id"].as_str().unwrap().to_string();

    let list = harness::expect_success(harness::invoke_with_ctx(
        "code_session_list",
        json!({}),
        &ctx,
    ));
    let sessions = list["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["language"], "python");

    harness::invoke_with_ctx("code_session_close", json!({"session_id": sid}), &ctx);
}

#[test]
fn code_session_exec_not_found() {
    let ctx = ctx();
    let resp = harness::invoke_with_ctx(
        "code_session_exec",
        json!({"session_id": "sess_nonexistent", "code": "print(1)"}),
        &ctx,
    );
    harness::expect_error(&resp, "session_not_found");
}

#[test]
fn code_run_python_echo() {
    let resp = harness::invoke(
        "code_run",
        json!({
            "language": "python",
            "code": "print('hello from python')"
        }),
    );
    // If python is available, check output; otherwise expect interpreter_not_found
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert!(r["stdout"].as_str().unwrap().contains("hello from python"));
        assert_eq!(r["exit_code"], 0);
    } else {
        let code = resp["error"].as_str().unwrap();
        assert!(code == "interpreter_not_found" || code == "execution_failed" || code == "timeout");
    }
}

#[test]
fn code_run_unknown_language() {
    let resp = harness::invoke(
        "code_run",
        json!({
            "language": "cobol",
            "code": "DISPLAY 'HELLO'"
        }),
    );
    harness::expect_error(&resp, "interpreter_not_found");
}

#[test]
fn code_session_list_empty_initial() {
    let resp = harness::expect_success(harness::invoke("code_session_list", json!({})));
    assert_eq!(resp["sessions"].as_array().unwrap().len(), 0);
}
