//! Which calls a resumed turn may re-issue.
//!
//! A turn killed between its tool call and that call's result is resumed by
//! asking for the tools again — the restart lost the results, so there is
//! nothing else to continue from. Re-issuing is only correct where running the
//! call twice leaves what running it once left, so every tool declares its own
//! answer ([`zend_tools::Tool::replay`]) and the default is refusal.
//!
//! The classification is a safety property, so these assert the actual answers
//! the dispatcher gives, not the flags behind them.

mod harness;

use zend_tools::Replay;

/// A patch is located by its context and a hunk already present is reported
/// rather than applied twice, so the same patch re-sent leaves the same file.
/// This is the whole reason `file_edit` takes a diff instead of a
/// find-and-replace: the old shape compounded on re-issue (`3` → `30` → `300`).
#[test]
fn a_file_patch_may_be_re_issued() {
    let args = serde_json::json!({
        "path": "src/main.rs",
        "patch": "@@ -1 +1 @@\n-old\n+new\n",
    });
    assert_eq!(harness::replay("file_edit", args), Replay::Safe);
}

/// Reads change nothing, so a resumed turn simply reads again.
#[test]
fn a_read_may_be_re_issued() {
    let args = serde_json::json!({ "path": "src/main.rs" });
    assert_eq!(harness::replay("file_read", args), Replay::Safe);
}

/// One request through an open session, under `method`.
fn with_method(method: &str) -> serde_json::Value {
    serde_json::json!({ "session_id": "s1", "path": "/v1/things", "method": method })
}

/// `http_request` is the one call whose answer depends on its arguments: the
/// retrieving verbs may be re-issued, the rest would do a second time whatever
/// the first request did at the far end. `post` in lower case is judged the
/// same as `POST` — the verb is upper-cased before it is read.
#[test]
fn an_http_request_is_judged_by_its_verb() {
    for verb in ["GET", "HEAD", "OPTIONS", "get"] {
        let answer = harness::replay("http_request", with_method(verb));
        assert_eq!(answer, Replay::Safe, "{verb}");
    }
    for verb in ["POST", "PUT", "PATCH", "DELETE", "post"] {
        let answer = harness::replay("http_request", with_method(verb));
        assert_eq!(answer, Replay::Unsafe, "{verb}");
    }
}

/// An omitted verb is a GET, exactly as the tool itself defaults it.
#[test]
fn an_http_request_with_no_verb_is_a_get() {
    let args = serde_json::json!({ "session_id": "s1", "path": "/v1/things" });
    assert_eq!(harness::replay("http_request", args), Replay::Safe);
}

/// A call that never reaches `run` has no effect to repeat: re-issuing it
/// produces the same `invalid_arguments` the first one did. Refusing it would
/// hide the real error behind a resume message.
#[test]
fn a_call_that_cannot_parse_is_not_worth_refusing() {
    let args = serde_json::json!({ "session_id": 17 });
    assert_eq!(harness::replay("http_request", args), Replay::Safe);
}

/// Likewise a name no tool answers to — `run` returns `unknown_tool` either
/// way.
#[test]
fn an_unknown_tool_is_not_worth_refusing() {
    let args = serde_json::json!({});
    assert_eq!(harness::replay("no_such_tool", args), Replay::Safe);
}
