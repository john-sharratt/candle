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

use harness::REPO;

/// A replace or delete re-sent after it landed names an anchor its part no
/// longer has and is refused, rather than compounding (`3` → `30` → `300`);
/// a move re-sent puts its part where it already stands. An insert leaves its
/// anchor as it was, so the same insert re-sent would land twice — it is the
/// one edit a resumed turn may not re-issue.
#[test]
fn a_file_edit_may_be_re_issued_unless_it_inserts() {
    let edit = |action: &str| {
        serde_json::json!({
            "repo": REPO,
            "path": "src/main.rs",
            "at": "Qm7x",
            "do": action,
            "text": "new",
        })
    };
    for safe in ["replace", "delete", "move"] {
        assert_eq!(
            harness::replay("file_edit", edit(safe)),
            Replay::Safe,
            "{safe}"
        );
    }
    for unsafe_ in ["insert_before", "insert_after"] {
        assert_eq!(
            harness::replay("file_edit", edit(unsafe_)),
            Replay::Unsafe,
            "{unsafe_}"
        );
    }
}

/// Reads change nothing, so a resumed turn simply reads again.
#[test]
fn a_read_may_be_re_issued() {
    let args = serde_json::json!({ "repo": REPO, "path": "src/main.rs", "page": 1 });
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
