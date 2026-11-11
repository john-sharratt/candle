//! Conversation refs through the aggregator (`docs/private_mode.md` §3, §10),
//! against the fake gateway (`crate::fake`): a call is refused without a live
//! ref of its caller's, carries its ref to the backend and not its arguments,
//! expires idle, and is live on either aggregator — through any client, Grok
//! opening a session for every call included.

use std::sync::Arc;

use mcp_server::http::now_ms;
use mcp_server::refs::{Ref, REF_IDLE_MS, REF_TOUCH_EVERY_MS};
use mcp_server::{Client, Server};
use serde_json::{json, Value};
use tempfile::TempDir;

use crate::app::App;
use crate::context;
use crate::conversation::{EXPIRED, FORGED, NO_REF};
use crate::fake::{self, boss, call, call_as, caller, ref_of, until, Fake, LOOK, MAKE};

fn refused(answer: &Value, why: &str) {
    assert_eq!(answer["isError"], true, "{answer}");
    assert_eq!(context::text_of(answer), Some(why), "{answer}");
}

async fn begun(app: &Arc<App>) -> String {
    ref_of(&App::call(app, &call("s0"), context::GET, &json!({})).await)
}

/// **A call without a ref, with one not issued, or with another account's
/// is refused**, and reaches no backend.
#[tokio::test]
async fn a_call_without_a_live_ref_of_its_callers_is_refused() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    refused(
        &App::call(&app, &call("s1"), MAKE, &json!({})).await,
        NO_REF,
    );
    refused(
        &App::call(&app, &call("s1"), MAKE, &json!({"ref": "nonsense"})).await,
        FORGED,
    );
    let r = begun(&app).await;
    let other = call_as(caller("someone-else"), Client::ClaudeCode, "s1");
    refused(
        &App::call(&app, &other, MAKE, &json!({"ref": r})).await,
        FORGED,
    );
    refused(
        &App::call(&app, &call("s1"), "guide_list", &json!({})).await,
        NO_REF,
    );
    assert!(
        backend.sent(MAKE).is_empty(),
        "nothing refused reached the backend"
    );
}

/// **A call with its ref is run in its conversation**: the backend is told
/// the ref in `_meta`, and its tool is called without it.
#[tokio::test]
async fn a_call_carries_its_ref_to_the_backend_and_not_in_its_arguments() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let r = begun(&app).await;
    let answer = App::call(&app, &call("s1"), LOOK, &json!({"what": "x", "ref": r})).await;
    assert_eq!(answer["isError"], false, "{answer}");
    let sent = backend.sent(LOOK);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].args, json!({"what": "x"}));
    assert_eq!(sent[0].conversation_ref.as_deref(), Some(r.as_str()));
    let guide = App::call(&app, &call("s1"), "guide_list", &json!({"ref": r})).await;
    assert_eq!(guide["isError"], false, "a guide tool takes the ref too");
}

/// **`get_user_context` with its ref goes on with the same conversation**:
/// the backend is asked with the ref, and answers it.
#[tokio::test]
async fn get_user_context_with_its_ref_answers_the_same_ref() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let r = begun(&app).await;
    let again = App::call(&app, &call("s9"), context::GET, &json!({"ref": r})).await;
    assert_eq!(ref_of(&again), r);
    assert_eq!(backend.sent(context::GET)[1].args, json!({"ref": r}));
}

/// **An expired ref given to `get_user_context` is renewed, never revived**:
/// the answer names a new ref, keeping a private one's flag, and the expired
/// one stays expired — its use is never told.
#[tokio::test]
async fn an_expired_ref_is_renewed_and_never_revived() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let five_hours_ago = (now_ms() - REF_IDLE_MS - 3_600_000) / 1000;
    for private in [false, true] {
        let old = app.refs.seal(
            &boss().key(),
            &Ref {
                id: [private as u8 + 1; 16],
                private,
                issued_s: five_hours_ago as u32,
            },
        );
        let renewed = App::call(&app, &call("s1"), context::GET, &json!({"ref": old})).await;
        let new = ref_of(&renewed);
        assert_ne!(new, old, "renewed");
        assert_eq!(app.refs.open(&boss().key(), &new).unwrap().private, private);
        refused(
            &App::call(&app, &call("s1"), MAKE, &json!({"ref": old})).await,
            EXPIRED,
        );
        let ok = App::call(&app, &call("s1"), LOOK, &json!({"ref": new})).await;
        assert_eq!(ok["isError"], false, "{ok}");
    }
    assert!(
        backend
            .sent(context::REF_SEEN)
            .iter()
            .all(|s| s.args.get("used_ms").is_none()),
        "no use of an expired ref was told"
    );
}

/// **The answer's first line is its ref**, the client's guide after it.
#[tokio::test]
async fn the_answers_first_line_is_its_ref_and_the_guide_follows() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let answer = App::call(&app, &call("s0"), context::GET, &json!({})).await;
    let text = context::text_of(&answer).unwrap();
    assert!(text.starts_with("This conversation's ref is `"), "{text}");
    assert!(
        text.contains("\n\nHow to work with this user here (claude-code) — the `claude` guide:"),
        "{text}"
    );
}

/// **Every listed tool but `get_user_context` requires `ref`.**
#[tokio::test]
async fn every_listed_tool_but_get_user_context_requires_a_ref() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let listed = App::tools(&app, &call("s1")).await.unwrap();
    for tool in listed.as_array().unwrap() {
        let name = tool["name"].as_str().unwrap();
        let required = tool["inputSchema"]["required"]
            .as_array()
            .is_some_and(|r| r.iter().any(|v| v == "ref"));
        assert_eq!(required, name != context::GET, "{name}");
    }
}

/// **An idle ref expires at four hours**: one this aggregator cannot vouch
/// for is asked of the record, and the record never told of a use leaves it
/// expired.
#[tokio::test]
async fn an_idle_ref_expires_at_four_hours() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let r = begun(&app).await;
    let opened: Ref = app.refs.open(&boss().key(), &r).unwrap();
    let issued = opened.issued_ms();
    let c = call("s1");
    let at = |ms| App::conversation(&app, &c, Some(&r), ms);
    assert!(at(issued + REF_IDLE_MS - 1).await.is_ok());
    assert_eq!(at(issued + 2 * REF_IDLE_MS).await.unwrap_err(), EXPIRED);
    let asked = backend
        .sent(context::REF_SEEN)
        .iter()
        .filter(|s| s.args.get("used_ms").is_none())
        .count();
    assert_eq!(asked, 1, "asked of the record once");
}

/// **A ref used on one aggregator is live on the other**: the first tells
/// the record its use, and the second, which never saw it, asks.
#[tokio::test]
async fn a_ref_used_on_one_aggregator_is_live_on_the_other() {
    let backend = Fake::open();
    let gateway = fake::gateway(&backend).await;
    let (sa, sb) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let (a, b) = (fake::app(&gateway, &sa), fake::app(&gateway, &sb));
    let r = begun(&a).await;
    let issued = a.refs.open(&boss().key(), &r).unwrap().issued_ms();
    let used = issued + REF_TOUCH_EVERY_MS;
    App::conversation(&a, &call("s1"), Some(&r), used)
        .await
        .unwrap();
    until("the use was told", || {
        !backend.used.lock().unwrap().is_empty()
    })
    .await;
    let later = used + REF_IDLE_MS - 1;
    assert!(
        App::conversation(&b, &call("s2"), Some(&r), later)
            .await
            .is_ok(),
        "live by the record, past its issue's four hours"
    );
    let sc = TempDir::new().unwrap();
    let c = fake::app(&gateway, &sc);
    let gone = later + REF_IDLE_MS + 1;
    until("the second's use was told", || {
        backend.used.lock().unwrap().values().any(|&at| at == later)
    })
    .await;
    assert_eq!(
        App::conversation(&c, &call("s3"), Some(&r), gone)
            .await
            .unwrap_err(),
        EXPIRED,
        "idle four hours since the last use either told"
    );
}

/// **A Grok client opening a session for every call is one conversation**:
/// its ref works whatever session carries it, as through any other client.
#[tokio::test]
async fn a_grok_conversation_spans_a_session_per_call() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let grok = |s: &str| call_as(boss(), Client::Grok, s);
    let r = ref_of(&App::call(&app, &grok("g0"), context::GET, &json!({})).await);
    for (session, tool) in [("g1", LOOK), ("g2", MAKE), ("g3", "guide_list")] {
        let answer = App::call(&app, &grok(session), tool, &json!({"ref": r})).await;
        assert_eq!(answer["isError"], false, "{tool}: {answer}");
    }
    let sent = backend.sent(MAKE);
    assert_eq!(sent[0].conversation_ref.as_deref(), Some(r.as_str()));
    refused(
        &App::call(&app, &grok("g4"), MAKE, &json!({})).await,
        NO_REF,
    );
}

/// **A client may not call the aggregator's own backend tools.**
#[tokio::test]
async fn a_client_cannot_call_the_aggregators_own_backend_tools() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let r = begun(&app).await;
    for tool in [context::EPISODE, context::REF_SEEN] {
        let answer = App::call(
            &app,
            &call("s1"),
            tool,
            &json!({"ref": r, "used_ms": now_ms()}),
        )
        .await;
        assert_eq!(answer["isError"], true, "{tool}");
    }
    assert!(backend.sent(context::REF_SEEN).is_empty());
}
