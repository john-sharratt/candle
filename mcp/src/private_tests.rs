//! Private mode through the aggregator (`docs/private_mode.md` §4, §10),
//! against the fake gateway (`crate::fake`): every write the table names is
//! refused in a private conversation, and only there are diaries reached; a
//! private conversation's calls record nothing, are logged nowhere, and its
//! flag reaches no other conversation — through any client.

use std::sync::Arc;

use mcp_server::access::{self, Access, DIARIES_PRIVATE, PRIVATE, TOOLS};
use mcp_server::http::now_ms;
use mcp_server::refs::{Ref, REF_IDLE_MS};
use mcp_server::{Client, Server};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

use crate::app::App;
use crate::context;
use crate::conversation::EXPIRED;
use crate::fake::{self, boss, call, call_as, ref_of, Fake, DIARY, LOOK, MAKE};

async fn begin(app: &Arc<App>, private: bool) -> String {
    ref_of(
        &App::call(
            app,
            &call("s0"),
            context::GET,
            &json!({"private_mode": private}),
        )
        .await,
    )
}

fn says(answer: &Value, why: &str) {
    assert_eq!(answer["isError"], true, "{answer}");
    assert_eq!(context::text_of(answer), Some(why), "{answer}");
}

/// **Every tool the table names a write is refused in a private
/// conversation**, and none reaches a backend; the list is the table's own,
/// so a write added to it is held to this too.
#[tokio::test]
async fn every_write_is_refused_in_a_private_conversation() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let r = begin(&app, true).await;
    let writes: Vec<&str> = TOOLS
        .iter()
        .filter(|(name, a)| *a == Access::Write && !context::is_internal(name))
        .map(|(name, _)| *name)
        .collect();
    assert!(writes.len() > 40, "the table names its writes: {writes:?}");
    for tool in &writes {
        says(
            &App::call(&app, &call("s1"), tool, &json!({"ref": r})).await,
            PRIVATE,
        );
    }
    let reached: Vec<String> = backend
        .sent
        .lock()
        .unwrap()
        .iter()
        .map(|s| s.tool.clone())
        .filter(|t| t != context::GET)
        .collect();
    assert!(reached.is_empty(), "refused writes reached: {reached:?}");
    says(
        &App::call(&app, &call("s1"), "a_tool_nobody_named", &json!({"ref": r})).await,
        PRIVATE,
    );
}

/// **A private conversation reads, quietly, and records nothing**: its read
/// reaches the backend asking not to be logged, and no episode follows.
#[tokio::test]
async fn a_private_conversation_reads_quietly_and_records_nothing() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let begun = App::call(
        &app,
        &call("s0"),
        context::GET,
        &json!({"private_mode": true}),
    )
    .await;
    let served = context::text_of(&begun).unwrap();
    assert!(served.contains("the `private-mode` guide:"), "{served}");
    assert!(
        !served.contains("the `claude` guide:"),
        "no guide that says to write: {served}"
    );
    let r = ref_of(&begun);
    let read = App::call(&app, &call("s1"), LOOK, &json!({"ref": r})).await;
    assert_eq!(read["isError"], false, "{read}");
    let guide = App::call(&app, &call("s1"), "guide_list", &json!({"ref": r})).await;
    assert_eq!(guide["isError"], false, "{guide}");
    let sent = backend.sent(LOOK);
    assert!(sent[0].quiet, "the backend is asked not to log it");
    assert!(backend.sent(context::GET)[0].quiet, "nor its beginning");
    sleep(Duration::from_millis(200)).await;
    assert!(backend.sent(context::EPISODE).is_empty());
    assert_eq!(app.counters.read(), (1, 2, 0));
}

/// **Diaries are reached only from a private conversation.**
#[tokio::test]
async fn diaries_are_reached_only_from_a_private_conversation() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let normal = begin(&app, false).await;
    says(
        &App::call(&app, &call("s1"), DIARY, &json!({"ref": normal})).await,
        DIARIES_PRIVATE,
    );
    assert!(backend.sent(DIARY).is_empty());
    let private = begin(&app, true).await;
    let read = App::call(&app, &call("s1"), DIARY, &json!({"ref": private})).await;
    assert_eq!(read["isError"], false, "{read}");
    assert!(backend.sent(DIARY)[0].quiet);
    assert_eq!(app.counters.read().2, 1, "counted as a diary call");
}

/// **A private conversation's flag reaches no other**: a normal one begun
/// beside it, on the same MCP session, writes, and its write is recorded.
#[tokio::test]
async fn private_reaches_no_other_conversation() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let private = begin(&app, true).await;
    let normal = begin(&app, false).await;
    says(
        &App::call(&app, &call("s0"), MAKE, &json!({"ref": private})).await,
        PRIVATE,
    );
    let made = App::call(&app, &call("s0"), MAKE, &json!({"ref": normal})).await;
    assert_eq!(made["isError"], false, "{made}");
    assert!(!backend.sent(MAKE)[0].quiet);
}

/// **`private_mode` with a live normal ref changes nothing**: the backend
/// goes on with the normal conversation, so it is served the client's own
/// guide — never the `private-mode` guide, which would tell the client it is
/// off the record while its writes still go through.
#[tokio::test]
async fn private_mode_with_a_live_normal_ref_is_served_as_normal() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let normal = begin(&app, false).await;
    let again = App::call(
        &app,
        &call("s1"),
        context::GET,
        &json!({"ref": normal, "private_mode": true}),
    )
    .await;
    assert_eq!(ref_of(&again), normal, "the same conversation");
    let served = context::text_of(&again).unwrap();
    assert!(served.contains("the `claude` guide:"), "{served}");
    assert!(!served.contains("the `private-mode` guide:"), "{served}");
    let made = App::call(&app, &call("s1"), MAKE, &json!({"ref": normal})).await;
    assert_eq!(made["isError"], false, "{made}");
}

/// **An expired private ref refuses writes still**: every write with it is
/// refused whole, never served as a normal conversation's, and renewing it
/// answers a private ref.
#[tokio::test]
async fn an_expired_private_ref_still_refuses_writes() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let five_hours_ago = (now_ms() - REF_IDLE_MS - 3_600_000) / 1000;
    let old = app.refs.seal(
        &boss().key(),
        &Ref {
            id: [7; 16],
            private: true,
            issued_s: five_hours_ago as u32,
        },
    );
    for tool in [MAKE, LOOK, DIARY] {
        says(
            &App::call(&app, &call("s1"), tool, &json!({"ref": old})).await,
            EXPIRED,
        );
    }
    assert!(backend.sent(MAKE).is_empty() && backend.sent(DIARY).is_empty());
    let renewed = ref_of(&App::call(&app, &call("s1"), context::GET, &json!({"ref": old})).await);
    assert!(app.refs.open(&boss().key(), &renewed).unwrap().private);
    says(
        &App::call(&app, &call("s1"), MAKE, &json!({"ref": renewed})).await,
        PRIVATE,
    );
}

/// **Nothing private, and no diary, is logged here**; anything else is.
#[tokio::test]
async fn nothing_private_and_no_diary_is_logged() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let private = begin(&app, true).await;
    let normal = begin(&app, false).await;
    let c = call("s1");
    assert!(!app.logs(&c, LOOK, &json!({"ref": private})));
    assert!(!app.logs(&c, DIARY, &json!({"ref": normal})));
    assert!(!app.logs(&c, context::GET, &json!({"private_mode": true})));
    assert!(!app.logs(&c, context::GET, &json!({"ref": private})));
    assert!(app.logs(&c, LOOK, &json!({"ref": normal})));
    assert!(app.logs(&c, context::GET, &json!({})));
    assert!(app.logs(&c, LOOK, &json!({})), "a refused call is logged");
}

/// **Grok, opening a session for every call, is held to the same**.
#[tokio::test]
async fn a_grok_private_conversation_is_held_to_the_same() {
    let backend = Fake::open();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let grok = |s: &str| call_as(boss(), Client::Grok, s);
    let begun = App::call(
        &app,
        &grok("g0"),
        context::GET,
        &json!({"private_mode": true}),
    )
    .await;
    let r = ref_of(&begun);
    says(
        &App::call(&app, &grok("g1"), MAKE, &json!({"ref": r})).await,
        PRIVATE,
    );
    let read = App::call(&app, &grok("g2"), LOOK, &json!({"ref": r})).await;
    assert_eq!(read["isError"], false);
    let diary = App::call(&app, &grok("g3"), DIARY, &json!({"ref": r})).await;
    assert_eq!(diary["isError"], false);
    assert_eq!(access::access(DIARY), Access::Diary);
}
