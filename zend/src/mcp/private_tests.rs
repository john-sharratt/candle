//! Conversations and private mode at zend (`docs/private_mode.md` §3.4,
//! §4.4, §10), against a real substrate and a real journal repository: a
//! conversation's journal opens at its first write and never on a read, a
//! private conversation opens none, writes nothing and records no episode, and
//! the server opens the ref the aggregator passes for this caller alone.

use std::sync::Arc;

use axum::http::HeaderMap;
use journals::slug::Slug;
use mcp_server::access::{self, Access, DIARIES_PRIVATE, PRIVATE, TOOLS};
use mcp_server::protocol::MODERN;
use mcp_server::refs::named_in;
use mcp_server::{Call, Client, Server, Session};
use serde_json::{json, Value};

use super::caller::Caller;
use super::memory::conversations::section;
use super::records::Records;
use super::test_session::{self, conversation, ctx, john};
use super::tools;
use crate::session::ZendSession;

/// How many conversation journals the group holds.
fn conversations(session: &ZendSession) -> usize {
    let tree = session.mcp().memory().tree().unwrap();
    let zen = Slug::parse("zen").unwrap();
    tree.get(&section(&zen)).map_or(0, |j| j.children.len())
}

/// **A normal conversation's journal opens at its first write, and not
/// before**: `get_user_context` and every read leave the group as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_conversation_opens_its_journal_at_its_first_write_only() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let (text, r) = conversation(&session, &john(), false);
    let c = ctx(&session, john(), Some(r));
    let begun = tools::call(&c, "get_user_context", &json!({"ref": text})).await;
    assert!(!begun.is_error, "{}", begun.text);
    let read = tools::call(&c, "journal_read", &json!({"journal": "here"})).await;
    assert!(!read.is_error, "{}", read.text);
    assert_eq!(conversations(&session), 0, "a read opens nothing");
    let noted = tools::call(
        &c,
        "journal_note",
        &json!({"journal": "here", "text": "The synthetic widget is blue."}),
    )
    .await;
    assert!(!noted.is_error, "{}", noted.text);
    assert_eq!(conversations(&session), 1, "the first write opens it");
}

/// **A private conversation opens no journal and writes nothing**: its
/// beginning names none, its reads read, and its writes — the memory's, an
/// episode — are refused at zend as at the aggregator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_private_conversation_opens_nothing_and_writes_nothing() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let fresh = ctx(&session, john(), None);
    let begun = tools::call(&fresh, "get_user_context", &json!({"private_mode": true})).await;
    assert!(!begun.is_error, "{}", begun.text);
    assert!(
        begun.text.contains("This conversation is private"),
        "{}",
        begun.text
    );
    assert!(
        !begun.text.contains("conversations/"),
        "no journal is named for it: {}",
        begun.text
    );
    let text = named_in(&begun.text).unwrap().to_string();
    let r = session.mcp().refs().open(&john().key(), &text).unwrap();
    assert!(r.private);
    let c = ctx(&session, john(), Some(r));
    let read = tools::call(&c, "journal_read", &json!({"journal": "people/john"})).await;
    assert!(!read.is_error, "{}", read.text);
    for (tool, args) in [
        (
            "journal_note",
            json!({"journal": "here", "text": "The synthetic gadget is red."}),
        ),
        ("journal_focus", json!({"speaker": "john"})),
        (
            "user_episode_note",
            json!({"tool": "image_generate", "summary": "a synthetic cat"}),
        ),
        ("thread_attach", json!({"thread": "synthetic-thread"})),
    ] {
        let refused = tools::call(&c, tool, &args).await;
        assert!(refused.is_error, "{tool}");
        assert_eq!(refused.text, PRIVATE, "{tool}");
    }
    let here = tools::call(&c, "journal_read", &json!({"journal": "here"})).await;
    assert!(
        !here.text.contains("conversations/"),
        "no journal is named for it: {}",
        here.text
    );
    assert_eq!(session.mcp().noted().waiting(), 0, "no episode was noted");
    assert_eq!(conversations(&session), 0);
    let again = tools::call(&c, "get_user_context", &json!({"ref": text})).await;
    assert!(
        again.text.contains("This conversation is private"),
        "still private"
    );
}

/// **In a private conversation zend refuses every write the table names,
/// and no read it serves leaves a record**: the lists are the table's and
/// zend's own, so a tool added to either is held to this too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_write_is_refused_and_no_read_leaves_a_record() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let (_, r) = conversation(&session, &john(), true);
    let c = ctx(&session, john(), Some(r));
    for (tool, _) in TOOLS.iter().filter(|(_, a)| *a == Access::Write) {
        let refused = tools::call(&c, tool, &json!({})).await;
        assert_eq!(refused.text, PRIVATE, "{tool}");
    }
    let records = session.shared_substrate().unwrap();
    let before = records.under("mcp/");
    let defs = tools::definitions();
    let reads: Vec<&str> = defs
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|d| d["name"].as_str())
        .filter(|name| access::access(name) == Access::Read)
        .collect();
    assert!(reads.len() > 15, "{reads:?}");
    for tool in reads {
        tools::call(&c, tool, &json!({"wait_s": 0})).await;
    }
    assert_eq!(
        records.under("mcp/"),
        before,
        "a private read wrote a record"
    );
    assert_eq!(conversations(&session), 0);
}

/// **A diary needs a private conversation**, at zend as at the aggregator.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_diary_needs_a_private_conversation() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let (_, normal) = conversation(&session, &john(), false);
    for c in [
        ctx(&session, john(), None),
        ctx(&session, john(), Some(normal)),
    ] {
        let refused = tools::call(&c, "diary_list", &json!({})).await;
        assert_eq!(refused.text, DIARIES_PRIVATE);
    }
}

/// A call by `caller` in the conversation `ref_text` names, as the aggregator
/// passes it on.
fn served(session: &Arc<ZendSession>, caller: Caller, ref_text: &str) -> Call<ZendSession> {
    Call::<ZendSession> {
        headers: HeaderMap::new(),
        session_id: "s1".into(),
        session: Session::new(&caller.key(), Client::ClaudeAi, MODERN, 0),
        caller,
        conversation_ref: Some(ref_text.to_string()),
        quiet: false,
        store: Arc::clone(&ctx(session, john(), None).store),
        progress: None,
    }
}

/// **A diary end to end through the server, as the aggregator reaches
/// it**: open finds none, create, write one entry now and one in the past,
/// read them in time order — whole, however long, never cut to fit — delete
/// one, close, reopen and read what stands; and no diary call is logged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_diary_end_to_end_through_the_server() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let (text, _) = conversation(&session, &john(), true);
    let mine = served(&session, john(), &text);
    for tool in ["diary_open", "diary_read", "diary_write"] {
        assert!(
            !session.logs(&mine, tool, &json!({})),
            "{tool} is logged nowhere"
        );
    }
    let call = |name: &'static str, args: Value| {
        let (session, mine) = (Arc::clone(&session), served(&session, john(), &text));
        async move {
            let a = <ZendSession as Server>::call(&session, &mine, name, &args).await;
            assert!(!a.is_error, "{name}: {}", a.text);
            a.text
        }
    };
    let pass = "synthetic diary passphrase";
    assert_eq!(
        call("diary_open", json!({"passphrase": pass})).await,
        "Opened 0 diaries."
    );
    call(
        "diary_create",
        json!({"topic": "synthetic-garden", "passphrase": pass}),
    )
    .await;
    call(
        "diary_write",
        json!({"topic": "synthetic-garden", "text": "Planted synthetic beans.", "author": "user"}),
    )
    .await;
    let past = call(
        "diary_write",
        json!({"topic": "synthetic-garden", "text": "Dug the synthetic bed.",
               "ts": "2026-11-01T08:00:00Z"}),
    )
    .await;
    let long: String = (0..2_000)
        .map(|i| format!("synthetic line {i:05}\n"))
        .collect();
    assert!(long.len() > 32_000, "longer than any answer is cut to");
    call(
        "diary_write",
        json!({"topic": "synthetic-garden", "text": long, "ts": "2026-11-20T08:00:00Z"}),
    )
    .await;
    let read = call("diary_read", json!({"topic": "synthetic-garden"})).await;
    let dug = read.find("Dug the synthetic bed.").unwrap();
    let planted = read.find("Planted synthetic beans.").unwrap();
    assert!(dug < planted, "the past entry reads first");
    assert!(read.contains(&long), "read whole, never cut");
    let id = past
        .strip_prefix("Wrote entry `")
        .and_then(|r| r.split('`').next())
        .unwrap()
        .to_string();
    assert_eq!(
        call(
            "diary_delete",
            json!({"topic": "synthetic-garden", "ids": [id]})
        )
        .await,
        "Deleted 1 entry."
    );
    call("diary_close", json!({})).await;
    let closed = <ZendSession as Server>::call(
        &session,
        &mine,
        "diary_read",
        &json!({"topic": "synthetic-garden"}),
    )
    .await;
    assert!(closed.is_error, "closed: {}", closed.text);
    assert_eq!(
        call("diary_open", json!({"passphrase": pass})).await,
        "Opened 1 diary."
    );
    let read = call("diary_read", json!({"topic": "synthetic-garden"})).await;
    assert!(read.contains("Planted synthetic beans."), "{read}");
    assert!(
        !read.contains("Dug the synthetic bed."),
        "the delete stands: {read}"
    );
    assert_eq!(conversations(&session), 0, "a diary opens no journal");
}

/// **The server opens the ref the aggregator passes, for its caller only**,
/// and logs no call of a private conversation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_server_opens_the_passed_ref_for_its_caller_only() {
    let (_dirs, session, _) = test_session::session(|_| {});
    let (text, _) = conversation(&session, &john(), true);
    let call = |caller: Caller| served(&session, caller, &text);
    let mine = call(john());
    assert!(
        !session.logs(&mine, "journal_read", &json!({})),
        "private: not logged"
    );
    let refused = <ZendSession as Server>::call(&session, &mine, "journal_note", &json!({})).await;
    assert_eq!(
        refused.text, PRIVATE,
        "opened as John's private conversation"
    );
    let mut someone = john();
    someone.person.sub = "someone-else".into();
    let theirs = call(someone);
    assert!(
        session.logs(&theirs, "journal_read", &json!({})),
        "not theirs to open"
    );
}
