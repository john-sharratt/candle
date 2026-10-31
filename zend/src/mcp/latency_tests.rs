//! The latency gate for `/mcp`'s tools that run no model: each driven through
//! the server's own dispatch, against a real substrate and a real journal
//! repository — grown past the real one's size, its sync thread pushing to
//! its origin as it does in the daemon — and refused when its 95th
//! percentile is over [`P95_BUDGET`].
//!
//! It times the tools themselves — what a member spends on a call — not the
//! gateway hops in front of them. It measures time, so it is run on its own,
//! on a quiet machine: `cargo test -p zend --lib latency -- --ignored
//! --test-threads=1`.

use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use candle_conversation::projection::Conversation;
use journals::actor::Actor;
use journals::ops::{OnTaken, Op};
use journals::path::JournalPath;
use journals::scope::{Channel, Section};
use journals::slug::Slug;
use journals::store::Store as JournalStore;
use journals::time::Stamp;
use mcp_server::protocol::MODERN;
use mcp_server::{Client, Session, Store};
use serde_json::{json, Value};
use tempfile::TempDir;
use web::auth::Role;

use super::caller::{Caller, Person};
use super::memory::tests::repo;
use super::state::RecordStore;
use super::tools::{self, Ctx};
use super::workers::now_ms;
use crate::config::DaemonConfig;
use crate::session::ZendSession;
use crate::workspace::single_repo;

/// The most a tool's 95th percentile may be.
const P95_BUDGET: Duration = Duration::from_millis(30);

/// How many times each tool is called.
const ROUNDS: usize = 40;

/// How large the journal repository is made before the gate runs: past the
/// real one's size, so a cost that grows with the repository shows. A
/// journal holds at most ten live sub-journals, so the topics are areas of
/// ten.
const AREAS: usize = 4;
const TOPICS_PER_AREA: usize = 10;
const NOTES_PER_TOPIC: usize = 40;

/// `dir`'s journal repository grown to [`AREAS`] × [`TOPICS_PER_AREA`]
/// topics of [`NOTES_PER_TOPIC`] lines each, committed through the store as
/// the service commits them.
fn grow(dir: &Path) {
    let mut store = JournalStore::open(dir).unwrap();
    let john = Actor::Person {
        person: Slug::parse("john").unwrap(),
        channel: Channel::ClaudeCode,
        admin: false,
    };
    let zen = Slug::parse("zen").unwrap();
    let topics = JournalPath::of_section(&zen, Section::Topics);
    let at = Stamp::parse("2026-10-30T09:00:00Z").unwrap();
    let mut written = 0i64;
    let make = |store: &mut JournalStore, under: &JournalPath, name: String| {
        let name = Slug::parse(&name).unwrap();
        let create = Op::Create {
            under: under.clone(),
            name: name.clone(),
            cue: format!("open for {name}"),
            on_taken: OnTaken::Refuse,
        };
        store.write(&john, &create, at).unwrap();
        under.child(name).unwrap()
    };
    for a in 0..AREAS {
        let area = make(&mut store, &topics, format!("area-{a}"));
        for t in 0..TOPICS_PER_AREA {
            let topic = make(&mut store, &area, format!("topic-{t}"));
            for n in 0..NOTES_PER_TOPIC {
                let note = Op::Note {
                    path: topic.clone(),
                    text: format!("Area {a} topic {t} holds line {n}."),
                    speaker: None,
                };
                written += 1;
                store.write(&john, &note, at.plus(written)).unwrap();
            }
        }
    }
}

/// A session on a real substrate, its memory open on a real journal
/// repository grown past the real one's size ([`grow`]), and a call context
/// for John — the repository's owner — as an admin, so every tool is his to
/// call.
fn session() -> (Vec<TempDir>, Ctx) {
    let work = TempDir::new().unwrap();
    let session = Arc::new(ZendSession::new(DaemonConfig::new(
        single_repo(work.path(), "project").unwrap(),
    )));
    session.hold_substrate(Conversation::new());
    let (journals, dir) = repo();
    grow(&dir);
    let memory = session.mcp().memory();
    memory.start(dir);
    let until = Instant::now() + Duration::from_secs(30);
    while memory.tree().is_err() {
        assert!(Instant::now() < until, "the memory never opened");
        thread::sleep(Duration::from_millis(20));
    }
    let caller = Caller {
        person: Person {
            provider: "google".into(),
            sub: "john".into(),
            email: "john@example.com".into(),
            name: "John".into(),
        },
        role: Role::Admin,
    };
    let records = session.shared_substrate().unwrap();
    let store: Arc<dyn Store> =
        Arc::new(RecordStore::new(records, Arc::clone(session.mcp().used())));
    let ctx = Ctx {
        mcp: Session::new(&caller.key(), Client::ClaudeCode, MODERN, now_ms()),
        mcp_id: "latency".into(),
        session,
        caller,
        store,
        progress: None,
    };
    (vec![work, journals], ctx)
}

/// The calls of one round, the `i`th: each tool that runs no model and needs
/// no other member, with what it writes new each time — each under the label
/// it is timed by, a thread's journal and the memory's apart.
fn round(i: usize) -> Vec<(&'static str, &'static str, Value)> {
    vec![
        ("get_user_context", "get_user_context", json!({})),
        (
            "thread_attach",
            "thread_attach",
            json!({"thread": "review"}),
        ),
        ("thread_list", "thread_list", json!({})),
        (
            "journal_note T:",
            "journal_note",
            json!({"journal": "T:review", "text": format!("Checked the build, round {i}.")}),
        ),
        (
            "journal_read T:",
            "journal_read",
            json!({"journal": "T:review"}),
        ),
        (
            "journal_grep T:",
            "journal_grep",
            json!({"journal": "T:review", "pattern": "build"}),
        ),
        (
            "journal_note memory",
            "journal_note",
            json!({"journal": "people/john", "text": format!("Prefers tea, cup {i}.")}),
        ),
        (
            "journal_read memory",
            "journal_read",
            json!({"journal": "people/john"}),
        ),
        (
            "journal_grep memory",
            "journal_grep",
            json!({"journal": "*", "pattern": "tea"}),
        ),
        // A focus that changes nothing is refused: each round moves it.
        (
            "journal_focus",
            "journal_focus",
            json!({"speaker": "john", "about": [if i.is_multiple_of(2) { "people/john" } else { "people" }]}),
        ),
    ]
}

/// The 95th percentile of `times`.
fn p95(times: &mut [Duration]) -> Duration {
    times.sort();
    times[(times.len() * 95).div_ceil(100) - 1]
}

/// **Every tool that runs no model answers within its budget at the 95th
/// percentile**, its answer no error, while the journal repository syncs
/// beside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a latency gate: run on its own, on a quiet machine"]
async fn every_tool_without_a_model_answers_within_its_budget() {
    let (_dirs, ctx) = session();
    let mut times: Vec<(&'static str, Vec<Duration>)> = Vec::new();
    for i in 0..ROUNDS {
        for (label, name, args) in round(i) {
            let started = Instant::now();
            let answer = tools::call(&ctx, name, &args).await;
            let took = started.elapsed();
            assert!(!answer.is_error, "{name} {args}: {}", answer.text);
            match times.iter_mut().find(|(l, _)| *l == label) {
                Some((_, t)) => t.push(took),
                None => times.push((label, vec![took])),
            }
        }
    }
    let over: Vec<String> = times
        .iter_mut()
        .map(|(name, t)| (*name, p95(t)))
        .inspect(|(name, p)| eprintln!("{name}: p95 {} us", p.as_micros()))
        .filter(|(_, p)| *p > P95_BUDGET)
        .map(|(name, p)| format!("{name} {} ms", p.as_millis()))
        .collect();
    assert!(
        over.is_empty(),
        "over {P95_BUDGET:?} at p95: {}",
        over.join(", ")
    );
}

#[test]
fn the_95th_percentile_is_the_time_95_in_100_are_within() {
    let ms = Duration::from_millis;
    let mut times: Vec<Duration> = (1..=100).map(ms).collect();
    assert_eq!(p95(&mut times), ms(95));
    let mut few = vec![ms(3), ms(1), ms(2)];
    assert_eq!(p95(&mut few), ms(3));
}
