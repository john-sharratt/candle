//! A call answers while the listing that decides its record is held, against a
//! fake gateway on a local port whose `tools/list` waits to be let go.

use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, RwLock};
use std::time::Duration;

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use mcp_server::protocol::MODERN;
use mcp_server::{Call, Client, Identity, SealedStore, Server, Session};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};

use super::Listing;
use crate::app::App;
use crate::backend::{Backend, Link};
use crate::caller::Caller;
use crate::context::{self, Conversations};
use crate::guides;
use crate::last_listed::LastListed;
use crate::reading::Readings;
use crate::routes::Routes;
use crate::served::Served;

/// The fake gateway's backend: how many listings it was asked for, the
/// episodes recorded on it, and the gate its listing waits at.
#[derive(Default)]
struct Fake {
    lists: AtomicUsize,
    episodes: StdMutex<Vec<String>>,
    open: AtomicBool,
    gate: Notify,
}

impl Fake {
    fn release(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.gate.notify_waiters();
    }

    async fn wait_open(&self) {
        loop {
            let opened = self.gate.notified();
            if self.open.load(Ordering::SeqCst) {
                return;
            }
            opened.await;
        }
    }
}

async fn answer(State(fake): State<Arc<Fake>>, Json(req): Json<Value>) -> Json<Value> {
    let id = req["id"].clone();
    let result = match req["method"].as_str() {
        Some("tools/list") => {
            fake.lists.fetch_add(1, Ordering::SeqCst);
            fake.wait_open().await;
            json!({"tools": [
                {"name": context::GET},
                {"name": "look", "annotations": {"readOnlyHint": true}},
                {"name": "make"},
            ]})
        }
        _ => {
            if req["params"]["name"] == context::EPISODE {
                let tool = req["params"]["arguments"]["tool"].as_str().unwrap_or("");
                fake.episodes.lock().unwrap().push(tool.to_string());
            }
            json!({"content": [{"type": "text", "text": "done"}], "isError": false})
        }
    };
    Json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// An aggregator over one backend behind the fake gateway, whose routes are
/// known from an earlier run and which has listed nothing yet.
async fn app(fake: &Arc<Fake>, state: &TempDir) -> Arc<App> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Router::new()
        .route("/v1/mcp", post(answer))
        .with_state(Arc::clone(fake));
    tokio::spawn(async move { axum::serve(listener, router).await });
    let routes = state.path().join("routes.json");
    fs::write(
        &routes,
        json!({"look": "zen", "make": "zen", "get_user_context": "zen"}).to_string(),
    )
    .unwrap();
    Arc::new(App {
        identity: Identity {
            name: "mcp".into(),
            title: "mcp".into(),
            description: String::new(),
            instructions: String::new(),
            icon: None,
            version: "0".into(),
            list_changed: true,
        },
        store: Arc::new(SealedStore),
        backends: vec![Backend {
            name: "zen".into(),
            host: "code.tokera.com".into(),
            path: "/v1/mcp".into(),
            memory: None,
        }],
        link: Link::new(&format!("http://{addr}")).unwrap(),
        routes: Routes::kept_in(&routes, vec!["zen".into()]),
        last_listed: LastListed::default(),
        served: Served::default(),
        library: guides::embedded().unwrap(),
        read_only: RwLock::new(BTreeSet::new()),
        conversations: Conversations::default(),
        readings: Readings::default(),
        listing: Listing::default(),
    })
}

fn call(session: &str) -> Call<App> {
    Call {
        caller: Caller {
            provider: "google".into(),
            sub: "boss".into(),
            token: "t0k".into(),
        },
        headers: Default::default(),
        session_id: session.to_string(),
        session: Session::new("o", Client::ClaudeCode, MODERN, 0),
        store: Arc::new(SealedStore),
        progress: None,
    }
}

/// Wait until `done` holds, for at most five seconds.
async fn until(what: &str, done: impl Fn() -> bool) {
    timeout(Duration::from_secs(5), async {
        while !done() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}"));
}

/// **A call answers while the listing is held**; three calls during one held
/// listing cause one listing; once it is let go, the read-only tool's calls
/// record nothing and the write tool's call records exactly one episode.
#[tokio::test]
async fn a_call_answers_while_its_record_waits_for_a_listing() {
    let fake = Arc::new(Fake::default());
    let state = TempDir::new().unwrap();
    let app = app(&fake, &state).await;

    for (session, tool) in [("s1", "look"), ("s2", "make"), ("s3", "look")] {
        let answered = timeout(
            Duration::from_secs(5),
            App::call(&app, &call(session), tool, &json!({"what": "x"})),
        )
        .await
        .expect("the call answered while the listing was held");
        assert_eq!(answered["isError"], false, "{answered}");
    }
    until("a listing began", || fake.lists.load(Ordering::SeqCst) == 1).await;
    assert!(fake.episodes.lock().unwrap().is_empty());

    fake.release();
    until("the write was recorded", || {
        !fake.episodes.lock().unwrap().is_empty()
    })
    .await;
    sleep(Duration::from_millis(200)).await;
    assert_eq!(*fake.episodes.lock().unwrap(), ["make"]);
    assert_eq!(fake.lists.load(Ordering::SeqCst), 1, "one listing");
    assert!(app.listing.listed());
}

/// **A task that waited while another listed does not list again**, even when
/// that listing found nothing.
#[tokio::test]
async fn a_waiting_task_takes_the_listing_it_waited_for() {
    let listing = Arc::new(Listing::default());
    let held = Arc::new(Notify::new());
    let ran = Arc::new(AtomicUsize::new(0));
    let first = {
        let (listing, held, ran) = (Arc::clone(&listing), Arc::clone(&held), Arc::clone(&ran));
        tokio::spawn(async move {
            listing
                .ensure(async {
                    ran.fetch_add(1, Ordering::SeqCst);
                    held.notified().await;
                })
                .await
        })
    };
    until("the first listing began", || {
        ran.load(Ordering::SeqCst) == 1
    })
    .await;
    let second = {
        let (listing, ran) = (Arc::clone(&listing), Arc::clone(&ran));
        tokio::spawn(async move {
            listing
                .ensure(async {
                    ran.fetch_add(1, Ordering::SeqCst);
                })
                .await
        })
    };
    sleep(Duration::from_millis(50)).await;
    held.notify_one();
    assert!(first.await.unwrap());
    assert!(!second.await.unwrap(), "it waited while another listed");
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert!(!listing.listed(), "nothing was listed successfully");

    assert!(listing.ensure(async {}).await, "a later call lists afresh");
    listing.mark_listed();
    assert!(!listing.ensure(async {}).await, "listed once, never again");
}
