//! A call answers while the listing that decides its record is held, against
//! the fake gateway (`crate::fake`) with its listing gated.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mcp_server::Server;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};

use super::Listing;
use crate::app::App;
use crate::context;
use crate::fake::{self, call, ref_of, until, Fake};

/// **A call answers while the listing is held**; three calls during one held
/// listing cause one listing; once it is let go, the read-only tool's calls
/// record nothing and the write tool's call records exactly one episode — on
/// its conversation.
#[tokio::test]
async fn a_call_answers_while_its_record_waits_for_a_listing() {
    let backend = Fake::gated();
    let state = TempDir::new().unwrap();
    let app = fake::app(&fake::gateway(&backend).await, &state);
    let begun = App::call(&app, &call("s0"), context::GET, &json!({})).await;
    let r = ref_of(&begun);

    for (session, tool) in [("s1", "look"), ("s2", "make"), ("s3", "look")] {
        let answered = timeout(
            Duration::from_secs(5),
            App::call(&app, &call(session), tool, &json!({"what": "x", "ref": r})),
        )
        .await
        .expect("the call answered while the listing was held");
        assert_eq!(answered["isError"], false, "{answered}");
    }
    until("a listing began", || {
        backend.lists.load(Ordering::SeqCst) == 1
    })
    .await;
    assert!(backend.sent(context::EPISODE).is_empty());

    backend.release();
    until("the write was recorded", || {
        !backend.sent(context::EPISODE).is_empty()
    })
    .await;
    sleep(Duration::from_millis(200)).await;
    let episodes = backend.sent(context::EPISODE);
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].args["tool"], "make");
    assert_eq!(episodes[0].conversation_ref.as_deref(), Some(r.as_str()));
    assert_eq!(backend.lists.load(Ordering::SeqCst), 1, "one listing");
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
