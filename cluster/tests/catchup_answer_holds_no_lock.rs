//! A catch-up's answer asks its store with the replicator's state free: the
//! store's answer can wait on a write whose tap takes that state.

mod support;

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::replicate::MemoryStore;

use support::tapping::TappingWhileAsked;
use support::{gateway, member, member_over, until};

const WITHIN: Duration = Duration::from_secs(20);

/// A deadlocked member holds its runtime's workers, so the scenario runs on a
/// runtime of its own and the test waits for it from outside.
const DEADLINE: Duration = Duration::from_secs(60);

/// **A member answers a catch-up while its store is writing.** The answer
/// reads the store's own high-water mark; a store whose answer waits on a
/// write meanwhile — whose tap takes the replicator's state — deadlocked an
/// answer that held that state across the read, and with it every thread of
/// the member that replicates.
#[test]
fn a_catchup_is_answered_while_the_store_writes() {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(caught_up_while_writing());
        let _ = done.send(());
    });
    finished
        .recv_timeout(DEADLINE)
        .expect("the catch-up answer deadlocked against the store's write");
}

async fn caught_up_while_writing() {
    let gw = gateway("gw");
    let a_store = TappingWhileAsked::new(1);
    let a = member_over(
        &[&gw],
        1,
        a_store.clone(),
        a_store.inner.clone(),
        MemoryVotes::new(),
    );
    until("a ready", WITHIN, || a.ready()).await;
    for k in 0..10u32 {
        a_store.inner.write(format!("a/{k}").as_bytes(), b"x");
    }
    let b = member(
        &[&gw],
        2,
        Arc::new(MemoryStore::new(2)),
        MemoryVotes::new(),
        true,
    );
    until("b caught up from a", WITHIN, || {
        b.ready() && (0..10u32).all(|k| b.store.value(format!("a/{k}").as_bytes()).is_some())
    })
    .await;
    assert!(a.app.replicator().unwrap().view().stats.answered >= 1);
}
