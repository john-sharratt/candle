//! Tier-2 integration test for the workspace watcher.
//!
//! Spins up the watcher against a temp workspace, performs file
//! operations, and asserts the debounced refresh callback fires (or
//! doesn't) per the operation's relevance to the file-name set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn watcher_fires_callback_on_file_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let workspace = dir.path().to_path_buf();
    std::fs::write(workspace.join("seed.rs"), b"// seed\n").unwrap();

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let cb: Arc<dyn Fn() + Send + Sync> =
        Arc::new(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

    let _watcher = zend::watcher::spawn(&workspace, cb).expect("watcher started");
    // Allow the watcher's background task a moment to arm before we
    // start writing — notify's recommended_watcher has a small
    // startup delay.
    tokio::time::sleep(Duration::from_millis(200)).await;

    std::fs::write(workspace.join("alpha.rs"), b"// new\n").unwrap();
    std::fs::write(workspace.join("bravo.rs"), b"// new\n").unwrap();

    // Wait past the debounce window for the callback to fire.
    tokio::time::sleep(zend::watcher::DEBOUNCE_WINDOW + Duration::from_millis(500))
        .await;

    assert!(
        counter.load(Ordering::SeqCst) >= 1,
        "callback should fire at least once after file creates"
    );
}

#[tokio::test]
async fn watcher_debounces_event_bursts_into_one_call() {
    let dir = tempfile::tempdir().expect("tempdir");
    let workspace = dir.path().to_path_buf();

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let cb: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    let _watcher = zend::watcher::spawn(&workspace, cb).expect("watcher started");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Fire a tight burst of 20 file creates over ~50 ms.
    for i in 0..20 {
        std::fs::write(workspace.join(format!("f_{i}.rs")), b"// new\n").unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    // Wait for the debounce to time out.
    tokio::time::sleep(zend::watcher::DEBOUNCE_WINDOW + Duration::from_millis(500))
        .await;

    let n = counter.load(Ordering::SeqCst);
    assert!(
        (1..=3).contains(&n),
        "burst of 20 creates should debounce to 1-3 callbacks, got {n}"
    );
}
