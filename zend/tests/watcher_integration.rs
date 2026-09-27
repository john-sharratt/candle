//! Tier-2 integration test for the workspace watcher.
//!
//! Spins up the watcher against a temp workspace of one repository, performs
//! file operations, and asserts the debounced refresh callback fires (or
//! doesn't) per the operation's relevance to the file-name set.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use zend_vfs::{RepoSpec, Workspace};

/// A workspace in `dir` holding one repository, returning the workspace and
/// the repository's folder.
fn workspace_in(dir: &tempfile::TempDir) -> (Workspace, PathBuf) {
    let repo = dir.path().join("r");
    std::fs::create_dir_all(&repo).unwrap();
    let ws = Workspace::new(dir.path(), vec![RepoSpec::named("r")]).unwrap();
    (ws, repo)
}

fn counting() -> (Arc<AtomicUsize>, Arc<dyn Fn() + Send + Sync>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let cb: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });
    (counter, cb)
}

#[tokio::test]
async fn watcher_fires_callback_on_file_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (workspace, repo) = workspace_in(&dir);
    std::fs::write(repo.join("seed.rs"), b"// seed\n").unwrap();

    let (counter, cb) = counting();
    let on_uploads: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
    let _watcher =
        zend::watcher::spawn(&workspace, &[], None, cb, on_uploads).expect("watcher started");
    // Allow the watcher's background task a moment to arm before we
    // start writing — notify's recommended_watcher has a small
    // startup delay.
    tokio::time::sleep(Duration::from_millis(200)).await;

    std::fs::write(repo.join("alpha.rs"), b"// new\n").unwrap();
    std::fs::write(repo.join("bravo.rs"), b"// new\n").unwrap();

    // Wait past the debounce window for the callback to fire.
    tokio::time::sleep(zend::watcher::DEBOUNCE_WINDOW + Duration::from_millis(500)).await;

    assert!(
        counter.load(Ordering::SeqCst) >= 1,
        "callback should fire at least once after file creates"
    );
}

/// **Only the repositories are watched.** A file written beside them in the
/// workspace folder — another checkout, a stray note — moves nothing any walk
/// reads, so it wakes nothing.
#[tokio::test]
async fn a_file_outside_every_repository_fires_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (workspace, _repo) = workspace_in(&dir);

    let (counter, cb) = counting();
    let on_uploads: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
    let _watcher =
        zend::watcher::spawn(&workspace, &[], None, cb, on_uploads).expect("watcher started");
    tokio::time::sleep(Duration::from_millis(200)).await;

    std::fs::create_dir_all(dir.path().join("other")).unwrap();
    std::fs::write(dir.path().join("other").join("x.rs"), b"// elsewhere\n").unwrap();
    std::fs::write(dir.path().join("notes.md"), b"beside\n").unwrap();
    tokio::time::sleep(zend::watcher::DEBOUNCE_WINDOW + Duration::from_millis(500)).await;

    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn watcher_debounces_event_bursts_into_one_call() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (workspace, repo) = workspace_in(&dir);

    let (counter, cb) = counting();
    let on_uploads: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
    let _watcher =
        zend::watcher::spawn(&workspace, &[], None, cb, on_uploads).expect("watcher started");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Fire a tight burst of 20 file creates over ~50 ms.
    for i in 0..20 {
        std::fs::write(repo.join(format!("f_{i}.rs")), b"// new\n").unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    // Wait for the debounce to time out.
    tokio::time::sleep(zend::watcher::DEBOUNCE_WINDOW + Duration::from_millis(500)).await;

    let n = counter.load(Ordering::SeqCst);
    assert!(
        (1..=3).contains(&n),
        "burst of 20 creates should debounce to 1-3 callbacks, got {n}"
    );
}
