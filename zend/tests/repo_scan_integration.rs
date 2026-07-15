//! Tier-2 integration test for the `repo_map` layer's directory
//! cluster ingestion.
//!
//! Drives `ingest_repo_map_into_sink` against synthetic workspaces
//! and asserts the produced cluster-shaped `(user, assistant)` turn
//! pairs have the shape the daemon expects.  No model load — the
//! sink is a `RecordingTurnSink` that captures every call into
//! memory.

use std::fs;
use std::path::Path;

use zend::loading::LoadProgress;
use zend::repo_scan::ingest_repo_map_into_sink;
use zend::turn_sink::RecordingTurnSink;

fn fixture(name: &str) -> tempfile::TempDir {
    let _ = name;
    tempfile::tempdir().expect("tempdir")
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

fn small_workspace() -> tempfile::TempDir {
    let dir = fixture("phase1");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "Cargo.toml",
        b"[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    );
    write(&root, "src/lib.rs", b"// lib root\npub fn hello() {}\n");
    write(&root, "src/handler.rs", b"pub fn handle() {}\n");
    write(&root, "README.md", b"# demo\n\nhello world\n");
    write(&root, ".gitignore", b"target/\n");
    write(&root, "target/should_be_skipped.rs", b"unreachable\n");
    dir
}

fn large_workspace() -> tempfile::TempDir {
    let dir = fixture("phase1_big");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "Cargo.toml",
        b"[package]\nname = \"big\"\nversion = \"0.1.0\"\n",
    );
    for sub in &["alpha", "bravo", "charlie", "delta"] {
        for i in 0..40 {
            let body = format!("// {sub} {i}\npub fn item_{i}() {{}}\n");
            write(&root, &format!("src/{sub}/file_{i:03}.rs"), body.as_bytes());
        }
    }
    dir
}

#[test]
fn repo_scan_emits_one_turn_pair_per_cluster() {
    let dir = small_workspace();
    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let _ = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();
    // Small workspace collapses to one cluster.
    assert_eq!(sink.turns.len(), 1);
    assert!(sink.turns[0].0.starts_with("Repository index — "));
}

#[test]
fn repo_scan_large_workspace_emits_multiple_clusters() {
    let dir = large_workspace();
    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let _ = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();
    assert!(
        sink.turns.len() >= 4,
        "expected ≥4 cluster turn pairs, got {}",
        sink.turns.len()
    );
    let prompts: Vec<&str> = sink.turns.iter().map(|(u, _)| u.as_str()).collect();
    for sub in &["src/alpha", "src/bravo", "src/charlie", "src/delta"] {
        assert!(
            prompts.iter().any(|p| p.contains(sub)),
            "expected prompt mentioning {sub} in {prompts:?}"
        );
    }
}

#[test]
fn repo_scan_assistant_text_lists_actual_files() {
    let dir = small_workspace();
    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let _ = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();

    let combined_listings: String = sink
        .turns
        .iter()
        .map(|(_, a)| a.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(combined_listings.contains("Cargo.toml"));
    assert!(combined_listings.contains("lib.rs"));
    assert!(combined_listings.contains("handler.rs"));
    assert!(combined_listings.contains("README.md"));
    // .gitignore exclusion holds across the cluster path.
    assert!(!combined_listings.contains("should_be_skipped"));
}

#[test]
fn repo_scan_reports_progress_at_completion() {
    let dir = small_workspace();
    let mut sink = RecordingTurnSink::new();
    let progress = LoadProgress::new();
    let _ = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();
    let snap = progress.snapshot().expect("still loading");
    assert!(snap.progress >= 0.999);
}

#[test]
fn repo_scan_is_byte_identical_on_repeat() {
    let dir = small_workspace();
    let progress = LoadProgress::new();

    let mut sink_a = RecordingTurnSink::new();
    ingest_repo_map_into_sink(&mut sink_a, dir.path(), &progress).unwrap();

    let mut sink_b = RecordingTurnSink::new();
    ingest_repo_map_into_sink(&mut sink_b, dir.path(), &progress).unwrap();

    assert_eq!(sink_a.turns, sink_b.turns);
}

#[test]
fn cluster_state_is_stable_when_file_contents_change() {
    let dir = small_workspace();
    let progress = LoadProgress::new();
    let mut sink_a = RecordingTurnSink::new();
    let (_, state_a) = ingest_repo_map_into_sink(&mut sink_a, dir.path(), &progress).unwrap();

    // Touch a file's CONTENT — no rename, no add, no delete.  The
    // hash is over file names, so the cluster state must not move.
    write(
        dir.path(),
        "src/lib.rs",
        b"// edited but same name\npub fn hello() {}\npub fn extra() {}\n",
    );

    let mut sink_b = RecordingTurnSink::new();
    let (_, state_b) = ingest_repo_map_into_sink(&mut sink_b, dir.path(), &progress).unwrap();
    assert_eq!(state_a, state_b, "file-content edit must not move the hash");
}

#[test]
fn cluster_state_moves_when_a_file_is_added() {
    let dir = small_workspace();
    let progress = LoadProgress::new();
    let mut sink_a = RecordingTurnSink::new();
    let (_, state_a) = ingest_repo_map_into_sink(&mut sink_a, dir.path(), &progress).unwrap();

    write(dir.path(), "src/new_module.rs", b"pub fn n() {}\n");

    let mut sink_b = RecordingTurnSink::new();
    let (_, state_b) = ingest_repo_map_into_sink(&mut sink_b, dir.path(), &progress).unwrap();
    assert_ne!(state_a, state_b);
}

// ── refresh_repo_map shape tests (no live engine) ────────────────────────────
//
// `refresh_repo_map` needs a real `Sequence`, but its inner logic is
// pure: equivalent_to / changed_dirs on `ClusterState`.  We exercise
// those primitives directly here so the no-op short-circuit and the
// changed-dirs reporting are covered without a model.

#[test]
fn cluster_state_equivalent_short_circuit_is_pure() {
    let dir = small_workspace();
    let progress = LoadProgress::new();
    let mut sink = RecordingTurnSink::new();
    let (_, state) = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();

    // Re-cluster against the unchanged workspace via the same path
    // refresh uses internally.
    let map = zend::repo_scan::walk_workspace(dir.path());
    let clusters = zend::repo_scan::build_clusters(&map);
    assert!(state.equivalent_to(&clusters));
}

#[test]
fn cluster_state_changed_dirs_after_rename() {
    let dir = small_workspace();
    let progress = LoadProgress::new();
    let mut sink = RecordingTurnSink::new();
    let (_, state_before) = ingest_repo_map_into_sink(&mut sink, dir.path(), &progress).unwrap();

    // Rename README.md -> CHANGELOG.md.
    let root = dir.path().to_path_buf();
    fs::rename(root.join("README.md"), root.join("CHANGELOG.md")).unwrap();

    let map = zend::repo_scan::walk_workspace(dir.path());
    let clusters = zend::repo_scan::build_clusters(&map);
    assert!(!state_before.equivalent_to(&clusters));
    let changed = state_before.changed_dirs(&clusters);
    assert!(!changed.is_empty(), "rename must surface in changed_dirs");
}

#[test]
fn refresh_returns_no_op_outcome_variant() {
    // The atomic-refresh helper returns `RefreshOutcome::{NoOp,
    // Replaced}` — we can't actually invoke it without a model
    // (Replaced mints a real Sequence), but constructing a `NoOp`
    // variant proves the symbol stays public and the enum exists.
    let _: zend::repo_scan::RefreshOutcome = zend::repo_scan::RefreshOutcome::NoOp;
}
