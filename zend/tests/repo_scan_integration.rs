//! Tier-2 integration test for the `repo_map` layer's per-directory ingest.
//!
//! Drives the real walk → unit → render pipeline against synthetic workspaces
//! and pushes each directory's chain through a [`RecordingTurnSink`], so the
//! turn shape the daemon will prefill is asserted end-to-end with no model load.
//! The engine-bound half (conversation minting, the summary decode, the resume
//! cache) is covered by the live daemon run; everything up to the turns is here.

use std::fs;
use std::path::Path;

use zend::repo_scan::render::{render_chain, CHAIN_TOOLS};
use zend::repo_scan::{build_units, walk_workspace, DirState, DirUnit};
use zend::turn_sink::{InsertTurnSink, RecordingTurnSink};
use zend_tools::ToolContext;

/// Same budget the daemon passes; irrelevant to a model-less sink but keeps the
/// call identical to the production one.
const SUMMARY_TOKENS: usize = 200;

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

fn small_workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    write(
        &root,
        "Cargo.toml",
        b"[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    );
    write(
        &root,
        "src/lib.rs",
        b"//! The demo crate.\n//! Says hello.\npub fn hello() {}\n",
    );
    write(&root, "src/handler.rs", b"pub fn handle() {}\n");
    write(&root, "README.md", b"# demo\n\nhello world\n");
    write(&root, ".gitignore", b"target/\n");
    write(&root, "target/should_be_skipped.rs", b"unreachable\n");
    dir
}

/// Walk + build the units the daemon would ingest.
fn units_of(root: &Path) -> Vec<DirUnit> {
    build_units(&walk_workspace(root), root)
}

/// Run every unit's chain through a recording sink, exactly as
/// `process_one_dir` runs it through the live one.
fn record(root: &Path) -> RecordingTurnSink {
    let ctx = ToolContext::with_workspace(root);
    let force: Vec<String> = CHAIN_TOOLS.iter().map(|t| t.to_string()).collect();
    let mut sink = RecordingTurnSink::new();
    for unit in units_of(root) {
        let (prefilled, decode_user) = render_chain(&ctx, &unit);
        sink.ingest_chain(
            &prefilled,
            &decode_user,
            vec!["repo_map".to_string(), unit.dir.clone()],
            SUMMARY_TOKENS,
            &force,
        )
        .unwrap();
    }
    sink
}

#[test]
fn one_unit_per_directory_holding_files() {
    let dir = small_workspace();
    let dirs: Vec<String> = units_of(dir.path()).into_iter().map(|u| u.dir).collect();
    // The root (Cargo.toml, README.md, .gitignore) and `src/`. `target/` is
    // gitignored, so it contributes no unit at all.
    assert_eq!(dirs, vec![".".to_string(), "src/".to_string()]);
}

/// One chain per folder: request → list → read → DECODE. It lists BEFORE it
/// reads — a `file_read` naming `lib.rs` before anything revealed the file
/// exists would teach the model to guess paths.
#[test]
fn each_directory_lists_before_it_reads() {
    let dir = small_workspace();
    let sink = record(dir.path());
    let src = sink
        .turns
        .iter()
        .filter(|(_, _, tags)| tags[1] == "src/")
        .collect::<Vec<_>>();
    assert_eq!(src.len(), 3, "request+list, listing+read, excerpt+summary");

    assert!(src[0].0.starts_with("Summarize the `src/` folder"));
    assert!(src[0].1.contains("\"name\":\"file_list\""));
    assert!(src[1].0.starts_with("<tool_response>{"), "the listing");
    assert!(src[1].1.contains("\"name\":\"file_read\""));
    assert!(src[2].0.contains("```rust"), "the anchor excerpt");
    assert!(src[2].1.is_empty(), "the folder summary is DECODED");
}

/// `src/lib.rs` carries a `//!` block, so the read is scoped to it rather than
/// pulling the whole file.
#[test]
fn the_anchor_excerpt_is_the_module_doc_block() {
    let dir = small_workspace();
    let sink = record(dir.path());
    let excerpt = sink
        .turns
        .iter()
        .find(|(u, _, tags)| tags[1] == "src/" && u.contains("```"))
        .map(|(u, _, _)| u.clone())
        .expect("the excerpt turn");
    assert!(
        excerpt.contains("src/lib.rs (lines 1-2 of 3):"),
        "{excerpt}"
    );
    assert!(excerpt.contains("//! The demo crate."));
    assert!(
        !excerpt.contains("pub fn hello"),
        "the code below the doc block is not the folder's description",
    );
}

/// The listing is produced by running the real `file_list`, so it names the
/// directory's own files and honours the walk's `.gitignore` exclusion.
#[test]
fn the_listing_names_the_directorys_files() {
    let dir = small_workspace();
    let sink = record(dir.path());
    let listings: String = sink
        .turns
        .iter()
        .map(|(u, _, _)| u.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(listings.contains("Cargo.toml"));
    assert!(listings.contains("README.md"));
    assert!(listings.contains("lib.rs"));
    assert!(listings.contains("handler.rs"));
    assert!(!listings.contains("should_be_skipped"));
}

/// Every turn carries `["repo_map", <dir>]` so a tag-scoped provenance gallery
/// admits exactly one folder's turns.
#[test]
fn every_turn_carries_the_layer_and_directory_tags() {
    let dir = small_workspace();
    for (_, _, tags) in &record(dir.path()).turns {
        assert_eq!(tags.len(), 2, "kind + dir tags: {tags:?}");
        assert_eq!(tags[0], "repo_map");
        assert!(!tags[1].is_empty(), "dir tag present");
    }
}

/// The resume cache and the refresh both key on rendering being deterministic.
#[test]
fn rendering_is_byte_identical_on_repeat() {
    let dir = small_workspace();
    assert_eq!(record(dir.path()).turns, record(dir.path()).turns);
}

/// A folder with no README / module root still gets a conversation — it just
/// summarises from the listing alone.
#[test]
fn a_directory_with_no_anchor_still_ingests() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "src/thing.rs", b"pub fn t() {}\n");
    let sink = record(dir.path());
    assert_eq!(sink.turns.len(), 2, "request+list, then listing+summary");
    assert!(sink.turns[0].1.contains("\"name\":\"file_list\""));
    assert!(sink.turns[1].0.starts_with("<tool_response>{"));
    assert!(sink.turns[1].1.is_empty(), "the summary is decoded");
}

// ── DirState: what re-ingests and what does not ──────────────────────────────

#[test]
fn state_is_stable_when_an_unshown_file_changes() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    // `handler.rs` is listed by name but its CONTENT is never shown, so the
    // folder's summary is still accurate and re-decoding it would cost for
    // nothing.
    write(dir.path(), "src/handler.rs", b"pub fn handle_v2() {}\n");
    assert!(before.equivalent_to(&units_of(dir.path())));
}

#[test]
fn state_moves_when_the_anchor_text_changes() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    // The module doc IS the summary's evidence — editing it must re-ingest.
    write(
        dir.path(),
        "src/lib.rs",
        b"//! The demo crate, rewritten.\n//! Now says goodbye.\npub fn hello() {}\n",
    );
    let after = units_of(dir.path());
    assert!(!before.equivalent_to(&after));
    assert_eq!(before.changed_dirs(&after), vec!["src/".to_string()]);
}

/// `file_list` matches a path PREFIX, so the root folder's listing spans the
/// whole tree: a file added under `src/` changes what BOTH folders show, and
/// both must re-ingest or one of them keeps a summary of a repo that moved on.
#[test]
fn state_moves_when_a_file_is_added() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    write(dir.path(), "src/new_module.rs", b"pub fn n() {}\n");
    let after = units_of(dir.path());
    assert!(!before.equivalent_to(&after));
    assert_eq!(
        before.changed_dirs(&after),
        vec![".".to_string(), "src/".to_string()],
    );
}

#[test]
fn a_removed_directory_is_reported_as_changed() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    fs::remove_dir_all(dir.path().join("src")).unwrap();
    let after = units_of(dir.path());
    // `src/` is gone entirely; the root's listing lost those files.
    assert_eq!(
        before.changed_dirs(&after),
        vec![".".to_string(), "src/".to_string()],
    );
}

#[test]
fn refresh_returns_no_op_outcome_variant() {
    // `refresh_repo_map` needs a live engine (it mints conversations), but the
    // outcome enum is the caller's contract — keep the symbol public.
    let _: zend::repo_scan::RefreshOutcome = zend::repo_scan::RefreshOutcome::NoOp;
}
