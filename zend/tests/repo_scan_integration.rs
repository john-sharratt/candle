//! Tier-2 integration test for the `repo_map` layer's per-directory ingest.
//!
//! Drives the real walk → unit → render pipeline against synthetic workspaces
//! and pushes each directory's chain through a [`RecordingTurnSink`], so the
//! turn shape the daemon will prefill is asserted end-to-end with no model load.
//! The engine-bound half (conversation minting, the summary decode, the resume
//! cache) is covered by the live daemon run; everything up to the turns is here.

use std::fs;
use std::path::Path;

use candle_conversation::models::Dialect;
use candle_conversation::stencil::ToolCallEnvelope;
use zend::repo_scan::render::render_chain;
use zend::repo_scan::{build_units, walk_workspace, DirState, DirUnit};
use zend::turn_sink::{InsertTurnSink, RecordingTurnSink};
use zend_tools::ToolContext;

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
    build_units(&walk_workspace(root, None))
}

/// Record every unit's SEED chain — the prefilled request/`file_list` pair and
/// the listing response whose assistant half `converse::run_folder_conversation`
/// decodes. The decode and any follow-up tool round it drives need a model, so
/// they are the live daemon's half; what is asserted here is the turn shape that
/// reaches the conversation before the first decode, which is where every
/// rendering defect lives.
fn record(root: &Path) -> RecordingTurnSink {
    let ctx = ToolContext::with_workspace(root);
    let mut sink = RecordingTurnSink::new();
    // ChatML's envelope, matching this suite's existing JSON-shaped
    // expectations; `render::tests::tool_calls_follow_the_dialects_call_style`
    // is what holds the other style.
    let env = ToolCallEnvelope::for_dialect(&Dialect::chat_ml());
    for unit in units_of(root) {
        let tags = vec!["repo_map".to_string(), unit.dir.clone()];
        let (prefilled, decode_user) = render_chain(&ctx, &unit, &env);
        for (user, assistant) in &prefilled {
            sink.insert_prefill_turn(user, assistant, tags.clone())
                .unwrap();
        }
        sink.insert_prefill_turn(&decode_user, "", tags).unwrap();
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

/// One chain per folder, and it is ONE pair: request → `file_list` /
/// listing → DECODED summary. A folder is described from names and paths alone,
/// so nothing in the chain reads a file — the listing is the whole evidence.
#[test]
fn each_directory_lists_once_and_then_summarises() {
    let dir = small_workspace();
    let sink = record(dir.path());
    let src = sink
        .turns
        .iter()
        .filter(|(_, _, tags)| tags[1] == "src/")
        .collect::<Vec<_>>();
    assert_eq!(src.len(), 2, "request+list, then listing+summary");

    assert!(src[0].0.starts_with("Summarize the `src/` folder"));
    assert!(src[0].1.contains("\"name\": \"file_list\""));
    assert!(src[1].0.starts_with("<tool_response>{"), "the listing");
    assert!(src[1].1.is_empty(), "the folder summary is DECODED");
}

/// No folder turn reads a file or carries file content. The read round-trip and
/// its fenced excerpt were removed: a folder is summarised from its listing, and
/// prefilling a `file_read` taught the model to read a file per folder.
#[test]
fn no_folder_turn_reads_a_file() {
    let dir = small_workspace();
    for (user, assistant, tags) in &record(dir.path()).turns {
        assert!(
            !assistant.contains("file_read"),
            "{tags:?} prefilled a file_read: {assistant}",
        );
        assert!(
            !user.contains("```"),
            "{tags:?} carries a file excerpt: {user}",
        );
    }
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
    assert!(sink.turns[0].1.contains("\"name\": \"file_list\""));
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

/// Rewriting `src/lib.rs`'s module doc does NOT re-ingest. It used to: the doc
/// block was the folder's anchor excerpt and part of the hash. The chain no
/// longer shows it, so the sealed summary still answers the request this unit
/// renders, and re-decoding would pay full cost for the same answer.
#[test]
fn state_is_stable_when_the_former_anchor_text_changes() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    write(
        dir.path(),
        "src/lib.rs",
        b"//! The demo crate, rewritten.\n//! Now says goodbye.\npub fn hello() {}\n",
    );
    assert!(before.equivalent_to(&units_of(dir.path())));
}

/// A unit's hash covers ONE level, because its turn shows one level: adding
/// `src/new_module.rs` moves `src/` and leaves the root alone, whose listing
/// still names `src/` and nothing else about it. Under the old subtree hash a
/// file three levels down moved every ancestor and re-decoded the whole spine.
#[test]
fn a_new_file_moves_only_its_own_directory() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    write(dir.path(), "src/new_module.rs", b"pub fn n() {}\n");
    let after = units_of(dir.path());
    assert!(!before.equivalent_to(&after));
    assert_eq!(before.changed_dirs(&after), vec!["src/".to_string()]);
}

/// A new SUBDIRECTORY does move its parent — the parent's listing names one
/// entry per subdirectory, so the entry is new content in the turn the parent
/// shows. This is the boundary of the one-level rule above.
#[test]
fn a_new_subdirectory_moves_its_parent() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    write(dir.path(), "src/inner/deep.rs", b"pub fn d() {}\n");
    let after = units_of(dir.path());
    assert!(before.changed_dirs(&after).contains(&"src/".to_string()));
}

/// The module hint is spliced into the request the model reads, so it is part of
/// what the turn shows and therefore part of the hash. A `[workspace]` table
/// added to the root manifest changes the question — `(Cargo workspace root)`
/// appears — without changing the listing, and the resume cache must not report
/// a hit on a summary that answered the older request.
#[test]
fn state_moves_when_the_module_hint_changes() {
    let dir = small_workspace();
    let before = DirState::from_units(&units_of(dir.path()));
    write(
        dir.path(),
        "Cargo.toml",
        b"[workspace]\nmembers = [\"a\", \"b\"]\n",
    );
    let after = units_of(dir.path());
    assert_eq!(before.changed_dirs(&after), vec![".".to_string()]);
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
