//! Tier-2 integration test for the `repo_map` layer's per-directory ingest.
//!
//! Drives the real branch walk → unit → render pipeline against synthetic
//! repositories and pushes each directory's seed chain through a
//! [`RecordingTurnSink`], so the turn shape the daemon will prefill is asserted
//! end-to-end with no model load. The engine-bound half (conversation minting,
//! the summary decode, the content keys landing) is covered by the live daemon
//! run; everything up to the turns is here.
//!
//! Every workspace here holds one git repository, [`REPO`], so every key and
//! tag is workspace-relative (`demo/src/`).

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use candle_conversation::models::Dialect;
use candle_conversation::stencil::ToolCallEnvelope;
use zend::branch_ingest::filter::IngestScope;
use zend::branch_ingest::walk::{walk, RepoBranches, TreeCache, UnitItem};
use zend::repo_scan::render::render_chain;
use zend::repo_scan::DirUnit;
use zend::turn_sink::{InsertTurnSink, RecordingTurnSink};
use zend_tools::ToolContext;
use zend_vfs::{Repo, RepoSpec, Workspace};

/// The one repository each test workspace holds.
const REPO: &str = "demo";

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.hooksPath=", "-c", "core.autocrlf=false"])
        .args(["-c", "user.name=T", "-c", "user.email=t@x"])
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(REPO).join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

/// Commit everything in the repository on `main`.
fn commit(root: &Path, message: &str) {
    let dir = root.join(REPO);
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-q", "--allow-empty", "-m", message]);
}

fn workspace_of(root: &Path) -> Workspace {
    Workspace::new(root, vec![RepoSpec::named(REPO)]).unwrap()
}

/// A repository on `main`, its first commit holding `files`.
fn repository(files: &[(&str, &[u8])]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = dir.path().join(REPO);
    fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    for (rel, body) in files {
        write(dir.path(), rel, body);
    }
    commit(dir.path(), "first");
    dir
}

fn small_workspace() -> tempfile::TempDir {
    repository(&[
        (
            "Cargo.toml",
            b"[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        ),
        (
            "src/lib.rs",
            b"//! The demo crate.\n//! Says hello.\npub fn hello() {}\n",
        ),
        ("src/handler.rs", b"pub fn handle() {}\n"),
        ("README.md", b"# demo\n\nhello world\n"),
        (".gitignore", b"target/\n"),
        ("target/should_be_skipped.rs", b"unreachable\n"),
    ])
}

/// The folder units every branch lists, as the daemon's pass walks them.
fn units_of(root: &Path) -> Vec<UnitItem> {
    let branches = RepoBranches::read(REPO, Repo::open(&root.join(REPO)).unwrap()).unwrap();
    walk(
        std::slice::from_ref(&branches),
        &IngestScope::new("", None),
        &mut TreeCache::default(),
    )
    .0
    .units
}

/// Each folder's key, by folder.
fn keys_of(root: &Path) -> Vec<(String, String)> {
    units_of(root)
        .into_iter()
        .map(|u| (u.unit.dir, u.unit.key))
        .collect()
}

/// Record every unit's SEED chain — the prefilled request/`file_list` pair and
/// the listing response whose assistant half `converse::run_folder_conversation`
/// decodes — exactly as `process_one_dir` renders it: tools reading the commit
/// the unit was found on, the hint the walk read from its manifest. The decode and any
/// follow-up tool round it drives need a model, so they are the live daemon's
/// half; what is asserted here is the turn shape that reaches the conversation
/// before the first decode, which is where every rendering defect lives.
fn record(root: &Path) -> RecordingTurnSink {
    let ctx = ToolContext::with_workspace(workspace_of(root));
    let mut sink = RecordingTurnSink::new();
    // ChatML's envelope, matching this suite's existing JSON-shaped
    // expectations; `render::tests::tool_calls_follow_the_dialects_call_style`
    // is what holds the other style.
    let env = ToolCallEnvelope::for_dialect(&Dialect::chat_ml());
    for item in units_of(root) {
        let files = ctx
            .files
            .fresh_at(REPO, &item.at)
            .expect("a git repository");
        let tools = ctx.with_files(Arc::new(files));
        let unit = DirUnit::of(&item.unit);
        let tags = vec!["repo_map".to_string(), unit.dir.clone()];
        let (prefilled, decode_user) = render_chain(&tools, &unit, &env);
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
    let dirs: Vec<String> = units_of(dir.path())
        .into_iter()
        .map(|u| u.unit.dir)
        .collect();
    // The repository's root (Cargo.toml, README.md) and its `src/` — no unit
    // for the workspace itself, which no `file_list` lists. `target/` is
    // ignored, so it was never committed and contributes no unit; `.gitignore`
    // is hidden.
    assert_eq!(dirs, vec!["demo/".to_string(), "demo/src/".to_string()]);
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
        .filter(|(_, _, tags)| tags[1] == "demo/src/")
        .collect::<Vec<_>>();
    assert_eq!(src.len(), 2, "request+list, then listing+summary");

    assert!(src[0]
        .0
        .starts_with("Summarize the `src/` folder in the `demo` repository"));
    assert!(src[0].1.contains(
        "{\"name\": \"file_list\", \"arguments\": {\"repo\": \"demo\", \"path\": \"src\"}}"
    ));
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

/// The repository's root carries its manifest's hint, read from the commit.
#[test]
fn the_manifest_hint_is_read_from_the_commit() {
    let dir = small_workspace();
    let sink = record(dir.path());
    let root = sink
        .turns
        .iter()
        .find(|(_, _, tags)| tags[1] == "demo/")
        .expect("the repository's root");
    assert!(
        root.0
            .starts_with("Summarize the `demo` repository (crate: demo)"),
        "{}",
        root.0
    );
}

/// **No folder call names `*`**: `file_list` works inside one repository, so
/// every chain — the repository's root included — names its repository.
#[test]
fn every_folder_call_names_its_repository() {
    let dir = small_workspace();
    for (_, assistant, tags) in &record(dir.path()).turns {
        if assistant.is_empty() {
            continue;
        }
        assert!(
            assistant.contains("\"repo\": \"demo\""),
            "{tags:?}: {assistant}"
        );
        assert!(!assistant.contains("\"*\""), "{tags:?}: {assistant}");
    }
}

/// The listing is produced by running the real `file_list` at the unit's
/// commit, so it names the directory's own committed files — never what the
/// folder holds uncommitted.
#[test]
fn the_listing_names_the_directorys_committed_files() {
    let dir = small_workspace();
    write(dir.path(), "src/uncommitted.rs", b"pub fn u() {}\n");
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
    assert!(!listings.contains("uncommitted.rs"));
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

/// A unit's key names what its turns show, so its rendering must be
/// deterministic.
#[test]
fn rendering_is_byte_identical_on_repeat() {
    let dir = small_workspace();
    assert_eq!(record(dir.path()).turns, record(dir.path()).turns);
}

/// A folder with no README / module root gets exactly the same chain as one
/// with — the listing is all any folder shows.
#[test]
fn a_directory_with_no_readme_ingests_the_same_way() {
    let dir = repository(&[("src/thing.rs", b"pub fn t() {}\n")]);
    let sink = record(dir.path());
    let src: Vec<_> = sink
        .turns
        .iter()
        .filter(|(_, _, tags)| tags[1] == "demo/src/")
        .collect();
    assert_eq!(src.len(), 2, "request+list, then listing+summary");
    assert!(src[0].1.contains("\"name\": \"file_list\""));
    assert!(src[1].0.starts_with("<tool_response>{"));
    assert!(src[1].1.is_empty(), "the summary is decoded");
}

// ── Keys: what re-ingests and what does not ──────────────────────────────────

/// The folders whose key differs between two walks.
fn moved(before: &[(String, String)], after: &[(String, String)]) -> Vec<String> {
    let mut out: Vec<String> = after
        .iter()
        .filter(|a| !before.contains(a))
        .map(|(dir, _)| dir.clone())
        .collect();
    for (dir, _) in before {
        if !after.iter().any(|(d, _)| d == dir) {
            out.push(dir.clone());
        }
    }
    out
}

#[test]
fn a_folder_key_is_stable_when_an_unshown_file_changes() {
    let dir = small_workspace();
    let before = keys_of(dir.path());
    // `handler.rs` is listed by name but its CONTENT is never shown, so the
    // folder's summary is still accurate and re-decoding it would cost for
    // nothing.
    write(dir.path(), "src/handler.rs", b"pub fn handle_v2() {}\n");
    commit(dir.path(), "handler");
    assert_eq!(keys_of(dir.path()), before);
}

/// Rewriting `src/lib.rs`'s module doc does NOT re-ingest. The doc block used
/// to be the folder's anchor excerpt and part of its key; the chain no longer
/// shows it, so the sealed summary still answers the request this unit
/// renders, and re-decoding would pay full cost for the same answer.
#[test]
fn a_folder_key_is_stable_when_the_former_anchor_text_changes() {
    let dir = small_workspace();
    let before = keys_of(dir.path());
    write(
        dir.path(),
        "src/lib.rs",
        b"//! The demo crate, rewritten.\n//! Now says goodbye.\npub fn hello() {}\n",
    );
    commit(dir.path(), "module doc");
    assert_eq!(keys_of(dir.path()), before);
}

/// `file_list` is one level deep, so a folder's listing is only its own direct
/// entries — a file added under `src/` moves only `src/`'s key. Neither the
/// repository's root nor the workspace root ever showed `src/`'s files, so
/// they have nothing to re-ingest.
#[test]
fn a_folder_key_moves_when_a_file_is_added() {
    let dir = small_workspace();
    let before = keys_of(dir.path());
    write(dir.path(), "src/new_module.rs", b"pub fn n() {}\n");
    commit(dir.path(), "added");
    assert_eq!(moved(&before, &keys_of(dir.path())), ["demo/src/"]);
}

/// The module hint is spliced into the request the model reads, so it is part
/// of what the turn shows and therefore part of the key. A `[workspace]` table
/// replacing the root manifest's package changes the question — `(Cargo
/// workspace root)` appears — without changing the listing, and the plan must
/// not treat the summary that answered the older request as current.
#[test]
fn a_folder_key_moves_when_the_module_hint_changes() {
    let dir = small_workspace();
    let before = keys_of(dir.path());
    write(
        dir.path(),
        "Cargo.toml",
        b"[workspace]\nmembers = [\"a\", \"b\"]\n",
    );
    commit(dir.path(), "workspace");
    assert_eq!(moved(&before, &keys_of(dir.path())), ["demo/"]);
}

/// **Branches sharing a lineage share their units.** A `topic` branch whose
/// only change is a `Cargo.toml` version bump lists every folder the way
/// `main` does and asks the same question of it — the hint is `(crate: demo)`
/// either way — so each folder is ONE unit that both branches hold, and every
/// file they share is one reading; only the manifest itself, whose bytes
/// differ, is read twice. Keyed on the manifest's bytes the repository's root
/// was two conversations saying the same thing, and that is what multiplied
/// `candle/` across its branches.
#[test]
fn branches_that_share_a_lineage_share_their_units() {
    let dir = small_workspace();
    let repo = dir.path().join(REPO);
    git(&repo, &["checkout", "-q", "-b", "topic"]);
    write(
        dir.path(),
        "Cargo.toml",
        b"[package]\nname = \"demo\"\nversion = \"0.2.0\"\n",
    );
    commit(dir.path(), "bump");
    git(&repo, &["checkout", "-q", "main"]);

    let branches = RepoBranches::read(REPO, Repo::open(&repo).unwrap()).unwrap();
    let (corpus, failed) = walk(
        std::slice::from_ref(&branches),
        &IngestScope::new("", None),
        &mut TreeCache::default(),
    );
    assert!(failed.is_empty());

    let folders: Vec<(&str, Vec<&str>)> = corpus
        .units
        .iter()
        .map(|u| {
            (
                u.unit.dir.as_str(),
                u.branches.iter().map(String::as_str).collect(),
            )
        })
        .collect();
    assert_eq!(
        folders,
        [
            ("demo/", vec!["main", "topic"]),
            ("demo/src/", vec!["main", "topic"]),
        ],
        "one unit per folder, held by both branches",
    );

    let readings = |path: &str| -> Vec<Vec<&str>> {
        corpus
            .files
            .iter()
            .filter(|f| f.file.path == path)
            .map(|f| f.branches.iter().map(String::as_str).collect())
            .collect()
    };
    assert_eq!(
        readings("demo/src/lib.rs"),
        [vec!["main", "topic"]],
        "an unchanged file is one reading on both branches"
    );
    assert_eq!(
        readings("demo/Cargo.toml"),
        [vec!["main"], vec!["topic"]],
        "the manifest's bytes differ, so it is read once per version"
    );
}

#[test]
fn a_removed_directory_leaves_the_corpus() {
    let dir = small_workspace();
    let before = keys_of(dir.path());
    fs::remove_dir_all(dir.path().join(REPO).join("src")).unwrap();
    commit(dir.path(), "removed");
    // `src/`'s unit vanishes entirely, and the repository's root — whose
    // listing showed `src/` as a folder — moves with it. The workspace root
    // listed only the repository, so it stays.
    assert_eq!(moved(&before, &keys_of(dir.path())), ["demo/", "demo/src/"]);
}
