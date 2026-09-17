//! `file_search` (find by name) and `file_grep` (find by content).

mod harness;

use serde_json::json;
use tempfile::TempDir;
use zend_tools::ToolContext;

/// A small tree with names and contents worth searching for.
fn workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for d in ["src", "src/tools", "docs", "vendor/copy/src/tools"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    std::fs::write(
        root.join("src/main.rs"),
        "fn main() {\n    let key = lookup();\n}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/tools/web_search.rs"),
        "pub fn run() {\n    // TODO: cache\n    let API_KEY = 1;\n}\n",
    )
    .unwrap();
    std::fs::write(root.join("src/tools/mod.rs"), "pub mod web_search;\n").unwrap();
    // A same-named file buried deeper: the shortest path must win the ordering.
    std::fs::write(root.join("vendor/copy/src/tools/mod.rs"), "// copy\n").unwrap();
    std::fs::write(root.join("docs/guide.md"), "# Guide\nsearch the web\n").unwrap();
    std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    dir
}

fn ctx(dir: &TempDir) -> ToolContext {
    ToolContext::with_workspace(dir.path())
}

// ── file_search ──────────────────────────────────────────────────────────────

#[test]
fn a_name_is_found_anywhere_in_the_tree() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "web_search"}),
        &ctx(&dir),
    ));
    assert_eq!(r["files"][0], "src/tools/web_search.rs");
    assert_eq!(r["paging"]["total"], 1);
}

/// Matching is case-insensitive over the whole path, so a caller who types a
/// name in the wrong case still finds it.
#[test]
fn the_query_is_case_insensitive() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "WEB_Search.RS"}),
        &ctx(&dir),
    ));
    assert_eq!(r["files"][0], "src/tools/web_search.rs");
}

/// **The shortest path wins.** A vendored copy of `mod.rs` must not outrank the
/// real one, because the shortest path bearing a name is nearly always the
/// definition rather than a copy of it.
#[test]
fn the_shallowest_match_sorts_first() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "mod.rs"}),
        &ctx(&dir),
    ));
    assert_eq!(r["files"][0], "src/tools/mod.rs");
    assert_eq!(r["files"][1], "vendor/copy/src/tools/mod.rs");
}

#[test]
fn a_glob_matches_by_extension() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "*.toml"}),
        &ctx(&dir),
    ));
    assert_eq!(r["files"][0], "Cargo.toml");
    assert_eq!(r["paging"]["total"], 1);
}

#[test]
fn a_prefix_narrows_the_search() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "mod.rs", "prefix": "vendor/"}),
        &ctx(&dir),
    ));
    assert_eq!(r["paging"]["total"], 1);
    assert_eq!(r["files"][0], "vendor/copy/src/tools/mod.rs");
}

/// An unmatched query is an empty result, not an error — a model must be able
/// to read "not present" off a successful call.
#[test]
fn an_unmatched_query_is_empty_not_an_error() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "nothing_like_this_exists"}),
        &ctx(&dir),
    ));
    assert_eq!(r["paging"]["total"], 0);
    assert_eq!(r["files"].as_array().unwrap().len(), 0);
}

/// A file this session wrote is findable alongside the workspace's own.
#[test]
fn a_session_file_is_searchable() {
    let dir = workspace();
    let c = ctx(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "notes/scratch.md", "content": "hello"}),
        &c,
    ));
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "scratch"}),
        &c,
    ));
    assert_eq!(r["files"][0], "notes/scratch.md");
}

// ── file_grep ────────────────────────────────────────────────────────────────

#[test]
fn a_literal_is_found_with_its_line_number() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "TODO"}),
        &ctx(&dir),
    ));
    assert_eq!(r["matches"][0]["path"], "src/tools/web_search.rs");
    assert_eq!(r["matches"][0]["line"], 2);
    assert_eq!(r["matches"][0]["text"], "    // TODO: cache");
}

/// The line number is the one `file_read` wants as `start_line`, so the pair
/// composes without the model doing arithmetic.
#[test]
fn the_line_number_feeds_file_read() {
    let dir = workspace();
    let c = ctx(&dir);
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "API_KEY"}),
        &c,
    ));
    let path = r["matches"][0]["path"].as_str().unwrap().to_string();
    let line = r["matches"][0]["line"].as_u64().unwrap();
    assert_eq!(line, 3);

    let excerpt = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": path, "start_line": line, "end_line": line}),
        &c,
    ));
    assert!(excerpt.as_str().unwrap().contains("API_KEY"));
}

#[test]
fn a_regex_alternation_matches_either_branch() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "^fn main|^pub fn run"}),
        &ctx(&dir),
    ));
    assert_eq!(r["paging"]["total"], 2);
}

#[test]
fn case_sensitivity_is_the_default_and_can_be_turned_off() {
    let dir = workspace();
    let c = ctx(&dir);

    let sensitive = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "api_key"}),
        &c,
    ));
    assert_eq!(sensitive["paging"]["total"], 0);

    let insensitive = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "api_key", "ignore_case": true}),
        &c,
    ));
    assert_eq!(insensitive["paging"]["total"], 1);
}

/// `files_searched` is what makes an empty result unambiguous: it separates
/// "searched the tree and it is absent" from "the prefix matched nothing".
#[test]
fn an_empty_result_reports_how_much_was_searched() {
    let dir = workspace();
    let c = ctx(&dir);

    let searched = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "definitely_not_present"}),
        &c,
    ));
    assert_eq!(searched["paging"]["total"], 0);
    assert!(searched["files_searched"].as_u64().unwrap() >= 5);

    let nothing_to_search = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "fn", "prefix": "no/such/dir/"}),
        &c,
    ));
    assert_eq!(nothing_to_search["files_searched"], 0);
}

#[test]
fn a_bad_pattern_is_rejected_with_its_reason() {
    let dir = workspace();
    let resp = harness::invoke_with_ctx("file_grep", json!({"pattern": "unclosed(["}), &ctx(&dir));
    let detail = harness::expect_error(&resp, "invalid_arguments");
    assert!(
        detail.contains("regex"),
        "the detail should say what was wrong: {detail}"
    );
}

/// A session's own edit is what gets searched for that path, not the file on
/// disk — the overlay's whole premise.
#[test]
fn a_session_edit_shadows_the_workspace_copy() {
    let dir = workspace();
    let c = ctx(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "src/main.rs", "content": "fn main() { unique_marker(); }\n"}),
        &c,
    ));
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "unique_marker"}),
        &c,
    ));
    assert_eq!(r["matches"][0]["path"], "src/main.rs");
    assert_eq!(r["matches"][0]["modified"], true);

    // The overwritten line is gone from the search, because the session's copy
    // is what the path now resolves to.
    let gone = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "lookup"}),
        &c,
    ));
    assert_eq!(gone["paging"]["total"], 0);
}
