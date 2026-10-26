//! `file_search` (find by name) and `file_grep` (find by content).

mod harness;

use doc_units::render::{pages, PAGE_LINES};
use doc_units::Doc;
use serde_json::json;
use tempfile::TempDir;
use zend_tools::ToolContext;

use harness::REPO;

/// A small tree with names and contents worth searching for, under the single
/// repository [`REPO`]'s folder.
fn workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
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
    harness::workspace_ctx(dir.path())
}

// ── file_search ──────────────────────────────────────────────────────────────

#[test]
fn a_name_is_found_anywhere_in_the_tree() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"web_search"}),
        &ctx(&dir),
    ));
    assert_eq!(r["repos"][0]["repo"], REPO);
    assert_eq!(r["repos"][0]["files"][0], "src/tools/web_search.rs");
    assert_eq!(r["paging"]["total"], 1);
}

/// Matching is case-insensitive over the whole path, so a caller who types a
/// name in the wrong case still finds it.
#[test]
fn the_query_is_case_insensitive() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"WEB_Search.RS"}),
        &ctx(&dir),
    ));
    assert_eq!(r["repos"][0]["files"][0], "src/tools/web_search.rs");
}

/// **The shortest path wins.** A vendored copy of `mod.rs` must not outrank the
/// real one, because the shortest path bearing a name is nearly always the
/// definition rather than a copy of it.
#[test]
fn the_shallowest_match_sorts_first() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"mod.rs"}),
        &ctx(&dir),
    ));
    assert_eq!(r["repos"][0]["files"][0], "src/tools/mod.rs");
    assert_eq!(r["repos"][0]["files"][1], "vendor/copy/src/tools/mod.rs");
}

#[test]
fn a_glob_matches_by_extension() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"*.toml"}),
        &ctx(&dir),
    ));
    assert_eq!(r["repos"][0]["files"][0], "Cargo.toml");
    assert_eq!(r["paging"]["total"], 1);
}

#[test]
fn a_prefix_narrows_the_search() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"mod.rs", "prefix": "vendor/"}),
        &ctx(&dir),
    ));
    assert_eq!(r["paging"]["total"], 1);
    assert_eq!(r["repos"][0]["files"][0], "vendor/copy/src/tools/mod.rs");
}

/// An unmatched query is an empty result, not an error — a model must be able
/// to read "not present" off a successful call.
#[test]
fn an_unmatched_query_is_empty_not_an_error() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"nothing_like_this_exists"}),
        &ctx(&dir),
    ));
    assert_eq!(r["paging"]["total"], 0);
    assert_eq!(r["repos"].as_array().unwrap().len(), 0);
}

/// A file this session wrote is findable alongside the workspace's own.
#[test]
fn a_session_file_is_searchable() {
    let dir = workspace();
    let c = ctx(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "notes/scratch.md", "content": "hello"}),
        &c,
    ));
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": REPO, "page": 1, "query":"scratch"}),
        &c,
    ));
    assert_eq!(r["repos"][0]["files"][0], "notes/scratch.md");
}

// ── file_grep ────────────────────────────────────────────────────────────────

#[test]
fn a_literal_is_found_with_its_line_number() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"TODO"}),
        &ctx(&dir),
    ));
    assert_eq!(r["repos"][0]["repo"], REPO);
    assert_eq!(
        r["repos"][0]["matches"][0]["path"],
        "src/tools/web_search.rs"
    );
    assert_eq!(r["repos"][0]["matches"][0]["line"], 2);
    assert_eq!(r["repos"][0]["matches"][0]["text"], "    // TODO: cache");
}

/// A hit names the part it is in under its anchor, and the `file_read` page
/// that holds it — so the pair composes without the model losing the hit, and
/// the part can be edited straight from the hit.
#[test]
fn a_hit_names_its_part_and_page() {
    let dir = workspace();
    let c = ctx(&dir);
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"API_KEY"}),
        &c,
    ));
    let hit = &r["repos"][0]["matches"][0];
    let path = hit["path"].as_str().unwrap().to_string();
    assert_eq!(hit["line"], 3);
    let text = "pub fn run() {\n    // TODO: cache\n    let API_KEY = 1;\n}\n";
    let a = &Doc::new(&path, text).anchors()[0];
    assert_eq!(hit["part"], format!("{a}| fn run"));
    assert_eq!(hit["page"], 1);

    let page = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": path, "page": hit["page"]}),
        &c,
    ));
    assert!(page.as_str().unwrap().contains(&format!("{a}| fn run\n")));

    // The hit's anchor edits the part with no read in between.
    harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": path, "at": a, "do": "replace",
               "old": "API_KEY = 1", "text": "API_KEY = 2"}),
        &c,
    ));
}

/// Pages hold whole parts, so a hit deep in a long file names the page its
/// part is on, not one derived from its line number.
#[test]
fn a_hit_in_a_long_file_names_the_page_its_part_is_on() {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
    std::fs::create_dir_all(&root).unwrap();
    let text: String = (0..120)
        .map(|i| format!("fn f{i}() {{\n    let _ = {i};\n}}\n\n"))
        .collect();
    std::fs::write(root.join("long.rs"), &text).unwrap();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"fn f99\\("}),
        &harness::workspace_ctx(dir.path()),
    ));
    let hit = &r["repos"][0]["matches"][0];
    let doc = Doc::new("long.rs", &text);
    let i = doc
        .units()
        .iter()
        .position(|u| u.label.as_deref() == Some("fn f99"))
        .unwrap();
    let page = pages(&doc, PAGE_LINES)
        .iter()
        .position(|p| p.contains(&i))
        .unwrap()
        + 1;
    assert_eq!(hit["page"], page);
    assert!(page > 1, "a page past the first: {page}");
    assert_eq!(hit["part"], format!("{}| fn f99", doc.units()[i].anchor));
}

#[test]
fn a_regex_alternation_matches_either_branch() {
    let dir = workspace();
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"^fn main|^pub fn run"}),
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
        json!({"repo": REPO, "page": 1, "pattern":"api_key"}),
        &c,
    ));
    assert_eq!(sensitive["paging"]["total"], 0);

    let insensitive = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"api_key", "ignore_case": true}),
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
        json!({"repo": REPO, "page": 1, "pattern":"definitely_not_present"}),
        &c,
    ));
    assert_eq!(searched["paging"]["total"], 0);
    assert!(searched["files_searched"].as_u64().unwrap() >= 5);

    let nothing_to_search = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"fn", "prefix": "no/such/dir/"}),
        &c,
    ));
    assert_eq!(nothing_to_search["files_searched"], 0);
}

#[test]
fn a_bad_pattern_is_rejected_with_its_reason() {
    let dir = workspace();
    let resp = harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"unclosed(["}),
        &ctx(&dir),
    );
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
        json!({"repo": REPO, "path": "src/main.rs", "content": "fn main() { unique_marker(); }\n"}),
        &c,
    ));
    let r = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"unique_marker"}),
        &c,
    ));
    assert_eq!(r["repos"][0]["matches"][0]["path"], "src/main.rs");
    assert_eq!(r["repos"][0]["matches"][0]["modified"], true);

    // The overwritten line is gone from the search, because the session's copy
    // is what the path now resolves to.
    let gone = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": REPO, "page": 1, "pattern":"lookup"}),
        &c,
    ));
    assert_eq!(gone["paging"]["total"], 0);
}
