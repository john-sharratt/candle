//! Multi-repository behaviour of the `file_*` tools, through a real
//! two-repository workspace on disk.
//!
//! Each repository is its own root: the same path in `a` and `b` is a
//! different file, a write into one is invisible through the other and never
//! reaches disk, and `..` cannot cross from one repository into its sibling.

mod harness;

use serde_json::json;
use tempfile::TempDir;
use zend_tools::ToolContext;
use zend_vfs::{RepoSpec, Workspace};

/// A workspace with two repositories, `a` and `b`, each holding a file at the
/// same path (`shared.txt`) with different content, plus one file unique to
/// each.
fn two_repo_workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for (repo, unique, shared_body) in [
        ("a", "only_a.txt", "content from a"),
        ("b", "only_b.txt", "content from b"),
    ] {
        let repo_dir = root.join(repo);
        std::fs::create_dir_all(repo_dir.join("src")).unwrap();
        std::fs::write(repo_dir.join("shared.txt"), shared_body).unwrap();
        std::fs::write(repo_dir.join(unique), "unique").unwrap();
        std::fs::write(
            repo_dir.join("src/lib.rs"),
            format!("// {repo}\npub fn marker_{repo}() {{}}\n"),
        )
        .unwrap();
    }
    dir
}

fn ctx(dir: &TempDir) -> ToolContext {
    let ws = Workspace::new(dir.path(), vec![RepoSpec::named("a"), RepoSpec::named("b")]).unwrap();
    ToolContext::with_workspace(ws)
}

/// A grouped search result — `{"repos": [{"repo", <items>: [...]}]}` — as its
/// groups: each repository with its items, in order.
fn groups<'a>(
    resp: &'a serde_json::Value,
    items: &str,
) -> Vec<(&'a str, Vec<&'a serde_json::Value>)> {
    resp["repos"]
        .as_array()
        .expect("results grouped by repository")
        .iter()
        .map(|g| {
            (
                g["repo"].as_str().unwrap(),
                g[items].as_array().unwrap().iter().collect(),
            )
        })
        .collect()
}

fn excerpt_source(resp: &serde_json::Value) -> String {
    let text = resp.as_str().expect("file_read returns a rendered string");
    let body = text
        .split_once("```")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .and_then(|(_, rest)| rest.rsplit_once("```"))
        .map(|(body, _)| body)
        .unwrap_or("");
    body.lines()
        .map(|l| l.split_once("  ").map(|(_, t)| t).unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

// ── file_read ────────────────────────────────────────────────────────────────

/// The same path in two repositories is two different files, and the fence
/// names which repository was read.
#[test]
fn file_read_of_the_same_path_returns_each_repositorys_own_content() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);

    let a = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "a", "path": "shared.txt", "page": 0}),
        &c,
    ));
    assert_eq!(excerpt_source(&a), "content from a");
    assert!(a.as_str().unwrap().contains(" file=a/shared.txt "));

    let b = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "b", "path": "shared.txt", "page": 0}),
        &c,
    ));
    assert_eq!(excerpt_source(&b), "content from b");
    assert!(b.as_str().unwrap().contains(" file=b/shared.txt "));
}

#[test]
fn an_unknown_repo_names_every_repo_the_workspace_lists() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "c", "path": "shared.txt", "page": 0}),
        &c,
    );
    let detail = harness::expect_error(&resp, "unknown_repo");
    assert!(detail.contains("no repository named \"c\""), "{detail}");
    assert!(detail.contains("repo must be one of: a, b"), "{detail}");
}

/// `..` stops at the repository's own root — it cannot reach into a sibling
/// repository, even though both live directly under the same workspace folder
/// on disk.
#[test]
fn parent_traversal_cannot_reach_a_sibling_repository() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "a", "path": "../b/only_b.txt", "page": 0}),
        &c,
    );
    harness::expect_error(&resp, "not_found");
}

// ── file_list ────────────────────────────────────────────────────────────────

/// `file_list` lists inside one repository, its paths relative to it.
#[test]
fn file_list_lists_inside_one_repository() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": "b"}),
        &c,
    ));
    assert_eq!(resp["repo"], "b");
    let paths: Vec<&str> = resp["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["only_b.txt", "shared.txt", "src"]);
}

/// **The scope is always stated.** Every file tool requires `repo`; a call
/// that leaves it out is refused, not read as "everywhere".
#[test]
fn every_file_tool_refuses_a_call_without_a_repo() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    for (tool, args) in [
        ("file_list", json!({})),
        ("file_search", json!({"query": "shared.txt"})),
        ("file_grep", json!({"pattern": "marker"})),
        ("file_read", json!({"path": "shared.txt", "page": 0})),
    ] {
        let resp = harness::invoke_with_ctx(tool, args, &c);
        harness::expect_error(&resp, "invalid_arguments");
    }
}

/// `*` covers every repository, and only the searches take it: listing and
/// reading are raw operations on one repository's files, and are refused it
/// as they would be any repository the workspace does not list.
#[test]
fn only_the_search_tools_take_all_repos() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    for (tool, args) in [
        (
            "file_read",
            json!({"repo": "*", "path": "shared.txt", "page": 0}),
        ),
        ("file_list", json!({"repo": "*"})),
    ] {
        let resp = harness::invoke_with_ctx(tool, args, &c);
        let detail = harness::expect_error(&resp, "unknown_repo");
        assert!(detail.contains("a, b"), "{tool}: {detail}");
    }
}

// ── file_search / file_grep across repositories ─────────────────────────────

/// **Hits across every repository come back grouped by repository**, in the
/// workspace's order, each path relative to its group's repository.
#[test]
fn file_search_of_all_repos_groups_the_hits_by_repo() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "shared.txt", "repo": "*"}),
        &c,
    ));
    let got: Vec<(&str, Vec<&str>)> = groups(&resp, "files")
        .into_iter()
        .map(|(repo, files)| (repo, files.iter().map(|f| f.as_str().unwrap()).collect()))
        .collect();
    assert_eq!(got, [("a", vec!["shared.txt"]), ("b", vec!["shared.txt"])]);
}

#[test]
fn file_search_with_a_repo_finds_only_that_repos_hits() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"repo": "b", "query": "shared.txt"}),
        &c,
    ));
    let got = groups(&resp, "files");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, "b");
    assert_eq!(got[0].1, [&json!("shared.txt")]);
}

#[test]
fn file_grep_of_all_repos_groups_the_hits_by_repo() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "pub fn marker_", "repo": "*"}),
        &c,
    ));
    let got: Vec<(&str, Vec<(&str, u64)>)> = groups(&resp, "matches")
        .into_iter()
        .map(|(repo, hits)| {
            (
                repo,
                hits.iter()
                    .map(|m| (m["path"].as_str().unwrap(), m["line"].as_u64().unwrap()))
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("a", vec![("src/lib.rs", 2)]),
            ("b", vec![("src/lib.rs", 2)])
        ]
    );
}

/// **A repository listed later is not starved by one listed earlier.** `a`
/// holds far more matches than the whole call's ceiling; `b` holds one. With
/// one ceiling filled in manifest order, `a` used all of it and `b` was never
/// searched — measured live, a workspace-wide grep reported two of three
/// repositories as the only ones mentioning a word the third held 2,413 times.
/// Each repository gets its share, so both are represented and the result
/// says it was truncated.
#[test]
fn a_workspace_grep_represents_every_repository_that_matches() {
    let dir = two_repo_workspace();
    let flood: String = (0..60)
        .map(|i| format!("pub fn marker_flood_{i}() {{}}\n"))
        .collect();
    // 40 files at the 20-hits-per-file cap is 800 hits in `a` alone, past the
    // call's 600 — enough to starve `b` under one shared, first-come ceiling.
    for f in 0..40 {
        std::fs::write(dir.path().join("a").join(format!("flood_{f}.rs")), &flood).unwrap();
    }
    let c = ctx(&dir);
    let mut repos = std::collections::BTreeSet::new();
    let mut page = 0;
    let truncated = loop {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_grep",
            json!({"pattern": "pub fn marker_", "repo": "*", "page": page}),
            &c,
        ));
        for (repo, _) in groups(&resp, "matches") {
            repos.insert(repo.to_string());
        }
        match resp["paging"]["next_page"].as_u64() {
            Some(next) => page = next,
            None => break resp["truncated"].as_bool().unwrap_or(false),
        }
    };
    assert_eq!(
        repos,
        std::collections::BTreeSet::from(["a".to_string(), "b".to_string()]),
        "every matching repository appears"
    );
    assert!(
        truncated,
        "the flooded repository was cut, and the result says so"
    );
}

#[test]
fn file_grep_with_a_repo_finds_only_that_repos_hits() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"repo": "b", "pattern": "pub fn marker_"}),
        &c,
    ));
    let got = groups(&resp, "matches");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, "b");
    assert_eq!(got[0].1.len(), 1);
    assert_eq!(got[0].1[0]["path"], "src/lib.rs");
}

// ── Isolation ────────────────────────────────────────────────────────────────

/// A write into one repository is invisible through the other, and never
/// reaches disk — the overlay's whole premise, now over two repositories'
/// worth of disk.
#[test]
fn a_write_into_one_repo_is_not_visible_through_the_other_and_never_reaches_disk() {
    let dir = two_repo_workspace();
    let c = ctx(&dir);

    harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": "a", "path": "planted.txt", "content": "from a"}),
        &c,
    ));

    // Visible in `a`...
    let read_a = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "a", "path": "planted.txt", "page": 0}),
        &c,
    ));
    assert_eq!(excerpt_source(&read_a), "from a");

    // ...absent in `b`.
    let read_b = harness::invoke_with_ctx(
        "file_read",
        json!({"repo": "b", "path": "planted.txt", "page": 0}),
        &c,
    );
    harness::expect_error(&read_b, "not_found");

    // ...and never written to disk in either repository.
    assert!(!dir.path().join("a/planted.txt").exists());
    assert!(!dir.path().join("b/planted.txt").exists());
}
