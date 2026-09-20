//! A `secrets/` directory is unreachable through the file tools.
//!
//! These assert the property the deployment's secrets actually rest on. The
//! `.gitignore` entry beside them keeps the files out of commits and does
//! nothing whatsoever about the model: a read resolves a normalised key straight
//! to a path under the workspace root and opens it, consulting no ignore rules
//! at any point. Before the guard existed, `secrets/tools.yaml` and
//! `web/secrets/auth.yaml` were both hidden from `file_list` and served in full
//! by `file_read`.
//!
//! The fixture deliberately writes **no `.gitignore`**, so nothing here can pass
//! by accident on the strength of the ignore rules — what is being tested is the
//! guard itself.

mod harness;

use serde_json::{json, Value};
use tempfile::TempDir;
use zend_tools::ToolContext;

/// The literal secrets. Any appearance of these in a tool response is the bug.
const TAVILY: &str = "tvly-TESTKEY-must-never-be-served";
const OAUTH: &str = "GOCSPX-testsecret-must-never-be-served";

/// A workspace shaped like the real one: the daemon's own secrets at
/// `secrets/tools.yaml`, the gateway's at `web/secrets/auth.yaml`, and ordinary
/// source beside them.
fn workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("secrets")).unwrap();
    std::fs::create_dir_all(root.join("web/secrets")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("secrets/tools.yaml"),
        format!("tavily_api_key: {TAVILY}\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("web/secrets/auth.yaml"),
        format!("client_secret: {OAUTH}\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.rs"),
        "fn main() { println!(\"hi\"); }\n",
    )
    .unwrap();
    dir
}

fn ctx(dir: &TempDir) -> ToolContext {
    ToolContext::with_workspace(dir.path())
}

/// Every rendering of a response, for leak assertions.
fn text(v: &Value) -> String {
    v.to_string()
}

#[test]
fn file_read_refuses_the_daemons_secrets() {
    let dir = workspace();
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "secrets/tools.yaml", "start_line": 1, "end_line": 200}),
        &ctx(&dir),
    );
    let detail = harness::expect_error(&resp, "forbidden");
    assert!(
        !text(&resp).contains(TAVILY),
        "the refusal must not quote the secret: {detail}"
    );
}

#[test]
fn file_read_refuses_the_gateways_secrets() {
    let dir = workspace();
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "web/secrets/auth.yaml", "start_line": 1, "end_line": 200}),
        &ctx(&dir),
    );
    harness::expect_error(&resp, "forbidden");
    assert!(!text(&resp).contains(OAUTH));
}

/// **Every spelling of the path is refused.** Normalisation runs before the
/// guard, so the mount prefix, a leading slash, backslashes and a `..` detour
/// all collapse onto the same key first. A guard applied to the raw string
/// instead would be bypassed by any one of these.
#[test]
fn no_spelling_of_the_path_gets_through() {
    let dir = workspace();
    let c = ctx(&dir);
    for path in [
        "secrets/tools.yaml",
        "/secrets/tools.yaml",
        "./secrets/tools.yaml",
        "workspace/secrets/tools.yaml",
        "/workspace/secrets/tools.yaml",
        "src/../secrets/tools.yaml",
        "../../secrets/tools.yaml",
        "secrets/../secrets/tools.yaml",
        r"secrets\tools.yaml",
        r"\workspace\secrets\tools.yaml",
    ] {
        let resp = harness::invoke_with_ctx(
            "file_read",
            json!({ "path": path, "start_line": 1, "end_line": 200 }),
            &c,
        );
        harness::expect_error(&resp, "forbidden");
        assert!(!text(&resp).contains(TAVILY), "leaked via {path:?}");
    }
}

/// The content search must not become the hole the read path was. This is the
/// one that would bite hardest: a `file_grep` that walked the workspace itself
/// would happily return the key as a matching line.
#[test]
fn file_grep_never_matches_inside_a_secrets_directory() {
    let dir = workspace();
    let c = ctx(&dir);
    for pattern in ["tvly", "TESTKEY", "GOCSPX", "client_secret", "api_key", "."] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_grep",
            json!({ "pattern": pattern }),
            &c,
        ));
        let body = text(&resp);
        assert!(!body.contains(TAVILY), "pattern {pattern:?} leaked the key");
        assert!(
            !body.contains(OAUTH),
            "pattern {pattern:?} leaked the secret"
        );
        assert!(
            !body.contains("secrets/"),
            "pattern {pattern:?} named a protected path: {body}"
        );
    }
}

/// Path search must not even reveal that the files are there.
#[test]
fn file_search_never_lists_a_protected_path() {
    let dir = workspace();
    let c = ctx(&dir);
    for query in ["tools.yaml", "auth.yaml", "secrets", "*.yaml"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_search",
            json!({ "query": query }),
            &c,
        ));
        assert!(
            !text(&resp).contains("secrets"),
            "query {query:?} surfaced a protected path: {resp}"
        );
    }
}

#[test]
fn file_list_never_lists_a_protected_path() {
    let dir = workspace();
    let c = ctx(&dir);
    for prefix in ["", "secrets", "secrets/", "web/", "web/secrets/"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &c,
        ));
        assert!(
            !text(&resp).contains("tools.yaml") && !text(&resp).contains("auth.yaml"),
            "prefix {prefix:?} surfaced a protected path: {resp}"
        );
    }
}

/// A session cannot plant a decoy at a protected path — which would otherwise
/// make later reads of that path start succeeding.
#[test]
fn writing_into_a_secrets_directory_is_refused() {
    let dir = workspace();
    let resp = harness::invoke_with_ctx(
        "write",
        json!({"path": "secrets/tools.yaml", "content": "tavily_api_key: mine\n"}),
        &ctx(&dir),
    );
    harness::expect_error(&resp, "forbidden");
}

/// The guard is narrow: ordinary files are untouched by it.
#[test]
fn ordinary_files_still_read_and_search() {
    let dir = workspace();
    let c = ctx(&dir);

    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 200}),
        &c,
    ));
    assert!(read.as_str().unwrap().contains("fn main()"));

    let found = harness::expect_success(harness::invoke_with_ctx(
        "file_search",
        json!({"query": "main.rs"}),
        &c,
    ));
    assert_eq!(found["files"][0], "src/main.rs");

    let hit = harness::expect_success(harness::invoke_with_ctx(
        "file_grep",
        json!({"pattern": "println"}),
        &c,
    ));
    assert_eq!(hit["matches"][0]["path"], "src/main.rs");
    assert_eq!(hit["matches"][0]["line"], 1);
}

/// A directory merely *named* like a secret elsewhere in a path is still
/// protected — the segment matches at any depth, which is what makes a new
/// `secrets/` directory safe the day it is created.
#[test]
fn the_segment_matches_at_any_depth() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("a/b/secrets/c")).unwrap();
    std::fs::write(dir.path().join("a/b/secrets/c/deep.txt"), "buried").unwrap();
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "a/b/secrets/c/deep.txt", "start_line": 1, "end_line": 200}),
        &ToolContext::with_workspace(dir.path()),
    );
    harness::expect_error(&resp, "forbidden");
}
