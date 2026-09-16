mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

/// Source lines out of a rendered `file_read` excerpt — header, fence and
/// `cat -n` numbering stripped. The format itself is pinned in `file_overlay.rs`.
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

#[test]
fn file_write_unicode() {
    let ctx = ctx();
    let content = "こんにちは 🌍 — Unicode test";
    harness::invoke_with_ctx(
        "write",
        json!({"path": "uni.txt", "content": content}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "uni.txt", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&resp), content);
}

#[test]
fn file_edit_not_found() {
    let resp = harness::invoke(
        "file_edit",
        json!({"path": "nonexistent.txt", "patch": "@@ -1 +1 @@\n-x\n+y\n"}),
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn file_list_prefix() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "alpha/a.txt", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"path": "alpha/b.txt", "content": "2"}),
        &ctx,
    );
    harness::invoke_with_ctx("write", json!({"path": "beta/c.txt", "content": "3"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "alpha/"}),
        &ctx,
    ));
    let files = resp["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    for f in files {
        assert!(f["path"].as_str().unwrap().starts_with("alpha/"));
    }
}

#[test]
fn file_delete_idempotent() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "idem.txt", "content": "x"}), &ctx);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "idem.txt"}),
        &ctx,
    ));
    let r2 = harness::invoke_with_ctx("file_delete", json!({"path": "idem.txt"}), &ctx);
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_write_overwrite_created_false() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "ow.txt", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "ow.txt", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "ow.txt", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&rd), "v2");
}

#[test]
fn file_edit_round_trip() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "rt.txt", "content": "hello world"}),
        &ctx,
    );
    harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "rt.txt", "patch": "@@ -1 +1 @@\n-hello world\n+hello Rust\n"}),
        &ctx,
    ));
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "rt.txt", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&rd), "hello Rust");
}

#[test]
fn file_write_read_roundtrip() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "hello.txt", "content": "hello world"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "hello.txt", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&resp), "hello world");
}

#[test]
fn file_write_creates_vs_overwrites() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "a.txt", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "a.txt", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
}

#[test]
fn file_read_not_found() {
    let resp = harness::invoke(
        "file_read",
        json!({"path": "nosuchfile.txt", "start_line": 1, "end_line": 200}),
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn file_edit_success() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "edit.txt", "content": "foo bar baz"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({
            "path": "edit.txt",
            "patch": "@@ -1 +1 @@\n-foo bar baz\n+foo qux baz\n"
        }),
        &ctx,
    ));
    assert_eq!(resp["bytes"], 11);
    assert_eq!(resp["hunks_applied"], 1);
    assert_eq!(resp["hunks_already_applied"], 0);
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "edit.txt", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), "foo qux baz");
}

#[test]
fn file_edit_ambiguous() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "dup.txt", "content": "aa\nbb\naa\n"}),
        &ctx,
    );
    // `aa` stands one line either side of the hunk's stated position, so the
    // line numbers cannot pick between them.
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({
            "path": "dup.txt",
            "patch": "@@ -2 +2 @@\n-aa\n+xx\n"
        }),
        &ctx,
    );
    harness::expect_error(&resp, "ambiguous");
}

/// The tool end to end over a real workspace: the patch lands in the session,
/// the file on disk is untouched, and sending the same patch a second time is a
/// no-op that reports the hunk as already applied instead of applying it again.
#[test]
fn file_edit_patches_through_the_overlay_and_leaves_disk_untouched() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    let on_disk = "fn main() {\n    let retries = 3;\n    run(retries);\n}\n";
    std::fs::write(dir.path().join("src/main.rs"), on_disk).unwrap();
    let ctx = ToolContext::with_workspace(dir.path());

    let patch = "@@ -1,3 +1,3 @@\n fn main() {\n-    let retries = 3;\n+    let retries = 30;\n     run(retries);\n";
    let first = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "src/main.rs", "patch": patch}),
        &ctx,
    ));
    assert_eq!(first["path"], "src/main.rs");
    assert_eq!(first["hunks_applied"], 1);
    assert_eq!(first["hunks_already_applied"], 0);
    assert_eq!(first["bytes"], 54);

    let patched = "fn main() {\n    let retries = 30;\n    run(retries);\n}\n";
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs"}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), patched.trim_end());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("src/main.rs")).unwrap(),
        on_disk,
        "the file on disk must be byte-for-byte what it was",
    );

    // The same patch again: the addition contains the removal, so a
    // substring-replacing edit would leave `retries = 300` here.
    let second = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "src/main.rs", "patch": patch}),
        &ctx,
    ));
    assert_eq!(second["hunks_applied"], 0);
    assert_eq!(second["hunks_already_applied"], 1);
    assert_eq!(second["bytes"], 54);
    let reread = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs"}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&reread), patched.trim_end());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("src/main.rs")).unwrap(),
        on_disk,
    );
}

/// A patch that is entirely already applied writes nothing at all — a workspace
/// file must not be copied up on account of an edit that did not happen.
#[test]
fn file_edit_already_applied_does_not_copy_the_file_up() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.toml"), "[net]\nport = 9090\n").unwrap();
    let ctx = ToolContext::with_workspace(dir.path());

    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({
            "path": "config.toml",
            "patch": "@@ -1,2 +1,2 @@\n [net]\n-port = 8080\n+port = 9090\n"
        }),
        &ctx,
    ));
    assert_eq!(resp["hunks_applied"], 0);
    assert_eq!(resp["hunks_already_applied"], 1);

    let listed = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "config.toml"}),
        &ctx,
    ));
    assert!(
        listed["files"][0].get("modified").is_none(),
        "nothing was written, so nothing was copied up",
    );
}

/// One failing hunk abandons the whole patch: the hunk that would have applied
/// leaves no trace, and the error names the one that failed.
#[test]
fn file_edit_writes_nothing_when_one_hunk_fails() {
    let ctx = ctx();
    let before = "alpha\nbeta\ngamma\ndelta\n";
    harness::invoke_with_ctx("write", json!({"path": "all.txt", "content": before}), &ctx);
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({
            "path": "all.txt",
            "patch": "@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n@@ -3,2 +3,2 @@\n gamma\n-absent\n+new\n"
        }),
        &ctx,
    );
    let detail = harness::expect_error(&resp, "not_found");
    assert!(
        detail.starts_with("hunk 2 (@@ -3,2 +3,2 @@) does not apply"),
        "{detail}",
    );
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "all.txt"}),
        &ctx,
    ));
    assert_eq!(
        excerpt_source(&read),
        before.trim_end(),
        "the first hunk must not have landed",
    );
}

/// A patch that is not a diff is rejected as bad arguments, not read as content.
#[test]
fn file_edit_rejects_a_patch_that_is_not_a_diff() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "p.txt", "content": "a\n"}), &ctx);
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "p.txt", "patch": "just change a into b"}),
        &ctx,
    );
    harness::expect_error(&resp, "invalid_arguments");
}

#[test]
fn file_list() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "a/b.txt", "content": "1"}), &ctx);
    harness::invoke_with_ctx("write", json!({"path": "a/c.txt", "content": "2"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "a/"}),
        &ctx,
    ));
    let files = resp["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
}

/// A context that has written nothing lists nothing, whatever the prefix — the
/// VFS is scratch space, never seeded from the real filesystem. This is the
/// exact response a model gets when it reaches for `file_list` expecting a
/// project directory listing: `{"files":[],"total_bytes":0}`.
#[test]
fn file_list_is_empty_until_something_is_written() {
    let ctx = ctx();
    for prefix in ["", "/", "/workspace/", "/workspace/src", "candle-examples/"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        assert_eq!(
            resp["files"].as_array().unwrap().len(),
            0,
            "prefix {prefix:?} listed files from an unwritten VFS",
        );
        assert_eq!(resp["total_bytes"], 0);
    }
}

/// `/` normalizes to the empty prefix, so it lists the whole VFS rather than
/// erroring or resolving to a real filesystem root.
#[test]
fn file_list_root_prefix_lists_everything() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "a.txt", "content": "1"}), &ctx);
    harness::invoke_with_ctx(
        "write",
        json!({"path": "nested/deep/b.txt", "content": "22"}),
        &ctx,
    );
    for prefix in ["/", ""] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        assert_eq!(
            resp["files"].as_array().unwrap().len(),
            2,
            "prefix {prefix:?}"
        );
        assert_eq!(resp["total_bytes"], 3);
    }
}

/// `/workspace` is the mount point of the working directory, so it normalises
/// away: a path written as `/workspace/src/main.rs` and one written as
/// `src/main.rs` are the same entry, and either spelling of the prefix selects
/// it. This is what lets the tool definitions' `/workspace/...` examples address
/// the same files as the bare repo-relative paths a model infers from a repo map.
#[test]
fn workspace_mount_prefix_normalises_to_the_same_entry() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"path": "/workspace/src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"path": "README.md", "content": "# hi\n"}),
        &ctx,
    );
    for prefix in ["/workspace/src", "src/", "/src"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        let files = resp["files"].as_array().unwrap();
        assert_eq!(files.len(), 1, "prefix {prefix:?}");
        assert_eq!(files[0]["path"], "src/main.rs");
        assert_eq!(files[0]["lines"], 1);
    }
    // The same file resolves under either spelling.
    let bare = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 1}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&bare), "fn main() {}");
}

#[test]
fn file_delete() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "del.txt", "content": "bye"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "del.txt"}),
        &ctx,
    ));
    assert_eq!(resp["deleted"], true);
    let r2 = harness::invoke_with_ctx("file_delete", json!({"path": "del.txt"}), &ctx);
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_present_found_and_missing() {
    let ctx = ctx();
    harness::invoke_with_ctx("write", json!({"path": "p.txt", "content": "hi"}), &ctx);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_present",
        json!({
            "paths": ["p.txt", "missing.txt"]
        }),
        &ctx,
    ));
    let presented = resp["presented"].as_array().unwrap();
    assert_eq!(presented.len(), 1);
    let missing = resp["missing"].as_array().unwrap();
    assert_eq!(missing.len(), 1);
}

#[test]
fn file_present_all_missing() {
    let resp = harness::invoke("file_present", json!({"paths": ["nope.txt"]}));
    harness::expect_error(&resp, "no_files_found");
}
