mod harness;

use serde_json::json;
use zend_tools::ToolContext;

use harness::REPO;

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
        json!({"repo": REPO, "path": "uni.txt", "content": content}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "uni.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&resp), content);
}

#[test]
fn file_edit_not_found() {
    let resp = harness::invoke(
        "file_edit",
        json!({"repo": REPO, "path": "nonexistent.txt", "old_text": "x", "new_text": "y"}),
    );
    let detail = harness::expect_error(&resp, "not_found");
    // The refusal names the way to create the file, not only the fault.
    assert!(detail.contains("nonexistent.txt"), "{detail}");
    assert!(detail.contains("call `write`"), "{detail}");
}

/// A URL is not a file: the refusal names `web_fetch` rather than reporting a
/// missing path the model would retry under other spellings.
#[test]
fn file_read_of_a_url_points_to_web_fetch() {
    let resp = harness::invoke(
        "file_read",
        json!({"repo": REPO, "path": "https://docs.rs/serde/latest/serde/", "page": 0}),
    );
    let detail = harness::expect_error(&resp, "invalid_arguments");
    assert!(
        detail.contains("https://docs.rs/serde/latest/serde/"),
        "{detail}"
    );
    assert!(detail.contains("`web_fetch`"), "{detail}");
}

#[test]
fn file_list_within_a_directory() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "alpha/a.txt", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "alpha/b.txt", "content": "2"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "beta/c.txt", "content": "3"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "alpha"}),
        &ctx,
    ));
    let entries = resp["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    for f in entries {
        assert!(f["path"].as_str().unwrap().starts_with("alpha/"));
    }
}

#[test]
fn file_delete_idempotent() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "idem.txt", "content": "x"}),
        &ctx,
    );
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"repo": REPO, "path": "idem.txt"}),
        &ctx,
    ));
    let r2 = harness::invoke_with_ctx(
        "file_delete",
        json!({"repo": REPO, "path": "idem.txt"}),
        &ctx,
    );
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_write_overwrite_created_false() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "ow.txt", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "ow.txt", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "ow.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&rd), "v2");
}

#[test]
fn file_edit_round_trip() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "rt.txt", "content": "hello world"}),
        &ctx,
    );
    harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "rt.txt", "old_text": "hello world", "new_text": "hello Rust"}),
        &ctx,
    ));
    let rd = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "rt.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&rd), "hello Rust");
}

#[test]
fn file_write_read_roundtrip() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "hello.txt", "content": "hello world"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "hello.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&resp), "hello world");
}

#[test]
fn file_write_creates_vs_overwrites() {
    let ctx = ctx();
    let r1 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "a.txt", "content": "v1"}),
        &ctx,
    ));
    assert_eq!(r1["created"], true);
    let r2 = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "a.txt", "content": "v2"}),
        &ctx,
    ));
    assert_eq!(r2["created"], false);
}

#[test]
fn file_read_not_found() {
    let resp = harness::invoke(
        "file_read",
        json!({"repo": REPO, "path": "nosuchfile.txt", "page": 0}),
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn file_edit_success() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "edit.txt", "content": "foo bar baz"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "edit.txt", "old_text": "bar", "new_text": "qux"}),
        &ctx,
    ));
    assert_eq!(resp["bytes"], 11);
    assert_eq!(resp["replacements"], 1);
    assert_eq!(resp["already_applied"], false);
    assert_eq!(resp["matched"], "exact");
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "edit.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), "foo qux baz");
}

#[test]
fn file_edit_ambiguous() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "dup.txt", "content": "aa\nbb\naa\n"}),
        &ctx,
    );
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "dup.txt", "old_text": "aa", "new_text": "xx"}),
        &ctx,
    );
    harness::expect_error(&resp, "ambiguous");
    // Asked for every one, both change.
    let all = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "dup.txt", "old_text": "aa", "new_text": "xx",
               "replace_all": true}),
        &ctx,
    ));
    assert_eq!(all["replacements"], 2);
}

/// **Text quoted at the wrong indentation is found, and the new text lands at
/// the file's** — the mistake a live model made against a two-space file,
/// quoting and writing at four.
#[test]
fn file_edit_finds_text_quoted_at_the_wrong_indentation() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "inv.js", "content":
               "class Inventory {\n  list() {\n    return [];\n  }\n}\n"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "inv.js",
               "old_text": "    list() {\n      return [];\n    }\n",
               "new_text": "    list() {\n      return [];\n    }\n\n    size() {\n      return 0;\n    }\n"}),
        &ctx,
    ));
    assert_eq!(resp["matched"], "indentation");
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "inv.js", "page": 0}),
        &ctx,
    ));
    assert_eq!(
        excerpt_source(&read),
        "class Inventory {\n  list() {\n    return [];\n  }\n\n  size() {\n    return 0;\n  }\n}"
    );
}

/// The tool end to end over a real workspace: the edit lands in the session,
/// the file on disk is untouched, and sending the same edit a second time is a
/// no-op that reports it already applied instead of applying it again.
#[test]
fn file_edit_replaces_through_the_overlay_and_leaves_disk_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let repo_dir = harness::repo_root(dir.path());
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let on_disk = "fn main() {\n    let retries = 3;\n    run(retries);\n}\n";
    std::fs::write(repo_dir.join("src/main.rs"), on_disk).unwrap();
    let ctx = harness::workspace_ctx(dir.path());

    let edit = json!({"repo": REPO, "path": "src/main.rs",
                      "old_text": "let retries = 3;", "new_text": "let retries = 30;"});
    let first = harness::expect_success(harness::invoke_with_ctx("file_edit", edit.clone(), &ctx));
    assert_eq!(first["repo"], REPO);
    assert_eq!(first["path"], "src/main.rs");
    assert_eq!(first["replacements"], 1);
    assert_eq!(first["already_applied"], false);
    assert_eq!(first["bytes"], 54);

    let patched = "fn main() {\n    let retries = 30;\n    run(retries);\n}\n";
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "src/main.rs", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), patched.trim_end());
    assert_eq!(
        std::fs::read_to_string(repo_dir.join("src/main.rs")).unwrap(),
        on_disk,
        "the file on disk must be byte-for-byte what it was",
    );

    // The same edit again: the new text contains the old, so a naive
    // substring replacement would leave `retries = 300` here.
    let second = harness::expect_success(harness::invoke_with_ctx("file_edit", edit, &ctx));
    assert_eq!(second["replacements"], 0);
    assert_eq!(second["already_applied"], true);
    assert_eq!(second["bytes"], 54);
    let reread = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "src/main.rs", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&reread), patched.trim_end());
    assert_eq!(
        std::fs::read_to_string(repo_dir.join("src/main.rs")).unwrap(),
        on_disk,
    );
}

/// **`file_edit` records the lines it changed, not the file**, under every
/// grant set. Two edits to a 200-line workspace file: the overlay holds two
/// edits of one splice each and the disk copy under it is untouched.
#[test]
fn file_edit_records_only_the_changed_lines() {
    use zend_tools::grants::Grants;
    use zend_vfs::FileDelta;

    let original: String = (1..=200).map(|i| format!("line {i}\n")).collect();
    let expected = original
        .replace("line 10\n", "line ten\n")
        .replace("line 190\n", "line 190\nline 190½\n");

    let dir = tempfile::tempdir().unwrap();
    let overlay_repo = harness::repo_root(dir.path());
    std::fs::create_dir_all(&overlay_repo).unwrap();
    std::fs::write(overlay_repo.join("big.txt"), &original).unwrap();
    // Every capability granted: a file edit still never reaches the disk.
    let overlay = harness::workspace_ctx(dir.path()).granting(Grants::ALL);
    for (old, new) in [
        ("line 9\nline 10\n", "line 9\nline ten\n"),
        ("line 190\n", "line 190\nline 190½\n"),
    ] {
        let out = harness::expect_success(harness::invoke_with_ctx(
            "file_edit",
            json!({"repo": REPO, "path": "big.txt", "old_text": old, "new_text": new}),
            &overlay,
        ));
        assert_eq!(out["replacements"], 1);
    }

    let store = overlay.files.repo(REPO).unwrap();
    assert_eq!(
        store.read("big.txt").unwrap().as_deref(),
        Some(expected.as_str())
    );
    let chain = store.deltas("big.txt").unwrap();
    for timed in &chain {
        let FileDelta::Edit { splices } = &timed.delta else {
            panic!("edits, not copies of the file: {chain:?}");
        };
        assert_eq!(splices.len(), 1, "{splices:?}");
    }
    assert_eq!(chain.len(), 2);
    // The insertion carries the next unchanged line as its anchor, removed and
    // re-inserted, so it cannot land anywhere but between 190 and 191.
    let first = "line 10\n".len() + "line ten\n".len();
    let second = "line 191\n".len() + "line 190½\nline 191\n".len();
    assert_eq!(store.total_bytes(), first + second);
    assert_eq!(
        std::fs::read_to_string(overlay_repo.join("big.txt")).unwrap(),
        original,
        "the overlay's disk is untouched"
    );
}

/// An edit already applied writes nothing at all — a workspace file must not
/// be copied up on account of an edit that did not happen.
#[test]
fn file_edit_already_applied_does_not_copy_the_file_up() {
    let dir = tempfile::tempdir().unwrap();
    let repo_dir = harness::repo_root(dir.path());
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("config.toml"), "[net]\nport = 9090\n").unwrap();
    let ctx = harness::workspace_ctx(dir.path());

    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "config.toml",
               "old_text": "port = 8080", "new_text": "port = 9090"}),
        &ctx,
    ));
    assert_eq!(resp["replacements"], 0);
    assert_eq!(resp["already_applied"], true);

    let listed = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO}),
        &ctx,
    ));
    let entry = listed["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["path"] == "config.toml")
        .expect("config.toml is listed");
    assert!(
        entry.get("modified").is_none(),
        "nothing was written, so nothing was copied up",
    );
}

/// Text that is not in the file writes nothing, and the refusal says where
/// its first line is, so the next call can quote the file as it stands.
#[test]
fn file_edit_writes_nothing_when_the_text_is_not_there() {
    let ctx = ctx();
    let before = "alpha\nbeta\ngamma\ndelta\n";
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "all.txt", "content": before}),
        &ctx,
    );
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "all.txt",
               "old_text": "beta\nabsent\n", "new_text": "BETA\nnew\n"}),
        &ctx,
    );
    let detail = harness::expect_error(&resp, "not_found");
    assert!(
        detail.starts_with("`old_text` is not in the file"),
        "{detail}"
    );
    assert!(detail.contains("line 2 of the file"), "{detail}");
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "all.txt", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), before.trim_end());
}

/// An edit that changes nothing is rejected as bad arguments.
#[test]
fn file_edit_rejects_an_edit_that_changes_nothing() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "p.txt", "content": "a\n"}),
        &ctx,
    );
    for (old, new) in [("  ", "b"), ("a", "a")] {
        let resp = harness::invoke_with_ctx(
            "file_edit",
            json!({"repo": REPO, "path": "p.txt", "old_text": old, "new_text": new}),
            &ctx,
        );
        harness::expect_error(&resp, "invalid_arguments");
    }
}

#[test]
fn file_list() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "a/b.txt", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "a/c.txt", "content": "2"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "a"}),
        &ctx,
    ));
    let entries = resp["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
}

/// The root always resolves, even from a context that has written nothing —
/// the VFS is scratch space, never seeded from the real filesystem — so an
/// empty session's root listing is `{"entries":[],"total_bytes":0}` rather
/// than an error.
#[test]
fn file_list_root_is_empty_until_something_is_written() {
    let ctx = ctx();
    for path in ["", "/"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path }),
            &ctx,
        ));
        assert_eq!(
            resp["entries"].as_array().unwrap().len(),
            0,
            "path {path:?} listed entries from an unwritten VFS",
        );
        assert_eq!(resp["total_bytes"], 0);
    }
}

/// A directory nothing has ever written into does not resolve — `file_list`
/// names a real directory to list, not a prefix that happens to match nothing.
#[test]
fn file_list_of_an_unwritten_directory_is_not_found() {
    let ctx = ctx();
    for path in ["src", "candle-examples/"] {
        let resp =
            harness::invoke_with_ctx("file_list", json!({ "repo": REPO, "path": path }), &ctx);
        harness::expect_error(&resp, "not_found");
    }
}

/// `/` normalizes to the empty path, so it lists the repository root rather
/// than erroring or resolving to a real filesystem root. One level deep: the
/// root-level file lists directly, the nested file's directory collapses to
/// its own entry rather than reaching all the way down to the leaf.
#[test]
fn file_list_root_path_lists_one_level() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "a.txt", "content": "1"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "nested/deep/b.txt", "content": "22"}),
        &ctx,
    );
    for path in ["/", ""] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path }),
            &ctx,
        ));
        let entries = resp["entries"].as_array().unwrap();
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e["path"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.txt", "nested"], "path {path:?}");
        assert_eq!(
            entries[1]["dir"], true,
            "the nested file's directory, not the leaf itself"
        );
        assert_eq!(resp["total_bytes"], 3);
    }
}

/// A leading slash and a `.` segment both normalise to the same entry as the
/// bare relative path — plain path normalisation, not a special mount alias
/// (no segment is special; see `VfsStore::normalize`).
#[test]
fn leading_slash_and_dot_segments_normalise_to_the_same_entry() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "README.md", "content": "# hi\n"}),
        &ctx,
    );
    for path in ["src", "src/", "/src", "./src", "src/../src"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path }),
            &ctx,
        ));
        let entries = resp["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1, "path {path:?}");
        assert_eq!(entries[0]["path"], "src/main.rs");
    }
    // The same file resolves under either spelling.
    let bare = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "src/main.rs", "page": 0}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&bare), "fn main() {}");
}

#[test]
fn file_delete() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "del.txt", "content": "bye"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"repo": REPO, "path": "del.txt"}),
        &ctx,
    ));
    assert_eq!(resp["deleted"], true);
    let r2 = harness::invoke_with_ctx(
        "file_delete",
        json!({"repo": REPO, "path": "del.txt"}),
        &ctx,
    );
    harness::expect_error(&r2, "not_found");
}

#[test]
fn file_present_found_and_missing() {
    let ctx = ctx();
    harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "p.txt", "content": "hi"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_present",
        json!({
            "repo": REPO,
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
    let resp = harness::invoke("file_present", json!({"repo": REPO, "paths": ["nope.txt"]}));
    harness::expect_error(&resp, "no_files_found");
}
