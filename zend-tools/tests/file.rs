mod harness;

use doc_units::Doc;
use serde_json::json;
use zend_tools::ToolContext;

use harness::REPO;

fn ctx() -> ToolContext {
    ToolContext::new()
}

/// The parts a rendered `file_read` page shows, between its fences. The
/// format itself is pinned in `file_overlay.rs` and `render.rs`.
fn page_body(resp: &serde_json::Value) -> String {
    let text = resp.as_str().expect("file_read returns a rendered string");
    text.split_once("```")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .and_then(|(_, rest)| rest.rsplit_once("```"))
        .map(|(body, _)| body.to_string())
        .unwrap_or_default()
}

/// A file's text as the session's store holds it.
fn text_of(ctx: &ToolContext, path: &str) -> String {
    ctx.files.repo(REPO).unwrap().read(path).unwrap().unwrap()
}

/// The anchor of the first part of `path` whose label or first line contains
/// `needle`.
fn anchor_of(ctx: &ToolContext, path: &str, needle: &str) -> String {
    let doc = Doc::new(path, &text_of(ctx, path));
    let i = (0..doc.units().len())
        .find(|&i| {
            doc.units()[i]
                .label
                .as_deref()
                .is_some_and(|l| l.contains(needle))
                || doc
                    .unit_text(i)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .contains(needle)
        })
        .unwrap_or_else(|| panic!("no part of {path} holds {needle:?}"));
    doc.units()[i].anchor.clone()
}

fn write(ctx: &ToolContext, path: &str, content: &str) {
    harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": path, "content": content}),
        ctx,
    ));
}

fn edit(ctx: &ToolContext, args: serde_json::Value) -> serde_json::Value {
    let mut args = args;
    args["repo"] = json!(REPO);
    harness::invoke_with_ctx("file_edit", args, ctx)
}

#[test]
fn file_write_unicode() {
    let ctx = ctx();
    let content = "こんにちは 🌍 — Unicode test";
    write(&ctx, "uni.txt", content);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "uni.txt", "page": 1}),
        &ctx,
    ));
    let anchor = anchor_of(&ctx, "uni.txt", "Unicode");
    assert_eq!(page_body(&resp), format!("{anchor}| {content}\n"));
}

#[test]
fn file_edit_not_found() {
    let resp = harness::invoke(
        "file_edit",
        json!({"repo": REPO, "path": "nonexistent.txt", "at": "abcd", "do": "replace", "text": "y"}),
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
        json!({"repo": REPO, "path": "https://docs.rs/serde/latest/serde/", "page": 1}),
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
    write(&ctx, "alpha/a.txt", "1");
    write(&ctx, "alpha/b.txt", "2");
    write(&ctx, "beta/c.txt", "3");
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "alpha", "page": 1}),
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
    write(&ctx, "idem.txt", "x");
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
    assert_eq!(text_of(&ctx, "ow.txt"), "v2");
}

/// The whole round trip a model makes: read, take an anchor off the page,
/// replace that part, and find the new text — and the new anchor — in the
/// reply.
#[test]
fn file_edit_round_trip() {
    let ctx = ctx();
    write(&ctx, "rt.txt", "hello world");
    let at = anchor_of(&ctx, "rt.txt", "hello");
    let resp = harness::expect_success(edit(
        &ctx,
        json!({"path": "rt.txt", "at": at, "do": "replace", "text": "hello Rust"}),
    ));
    assert_eq!(text_of(&ctx, "rt.txt"), "hello Rust");
    let now = anchor_of(&ctx, "rt.txt", "hello");
    assert_eq!(
        resp["result"],
        format!("Replaced {at}.\n{now}| hello Rust\n")
    );
    assert_eq!(resp["bytes"], 10);
}

#[test]
fn file_write_read_roundtrip() {
    let ctx = ctx();
    write(&ctx, "hello.txt", "hello world");
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "hello.txt", "page": 1}),
        &ctx,
    ));
    let a = anchor_of(&ctx, "hello.txt", "hello");
    assert_eq!(page_body(&resp), format!("{a}| hello world\n"));
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
        json!({"repo": REPO, "path": "nosuchfile.txt", "page": 1}),
    );
    harness::expect_error(&resp, "not_found");
}

/// `old` changes only the words named, inside the part named.
#[test]
fn file_edit_with_old_changes_only_those_words() {
    let ctx = ctx();
    write(&ctx, "edit.txt", "foo bar baz");
    let at = anchor_of(&ctx, "edit.txt", "foo");
    let resp = harness::expect_success(edit(
        &ctx,
        json!({"path": "edit.txt", "at": at, "do": "replace", "old": "bar", "text": "qux"}),
    ));
    assert_eq!(resp["bytes"], 11);
    assert!(
        resp["result"]
            .as_str()
            .unwrap()
            .starts_with(&format!("Changed {at}.\n")),
        "{resp}"
    );
    assert_eq!(text_of(&ctx, "edit.txt"), "foo qux baz");
}

#[test]
fn file_edit_with_old_twice_in_the_part_is_ambiguous() {
    let ctx = ctx();
    write(&ctx, "dup.txt", "aa bb aa\n");
    let at = anchor_of(&ctx, "dup.txt", "aa");
    let resp = edit(
        &ctx,
        json!({"path": "dup.txt", "at": at, "do": "replace", "old": "aa", "text": "xx"}),
    );
    harness::expect_error(&resp, "ambiguous");
    assert_eq!(text_of(&ctx, "dup.txt"), "aa bb aa\n");
}

/// A method inserted after another lands in the class, at the file's own
/// spacing, and comes back as its own part.
#[test]
fn file_edit_inserts_a_method_after_another() {
    let ctx = ctx();
    write(
        &ctx,
        "inv.js",
        "class Inventory {\n  list() {\n    return [];\n  }\n}\n",
    );
    let at = anchor_of(&ctx, "inv.js", "list");
    let resp = harness::expect_success(edit(
        &ctx,
        json!({"path": "inv.js", "at": at, "do": "insert_after",
               "text": "  size() {\n    return 0;\n  }"}),
    ));
    assert_eq!(
        text_of(&ctx, "inv.js"),
        "class Inventory {\n  list() {\n    return [];\n  }\n  size() {\n    return 0;\n  }\n}\n"
    );
    let size = anchor_of(&ctx, "inv.js", "size");
    let result = resp["result"].as_str().unwrap();
    assert!(
        result.starts_with(&format!("Inserted after {at}.\n")),
        "{result}"
    );
    assert!(
        result.contains(&format!("{size}| class Inventory > size\n  size() {{\n")),
        "{result}"
    );
}

#[test]
fn file_edit_deletes_and_moves_parts() {
    let ctx = ctx();
    write(&ctx, "s.md", "One.\n\nTwo.\n\nThree.\n");
    let (one, two) = (
        anchor_of(&ctx, "s.md", "One"),
        anchor_of(&ctx, "s.md", "Two"),
    );
    let three = anchor_of(&ctx, "s.md", "Three");
    harness::expect_success(edit(
        &ctx,
        json!({"path": "s.md", "at": one, "do": "move", "dest": three, "where": "after"}),
    ));
    assert_eq!(text_of(&ctx, "s.md"), "Two.\n\nThree.\n\nOne.\n");
    let resp = harness::expect_success(edit(
        &ctx,
        json!({"path": "s.md", "at": two, "do": "delete"}),
    ));
    assert_eq!(text_of(&ctx, "s.md"), "Three.\n\nOne.\n");
    assert_eq!(resp["result"], format!("Deleted {two}.\n{three}| Three.\n"));
}

/// The tool end to end over a real workspace: the edit lands in the session,
/// the file on disk is untouched, and the same edit sent a second time names
/// an anchor the file no longer has — refused, never applied twice.
#[test]
fn file_edit_replaces_through_the_overlay_and_leaves_disk_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let repo_dir = harness::repo_root(dir.path());
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let on_disk = "fn main() {\n    let retries = 3;\n    run(retries);\n}\n";
    std::fs::write(repo_dir.join("src/main.rs"), on_disk).unwrap();
    let ctx = harness::workspace_ctx(dir.path());

    let at = anchor_of(&ctx, "src/main.rs", "fn main");
    let args = json!({"path": "src/main.rs", "at": at, "do": "replace",
                      "old": "let retries = 3;", "text": "let retries = 30;"});
    let first = harness::expect_success(edit(&ctx, args.clone()));
    assert_eq!(first["repo"], REPO);
    assert_eq!(first["path"], "src/main.rs");
    assert_eq!(first["bytes"], 54);

    let patched = "fn main() {\n    let retries = 30;\n    run(retries);\n}\n";
    assert_eq!(text_of(&ctx, "src/main.rs"), patched);
    assert_eq!(
        std::fs::read_to_string(repo_dir.join("src/main.rs")).unwrap(),
        on_disk,
        "the file on disk must be byte-for-byte what it was",
    );

    // The same edit again: its anchor named the part as it was, so it is
    // stale now — and a naive substring replacement would have left
    // `retries = 300` here.
    let second = edit(&ctx, args);
    let detail = harness::expect_error(&second, "stale");
    assert!(detail.contains("read it again"), "{detail}");
    assert_eq!(text_of(&ctx, "src/main.rs"), patched);
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
    // A 200-line paragraph is cut into parts at the unit cap; each edit names
    // the part its line is in.
    for (line, old, new) in [
        (10, "line 9\nline 10\n", "line 9\nline ten\n"),
        (190, "line 190\n", "line 190\nline 190½\n"),
    ] {
        let doc = Doc::new("big.txt", &text_of(&overlay, "big.txt"));
        let at = doc.units()[doc.unit_at(line - 1).unwrap()].anchor.clone();
        harness::expect_success(edit(
            &overlay,
            json!({"path": "big.txt", "at": at, "do": "replace", "old": old, "text": new}),
        ));
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

/// A refused edit writes nothing at all — a workspace file must not be copied
/// up on account of an edit that did not happen.
#[test]
fn a_stale_edit_does_not_copy_the_file_up() {
    let dir = tempfile::tempdir().unwrap();
    let repo_dir = harness::repo_root(dir.path());
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("config.toml"), "[net]\nport = 9090\n").unwrap();
    let ctx = harness::workspace_ctx(dir.path());

    let resp = edit(
        &ctx,
        json!({"path": "config.toml", "at": "zzzz", "do": "replace", "text": "[net]\nport = 1\n"}),
    );
    harness::expect_error(&resp, "stale");

    let listed = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "page": 1}),
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

/// `old` that is not in its part writes nothing, and the refusal shows the
/// part as it stands, so the next call can quote it.
#[test]
fn file_edit_writes_nothing_when_old_is_not_there() {
    let ctx = ctx();
    let before = "alpha\nbeta\ngamma\ndelta\n";
    write(&ctx, "all.txt", before);
    let at = anchor_of(&ctx, "all.txt", "alpha");
    let resp = edit(
        &ctx,
        json!({"path": "all.txt", "at": at, "do": "replace", "old": "absent", "text": "new"}),
    );
    let detail = harness::expect_error(&resp, "not_found");
    assert_eq!(
        detail,
        format!("`old` does not occur in {at}, which reads:\nalpha\nbeta\ngamma\ndelta")
    );
    assert_eq!(text_of(&ctx, "all.txt"), before);
}

/// An edit that names nothing to do is rejected as bad arguments.
#[test]
fn file_edit_rejects_an_edit_that_changes_nothing() {
    let ctx = ctx();
    write(&ctx, "p.txt", "a\n");
    let at = anchor_of(&ctx, "p.txt", "a");
    for args in [
        json!({"path": "p.txt", "at": at, "do": "replace", "old": "a", "text": "a"}),
        json!({"path": "p.txt", "at": at, "do": "replace", "text": "  \n"}),
        json!({"path": "p.txt", "at": at, "do": "replace"}),
        json!({"path": "p.txt", "at": at, "do": "move", "dest": at, "where": "after"}),
        json!({"path": "p.txt", "at": at, "do": "rewrite", "text": "b"}),
    ] {
        harness::expect_error(&edit(&ctx, args), "invalid_arguments");
    }
    assert_eq!(text_of(&ctx, "p.txt"), "a\n");
}

#[test]
fn file_list() {
    let ctx = ctx();
    write(&ctx, "a/b.txt", "1");
    write(&ctx, "a/c.txt", "2");
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "a", "page": 1}),
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
            json!({ "repo": REPO, "path": path, "page": 1 }),
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
        let resp = harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path, "page": 1 }),
            &ctx,
        );
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
    write(&ctx, "a.txt", "1");
    write(&ctx, "nested/deep/b.txt", "22");
    for path in ["/", ""] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path, "page": 1 }),
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
    write(&ctx, "src/main.rs", "fn main() {}\n");
    write(&ctx, "README.md", "# hi\n");
    for path in ["src", "src/", "/src", "./src", "src/../src"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path, "page": 1 }),
            &ctx,
        ));
        let entries = resp["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1, "path {path:?}");
        assert_eq!(entries[0]["path"], "src/main.rs");
    }
    // The same file resolves under either spelling.
    let bare = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "/src/./main.rs", "page": 1}),
        &ctx,
    ));
    let a = anchor_of(&ctx, "src/main.rs", "main");
    assert_eq!(page_body(&bare), format!("{a}| fn main\nfn main() {{}}\n"));
}

#[test]
fn file_delete() {
    let ctx = ctx();
    write(&ctx, "del.txt", "bye");
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
    write(&ctx, "p.txt", "hi");
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
