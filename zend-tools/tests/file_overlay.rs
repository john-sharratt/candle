//! Overlay semantics for the `file_*` tools: the session layer stacked over a
//! real working directory.
//!
//! Every test builds a throwaway workspace on disk and points a `ToolContext` at
//! it, so these exercise the lower layer that `file.rs` (upper-layer only) does
//! not. The invariant running through all of them: **no tool ever modifies the
//! workspace**. Each test that mutates asserts the on-disk bytes afterwards.

mod harness;

use std::path::Path;

use serde_json::json;
use tempfile::TempDir;
use zend_tools::ToolContext;

/// A workspace with a small, predictable tree.
fn workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_disk(root, "README.md", "# project\n");
    write_disk(
        root,
        "src/main.rs",
        "fn main() {\n    println!(\"hi\");\n}\n",
    );
    write_disk(root, "src/lib.rs", "pub mod util;\n");
    write_disk(root, "docs/guide.md", "guide\n");
    dir
}

fn write_disk(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

fn read_disk(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap()
}

fn ctx_for(dir: &TempDir) -> ToolContext {
    ToolContext::with_workspace(dir.path())
}

/// The source lines out of a rendered `file_read` excerpt, with the header,
/// fence and `cat -n` numbering stripped — so a test can assert on content
/// without restating the format (which `read_returns_a_numbered_fenced_excerpt`
/// pins separately).
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

fn paths(resp: &serde_json::Value) -> Vec<String> {
    resp["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap().to_string())
        .collect()
}

// ── Read-through ─────────────────────────────────────────────────────────────

#[test]
fn read_falls_through_to_the_workspace() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 200}),
        &ctx,
    ));
    assert_eq!(
        excerpt_source(&resp),
        "fn main() {
    println!(\"hi\");
}",
    );
}

/// The `/workspace` mount point the tool definitions document resolves to the
/// same entry as a bare relative path.
#[test]
fn workspace_mount_prefix_is_an_alias_for_the_root() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    for path in [
        "src/main.rs",
        "/src/main.rs",
        "workspace/src/main.rs",
        "/workspace/src/main.rs",
        "/workspace/src/../src/main.rs",
    ] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_read",
            json!({ "path": path, "start_line": 1, "end_line": 200 }),
            &ctx,
        ));
        assert!(
            resp.as_str().unwrap().contains("fn main() {"),
            "path {path:?} did not resolve",
        );
    }
}

#[test]
fn read_of_a_missing_workspace_path_is_not_found() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/nope.rs", "start_line": 1, "end_line": 200}),
        &ctx,
    );
    harness::expect_error(&resp, "not_found");
}

/// A traversal attempt normalises back inside the root rather than escaping it.
#[test]
fn parent_traversal_cannot_escape_the_workspace() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "../../../../etc/passwd", "start_line": 1, "end_line": 200}),
        &ctx,
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn non_utf8_workspace_file_is_unreadable_not_missing() {
    let dir = workspace();
    std::fs::write(dir.path().join("blob.bin"), [0xffu8, 0xfe, 0x00, 0x01]).unwrap();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "blob.bin", "start_line": 1, "end_line": 200}),
        &ctx,
    );
    harness::expect_error(&resp, "unreadable");
}

// ── Listing ──────────────────────────────────────────────────────────────────

#[test]
fn list_enumerates_the_workspace_from_the_root() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "/"}),
        &ctx,
    ));
    assert_eq!(
        paths(&resp),
        vec!["README.md", "docs/guide.md", "src/lib.rs", "src/main.rs"],
    );
    // Nothing has been written this session, so the budget is untouched even
    // though the listing is non-empty.
    assert_eq!(resp["total_bytes"], 0);
}

#[test]
fn list_narrows_the_workspace_by_prefix() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    for prefix in ["src", "src/", "/workspace/src"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        assert_eq!(
            paths(&resp),
            vec!["src/lib.rs", "src/main.rs"],
            "prefix {prefix:?}",
        );
    }
}

#[test]
fn list_omits_gitignored_and_hidden_paths() {
    let dir = workspace();
    write_disk(dir.path(), ".gitignore", "target/\nsecret.txt\n");
    write_disk(dir.path(), "target/debug/huge.bin", "x");
    write_disk(dir.path(), "secret.txt", "shh");
    let ctx = ctx_for(&dir);
    let listed = paths(&harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": ""}),
        &ctx,
    )));
    assert!(
        !listed.iter().any(|p| p.starts_with("target/")),
        "{listed:?}"
    );
    assert!(!listed.contains(&"secret.txt".to_string()), "{listed:?}");
    // Hidden files are excluded from listings the way `ls` excludes them...
    assert!(!listed.contains(&".gitignore".to_string()), "{listed:?}");
    // ...but still resolve by exact path, which is what file_read's own examples
    // (`/workspace/.gitignore`) depend on.
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "/workspace/.gitignore", "start_line": 1, "end_line": 200}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&resp), "target/\nsecret.txt");
}

#[test]
fn list_marks_session_written_entries_as_modified() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::invoke_with_ctx(
        "write",
        json!({"path": "src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "src/"}),
        &ctx,
    ));
    let files = resp["files"].as_array().unwrap();
    assert_eq!(files.len(), 2, "the shadowed file must not be listed twice");
    let main = files.iter().find(|f| f["path"] == "src/main.rs").unwrap();
    let lib = files.iter().find(|f| f["path"] == "src/lib.rs").unwrap();
    assert_eq!(main["modified"], true);
    assert_eq!(main["bytes"], 13);
    assert!(
        lib.get("modified").is_none(),
        "unchanged entries omit the flag"
    );
}

// ── Copy-up ──────────────────────────────────────────────────────────────────

#[test]
fn edit_copies_the_workspace_file_up_and_leaves_disk_untouched() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let before = read_disk(dir.path(), "src/main.rs");

    harness::expect_success(harness::invoke_with_ctx(
        "file_edit",
        json!({
            "path": "src/main.rs",
            "patch": "@@ -2 +2 @@\n-    println!(\"hi\");\n+    println!(\"hello\");\n"
        }),
        &ctx,
    ));

    // The session now sees the edited copy...
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 200}),
        &ctx,
    ));
    assert_eq!(
        excerpt_source(&resp),
        "fn main() {\n    println!(\"hello\");\n}",
    );
    // ...and the file on disk is byte-for-byte what it was.
    assert_eq!(read_disk(dir.path(), "src/main.rs"), before);
}

#[test]
fn edit_of_a_workspace_file_is_ambiguous_when_the_hunk_matches_twice() {
    let dir = workspace();
    write_disk(dir.path(), "dup.txt", "aa\nbb\naa\n");
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({"path": "dup.txt", "patch": "@@ -2 +2 @@\n-aa\n+cc\n"}),
        &ctx,
    );
    harness::expect_error(&resp, "ambiguous");
    // A rejected edit must not leave a copy-up behind that shadows the original.
    let listed = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "dup.txt"}),
        &ctx,
    ));
    assert!(
        listed["files"][0].get("modified").is_none(),
        "a rejected edit leaves the file unmodified",
    );
}

#[test]
fn write_over_a_workspace_file_reports_overwrite_not_creation() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "README.md", "content": "# replaced\n"}),
        &ctx,
    ));
    assert_eq!(resp["created"], false, "the path already resolved on disk");
    assert_eq!(read_disk(dir.path(), "README.md"), "# project\n");

    let fresh = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "NOTES.md", "content": "new\n"}),
        &ctx,
    ));
    assert_eq!(fresh["created"], true);
}

// ── Whiteouts ────────────────────────────────────────────────────────────────

#[test]
fn delete_hides_a_workspace_file_without_erasing_it() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "docs/guide.md"}),
        &ctx,
    ));
    assert_eq!(resp["deleted"], true);

    // Gone from this session's view, both ways...
    harness::expect_error(
        &harness::invoke_with_ctx(
            "file_read",
            json!({"path": "docs/guide.md", "start_line": 1, "end_line": 200}),
            &ctx,
        ),
        "not_found",
    );
    let listed = paths(&harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": ""}),
        &ctx,
    )));
    assert!(!listed.contains(&"docs/guide.md".to_string()), "{listed:?}");

    // ...and still on disk.
    assert_eq!(read_disk(dir.path(), "docs/guide.md"), "guide\n");
}

#[test]
fn deleting_twice_reports_not_found_the_second_time() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "README.md"}),
        &ctx,
    ));
    harness::expect_error(
        &harness::invoke_with_ctx("file_delete", json!({"path": "README.md"}), &ctx),
        "not_found",
    );
}

#[test]
fn writing_over_a_whiteout_resurrects_the_path_as_a_creation() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "README.md"}),
        &ctx,
    ));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"path": "README.md", "content": "# back\n"}),
        &ctx,
    ));
    assert_eq!(
        resp["created"], true,
        "the path did not resolve while the whiteout stood",
    );
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"path": "README.md", "start_line": 1, "end_line": 200}),
        &ctx,
    ));
    assert_eq!(excerpt_source(&read), "# back");
}

#[test]
fn edit_after_delete_reports_not_found() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"path": "src/lib.rs"}),
        &ctx,
    ));
    harness::expect_error(
        &harness::invoke_with_ctx(
            "file_edit",
            json!({
                "path": "src/lib.rs",
                "patch": "@@ -1 +1 @@\n-pub mod util;\n+pub mod helper;\n"
            }),
            &ctx,
        ),
        "not_found",
    );
}

// ── Presentation ─────────────────────────────────────────────────────────────

#[test]
fn present_resolves_workspace_files_and_reports_the_rest_missing() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_present",
        json!({"paths": ["/workspace/README.md", "src/absent.rs"]}),
        &ctx,
    ));
    assert_eq!(resp["presented"], json!(["/workspace/README.md"]));
    assert_eq!(resp["missing"], json!(["src/absent.rs"]));
}

// ── Paging ───────────────────────────────────────────────────────────────────

/// The failure that started this: `file_list` over a real source directory
/// returned 175 files as one 5.7k-token JSON blob. Results are now capped to a
/// page, with the total and a `next_page` reported so nothing is silently lost.
#[test]
fn list_caps_a_large_directory_to_one_page() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..175 {
        write_disk(dir.path(), &format!("src/f{i:03}.rs"), "fn a() {}\n");
    }
    let ctx = ToolContext::with_workspace(dir.path());

    let first = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "src/"}),
        &ctx,
    ));
    assert_eq!(first["files"].as_array().unwrap().len(), 50);
    assert_eq!(first["paging"]["total"], 175);
    assert_eq!(first["paging"]["pages"], 4);
    assert_eq!(first["paging"]["page"], 0);
    assert_eq!(first["paging"]["next_page"], 1);
    assert_eq!(paths(&first)[0], "src/f000.rs");

    // The advertised next page continues exactly where the first stopped.
    let second = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "src/", "page": 1}),
        &ctx,
    ));
    assert_eq!(paths(&second)[0], "src/f050.rs");
    assert_eq!(second["paging"]["next_page"], 2);

    // The last page is partial and terminates the chain.
    let last = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "src/", "page": 3}),
        &ctx,
    ));
    assert_eq!(last["files"].as_array().unwrap().len(), 25);
    assert_eq!(paths(&last)[0], "src/f150.rs");
    assert!(last["paging"]["next_page"].is_null());
}

/// Over-shooting the page count clamps to the last page rather than returning an
/// empty list, which a model would read as "the directory is empty".
#[test]
fn list_page_past_the_end_clamps_to_the_last_page() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..60 {
        write_disk(dir.path(), &format!("a/f{i:02}.txt"), "x\n");
    }
    let ctx = ToolContext::with_workspace(dir.path());
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "a/", "page": 99}),
        &ctx,
    ));
    assert_eq!(resp["paging"]["page"], 1, "clamped to the last page");
    assert_eq!(resp["files"].as_array().unwrap().len(), 10);
    assert!(!resp["files"].as_array().unwrap().is_empty());
}

/// A listing that fits reports a single page and no continuation.
#[test]
fn list_within_one_page_reports_no_next_page() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": ""}),
        &ctx,
    ));
    assert_eq!(resp["paging"]["pages"], 1);
    assert_eq!(resp["paging"]["total"], 4);
    assert!(resp["paging"]["next_page"].is_null());
}

/// `modified` is omitted when false — it is a third of an entry's JSON and the
/// common case is an untouched project file.
#[test]
fn list_entries_omit_the_modified_flag_when_unchanged() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::invoke_with_ctx(
        "write",
        json!({"path": "src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"prefix": "src/"}),
        &ctx,
    ));
    let files = resp["files"].as_array().unwrap();
    let main = files.iter().find(|f| f["path"] == "src/main.rs").unwrap();
    let lib = files.iter().find(|f| f["path"] == "src/lib.rs").unwrap();
    assert_eq!(main["modified"], true);
    assert!(lib.get("modified").is_none(), "unchanged entries stay lean");
}

// ── file_read excerpt format ─────────────────────────────────────────────────

/// A read returns the same shape the `code_reading` ingest prefills: header with
/// path and absolute line range, then a language-tagged fence with `cat -n`
/// numbering. Not JSON — the runner places a string result verbatim.
#[test]
fn read_returns_a_numbered_fenced_excerpt() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 3}),
        &ctx,
    );
    let text = resp.as_str().expect("file_read returns a rendered string");
    assert_eq!(
        text,
        "\nsrc/main.rs (lines 1-3):\n\n```rust\n1  fn main() {\n2      println!(\"hi\");\n3  }\n```\n",
    );
}

/// **There is no whole-file read.** A call that names no range — or only one
/// bound — is refused before it runs, so the unbounded read that put 144 KB of
/// one module into a live conversation cannot be expressed. The model is told
/// which field is missing, which is what it needs to reissue the call.
#[test]
fn a_read_without_both_bounds_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(dir.path(), "big.rs", &body);
    let ctx = ToolContext::with_workspace(dir.path());

    for args in [
        json!({"path": "big.rs"}),
        json!({"path": "big.rs", "start_line": 1}),
        json!({"path": "big.rs", "end_line": 200}),
    ] {
        let resp = harness::invoke_with_ctx("file_read", args.clone(), &ctx);
        harness::expect_error(&resp, "invalid_arguments");
        let detail = resp["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains("start_line") || detail.contains("end_line"),
            "the refusal must name the missing bound, got {detail:?} for {args}",
        );
    }
}

/// A range wider than the cap is **served, not refused**: the first
/// `MAX_READ_LINES` lines come back, and because the excerpt now stops short of
/// the end the header carries `of N` — which says both that it was cut and where
/// to resume. Asking for the whole of a 900-line file therefore costs 200 lines
/// and a continuation hint, not 900 lines.
#[test]
fn a_range_wider_than_the_cap_returns_the_first_200_lines() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(dir.path(), "big.rs", &body);
    let ctx = ToolContext::with_workspace(dir.path());

    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "big.rs", "start_line": 1, "end_line": 900}),
        &ctx,
    );
    let numbered: String = (1..=200).map(|i| format!("{i:3}  line {i}\n")).collect();
    assert_eq!(
        resp.as_str().expect("file_read returns a rendered string"),
        format!("\nbig.rs (lines 1-200 of 900):\n\n```rust\n{numbered}```\n"),
    );

    // The header's own arithmetic is the continuation protocol: resuming at 201
    // picks up exactly where it stopped, with no gap and no repeated line.
    let next = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "big.rs", "start_line": 201, "end_line": 400}),
        &ctx,
    );
    let text = next.as_str().unwrap();
    assert!(
        text.starts_with("\nbig.rs (lines 201-400 of 900):"),
        "{text:.60}"
    );
    assert!(text.contains("201  line 201"), "resumes at the next line");
}

/// Exactly the cap is not truncated — the boundary is inclusive, so a 200-line
/// range returns 200 lines. Off by one here would silently shorten every read.
#[test]
fn a_range_of_exactly_the_cap_is_returned_whole() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(dir.path(), "big.rs", &body);
    let ctx = ToolContext::with_workspace(dir.path());

    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "big.rs", "start_line": 51, "end_line": 250}),
        &ctx,
    );
    let text = resp.as_str().unwrap();
    assert!(
        text.starts_with("\nbig.rs (lines 51-250 of 900):"),
        "{text:.60}"
    );
    assert_eq!(
        text.lines().filter(|l| l.contains("  line ")).count(),
        200,
        "a 200-line range is the cap, not one past it",
    );
}

/// The cap bounds the SPAN, not the file: a short file read with a wide range
/// still ends at the last line, and the header stays in the plain `(lines a-b)`
/// form that says there is no more to fetch.
#[test]
fn a_wide_range_over_a_short_file_ends_at_the_last_line() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 1, "end_line": 200}),
        &ctx,
    );
    assert_eq!(
        resp.as_str().unwrap(),
        "\nsrc/main.rs (lines 1-3):\n\n```rust\n1  fn main() {\n2      println!(\"hi\");\n3  }\n```\n",
    );
}

/// An explicit range is honoured exactly, and a range running past the end of
/// the file clamps to it rather than failing.
#[test]
fn read_honours_a_line_range_and_clamps_a_wide_one() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(dir.path(), "big.rs", &body);
    let ctx = ToolContext::with_workspace(dir.path());

    let exact = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "big.rs", "start_line": 47, "end_line": 93}),
        &ctx,
    );
    // Column width tracks the widest line number in the excerpt (93 → 2), the
    // same rule the ingest renderer uses.
    let numbered: String = (47..=93).map(|i| format!("{i}  line {i}\n")).collect();
    assert_eq!(
        exact.as_str().unwrap(),
        format!("\nbig.rs (lines 47-93 of 900):\n\n```rust\n{numbered}```\n"),
    );

    let wide = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "big.rs", "start_line": 850, "end_line": 5000}),
        &ctx,
    );
    let numbered: String = (850..=900).map(|i| format!("{i}  line {i}\n")).collect();
    assert_eq!(
        wide.as_str().unwrap(),
        format!("\nbig.rs (lines 850-900):\n\n```rust\n{numbered}```\n"),
    );
}

/// An out-of-range start clamps into the file rather than returning nothing,
/// which a model would read as "the file is empty".
#[test]
fn read_past_the_end_clamps_into_the_file() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "src/main.rs", "start_line": 9999, "end_line": 9999}),
        &ctx,
    );
    let text = resp.as_str().unwrap();
    assert!(
        text.starts_with("\nsrc/main.rs (lines 3-3):\n"),
        "{text:.60}"
    );
}

/// **The read header is the only place a file's length is reported, so it has to
/// be right for every shape of ending.**
///
/// A listing carries no line count, so the header's `of N` is the single source
/// of a file's length — the number the model pages against — and an off-by-one over
/// a trailing newline would misplace the last range of every file.
#[test]
fn the_read_header_reports_the_files_true_length() {
    let dir = tempfile::tempdir().unwrap();
    // Each shape that has ever made a line count ambiguous.
    let cases = [
        ("trailing.rs", "a\nb\n", 2),
        ("no_trailing.rs", "a\nb", 2),
        ("blank_last.rs", "a\n\n", 2),
        ("just_newline.rs", "\n", 1),
        ("one_line.rs", "only", 1),
        ("crlf.rs", "a\r\nb\r\n", 2),
    ];
    for (name, body, _) in cases {
        write_disk(dir.path(), name, body);
    }
    let ctx = ToolContext::with_workspace(dir.path());

    for (name, _, expected) in cases {
        // Read one line only, so the header is forced to state `of N` — reading
        // the whole file would render the short form and prove nothing.
        let resp = harness::invoke_with_ctx(
            "file_read",
            json!({"path": name, "start_line": 1, "end_line": 1}),
            &ctx,
        );
        let text = resp.as_str().unwrap();
        let header_total = text
            .split_once(" of ")
            .and_then(|(_, rest)| rest.split_once(')'))
            .map(|(n, _)| n.parse::<i64>().unwrap());
        match header_total {
            // A multi-line file reports its total, and it must be the true one.
            Some(total) => assert_eq!(
                total, expected,
                "{name}: header total {total} != true length {expected}",
            ),
            // A one-line file's read reached the end, so the header carries no
            // total — which is only correct when the file really is one line.
            None => assert_eq!(expected, 1, "{name}: header omitted `of N`: {text:?}"),
        }
    }
}

/// An empty file reads as empty rather than as an impossible line range.
#[test]
fn read_of_an_empty_file_reports_empty() {
    let dir = workspace();
    write_disk(dir.path(), "blank.rs", "");
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"path": "blank.rs", "start_line": 1, "end_line": 200}),
        &ctx,
    );
    assert_eq!(
        resp.as_str().unwrap(),
        "\nblank.rs (empty):\n\n```rust\n```\n"
    );
}

/// **Every spelling of a directory prefix lists the same files.** A trailing
/// slash, a leading slash, the `/workspace` mount, and the bare name are the
/// four ways a model writes the same directory, and a listing that answered
/// only some of them would look to the caller like an empty directory — the one
/// result that is indistinguishable from a wrong guess.
#[test]
fn every_spelling_of_a_directory_prefix_lists_the_same_files() {
    let dir = tempfile::tempdir().unwrap();
    // A workspace shaped like this repo: crates at the root, sources nested.
    write_disk(dir.path(), "zend/src/main.rs", "fn main() {}\n");
    write_disk(dir.path(), "zend/src/session.rs", "// session\n");
    write_disk(dir.path(), "zend/Cargo.toml", "[package]\n");
    write_disk(dir.path(), "candle-core/src/lib.rs", "// core\n");
    let ctx = ToolContext::with_workspace(dir.path());

    let expected = vec!["zend/src/main.rs", "zend/src/session.rs"];
    for prefix in [
        "zend/src",
        "zend/src/",
        "/zend/src",
        "/zend/src/",
        "workspace/zend/src",
        "/workspace/zend/src/",
        r"zend\src",
    ] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        assert_eq!(paths(&resp), expected, "prefix {prefix:?}");
    }

    // The root, however it is spelled, lists everything.
    for prefix in ["", "/", "workspace", "/workspace/"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "prefix": prefix }),
            &ctx,
        ));
        assert_eq!(paths(&resp).len(), 4, "prefix {prefix:?}");
    }
}
