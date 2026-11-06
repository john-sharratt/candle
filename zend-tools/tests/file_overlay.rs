//! Overlay semantics for the `file_*` tools: the session layer stacked over a
//! real working directory.
//!
//! Every test builds a throwaway workspace on disk and points a `ToolContext` at
//! it, so these exercise the lower layer that `file.rs` (upper-layer only) does
//! not. The invariant running through all of them: **no tool ever modifies the
//! workspace**. Each test that mutates asserts the on-disk bytes afterwards.

mod harness;

use std::path::Path;

use doc_units::render::{blocks, entry, Source};
use doc_units::Doc;
use serde_json::json;
use tempfile::TempDir;
use zend_tools::ToolContext;

use harness::REPO;

/// A workspace with a small, predictable tree, under the single repository
/// [`REPO`]'s folder.
fn workspace() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
    write_disk(&root, "README.md", "# project\n");
    write_disk(
        &root,
        "src/main.rs",
        "fn main() {\n    println!(\"hi\");\n}\n",
    );
    write_disk(&root, "src/lib.rs", "pub mod util;\n");
    write_disk(&root, "docs/guide.md", "guide\n");
    dir
}

fn write_disk(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

fn read_disk(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(harness::repo_root(root).join(rel)).unwrap()
}

fn ctx_for(dir: &TempDir) -> ToolContext {
    harness::workspace_ctx(dir.path())
}

/// The text of a rendered `file_read` of a plain-text file: its parts' lines,
/// without their blocks' fences, the `+N` before each, or the closing line — so
/// a test can assert on content without restating the format (which
/// `read_returns_each_part_as_a_block` pins separately).
fn excerpt_source(resp: &serde_json::Value) -> String {
    let text = resp.as_str().expect("file_read returns a rendered string");
    text.lines()
        .filter(|l| !l.starts_with("```") && !l.starts_with("end of "))
        .map(|l| {
            let numbered = l.trim_start().trim_start_matches('+');
            let line = numbered.trim_start_matches(|c: char| c.is_ascii_digit());
            line.strip_prefix(' ').unwrap_or(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The anchors `content`, at `path`, reads with.
fn anchors(path: &str, content: &str) -> Vec<String> {
    Doc::new(path, content).anchors()
}

const MAIN_RS: &str = "fn main() {\n    println!(\"hi\");\n}\n";

fn paths(resp: &serde_json::Value) -> Vec<String> {
    resp["entries"]
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
        json!({"repo": REPO, "path": "src/main.rs", "list": 1}),
        &ctx,
    ));
    let a = &anchors("src/main.rs", MAIN_RS)[0];
    assert!(
        resp.as_str().unwrap().contains(&format!(
            "part={a} lines=1-3 scope=\"fn main\"\n +1 fn main() {{\n +2     println!(\"hi\");\n +3 }}\n"
        )),
        "{resp}"
    );
}

/// A leading slash and a `.`/`..` detour both normalise to the same entry as
/// the bare relative path — plain path normalisation (no segment, including a
/// folder that happened to be named `workspace`, is special).
#[test]
fn alternate_spellings_of_a_path_resolve_to_the_same_entry() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    for path in [
        "src/main.rs",
        "/src/main.rs",
        "./src/main.rs",
        "src/../src/main.rs",
    ] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_read",
            json!({ "repo": REPO, "path": path, "list": 1 }),
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
        json!({"repo": REPO, "path": "src/nope.rs", "list": 1}),
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
        json!({"repo": REPO, "path": "../../../../etc/passwd", "list": 1}),
        &ctx,
    );
    harness::expect_error(&resp, "not_found");
}

#[test]
fn non_utf8_workspace_file_is_unreadable_not_missing() {
    let dir = workspace();
    std::fs::write(
        harness::repo_root(dir.path()).join("blob.bin"),
        [0xffu8, 0xfe, 0x00, 0x01],
    )
    .unwrap();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "blob.bin", "list": 1}),
        &ctx,
    );
    harness::expect_error(&resp, "unreadable");
}

// ── Listing ──────────────────────────────────────────────────────────────────

#[test]
fn list_enumerates_the_workspace_one_level_from_the_root() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "/", "page": 1}),
        &ctx,
    ));
    assert_eq!(paths(&resp), vec!["README.md", "docs", "src"]);
    let entries = resp["entries"].as_array().unwrap();
    let dirs: Vec<&str> = entries
        .iter()
        .filter(|e| e["dir"].as_bool().unwrap_or(false))
        .map(|e| e["path"].as_str().unwrap())
        .collect();
    assert_eq!(dirs, vec!["docs", "src"], "subdirectories, not flattened");
    // Nothing has been written this session, so the budget is untouched even
    // though the listing is non-empty.
    assert_eq!(resp["total_bytes"], 0);
}

#[test]
fn list_narrows_the_workspace_by_directory() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    for path in ["src", "src/", "/src"] {
        let resp = harness::expect_success(harness::invoke_with_ctx(
            "file_list",
            json!({ "repo": REPO, "path": path, "page": 1 }),
            &ctx,
        ));
        assert_eq!(
            paths(&resp),
            vec!["src/lib.rs", "src/main.rs"],
            "path {path:?}",
        );
    }
}

#[test]
fn list_omits_gitignored_and_hidden_paths() {
    let dir = workspace();
    let root = harness::repo_root(dir.path());
    write_disk(&root, ".gitignore", "target/\nsecret.txt\n");
    write_disk(&root, "target/debug/huge.bin", "x");
    write_disk(&root, "secret.txt", "shh");
    let ctx = ctx_for(&dir);
    let listed = paths(&harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "", "page": 1}),
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
    // depend on.
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": ".gitignore", "list": 1}),
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
        json!({"repo": REPO, "path": "src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "src", "page": 1}),
        &ctx,
    ));
    let entries = resp["entries"].as_array().unwrap();
    assert_eq!(
        entries.len(),
        2,
        "the shadowed file must not be listed twice"
    );
    let main = entries.iter().find(|f| f["path"] == "src/main.rs").unwrap();
    let lib = entries.iter().find(|f| f["path"] == "src/lib.rs").unwrap();
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
            "repo": REPO,
            "path": "src/main.rs",
            "part": anchors("src/main.rs", MAIN_RS)[0],
            "do": "replace",
            "old": "println!(\"hi\");",
            "text": "println!(\"hello\");"
        }),
        &ctx,
    ));

    // The session now sees the edited copy...
    assert_eq!(
        ctx.files
            .repo(REPO)
            .unwrap()
            .read("src/main.rs")
            .unwrap()
            .unwrap(),
        "fn main() {\n    println!(\"hello\");\n}\n",
    );
    // ...and the file on disk is byte-for-byte what it was.
    assert_eq!(read_disk(dir.path(), "src/main.rs"), before);
}

#[test]
fn edit_of_a_workspace_file_is_ambiguous_when_the_text_occurs_twice() {
    let dir = workspace();
    write_disk(&harness::repo_root(dir.path()), "dup.txt", "aa\nbb\naa\n");
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_edit",
        json!({"repo": REPO, "path": "dup.txt", "part": anchors("dup.txt", "aa\nbb\naa\n")[0],
               "do": "replace", "old": "aa", "text": "cc"}),
        &ctx,
    );
    harness::expect_error(&resp, "ambiguous");
    // A rejected edit must not leave a copy-up behind that shadows the original.
    let listed = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "", "page": 1}),
        &ctx,
    ));
    let entries = listed["entries"].as_array().unwrap();
    let dup = entries.iter().find(|e| e["path"] == "dup.txt").unwrap();
    assert!(
        dup.get("modified").is_none(),
        "a rejected edit leaves the file unmodified",
    );
}

#[test]
fn write_over_a_workspace_file_reports_overwrite_not_creation() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "README.md", "content": "# replaced\n"}),
        &ctx,
    ));
    assert_eq!(resp["created"], false, "the path already resolved on disk");
    assert_eq!(read_disk(dir.path(), "README.md"), "# project\n");

    let fresh = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "NOTES.md", "content": "new\n"}),
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
        json!({"repo": REPO, "path": "docs/guide.md"}),
        &ctx,
    ));
    assert_eq!(resp["deleted"], true);

    // Gone from this session's view, both ways...
    harness::expect_error(
        &harness::invoke_with_ctx(
            "file_read",
            json!({"repo": REPO, "path": "docs/guide.md", "list": 1}),
            &ctx,
        ),
        "not_found",
    );
    let listed = paths(&harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "docs", "page": 1}),
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
        json!({"repo": REPO, "path": "README.md"}),
        &ctx,
    ));
    harness::expect_error(
        &harness::invoke_with_ctx(
            "file_delete",
            json!({"repo": REPO, "path": "README.md"}),
            &ctx,
        ),
        "not_found",
    );
}

#[test]
fn writing_over_a_whiteout_resurrects_the_path_as_a_creation() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    harness::expect_success(harness::invoke_with_ctx(
        "file_delete",
        json!({"repo": REPO, "path": "README.md"}),
        &ctx,
    ));
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "write",
        json!({"repo": REPO, "path": "README.md", "content": "# back\n"}),
        &ctx,
    ));
    assert_eq!(
        resp["created"], true,
        "the path did not resolve while the whiteout stood",
    );
    let read = harness::expect_success(harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "README.md", "list": 1}),
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
        json!({"repo": REPO, "path": "src/lib.rs"}),
        &ctx,
    ));
    harness::expect_error(
        &harness::invoke_with_ctx(
            "file_edit",
            json!({
                "repo": REPO,
                "path": "src/lib.rs",
                "part": anchors("src/lib.rs", "pub mod util;\n")[0],
                "do": "replace",
                "text": "pub mod helper;"
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
        json!({"repo": REPO, "paths": ["README.md", "src/absent.rs"]}),
        &ctx,
    ));
    assert_eq!(resp["presented"], json!(["README.md"]));
    assert_eq!(resp["missing"], json!(["src/absent.rs"]));
}

// ── Paging ───────────────────────────────────────────────────────────────────

/// The failure that started this: `file_list` over a real source directory
/// returned 175 files as one 5.7k-token JSON blob. Results are now capped to a
/// page, with the total and a `next_page` reported so nothing is silently lost.
#[test]
fn list_caps_a_large_directory_to_one_page() {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
    for i in 0..175 {
        write_disk(&root, &format!("src/f{i:03}.rs"), "fn a() {}\n");
    }
    let ctx = harness::workspace_ctx(dir.path());

    let first = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "src", "page": 1}),
        &ctx,
    ));
    assert_eq!(first["entries"].as_array().unwrap().len(), 50);
    assert_eq!(first["paging"]["total"], 175);
    assert_eq!(first["paging"]["pages"], 4);
    assert_eq!(first["paging"]["page"], 1);
    assert_eq!(first["paging"]["next_page"], 2);
    assert_eq!(paths(&first)[0], "src/f000.rs");

    // The advertised next page continues exactly where the first stopped.
    let second = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "src", "page": 2}),
        &ctx,
    ));
    assert_eq!(paths(&second)[0], "src/f050.rs");
    assert_eq!(second["paging"]["next_page"], 3);

    // The last page is partial, is numbered `pages`, and ends the chain.
    let last = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "src", "page": 4}),
        &ctx,
    ));
    assert_eq!(last["entries"].as_array().unwrap().len(), 25);
    assert_eq!(paths(&last)[0], "src/f150.rs");
    assert_eq!(last["paging"]["page"], last["paging"]["pages"]);
    assert!(last["paging"]["next_page"].is_null());
}

/// **There is no page 0**: a call for one is refused with a reason, never
/// read as some other page.
#[test]
fn list_page_zero_is_refused() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "", "page": 0}),
        &ctx,
    );
    harness::expect_error(&resp, "invalid_arguments");
}

/// Over-shooting the page count clamps to the last page rather than returning an
/// empty list, which a model would read as "the directory is empty".
#[test]
fn list_page_past_the_end_clamps_to_the_last_page() {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
    for i in 0..60 {
        write_disk(&root, &format!("a/f{i:02}.txt"), "x\n");
    }
    let ctx = harness::workspace_ctx(dir.path());
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "a", "page": 99}),
        &ctx,
    ));
    assert_eq!(resp["paging"]["page"], 2, "clamped to the last page");
    assert_eq!(resp["entries"].as_array().unwrap().len(), 10);
    assert!(!resp["entries"].as_array().unwrap().is_empty());
}

/// A listing that fits reports a single page and no continuation.
#[test]
fn list_within_one_page_reports_no_next_page() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "", "page": 1}),
        &ctx,
    ));
    assert_eq!(resp["paging"]["pages"], 1);
    assert_eq!(
        resp["paging"]["total"], 3,
        "one level: README.md, docs, src"
    );
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
        json!({"repo": REPO, "path": "src/main.rs", "content": "fn main() {}\n"}),
        &ctx,
    );
    let resp = harness::expect_success(harness::invoke_with_ctx(
        "file_list",
        json!({"repo": REPO, "path": "src", "page": 1}),
        &ctx,
    ));
    let entries = resp["entries"].as_array().unwrap();
    let main = entries.iter().find(|f| f["path"] == "src/main.rs").unwrap();
    let lib = entries.iter().find(|f| f["path"] == "src/lib.rs").unwrap();
    assert_eq!(main["modified"], true);
    assert!(lib.get("modified").is_none(), "unchanged entries stay lean");
}

// ── file_read format ─────────────────────────────────────────────────────────

/// A read returns the same shape the `code_reading` ingest prefills: each part
/// a language-tagged block whose fence names the file, the part's anchor, its
/// lines and its scope, each line behind its number in the part — then the
/// closing line. Not JSON — the runner places a string result verbatim.
#[test]
fn read_returns_each_part_as_a_block() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    let resp = harness::invoke_with_ctx(
        "file_read",
        json!({"repo": REPO, "path": "src/main.rs", "list": 1}),
        &ctx,
    );
    let text = resp.as_str().expect("file_read returns a rendered string");
    let a = &anchors("src/main.rs", MAIN_RS)[0];
    assert_eq!(
        text,
        format!(
            "```rust file=proj/src/main.rs part={a} lines=1-3 scope=\"fn main\"\n +1 fn main() {{\n \
             +2     println!(\"hi\");\n +3 }}\n```\nend of proj/src/main.rs lines 1-3 of 3 — the end \
             of the file\n"
        ),
    );
}

/// **A read names exactly one of `part`, `lines` and `list`**: none, or two,
/// is refused.
#[test]
fn a_read_naming_none_or_two_things_is_refused() {
    let dir = workspace();
    let ctx = ctx_for(&dir);
    for args in [
        json!({"repo": REPO, "path": "src/main.rs"}),
        json!({"repo": REPO, "path": "src/main.rs", "list": 1, "lines": "1"}),
        json!({"repo": REPO, "path": "src/main.rs", "list": 0}),
        json!({"repo": REPO, "path": "src/main.rs", "lines": "0"}),
        json!({"repo": REPO, "path": "src/main.rs", "lines": "+3"}),
    ] {
        let resp = harness::invoke_with_ctx("file_read", args.clone(), &ctx);
        harness::expect_error(&resp, "invalid_arguments");
    }
}

/// **A read of lines holds the whole parts they touch, up to its budget**, and
/// names the range to read next; one past the end reads the last part.
#[test]
fn a_read_of_lines_holds_whole_parts_and_names_the_next() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(&harness::repo_root(dir.path()), "big.txt", &body);
    let ctx = harness::workspace_ctx(dir.path());
    // One 900-line paragraph, cut at the unit cap into eleven parts of 80
    // lines and one of 20; two 80-line parts fill a read.
    let doc = Doc::new("big.txt", &body);
    assert_eq!(doc.units().len(), 12);
    let source = Source {
        file: "proj/big.txt",
        tag: "",
    };
    let read = |lines: &str| {
        harness::invoke_with_ctx(
            "file_read",
            json!({"repo": REPO, "path": "big.txt", "lines": lines}),
            &ctx,
        )
        .as_str()
        .unwrap()
        .to_string()
    };

    assert_eq!(
        read("161-320"),
        format!(
            "{}end of proj/big.txt lines 161-320 of 900 — next: lines=321-480\n",
            blocks(&doc, 2..4, source)
        ),
    );
    assert!(
        read("161-320").contains("lines=161-240 scope=\"line 161\"\n +1 line 161\n +2 line 162\n")
    );
    // Asked for more than a read holds, it holds what fits and says where on.
    assert_eq!(
        read("1-900"),
        format!(
            "{}end of proj/big.txt lines 1-160 of 900 — next: lines=161-320\n",
            blocks(&doc, 0..2, source)
        ),
    );
    // One line is its part, named with its line in it.
    let a10 = &doc.units()[10].anchor;
    assert_eq!(
        read("850"),
        format!(
            "line 850 is part={a10} +50\n{}end of proj/big.txt lines 801-880 of 900 — next: \
             lines=881-900\n",
            blocks(&doc, 10..11, source)
        ),
    );
    assert!(read("9999").starts_with("proj/big.txt has 900 lines\n```text file=proj/big.txt"));
    assert!(
        read("9999").ends_with("end of proj/big.txt lines 881-900 of 900 — the end of the file\n")
    );
}

/// **A one-line read says which part and line it is**, and a blank line
/// between parts says so.
#[test]
fn a_one_line_read_names_its_part_and_line() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (0..300).map(|i| format!("para {i}\n\n")).collect();
    write_disk(&harness::repo_root(dir.path()), "p.md", &body);
    let ctx = harness::workspace_ctx(dir.path());
    let doc = Doc::new("p.md", &body);
    let read = |line: usize| {
        harness::invoke_with_ctx(
            "file_read",
            json!({"repo": REPO, "path": "p.md", "lines": line.to_string()}),
            &ctx,
        )
        .as_str()
        .unwrap()
        .to_string()
    };
    let a = |i: usize| doc.units()[i].anchor.clone();
    assert!(read(1).starts_with(&format!("line 1 is part={} +1\n", a(0))));
    assert!(read(401).starts_with(&format!("line 401 is part={} +1\n", a(200))));
    for blank in [2, 600] {
        assert!(
            read(blank).starts_with(&format!("line {blank} is a blank line between parts\n")),
            "line {blank}"
        );
    }
    assert!(read(9999).starts_with("proj/p.md has 600 lines\n"));
}

/// **A list names every part in one line**, and a page past its end reads the
/// last page rather than nothing, which a model would read as "the file is
/// empty".
#[test]
fn a_list_names_the_parts_and_clamps_past_the_end() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
    write_disk(&harness::repo_root(dir.path()), "big.txt", &body);
    let ctx = harness::workspace_ctx(dir.path());
    let doc = Doc::new("big.txt", &body);
    let entries: String = (0..12).map(|i| entry(&doc, i)).collect();
    let expected = format!(
        "proj/big.txt: 900 lines in 12 parts. Read one with `part`, or a run of them with \
         `lines`.\n{entries}end of the parts of proj/big.txt page 1 of 1 — the last page\n"
    );
    for page in [1, 99] {
        let resp = harness::invoke_with_ctx(
            "file_read",
            json!({"repo": REPO, "path": "big.txt", "list": page}),
            &ctx,
        );
        assert_eq!(resp.as_str().unwrap(), expected, "list {page}");
    }
    assert!(entries.starts_with(&format!(
        "part={} lines=1-80 line 1\npart={} lines=81-160 line 81\n",
        doc.units()[0].anchor,
        doc.units()[1].anchor
    )));
}

/// **A read's closing line is the only place a file's length is reported, so
/// it has to be right for every shape of ending.**
///
/// A listing carries no line count, so the closing line's `of N` is the single
/// source of a file's length — the number the model reads on against — and an
/// off-by-one over a trailing newline would misplace the last range of every
/// file.
#[test]
fn the_read_reports_the_files_true_length() {
    let dir = tempfile::tempdir().unwrap();
    let root = harness::repo_root(dir.path());
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
        write_disk(&root, name, body);
    }
    let ctx = harness::workspace_ctx(dir.path());

    for (name, _, expected) in cases {
        // Every case is short, so a list reads it whole and closes stating its
        // length — or, with nothing but blank lines, says how many.
        let resp = harness::invoke_with_ctx(
            "file_read",
            json!({"repo": REPO, "path": name, "list": 1}),
            &ctx,
        );
        let text = resp.as_str().unwrap();
        let total: i64 = text
            .lines()
            .find(|l| l.starts_with("end of "))
            .and_then(|l| l.split_once(" — "))
            .and_then(|(said, _)| said.rsplit_once(" of "))
            .map(|(_, n)| n.parse().unwrap())
            .or_else(|| {
                text.split_once("holds no text — ")
                    .and_then(|(_, rest)| rest.split_once(' '))
                    .map(|(n, _)| n.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("{name}: the read states no length: {text:?}"));
        assert_eq!(
            total, expected,
            "{name}: stated length {total} != true length {expected}",
        );
    }
}

/// An empty file reads as empty rather than as an impossible range.
#[test]
fn read_of_an_empty_file_reports_empty() {
    let dir = workspace();
    write_disk(&harness::repo_root(dir.path()), "blank.rs", "");
    let ctx = ctx_for(&dir);
    for args in [
        json!({"repo": REPO, "path": "blank.rs", "list": 1}),
        json!({"repo": REPO, "path": "blank.rs", "lines": "1"}),
    ] {
        let resp = harness::invoke_with_ctx("file_read", args, &ctx);
        assert_eq!(
            resp.as_str().unwrap(),
            "proj/blank.rs holds no text — 0 line(s), all blank.\n"
        );
    }
}
