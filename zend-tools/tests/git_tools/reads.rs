//! The readers: git_status, git_log, git_show, git_grep, git_refs.
//!
//! None changes the repository, and each runs here on the read-only context —
//! Comprehensive's grants, which hold no `DiskWrite` at all.

use serde_json::{json, Value};
use zend_tools::registry::find;

use crate::harness::{branch_rev, commit_rev, head_rev, parent_rev, tag_rev, GitWorkspace};

// ── git_status ───────────────────────────────────────────────────────────────

#[test]
fn a_conversation_with_no_changes_is_clean_and_names_its_branch() {
    let ws = GitWorkspace::new();
    let out = ws.read("git_status", json!({"repo": "app", "page": 0}));
    assert_eq!(out["clean"], true, "{out}");
    assert_eq!(out["branch"], "main");
    assert_eq!(out["head"], ws.oid("HEAD"));
    assert!(out["changes"].as_array().unwrap().is_empty());
    assert_eq!(out["counts"]["modified"], 0);
}

/// **What the conversation changed is reported, and never what is on disk**
/// — the repository's folder is the sandbox's, and holds whatever a job left.
#[test]
fn the_conversations_changes_are_reported_and_never_the_folders() {
    let ws = GitWorkspace::new();
    ws.write_worktree("README.md", "# left on disk by a job\n");
    ws.write_worktree("stray.txt", "stray\n");
    let conv = ws.conversation();
    assert_eq!(
        conv.status()["clean"],
        true,
        "the folder is not the conversation"
    );

    conv.write("README.md", "# app\n\nedited.\n");
    conv.write("NOTES.md", "notes\n");
    conv.delete("src/lib.rs");
    let out = conv.status();
    assert_eq!(out["clean"], false, "{out}");
    let changes: Vec<(String, String)> = out["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["path"].as_str().unwrap().to_string(),
                c["status"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        changes,
        [
            ("NOTES.md".to_string(), "added".to_string()),
            ("README.md".to_string(), "modified".to_string()),
            ("src/lib.rs".to_string(), "deleted".to_string()),
        ]
    );
    assert_eq!(out["counts"]["added"], 1);
    assert_eq!(out["counts"]["modified"], 1);
    assert_eq!(out["counts"]["deleted"], 1);
}

/// **"How far ahead am I" is answered by the tool asked it first.** Live, the
/// model went to git_status for it, found nothing, and spent three rounds
/// elsewhere.
#[test]
fn status_reports_how_far_ahead_of_its_upstream_the_branch_is() {
    let ws = GitWorkspace::new();
    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://example.com/acme/app.git",
    ]);
    ws.git(&["update-ref", "refs/remotes/origin/main", "HEAD~1"]);
    ws.git(&["config", "branch.main.remote", "origin"]);
    ws.git(&["config", "branch.main.merge", "refs/heads/main"]);
    let out = ws.read("git_status", json!({"repo": "app", "page": 0}));
    assert_eq!(out["upstream"]["remote"], "origin", "{out}");
    assert_eq!(out["upstream"]["branch"], "main");
    assert_eq!(out["upstream"]["ahead"], 1);
    assert_eq!(out["upstream"]["behind"], 0);
    assert!(out.get("no_upstream").is_none(), "{out}");
}

/// Without a copy on origin the reply says why there is nothing to compare
/// against, so the question still has an answer rather than a dead end.
#[test]
fn status_without_an_upstream_says_why() {
    let ws = GitWorkspace::new();
    let out = ws.read("git_status", json!({"repo": "app", "page": 0}));
    assert!(out.get("upstream").is_none(), "{out}");
    let hint = out["no_upstream"].as_str().unwrap();
    assert!(hint.contains("no origin"), "{hint}");

    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://example.com/acme/app.git",
    ]);
    let out = ws.read("git_status", json!({"repo": "app", "page": 0}));
    let hint = out["no_upstream"].as_str().unwrap();
    assert!(hint.contains("origin has no copy"), "{hint}");
}

/// **Counts cover every changed path, not just the page.** They exist because
/// the model's own totals were measured wrong repeatedly — it should read a
/// number, not count a list.
#[test]
fn status_pages_and_the_counts_cover_everything() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation();
    for i in 0..100 {
        conv.write(&format!("f{i}.txt"), "x\n");
    }
    let first = conv.status();
    assert_eq!(first["counts"]["added"], 100, "{}", first["counts"]);
    assert_eq!(first["paging"]["total"], 100);
    assert_eq!(first["changes"].as_array().unwrap().len(), 80);
    assert_eq!(first["paging"]["next_page"], 1);

    let second = conv.call("git_status", json!({"repo": "app", "page": 1}));
    assert_eq!(second["changes"].as_array().unwrap().len(), 20);
    assert!(second["paging"]["next_page"].is_null());
}

/// **An over-shot page clamps to the last one** rather than returning an
/// empty list the model would read as "nothing there" — a page number is a
/// way out, and it has to stay one.
#[test]
fn a_page_past_the_end_clamps_to_the_last_page() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation();
    conv.write("one.txt", "x\n");
    let out = conv.call("git_status", json!({"repo": "app", "page": 99}));
    assert_eq!(out["paging"]["page"], 0, "{out}");
    assert_eq!(out["changes"].as_array().unwrap().len(), 1);
}

#[test]
fn an_unknown_repo_names_the_ones_that_exist() {
    let ws = GitWorkspace::new();
    let out = ws.read("git_status", json!({"repo": "nope", "page": 0}));
    assert_eq!(out["error"], "unknown_repo", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("app"), "{out}");
}

// ── git_log ──────────────────────────────────────────────────────────────────

#[test]
fn history_comes_back_newest_first_with_full_ids_and_a_count() {
    let ws = GitWorkspace::new();
    let out = ws.read("git_log", json!({"repo": "app", "page": 0}));
    let commits = out["commits"].as_array().unwrap();
    assert_eq!(out["count"], 2, "{out}");
    assert_eq!(out["counted_all"], true);
    assert_eq!(commits[0]["subject"], "say hello properly");
    assert_eq!(commits[1]["subject"], "initial commit");
    assert_eq!(commits[0]["id"], ws.oid("HEAD"));
    assert_eq!(commits[0]["id"].as_str().unwrap().len(), 40, "ids are full");
    assert!(
        commits[0].get("merge_of").is_none(),
        "an ordinary commit's parent is the next entry: {}",
        commits[0]
    );
}

/// **A merge names its parents** — the one entry whose parents say something
/// the order of the listing does not.
#[test]
fn a_merge_lists_its_parents() {
    let ws = GitWorkspace::new();
    let base = ws.oid("HEAD");
    ws.git(&["checkout", "-q", "-b", "side"]);
    ws.write_worktree("side.txt", "side\n");
    let side = ws.commit_all("side work");
    ws.git(&["checkout", "-q", "main"]);
    ws.git(&["merge", "-q", "--no-ff", "-m", "merge side", "side"]);
    let out = ws.read("git_log", json!({"repo": "app", "page": 0}));
    let merge = &out["commits"][0];
    assert_eq!(merge["subject"], "merge side", "{out}");
    assert_eq!(merge["merge_of"], json!([base, side]));
}

#[test]
fn a_commit_carries_an_iso_date_and_its_author() {
    let ws = GitWorkspace::new();
    let a = &ws.read("git_log", json!({"repo": "app", "page": 0}))["commits"][0]["author"];
    assert_eq!(a["name"], "Setup");
    assert_eq!(a["date"], "2023-11-14T22:13:20+00:00");
}

/// **`since` is what answers "how far ahead am I".** Live, the model invented
/// a count rather than using it; the `count` field now hands it the number.
#[test]
fn since_excludes_what_the_other_branch_already_has_and_counts_it() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "feature"]);
    ws.write_worktree("src/feature.rs", "pub fn f() {}\n");
    ws.commit_all("add the feature");

    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("feature"), "since": branch_rev("main"),
               "page": 0}),
    );
    assert_eq!(out["count"], 1, "only the commit main lacks: {out}");
    assert_eq!(out["commits"][0]["subject"], "add the feature");

    // Nothing ahead reads as zero rather than as an absence.
    let none = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("main"), "since": branch_rev("feature"),
               "page": 0}),
    );
    assert_eq!(none["count"], 0, "{none}");
}

#[test]
fn paths_narrow_history_to_the_files_that_changed() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["README.md"], "page": 0}),
    );
    assert_eq!(out["count"], 1, "{out}");
    assert_eq!(out["commits"][0]["subject"], "initial commit");
}

#[test]
fn follow_renames_reaches_back_past_the_rename() {
    let ws = GitWorkspace::new();
    ws.git(&["mv", "src/lib.rs", "src/core.rs"]);
    ws.commit_all("rename lib to core");

    let plain = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/core.rs"], "page": 0}),
    );
    assert_eq!(plain["count"], 1);
    let followed = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/core.rs"], "follow_renames": true, "page": 0}),
    );
    assert_eq!(followed["count"], 3, "{followed}");
}

#[test]
fn follow_renames_refuses_the_combinations_it_cannot_honour() {
    let ws = GitWorkspace::new();
    for args in [
        json!({"repo": "app", "paths": ["a", "b"], "follow_renames": true, "page": 0}),
        json!({"repo": "app", "paths": ["README.md"], "follow_renames": true,
               "since": branch_rev("main"), "page": 0}),
    ] {
        assert_eq!(
            ws.read("git_log", args.clone())["error"],
            "invalid_arguments",
            "{args}"
        );
    }
}

/// **`lines` answers "who last changed these lines", not "who last touched the
/// file".** The fixture's second commit rewrote only line 2 of `src/lib.rs`:
/// for line 1 the file's latest commit is the wrong answer — which is exactly
/// the answer a history listing gave live.
#[test]
fn lines_names_the_commits_behind_those_lines_not_the_files_latest() {
    let ws = GitWorkspace::new();
    let initial = ws.oid("HEAD~1");
    let latest = ws.oid("HEAD");

    let top = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 1, "end": 1},
               "page": 0}),
    );
    assert_eq!(top["count"], 1, "{top}");
    assert_eq!(
        top["commits"][0]["id"], initial,
        "not the file's latest: {top}"
    );
    assert_eq!(top["commits"][0]["subject"], "initial commit");
    assert_eq!(
        top["line_runs"],
        json!([{"from": 1, "to": 1, "commit": initial}])
    );

    let all = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 1, "end": 3},
               "page": 0}),
    );
    assert_eq!(all["count"], 2, "{all}");
    // Newest first, as a history reads.
    assert_eq!(all["commits"][0]["id"], latest);
    assert_eq!(all["commits"][1]["id"], initial);
    assert_eq!(
        all["line_runs"],
        json!([
            {"from": 1, "to": 1, "commit": initial},
            {"from": 2, "to": 2, "commit": latest},
            {"from": 3, "to": 3, "commit": initial},
        ])
    );
}

/// A span running past the end is clamped to the lines that exist; one
/// starting past the end is refused with the file's length, so the next call
/// can get it right.
#[test]
fn lines_past_the_end_clamp_or_say_how_long_the_file_is() {
    let ws = GitWorkspace::new();
    let clamped = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 2, "end": 99},
               "page": 0}),
    );
    assert_eq!(
        clamped["line_runs"].as_array().unwrap().len(),
        2,
        "{clamped}"
    );

    let beyond = ws.read(
        "git_log",
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 50, "end": 60},
               "page": 0}),
    );
    assert_eq!(beyond["error"], "invalid_arguments", "{beyond}");
    assert!(
        beyond["detail"].as_str().unwrap().contains("only 3 line"),
        "{beyond}"
    );
}

#[test]
fn lines_refuses_what_it_cannot_honour() {
    let ws = GitWorkspace::new();
    for args in [
        json!({"repo": "app", "lines": {"start": 1, "end": 2}, "page": 0}),
        json!({"repo": "app", "paths": ["src/lib.rs", "README.md"],
               "lines": {"start": 1, "end": 2}, "page": 0}),
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 1, "end": 2},
               "since": branch_rev("main"), "page": 0}),
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 0, "end": 2},
               "page": 0}),
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 3, "end": 1},
               "page": 0}),
        json!({"repo": "app", "paths": ["src/lib.rs"], "lines": {"start": 1, "end": 500},
               "page": 0}),
    ] {
        assert_eq!(
            ws.read("git_log", args.clone())["error"],
            "invalid_arguments",
            "{args}"
        );
    }
}

#[test]
fn an_unknown_revision_says_so() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("nope"), "page": 0}),
    );
    assert_eq!(out["error"], "unknown_revision", "{out}");
}

/// A field the schema does not define is refused rather than ignored.
#[test]
fn an_unknown_field_is_refused() {
    let ws = GitWorkspace::new();
    let out = ws.read("git_log", json!({"repo": "app", "max_count": 5, "page": 0}));
    assert_eq!(out["error"], "invalid_arguments", "{out}");
}

/// **Every reader's page is required**, as `file_read`'s is: a call that
/// leaves it out is refused rather than quietly served page 0, so the decoder's
/// grammar and the executor agree that there is no unpaged read.
#[test]
fn a_reader_called_without_a_page_is_refused() {
    let ws = GitWorkspace::new();
    for (tool, args) in [
        ("git_status", json!({"repo": "app"})),
        ("git_log", json!({"repo": "app"})),
        ("git_show", json!({"repo": "app", "what": "changes"})),
        ("git_grep", json!({"repo": "app", "pattern": "hello"})),
        ("git_refs", json!({"repo": "app", "kind": "branches"})),
    ] {
        let out = ws.read(tool, args);
        assert_eq!(out["error"], "invalid_arguments", "{tool}: {out}");
    }
}

/// **A history page lists subjects, never bodies.** Twenty-five commits with
/// full bodies measured 11,761 tokens live — in a repository whose messages
/// run to paragraphs, the body is the whole cost of the page, and a listing
/// is for choosing a commit, not reading one.
#[test]
fn a_history_page_carries_subjects_and_no_bodies() {
    let ws = GitWorkspace::new();
    let body = "a paragraph of explanation. ".repeat(40);
    for i in 0..30 {
        ws.write_worktree("churn.txt", &format!("{i}\n"));
        ws.git(&["add", "-A"]);
        ws.git(&["commit", "-q", "-m", &format!("change {i}"), "-m", &body]);
    }
    let out = ws.read("git_log", json!({"repo": "app", "page": 0}));
    let commits = out["commits"].as_array().unwrap();
    assert_eq!(commits.len(), 20, "{}", out["paging"]);
    assert_eq!(out["count"], 32);
    assert_eq!(out["paging"]["next_page"], 1);
    assert_eq!(commits[0]["subject"], "change 29");
    assert!(commits[0].get("message").is_none(), "{}", commits[0]);
    let text = out.to_string();
    assert!(!text.contains("a paragraph of explanation"), "{text}");
    // Twenty entries of two ids, an author and a subject: the page stays in
    // the few-kilobyte range however long the messages behind it run.
    assert!(text.len() < 8 * 1024, "page is {} bytes", text.len());
}

/// **The body is git_show's**, on the first page of a commit's changes — the
/// one place a single commit worth reading is read — and not repeated on the
/// pages after it.
#[test]
fn a_commits_full_message_comes_with_the_first_page_of_its_changes() {
    let ws = GitWorkspace::new();
    for i in 0..70 {
        ws.write_worktree(&format!("f{i}.txt"), "x\n");
    }
    ws.git(&["add", "-A"]);
    ws.git(&[
        "commit",
        "-q",
        "-m",
        "add many files",
        "-m",
        "why they were added",
    ]);

    let first = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "page": 0}),
    );
    let message = first["message"].as_str().unwrap();
    assert!(message.starts_with("add many files"), "{first}");
    assert!(message.contains("why they were added"), "{first}");
    assert_eq!(first["paging"]["next_page"], 1);

    let second = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "page": 1}),
    );
    assert!(second.get("message").is_none(), "{second}");
    assert_eq!(second["subject"], "add many files");
}

// ── revisions ────────────────────────────────────────────────────────────────

/// **`parent` is the escape hatch.** Live, asked for "the previous commit",
/// the model had no id to give, could not revise the arm it had chosen, and
/// fabricated. This arm means the question is always answerable.
#[test]
fn a_parent_revision_needs_no_object_id() {
    let ws = GitWorkspace::new();
    let previous = ws.oid("HEAD~1");

    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "src/lib.rs",
               "rev": parent_rev("HEAD", None), "page": 0}),
    );
    assert_eq!(out["commit"], previous, "{out}");
    assert!(out["content"].as_str().unwrap().contains("\"hi\""));

    // `back` counts further, and a branch name works as the base.
    ws.write_worktree("x.txt", "x\n");
    ws.commit_all("third");
    let two = ws.read(
        "git_log",
        json!({"repo": "app", "rev": parent_rev("main", Some(2)), "page": 0}),
    );
    assert_eq!(two["commits"][0]["subject"], "initial commit", "{two}");
}

/// **A remote branch is named the way it is spoken of.** Measured live: asked
/// how far ahead of `origin/main` it was, the model wrote the right call in
/// prose — `since="origin/main"` — and emitted `since: null`, because the only
/// correct spelling was the full `refs/remotes/origin/main` and no arm fitted
/// what it meant. `null` satisfied the grammar and answered nothing, which is
/// the dead end one level down from the one `parent` closed.
#[test]
fn a_remote_branch_is_named_as_origin_slash_branch() {
    let ws = GitWorkspace::new();
    ws.git(&["update-ref", "refs/remotes/origin/main", "HEAD~1"]);
    let older = ws.oid("HEAD~1");

    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": {"kind": "remote_branch", "name": "origin/main"},
               "page": 0}),
    );
    assert_eq!(out["commits"][0]["id"], older, "{out}");

    // The full ref still works, for a caller that already holds it.
    let full = ws.read(
        "git_log",
        json!({"repo": "app",
               "rev": {"kind": "remote_branch", "name": "refs/remotes/origin/main"},
               "page": 0}),
    );
    assert_eq!(full["commits"][0]["id"], older, "{full}");

    // And this is what the failing question actually needed.
    let ahead = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("main"),
               "since": {"kind": "remote_branch", "name": "origin/main"}, "page": 0}),
    );
    assert_eq!(ahead["count"], 1, "one commit ahead of the remote: {ahead}");
}

/// **`upstream` resolves what a branch tracks**, which is the other half of
/// "how far ahead am I".
#[test]
fn an_upstream_revision_resolves_what_the_branch_tracks() {
    let ws = GitWorkspace::new();
    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://example.com/acme/app.git",
    ]);
    ws.git(&["update-ref", "refs/remotes/origin/main", "HEAD~1"]);
    ws.git(&["config", "branch.main.remote", "origin"]);
    ws.git(&["config", "branch.main.merge", "refs/heads/main"]);

    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("main"), "since": {"kind": "upstream"},
               "page": 0}),
    );
    assert_eq!(out["count"], 1, "{out}");

    // Naming the branch explicitly is equivalent.
    let named = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("main"),
               "since": {"kind": "upstream", "name": "main"}, "page": 0}),
    );
    assert_eq!(named["count"], 1, "{named}");
}

/// A branch with no upstream is the common case on a topic branch, and the
/// error names the form that does work instead of leaving it to be guessed.
#[test]
fn an_upstream_that_does_not_exist_names_the_alternative() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": branch_rev("main"), "since": {"kind": "upstream"},
               "page": 0}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    let detail = out["detail"].as_str().unwrap();
    assert!(detail.contains("no upstream"), "{detail}");
    assert!(
        detail.contains("remote_branch"),
        "it names the way out: {detail}"
    );
}

/// Asking further back than the history goes is an error that says how far it
/// actually goes — a content error with a way forward, not a dead end.
#[test]
fn a_parent_past_the_root_explains_how_far_back_it_goes() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": parent_rev("HEAD", Some(9)), "page": 0}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("only"), "{out}");
}

/// **"The previous commit" of a merge is the mainline's**, not whichever
/// commit on the merged branch happens to sort first.
#[test]
fn a_parent_of_a_merge_is_its_first_parent() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "side"]);
    ws.write_worktree("side.txt", "side\n");
    ws.commit_all("side one");
    ws.write_worktree("side.txt", "side two\n");
    ws.commit_all("side two");
    ws.git(&["checkout", "-q", "main"]);
    ws.write_worktree("main.txt", "main\n");
    let mainline = ws.commit_all("mainline");
    ws.git(&["merge", "-q", "--no-ff", "-m", "merge side", "side"]);

    let out = ws.read(
        "git_log",
        json!({"repo": "app", "rev": parent_rev("HEAD", None), "page": 0}),
    );
    assert_eq!(out["commits"][0]["id"], mainline, "{out}");
}

/// Every other form still resolves, and a kind that needs a name says so
/// rather than failing obscurely.
#[test]
fn each_revision_kind_resolves_or_explains_itself() {
    let ws = GitWorkspace::new();
    ws.git(&["tag", "v1.0"]);
    let head = ws.oid("HEAD");
    for rev in [
        head_rev(),
        branch_rev("main"),
        tag_rev("v1.0"),
        commit_rev(&head),
        json!({"kind": "ref", "name": "refs/heads/main"}),
    ] {
        let out = ws.read(
            "git_log",
            json!({"repo": "app", "rev": rev.clone(), "page": 0}),
        );
        assert_eq!(out["commits"][0]["id"], head, "{rev}: {out}");
    }
    let nameless = ws.read(
        "git_log",
        json!({"repo": "app", "rev": {"kind": "branch"}, "page": 0}),
    );
    assert_eq!(nameless["error"], "invalid_arguments", "{nameless}");
    assert!(
        nameless["detail"].as_str().unwrap().contains("name"),
        "{nameless}"
    );
}

// ── git_show ─────────────────────────────────────────────────────────────────

fn texts(file: &Value, kind: &str) -> Vec<String> {
    file["hunks"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|h| h["lines"].as_array().unwrap())
        .filter(|l| l["kind"] == kind)
        .map(|l| l["text"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn changes_lists_the_files_a_commit_touched() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "page": 0}),
    );
    assert_eq!(out["subject"], "say hello properly", "{out}");
    let changes = out["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0]["path"], "src/lib.rs");
    assert_eq!(changes[0]["status"], "modified");
}

#[test]
fn patch_gives_the_changed_lines() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "patch", "page": 0}),
    );
    let f = &out["files"][0];
    assert_eq!(f["path"], "src/lib.rs", "{out}");
    assert_eq!(texts(f, "added"), vec!["    \"hello\""]);
    assert_eq!(texts(f, "removed"), vec!["    \"hi\""]);
}

/// **A root commit has no parent**, and shows against the empty tree rather
/// than failing.
#[test]
fn the_first_commit_shows_as_all_additions() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "rev": commit_rev(&ws.oid("HEAD~1")),
               "page": 0}),
    );
    let changes = out["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 2, "{out}");
    for c in changes {
        assert_eq!(c["status"], "added");
    }
}

#[test]
fn from_compares_two_branches() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "feature"]);
    ws.write_worktree("src/feature.rs", "pub fn f() {}\n");
    ws.commit_all("add the feature");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes",
               "rev": branch_rev("feature"), "from": branch_rev("main"), "page": 0}),
    );
    let changes = out["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "{out}");
    assert_eq!(changes[0]["path"], "src/feature.rs");
}

/// A rename is one change with both paths, not a delete plus an add.
#[test]
fn a_rename_reports_both_paths() {
    let ws = GitWorkspace::new();
    ws.git(&["mv", "src/lib.rs", "src/core.rs"]);
    ws.commit_all("rename lib to core");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "page": 0}),
    );
    let changes = out["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "{out}");
    assert_eq!(changes[0]["status"], "renamed");
    assert_eq!(changes[0]["path"], "src/core.rs");
    assert_eq!(changes[0]["from_path"], "src/lib.rs");
}

/// **`file` is the mode that kept being reached for with `patch`.** As an
/// enum inside one tool the distinction cannot be got wrong any more.
#[test]
fn file_returns_contents_as_that_revision_holds_them() {
    let ws = GitWorkspace::new();
    let now = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "src/lib.rs", "page": 0}),
    );
    assert!(
        now["content"].as_str().unwrap().contains("\"hello\""),
        "{now}"
    );

    let before = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "src/lib.rs",
               "rev": commit_rev(&ws.oid("HEAD~1")), "page": 0}),
    );
    assert!(
        before["content"].as_str().unwrap().contains("\"hi\""),
        "{before}"
    );
}

/// Reading history does not read the working tree — what distinguishes this
/// from file_read.
#[test]
fn an_uncommitted_edit_is_not_visible_at_head() {
    let ws = GitWorkspace::new();
    ws.write_worktree("src/lib.rs", "pub fn hello() { todo!() }\n");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "src/lib.rs", "rev": head_rev(),
               "page": 0}),
    );
    assert!(!out["content"].as_str().unwrap().contains("todo!"), "{out}");
}

/// **A big file pages rather than truncating**, and the page is a way out.
#[test]
fn a_large_file_pages() {
    let ws = GitWorkspace::new();
    let long: String = (0..900).map(|i| format!("line {i}\n")).collect();
    ws.write_worktree("long.txt", &long);
    ws.commit_all("add a long file");

    let first = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "long.txt", "page": 0}),
    );
    assert_eq!(first["paging"]["total"], 900, "{}", first["paging"]);
    assert_eq!(
        first["content"].as_str().unwrap().lines().count(),
        200,
        "the same 200-line page file_read serves"
    );
    assert_eq!(first["paging"]["next_page"], 1);

    let last = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "long.txt", "page": 4}),
    );
    assert_eq!(last["content"].as_str().unwrap().lines().count(), 100);
    assert!(last["paging"]["next_page"].is_null());
    assert!(last["content"].as_str().unwrap().starts_with("line 800"));
}

#[test]
fn a_binary_file_is_reported_not_mangled() {
    let ws = GitWorkspace::new();
    std::fs::write(ws.repo_dir().join("blob.bin"), [0u8, 159, 146, 150, 0]).unwrap();
    ws.commit_all("add a binary file");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "blob.bin", "page": 0}),
    );
    assert_eq!(out["binary"], true, "{out}");
    assert_eq!(out["bytes"], 5);
    assert!(out.get("content").is_none(), "{out}");
}

#[test]
fn tree_lists_a_directory_at_a_revision() {
    let ws = GitWorkspace::new();
    let root = ws.read(
        "git_show",
        json!({"repo": "app", "what": "tree", "page": 0}),
    );
    let entries = root["entries"].as_array().unwrap();
    let by = |p: &str| entries.iter().find(|e| e["path"] == p).cloned().unwrap();
    assert_eq!(by("README.md")["kind"], "file", "{root}");
    assert_eq!(by("src")["kind"], "directory");

    let src = ws.read(
        "git_show",
        json!({"repo": "app", "what": "tree", "path": "src", "page": 0}),
    );
    let paths: Vec<&str> = src["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, vec!["src/lib.rs"], "{src}");
}

#[test]
fn blame_attributes_each_line_and_follows_renames() {
    let ws = GitWorkspace::new();
    let first = ws.oid("HEAD~1");
    let second = ws.oid("HEAD");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "blame", "path": "src/lib.rs", "page": 0}),
    );
    let lines = out["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 3, "{out}");
    assert_eq!(lines[0]["commit"], first);
    assert_eq!(lines[1]["commit"], second);
    assert_eq!(lines[1]["summary"], "say hello properly");

    ws.git(&["mv", "src/lib.rs", "src/core.rs"]);
    ws.commit_all("rename lib to core");
    let renamed = ws.read(
        "git_show",
        json!({"repo": "app", "what": "blame", "path": "src/core.rs", "page": 0}),
    );
    let hello = renamed["lines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["text"].as_str().unwrap().contains("\"hello\""))
        .unwrap();
    assert_eq!(
        hello["commit"], second,
        "the rename is not the author: {renamed}"
    );
    assert_eq!(hello["from_path"], "src/lib.rs");
}

/// **Blame pages.** It is the one read costing a record per line — ~70 tokens
/// each, with a full commit id, author, date and summary beside the text — and
/// an unbounded whole-file blame measured at seventeen minutes on a live turn.
#[test]
fn blame_pages_a_long_file() {
    let ws = GitWorkspace::new();
    let long: String = (0..400).map(|i| format!("line {i}\n")).collect();
    ws.write_worktree("long.txt", &long);
    ws.commit_all("add a long file");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "blame", "path": "long.txt", "page": 0}),
    );
    assert_eq!(out["paging"]["total"], 400, "{}", out["paging"]);
    assert_eq!(out["lines"].as_array().unwrap().len(), 40);
    assert_eq!(out["paging"]["next_page"], 1);

    // Each page blames its own window of the file, numbered as the file is.
    let last = ws.read(
        "git_show",
        json!({"repo": "app", "what": "blame", "path": "long.txt", "page": 9}),
    );
    let lines = last["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 40, "{last}");
    assert_eq!(lines[0]["line"], 361);
    assert_eq!(lines[0]["text"], "line 360");
    assert_eq!(lines[39]["line"], 400);
    assert_eq!(last["paging"]["next_page"], Value::Null);
}

/// **Patch pages are bounded and disjoint** even when one hunk is longer
/// than a page: it is cut at the boundary, and each piece says where in the
/// file it starts. Whole overlapping hunks came back on every page they
/// touched — one 900-line hunk, in full, five times.
#[test]
fn a_hunk_longer_than_a_page_is_split_across_pages() {
    let ws = GitWorkspace::new();
    let long: String = (0..450).map(|i| format!("line {i}\n")).collect();
    ws.write_worktree("long.txt", &long);
    ws.commit_all("add a long file");

    let mut texts = Vec::new();
    for page in 0..3 {
        let out = ws.read(
            "git_show",
            json!({"repo": "app", "what": "patch", "page": page}),
        );
        assert_eq!(out["paging"]["total"], 450, "{}", out["paging"]);
        let hunks = out["files"][0]["hunks"].as_array().unwrap();
        assert_eq!(hunks.len(), 1, "{out}");
        assert_eq!(
            hunks[0]["new_start"],
            1 + page * 200,
            "{}",
            hunks[0]["new_start"]
        );
        let lines = hunks[0]["lines"].as_array().unwrap();
        assert_eq!(lines.len(), if page < 2 { 200 } else { 50 });
        texts.extend(
            lines
                .iter()
                .map(|l| l["text"].as_str().unwrap().to_string()),
        );
    }
    let expected: Vec<String> = (0..450).map(|i| format!("line {i}")).collect();
    assert_eq!(texts, expected, "every line exactly once, in order");
}

/// **`paths` narrows `changes`**, as it does `patch`.
#[test]
fn changes_are_limited_to_the_paths_asked_for() {
    let ws = GitWorkspace::new();
    ws.write_worktree("a.txt", "a\n");
    ws.write_worktree("b.txt", "b\n");
    ws.commit_all("two files");

    let all = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "page": 0}),
    );
    assert_eq!(all["changes"].as_array().unwrap().len(), 2, "{all}");
    let one = ws.read(
        "git_show",
        json!({"repo": "app", "what": "changes", "paths": ["b.txt"], "page": 0}),
    );
    let changes = one["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "{one}");
    assert!(changes[0].to_string().contains("b.txt"), "{one}");
}

/// A tree entry carries a full object id, so a directory pages at a size that
/// costs what a file page does rather than at the count a path list could
/// afford.
#[test]
fn tree_pages_a_large_directory() {
    let ws = GitWorkspace::new();
    for i in 0..120 {
        ws.write_worktree(&format!("many/f{i:03}.txt"), "x\n");
    }
    ws.commit_all("add a large directory");
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "tree", "path": "many", "page": 0}),
    );
    assert_eq!(out["paging"]["total"], 120, "{}", out["paging"]);
    assert_eq!(out["entries"].as_array().unwrap().len(), 50);
    assert_eq!(out["paging"]["next_page"], 1);
}

#[test]
fn a_mode_that_needs_a_path_says_so() {
    let ws = GitWorkspace::new();
    for what in ["file", "blame"] {
        let out = ws.read("git_show", json!({"repo": "app", "what": what, "page": 0}));
        assert_eq!(out["error"], "invalid_arguments", "{what}: {out}");
        assert!(out["detail"].as_str().unwrap().contains("path"), "{out}");
    }
}

#[test]
fn a_path_the_revision_does_not_hold_says_so() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_show",
        json!({"repo": "app", "what": "file", "path": "nope.txt", "page": 0}),
    );
    assert_eq!(out["error"], "unknown_revision", "{out}");
}

// ── protected paths ──────────────────────────────────────────────────────────

/// **A key committed once lives in the object store forever**, so the live
/// path's rule has to reach backwards through history.
#[test]
fn a_protected_path_is_refused_in_every_mode_and_at_every_revision() {
    let ws = GitWorkspace::new();
    ws.write_worktree("secrets/keys.yaml", "token: ghp_abcd\n");
    ws.commit_all("add a secret");
    let had_it = ws.oid("HEAD");
    ws.git(&["rm", "-q", "secrets/keys.yaml"]);
    ws.commit_all("remove the secret");

    for args in [
        json!({"repo": "app", "what": "file", "path": "secrets/keys.yaml",
               "rev": commit_rev(&had_it), "page": 0}),
        json!({"repo": "app", "what": "blame", "path": "secrets/keys.yaml",
               "rev": commit_rev(&had_it), "page": 0}),
    ] {
        let out = ws.read("git_show", args.clone());
        assert_eq!(out["error"], "invalid_arguments", "{args}: {out}");
        assert!(!out.to_string().contains("ghp_abcd"), "{out}");
    }

    // A tree listing leaves the folder out entirely.
    let tree = ws.read(
        "git_show",
        json!({"repo": "app", "what": "tree", "rev": commit_rev(&had_it), "page": 0}),
    );
    assert!(!tree.to_string().contains("secrets"), "{tree}");

    // And a patch of the commit that added it carries none of its content.
    let patch = ws.read(
        "git_show",
        json!({"repo": "app", "what": "patch", "rev": commit_rev(&had_it), "page": 0}),
    );
    assert!(!patch.to_string().contains("ghp_abcd"), "{patch}");
}

// ── git_grep ─────────────────────────────────────────────────────────────────

#[test]
fn a_match_reports_its_file_line_and_text_with_a_count() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "pub fn hello", "page": 0}),
    );
    let m = &out["matches"][0];
    assert_eq!(m["path"], "src/lib.rs", "{out}");
    assert_eq!(m["line"], 1);
    assert_eq!(out["count"], 1);
    assert_eq!(out["files_matched"], 1);
}

/// **Searching an older revision finds what the current one lost** — the
/// whole reason this is not file_grep.
#[test]
fn an_older_revision_searches_its_own_content() {
    let ws = GitWorkspace::new();
    let now = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "\"hi\"", "fixed": true, "page": 0}),
    );
    assert_eq!(now["count"], 0, "{now}");
    let before = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "\"hi\"", "fixed": true,
               "rev": commit_rev(&ws.oid("HEAD~1")), "page": 0}),
    );
    assert_eq!(before["count"], 1, "{before}");
}

#[test]
fn fixed_turns_the_pattern_into_a_literal() {
    let ws = GitWorkspace::new();
    ws.write_worktree("re.txt", "a.c\nabc\n");
    ws.commit_all("add re");
    let regex = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "a.c", "paths": ["re.txt"], "page": 0}),
    );
    assert_eq!(regex["count"], 2, "{regex}");
    let literal = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "a.c", "fixed": true, "paths": ["re.txt"],
               "page": 0}),
    );
    assert_eq!(literal["count"], 1, "{literal}");
}

#[test]
fn a_single_file_is_capped_so_other_files_still_appear() {
    let ws = GitWorkspace::new();
    let flood: String = (0..80).map(|i| format!("needle {i}\n")).collect();
    ws.write_worktree("generated.txt", &flood);
    ws.write_worktree("real.txt", "needle here\n");
    ws.commit_all("add both");
    let out = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "needle", "max_per_file": 3, "page": 0}),
    );
    let matches = out["matches"].as_array().unwrap();
    assert_eq!(
        matches
            .iter()
            .filter(|m| m["path"] == "generated.txt")
            .count(),
        3
    );
    assert!(matches.iter().any(|m| m["path"] == "real.txt"), "{out}");
    assert_eq!(out["files_matched"], 2);
}

/// **A very long line is clipped and says so**, exactly as file_grep clips —
/// one hit in a minified file is otherwise the whole bundle.
#[test]
fn a_very_long_matching_line_is_clipped() {
    let ws = GitWorkspace::new();
    let bundle = format!("needle{}\n", "x".repeat(5_000));
    ws.write_worktree("bundle.min.js", &bundle);
    ws.commit_all("add a minified bundle");
    let out = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "needle", "page": 0}),
    );
    let text = out["matches"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("needle"), "{text}");
    assert!(text.ends_with("[line truncated]"), "{text}");
    assert!(text.chars().count() < 500, "{} chars", text.chars().count());
}

/// A wildcard search must not sweep up a protected file's lines — the call
/// never named it.
#[test]
fn a_search_with_no_paths_does_not_return_a_protected_files_lines() {
    let ws = GitWorkspace::new();
    ws.write_worktree("secrets/keys.yaml", "token: ghp_needle_value\n");
    ws.write_worktree("ok.txt", "needle in the open\n");
    ws.commit_all("add both");
    let out = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "needle", "page": 0}),
    );
    assert!(!out.to_string().contains("ghp_needle_value"), "{out}");
    assert_eq!(out["files_matched"], 1, "{out}");
}

#[test]
fn a_pattern_that_matches_nothing_is_an_empty_result_not_an_error() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_grep",
        json!({"repo": "app", "pattern": "zzzznotpresent", "page": 0}),
    );
    assert!(out.get("error").is_none(), "{out}");
    assert_eq!(out["count"], 0);
}

// ── git_refs ─────────────────────────────────────────────────────────────────

#[test]
fn branches_list_with_the_current_one_marked_and_a_count() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "feature"]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "branches", "page": 0}),
    );
    assert_eq!(out["current"], "main", "{out}");
    assert_eq!(out["count"], 2);
    let listed = out["branches"].as_array().unwrap();
    let main = listed.iter().find(|b| b["name"] == "main").unwrap();
    assert_eq!(main["head"], true);
    assert_eq!(main["id"], ws.oid("HEAD"));
    assert_eq!(
        listed.iter().find(|b| b["name"] == "feature").unwrap()["head"],
        false
    );
}

/// Ahead and behind are the question this answers; live, the model invented
/// the number instead.
#[test]
fn a_tracking_branch_reports_how_far_ahead_and_behind() {
    let ws = GitWorkspace::new();
    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://example.com/acme/app.git",
    ]);
    ws.git(&["update-ref", "refs/remotes/origin/main", "HEAD~1"]);
    ws.git(&["config", "branch.main.remote", "origin"]);
    ws.git(&["config", "branch.main.merge", "refs/heads/main"]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "branches", "page": 0}),
    );
    let up = &out["branches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["name"] == "main")
        .unwrap()["upstream"];
    assert_eq!(up["remote"], "origin", "{out}");
    assert_eq!(up["ahead"], 1);
    assert_eq!(up["behind"], 0);
}

#[test]
fn a_detached_head_names_no_current_branch() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "--detach", "HEAD"]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "branches", "page": 0}),
    );
    assert!(out.get("current").is_none(), "{out}");
}

#[test]
fn tags_tell_annotated_from_lightweight() {
    let ws = GitWorkspace::new();
    ws.git(&["tag", "v1.0"]);
    ws.git(&["tag", "-a", "v2.0", "-m", "release two"]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "tags", "page": 0}),
    );
    assert_eq!(out["count"], 2, "{out}");
    let tags = out["tags"].as_array().unwrap();
    let light = tags.iter().find(|t| t["name"] == "v1.0").unwrap();
    let annotated = tags.iter().find(|t| t["name"] == "v2.0").unwrap();
    assert_eq!(light["annotated"], false);
    assert_eq!(light["id"], light["target"]);
    assert_eq!(annotated["annotated"], true);
    assert_ne!(annotated["id"], annotated["target"]);
    assert_eq!(annotated["target"], ws.oid("HEAD"));
}

#[test]
fn remotes_list_with_credentials_redacted() {
    let ws = GitWorkspace::new();
    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://someone:ghp_verysecrettokenvalue@example.com/acme/app.git",
    ]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "remotes", "page": 0}),
    );
    assert_eq!(out["count"], 1, "{out}");
    assert!(
        !out.to_string().contains("ghp_verysecrettokenvalue"),
        "{out}"
    );
    assert!(out["remotes"][0]["fetch_url"]
        .as_str()
        .unwrap()
        .contains("example.com/acme/app.git"));
}

/// **Remote branches carry their remotes' URLs.** Asked where a repository
/// pushes, the model listed remote branches rather than remotes, found no URL,
/// and answered from a typo in the README — so this kind answers it too.
#[test]
fn remote_branches_list_what_the_last_fetch_left_and_where_it_came_from() {
    let ws = GitWorkspace::new();
    ws.git(&[
        "remote",
        "add",
        "origin",
        "https://example.com/acme/app.git",
    ]);
    ws.git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "remote_branches", "page": 0}),
    );
    assert_eq!(out["count"], 1, "{out}");
    assert_eq!(out["remote_branches"][0]["remote"], "origin");
    assert_eq!(out["remote_branches"][0]["branch"], "main");
    assert_eq!(out["remotes"][0]["name"], "origin");
    assert_eq!(
        out["remotes"][0]["fetch_url"],
        "https://example.com/acme/app.git"
    );
}

/// A name about a remote is, to the model, a question about the remote — so
/// `git_remote_update` answers with the remotes rather than attempting a
/// fetch the conversation may not be allowed to make.
#[test]
fn a_remote_lookup_name_reaches_the_refs_listing() {
    for name in ["git_remote_update", "git_remote", "remote_url"] {
        let tool = find(name).unwrap();
        assert_eq!(tool.name, "git_refs", "{name}");
    }
}

#[test]
fn refs_page() {
    let ws = GitWorkspace::new();
    for i in 0..70 {
        ws.git(&["branch", &format!("b{i}")]);
    }
    let out = ws.read(
        "git_refs",
        json!({"repo": "app", "kind": "branches", "page": 0}),
    );
    assert_eq!(out["count"], 71, "{}", out["count"]);
    assert_eq!(out["branches"].as_array().unwrap().len(), 40);
    assert_eq!(out["paging"]["next_page"], 1);
}
