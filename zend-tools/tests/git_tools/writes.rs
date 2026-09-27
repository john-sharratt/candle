//! The local writers: git_commit and git_ref.
//!
//! Two properties are asserted throughout, because they are what make these
//! safe to offer at all:
//!
//! - **The developer's working tree is never touched**, and a branch that is
//!   checked out cannot be moved.
//! - **Comprehensive cannot run either of them** — each tool's last test
//!   calls it on the read-only context and asserts the refusal, so a tool
//!   that lost its `DiskWrite` declaration would fail here.

use serde_json::{json, Value};
use zend_tools::registry::find;

use crate::harness::{branch_rev, commit_rev, parent_rev, GitWorkspace};

fn commit(ws: &GitWorkspace, args: Value) -> Value {
    ws.write("git_commit", args)
}

fn git_ref(ws: &GitWorkspace, args: Value) -> Value {
    ws.write("git_ref", args)
}

// ── git_commit: from files ───────────────────────────────────────────────────

/// **`take` is the arm that exists so nothing has to be reproduced from
/// memory.** The edit is already in the conversation's files; the commit
/// picks it up — and never what the repository's folder holds instead, which
/// is the sandbox's, not the conversation's.
#[test]
fn take_commits_the_file_as_the_conversation_holds_it() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    ws.write_worktree("src/lib.rs", "a job left this in the folder\n");
    let ctx = ws.mutable_ctx();
    ctx.files
        .repo("app")
        .unwrap()
        .write("src/lib.rs", "pub fn hello() -> u8 {\n    7\n}\n".into())
        .unwrap();

    let out = find("git_commit").unwrap().call(
        &ctx,
        &json!({"repo": "app", "branch": "work", "from": "files", "message": "take the edit",
               "changes": [{"action": "take", "path": "src/lib.rs"}]}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(
        ws.git(&["show", "work:src/lib.rs"]),
        "pub fn hello() -> u8 {\n    7\n}\n"
    );
    assert_eq!(out["author"], "Setup <setup@example.com>");
}

#[test]
fn write_replaces_the_file_with_the_given_content() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "add a greeting",
               "changes": [{"action": "write", "path": "GREETING.md",
                            "content": "hello there\n"}]}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(ws.git(&["show", "work:GREETING.md"]), "hello there\n");
}

/// `write` without content points at `take` rather than failing blankly —
/// the error is a way forward, not a wall.
#[test]
fn write_without_content_names_take_as_the_alternative() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "m",
               "changes": [{"action": "write", "path": "x.txt"}]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("take"), "{out}");
}

#[test]
fn delete_removes_the_file_from_the_new_tree() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "drop the readme",
               "changes": [{"action": "delete", "path": "README.md"}]}),
    );
    let listed = ws.git(&["ls-tree", "--name-only", "work"]);
    assert!(!listed.contains("README.md"), "{listed}");
}

#[test]
fn one_commit_can_carry_several_files_and_set_a_mode() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "add scripts",
               "changes": [
                   {"action": "write", "path": "run.sh", "content": "#!/bin/sh\necho hi\n",
                    "executable": true},
                   {"action": "write", "path": "notes.md", "content": "notes\n"},
                   {"action": "delete", "path": "README.md"}]}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert!(ws.git(&["ls-tree", "work"]).contains("100755"));
}

/// **The developer's working tree is never touched.**
#[test]
fn an_uncommitted_edit_on_disk_survives_a_commit() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    ws.write_worktree("README.md", "# app\n\nwork in progress\n");
    commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "unrelated",
               "changes": [{"action": "write", "path": "other.txt", "content": "x\n"}]}),
    );
    assert_eq!(
        std::fs::read_to_string(ws.repo_dir().join("README.md")).unwrap(),
        "# app\n\nwork in progress\n"
    );
    assert_eq!(ws.oid("HEAD"), ws.oid("main"), "HEAD must not have moved");
}

/// **Committing your changes on your branch clears them**: the files now read
/// from the branch itself, nothing is left uncommitted, and the repository's
/// folder — the sandbox's — is not what was committed from.
#[test]
fn committing_your_changes_on_your_branch_clears_them() {
    let ws = GitWorkspace::new();
    ws.write_worktree("README.md", "# left on disk by a job\n");
    let conv = ws.conversation();
    conv.write("README.md", "# app\n\ncommitted.\n");
    conv.write("NOTES.md", "notes\n");
    conv.delete("src/lib.rs");

    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "changes", "message": "my work"}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(out["branch"], "main");
    assert_eq!(out["on"], "local", "no origin here");
    assert_eq!(ws.oid("main"), out["commit"].as_str().unwrap());
    assert_eq!(ws.git(&["show", "main:README.md"]), "# app\n\ncommitted.\n");
    assert_eq!(ws.git(&["show", "main:NOTES.md"]), "notes\n");
    assert!(ws
        .git(&["ls-tree", "--name-only", "-r", "main"])
        .lines()
        .all(|l| l != "src/lib.rs"));

    let status = conv.status();
    assert_eq!(status["clean"], true, "{status}");
    assert_eq!(
        conv.read("README.md").as_deref(),
        Some("# app\n\ncommitted.\n")
    );

    let again = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "changes", "message": "nothing"}),
    );
    assert_eq!(again["error"], "invalid_arguments", "{again}");
}

/// **Committing some files leaves the rest uncommitted.**
#[test]
fn committing_some_files_leaves_the_rest() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation();
    conv.write("a.txt", "a\n");
    conv.write("b.txt", "b\n");
    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "files", "message": "just a",
               "changes": [{"action": "take", "path": "a.txt"}]}),
    );
    assert_eq!(out["applied"], true, "{out}");
    let status = conv.status();
    assert_eq!(status["counts"]["added"], 1, "{status}");
    assert_eq!(status["changes"][0]["path"], "b.txt");
}

/// **`expected_head` is optional, and both paths work.** Omitted, the tip is
/// read here and still swapped atomically; given and stale, it is refused.
#[test]
fn expected_head_is_optional_and_a_stale_one_is_refused() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);

    let without = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "no guard",
               "changes": [{"action": "write", "path": "a.txt", "content": "a\n"}]}),
    );
    assert_eq!(without["applied"], true, "{without}");

    let now = ws.oid("work");
    let matching = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "guarded",
               "expected_head": now,
               "changes": [{"action": "write", "path": "b.txt", "content": "b\n"}]}),
    );
    assert_eq!(matching["applied"], true, "{matching}");

    let after = ws.oid("work");
    let stale = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "should not land",
               "expected_head": now,
               "changes": [{"action": "write", "path": "c.txt", "content": "c\n"}]}),
    );
    assert_eq!(stale["error"], "stale_ref", "{stale}");
    assert_eq!(ws.oid("work"), after, "the branch must not have moved");
}

#[test]
fn committing_on_a_branch_that_does_not_exist_points_at_git_switch() {
    let ws = GitWorkspace::new();
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "nope", "from": "files", "message": "m",
               "changes": [{"action": "write", "path": "x.txt", "content": "x\n"}]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("git_switch"),
        "{out}"
    );
}

/// A protected path cannot be committed — committing is not a way to launder
/// a secret into history.
#[test]
fn a_secrets_path_cannot_be_committed() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "sneak",
               "changes": [{"action": "write", "path": "secrets/token.txt",
                            "content": "ghp_abc\n"}]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("secrets"), "{out}");
}

/// **`take` reads nothing outside the repository.** A drive-qualified path
/// is a valid repository path, and joined onto the checkout it would
/// replace it on Windows — this one names a file that really exists.
#[test]
fn take_cannot_reach_outside_the_repository() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let before = ws.oid("work");
    let outside = ws.repo_dir().parent().unwrap().join("outside.txt");
    std::fs::write(&outside, "private\n").unwrap();
    let spelled = outside.to_string_lossy().replace('\\', "/");
    let spelled = spelled.trim_start_matches('/');

    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "files", "message": "reach out",
               "changes": [{"action": "take", "path": spelled}]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert_eq!(ws.oid("work"), before, "nothing may have been committed");
}

// ── git_commit: from patch ───────────────────────────────────────────────────

const PATCH: &str = concat!(
    "diff --git a/README.md b/README.md\n",
    "--- a/README.md\n",
    "+++ b/README.md\n",
    "@@ -1,3 +1,4 @@\n",
    " # app\n",
    " \n",
    " the app.\n",
    "+patched line\n",
);

#[test]
fn a_patch_becomes_a_commit_on_the_branch() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "patch", "patch": PATCH,
               "message": "apply the patch"}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert!(ws.git(&["show", "work:README.md"]).contains("patched line"));
}

/// A patch that does not apply is an outcome, not an error — the model gets
/// git's reason and can decide what to do, and nothing moved.
#[test]
fn a_patch_that_does_not_apply_reports_why_and_changes_nothing() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let before = ws.oid("work");
    let wrong = concat!(
        "--- a/README.md\n+++ b/README.md\n@@ -1,2 +1,3 @@\n",
        " this context is not in the file\n nor is this\n+added\n",
    );
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "patch", "patch": wrong,
               "message": "no"}),
    );
    assert!(
        out.get("error").is_none(),
        "a rejection is not an error: {out}"
    );
    assert_eq!(out["applied"], false, "{out}");
    assert!(!out["reason"].as_str().unwrap().is_empty());
    assert_eq!(ws.oid("work"), before);
}

/// **A patch is no way round the `secrets` rule** that `from: files`
/// enforces.
#[test]
fn a_patch_cannot_commit_a_secrets_path() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let before = ws.oid("work");
    let sneak = concat!(
        "diff --git a/secrets/token.txt b/secrets/token.txt\n",
        "new file mode 100644\n",
        "--- /dev/null\n",
        "+++ b/secrets/token.txt\n",
        "@@ -0,0 +1 @@\n",
        "+ghp_abc\n",
    );
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "patch", "patch": sneak,
               "message": "sneak"}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("secrets"), "{out}");
    assert_eq!(ws.oid("work"), before);
}

// ── git_commit: cherry-pick and revert ───────────────────────────────────────

#[test]
fn a_cherry_pick_lands_the_change_with_its_original_message() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "side", "HEAD~1"]);
    ws.write_worktree("fix.txt", "the fix\n");
    let fix = ws.commit_all("fix the thing");
    ws.git(&["checkout", "-q", "main"]);
    ws.git(&["branch", "release"]);

    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "release", "from": "cherry_pick", "commit": fix}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(ws.git(&["show", "release:fix.txt"]), "the fix\n");
    assert_eq!(
        ws.git(&["log", "-1", "--format=%s", "release"]).trim(),
        "fix the thing"
    );
}

#[test]
fn a_revert_undoes_the_change_and_says_what_it_reverted() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let bad = ws.oid("HEAD");
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "revert", "commit": bad}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert!(ws.git(&["show", "work:src/lib.rs"]).contains("\"hi\""));
    assert!(ws.git(&["log", "-1", "--format=%B", "work"]).contains(&bad));
}

/// **A conflict is reported with its paths and commits nothing** — there is
/// no half-finished state to clean up, unlike git's own cherry-pick.
#[test]
fn a_conflict_names_the_paths_and_leaves_the_branch_alone() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "side", "HEAD~1"]);
    ws.write_worktree(
        "src/lib.rs",
        "pub fn hello() -> &'static str {\n    \"side\"\n}\n",
    );
    let side = ws.commit_all("say side");
    ws.git(&["checkout", "-q", "main"]);
    ws.git(&["branch", "work"]);
    let before = ws.oid("work");

    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "cherry_pick", "commit": side}),
    );
    assert!(
        out.get("error").is_none(),
        "a conflict is not an error: {out}"
    );
    assert_eq!(out["applied"], false, "{out}");
    assert_eq!(out["conflicts"][0], "src/lib.rs");
    assert_eq!(ws.oid("work"), before);
    assert!(!ws.repo_dir().join(".git/CHERRY_PICK_HEAD").exists());
}

#[test]
fn a_merge_commit_cannot_be_replayed_and_says_why() {
    let ws = GitWorkspace::new();
    ws.git(&["checkout", "-q", "-b", "side", "HEAD~1"]);
    ws.write_worktree("side.txt", "side\n");
    ws.commit_all("side work");
    ws.git(&["checkout", "-q", "main"]);
    ws.git(&["merge", "--no-ff", "-q", "-m", "merge side", "side"]);
    let merge = ws.oid("HEAD");
    ws.git(&["branch", "work"]);
    let out = commit(
        &ws,
        json!({"repo": "app", "branch": "work", "from": "cherry_pick", "commit": merge}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("merge"), "{out}");
}

/// Each source names what it is missing rather than failing obscurely.
#[test]
fn each_source_names_what_it_needs() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    for (args, want) in [
        (
            json!({"repo": "app", "branch": "work", "from": "files", "message": "m"}),
            "changes",
        ),
        (
            json!({"repo": "app", "branch": "work", "from": "patch", "message": "m"}),
            "patch",
        ),
        (
            json!({"repo": "app", "branch": "work", "from": "cherry_pick"}),
            "commit",
        ),
        (
            json!({"repo": "app", "branch": "work", "from": "files",
                   "changes": [{"action": "take", "path": "README.md"}]}),
            "message",
        ),
    ] {
        let out = commit(&ws, args.clone());
        assert_eq!(out["error"], "invalid_arguments", "{args}: {out}");
        assert!(
            out["detail"].as_str().unwrap().contains(want),
            "{args}: should mention {want}: {out}"
        );
    }
}

#[test]
fn a_comprehensive_context_cannot_commit() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work"]);
    let before = ws.oid("work");
    let out = ws.read(
        "git_commit",
        json!({"repo": "app", "branch": "work", "from": "files", "message": "no",
               "changes": [{"action": "write", "path": "x.txt", "content": "x\n"}]}),
    );
    assert_eq!(out["error"], "not_permitted", "{out}");
    assert_eq!(ws.oid("work"), before, "nothing was written");
}

// ── git_ref ──────────────────────────────────────────────────────────────────

#[test]
fn a_branch_is_created_at_head_by_default() {
    let ws = GitWorkspace::new();
    let out = git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "feature/login"}),
    );
    assert_eq!(out["action"], "created", "{out}");
    assert_eq!(ws.oid("feature/login"), ws.oid("HEAD"));
}

/// A branch can start anywhere a revision can name — including a `parent`,
/// which needs no object id.
#[test]
fn a_branch_can_start_at_a_parent_revision() {
    let ws = GitWorkspace::new();
    let older = ws.oid("HEAD~1");
    let out = git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "from-the-start",
               "at": parent_rev("HEAD", None)}),
    );
    assert_eq!(out["id"], older, "{out}");
    assert_eq!(ws.oid("from-the-start"), older);
}

/// **Creating a branch changes no file on disk.** This moves a pointer.
#[test]
fn creating_a_branch_does_not_switch_the_working_tree() {
    let ws = GitWorkspace::new();
    let before = std::fs::read_to_string(ws.repo_dir().join("src/lib.rs")).unwrap();
    git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "elsewhere",
               "at": commit_rev(&ws.oid("HEAD~1"))}),
    );
    assert_eq!(
        ws.git(&["rev-parse", "--abbrev-ref", "HEAD"]).trim(),
        "main"
    );
    assert_eq!(
        std::fs::read_to_string(ws.repo_dir().join("src/lib.rs")).unwrap(),
        before
    );
}

/// **`expected` is optional and both paths work**: omitted, the value is read
/// here and swapped atomically; given and stale, the move is refused.
#[test]
fn moving_a_branch_works_without_expected_and_refuses_a_stale_one() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "work", "HEAD~1"]);
    let start = ws.oid("work");

    let moved = git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "move", "name": "work",
               "at": branch_rev("main")}),
    );
    assert_eq!(moved["action"], "moved", "{moved}");
    assert_eq!(ws.oid("work"), ws.oid("main"));

    let stale = git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "move", "name": "work",
               "at": commit_rev(&start), "expected": start}),
    );
    assert_eq!(stale["error"], "stale_ref", "{stale}");
}

#[test]
fn a_branch_is_deleted_and_reports_what_it_held() {
    let ws = GitWorkspace::new();
    ws.git(&["branch", "doomed"]);
    let at = ws.oid("doomed");
    let out = git_ref(
        &ws,
        json!({"repo": "app", "kind": "branch", "action": "delete", "name": "doomed"}),
    );
    assert_eq!(out["action"], "deleted", "{out}");
    assert_eq!(out["previous"], at);
    assert!(!ws.git(&["branch"]).contains("doomed"));
}

/// **The branch you are on is neither deleted nor moved by `git_ref`** —
/// switch away first, or move it with `git_reset`, which moves your files
/// with it.
#[test]
fn the_branch_you_are_on_is_neither_deleted_nor_moved_here() {
    let ws = GitWorkspace::new();
    let at = ws.oid("main");
    let conv = ws.conversation();
    let out = conv.call(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "delete", "name": "main"}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("git_switch"),
        "{out}"
    );
    assert_eq!(ws.oid("main"), at, "main must be untouched");

    let back = ws.oid("HEAD~1");
    let out = conv.call(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "move", "name": "main",
               "at": commit_rev(&back), "expected": at}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("git_reset"),
        "{out}"
    );
    assert_eq!(ws.oid("main"), at, "main must be untouched");
}

#[test]
fn a_message_is_what_makes_a_tag_annotated() {
    let ws = GitWorkspace::new();
    let head = ws.oid("HEAD");
    let light = git_ref(
        &ws,
        json!({"repo": "app", "kind": "tag", "action": "create", "name": "v1.0"}),
    );
    assert_eq!(light["id"], head, "{light}");
    assert_eq!(light["target"], head);

    let annotated = git_ref(
        &ws,
        json!({"repo": "app", "kind": "tag", "action": "create", "name": "v2.0",
               "message": "release two"}),
    );
    assert_eq!(annotated["target"], head, "{annotated}");
    assert_ne!(annotated["id"], annotated["target"]);
    assert_eq!(
        ws.git(&["cat-file", "-t", annotated["id"].as_str().unwrap()])
            .trim(),
        "tag"
    );
}

#[test]
fn a_tag_is_deleted_and_moving_one_explains_the_alternative() {
    let ws = GitWorkspace::new();
    ws.git(&["tag", "v1.0"]);
    let moved = git_ref(
        &ws,
        json!({"repo": "app", "kind": "tag", "action": "move", "name": "v1.0"}),
    );
    assert_eq!(moved["error"], "invalid_arguments", "{moved}");
    assert!(
        moved["detail"].as_str().unwrap().contains("delete"),
        "{moved}"
    );

    let deleted = git_ref(
        &ws,
        json!({"repo": "app", "kind": "tag", "action": "delete", "name": "v1.0"}),
    );
    assert_eq!(deleted["action"], "deleted", "{deleted}");
    assert!(!ws.git(&["tag", "-l"]).contains("v1.0"));
}

#[test]
fn an_invalid_ref_name_is_refused() {
    let ws = GitWorkspace::new();
    for name in ["has space", "--force", "ends.lock", "back\\slash"] {
        let out = git_ref(
            &ws,
            json!({"repo": "app", "kind": "branch", "action": "create", "name": name}),
        );
        assert_eq!(out["error"], "invalid_arguments", "{name}: {out}");
    }
}

#[test]
fn a_comprehensive_context_cannot_change_a_ref() {
    let ws = GitWorkspace::new();
    let out = ws.read(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "nope"}),
    );
    assert_eq!(out["error"], "not_permitted", "{out}");
    assert!(!ws.git(&["branch"]).contains("nope"));
}
