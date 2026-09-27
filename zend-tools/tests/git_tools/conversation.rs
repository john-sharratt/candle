//! A conversation working in a repository: the branch it is on, its
//! uncommitted work, and origin as the record every write lands on.
//!
//! Each test holds one conversation — one context kept across calls — so what
//! it writes stays its own uncommitted work and the branch it switches to
//! stays its branch.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::harness::{branch_rev, git_in, parent_rev, remote_oid, Conversation, GitWorkspace};

const HI: &str = "pub fn hello() -> &'static str {\n    \"hi\"\n}\n";
const HELLO: &str = "pub fn hello() -> &'static str {\n    \"hello\"\n}\n";

/// A workspace whose `main` is on a fresh origin.
fn on_origin() -> (GitWorkspace, PathBuf) {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    ws.git(&["push", "-q", "origin", "main"]);
    (ws, origin)
}

/// Another developer pushes `file` on `branch` to origin.
fn someone_else_pushes(ws: &GitWorkspace, origin: &Path, branch: &str, file: &str) -> String {
    let other = ws.other_clone(origin);
    git_in(&other, &["checkout", "-q", "-B", branch]);
    std::fs::write(other.join(file), "theirs\n").unwrap();
    git_in(&other, &["add", "-A"]);
    git_in(&other, &["commit", "-q", "-m", "their work"]);
    git_in(&other, &["push", "-q", "origin", branch]);
    git_in(&other, &["rev-parse", "HEAD"]).trim().to_string()
}

// ── Writes land on origin ────────────────────────────────────────────────────

/// **A commit lands on origin and the local branch follows it there.**
#[test]
fn a_commit_lands_on_origin() {
    let (ws, origin) = on_origin();
    let before = ws.oid("main");
    let conv = ws.conversation();
    conv.write("mine.txt", "mine\n");
    let out = commit(&conv, "mine");
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(out["on"], "origin");
    assert_eq!(out["parent"], before);
    let commit = out["commit"].as_str().unwrap();
    assert_eq!(remote_oid(&origin, "refs/heads/main"), commit);
    assert_eq!(ws.oid("main"), commit);
    let status = conv.status();
    assert_eq!(status["clean"], true, "{status}");
    assert_eq!(status["head"], commit);
}

/// **Someone else pushed first: the commit is refused, with nothing written
/// anywhere** — origin keeps their commit, the conversation its files — and
/// says to merge. git_merge brings their commit into the conversation's
/// files, and the commit then lands on top of it.
#[test]
fn a_refused_commit_merges_and_lands_on_top() {
    let (ws, origin) = on_origin();
    let conv = ws.conversation();
    conv.write("mine.txt", "mine\n");
    let theirs = someone_else_pushes(&ws, &origin, "main", "theirs.txt");

    let out = commit(&conv, "mine");
    assert_eq!(out["applied"], false, "{out}");
    let reason = out["reason"].as_str().unwrap();
    assert!(reason.contains("git_merge"), "{reason}");
    assert_eq!(remote_oid(&origin, "refs/heads/main"), theirs);
    assert_eq!(conv.read("mine.txt").as_deref(), Some("mine\n"));
    assert_eq!(conv.read("theirs.txt"), None, "not merged by the refusal");
    assert_eq!(conv.status()["incoming"], 1);

    let merged = conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(merged["merged"], "fast_forward", "{merged}");
    assert_eq!(merged["from"], theirs);
    assert_eq!(merged["conflicts"], json!([]));
    assert_eq!(conv.read("theirs.txt").as_deref(), Some("theirs\n"));
    assert!(conv.status().get("incoming").is_none());

    let out = commit(&conv, "mine");
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(out["parent"], theirs, "on top of their commit");
    assert_eq!(
        remote_oid(&origin, "refs/heads/main"),
        out["commit"].as_str().unwrap()
    );
    assert_eq!(
        conv.call("git_merge", json!({"repo": "app"}))["merged"],
        "up_to_date"
    );
}

/// **Both sides changed the same lines: the merge keeps both in the
/// conversation's file, marked; every commit is refused, naming the file,
/// until it is settled** — then the settled file lands.
#[test]
fn a_conflict_is_settled_in_the_conversations_files_before_it_commits() {
    let (ws, origin) = on_origin();
    let conv = ws.conversation();
    conv.write("README.md", "# app\n\nmine.\n");
    let other = ws.other_clone(&origin);
    std::fs::write(other.join("README.md"), "# app\n\ntheirs.\n").unwrap();
    git_in(&other, &["commit", "-q", "-am", "theirs"]);
    git_in(&other, &["push", "-q", "origin", "main"]);

    let merged = conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(merged["conflicts"], json!(["README.md"]), "{merged}");
    let marked = conv.read("README.md").unwrap();
    assert!(
        marked.contains("<<<<<<< yours\nmine.\n=======\ntheirs.\n>>>>>>> origin/main\n"),
        "{marked}"
    );
    assert_eq!(conv.status()["conflicts"], json!(["README.md"]));

    let out = commit(&conv, "mine");
    assert_eq!(out["applied"], false, "{out}");
    assert_eq!(out["conflicts"], json!(["README.md"]));
    assert_eq!(
        remote_oid(&origin, "refs/heads/main"),
        git_in(&other, &["rev-parse", "HEAD"]).trim()
    );

    conv.write("README.md", "# app\n\nmine and theirs.\n");
    assert!(conv.status().get("conflicts").is_none());
    let out = commit(&conv, "settled");
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(
        ws.git(&["show", "main:README.md"]),
        "# app\n\nmine and theirs.\n"
    );
}

/// **History on both sides: the merge is finished by the next commit, which
/// records both parents** — a commit made on this machine and never pushed
/// is kept, and so is everyone else's.
#[test]
fn a_merge_of_diverged_history_is_recorded_by_the_next_commit() {
    let (ws, origin) = on_origin();
    ws.write_worktree("local.txt", "made here, never pushed\n");
    let local = ws.commit_all("local only");
    let conv = ws.conversation();
    let theirs = someone_else_pushes(&ws, &origin, "main", "theirs.txt");

    let merged = conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(merged["merged"], "merging", "{merged}");
    let status = conv.status();
    assert_eq!(status["merging"], theirs, "{status}");
    assert_eq!(status["clean"], false);
    assert_eq!(
        conv.call("git_switch", json!({"repo": "app", "branch": "main"}))["error"],
        "invalid_arguments",
        "no switching away from a merge being finished"
    );

    let out = commit(&conv, "merge origin");
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(out["parent"], local);
    assert_eq!(out["merged"], theirs);
    let commit = out["commit"].as_str().unwrap();
    assert_eq!(remote_oid(&origin, "refs/heads/main"), commit);
    assert_eq!(
        ws.git(&["show", "main:local.txt"]),
        "made here, never pushed\n"
    );
    assert_eq!(ws.git(&["show", "main:theirs.txt"]), "theirs\n");
    assert_eq!(conv.status()["clean"], true);
}

/// **A hard reset abandons a merge being finished**, and the conversation
/// reads its own commit again.
#[test]
fn a_hard_reset_abandons_a_merge() {
    let (ws, origin) = on_origin();
    ws.write_worktree("local.txt", "local\n");
    let local = ws.commit_all("local only");
    let conv = ws.conversation();
    someone_else_pushes(&ws, &origin, "main", "theirs.txt");
    conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(conv.read("theirs.txt").as_deref(), Some("theirs\n"));

    let out = conv.call("git_reset", json!({"repo": "app", "mode": "hard"}));
    assert!(out.get("error").is_none(), "{out}");
    let status = conv.status();
    assert!(status.get("merging").is_none(), "{status}");
    assert_eq!(status["head"], local);
    assert_eq!(conv.read("theirs.txt"), None);
}

/// **A reset that would move the branch past commits the conversation does
/// not have is refused**: they would be taken off it.
#[test]
fn a_reset_over_someone_elses_commit_is_refused() {
    let (ws, origin) = on_origin();
    let conv = ws.conversation();
    conv.status();
    let theirs = someone_else_pushes(&ws, &origin, "main", "theirs.txt");
    let out = conv.call(
        "git_reset",
        json!({"repo": "app", "mode": "hard", "to": parent_rev("HEAD", Some(1))}),
    );
    assert_eq!(out["error"], "stale_ref", "{out}");
    assert_eq!(remote_oid(&origin, "refs/heads/main"), theirs);
}

/// **Uncommitted changes commit only on the branch they are made on.**
#[test]
fn changes_commit_only_on_their_own_branch() {
    let (ws, _origin) = on_origin();
    ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "release"}),
    );
    let conv = ws.conversation();
    conv.write("mine.txt", "mine\n");
    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "branch": "release", "from": "changes", "message": "m"}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("git_switch"),
        "{out}"
    );
}

/// **A `git_ref` move that would take commits off a branch is refused unless
/// `expected` names the tip it takes them from.**
#[test]
fn a_move_that_drops_commits_needs_expected() {
    let (ws, origin) = on_origin();
    ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "topic"}),
    );
    let conv = ws.conversation();
    conv.status();
    let theirs = someone_else_pushes(&ws, &origin, "topic", "theirs.txt");
    let main = branch_rev("main");
    let refused = conv.call(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "move", "name": "topic",
               "at": main}),
    );
    assert_eq!(refused["error"], "invalid_arguments", "{refused}");
    assert!(refused["detail"].as_str().unwrap().contains("expected"));
    assert_eq!(remote_oid(&origin, "refs/heads/topic"), theirs);

    let moved = conv.call(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "move", "name": "topic",
               "at": main, "expected": theirs}),
    );
    assert_eq!(moved["action"], "moved", "{moved}");
    assert_eq!(
        remote_oid(&origin, "refs/heads/topic"),
        remote_oid(&origin, "refs/heads/main")
    );
}

/// **Committing a file onto another branch that has changed it since the
/// conversation's work started is refused**, until `expected_head` says the
/// branch was looked at.
#[test]
fn a_file_committed_over_another_branchs_change_is_refused() {
    let (ws, origin) = on_origin();
    ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "release"}),
    );
    let conv = ws.conversation();
    conv.write("README.md", "# app\n\nmine.\n");
    let release = someone_else_pushes(&ws, &origin, "release", "README.md");

    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "branch": "release", "from": "files", "message": "m",
               "changes": [{"action": "take", "path": "README.md"}]}),
    );
    assert_eq!(out["applied"], false, "{out}");
    assert_eq!(out["conflicts"], json!(["README.md"]));
    assert_eq!(remote_oid(&origin, "refs/heads/release"), release);

    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "branch": "release", "from": "files", "message": "m",
               "expected_head": release,
               "changes": [{"action": "take", "path": "README.md"}]}),
    );
    assert_eq!(out["applied"], true, "{out}");
}

/// **A merge is committed whole**: `files` or a patch while one is being
/// finished is refused.
#[test]
fn a_merge_is_committed_whole() {
    let (ws, origin) = on_origin();
    ws.write_worktree("local.txt", "local\n");
    ws.commit_all("local only");
    let conv = ws.conversation();
    someone_else_pushes(&ws, &origin, "main", "theirs.txt");
    conv.call("git_merge", json!({"repo": "app"}));
    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "files", "message": "part",
               "changes": [{"action": "write", "path": "x.txt", "content": "x\n"}]}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(out["detail"].as_str().unwrap().contains("whole"), "{out}");
}

/// **A soft reset past a file that is not text is refused before origin
/// moves**: it could not be kept.
#[test]
fn a_soft_reset_past_a_binary_file_is_refused_first() {
    let (ws, origin) = on_origin();
    std::fs::write(ws.repo_dir().join("logo.bin"), [0u8, 1, 2, 0, 255]).unwrap();
    ws.commit_all("binary");
    ws.git(&["push", "-q", "origin", "main"]);
    let before = remote_oid(&origin, "refs/heads/main");
    let conv = ws.conversation();
    let out = conv.call(
        "git_reset",
        json!({"repo": "app", "mode": "soft", "to": parent_rev("HEAD", Some(1))}),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("logo.bin"),
        "{out}"
    );
    assert_eq!(
        remote_oid(&origin, "refs/heads/main"),
        before,
        "origin untouched"
    );
}

/// **A hard reset gets a conversation whose base is gone from the
/// repository going again** — on its branch as it stands.
#[test]
fn a_hard_reset_recovers_from_a_base_that_is_gone() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation();
    let gone = "ce013625030ba8dba906f756967f9e9ca394464a";
    let snapshot = serde_json::from_value(json!({
        "base": { "tree": gone, "parents": [gone] },
        "chains": { "x.txt": { "deltas": [
            { "at_ns": 1, "kind": "replace", "content": "x\n" }
        ], "size": 2 } }
    }))
    .unwrap();
    conv.ctx
        .files
        .repo("app")
        .unwrap()
        .restore(snapshot)
        .unwrap();
    let soft = conv.call("git_reset", json!({"repo": "app", "mode": "soft"}));
    assert_eq!(soft["error"], "invalid_arguments", "{soft}");
    let hard = conv.call("git_reset", json!({"repo": "app", "mode": "hard"}));
    assert_eq!(hard["id"], ws.oid("main"), "{hard}");
    assert_eq!(hard["discarded"], 1);
    assert_eq!(conv.status()["clean"], true);
}

/// **A file in conflict goes nowhere until it is settled**: not merged
/// again as though its markers were content, not carried to another branch
/// by a switch, not committed onto another branch.
#[test]
fn a_file_in_conflict_holds_everything_until_it_is_settled() {
    let (ws, origin) = on_origin();
    ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "release"}),
    );
    let conv = ws.conversation();
    conv.write("README.md", "# app\n\nmine.\n");
    let other = ws.other_clone(&origin);
    std::fs::write(other.join("README.md"), "# app\n\ntheirs.\n").unwrap();
    git_in(&other, &["commit", "-q", "-am", "theirs"]);
    git_in(&other, &["push", "-q", "origin", "main"]);
    let merged = conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(merged["conflicts"], json!(["README.md"]), "{merged}");

    let again = conv.call("git_merge", json!({"repo": "app"}));
    assert_eq!(again["error"], "invalid_arguments", "{again}");
    assert!(again["detail"].as_str().unwrap().contains("README.md"));

    let switched = conv.call("git_switch", json!({"repo": "app", "branch": "release"}));
    assert_eq!(switched["error"], "invalid_arguments", "{switched}");

    let release = remote_oid(&origin, "refs/heads/release");
    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "branch": "release", "from": "files", "message": "m",
               "changes": [{"action": "take", "path": "README.md"}]}),
    );
    assert_eq!(out["applied"], false, "{out}");
    assert_eq!(out["conflicts"], json!(["README.md"]));
    assert_eq!(remote_oid(&origin, "refs/heads/release"), release);
}

/// **A tag origin never takes is not kept here either** — refused or
/// never reached alike.
#[test]
fn a_tag_origin_cannot_be_reached_for_is_not_kept() {
    let (ws, _origin) = on_origin();
    let gone = ws.repo_dir().join("no-such-origin");
    ws.git(&["remote", "set-url", "origin", gone.to_str().unwrap()]);
    let conv = ws.conversation();
    let out = conv.call(
        "git_ref",
        json!({"repo": "app", "kind": "tag", "action": "create", "name": "v1"}),
    );
    assert!(out.get("error").is_some(), "{out}");
    assert_eq!(ws.git(&["tag", "-l"]).trim(), "", "no tag left behind");
}

fn commit(conv: &Conversation, message: &str) -> Value {
    conv.call(
        "git_commit",
        json!({"repo": "app", "from": "changes", "message": message}),
    )
}

/// **A branch made with git_ref is made on origin.**
#[test]
fn a_branch_made_with_git_ref_is_made_on_origin() {
    let (ws, origin) = on_origin();
    let out = ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "release"}),
    );
    assert_eq!(out["on"], "origin", "{out}");
    assert_eq!(remote_oid(&origin, "refs/heads/release"), ws.oid("main"));
    assert_eq!(ws.oid("release"), ws.oid("main"));

    let again = ws.write(
        "git_ref",
        json!({"repo": "app", "kind": "branch", "action": "create", "name": "release"}),
    );
    assert_eq!(again["error"], "invalid_arguments", "{again}");
}

// ── git_switch ───────────────────────────────────────────────────────────────

/// **Switching to a new branch makes it on origin and carries the changes**;
/// from then on the conversation reads and commits there, and `main` is left
/// alone.
#[test]
fn switching_to_a_new_branch_makes_it_on_origin_and_carries_the_changes() {
    let (ws, origin) = on_origin();
    let main = ws.oid("main");
    let conv = ws.conversation();
    conv.write("README.md", "# app\n\nwork in progress.\n");

    let out = conv.call(
        "git_switch",
        json!({"repo": "app", "branch": "feature", "create": true}),
    );
    assert_eq!(out["created"], true, "{out}");
    assert_eq!(out["on"], "origin");
    assert_eq!(out["previous"], "main");
    assert_eq!(out["carried"], 1);
    assert_eq!(remote_oid(&origin, "refs/heads/feature"), main);

    let status = conv.status();
    assert_eq!(status["branch"], "feature", "{status}");
    assert_eq!(status["counts"]["modified"], 1);

    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "changes", "message": "on the feature"}),
    );
    assert_eq!(out["branch"], "feature", "{out}");
    assert_eq!(
        remote_oid(&origin, "refs/heads/feature"),
        out["commit"].as_str().unwrap()
    );
    assert_eq!(ws.oid("main"), main, "main is untouched");
    assert_eq!(remote_oid(&origin, "refs/heads/main"), main);
}

/// **Switching to a branch reads that branch**, one only on origin included,
/// and every read tool's `HEAD` follows.
#[test]
fn switching_reads_the_other_branch_and_head_follows() {
    let (ws, origin) = on_origin();
    let theirs = someone_else_pushes(&ws, &origin, "theirs", "theirs.txt");
    let conv = ws.conversation();

    let out = conv.call("git_switch", json!({"repo": "app", "branch": "theirs"}));
    assert_eq!(out["created"], false, "{out}");
    assert_eq!(out["id"], theirs);
    assert_eq!(conv.read("theirs.txt").as_deref(), Some("theirs\n"));

    let log = conv.call("git_log", json!({"repo": "app", "page": 0}));
    assert_eq!(log["commits"][0]["id"], theirs, "{log}");
    let refs = conv.call(
        "git_refs",
        json!({"repo": "app", "kind": "branches", "page": 0}),
    );
    assert_eq!(refs["current"], "theirs", "{refs}");

    let missing = conv.call("git_switch", json!({"repo": "app", "branch": "nope"}));
    assert_eq!(missing["error"], "invalid_arguments", "{missing}");
    assert!(
        missing["detail"].as_str().unwrap().contains("create"),
        "{missing}"
    );
}

// ── git_reset ────────────────────────────────────────────────────────────────

/// **A hard reset with nowhere to go discards the conversation's changes**
/// and moves nothing.
#[test]
fn a_hard_reset_discards_the_changes() {
    let ws = GitWorkspace::new();
    let main = ws.oid("main");
    let conv = ws.conversation();
    conv.write("README.md", "# scrap this\n");
    conv.write("scratch.txt", "x\n");

    let out = conv.call("git_reset", json!({"repo": "app", "mode": "hard"}));
    assert_eq!(out["discarded"], 2, "{out}");
    assert!(out.get("on").is_none(), "nothing moved: {out}");
    assert_eq!(ws.oid("main"), main);
    assert_eq!(conv.status()["clean"], true);
    assert_eq!(conv.read("scratch.txt"), None);
}

/// **A soft reset back one commit keeps what you see**: the commit's change
/// becomes uncommitted, ready to commit again — and origin's copy is
/// rewound with it.
#[test]
fn a_soft_reset_back_one_keeps_the_view_and_rewinds_origin() {
    let (ws, origin) = on_origin();
    let before = ws.oid("main");
    let parent = ws.oid("main~1");
    let conv = ws.conversation();

    let out = conv.call(
        "git_reset",
        json!({"repo": "app", "mode": "soft", "to": parent_rev("HEAD", Some(1))}),
    );
    assert_eq!(out["on"], "origin", "{out}");
    assert_eq!(out["previous"], before);
    assert_eq!(out["id"], parent);
    assert_eq!(out["kept"], 1);
    assert_eq!(remote_oid(&origin, "refs/heads/main"), parent);
    assert_eq!(ws.oid("main"), parent);

    assert_eq!(conv.read("src/lib.rs").as_deref(), Some(HELLO));
    let status = conv.status();
    assert_eq!(status["changes"][0]["path"], "src/lib.rs", "{status}");

    let out = conv.call(
        "git_commit",
        json!({"repo": "app", "from": "changes", "message": "say hello again"}),
    );
    assert_eq!(out["applied"], true, "{out}");
    assert_eq!(ws.git(&["show", "main:src/lib.rs"]), HELLO);
}

/// **A hard reset back one commit shows the branch as it now stands.**
#[test]
fn a_hard_reset_back_one_shows_the_older_commit() {
    let ws = GitWorkspace::new();
    let conv = ws.conversation();
    conv.write("scratch.txt", "x\n");
    let out = conv.call(
        "git_reset",
        json!({"repo": "app", "mode": "hard", "to": parent_rev("HEAD", Some(1))}),
    );
    assert_eq!(out["on"], "local", "{out}");
    assert_eq!(out["discarded"], 1);
    assert_eq!(conv.read("src/lib.rs").as_deref(), Some(HI));
    assert_eq!(conv.status()["clean"], true);
}

/// A reset refuses a branch that moved since the caller looked.
#[test]
fn a_reset_with_a_stale_expected_head_is_refused() {
    let ws = GitWorkspace::new();
    let main = ws.oid("main");
    let conv = ws.conversation();
    let out = conv.call(
        "git_reset",
        json!({"repo": "app", "mode": "hard", "to": parent_rev("HEAD", Some(1)),
               "expected_head": ws.oid("main~1")}),
    );
    assert_eq!(out["error"], "stale_ref", "{out}");
    assert_eq!(ws.oid("main"), main);
}
