//! The two tools that reach a remote: git_fetch and git_push.
//!
//! Both run against a second repository on disk serving as `origin` over
//! `file://`, so git's real transport is exercised. A third clone stands in
//! for another developer — the only way to test what these exist to get
//! right: what happens when the remote moved while you were not looking.

use serde_json::{json, Value};

use crate::harness::{branch_rev, git_in, remote_oid, GitWorkspace};

fn push(ws: &GitWorkspace, pushes: Value) -> Value {
    ws.write(
        "git_push",
        json!({"repo": "app", "remote": "origin", "pushes": pushes}),
    )
}

/// Publish `main` to a fresh origin.
fn publish_main(ws: &GitWorkspace) -> Value {
    push(
        ws,
        json!([{"action": "update_branch", "name": "main", "new": true}]),
    )
}

/// Another developer pushes a commit to origin.
fn someone_else_pushes(ws: &GitWorkspace, origin: &std::path::Path, file: &str) {
    let other = ws.other_clone(origin);
    std::fs::write(other.join(file), "theirs\n").unwrap();
    git_in(&other, &["add", "-A"]);
    git_in(&other, &["commit", "-q", "-m", "their work"]);
    git_in(&other, &["push", "-q", "origin", "main"]);
}

// ── git_fetch ────────────────────────────────────────────────────────────────

#[test]
fn a_fetch_with_nothing_new_is_up_to_date() {
    let ws = GitWorkspace::new();
    ws.with_origin();
    publish_main(&ws);
    let out = ws.write("git_fetch", json!({"repo": "app", "remote": "origin"}));
    assert!(out.get("error").is_none(), "{out}");
    assert_eq!(out["up_to_date"], true, "{out}");
    assert_eq!(out["count"], 0);
}

/// **A fetch moves tracking refs and never a local branch.** That is what
/// makes it safe to call before deciding anything.
#[test]
fn a_new_upstream_commit_moves_only_the_tracking_ref() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    someone_else_pushes(&ws, &origin, "upstream.txt");

    let local_before = ws.oid("main");
    let out = ws.write("git_fetch", json!({"repo": "app", "remote": "origin"}));
    assert_eq!(out["up_to_date"], false, "{out}");
    assert_eq!(out["updates"][0]["ref_name"], "refs/remotes/origin/main");
    assert_eq!(out["updates"][0]["change"], "fast_forward");
    assert_eq!(
        ws.oid("main"),
        local_before,
        "the local branch must not move"
    );
    assert_ne!(ws.oid("refs/remotes/origin/main"), local_before);
}

#[test]
fn a_fetch_changes_no_file_on_disk() {
    let ws = GitWorkspace::new();
    ws.with_origin();
    publish_main(&ws);
    let before = std::fs::read_to_string(ws.repo_dir().join("src/lib.rs")).unwrap();
    ws.write("git_fetch", json!({"repo": "app", "remote": "origin"}));
    assert_eq!(
        std::fs::read_to_string(ws.repo_dir().join("src/lib.rs")).unwrap(),
        before
    );
}

/// A remote that is not configured is refused by name — git would otherwise
/// treat an unknown name as a path or a URL.
#[test]
fn fetching_an_unconfigured_remote_is_refused() {
    let ws = GitWorkspace::new();
    let out = ws.write("git_fetch", json!({"repo": "app", "remote": "nowhere"}));
    assert_eq!(out["error"], "unknown_remote", "{out}");
}

#[test]
fn a_restricted_context_cannot_fetch() {
    let ws = GitWorkspace::new();
    ws.with_origin();
    let out = ws.read("git_fetch", json!({"repo": "app", "remote": "origin"}));
    assert_eq!(out["error"], "not_permitted", "{out}");
}

// ── git_push ─────────────────────────────────────────────────────────────────

/// `new: true` is how something is published for the first time.
#[test]
fn a_new_branch_is_created_on_the_remote() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    let out = publish_main(&ws);
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(out["results"][0]["outcome"], "created");
    assert_eq!(remote_oid(&origin, "main"), ws.oid("main"));
}

/// **The lease is not something the caller has to supply.** Omitted, it comes
/// from the tracking ref — which is the value that actually protects a
/// concurrent push, and one the model has no way to hold.
#[test]
fn an_omitted_lease_comes_from_the_tracking_ref_and_fast_forwards() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    ws.write_worktree("more.txt", "more\n");
    ws.commit_all("more work");

    let out = push(&ws, json!([{"action": "update_branch", "name": "main"}]));
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(out["results"][0]["outcome"], "fast_forward");
    assert_eq!(remote_oid(&origin, "main"), ws.oid("main"));
}

/// **And that same default is what refuses to bury someone else's commit.**
/// The tracking ref is stale after they push, so the lease no longer matches.
#[test]
fn an_omitted_lease_still_refuses_to_overwrite_a_concurrent_push() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    someone_else_pushes(&ws, &origin, "theirs.txt");
    let theirs = remote_oid(&origin, "main");

    ws.write_worktree("mine.txt", "mine\n");
    ws.commit_all("my work");

    let out = push(&ws, json!([{"action": "update_branch", "name": "main"}]));
    assert_eq!(out["accepted"], false, "{out}");
    assert_eq!(out["results"][0]["outcome"], "rejected");
    assert!(
        out["results"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("fetch"),
        "the reason says what to do: {out}"
    );
    assert_eq!(remote_oid(&origin, "main"), theirs, "their commit survives");
}

/// After a fetch the tracking ref is current again, so the same push is
/// accepted — the way out of the rejection above.
#[test]
fn a_fetch_makes_the_rejected_push_possible_again() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    someone_else_pushes(&ws, &origin, "theirs.txt");
    ws.write_worktree("mine.txt", "mine\n");
    ws.commit_all("my work");
    assert_eq!(
        push(&ws, json!([{"action": "update_branch", "name": "main"}]))["accepted"],
        false
    );

    ws.write("git_fetch", json!({"repo": "app", "remote": "origin"}));
    // Rebase local work on top of what they pushed, then push again.
    ws.git(&["rebase", "-q", "refs/remotes/origin/main"]);
    let out = push(&ws, json!([{"action": "update_branch", "name": "main"}]));
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(remote_oid(&origin, "main"), ws.oid("main"));
}

/// An explicit `expected` still works for a caller that holds the value.
#[test]
fn an_explicit_expected_is_honoured() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    let published = remote_oid(&origin, "main");
    ws.write_worktree("more.txt", "more\n");
    ws.commit_all("more work");

    let wrong = push(
        &ws,
        json!([{"action": "update_branch", "name": "main",
                "expected": ws.oid("HEAD~2")}]),
    );
    assert_eq!(wrong["accepted"], false, "{wrong}");
    assert_eq!(remote_oid(&origin, "main"), published);

    let right = push(
        &ws,
        json!([{"action": "update_branch", "name": "main", "expected": published}]),
    );
    assert_eq!(right["accepted"], true, "{right}");
}

/// A push that would change nothing is accepted without the lease being
/// consulted: a push that writes nothing cannot overwrite anything.
#[test]
fn a_push_that_changes_nothing_is_up_to_date() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    let published = remote_oid(&origin, "main");
    let out = push(&ws, json!([{"action": "update_branch", "name": "main"}]));
    assert_eq!(out["results"][0]["outcome"], "up_to_date", "{out}");
    assert_eq!(remote_oid(&origin, "main"), published);
}

#[test]
fn a_tag_is_published_and_can_be_deleted_again() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    ws.git(&["tag", "v1.0"]);

    let out = push(&ws, json!([{"action": "update_tag", "name": "v1.0"}]));
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(out["results"][0]["kind"], "tag");

    let deleted = push(&ws, json!([{"action": "delete_tag", "name": "v1.0"}]));
    assert_eq!(deleted["accepted"], true, "{deleted}");
    assert!(!git_in(&origin, &["tag", "-l"]).contains("v1.0"));
}

/// **An annotated tag reaches the remote as itself**, message and tagger
/// intact — not as a lightweight tag on its commit — and can be deleted
/// there again without naming what it holds.
#[test]
fn an_annotated_tag_is_published_whole_and_deleted_again() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    ws.git(&["tag", "-a", "v2.0", "-m", "the release"]);
    let tag_object = ws.oid("refs/tags/v2.0");

    let out = push(&ws, json!([{"action": "update_tag", "name": "v2.0"}]));
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(remote_oid(&origin, "refs/tags/v2.0"), tag_object);
    assert_eq!(
        git_in(&origin, &["cat-file", "-t", "refs/tags/v2.0"]).trim(),
        "tag"
    );

    let deleted = push(&ws, json!([{"action": "delete_tag", "name": "v2.0"}]));
    assert_eq!(deleted["accepted"], true, "{deleted}");
    assert!(!git_in(&origin, &["tag", "-l"]).contains("v2.0"));
}

#[test]
fn a_branch_can_be_deleted_on_the_remote() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    ws.git(&["branch", "temp"]);
    push(
        &ws,
        json!([{"action": "update_branch", "name": "temp", "new": true}]),
    );
    ws.write("git_fetch", json!({"repo": "app", "remote": "origin"}));

    let out = push(&ws, json!([{"action": "delete_branch", "name": "temp"}]));
    assert_eq!(out["accepted"], true, "{out}");
    assert!(!git_in(&origin, &["branch"]).contains("temp"));
}

/// Deleting a branch nothing is known about says what to do rather than
/// guessing.
#[test]
fn deleting_an_unknown_remote_branch_says_to_fetch() {
    let ws = GitWorkspace::new();
    ws.with_origin();
    publish_main(&ws);
    let out = push(
        &ws,
        json!([{"action": "delete_branch", "name": "never-there"}]),
    );
    assert_eq!(out["error"], "invalid_arguments", "{out}");
    assert!(
        out["detail"].as_str().unwrap().contains("git_fetch"),
        "{out}"
    );
}

/// **A push is atomic**: one rejected ref means none of them changed.
#[test]
fn one_rejected_ref_aborts_the_whole_push() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    ws.git(&["branch", "extra"]);
    someone_else_pushes(&ws, &origin, "theirs.txt");
    ws.write_worktree("mine.txt", "mine\n");
    ws.commit_all("my work");

    let out = push(
        &ws,
        json!([
            {"action": "update_branch", "name": "extra", "new": true},
            {"action": "update_branch", "name": "main"}
        ]),
    );
    assert_eq!(out["accepted"], false, "{out}");
    assert!(
        !git_in(&origin, &["branch"]).contains("extra"),
        "the good ref landed despite the atomic push failing"
    );
}

#[test]
fn pushing_to_an_unconfigured_remote_is_refused() {
    let ws = GitWorkspace::new();
    let out = ws.write(
        "git_push",
        json!({"repo": "app", "remote": "nowhere",
               "pushes": [{"action": "update_branch", "name": "main", "new": true}]}),
    );
    assert_eq!(out["error"], "unknown_remote", "{out}");
}

/// A source other than the branch of the same name still works.
#[test]
fn a_branch_can_be_published_from_another_revision() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    publish_main(&ws);
    let out = push(
        &ws,
        json!([{"action": "update_branch", "name": "release", "new": true,
                "source": branch_rev("main")}]),
    );
    assert_eq!(out["accepted"], true, "{out}");
    assert_eq!(remote_oid(&origin, "release"), ws.oid("main"));
}

/// The one tool whose effect leaves the machine is Comprehensive's alone.
#[test]
fn a_restricted_context_cannot_push() {
    let ws = GitWorkspace::new();
    let origin = ws.with_origin();
    let out = ws.read(
        "git_push",
        json!({"repo": "app", "remote": "origin",
               "pushes": [{"action": "update_branch", "name": "main", "new": true}]}),
    );
    assert_eq!(out["error"], "not_permitted", "{out}");
    assert!(git_in(&origin, &["branch"]).trim().is_empty());
}
