//! Sandbox runs end to end: a real repository, a conversation's overlay over
//! its branch, and real commands run through the platform's shell.
//!
//! Each test pins one promise of a run: what the command sees, what comes
//! back into the conversation's store, that whatever the folder held before —
//! someone's own work in it — is exactly as it was afterwards however the run
//! ended, and that a refused, failed, timed-out or abandoned run leaves the
//! lock free.

mod support;

use std::time::Duration;

use support::*;
use zend_vfs::checkout::CheckoutError;
use zend_vfs::file_delta::FileDelta;
use zend_vfs::sandbox::Refused;
use zend_vfs::{CommandPolicy, Repo, Sandbox, SandboxCommand, SandboxError, VfsStore};

// ── A run ────────────────────────────────────────────────────────────────────

/// **A run lays the conversation down, runs the command there, and records
/// what it changed** — the command sees the conversation's edit and its new
/// file, what it creates and deletes comes back into the store — and the
/// folder is then exactly as it was before.
#[tokio::test]
async fn a_run_sees_the_conversation_and_records_what_the_command_changed() {
    let f = fixture();
    let files = f.store();
    files
        .edit("README.md", "# app\n\nthe conversation's app.\n".into())
        .unwrap();
    files
        .write("src/new.rs", "pub fn new() {}\n".into())
        .unwrap();

    let command = shell(
        "type README.md & echo made> made.txt & del src\\new.rs",
        "cat README.md; echo made > made.txt; rm src/new.rs",
    );
    let done = run(&f, "main", &files, &command).await;

    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert!(!done.timed_out);
    assert!(done.printed.contains("the conversation's app."), "{done:?}");
    assert_eq!(done.output.bytes, done.printed.len() as u64);
    let paths: Vec<&str> = done.changed.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, ["made.txt", "src/new.rs"]);
    assert!(done.unrecorded.is_empty(), "{:?}", done.unrecorded);
    assert!(matches!(done.changed[1].delta.delta, FileDelta::Delete));

    assert_eq!(read(&files, "made.txt").unwrap().trim_end(), "made");
    assert_eq!(read(&files, "src/new.rs"), None);
    assert_eq!(
        read(&files, "README.md").unwrap(),
        "# app\n\nthe conversation's app.\n"
    );

    assert_untouched(&f, "main");
    assert_eq!(disk(&f.root, "README.md").unwrap(), README);
    assert_eq!(disk(&f.root, "made.txt"), None);
}

/// **A command that changes nothing records nothing**, and what it printed —
/// both streams — and its exit code still come back.
#[tokio::test]
async fn a_command_that_changes_nothing_records_nothing() {
    let f = fixture();
    let files = f.store();
    files.write("notes.txt", "kept\n".into()).unwrap();
    let before = files.changes().unwrap();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "echo out & echo err 1>&2 & exit 7",
            "echo out; echo err 1>&2; exit 7",
        ),
    )
    .await;
    assert_eq!(done.exit_code, Some(7));
    let printed: Vec<&str> = done.printed.lines().map(str::trim_end).collect();
    assert_eq!(printed.len(), 2, "{printed:?}");
    assert!(
        printed.contains(&"out") && printed.contains(&"err"),
        "{printed:?}"
    );
    assert!(done.changed.is_empty() && done.unrecorded.is_empty());
    assert_eq!(files.changes().unwrap(), before);
    assert_untouched(&f, "main");
}

/// **The command runs on the conversation's branch**, and the folder goes
/// back to the branch it was on.
#[tokio::test]
async fn the_command_runs_on_the_conversations_branch() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "feature",
        &files,
        &shell("type FEATURE.md", "cat FEATURE.md"),
    )
    .await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert!(done.printed.contains("only on the feature branch"));
    assert_untouched(&f, "main");
    assert_eq!(disk(&f.root, "FEATURE.md"), None);
}

/// **A binary file the command writes is reported, not lost** — the store
/// holds text only — while its text changes are recorded.
#[tokio::test]
async fn a_binary_file_the_store_cannot_hold_is_reported() {
    let f = fixture();
    let files = f.store();
    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "copy /Y assets\\logo.bin out.bin >NUL & echo text> out.txt",
            "cp assets/logo.bin out.bin; echo text > out.txt",
        ),
    )
    .await;
    assert_eq!(done.changed.len(), 1, "{done:?}");
    assert_eq!(done.changed[0].path, "out.txt");
    assert_eq!(done.unrecorded.len(), 1);
    assert_eq!(done.unrecorded[0].path, "out.bin");
    assert_eq!(
        done.unrecorded[0].delta.delta,
        FileDelta::ReplaceBinary {
            content: BINARY.to_vec()
        }
    );
    assert_eq!(read(&files, "out.bin"), None);
    assert_untouched(&f, "main");
}

// ── The folder's own state ───────────────────────────────────────────────────

/// Someone's own work in the folder: on `feature`, a staged edit with an
/// unstaged one over it, a new file not yet added, a deleted file, and an
/// ignored log — returned as `(status, staged diff)` to compare against.
fn someones_work(f: &Fixture) -> (String, String) {
    git(&f.root, &["checkout", "-q", "feature"]);
    put(&f.root, "README.md", b"# app\n\nstaged.\n");
    git(&f.root, &["add", "README.md"]);
    put(&f.root, "README.md", b"# app\n\nstaged, then more.\n");
    put(&f.root, "mine/draft.txt", b"not added yet\n");
    std::fs::remove_file(f.root.join("src/lib.rs")).unwrap();
    put(&f.root, "notes.log", b"my ignored notes\n");
    (worktree_state(f), git(&f.root, &["diff", "--cached"]))
}

fn worktree_state(f: &Fixture) -> String {
    git(
        &f.root,
        &[
            "status",
            "--porcelain=v1",
            "--branch",
            "--untracked-files=all",
        ],
    )
}

/// **Someone's work in the folder is exactly as it was after a job** — the
/// branch they were on, their staged and unstaged edits, their new and
/// deleted files, their ignored file even where the conversation writes the
/// same path — and the job never saw any of it.
#[tokio::test]
async fn someones_work_in_the_folder_survives_a_job() {
    let f = fixture();
    let (before, staged) = someones_work(&f);
    let files = f.store();
    files
        .write("notes.log", "the conversation's log\n".into())
        .unwrap();
    files
        .write("mine/draft.txt", "the conversation's draft\n".into())
        .unwrap();

    let done = run(
        &f,
        "main",
        &files,
        &shell(
            "type README.md & type mine\\draft.txt & type notes.log",
            "cat README.md mine/draft.txt notes.log",
        ),
    )
    .await;
    assert!(done.printed.contains("the app."), "{done:?}");
    assert!(
        done.printed.contains("the conversation's draft"),
        "{done:?}"
    );
    assert!(done.printed.contains("the conversation's log"), "{done:?}");
    assert!(!done.printed.contains("staged"), "{done:?}");

    assert_eq!(worktree_state(&f), before);
    assert_eq!(git(&f.root, &["diff", "--cached"]), staged);
    assert_eq!(
        disk(&f.root, "README.md").unwrap(),
        b"# app\n\nstaged, then more.\n"
    );
    assert_eq!(disk(&f.root, "mine/draft.txt").unwrap(), b"not added yet\n");
    assert_eq!(disk(&f.root, "notes.log").unwrap(), b"my ignored notes\n");
    assert_eq!(disk(&f.root, "src/lib.rs"), None);
    assert_nothing_set_aside(&f);
}

/// **A job that fails part way — its changes not fitting the branch —
/// still puts the folder back.**
#[tokio::test]
async fn a_failed_job_puts_the_folder_back() {
    let f = fixture();
    let (before, staged) = someones_work(&f);
    let failed = try_run(&f, "main", &diverged_store(&f), &shell("echo x", "echo x")).await;
    assert!(
        matches!(
            failed,
            Err(SandboxError::Checkout(CheckoutError::Diverged { .. }))
        ),
        "{failed:?}"
    );
    assert_eq!(worktree_state(&f), before);
    assert_eq!(git(&f.root, &["diff", "--cached"]), staged);
    assert_nothing_set_aside(&f);
}

/// **A job abandoned while its command runs — its future dropped — kills the
/// command and puts the folder back before the next job may start.**
#[tokio::test]
async fn an_abandoned_job_puts_the_folder_back() {
    let f = fixture();
    let (before, staged) = someones_work(&f);
    let files = f.store();
    files.write("slow.txt", "mine\n".into()).unwrap();
    let slow = shell(
        "echo started> started.txt & ping -n 30 127.0.0.1 >NUL",
        "echo started > started.txt; sleep 30",
    );
    let abandoned = tokio::time::timeout(
        Duration::from_millis(3000),
        run_job(&f.sandbox, "abandoned", "main", &files, &slow),
    )
    .await;
    assert!(abandoned.is_err(), "the job should still have been running");

    // The next job waits for the lock, which is let go only once the folder
    // is back — wherever the abandoned job was when it was dropped.
    let next = run(&f, "main", &f.store(), &shell("echo ok", "echo ok")).await;
    assert_eq!(next.printed.trim_end(), "ok");
    assert_eq!(worktree_state(&f), before);
    assert_eq!(git(&f.root, &["diff", "--cached"]), staged);
    assert_eq!(disk(&f.root, "started.txt"), None);
    assert_nothing_set_aside(&f);
}

/// **A refused command never sets anything aside**: the folder is not
/// touched at all.
#[tokio::test]
async fn a_refused_command_never_touches_the_folder() {
    let f = fixture();
    let (before, _) = someones_work(&f);
    let refused = try_run(&f, "main", &f.store(), &SandboxCommand::new("python")).await;
    assert!(matches!(refused, Err(SandboxError::Refused(_))));
    assert_eq!(worktree_state(&f), before);
    assert_nothing_set_aside(&f);
}

/// A conversation's store holding an edit that fits no copy of `README.md`
/// the branch has held: laying it down fails before any command runs.
fn diverged_store(f: &Fixture) -> VfsStore {
    let other = VfsStore::new();
    other.write("README.md", "# other\n".into()).unwrap();
    other
        .edit("README.md", "# other\nedited.\n".into())
        .unwrap();
    let edit = other.deltas("README.md").unwrap().remove(1);
    let files = f.store();
    let saved = serde_json::json!({
        "chains": { "README.md": { "deltas": [edit], "size": 16 } }
    });
    files
        .restore(serde_json::from_value(saved).unwrap())
        .unwrap();
    files
}

// ── Failures ─────────────────────────────────────────────────────────────────

/// **A refused command never runs, records nothing, and touches nothing** —
/// whether the program is unlisted, git, or pointed outside.
#[tokio::test]
async fn a_refused_command_never_runs_and_touches_nothing() {
    let f = fixture();
    let files = f.store();
    files.write("mine.txt", "mine\n".into()).unwrap();
    let before = files.changes().unwrap();
    for (command, want) in [
        (
            SandboxCommand::new("python").arg("x.py"),
            Refused::NotAllowed {
                program: "python".into(),
                allowed: f.sandbox.policy().allowed(),
            },
        ),
        (
            SandboxCommand::new("git").arg("status"),
            Refused::Git {
                command: "git status".into(),
            },
        ),
        (
            SandboxCommand::new(SHELL).args(["-c", "../escape.txt"]),
            Refused::Escapes {
                arg: "../escape.txt".into(),
            },
        ),
    ] {
        match try_run(&f, "main", &files, &command).await {
            Err(SandboxError::Refused(got)) => assert_eq!(got, want),
            other => panic!("{command:?}: {other:?}"),
        }
        assert_eq!(files.changes().unwrap(), before);
        assert_eq!(disk(&f.root, "mine.txt"), None);
        assert!(f.sandbox.repo().status().unwrap().is_empty());
    }
}

/// **A command past its timeout is killed** — and what it changed before
/// that is still recorded.
#[tokio::test]
async fn a_timed_out_command_is_killed_and_its_changes_recorded() {
    let f = fixture();
    let files = f.store();
    let command = shell(
        "echo partial> partial.txt & ping -n 30 127.0.0.1 >NUL",
        "echo partial > partial.txt; sleep 30",
    )
    .timeout(Duration::from_millis(300));
    let done = run(&f, "main", &files, &command).await;
    assert!(done.timed_out);
    assert_eq!(done.exit_code, None);
    assert_eq!(read(&files, "partial.txt").unwrap().trim_end(), "partial");
    assert_untouched(&f, "main");
}

/// **Changes that no longer fit the branch fail the run before the command
/// starts**, and free the lock for the next run.
#[tokio::test]
async fn changes_that_do_not_fit_fail_before_the_command_runs() {
    let f = fixture();
    let files = diverged_store(&f);

    let command = shell("echo ran> ran.txt", "echo ran > ran.txt");
    match try_run(&f, "main", &files, &command).await {
        Err(SandboxError::Checkout(CheckoutError::Diverged { path })) => {
            assert_eq!(path, "README.md")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(disk(&f.root, "ran.txt"), None, "the command ran");

    let fresh = f.store();
    let done = run(&f, "main", &fresh, &shell("echo ok", "echo ok")).await;
    assert_eq!(done.printed.trim_end(), "ok");
}

/// **A store that is not an overlay over this repository's branch is
/// refused** before anything is touched.
#[tokio::test]
async fn a_store_over_anything_else_is_refused() {
    let f = fixture();
    let other = tempfile::tempdir().unwrap();
    let command = shell("echo x", "echo x");
    for files in [VfsStore::new(), VfsStore::with_root(other.path())] {
        assert!(matches!(
            try_run(&f, "main", &files, &command).await,
            Err(SandboxError::WrongStore { .. })
        ));
    }
    // The repository's own folder, read as a folder rather than through the
    // branch, is refused too: it is what the jobs run in.
    assert!(matches!(
        try_run(&f, "main", &VfsStore::with_root(&f.root), &command).await,
        Err(SandboxError::WrongBranch { .. })
    ));
}

// ── Several conversations ────────────────────────────────────────────────────

/// **Runs on one repository take turns.** Two conversations started together
/// each see only their own file, and each records only its own change.
#[tokio::test]
async fn runs_on_one_repository_take_turns() {
    let f = fixture();
    let a = f.store();
    let b = f.store();
    a.write("who.txt", "conversation a\n".into()).unwrap();
    b.write("who.txt", "conversation b\n".into()).unwrap();
    let command = shell(
        "type who.txt & echo touched> touched.log & echo seen>> who.txt",
        "cat who.txt; echo touched > touched.log; echo seen >> who.txt",
    );

    let (ran_a, ran_b) = tokio::join!(run(&f, "main", &a, &command), run(&f, "main", &b, &command));
    assert!(ran_a.printed.starts_with("conversation a"), "{ran_a:?}");
    assert!(ran_b.printed.starts_with("conversation b"), "{ran_b:?}");
    for (files, who) in [(&a, "a"), (&b, "b")] {
        let text = read(files, "who.txt").unwrap();
        assert!(
            text.starts_with(&format!("conversation {who}\n")),
            "{text:?}"
        );
        assert!(text.contains("seen"), "{text:?}");
        // An ignored by-product — a build output — stays in the folder like
        // any build cache, and is not the conversation's change.
        assert!(!files.is_modified("touched.log"));
    }
    assert!(disk(&f.root, "touched.log").is_some(), "build output kept");
    assert_nothing_set_aside(&f);
}

/// **A program the conversation wrote runs from where it now stands** — the
/// check comes after the conversation's changes are laid down.
#[cfg(windows)]
#[tokio::test]
async fn a_program_the_conversation_wrote_runs() {
    let f = fixture();
    let sandbox = Sandbox::new(
        Repo::open(&f.root).unwrap(),
        CommandPolicy::allowing(["./tools/hello.cmd"]),
    );
    let files = f.store();
    files
        .write("tools/hello.cmd", "@echo from the conversation\r\n".into())
        .unwrap();
    let hello = SandboxCommand::new("./tools/hello.cmd");
    let done = run_job(&sandbox, "one", "main", &files, &hello)
        .await
        .unwrap();
    assert_eq!(done.printed.trim_end(), "from the conversation");

    let fresh = f.store();
    assert!(matches!(
        run_job(&sandbox, "two", "main", &fresh, &hello).await,
        Err(SandboxError::Refused(Refused::Program { .. }))
    ));
    assert_eq!(disk(&f.root, "tools/hello.cmd"), None);
}
