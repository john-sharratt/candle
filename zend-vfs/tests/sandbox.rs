//! Sandbox runs end to end: a real repository, a conversation's overlay over
//! it, and real commands run through the platform's shell.
//!
//! Each test pins one promise of a run: what the command sees, what comes
//! back into the conversation's store, what the checkout holds afterwards,
//! and that a refused, failed or timed-out run leaves the checkout reset and
//! the lock free.

mod support;

use std::time::Duration;

use support::*;
use zend_vfs::checkout::CheckoutError;
use zend_vfs::file_delta::FileDelta;
use zend_vfs::sandbox::Refused;
use zend_vfs::{CommandPolicy, Head, Repo, Sandbox, SandboxCommand, SandboxError, VfsStore};

// ── A run ────────────────────────────────────────────────────────────────────

/// **A run lays the conversation down, runs the command there, and records
/// what it changed** — the command sees the conversation's edit and its new
/// file, what it creates and deletes comes back into the store, and the
/// checkout is the branch again afterwards.
#[tokio::test]
async fn a_run_sees_the_conversation_and_records_what_the_command_changed() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
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
    assert!(
        done.stdout.text.contains("the conversation's app."),
        "{:?}",
        done.stdout
    );
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

    assert_reset(&f, "main");
    assert_eq!(disk(&f.root, "README.md").unwrap(), README);
    assert_eq!(disk(&f.root, "made.txt"), None);
    assert_eq!(disk(&f.root, "src/new.rs"), None);
}

/// **A command that changes nothing records nothing**, and its streams and
/// exit code still come back.
#[tokio::test]
async fn a_command_that_changes_nothing_records_nothing() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    files.write("notes.txt", "kept\n".into()).unwrap();
    let before = files.changes();
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
    assert_eq!(done.stdout.text.trim_end(), "out");
    assert_eq!(done.stderr.text.trim_end(), "err");
    assert!(done.changed.is_empty() && done.unrecorded.is_empty());
    assert_eq!(files.changes(), before);
    assert_reset(&f, "main");
}

/// **The checkout is switched to the conversation's branch** when it is on
/// another, and stays there.
#[tokio::test]
async fn the_checkout_is_switched_to_the_conversations_branch() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    let done = run(
        &f,
        "feature",
        &files,
        &shell("type FEATURE.md", "cat FEATURE.md"),
    )
    .await;
    assert_eq!(done.exit_code, Some(0), "{done:?}");
    assert!(done.stdout.text.contains("only on the feature branch"));
    assert_reset(&f, "feature");
    assert!(matches!(
        f.sandbox.repo().head().unwrap(),
        Head::Branch { .. }
    ));

    // And back.
    let back = run(&f, "main", &files, &shell("echo x", "echo x")).await;
    assert_eq!(back.exit_code, Some(0));
    assert_reset(&f, "main");
    assert_eq!(disk(&f.root, "FEATURE.md"), None);
}

/// **A binary file the command writes is reported, not lost** — the store
/// holds text only — while its text changes are recorded.
#[tokio::test]
async fn a_binary_file_the_store_cannot_hold_is_reported() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
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
    assert_reset(&f, "main");
}

// ── Failures ─────────────────────────────────────────────────────────────────

/// **A refused command never runs, records nothing, and leaves the checkout
/// reset** — whether the program is unlisted, git, or pointed outside.
#[tokio::test]
async fn a_refused_command_never_runs_and_leaves_the_checkout_reset() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    files.write("mine.txt", "mine\n".into()).unwrap();
    let before = files.changes();
    for (command, want) in [
        (
            SandboxCommand::new("python").arg("x.py"),
            Refused::NotAllowed {
                program: "python".into(),
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
        match f
            .sandbox
            .run(&grant(), &branch("main"), &files, &command)
            .await
        {
            Err(SandboxError::Refused(got)) => assert_eq!(got, want),
            other => panic!("{command:?}: {other:?}"),
        }
        assert_eq!(files.changes(), before);
        assert_reset(&f, "main");
        assert_eq!(disk(&f.root, "mine.txt"), None);
    }
}

/// **A command past its timeout is killed** — and what it changed before
/// that is still recorded.
#[tokio::test]
async fn a_timed_out_command_is_killed_and_its_changes_recorded() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    let command = shell(
        "echo partial> partial.txt & ping -n 30 127.0.0.1 >NUL",
        "echo partial > partial.txt; sleep 30",
    )
    .timeout(Duration::from_millis(300));
    let done = run(&f, "main", &files, &command).await;
    assert!(done.timed_out);
    assert_eq!(done.exit_code, None);
    assert_eq!(read(&files, "partial.txt").unwrap().trim_end(), "partial");
    assert_reset(&f, "main");
}

/// **Changes that no longer fit the branch fail the run before the command
/// starts**, leave the checkout reset, and free the lock for the next run.
#[tokio::test]
async fn changes_that_do_not_fit_fail_before_the_command_runs() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    files
        .edit("README.md", "# app\n\nedited.\n".into())
        .unwrap();
    // The branch moves under the conversation's edit.
    put(&f.root, "README.md", b"# rewritten upstream\n");
    git(&f.root, &["commit", "-q", "-am", "upstream"]);

    let command = shell("echo ran> ran.txt", "echo ran > ran.txt");
    match f
        .sandbox
        .run(&grant(), &branch("main"), &files, &command)
        .await
    {
        Err(SandboxError::Checkout(CheckoutError::Diverged { path })) => {
            assert_eq!(path, "README.md")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(disk(&f.root, "ran.txt"), None, "the command ran");
    assert_reset(&f, "main");

    let fresh = VfsStore::with_root(&f.root);
    let done = run(&f, "main", &fresh, &shell("echo ok", "echo ok")).await;
    assert_eq!(done.stdout.text.trim_end(), "ok");
}

/// **A store that is not an overlay over this repository is refused** before
/// anything is touched.
#[tokio::test]
async fn a_store_over_anything_else_is_refused() {
    let f = fixture();
    let other = tempfile::tempdir().unwrap();
    let command = shell("echo x", "echo x");
    for files in [VfsStore::new(), VfsStore::with_root(other.path())] {
        assert!(matches!(
            f.sandbox
                .run(&grant(), &branch("main"), &files, &command)
                .await,
            Err(SandboxError::WrongStore { .. })
        ));
    }
    let direct = VfsStore::direct(&f.root, &grant());
    assert!(matches!(
        f.sandbox
            .run(&grant(), &branch("main"), &direct, &command)
            .await,
        Err(SandboxError::DirectStore)
    ));
}

// ── Several conversations ────────────────────────────────────────────────────

/// **Runs on one repository take turns.** Two conversations started together
/// each see only their own file, each records only its own change, and the
/// checkout ends as the branch.
#[tokio::test]
async fn runs_on_one_repository_take_turns() {
    let f = fixture();
    let a = VfsStore::with_root(&f.root);
    let b = VfsStore::with_root(&f.root);
    a.write("who.txt", "conversation a\n".into()).unwrap();
    b.write("who.txt", "conversation b\n".into()).unwrap();
    let command = shell(
        "type who.txt & echo touched> touched.log & echo seen>> who.txt",
        "cat who.txt; echo touched > touched.log; echo seen >> who.txt",
    );

    let (ran_a, ran_b) = tokio::join!(run(&f, "main", &a, &command), run(&f, "main", &b, &command));
    assert!(
        ran_a.stdout.text.starts_with("conversation a"),
        "{:?}",
        ran_a.stdout
    );
    assert!(
        ran_b.stdout.text.starts_with("conversation b"),
        "{:?}",
        ran_b.stdout
    );
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
    assert_reset(&f, "main");
    assert_eq!(disk(&f.root, "who.txt"), None);
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
    let files = VfsStore::with_root(&f.root);
    files
        .write("tools/hello.cmd", "@echo from the conversation\r\n".into())
        .unwrap();
    let done = sandbox
        .run(
            &grant(),
            &branch("main"),
            &files,
            &SandboxCommand::new("./tools/hello.cmd"),
        )
        .await
        .unwrap();
    assert_eq!(done.stdout.text.trim_end(), "from the conversation");
    assert_eq!(disk(&f.root, "tools/hello.cmd"), None);

    let fresh = VfsStore::with_root(&f.root);
    assert!(matches!(
        sandbox
            .run(
                &grant(),
                &branch("main"),
                &fresh,
                &SandboxCommand::new("./tools/hello.cmd"),
            )
            .await,
        Err(SandboxError::Refused(Refused::Program { .. }))
    ));
}
