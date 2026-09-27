//! git run directly through the sandbox: refused before anything starts, with
//! an error that sends the caller to the git tools — and the word `git` in a
//! command that does not run it passes untouched.
//!
//! The reading of commands itself is pinned case by case in the sandbox's
//! `git_use` tests; these check what a run does with the answer.

mod support;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use support::*;
use zend_vfs::sandbox::Refused;
use zend_vfs::{SandboxCommand, SandboxError, VfsStore};

/// Run `command` expecting the git refusal; returns what it named.
async fn refused(f: &Fixture, files: &VfsStore, command: &SandboxCommand) -> String {
    match f
        .sandbox
        .run(&grant(), &branch("main"), files, command)
        .await
    {
        Err(SandboxError::Refused(Refused::Git { command })) => command,
        other => panic!("{command:?} was not refused as git: {other:?}"),
    }
}

/// **git as the program is refused, and the error tells the caller to use
/// the git tools** — nothing runs, nothing is recorded, the checkout is reset
/// and the next run goes ahead.
#[tokio::test]
async fn git_as_the_program_is_refused_towards_the_git_tools() {
    let f = fixture_allowing(&[SHELL, "git"]);
    let files = VfsStore::with_root(&f.root);
    files.write("mine.txt", "mine\n".into()).unwrap();
    let before = files.changes();

    let result = f
        .sandbox
        .run(
            &grant(),
            &branch("main"),
            &files,
            &SandboxCommand::new("git").args(["commit", "-am", "wip"]),
        )
        .await;
    let error = result.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("`git commit -am wip` runs git directly"),
        "{message}"
    );
    assert!(message.contains("use the git tools instead"), "{message}");
    for tool in [
        "git_status",
        "git_log",
        "git_show",
        "git_grep",
        "git_refs",
        "git_commit",
        "git_ref",
        "git_fetch",
        "git_push",
    ] {
        assert!(message.contains(tool), "{tool} missing from: {message}");
    }
    assert_eq!(files.changes(), before);
    assert_reset(&f, "main");
    assert_eq!(disk(&f.root, "mine.txt"), None);

    let next = run(&f, "main", &files, &shell("echo next", "echo next")).await;
    assert_eq!(next.stdout.text.trim_end(), "next");
}

/// **git anywhere in a script is refused before the script starts** — the
/// commands before it never run either.
#[tokio::test]
async fn git_in_a_script_is_refused_before_anything_runs() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    let named = refused(
        &f,
        &files,
        &shell(
            "echo ran> ran.txt & git status & echo after> after.txt",
            "echo ran > ran.txt; git status; echo after > after.txt",
        ),
    )
    .await;
    assert_eq!(named, "git status");
    assert_eq!(disk(&f.root, "ran.txt"), None, "the script started");
    assert!(!files.is_modified("ran.txt"));
    assert_reset(&f, "main");
}

/// **Every way of running git directly through a command is refused**, on
/// any platform — the refusal comes before anything is started, so the
/// shells named need not exist here.
#[tokio::test]
async fn every_direct_route_to_git_is_refused() {
    let f = fixture_allowing(&["sh", "bash", "cmd", "powershell", "pwsh", "env", "timeout"]);
    let files = VfsStore::with_root(&f.root);
    // PowerShell's `-EncodedCommand`: base64 of UTF-16LE.
    let encoded = {
        let utf16: Vec<u8> = "Write-Host hi; git fetch"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        BASE64.encode(utf16)
    };
    for (command, named) in [
        (
            SandboxCommand::new("sh").args(["-c", "cargo build && git commit -am wip"]),
            "git commit -am wip",
        ),
        (
            SandboxCommand::new("bash").args(["-lc", "cd sub; git log"]),
            "git log",
        ),
        (
            SandboxCommand::new("sh").args(["-c", "echo $(git rev-parse HEAD)"]),
            "git rev-parse HEAD",
        ),
        (
            SandboxCommand::new("cmd").args(["/D", "/C", "dir & git status"]),
            "git status",
        ),
        (
            SandboxCommand::new("cmd").args(["/C", "if exist x git log"]),
            "if exist x git log",
        ),
        (
            SandboxCommand::new("cmd").args(["/C", "bash -c 'git push'"]),
            "bash -c 'git push'",
        ),
        (
            SandboxCommand::new("pwsh").args(["-Command", "Write-Host $(git log -1)"]),
            "git log -1",
        ),
        (
            SandboxCommand::new("powershell").args(["-EncodedCommand", &encoded]),
            "git fetch",
        ),
        (
            SandboxCommand::new("env").args(["GIT_TRACE=1", "git", "status"]),
            "env GIT_TRACE=1 git status",
        ),
        (
            SandboxCommand::new("timeout").args(["30", "git", "fetch"]),
            "timeout 30 git fetch",
        ),
        (
            SandboxCommand::new("/usr/bin/git").arg("log"),
            "/usr/bin/git log",
        ),
        (SandboxCommand::new("GIT.EXE").arg("log"), "GIT.EXE log"),
    ] {
        assert_eq!(refused(&f, &files, &command).await, named, "{command:?}");
    }
    // Refused before the checkout was touched at all.
    assert_reset(&f, "main");
}

/// **git is refused as git even where the program is not allowed at all** —
/// the answer the caller needs is the git tools, not "not allowed".
#[tokio::test]
async fn git_is_named_as_git_before_the_allow_list() {
    let f = fixture_allowing(&[]);
    let files = VfsStore::with_root(&f.root);
    assert_eq!(
        refused(&f, &files, &SandboxCommand::new("git").arg("status")).await,
        "git status"
    );
    assert_eq!(
        refused(
            &f,
            &files,
            &SandboxCommand::new("sh").args(["-c", "git status"])
        )
        .await,
        "git status"
    );
    // Without git, an unlisted program is refused as unlisted.
    assert!(matches!(
        f.sandbox
            .run(
                &grant(),
                &branch("main"),
                &files,
                &SandboxCommand::new("sh").args(["-c", "echo git"]),
            )
            .await,
        Err(SandboxError::Refused(Refused::NotAllowed { .. }))
    ));
}

/// **Commands that only mention git run as normal** — its name echoed, its
/// files read, text about it quoted and escaped, its presence asked about, a
/// comment — all in one script, each printing its mark.
#[tokio::test]
async fn commands_that_only_mention_git_run() {
    let f = fixture();
    let files = VfsStore::with_root(&f.root);
    let command = shell(
        "echo git status & type .gitignore & echo \"quoted & git log\" & \
         echo escaped ^& git log & where git >NUL 2>&1 & echo checked & echo (git) & \
         echo commented & rem git status",
        "echo git status; cat .gitignore; echo \"quoted; git log\"; \
         echo 'escaped && git log'; command -v git >/dev/null; echo checked; echo '(git)'; \
         echo commented # git status",
    );
    let done = run(&f, "main", &files, &command).await;
    for mark in [
        "git status",
        "target/",
        "quoted",
        "escaped",
        "checked",
        "(git)",
        "commented",
    ] {
        assert!(
            done.stdout.text.contains(mark),
            "{mark:?} missing from {:?}",
            done.stdout.text
        );
    }
    assert_reset(&f, "main");
}
