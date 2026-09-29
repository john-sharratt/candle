//! The workspace's sandboxes over a real git repository.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use tempfile::TempDir;
use zend_vfs::{
    BranchName, CommandPolicy, DiskWriteGrant, GitSource, JobId, JobRequest, RepoSpec, Rev,
    SandboxCommand, VfsStore, Workspace, JOBS_DIR,
};

use super::{Sandboxes, SandboxesError};

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.hooksPath=", "-c", "commit.gpgSign=false"])
        .args(args)
        .env("GIT_AUTHOR_NAME", "Setup")
        .env("GIT_AUTHOR_EMAIL", "setup@example.com")
        .env("GIT_COMMITTER_NAME", "Setup")
        .env("GIT_COMMITTER_EMAIL", "setup@example.com")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .expect("git runs");
    assert!(out.status.success(), "git {args:?}: {out:?}");
}

/// A workspace of `app`, a git repository on `main` with one commit, and
/// `plain`, a folder that is not one.
fn workspace() -> (TempDir, Workspace) {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("app");
    std::fs::create_dir_all(&app).unwrap();
    std::fs::create_dir_all(dir.path().join("plain")).unwrap();
    git(&app, &["init", "-q"]);
    git(&app, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    git(&app, &["config", "core.autocrlf", "false"]);
    std::fs::write(app.join("a.txt"), "a\n").unwrap();
    git(&app, &["add", "-A"]);
    git(&app, &["commit", "-q", "-m", "base"]);
    let ws = Workspace::new(
        dir.path(),
        vec![RepoSpec::named("app"), RepoSpec::named("plain")],
    )
    .unwrap();
    (dir, ws)
}

/// **A `.git` that cannot be opened is an error**, naming the repository —
/// not a folder quietly left without a sandbox.
#[test]
fn an_unopenable_repository_is_an_error() {
    let (dir, ws) = workspace();
    std::fs::write(dir.path().join("plain").join(".git"), "not a repository\n").unwrap();
    let err = match Sandboxes::for_workspace(&ws, &CommandPolicy::allowing(["cmd", "sh"])) {
        Ok(_) => panic!("an unopenable .git was skipped"),
        Err(err) => err,
    };
    assert!(err.to_string().starts_with("repository plain:"), "{err}");
}

/// The platform's shell, and a script for it.
fn shell(script: &str) -> SandboxCommand {
    if cfg!(windows) {
        SandboxCommand::new("cmd").args(["/D", "/C", script])
    } else {
        SandboxCommand::new("sh").args(["-c", script])
    }
}

fn policy() -> CommandPolicy {
    CommandPolicy::allowing([if cfg!(windows) { "cmd" } else { "sh" }])
}

fn store(ws: &Workspace) -> Arc<VfsStore> {
    let main = BranchName::parse("main").unwrap();
    Arc::new(VfsStore::on_branch(
        GitSource::open(&ws.repo("app").unwrap().dir).unwrap(),
        Rev::Branch(main),
    ))
}

fn request(files: &Arc<VfsStore>, command: SandboxCommand) -> JobRequest {
    JobRequest {
        branch: BranchName::parse("main").unwrap(),
        files: Arc::clone(files),
        command,
    }
}

/// **A git repository has a sandbox; a plain folder has none**, and the
/// jobs folder is the workspace folder's.
#[test]
fn only_git_repositories_have_a_sandbox() {
    let (dir, ws) = workspace();
    let boxes = Sandboxes::for_workspace(&ws, &policy()).unwrap();
    assert_eq!(boxes.repos().collect::<Vec<_>>(), ["app"]);
    assert_eq!(boxes.jobs_dir(), dir.path().join(JOBS_DIR));
    assert!(boxes.jobs_dir().is_dir());
    let files = store(&ws);
    let err = boxes
        .run(
            DiskWriteGrant::issue(),
            "plain",
            request(&files, shell("echo x")),
        )
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "plain has no sandbox — commands run only in the workspace's git repositories: app"
    );
}

/// **A job runs from a thread with no runtime, what it writes comes back into
/// the store, and its output reads back a page at a time** — while the
/// repository's folder is left as it was.
#[test]
fn a_job_runs_records_what_it_wrote_and_pages_its_output() {
    let (_dir, ws) = workspace();
    let boxes = Sandboxes::for_workspace(&ws, &policy()).unwrap();
    let files = store(&ws);
    let script = if cfg!(windows) {
        // `cmd` echoes the space before an `&`, so there is none.
        "echo made> out.txt& echo first& echo second"
    } else {
        "echo made > out.txt; echo first; echo second"
    };
    let ran = boxes
        .run(
            DiskWriteGrant::issue(),
            "app",
            request(&files, shell(script)),
        )
        .unwrap();
    assert_eq!(ran.outcome.exit_code, Some(0));
    assert_eq!(
        ran.outcome
            .changed
            .iter()
            .map(|c| c.path.as_str())
            .collect::<Vec<_>>(),
        ["out.txt"]
    );
    let written = files.read("out.txt").unwrap().unwrap();
    assert_eq!(written.trim_end(), "made");
    assert!(
        !ws.repo("app").unwrap().dir.join("out.txt").exists(),
        "the folder is put back"
    );

    let (info, page) = boxes.output("app", &ran.job, 0).unwrap();
    assert_eq!(info.log, ran.log);
    assert_eq!(page.text.trim_end(), "first\nsecond");
    assert_eq!((page.page, page.pages, page.total_lines), (0, 1, 2));
}

/// **A refused command is the policy's answer**, and it never runs.
#[test]
fn a_refused_command_never_runs() {
    let (_dir, ws) = workspace();
    let boxes = Sandboxes::for_workspace(&ws, &policy()).unwrap();
    let files = store(&ws);
    let err = boxes
        .run(
            DiskWriteGrant::issue(),
            "app",
            request(&files, SandboxCommand::new("python")),
        )
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        format!(
            "refused: python is not a program this repository's sandbox runs; it runs: {}",
            policy().allowed()
        )
    );
    assert!(files.changes().unwrap().paths().next().is_none());
}

/// A job the sandbox never ran is not found.
#[test]
fn an_unknown_job_is_not_found() {
    let (_dir, ws) = workspace();
    let boxes = Sandboxes::for_workspace(&ws, &policy()).unwrap();
    let id = JobId::parse("AAAAAAAAAAA").unwrap();
    assert!(matches!(
        boxes.output("app", &id, 0),
        Err(SandboxesError::NotFound(_))
    ));
}
