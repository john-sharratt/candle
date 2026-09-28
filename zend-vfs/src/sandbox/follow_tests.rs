//! A job whose conversation pinned origin's copy of its branch while the
//! local branch — checked out, clean, in the owner's folder — lags behind it.

use super::*;
use crate::checkout::CheckoutError;
use crate::testing::TestRepo;
use crate::{GitSource, Oid};

/// The shell and arguments that print one line, per platform.
fn echo() -> (&'static str, [&'static str; 2]) {
    if cfg!(windows) {
        ("cmd", ["/C", "echo x"])
    } else {
        ("sh", ["-c", "echo x"])
    }
}

/// The owner's repository, clean on `main` at `first`, with origin moved on
/// to `second` by someone else and fetched.
fn behind_origin() -> (TestRepo, TestRepo, TestRepo, Oid, Oid) {
    let origin = TestRepo::bare();
    let owner = TestRepo::init();
    owner.write("a.txt", b"one\n");
    let first = owner.commit_all("first");
    owner.git(&["remote", "add", "origin", &origin.url()]);
    owner.git(&["push", "-q", "origin", "main"]);

    let other = TestRepo::init();
    other.git(&["remote", "add", "origin", &origin.url()]);
    other.git(&["fetch", "-q", "origin"]);
    other.git(&["reset", "-q", "--hard", "origin/main"]);
    other.write("a.txt", b"two\n");
    let second = other.commit_all("second");
    other.git(&["push", "-q", "origin", "main"]);

    owner.git(&["fetch", "-q", "origin"]);
    (origin, owner, other, first, second)
}

async fn run_echo(owner: &TestRepo, files: &VfsStore) -> Result<RunOutcome, SandboxError> {
    let (shell, args) = echo();
    let sandbox = Sandbox::new(owner.repo(), CommandPolicy::allowing([shell]));
    let main = BranchName::parse("main").unwrap();
    let command = SandboxCommand::new(shell).args(args);
    let mut out = tokio::io::sink();
    sandbox
        .run(
            &DiskWriteGrant::issue(),
            Job {
                id: "follow",
                branch: &main,
                files,
                command: &command,
                on_lock: None,
            },
            &mut out,
        )
        .await
}

/// **The owner's clean checkout comes back as a branch that moved on**:
/// `main` at origin's commit, `HEAD` on it, nothing uncommitted — never the
/// old files kept as work that reverts what origin gained.
#[tokio::test]
async fn a_lagging_checked_out_branch_follows_origin_and_stays_clean() {
    let (_origin, owner, _other, first, second) = behind_origin();
    let files = VfsStore::on_branch(
        GitSource::open(&owner.path).unwrap(),
        Rev::Branch(BranchName::parse("main").unwrap()),
    );
    assert_eq!(
        files.base().unwrap().unwrap().commit(),
        Some(&second),
        "origin's copy"
    );
    assert_eq!(owner.oid("main"), first);

    run_echo(&owner, &files).await.expect("the job runs");

    assert_eq!(
        owner.oid("main"),
        second,
        "the local branch followed origin"
    );
    assert_eq!(
        owner.git(&["symbolic-ref", "HEAD"]).trim(),
        "refs/heads/main"
    );
    assert_eq!(
        owner.git(&["status", "--porcelain", "--untracked-files=all"]),
        "",
        "no uncommitted change was left behind"
    );
    assert_eq!(owner.read("a.txt"), b"two\n");
}

/// **The owner's own work comes back over the branch that moved on** —
/// untouched by what origin gained, and nothing of origin's reverted.
#[tokio::test]
async fn the_owners_own_work_comes_back_over_the_followed_branch() {
    let (_origin, owner, _other, _first, second) = behind_origin();
    owner.write("notes.txt", b"mine\n");
    let files = VfsStore::on_branch(
        GitSource::open(&owner.path).unwrap(),
        Rev::Branch(BranchName::parse("main").unwrap()),
    );

    run_echo(&owner, &files).await.expect("the job runs");

    assert_eq!(owner.oid("main"), second);
    assert_eq!(
        owner.git(&["status", "--porcelain", "--untracked-files=all"]),
        "?? notes.txt\n"
    );
    assert_eq!(owner.read("notes.txt"), b"mine\n");
    assert_eq!(owner.read("a.txt"), b"two\n");
}

/// **The owner's uncommitted change to a file origin changed too stops the
/// follow**, as `git merge --ff-only` would: nothing moves, and the change
/// is back as it was.
#[tokio::test]
async fn an_owners_change_to_a_file_origin_changed_refuses_the_follow() {
    let (_origin, owner, _other, first, _second) = behind_origin();
    owner.write("a.txt", b"my edit\n");
    let files = VfsStore::on_branch(
        GitSource::open(&owner.path).unwrap(),
        Rev::Branch(BranchName::parse("main").unwrap()),
    );

    let refused = run_echo(&owner, &files).await.unwrap_err();
    assert!(
        matches!(
            &refused,
            SandboxError::Checkout(CheckoutError::OwnWorkInTheWay { paths, .. })
                if paths == &["a.txt".to_string()]
        ),
        "{refused}"
    );
    assert_eq!(owner.oid("main"), first, "nothing moved");
    assert_eq!(owner.read("a.txt"), b"my edit\n");
    assert_eq!(
        owner.git(&["status", "--porcelain", "--untracked-files=all"]),
        " M a.txt\n"
    );
}

/// **A local branch holding commits origin lacks is never moved**, and the
/// job is refused as before.
#[tokio::test]
async fn a_branch_with_unpushed_commits_is_not_moved() {
    let (_origin, owner, _other, _first, second) = behind_origin();
    owner.write("b.txt", b"mine\n");
    let mine = owner.commit_all("unpushed");
    let files = VfsStore::on_branch(
        GitSource::open(&owner.path).unwrap(),
        Rev::Branch(BranchName::parse("main").unwrap()),
    );
    assert_eq!(files.base().unwrap().unwrap().commit(), Some(&second));

    let refused = run_echo(&owner, &files).await.unwrap_err();
    assert!(matches!(refused, SandboxError::Behind { .. }), "{refused}");
    assert_eq!(owner.oid("main"), mine);
}
