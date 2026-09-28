//! A job whose conversation pinned origin's copy of its branch while the
//! local branch — in the owner's folder — stands elsewhere: behind origin's
//! copy, or holding commits of the owner's origin never had.

use super::*;
use crate::checkout::CheckoutError;
use crate::testing::TestRepo;
use crate::{GitSource, Oid};

/// What a job runs: a shell script, per platform.
struct Script {
    windows: &'static str,
    unix: &'static str,
}

/// Prints one line.
const ECHO: Script = Script {
    windows: "echo x",
    unix: "echo x",
};

/// Prints whether the checkout the job runs on holds `mine.txt`.
const MINE_THERE: Script = Script {
    windows: "if exist mine.txt (echo present) else (echo absent)",
    unix: "if [ -e mine.txt ]; then echo present; else echo absent; fi",
};

/// Prints `a.txt` as the checkout the job runs on holds it.
const PRINT_A: Script = Script {
    windows: "type a.txt",
    unix: "cat a.txt",
};

fn main_branch() -> BranchName {
    BranchName::parse("main").unwrap()
}

/// Origin and the owner's repository, both on `main` at `first`.
fn published() -> (TestRepo, TestRepo, Oid) {
    let origin = TestRepo::bare();
    let owner = TestRepo::init();
    owner.write("a.txt", b"one\n");
    let first = owner.commit_all("first");
    owner.git(&["remote", "add", "origin", &origin.url()]);
    owner.git(&["push", "-q", "origin", "main"]);
    (origin, owner, first)
}

/// Someone else's clone of `origin`, pushing `a.txt` as `content` onto
/// origin's `main` as a new commit.
fn push_from_elsewhere(origin: &TestRepo, content: &[u8]) -> (TestRepo, Oid) {
    let other = TestRepo::init();
    other.git(&["remote", "add", "origin", &origin.url()]);
    other.git(&["fetch", "-q", "origin"]);
    other.git(&["reset", "-q", "--hard", "origin/main"]);
    other.write("a.txt", content);
    let pushed = other.commit_all("from elsewhere");
    other.git(&["push", "-q", "origin", "main"]);
    (other, pushed)
}

/// The owner's repository, clean on `main` at `first`, with origin moved on
/// to `second` by someone else and fetched.
fn behind_origin() -> (TestRepo, TestRepo, TestRepo, Oid, Oid) {
    let (origin, owner, first) = published();
    let (other, second) = push_from_elsewhere(&origin, b"two\n");
    owner.git(&["fetch", "-q", "origin"]);
    (origin, owner, other, first, second)
}

/// A commit of the owner's on the branch checked out, never pushed: it adds
/// `mine.txt`.
fn commit_unpushed(owner: &TestRepo) -> Oid {
    owner.write("mine.txt", b"never pushed\n");
    owner.commit_all("unpushed")
}

fn store_on_main(owner: &TestRepo) -> VfsStore {
    VfsStore::on_branch(
        GitSource::open(&owner.path).unwrap(),
        Rev::Branch(main_branch()),
    )
}

/// The commit `files` pinned.
fn pinned(files: &VfsStore) -> Oid {
    files.base().unwrap().unwrap().commit().cloned().unwrap()
}

/// Run `script` as a job on `main`, returning what it printed, trimmed.
async fn run(
    owner: &TestRepo,
    files: &VfsStore,
    script: &Script,
) -> Result<(RunOutcome, String), SandboxError> {
    let (shell, flag, line) = if cfg!(windows) {
        ("cmd", "/C", script.windows)
    } else {
        ("sh", "-c", script.unix)
    };
    let sandbox = Sandbox::new(owner.repo(), CommandPolicy::allowing([shell]));
    let main = main_branch();
    let command = SandboxCommand::new(shell).args([flag, line]);
    let mut out: Vec<u8> = Vec::new();
    let outcome = sandbox
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
        .await?;
    Ok((outcome, String::from_utf8(out).unwrap().trim().to_string()))
}

fn status(owner: &TestRepo) -> String {
    owner.git(&["status", "--porcelain", "--untracked-files=all"])
}

/// Nothing of the job is left in the owner's repository: no preservation
/// ref, no preservation folder.
fn assert_nothing_left(owner: &TestRepo) {
    assert_eq!(owner.git(&["for-each-ref", "refs/zend/preserved/"]), "");
    let left = std::fs::read_dir(owner.path.join(".git").join("zend-preserved"))
        .map(|entries| {
            entries
                .filter(|e| e.as_ref().is_ok_and(|e| e.path().is_dir()))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(left, 0, "a preservation folder was left");
}

/// **The owner's clean checkout comes back as a branch that moved on**:
/// `main` at origin's commit, `HEAD` on it, nothing uncommitted — never the
/// old files kept as work that reverts what origin gained.
#[tokio::test]
async fn a_lagging_checked_out_branch_follows_origin_and_stays_clean() {
    let (_origin, owner, _other, first, second) = behind_origin();
    let files = store_on_main(&owner);
    assert_eq!(pinned(&files), second, "origin's copy");
    assert_eq!(owner.oid("main"), first);

    run(&owner, &files, &ECHO).await.expect("the job runs");

    assert_eq!(
        owner.oid("main"),
        second,
        "the local branch followed origin"
    );
    assert_eq!(
        owner.git(&["symbolic-ref", "HEAD"]).trim(),
        "refs/heads/main"
    );
    assert_eq!(status(&owner), "", "no uncommitted change was left behind");
    assert_eq!(owner.read("a.txt"), b"two\n");
}

/// **The owner's own work comes back over the branch that moved on** —
/// untouched by what origin gained, and nothing of origin's reverted.
#[tokio::test]
async fn the_owners_own_work_comes_back_over_the_followed_branch() {
    let (_origin, owner, _other, _first, second) = behind_origin();
    owner.write("notes.txt", b"mine\n");
    let files = store_on_main(&owner);

    run(&owner, &files, &ECHO).await.expect("the job runs");

    assert_eq!(owner.oid("main"), second);
    assert_eq!(status(&owner), "?? notes.txt\n");
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
    let files = store_on_main(&owner);

    let refused = run(&owner, &files, &ECHO).await.unwrap_err();
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
    assert_eq!(status(&owner), " M a.txt\n");
}

/// **Commits the owner never pushed are set aside for the job and come
/// back**: the job runs on origin's copy — the one the conversation pinned —
/// without them, and afterwards `main` is back at them, `HEAD` on it,
/// nothing uncommitted, and origin never got them.
#[tokio::test]
async fn unpushed_commits_are_set_aside_for_the_job_and_come_back() {
    let (origin, owner, first) = published();
    let mine = commit_unpushed(&owner);
    let files = store_on_main(&owner);
    assert_eq!(pinned(&files), first, "origin's copy, not the unpushed one");

    let (_, printed) = run(&owner, &files, &MINE_THERE)
        .await
        .expect("the job runs");

    assert_eq!(printed, "absent", "the job ran without the unpushed commit");
    assert_eq!(owner.oid("main"), mine, "the unpushed commit is back");
    assert_eq!(
        owner.git(&["symbolic-ref", "HEAD"]).trim(),
        "refs/heads/main"
    );
    assert_eq!(status(&owner), "");
    assert_eq!(owner.read("mine.txt"), b"never pushed\n");
    assert_eq!(origin.oid("main"), first, "nothing was published");
    assert_nothing_left(&owner);
}

/// **The owner's uncommitted work over unpushed commits comes back with
/// them** — a staged edit staged, an unstaged one unstaged, an untracked
/// file untracked.
#[tokio::test]
async fn the_owners_work_over_unpushed_commits_comes_back() {
    let (_origin, owner, _first) = published();
    let mine = commit_unpushed(&owner);
    owner.write("mine.txt", b"never pushed, staged edit\n");
    owner.git(&["add", "mine.txt"]);
    owner.write("a.txt", b"one, edited\n");
    owner.write("notes.txt", b"notes\n");
    let before = status(&owner);
    let files = store_on_main(&owner);

    let (_, printed) = run(&owner, &files, &PRINT_A).await.expect("the job runs");

    assert_eq!(printed, "one", "the job read origin's a.txt, not the edit");
    assert_eq!(owner.oid("main"), mine);
    assert_eq!(status(&owner), before);
    assert_eq!(before, " M a.txt\nM  mine.txt\n?? notes.txt\n");
    assert_eq!(owner.read("mine.txt"), b"never pushed, staged edit\n");
    assert_eq!(owner.read("a.txt"), b"one, edited\n");
    assert_eq!(owner.read("notes.txt"), b"notes\n");
    assert_nothing_left(&owner);
}

/// **A branch that diverged — the owner's unpushed commit on one side,
/// someone else's pushed one on the other — runs on origin's copy**, and
/// comes back at the owner's commit; origin keeps its own.
#[tokio::test]
async fn a_diverged_branch_runs_on_origins_copy_and_comes_back() {
    let (origin, owner, _other, _first, second) = behind_origin();
    let mine = commit_unpushed(&owner);
    let files = store_on_main(&owner);
    assert_eq!(pinned(&files), second);

    let (_, printed) = run(&owner, &files, &PRINT_A).await.expect("the job runs");

    assert_eq!(printed, "two", "origin's a.txt");
    assert_eq!(owner.oid("main"), mine, "the owner's side is back");
    assert_eq!(status(&owner), "");
    assert_eq!(owner.read("a.txt"), b"one\n", "the owner's own a.txt");
    assert_eq!(origin.oid("main"), second);
    assert_nothing_left(&owner);
}

/// **A file the owner has untracked where origin's copy tracks one is
/// theirs after the job** — the job's checkout wrote origin's over it, and
/// the owner's bytes come back.
#[tokio::test]
async fn an_untracked_file_origin_tracks_comes_back_as_the_owners() {
    let (origin, owner, _first) = published();
    let other = TestRepo::init();
    other.git(&["remote", "add", "origin", &origin.url()]);
    other.git(&["fetch", "-q", "origin"]);
    other.git(&["reset", "-q", "--hard", "origin/main"]);
    other.write("shared.txt", b"origin's\n");
    let theirs = other.commit_all("adds shared.txt");
    other.git(&["push", "-q", "origin", "main"]);
    owner.git(&["fetch", "-q", "origin"]);
    let mine = commit_unpushed(&owner);
    owner.write("shared.txt", b"the owner's, untracked\n");
    let files = store_on_main(&owner);
    assert_eq!(pinned(&files), theirs);

    run(&owner, &files, &ECHO).await.expect("the job runs");

    assert_eq!(owner.oid("main"), mine);
    assert_eq!(owner.read("shared.txt"), b"the owner's, untracked\n");
    assert_eq!(status(&owner), "?? shared.txt\n");
    assert_nothing_left(&owner);
}

/// **A branch with unpushed commits that the owner has not checked out
/// comes back at them too** — and the owner's checkout, on another branch,
/// comes back as it was.
#[tokio::test]
async fn an_unpushed_branch_not_checked_out_comes_back() {
    let (_origin, owner, first) = published();
    let mine = commit_unpushed(&owner);
    owner.git(&["checkout", "-q", "-b", "topic", first.as_str()]);
    owner.write("a.txt", b"topic edit\n");
    let files = store_on_main(&owner);

    let (_, printed) = run(&owner, &files, &MINE_THERE)
        .await
        .expect("the job runs");

    assert_eq!(printed, "absent");
    assert_eq!(owner.oid("main"), mine);
    assert_eq!(
        owner.git(&["symbolic-ref", "HEAD"]).trim(),
        "refs/heads/topic"
    );
    assert_eq!(status(&owner), " M a.txt\n");
    assert_eq!(owner.read("a.txt"), b"topic edit\n");
    assert_nothing_left(&owner);
}

/// **A job that fails once the branch is set aside puts it back all the
/// same** — here a program that cannot be found.
#[tokio::test]
async fn a_failing_job_puts_the_unpushed_commits_back() {
    let (_origin, owner, _first) = published();
    let mine = commit_unpushed(&owner);
    let files = store_on_main(&owner);
    let sandbox = Sandbox::new(
        owner.repo(),
        CommandPolicy::allowing(["no-such-program-anywhere"]),
    );
    let main = main_branch();
    let command = SandboxCommand::new("no-such-program-anywhere");
    let failed = sandbox
        .run(
            &DiskWriteGrant::issue(),
            Job {
                id: "fails",
                branch: &main,
                files: &files,
                command: &command,
                on_lock: None,
            },
            &mut tokio::io::sink(),
        )
        .await;

    assert!(
        matches!(failed, Err(SandboxError::Start { .. })),
        "{failed:?}"
    );
    assert_eq!(owner.oid("main"), mine);
    assert_eq!(status(&owner), "");
    assert_nothing_left(&owner);
}

/// **A job a crash cut short is put back when the sandbox is brought up** —
/// the unpushed commit on its branch again, the owner's edit over it —
/// without waiting for the next job.
#[test]
fn a_crash_is_recovered_when_the_sandbox_comes_up() {
    let (_origin, owner, first) = published();
    let mine = commit_unpushed(&owner);
    owner.write("a.txt", b"my edit\n");
    let repo = Arc::new(owner.repo());
    let mut kept = checkout::preserve(repo, "a job ran", &[], Some(&first)).unwrap();
    kept.set_branch_aside(&main_branch(), &mine, &first)
        .unwrap();
    kept.crash();
    assert_eq!(owner.oid("main"), first, "left set aside");

    let sandbox = Sandbox::new(owner.repo(), CommandPolicy::allowing(["sh"]));
    sandbox.recover().unwrap();
    assert_eq!(owner.oid("main"), mine);
    assert_eq!(
        owner.git(&["symbolic-ref", "HEAD"]).trim(),
        "refs/heads/main"
    );
    assert_eq!(status(&owner), " M a.txt\n");
    assert_eq!(owner.read("a.txt"), b"my edit\n");
    assert_eq!(owner.read("mine.txt"), b"never pushed\n");
    assert_nothing_left(&owner);
}

/// **A conversation made on a commit origin has since moved past is still
/// behind, unpushed commits or not**: it merges first, and nothing moves —
/// the refusal names what origin holds, not the owner's commit.
#[tokio::test]
async fn a_conversation_behind_origin_is_refused_whatever_is_unpushed() {
    let (origin, owner, first) = published();
    let files = store_on_main(&owner);
    assert_eq!(pinned(&files), first);
    let (_other, second) = push_from_elsewhere(&origin, b"two\n");
    owner.git(&["fetch", "-q", "origin"]);
    owner.git(&["merge", "-q", "--ff-only", "origin/main"]);
    let mine = commit_unpushed(&owner);

    let refused = run(&owner, &files, &ECHO).await.unwrap_err();
    assert!(
        matches!(&refused, SandboxError::Behind { tip, .. } if tip == second.as_str()),
        "{refused}"
    );
    assert_eq!(owner.oid("main"), mine, "nothing moved");
    assert_eq!(status(&owner), "");
}
