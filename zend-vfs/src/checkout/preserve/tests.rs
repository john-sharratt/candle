#[cfg(windows)]
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;

use super::journal::PRESERVED_DIR;
use super::*;
use crate::testing::TestRepo;

/// `main` with two files and an ignore rule for `target/` and `*.local`, and
/// a branch `job` with `a.txt` changed.
fn repo() -> TestRepo {
    let t = TestRepo::init();
    t.write(".gitignore", b"target/\n*.local\n");
    t.write("a.txt", b"a\n");
    t.write("b.txt", b"b\n");
    t.commit_all("base");
    t.git(&["branch", "job"]);
    t
}

fn keep(t: &TestRepo, writes: &[&str]) -> Preserved {
    let writes: Vec<String> = writes.iter().map(|w| w.to_string()).collect();
    preserve(Arc::new(t.repo()), "a test ran", &writes, None).unwrap()
}

/// What a run does to the checkout: another branch, files written, changed
/// and deleted, a stray file, a build output.
fn run_on(t: &TestRepo) {
    t.git(&["checkout", "-q", "-f", "job"]);
    t.write("a.txt", b"the job's\n");
    std::fs::remove_file(t.path.join("b.txt")).unwrap();
    t.write("stray.txt", b"left by the job\n");
    t.write("target/out.bin", b"built\n");
}

fn status(t: &TestRepo) -> String {
    t.git(&[
        "status",
        "--porcelain=v1",
        "--branch",
        "--untracked-files=all",
    ])
}

/// Nothing of any preservation is left: no folder, no ref holding one. The
/// lock file stays, free; what a recovery kept of the checkout stays too.
fn assert_released(t: &TestRepo) {
    let root = t.path.join(".git").join(PRESERVED_DIR);
    let folders = match std::fs::read_dir(&root) {
        Ok(entries) => entries
            .filter(|e| e.as_ref().is_ok_and(|e| e.path().is_dir()))
            .count(),
        Err(_) => 0,
    };
    assert_eq!(folders, 0, "a preservation folder was left");
    assert_eq!(
        t.git(&["for-each-ref", "refs/zend/preserved/"]),
        "",
        "a ref was left"
    );
}

/// **Everything in the checkout comes back exactly** — the branch, a staged
/// edit staged, an unstaged edit over it, a new file still untracked, a
/// deleted file still deleted — whatever a run did in between; nothing is
/// left behind, the stash list is never touched, and a build output stays.
#[test]
fn everything_in_the_checkout_comes_back_exactly() {
    let t = repo();
    t.write("x.txt", b"theirs\n");
    t.git(&[
        "stash",
        "push",
        "-q",
        "--include-untracked",
        "-m",
        "someone's own",
    ]);
    let stashes = t.git(&["stash", "list"]);
    t.git(&["checkout", "-q", "-b", "mine"]);
    t.write("a.txt", b"staged\n");
    t.git(&["add", "a.txt"]);
    t.write("a.txt", b"staged, then more\n");
    t.write("new.txt", b"not added yet\n");
    std::fs::remove_file(t.path.join("b.txt")).unwrap();
    let before = status(&t);
    let staged = t.git(&["diff", "--cached"]);

    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();

    assert_eq!(status(&t), before);
    assert_eq!(t.git(&["diff", "--cached"]), staged, "the index as it was");
    assert_eq!(t.read("a.txt"), b"staged, then more\n");
    assert_eq!(t.read("new.txt"), b"not added yet\n");
    assert!(!t.path.join("b.txt").exists());
    assert!(!t.path.join("stray.txt").exists());
    assert_eq!(
        t.read("target/out.bin"),
        b"built\n",
        "the build output stays"
    );
    assert_eq!(
        t.git(&["stash", "list"]),
        stashes,
        "the stash list untouched"
    );
    assert_released(&t);
}

/// **Files come back byte for byte**: no line-ending conversion or filter
/// is applied on the way out or back, whatever the repository's settings.
#[test]
fn files_come_back_byte_for_byte() {
    let t = repo();
    t.git(&["config", "core.autocrlf", "true"]);
    let mixed: &[u8] = b"lf\ncrlf\r\nlone cr\rend";
    t.write("a.txt", mixed);
    t.write("untracked.txt", b"only lf\nhere\n");
    t.write("bin.dat", &[0, 13, 10, 10, 13, 255]);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(t.read("a.txt"), mixed);
    assert_eq!(t.read("untracked.txt"), b"only lf\nhere\n");
    assert_eq!(t.read("bin.dat"), [0, 13, 10, 10, 13, 255]);
}

/// **Dropped without restoring — a failed or abandoned run — the guard puts
/// the checkout back all the same.**
#[test]
fn dropping_the_guard_puts_the_checkout_back() {
    let t = repo();
    t.write("a.txt", b"my edit\n");
    let kept = keep(&t, &[]);
    run_on(&t);
    drop(kept);
    assert_eq!(t.git(&["symbolic-ref", "--short", "HEAD"]).trim(), "main");
    assert_eq!(t.read("a.txt"), b"my edit\n");
    assert_eq!(t.read("b.txt"), b"b\n");
    assert_released(&t);
}

/// **After a crash, the next preservation puts the last one back first** —
/// from its journal alone.
#[test]
fn a_crashed_runs_state_is_put_back_by_the_next() {
    let t = repo();
    t.git(&["checkout", "-q", "-b", "mine"]);
    t.write("a.txt", b"my edit\n");
    t.write("new.txt", b"mine\n");
    let before = status(&t);
    keep(&t, &[]).crash();
    run_on(&t);

    let mut next = keep(&t, &[]);
    assert_eq!(
        status(&t),
        before,
        "the crashed run's state was put back first"
    );
    next.restore().unwrap();
    assert_eq!(status(&t), before);
    assert_eq!(t.read("new.txt"), b"mine\n");
    assert_released(&t);

    keep(&t, &[]).crash();
    run_on(&t);
    recover(&Arc::new(t.repo())).unwrap();
    assert_eq!(status(&t), before, "recovered directly");
    assert_released(&t);
}

/// **A clean checkout is put back on its `HEAD`; a detached one goes back
/// detached at its commit.**
#[test]
fn a_clean_or_detached_checkout_comes_back() {
    let t = repo();
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(status(&t), "## main\n");

    let at = t.oid("main");
    t.git(&["checkout", "-q", "--detach"]);
    t.write("a.txt", b"detached edit\n");
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(t.git(&["rev-parse", "HEAD"]).trim(), at.as_str());
    assert_eq!(t.read("a.txt"), b"detached edit\n");
}

/// **A branch the run deleted is made again at the commit it was at**, and
/// `HEAD` goes back on it.
#[test]
fn a_branch_the_run_deleted_is_made_again() {
    let t = repo();
    t.git(&["checkout", "-q", "-b", "mine"]);
    let at = t.oid("mine");
    t.write("a.txt", b"my edit\n");
    let mut kept = keep(&t, &[]);
    run_on(&t);
    t.git(&["branch", "-D", "mine"]);
    kept.restore().unwrap();
    assert_eq!(t.git(&["symbolic-ref", "--short", "HEAD"]).trim(), "mine");
    assert_eq!(t.oid("mine"), at);
    assert_eq!(t.read("a.txt"), b"my edit\n");
}

/// **An edit git was told to leave alone comes back** — `skip-worktree`
/// and `assume-unchanged` files, which no status shows, their flags too.
#[test]
fn an_edit_git_leaves_alone_comes_back() {
    let t = repo();
    t.git(&["update-index", "--skip-worktree", "a.txt"]);
    t.git(&["update-index", "--assume-unchanged", "b.txt"]);
    t.write("a.txt", b"my local config\n");
    t.write("b.txt", b"my other local config\n");
    let mut kept = keep(&t, &["a.txt", "b.txt"]);
    t.git(&["checkout", "-q", "-f", "job"]);
    t.write("a.txt", b"the conversation's\n");
    t.write("b.txt", b"the conversation's\n");
    kept.restore().unwrap();
    assert_eq!(t.read("a.txt"), b"my local config\n");
    assert_eq!(t.read("b.txt"), b"my other local config\n");
    let flags = t.git(&["ls-files", "-v", "a.txt", "b.txt"]);
    assert_eq!(flags, "S a.txt\nh b.txt\n");
}

/// **An ignored file at a path the run writes is moved aside and comes back**
/// — never overwritten, never removed — while a run's ignored file where
/// nothing stood is removed, and an ignored file elsewhere is never touched.
#[test]
fn an_ignored_file_the_run_writes_over_comes_back() {
    let t = repo();
    t.write(".env.local", b"SECRET=mine\n");
    t.write("other.local", b"untouched\n");
    t.write("target/keep.bin", b"mine\n");
    let mut kept = keep(&t, &[".env.local", "fresh.local", "a.txt"]);
    assert!(
        !t.path.join(".env.local").exists(),
        "moved aside for the run"
    );
    run_on(&t);
    t.write(".env.local", b"SECRET=the conversation's\n");
    t.write("fresh.local", b"the conversation's\n");
    kept.restore().unwrap();
    assert_eq!(t.read(".env.local"), b"SECRET=mine\n");
    assert!(!t.path.join("fresh.local").exists());
    assert_eq!(t.read("other.local"), b"untouched\n");
    assert_eq!(t.read("target/keep.bin"), b"mine\n");
    assert_released(&t);
}

/// **An ignored file the run's branch tracks survives the switch there and
/// back** — a forced checkout would overwrite it, and the checkout back
/// delete it — and so does an ignored file where the branch needs a folder.
#[test]
fn an_ignored_file_the_runs_branch_tracks_survives() {
    let t = repo();
    t.git(&["checkout", "-q", "job"]);
    t.write("dist/app.local", b"tracked on job\n");
    t.write("lib.local/inner.txt", b"a folder on job\n");
    t.git(&["add", "-f", "dist/app.local", "lib.local/inner.txt"]);
    t.git(&["commit", "-q", "-m", "job tracks it"]);
    let job = t.oid("job");
    t.git(&["checkout", "-q", "main"]);
    t.write("dist/app.local", b"my local build\n");
    t.write("lib.local", b"my ignored file\n");

    let writes: Vec<String> = Vec::new();
    let mut kept = preserve(Arc::new(t.repo()), "a test ran", &writes, Some(&job)).unwrap();
    t.git(&["checkout", "-q", "-f", "job"]);
    assert_eq!(t.read("dist/app.local"), b"tracked on job\n");
    kept.restore().unwrap();
    assert_eq!(t.read("dist/app.local"), b"my local build\n");
    assert_eq!(t.read("lib.local"), b"my ignored file\n");
    assert_released(&t);
}

/// **A file that could not be moved aside is left exactly where it was** —
/// the preservation fails whole, and nothing that was not moved is removed.
#[cfg(windows)]
#[test]
fn a_file_that_cannot_be_moved_aside_is_left() {
    let t = repo();
    t.write("first.local", b"first\n");
    t.write("second.local", b"second, held open\n");
    let held = hold(&t.path.join("second.local"));
    let writes = vec!["first.local".to_string(), "second.local".to_string()];
    let refused = preserve(Arc::new(t.repo()), "a test ran", &writes, None);
    assert!(refused.is_err());
    drop(held);
    assert_eq!(t.read("first.local"), b"first\n");
    assert_eq!(t.read("second.local"), b"second, held open\n");
    assert_released(&t);
}

/// **A restore that keeps failing leaves everything journalled and says
/// where**; once the cause is gone, the next preservation finishes it.
#[cfg(windows)]
#[test]
fn a_failed_restore_is_kept_and_finished_later() {
    let t = repo();
    t.write("mine.txt", b"my new file\n");
    let mut kept = keep(&t, &[]);
    run_on(&t);
    let held = hold(&t.path.join("stray.txt"));
    match kept.restore() {
        Err(CheckoutError::NotPutBack { journal, .. }) => {
            assert!(journal.contains(PRESERVED_DIR), "{journal}")
        }
        other => panic!("{other:?}"),
    }
    assert_ne!(
        t.git(&["for-each-ref", "refs/zend/"]),
        "",
        "the refs are kept"
    );
    drop(held);
    let mut next = keep(&t, &[]);
    next.restore().unwrap();
    assert_eq!(t.read("mine.txt"), b"my new file\n");
    assert_eq!(t.git(&["symbolic-ref", "--short", "HEAD"]).trim(), "main");
    assert_released(&t);
}

/// **A file moved aside comes back to its own path, never through a link the
/// run planted on the way**, and the link's target is untouched.
#[test]
fn a_file_moved_aside_never_goes_back_through_a_link() {
    let t = repo();
    let outside = tempfile::tempdir().unwrap();
    t.write("conf.local/.env", b"SECRET=mine\n");
    let mut kept = keep(&t, &["conf.local/.env"]);
    assert!(
        !t.path.join("conf.local").exists(),
        "the ignored folder is moved aside whole"
    );
    run_on(&t);
    link_dir(outside.path(), &t.path.join("conf.local"));
    kept.restore().unwrap();
    assert_eq!(t.read("conf.local/.env"), b"SECRET=mine\n");
    assert!(
        std::fs::read_dir(outside.path()).unwrap().next().is_none(),
        "nothing landed outside the repository"
    );
}

/// **Ignore rules the run changed are put back before anything untracked is
/// removed**, so a file ignored only by them is never taken for the run's.
#[test]
fn ignore_rules_the_run_changed_protect_nothing_of_the_runs() {
    let t = repo();
    t.write(".git/info/exclude", b"notes.txt\n");
    t.write("notes.txt", b"my notes, ignored locally\n");
    let mut kept = keep(&t, &[]);
    run_on(&t);
    t.write(".git/info/exclude", b"");
    t.write(".gitignore", b"");
    kept.restore().unwrap();
    assert_eq!(t.read("notes.txt"), b"my notes, ignored locally\n");
    assert_eq!(t.read(".git/info/exclude"), b"notes.txt\n");
}

/// **A checkout part way through a merge is refused, untouched** — its
/// conflicted files, and the merge itself, exactly as they were.
#[test]
fn a_checkout_part_way_through_a_merge_is_refused_untouched() {
    let t = repo();
    t.git(&["checkout", "-q", "-b", "side"]);
    t.write("a.txt", b"side\n");
    t.commit_all("side");
    t.git(&["checkout", "-q", "main"]);
    t.write("a.txt", b"main\n");
    t.commit_all("main");
    let merged = std::process::Command::new("git")
        .arg("-C")
        .arg(&t.path)
        .args(["merge", "-q", "side"])
        .output()
        .unwrap();
    assert!(!merged.status.success(), "the merge conflicts");
    let before = status(&t);
    let refused = preserve(Arc::new(t.repo()), "a test ran", &[], None);
    assert!(refused.unwrap_err().to_string().contains("a merge"));
    assert_eq!(status(&t), before);
    assert!(t.path.join(".git/MERGE_HEAD").exists());
    assert_released(&t);
}

/// **A branch with no commit yet is refused**, the checkout untouched.
#[test]
fn an_unborn_branch_is_refused_untouched() {
    let t = TestRepo::init();
    t.write("draft.txt", b"draft\n");
    assert!(preserve(Arc::new(t.repo()), "a test ran", &[], None).is_err());
    assert_eq!(t.read("draft.txt"), b"draft\n");
}

/// **A file only an uncommitted `.gitignore` edit ignores is never taken for
/// the run's** — checking out `HEAD` puts the committed rules back, under
/// which it reads as untracked — and neither is one a `.gitignore` of its
/// own folder, never committed, ignores.
#[test]
fn a_file_ignored_by_uncommitted_rules_is_never_removed() {
    let t = repo();
    t.write(".gitignore", b"target/\n*.local\nlocal.env\n");
    t.write("local.env", b"SECRET=mine\n");
    t.write("tools/.gitignore", b"secret\n");
    t.write("tools/secret", b"my tool's key\n");
    let before = status(&t);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    // As putting the conversation on the checkout clears untracked files.
    std::fs::remove_file(t.path.join("tools/.gitignore")).unwrap();
    kept.restore().unwrap();
    assert_eq!(t.read("local.env"), b"SECRET=mine\n");
    assert_eq!(t.read("tools/secret"), b"my tool's key\n");
    assert_eq!(status(&t), before);
    assert_released(&t);
}

/// **A file the run's branch does not ignore, but the checkout's own rules
/// did, is neither removed nor read back as the conversation's** — through
/// the whole run: put on, captured, put back.
#[test]
fn a_file_the_runs_branch_does_not_ignore_is_left_alone_throughout() {
    use crate::checkout::{capture, materialize};
    use crate::{DiskWriteGrant, FileChanges};

    let t = repo();
    t.git(&["checkout", "-q", "job"]);
    t.write(".gitignore", b"");
    let job = t.commit_all("job ignores nothing");
    t.git(&["checkout", "-q", "main"]);
    t.write("mine.local", b"my local file\n");
    t.write("target/cache.bin", b"my build cache\n");

    let writes: Vec<String> = Vec::new();
    let repo = Arc::new(t.repo());
    let mut kept = preserve(Arc::clone(&repo), "a test ran", &writes, Some(&job)).unwrap();
    let branch = BranchName::parse("job").unwrap();
    let (mut ledger, done) = materialize(
        &repo,
        &DiskWriteGrant::issue(),
        &branch,
        &FileChanges::new(),
    )
    .unwrap();
    assert!(!done
        .removed
        .iter()
        .any(|p| p.contains("local") || p.contains("target")));
    assert_eq!(t.read("mine.local"), b"my local file\n");
    let captured = capture(&repo, &branch, &mut ledger).unwrap();
    assert!(captured.is_empty(), "{captured:?}");
    kept.restore().unwrap();
    assert_eq!(t.read("mine.local"), b"my local file\n");
    assert_eq!(t.read("target/cache.bin"), b"my build cache\n");
    assert_released(&t);
}

/// **An ignored folder where the run's branch has a file is moved aside and
/// comes back whole.**
#[test]
fn an_ignored_folder_where_the_branch_has_a_file_comes_back() {
    let t = repo();
    t.git(&["checkout", "-q", "job"]);
    t.write("build.local", b"a file on job\n");
    t.git(&["add", "-f", "build.local"]);
    let job = t.commit_all("job tracks a file there");
    t.git(&["checkout", "-q", "main"]);
    t.write("build.local/out/app", b"my build\n");

    let writes: Vec<String> = Vec::new();
    let mut kept = preserve(Arc::new(t.repo()), "a test ran", &writes, Some(&job)).unwrap();
    t.git(&["checkout", "-q", "-f", "job"]);
    assert_eq!(t.read("build.local"), b"a file on job\n");
    kept.restore().unwrap();
    assert_eq!(t.read("build.local/out/app"), b"my build\n");
    assert_released(&t);
}

/// **A checkout someone has worked in since a run crashed is refused, left
/// as they have it**, with what was set aside kept and named.
#[test]
fn a_checkout_worked_in_since_a_crash_is_refused_untouched() {
    let t = repo();
    t.write("a.txt", b"before the run\n");
    keep(&t, &[]).crash();
    run_on(&t);
    t.git(&["checkout", "-q", "-f", "main"]);
    t.write("a.txt", b"my later work\n");
    t.commit_all("my later work");
    t.write("b.txt", b"uncommitted, later\n");

    let refused = preserve(Arc::new(t.repo()), "a test ran", &[], None).unwrap_err();
    match refused {
        CheckoutError::NotPutBack { detail, .. } => {
            assert!(detail.contains("worked in it"), "{detail}")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(t.read("a.txt"), b"my later work\n");
    assert_eq!(t.read("b.txt"), b"uncommitted, later\n");
    assert_ne!(t.git(&["for-each-ref", "refs/zend/preserved/"]), "");
}

/// **A recovery keeps what the checkout held before putting it back** —
/// under `refs/zend/recovered/`, where nothing lets go of it.
#[test]
fn a_recovery_keeps_what_stood_first() {
    let t = repo();
    let before = status(&t);
    keep(&t, &[]).crash();
    run_on(&t);
    recover(&Arc::new(t.repo())).unwrap();
    assert_eq!(status(&t), before);
    let kept = t.git(&[
        "for-each-ref",
        "--format=%(refname)",
        "refs/zend/recovered/",
    ]);
    let files = kept.lines().find(|r| r.ends_with("/files")).unwrap();
    assert_eq!(
        t.git(&["show", &format!("{files}:stray.txt")]),
        "left by the job\n"
    );
    assert_released(&t);
}

/// **A move journalled but never made loses nothing**: the file is still at
/// home, and a recovery leaves it there.
#[test]
fn a_move_journalled_but_never_made_leaves_the_file() {
    let t = repo();
    t.write("x.local", b"mine\n");
    let place = Place::new(&t.path.join(".git"));
    let journal = Journal {
        why: "a test ran".into(),
        phase: Phase::Capturing,
        head: SavedHead {
            branch: Some("main".into()),
            commit: t.oid("main").as_str().to_string(),
        },
        target: None,
        set_aside: None,
        index: None,
        files: None,
        deleted: Vec::new(),
        flags: Vec::new(),
        intent_to_add: Vec::new(),
        perms: Default::default(),
        link_dirs: Vec::new(),
        unchanged: Vec::new(),
        exclude: None,
        vacant: vec!["fresh.local".into()],
        moved: vec!["x.local".into()],
    };
    place.save(&journal).unwrap();
    t.write("fresh.local", b"made after the crash\n");
    recover(&Arc::new(t.repo())).unwrap();
    assert_eq!(t.read("x.local"), b"mine\n");
    assert_eq!(t.read("fresh.local"), b"made after the crash\n");
    assert_released(&t);
}

/// **One preservation of a checkout at a time**: a second is refused while
/// the first holds it, and runs once it is let go.
#[test]
fn a_second_preservation_is_refused_while_one_holds() {
    let t = repo();
    let mut first = keep(&t, &[]);
    let second = preserve(Arc::new(t.repo()), "a test ran", &[], None).unwrap_err();
    assert!(second.to_string().contains("another process"), "{second}");
    first.restore().unwrap();
    drop(first);
    keep(&t, &[]).restore().unwrap();
}

/// **A file swapped for a folder, and a folder for a file, come back** —
/// what held no file cleared before the captured files are written.
#[test]
fn files_and_folders_swapped_come_back() {
    let t = repo();
    t.write("dir/x.txt", b"x\n");
    t.commit_all("a folder");
    std::fs::remove_file(t.path.join("b.txt")).unwrap();
    t.write("b.txt/inner.txt", b"a folder where a file was\n");
    std::fs::remove_dir_all(t.path.join("dir")).unwrap();
    t.write("dir", b"a file where a folder was\n");
    let before = status(&t);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(t.read("b.txt/inner.txt"), b"a folder where a file was\n");
    assert_eq!(t.read("dir"), b"a file where a folder was\n");
    assert_eq!(status(&t), before);
    assert_released(&t);
}

/// **Files `git rm`'d and renamed with `git mv` come back as they were** —
/// no old copy left behind as an untracked file.
#[test]
fn removed_and_renamed_files_come_back_so() {
    let t = repo();
    t.git(&["rm", "-q", "b.txt"]);
    t.git(&["mv", "a.txt", "c.txt"]);
    let before = status(&t);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(status(&t), before);
    assert!(!t.path.join("a.txt").exists());
    assert!(!t.path.join("b.txt").exists());
    assert_released(&t);
}

/// **A branch the run took commits off is put back; one that only moved on
/// keeps what it gained.**
#[test]
fn a_branch_the_run_rewound_is_put_back() {
    let t = repo();
    t.git(&["checkout", "-q", "-b", "mine"]);
    let first = t.oid("mine");
    t.write("a.txt", b"second\n");
    let second = t.commit_all("second");
    let mut kept = keep(&t, &[]);
    run_on(&t);
    t.git(&["branch", "-f", "mine", first.as_str()]);
    kept.restore().unwrap();
    assert_eq!(t.oid("mine"), second);
    assert_eq!(t.read("a.txt"), b"second\n");

    // A commit landing on the branch while a run has the checkout — as a
    // conversation's published commit moves it — is kept.
    let mut kept = keep(&t, &[]);
    t.write("later.txt", b"later\n");
    t.git(&["add", "later.txt"]);
    t.git(&["commit", "-q", "-m", "later"]);
    let gained = t.oid("mine");
    t.git(&["checkout", "-q", "-f", "job"]);
    kept.restore().unwrap();
    assert_eq!(t.oid("mine"), gained, "moved on, so kept");
    assert_eq!(status(&t), "## mine\n", "and nothing it gained staged away");
}

/// **The checkout's own changes are carried onto a branch that moved on**
/// while it was set aside — a staged change staged, an untracked file
/// untracked — and nothing the branch gained comes back reverted, not even
/// a file captured only to keep its bytes.
#[test]
fn own_changes_are_carried_onto_a_branch_that_moved_on() {
    let t = repo();
    t.git(&["checkout", "-q", "-b", "mine"]);
    t.write("a.txt", b"staged\n");
    t.git(&["add", "a.txt"]);
    t.write("new.txt", b"not added yet\n");
    let mut kept = keep(&t, &["b.txt"]);

    // While it is set aside the branch moves on: `b.txt` changed, `c.txt`
    // added.
    t.git(&["checkout", "-q", "-f", "mine"]);
    t.write("b.txt", b"b, moved on\n");
    t.write("c.txt", b"c\n");
    t.git(&["add", "b.txt", "c.txt"]);
    t.git(&["commit", "-q", "-m", "moved on"]);
    let moved = t.oid("mine");
    t.git(&["checkout", "-q", "-f", "job"]);
    kept.restore().unwrap();

    assert_eq!(t.oid("mine"), moved);
    assert_eq!(status(&t), "## mine\nM  a.txt\n?? new.txt\n");
    assert_eq!(t.read("a.txt"), b"staged\n");
    assert_eq!(t.read("b.txt"), b"b, moved on\n");
    assert_eq!(t.read("c.txt"), b"c\n");
    assert_eq!(t.read("new.txt"), b"not added yet\n");
    assert_released(&t);
}

/// `job` with a commit of its own on top of `main`, as unpushed work sits on
/// a branch; returns `(main, job)`.
fn job_ahead(t: &TestRepo) -> (Oid, Oid) {
    let base = t.oid("main");
    t.git(&["checkout", "-q", "job"]);
    t.write("a.txt", b"unpushed\n");
    let ahead = t.commit_all("unpushed");
    t.git(&["checkout", "-q", "main"]);
    (base, ahead)
}

fn held_for_the_branch(t: &TestRepo) -> String {
    t.git(&[
        "for-each-ref",
        "--format=%(objectname)",
        "refs/zend/preserved/*/branch",
    ])
    .trim()
    .to_string()
}

/// **A branch set aside for the run comes back at its own commit**, held by
/// a ref meanwhile so nothing of it can be pruned; one that only moved on
/// from there since keeps what it gained.
#[test]
fn a_branch_set_aside_comes_back_at_its_own_commit() {
    let t = repo();
    let (base, ahead) = job_ahead(&t);
    let job = BranchName::parse("job").unwrap();

    let mut kept = keep(&t, &[]);
    kept.set_branch_aside(&job, &ahead, &base).unwrap();
    assert_eq!(t.oid("job"), base, "moved for the run");
    assert_eq!(held_for_the_branch(&t), ahead.as_str());
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(t.oid("job"), ahead, "back at its own commit");
    assert_eq!(status(&t), "## main\n");
    assert_released(&t);

    let mut kept = keep(&t, &[]);
    kept.set_branch_aside(&job, &ahead, &base).unwrap();
    t.git(&["checkout", "-q", "-f", "job"]);
    t.git(&["reset", "-q", "--hard", ahead.as_str()]);
    t.write("later.txt", b"later\n");
    let gained = t.commit_all("later");
    kept.restore().unwrap();
    assert_eq!(t.oid("job"), gained, "moved on from it, so kept");
    assert_released(&t);
}

/// **A branch that has left the commit it was looked at is not set aside**:
/// refused, nothing moved, and nothing for the restore to put back.
#[test]
fn a_branch_that_moved_is_not_set_aside() {
    let t = repo();
    let (base, ahead) = job_ahead(&t);
    let job = BranchName::parse("job").unwrap();
    let mut kept = keep(&t, &[]);
    assert!(kept.set_branch_aside(&job, &base, &ahead).is_err());
    assert_eq!(t.oid("job"), ahead);
    assert_eq!(held_for_the_branch(&t), "");
    kept.restore().unwrap();
    assert_eq!(t.oid("job"), ahead);
    assert_released(&t);
}

/// **A branch set aside by a run that crashed comes back with the rest of
/// the checkout** — the owner's `HEAD` on it, their edit over it.
#[test]
fn a_branch_set_aside_comes_back_after_a_crash() {
    let t = repo();
    let (base, ahead) = job_ahead(&t);
    t.git(&["checkout", "-q", "job"]);
    t.write("b.txt", b"my edit\n");
    let job = BranchName::parse("job").unwrap();
    // As a sandbox job does: the run's target is the commit the branch is
    // set aside onto, which is where a crash leaves the owner's `HEAD`.
    let mut kept = preserve(Arc::new(t.repo()), "a test ran", &[], Some(&base)).unwrap();
    kept.set_branch_aside(&job, &ahead, &base).unwrap();
    kept.crash();
    t.git(&["checkout", "-q", "-f", "job"]);

    recover(&Arc::new(t.repo())).unwrap();
    assert_eq!(t.oid("job"), ahead);
    assert_eq!(status(&t), "## job\n M b.txt\n");
    assert_eq!(t.read("a.txt"), b"unpushed\n");
    assert_eq!(t.read("b.txt"), b"my edit\n");
    assert_released(&t);
}

/// **A file added with `git add -N` comes back added so.**
#[test]
fn an_intent_to_add_comes_back() {
    let t = repo();
    t.write("later.txt", b"to be added\n");
    t.git(&["add", "-N", "later.txt"]);
    let before = status(&t);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    assert_eq!(status(&t), before);
    assert_eq!(t.read("later.txt"), b"to be added\n");
}

/// **A tracked file the run rewrites comes back with the line endings it
/// had**, even where they differ from what a checkout writes.
#[test]
fn a_rewritten_files_line_endings_come_back() {
    let t = repo();
    t.git(&["checkout", "-q", "job"]);
    t.write("a.txt", b"job\n");
    let job = t.commit_all("job changes a");
    t.git(&["checkout", "-q", "main"]);
    t.git(&["config", "core.autocrlf", "true"]);
    let writes: Vec<String> = Vec::new();
    let mut kept = preserve(Arc::new(t.repo()), "a test ran", &writes, Some(&job)).unwrap();
    t.git(&["checkout", "-q", "-f", "job"]);
    kept.restore().unwrap();
    assert_eq!(t.read("a.txt"), b"a\n", "LF, as it stood — not CRLF");
}

/// **A file's permissions come back exactly.**
#[cfg(unix)]
#[test]
fn permissions_come_back() {
    use std::os::unix::fs::PermissionsExt;
    let t = repo();
    t.write("secret.key", b"key\n");
    let path = t.path.join("secret.key");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode, 0o600);
}

/// **A file the run named as no conversation could is cleared all the
/// same**, and the checkout comes back.
#[cfg(unix)]
#[test]
fn a_file_named_as_no_conversation_could_is_cleared() {
    let t = repo();
    let before = status(&t);
    let mut kept = keep(&t, &[]);
    run_on(&t);
    t.write("odd:name", b"x\n");
    t.write("trailing.", b"x\n");
    kept.restore().unwrap();
    assert!(!t.path.join("odd:name").exists());
    assert!(!t.path.join("trailing.").exists());
    assert_eq!(status(&t), before);
    assert_released(&t);
}

/// **A link to a folder comes back as a link to that folder** — its
/// target's files untouched.
#[test]
fn a_link_to_a_folder_comes_back() {
    let t = repo();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("there.txt"), b"outside\n").unwrap();
    link_dir(outside.path(), &t.path.join("linked"));
    let mut kept = keep(&t, &[]);
    run_on(&t);
    kept.restore().unwrap();
    let linked = t.path.join("linked");
    assert!(std::fs::symlink_metadata(&linked)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::read(linked.join("there.txt")).unwrap(),
        b"outside\n"
    );
    assert_eq!(
        std::fs::read(outside.path().join("there.txt")).unwrap(),
        b"outside\n"
    );
}

/// `path` held open with no sharing, as a process still running holds a
/// file: it can be neither moved nor deleted until the handle goes.
#[cfg(windows)]
fn hold(path: &Path) -> File {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(path)
        .unwrap()
}

/// Link folder `link` to `target`: a symlink on Unix, a junction on Windows.
fn link_dir(target: &Path, link: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).unwrap();
    #[cfg(windows)]
    {
        let out = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .unwrap();
        assert!(out.status.success(), "mklink /J failed");
    }
}
