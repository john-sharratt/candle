//! The version against scratch git repositories.

use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use super::compute::{compute, rerun_paths};
use super::this_build;

fn git(root: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .expect("git runs")
        .status
        .success();
    assert!(ok, "git {args:?}");
}

fn commit(root: &Path, file: &str, text: &str) {
    let path = root.join(file);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(path, text).unwrap();
    git(root, &["add", file]);
    git(root, &["commit", "-q", "-m", file]);
}

/// A repository with one workspace crate, `alpha`, and a doc beside it, at two
/// commits on `origin/main`.
fn repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    git(dir.path(), &["init", "-q"]);
    commit(dir.path(), "alpha/Cargo.toml", "[package]");
    commit(dir.path(), "alpha/src/lib.rs", "one");
    commit(dir.path(), "docs/notes.md", "notes");
    git(
        dir.path(),
        &["update-ref", "refs/remotes/origin/main", "HEAD"],
    );
    dir
}

/// At the merge base `local` is empty; a local commit, an edit and an
/// untracked file in a crate each change it, and none of them moves `count`.
#[test]
fn count_is_the_merge_base_and_local_names_everything_beyond_it() {
    let r = repo();
    let at_base = compute(r.path());
    assert_eq!(at_base.count, 3);
    assert_eq!(at_base.hash.len(), 7);
    assert_eq!(at_base.local, "");

    commit(r.path(), "alpha/src/more.rs", "local work");
    let committed = compute(r.path());
    assert_eq!((committed.count, &committed.hash), (3, &at_base.hash));
    assert_eq!(committed.local.len(), 16);

    std::fs::write(r.path().join("alpha/src/lib.rs"), "edited").unwrap();
    let edited = compute(r.path());
    assert_eq!(edited.count, 3);
    assert_ne!(edited.local, committed.local);

    std::fs::write(r.path().join("alpha/src/new.rs"), "untracked").unwrap();
    let untracked = compute(r.path());
    assert_ne!(untracked.local, edited.local);

    // The same tree computes the same version.
    assert_eq!(compute(r.path()), untracked);
}

/// **A file the build is not made from does not make it local.** A doc edit,
/// a tool's lock file, a local commit touching only docs: none of them can
/// change the binary, so the build is still the clean one.
#[test]
fn changes_outside_the_build_inputs_leave_the_build_clean() {
    let r = repo();
    std::fs::write(r.path().join("docs/notes.md"), "edited").unwrap();
    std::fs::create_dir_all(r.path().join(".claude")).unwrap();
    std::fs::write(r.path().join(".claude/scheduled_tasks.lock"), "{}").unwrap();
    assert_eq!(compute(r.path()).local, "", "an edit and an untracked file");

    commit(r.path(), "docs/more.md", "a local commit");
    assert_eq!(compute(r.path()).local, "", "a local commit of docs only");
}

/// **The same sources are the same build, however they were reached.** An
/// edit left uncommitted and the same edit committed locally name the same
/// `local`; pushed as it was built, it is a clean build at the new count.
#[test]
fn the_same_sources_are_the_same_build_by_any_route() {
    let r = repo();
    std::fs::write(r.path().join("alpha/src/lib.rs"), "two").unwrap();
    let uncommitted = compute(r.path());
    assert_eq!(uncommitted.local.len(), 16);

    git(r.path(), &["commit", "-q", "-am", "the same edit"]);
    let committed = compute(r.path());
    assert_eq!(committed, uncommitted);

    git(
        r.path(),
        &["update-ref", "refs/remotes/origin/main", "HEAD"],
    );
    let pushed = compute(r.path());
    assert_eq!((pushed.count, pushed.local.as_str()), (4, ""));
}

/// A pushed commit fetched as `origin/main` moves `count`, and a build at it
/// is no longer local.
#[test]
fn a_fetched_commit_moves_the_count() {
    let r = repo();
    commit(r.path(), "alpha/src/pushed.rs", "pushed");
    assert!(!compute(r.path()).local.is_empty());
    git(
        r.path(),
        &["update-ref", "refs/remotes/origin/main", "HEAD"],
    );
    let v = compute(r.path());
    assert_eq!((v.count, v.local.as_str()), (4, ""));
}

#[test]
fn a_tree_with_no_origin_main_is_a_local_build_at_count_zero() {
    let dir = TempDir::new().unwrap();
    git(dir.path(), &["init", "-q"]);
    commit(dir.path(), "a.txt", "one");
    let v = compute(dir.path());
    assert_eq!((v.count, v.hash.as_str()), (0, "none"));
    assert_eq!(v.local.len(), 16);
}

/// The script reruns on git's refs and every crate's sources and UI, never on
/// `target/`.
#[test]
fn the_build_script_watches_sources_and_refs_not_target() {
    let r = repo();
    for dir in ["alpha", "target", "zend"] {
        std::fs::create_dir_all(r.path().join(dir).join("src")).unwrap();
        std::fs::write(r.path().join(dir).join("Cargo.toml"), "").unwrap();
    }
    std::fs::create_dir_all(r.path().join("zend").join("web")).unwrap();
    let paths = rerun_paths(r.path());
    let has = |p: &Path| paths.iter().any(|x| x == p);
    assert!(has(&r.path().join("alpha").join("src")));
    assert!(has(&r.path().join("zend").join("web")));
    assert!(has(&r.path().join(".git").join("HEAD")));
    assert!(paths
        .iter()
        .any(|p| p.ends_with(Path::new("refs/remotes/origin/main"))));
    assert!(!paths.iter().any(|p| p.starts_with(r.path().join("target"))));
}

#[test]
fn this_build_reads_what_the_script_stamped() {
    let b = this_build();
    assert!(!b.hash.is_empty());
}
