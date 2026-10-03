//! Test repositories, inside `<target>/tmp/zend-vfs-scratch` — the build's own
//! scratch space, nested in the candle checkout and ignored by it — and deleted
//! when the test ends. Each is a copy of an empty repository built with
//! `git init` once per test process.
//!
//! Setup runs git directly (not through the layer under test) with a fixed
//! identity and fixed dates, so every setup commit id is reproducible.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use tempfile::TempDir;

use crate::types::Oid;
use crate::Repo;

/// The fixed date every setup commit carries.
pub(crate) const SETUP_DATE: &str = "1700000000 +0000";

/// The folder test repositories are created in: `<target>/tmp`, the directory
/// cargo hands integration tests as `CARGO_TARGET_TMPDIR`, reached from the test
/// binary's own path because cargo does not set that variable for unit tests.
///
/// Never the crate directory. A test's leftovers there are files in the source
/// tree, and this module's per-process templates outlive the process that made
/// them by design — swept only once they are an hour old.
pub(crate) fn scratch() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary's path");
    // `<target>/<profile>/deps/<binary>`.
    let target = exe
        .ancestors()
        .nth(3)
        .expect("a test binary sits in <target>/<profile>/deps");
    let dir = target.join("tmp").join("zend-vfs-scratch");
    std::fs::create_dir_all(&dir).expect("scratch folder");
    dir
}

/// Run git in `dir` for test setup, returning stdout; panics on failure.
pub(crate) fn git_in(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.hooksPath=", "-c", "init.defaultBranch=main"])
        .args(args)
        .env("LC_ALL", "C")
        .env("GIT_AUTHOR_NAME", "Setup")
        .env("GIT_AUTHOR_EMAIL", "setup@example.com")
        .env("GIT_AUTHOR_DATE", SETUP_DATE)
        .env("GIT_COMMITTER_NAME", "Setup")
        .env("GIT_COMMITTER_EMAIL", "setup@example.com")
        .env("GIT_COMMITTER_DATE", SETUP_DATE)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8")
}

pub(crate) struct TestRepo {
    _dir: TempDir,
    pub path: PathBuf,
}

/// How long a template folder from an earlier test run is kept before a new
/// run sweeps it away: long enough that a run still going is never robbed.
const STALE_TEMPLATE: Duration = Duration::from_secs(60 * 60);

/// An empty repository of each kind — working tree or bare — built once per
/// test process and copied for every [`TestRepo`]. Building one is eight git
/// processes; copying it is a handful of small files.
fn template(bare: bool) -> &'static Path {
    static WORK: OnceLock<PathBuf> = OnceLock::new();
    static BARE: OnceLock<PathBuf> = OnceLock::new();
    let cell = if bare { &BARE } else { &WORK };
    cell.get_or_init(|| {
        sweep_stale_templates();
        let kind = if bare { "bare" } else { "work" };
        let path = scratch().join(format!("template-{}-{kind}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("template dir");
        build(&path, bare);
        path
    })
}

/// Remove template folders earlier test runs left behind.
fn sweep_stale_templates() {
    let Ok(entries) = std::fs::read_dir(scratch()) else {
        return;
    };
    for entry in entries.flatten() {
        let stale = entry.file_name().to_string_lossy().starts_with("template-")
            && entry
                .metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|at| at.elapsed().is_ok_and(|age| age > STALE_TEMPLATE));
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// An empty repository at `path`, on `main`, configured for reproducible
/// setup: no sample hooks, no background maintenance after a commit.
fn build(path: &Path, bare: bool) {
    // `init -b` needs 2.28; the suite also runs against 2.24.
    let mut init = vec!["init", "-q", "--template="];
    if bare {
        init.push("--bare");
    }
    git_in(path, &init);
    git_in(path, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    for (key, value) in [
        ("core.autocrlf", "false"),
        ("core.filemode", "true"),
        ("commit.gpgSign", "false"),
        ("user.name", "Setup"),
        ("user.email", "setup@example.com"),
        ("maintenance.auto", "false"),
        ("gc.auto", "0"),
    ] {
        git_in(path, &["config", key, value]);
    }
    if !bare {
        // The repository must be its own top level, never the candle
        // checkout it is nested in.
        let top = git_in(path, &["rev-parse", "--show-toplevel"]);
        assert_eq!(
            PathBuf::from(top.trim()).canonicalize().unwrap(),
            path.canonicalize().unwrap(),
            "test repository resolved to an enclosing repository"
        );
    }
}

/// Copy the folder `from` to `to`, everything in it included.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

impl TestRepo {
    fn create(bare: bool) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(if bare { "origin-" } else { "repo-" })
            .tempdir_in(scratch())
            .expect("temp repo dir");
        let path = dir.path().to_path_buf();
        copy_tree(template(bare), &path);
        Self { _dir: dir, path }
    }

    /// A fresh working tree on branch `main`, with no commits.
    pub(crate) fn init() -> Self {
        Self::create(false)
    }

    /// A fresh bare repository, to act as origin.
    pub(crate) fn bare() -> Self {
        Self::create(true)
    }

    pub(crate) fn git(&self, args: &[&str]) -> String {
        git_in(&self.path, args)
    }

    pub(crate) fn write(&self, rel: &str, bytes: &[u8]) {
        let p = self.path.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    pub(crate) fn read(&self, rel: &str) -> Vec<u8> {
        std::fs::read(self.path.join(rel)).unwrap()
    }

    /// Stage everything and commit it, returning the commit.
    pub(crate) fn commit_all(&self, message: &str) -> Oid {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "--allow-empty", "-m", message]);
        self.oid("HEAD")
    }

    pub(crate) fn oid(&self, rev: &str) -> Oid {
        Oid::parse(self.git(&["rev-parse", rev]).trim()).unwrap()
    }

    pub(crate) fn repo(&self) -> Repo {
        Repo::open(&self.path).expect("test repo opens")
    }

    /// This repository as a `file://` URL, so fetch and push use git's
    /// smart transport rather than the local shortcut.
    pub(crate) fn url(&self) -> String {
        let p = self.path.canonicalize().unwrap();
        let s = p.to_string_lossy().replace('\\', "/");
        let s = s.trim_start_matches("//?/");
        if s.starts_with('/') {
            format!("file://{s}")
        } else {
            format!("file:///{s}")
        }
    }
}
