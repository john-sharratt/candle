//! Test repositories, created with `git init` inside this crate's `scratch/`
//! folder — nested in the candle checkout, ignored by it — and deleted when
//! the test ends.
//!
//! Setup runs git directly (not through the layer under test) with a fixed
//! identity and fixed dates, so every setup commit id is reproducible.

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

use crate::types::Oid;
use crate::Repo;

/// The fixed date every setup commit carries.
pub(crate) const SETUP_DATE: &str = "1700000000 +0000";

/// The folder test repositories are created in.
pub(crate) fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("scratch");
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

impl TestRepo {
    fn create(bare: bool) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(if bare { "origin-" } else { "repo-" })
            .tempdir_in(scratch())
            .expect("temp repo dir");
        let path = dir.path().to_path_buf();
        // `init -b` needs 2.28; the suite also runs against 2.24.
        let mut init = vec!["init", "-q"];
        if bare {
            init.push("--bare");
        }
        git_in(&path, &init);
        git_in(&path, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        for (key, value) in [
            ("core.autocrlf", "false"),
            ("core.filemode", "true"),
            ("commit.gpgSign", "false"),
            ("user.name", "Setup"),
            ("user.email", "setup@example.com"),
        ] {
            git_in(&path, &["config", key, value]);
        }
        if !bare {
            // The repository must be its own top level, never the candle
            // checkout it is nested in.
            let top = git_in(&path, &["rev-parse", "--show-toplevel"]);
            assert_eq!(
                PathBuf::from(top.trim()).canonicalize().unwrap(),
                path.canonicalize().unwrap(),
                "test repository resolved to an enclosing repository"
            );
        }
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
