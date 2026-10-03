//! Test repositories, inside this crate's `scratch/` folder — nested in the
//! candle checkout, ignored by it — and deleted when the test ends. Each is
//! initialised in this process, so a test leaves nothing behind.
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

/// An empty repository at `path`, on `main`, configured for reproducible
/// setup: no sample hooks, no background maintenance after a commit. Made in
/// this process — no template folder is left behind and no `git` is started.
fn init_repository(path: &Path, bare: bool) {
    let mut options = git2::RepositoryInitOptions::new();
    options
        .bare(bare)
        .external_template(false)
        .initial_head("main");
    let repository = git2::Repository::init_opts(path, &options).expect("init repository");
    let mut config = repository.config().expect("repository config");
    for (key, value) in [
        ("core.autocrlf", "false"),
        ("core.filemode", "true"),
        ("commit.gpgSign", "false"),
        ("user.name", "Setup"),
        ("user.email", "setup@example.com"),
        ("maintenance.auto", "false"),
        ("gc.auto", "0"),
    ] {
        config.set_str(key, value).expect("repository setting");
    }
    if !bare {
        // The repository must be its own top level, never the candle
        // checkout it is nested in.
        assert_eq!(
            repository.workdir().unwrap().canonicalize().unwrap(),
            path.canonicalize().unwrap(),
            "test repository resolved to an enclosing repository"
        );
    }
}

impl TestRepo {
    fn create(bare: bool) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(if bare { "origin-" } else { "repo-" })
            .tempdir_in(scratch())
            .expect("temp repo dir");
        let path = dir.path().to_path_buf();
        init_repository(&path, bare);
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
    ///
    /// In this process — `git add -A` and `git commit` are two `git` processes,
    /// and a fixture makes hundreds of commits — unless a merge, cherry-pick or
    /// revert is under way, which `git commit` finishes and libgit2's `commit`
    /// does not.
    pub(crate) fn commit_all(&self, message: &str) -> Oid {
        if let Some(commit) = self.commit_all_in_process(message) {
            return commit;
        }
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "--allow-empty", "-m", message]);
        self.oid("HEAD")
    }

    /// [`Self::commit_all`] through libgit2, as `git add -A && git commit
    /// --allow-empty -m` would make it: the same tree, the setup identity at the
    /// setup date, and the message with the newline `git commit` ends it with —
    /// so the commit's id is the one the `git` programs give.
    fn commit_all_in_process(&self, message: &str) -> Option<Oid> {
        let git_dir = self.path.join(".git");
        let mid_operation = ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"]
            .iter()
            .any(|marker| git_dir.join(marker).exists());
        if mid_operation {
            return None;
        }
        let repo = git2::Repository::open(&self.path).ok()?;
        let mut index = repo.index().ok()?;
        index
            .add_all(["*"], git2::IndexAddOption::DEFAULT, None)
            .ok()?;
        index.update_all(["*"], None).ok()?;
        index.write().ok()?;
        let tree = repo.find_tree(index.write_tree().ok()?).ok()?;
        let setup = git2::Signature::new(
            "Setup",
            "setup@example.com",
            &git2::Time::new(1_700_000_000, 0),
        )
        .ok()?;
        let parent = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
        let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
        let id = repo
            .commit(
                Some("HEAD"),
                &setup,
                &setup,
                &format!("{message}\n"),
                &tree,
                &parents,
            )
            .ok()?;
        Oid::parse(&id.to_string()).ok()
    }

    pub(crate) fn oid(&self, rev: &str) -> Oid {
        if rev == "HEAD" {
            if let Some(oid) = self.head_from_files() {
                return oid;
            }
        }
        Oid::parse(self.git(&["rev-parse", rev]).trim()).unwrap()
    }

    /// The commit `HEAD` names, read from the files: the branch ref a
    /// fixture's commit has just written, or a detached id. A fixture asks
    /// this after every commit, and a process per answer is most of what a
    /// commit costs. `None` when the answer is not in a file — a packed ref, an
    /// unborn branch — and git is asked instead.
    fn head_from_files(&self) -> Option<Oid> {
        let git_dir = self.path.join(".git");
        let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
        let head = head.trim();
        let target = match head.strip_prefix("ref: ") {
            Some(name) => std::fs::read_to_string(git_dir.join(name)).ok()?,
            None => head.to_string(),
        };
        Oid::parse(target.trim()).ok()
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
