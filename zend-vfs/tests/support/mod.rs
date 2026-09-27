//! The repository, sandbox and helpers the sandbox test suites share.

#![allow(dead_code)] // Each test binary uses the helpers it needs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use tempfile::TempDir;
use zend_vfs::{
    BranchName, CommandPolicy, DiskWriteGrant, Oid, Repo, RunOutcome, Sandbox, SandboxCommand,
    VfsStore,
};

pub fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "core.hooksPath=",
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgSign=false",
        ])
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

pub fn put(root: &Path, rel: &str, bytes: &[u8]) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, bytes).unwrap();
}

pub fn disk(root: &Path, rel: &str) -> Option<Vec<u8>> {
    std::fs::read(root.join(rel)).ok()
}

pub const SHELL: &str = if cfg!(windows) { "cmd" } else { "sh" };
pub const README: &[u8] = b"# app\n\nthe app.\n";
pub const LIB: &[u8] = b"pub fn one() -> u8 {\n    1\n}\n";
pub const BINARY: &[u8] = &[0, 159, 146, 150, 255, b'\n', 7];

/// A fresh copy of the repository `build` makes in the folder it is given,
/// at `<tempdir>/work`, and `main`'s commit there.
///
/// The repository is built once per test binary — under Cargo's temporary
/// folder for this binary, replaced at each run — and copied for each test.
/// Building takes a dozen git processes; copying takes none, and a test's
/// own repository is still its alone.
pub fn cached_repo(name: &str, build: fn(&Path)) -> (TempDir, PathBuf, Oid) {
    static BUILT: Mutex<Vec<(String, PathBuf, Oid)>> = Mutex::new(Vec::new());
    let (template, base) = {
        let mut built = BUILT.lock().unwrap();
        match built.iter().find(|(n, ..)| n == name) {
            Some((_, template, base)) => (template.clone(), base.clone()),
            None => {
                let exe = std::env::current_exe().unwrap();
                let binary = exe.file_stem().unwrap().to_string_lossy().into_owned();
                let dir =
                    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("fixture-{binary}-{name}"));
                let _ = std::fs::remove_dir_all(&dir);
                let template = dir.join("work");
                std::fs::create_dir_all(&template).unwrap();
                build(&template);
                let base = Oid::parse(git(&template, &["rev-parse", "main"]).trim()).unwrap();
                built.push((name.to_string(), template.clone(), base.clone()));
                (template, base)
            }
        }
    };
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("work");
    copy_tree(&template, &root);
    (dir, root, base)
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

pub struct Fixture {
    pub dir: TempDir,
    pub root: PathBuf,
    pub sandbox: Sandbox,
    /// `main`'s commit.
    pub base: Oid,
}

/// A repository on `main` — text, a binary file, ignore rules, and a copy of
/// `src/lib.rs` under `templates/` — with a second branch `feature` that adds
/// `FEATURE.md`, and a sandbox over it that may run the platform's shell.
pub fn fixture() -> Fixture {
    fixture_allowing(&[SHELL])
}

/// [`fixture`], with a sandbox that may run exactly `programs`.
pub fn fixture_allowing(programs: &[&str]) -> Fixture {
    let (dir, root, base) = cached_repo("sandbox", build_sandbox_repo);
    let sandbox = Sandbox::new(
        Repo::open(&root).unwrap(),
        CommandPolicy::allowing(programs.iter().copied()),
    );
    Fixture {
        dir,
        root,
        sandbox,
        base,
    }
}

fn build_sandbox_repo(root: &Path) {
    // No template: git's sample hooks are files every copy would carry.
    git(root, &["init", "-q", "--template="]);
    git(root, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    git(root, &["config", "core.autocrlf", "false"]);
    put(root, ".gitignore", b"target/\n*.log\n");
    put(root, "README.md", README);
    put(root, "src/lib.rs", LIB);
    put(root, "templates/lib.rs.orig", LIB);
    put(root, "assets/logo.bin", BINARY);
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "base"]);
    git(root, &["checkout", "-q", "-b", "feature"]);
    put(root, "FEATURE.md", b"only on the feature branch\n");
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "feature"]);
    git(root, &["checkout", "-q", "main"]);
}

pub fn branch(name: &str) -> BranchName {
    BranchName::parse(name).unwrap()
}

pub fn grant() -> DiskWriteGrant {
    DiskWriteGrant::issue()
}

/// `windows` in `cmd`, or `unix` in `sh`.
pub fn shell(windows: &str, unix: &str) -> SandboxCommand {
    if cfg!(windows) {
        SandboxCommand::new("cmd").args(["/D", "/C", windows])
    } else {
        SandboxCommand::new("sh").args(["-c", unix])
    }
}

pub async fn run(f: &Fixture, on: &str, files: &VfsStore, command: &SandboxCommand) -> RunOutcome {
    f.sandbox
        .run(&grant(), &branch(on), files, command)
        .await
        .unwrap_or_else(|e| panic!("{e}: {e:?}"))
}

/// **The checkout is the branch and nothing else**: `HEAD` on it, no file
/// differing from its commit, nothing a run added left behind.
pub fn assert_reset(f: &Fixture, on: &str) {
    let (head, status) = f.sandbox.repo().status_with_head().unwrap();
    assert_eq!(head.branch(), Some(&branch(on)));
    assert!(
        status.is_empty(),
        "the checkout was left changed: {status:?}"
    );
}

pub fn read(files: &VfsStore, path: &str) -> Option<String> {
    files.read(path).unwrap()
}

/// The paths a run recorded, in order.
pub fn changed(done: &RunOutcome) -> Vec<&str> {
    done.changed.iter().map(|c| c.path.as_str()).collect()
}

/// The paths a run could not record, in order.
pub fn unrecorded(done: &RunOutcome) -> Vec<&str> {
    done.unrecorded.iter().map(|c| c.path.as_str()).collect()
}
