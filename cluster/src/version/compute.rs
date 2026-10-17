//! Computes a build's version from its git checkout (§3.5). Compiled twice: by
//! `build.rs`, which stamps the result into the crate, and by the crate's tests,
//! which drive it against scratch repositories.
//!
//! - `count`: commits on `origin/main` that HEAD contains — the commit count of
//!   `git merge-base HEAD origin/main`. A local commit does not move it.
//! - `hash`: that merge base, short.
//! - `local`: empty when the **build's inputs** are exactly the merge base's;
//!   otherwise the first 16 hex digits of a SHA-256 over how they differ — the
//!   working tree's diff against the merge base, and every untracked input file,
//!   by path and contents.
//!
//! **The inputs, not the checkout.** The build's inputs are the workspace's
//! `Cargo.toml` and `Cargo.lock` and every workspace crate's directory
//! ([`build_inputs`]). A file outside them — a doc, an editor's or a harness's
//! lock file — cannot change the binary, so it cannot make a build local: a
//! checkout whose only change was a tool's lock file used to report a local
//! build at a clean commit.
//!
//! **The content, not the route to it.** The diff is taken against the merge
//! base, not built from HEAD's commits, so the same sources on the same base
//! are the same `local` however they were reached — committed locally, left
//! uncommitted, or built again. Sources committed and pushed as they were built
//! are a clean build at the new count.
//!
//! A tree with no git, or no `origin/main`, has no count to give: it is count
//! 0, hash `none`, and a local build named by everything it can see.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub count: u32,
    pub hash: String,
    pub local: String,
}

fn git(root: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

fn git_text(root: &Path, args: &[&str]) -> Option<String> {
    Some(
        String::from_utf8_lossy(&git(root, args)?)
            .trim()
            .to_string(),
    )
}

pub fn compute(root: &Path) -> Version {
    let base = git_text(root, &["merge-base", "HEAD", "origin/main"]);
    let (count, hash) = match &base {
        Some(base) => (
            git_text(root, &["rev-list", "--count", base])
                .and_then(|c| c.parse().ok())
                .unwrap_or(0),
            git_text(root, &["rev-parse", "--short", base]).unwrap_or_else(|| "none".into()),
        ),
        None => (0, "none".into()),
    };

    let inputs = build_inputs(root);
    let scoped = |args: &[&str]| -> Option<Vec<u8>> {
        let mut all: Vec<&str> = args.to_vec();
        all.push("--");
        all.extend(inputs.iter().map(String::as_str));
        git(root, &all)
    };
    let mut digest = Sha256::new();
    let mut beyond = false;
    // With no base there is nothing to diff against: the build is named by
    // HEAD's history, as far as there is one.
    let against = match &base {
        Some(base) => base.clone(),
        None => {
            beyond = true;
            digest.update(b"commits\0");
            digest.update(git(root, &["rev-list", "HEAD"]).unwrap_or_else(|| b"no git".to_vec()));
            "HEAD".to_string()
        }
    };
    let diff = scoped(&["diff", &against, "--binary"]).unwrap_or_default();
    if !diff.is_empty() {
        beyond = true;
        digest.update(b"diff\0");
        digest.update(&diff);
    }
    let untracked =
        scoped(&["ls-files", "--others", "--exclude-standard", "-z"]).unwrap_or_default();
    for path in untracked.split(|b| *b == 0).filter(|p| !p.is_empty()) {
        beyond = true;
        digest.update(b"untracked\0");
        digest.update(path);
        digest.update(b"\0");
        let file = root.join(String::from_utf8_lossy(path).as_ref());
        digest.update(std::fs::read(file).unwrap_or_default());
    }
    let local = if beyond {
        digest
            .finalize()
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect()
    } else {
        String::new()
    };
    Version { count, hash, local }
}

/// The workspace's crate directories, by name, in order: every directory at
/// the root with a `Cargo.toml`, except `target/` and the hidden ones.
fn crate_dirs(root: &Path) -> Vec<PathBuf> {
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name != "target" && !name.starts_with('.') && p.join("Cargo.toml").is_file()
        })
        .collect();
    crates.sort();
    crates
}

/// What a build is made from, as git pathspecs relative to the root: the
/// workspace's `Cargo.toml` and `Cargo.lock`, and every crate's directory.
pub fn build_inputs(root: &Path) -> Vec<String> {
    let mut inputs = vec!["Cargo.toml".to_string(), "Cargo.lock".to_string()];
    inputs.extend(
        crate_dirs(root)
            .iter()
            .filter_map(|p| p.file_name()?.to_str().map(str::to_string)),
    );
    inputs
}

/// What the build script reruns on: git's HEAD, the refs HEAD and
/// `origin/main` name, the packed refs, and every workspace crate's `src/` and
/// `Cargo.toml` plus the UI directories. Never `target/`.
pub fn rerun_paths(root: &Path) -> Vec<PathBuf> {
    // A worktree keeps its HEAD in its own git directory and its refs in the
    // repository's common one.
    let resolve = |p: String| {
        let p = PathBuf::from(p);
        if p.is_absolute() {
            p
        } else {
            root.join(p)
        }
    };
    let git_dir = git_text(root, &["rev-parse", "--git-dir"])
        .map(resolve)
        .unwrap_or_else(|| root.join(".git"));
    let common = git_text(root, &["rev-parse", "--git-common-dir"])
        .map(resolve)
        .unwrap_or_else(|| git_dir.clone());
    let mut paths = vec![
        git_dir.join("HEAD"),
        common.join("packed-refs"),
        common
            .join("refs")
            .join("remotes")
            .join("origin")
            .join("main"),
    ];
    if let Ok(head) = std::fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(r) = head.trim().strip_prefix("ref: ") {
            paths.push(common.join(r));
        }
    }
    paths.push(root.join("Cargo.toml"));
    paths.push(root.join("Cargo.lock"));
    for dir in crate_dirs(root) {
        paths.push(dir.join("Cargo.toml"));
        paths.push(dir.join("src"));
        for ui in ["web", "content"] {
            if dir.join(ui).is_dir() {
                paths.push(dir.join(ui));
            }
        }
    }
    paths
}
