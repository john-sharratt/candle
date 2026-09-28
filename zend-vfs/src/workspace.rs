//! The workspace: one folder holding the repositories a daemon serves.
//!
//! A workspace is a folder with a [`MANIFEST_FILE`] in it naming the
//! repositories in scope. Each repository is a folder directly under the
//! workspace — a git checkout or any other working tree — and its name is that
//! folder's name. Tools take the name as their `repo` argument and a path
//! relative to that repository, so `candle/src/lib.rs` on disk is
//! `repo: candle, path: src/lib.rs` to a tool.
//!
//! ```yaml
//! repos:
//!   - name: candle
//!   - name: battle-cities
//!   - name: mind
//! ```
//!
//! Only the listed folders are visible. Anything else in the workspace folder —
//! other checkouts, the daemon's own state — is outside every repository and
//! unreachable through the tools.
//!
//! # Rules, checked when the workspace is built
//!
//! * At least one repository.
//! * A name is one plain path segment — no separators, no `.`/`..`, nothing a
//!   Windows open would resolve to another name (the rule [`VfsStore`] applies
//!   to every path segment), not a `secrets` directory, and not `jobs`, where
//!   the command sandboxes log their jobs. Names are unique,
//!   so no two repositories share a folder and none nests inside another.
//! * [`Workspace::load`] also requires each folder to exist.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use super::sandbox::JOBS_DIR;
use super::vfs::VfsStore;

/// The manifest's file name, in the workspace folder.
pub const MANIFEST_FILE: &str = "workspace.yaml";

/// Why a workspace could not be built.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkspaceError {
    #[error("{MANIFEST_FILE} not found in {0} — a workspace folder must list its repositories")]
    Missing(PathBuf),
    #[error("{MANIFEST_FILE} in {path} could not be read: {why}")]
    Unreadable { path: PathBuf, why: String },
    #[error("{MANIFEST_FILE} lists no repositories")]
    NoRepos,
    #[error("repository name {0:?} is not a single plain folder name")]
    BadName(String),
    #[error("repository name {0:?} is listed twice")]
    DuplicateName(String),
    #[error("repository {name:?}: {dir} is not a directory")]
    NotADirectory { name: String, dir: PathBuf },
}

/// One repository as the manifest names it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepoSpec {
    /// What tools call it, and its folder's name under the workspace.
    pub name: String,
}

impl RepoSpec {
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    repos: Vec<RepoSpec>,
}

/// One repository in a built workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repo {
    /// What tools call it — the `repo` argument, and the first segment of every
    /// workspace-relative path inside it.
    pub name: String,
    /// Its folder on disk: the workspace root joined with the name.
    pub dir: PathBuf,
}

/// A validated workspace: its root folder and the repositories in scope, in
/// manifest order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    root: PathBuf,
    repos: Vec<Repo>,
}

impl Workspace {
    /// Read and validate `root`'s [`MANIFEST_FILE`], and check every
    /// repository's folder exists.
    pub fn load(root: impl Into<PathBuf>) -> Result<Self, WorkspaceError> {
        let root = root.into();
        let file = root.join(MANIFEST_FILE);
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(WorkspaceError::Missing(root));
            }
            Err(e) => {
                return Err(WorkspaceError::Unreadable {
                    path: root,
                    why: e.to_string(),
                });
            }
        };
        let manifest: Manifest =
            serde_yaml::from_str(&text).map_err(|e| WorkspaceError::Unreadable {
                path: root.clone(),
                why: e.to_string(),
            })?;
        let workspace = Self::new(root, manifest.repos)?;
        for repo in &workspace.repos {
            if !repo.dir.is_dir() {
                return Err(WorkspaceError::NotADirectory {
                    name: repo.name.clone(),
                    dir: repo.dir.clone(),
                });
            }
        }
        Ok(workspace)
    }

    /// Validate `specs` against `root` without touching the disk.
    pub fn new(root: impl Into<PathBuf>, specs: Vec<RepoSpec>) -> Result<Self, WorkspaceError> {
        let root = root.into();
        if specs.is_empty() {
            return Err(WorkspaceError::NoRepos);
        }
        let mut repos: Vec<Repo> = Vec::with_capacity(specs.len());
        for spec in specs {
            if !is_plain_name(&spec.name) {
                return Err(WorkspaceError::BadName(spec.name));
            }
            if repos.iter().any(|r| r.name == spec.name) {
                return Err(WorkspaceError::DuplicateName(spec.name));
            }
            repos.push(Repo {
                dir: root.join(&spec.name),
                name: spec.name,
            });
        }
        Ok(Self { root, repos })
    }

    /// This workspace with one more repository, `name`, after the listed ones —
    /// for a repository the daemon owns rather than the manifest. The same
    /// rules as a listed repository apply, so a manifest that already lists
    /// `name` is refused as a duplicate.
    pub fn with_repo(mut self, name: &str) -> Result<Self, WorkspaceError> {
        if !is_plain_name(name) {
            return Err(WorkspaceError::BadName(name.to_string()));
        }
        if self.repo(name).is_some() {
            return Err(WorkspaceError::DuplicateName(name.to_string()));
        }
        self.repos.push(Repo {
            name: name.to_string(),
            dir: self.root.join(name),
        });
        Ok(self)
    }

    /// The workspace folder.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Every repository, in manifest order.
    pub fn repos(&self) -> &[Repo] {
        &self.repos
    }

    /// The repository called `name`.
    pub fn repo(&self, name: &str) -> Option<&Repo> {
        self.repos.iter().find(|r| r.name == name)
    }

    /// Every repository's name, in manifest order.
    pub fn names(&self) -> Vec<String> {
        self.repos.iter().map(|r| r.name.clone()).collect()
    }

    /// Split a workspace-relative path (`candle/src/lib.rs`, `/` separated)
    /// into its repository and the path inside it (`src/lib.rs`, empty for the
    /// repository's own root). `None` when the first segment names no listed
    /// repository.
    pub fn split<'p>(&self, path: &'p str) -> Option<(&Repo, &'p str)> {
        let (first, rest) = path.split_once('/').unwrap_or((path, ""));
        self.repo(first).map(|r| (r, rest))
    }
}

/// The `repo` value that means every repository at once — what `file_list`,
/// `file_search` and `file_grep` take to cover the whole workspace. Never a
/// repository's name: `*` is not a character a folder name may hold, so the
/// manifest refuses it (see [`is_plain_name`]).
pub const ALL_REPOS: &str = "*";

/// Whether `name` is one plain segment a repository can be called — a name
/// a folder on any platform may carry, so never [`ALL_REPOS`], and never the
/// folder the command sandboxes log their jobs to ([`JOBS_DIR`]), in any case:
/// a repository there would have every job's log written into its checkout.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '<', '>', ':', '"', '|', '?', '*'])
        && VfsStore::addressable(name)
        && !VfsStore::is_protected(name)
        && !name.eq_ignore_ascii_case(JOBS_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A repository's folder is its name**, and the order is the manifest's.
    #[test]
    fn repos_resolve_under_the_root_in_manifest_order() {
        let ws = Workspace::new(
            "/w",
            vec![RepoSpec::named("candle"), RepoSpec::named("mind")],
        )
        .unwrap();
        assert_eq!(ws.names(), vec!["candle", "mind"]);
        assert_eq!(
            ws.repo("candle").unwrap().dir,
            Path::new("/w").join("candle")
        );
        assert_eq!(ws.repo("mind").unwrap().dir, Path::new("/w").join("mind"));
        assert!(ws.repo("other").is_none());
    }

    #[test]
    fn a_workspace_path_splits_into_repository_and_inner_path() {
        let ws = Workspace::new("/w", vec![RepoSpec::named("candle")]).unwrap();
        let (repo, inner) = ws.split("candle/src/lib.rs").unwrap();
        assert_eq!((repo.name.as_str(), inner), ("candle", "src/lib.rs"));
        let (repo, inner) = ws.split("candle").unwrap();
        assert_eq!((repo.name.as_str(), inner), ("candle", ""));
        assert!(ws.split("other/x.rs").is_none());
        assert!(ws.split("").is_none());
    }

    #[test]
    fn a_repository_added_after_the_manifest_follows_the_same_rules() {
        let ws = Workspace::new("/w", vec![RepoSpec::named("candle")])
            .unwrap()
            .with_repo("uploads")
            .unwrap();
        assert_eq!(ws.names(), vec!["candle", "uploads"]);
        assert_eq!(
            ws.repo("uploads").unwrap().dir,
            Path::new("/w").join("uploads")
        );
        assert_eq!(
            ws.clone().with_repo("uploads"),
            Err(WorkspaceError::DuplicateName("uploads".to_string()))
        );
        assert_eq!(
            ws.with_repo("a/b"),
            Err(WorkspaceError::BadName("a/b".to_string()))
        );
    }

    #[test]
    fn an_empty_manifest_is_refused() {
        assert_eq!(Workspace::new("/w", vec![]), Err(WorkspaceError::NoRepos));
    }

    #[test]
    fn names_must_be_single_plain_segments() {
        for bad in [
            "", ".", "..", "a/b", "a\\b", "c:", "secrets", "SECRETS", "x.", "LONG~1", ALL_REPOS,
            "a*", "a?", "a|b", "<a>", "\"a\"", "jobs", "Jobs",
        ] {
            assert_eq!(
                Workspace::new("/w", vec![RepoSpec::named(bad)]),
                Err(WorkspaceError::BadName(bad.to_string())),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn a_name_listed_twice_is_refused() {
        assert_eq!(
            Workspace::new("/w", vec![RepoSpec::named("a"), RepoSpec::named("a")]),
            Err(WorkspaceError::DuplicateName("a".to_string()))
        );
    }

    #[test]
    fn load_reads_the_manifest_and_checks_each_folder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("candle")).unwrap();
        std::fs::create_dir(dir.path().join("mind")).unwrap();
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
            "repos:\n  - name: candle\n  - name: mind\n",
        )
        .unwrap();
        let ws = Workspace::load(dir.path()).unwrap();
        assert_eq!(ws.names(), vec!["candle", "mind"]);
        assert_eq!(ws.root(), dir.path());
    }

    #[test]
    fn load_refuses_a_missing_manifest_and_a_missing_folder() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            Workspace::load(dir.path()),
            Err(WorkspaceError::Missing(dir.path().to_path_buf()))
        );
        std::fs::write(dir.path().join(MANIFEST_FILE), "repos:\n  - name: gone\n").unwrap();
        assert_eq!(
            Workspace::load(dir.path()),
            Err(WorkspaceError::NotADirectory {
                name: "gone".to_string(),
                dir: dir.path().join("gone"),
            })
        );
    }

    /// A misspelt or unsupported key is an error, not an ignored field.
    #[test]
    fn load_refuses_unknown_keys() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("a")).unwrap();
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
            "repos:\n  - name: a\n    path: b\n",
        )
        .unwrap();
        assert!(matches!(
            Workspace::load(dir.path()),
            Err(WorkspaceError::Unreadable { .. })
        ));
    }
}
