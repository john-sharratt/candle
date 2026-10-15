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
//!
//! # Repositories read from the substrate
//!
//! A repository can instead be read from the substrate
//! ([`Workspace::in_substrate`], `docs/substrate_git.md` §7): it keeps its
//! name, so every tool and path is unchanged, but it has no folder. Its object
//! database is opened once the daemon's substrate is, and [`Workspace::attach`]
//! hands it to every clone of the workspace at once.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::Deserialize;
use thiserror::Error;

use super::sandbox::JOBS_DIR;
use super::vfs::VfsStore;
use crate::{Fetch, GitError, ObjectDb, Repo as GitRepo};

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
    #[error("the workspace lists no repository {0:?}")]
    Unknown(String),
    #[error("repository {0:?} is not read from the substrate")]
    NotInSubstrate(String),
    #[error("repository {0:?} has already been attached to its object database")]
    AlreadyAttached(String),
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
    /// Where it is.
    pub place: RepoPlace,
}

/// Where a workspace repository is.
#[derive(Clone)]
pub enum RepoPlace {
    /// A folder on disk: the workspace root joined with the name.
    Folder(PathBuf),
    /// Read from the substrate: its object database, once
    /// [`Workspace::attach`] has given it one. Shared by every clone of the
    /// workspace.
    Substrate(Arc<OnceLock<Attached>>),
}

/// A repository read from the substrate, as [`Workspace::attach`] gives it:
/// its object database, and what makes its fetches.
#[derive(Clone)]
pub struct Attached {
    pub db: ObjectDb,
    pub fetcher: Arc<dyn Fetch>,
}

impl fmt::Debug for RepoPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Folder(dir) => f.debug_tuple("Folder").field(dir).finish(),
            Self::Substrate(slot) => f
                .debug_tuple("Substrate")
                .field(&slot.get().map(|a| a.db.name()))
                .finish(),
        }
    }
}

/// Two folders are the same place when their paths are; two repositories read
/// from the substrate when they share one slot.
impl PartialEq for RepoPlace {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Folder(a), Self::Folder(b)) => a == b,
            (Self::Substrate(a), Self::Substrate(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for RepoPlace {}

impl Repo {
    /// Its folder; `None` for a repository read from the substrate.
    pub fn folder(&self) -> Option<&Path> {
        match &self.place {
            RepoPlace::Folder(dir) => Some(dir),
            RepoPlace::Substrate(_) => None,
        }
    }

    /// Whether it is a git repository: a folder with a `.git` in it — read
    /// from the folder, never from a failure to open it — or a repository
    /// read from the substrate.
    pub fn is_git(&self) -> bool {
        match &self.place {
            RepoPlace::Folder(dir) => dir.join(".git").exists(),
            RepoPlace::Substrate(_) => true,
        }
    }

    /// It, as a git repository: the folder's, or one over its object
    /// database, fetching through what [`Workspace::attach`] gave it. Refused
    /// for a repository read from the substrate that has not been attached.
    pub fn open(&self) -> Result<GitRepo, GitError> {
        match &self.place {
            RepoPlace::Folder(dir) => GitRepo::open(dir),
            RepoPlace::Substrate(slot) => {
                let attached = slot.get().ok_or_else(|| {
                    GitError::invalid(format!(
                        "repository {} has not loaded from the substrate yet",
                        self.name
                    ))
                })?;
                Ok(GitRepo::in_substrate(&attached.db)?
                    .fetching_through(Arc::clone(&attached.fetcher)))
            }
        }
    }
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
        let workspace = Self::new(root.clone(), Self::manifest(&root)?)?;
        for repo in &workspace.repos {
            if let Some(dir) = repo.folder().filter(|d| !d.is_dir()) {
                return Err(WorkspaceError::NotADirectory {
                    name: repo.name.clone(),
                    dir: dir.to_path_buf(),
                });
            }
        }
        Ok(workspace)
    }

    /// The repositories `root`'s [`MANIFEST_FILE`] lists, read but not checked
    /// against the disk: for a member whose repositories are not folders under
    /// the workspace (`docs/substrate_git.md` §7).
    pub fn manifest(root: &Path) -> Result<Vec<RepoSpec>, WorkspaceError> {
        let file = root.join(MANIFEST_FILE);
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(WorkspaceError::Missing(root.to_path_buf()));
            }
            Err(e) => {
                return Err(WorkspaceError::Unreadable {
                    path: root.to_path_buf(),
                    why: e.to_string(),
                });
            }
        };
        let manifest: Manifest =
            serde_yaml::from_str(&text).map_err(|e| WorkspaceError::Unreadable {
                path: root.to_path_buf(),
                why: e.to_string(),
            })?;
        Ok(manifest.repos)
    }

    /// This workspace with repository `name` read from the substrate instead
    /// of its folder under the root (`docs/substrate_git.md` §7): no folder,
    /// and nothing to read until [`Self::attach`] gives it its object
    /// database. The repository keeps its name, so every workspace-relative
    /// path is unchanged.
    pub fn in_substrate(mut self, name: &str) -> Result<Self, WorkspaceError> {
        let Some(repo) = self.repos.iter_mut().find(|r| r.name == name) else {
            return Err(WorkspaceError::Unknown(name.to_string()));
        };
        repo.place = RepoPlace::Substrate(Arc::new(OnceLock::new()));
        Ok(self)
    }

    /// Give repository `name`, read from the substrate, its object database
    /// and what makes its fetches — in this workspace and every clone of it.
    /// Once: a repository's database is the one it keeps.
    pub fn attach(
        &self,
        name: &str,
        db: ObjectDb,
        fetcher: Arc<dyn Fetch>,
    ) -> Result<(), WorkspaceError> {
        let repo = self
            .repo(name)
            .ok_or_else(|| WorkspaceError::Unknown(name.to_string()))?;
        let RepoPlace::Substrate(slot) = &repo.place else {
            return Err(WorkspaceError::NotInSubstrate(name.to_string()));
        };
        slot.set(Attached { db, fetcher })
            .map_err(|_| WorkspaceError::AlreadyAttached(name.to_string()))
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
                place: RepoPlace::Folder(root.join(&spec.name)),
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
            place: RepoPlace::Folder(self.root.join(name)),
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
    use crate::fetch_here;
    use crate::odb::memory_store::MemoryStore;

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
            ws.repo("candle").unwrap().folder(),
            Some(Path::new("/w").join("candle").as_path())
        );
        assert_eq!(
            ws.repo("mind").unwrap().folder(),
            Some(Path::new("/w").join("mind").as_path())
        );
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
            ws.repo("uploads").unwrap().folder(),
            Some(Path::new("/w").join("uploads").as_path())
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

    /// **A repository read from the substrate has no folder, and opens once
    /// attached** — in every clone of the workspace, since they share it —
    /// and only once.
    #[test]
    fn a_repository_in_the_substrate_opens_in_every_clone_once_attached() {
        let ws = Workspace::new("/w", vec![RepoSpec::named("candle")])
            .unwrap()
            .in_substrate("candle")
            .unwrap();
        let clone = ws.clone();
        let repo = clone.repo("candle").unwrap();
        assert_eq!(repo.folder(), None);
        assert!(repo.is_git());
        assert!(
            repo.open().is_err(),
            "nothing to read before it is attached"
        );

        let db = ObjectDb::open("candle", Arc::new(MemoryStore::default()), 1 << 20).unwrap();
        ws.attach("candle", db.clone(), fetch_here()).unwrap();
        let opened = clone.repo("candle").unwrap().open().unwrap();
        assert_eq!(opened.objects().map(ObjectDb::name), Some("candle"));
        assert_eq!(
            ws.attach("candle", db.clone(), fetch_here()),
            Err(WorkspaceError::AlreadyAttached("candle".into()))
        );
        assert_eq!(
            ws.attach("other", db, fetch_here()),
            Err(WorkspaceError::Unknown("other".into()))
        );
        assert_eq!(
            Workspace::new("/w", vec![RepoSpec::named("a")])
                .unwrap()
                .in_substrate("b"),
            Err(WorkspaceError::Unknown("b".into()))
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
