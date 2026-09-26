//! The daemon's workspace: the manifest's repositories plus `uploads`.
//!
//! The manifest ([`zend_tools::state::workspace`]) lists the repositories a
//! deployment works on. The daemon adds one of its own: [`UPLOADS_REPO`], the
//! folder the upload endpoint writes a user's files into. As a repository, an
//! uploaded file is addressed like every other file — `repo: uploads, path:
//! notes.py` — and its workspace-relative key, `uploads/notes.py`, is the one
//! the `code_reading` layer ingests it under.

use std::path::Path;

use zend_tools::state::{RepoSpec, Workspace, WorkspaceError};

/// The daemon-owned repository uploaded files are written into.
pub const UPLOADS_REPO: &str = "uploads";

/// Load `root`'s manifest and add the uploads repository, creating its folder.
pub fn open(root: &Path) -> anyhow::Result<Workspace> {
    let workspace = Workspace::load(root)?;
    with_uploads(workspace)
}

/// A workspace at `root` of the one repository `repo`, plus uploads — both
/// folders created. For a harness or a tool that serves a single checkout
/// without a manifest on disk.
pub fn single_repo(root: &Path, repo: &str) -> anyhow::Result<Workspace> {
    let dir = root.join(repo);
    std::fs::create_dir_all(&dir)
        .map_err(|e| anyhow::anyhow!("could not create {}: {e}", dir.display()))?;
    with_uploads(Workspace::new(root, vec![RepoSpec::named(repo)])?)
}

/// `workspace` with the uploads repository added, its folder created.
pub fn with_uploads(workspace: Workspace) -> anyhow::Result<Workspace> {
    let workspace = workspace.with_repo(UPLOADS_REPO).map_err(|e| match e {
        WorkspaceError::DuplicateName(_) => anyhow::anyhow!(
            "the manifest lists a repository named `{UPLOADS_REPO}`, which the daemon \
             reserves for uploaded files — rename it"
        ),
        other => other.into(),
    })?;
    let dir = workspace.root().join(UPLOADS_REPO);
    std::fs::create_dir_all(&dir)
        .map_err(|e| anyhow::anyhow!("could not create {}: {e}", dir.display()))?;
    Ok(workspace)
}

#[cfg(test)]
mod tests {
    use super::*;

    use zend_tools::state::MANIFEST_FILE;

    #[test]
    fn open_adds_the_uploads_repository_and_creates_its_folder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("candle")).unwrap();
        std::fs::write(dir.path().join(MANIFEST_FILE), "repos:\n  - name: candle\n").unwrap();
        let ws = open(dir.path()).unwrap();
        assert_eq!(ws.names(), vec!["candle", UPLOADS_REPO]);
        assert!(dir.path().join(UPLOADS_REPO).is_dir());
    }

    #[test]
    fn a_manifest_that_lists_uploads_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("uploads")).unwrap();
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
            "repos:\n  - name: uploads\n",
        )
        .unwrap();
        let err = open(dir.path()).unwrap_err().to_string();
        assert!(err.contains("reserves for uploaded files"), "{err}");
    }

    #[test]
    fn a_single_repository_workspace_creates_both_folders() {
        let dir = tempfile::tempdir().unwrap();
        let ws = single_repo(dir.path(), "project").unwrap();
        assert_eq!(ws.names(), vec!["project", UPLOADS_REPO]);
        assert!(dir.path().join("project").is_dir());
        assert!(dir.path().join(UPLOADS_REPO).is_dir());
    }

    #[test]
    fn a_missing_manifest_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = open(dir.path()).unwrap_err().to_string();
        assert!(err.contains(MANIFEST_FILE), "{err}");
    }
}
