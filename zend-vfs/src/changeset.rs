//! A set of file changes to commit onto a base.

use std::collections::BTreeMap;

use crate::error::GitError;
use crate::types::{FileMode, RepoPath};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// The file's new contents, as the working tree would hold them. `mode`
    /// `None` keeps the base's mode for an existing file and is
    /// [`FileMode::Regular`] for a new one.
    Write {
        content: Vec<u8>,
        mode: Option<FileMode>,
    },
    Delete,
}

/// At most one change per path, in path order.
///
/// Refuses a path under the protected `secrets` segment: nothing the file
/// tools will not serve may be committed on the model's behalf.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeSet {
    changes: BTreeMap<RepoPath, Change>,
}

impl ChangeSet {
    pub fn new() -> Self {
        Self::default()
    }

    fn check(path: &RepoPath) -> Result<(), GitError> {
        if path.is_protected() {
            return Err(GitError::invalid(format!(
                "{path} is under a protected `secrets` folder"
            )));
        }
        Ok(())
    }

    /// Set `path`'s contents, replacing any earlier change to it. `mode` is
    /// `Regular` or `Executable`; a symlink is only ever written through
    /// [`Self::symlink`].
    ///
    /// A write with no mode over a path the base holds as a symlink is
    /// refused when committed ([`crate::Repo::commit_changes`]): inheriting
    /// the link's mode would turn the written text into a link target.
    pub fn write(
        &mut self,
        path: RepoPath,
        content: Vec<u8>,
        mode: Option<FileMode>,
    ) -> Result<(), GitError> {
        Self::check(&path)?;
        if let Some(mode) = mode {
            if !matches!(mode, FileMode::Regular | FileMode::Executable) {
                return Err(GitError::invalid(format!(
                    "{path}: a file cannot be written with mode {}",
                    mode.as_str()
                )));
            }
        }
        self.changes.insert(path, Change::Write { content, mode });
        Ok(())
    }

    /// Make `path` a symlink to `target`. Deliberately its own call: a
    /// symlink's content is where it points, and one committed to a
    /// repository can lead any later reader of the checkout outside it.
    pub fn symlink(&mut self, path: RepoPath, target: &str) -> Result<(), GitError> {
        Self::check(&path)?;
        if target.is_empty() || target.contains('\0') {
            return Err(GitError::invalid(format!(
                "{path}: a symlink target must be non-empty and hold no NUL"
            )));
        }
        self.changes.insert(
            path,
            Change::Write {
                content: target.as_bytes().to_vec(),
                mode: Some(FileMode::Symlink),
            },
        );
        Ok(())
    }

    /// Delete `path`, replacing any earlier change to it.
    pub fn delete(&mut self, path: RepoPath) -> Result<(), GitError> {
        Self::check(&path)?;
        self.changes.insert(path, Change::Delete);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&RepoPath, &Change)> {
        self.changes.iter()
    }

    pub fn len(&self) -> usize {
        self.changes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> RepoPath {
        RepoPath::parse(s).unwrap()
    }

    #[test]
    fn one_change_per_path_in_path_order() {
        let mut c = ChangeSet::new();
        c.write(p("b.txt"), b"1".to_vec(), None).unwrap();
        c.write(p("a.txt"), b"2".to_vec(), None).unwrap();
        c.delete(p("b.txt")).unwrap();
        let got: Vec<(&str, &Change)> = c.iter().map(|(p, c)| (p.as_str(), c)).collect();
        assert_eq!(
            got,
            vec![
                (
                    "a.txt",
                    &Change::Write {
                        content: b"2".to_vec(),
                        mode: None
                    }
                ),
                ("b.txt", &Change::Delete),
            ]
        );
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn protected_paths_are_refused_for_writes_and_deletes() {
        let mut c = ChangeSet::new();
        assert!(c.write(p("secrets/key"), b"x".to_vec(), None).is_err());
        assert!(c.delete(p("web/Secrets/auth.yaml")).is_err());
        assert!(c.is_empty());
    }

    #[test]
    fn a_folder_submodule_or_symlink_mode_is_refused_on_write() {
        let mut c = ChangeSet::new();
        for mode in [FileMode::Tree, FileMode::Submodule, FileMode::Symlink] {
            assert!(c.write(p("x"), vec![], Some(mode)).is_err(), "{mode:?}");
        }
        assert!(c.write(p("x"), vec![], Some(FileMode::Executable)).is_ok());
    }

    #[test]
    fn a_symlink_is_its_own_explicit_call() {
        let mut c = ChangeSet::new();
        c.symlink(p("link"), "../target").unwrap();
        assert_eq!(
            c.iter().next().unwrap().1,
            &Change::Write {
                content: b"../target".to_vec(),
                mode: Some(FileMode::Symlink)
            }
        );
        assert!(c.symlink(p("empty"), "").is_err());
        assert!(c.symlink(p("secrets/link"), "x").is_err());
    }
}
