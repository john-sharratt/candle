//! Creating repositories: a fresh one, or a clone of a remote.

pub mod clone;
pub mod init;

use std::path::Path;

use crate::error::GitError;

/// Refuse a target folder that already holds something: creating a
/// repository there would mix with, or overwrite, what is already there.
pub(crate) fn require_empty_target(dir: &Path) -> Result<(), GitError> {
    if !dir.is_absolute() {
        return Err(GitError::invalid(format!(
            "{} is not an absolute path",
            dir.display()
        )));
    }
    if dir.exists() {
        let empty = std::fs::read_dir(dir)?.next().is_none();
        if !dir.is_dir() || !empty {
            return Err(GitError::invalid(format!(
                "{} already exists and is not an empty folder",
                dir.display()
            )));
        }
    }
    Ok(())
}
