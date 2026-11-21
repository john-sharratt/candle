//! The reasons a store refuses a read or a change.
//!
//! A workspace file that exists but cannot be served as text is reported apart
//! from one that is missing. A protected path is refused whether or not the file
//! exists, so the error cannot be used to learn what is there. A change the store
//! cannot make, and an edit whose workspace copy has changed on disk since, each
//! have a variant of their own, so the caller learns why nothing was applied.

use super::*;

#[derive(Debug)]
pub enum VfsError {
    Full,
    /// A workspace file exists but cannot be served as text — too large, or not
    /// valid UTF-8.
    Unreadable(String),
    /// The path is under a [`PROTECTED_SEGMENT`] directory. Refused whether or
    /// not it exists: saying "not found" for a real file and "forbidden" for a
    /// missing one would turn the error into an oracle for what is there.
    Forbidden(String),
    /// A change this store cannot make — a base moved on a store that reads
    /// no branch.
    Unwritable(String),
    /// The session edited the workspace's copy of a file, and that copy has
    /// since changed on disk so the edit no longer fits it.
    Diverged(String),
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VfsError::Full => write!(f, "VFS storage limit exceeded (10 MiB)"),
            VfsError::Unreadable(why) => write!(f, "{why}"),
            VfsError::Forbidden(path) => write!(
                f,
                "{path} is under a protected directory ({PROTECTED_SEGMENT}/ or \
                 {PROTECTED_GIT_DIR}/) and cannot be read, written or listed by \
                 tools; for a repository's branches, tags, history or file \
                 contents at a revision, use the git_* tools rather than its \
                 {PROTECTED_GIT_DIR}/ folder"
            ),
            VfsError::Unwritable(why) | VfsError::Diverged(why) => write!(f, "{why}"),
        }
    }
}

impl VfsStore {
    pub(super) fn not_text(norm: &str) -> VfsError {
        VfsError::Unreadable(format!("{norm} is not valid UTF-8 text"))
    }

    pub(super) fn diverged(norm: &str) -> VfsError {
        VfsError::Diverged(format!(
            "{norm} has changed on disk since this conversation edited it, so the edit no \
             longer fits it — write the whole file to set its content"
        ))
    }
}
