//! VFS file tools: `file_{write,read,edit,list,delete,present}`.
//!
//! All operations target the per-session in-memory virtual filesystem
//! ([`crate::state::VfsStore`]).  Nothing written here ever touches disk.
//!
//! # Path semantics
//!
//! Paths are normalised before use (see [`crate::state::VfsStore`]).  `./src/../main.rs`,
//! `/main.rs`, and `main.rs` all resolve to the same entry.
//!
//! # `file_edit` uniqueness requirement
//!
//! `file_edit` replaces `old_str` only if it appears exactly once in the file.
//! If it appears zero times → `not_found`; if it appears more than once →
//! `ambiguous` with a count.  This matches Claude Code's `str_replace` semantics
//! and forces the model to provide enough context to identify a single edit site.
//!
//! # `file_present`
//!
//! An explicit foreground gesture: the model calls this to draw the user's
//! attention to specific files as deliverables.  Distinct from passive Files-panel
//! visibility — `file_present` emits an SSE `file_present` frame; the panel is
//! driven by `write` / `file_edit` / `file_delete` events separately.
//!
//! # Size cap
//!
//! 10 MiB total VFS content per session.  `write` returns `vfs_full` if the
//! cap would be exceeded.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `not_found` | Path does not exist in VFS (`file_read`, `file_edit`, `file_delete`) |
//! | `vfs_full` | Write would exceed the 10 MiB session cap |
//! | `ambiguous` | `old_str` appears more than once in the file (`file_edit`) |
//! | `no_files_found` | All requested paths are missing (`file_present`) |

use thiserror::Error;

use crate::ToolError;

pub mod delete;
pub mod edit;
pub mod list;
pub mod present;
pub mod read;
pub mod write;

pub use delete::FILE_DELETE;
pub use edit::FILE_EDIT;
pub use list::FILE_LIST;
pub use present::FILE_PRESENT;
pub use read::FILE_READ;
pub use write::FILE_WRITE;

#[derive(Debug, Error)]
pub enum FileError {
    #[error("file not found: {0}")]
    NotFound(String),
    #[error("VFS storage limit exceeded")]
    VfsFull,
    #[error("ambiguous: old_str appears multiple times in file")]
    Ambiguous,
    #[error("no files found")]
    NoFilesFound,
}

impl ToolError for FileError {
    fn code(&self) -> &'static str {
        match self {
            FileError::NotFound(_) => "not_found",
            FileError::VfsFull => "vfs_full",
            FileError::Ambiguous => "ambiguous",
            FileError::NoFilesFound => "no_files_found",
        }
    }
}
