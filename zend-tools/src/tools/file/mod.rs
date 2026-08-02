//! Overlay file tools: `file_{write,read,edit,list,delete,present}`.
//!
//! All operations target the overlay filesystem ([`crate::state::VfsStore`]): an
//! in-memory session layer stacked over the daemon's working directory. Reads
//! resolve session-first and fall through to the real project; writes, edits, and
//! deletes stay in memory. **Nothing here ever modifies a file on disk.**
//!
//! Editing a file that exists only in the workspace copies it into the session
//! layer first, so the edit applies to the session's own copy and every later read
//! of that path sees it. Deleting a workspace-backed file records a whiteout — the
//! path stops resolving and stops listing, the file on disk is untouched.
//!
//! # Path semantics
//!
//! Paths are normalised before use (see [`crate::state::VfsStore`]). `/workspace`
//! is the mount point of the working directory, so `/workspace/src/main.rs`,
//! `./src/../src/main.rs`, `/src/main.rs`, and `src/main.rs` are all one entry.
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
//! | `not_found` | Path resolves in neither layer (`file_read`, `file_edit`, `file_delete`) |
//! | `vfs_full` | Write or copy-up would exceed the 10 MiB session cap |
//! | `ambiguous` | `old_str` appears more than once in the file (`file_edit`) |
//! | `no_files_found` | All requested paths are missing (`file_present`) |
//! | `unreadable` | Workspace file is above the read limit or is not UTF-8 text |

use serde::Serialize;
use thiserror::Error;

use crate::state::vfs::VfsError;
use crate::ToolError;

/// Which slice of a larger listing a response carries, and how to get the rest.
///
/// Tool results are injected into the conversation verbatim, so an unbounded one
/// is a context hazard — a single `file_list` over `zend/src/` produced a 5.7k
/// token turn. Listings are therefore paged and report here how much they held
/// back. `file_read` bounds itself differently, by line range: see
/// [`read::MAX_READ_LINES`], whose continuation signal rides in the excerpt
/// header so it matches the `code_reading` ingest's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Paging {
    /// Zero-based index of the page returned. Clamped into range, so asking past
    /// the end yields the last page rather than an error or an empty result.
    pub page: u32,
    /// Total number of pages available for this request.
    pub pages: u32,
    /// Items per page — the cap this tool applies.
    pub per_page: usize,
    /// Total items matching the request across all pages.
    pub total: usize,
    /// The page to request next, or `null` when this is the last one.
    pub next_page: Option<u32>,
}

impl Paging {
    /// Describe page `requested` of `total` items at `per_page`. An out-of-range
    /// request clamps to the last page: a model that over-shoots gets the tail of
    /// the data rather than a silent empty list it would read as "nothing there".
    pub fn of(total: usize, requested: u32, per_page: usize) -> Self {
        let per_page = per_page.max(1);
        let pages = total.div_ceil(per_page).max(1) as u32;
        let page = requested.min(pages - 1);
        Paging {
            page,
            pages,
            per_page,
            total,
            next_page: (page + 1 < pages).then_some(page + 1),
        }
    }

    /// Items to skip to reach this page.
    pub fn skipped(&self) -> usize {
        self.page as usize * self.per_page
    }
}

pub mod delete;
pub mod edit;
pub mod list;
pub mod present;
pub mod read;
pub mod render;
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
    #[error("{0}")]
    Unreadable(String),
}

impl ToolError for FileError {
    fn code(&self) -> &'static str {
        match self {
            FileError::NotFound(_) => "not_found",
            FileError::VfsFull => "vfs_full",
            FileError::Ambiguous => "ambiguous",
            FileError::NoFilesFound => "no_files_found",
            FileError::Unreadable(_) => "unreadable",
        }
    }
}

impl From<VfsError> for FileError {
    fn from(e: VfsError) -> Self {
        match e {
            VfsError::Full => FileError::VfsFull,
            VfsError::Unreadable(why) => FileError::Unreadable(why),
        }
    }
}
