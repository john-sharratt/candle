//! Overlay file tools: `file_{write,read,edit,list,search,grep,delete,present}`.
//!
//! # Finding things
//!
//! [`search`] finds files by **name or path**, [`grep`] finds them by
//! **content**, and both cover the whole project in one call. They exist
//! because without them the only way to locate anything was to guess directory
//! names at [`list`] and read whole files to check — a real session spent
//! eighteen turns and 360 KB of context doing exactly that, and the largest
//! file it read was the wrong one.
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
//! # `file_edit` patches
//!
//! `file_edit` takes a unified diff. Hunks are located by their context, not by
//! the `@@` line numbers, so a stale line number costs nothing while a hunk that
//! matches in more than one place is `ambiguous` rather than a guess. A hunk
//! whose change is already in the file counts as already applied, which is what
//! makes sending the same patch twice a no-op; a hunk that matches nowhere is
//! `not_found`, and a patch that is not a readable diff is `invalid_arguments`.
//! Either every hunk lands or the file is left exactly as it was. The engine is
//! [`patch`], where the format and each failure are documented.
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
//! | `not_found` | Path resolves in neither layer (`file_read`, `file_edit`, `file_delete`), or a `file_edit` hunk matches nothing |
//! | `vfs_full` | Write or copy-up would exceed the 10 MiB session cap |
//! | `ambiguous` | A `file_edit` hunk matches in more than one place |
//! | `no_files_found` | All requested paths are missing (`file_present`) |
//! | `unreadable` | Workspace file is above the read limit or is not UTF-8 text |
//! | `invalid_arguments` | The `file_edit` patch is not a readable unified diff, or a `file_grep` pattern is not a valid regex |
//! | `forbidden` | The path is under a `secrets/` directory — see [`crate::state::vfs`] |

use serde::Serialize;
use thiserror::Error;

use self::patch::PatchError;
use crate::state::vfs::VfsError;
use crate::ToolError;

/// Which slice of a larger listing a response carries, and how to get the rest.
///
/// Tool results are injected into the conversation verbatim, so an unbounded one
/// is a context hazard — a single `file_list` over `zend/src/` produced a 5.7k
/// token turn. Listings are therefore paged and report here how much they held
/// back.
///
/// `file_read` is bounded too, but carries no `Paging`: its range is required
/// and capped at [`read::MAX_READ_LINES`], and the excerpt header is its own
/// paging record — `(lines a-b of N)` when it stops short of the end, the plain
/// `(lines a-b)` when it reached it, in the `code_reading` ingest's format. The
/// header serves the model directly, in the text it is already reading, where a
/// structured field beside a rendered string would have to be correlated with it.
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
pub mod grep;
pub mod list;
pub mod patch;
pub mod present;
pub mod read;
pub mod render;
pub mod search;
pub mod write;

pub use delete::FILE_DELETE;
pub use edit::FILE_EDIT;
pub use grep::FILE_GREP;
pub use list::FILE_LIST;
pub use present::FILE_PRESENT;
pub use read::FILE_READ;
pub use search::FILE_SEARCH;
pub use write::FILE_WRITE;

#[derive(Debug, Error)]
pub enum FileError {
    #[error("file not found: {0}")]
    NotFound(String),
    #[error("VFS storage limit exceeded")]
    VfsFull,
    /// A `file_edit` hunk matched nowhere in the file. It shares the
    /// `not_found` code with a missing path because it is the same answer —
    /// what the call named is not there — and the detail says which.
    #[error("{0}")]
    HunkUnmatched(String),
    /// A `file_edit` hunk matched in more than one place.
    #[error("{0}")]
    Ambiguous(String),
    #[error("no files found")]
    NoFilesFound,
    #[error("{0}")]
    Unreadable(String),
    /// The `file_edit` patch is not a unified diff the engine can read, or a
    /// `file_grep` pattern is not a valid regular expression.
    #[error("{0}")]
    InvalidArguments(String),
    /// The path is under a `secrets/` directory. Named distinctly from
    /// `not_found` so a model reads it as "I may not look here" and stops,
    /// rather than as "wrong path" and tries six more spellings.
    #[error("{0}")]
    Forbidden(String),
    /// A write to the workspace on disk (the Mutable tools mode) failed.
    #[error("{0}")]
    Unwritable(String),
}

impl ToolError for FileError {
    fn code(&self) -> &'static str {
        match self {
            FileError::NotFound(_) | FileError::HunkUnmatched(_) => "not_found",
            FileError::VfsFull => "vfs_full",
            FileError::Ambiguous(_) => "ambiguous",
            FileError::NoFilesFound => "no_files_found",
            FileError::Unreadable(_) => "unreadable",
            FileError::InvalidArguments(_) => "invalid_arguments",
            FileError::Forbidden(_) => "forbidden",
            FileError::Unwritable(_) => "unwritable",
        }
    }
}

impl From<VfsError> for FileError {
    fn from(e: VfsError) -> Self {
        match e {
            VfsError::Full => FileError::VfsFull,
            VfsError::Unreadable(why) => FileError::Unreadable(why),
            // Keep the store's own wording: it names the path and says the
            // directory is off limits, which is exactly what the model needs to
            // stop rather than retry.
            VfsError::Forbidden(path) => {
                FileError::Forbidden(VfsError::Forbidden(path).to_string())
            }
            VfsError::Unwritable(why) => FileError::Unwritable(why),
        }
    }
}

impl From<PatchError> for FileError {
    fn from(e: PatchError) -> Self {
        match e {
            PatchError::Malformed(why) => FileError::InvalidArguments(why),
            PatchError::Ambiguous(why) => FileError::Ambiguous(why),
            PatchError::Unmatched(why) => FileError::HunkUnmatched(why),
        }
    }
}
