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
//! All operations target the overlay filesystem ([`VfsStore`]): an
//! in-memory session layer stacked over the daemon's working directory. Reads
//! resolve session-first and fall through to the real project; writes, edits, and
//! deletes stay in memory. **Nothing here ever modifies a file on disk.**
//!
//! Editing a file that exists only in the workspace copies it into the session
//! layer first, so the edit applies to the session's own copy and every later read
//! of that path sees it. Deleting a workspace-backed file records a whiteout — the
//! path stops resolving and stops listing, the file on disk is untouched.
//!
//! # Repositories and paths
//!
//! Every call names a `repo` — one of the repositories the workspace lists —
//! and its paths are relative to that repository's folder
//! ([`zend_vfs::RepoFiles`]). `file_list`, `file_search` and `file_grep`
//! also take [`ALL_REPOS`] (`"*"`) to cover every repository at once; their
//! results then name the repository each entry came from. The scope is always
//! stated: a call that means the whole workspace says so.
//!
//! Paths are normalised before use (see [`VfsStore`]), so
//! `./src/../src/main.rs`, `/src/main.rs`, and `src/main.rs` are all one entry.
//!
//! # `file_edit` replacements
//!
//! `file_edit` replaces `old_text` — quoted from the file as it stands — with
//! `new_text`. Text found exactly is replaced; text quoted at the wrong
//! indentation is found line by line and the replacement re-indented to the
//! file's. Text that occurs more than once is `ambiguous` unless every
//! occurrence is asked for; text that is nowhere is `not_found`. An edit whose
//! result is already in the file counts as already applied, which is what makes
//! sending it twice a no-op. The engine is [`zend_vfs::replace`], where the
//! matching and each failure are documented.
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
//! | `not_found` | Path resolves in neither layer (`file_read`, `file_edit`, `file_delete`), or a `file_edit` `old_text` is not in the file |
//! | `vfs_full` | Write or copy-up would exceed the 10 MiB session cap |
//! | `ambiguous` | A `file_edit` `old_text` occurs more than once and `replace_all` is not set |
//! | `no_files_found` | All requested paths are missing (`file_present`) |
//! | `unreadable` | Workspace file is above the read limit or is not UTF-8 text |
//! | `invalid_arguments` | A `file_edit` `old_text` is empty or the same as `new_text`, a `file_grep` pattern is not a valid regex, or a `file_read` path is a web address |
//! | `forbidden` | The path is under a `secrets/` directory — see [`zend_vfs::vfs`] |
//! | `unknown_repo` | `repo` names no repository in the workspace; the message lists the ones it does |

use std::sync::Arc;

use serde::Serialize;
use thiserror::Error;

use zend_vfs::replace::ReplaceError;
use zend_vfs::{UnknownRepo, VfsError, VfsStore, ALL_REPOS};

use crate::tools::code::UNKNOWN_REPO;
use crate::{ToolContext, ToolError};

/// The stores a call covers: the named repository's alone, or — for
/// [`ALL_REPOS`] — every repository's, in manifest order.
pub(crate) fn stores_for(
    ctx: &ToolContext,
    repo: &str,
) -> Result<Vec<(String, Arc<VfsStore>)>, FileError> {
    if repo == ALL_REPOS {
        return Ok(ctx.files.all());
    }
    Ok(vec![(repo.to_string(), ctx.files.repo(repo)?)])
}

/// A file somewhere in the workspace: the repository it is in and its path
/// inside that repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RepoPath {
    pub repo: String,
    pub path: String,
}

/// Which slice of a larger listing a response carries, and how to get the rest.
///
/// Tool results are injected into the conversation verbatim, so an unbounded one
/// is a context hazard — a single `file_list` over `zend/src/` produced a 5.7k
/// token turn. Listings are therefore paged and report here how much they held
/// back.
///
/// `file_read` is bounded too, but carries no `Paging`: its page is required and
/// fixed at [`zend_vfs::vfs::PAGE_LINES`] lines, and the excerpt header is
/// its own paging record — `(page P of N, lines a-b of total)` — in the
/// `code_reading` ingest's format. The header serves the model directly, in the
/// text it is already reading, where a structured field beside a rendered
/// string would have to be correlated with it.
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
    /// `file_edit` named a file that does not exist. Worded as the way out, not
    /// only the fault: a model told "not found" alone rewrote its patch as an
    /// add-file diff ten times over instead of calling `write`.
    #[error(
        "file not found: {0} — file_edit only changes a file that exists; to create \
         it, call `write` with `path` and the whole file as `content`"
    )]
    NothingToEdit(String),
    /// `file_read` was given a web address. A model that read a URL as a path
    /// got `not_found` and tried the next spelling of the same URL; this names
    /// the tool that reads it.
    #[error(
        "{0} is a web address, not a file — file_read reads files in the project; \
         to read a web page, call `web_fetch` with it as `url`"
    )]
    IsUrl(String),
    #[error("VFS storage limit exceeded")]
    VfsFull,
    /// A `file_edit` `old_text` is nowhere in the file. It shares the
    /// `not_found` code with a missing path because it is the same answer —
    /// what the call named is not there — and the detail says which.
    #[error("{0}")]
    TextUnmatched(String),
    /// A `file_edit` `old_text` occurs more than once.
    #[error("{0}")]
    Ambiguous(String),
    #[error("no files found")]
    NoFilesFound,
    #[error("{0}")]
    Unreadable(String),
    /// A `file_edit` that changes nothing — an empty `old_text`, or a
    /// `new_text` the same as it — or a `file_grep` pattern that is not a
    /// valid regular expression.
    #[error("{0}")]
    InvalidArguments(String),
    /// The path is under a `secrets/` directory. Named distinctly from
    /// `not_found` so a model reads it as "I may not look here" and stops,
    /// rather than as "wrong path" and tries six more spellings.
    #[error("{0}")]
    Forbidden(String),
    /// A change the store cannot make — the file's base moved on a store that
    /// reads no branch.
    #[error("{0}")]
    Unwritable(String),
    /// The conversation edited a file whose copy on disk has since changed, so
    /// its edit no longer fits. The detail names the way out.
    #[error("{0}")]
    Diverged(String),
    #[error(transparent)]
    UnknownRepo(#[from] UnknownRepo),
}

impl ToolError for FileError {
    fn code(&self) -> &'static str {
        match self {
            FileError::NotFound(_) | FileError::NothingToEdit(_) | FileError::TextUnmatched(_) => {
                "not_found"
            }
            FileError::VfsFull => "vfs_full",
            FileError::Ambiguous(_) => "ambiguous",
            FileError::NoFilesFound => "no_files_found",
            FileError::Unreadable(_) => "unreadable",
            FileError::InvalidArguments(_) | FileError::IsUrl(_) => "invalid_arguments",
            FileError::Forbidden(_) => "forbidden",
            FileError::Unwritable(_) => "unwritable",
            FileError::Diverged(_) => "diverged",
            FileError::UnknownRepo(_) => UNKNOWN_REPO,
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
            VfsError::Diverged(why) => FileError::Diverged(why),
        }
    }
}

impl From<ReplaceError> for FileError {
    fn from(e: ReplaceError) -> Self {
        match e {
            ReplaceError::Invalid(why) => FileError::InvalidArguments(why),
            ReplaceError::Ambiguous(why) => FileError::Ambiguous(why),
            ReplaceError::Unmatched(why) => FileError::TextUnmatched(why),
        }
    }
}
