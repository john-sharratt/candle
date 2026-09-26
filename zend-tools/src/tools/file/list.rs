//! file_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{FileError, Paging};
use crate::state::ALL_REPOS;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Entries per page. A listing goes into the conversation verbatim, so an
/// unbounded one is a context hazard: `zend/src/` alone is 175 files ≈ 5.7k
/// tokens as JSON, larger than most whole turns. At ~30 tokens per entry this
/// keeps a page near 1.5k tokens, and the model pages when it needs more.
pub const LIST_PAGE_ENTRIES: usize = 50;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// The repository to list inside, or `*` to list the workspace's
    /// repositories themselves. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Directory to list, relative to the repository, e.g. `src/util`.
    /// Defaults to "" — the repository's root. Must name a real directory (or
    /// the root); a file path does not resolve. Not taken with repo `*`.
    pub path: Option<String>,
    /// Zero-based page of results to return. Defaults to 0. When the response's
    /// `paging.next_page` is set, pass it here to read the following page.
    pub page: Option<u32>,
}

#[derive(Serialize)]
pub struct FileEntry {
    /// Set only on an entry of the workspace listing, where each entry is a
    /// repository rather than a path inside one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Omitted on a repository entry, which is the repository's own root.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub path: String,
    /// Size from the directory entry's metadata — the only measure of a file a
    /// listing can give without opening it. There is deliberately no line count
    /// beside it: see [`crate::state::vfs::ListEntry`]. Omitted for a
    /// subdirectory entry — a directory has no size of its own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<usize>,
    /// `true` when this entry is a subdirectory. List it in turn to see what's
    /// inside — a listing is one level deep and never expands one for you.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dir: bool,
    /// `true` when this session has written or edited the file, so the content
    /// differs from what is on disk in the workspace. Omitted when false, which
    /// is the common case — it would otherwise be a third of the payload.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub modified: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    /// The repository listed inside, or `*` for the workspace listing.
    pub repo: String,
    pub entries: Vec<FileEntry>,
    /// Which slice of this directory's entries this is, and how to get the rest.
    pub paging: Paging,
    /// Bytes held in the session layer — the 10 MiB budget's denominator.
    /// Workspace files are read on demand and cost nothing against it.
    pub total_bytes: usize,
}

pub struct FileList;

impl Tool for FileList {
    const NAME: &'static str = "file_list";
    const DESCRIPTION: &'static str =
        "List one directory's immediate contents, as visible to this session: \
         the repository on disk plus anything written or edited during the \
         session, which shadows the file of the same path on disk. `repo` is \
         required: `*` lists the workspace's repositories; a repository alone \
         lists its root; add a directory `path` to list inside it. \
         A listing is one level deep — a subdirectory appears as its own \
         entry (`dir: true`), never expanded — so list it in turn to go \
         further. `path` must name a real directory; a file path does not \
         resolve. Ignored paths (per .gitignore and friends) never appear. \
         Results are paged: the response's `paging` reports the total and, \
         when more remain, a `next_page` to pass back as `page`. Returns names \
         and byte sizes, not file contents or line counts; a file entry carries \
         `modified: true` when this session has changed it. Use file_read to \
         get a file's contents — read page 0 and its header reports the \
         file's length in pages, so there is no need to size a file before \
         reading it.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = FileError;

    /// Lists a directory; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<ListResponse, FileError> {
        let path = req.path.as_deref().unwrap_or("");
        let total_bytes = ctx.files.total_bytes();
        let repo = req.repo;
        if repo == ALL_REPOS {
            if !path.is_empty() {
                return Err(FileError::InvalidArguments(format!(
                    "file_list with repo `*` lists the repositories and takes no path — \
                     {path:?} is relative to one repository; name it: {}",
                    ctx.files.names().join(", ")
                )));
            }
            let names = ctx.files.names();
            let paging = Paging::of(names.len(), req.page.unwrap_or(0), LIST_PAGE_ENTRIES);
            let entries = names
                .into_iter()
                .skip(paging.skipped())
                .take(LIST_PAGE_ENTRIES)
                .map(|name| FileEntry {
                    repo: Some(name),
                    path: String::new(),
                    bytes: None,
                    dir: true,
                    modified: false,
                })
                .collect();
            return Ok(ListResponse {
                repo,
                entries,
                paging,
                total_bytes,
            });
        }
        let all = ctx
            .files
            .repo(&repo)?
            .list_dir(path)?
            .ok_or_else(|| FileError::NotFound(path.to_string()))?;
        let paging = Paging::of(all.len(), req.page.unwrap_or(0), LIST_PAGE_ENTRIES);
        let entries = all
            .into_iter()
            .skip(paging.skipped())
            .take(LIST_PAGE_ENTRIES)
            .map(|e| FileEntry {
                repo: None,
                path: e.path,
                bytes: e.bytes,
                dir: e.dir,
                modified: e.modified,
            })
            .collect();
        Ok(ListResponse {
            repo,
            entries,
            paging,
            total_bytes,
        })
    }
}

pub const FILE_LIST: RegisteredTool = RegisteredTool::new::<FileList>();
