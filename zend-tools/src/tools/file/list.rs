//! file_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{FileError, Paging};
use crate::{RegisteredTool, Tool, ToolContext};

/// Entries per page. A listing goes into the conversation verbatim, so an
/// unbounded one is a context hazard: `zend/src/` alone is 175 files ≈ 5.7k
/// tokens as JSON, larger than most whole turns. At ~30 tokens per entry this
/// keeps a page near 1.5k tokens, and the model pages when it needs more.
pub const LIST_PAGE_ENTRIES: usize = 50;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// Path prefix to filter results (e.g. `src/`). Defaults to "" — the project root.
    pub prefix: Option<String>,
    /// Zero-based page of results to return. Defaults to 0. When the response's
    /// `paging.next_page` is set, pass it here to read the following page.
    pub page: Option<u32>,
}

#[derive(Serialize)]
pub struct FileEntry {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
    /// `true` when this session has written or edited the file, so the content
    /// differs from what is on disk in the workspace. Omitted when false, which
    /// is the common case — it would otherwise be a third of the payload.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub modified: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub files: Vec<FileEntry>,
    /// Which slice of the matching files this is, and how to get the rest.
    pub paging: Paging,
    /// Bytes held in the session layer — the 10 MiB budget's denominator.
    /// Workspace files are read on demand and cost nothing against it.
    pub total_bytes: usize,
}

pub struct FileList;

impl Tool for FileList {
    const NAME: &'static str = "file_list";
    const DESCRIPTION: &'static str =
        "List the files visible to this session: the project's working directory \
         plus anything written or edited during the session, which shadows the \
         file of the same path on disk. Optionally narrowed to a path prefix — \
         omit it to list from the project root. Ignored paths (per .gitignore and \
         friends) never appear. Results are paged: the response's `paging` reports \
         the total and, when more remain, a `next_page` to pass back as `page`. \
         Returns names, sizes, and line counts, not file contents; an entry \
         carries `modified: true` when this session has changed it. Use file_read \
         to get a file's contents.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<ListResponse, FileError> {
        let prefix = req.prefix.as_deref().unwrap_or("");
        let total_bytes = ctx.vfs.total_bytes();
        let all = ctx.vfs.list(prefix);
        let paging = Paging::of(all.len(), req.page.unwrap_or(0), LIST_PAGE_ENTRIES);
        let files = all
            .into_iter()
            .skip(paging.skipped())
            .take(LIST_PAGE_ENTRIES)
            .map(|e| FileEntry {
                path: e.path,
                bytes: e.bytes,
                lines: e.lines,
                modified: e.modified,
            })
            .collect();
        Ok(ListResponse {
            files,
            paging,
            total_bytes,
        })
    }
}

pub const FILE_LIST: RegisteredTool = RegisteredTool::new::<FileList>();
