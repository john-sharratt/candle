//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A page names itself. `repo`, `path` and `page` are all required — that is
/// the order the `required` list declares them in, and so the order the
/// constrained decoder offers them in, which is also the order a call reads
/// in.
///
/// **A page is required because an unbounded read is not a slower read, it is
/// a different failure.** One `file_read` of a 2,499-line module put 144 KB
/// into a live conversation; three such reads made the next turn a
/// 53,288-token prefill, which the scheduler delivered in 8,192-token chunks
/// while the KV pool ratcheted 6 GB against a card already at 99% — three
/// minutes and fifty seconds of wall clock for one turn. Guidance in the tool
/// description did not prevent it: nine of nine reads in that conversation
/// asked for whole files. The schema does, because a call with no page cannot
/// be decoded against this stencil at all — and a fixed page, rather than an
/// arbitrary caller-chosen range, means every call costs the same
/// [`zend_vfs::vfs::PAGE_LINES`] lines regardless of what the model asks
/// for, with no clamp-and-explain step to get there.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// The repository the file belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the file to read, relative to the repository — a project file, or one this session created (e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Page of the file to return, from 1, 200 lines a page. Required — pass 1
    /// for the top of the file. Its fence names how many pages there are and
    /// the next one to read, or says it is the last. A page past the end
    /// returns the last one.
    #[validate(range(min = 1))]
    #[schemars(range(min = 1))]
    pub page: u32,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file, one 200-line page at a time. Resolves against this session's \
         edits first, then falls through to the repository on disk, so real \
         project files can be read directly. The repo, the path within it, and \
         the page are all required. Pages count from 1: page 1 is lines 1-200, \
         page 2 is lines 201-400. Returns the page as numbered source in a fenced \
         block whose opening fence names the file, the page, how many pages there \
         are and the file's length — `rust file=candle/src/lib.rs page=2 pages=8 \
         lines=1420` — and which ends naming the next page to read, `end of \
         candle/src/lib.rs page 2 of 8 — next: page=3`, or saying it is the \
         last, `end of candle/src/lib.rs page 8 of 8 — the last page`. A short \
         file is entirely on page 1. There is no need to find a file's length \
         first: read page 1 and its fence reports it. To find the \
         page worth reading, use file_grep \
         rather than paging a large file to look for it. For remote filesystems \
         use remote_fs_session_get to download first, then file_read.";

    type Request = ReadRequest;
    /// A rendered excerpt, not a JSON object: the runner places a string result
    /// into the `<tool_response>` verbatim, so a live read is byte-identical to
    /// the `code_reading` ingest's prefilled responses.
    type Response = String;
    type Error = FileError;

    /// Reads a file; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<String, FileError> {
        let lower = req.path.trim_start().to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") {
            return Err(FileError::IsUrl(req.path));
        }
        // The request counts from 1, as the model does; the store's page is an
        // index from 0. `page` is validated at least 1 before it gets here.
        let page = ctx
            .files
            .repo(&req.repo)?
            .read_page(&req.path, req.page - 1)?
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        Ok(numbered_excerpt(
            &req.repo,
            &req.path,
            page.page + 1,
            page.total_pages,
            page.start_line,
            page.end_line,
            page.total_lines,
            fence_tag_for_path(&req.path),
            &page.body,
        ))
    }
}

pub const FILE_READ: RegisteredTool = RegisteredTool::new::<FileRead>();
