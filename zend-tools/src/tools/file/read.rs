//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A page names itself. `path` and `page` are both required — that is the
/// order the `required` list declares them in, and so the order the
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
/// [`crate::state::vfs::PAGE_LINES`] lines regardless of what the model asks
/// for, with no clamp-and-explain step to get there.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Path of the file to read — a project file from the working directory, or one this session created (e.g. `src/main.rs`, or `/workspace/src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Zero-based page of the file to return, 200 lines a page. Required —
    /// pass 0 to start at the top of the file. A page past the end clamps to
    /// the last one — read page 0 first, its header names the total, then
    /// keep incrementing until a response's own page number stops advancing.
    pub page: u32,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file, one 200-line page at a time. Resolves against this session's \
         edits first, then falls through to the project's working directory, so \
         real project files can be read directly. Both path and page are required \
         — pass page 0 to read the top of the file; it is zero-based, so page 1 is \
         lines 201-400. Returns the page as numbered source in a fenced block, \
         headed by the path, the page and total page count, and the line range \
         covered — `(page 1 of 5, lines 201-400 of 1420)`. Keep incrementing page \
         until the header's page number stops advancing; a short file is entirely \
         on page 0. There is no need to find a file's length first: read page 0 \
         and the header reports it. To find the page worth reading, use file_grep \
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
        let page = ctx
            .vfs
            .read_page(&req.path, req.page)?
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        Ok(numbered_excerpt(
            &req.path,
            page.page,
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
