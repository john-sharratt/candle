//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Path of the file to read — a project file from the working directory, or one this session created (e.g. `src/main.rs`, or `/workspace/src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Zero-based page of the file to return, 300 lines a page. Required —
    /// pass 0 to start at the top of the file. A page past the end clamps to
    /// the last one — read page 0 first, its header names the total, then
    /// keep incrementing until a response's own page number stops advancing.
    pub page: u32,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file, one 300-line page at a time. Resolves against this session's \
         edits first, then falls through to the project's working directory, so \
         real project files can be read directly. Both path and page are required \
         — pass page 0 to read the top of the file; it is zero-based, so page 1 is \
         lines 301-600. Returns the page as numbered source in a fenced block, \
         headed by the path, the page and total page count, and the line range \
         covered — `(page 1 of 5, lines 301-600 of 1420)`. Keep incrementing page \
         until the header's page number stops advancing; a short file is entirely \
         on page 0. For remote filesystems use remote_fs_session_get to download \
         first, then file_read.";

    type Request = ReadRequest;
    /// A rendered excerpt, not a JSON object: the runner places a string result
    /// into the `<tool_response>` verbatim, so a live read is byte-identical to
    /// the `code_reading` ingest's prefilled responses.
    type Response = String;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<String, FileError> {
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
