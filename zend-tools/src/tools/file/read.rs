//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Tool, ToolContext};

/// Most lines one call returns. A file is unbounded — `zend/src/session.rs` is
/// 4765 lines — and a tool result lands in the conversation verbatim, so an
/// uncapped read is a context hazard the same way an uncapped listing was. At
/// ~10 tokens a line this keeps a response near 2k tokens.
///
/// A caller that requests 1-5000 gets the first 200 with the header saying so,
/// so the cap cannot be bypassed by naming a wide range.
pub const MAX_READ_LINES: u32 = 200;

/// Every field is required, the range included. With the range optional the
/// model wrote `{"path": …}` alone even after planning a range in its reasoning
/// — measured live, a request for lines 598-630 read lines 1-200 six times over
/// — so a read now always names the lines it wants, and the constrained decoder
/// forces them in `path, start_line, end_line` order.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Path of the file to read — a project file from the working directory, or one this session created (e.g. `src/main.rs`, or `/workspace/src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// First line to return, 1-based. Use 1 to read from the top of the file. Required.
    #[validate(range(min = 1))]
    pub start_line: u32,
    /// Last line to return, 1-based and inclusive. At most 200 lines come back per call, so a range wider than that stops at start_line + 199. Required.
    #[validate(range(min = 1))]
    pub end_line: u32,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a range of lines from a file. Resolves against this session's edits \
         first, then falls through to the project's working directory, so real \
         project files can be read directly. path, start_line and end_line are all \
         required: to read a file from the top ask for lines 1-200. Returns the \
         excerpt as numbered source in a fenced block, headed by the path and the \
         line range it covers. At most 200 lines come back per call: when the \
         header reads `(lines 1-200 of 900)` there is more, and the next call asks \
         for start_line 201. For remote filesystems use remote_fs_session_get to \
         download first, then file_read.";

    type Request = ReadRequest;
    /// A rendered excerpt, not a JSON object: the runner places a string result
    /// into the `<tool_response>` verbatim, so a live read is byte-identical to
    /// the `code_reading` ingest's prefilled responses.
    type Response = String;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<String, FileError> {
        let content = ctx
            .vfs
            .read(&req.path)?
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        // Split on '\n' rather than `lines()`: a trailing newline must not shift
        // the numbering, and the renderer handles the final empty element.
        let all: Vec<&str> = content.split('\n').collect();
        let total = if all.last() == Some(&"") {
            all.len().saturating_sub(1)
        } else {
            all.len()
        } as u32;
        if total == 0 {
            return Ok(numbered_excerpt(
                &req.path,
                1,
                0,
                0,
                fence_tag_for_path(&req.path),
                "",
            ));
        }

        // Clamp into the file, then cap the span. `start` past the end reads the
        // last line rather than returning nothing a model would read as "empty".
        let start = req.start_line.clamp(1, total);
        let requested_end = req.end_line.clamp(start, total);
        let end = requested_end.min(start + MAX_READ_LINES - 1);

        let body = all[(start - 1) as usize..end as usize].join("\n");
        Ok(numbered_excerpt(
            &req.path,
            start,
            end,
            total,
            fence_tag_for_path(&req.path),
            &body,
        ))
    }
}

pub const FILE_READ: RegisteredTool = RegisteredTool::new::<FileRead>();
