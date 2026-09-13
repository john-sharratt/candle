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
    /// First line to return, 1-based. Defaults to 1 (the start of the file).
    pub start_line: Option<u32>,
    /// Last line to return, 1-based and inclusive. Defaults to the end of the
    /// file.
    pub end_line: Option<u32>,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file's content. Resolves against this session's edits first, then \
         falls through to the project's working directory, so real project files \
         can be read directly. Use for: reading a source file from the project, \
         looking at what was previously written, inspecting a file the user \
         uploaded into the chat, checking the current state of a draft after \
         edits. Triggered by \"show me the file\", \"read\", \"what's in\", \"open \
         the file\", \"cat\", \"display the contents of\". Returns the excerpt as \
         numbered source in a fenced block, headed by the path and the line range \
         it covers. With no range the whole file comes back; start_line and \
         end_line narrow it to a slice, and a header like `(lines 47-93 of 900)` \
         marks the slice as part of a longer file. For remote filesystems use \
         remote_fs_session_get to download first, then file_read.";

    type Request = ReadRequest;
    /// A rendered excerpt, not a JSON object: the runner places a string result
    /// into the `<tool_response>` verbatim, so a live read is byte-identical to
    /// the `code_reading` ingest's prefilled responses.
    type Response = String;
    type Error = FileError;

    /// The whole file, or the range asked for — there is no line cap. The only
    /// bound on a read is the workspace layer's per-file byte limit
    /// ([`MAX_LOWER_FILE_BYTES`](crate::state::vfs::MAX_LOWER_FILE_BYTES)).
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

        // Clamp into the file. `start` past the end reads the last line rather
        // than returning nothing a model would read as "empty".
        let start = req.start_line.unwrap_or(1).clamp(1, total);
        let end = req.end_line.unwrap_or(total).clamp(start, total);

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
