//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// The range is optional, and a missing bound is the file's own edge: no range
/// reads the whole file, `start_line` alone reads from there to the end, and
/// `end_line` alone reads from the top. Declared `path, start_line, end_line`,
/// the order the constrained decoder offers them in.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Path of the file to read — a project file from the working directory, or one this session created (e.g. `src/main.rs`, or `/workspace/src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// First line to return, 1-based. Omit to read from the top of the file.
    #[validate(range(min = 1))]
    pub start_line: Option<u32>,
    /// Last line to return, 1-based and inclusive. Omit to read to the end of the file.
    #[validate(range(min = 1))]
    pub end_line: Option<u32>,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file, or a range of its lines. Resolves against this session's \
         edits first, then falls through to the project's working directory, so \
         real project files can be read directly. Give only path to read the whole \
         file; add start_line and/or end_line to read part of it — start_line alone \
         reads to the end, end_line alone reads from the top. Returns the lines as \
         numbered source in a fenced block, headed by the path and the line range it \
         covers; a partial read's header reads `(lines 47-93 of 900)`. For remote \
         filesystems use remote_fs_session_get to download first, then file_read.";

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

        // A missing bound is the file's edge; a given one clamps into the file.
        // `start` past the end reads the last line rather than returning nothing
        // a model would read as "empty".
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
