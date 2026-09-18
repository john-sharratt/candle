//! file_read tool.

use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{fence_tag_for_path, numbered_excerpt};
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Most lines one `file_read` call returns, however wide a range it asks for.
///
/// The same size the rest of the system already treats as one excerpt, so a live
/// read and the prefilled excerpts the model was conditioned on are the same
/// kind of object — a scope, not a module. `zend`'s `repo_scan::anchor` bounds
/// its anchor excerpts at 200 by the identical `start + LIMIT - 1` clamp, and
/// the `code_reading` ingest carves scopes at 150 (`MAX_SCOPE_LINES`), so every
/// `file_read` exchange in the corpus already fits inside this cap and none had
/// to be re-cut for it.
pub const MAX_READ_LINES: u32 = 200;

/// Every read names a range. `path`, `start_line` and `end_line` are all
/// required — that is the order the `required` list declares them in, and so the
/// order the constrained decoder offers them in, which is also the order a call
/// reads in.
///
/// **A range is required because an unbounded read is not a slower read, it is a
/// different failure.** One `file_read` of a 2,499-line module put 144 KB into a
/// live conversation; three such reads made the next turn a 53,288-token prefill,
/// which the scheduler delivered in 8,192-token chunks while the KV pool ratcheted
/// 6 GB against a card already at 99% — three minutes and fifty seconds of wall
/// clock for one turn. Guidance in the tool description did not prevent it: nine
/// of nine reads in that conversation asked for whole files. The schema does,
/// because a call with no range cannot be decoded against this stencil at all.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Path of the file to read — a project file from the working directory, or one this session created (e.g. `src/main.rs`, or `/workspace/src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// First line of the range to return, 1-based. Required — every read names a range. Start where the answer is: a file_grep hit's line, a line named in an error, or the line after the previous excerpt ended.
    #[validate(range(min = 1))]
    pub start_line: u32,
    /// Last line of the range, 1-based and inclusive. Required. At most 200 lines come back per call; a wider range is served from start_line and the header says how much of the file is left.
    #[validate(range(min = 1))]
    pub end_line: u32,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a range of lines from a file. Resolves against this session's edits \
         first, then falls through to the project's working directory, so real \
         project files can be read directly. path, start_line and end_line are ALL \
         REQUIRED: there is no whole-file read, and at most 200 lines come back per \
         call. Aim the range at the answer — a file_grep hit's line number, a line \
         named in an error, the line after the previous excerpt ended — and read on \
         if it proves too narrow. Returns the lines as numbered source in a fenced \
         block, headed by the path and the range it covers: `(lines 1-200 of 2499)` \
         means 200 lines came back and the file runs to 2499, so the next call \
         starts at 201. A range wider than 200 lines is served from start_line and \
         the header says what is left. There is no need to find a file's length \
         first: read from line 1 and the header reports it. To find the line \
         worth reading, use file_grep rather than paging a large file to look \
         for it. For remote filesystems use remote_fs_session_get to download \
         first, then file_read.";

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

        // Both bounds clamp into the file. `start` past the end reads the last
        // line rather than returning nothing a model would read as "empty", and
        // `end` below `start` collapses to a one-line read rather than an error
        // — a transposed pair costs a narrow excerpt, not a wasted round trip.
        let start = req.start_line.clamp(1, total);
        let end = req.end_line.clamp(start, total);

        // The span cap, applied last so it bounds what the clamps produced.
        // Served rather than refused: the model asked for a region and gets its
        // first MAX_READ_LINES lines, and because `end` now falls short of
        // `total` the header renders `(lines 1-200 of 2499)` — which states both
        // that the excerpt was cut and where to resume. An `invalid_arguments`
        // here would spend a whole turn saying the same thing.
        let end = end.min(start.saturating_add(MAX_READ_LINES - 1));

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
