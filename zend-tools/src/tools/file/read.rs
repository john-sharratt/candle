//! file_read tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// VFS path of the file to read (session-relative, e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
}

#[derive(Serialize)]
pub struct ReadResponse {
    pub path: String,
    pub content: String,
    pub lines: usize,
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file's content from the session virtual filesystem (VFS). Use for: looking at \
         what was previously written, inspecting a file the user uploaded into the chat, \
         retrieving content the model needs to reference for editing or summarising, checking \
         the current state of a draft after edits. Triggered by \"show me the file\", \"read\", \
         \"what's in\", \"open the file\", \"cat\", \"display the contents of\". Returns path, \
         full content, and line count. For remote filesystems use remote_fs_session_get to \
         download first, then file_read.";

    type Request = ReadRequest;
    type Response = ReadResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<ReadResponse, FileError> {
        let content = ctx
            .vfs
            .read(&req.path)
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        let lines = content.lines().count();
        Ok(ReadResponse {
            path: req.path,
            content,
            lines,
        })
    }
}

pub const FILE_READ: RegisteredTool = RegisteredTool::new::<FileRead>();
