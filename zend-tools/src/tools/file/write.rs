//! write tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct WriteRequest {
    /// Path to create or overwrite (e.g. `src/main.rs`). The result is held in this session; a project file of the same path is shadowed, never modified on disk. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Full file content to write, replacing any existing content. Required.
    pub content: String,
}

#[derive(Serialize)]
pub struct WriteResponse {
    pub path: String,
    pub bytes: usize,
    pub created: bool,
}

pub struct FileWrite;

impl Tool for FileWrite {
    const NAME: &'static str = "write";
    const DESCRIPTION: &'static str =
        "Create a new file or overwrite an existing one in the in-memory virtual filesystem \
         (VFS) for this session. Use for: drafting code, writing notes, saving intermediate \
         output the model wants to reference later, creating files the user will download or \
         transfer, replacing a file's full content. Triggered by \"create a file\", \"save \
         this as\", \"write to\", \"put this in a file called\", \"make a file with\". Returns \
         path, byte count, and whether the file was newly created vs overwritten. For partial \
         edits use file_edit; for pushing files to a real remote system use \
         remote_fs_session_put.";

    type Request = WriteRequest;
    type Response = WriteResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: WriteRequest) -> Result<WriteResponse, FileError> {
        let bytes = req.content.len();
        let created = ctx.vfs.write(&req.path, req.content)?;
        Ok(WriteResponse {
            path: req.path,
            bytes,
            created,
        })
    }
}

pub const FILE_WRITE: RegisteredTool = RegisteredTool::new::<FileWrite>();
