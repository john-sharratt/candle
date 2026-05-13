//! file_delete tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::FileError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct DeleteRequest {
    #[validate(length(min = 1))]
    pub path: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub path: String,
    pub deleted: bool,
}

pub struct FileDelete;

impl Tool for FileDelete {
    const NAME: &'static str = "file_delete";
    const DESCRIPTION: &'static str =
        "Delete a file from the session virtual filesystem (VFS). Use for: removing a draft \
         that's no longer needed, cleaning up before exporting, getting rid of an uploaded \
         file the user wants gone, freeing space within the 10 MiB VFS budget. Triggered by \
         \"delete the file\", \"remove\", \"rm\", \"get rid of the file called\". Returns the \
         path and a deleted flag. VFS contents disappear at session end anyway — explicit \
         deletion is for in-session cleanup. For removing files on remote systems use \
         remote_fs_session_delete.";

    type Request = DeleteRequest;
    type Response = DeleteResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: DeleteRequest) -> Result<DeleteResponse, FileError> {
        let deleted = ctx.vfs.delete(&req.path);
        if !deleted {
            return Err(FileError::NotFound(req.path));
        }
        Ok(DeleteResponse { path: req.path, deleted: true })
    }
}

pub const FILE_DELETE: RegisteredTool = RegisteredTool::new::<FileDelete>();
