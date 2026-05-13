//! remote_fs_session_delete tool.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::RemoteFsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct DeleteRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub path: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub path: String,
    pub deleted: bool,
}

pub struct RemoteFsSessionDelete;

impl Tool for RemoteFsSessionDelete {
    const NAME: &'static str = "remote_fs_session_delete";
    const DESCRIPTION: &'static str = "Delete a file on the remote SFTP filesystem.";
    type Request = DeleteRequest;
    type Response = DeleteResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: DeleteRequest) -> Result<DeleteResponse, RemoteFsError> {
        let entry = ctx
            .sessions
            .get_remote_fs(&req.session_id)
            .ok_or_else(|| RemoteFsError::SessionNotFound(req.session_id.clone()))?;
        let guard = entry.lock().unwrap();
        let sftp = guard
            .conn
            .ssh
            .session
            .sftp()
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        sftp.unlink(Path::new(&req.path))
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        Ok(DeleteResponse {
            path: req.path,
            deleted: true,
        })
    }
}

pub const REMOTE_FS_SESSION_DELETE: RegisteredTool =
    RegisteredTool::new::<RemoteFsSessionDelete>();
