//! remote_fs_session_rename tool.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::RemoteFsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RenameRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub from: String,
    #[validate(length(min = 1))]
    pub to: String,
}

#[derive(Serialize)]
pub struct RenameResponse {
    pub from: String,
    pub to: String,
}

pub struct RemoteFsSessionRename;

impl Tool for RemoteFsSessionRename {
    const NAME: &'static str = "remote_fs_session_rename";
    const DESCRIPTION: &'static str =
        "Rename or move a file or directory on a remote host's filesystem over SFTP.";
    type Request = RenameRequest;
    type Response = RenameResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: RenameRequest) -> Result<RenameResponse, RemoteFsError> {
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
        sftp.rename(Path::new(&req.from), Path::new(&req.to), None)
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        Ok(RenameResponse {
            from: req.from,
            to: req.to,
        })
    }
}

pub const REMOTE_FS_SESSION_RENAME: RegisteredTool =
    RegisteredTool::new::<RemoteFsSessionRename>();
