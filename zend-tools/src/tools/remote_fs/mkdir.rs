//! remote_fs_session_mkdir tool.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::RemoteFsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct MkdirRequest {
    /// The session id returned by the remote_fs_session_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// Remote directory path to create.
    #[validate(length(min = 1))]
    pub path: String,
    /// Unix permission bits for the new directory. Defaults to 0o755.
    pub mode: Option<u32>,
}

#[derive(Serialize)]
pub struct MkdirResponse {
    pub path: String,
    pub created: bool,
}

pub struct RemoteFsSessionMkdir;

impl Tool for RemoteFsSessionMkdir {
    const NAME: &'static str = "remote_fs_session_mkdir";
    const DESCRIPTION: &'static str =
        "Create a new directory on a remote host's filesystem over SFTP.";
    type Request = MkdirRequest;
    type Response = MkdirResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: MkdirRequest) -> Result<MkdirResponse, RemoteFsError> {
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
        let mode = req.mode.unwrap_or(0o755);
        sftp.mkdir(Path::new(&req.path), mode as i32)
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        Ok(MkdirResponse {
            path: req.path,
            created: true,
        })
    }
}

pub const REMOTE_FS_SESSION_MKDIR: RegisteredTool = RegisteredTool::new::<RemoteFsSessionMkdir>();
