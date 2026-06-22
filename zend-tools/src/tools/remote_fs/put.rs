//! remote_fs_session_put tool.

use std::io::Write as _;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::RemoteFsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PutRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub local_vfs_path: String,
    #[validate(length(min = 1))]
    pub remote_path: String,
}

#[derive(Serialize)]
pub struct PutResponse {
    pub remote_path: String,
    pub bytes: usize,
}

pub struct RemoteFsSessionPut;

impl Tool for RemoteFsSessionPut {
    const NAME: &'static str = "remote_fs_session_put";
    const DESCRIPTION: &'static str =
        "Upload a file from the session VFS to the remote SFTP filesystem. \
         The file must exist in the VFS (use write first).";
    type Request = PutRequest;
    type Response = PutResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: PutRequest) -> Result<PutResponse, RemoteFsError> {
        let content = ctx
            .vfs
            .read(&req.local_vfs_path)
            .ok_or_else(|| RemoteFsError::VfsError(format!("{} not in VFS", req.local_vfs_path)))?;
        let bytes = content.len();
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
        let mut remote_file = sftp
            .create(Path::new(&req.remote_path))
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        remote_file
            .write_all(content.as_bytes())
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        Ok(PutResponse {
            remote_path: req.remote_path,
            bytes,
        })
    }
}

pub const REMOTE_FS_SESSION_PUT: RegisteredTool = RegisteredTool::new::<RemoteFsSessionPut>();
