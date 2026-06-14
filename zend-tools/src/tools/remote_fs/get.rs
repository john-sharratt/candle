//! remote_fs_session_get tool.

use std::io::Read as _;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::RemoteFsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct GetRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub remote_path: String,
    pub local_vfs_path: Option<String>,
}

#[derive(Serialize)]
pub struct GetResponse {
    pub remote_path: String,
    pub local_vfs_path: String,
    pub bytes: usize,
}

pub struct RemoteFsSessionGet;

impl Tool for RemoteFsSessionGet {
    const NAME: &'static str = "remote_fs_session_get";
    const DESCRIPTION: &'static str =
        "Download a file from the remote SFTP filesystem into the session VFS. \
         Returns the VFS path where content was written.";
    type Request = GetRequest;
    type Response = GetResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: GetRequest) -> Result<GetResponse, RemoteFsError> {
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
            .open(Path::new(&req.remote_path))
            .map_err(|e| RemoteFsError::NotFound(e.to_string()))?;
        let mut contents = String::new();
        remote_file
            .read_to_string(&mut contents)
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        let bytes = contents.len();
        let vfs_path = req.local_vfs_path.clone().unwrap_or_else(|| {
            Path::new(&req.remote_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "downloaded_file".into())
        });
        ctx.vfs
            .write(&vfs_path, contents)
            .map_err(|e| RemoteFsError::VfsError(format!("{e:?}")))?;
        Ok(GetResponse {
            remote_path: req.remote_path,
            local_vfs_path: vfs_path,
            bytes,
        })
    }
}

pub const REMOTE_FS_SESSION_GET: RegisteredTool = RegisteredTool::new::<RemoteFsSessionGet>();
