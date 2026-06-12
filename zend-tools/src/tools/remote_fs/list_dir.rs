//! remote_fs_session_list_dir tool.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::RemoteFsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListDirRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    pub path: Option<String>,
}

#[derive(Serialize)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub permissions: Option<u32>,
    pub modified: Option<String>,
}

#[derive(Serialize)]
pub struct ListDirResponse {
    pub path: String,
    pub entries: Vec<DirEntry>,
}

pub struct RemoteFsSessionListDir;

impl Tool for RemoteFsSessionListDir {
    const NAME: &'static str = "remote_fs_session_list_dir";
    const DESCRIPTION: &'static str =
        "List entries in a remote directory via SFTP. Returns name, size, is_dir, permissions, \
         and modification time for each entry.";
    type Request = ListDirRequest;
    type Response = ListDirResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: ListDirRequest) -> Result<ListDirResponse, RemoteFsError> {
        let entry = ctx
            .sessions
            .get_remote_fs(&req.session_id)
            .ok_or_else(|| RemoteFsError::SessionNotFound(req.session_id.clone()))?;
        let guard = entry.lock().unwrap();
        let path_str = req
            .path
            .as_deref()
            .unwrap_or(&guard.remote_prefix)
            .to_string();
        let sftp = guard
            .conn
            .ssh
            .session
            .sftp()
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        let entries = sftp
            .readdir(Path::new(&path_str))
            .map_err(|e| RemoteFsError::SftpError(e.to_string()))?;
        let result = entries
            .into_iter()
            .map(|(pb, stat)| {
                let name = pb
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                DirEntry {
                    name,
                    is_dir: stat.is_dir(),
                    size: stat.size.unwrap_or(0),
                    permissions: stat.perm,
                    modified: stat.mtime.and_then(|t| {
                        chrono::DateTime::from_timestamp(t as i64, 0).map(|dt| dt.to_rfc3339())
                    }),
                }
            })
            .collect();
        Ok(ListDirResponse {
            path: path_str,
            entries: result,
        })
    }
}

pub const REMOTE_FS_SESSION_LIST_DIR: RegisteredTool =
    RegisteredTool::new::<RemoteFsSessionListDir>();
