//! remote_fs_session_stat tool.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::RemoteFsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct StatRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub path: String,
}

#[derive(Serialize)]
pub struct StatResponse {
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
    pub permissions: Option<u32>,
    pub modified: Option<String>,
}

pub struct RemoteFsSessionStat;

impl Tool for RemoteFsSessionStat {
    const NAME: &'static str = "remote_fs_session_stat";
    const DESCRIPTION: &'static str = "Stat a file or directory on a remote SFTP filesystem.";
    type Request = StatRequest;
    type Response = StatResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: StatRequest) -> Result<StatResponse, RemoteFsError> {
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
        let stat = sftp
            .stat(Path::new(&req.path))
            .map_err(|e| RemoteFsError::NotFound(e.to_string()))?;
        Ok(StatResponse {
            path: req.path,
            is_dir: stat.is_dir(),
            size: stat.size.unwrap_or(0),
            permissions: stat.perm,
            modified: stat.mtime.and_then(|t| {
                chrono::DateTime::from_timestamp(t as i64, 0).map(|dt| dt.to_rfc3339())
            }),
        })
    }
}

pub const REMOTE_FS_SESSION_STAT: RegisteredTool = RegisteredTool::new::<RemoteFsSessionStat>();
