//! remote_fs_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::RemoteFsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct FsListRequest {}

#[derive(Serialize)]
pub struct FsListResponse {
    pub sessions: Vec<serde_json::Value>,
}

pub struct RemoteFsSessionList;

impl Tool for RemoteFsSessionList {
    const NAME: &'static str = "remote_fs_session_list";
    const DESCRIPTION: &'static str = "List open remote filesystem sessions.";
    type Request = FsListRequest;
    type Response = FsListResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, _req: FsListRequest) -> Result<FsListResponse, RemoteFsError> {
        let sessions = ctx
            .sessions
            .list_remote_fs()
            .into_iter()
            .map(|e| {
                let g = e.lock().unwrap();
                serde_json::json!({
                    "session_id": g.meta.session_id,
                    "protocol": g.protocol,
                    "host": g.host,
                    "port": g.port,
                    "opened_at": g.meta.opened_at,
                    "alive": g.meta.alive,
                })
            })
            .collect();
        Ok(FsListResponse { sessions })
    }
}

pub const REMOTE_FS_SESSION_LIST: RegisteredTool = RegisteredTool::new::<RemoteFsSessionList>();
