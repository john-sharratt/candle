//! remote_fs_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::RemoteFsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CloseRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
}

#[derive(Serialize)]
pub struct CloseResponse {
    pub session_id: String,
    pub closed: bool,
}

pub struct RemoteFsSessionClose;

impl Tool for RemoteFsSessionClose {
    const NAME: &'static str = "remote_fs_session_close";
    const DESCRIPTION: &'static str =
        "Close a remote filesystem session, tearing down the SFTP/SSH connection.";
    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, RemoteFsError> {
        let removed = ctx.sessions.remove_remote_fs(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed: removed,
        })
    }
}

pub const REMOTE_FS_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<RemoteFsSessionClose>();
