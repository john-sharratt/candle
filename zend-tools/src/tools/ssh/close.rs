//! ssh_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::SshError;

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

pub struct SshSessionClose;

impl Tool for SshSessionClose {
    const NAME: &'static str = "ssh_session_close";
    const DESCRIPTION: &'static str = "Close an SSH session and free its resources.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = SshError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, SshError> {
        let closed = ctx.sessions.remove_ssh(&req.session_id);
        Ok(CloseResponse { session_id: req.session_id, closed })
    }
}

pub const SSH_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<SshSessionClose>();
