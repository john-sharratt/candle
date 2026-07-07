//! ssh_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SshError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CloseRequest {
    /// The session id returned by the corresponding ssh_open tool.
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
    const DESCRIPTION: &'static str = "End a secure shell session on a remote server, terminating the encrypted login and freeing its working state.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = SshError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, SshError> {
        let closed = ctx.sessions.remove_ssh(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed,
        })
    }
}

pub const SSH_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<SshSessionClose>();
