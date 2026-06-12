//! tcp_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TcpError;
use crate::{RegisteredTool, Tool, ToolContext};

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

pub struct TcpSessionClose;

impl Tool for TcpSessionClose {
    const NAME: &'static str = "tcp_session_close";
    const DESCRIPTION: &'static str = "Close a raw TCP stream socket, dropping the connection and freeing its buffers. Use when a custom-protocol or non-HTTP byte-stream exchange is finished.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = TcpError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, TcpError> {
        let closed = ctx.sessions.remove_tcp(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed,
        })
    }
}

pub const TCP_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<TcpSessionClose>();
