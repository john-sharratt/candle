//! udp_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::UdpError;

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

pub struct UdpSessionClose;

impl Tool for UdpSessionClose {
    const NAME: &'static str = "udp_session_close";
    const DESCRIPTION: &'static str = "Release a UDP datagram socket and free its bound port. Connectionless — no peer is notified.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = UdpError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, UdpError> {
        let closed = ctx.sessions.remove_udp(&req.session_id);
        Ok(CloseResponse { session_id: req.session_id, closed })
    }
}

pub const UDP_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<UdpSessionClose>();
