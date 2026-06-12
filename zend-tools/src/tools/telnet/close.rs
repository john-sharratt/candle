//! telnet_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TelnetError;
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

pub struct TelnetSessionClose;

impl Tool for TelnetSessionClose {
    const NAME: &'static str = "telnet_session_close";
    const DESCRIPTION: &'static str = "End a Telnet terminal session to a network device, releasing the plaintext terminal connection.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = TelnetError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, TelnetError> {
        let closed = ctx.sessions.remove_telnet(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed,
        })
    }
}

pub const TELNET_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<TelnetSessionClose>();
