//! tls_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TlsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CloseRequest {
    /// The session id returned by tls_session_open.
    #[validate(length(min = 1))]
    pub session_id: String,
}

#[derive(Serialize)]
pub struct CloseResponse {
    pub session_id: String,
    pub closed: bool,
}

pub struct TlsSessionClose;

impl Tool for TlsSessionClose {
    const NAME: &'static str = "tls_session_close";
    const DESCRIPTION: &'static str = "Shut a TLS-encrypted connection, sending close-notify and freeing the certificate-verified channel.";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = TlsError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, TlsError> {
        let closed = ctx.sessions.remove_tls(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed,
        })
    }
}

pub const TLS_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<TlsSessionClose>();
