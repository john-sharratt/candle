//! http_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::HttpSessionError;
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

pub struct HttpSessionClose;

impl Tool for HttpSessionClose {
    const NAME: &'static str = "http_session_close";
    const DESCRIPTION: &'static str =
        "Release the cookie jar and connection pool for an HTTP session. Idempotent. Use when \
         finished with an API workflow or before starting a fresh session with different auth. \
         Triggered by \"close the HTTP session\", \"disconnect from the API\", \"end the \
         session\", \"log out of the API\".";

    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = HttpSessionError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, HttpSessionError> {
        let closed = ctx.sessions.remove_http(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed,
        })
    }
}

pub const HTTP_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<HttpSessionClose>();
