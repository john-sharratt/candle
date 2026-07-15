//! code_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::CodeError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionCloseReq {
    /// The session id returned by the code_session_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
}

#[derive(Serialize)]
pub struct SessionCloseResp {
    pub session_id: String,
    pub closed: bool,
}

pub struct CodeSessionClose;

impl Tool for CodeSessionClose {
    const NAME: &'static str = "code_session_close";
    const DESCRIPTION: &'static str =
        "Close a JavaScript code session, discarding its accumulated state. Use when finished \
         with a coding workflow, when the session state has become corrupted and needs a fresh \
         start, or when freeing up session slots. Triggered by \"close the code session\", \
         \"end the sandbox\", \"reset the interpreter\". Idempotent — closing an already-closed \
         session returns success.";
    type Request = SessionCloseReq;
    type Response = SessionCloseResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, req: SessionCloseReq) -> Result<SessionCloseResp, CodeError> {
        let removed = ctx.sessions.remove_code(&req.session_id);
        Ok(SessionCloseResp {
            session_id: req.session_id,
            closed: removed,
        })
    }
}

pub const CODE_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<CodeSessionClose>();
