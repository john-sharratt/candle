//! code_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::CodeError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionCloseReq {
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
        "Close a code execution sandbox session, terminating the interpreter subprocess and \
         freeing its resources. Use when finished with a coding workflow, when the interpreter \
         state has become corrupted and needs a fresh start, or when freeing up session slots. \
         Triggered by \"close the code session\", \"end the sandbox\", \"terminate the \
         interpreter\". Idempotent — closing an already-closed session returns success.";
    type Request = SessionCloseReq;
    type Response = SessionCloseResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, req: SessionCloseReq) -> Result<SessionCloseResp, CodeError> {
        if let Some(entry) = ctx.sessions.get_code(&req.session_id) {
            let mut guard = entry.lock().unwrap();
            let _ = guard.child.kill();
        }
        let removed = ctx.sessions.remove_code(&req.session_id);
        Ok(SessionCloseResp {
            session_id: req.session_id,
            closed: removed,
        })
    }
}

pub const CODE_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<CodeSessionClose>();
