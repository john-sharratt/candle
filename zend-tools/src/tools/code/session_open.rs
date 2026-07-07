//! code_session_open tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::{is_javascript, now, CodeError};
use crate::state::sessions::{CodeEntry, SessionMeta};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionOpenReq {
    /// Language for the session. Only JavaScript is supported (aliases:
    /// javascript, js, node).
    #[validate(length(min = 1))]
    pub language: String,
}

#[derive(Serialize)]
pub struct SessionOpenResp {
    pub session_id: String,
    pub language: String,
}

pub struct CodeSessionOpen;

impl Tool for CodeSessionOpen {
    const NAME: &'static str = "code_session_open";
    const DESCRIPTION: &'static str =
        "Open a persistent JavaScript session. Runs in-process on an embedded, sandboxed \
         pure-Rust engine — no Node or external interpreter required. Variable, function, and \
         const definitions persist across code_session_exec calls. Close with \
         code_session_close when done.";
    type Request = SessionOpenReq;
    type Response = SessionOpenResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, req: SessionOpenReq) -> Result<SessionOpenResp, CodeError> {
        if !is_javascript(&req.language) {
            return Err(CodeError::InterpreterNotFound(req.language));
        }

        let sid = format!("sess_{}", Uuid::new_v4().simple());
        ctx.sessions.insert_code(CodeEntry {
            meta: SessionMeta {
                session_id: sid.clone(),
                opened_at: now(),
                last_activity: now(),
                alive: true,
            },
            language: "javascript".to_string(),
            history: String::new(),
        });

        Ok(SessionOpenResp {
            session_id: sid,
            language: "javascript".to_string(),
        })
    }
}

pub const CODE_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<CodeSessionOpen>();
