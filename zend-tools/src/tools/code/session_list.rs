//! code_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::CodeError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionListReq {}

#[derive(Serialize)]
pub struct SessionListResp {
    pub sessions: Vec<serde_json::Value>,
}

pub struct CodeSessionList;

impl Tool for CodeSessionList {
    const NAME: &'static str = "code_list";
    const DESCRIPTION: &'static str =
        "List open code execution sandbox sessions for the current conversation. Use for: \
         checking which sandboxes are currently running, finding a session_id before \
         issuing code_session_exec, seeing which languages are active, identifying stale \
         sessions to clean up. Triggered by \"what code sessions do I have\", \"list active \
         sandboxes\", \"show running execution environments\". Returns session_id, language, \
         opened_at, last_activity, and alive flag for each session.";
    type Request = SessionListReq;
    type Response = SessionListResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, _req: SessionListReq) -> Result<SessionListResp, CodeError> {
        let sessions = ctx
            .sessions
            .list_code()
            .into_iter()
            .map(|e| {
                let g = e.lock().unwrap();
                serde_json::json!({
                    "session_id": g.meta.session_id,
                    "language": g.language,
                    "opened_at": g.meta.opened_at,
                    "last_activity": g.meta.last_activity,
                    "alive": g.meta.alive,
                })
            })
            .collect();
        Ok(SessionListResp { sessions })
    }
}

pub const CODE_SESSION_LIST: RegisteredTool = RegisteredTool::new::<CodeSessionList>();
