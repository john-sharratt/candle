//! code_session_exec tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::engine::run_js;
use super::{now, CodeError};
use crate::{RegisteredTool, Tool, ToolContext};

/// Most successful snippets one session holds. Every exec replays all of them,
/// each within its own file-call budget, so this bounds the work one exec can
/// replay — 64 snippets is a long interactive session.
pub const MAX_SESSION_SNIPPETS: usize = 64;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionExecReq {
    /// The session id returned by the code_session_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// JavaScript to execute in the session. Definitions persist across calls.
    #[validate(length(min = 1))]
    pub code: String,
    /// Advisory wall-clock hint. Runaway scripts are bounded by the VM's
    /// loop/recursion limits rather than a wall-clock timeout.
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct SessionExecResp {
    pub stdout: String,
    pub stderr: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The final expression's value, stringified. Absent when it was `undefined`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

pub struct CodeSessionExec;

impl Tool for CodeSessionExec {
    const NAME: &'static str = "code_session_exec";
    const DESCRIPTION: &'static str =
        "Execute JavaScript in a persistent session. Variable/function/const definitions from \
         earlier calls remain in scope. Returns stdout, stderr, an ok flag, the final \
         expression's value in result, and error details when ok=false.";
    type Request = SessionExecReq;
    type Response = SessionExecResp;
    type Error = CodeError;

    fn run(ctx: &ToolContext, req: SessionExecReq) -> Result<SessionExecResp, CodeError> {
        let entry = ctx
            .sessions
            .get_code(&req.session_id)
            .ok_or_else(|| CodeError::SessionNotFound(req.session_id.clone()))?;
        let mut guard = entry.lock().unwrap();
        guard.meta.last_activity = now();
        if guard.history.len() >= MAX_SESSION_SNIPPETS {
            return Err(CodeError::SessionFull(
                req.session_id.clone(),
                MAX_SESSION_SNIPPETS,
            ));
        }

        // Replay accumulated history (silently) to rebuild state, then run the
        // new snippet.
        let outcome = run_js(ctx.grants(), &ctx.vfs, &guard.history, &req.code)?;
        let ok = outcome.error.is_none();

        // Only successful snippets join the history — a throwing snippet must not
        // poison every future replay.
        if ok {
            guard.history.push(req.code);
        }

        Ok(SessionExecResp {
            stdout: outcome.stdout,
            stderr: outcome.stderr,
            ok,
            error: outcome.error,
            result: outcome.result,
        })
    }
}

pub const CODE_SESSION_EXEC: RegisteredTool = RegisteredTool::new::<CodeSessionExec>();
