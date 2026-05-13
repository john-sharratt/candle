//! code_session_exec tool.

use std::io::{BufRead as _, Write as _};
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{now, CodeError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SessionExecReq {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub code: String,
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct SessionExecResp {
    pub stdout: String,
    pub stderr: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub struct CodeSessionExec;

impl Tool for CodeSessionExec {
    const NAME: &'static str = "code_session_exec";
    const DESCRIPTION: &'static str =
        "Execute code in a persistent session. State is shared between calls. \
         Returns stdout, stderr, and ok flag. Error details in the error field if ok=false.";
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

        let timeout = req.timeout_sec.unwrap_or(30) as u64;
        let deadline = Instant::now() + std::time::Duration::from_secs(timeout);

        let lang = guard.language.clone();
        let is_repl = matches!(
            lang.as_str(),
            "python" | "python3" | "javascript" | "js" | "node"
        );

        if is_repl {
            let code_bytes = req.code.as_bytes();
            let header = format!("{}\n", code_bytes.len());
            guard
                .stdin
                .write_all(header.as_bytes())
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            guard
                .stdin
                .write_all(code_bytes)
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            guard
                .stdin
                .flush()
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;

            let sentinel = "__ZEND_DONE__";
            let mut json_line = String::new();
            loop {
                if Instant::now() > deadline {
                    return Err(CodeError::Timeout);
                }
                let mut line = String::new();
                guard
                    .stdout_reader
                    .read_line(&mut line)
                    .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
                if line.is_empty() {
                    return Err(CodeError::ExecutionFailed("process exited unexpectedly".into()));
                }
                let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
                if trimmed == sentinel {
                    break;
                }
                json_line = trimmed.to_string();
            }

            let parsed: serde_json::Value = serde_json::from_str(&json_line)
                .unwrap_or(serde_json::json!({"ok": false, "error": "invalid response", "stdout": "", "stderr": ""}));
            Ok(SessionExecResp {
                stdout: parsed["stdout"].as_str().unwrap_or("").to_string(),
                stderr: parsed["stderr"].as_str().unwrap_or("").to_string(),
                ok: parsed["ok"].as_bool().unwrap_or(false),
                error: parsed["error"].as_str().map(|s| s.to_string()),
            })
        } else {
            let exit_sentinel = "__ZEND_EXIT__";
            let payload = format!("{}\necho '{}:'$?\n", req.code, exit_sentinel);
            guard
                .stdin
                .write_all(payload.as_bytes())
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            guard
                .stdin
                .flush()
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;

            let mut stdout_lines: Vec<String> = Vec::new();
            #[allow(unused_assignments)]
            let mut exit_code: Option<i32> = None;
            loop {
                if Instant::now() > deadline {
                    return Err(CodeError::Timeout);
                }
                let mut line = String::new();
                guard
                    .stdout_reader
                    .read_line(&mut line)
                    .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
                if line.is_empty() {
                    return Err(CodeError::ExecutionFailed("process exited unexpectedly".into()));
                }
                let trimmed = line.trim_end_matches('\n').trim_end_matches('\r').to_string();
                if let Some(rest) = trimmed.strip_prefix(&format!("{exit_sentinel}:")) {
                    exit_code = rest.parse::<i32>().ok();
                    break;
                }
                stdout_lines.push(trimmed);
            }

            let ok = exit_code == Some(0);
            let stdout = stdout_lines.join("\n");
            Ok(SessionExecResp {
                stdout,
                stderr: String::new(),
                ok,
                error: exit_code
                    .filter(|&c| c != 0)
                    .map(|c| format!("exited with code {c}")),
            })
        }
    }
}

pub const CODE_SESSION_EXEC: RegisteredTool = RegisteredTool::new::<CodeSessionExec>();
