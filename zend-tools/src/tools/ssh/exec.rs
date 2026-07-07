//! ssh_session_exec tool.

use std::io::Read;
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{exec_simple, SshError, MAX_OUTPUT};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ExecRequest {
    /// The session id returned by the corresponding ssh_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// The shell command to run on the remote host.
    #[validate(length(min = 1))]
    pub command: String,
    /// Maximum seconds to wait for the command. No default is applied.
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct ExecResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub cwd_after: String,
    pub duration_ms: u64,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
}

pub struct SshSessionExec;

impl Tool for SshSessionExec {
    const NAME: &'static str = "ssh_session_exec";
    const DESCRIPTION: &'static str =
        "Run a command on a remote server through an open SSH session. \
         Returns stdout, stderr, exit code, working directory, and duration. \
         Use ssh_open first. Every command requires confirmation.";

    type Request = ExecRequest;
    type Response = ExecResponse;
    type Error = SshError;

    fn confirmation(req: &ExecRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Execute: {}", req.command))
                .with_field("session_id", req.session_id.clone())
                .with_field("command", req.command.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: ExecRequest) -> Result<ExecResponse, SshError> {
        let entry_arc = ctx
            .sessions
            .get_ssh(&req.session_id)
            .ok_or_else(|| SshError::SessionNotFound(req.session_id.clone()))?;
        let entry = entry_arc.lock().unwrap();

        if !entry.meta.alive {
            return Err(SshError::SessionDead);
        }

        let start = Instant::now();
        let session = &entry.conn.session;

        let mut channel = session
            .channel_session()
            .map_err(|e| SshError::ConnectionFailed(e.to_string()))?;
        channel
            .exec(&req.command)
            .map_err(|e| SshError::ConnectionFailed(e.to_string()))?;

        let mut stdout_bytes = Vec::new();
        let mut stderr_bytes = Vec::new();
        channel.read_to_end(&mut stdout_bytes).ok();
        {
            let mut stderr_stream = channel.stderr();
            stderr_stream.read_to_end(&mut stderr_bytes).ok();
        }
        channel.wait_close().ok();
        let exit_code = channel.exit_status().unwrap_or(-1);
        let duration_ms = start.elapsed().as_millis() as u64;

        let stdout_truncated = stdout_bytes.len() > MAX_OUTPUT;
        let stderr_truncated = stderr_bytes.len() > MAX_OUTPUT;
        stdout_bytes.truncate(MAX_OUTPUT);
        stderr_bytes.truncate(MAX_OUTPUT);

        let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
        let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();

        let cwd_after = exec_simple(session, "pwd")
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        Ok(ExecResponse {
            stdout,
            stderr,
            exit_code,
            cwd_after,
            duration_ms,
            stdout_truncated,
            stderr_truncated,
        })
    }
}

pub const SSH_SESSION_EXEC: RegisteredTool = RegisteredTool::new::<SshSessionExec>();
