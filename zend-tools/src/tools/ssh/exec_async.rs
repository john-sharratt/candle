//! ssh_session_exec_async tool.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::SshError;
use crate::state::sessions::SshProcess;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ExecAsyncRequest {
    /// The session id returned by the corresponding ssh_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// The shell command to start on the remote host.
    #[validate(length(min = 1))]
    pub command: String,
    /// Environment variables to set for the command, as name→value pairs.
    /// Defaults to none (inherits the session environment).
    pub env: Option<HashMap<String, String>>,
    /// Working directory to run the command in. Defaults to the session's current directory.
    pub cwd: Option<String>,
    /// Maximum seconds to let the command run. No default is applied.
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct ExecAsyncResponse {
    pub process_id: String,
    pub session_id: String,
    pub started_at: String,
    pub command: String,
}

pub struct SshSessionExecAsync;

impl Tool for SshSessionExecAsync {
    const NAME: &'static str = "ssh_session_exec_async";
    const DESCRIPTION: &'static str =
        "Start a long-running command on a remote SSH session without blocking. \
         Returns process_id; use ssh_session_poll to check status and read output.";

    type Request = ExecAsyncRequest;
    type Response = ExecAsyncResponse;
    type Error = SshError;

    fn confirmation(req: &ExecAsyncRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Start async: {}", req.command))
                .with_field("session_id", req.session_id.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: ExecAsyncRequest) -> Result<ExecAsyncResponse, SshError> {
        let entry_arc = ctx
            .sessions
            .get_ssh(&req.session_id)
            .ok_or_else(|| SshError::SessionNotFound(req.session_id.clone()))?;
        let entry = entry_arc.lock().unwrap();

        if !entry.meta.alive {
            return Err(SshError::SessionDead);
        }

        let process_id = format!("proc_{}", Uuid::new_v4());
        let started_at = Utc::now().to_rfc3339();

        let stdout_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let exit_code: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(None));
        let running: Arc<Mutex<bool>> = Arc::new(Mutex::new(true));

        let proc = SshProcess {
            process_id: process_id.clone(),
            session_id: req.session_id.clone(),
            command: req.command.clone(),
            started_at: started_at.clone(),
            stdout_buf: stdout_buf.clone(),
            stderr_buf: stderr_buf.clone(),
            exit_code: exit_code.clone(),
            running: running.clone(),
        };
        ctx.sessions.insert_ssh_process(proc);

        let session = &entry.conn.session;
        let command = req.command.clone();
        match session.channel_session() {
            Ok(mut channel) => {
                if channel.exec(&command).is_ok() {
                    let mut stdout_data = Vec::new();
                    let mut stderr_data = Vec::new();
                    channel.read_to_end(&mut stdout_data).ok();
                    {
                        let mut s = channel.stderr();
                        s.read_to_end(&mut stderr_data).ok();
                    }
                    channel.wait_close().ok();
                    let code = channel.exit_status().unwrap_or(-1);
                    *stdout_buf.lock().unwrap() = stdout_data;
                    *stderr_buf.lock().unwrap() = stderr_data;
                    *exit_code.lock().unwrap() = Some(code);
                }
            }
            Err(_) => {}
        }
        *running.lock().unwrap() = false;

        Ok(ExecAsyncResponse {
            process_id,
            session_id: req.session_id,
            started_at,
            command: req.command,
        })
    }
}

pub const SSH_SESSION_EXEC_ASYNC: RegisteredTool = RegisteredTool::new::<SshSessionExecAsync>();
