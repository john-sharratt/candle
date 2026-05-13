//! ssh_session_poll tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{SshError, MAX_OUTPUT};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PollRequest {
    #[validate(length(min = 1))]
    pub process_id: String,
    pub recv_wait_sec: Option<f64>,
    pub signal: Option<String>,
    pub format: Option<String>,
}

#[derive(Serialize)]
pub struct PollResponse {
    pub process_id: String,
    pub running: bool,
    pub exit_code: Option<i32>,
    pub stdout_chunk: String,
    pub stderr_chunk: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub stdout_total_bytes: usize,
    pub stderr_total_bytes: usize,
    pub duration_so_far_ms: u64,
    pub signal_sent: bool,
}

pub struct SshSessionPoll;

impl Tool for SshSessionPoll {
    const NAME: &'static str = "ssh_session_poll";
    const DESCRIPTION: &'static str =
        "Poll a running async SSH process for output. \
         Returns stdout/stderr chunks, running status, and exit code when done.";

    type Request = PollRequest;
    type Response = PollResponse;
    type Error = SshError;

    fn run(ctx: &ToolContext, req: PollRequest) -> Result<PollResponse, SshError> {
        let proc_arc = ctx.sessions.get_ssh_process(&req.process_id)
            .ok_or_else(|| SshError::ProcessNotFound(req.process_id.clone()))?;
        let proc = proc_arc.lock().unwrap();

        let running = *proc.running.lock().unwrap();
        let exit_code = *proc.exit_code.lock().unwrap();

        let stdout_all = proc.stdout_buf.lock().unwrap().clone();
        let stderr_all = proc.stderr_buf.lock().unwrap().clone();

        let stdout_total = stdout_all.len();
        let stderr_total = stderr_all.len();
        let stdout_truncated = stdout_all.len() > MAX_OUTPUT;
        let stderr_truncated = stderr_all.len() > MAX_OUTPUT;

        let stdout_chunk = String::from_utf8_lossy(
            &stdout_all[..stdout_all.len().min(MAX_OUTPUT)]
        ).into_owned();
        let stderr_chunk = String::from_utf8_lossy(
            &stderr_all[..stderr_all.len().min(MAX_OUTPUT)]
        ).into_owned();

        Ok(PollResponse {
            process_id: req.process_id,
            running,
            exit_code,
            stdout_chunk,
            stderr_chunk,
            stdout_truncated,
            stderr_truncated,
            stdout_total_bytes: stdout_total,
            stderr_total_bytes: stderr_total,
            duration_so_far_ms: 0,
            signal_sent: false,
        })
    }
}

pub const SSH_SESSION_POLL: RegisteredTool = RegisteredTool::new::<SshSessionPoll>();
