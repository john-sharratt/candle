//! ssh_session_poll tool.

use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{SshError, MAX_OUTPUT};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PollRequest {
    /// The async process handle returned by ssh_session_exec_async (a `proc_…`
    /// id). This is NOT an ssh session id — poll the job, not the session.
    #[validate(length(min = 1))]
    pub process_id: String,
    /// Seconds to wait for the process to finish before returning. Defaults to 0
    /// (return immediately with whatever output has accumulated).
    pub recv_wait_sec: Option<f64>,
    /// Signal to deliver if the process is still running: "TERM", "KILL", or "INT".
    pub signal: Option<String>,
    /// How to return output: "auto" (default — text if valid UTF-8, else hex),
    /// "text", or "hex".
    pub format: Option<String>,
}

/// Encode an output chunk per the requested format.
fn encode_chunk(bytes: &[u8], format: &str) -> String {
    match format {
        "hex" => hex::encode(bytes),
        "text" => String::from_utf8_lossy(bytes).into_owned(),
        // auto: text when valid UTF-8, else hex.
        _ => match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => hex::encode(bytes),
        },
    }
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
    const DESCRIPTION: &'static str = "Poll a running async SSH process for output. \
         Returns stdout/stderr chunks, running status, and exit code when done.";

    type Request = PollRequest;
    type Response = PollResponse;
    type Error = SshError;

    fn run(ctx: &ToolContext, req: PollRequest) -> Result<PollResponse, SshError> {
        let proc_arc = ctx
            .sessions
            .get_ssh_process(&req.process_id)
            .ok_or_else(|| SshError::ProcessNotFound(req.process_id.clone()))?;
        let proc = proc_arc.lock().unwrap();

        // Honor recv_wait_sec: poll for the process to finish, up to the budget.
        if let Some(wait) = req.recv_wait_sec {
            if wait > 0.0 {
                let step = Duration::from_millis(20);
                let budget = Duration::from_secs_f64(wait);
                let mut waited = Duration::ZERO;
                while *proc.running.lock().unwrap() && waited < budget {
                    std::thread::sleep(step);
                    waited += step;
                }
            }
        }

        let running = *proc.running.lock().unwrap();
        let exit_code = *proc.exit_code.lock().unwrap();

        // A signal can only be delivered to a process that is still running.
        let signal_sent = req.signal.is_some() && running;

        // Elapsed wall-clock since the process started.
        let duration_so_far_ms = chrono::DateTime::parse_from_rfc3339(&proc.started_at)
            .map(|t| {
                (chrono::Utc::now() - t.with_timezone(&chrono::Utc))
                    .num_milliseconds()
                    .max(0) as u64
            })
            .unwrap_or(0);

        let stdout_all = proc.stdout_buf.lock().unwrap().clone();
        let stderr_all = proc.stderr_buf.lock().unwrap().clone();

        let stdout_total = stdout_all.len();
        let stderr_total = stderr_all.len();
        let stdout_truncated = stdout_all.len() > MAX_OUTPUT;
        let stderr_truncated = stderr_all.len() > MAX_OUTPUT;

        let format = req.format.as_deref().unwrap_or("auto");
        let stdout_chunk = encode_chunk(&stdout_all[..stdout_all.len().min(MAX_OUTPUT)], format);
        let stderr_chunk = encode_chunk(&stderr_all[..stderr_all.len().min(MAX_OUTPUT)], format);

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
            duration_so_far_ms,
            signal_sent,
        })
    }
}

pub const SSH_SESSION_POLL: RegisteredTool = RegisteredTool::new::<SshSessionPoll>();
