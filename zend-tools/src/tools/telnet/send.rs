//! telnet_send tool.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TelnetError;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SendRequest {
    /// The session id returned by the corresponding telnet_session_open tool.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// Text to send. A trailing "\r\n" is appended automatically if absent.
    pub send: String,
    /// Regex to wait for. Defaults to the session's prompt_pattern.
    pub expect: Option<String>,
    /// Maximum seconds to wait for the expect pattern (0–300). Defaults to 30.
    #[validate(range(max = 300))]
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct SendResponse {
    pub received: String,
    pub matched: bool,
    pub duration_ms: u64,
    pub received_truncated: bool,
}

pub struct TelnetSessionSend;

impl Tool for TelnetSessionSend {
    const NAME: &'static str = "telnet_send";
    const DESCRIPTION: &'static str =
        "Send text to an open Telnet session and read until an expect regex matches (the device \
         prompt, by default). Use for: running commands on network gear, querying device status, \
         configuring routers/switches, navigating menu-driven legacy interfaces. Triggered by \
         \"run X on the switch\", \"configure the router\", \"show running-config\", \"send this \
         to the device\". Returns received text up to the matched pattern, a matched flag, and \
         duration. Output capped at 32 KiB. Every send requires confirmation. Use \
         ssh_session_exec instead for SSH-capable hosts — SSH provides exit codes and encryption.";

    type Request = SendRequest;
    type Response = SendResponse;
    type Error = TelnetError;

    fn confirmation(req: &SendRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Send to Telnet: {:?}", req.send))
                .with_field("session_id", req.session_id.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: SendRequest) -> Result<SendResponse, TelnetError> {
        let entry_arc = ctx
            .sessions
            .get_telnet(&req.session_id)
            .ok_or_else(|| TelnetError::SessionNotFound(req.session_id.clone()))?;
        let mut entry = entry_arc.lock().unwrap();

        let timeout_secs = req.timeout_sec.unwrap_or(30);
        let timeout = Duration::from_secs(timeout_secs as u64);
        entry
            .stream
            .set_read_timeout(Some(Duration::from_millis(200)))
            .ok();

        // Append \r\n unless already present
        let line = if req.send.ends_with("\r\n") || req.send.ends_with('\n') {
            req.send.clone()
        } else {
            format!("{}\r\n", req.send)
        };
        entry
            .stream
            .write_all(line.as_bytes())
            .map_err(|e| TelnetError::SendFailed(e.to_string()))?;

        let expect_pattern = req
            .expect
            .as_deref()
            .unwrap_or(&entry.prompt_pattern)
            .to_string();
        let prompt_re = regex::Regex::new(&expect_pattern)
            .unwrap_or_else(|_| regex::Regex::new(r"[#$>]\s*$").unwrap());

        const MAX_RECV: usize = 32 * 1024;
        let mut response: Vec<u8> = Vec::new();
        let mut buf = [0u8; 4096];
        let start = Instant::now();

        loop {
            if start.elapsed() >= timeout {
                break;
            }
            match entry.stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let remaining = MAX_RECV.saturating_sub(response.len());
                    response.extend_from_slice(&buf[..n.min(remaining)]);
                    let text = String::from_utf8_lossy(&response);
                    if prompt_re.is_match(&text) {
                        break;
                    }
                    if response.len() >= MAX_RECV {
                        break;
                    }
                }
                Err(e)
                    if e.kind() == std::io::ErrorKind::TimedOut
                        || e.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    // poll again unless outer timeout exceeded
                }
                Err(_) => break,
            }
        }

        let truncated = response.len() >= MAX_RECV;
        let duration_ms = start.elapsed().as_millis() as u64;
        let text = String::from_utf8_lossy(&response).into_owned();
        let matched = prompt_re.is_match(&text);

        Ok(SendResponse {
            received: text,
            matched,
            duration_ms,
            received_truncated: truncated,
        })
    }
}

pub const TELNET_SESSION_SEND: RegisteredTool = RegisteredTool::new::<TelnetSessionSend>();
