//! tcp_session_recv tool.

use std::io::Read;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TcpError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RecvRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    /// recv_amt mode: read exactly this many bytes (blocks until satisfied or EOF).
    /// Provide exactly one of recv_amt or recv_wait.
    #[validate(range(min = 1, max = 1048576))]
    pub recv_amt: Option<usize>,
    /// recv_wait mode: read whatever arrives within this many seconds (0.1–60).
    /// Provide exactly one of recv_amt or recv_wait.
    #[validate(range(min = 0.1, max = 60.0))]
    pub recv_wait: Option<f64>,
    /// In recv_amt mode only, a hard read deadline in seconds (default 30).
    /// Ignored in recv_wait mode, where recv_wait is the deadline.
    pub timeout_sec: Option<u32>,
    /// How to return the bytes: "auto" (default — text if valid UTF-8, else hex),
    /// "text" (lossy UTF-8), or "hex".
    #[schemars(with = "Option<super::RecvFormat>")]
    pub format: Option<String>,
}

#[derive(Serialize)]
pub struct RecvResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_hex: Option<String>,
    pub bytes_received: usize,
    pub eof: bool,
    pub timed_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub had_invalid_bytes: Option<bool>,
}

pub struct TcpSessionRecv;

impl Tool for TcpSessionRecv {
    const NAME: &'static str = "tcp_session_recv";
    const DESCRIPTION: &'static str =
        "Read bytes from an open TCP stream. Two modes: recv_amt (wait for N bytes, use when \
         the protocol has a known message size) or recv_wait (drain for N seconds, use for \
         banner grabs or unknown-size responses). Triggered by \"read N bytes\", \"wait for X \
         bytes\", \"grab the banner\", \"drain the connection for N seconds\", \"wait and see \
         what comes back\". Returns data (printable UTF-8) or data_hex (binary) per the format \
         parameter, bytes_received, eof, and timed_out. No confirmation — receiving has no \
         remote side effects. Provide exactly one of recv_amt or recv_wait.";

    type Request = RecvRequest;
    type Response = RecvResponse;
    type Error = TcpError;

    fn run(ctx: &ToolContext, req: RecvRequest) -> Result<RecvResponse, TcpError> {
        match (req.recv_amt.is_some(), req.recv_wait.is_some()) {
            (false, false) => return Err(TcpError::MissingRecvMode),
            (true, true) => return Err(TcpError::ConflictingRecvModes),
            _ => {}
        }

        let entry_arc = ctx
            .sessions
            .get_tcp(&req.session_id)
            .ok_or_else(|| TcpError::SessionNotFound(req.session_id.clone()))?;
        let mut entry = entry_arc.lock().unwrap();

        let timeout = if let Some(wait) = req.recv_wait {
            Duration::from_millis((wait * 1000.0) as u64)
        } else {
            Duration::from_secs(req.timeout_sec.unwrap_or(30) as u64)
        };
        entry.stream.set_read_timeout(Some(timeout)).ok();

        let cap = req.recv_amt.unwrap_or(65536).min(65536);
        let mut buf = vec![0u8; cap];
        let (n, eof, timed_out) = match entry.stream.read(&mut buf) {
            Ok(0) => (0, true, false),
            Ok(n) => (n, false, false),
            Err(e)
                if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                (0, false, true)
            }
            Err(e) => return Err(TcpError::RecvFailed(e.to_string())),
        };

        let bytes = &buf[..n];
        let format = req.format.as_deref().unwrap_or("auto");

        let (data, data_hex, had_invalid) = match format {
            "hex" => (None, Some(hex::encode(bytes)), None),
            "text" => match std::str::from_utf8(bytes) {
                Ok(s) => (Some(s.to_string()), None, Some(false)),
                Err(_) => {
                    let s = String::from_utf8_lossy(bytes).into_owned();
                    (Some(s), None, Some(true))
                }
            },
            _ => {
                // auto: text if valid UTF-8, else hex
                match std::str::from_utf8(bytes) {
                    Ok(s) => (Some(s.to_string()), None, None),
                    Err(_) => (None, Some(hex::encode(bytes)), None),
                }
            }
        };

        Ok(RecvResponse {
            data,
            data_hex,
            bytes_received: n,
            eof,
            timed_out,
            had_invalid_bytes: had_invalid,
        })
    }
}

pub const TCP_SESSION_RECV: RegisteredTool = RegisteredTool::new::<TcpSessionRecv>();
