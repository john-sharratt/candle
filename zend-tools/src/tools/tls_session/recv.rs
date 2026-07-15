//! tls_session_recv tool.

use std::io::Read;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TlsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RecvRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    /// Maximum number of bytes to read in one call (capped at 65536).
    pub recv_amt: Option<usize>,
    /// Seconds to wait for data before returning with `timed_out: true`. Defaults to 30.
    pub recv_wait: Option<f64>,
    /// Encoding for the returned bytes. "hex" forces hex; "auto" (default) returns UTF-8 text when valid, else hex.
    #[schemars(with = "Option<super::super::RecvEncoding>")]
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct RecvResponse {
    pub data: String,
    pub bytes_received: usize,
    pub eof: bool,
    pub timed_out: bool,
}

pub struct TlsSessionRecv;

impl Tool for TlsSessionRecv {
    const NAME: &'static str = "tls_session_recv";
    const DESCRIPTION: &'static str = "Read decrypted application data from an open TLS channel; decryption and certificate handling are transparent.";

    type Request = RecvRequest;
    type Response = RecvResponse;
    type Error = TlsError;

    fn run(ctx: &ToolContext, req: RecvRequest) -> Result<RecvResponse, TlsError> {
        let entry_arc = ctx
            .sessions
            .get_tls(&req.session_id)
            .ok_or_else(|| TlsError::SessionNotFound(req.session_id.clone()))?;
        let mut entry = entry_arc.lock().unwrap();

        // Bound the blocking read so a quiet peer returns `timed_out` instead of
        // hanging. The timeout is set on the TcpStream beneath the TLS layer.
        let timeout = Duration::from_secs_f64(req.recv_wait.unwrap_or(30.0));
        entry.stream.get_ref().set_read_timeout(Some(timeout)).ok();

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
            Err(e) => return Err(TlsError::RecvFailed(e.to_string())),
        };

        let encoding = req.output_encoding.as_deref().unwrap_or("auto");
        let data = if encoding == "hex" {
            hex::encode(&buf[..n])
        } else {
            match std::str::from_utf8(&buf[..n]) {
                Ok(s) => s.to_string(),
                Err(_) => hex::encode(&buf[..n]),
            }
        };

        Ok(RecvResponse {
            data,
            bytes_received: n,
            eof,
            timed_out,
        })
    }
}

pub const TLS_SESSION_RECV: RegisteredTool = RegisteredTool::new::<TlsSessionRecv>();
