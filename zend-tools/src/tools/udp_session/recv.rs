//! udp_session_recv tool.

use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::UdpError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RecvRequest {
    /// The session id returned by udp_session_open.
    #[validate(length(min = 1))]
    pub session_id: String,
    /// Receive timeout in seconds. Default 5.
    pub timeout_sec: Option<f64>,
    /// Encoding for the returned bytes. "hex" forces hex; "auto" (default) returns UTF-8 text when valid, else hex.
    #[schemars(with = "Option<super::super::RecvEncoding>")]
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct RecvResponse {
    pub data: String,
    pub bytes_received: usize,
    pub from: String,
    pub timed_out: bool,
}

pub struct UdpSessionRecv;

impl Tool for UdpSessionRecv {
    const NAME: &'static str = "udp_session_recv";
    const DESCRIPTION: &'static str =
        "Read one incoming UDP datagram, returning its packet payload and the sender's address.";

    type Request = RecvRequest;
    type Response = RecvResponse;
    type Error = UdpError;

    fn run(ctx: &ToolContext, req: RecvRequest) -> Result<RecvResponse, UdpError> {
        let entry_arc = ctx
            .sessions
            .get_udp(&req.session_id)
            .ok_or_else(|| UdpError::SessionNotFound(req.session_id.clone()))?;
        let entry = entry_arc.lock().unwrap();

        let timeout = req
            .timeout_sec
            .map(|s| Duration::from_millis((s * 1000.0) as u64))
            .unwrap_or(Duration::from_secs(5));
        entry.socket.set_read_timeout(Some(timeout)).ok();

        let mut buf = vec![0u8; 65536];
        match entry.socket.recv_from(&mut buf) {
            Ok((n, from)) => {
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
                    from: from.to_string(),
                    timed_out: false,
                })
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                Ok(RecvResponse {
                    data: String::new(),
                    bytes_received: 0,
                    from: String::new(),
                    timed_out: true,
                })
            }
            Err(e) => Err(UdpError::RecvFailed(e.to_string())),
        }
    }
}

pub const UDP_SESSION_RECV: RegisteredTool = RegisteredTool::new::<UdpSessionRecv>();
