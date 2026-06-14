//! tcp_session_send tool.

use std::io::Write;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TcpError;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SendRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    pub data: Option<String>,
    pub data_hex: Option<String>,
}

#[derive(Serialize)]
pub struct SendResponse {
    pub bytes_written: usize,
}

pub struct TcpSessionSend;

impl Tool for TcpSessionSend {
    const NAME: &'static str = "tcp_session_send";
    const DESCRIPTION: &'static str =
        "Write bytes to an open TCP stream. Accepts text via data or arbitrary bytes via \
         data_hex (hex-encoded, whitespace allowed — '16 03 01 00 ff' and '160301 00ff' both \
         parse). Use for: sending a protocol message, writing a command to a custom service, \
         transmitting a binary payload. Triggered by \"send bytes to the connection\", \"write \
         to the socket\", \"transmit\", \"send this packet\", \"send these hex bytes\". Returns \
         bytes_written. Every send requires user confirmation. For the response use \
         tcp_session_recv. For TLS-protected streams use tls_session_send instead.";

    type Request = SendRequest;
    type Response = SendResponse;
    type Error = TcpError;

    fn confirmation(req: &SendRequest) -> Option<ConfirmationDetails> {
        let preview = req
            .data
            .as_deref()
            .or(req.data_hex.as_deref())
            .unwrap_or("(empty)");
        Some(
            ConfirmationDetails::new("Send to TCP session")
                .with_field("session_id", req.session_id.clone())
                .with_field("data", preview[..preview.len().min(80)].to_string()),
        )
    }

    fn run(ctx: &ToolContext, req: SendRequest) -> Result<SendResponse, TcpError> {
        let bytes = if let Some(hex) = &req.data_hex {
            hex::decode(hex).map_err(|e| TcpError::SendFailed(e.to_string()))?
        } else if let Some(text) = &req.data {
            text.as_bytes().to_vec()
        } else {
            return Err(TcpError::InvalidParams(
                "data or data_hex required".to_string(),
            ));
        };

        let entry_arc = ctx
            .sessions
            .get_tcp(&req.session_id)
            .ok_or_else(|| TcpError::SessionNotFound(req.session_id.clone()))?;
        let mut entry = entry_arc.lock().unwrap();
        entry
            .stream
            .write_all(&bytes)
            .map_err(|e| TcpError::SendFailed(e.to_string()))?;

        Ok(SendResponse {
            bytes_written: bytes.len(),
        })
    }
}

pub const TCP_SESSION_SEND: RegisteredTool = RegisteredTool::new::<TcpSessionSend>();
