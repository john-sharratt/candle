//! tls_session_send tool.

use std::io::Write;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};
use super::TlsError;

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

pub struct TlsSessionSend;

impl Tool for TlsSessionSend {
    const NAME: &'static str = "tls_session_send";
    const DESCRIPTION: &'static str = "Send data over an open TLS session.";

    type Request = SendRequest;
    type Response = SendResponse;
    type Error = TlsError;

    fn confirmation(req: &SendRequest) -> Option<ConfirmationDetails> {
        Some(ConfirmationDetails::new("Send over TLS")
            .with_field("session_id", req.session_id.clone()))
    }

    fn run(ctx: &ToolContext, req: SendRequest) -> Result<SendResponse, TlsError> {
        let bytes = if let Some(hex) = &req.data_hex {
            hex::decode(hex).map_err(|e| TlsError::SendFailed(e.to_string()))?
        } else if let Some(text) = &req.data {
            text.as_bytes().to_vec()
        } else {
            return Err(TlsError::InvalidParams("data or data_hex required".to_string()));
        };

        let entry_arc = ctx.sessions.get_tls(&req.session_id)
            .ok_or_else(|| TlsError::SessionNotFound(req.session_id.clone()))?;
        let mut entry = entry_arc.lock().unwrap();
        entry.stream.write_all(&bytes)
            .map_err(|e| TlsError::SendFailed(e.to_string()))?;

        Ok(SendResponse { bytes_written: bytes.len() })
    }
}

pub const TLS_SESSION_SEND: RegisteredTool = RegisteredTool::new::<TlsSessionSend>();
