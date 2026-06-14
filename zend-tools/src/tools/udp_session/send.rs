//! udp_session_send tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::UdpError;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SendRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    pub data: Option<String>,
    pub data_hex: Option<String>,
    pub peer: Option<String>,
}

#[derive(Serialize)]
pub struct SendResponse {
    pub bytes_sent: usize,
}

pub struct UdpSessionSend;

impl Tool for UdpSessionSend {
    const NAME: &'static str = "udp_session_send";
    const DESCRIPTION: &'static str = "Transmit one fire-and-forget UDP datagram to a peer, payload as text or hex. No delivery acknowledgement.";

    type Request = SendRequest;
    type Response = SendResponse;
    type Error = UdpError;

    fn confirmation(req: &SendRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new("Send UDP datagram")
                .with_field("session_id", req.session_id.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: SendRequest) -> Result<SendResponse, UdpError> {
        let bytes = if let Some(hex) = &req.data_hex {
            hex::decode(hex).map_err(|e| UdpError::SendFailed(e.to_string()))?
        } else if let Some(text) = &req.data {
            text.as_bytes().to_vec()
        } else {
            return Err(UdpError::InvalidParams(
                "data or data_hex required".to_string(),
            ));
        };

        let entry_arc = ctx
            .sessions
            .get_udp(&req.session_id)
            .ok_or_else(|| UdpError::SessionNotFound(req.session_id.clone()))?;
        let entry = entry_arc.lock().unwrap();

        let n = if let Some(peer) = &req.peer {
            entry
                .socket
                .send_to(&bytes, peer)
                .map_err(|e| UdpError::SendFailed(e.to_string()))?
        } else {
            entry
                .socket
                .send(&bytes)
                .map_err(|e| UdpError::SendFailed(e.to_string()))?
        };

        Ok(SendResponse { bytes_sent: n })
    }
}

pub const UDP_SESSION_SEND: RegisteredTool = RegisteredTool::new::<UdpSessionSend>();
