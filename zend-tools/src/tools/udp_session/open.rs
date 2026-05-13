//! udp_session_open tool.

use std::net::UdpSocket;

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::state::sessions::{SessionMeta, UdpEntry};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};
use super::UdpError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    pub default_peer: Option<String>,
    pub bind_addr: Option<String>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub local_addr: String,
    pub default_peer: String,
}

pub struct UdpSessionOpen;

impl Tool for UdpSessionOpen {
    const NAME: &'static str = "udp_session_open";
    const DESCRIPTION: &'static str =
        "Open a UDP socket for datagrams. Optionally connect to a default peer. \
         Returns session_id for subsequent sends/receives.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = UdpError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        Some(ConfirmationDetails::new("Open UDP socket")
            .with_field("default_peer", req.default_peer.as_deref().unwrap_or("(none)").to_string()))
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, UdpError> {
        let bind = req.bind_addr.as_deref().unwrap_or("0.0.0.0:0");
        let socket = UdpSocket::bind(bind)
            .map_err(|e| UdpError::BindFailed(e.to_string()))?;

        let default_peer = req.default_peer.clone().unwrap_or_default();
        if !default_peer.is_empty() {
            socket.connect(&default_peer)
                .map_err(|e| UdpError::BindFailed(e.to_string()))?;
        }

        let local_addr = socket.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = UdpEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            default_peer: default_peer.clone(),
            local_addr: local_addr.clone(),
            socket,
        };
        ctx.sessions.insert_udp(entry);

        Ok(OpenResponse { session_id, local_addr, default_peer })
    }
}

pub const UDP_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<UdpSessionOpen>();
