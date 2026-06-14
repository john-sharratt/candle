//! tcp_session_open tool.

use std::net::TcpStream;
use std::time::Duration;

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::TcpError;
use crate::state::sessions::{SessionMeta, TcpEntry};
use crate::tools::web_fetch::is_private_ip;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    #[validate(length(min = 1))]
    pub host: String,
    pub port: u16,
    pub timeout_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub peer_addr: String,
    pub local_addr: String,
}

pub struct TcpSessionOpen;

impl Tool for TcpSessionOpen {
    const NAME: &'static str = "tcp_session_open";
    const DESCRIPTION: &'static str =
        "Open a raw TCP connection. Use for: custom protocols, debugging, \
         non-HTTP services. Use tls_session_open for TLS. Returns session_id.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = TcpError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Open TCP to {}:{}", req.host, req.port))
                .with_field("host", req.host.clone())
                .with_field("port", req.port.to_string()),
        )
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, TcpError> {
        if let Ok(ip) = req.host.parse::<std::net::IpAddr>() {
            if is_private_ip(ip) && req.host != "127.0.0.1" {
                return Err(TcpError::UrlBlocked(format!(
                    "{} is a private IP",
                    req.host
                )));
            }
        }

        let addr = format!("{}:{}", req.host, req.port);
        let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(5000));
        let addr_parsed = addr
            .parse::<std::net::SocketAddr>()
            .or_else(|_| {
                use std::net::ToSocketAddrs;
                addr.to_socket_addrs().map(|mut a| a.next().unwrap())
            })
            .map_err(|e| TcpError::ConnectionFailed(e.to_string()))?;

        let stream = TcpStream::connect_timeout(&addr_parsed, timeout)
            .map_err(|e| TcpError::ConnectionFailed(e.to_string()))?;
        let peer_addr = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        let local_addr = stream
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();

        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = TcpEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            peer_addr: peer_addr.clone(),
            local_addr: local_addr.clone(),
            stream,
        };
        ctx.sessions.insert_tcp(entry);

        Ok(OpenResponse {
            session_id,
            peer_addr,
            local_addr,
        })
    }
}

pub const TCP_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<TcpSessionOpen>();
