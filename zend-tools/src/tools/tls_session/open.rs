//! tls_session_open tool.

use std::net::TcpStream;
use std::time::Duration;

use chrono::Utc;
use native_tls::TlsConnector;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::TlsError;
use crate::state::sessions::{SessionMeta, TlsEntry};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    /// Target hostname (also used for TLS certificate verification / SNI).
    #[validate(length(min = 1))]
    pub host: String,
    /// Target TCP port (0–65535).
    pub port: u16,
    /// If true, skip certificate validation (accept invalid/self-signed certs).
    /// Default false.
    pub accept_invalid_certs: Option<bool>,
    /// Name of a stored client credential to present. Default none.
    pub credential_name: Option<String>,
    /// Connection timeout in milliseconds. Default 5000.
    pub timeout_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub local_addr: String,
}

pub struct TlsSessionOpen;

impl Tool for TlsSessionOpen {
    const NAME: &'static str = "tls_session_open";
    const DESCRIPTION: &'static str =
        "Open a TLS-encrypted TCP connection. Use for LDAPS, IMAPS, SMTPS, MQTTS, \
         or any TLS service. For HTTP over TLS, prefer http_session_open. \
         Returns session_id for subsequent sends/receives.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = TlsError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Open TLS to {}:{}", req.host, req.port))
                .with_field("host", req.host.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, TlsError> {
        let addr = format!("{}:{}", req.host, req.port);
        let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(5000));
        let addr_parsed: std::net::SocketAddr = addr
            .parse()
            .or_else(|_| {
                use std::net::ToSocketAddrs;
                addr.to_socket_addrs().map(|mut a| a.next().unwrap())
            })
            .map_err(|e| TlsError::ConnectionFailed(e.to_string()))?;

        let stream = TcpStream::connect_timeout(&addr_parsed, timeout)
            .map_err(|e| TlsError::ConnectionFailed(e.to_string()))?;
        let local_addr = stream
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();

        let mut connector_builder = TlsConnector::builder();
        if req.accept_invalid_certs == Some(true) {
            connector_builder.danger_accept_invalid_certs(true);
        }
        let connector = connector_builder
            .build()
            .map_err(|e| TlsError::HandshakeFailed(e.to_string()))?;

        let tls_stream = connector
            .connect(&req.host, stream)
            .map_err(|e| TlsError::HandshakeFailed(e.to_string()))?;

        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = TlsEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            host: req.host.clone(),
            port: req.port,
            local_addr: local_addr.clone(),
            stream: tls_stream,
        };
        ctx.sessions.insert_tls(entry);

        Ok(OpenResponse {
            session_id,
            host: req.host,
            port: req.port,
            local_addr,
        })
    }
}

pub const TLS_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<TlsSessionOpen>();
