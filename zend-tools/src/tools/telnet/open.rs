//! telnet_session_open tool.

use std::io::Read;
use std::net::TcpStream;
use std::time::Duration;

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::state::sessions::{SessionMeta, TelnetEntry};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};
use super::TelnetError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    #[validate(length(min = 1))]
    pub host: String,
    pub port: Option<u16>,
    pub credential_name: Option<String>,
    pub prompt_pattern: Option<String>,
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub banner: String,
}

pub struct TelnetSessionOpen;

impl Tool for TelnetSessionOpen {
    const NAME: &'static str = "telnet_session_open";
    const DESCRIPTION: &'static str =
        "Open a Telnet session to a network device or host. \
         Use for legacy network equipment, serial consoles, or any TCP text protocol. \
         Returns session_id for subsequent sends.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = TelnetError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        Some(ConfirmationDetails::new(format!("Open Telnet to {}:{}", req.host, req.port.unwrap_or(23)))
            .with_field("host", req.host.clone()))
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, TelnetError> {
        let port = req.port.unwrap_or(23);
        let addr = format!("{}:{}", req.host, port);
        let timeout = Duration::from_secs(req.timeout_sec.unwrap_or(10) as u64);
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e: std::net::AddrParseError| TelnetError::ConnectionFailed(e.to_string()))?,
            timeout,
        ).map_err(|e| TelnetError::ConnectionFailed(e.to_string()))?;
        stream.set_read_timeout(Some(Duration::from_secs(3))).ok();

        let prompt_pattern = req.prompt_pattern.clone().unwrap_or_else(|| "[$#>] *$".to_string());

        let mut banner_bytes = vec![0u8; 4096];
        let mut banner = String::new();
        if let Ok(n) = stream.try_clone().unwrap().read(&mut banner_bytes) {
            banner = String::from_utf8_lossy(&banner_bytes[..n]).into_owned();
        }

        if let Some(cred_name) = &req.credential_name {
            if let Some(cred) = ctx.credentials.get_by_name(cred_name) {
                use std::io::Write;
                let mut stream_clone = stream.try_clone().unwrap();
                if let Some(username) = &cred.username {
                    let _ = stream_clone.write_all(format!("{}\r\n", username).as_bytes());
                    std::thread::sleep(Duration::from_millis(300));
                }
                let _ = stream_clone.write_all(format!("{}\r\n", cred.secret).as_bytes());
                std::thread::sleep(Duration::from_millis(300));
                let mut resp = vec![0u8; 4096];
                if let Ok(n) = stream_clone.read(&mut resp) {
                    banner.push_str(&String::from_utf8_lossy(&resp[..n]));
                }
            }
        }

        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = TelnetEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            host: req.host.clone(),
            port,
            prompt_pattern,
            stream,
        };
        ctx.sessions.insert_telnet(entry);

        Ok(OpenResponse {
            session_id,
            host: req.host,
            port,
            banner: banner.trim().to_string(),
        })
    }
}

pub const TELNET_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<TelnetSessionOpen>();
