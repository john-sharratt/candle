//! ssh_session_open tool.

use std::net::TcpStream;

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::state::sessions::{SessionMeta, SshConn, SshEntry};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};
use super::{SshError, exec_simple};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    #[validate(length(min = 1))]
    pub credential_name: String,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub idle_timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub host: String,
    pub cwd: String,
    pub shell: String,
}

pub struct SshSessionOpen;

impl Tool for SshSessionOpen {
    const NAME: &'static str = "ssh_session_open";
    const DESCRIPTION: &'static str =
        "Open a persistent SSH shell session on a remote host using a stored credential. Use \
         to start a working session before issuing commands, when the user names an SSH host \
         to connect to, or before any operation phrased as \"ssh in\", \"connect to the \
         server\", \"log into\". Triggered by \"ssh into\", \"open a connection to\", \
         \"connect to the bastion\", \"log into the server\", \"start a shell on\", \"open a \
         session on prod\". Returns session_id, host, initial working directory, and shell \
         path. The session persists until closed or idle-timed-out (15 min default). \
         Subsequent commands run via ssh_session_exec; close with ssh_session_close when done.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = SshError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        let host = req.host.as_deref().unwrap_or("default");
        Some(ConfirmationDetails::new(format!("Open SSH session to {host}"))
            .with_field("credential_name", req.credential_name.clone())
            .with_field("host", host.to_string())
            .with_field("port", req.port.unwrap_or(22).to_string()))
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, SshError> {
        let cred = ctx.credentials.get_by_name(&req.credential_name)
            .ok_or_else(|| SshError::CredentialNotFound(req.credential_name.clone()))?;

        let host = req.host.as_deref()
            .or(cred.default_host.as_deref())
            .ok_or_else(|| SshError::ConnectionFailed("no host specified".to_string()))?
            .to_string();
        let port = req.port.or(cred.default_port).unwrap_or(22);

        let addr = format!("{host}:{port}");
        let stream = TcpStream::connect(&addr)
            .map_err(|e| SshError::ConnectionFailed(format!("{addr}: {e}")))?;

        let mut session = ssh2::Session::new()
            .map_err(|e| SshError::ConnectionFailed(e.to_string()))?;
        session.set_tcp_stream(stream.try_clone().unwrap());
        session.handshake()
            .map_err(|e| SshError::ConnectionFailed(e.to_string()))?;

        let username = cred.username.as_deref().unwrap_or("root");
        match cred.cred_type.as_str() {
            "ssh_password" => {
                session.userauth_password(username, &cred.secret)
                    .map_err(|e| SshError::AuthFailed(e.to_string()))?;
            }
            "ssh_key" => {
                #[cfg(unix)]
                {
                    let passphrase = cred.passphrase.as_deref();
                    session.userauth_pubkey_memory(username, None, &cred.secret, passphrase)
                        .map_err(|e| SshError::AuthFailed(e.to_string()))?;
                }
                #[cfg(not(unix))]
                {
                    let tmp = std::env::temp_dir().join(format!("zend_key_{}.pem", uuid::Uuid::new_v4()));
                    std::fs::write(&tmp, cred.secret.as_bytes())
                        .map_err(|e| SshError::AuthFailed(e.to_string()))?;
                    let result = session.userauth_pubkey_file(
                        username,
                        None,
                        std::path::Path::new(&tmp),
                        cred.passphrase.as_deref(),
                    );
                    let _ = std::fs::remove_file(&tmp);
                    result.map_err(|e| SshError::AuthFailed(e.to_string()))?;
                }
            }
            _ => {
                return Err(SshError::AuthFailed(format!("unsupported credential type: {}", cred.cred_type)));
            }
        }

        let cwd = exec_simple(&session, "pwd").unwrap_or_else(|_| "/".to_string());
        let shell = exec_simple(&session, "echo $SHELL").unwrap_or_else(|_| "/bin/sh".to_string());

        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = SshEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            host: host.clone(),
            port,
            credential_id: cred.id,
            credential_name: cred.name,
            cwd: cwd.trim().to_string(),
            conn: SshConn { session, _stream: stream },
        };
        ctx.sessions.insert_ssh(entry);

        Ok(OpenResponse {
            session_id,
            host,
            cwd: cwd.trim().to_string(),
            shell: shell.trim().to_string(),
        })
    }
}

pub const SSH_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<SshSessionOpen>();
