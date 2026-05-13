//! remote_fs_session_open tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::state::sessions::{RemoteFsConn, RemoteFsEntry, SessionMeta, SshConn};
use crate::{RegisteredTool, Tool, ToolContext};
use super::{now, RemoteFsError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    /// URI — only sftp:// is supported, e.g. sftp://user@host:22/home/user
    #[validate(length(min = 1))]
    pub uri: String,
    pub credential_name: Option<String>,
    pub idle_timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub protocol: String,
    pub host: String,
    pub port: u16,
    pub remote_prefix: String,
}

pub struct RemoteFsSessionOpen;

impl Tool for RemoteFsSessionOpen {
    const NAME: &'static str = "remote_fs_session_open";
    const DESCRIPTION: &'static str =
        "Open a remote filesystem session via URI. Only sftp:// is supported \
         (e.g. sftp://user@host:22/base/path). Requires an ssh_key or ssh_password \
         credential. Operations: list_dir, stat, get (→ VFS), put (← VFS), delete, \
         mkdir, rename. Use ssh_session_exec for arbitrary remote commands.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = RemoteFsError;

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, RemoteFsError> {
        let parsed = url::Url::parse(&req.uri)
            .map_err(|e| RemoteFsError::NotSupported(e.to_string()))?;

        match parsed.scheme() {
            "sftp" => {}
            other => return Err(RemoteFsError::NotSupported(other.to_string())),
        }

        let host = parsed
            .host_str()
            .ok_or_else(|| RemoteFsError::ConnectionFailed("no host in URI".into()))?
            .to_string();
        let port = parsed.port().unwrap_or(22);
        let remote_prefix = parsed.path().to_string();

        if ctx.sessions.list_remote_fs().len() >= 5 {
            return Err(RemoteFsError::SessionLimitExceeded);
        }

        let cred_name = req.credential_name.unwrap_or_default();
        let cred = ctx
            .credentials
            .get_by_name(&cred_name)
            .ok_or_else(|| RemoteFsError::CredentialNotFound(cred_name.clone()))?;

        let addr = format!("{host}:{port}");
        let stream = std::net::TcpStream::connect(&addr)
            .map_err(|e| RemoteFsError::ConnectionFailed(e.to_string()))?;

        let mut session = ssh2::Session::new()
            .map_err(|e| RemoteFsError::ConnectionFailed(e.to_string()))?;
        session.set_tcp_stream(stream.try_clone().unwrap());
        session
            .handshake()
            .map_err(|e| RemoteFsError::ConnectionFailed(e.to_string()))?;

        let username = cred.username.as_deref().unwrap_or("root");
        match cred.cred_type.as_str() {
            "ssh_password" | "remote_fs_password" => {
                session
                    .userauth_password(username, &cred.secret)
                    .map_err(|e| RemoteFsError::AuthFailed(e.to_string()))?;
            }
            "ssh_key" => {
                let tmp_key = std::env::temp_dir()
                    .join(format!("zend_key_{}", uuid::Uuid::new_v4().simple()));
                std::fs::write(&tmp_key, &cred.secret)
                    .map_err(|e| RemoteFsError::ConnectionFailed(e.to_string()))?;
                let result = session.userauth_pubkey_file(
                    username,
                    None,
                    tmp_key.as_path(),
                    cred.passphrase.as_deref(),
                );
                let _ = std::fs::remove_file(&tmp_key);
                result.map_err(|e| RemoteFsError::AuthFailed(e.to_string()))?;
            }
            t => {
                return Err(RemoteFsError::AuthFailed(format!(
                    "unsupported credential type: {t}"
                )))
            }
        }

        let sid = format!("sess_{}", Uuid::new_v4().simple());
        ctx.sessions.insert_remote_fs(RemoteFsEntry {
            meta: SessionMeta {
                session_id: sid.clone(),
                opened_at: now(),
                last_activity: now(),
                alive: true,
            },
            uri: req.uri,
            protocol: "sftp".into(),
            host: host.clone(),
            port,
            credential_id: cred.id.clone(),
            credential_name: cred.name,
            remote_prefix: remote_prefix.clone(),
            conn: RemoteFsConn {
                ssh: SshConn {
                    session,
                    _stream: stream,
                },
            },
        });

        Ok(OpenResponse {
            session_id: sid,
            protocol: "sftp".into(),
            host,
            port,
            remote_prefix,
        })
    }
}

pub const REMOTE_FS_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<RemoteFsSessionOpen>();
