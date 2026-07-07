//! credential_save tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::CredError;
use crate::state::credentials::Credential;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SaveRequest {
    /// Unique friendly name to store and later reference this credential by.
    #[validate(length(min = 1))]
    pub name: String,
    /// Credential type. Infer it from the secret: an API token (e.g. one starting
    /// `sk-`, `ghp_`, `sk_live_`, `xoxb-`) is `http_bearer`; an SSH PEM/OpenSSH
    /// private key is `ssh_key`; a PEM signing key is `signing_key`; a base32 2FA
    /// seed is `totp_secret`; a DB login is `sql_password`.
    #[schemars(with = "super::CredType")]
    #[validate(length(min = 1))]
    #[serde(rename = "type")]
    pub cred_type: String,
    /// Login name. Required for ssh_key, ssh_password, telnet_password, http_basic,
    /// sql_password, and remote_fs_password.
    pub username: Option<String>,
    /// The secret material: API token, password, base32 TOTP seed, or PEM key block.
    #[validate(length(min = 1))]
    pub secret: String,
    /// Passphrase protecting an encrypted `ssh_key`, if any.
    pub passphrase: Option<String>,
    /// HTTP header name (e.g. "X-API-Key"). Required for type http_header.
    pub header_name: Option<String>,
    /// Authentication domain (e.g. SMB/AD domain for remote_fs_password).
    pub domain: Option<String>,
    /// Default host this credential connects to, if any.
    pub default_host: Option<String>,
    /// Default port this credential connects to, if any.
    pub default_port: Option<u16>,
    /// Default database name (for sql_password).
    pub default_database: Option<String>,
}

#[derive(Serialize)]
pub struct SaveResponse {
    pub id: String,
    pub name: String,
    pub created: bool,
}

pub struct CredentialSave;

impl Tool for CredentialSave {
    const NAME: &'static str = "credential_save";
    const DESCRIPTION: &'static str =
        "Save a new credential — SSH key, password, API token, TOTP secret, database login, \
         or remote-filesystem auth — to the encrypted credential store for later use by session \
         and security tools. Supported types: ssh_key, ssh_password, telnet_password, \
         http_bearer, http_basic, http_header, totp_secret, sql_password, remote_fs_password, \
         tls_client_cert, signing_key. Triggered by \"save this key\", \"remember this \
         password\", \"store my credentials for\", \"add a credential called\", \"set up auth \
         for\". Returns the credential ID and name. Use cred_list to find existing \
         credentials; credential_delete to revoke. Note: secrets passed here enter conversation \
         history.";

    type Request = SaveRequest;
    type Response = SaveResponse;
    type Error = CredError;

    fn confirmation(req: &SaveRequest) -> Option<ConfirmationDetails> {
        Some(
            ConfirmationDetails::new(format!("Save credential {:?}", req.name))
                .with_field("type", req.cred_type.clone())
                .with_field("name", req.name.clone()),
        )
    }

    fn run(ctx: &ToolContext, req: SaveRequest) -> Result<SaveResponse, CredError> {
        const VALID_TYPES: &[&str] = &[
            "ssh_key",
            "ssh_password",
            "telnet_password",
            "http_bearer",
            "http_basic",
            "http_header",
            "totp_secret",
            "sql_password",
            "remote_fs_password",
            "tls_client_cert",
            "signing_key",
            // aliases accepted for compatibility
            "api_key",
            "ed25519_key",
        ];
        if !VALID_TYPES.contains(&req.cred_type.as_str()) {
            return Err(CredError::InvalidType(format!(
                "unknown type {:?}; valid types: {}",
                req.cred_type,
                VALID_TYPES.join(", ")
            )));
        }

        let needs_username = matches!(
            req.cred_type.as_str(),
            "ssh_key"
                | "ssh_password"
                | "http_basic"
                | "sql_password"
                | "telnet_password"
                | "remote_fs_password"
        );
        let needs_header_name = req.cred_type == "http_header";

        if needs_username && req.username.is_none() {
            return Err(CredError::MissingField(format!(
                "username required for type {}; provide the login name for this credential",
                req.cred_type
            )));
        }
        if needs_header_name && req.header_name.is_none() {
            return Err(CredError::MissingField(
                "header_name required for type http_header (e.g. \"X-API-Key\")".to_string(),
            ));
        }

        // Validate that ssh_key secret looks like a PEM or OpenSSH key
        if req.cred_type == "ssh_key"
            && !req.secret.contains("BEGIN")
            && !req.secret.contains("openssh")
        {
            return Err(CredError::InvalidKey(
                "ssh_key secret must be a PEM or OpenSSH private key (should begin with \
                 '-----BEGIN ... PRIVATE KEY-----' or 'openssh-key-v1')"
                    .to_string(),
            ));
        }

        let id = format!("cred_{}", Uuid::new_v4());
        let cred = Credential {
            id: id.clone(),
            name: req.name.clone(),
            cred_type: req.cred_type,
            username: req.username,
            header_name: req.header_name,
            domain: req.domain,
            secret: req.secret,
            passphrase: req.passphrase,
            default_host: req.default_host,
            default_port: req.default_port,
            default_database: req.default_database,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        ctx.credentials
            .save(cred)
            .map_err(CredError::DuplicateName)?;

        Ok(SaveResponse {
            id,
            name: req.name,
            created: true,
        })
    }
}

pub const CREDENTIAL_SAVE: RegisteredTool = RegisteredTool::new::<CredentialSave>();
