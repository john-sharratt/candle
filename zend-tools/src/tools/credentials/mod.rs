//! Credential management tools: `credential_save`, `cred_list`, `credential_delete`.
//!
//! Credentials are named auth material stored in [`crate::state::CredentialStore`] for
//! use by session-open tools (`ssh_open`, `http_session_open`, etc.).  Tools
//! reference credentials by the friendly `name`, not by UUID.
//!
//! # Supported credential types
//!
//! | Type | Required extra fields |
//! |------|-----------------------|
//! | `ssh_key` | `username` + PEM/OpenSSH key in `secret`, optional `passphrase` |
//! | `ssh_password` | `username` |
//! | `telnet_password` | `username` |
//! | `http_bearer` | — (token goes in `secret`) |
//! | `http_basic` | `username` |
//! | `http_header` | `header_name` |
//! | `totp_secret` | — (base32 TOTP seed in `secret`) |
//! | `sql_password` | `username`, optional `default_database` |
//! | `remote_fs_password` | `username`, optional `domain` (SMB AD) |
//! | `tls_client_cert` | — (combined cert+key PEM bundle in `secret`) |
//! | `signing_key` | — (PEM private key in `secret`) |
//!
//! Aliases `api_key` and `ed25519_key` are also accepted.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `duplicate_name` | A credential with that name already exists |
//! | `not_found` | Name not in the store (delete) |
//! | `missing_field` | A type-required field (`username`, `header_name`) was absent |
//! | `invalid_credential_type` | `type` field is not in the accepted list |
//! | `invalid_key` | `ssh_key` secret does not contain a recognisable PEM/OpenSSH header |
//!
//! # Confirmation policy
//!
//! `credential_save` confirms before saving (shows type + name).
//! `cred_list` and `credential_delete` do not confirm.

use thiserror::Error;

use crate::ToolError;

pub mod delete;
pub mod list;
pub mod save;

pub use delete::CREDENTIAL_DELETE;
pub use list::CREDENTIAL_LIST;
pub use save::CREDENTIAL_SAVE;

#[derive(Debug, Error)]
pub enum CredError {
    #[error("credential name already exists: {0}")]
    DuplicateName(String),
    #[error("credential not found: {0}")]
    NotFound(String),
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid credential type: {0}")]
    InvalidType(String),
    #[error("invalid key material: {0}")]
    InvalidKey(String),
}

impl ToolError for CredError {
    fn code(&self) -> &'static str {
        match self {
            CredError::DuplicateName(_) => "duplicate_name",
            CredError::NotFound(_) => "not_found",
            CredError::MissingField(_) => "missing_field",
            CredError::InvalidType(_) => "invalid_credential_type",
            CredError::InvalidKey(_) => "invalid_key",
        }
    }
}
