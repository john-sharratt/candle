//! In-memory credential store for `credential_*` tools and session opens.
//!
//! Credentials are keyed by `name` (the user-facing friendly label, unique per
//! user).  They also carry a UUID `id` for backward-compatible lookups.
//!
//! # What is stored
//!
//! Each [`Credential`] holds metadata (type, username, header_name, default_host,
//! default_port, default_database) plus the raw `secret` and optional `passphrase`
//! as plain strings.  In the production daemon these fields would be encrypted at
//! rest with chacha20poly1305; in this in-process store they are held in memory
//! and scoped to a single conversation.
//!
//! # Accepted types
//!
//! `ssh_key`, `ssh_password`, `telnet_password`, `http_bearer`, `http_basic`,
//! `http_header`, `totp_secret`, `sql_password`, `remote_fs_password`,
//! `tls_client_cert`, `signing_key`.  Aliases `api_key` and `ed25519_key` are
//! also accepted for backward compatibility.
//!
//! # Lookup
//!
//! - `get_by_name` — O(1) primary path
//! - `get_by_id` — O(n) linear scan; kept for compatibility
//! - `delete` operates by name; returns `true` if the credential was present

use std::collections::HashMap;
use std::sync::RwLock;

use chrono::Utc;
use uuid::Uuid;

#[derive(Clone)]
pub struct Credential {
    pub id: String,
    pub name: String,
    pub cred_type: String,
    pub username: Option<String>,
    pub header_name: Option<String>,
    pub domain: Option<String>,
    pub secret: String,
    pub passphrase: Option<String>,
    pub default_host: Option<String>,
    pub default_port: Option<u16>,
    pub default_database: Option<String>,
    pub created_at: String,
}

#[derive(Default)]
pub struct CredentialStore {
    inner: RwLock<HashMap<String, Credential>>,
}

impl CredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save(&self, mut cred: Credential) -> Result<(), String> {
        let mut guard = self.inner.write().unwrap();
        // Check for duplicate name (exclude self-update by same name key)
        if guard.contains_key(&cred.name) && guard[&cred.name].id != cred.id {
            return Err(format!(
                "credential with name {:?} already exists",
                cred.name
            ));
        }
        if cred.id.is_empty() {
            cred.id = format!("cred_{}", Uuid::new_v4());
        }
        if cred.created_at.is_empty() {
            cred.created_at = Utc::now().to_rfc3339();
        }
        let key = cred.name.clone();
        guard.insert(key, cred);
        Ok(())
    }

    pub fn list(&self, type_filter: Option<&str>) -> Vec<Credential> {
        let guard = self.inner.read().unwrap();
        let mut creds: Vec<Credential> = guard
            .values()
            .filter(|c| type_filter.is_none_or(|t| c.cred_type == t))
            .cloned()
            .collect();
        creds.sort_by(|a, b| a.name.cmp(&b.name));
        creds
    }

    /// Delete by name (primary key). Returns true if removed.
    pub fn delete(&self, name: &str) -> bool {
        self.inner.write().unwrap().remove(name).is_some()
    }

    /// O(1) lookup by name.
    pub fn get_by_name(&self, name: &str) -> Option<Credential> {
        self.inner.read().unwrap().get(name).cloned()
    }

    /// Linear scan by id — kept for backward-compat within this crate.
    pub fn get_by_id(&self, id: &str) -> Option<Credential> {
        self.inner
            .read()
            .unwrap()
            .values()
            .find(|c| c.id == id)
            .cloned()
    }
}
