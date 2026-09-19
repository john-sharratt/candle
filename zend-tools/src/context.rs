//! Shared state bundle handed to every tool's `run` call.
//!
//! [`ToolContext`] is cheap to clone (`Arc`-wrapped stores) and is passed by
//! reference into each tool invocation.  Tools read and mutate the stores they
//! need; they never allocate or own state themselves.
//!
//! # Stores
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `vfs` | [`state::VfsStore`] | Overlay filesystem for `file_*` tools — session writes over the workspace |
//! | `credentials` | [`state::CredentialStore`] | Named auth material for session opens — reached only through [`ToolContext::credentials`] |
//! | `notes` | [`state::NotesStore`] | Cross-conversation persistent key-value store |
//! | `sessions` | [`state::SessionRegistry`] | All open protocol sessions (SSH, TCP, …) |
//! | `hash_states` | [`state::HashStateStore`] | Running hash contexts for `hash_state_*` tools |
//! | `http_client` | `reqwest::blocking::Client` | Shared HTTP client for `web_fetch`, `weather`, etc. — reached only through [`ToolContext::http`] |
//! | `secrets` | [`state::ToolSecrets`] | Deployment API keys (Tavily) for the tools that call third-party services |
//! | `subagent_runner` | `Option<Arc<dyn SubagentRunner>>` | Injected by daemon to run nested agent loops |
//!
//! # Grants
//!
//! A context also carries the [`Grants`] its calls run under — see
//! [`crate::grants`]. Every constructor grants **nothing**; the daemon grants
//! what a caller is entitled to with [`ToolContext::granting`]. The field is
//! private, so a tool can read its grants but never widen them.
//!
//! # Construction
//!
//! In production the daemon calls [`ToolContext::with_workspace`] once at startup,
//! passing its working directory so the `file_*` tools resolve real project files
//! through the VFS overlay. [`ToolContext::new`] leaves the overlay upper-only,
//! which is what most tests want; a test needing the lower layer points
//! `with_workspace` at a temp dir.

use std::path::PathBuf;
use std::sync::Arc;

use crate::grants::{Capability, Grants, NotPermitted};
use crate::state::{
    CredentialStore, HashStateStore, NotesStore, SessionRegistry, ToolSecrets, VfsStore,
};

/// Read-only handle bundle passed by the runner into each tool invocation.
/// All stores are wrapped in `Arc` so cloning the context is cheap.
#[derive(Clone)]
pub struct ToolContext {
    pub vfs: Arc<VfsStore>,
    credentials: Arc<CredentialStore>,
    pub notes: Arc<NotesStore>,
    pub sessions: Arc<SessionRegistry>,
    pub hash_states: Arc<HashStateStore>,
    http_client: reqwest::blocking::Client,
    pub secrets: Arc<ToolSecrets>,
    pub subagent_runner: Option<Arc<dyn crate::SubagentRunner>>,
    grants: Grants,
}

impl ToolContext {
    /// Construct a context with default-initialized stores, no workspace
    /// layer — `file_*` tools see only what this session writes — and no
    /// grants.
    pub fn new() -> Self {
        Self::build(VfsStore::new())
    }

    /// Construct a context whose VFS overlays `workspace`, the daemon's working
    /// directory: `file_*` reads fall through to real project files, writes and
    /// edits stay in memory. Grants nothing.
    pub fn with_workspace(workspace: impl Into<PathBuf>) -> Self {
        Self::build(VfsStore::with_workspace(workspace))
    }

    fn build(vfs: VfsStore) -> Self {
        Self {
            vfs: Arc::new(vfs),
            credentials: Arc::new(CredentialStore::new()),
            notes: Arc::new(NotesStore::new()),
            sessions: Arc::new(SessionRegistry::new()),
            hash_states: Arc::new(HashStateStore::new()),
            http_client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
            // Unset unless the daemon supplies them: a test, and any caller that
            // is not the daemon, gets a context whose third-party tools report
            // themselves unconfigured rather than reaching the network.
            secrets: Arc::new(ToolSecrets::empty()),
            subagent_runner: None,
            grants: Grants::NONE,
        }
    }

    /// The grants this context's calls run under.
    pub fn grants(&self) -> Grants {
        self.grants
    }

    /// This context under `grants` — replacing, not adding to, what it held.
    pub fn granting(mut self, grants: Grants) -> Self {
        self.grants = grants;
        self
    }

    /// The shared HTTP client, when the context may use the network.
    pub fn http(&self) -> Result<&reqwest::blocking::Client, NotPermitted> {
        self.grants.require(Capability::Network)?;
        Ok(&self.http_client)
    }

    /// The stored credentials, when the context may use them.
    pub fn credentials(&self) -> Result<&CredentialStore, NotPermitted> {
        self.grants.require(Capability::Secrets)?;
        Ok(&self.credentials)
    }

    /// This context with its `file_*` tools working on the workspace on disk
    /// ([`VfsStore::direct`]) instead of through the overlay. Every other store
    /// is shared with `self` — sessions, notes and credentials stay one set,
    /// whichever way a round's files are handled.
    ///
    /// `Ok(None)` for a context with no workspace, which has no disk to work on;
    /// refused outright unless this context holds [`Capability::DiskWrite`].
    pub fn with_direct_files(&self) -> Result<Option<Self>, NotPermitted> {
        let grant = self.grants.disk_write()?;
        let Some(root) = self.vfs.workspace().map(PathBuf::from) else {
            return Ok(None);
        };
        Ok(Some(Self {
            vfs: Arc::new(VfsStore::direct(root, grant)),
            ..self.clone()
        }))
    }

    /// Attach the deployment's secrets, read once by the daemon at startup.
    pub fn with_secrets(mut self, secrets: ToolSecrets) -> Self {
        self.secrets = Arc::new(secrets);
        self
    }

    /// Attach a subagent runner to this context.
    pub fn with_subagent_runner(mut self, runner: Arc<dyn crate::SubagentRunner>) -> Self {
        self.subagent_runner = Some(runner);
        self
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_context_grants_nothing_and_cannot_reach_the_network() {
        let ctx = ToolContext::new();
        assert_eq!(ctx.grants(), Grants::NONE);
        assert!(ctx.http().is_err());
        assert_eq!(
            ctx.credentials().err(),
            Some(NotPermitted(Capability::Secrets))
        );
        assert!(ctx
            .clone()
            .granting(Grants::NONE.with(Capability::Secrets))
            .credentials()
            .is_ok());
        assert!(ctx
            .granting(Grants::NONE.with(Capability::Network))
            .http()
            .is_ok());
    }

    /// **A disk-writing store cannot be had without the grant.**
    #[test]
    fn direct_files_need_the_disk_write_grant() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::with_workspace(dir.path());
        assert_eq!(
            ctx.with_direct_files().err(),
            Some(NotPermitted(Capability::DiskWrite))
        );
        let ctx = ctx.granting(Grants::NONE.with(Capability::DiskWrite));
        let direct = ctx.with_direct_files().unwrap().expect("has a workspace");
        assert!(direct.vfs.is_direct());
        assert_eq!(direct.grants(), ctx.grants(), "grants carry over");
        assert!(ToolContext::new()
            .granting(Grants::ALL)
            .with_direct_files()
            .unwrap()
            .is_none());
    }
}
