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
//! | `files` | [`RepoFiles`] | Overlay filesystem for `file_*` tools — one store per repository, one conversation's changes over each |
//! | `credentials` | [`state::CredentialStore`] | Named auth material for session opens — reached only through [`ToolContext::credentials`] |
//! | `notes` | [`state::NotesStore`] | Cross-conversation persistent key-value store |
//! | `sessions` | [`state::SessionRegistry`] | All open protocol sessions (SSH, TCP, …) |
//! | `hash_states` | [`state::HashStateStore`] | Running hash contexts for `hash_state_*` tools |
//! | `http_client` | `reqwest::blocking::Client` | Shared HTTP client for `web_fetch`, `weather`, etc. — reached only through [`ToolContext::http`] |
//! | `secrets` | [`state::Secrets`] | The daemon's API keys and tokens (Tavily, GitHub), read once from a per-user file |
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
//! passing its [`Workspace`] so the `file_*` tools resolve real project files in
//! each repository through the VFS overlay, and gives each conversation its own
//! overlay with [`ToolContext::with_files`]. [`ToolContext::new`] leaves every
//! repository's store upper-only ([`RepoFiles::detached`]), which is what most
//! tests want; a test needing the lower layer builds a `Workspace` over a temp
//! dir.

use std::sync::Arc;

use crate::grants::{Capability, Grants, NotPermitted};
use crate::state::{CredentialStore, HashStateStore, NotesStore, Secrets, SessionRegistry};
use zend_vfs::{RepoFiles, Workspace};

/// Read-only handle bundle passed by the runner into each tool invocation.
/// All stores are wrapped in `Arc` so cloning the context is cheap.
#[derive(Clone)]
pub struct ToolContext {
    pub files: Arc<RepoFiles>,
    credentials: Arc<CredentialStore>,
    pub notes: Arc<NotesStore>,
    pub sessions: Arc<SessionRegistry>,
    pub hash_states: Arc<HashStateStore>,
    http_client: reqwest::blocking::Client,
    pub secrets: Arc<Secrets>,
    pub subagent_runner: Option<Arc<dyn crate::SubagentRunner>>,
    grants: Grants,
}

impl ToolContext {
    /// Construct a context with default-initialized stores, no workspace —
    /// `file_*` tools see only what this session writes, in whatever
    /// repository it names — and no grants.
    pub fn new() -> Self {
        Self::build(RepoFiles::detached())
    }

    /// Construct a context whose file stores overlay `workspace`'s
    /// repositories: `file_*` reads fall through to real project files, writes
    /// and edits stay in memory. Grants nothing.
    pub fn with_workspace(workspace: Workspace) -> Self {
        Self::build(RepoFiles::overlay(workspace))
    }

    fn build(files: RepoFiles) -> Self {
        Self {
            files: Arc::new(files),
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
            secrets: Arc::new(Secrets::empty()),
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

    /// This context with `files` as its file stores — one conversation's own,
    /// so the changes its tool calls make are its alone. Every other store is
    /// shared with `self`.
    pub fn with_files(&self, files: Arc<RepoFiles>) -> Self {
        Self {
            files,
            ..self.clone()
        }
    }

    /// This context with its `file_*` tools working on each repository on disk
    /// ([`RepoFiles::direct`]) instead of through the overlay. Every other
    /// store is shared with `self` — sessions, notes and credentials stay one
    /// set, whichever way a round's files are handled.
    ///
    /// `Ok(None)` for a context with no workspace, which has no disk to work on;
    /// refused outright unless this context holds [`Capability::DiskWrite`].
    pub fn with_direct_files(&self) -> Result<Option<Self>, NotPermitted> {
        let grant = self.grants.disk_write()?;
        let Some(workspace) = self.files.workspace().cloned() else {
            return Ok(None);
        };
        Ok(Some(Self {
            files: Arc::new(RepoFiles::direct(workspace, grant)),
            ..self.clone()
        }))
    }

    /// Attach the daemon's secrets, read once at startup and shared by every
    /// context it builds.
    pub fn with_secrets(mut self, secrets: Arc<Secrets>) -> Self {
        self.secrets = secrets;
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

    use zend_vfs::RepoSpec;

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
        let workspace = Workspace::new(dir.path(), vec![RepoSpec::named("r")]).unwrap();
        let ctx = ToolContext::with_workspace(workspace);
        assert_eq!(
            ctx.with_direct_files().err(),
            Some(NotPermitted(Capability::DiskWrite))
        );
        let ctx = ctx.granting(Grants::NONE.with(Capability::DiskWrite));
        let direct = ctx.with_direct_files().unwrap().expect("has a workspace");
        assert!(direct.files.is_direct());
        assert!(direct.files.repo("r").unwrap().is_direct());
        assert_eq!(direct.grants(), ctx.grants(), "grants carry over");
        assert!(ToolContext::new()
            .granting(Grants::ALL)
            .with_direct_files()
            .unwrap()
            .is_none());
    }
}
