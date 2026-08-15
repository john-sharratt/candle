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
//! | `credentials` | [`state::CredentialStore`] | Named auth material for session opens |
//! | `notes` | [`state::NotesStore`] | Cross-conversation persistent key-value store |
//! | `sessions` | [`state::SessionRegistry`] | All open protocol sessions (SSH, TCP, …) |
//! | `hash_states` | [`state::HashStateStore`] | Running hash contexts for `hash_state_*` tools |
//! | `http_client` | `reqwest::blocking::Client` | Shared HTTP client for `web_fetch`, `weather`, etc. |
//! | `subagent_runner` | `Option<Arc<dyn SubagentRunner>>` | Injected by daemon to run nested agent loops |
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

use crate::state::{CredentialStore, HashStateStore, NotesStore, SessionRegistry, VfsStore};

/// Read-only handle bundle passed by the runner into each tool invocation.
/// All stores are wrapped in `Arc` so cloning the context is cheap.
#[derive(Clone)]
pub struct ToolContext {
    pub vfs: Arc<VfsStore>,
    pub credentials: Arc<CredentialStore>,
    pub notes: Arc<NotesStore>,
    pub sessions: Arc<SessionRegistry>,
    pub hash_states: Arc<HashStateStore>,
    pub http_client: reqwest::blocking::Client,
    pub subagent_runner: Option<Arc<dyn crate::SubagentRunner>>,
}

impl ToolContext {
    /// Construct a context with default-initialized stores and no workspace
    /// layer — `file_*` tools see only what this session writes.
    pub fn new() -> Self {
        Self::build(VfsStore::new())
    }

    /// Construct a context whose VFS overlays `workspace`, the daemon's working
    /// directory: `file_*` reads fall through to real project files, writes and
    /// edits stay in memory.
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
            subagent_runner: None,
        }
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
