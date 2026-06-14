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
//! | `vfs` | [`state::VfsStore`] | In-memory virtual filesystem for `file_*` tools |
//! | `credentials` | [`state::CredentialStore`] | Named auth material for session opens |
//! | `notes` | [`state::NotesStore`] | Cross-conversation persistent key-value store |
//! | `sessions` | [`state::SessionRegistry`] | All open protocol sessions (SSH, TCP, …) |
//! | `hash_states` | [`state::HashStateStore`] | Running hash contexts for `hash_state_*` tools |
//! | `http_client` | `reqwest::blocking::Client` | Shared HTTP client for `web_fetch`, `weather`, etc. |
//! | `subagent_runner` | `Option<Arc<dyn SubagentRunner>>` | Injected by daemon to run nested agent loops |
//!
//! # Construction
//!
//! In production the daemon calls `ToolContext::new()` once at startup.
//! In tests each test case typically calls `ToolContext::new()` for isolation,
//! or shares a context explicitly for lifecycle tests (open → use → close).

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
    /// Construct a context with default-initialized stores.
    /// Suitable for tests and for one-shot daemon startup.
    pub fn new() -> Self {
        Self {
            vfs: Arc::new(VfsStore::new()),
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
