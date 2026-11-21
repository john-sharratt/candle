//! Gives the rest of the daemon access to a session's model, substrate,
//! cluster membership and roles.
//!
//! A caller that needs only the substrate, the cluster or the admin roles
//! reaches them through these accessors without holding the engine. The
//! `conversation` handle is a clone taken past which the engine lock is not
//! held, so a maintenance command never blocks a turn's commits. The
//! `shared_substrate` handle is published as soon as the redo log has replayed
//! and kept for the process's life, so reads work while the model still loads.
//! `hold_substrate` lets the `/mcp` tests run without a model.

use super::*;

impl ZendSession {
    /// Who is an admin — see [`crate::access`].
    pub fn roles(&self) -> &Roles {
        &self.config.roles
    }

    /// A handle that scores probes raw on the scheduler's GPU gallery arena
    /// without holding the engine, or `None` until the model is loaded.
    pub fn group_scanner(&self) -> Option<GroupScanner> {
        let state = self.inference.load_full()?;
        let scanner = state.engine.group_scanner();
        Some(scanner)
    }

    /// This process's membership of zend's cluster, when it was given one and
    /// joined: what the operations tools send their requests up
    /// (`docs/claude_zend_gateway.md` §8).
    pub fn cluster_member(&self) -> Option<Arc<AppMember>> {
        self.config.cluster.as_ref()?.link().member().cloned()
    }

    pub fn new(config: DaemonConfig) -> Self {
        let projection_builder = build_projection_builder(config.workspace.root());
        tracing::info!(
            workspace = %config.workspace.root().display(),
            repos = %config.workspace.names().join(", "),
            "session initialised",
        );
        let (ready_tx, _) = tokio::sync::watch::channel(false);
        let (status_tx, status_rx) = tokio::sync::watch::channel(String::new());
        let started_at_ms = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mcp = Arc::new(McpState::default());
        // The memory backbone, when the agent holds the journal repository —
        // and the workspace repository that is it, by its folder, so retrieval
        // and ingest keep the memory's boundary whatever it is called.
        if let Some(journal) = config
            .agent
            .as_ref()
            .and_then(|jobs| jobs.checkouts().journal())
        {
            mcp.memory().start(journal.dir.clone());
            let same = |dir: &Path| match (dir.canonicalize(), journal.dir.canonicalize()) {
                (Ok(a), Ok(b)) => a == b,
                _ => dir == journal.dir,
            };
            let named = config
                .workspace
                .repos()
                .iter()
                .find(|r| r.folder().is_some_and(same))
                .map_or(journal.name.as_str(), |r| r.name.as_str());
            memory_paths::name_repo(named);
        }
        Self {
            inference: Arc::new(ArcSwapOption::empty()),
            loading_substrate: Arc::new(OnceLock::new()),
            config,
            projection_builder,
            ready_tx,
            status_tx,
            status_rx,
            load_progress: Arc::new(LoadProgress::new()),
            live_turns: Arc::new(LiveTurns::default()),
            started_at_ms,
            load_thread: Mutex::new(None),
            ingest_worker: Mutex::new(None),
            mcp,
        }
    }

    pub(super) fn loaded(&self) -> Option<Arc<InferenceState>> {
        self.inference.load_full()
    }

    /// What `/mcp` holds in memory.
    pub fn mcp(&self) -> &Arc<McpState> {
        &self.mcp
    }

    /// The substrate the model's conversations live in, for records kept
    /// beside them (`crate::mcp::records`). The load publishes the engine's
    /// conversation handle as soon as the redo log has replayed and keeps it
    /// for the process's life, so a caller can read while zend is still
    /// loading. `None` before that.
    pub fn shared_substrate(&self) -> Option<SharedSubstrate> {
        let conv = self.loading_substrate.get().cloned()?;
        Some(conv.shared())
    }

    /// Hold `conv` as the substrate the load would have opened: what a test
    /// of the `/mcp` tools runs on without a model.
    #[cfg(test)]
    pub(crate) fn hold_substrate(&self, conv: Conversation) {
        let _ = self.loading_substrate.set(conv);
    }

    /// Whether the model and its substrate are loaded: what a conversation's
    /// files need.
    pub fn is_loaded(&self) -> bool {
        self.loaded().is_some()
    }

    /// The agents this daemon's commands, pushes and fetches run on
    /// (`docs/replicated_substrate.md` §11.5, `docs/substrate_git.md` §9): the
    /// cluster's, or with no cluster the one in this process. `None` with
    /// neither.
    pub fn agents(&self) -> Option<Agents> {
        substrate_git::agents(self.config.cluster.as_ref(), self.config.agent.as_ref())
    }

    /// The substrate, once the model is loaded: what the host's maintenance
    /// commands run on. A cloned handle, taken without holding the engine
    /// lock past the clone, so a command never blocks a turn's commits.
    pub fn conversation(&self) -> Option<Conversation> {
        let state = self.inference.load_full()?;
        let conv = state.engine.conversation();
        Some(conv)
    }
}
