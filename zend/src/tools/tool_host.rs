//! Holds the daemon's tool contexts, one per mode, and builds the context a
//! round runs in for a conversation.
//!
//! One context per tools mode shares every store and differs only in its grants,
//! so what a round may do is fixed by the context it is handed rather than by
//! the tools its prompt offered. Each conversation has its own file stores,
//! layered over the workspace's repositories read-only. A conversation owned
//! through `/mcp` may push only to branches, since each repository's default
//! branch is closed to it. The memory's repository is visible only to the
//! comprehensive mode, so a person's dialogue reads memory through the memory's
//! own tools.

use super::*;

impl ToolHost {
    /// What decides which tools are out of a round
    /// ([`crate::tool_availability`]): every context shares these stores.
    pub fn availability_facts(&self) -> Facts {
        let context = &self.contexts[ToolMode::Comprehensive.level() as usize];
        Facts {
            open: context.sessions.open(),
            search_key: context.secrets.tavily_api_key().is_some(),
        }
    }

    /// Build a host whose file tools overlay `workspace`'s repositories, which
    /// reads fall through to when the session layer has no entry, whose every
    /// context carries `secrets` — read once by the daemon at launch
    /// ([`crate::secrets::load`]) — and whose commands run in a sandbox per git
    /// repository, starting the programs [`programs`] lists. Fails when the
    /// workspace's jobs folder cannot be made, or a repository opened.
    pub fn new(workspace: &Workspace, secrets: Arc<Secrets>) -> io::Result<Self> {
        let sandboxes = Sandboxes::for_workspace(workspace, &programs::policy())?;
        Self::on(workspace, secrets, Arc::new(sandboxes)).map_err(io::Error::other)
    }

    /// The host this daemon runs with: commands and pushes on agents — the
    /// cluster's when it is a member of `cluster` (§11.5), reaching the agent
    /// in its own process while cut off, else the one in its own process. A
    /// workspace naming no repository needs no agent: its commands run in
    /// [`ToolHost::new`]'s sandboxes.
    pub fn deployed(
        workspace: &Workspace,
        secrets: Arc<Secrets>,
        cluster: Option<&Join>,
        agent: Option<&Arc<Jobs>>,
    ) -> anyhow::Result<Self> {
        // Every command and push on an agent — the cluster's, or the one in
        // this process alone (`docs/substrate_git.md` §9): a repository is
        // a store with no working tree, which nothing runs in.
        let Some(agents) = agents_for(cluster, agent) else {
            if cluster.is_some() {
                anyhow::bail!(
                    "commands and pushes run on the cluster's agents, and its member never started"
                );
            }
            let repos = workspace.repos().iter().any(|r| r.name != UPLOADS_REPO);
            if repos {
                anyhow::bail!(
                    "commands and pushes run on an agent, and there is none: give the process \
                     an `agent:` file"
                );
            }
            return Ok(Self::new(workspace, secrets)?);
        };
        let executor =
            AgentExecutor::new(agents, Handle::current(), workspace.root().join(JOBS_DIR))
                .gating(workspace);
        Ok(Self::on(workspace, secrets, Arc::new(executor))?)
    }

    /// A host whose commands run, and whose pushes are made, by `executor` —
    /// agents, for a daemon given `--cluster` (§11.5). Refused when a
    /// repository of `workspace` cannot be opened.
    pub fn on(
        workspace: &Workspace,
        secrets: Arc<Secrets>,
        executor: Arc<dyn Executor>,
    ) -> Result<Self, GitError> {
        let base = ToolContext::with_workspace(workspace.clone())?
            .with_secrets(secrets)
            .with_executor(executor);
        let contexts =
            ToolMode::ALL.map(|mode| Arc::new(base.clone().granting(access::grants(mode))));
        Ok(Self {
            contexts,
            default_branches: Arc::default(),
        })
    }

    /// This host with `runner` running every context's build and deploy tools
    /// (`docs/jobs.md` §8).
    pub fn running_ops(self, runner: Arc<dyn OpsRunner>) -> Self {
        Self {
            contexts: self.contexts.map(|ctx| {
                Arc::new(Arc::unwrap_or_clone(ctx).with_ops_runner(Arc::clone(&runner)))
            }),
            ..self
        }
    }

    /// This host closing each repository's base branch in `bases`, and the
    /// [`CONVENTIONAL_DEFAULTS`], to the pushes of every conversation owned
    /// through `/mcp`. A repository `bases` does not name takes no push from
    /// such a conversation.
    pub fn guarding_defaults(self, bases: &BTreeMap<String, String>) -> Self {
        let closed = bases
            .iter()
            .map(|(repo, base)| {
                let mut names: BTreeSet<String> = CONVENTIONAL_DEFAULTS
                    .iter()
                    .map(|n| n.to_string())
                    .collect();
                names.insert(base.clone());
                (repo.clone(), names)
            })
            .collect();
        Self {
            default_branches: Arc::new(closed),
            ..self
        }
    }

    /// The context of `mode` for the conversation `lineage`, with its file
    /// stores `files`: branches-only when an `/mcp` caller owns it.
    pub(super) fn bound(
        &self,
        mode: ToolMode,
        files: Arc<RepoFiles>,
        lineage: &str,
    ) -> ToolContext {
        let ctx = self.contexts[mode.level() as usize].with_files(files, lineage);
        if owner_of(lineage).is_some() {
            ctx.branches_only(Arc::clone(&self.default_branches))
        } else {
            ctx
        }
    }

    /// A new conversation's own file stores: an overlay over each repository
    /// with no changes yet.
    pub fn conversation_files(&self) -> Arc<RepoFiles> {
        Arc::new(
            self.contexts[ToolMode::Restricted.level() as usize]
                .files
                .fresh(),
        )
    }

    /// The context a round of tools in `mode` runs in, for the conversation
    /// `lineage` whose file stores are `files`.
    pub fn context_for(
        &self,
        mode: ToolMode,
        files: &Arc<RepoFiles>,
        lineage: &str,
    ) -> Arc<ToolContext> {
        Arc::new(self.bound(mode, Arc::clone(files), lineage))
    }

    /// [`Self::context_for`] a dialogue's round, whose journal tools read
    /// `journals`.
    ///
    /// **The memory's repository is an admin's to read file by file.** Below
    /// [`ToolMode::Comprehensive`] — the mode only an admin holds — a
    /// dialogue's stores are a view without it ([`memory_hidden`]), so a
    /// person's conversation, and every task it starts, reads the memory
    /// through the memory's tools and its retrieval scope alone, never
    /// another group's journal by path or by `*`. The ingest's own contexts
    /// ([`Self::context_for`]) read the memory whole: reading it is how it
    /// enters the substrate.
    pub fn dialogue_context(
        &self,
        mode: ToolMode,
        files: &Arc<RepoFiles>,
        lineage: &str,
        journals: Arc<dyn JournalShelf>,
    ) -> Arc<ToolContext> {
        let files = match mode {
            ToolMode::Comprehensive => Arc::clone(files),
            _ => Arc::new(memory_hidden(files)),
        };
        Arc::new(self.bound(mode, files, lineage).with_journals(journals))
    }

    /// Close every tool session the conversation `lineage` opened — sockets,
    /// HTTP sessions and the credentials they carry, SSH, SQL, code — for a
    /// conversation that is archived or deleted and so never runs another
    /// round. Answers how many it closed. Every mode's context shares one
    /// registry.
    pub fn close_sessions(&self, lineage: &str) -> usize {
        self.contexts[ToolMode::Restricted.level() as usize]
            .sessions
            .close_conversation(lineage)
    }
}
