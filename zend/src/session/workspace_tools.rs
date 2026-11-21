//! Gives the connector's repository tools a view of the workspace, either as
//! committed or as a conversation left it.
//!
//! The committed view belongs to no conversation, so its pushes may go to any
//! branch. Those pushes are gated and announced as every push is, and the
//! memory's repository is hidden from anyone below an admin. A conversation's
//! view is rebuilt by replaying its file events, with each repository on the
//! conversation's branch and its uncommitted changes. Building it writes
//! nothing, and the repositories that cannot be restored are reported with why.

use super::*;

/// A context over a conversation's files as it left them, and the
/// repositories that could not be put back, each with why
/// ([`ZendSession::conversation_tools`]).
pub type ConversationTools = (Arc<ToolContext>, Vec<(String, String)>);

impl ZendSession {
    /// The workspace folder — where `substrate/` lives.
    pub fn workspace_root(&self) -> &Path {
        self.config.workspace.root()
    }

    /// The workspace's repositories by name, the uploads store left out: the
    /// projects a `P:<repo>` journal can be kept for.
    pub fn repository_names(&self) -> Vec<String> {
        self.config
            .workspace
            .repos()
            .iter()
            .map(|r| r.name.clone())
            .filter(|name| name != UPLOADS_REPO)
            .collect()
    }

    /// What the connector's repository tools work with
    /// (`docs/mcp_file_tools.md` §2–§3): a context over the workspace as it is
    /// committed — no conversation's changes — whose pushes are made, gated
    /// and announced as every push of Zen's is, and each repository's base
    /// branch. It belongs to no conversation, so it pushes to any branch.
    /// For a caller below an admin the memory's repository is not among them
    /// ([`memory_hidden`]): an admin's alone to read or commit file by file.
    /// `None` while the model is not loaded.
    pub fn repository_tools(&self, role: AccessRole) -> Option<(Arc<ToolContext>, BaseBranches)> {
        let state = self.loaded()?;
        let files = state.tool_host.conversation_files();
        let files = match role.at_least(AccessRole::Admin) {
            true => files,
            false => Arc::new(memory_hidden(&files)),
        };
        let ctx = state
            .tool_host
            .context_for(ToolMode::Comprehensive, &files, "");
        Some((ctx, state.base_branches.clone()))
    }

    /// What the connector's repository tools read a conversation's files
    /// through (`docs/mcp_file_tools.md` §2.1): a context over the
    /// repositories as `conv_id` left them — each on the conversation's
    /// branch, with its uncommitted changes — made by replaying its file
    /// events, which writes nothing. Answers the repositories that could not
    /// be put back, each with why. The caller has already been checked against
    /// the conversation (`ownership`); the memory's repository is hidden from
    /// a caller below an admin, as [`Self::repository_tools`] hides it.
    /// `None` while the model is not loaded.
    pub fn conversation_tools(
        &self,
        role: AccessRole,
        conv_id: &str,
    ) -> Option<Result<ConversationTools, String>> {
        let state = self.loaded()?;
        let files = state.tool_host.conversation_files();
        let refused =
            match conv_overlay::view(&state.engine, conversation_timeline(conv_id), &files) {
                Ok(refused) => refused,
                Err(e) => return Some(Err(e)),
            };
        let files = match role.at_least(AccessRole::Admin) {
            true => files,
            false => Arc::new(memory_hidden(&files)),
        };
        let ctx = state
            .tool_host
            .context_for(ToolMode::Comprehensive, &files, "");
        Some(Ok((ctx, refused)))
    }
}
