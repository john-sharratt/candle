//! A member's pushes, made by an agent (§11.5): the [`Push`] a repository is
//! opened [`pushing through`](zend_vfs::Repo::pushing_through) when its daemon
//! runs with `--cluster`. The member bundles what origin lacks of the push and
//! an agent holding the repository with push rights pushes the same specs
//! under the same leases.
//!
//! A push is made from a tool's own thread, which may block; the job itself
//! runs on the member's runtime, where its socket lives.

use std::io;
use std::sync::mpsc;
use std::sync::Arc;

use tokio::runtime::Handle;
use zend_vfs::{GitError, Push, PushResult, PushSpec, RemoteName, Repo};

use crate::client::Agents;
use crate::wire::{encode_bundle, PushJob, PushLine};

/// One repository's pushes, made by an agent.
pub struct AgentPush {
    agents: Agents,
    runtime: Handle,
    /// The repository's name, which the agent holds it under.
    repo: String,
}

impl AgentPush {
    /// The pushes of the repository the agents hold as `repo`, made through
    /// `agents` on `runtime`.
    pub fn shared(agents: Agents, runtime: Handle, repo: impl Into<String>) -> Arc<dyn Push> {
        Arc::new(AgentPush {
            agents,
            runtime,
            repo: repo.into(),
        })
    }
}

fn failed(why: String) -> GitError {
    GitError::Io(io::Error::other(why))
}

impl Push for AgentPush {
    fn push(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        let bundle = repo.bundle_for_push(remote, specs)?;
        let job = PushJob {
            repo: self.repo.clone(),
            bundle: bundle.as_deref().map(encode_bundle),
            specs: specs.iter().map(PushLine::of).collect(),
        };
        let agents = self.agents.clone();
        let (tx, rx) = mpsc::channel();
        self.runtime.spawn(async move {
            let _ = tx.send(agents.push(job).await);
        });
        rx.recv()
            .map_err(|_| failed("the push's job was dropped".into()))?
            .map_err(|e| failed(e.to_string()))
    }
}
