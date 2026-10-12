//! A member's fetches, made by an agent (§10.3): the [`Fetch`] a repository
//! that lives with an agent is opened [`fetching through`](Repo::fetching_through).
//! The member names the commits its tracking refs already hold, and an agent
//! holding the repository fetches its origin and answers with what origin
//! holds and a bundle of the rest, which the member takes in as its own
//! tracking refs.
//!
//! A fetch is made from a thread that may block; the job itself runs on the
//! member's runtime, where its socket lives.

use std::sync::mpsc;
use std::sync::Arc;

use tokio::runtime::Handle;
use zend_vfs::{Fetch, FetchSpec, GitError, RefUpdate, RemoteName, Repo};

use crate::client::Agents;
use crate::wire::FetchJob;

/// One repository's fetches, made by an agent.
pub struct AgentFetch {
    agents: Agents,
    runtime: Handle,
    /// The repository's name, which the agent holds it under.
    repo: String,
}

impl AgentFetch {
    /// The fetches of the repository the agents hold as `repo`, made through
    /// `agents` on `runtime`.
    pub fn shared(agents: Agents, runtime: Handle, repo: impl Into<String>) -> Arc<dyn Fetch> {
        Arc::new(AgentFetch {
            agents,
            runtime,
            repo: repo.into(),
        })
    }
}

/// The agent could not be reached or would not fetch: the remote is
/// unreachable from here, which a publish waits out and the record's follow
/// tries again (§10.3).
fn failed(remote: &RemoteName, why: String) -> GitError {
    GitError::RemoteUnreachable {
        remote: remote.clone(),
        detail: why,
    }
}

impl Fetch for AgentFetch {
    fn fetch(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        spec: &FetchSpec,
    ) -> Result<Vec<RefUpdate>, GitError> {
        let have = repo
            .tracking_tips(remote)?
            .into_iter()
            .map(|o| o.to_string())
            .collect();
        let job = FetchJob {
            repo: self.repo.clone(),
            branch: match spec {
                FetchSpec::Branch(b) => Some(b.as_str().to_string()),
                FetchSpec::AllBranches => None,
            },
            have,
        };
        let agents = self.agents.clone();
        let (tx, rx) = mpsc::channel();
        self.runtime.spawn(async move {
            let _ = tx.send(agents.fetch(job).await);
        });
        let fetched = rx
            .recv()
            .map_err(|_| failed(remote, "the fetch's job was dropped".into()))?
            .map_err(|e| failed(remote, e.to_string()))?;
        repo.take_fetch(remote, spec, &fetched)
    }
}
