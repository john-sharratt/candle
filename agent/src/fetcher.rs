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
use zend_vfs::{
    Fetch, FetchSpec, GitError, Oid, RefName, RefOp, RefTransaction, RefUpdate, RemoteName, Repo,
    ORIGIN,
};

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
        // Everything this repository already holds of origin's history: the
        // fetching remote's tracking refs, the record's — a store's
        // `refs/remotes/origin/*`, which the log's packs filled — and the
        // slices a run cut short already took.
        let mut slices: Vec<(RefName, Oid)> = repo.refs_in(SLICES)?;
        let mut have: Vec<String> = Vec::new();
        let held = repo
            .tracking_tips(remote)?
            .into_iter()
            .chain(repo.tracking_tips(&RemoteName::parse(ORIGIN)?)?)
            .chain(slices.iter().map(|(_, o)| o.clone()));
        for oid in held {
            let oid = oid.to_string();
            if !have.contains(&oid) {
                have.push(oid);
            }
        }
        // A long history comes in slices: each one's objects are taken and its
        // commits named as held, and only the last answer moves the tracking
        // refs, so they never stand anywhere origin's branches do not. Each
        // slice's commits are kept under `SLICES` too, so a run that fails part
        // way resumes where it stopped; the last answer drops them.
        loop {
            let job = FetchJob {
                repo: self.repo.clone(),
                branch: match spec {
                    FetchSpec::Branch(b) => Some(b.as_str().to_string()),
                    FetchSpec::AllBranches => None,
                },
                have: have.clone(),
                max_commits: Some(SLICE),
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
            if !fetched.partial {
                let taken = repo.take_fetch(remote, spec, &fetched)?;
                drop_slices(repo, &slices);
                return Ok(taken);
            }
            let Some(bundle) = &fetched.bundle else {
                return Err(failed(remote, "a partial answer carried no objects".into()));
            };
            repo.take_bundle(bundle)?;
            let mut txn = RefTransaction::new();
            for tip in fetched.tips.values() {
                let named = tip.to_string();
                if have.contains(&named) {
                    continue;
                }
                have.push(named);
                let slice = RefName::parse(&format!("{SLICES}{tip}"))?;
                if !slices.iter().any(|(n, _)| *n == slice) {
                    txn = txn.push(RefOp::Create {
                        name: slice.clone(),
                        new: tip.clone(),
                    });
                    slices.push((slice, tip.clone()));
                }
            }
            repo.update_refs(&txn)?;
        }
    }
}

/// Where a sliced fetch keeps the commits its slices took, one ref per commit,
/// named by it, until its last answer lands.
const SLICES: &str = "refs/substrate-slices/";

/// Drop a fetch's slice refs once its tracking refs reach past them. Left
/// behind they only name commits already held, so failing here fails no
/// fetch.
fn drop_slices(repo: &Repo, slices: &[(RefName, Oid)]) {
    let mut txn = RefTransaction::new();
    for (name, old) in slices {
        txn = txn.push(RefOp::Delete {
            name: name.clone(),
            old: old.clone(),
        });
    }
    if let Err(e) = repo.update_refs(&txn) {
        tracing::warn!("dropping a fetch's slice refs: {e}");
    }
}

/// The most commits one answer carries. A first fetch of candle's history —
/// 6,500 commits, 1.3 GB packed — comes in about 26 answers of this.
const SLICE: u32 = 256;
