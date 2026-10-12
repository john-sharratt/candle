//! Where a repository's fetches are made (`docs/replicated_substrate.md`
//! §10.3, §11.4). Every fetch from a remote — a pull, a publish reading the
//! record, the mind following it — goes through [`Repo::fetch`], which hands it
//! to the repository's [`Fetch`]. A daemon on its own fetches from its own
//! checkout's remote ([`FetchHere`]); a cluster member whose repository lives
//! with an agent asks the agent, which fetches its origin and bundles back what
//! the member lacks ([`Repo::bundle_for_fetch`], [`Repo::take_fetch`]).

use std::sync::Arc;

use crate::error::GitError;
use crate::remote::fetch::{FetchSpec, RefUpdate};
use crate::types::RemoteName;
use crate::Repo;

/// What makes a repository's fetches.
pub trait Fetch: Send + Sync {
    /// Fetch `spec` from `remote` into `repo`'s remote-tracking refs, as
    /// [`Repo::fetch_here`] does, and report what changed.
    fn fetch(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        spec: &FetchSpec,
    ) -> Result<Vec<RefUpdate>, GitError>;
}

/// Fetching from this checkout's remote, over this machine's network and
/// credentials.
pub struct FetchHere;

impl Fetch for FetchHere {
    fn fetch(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        spec: &FetchSpec,
    ) -> Result<Vec<RefUpdate>, GitError> {
        repo.fetch_here(remote, spec)
    }
}

/// [`FetchHere`], shared.
pub fn fetch_here() -> Arc<dyn Fetch> {
    Arc::new(FetchHere)
}
