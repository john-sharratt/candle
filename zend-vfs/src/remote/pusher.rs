//! Where a repository's pushes are made (`docs/replicated_substrate.md`
//! §11.5). Every push to a remote — a commit landing, a branch published or
//! removed — goes through [`Repo::push`], which hands it to the repository's
//! [`Push`]. A daemon on its own pushes from its own checkout ([`Here`]); a
//! cluster member holds no push credentials and hands each push to an agent,
//! which pushes the same specs under the same leases from its checkout.

use std::sync::Arc;

use crate::error::GitError;
use crate::remote::push::{PushResult, PushSpec};
use crate::types::RemoteName;
use crate::Repo;

/// What makes a repository's pushes.
pub trait Push: Send + Sync {
    /// Push `specs` from `repo` to `remote` in one atomic push, as
    /// [`Repo::push_here`] does.
    fn push(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError>;
}

/// Pushing from this checkout, with this machine's credentials.
pub struct Here;

impl Push for Here {
    fn push(
        &self,
        repo: &Repo,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Vec<PushResult>, GitError> {
        repo.push_here(remote, specs)
    }
}

/// [`Here`], shared.
pub fn here() -> Arc<dyn Push> {
    Arc::new(Here)
}
