//! A repository's first packs, from a checkout a member already has
//! (`docs/substrate_git.md` §9).
//!
//! The checkout's view of `origin` — its `refs/remotes/origin/*` — becomes the
//! log's branches; a checkout with no `origin` gives its own branches instead.
//! A log that already names a branch of the repository has been adopted, by
//! this member or another, and is left alone: the importer moves it from
//! there.

use zend_vfs::{Oid, RefName, Repo};

use crate::publish::{publish, Published};
use crate::records::{Log, Records};

const TRACKING: &str = "refs/remotes/origin/";

/// Publish `source`'s branches as repository `repo`'s, unless the log already
/// holds one.
pub fn adopt(records: &dyn Records, repo: &str, source: &Repo) -> anyhow::Result<Published> {
    if !Log::new(records, repo)?.refs()?.is_empty() {
        return Ok(Published::default());
    }
    let mut refs: Vec<(RefName, Option<Oid>)> = Vec::new();
    for (name, oid) in source.refs_in(TRACKING)? {
        let branch = &name.as_str()[TRACKING.len()..];
        if branch != "HEAD" {
            refs.push((RefName::parse(&format!("refs/heads/{branch}"))?, Some(oid)));
        }
    }
    if refs.is_empty() {
        refs = source
            .published_refs()?
            .into_iter()
            .filter(|(name, _)| name.as_str().starts_with("refs/heads/"))
            .map(|(name, oid)| (name, Some(oid)))
            .collect();
    }
    publish(records, repo, source, &refs, false)
}
