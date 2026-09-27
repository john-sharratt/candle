//! Reading back what a tool run changed on a checkout, as the conversation's
//! next deltas.
//!
//! After [`materialize`](super::materialize) the checkout holds a
//! conversation's state on its branch, and its [`Ledger`] says what that state
//! is beyond the branch's commit. A tool then runs and changes what it
//! changes — possibly the checkout's git state too: a tool that stages,
//! commits, switches branch or deletes the branch. First, then, the branch and
//! `HEAD` are put back where the run started ([`reclaim`](super::reclaim)), so
//! whatever the tool did is a difference in the index and the working tree
//! against that commit. Capture then looks at exactly the files that could
//! differ from the conversation's state:
//!
//! - every file `git status` reports against the base — which covers anything
//!   the tool changed, added, deleted, staged or committed among the files git
//!   sees;
//! - every file the ledger names — which covers a file the tool put back to
//!   the base's content (status is then silent on it, but the conversation had
//!   it changed) and an ignored file the conversation wrote.
//!
//! A file whose stamp the ledger vouches for is not read at all. Every other
//! one is read and compared with what the conversation had — the ledger's
//! content, or the base's — and a difference becomes one delta
//! ([`file_delta::between`]) made against the conversation's own state, ready
//! to append to its changes. The ledger is brought up to what is on disk, so
//! the next [`materialize`](super::materialize) starts from the truth.
//!
//! A file the tool replaced with a folder or a link, or put behind a link, is
//! gone as far as the conversation is concerned: nothing is read through a
//! link. The files inside a new folder are paths of their own. Ignored files
//! the conversation never wrote — build outputs, caches — are not captured:
//! they are the tool's by-products, not the conversation's changes.

use std::collections::BTreeSet;

use super::error::CheckoutError;
use super::ledger::Ledger;
use super::materialize::{read, stamp};
use super::reclaim::reclaim;
use super::target::{self, Found};
use crate::file_delta::{self, TimedDelta};
use crate::{BranchName, Oid, Repo, RepoPath, Rev, StatusEntry};

/// What changed on the checkout since its ledger was taken, as one delta per
/// changed path, in path order, each against the conversation's content for
/// that path and stamped with the moment it was computed. `branch` is the
/// branch the checkout was put on; it and `HEAD` are put back at the ledger's
/// commit first. `ledger` is updated to what the checkout now holds.
pub fn capture(
    repo: &Repo,
    branch: &BranchName,
    ledger: &mut Ledger,
) -> Result<Vec<(String, TimedDelta)>, CheckoutError> {
    let root = repo.dir().to_path_buf();
    let base_oid = Oid::parse(ledger.base())?;
    let (head, mut status) = repo.status_with_head()?;
    if reclaim(repo, branch, &base_oid, &head)? {
        // Measured against the commit the run started at, not wherever the
        // tool left `HEAD`.
        status = repo.status()?;
    }
    let base = Rev::Oid(base_oid);

    let mut candidates: BTreeSet<String> = ledger.paths().map(str::to_string).collect();
    // Paths git reports untracked and nothing else: the base holds no copy of
    // them. (`git rm --cached` makes a path both a staged delete and
    // untracked — the base has that one.)
    let mut untracked: BTreeSet<String> = BTreeSet::new();
    let mut tracked: BTreeSet<String> = BTreeSet::new();
    for entry in status {
        if let StatusEntry::Renamed { from, .. } = &entry {
            candidates.insert(from.as_str().to_string());
            tracked.insert(from.as_str().to_string());
        }
        let path = entry.path().as_str().to_string();
        if matches!(entry, StatusEntry::Untracked { .. }) {
            untracked.insert(path.clone());
        } else {
            tracked.insert(path.clone());
        }
        candidates.insert(path);
    }
    untracked.retain(|path| !tracked.contains(path));

    // What each changed path holds now.
    let mut changed = Vec::new();
    for path in candidates {
        let (target, found) = target::inspect(&root, &path)?;
        let (now_stamp, now) = match found {
            Found::File => {
                let now_stamp = stamp(&target.abs, &path)?;
                if ledger.verified(&path, now_stamp).is_some() {
                    continue;
                }
                (now_stamp, read(&target.abs, &path)?)
            }
            // No file there — or a folder or a link in its place, or a link
            // on the way to it, none of which is a file of the conversation's.
            _ => (None, None),
        };
        changed.push((path, target, now_stamp, now));
    }
    // What the conversation had: the ledger's content, or the base's — every
    // base copy read in one pass, none for an untracked path.
    let from_base: Vec<&RepoPath> = changed
        .iter()
        .filter(|(path, ..)| ledger.entry(path).is_none() && !untracked.contains(path))
        .map(|(_, target, ..)| &target.repo_path)
        .collect();
    let mut at_base = repo.read_checked_out_all(&base, &from_base)?;

    let mut captured = Vec::new();
    for (path, target, now_stamp, now) in changed {
        let before = match ledger.entry(&path) {
            Some(entry) => entry.content.clone(),
            None => at_base.remove(target.repo_path.as_str()),
        };
        if let Some(delta) = file_delta::between(before.as_deref(), now.as_deref()) {
            captured.push((path.clone(), TimedDelta::now(delta)));
        }
        ledger.record(path, now, now_stamp);
    }
    ledger.seal();
    Ok(captured)
}
