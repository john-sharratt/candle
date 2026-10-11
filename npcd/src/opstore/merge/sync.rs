//! `merge_sync`: the record's newer commit brought into an operation's store,
//! each overlap marked in the document for the Maker to settle at the bench.

use zend_vfs::{merge_into, GitError, Merged};

use super::{unreachable, Merge, RECORD};

/// What a sync came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Synced {
    /// The store already has everything the record holds.
    UpToDate,
    /// The record's commit is in the store; `conflicts` hold its marks.
    Merged { conflicts: Vec<String> },
    /// `origin` could not be reached; nothing changed.
    Unreachable(String),
    /// It could not be merged, for this reason; nothing changed.
    Refused(String),
}

impl Merge<'_> {
    /// Bring the record's commit into the store, as `git merge` brings a
    /// branch into a working tree — but into the store, never a folder.
    pub fn sync(&self) -> Synced {
        let failed = |e: GitError| match unreachable(&e) {
            true => Synced::Unreachable(e.to_string()),
            false => Synced::Refused(e.to_string()),
        };
        let record = match self.record() {
            Ok(Some(record)) => record,
            Ok(None) => return Synced::UpToDate,
            Err(e) => return failed(e),
        };
        match merge_into(&self.repo, self.store, &record, RECORD) {
            Ok(Merged::UpToDate) => Synced::UpToDate,
            Ok(merged) => Synced::Merged {
                conflicts: merged.conflicts().to_vec(),
            },
            Err(e) => failed(e),
        }
    }
}
