//! `merge_publish`: an operation's store landed on the record as one commit,
//! or refused with why.

use zend_vfs::{
    merge_into, ChangeSet, Committing, FileState, GitError, NotCommitted, Oid, Rejection, RepoPath,
    Signature,
};

use super::{unreachable, Merge, RECORD};

/// What a publish came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Publish {
    /// The store is on the record as `commit`, which changed `paths`.
    Landed { commit: String, paths: Vec<String> },
    /// The store changes nothing.
    Nothing,
    /// The record moved under the store at lines it changes: these paths
    /// overlap, and nothing changed.
    Conflicts(Vec<String>),
    /// These paths still hold a merge's marks, to be settled at the bench.
    Unsettled(Vec<String>),
    /// `origin`, or the agent pushing to it, could not be reached; nothing
    /// changed and the publish waits (§10.3).
    Unreachable(String),
    /// It cannot land as it stands, for this reason; nothing changed.
    Refused(String),
}

/// How many times a publish meets a record that moved between its check and
/// its push before it gives the turn back.
const ATTEMPTS: usize = 3;

impl Merge<'_> {
    /// Land the store on the record as one commit with `message`, made as the
    /// mind repository's own identity. A record that moved since the store's
    /// base is merged in first when it merges cleanly; one that overlaps it
    /// is refused, naming the paths, and the store is left exactly as it was.
    pub fn publish(&self, message: &str) -> Publish {
        let author = match self.repo.identity() {
            Ok(author) => author,
            Err(e) => return Publish::Refused(e.to_string()),
        };
        self.publish_as(message, &author)
    }

    fn publish_as(&self, message: &str, author: &Signature) -> Publish {
        for _ in 0..ATTEMPTS {
            // Read afresh each time: a merge taken in may leave less to land,
            // or nothing — the same text published from elsewhere.
            let paths: Vec<String> = self.store.status().into_iter().map(|(p, _)| p).collect();
            let committing = match Committing::begin(&self.repo, self.store, &self.branch) {
                Ok(Ok(committing)) => committing,
                Ok(Err(NotCommitted::Behind { record })) => match self.merge_clean(&record) {
                    Ok(None) => continue,
                    Ok(Some(conflicts)) => return Publish::Conflicts(conflicts),
                    Err(publish) => return publish,
                },
                Ok(Err(why)) => return not_committed(why),
                Err(e) => return failed(e),
            };
            if paths.is_empty() && !committing.finishes_merge() && !committing.ahead() {
                return Publish::Nothing;
            }
            let landed = if paths.is_empty() && committing.ahead() {
                committing.publish_base(&self.repo)
            } else {
                let built = match self.changes().and_then(|set| {
                    self.repo
                        .commit_changes(committing.onto(), &set, message, author, author)
                }) {
                    Ok(built) => built,
                    Err(e) => return failed(e),
                };
                committing.land(&self.repo, self.store, &built, message, author, author)
            };
            match landed {
                Ok(Ok(landed)) => {
                    return Publish::Landed {
                        commit: landed.commit.to_string(),
                        paths,
                    }
                }
                Ok(Err(NotCommitted::Behind { .. })) => continue,
                Ok(Err(why)) => return not_committed(why),
                Err(e) => return failed(e),
            }
        }
        Publish::Refused(
            "the record kept moving while this was being published; publish again".into(),
        )
    }

    /// Merge `record` into the store when it merges cleanly: `None` once it
    /// has, or the paths that overlap — the store put back exactly as it was.
    fn merge_clean(&self, record: &Oid) -> Result<Option<Vec<String>>, Publish> {
        let before = self.store.snapshot();
        let merged = merge_into(&self.repo, self.store, record, RECORD).map_err(failed)?;
        if merged.conflicts().is_empty() {
            return Ok(None);
        }
        self.put_back(before, &merged).map_err(Publish::Refused)?;
        Ok(Some(merged.conflicts().to_vec()))
    }

    /// Every change the store holds, as a commit's change set.
    fn changes(&self) -> Result<ChangeSet, GitError> {
        let mut set = ChangeSet::new();
        for (path, state) in self.store.status() {
            let at = RepoPath::parse(&path)?;
            match state {
                FileState::Deleted => set.delete(at)?,
                FileState::Added | FileState::Modified => {
                    let bytes = self
                        .store
                        .read_bytes(&path)
                        .map_err(|e| GitError::invalid(format!("{path}: {e}")))?
                        .ok_or_else(|| GitError::invalid(format!("{path} is not in the store")))?;
                    set.write(at, bytes, None)?
                }
            }
        }
        Ok(set)
    }
}

fn failed(e: GitError) -> Publish {
    match unreachable(&e) {
        true => Publish::Unreachable(e.to_string()),
        false => Publish::Refused(e.to_string()),
    }
}

fn not_committed(why: NotCommitted) -> Publish {
    match why {
        NotCommitted::Conflicts(paths) => Publish::Unsettled(paths),
        NotCommitted::Behind { record } => Publish::Refused(format!(
            "the record moved to {record} while this was published"
        )),
        NotCommitted::Refused(rejection) => Publish::Refused(format!(
            "origin refused the push: {}",
            match rejection {
                Rejection::Stale => "the record moved while it was pushed".to_string(),
                Rejection::AtomicAborted => "another ref in the push was refused".to_string(),
                Rejection::Remote(why) | Rejection::Other(why) => why,
            }
        )),
    }
}
