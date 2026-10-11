//! `merge_status`: what an operation's store changes against its base, and
//! whether it merges cleanly with the record as it stands.

use zend_vfs::{FileState, Rev};

use super::{unreachable, Merge};

/// What the merge table shows of a store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// The store changes nothing.
    Nothing,
    /// It changes `changed`; the record has not moved since its base, or
    /// moved only at lines it does not touch (`clean`), or it overlaps the
    /// record at `conflicts`.
    Changes {
        changed: Vec<(String, FileState)>,
        record_moved: bool,
        conflicts: Vec<String>,
    },
    /// `origin` could not be reached, so whether it merges is not known.
    Unknown {
        changed: Vec<(String, FileState)>,
        why: String,
    },
}

impl Merge<'_> {
    /// What the store changes, and whether it lands cleanly.
    pub fn status(&self) -> Result<Status, String> {
        let changed = self.store.status();
        if changed.is_empty() {
            return Ok(Status::Nothing);
        }
        let record = match self.record() {
            Ok(record) => record,
            Err(e) if unreachable(&e) => {
                return Ok(Status::Unknown {
                    changed,
                    why: e.to_string(),
                })
            }
            Err(e) => return Err(e.to_string()),
        };
        let moved = match (&record, self.base()) {
            (Some(record), Some(base)) => {
                record != &base
                    && !self
                        .repo
                        .is_ancestor(&Rev::Oid(record.clone()), &Rev::Oid(base))
                        .map_err(|e| e.to_string())?
            }
            (Some(_), None) => true,
            (None, _) => false,
        };
        let conflicts = match (&record, moved) {
            (Some(record), true) => self.trial(record)?,
            _ => Vec::new(),
        };
        Ok(Status::Changes {
            changed,
            record_moved: moved,
            conflicts,
        })
    }
}

impl Status {
    /// The status as the Maker at the merge table reads it.
    pub fn render(&self) -> String {
        let listed = |changed: &[(String, FileState)]| {
            changed
                .iter()
                .map(|(path, state)| format!("{path} ({})", state_word(state)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            Status::Nothing => {
                "This operation changes nothing in the record: there is nothing to publish."
                    .to_string()
            }
            Status::Changes {
                changed,
                record_moved: false,
                ..
            } => format!(
                "This operation changes {}. The record has not moved since its work began: \
                 `merge_publish` lands it.",
                listed(changed)
            ),
            Status::Changes {
                changed, conflicts, ..
            } if conflicts.is_empty() => format!(
                "This operation changes {}. The record has moved since its work began, at lines \
                 it does not touch: `merge_publish` merges and lands it.",
                listed(changed)
            ),
            Status::Changes {
                changed, conflicts, ..
            } => format!(
                "This operation changes {}. The record has moved since its work began, and \
                 overlaps it in {}. `merge_sync` brings the record's text in, with each overlap \
                 marked; settle every mark at the bench, `bench_commit`, and publish.",
                listed(changed),
                conflicts.join(", ")
            ),
            Status::Unknown { changed, why } => format!(
                "This operation changes {}. Whether it merges with the record is not known: {why}.",
                listed(changed)
            ),
        }
    }
}

fn state_word(state: &FileState) -> &'static str {
    match state {
        FileState::Added => "new",
        FileState::Modified => "changed",
        FileState::Deleted => "taken out",
    }
}
