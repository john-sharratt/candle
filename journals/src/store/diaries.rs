//! The diaries' files in the store (`docs/private_mode.md` §5.2): read from
//! what the store holds, written as one commit onto the service's head whose
//! message names nothing ([`diaries_message`]), and carried across a replay
//! whole — each write is a diary's whole file, so the last one stands.

use std::collections::BTreeMap;

use zend_vfs::{ChangeSet, CommitInfo, Oid, RefOp, RefTransaction, Repo, RepoPath, Signature};

use super::diff::{apply_changes, Changes};
use super::message::diaries_message;
use super::read::files_at;
use super::{git_time, journal_ref, Store, StoreError};
use crate::diaries::layout::{in_diaries, DiaryId, DIR, SALT};
use crate::error::Error;
use crate::time::Stamp;

impl Store {
    /// Every diary's file, by its id.
    pub fn diary_files(&self) -> Vec<(DiaryId, &[u8])> {
        let under = format!("{DIR}/");
        self.files
            .range(under.clone()..)
            .take_while(|(path, _)| path.starts_with(&under))
            .filter_map(|(path, bytes)| Some((DiaryId::of_path(path)?, bytes.as_slice())))
            .collect()
    }

    /// One diary's file.
    pub fn diary_file(&self, id: &DiaryId) -> Option<&[u8]> {
        self.files.get(&id.path()).map(Vec::as_slice)
    }

    /// The repository's salt, once the first diary made it.
    pub fn diary_salt(&self) -> Option<&[u8]> {
        self.files.get(SALT).map(Vec::as_slice)
    }

    /// Write `changes` — every one a file under `diaries/` — as one commit
    /// onto the service's head, which the next sync pushes. The tree is
    /// unchanged: it never reads `diaries/`.
    pub fn write_diaries(&mut self, changes: &Changes) -> Result<(), StoreError> {
        if let Some(path) = changes.keys().find(|p| !in_diaries(p)) {
            return Err(Error::refused(format!("{path} is no diary's file")).into());
        }
        let changes: Changes = changes
            .iter()
            .filter(|(path, change)| self.files.get(*path) != change.as_ref())
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect();
        if changes.is_empty() {
            return Err(Error::refused("that changes nothing").into());
        }
        let new = commit_diaries(&self.repo, &self.head, &changes)?;
        let txn = RefTransaction::new().push(RefOp::Update {
            name: journal_ref(),
            new: new.clone(),
            old: self.head.clone(),
        });
        self.repo.update_refs(&txn)?;
        self.head = new;
        apply_changes(&mut self.files, &changes);
        Ok(())
    }
}

/// `changes` committed onto `base` as a diaries commit: written by the
/// service, at the time it was made, saying nothing.
pub(super) fn commit_diaries(repo: &Repo, base: &Oid, changes: &Changes) -> Result<Oid, StoreError> {
    let mut set = ChangeSet::new();
    for (path, change) in changes {
        let path = RepoPath::parse(path)?;
        match change {
            Some(bytes) => set.write_blob(path, bytes.clone(), None)?,
            None => set.delete(path)?,
        }
    }
    let at = git_time(Stamp::now());
    let service = Signature::new("journal service", "journals@tokera", at)?;
    Ok(repo.commit_changes(base, &set, &diaries_message(), &service, &service)?)
}

/// What diaries commit `c` changed against its first parent: what a replay
/// carries onto the newer base.
pub(super) fn diaries_changed(repo: &Repo, c: &CommitInfo) -> Result<Changes, StoreError> {
    let diaries = |files: BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Vec<u8>> {
        files.into_iter().filter(|(p, _)| in_diaries(p)).collect()
    };
    let after = diaries(files_at(repo, &c.oid)?);
    let before = match c.parents.first() {
        Some(parent) => diaries(files_at(repo, parent)?),
        None => BTreeMap::new(),
    };
    let mut changes = Changes::new();
    for (path, bytes) in &after {
        if before.get(path) != Some(bytes) {
            changes.insert(path.clone(), Some(bytes.clone()));
        }
    }
    for path in before.keys().filter(|p| !after.contains_key(*p)) {
        changes.insert(path.clone(), None);
    }
    Ok(changes)
}
