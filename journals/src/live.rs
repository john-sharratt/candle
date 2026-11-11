//! A journal checkout kept open for a service's life: the [`Store`] behind a
//! lock, the tree it holds published for readers who never take the lock,
//! and a thread that keeps the store in step with origin every
//! [`SYNC_EVERY`].
//!
//! zend's memory and npcd's bots each open one, on checkouts of their own: a
//! store commits on the private [`JOURNAL_REF`](crate::store::JOURNAL_REF), so
//! two never share a checkout, and they meet through origin as two members'
//! stores do.
//!
//! Reads answer from the published snapshot. Writes take the store, commit at
//! once, and publish the new snapshot. The thread holds the store only between
//! the push and the fetch, never across them ([`sync`]), so no write waits on
//! the network.

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::actor::Actor;
use crate::diaries::layout::DiaryId;
use crate::error::Error;
use crate::ops::{Applied, Op};
use crate::store::diff::Changes;
use crate::store::sync::sync;
use crate::store::{Store, StoreError};
use crate::time::Stamp;
use crate::tree::Tree;

/// How often the store pushes what it wrote and fetches what origin has.
pub const SYNC_EVERY: Duration = Duration::from_secs(5);

/// Where the checkout stands, read without the store.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Status {
    /// No checkout on this machine.
    Absent,
    /// Being read; callers wait for it.
    Opening,
    Failed(String),
    Open,
}

/// Why a write was not made.
#[derive(Debug)]
pub enum WriteError {
    /// The checkout cannot be written now: none here, or not yet read.
    Unavailable(String),
    /// The store refused it, or failed to commit it.
    Store(StoreError),
}

impl WriteError {
    /// What it was based on changed underneath it: made again over the
    /// journal as it now stands, it may be kept.
    pub fn is_conflict(&self) -> bool {
        matches!(
            self,
            WriteError::Store(StoreError::Journal(Error::Conflict(_)))
        )
    }

    /// The rules' answer to the write as made: made again unchanged, it is
    /// refused again.
    pub fn is_refusal(&self) -> bool {
        matches!(
            self,
            WriteError::Store(StoreError::Journal(
                Error::Invalid(_) | Error::NotFound(_) | Error::Forbidden(_) | Error::Refused(_)
            ))
        )
    }

    /// A file the write touches is broken: refused until another commit
    /// mends it, which may be a file the write's own journal does not show.
    /// Nothing about the store as a whole.
    pub fn is_broken(&self) -> bool {
        matches!(
            self,
            WriteError::Store(StoreError::Journal(Error::Broken(_)))
        )
    }
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Unavailable(why) => f.write_str(why),
            WriteError::Store(e) => write!(f, "{e}"),
        }
    }
}

impl From<WriteError> for String {
    fn from(e: WriteError) -> String {
        e.to_string()
    }
}

/// `m` locked, whatever a panic elsewhere left in it: the store's state is
/// changed only once a write has committed, so what it holds is sound.
fn held<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Live {
    /// What a caller is told when this machine has no checkout.
    absent: &'static str,
    status: RwLock<Status>,
    store: Mutex<Option<Store>>,
    tree: RwLock<Arc<Tree>>,
}

impl Live {
    /// A checkout not opened yet; until [`Self::start`], every call is told
    /// `absent`.
    pub fn new(absent: &'static str) -> Live {
        Live {
            absent,
            status: RwLock::new(Status::Absent),
            store: Mutex::new(None),
            tree: RwLock::new(Arc::new(Tree::default())),
        }
    }

    fn set_status(&self, s: Status) {
        *self.status.write().unwrap_or_else(|e| e.into_inner()) = s;
    }

    /// Open the checkout at `dir` on a thread named `name`, and keep it in
    /// step with origin there for the process's life. `enter` runs first on
    /// that thread: what the service marks its threads with.
    pub fn start(
        self: &Arc<Self>,
        dir: PathBuf,
        name: &str,
        enter: impl FnOnce() + Send + 'static,
    ) {
        self.set_status(Status::Opening);
        let me = Arc::clone(self);
        let spawned = thread::Builder::new().name(name.into()).spawn(move || {
            enter();
            me.run(dir)
        });
        if let Err(e) = spawned {
            self.set_status(Status::Failed(format!(
                "the journal thread did not start: {e}"
            )));
        }
    }

    fn run(self: Arc<Self>, dir: PathBuf) {
        match Store::open(&dir) {
            Ok(store) => {
                tracing::info!(dir = %dir.display(), head = %store.head(), "journals: open");
                self.publish(store.shared());
                *held(&self.store) = Some(store);
                self.set_status(Status::Open);
            }
            Err(e) => {
                tracing::error!(dir = %dir.display(), "journals: could not open: {e}");
                self.set_status(Status::Failed(format!(
                    "the journal repository could not be read: {e}"
                )));
                return;
            }
        }
        loop {
            thread::sleep(SYNC_EVERY);
            let Some(synced) = sync(&self.store) else {
                return;
            };
            match synced {
                Ok(synced) => {
                    for (subject, why) in &synced.dropped {
                        tracing::warn!(
                            "journals: a write origin refused on replay is gone: {subject}: {why}"
                        );
                    }
                    for (was, is) in &synced.renamed {
                        tracing::warn!(
                            "journals: a replayed line's id was taken: {was} is now {is}"
                        );
                    }
                    if synced.fetched || synced.replayed > 0 {
                        // Under the store, so a write's newer tree is never
                        // published over.
                        if let Some(store) = held(&self.store).as_ref() {
                            self.publish(store.shared());
                        }
                    }
                }
                Err(e) => tracing::warn!("journals: sync: {e}"),
            }
        }
    }

    fn publish(&self, tree: Arc<Tree>) {
        *self.tree.write().unwrap_or_else(|e| e.into_inner()) = tree;
    }

    /// Whether the checkout opened, waiting up to `within` while it is still
    /// being read.
    pub fn opened_within(&self, within: Duration) -> bool {
        let until = Instant::now() + within;
        loop {
            match &*self.status.read().unwrap_or_else(|e| e.into_inner()) {
                Status::Open => return true,
                Status::Absent | Status::Failed(_) => return false,
                Status::Opening => {}
            }
            if Instant::now() >= until {
                return false;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// This machine has the checkout, and it opened or is opening.
    pub fn serves(&self) -> bool {
        !matches!(
            *self.status.read().unwrap_or_else(|e| e.into_inner()),
            Status::Absent | Status::Failed(_)
        )
    }

    /// Why the checkout cannot be used now, when it cannot.
    pub fn unavailable(&self) -> Option<String> {
        match &*self.status.read().unwrap_or_else(|e| e.into_inner()) {
            Status::Open => None,
            Status::Absent => Some(self.absent.into()),
            Status::Opening => {
                Some("the journal repository is still being read; try again shortly".into())
            }
            Status::Failed(why) => Some(why.clone()),
        }
    }

    /// The tree as last published.
    pub fn tree(&self) -> Result<Arc<Tree>, String> {
        if let Some(why) = self.unavailable() {
            return Err(why);
        }
        Ok(Arc::clone(
            &self.tree.read().unwrap_or_else(|e| e.into_inner()),
        ))
    }

    /// Apply `op` for `actor` now, commit it, and publish the new tree.
    pub fn write(&self, actor: &Actor, op: &Op) -> Result<Applied, WriteError> {
        self.write_at(actor, op, Stamp::now())
    }

    /// [`Self::write`], recorded as happening at `at`.
    pub fn write_at(&self, actor: &Actor, op: &Op, at: Stamp) -> Result<Applied, WriteError> {
        if let Some(why) = self.unavailable() {
            return Err(WriteError::Unavailable(why));
        }
        let mut guard = held(&self.store);
        let Some(store) = guard.as_mut() else {
            return Err(WriteError::Unavailable(self.absent.into()));
        };
        let applied = store.write(actor, op, at).map_err(WriteError::Store)?;
        self.publish(store.shared());
        Ok(applied)
    }

    /// Every diary's sealed file, by its id, and the repository's salt once
    /// there is one (`docs/private_mode.md` §5.2): read under the store, which
    /// holds them, never through the tree.
    pub fn diaries(&self) -> Result<(DiaryFiles, Option<Vec<u8>>), String> {
        if let Some(why) = self.unavailable() {
            return Err(why);
        }
        let guard = held(&self.store);
        let store = guard.as_ref().ok_or(self.absent)?;
        let files = store
            .diary_files()
            .into_iter()
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect();
        Ok((files, store.diary_salt().map(<[u8]>::to_vec)))
    }

    /// The repository's diary salt, once there is one: read alone, without
    /// copying any diary's file.
    pub fn diary_salt(&self) -> Result<Option<Vec<u8>>, String> {
        if let Some(why) = self.unavailable() {
            return Err(why);
        }
        let guard = held(&self.store);
        let store = guard.as_ref().ok_or(self.absent)?;
        Ok(store.diary_salt().map(<[u8]>::to_vec))
    }

    /// One diary's sealed file, when it is there.
    pub fn diary_file(&self, id: &DiaryId) -> Result<Option<Vec<u8>>, String> {
        if let Some(why) = self.unavailable() {
            return Err(why);
        }
        let guard = held(&self.store);
        let store = guard.as_ref().ok_or(self.absent)?;
        Ok(store.diary_file(id).map(<[u8]>::to_vec))
    }

    /// Write `changes` to diaries' files as one commit, pushed at the next
    /// sync. The tree is unchanged: it never reads diaries.
    pub fn write_diaries(&self, changes: &Changes) -> Result<(), WriteError> {
        if let Some(why) = self.unavailable() {
            return Err(WriteError::Unavailable(why));
        }
        let mut guard = held(&self.store);
        let Some(store) = guard.as_mut() else {
            return Err(WriteError::Unavailable(self.absent.into()));
        };
        store.write_diaries(changes).map_err(WriteError::Store)
    }

    /// The store, held: for a test that proves a read never waits on it.
    pub fn hold_store(&self) -> MutexGuard<'_, Option<Store>> {
        held(&self.store)
    }
}

/// Every diary's sealed file, by its id.
pub type DiaryFiles = Vec<(DiaryId, Vec<u8>)>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::JournalPath;
    use crate::slug::Slug;

    const ABSENT: &str = "no checkout here";

    #[test]
    fn a_checkout_not_started_says_it_is_absent() {
        let live = Live::new(ABSENT);
        assert!(!live.serves());
        assert_eq!(live.tree().unwrap_err(), ABSENT);
        let op = Op::MarkStale {
            path: JournalPath::root(Slug::parse("x").unwrap()),
            why: "over its cap".into(),
        };
        let e = live.write(&Actor::Zend, &op).unwrap_err();
        assert_eq!(e.to_string(), ABSENT);
        assert!(!e.is_conflict() && !e.is_refusal() && !e.is_broken());
        assert!(!live.opened_within(Duration::from_millis(1)));
    }

    #[test]
    fn a_checkout_that_cannot_open_says_why() {
        let root = tempfile::tempdir().unwrap();
        let live = Arc::new(Live::new(ABSENT));
        live.start(root.path().to_path_buf(), "journal-test", || {});
        assert!(!live.opened_within(Duration::from_secs(30)));
        assert!(live
            .tree()
            .unwrap_err()
            .starts_with("the journal repository could not be read"));
    }
}
