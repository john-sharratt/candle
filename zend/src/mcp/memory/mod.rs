//! The memory backbone as zend serves it (`docs/memory_backbone.md`): the
//! journal repository the agent holds as its `journal: true` checkout, opened
//! as a [`Store`] and kept in step with origin, behind the `journal_*` tools,
//! `get_user_context` and the episodes the aggregator records.
//!
//! | Module | |
//! |---|---|
//! | [`who`] | who a caller is — a person, binding a new account by its email, or a guest |
//! | [`conversations`] | each conversation's own journal, opened at its first call |
//! | [`moved`] | a person's old `U:` notes, moved onto their own journal once |
//! | [`context`] | `get_user_context`: the frame, within the client's budget |
//!
//! Reads answer from a published snapshot of the tree and never take the
//! store. Writes take the store, commit at once, and publish the new
//! snapshot. The thread keeps the store in step with origin every
//! [`SYNC_EVERY`], holding it only between the push and fetch, never across
//! them (`journals::store::sync`), so no write waits on the network.

pub mod context;
pub mod conversations;
pub mod moved;
pub mod who;

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use journals::actor::{need_read, Actor};
use journals::error::Error;
use journals::ops::{Applied, Op};
use journals::path::{JournalPath, HERE};
use journals::scope::Section;
use journals::store::sync::sync;
use journals::store::{Store, StoreError};
use journals::time::Stamp;
use journals::tree::{Journal, Tree};

use conversations::Conversations;

/// How often the store pushes what it wrote and fetches what origin has.
pub const SYNC_EVERY: Duration = Duration::from_secs(5);

/// What a tool says when this zend serves no journal repository.
pub const NO_MEMORY: &str =
    "this zend serves no journal repository: its agent has no `journal: true` checkout";

/// Where the memory stands, read without the store.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Status {
    /// No journal checkout on this machine.
    Absent,
    /// Being read; tools wait for it.
    Opening,
    Failed(String),
    Open,
}

/// Why a write was not made.
#[derive(Debug)]
pub enum WriteError {
    /// The memory cannot be written now: none here, or not yet read.
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

pub struct Memory {
    status: RwLock<Status>,
    store: Mutex<Option<Store>>,
    tree: RwLock<Arc<Tree>>,
    pub(crate) conversations: Conversations,
    /// The accounts whose old notes were moved, or are moving, in this
    /// process: each account's own, since each kept its notes under its key.
    pub(crate) moved: Mutex<BTreeSet<String>>,
}

impl Default for Memory {
    fn default() -> Self {
        Memory {
            status: RwLock::new(Status::Absent),
            store: Mutex::new(None),
            tree: RwLock::new(Arc::new(Tree::default())),
            conversations: Conversations::default(),
            moved: Mutex::new(BTreeSet::new()),
        }
    }
}

/// `m` locked, whatever a panic elsewhere left in it: the store's state is
/// changed only once a write has committed, so what it holds is sound.
pub(crate) fn held<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Memory {
    fn set_status(&self, s: Status) {
        *self.status.write().unwrap_or_else(|e| e.into_inner()) = s;
    }

    /// Open the journal checkout at `dir` on a thread of its own, and keep it
    /// in step with origin there for the process's life.
    pub fn start(self: &Arc<Self>, dir: PathBuf) {
        self.set_status(Status::Opening);
        let me = Arc::clone(self);
        let spawned = thread::Builder::new()
            .name("journal-sync".into())
            .spawn(move || me.run(dir));
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

    /// Whether the memory opened, waiting up to `within` while it is still
    /// being read: what decides whether this member declares the memory's
    /// routes, so a member whose checkout could not be read never draws a
    /// memory call, nor one still reading it.
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

    /// This zend serves the memory: its agent holds the journal checkout, and
    /// it opened or is opening.
    pub fn serves(&self) -> bool {
        !matches!(
            *self.status.read().unwrap_or_else(|e| e.into_inner()),
            Status::Absent | Status::Failed(_)
        )
    }

    /// Why the memory cannot be used now, when it cannot.
    fn unavailable(&self) -> Option<String> {
        match &*self.status.read().unwrap_or_else(|e| e.into_inner()) {
            Status::Open => None,
            Status::Absent => Some(NO_MEMORY.into()),
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
            return Err(WriteError::Unavailable(NO_MEMORY.into()));
        };
        let applied = store.write(actor, op, at).map_err(WriteError::Store)?;
        self.publish(store.shared());
        Ok(applied)
    }

    /// The journal at `path`, which `actor` may read.
    pub fn readable<'a>(
        tree: &'a Tree,
        actor: &Actor,
        path: &JournalPath,
    ) -> Result<&'a Journal, String> {
        need_read(tree, actor, path).map_err(|e| e.to_string())?;
        tree.need(path).map_err(|e| e.to_string())
    }
}

/// A journal path written the memory's way rather than a thread's: `@group`,
/// `here`, or one of the sections — and `~`, which the memory's reader
/// answers with where a person's own journal now is.
pub fn is_memory_path(text: &str) -> bool {
    let first = text.split(['/', '#']).next().unwrap_or_default();
    text.starts_with('~')
        || text.starts_with('@')
        || first == HERE
        || Section::parse(first).is_some()
}

#[cfg(test)]
pub(crate) mod tests;
