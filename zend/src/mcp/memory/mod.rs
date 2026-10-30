//! The memory backbone as zend serves it (`docs/memory_backbone.md`): the
//! journal repository the agent holds as its `journal: true` checkout, opened
//! as a [`Store`] and kept in step with origin, behind the `journal_*` tools,
//! `get_user_context` and the episodes the aggregator records.
//!
//! | Module | |
//! |---|---|
//! | [`who`] | which person a caller is, binding a new account by its email |
//! | [`conversations`] | the journal each conversation's episodes go on |
//! | [`moved`] | a person's old `U:` notes, moved onto their tree once |
//! | [`context`] | `get_user_context`: the frame, within the client's budget |
//!
//! Reads answer from a published snapshot of the tree and never take the
//! store, so a voice client's read never waits behind a push. Writes take the
//! store, commit at once, and publish the new snapshot; a write made while the
//! sync thread is pushing waits for the push. The thread keeps the store in
//! step with origin every [`SYNC_EVERY`].

pub mod context;
pub mod conversations;
pub mod moved;
pub mod who;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::thread;
use std::time::Duration;

use journals::actor::{need_read, Actor};
use journals::ops::{Applied, Op};
use journals::path::JournalPath;
use journals::scope::Section;
use journals::store::Store;
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
            let mut guard = held(&self.store);
            let Some(store) = guard.as_mut() else { return };
            match store.sync() {
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
                        self.publish(store.shared());
                    }
                }
                Err(e) => tracing::warn!("journals: sync: {e}"),
            }
        }
    }

    fn publish(&self, tree: Arc<Tree>) {
        *self.tree.write().unwrap_or_else(|e| e.into_inner()) = tree;
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
    pub fn write(&self, actor: &Actor, op: &Op) -> Result<Applied, String> {
        self.write_at(actor, op, Stamp::now())
    }

    /// [`Self::write`], recorded as happening at `at`.
    pub fn write_at(&self, actor: &Actor, op: &Op, at: Stamp) -> Result<Applied, String> {
        if let Some(why) = self.unavailable() {
            return Err(why);
        }
        let mut guard = held(&self.store);
        let Some(store) = guard.as_mut() else {
            return Err(NO_MEMORY.into());
        };
        let applied = store.write(actor, op, at).map_err(|e| e.to_string())?;
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

/// A journal path written the memory's way rather than a thread's: `~`, `@`,
/// or one of the sections.
pub fn is_memory_path(text: &str) -> bool {
    let first = text.split(['/', '#']).next().unwrap_or_default();
    text.starts_with('~') || text.starts_with('@') || Section::parse(first).is_some()
}

#[cfg(test)]
mod tests;
