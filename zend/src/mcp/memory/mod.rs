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
//! | [`readers`] | whom each conversation reads the memory as, for its retrieval scope |
//!
//! Reads answer from a published snapshot of the tree and never take the
//! store; writes take it, commit at once, and publish the new snapshot. The
//! checkout is a [`Live`], kept in step with origin every
//! [`journals::live::SYNC_EVERY`] without any write waiting on the network.

pub mod context;
pub mod conversations;
pub mod moved;
pub mod readers;
pub mod who;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

#[cfg(feature = "memory-census")]
use heap_census::{set_thread, Scope};
use journals::actor::{need_read, Actor};
use journals::live::Live;
use journals::ops::{Applied, Op};
use journals::path::{JournalPath, HERE};
use journals::scope::Section;
use journals::time::Stamp;
use journals::tree::{Journal, Tree};

use conversations::Conversations;

pub use journals::live::WriteError;

/// What a tool says when this zend serves no journal repository.
pub const NO_MEMORY: &str =
    "this zend serves no journal repository: its agent has no `journal: true` checkout";

pub struct Memory {
    pub(crate) live: Arc<Live>,
    pub(crate) conversations: Conversations,
    /// The accounts whose old notes were moved, or are moving, in this
    /// process: each account's own, since each kept its notes under its key.
    pub(crate) moved: Mutex<BTreeSet<String>>,
}

impl Default for Memory {
    fn default() -> Self {
        Memory {
            live: Arc::new(Live::new(NO_MEMORY)),
            conversations: Conversations::default(),
            moved: Mutex::new(BTreeSet::new()),
        }
    }
}

/// `m` locked, whatever a panic elsewhere left in it.
pub(crate) fn held<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Memory {
    /// Open the journal checkout at `dir` on a thread of its own, and keep it
    /// in step with origin there for the process's life.
    pub fn start(self: &Arc<Self>, dir: PathBuf) {
        self.live.start(dir, "journal-sync", || {
            #[cfg(feature = "memory-census")]
            set_thread(Scope::Journals);
        });
    }

    /// Whether the memory opened, waiting up to `within` while it is still
    /// being read: what decides whether this member declares the memory's
    /// routes, so a member whose checkout could not be read never draws a
    /// memory call, nor one still reading it.
    pub fn opened_within(&self, within: Duration) -> bool {
        self.live.opened_within(within)
    }

    /// This zend serves the memory: its agent holds the journal checkout, and
    /// it opened or is opening.
    pub fn serves(&self) -> bool {
        self.live.serves()
    }

    /// The tree as last published.
    pub fn tree(&self) -> Result<Arc<Tree>, String> {
        self.live.tree()
    }

    /// Apply `op` for `actor` now, commit it, and publish the new tree.
    pub fn write(&self, actor: &Actor, op: &Op) -> Result<Applied, WriteError> {
        self.live.write(actor, op)
    }

    /// [`Self::write`], recorded as happening at `at`.
    pub fn write_at(&self, actor: &Actor, op: &Op, at: Stamp) -> Result<Applied, WriteError> {
        self.live.write_at(actor, op, at)
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
