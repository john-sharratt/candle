//! An author's writes to the mind (`docs/replicated_substrate.md` §10.3).
//!
//! Everything an admin writes through the console — a world, a personality, an
//! entry of the corpus, a life, a reshaped map — is a change to a document of
//! the mind repository. Over a mind under git it is a **draft**: a `VfsEvent`
//! in the author's own store (one per account), seen by that author and by no
//! one else, and landed on the record only when they publish it — through the
//! same merge an operation's publish step runs. Over a mind folder that is not
//! a git repository there is no record to land on, and what is written is what
//! stands.
//!
//! | Module | |
//! |---|---|
//! | [`overlay`] | the published mind with an author's drafts laid over it |
//! | [`disk`] | the mind folder written directly |
//! | [`caller`] | as whom a request reads and writes |
//! | [`records`] | the worlds and personalities as an author sees them |
//! | [`routes`] | the drafts, publish, sync and discard routes |

pub mod caller;
pub mod disk;
pub mod overlay;
pub mod records;
pub mod routes;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use crate::mind::Mind;
use crate::opstore::owner::Owner;
use crate::opstore::shelf::StoreShelf;
use crate::opstore::MindStores;
use crate::sim::shelf::{Published, Shelf};

use self::disk::Disk;
use self::overlay::{Drafted, Overlay};

/// Why a write is refused while the mind has not arrived.
const PENDING: &str = "the mind has not arrived from the substrate yet — nothing can be written until it has";

/// One account's view of the mind, and where its writes go.
pub enum Author {
    /// A mind under git: the published mind under the account's drafts, its
    /// writes committed into its store.
    Drafts { view: Overlay, store: StoreShelf },
    /// A mind folder that is not a git repository: read and written as it
    /// stands.
    Disk(Disk),
    /// A mind read from the substrate that has not arrived yet: nothing to
    /// read, and every write refused rather than landed somewhere else.
    Pending,
}

impl Author {
    /// Account `account`'s view of `mind`, over `stores` when the daemon has
    /// them. A mind with no folder of its own is written in `authored` — the
    /// folder the worlds and personalities are kept in — as it stands.
    pub fn open(
        mind: &Mind,
        authored: &Path,
        stores: Option<&Arc<MindStores>>,
        account: &str,
    ) -> Author {
        let files = mind.files();
        if files.is_none() && mind.repo_dir().is_some() {
            return Author::Pending;
        }
        let published = files
            .map(Published::of)
            .unwrap_or_else(|| Published::folder(authored));
        let Some(stores) = stores.filter(|s| s.under_git()) else {
            let root = published
                .files()
                .folder_root()
                .unwrap_or(authored)
                .to_path_buf();
            return Author::Disk(Disk::new(root));
        };
        let store = StoreShelf {
            stores: Arc::clone(stores),
            owner: Owner::author(account),
        };
        let drafts: Drafted = stores
            .read(&store.owner, |s| {
                s.changed_paths()
                    .into_iter()
                    .map(|path| {
                        let text = s.read(&path).ok().flatten();
                        (path, text)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Author::Drafts {
            view: Overlay::new(published, drafts),
            store,
        }
    }

    /// Whether writes are drafts, to be published, rather than what stands.
    pub fn keeps_drafts(&self) -> bool {
        matches!(self, Author::Drafts { .. })
    }

    /// Every drafted path, with its text or `None` where the draft removes
    /// it; empty over a mind that keeps no drafts.
    pub fn drafted(&self) -> Drafted {
        match self {
            Author::Drafts { view, .. } => view.drafts().clone(),
            Author::Disk(_) | Author::Pending => Drafted::new(),
        }
    }

    /// The drafted documents directly inside `folder`, by file name, with
    /// their text or `None` where the draft removes them.
    pub fn drafted_in(&self, folder: &str) -> Vec<(String, Option<String>)> {
        let prefix = format!("{}/", folder.trim_matches('/'));
        self.drafted()
            .into_iter()
            .filter_map(|(path, text)| {
                let name = path.strip_prefix(&prefix)?;
                (!name.contains('/')).then(|| (name.to_string(), text))
            })
            .collect()
    }

    /// The owner of this author's store, over a mind that keeps drafts.
    pub fn owner(&self) -> Option<&Owner> {
        match self {
            Author::Drafts { store, .. } => Some(&store.owner),
            Author::Disk(_) | Author::Pending => None,
        }
    }

    /// Whether something that is not a document — a folder, a link — stands
    /// at `key` in the folder, where no draft of the author's replaces it.
    pub fn not_a_document(&self, key: &str) -> bool {
        match self {
            Author::Drafts { view, .. } => {
                !view.drafts().contains_key(key) && view.published().not_a_document(key)
            }
            Author::Disk(disk) => disk.folder.not_a_document(key),
            Author::Pending => false,
        }
    }
}

impl Shelf for Author {
    fn read(&self, key: &str) -> Option<String> {
        match self {
            Author::Drafts { view, .. } => view.read(key),
            Author::Disk(disk) => disk.read(key),
            Author::Pending => None,
        }
    }

    fn list(&self, dir: &str) -> BTreeSet<String> {
        match self {
            Author::Drafts { view, .. } => view.list(dir),
            Author::Disk(disk) => disk.list(dir),
            Author::Pending => BTreeSet::new(),
        }
    }

    fn size(&self, key: &str) -> Option<u64> {
        match self {
            Author::Drafts { view, .. } => view.size(key),
            Author::Disk(disk) => disk.size(key),
            Author::Pending => None,
        }
    }

    fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String> {
        match self {
            Author::Drafts { view, store } => {
                store.commit(changes)?;
                view.lay(changes);
                Ok(())
            }
            Author::Disk(disk) => disk.commit(changes),
            Author::Pending => Err(PENDING.to_string()),
        }
    }
}

#[cfg(test)]
mod tests;
