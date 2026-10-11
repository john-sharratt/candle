//! A store over the mind as a shelf (`docs/replicated_substrate.md` §10.3):
//! reads fall through to the store, and a commit records into it. An
//! operation's is the shelf of a body on one of its steps; an author's is the
//! shelf the console writes through.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::sim::shelf::Shelf;

use super::owner::Owner;
use super::MindStores;

/// `owner`'s store.
pub struct StoreShelf {
    pub stores: Arc<MindStores>,
    pub owner: Owner,
}

impl Shelf for StoreShelf {
    fn read(&self, key: &str) -> Option<String> {
        self.stores
            .read(&self.owner, |s| s.read(key).ok().flatten())
            .flatten()
    }

    fn list(&self, dir: &str) -> BTreeSet<String> {
        self.stores
            .read(&self.owner, |s| {
                s.list_dir(dir)
                    .ok()
                    .flatten()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|e| {
                        let name = e.path.rsplit('/').next().unwrap_or(&e.path).to_string();
                        match e.dir {
                            true => format!("{name}/"),
                            false => name,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String> {
        self.stores
            .change(&self.owner, |s| {
                // All or nothing: one refused takes the batch back out.
                let before = s.snapshot();
                for (key, now) in changes {
                    let kept = match now {
                        Some(text) => s.write(key, text.clone()).map(|_| ()),
                        None => {
                            s.delete(key);
                            Ok(())
                        }
                    };
                    if let Err(e) = kept {
                        if before.is_empty() {
                            s.discard();
                        } else {
                            let _ = s.restore(before);
                        }
                        return Err(format!("{key} could not be kept: {e}"));
                    }
                }
                Ok(())
            })
            .unwrap_or_else(|| Err("There are no documents in this world to work on.".into()))
    }
}
