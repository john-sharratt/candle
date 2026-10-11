//! The published mind with an author's drafts laid over it
//! (`docs/replicated_substrate.md` §10.3).
//!
//! What an author sees of the mind: every document they have drafted as their
//! draft says it — gone, where they removed it — and everything else as the
//! record stands now. The drafts are read out of the author's store once, when
//! the view is made, and each commit made through the view is laid into it, so
//! a listing of six hundred documents opens the store once rather than six
//! hundred times.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{RwLock, RwLockReadGuard};

use crate::sim::shelf::{Published, Shelf, NO_OPERATION};

/// Each drafted path's text, or `None` where the draft removes it.
pub type Drafted = BTreeMap<String, Option<String>>;

/// The published mind under an author's drafts.
pub struct Overlay {
    published: Published,
    drafts: RwLock<Drafted>,
}

impl Overlay {
    pub fn new(published: Published, drafts: Drafted) -> Overlay {
        Overlay {
            published,
            drafts: RwLock::new(drafts),
        }
    }

    /// The published mind the drafts are laid over.
    pub fn published(&self) -> &Published {
        &self.published
    }

    /// The drafts as they stand.
    pub fn drafts(&self) -> RwLockReadGuard<'_, Drafted> {
        self.drafts.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Lay `changes`, just committed to the author's store, over the view.
    pub fn lay(&self, changes: &[(String, Option<String>)]) {
        let mut drafts = self.drafts.write().unwrap_or_else(|e| e.into_inner());
        for (key, now) in changes {
            drafts.insert(key.clone(), now.clone());
        }
    }

    /// `dir`'s path with a trailing separator, so `worlds` does not match
    /// `worlds-old/`; the mind's root is the empty prefix.
    fn prefix(dir: &str) -> String {
        let dir = dir.trim_matches('/');
        match dir.is_empty() {
            true => String::new(),
            false => format!("{dir}/"),
        }
    }
}

impl Shelf for Overlay {
    fn read(&self, key: &str) -> Option<String> {
        match self.drafts().get(key) {
            Some(drafted) => drafted.clone(),
            None => self.published.read(key),
        }
    }

    /// The folder as published, with each drafted document — and each folder
    /// a drafted document makes — added, and each removed one taken out. A
    /// folder whose every document a draft removes still lists: the drafts do
    /// not know what else the record keeps inside it.
    fn list(&self, dir: &str) -> BTreeSet<String> {
        let prefix = Overlay::prefix(dir);
        let mut names = self.published.list(dir);
        for (path, drafted) in self.drafts().iter() {
            let Some(rest) = path.strip_prefix(&prefix) else {
                continue;
            };
            match (rest.split_once('/'), drafted) {
                (Some((folder, _)), Some(_)) => {
                    names.insert(format!("{folder}/"));
                }
                (None, Some(_)) => {
                    names.insert(rest.to_string());
                }
                (None, None) => {
                    names.remove(rest);
                }
                (Some(_), None) => {}
            }
        }
        names
    }

    fn size(&self, key: &str) -> Option<u64> {
        match self.drafts().get(key) {
            Some(drafted) => drafted.as_ref().map(|text| text.len() as u64),
            None => self.published.size(key),
        }
    }

    /// A view takes no commit of its own: the author's store does, and the
    /// commit is then laid over the view ([`Overlay::lay`]).
    fn commit(&self, _changes: &[(String, Option<String>)]) -> Result<(), String> {
        Err(NO_OPERATION.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn overlay(dir: &tempfile::TempDir, drafts: &[(&str, Option<&str>)]) -> Overlay {
        Overlay::new(
            Published {
                root: dir.path().to_path_buf(),
            },
            drafts
                .iter()
                .map(|(k, v)| (k.to_string(), v.map(str::to_string)))
                .collect(),
        )
    }

    /// **A draft reads over the record, a removal hides it, and what was not
    /// drafted reads as published.**
    #[test]
    fn drafts_read_over_the_record() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("worlds")).unwrap();
        fs::write(dir.path().join("worlds/a.yaml"), "name: A\n").unwrap();
        fs::write(dir.path().join("worlds/b.yaml"), "name: B\n").unwrap();
        let view = overlay(
            &dir,
            &[
                ("worlds/a.yaml", Some("name: A2\n")),
                ("worlds/b.yaml", None),
                ("worlds/c.yaml", Some("name: C\n")),
                ("worlds-old/d.yaml", Some("name: D\n")),
                ("layers/life/x/1998 One.md", Some("one\n")),
            ],
        );
        assert_eq!(view.read("worlds/a.yaml").as_deref(), Some("name: A2\n"));
        assert_eq!(view.read("worlds/b.yaml"), None);
        assert_eq!(view.size("worlds/b.yaml"), None);
        assert_eq!(view.size("worlds/c.yaml"), Some(8));
        assert_eq!(
            view.list("worlds").into_iter().collect::<Vec<_>>(),
            ["a.yaml", "c.yaml"]
        );
        assert_eq!(
            view.list("").into_iter().collect::<Vec<_>>(),
            ["layers/", "worlds-old/", "worlds/"]
        );
        assert_eq!(
            view.list("layers/life").into_iter().collect::<Vec<_>>(),
            ["x/"]
        );
    }

    /// **A commit is laid over the view**, so what was just written reads
    /// back through the view it was written through.
    #[test]
    fn a_commit_is_laid_over_the_view() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.md"), "a\n").unwrap();
        let view = overlay(&dir, &[]);
        view.lay(&[("a.md".into(), None), ("b.md".into(), Some("b\n".into()))]);
        assert_eq!(view.read("a.md"), None);
        assert_eq!(view.read("b.md").as_deref(), Some("b\n"));
        assert_eq!(view.list("").into_iter().collect::<Vec<_>>(), ["b.md"]);
    }
}
