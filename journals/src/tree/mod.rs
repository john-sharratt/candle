//! The whole repository in memory: each group's tree beneath its master, and
//! each person's identity, read from the files and written back to them.
//!
//! | Module | |
//! |---|---|
//! | [`journal`] | one journal: metadata, window, archives, children, problems |
//! | [`children`] | a journal's sub-journals, shared with earlier versions of the tree |
//! | [`load`] | the files into a tree, every fault kept as a problem |
//! | [`check`] | the rules a loaded tree is held to |
//! | [`render`] | a tree, or one journal, back into files |
//! | [`fingerprint`] | what a summary is built over, and so what is dirty |
//! | [`window`] | which closed period of a window folds next |
//! | [`accounts`] | which person an account is |

pub mod accounts;
pub mod check;
pub mod children;
pub mod fingerprint;
pub mod journal;
pub mod load;
mod problems;
pub mod render;
pub mod window;

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{Error, Problem, Result};
use crate::files::{GroupFile, PersonFile};
use crate::path::{JournalPath, Place};
use crate::scope::Section;
use crate::slug::Slug;

pub use journal::{Held, Journal, MAX_CHILDREN, WINDOW};

/// A person: their `person.yaml`, or why it cannot be read. Identity alone;
/// their journals are their groups'.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Person {
    pub file: Result<PersonFile, Problem>,
    /// Faults in `person.yaml` against the rest of the repository.
    pub problems: Vec<Problem>,
}

/// A group: its `group.yaml` (or why it cannot be read) and its tree, its
/// master shared with earlier versions of the tree as every journal below
/// it is ([`children`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub file: Result<GroupFile, Problem>,
    pub root: Arc<Journal>,
    pub problems: Vec<Problem>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tree {
    pub persons: BTreeMap<Slug, Person>,
    pub groups: BTreeMap<Slug, Group>,
    /// Files that are no part of the layout, and faults across owners.
    pub problems: Vec<Problem>,
}

/// A master and its fixed sections, with no files yet.
pub fn skeleton() -> Journal {
    let mut root = Journal::new(Place::Root, None);
    for section in Section::ALL {
        root.children.insert(
            Slug::parse(section.name()).expect("a section's name is a name"),
            Journal::new(Place::Section(section), None),
        );
    }
    root
}

impl Tree {
    pub fn root(&self, group: &Slug) -> Option<&Journal> {
        self.groups.get(group).map(|g| &*g.root)
    }

    /// `group`'s master, to change: copied first when an earlier tree still
    /// shares it.
    pub fn root_mut(&mut self, group: &Slug) -> Option<&mut Journal> {
        self.groups
            .get_mut(group)
            .map(|g| Arc::make_mut(&mut g.root))
    }

    /// The journal at exactly `path`, references not followed.
    pub fn get(&self, path: &JournalPath) -> Option<&Journal> {
        let mut j = self.root(path.group())?;
        for n in path.names() {
            j = j.children.get(n)?;
        }
        Some(j)
    }

    pub fn get_mut(&mut self, path: &JournalPath) -> Option<&mut Journal> {
        let mut j = self.root_mut(path.group())?;
        for n in path.names() {
            j = j.children.get_mut(n)?;
        }
        Some(j)
    }

    /// The journal at exactly `path`, or an error naming what is not there.
    pub fn need(&self, path: &JournalPath) -> Result<&Journal> {
        self.get(path)
            .ok_or_else(|| Error::not_found(format!("there is no journal {path}")))
    }

    pub fn need_mut(&mut self, path: &JournalPath) -> Result<&mut Journal> {
        self.get_mut(path)
            .ok_or_else(|| Error::not_found(format!("there is no journal {path}")))
    }

    /// The real path a caller's `path` names, each reference on the way
    /// followed to its target (§2.5). A name that is neither a child nor a
    /// reference is an error naming the deepest journal reached, so a wrong
    /// turn costs one call.
    pub fn resolve(&self, path: &JournalPath) -> Result<JournalPath> {
        let group = path.group();
        let mut at = JournalPath::root(group.clone());
        let mut j = self
            .root(group)
            .ok_or_else(|| Error::not_found(format!("there is no group @{group}")))?;
        for n in path.names() {
            if let Some(child) = j.children.get(n) {
                at = at.child(n.clone())?;
                j = child;
                continue;
            }
            let link = j
                .meta
                .as_ref()
                .and_then(|m| m.refs.iter().find(|r| &r.name == n));
            let Some(link) = link else {
                return Err(Error::not_found(format!(
                    "{at} has no sub-journal or reference `{n}`"
                )));
            };
            at = link.to.at(group)?;
            j = self.get(&at).ok_or_else(|| {
                Error::not_found(format!(
                    "the reference `{n}` leads to {at}, which is not there"
                ))
            })?;
        }
        Ok(at)
    }

    /// The deepest journal a caller's `path` reaches, references followed:
    /// where a path that does not resolve went wrong, so its answer can show
    /// what is there instead (§6.8). `None` when not even its group is.
    pub fn reached(&self, path: &JournalPath) -> Option<JournalPath> {
        (0..=path.names().len()).rev().find_map(|keep| {
            let prefix = JournalPath::new(path.group().clone(), path.names()[..keep].to_vec());
            prefix.ok().and_then(|p| self.resolve(&p).ok())
        })
    }

    pub fn person_file(&self, person: &Slug) -> Option<&PersonFile> {
        self.persons.get(person).and_then(|p| p.file.as_ref().ok())
    }

    pub fn group_file(&self, group: &Slug) -> Option<&GroupFile> {
        self.groups.get(group).and_then(|g| g.file.as_ref().ok())
    }

    /// Every journal of every group, with its path.
    pub fn journals(&self) -> Vec<(JournalPath, &Journal)> {
        let mut out = Vec::new();
        for (group, g) in &self.groups {
            for (names, j) in g.root.walk() {
                let path = JournalPath::new(group.clone(), names)
                    .expect("a loaded journal is at a place in the tree");
                out.push((path, j));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{Meta, Ref};
    use crate::path::Below;
    use crate::testing::{at, slug};

    #[test]
    fn a_skeleton_has_the_five_sections() {
        let root = skeleton();
        let names: Vec<&str> = root.children.keys().map(Slug::as_str).collect();
        assert_eq!(
            names,
            ["bots", "conversations", "people", "threads", "topics"]
        );
        assert!(root.walk().iter().all(|(_, j)| j.is_fixed()));
    }

    fn named(place: Place, name: &str) -> Journal {
        Journal::new(
            place,
            Some(Meta::new(
                name,
                Some(format!("open for {name}")),
                at("2026-10-30T00:00:00Z"),
            )),
        )
    }

    /// **A path through a reference is its target's**, one hop at a time;
    /// a wrong turn names the deepest journal reached.
    #[test]
    fn references_resolve_to_their_targets() {
        let mut root = skeleton();
        let topic = |depth| Place::Named {
            section: Section::Topics,
            depth,
        };
        let mut candle = named(topic(1), "candle");
        candle.meta.as_mut().unwrap().refs.push(Ref {
            name: slug("wasmer"),
            to: Below::parse("topics/wasmer").unwrap(),
            cue: "open for the WebAssembly work".into(),
        });
        let mut wasmer = named(topic(1), "wasmer");
        wasmer
            .children
            .insert(slug("wasix"), named(topic(2), "wasix"));
        let topics = root.children.get_mut(&slug("topics")).unwrap();
        topics.children.insert(slug("candle"), candle);
        topics.children.insert(slug("wasmer"), wasmer);
        let mut tree = Tree::default();
        tree.groups.insert(
            slug("zen"),
            Group {
                file: Err(Problem::new("x", None, "unread")),
                root: Arc::new(root),
                problems: vec![],
            },
        );
        let p = |names: &[&str]| {
            JournalPath::new(slug("zen"), names.iter().map(|n| slug(n)).collect()).unwrap()
        };
        assert_eq!(
            tree.resolve(&p(&["topics", "candle", "wasmer", "wasix"]))
                .unwrap()
                .show(),
            "@zen/topics/wasmer/wasix"
        );
        assert_eq!(
            tree.resolve(&p(&["topics", "candle"])).unwrap().show(),
            "@zen/topics/candle"
        );
        assert_eq!(
            tree.resolve(&p(&["topics", "candle", "kv"]))
                .unwrap_err()
                .to_string(),
            "@zen/topics/candle has no sub-journal or reference `kv`"
        );
        let reached = |names: &[&str]| tree.reached(&p(names)).map(|p| p.show());
        assert_eq!(
            reached(&["topics", "candle", "kv", "paging"]).as_deref(),
            Some("@zen/topics/candle")
        );
        assert_eq!(
            reached(&["topics", "candle", "wasmer", "wasi"]).as_deref(),
            Some("@zen/topics/wasmer"),
            "a wrong turn past a reference stops at its target"
        );
        assert_eq!(reached(&["people", "yen"]).as_deref(), Some("@zen/people"));
        assert_eq!(tree.reached(&JournalPath::root(slug("nobody"))), None);
        tree.root_mut(&slug("zen"))
            .unwrap()
            .children
            .get_mut(&slug("topics"))
            .unwrap()
            .children
            .remove(&slug("wasmer"));
        assert_eq!(
            tree.resolve(&p(&["topics", "candle", "wasmer"]))
                .unwrap_err()
                .to_string(),
            "the reference `wasmer` leads to @zen/topics/wasmer, which is not there"
        );
        assert_eq!(
            tree.resolve(&JournalPath::root(slug("nobody")))
                .unwrap_err()
                .to_string(),
            "there is no group @nobody"
        );
    }
}
