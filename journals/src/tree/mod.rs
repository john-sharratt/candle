//! The whole repository in memory: each person's tree and each group's, read
//! from the files and written back to them.
//!
//! | Module | |
//! |---|---|
//! | [`journal`] | one journal: metadata, window, archives, children, problems |
//! | [`load`] | the files into a tree, every fault kept as a problem |
//! | [`check`] | the rules a loaded tree is held to |
//! | [`render`] | a tree, or one journal, back into files |
//! | [`fingerprint`] | what a summary is built over, and so what is dirty |
//! | [`window`] | which closed period of a window folds next |
//! | [`accounts`] | which person an account is |

pub mod accounts;
pub mod check;
pub mod fingerprint;
pub mod journal;
pub mod load;
pub mod render;
pub mod window;

use std::collections::BTreeMap;

use crate::error::{Error, Problem, Result};
use crate::files::{GroupFile, PersonFile};
use crate::path::{JournalPath, Owner, Place};
use crate::scope::{Channel, Scope, Section};
use crate::slug::Slug;

pub use journal::{Held, Journal, MAX_CHILDREN, WINDOW};

/// A person: their `person.yaml` (or why it cannot be read) and their tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Person {
    pub file: Result<PersonFile, Problem>,
    pub root: Journal,
    /// Faults in `person.yaml` against the rest of the repository.
    pub problems: Vec<Problem>,
}

/// A group: its `group.yaml` (or why it cannot be read) and its tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub file: Result<GroupFile, Problem>,
    pub root: Journal,
    pub problems: Vec<Problem>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tree {
    pub persons: BTreeMap<Slug, Person>,
    pub groups: BTreeMap<Slug, Group>,
    /// Files that are no part of the layout, and faults across owners.
    pub problems: Vec<Problem>,
}

/// A root and its fixed sections — and under `channels`, a journal per
/// channel — with no files yet.
pub fn skeleton(scope: Scope) -> Journal {
    let mut root = Journal::new(Place::Root, None);
    for section in Section::of(scope) {
        let mut s = Journal::new(Place::Section(section), None);
        if section == Section::Channels {
            for c in Channel::ALL {
                s.children.insert(
                    Slug::parse(c.name()).expect("a channel's name is a name"),
                    Journal::new(Place::Channel(c), None),
                );
            }
        }
        root.children.insert(
            Slug::parse(section.name()).expect("a section's name is a name"),
            s,
        );
    }
    root
}

impl Tree {
    pub fn root(&self, owner: &Owner) -> Option<&Journal> {
        match owner {
            Owner::Person(p) => self.persons.get(p).map(|x| &x.root),
            Owner::Group(g) => self.groups.get(g).map(|x| &x.root),
        }
    }

    pub fn root_mut(&mut self, owner: &Owner) -> Option<&mut Journal> {
        match owner {
            Owner::Person(p) => self.persons.get_mut(p).map(|x| &mut x.root),
            Owner::Group(g) => self.groups.get_mut(g).map(|x| &mut x.root),
        }
    }

    pub fn get(&self, path: &JournalPath) -> Option<&Journal> {
        let mut j = self.root(path.owner())?;
        for n in path.names() {
            j = j.children.get(n)?;
        }
        Some(j)
    }

    pub fn get_mut(&mut self, path: &JournalPath) -> Option<&mut Journal> {
        let mut j = self.root_mut(path.owner())?;
        for n in path.names() {
            j = j.children.get_mut(n)?;
        }
        Some(j)
    }

    /// The journal at `path`, or an error naming what is not there.
    pub fn need(&self, path: &JournalPath) -> Result<&Journal> {
        self.get(path)
            .ok_or_else(|| Error::not_found(format!("there is no journal {path}")))
    }

    pub fn need_mut(&mut self, path: &JournalPath) -> Result<&mut Journal> {
        self.get_mut(path)
            .ok_or_else(|| Error::not_found(format!("there is no journal {path}")))
    }

    pub fn person_file(&self, person: &Slug) -> Option<&PersonFile> {
        self.persons.get(person).and_then(|p| p.file.as_ref().ok())
    }

    pub fn group_file(&self, group: &Slug) -> Option<&GroupFile> {
        self.groups.get(group).and_then(|g| g.file.as_ref().ok())
    }

    /// Every journal of every owner, with its path.
    pub fn journals(&self) -> Vec<(JournalPath, &Journal)> {
        let owners = self
            .persons
            .iter()
            .map(|(p, x)| (Owner::Person(p.clone()), &x.root))
            .chain(
                self.groups
                    .iter()
                    .map(|(g, x)| (Owner::Group(g.clone()), &x.root)),
            );
        let mut out = Vec::new();
        for (owner, root) in owners {
            for (names, j) in root.walk() {
                let path = JournalPath::new(owner.clone(), names)
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
    use crate::testing::slug;

    #[test]
    fn a_skeleton_has_the_fixed_sections_and_channels() {
        let person = skeleton(Scope::Personal);
        let names: Vec<&str> = person.children.keys().map(Slug::as_str).collect();
        assert_eq!(names, ["channels", "people"]);
        let channels: Vec<&str> = person.children[&slug("channels")]
            .children
            .keys()
            .map(Slug::as_str)
            .collect();
        assert_eq!(channels, ["claude-code", "grok", "other"]);
        let group = skeleton(Scope::Group);
        let names: Vec<&str> = group.children.keys().map(Slug::as_str).collect();
        assert_eq!(names, ["bots", "projects", "threads", "topics"]);
        assert!(group.walk().iter().all(|(_, j)| j.is_fixed()));
    }
}
