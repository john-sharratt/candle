//! One journal in memory: its metadata, window, archives and children, and
//! whatever is wrong with its files.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::children::Children;
use crate::error::{Error, Problem, Result};
use crate::files::{Meta, Summary};
use crate::line::{Line, LineId};
use crate::path::Place;
use crate::period::Period;
use crate::slug::Slug;
use crate::time::Stamp;

/// The most live sub-journals a journal may have (§2.2).
pub const MAX_CHILDREN: usize = 10;

/// The most lines a window shows before its oldest closed periods fold (§4).
pub const WINDOW: usize = 50;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Journal {
    pub place: Place,
    /// `journal.yaml`. A fixed journal — a master or a section — may have
    /// none until it has a summary; a named journal always has one, and is
    /// broken without it.
    pub meta: Option<Meta>,
    pub window: Vec<Line>,
    /// Each period's lines, shared with earlier versions of the tree as its
    /// children are: a write to the journal copies its window, never its
    /// archives, unless it changes one.
    pub archives: BTreeMap<Period, Arc<Vec<Line>>>,
    /// Shared with earlier versions of the tree ([`Children`]).
    pub children: Children,
    /// What is wrong with its own files; a journal with any is read as
    /// unreadable and refuses writes until a commit fixes them.
    pub problems: Vec<Problem>,
}

/// Where a line is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    Window(usize),
    Archive(Period, usize),
}

impl Journal {
    pub fn new(place: Place, meta: Option<Meta>) -> Journal {
        Journal {
            place,
            meta,
            window: Vec::new(),
            archives: BTreeMap::new(),
            children: Children::new(),
            problems: Vec::new(),
        }
    }

    pub fn is_fixed(&self) -> bool {
        !matches!(self.place, Place::Named { .. })
    }

    pub fn summary(&self) -> Option<&Summary> {
        self.meta.as_ref().and_then(|m| m.summary.as_ref())
    }

    pub fn is_folded(&self) -> bool {
        self.meta.as_ref().is_some_and(|m| m.folded)
    }

    pub fn is_closed(&self) -> bool {
        self.meta.as_ref().is_some_and(|m| m.closed.is_some())
    }

    /// Its own cue: written for a named journal, fixed for a fixed one.
    pub fn cue(&self) -> Option<String> {
        match self.place {
            Place::Section(s) => Some(s.cue().into()),
            Place::Root => None,
            Place::Named { .. } => self.meta.as_ref().and_then(|m| m.cue.clone()),
        }
    }

    /// Nothing is written in it or below it: a fixed journal with no summary,
    /// no line and no named journal below. It writes no files. A named
    /// journal is never blank: made, it is there to be found.
    pub fn is_blank(&self) -> bool {
        self.is_fixed()
            && self.summary().is_none()
            && self.window.is_empty()
            && self.archives.is_empty()
            && self.children.values().all(Journal::is_blank)
    }

    /// Its live children, by name: not folded.
    pub fn live(&self) -> impl Iterator<Item = (&Slug, &Journal)> {
        self.children.iter().filter(|(_, c)| !c.is_folded())
    }

    pub fn live_count(&self) -> usize {
        self.live().count()
    }

    /// When anything last happened in it or below it: its newest line, or
    /// when it was created or closed.
    pub fn last_active(&self) -> Stamp {
        let own = self.window.iter().map(|l| l.at).max();
        let meta = self
            .meta
            .as_ref()
            .map(|m| m.closed.unwrap_or(m.created).max(m.created));
        let below = self.children.values().map(Journal::last_active).max();
        [own, meta, below]
            .into_iter()
            .flatten()
            .max()
            .unwrap_or(Stamp::MIN)
    }

    /// Where the line `id` is held: its window, or an archive.
    pub fn find(&self, id: &LineId) -> Option<Held> {
        if let Some(i) = self.window.iter().position(|l| &l.id == id) {
            return Some(Held::Window(i));
        }
        self.archives.iter().find_map(|(p, lines)| {
            lines
                .iter()
                .position(|l| &l.id == id)
                .map(|i| Held::Archive(*p, i))
        })
    }

    pub fn line(&self, id: &LineId) -> Option<&Line> {
        match self.find(id)? {
            Held::Window(i) => self.window.get(i),
            Held::Archive(p, i) => self.archives.get(&p)?.get(i),
        }
    }

    pub fn line_mut(&mut self, id: &LineId) -> Option<&mut Line> {
        match self.find(id)? {
            Held::Window(i) => self.window.get_mut(i),
            Held::Archive(p, i) => self.archives.get_mut(&p).map(Arc::make_mut)?.get_mut(i),
        }
    }

    /// Every line it holds, window first, then each archive.
    pub fn all_lines(&self) -> impl Iterator<Item = &Line> {
        self.window
            .iter()
            .chain(self.archives.values().flat_map(|lines| lines.iter()))
    }

    pub fn has_id(&self, id: &LineId) -> bool {
        self.find(id).is_some()
    }

    /// Refuses a write to a journal whose files are broken.
    pub fn writable(&self, shown: &str) -> Result<()> {
        match self.problems.first() {
            Some(p) => Err(Error::refused(format!(
                "{shown} cannot be written until its files are fixed: {p}"
            ))),
            None => Ok(()),
        }
    }

    /// Every journal below it, depth first, with the names leading to it.
    pub fn walk(&self) -> Vec<(Vec<Slug>, &Journal)> {
        let mut out = vec![(Vec::new(), self)];
        for (name, child) in &self.children {
            for (mut names, j) in child.walk() {
                names.insert(0, name.clone());
                out.push((names, j));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::Meta;
    use crate::scope::Section;
    use crate::testing::{at, note, slug};

    fn named(name: &str, created: &str) -> Journal {
        Journal::new(
            Place::Named {
                section: Section::Topics,
                depth: 1,
            },
            Some(Meta::new(
                name,
                Some(format!("open for {name}")),
                at(created),
            )),
        )
    }

    #[test]
    fn lines_are_found_in_the_window_and_the_archives() {
        let mut j = named("permits", "2026-10-01T00:00:00Z");
        let a = note("2026-10-29T10:00:00Z", "archived");
        let b = note("2026-10-30T10:00:00Z", "in the window");
        let day = Period::parse("2026-10-29").unwrap();
        j.archives.insert(day, Arc::new(vec![a.clone()]));
        j.window.push(b.clone());
        assert_eq!(j.find(&b.id), Some(Held::Window(0)));
        assert_eq!(j.find(&a.id), Some(Held::Archive(day, 0)));
        assert_eq!(j.line(&a.id), Some(&a));
        j.line_mut(&a.id).unwrap().text = "edited".into();
        assert_eq!(j.line(&a.id).unwrap().text, "edited");
        assert_eq!(j.all_lines().count(), 2);
        assert!(!j.has_id(&note("2026-10-30T10:00:00Z", "other").id));
    }

    #[test]
    fn live_children_are_the_unfolded_ones_and_activity_runs_up() {
        let mut parent = named("house-build", "2026-10-01T00:00:00Z");
        let mut a = named("permits", "2026-10-02T00:00:00Z");
        a.window.push(note("2026-10-30T10:00:00Z", "latest"));
        let mut b = named("budget", "2026-10-03T00:00:00Z");
        b.meta.as_mut().unwrap().folded = true;
        parent.children.insert(slug("permits"), a);
        parent.children.insert(slug("budget"), b);
        assert_eq!(parent.live_count(), 1);
        assert_eq!(parent.live().next().unwrap().0, &slug("permits"));
        assert_eq!(parent.last_active(), at("2026-10-30T10:00:00Z"));
        assert_eq!(parent.walk().len(), 3);
        assert_eq!(parent.walk()[1].0, vec![slug("budget")]);
    }

    #[test]
    fn a_broken_journal_refuses_writes() {
        let mut j = named("permits", "2026-10-01T00:00:00Z");
        assert!(j.writable("x").is_ok());
        j.problems.push(Problem::new("f", Some(2), "bad"));
        assert_eq!(
            j.writable("@zen/topics/permits").unwrap_err().to_string(),
            "@zen/topics/permits cannot be written until its files are fixed: f:2: bad"
        );
    }

    #[test]
    fn fixed_journals_have_fixed_cues() {
        let s = Journal::new(Place::Section(Section::Topics), None);
        assert_eq!(s.cue().as_deref(), Some(Section::Topics.cue()));
        assert!(s.is_fixed());
        assert_eq!(Journal::new(Place::Root, None).cue(), None);
        assert_eq!(s.last_active(), Stamp::MIN);
    }
}
