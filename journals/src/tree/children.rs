//! A journal's sub-journals by name, each held behind an [`Arc`] shared with
//! every earlier version of the tree: a write makes a new tree from the one
//! readers hold (`ops::applied_to`), and cloning it copies these maps, never
//! the journals in them. Changing a journal copies only the journals on its
//! path ([`Children::get_mut`]); every other one stays shared.

use std::collections::btree_map;
use std::collections::BTreeMap;
use std::iter::Map;
use std::ops::Index;
use std::sync::Arc;

use super::Journal;
use crate::slug::Slug;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Children(BTreeMap<Slug, Arc<Journal>>);

impl Children {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &Slug) -> Option<&Journal> {
        self.0.get(name).map(|j| &**j)
    }

    /// The journal `name`, to change: copied first when an earlier tree
    /// still shares it.
    pub fn get_mut(&mut self, name: &Slug) -> Option<&mut Journal> {
        self.0.get_mut(name).map(Arc::make_mut)
    }

    pub fn insert(&mut self, name: Slug, journal: Journal) {
        self.0.insert(name, Arc::new(journal));
    }

    /// The journal `name`, to change — made by `make` when there is none.
    pub fn get_or_insert_with(
        &mut self,
        name: Slug,
        make: impl FnOnce() -> Journal,
    ) -> &mut Journal {
        Arc::make_mut(self.0.entry(name).or_insert_with(|| Arc::new(make())))
    }

    pub fn remove(&mut self, name: &Slug) -> Option<Journal> {
        self.0.remove(name).map(Arc::unwrap_or_clone)
    }

    pub fn contains_key(&self, name: &Slug) -> bool {
        self.0.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn keys(&self) -> btree_map::Keys<'_, Slug, Arc<Journal>> {
        self.0.keys()
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &Journal> + '_ {
        self.0.values().map(|j| &**j)
    }

    /// Every journal, to change: each copied first when an earlier tree
    /// still shares it.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Journal> + '_ {
        self.0.values_mut().map(Arc::make_mut)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&Slug, &Journal)> + '_ {
        self.0.iter().map(|(n, j)| (n, &**j))
    }
}

impl<'a> IntoIterator for &'a Children {
    type Item = (&'a Slug, &'a Journal);
    type IntoIter = Map<
        btree_map::Iter<'a, Slug, Arc<Journal>>,
        fn((&'a Slug, &'a Arc<Journal>)) -> (&'a Slug, &'a Journal),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|(n, j)| (n, &**j))
    }
}

impl Index<&Slug> for Children {
    type Output = Journal;

    fn index(&self, name: &Slug) -> &Journal {
        self.get(name).expect("no sub-journal by that name")
    }
}

impl FromIterator<(Slug, Journal)> for Children {
    fn from_iter<I: IntoIterator<Item = (Slug, Journal)>>(iter: I) -> Self {
        Self(iter.into_iter().map(|(n, j)| (n, Arc::new(j))).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::Place;
    use crate::scope::Section;

    fn slug(s: &str) -> Slug {
        Slug::parse(s).unwrap()
    }

    /// **A copy shares every journal until one is changed, and then only
    /// that one is copied**: the other copy still reads it as it was.
    #[test]
    fn a_copy_shares_its_journals_until_one_changes() {
        let mut a = Children::new();
        a.insert(
            slug("topics"),
            Journal::new(Place::Section(Section::Topics), None),
        );
        a.insert(
            slug("people"),
            Journal::new(Place::Section(Section::People), None),
        );
        let mut b = a.clone();
        assert!(Arc::ptr_eq(&a.0[&slug("topics")], &b.0[&slug("topics")]));
        b.get_mut(&slug("topics")).unwrap().problems.clear();
        b.get_mut(&slug("topics"))
            .unwrap()
            .children
            .insert(slug("candle"), Journal::new(Place::Root, None));
        assert!(!Arc::ptr_eq(&a.0[&slug("topics")], &b.0[&slug("topics")]));
        assert!(
            Arc::ptr_eq(&a.0[&slug("people")], &b.0[&slug("people")]),
            "not on the path"
        );
        assert!(
            a[&slug("topics")].children.is_empty(),
            "the first reads as it was"
        );
        assert_eq!(b[&slug("topics")].children.len(), 1);
        let names: Vec<&str> = b.keys().map(Slug::as_str).collect();
        assert_eq!(names, ["people", "topics"]);
        assert!(b.remove(&slug("people")).is_some());
        assert!(!b.contains_key(&slug("people")) && a.contains_key(&slug("people")));
    }
}
