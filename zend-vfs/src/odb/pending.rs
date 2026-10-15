//! Objects this member made that are in no pack yet: the blobs, trees,
//! commits and tags libgit2 writes, held in memory until a publish packs them
//! (`docs/substrate_git.md` §4.4).

use std::collections::HashMap;
use std::sync::Arc;

use super::pack::{Object, RawId};

/// The objects written and not yet packed, by id.
#[derive(Debug, Default)]
pub(crate) struct Pending {
    objects: HashMap<RawId, Arc<Object>>,
}

impl Pending {
    /// Hold `object`; its id. One already held is held once.
    pub(crate) fn insert(&mut self, object: Object) -> RawId {
        let id = object.id();
        self.objects.entry(id).or_insert_with(|| Arc::new(object));
        id
    }

    /// The object `id`, when held.
    pub(crate) fn get(&self, id: &RawId) -> Option<Arc<Object>> {
        self.objects.get(id).cloned()
    }

    /// Every held id.
    pub(crate) fn ids(&self) -> Vec<RawId> {
        self.objects.keys().copied().collect()
    }

    /// Let go of every object `packed` says a pack now holds.
    pub(crate) fn drop_packed(&mut self, packed: impl Fn(&RawId) -> bool) {
        self.objects.retain(|id, _| !packed(id));
    }

    /// How many objects are held.
    pub(crate) fn len(&self) -> usize {
        self.objects.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odb::pack::ObjectType;

    fn blob(text: &str) -> Object {
        Object {
            kind: ObjectType::Blob,
            data: text.as_bytes().to_vec(),
        }
    }

    #[test]
    fn an_object_is_held_once_by_its_id() {
        let mut p = Pending::default();
        let a = p.insert(blob("hello\n"));
        assert_eq!(a.to_string(), "ce013625030ba8dba906f756967f9e9ca394464a");
        assert_eq!(p.insert(blob("hello\n")), a);
        assert_eq!(p.len(), 1);
        assert_eq!(p.get(&a).unwrap().data, b"hello\n");
    }

    #[test]
    fn packed_objects_are_let_go() {
        let mut p = Pending::default();
        let a = p.insert(blob("a"));
        let b = p.insert(blob("bb"));
        p.drop_packed(|id| *id == a);
        assert!(p.get(&a).is_none());
        assert!(p.get(&b).is_some());
        assert_eq!(p.len(), 1);
    }
}
