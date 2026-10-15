//! Decoded objects a pack's delta chains were resolved through, kept within a
//! byte budget: the next read down the same chain starts from the nearest one
//! instead of the chain's whole base.
//!
//! Least recently used goes first. One cache serves one pack, keyed by entry
//! offset.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use super::Object;

/// The decoded objects of one pack, by entry offset.
#[derive(Debug)]
pub(crate) struct BaseCache {
    /// Each object with the stamp of its latest use.
    held: HashMap<u64, (Arc<Object>, u64)>,
    /// Every use, oldest first, as `(offset, stamp)`: a use whose stamp is no
    /// longer its object's latest is stale and skipped.
    order: VecDeque<(u64, u64)>,
    stamp: u64,
    bytes: usize,
    budget: usize,
}

impl BaseCache {
    /// A cache holding at most `budget` bytes of object data.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            held: HashMap::new(),
            order: VecDeque::new(),
            stamp: 0,
            bytes: 0,
            budget,
        }
    }

    fn touch(&mut self, offset: u64) -> u64 {
        self.stamp += 1;
        self.order.push_back((offset, self.stamp));
        self.stamp
    }

    /// The object decoded at `offset`, when held.
    pub(crate) fn get(&mut self, offset: u64) -> Option<Arc<Object>> {
        let object = Arc::clone(&self.held.get(&offset)?.0);
        let stamp = self.touch(offset);
        self.held.get_mut(&offset).expect("held above").1 = stamp;
        self.compact();
        Some(object)
    }

    /// Keep `object`, decoded at `offset`, making room as needed. An object
    /// larger than the whole budget is not kept.
    pub(crate) fn put(&mut self, offset: u64, object: Arc<Object>) {
        let size = object.data.len();
        if size > self.budget {
            return;
        }
        let stamp = self.touch(offset);
        if let Some((old, _)) = self.held.insert(offset, (object, stamp)) {
            self.bytes -= old.data.len();
        }
        self.bytes += size;
        while self.bytes > self.budget {
            let Some((oldest, stamp)) = self.order.pop_front() else {
                break;
            };
            if self
                .held
                .get(&oldest)
                .is_some_and(|(_, latest)| *latest == stamp)
            {
                let (gone, _) = self.held.remove(&oldest).expect("held above");
                self.bytes -= gone.data.len();
            }
        }
        self.compact();
    }

    /// Drop stale uses once they outnumber the live ones, so a run of gets
    /// never grows the queue without bound.
    fn compact(&mut self) {
        if self.order.len() <= 2 * self.held.len() + 64 {
            return;
        }
        let held = &self.held;
        self.order
            .retain(|(offset, stamp)| held.get(offset).is_some_and(|(_, l)| l == stamp));
    }

    /// How many bytes of object data are held.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odb::pack::ObjectType;

    fn blob(n: usize) -> Arc<Object> {
        Arc::new(Object {
            kind: ObjectType::Blob,
            data: vec![0; n],
        })
    }

    /// **The least recently used goes first**, and a use counts as recent.
    #[test]
    fn the_least_recently_used_object_goes_first() {
        let mut cache = BaseCache::new(30);
        cache.put(1, blob(10));
        cache.put(2, blob(10));
        cache.put(3, blob(10));
        assert!(cache.get(1).is_some(), "1 is now the most recent");
        cache.put(4, blob(10));
        assert!(cache.get(2).is_none(), "2 was least recently used");
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
        assert_eq!(cache.bytes(), 30);
    }

    #[test]
    fn an_object_over_the_budget_is_not_kept() {
        let mut cache = BaseCache::new(5);
        cache.put(1, blob(6));
        assert!(cache.get(1).is_none());
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn many_gets_do_not_grow_the_order_without_bound() {
        let mut cache = BaseCache::new(100);
        cache.put(1, blob(1));
        for _ in 0..10_000 {
            cache.get(1);
        }
        cache.put(2, blob(1));
        assert!(cache.order.len() < 100);
        assert!(cache.get(1).is_some() && cache.get(2).is_some());
    }

    #[test]
    fn putting_an_offset_again_replaces_it() {
        let mut cache = BaseCache::new(100);
        cache.put(1, blob(10));
        cache.put(1, blob(20));
        assert_eq!(cache.bytes(), 20);
        assert_eq!(cache.get(1).unwrap().data.len(), 20);
    }
}
