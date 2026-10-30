//! The persistent collections the substrate's versioned root is built from
//! (`docs/substrate_lock_free.md` §3).
//!
//! Every commit clones the root and edits the clone; these collections make
//! the clone O(1) and an edit copy only the path it changes, so a write costs
//! what it touches whatever the substrate's size. Large values are held behind
//! an `Arc` and edited through [`arc_mut`], so a path copy shares them.

use std::hash::Hash;
use std::sync::Arc;

use imbl::shared_ptr::DefaultSharedPtr;

/// Hash map, `ahash`-hashed.
pub type PMap<K, V> = imbl::GenericHashMap<K, V, ahash::RandomState, DefaultSharedPtr>;
/// Hash set, `ahash`-hashed.
pub type PSet<T> = imbl::GenericHashSet<T, ahash::RandomState, DefaultSharedPtr>;
/// Ordered map.
pub type POrdMap<K, V> = imbl::OrdMap<K, V>;
/// Ordered set.
pub type POrdSet<T> = imbl::OrdSet<T>;
/// Vector.
pub type PVec<T> = imbl::Vector<T>;

/// `map[key]`, unshared for editing: the map's path to it is copied and the
/// value itself is cloned only if another version still holds it.
pub fn arc_mut<'a, K, V>(map: &'a mut PMap<K, Arc<V>>, key: &K) -> Option<&'a mut V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    map.get_mut(key).map(Arc::make_mut)
}

/// [`arc_mut`] for an ordered map.
pub fn ord_arc_mut<'a, K, V>(map: &'a mut POrdMap<K, Arc<V>>, key: &K) -> Option<&'a mut V>
where
    K: Ord + Clone,
    V: Clone,
{
    map.get_mut(key).map(Arc::make_mut)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An edit through a clone leaves the original version as it was: the
    /// property every reader of an older root relies on.
    #[test]
    fn an_edit_never_reaches_an_older_version() {
        let mut a: PMap<u32, Arc<Vec<u32>>> = PMap::default();
        a.insert(1, Arc::new(vec![1, 2, 3]));
        let b = a.clone();
        arc_mut(&mut a, &1).unwrap().push(4);
        assert_eq!(*a[&1], vec![1, 2, 3, 4]);
        assert_eq!(*b[&1], vec![1, 2, 3]);
        let mut o: POrdMap<u32, Arc<String>> = POrdMap::new();
        o.insert(7, Arc::new("x".into()));
        let p = o.clone();
        ord_arc_mut(&mut o, &7).unwrap().push('y');
        assert_eq!(o[&7].as_str(), "xy");
        assert_eq!(p[&7].as_str(), "x");
    }
}
