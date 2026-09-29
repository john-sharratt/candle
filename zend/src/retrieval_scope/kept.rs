//! A few values worked out recently, by key: the least recently used is let
//! go first once the capacity is reached.

/// At most `capacity` values, least recently used first.
pub struct Kept<K, V> {
    capacity: usize,
    entries: Vec<(K, V)>,
}

impl<K: PartialEq, V: Clone> Kept<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity + 1),
        }
    }

    /// The value kept under `key`, now the most recently used.
    pub fn get(&mut self, key: &K) -> Option<V> {
        let at = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(at);
        let value = entry.1.clone();
        self.entries.push(entry);
        Some(value)
    }

    /// Keep `value` under `key`, replacing what was kept under it.
    pub fn insert(&mut self, key: K, value: V) {
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, value));
        if self.entries.len() > self.capacity {
            self.entries.remove(0);
        }
    }

    /// Let go of every value whose key `keep` refuses.
    pub fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        self.entries.retain(|(k, _)| keep(k));
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<&K> {
        self.entries.iter().map(|(k, _)| k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The least recently used goes first**, and reading a value counts as
    /// using it.
    #[test]
    fn the_least_recently_used_goes_first() {
        let mut kept = Kept::new(2);
        kept.insert(1, "one");
        kept.insert(2, "two");
        assert_eq!(kept.get(&1), Some("one"));
        kept.insert(3, "three");
        assert_eq!(kept.keys(), [&1, &3], "2 was the least recently used");
        assert_eq!(kept.get(&2), None);
    }

    /// A key inserted again is replaced, not kept twice.
    #[test]
    fn a_key_is_kept_once() {
        let mut kept = Kept::new(2);
        kept.insert(1, "one");
        kept.insert(1, "uno");
        assert_eq!(kept.keys(), [&1]);
        assert_eq!(kept.get(&1), Some("uno"));
    }

    #[test]
    fn retain_lets_go_of_what_it_refuses() {
        let mut kept = Kept::new(4);
        for n in 1..=4 {
            kept.insert(n, n * 10);
        }
        kept.retain(|k| k % 2 == 0);
        assert_eq!(kept.keys(), [&2, &4]);
    }
}
