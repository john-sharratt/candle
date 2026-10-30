//! The turns in a gather scope, indexed by the tag that puts them there.
//!
//! A tagged collection's belief gallery is drawn from the turns carrying one
//! of its tags — the tool catalog from the few thousand `tool` calibration
//! turns. Every gather scope shares one set of tagged turns, though, and the
//! calibration questions alone are tens of thousands of them, so a gallery
//! that walked them all to keep its own spent ~75 ms a scan, and was rebuilt
//! on every scan that followed a seal. Indexed by tag, the walk reads only the
//! turns that carry the collection's tags.
//!
//! A turn's tags are its gather tags: a working-set mark puts no turn in a
//! scope (`working_set::marks`), so a turn carrying only marks is not here.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::pcoll::{PMap, POrdSet};
use super::stream_index::StreamKey;
use crate::working_set::marks::gather_tags;

/// See the module documentation. Persistent: a clone is O(1).
#[derive(Debug, Default, Clone)]
pub struct TagIndex {
    by_tag: PMap<String, POrdSet<StreamKey>>,
    tags_of: PMap<StreamKey, Arc<Vec<String>>>,
}

impl TagIndex {
    /// `key` is declared a turn with `tags`, replacing whatever it was declared
    /// with before. A turn with no gather tag leaves the index.
    pub fn set(&mut self, key: StreamKey, tags: &[String]) {
        self.remove(key);
        let gather: Vec<String> = gather_tags(tags).cloned().collect();
        if gather.is_empty() {
            return;
        }
        for tag in &gather {
            self.by_tag.entry(tag.clone()).or_default().insert(key);
        }
        self.tags_of.insert(key, Arc::new(gather));
    }

    /// `key` is no longer declared a turn.
    pub fn remove(&mut self, key: StreamKey) {
        let Some(tags) = self.tags_of.remove(&key) else {
            return;
        };
        for tag in tags.iter() {
            if let Some(keys) = self.by_tag.get_mut(tag) {
                keys.remove(&key);
                if keys.is_empty() {
                    self.by_tag.remove(tag);
                }
            }
        }
    }

    /// Every turn carrying one of `tags`, each once, in key order — the order
    /// a walk of every tagged turn keeping these would visit them in.
    pub fn with_any(&self, tags: &[String]) -> Vec<StreamKey> {
        let mut sets = tags.iter().filter_map(|t| self.by_tag.get(t));
        match (sets.next(), sets.next()) {
            (None, _) => Vec::new(),
            (Some(one), None) => one.iter().copied().collect(),
            (Some(a), Some(b)) => {
                let mut all: BTreeSet<StreamKey> = a.iter().chain(b.iter()).copied().collect();
                for s in sets {
                    all.extend(s.iter().copied());
                }
                all.into_iter().collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::streams::StreamId;

    fn key(id: u64) -> StreamKey {
        StreamKey {
            stream_id: StreamId(id),
            writer: 0,
        }
    }

    fn tags(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    /// **A collection reads the turns carrying its tags, and only those**, in
    /// key order, each once however many of its tags a turn carries.
    #[test]
    fn a_lookup_takes_the_turns_carrying_any_tag_in_key_order() {
        let mut idx = TagIndex::default();
        idx.set(key(5), &tags(&["tool"]));
        idx.set(key(2), &tags(&["calibration"]));
        idx.set(key(3), &tags(&["tool", "calibration"]));
        idx.set(key(1), &tags(&["tool"]));
        assert_eq!(idx.with_any(&tags(&["tool"])), vec![key(1), key(3), key(5)]);
        assert_eq!(
            idx.with_any(&tags(&["calibration", "tool"])),
            vec![key(1), key(2), key(3), key(5)]
        );
        assert_eq!(idx.with_any(&tags(&["code"])), Vec::<StreamKey>::new());
        assert_eq!(idx.with_any(&[]), Vec::<StreamKey>::new());
    }

    /// **A redeclaration moves a turn between tags; marks scope nothing.** A
    /// turn redeclared under another tag leaves the old one, a turn carrying
    /// only working-set marks is in no scope, and a removed or cleared turn is
    /// in none.
    #[test]
    fn a_redeclaration_moves_a_turn_and_marks_scope_nothing() {
        let mut idx = TagIndex::default();
        idx.set(key(1), &tags(&["tool"]));
        idx.set(key(1), &tags(&["code", "working_set:release"]));
        assert_eq!(idx.with_any(&tags(&["tool"])), Vec::<StreamKey>::new());
        assert_eq!(idx.with_any(&tags(&["code"])), vec![key(1)]);
        assert_eq!(
            idx.with_any(&tags(&["working_set:release"])),
            Vec::<StreamKey>::new()
        );

        idx.set(key(1), &tags(&["working_set:release"]));
        assert_eq!(idx.with_any(&tags(&["code"])), Vec::<StreamKey>::new());
        assert!(idx.by_tag.is_empty() && idx.tags_of.is_empty());

        idx.set(key(2), &tags(&["tool"]));
        idx.remove(key(2));
        idx.remove(key(9));
        assert_eq!(idx.with_any(&tags(&["tool"])), Vec::<StreamKey>::new());
    }
}
