//! The identity of the record behind every replicated value the substrate
//! holds (`docs/replicated_substrate.md` §6.2, §6.5).
//!
//! A value is replaced only by a record whose stamp is at least as great
//! ([`RecordIdentity::supersedes`]), so what the substrate holds is a maximum
//! over the records it has applied — the same whatever order they came in. The
//! identity also travels with the value when a rewrite re-encodes it: a
//! compaction or a maintenance re-emission writes the record back under the
//! identity it had.

use crate::persistence::identity::RecordIdentity;
use crate::projection::TurnIndex;

use super::pcoll::POrdMap;

/// The records behind one stream's keyed values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamIds {
    pub decl: RecordIdentity,
    pub tokens: RecordIdentity,
    pub projection_events: RecordIdentity,
    pub wide_q_sigs: RecordIdentity,
    pub index_page: RecordIdentity,
    /// The `Commit` that set `committed_through`, the greatest.
    pub commit: RecordIdentity,
    pub chunks: POrdMap<u64, RecordIdentity>,
}

/// The records behind one timeline's keyed values.
///
/// A `Label` record sets any of `conv_id`, `label` and the `custom` keys, and
/// each field keeps the value of the greatest record that set it, so one
/// timeline's label is assembled from several records.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimelineIds {
    pub conv_id: RecordIdentity,
    pub label: RecordIdentity,
    pub custom: POrdMap<String, RecordIdentity>,
    /// The least record that named a `conv_id` — the conversation's place in
    /// creation order — and the `conv_id` it named. A rewrite writes it back
    /// even when a greater record holds the field, so the order survives.
    pub named: Option<RecordIdentity>,
    pub named_conv_id: String,
    pub state: RecordIdentity,
    pub debug_id: RecordIdentity,
    pub tree_meta: POrdMap<TurnIndex, RecordIdentity>,
    /// Each coupling's record, the greatest that stated it.
    pub couplings: POrdMap<u32, RecordIdentity>,
}

/// Put `id` in `slot` when it supersedes what the slot holds; answers whether
/// it did.
pub fn take(slot: &mut RecordIdentity, id: RecordIdentity) -> bool {
    if id.supersedes(slot) {
        *slot = id;
        true
    } else {
        false
    }
}

/// [`take`] for a keyed slot that may not exist yet.
pub fn take_keyed<K: Ord + Clone>(
    map: &mut POrdMap<K, RecordIdentity>,
    key: K,
    id: RecordIdentity,
) -> bool {
    match map.get_mut(&key) {
        Some(slot) => take(slot, id),
        None => {
            map.insert(key, id);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 1,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        }
    }

    #[test]
    fn a_slot_keeps_the_greatest_stamp() {
        let mut slot = RecordIdentity::NONE;
        assert!(take(&mut slot, id(5)));
        assert!(!take(&mut slot, id(3)));
        assert!(take(&mut slot, id(5)), "the same record again");
        assert!(take(&mut slot, id(9)));
        assert_eq!(slot, id(9));
    }

    #[test]
    fn a_keyed_slot_starts_with_the_first_record() {
        let mut map = POrdMap::new();
        assert!(take_keyed(&mut map, 4u64, id(2)));
        assert!(!take_keyed(&mut map, 4u64, id(1)));
        assert_eq!(map[&4], id(2));
    }
}
