//! A world's records as this member last wrote or applied them, with the NPC
//! leases among them parsed once, and the characters those leases lend this
//! member.
//!
//! Kept behind an `Arc` in `Standing`, so a reader — the cast reconcile on
//! every beat, a report, a restore — shares the value instead of copying a
//! whole world's records and parsing every lease again.

use std::collections::{BTreeMap, BTreeSet};

use super::npc_lease::{held_by, leases_of, NpcLease};
use crate::world_state::Records;

/// A world's records and the leases in them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Written {
    pub records: Records,
    pub leases: BTreeMap<u64, NpcLease>,
}

impl Written {
    /// `records`, with their leases parsed from them.
    pub fn of(records: Records) -> Self {
        let leases = leases_of(&records);
        Self { records, leases }
    }

    /// `records`, whose leases are `leases` — what an authority's write
    /// already holds, having just lent them.
    pub fn new(records: Records, leases: BTreeMap<u64, NpcLease>) -> Self {
        Self { records, leases }
    }
}

/// The characters whose lease, in any of `worlds`, names `me` as holder: the
/// characters this member runs.
pub fn held_of<'a>(worlds: impl IntoIterator<Item = &'a Written>, me: u64) -> BTreeSet<u64> {
    worlds
        .into_iter()
        .flat_map(|w| held_by(&w.leases, me))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::authority::npc_lease::hex;

    fn lease(holder: u64) -> NpcLease {
        NpcLease {
            holder: hex(holder),
            expires_ms: 60_000,
            next: None,
            waits: Vec::new(),
        }
    }

    /// **The leases are the `npclease/` records, parsed**; every other record
    /// is not a lease.
    #[test]
    fn a_world_s_leases_are_its_lease_records() {
        let records = Records::from([
            (
                "npclease/7".to_string(),
                json!({"holder": hex(3), "expires_ms": 60_000}),
            ),
            ("sim/meta".to_string(), json!({})),
        ]);
        let w = Written::of(records.clone());
        assert_eq!(w.records, records);
        assert_eq!(w.leases, BTreeMap::from([(7, lease(3))]));
    }

    /// **This member runs the characters lent to it in any world**, and none
    /// lent elsewhere.
    #[test]
    fn the_held_set_is_every_lease_naming_this_member() {
        let a = Written::new(
            Records::new(),
            BTreeMap::from([(1, lease(5)), (2, lease(9))]),
        );
        let b = Written::new(Records::new(), BTreeMap::from([(3, lease(5))]));
        assert_eq!(held_of([&a, &b], 5), BTreeSet::from([1, 3]));
        assert_eq!(held_of([&a, &b], 9), BTreeSet::from([2]));
        assert!(held_of([&a, &b], 4).is_empty());
    }
}
