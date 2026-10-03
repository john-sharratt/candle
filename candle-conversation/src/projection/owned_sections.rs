//! Sections a single conversation owns, added after it opened.
//!
//! A schema's sections are sealed when a conversation is created and belong to
//! every conversation built from that schema. An **owned** section is different:
//! it is submitted at runtime, joins one collection for one conversation only,
//! and goes away with that conversation (or when it is removed by reference).
//!
//! This module holds the two pieces of bookkeeping that make that safe:
//!
//! - **Ids.** The substrate's section map is keyed by [`SectionId`] across every
//!   conversation, and a [`super::Builder`] numbers its own additions from its
//!   own highest id, so two conversations adding sections to their own builders
//!   would collide. Owned sections draw from one counter in a partition of the id
//!   space of their own, below the transient plain-prompt frames. An id is never
//!   handed out twice by one substrate, so a released section's id can never
//!   alias a later one's still-resident K/V or side-table entry.
//! - **Ownership.** For each owning timeline, the sections it owns. A timeline
//!   tombstone releases everything it owns, so the cascade needs no handle to the
//!   conversation that submitted them.

use std::collections::HashMap;

use super::plain_prompt::TRANSIENT_FLOOR;
use super::{Reserved, SectionId, TimelineId};

/// Ids in the owned partition.
const SLOTS: u32 = 1 << 24;

/// The lowest id an owned section can take.
const FLOOR: u32 = TRANSIENT_FLOOR - SLOTS;

const _: () = {
    assert!(
        TRANSIENT_FLOOR < u32::MAX - Reserved::COUNT,
        "the owned partition reaches the reserved band"
    );
    assert!(
        FLOOR > 1 << 31,
        "the owned partition reaches into schema-allocated ids"
    );
};

/// One section a timeline owns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedSection {
    /// The id the section is registered under in the substrate.
    pub section: SectionId,
    /// The caller's name for it, unique among the owner's sections.
    pub name: String,
}

/// The owned-section id counter and the per-timeline ownership registry.
#[derive(Debug, Default)]
pub struct OwnedSections {
    /// Ids handed out so far.
    issued: u32,
    by_owner: HashMap<TimelineId, Vec<OwnedSection>>,
}

impl OwnedSections {
    /// The next unused id in the owned partition, or `None` when the partition
    /// is exhausted.
    pub fn allocate(&mut self) -> Option<SectionId> {
        if self.issued >= SLOTS {
            return None;
        }
        let id = SectionId::new(FLOOR + self.issued);
        self.issued += 1;
        Some(id)
    }

    /// Record that `owner` owns `section` under `name`. A second section under
    /// the same name replaces the first's entry; the caller releases the first.
    pub fn register(&mut self, owner: TimelineId, name: &str, section: SectionId) {
        let owned = self.by_owner.entry(owner).or_default();
        owned.retain(|o| o.name != name);
        owned.push(OwnedSection {
            section,
            name: name.to_string(),
        });
    }

    /// What `owner` owns, in submission order.
    pub fn of(&self, owner: TimelineId) -> &[OwnedSection] {
        self.by_owner.get(&owner).map_or(&[], Vec::as_slice)
    }

    /// The section `owner` owns under `name`.
    pub fn named(&self, owner: TimelineId, name: &str) -> Option<SectionId> {
        self.of(owner)
            .iter()
            .find(|o| o.name == name)
            .map(|o| o.section)
    }

    /// Whether `owner` owns `section`.
    pub fn owns(&self, owner: TimelineId, section: SectionId) -> bool {
        self.of(owner).iter().any(|o| o.section == section)
    }

    /// Forget `owner`'s claim on `section`. Returns whether it held one.
    pub fn release(&mut self, owner: TimelineId, section: SectionId) -> bool {
        let Some(owned) = self.by_owner.get_mut(&owner) else {
            return false;
        };
        let before = owned.len();
        owned.retain(|o| o.section != section);
        let released = owned.len() != before;
        if owned.is_empty() {
            self.by_owner.remove(&owner);
        }
        released
    }

    /// Forget every owner's claims. The counter keeps its place, so ids issued
    /// before the clear are not issued again.
    pub fn clear_ownership(&mut self) {
        self.by_owner.clear();
    }

    /// Forget everything `owner` owns, returning it.
    pub fn release_all(&mut self, owner: TimelineId) -> Vec<OwnedSection> {
        self.by_owner.remove(&owner).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timeline(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    #[test]
    fn ids_are_distinct_and_inside_the_partition() {
        let mut owned = OwnedSections::default();
        let a = owned.allocate().unwrap();
        let b = owned.allocate().unwrap();
        assert_ne!(a, b);
        assert_eq!(a.raw(), FLOOR);
        assert_eq!(b.raw(), FLOOR + 1);
        assert!(b.raw() < TRANSIENT_FLOOR);
    }

    #[test]
    fn the_partition_is_finite() {
        let mut owned = OwnedSections {
            issued: SLOTS - 1,
            ..Default::default()
        };
        assert_eq!(owned.allocate().unwrap().raw(), TRANSIENT_FLOOR - 1);
        assert!(owned.allocate().is_none());
    }

    #[test]
    fn ownership_is_per_timeline() {
        let mut owned = OwnedSections::default();
        let (a, b) = (timeline(1), timeline(2));
        let s = owned.allocate().unwrap();
        owned.register(a, "mail", s);
        assert!(owned.owns(a, s));
        assert!(!owned.owns(b, s));
        assert_eq!(owned.named(a, "mail"), Some(s));
        assert_eq!(owned.named(b, "mail"), None);
    }

    #[test]
    fn a_name_is_held_by_one_section() {
        let mut owned = OwnedSections::default();
        let a = timeline(1);
        let (first, second) = (owned.allocate().unwrap(), owned.allocate().unwrap());
        owned.register(a, "mail", first);
        owned.register(a, "mail", second);
        assert_eq!(owned.named(a, "mail"), Some(second));
        assert_eq!(owned.of(a).len(), 1);
    }

    #[test]
    fn release_drops_one_claim_and_reports_it() {
        let mut owned = OwnedSections::default();
        let a = timeline(1);
        let (x, y) = (owned.allocate().unwrap(), owned.allocate().unwrap());
        owned.register(a, "x", x);
        owned.register(a, "y", y);
        assert!(owned.release(a, x));
        assert!(!owned.release(a, x), "a second release finds nothing");
        assert_eq!(
            owned.of(a),
            &[OwnedSection {
                section: y,
                name: "y".into()
            }]
        );
        assert!(owned.release(a, y));
        assert!(owned.of(a).is_empty());
    }

    #[test]
    fn release_all_returns_everything_in_order() {
        let mut owned = OwnedSections::default();
        let (a, b) = (timeline(1), timeline(2));
        let (x, y, z) = (
            owned.allocate().unwrap(),
            owned.allocate().unwrap(),
            owned.allocate().unwrap(),
        );
        owned.register(a, "x", x);
        owned.register(a, "y", y);
        owned.register(b, "z", z);
        let released: Vec<_> = owned
            .release_all(a)
            .into_iter()
            .map(|o| o.section)
            .collect();
        assert_eq!(released, vec![x, y]);
        assert!(owned.of(a).is_empty());
        assert!(owned.owns(b, z), "another owner is untouched");
        assert!(owned.release_all(a).is_empty());
    }
}
