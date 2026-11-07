//! A reading's face: the opening turn a dialogue holds an ingest unit by
//! (`docs/zend_working_set.md` §4.9).
//!
//! An ingest unit opens on the request that commissioned it — "Read the entire
//! contents of …", "Summarize …" — because the model doing the reading has to
//! be asked. A dialogue that places the reading must not hold that request: a
//! user turn reads as the user's, and a worker placed beside a dozen of them
//! answered one of them in place of its own task (6 of 12 workers, 2026-11-07).
//!
//! So a unit has a face: a one-turn conversation sealed beside it whose user
//! half names the reading by its anchor alone and whose assistant half is the
//! unit's own opening, and a dialogue's projection emits the face where the
//! unit's opening turn would go. The face is a turn like any other — its own
//! K/V, its own index pages — so nothing is windowed and nothing goes in
//! unindexed.
//!
//! The unit names its face under [`Substrate::FACE_KEY`] and the face names its
//! unit under [`Substrate::FACE_OF_KEY`] — plain `custom` entries like
//! [`Substrate::FORKED_FROM_KEY`], durable through the Label record they ride.
//! The two must agree: a face whose unit names another (a crash between the
//! face sealing and its unit naming it) stands for nothing.
//!
//! A face only stands in for the unit's K/V. What was SELECTED is still the
//! unit's opening, so a projection's record names the unit
//! ([`Substrate::unit_of`]): its score, its origin and the belief the next
//! reprojection is seeded from all stay the unit's.

use super::Substrate;
use crate::projection::{TimelineId, TurnIndex, TurnKey};

/// The turn a unit opens on, and the one a face holds.
const OPENING: TurnIndex = TurnIndex(0);

impl Substrate {
    /// Metadata key under which an ingest unit names its face.
    pub const FACE_KEY: &'static str = "face";

    /// Metadata key under which a face names its unit.
    pub const FACE_OF_KEY: &'static str = "face_of";

    /// The face `unit` names, live or not.
    pub fn face_timeline(&self, unit: TimelineId) -> Option<TimelineId> {
        let raw: u64 = self.custom_of(unit)?.get(Self::FACE_KEY)?.parse().ok()?;
        TimelineId::from_raw(raw)
    }

    /// What a dialogue holds in place of `turn`: its unit's face, when `turn`
    /// is a unit's opening and the face it names is live and holds its turn.
    ///
    /// A face that is gone or not yet sealed reads as absent, and the unit's
    /// own opening stands, exactly as it did before the unit had one.
    pub fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        if turn.index != OPENING {
            return None;
        }
        let face = self.face_timeline(turn.timeline)?;
        if self.is_tombstoned(face) || self.turn_token_count_of(face, OPENING) == 0 {
            return None;
        }
        Some(TurnKey::new(face, OPENING))
    }

    /// The unit opening `face` stands for — [`Self::face_of`] read backwards —
    /// when `face` is a face's turn and its unit names it.
    pub fn unit_of(&self, face: TurnKey) -> Option<TurnKey> {
        if face.index != OPENING {
            return None;
        }
        let raw: u64 = self
            .custom_of(face.timeline)?
            .get(Self::FACE_OF_KEY)?
            .parse()
            .ok()?;
        let unit = TimelineId::from_raw(raw)?;
        (self.face_timeline(unit) == Some(face.timeline)).then(|| TurnKey::new(unit, OPENING))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::persistence::identity::RecordIdentity;
    use crate::projection::{GroupId, LayerId, TimelineAllocator};

    /// Record `key = value` in `timeline`'s `custom` metadata.
    fn name(sub: &mut Substrate, timeline: TimelineId, key: &str, value: TimelineId) {
        let mut kv = BTreeMap::new();
        kv.insert(key.to_string(), value.raw().to_string());
        sub.merge_custom(timeline, &kv, RecordIdentity::NONE);
    }

    /// A unit of three turns and a face of one, each naming the other.
    fn unit_and_face() -> (Substrate, TimelineId, TimelineId) {
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let (unit, face) = (alloc.next(), alloc.next());
        sub.register_timeline(unit, LayerId::for_test(1), GroupId::for_test(1));
        sub.register_timeline(face, LayerId::for_test(2), GroupId::for_test(2));
        for _ in 0..3 {
            sub.append_with_blocks(unit, 10, 0, 1);
        }
        sub.append_with_blocks(face, 4, 0, 1);
        name(&mut sub, unit, Substrate::FACE_KEY, face);
        name(&mut sub, face, Substrate::FACE_OF_KEY, unit);
        (sub, unit, face)
    }

    /// **A face's turn reads back as its unit's opening**, and no other turn
    /// of either does.
    #[test]
    fn a_face_stands_for_its_units_opening() {
        let (sub, unit, face) = unit_and_face();
        assert_eq!(
            sub.unit_of(TurnKey::new(face, TurnIndex(0))),
            Some(TurnKey::new(unit, TurnIndex(0)))
        );
        assert_eq!(sub.unit_of(TurnKey::new(unit, TurnIndex(0))), None);
        assert_eq!(sub.unit_of(TurnKey::new(face, TurnIndex(1))), None);
    }

    /// **A face whose unit names another stands for nothing** — the face a
    /// crash left sealed before its unit named it.
    #[test]
    fn a_face_its_unit_does_not_name_stands_for_nothing() {
        let (mut sub, unit, face) = unit_and_face();
        let other = TimelineAllocator::new().next();
        name(&mut sub, unit, Substrate::FACE_KEY, other);
        assert_eq!(sub.unit_of(TurnKey::new(face, TurnIndex(0))), None);
    }

    /// **A unit's opening is held by its face, and nothing else is.**
    #[test]
    fn only_the_opening_turn_has_a_face() {
        let (sub, unit, face) = unit_and_face();
        assert_eq!(
            sub.face_of(TurnKey::new(unit, TurnIndex(0))),
            Some(TurnKey::new(face, TurnIndex(0)))
        );
        assert_eq!(sub.face_of(TurnKey::new(unit, TurnIndex(1))), None);
        assert_eq!(sub.face_of(TurnKey::new(unit, TurnIndex(2))), None);
    }

    /// **A tombstoned face is no face**: the unit's own opening stands.
    #[test]
    fn a_tombstoned_face_is_none() {
        let (mut sub, unit, face) = unit_and_face();
        sub.tombstone_timeline(face);
        assert_eq!(sub.face_of(TurnKey::new(unit, TurnIndex(0))), None);
        assert_eq!(sub.face_timeline(unit), Some(face));
    }

    /// **A face that holds no turn yet is none either** — one whose turn has
    /// not sealed.
    #[test]
    fn a_face_without_its_turn_is_none() {
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let (unit, face) = (alloc.next(), alloc.next());
        sub.register_timeline(unit, LayerId::for_test(1), GroupId::for_test(1));
        sub.register_timeline(face, LayerId::for_test(2), GroupId::for_test(2));
        sub.append_with_blocks(unit, 10, 0, 1);
        name(&mut sub, unit, Substrate::FACE_KEY, face);
        assert_eq!(sub.face_of(TurnKey::new(unit, TurnIndex(0))), None);
    }

    /// **A unit that names no face has none.**
    #[test]
    fn a_unit_without_a_face_has_none() {
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let unit = alloc.next();
        sub.register_timeline(unit, LayerId::for_test(1), GroupId::for_test(1));
        sub.append_with_blocks(unit, 10, 0, 1);
        assert_eq!(sub.face_of(TurnKey::new(unit, TurnIndex(0))), None);
        assert_eq!(sub.face_timeline(unit), None);
    }
}
