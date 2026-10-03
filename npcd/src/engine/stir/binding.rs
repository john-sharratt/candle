//! Fixtures that are objects in the room, and the objects that are fixtures.
//!
//! # What this is for
//!
//! A stirring is a sentence. Left at that, the building could say "a breaker
//! trips" while there was no breaker anywhere in the world to look at, and a
//! character that went to deal with it had nothing to deal with. A fixture that
//! has a [`Bound`] is different: its fault is the mode of a real, operable part
//! standing in the room, and the two are kept in step.
//!
//! * **The fixture changes first.** The coolant starts weeping, and the
//!   `coolant-valve` in the room goes to its `weeping` mode. A character that
//!   looks sees it, and can set it back.
//! * **The object changes first.** A character operates the valve to `tight`.
//!   The next [`Building::tend`] sees the object disagree with the fixture,
//!   takes the object as the truth, and asks the fixture to bring its own state
//!   into line ([`Fixture::set_fault`]) — and the room hears what that sounded
//!   like. The fixture cannot mend itself; it can be mended.
//!
//! The device is the source of truth in a disagreement, because it is the thing
//! a character acted on.
//!
//! # Gating
//!
//! [`Fixture::needs`] names the parts a fixture cannot run without. A building
//! is fitted only with the fixtures whose parts stand in the room, so the
//! building never describes a system the room does not hold.
//!
//! # The board
//!
//! The status board is the one fixture that speaks of *other* rooms. It is told
//! the standing faults of the world ([`Building::hear_of`]) as [`Report`]s and
//! files an entry only for one that exists, and takes it down when it is gone.

use std::time::Duration;

use crate::engine::stir::{Building, Stirring};

/// The operable object a fixture's fault is the state of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    /// The part id of the object, as a map places it.
    pub part: &'static str,
    /// The mode the object rests in while the fixture is well.
    pub ok: &'static str,
    /// The mode the object is in while the fixture is faulted.
    pub fault: &'static str,
    /// What a fault board files this under, e.g. "the main supply bus".
    pub system: &'static str,
    /// What is wrong, as a board entry says it, e.g. "a breaker has tripped".
    pub trouble: &'static str,
    /// The object, named as a person names it, e.g. "the breaker panel".
    pub object: &'static str,
}

/// A fault standing on a real object somewhere in the world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// The room it stands in.
    pub room: String,
    pub object: &'static str,
    pub system: &'static str,
    pub trouble: &'static str,
}

/// A mode a fixture has put its object into, for the world to apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Set {
    pub part: &'static str,
    pub mode: &'static str,
}

impl Building {
    /// The ids of the fixtures this building was fitted with.
    pub fn fitted_ids(&self) -> Vec<&'static str> {
        self.fixtures.iter().map(|f| f.id()).collect()
    }

    /// The faults standing in this room right now, filed against `room`.
    pub fn standing(&self, room: &str) -> Vec<Report> {
        self.fixtures
            .iter()
            .filter(|f| f.faulted())
            .filter_map(|f| f.bound())
            .map(|b| Report {
                room: room.to_string(),
                object: b.object,
                system: b.system,
                trouble: b.trouble,
            })
            .collect()
    }

    /// Tell the building which faults stand in the world, for its board.
    pub fn hear_of(&mut self, standing: Vec<Report>) {
        self.standing = standing;
    }

    /// The object modes the fixtures have changed since the last call, for the
    /// world to apply to the room's objects.
    pub fn take_sets(&mut self) -> Vec<Set> {
        std::mem::take(&mut self.outbox)
    }

    /// Reconcile the room's objects with its fixtures, at `since_start`.
    ///
    /// `seen` is each bound part's current mode in this room, as `(part,
    /// mode)`. Where the object disagrees with what was last written to it, a
    /// character operated it: the fixture is told, and what the room notices of
    /// that is returned. Parts not in `seen` are left alone.
    pub fn tend(&mut self, since_start: Duration, seen: &[(&str, &str)]) -> Vec<Stirring> {
        let w = self.watch(since_start);
        let mut said = Vec::new();
        for i in 0..self.fixtures.len() {
            let Some(bound) = self.fixtures[i].bound() else {
                continue;
            };
            let Some((_, mode)) = seen.iter().find(|(part, _)| *part == bound.part) else {
                continue;
            };
            let faulted = *mode == bound.fault;
            if faulted == self.synced[i] {
                continue;
            }
            self.synced[i] = faulted;
            if let Some(s) = self.fixtures[i].set_fault(faulted) {
                said.push(s);
            }
        }
        for s in &said {
            for f in &mut self.fixtures {
                f.notice(s, &w);
            }
        }
        self.sync_out();
        said
    }

    /// Queue a write to the object of every fixture whose own state moved since
    /// it was last written.
    pub(super) fn sync_out(&mut self) {
        for i in 0..self.fixtures.len() {
            let Some(bound) = self.fixtures[i].bound() else {
                continue;
            };
            let faulted = self.fixtures[i].faulted();
            if faulted == self.synced[i] {
                continue;
            }
            self.synced[i] = faulted;
            let mode = if faulted { bound.fault } else { bound.ok };
            self.outbox.push(Set {
                part: bound.part,
                mode,
            });
        }
    }
}

#[cfg(test)]
mod tests;
