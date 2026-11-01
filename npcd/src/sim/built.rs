//! What the world's builders made (`docs/zen_moon_mind.md` §14): every change
//! to its map, in the order it was made, and how far this member's copy of the
//! world has laid them.
//!
//! **Only ever appended to.** A change and a later one that undoes it are both
//! kept, so a member that laid the first lays the second too: what a member
//! lays is what it has not laid yet ([`Built::unlaid`]), in order, never the
//! whole list again.

use npc_map::MapEdit;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Built {
    /// Every change, oldest first.
    edits: Vec<MapEdit>,
    /// How many of `edits` this member's world has laid — this machine's own
    /// count, never the world's, so it is not written with it.
    #[serde(skip)]
    laid: usize,
}

/// Two worlds' builds are the same when they made the same changes, however far
/// each member has laid them.
impl PartialEq for Built {
    fn eq(&self, other: &Built) -> bool {
        self.edits == other.edits
    }
}

impl Eq for Built {}

impl Built {
    /// Every change made, oldest first.
    pub fn edits(&self) -> &[MapEdit] {
        &self.edits
    }

    /// A change just laid in this world by its builders.
    pub fn laid_now(&mut self, edit: MapEdit) {
        self.edits.push(edit);
        self.laid = self.edits.len();
    }

    /// The changes this world has not laid yet, oldest first — counted laid
    /// from here on.
    pub fn unlaid(&mut self) -> Vec<MapEdit> {
        let fresh = self.edits[self.laid.min(self.edits.len())..].to_vec();
        self.laid = self.edits.len();
        fresh
    }

    /// Carry how far this world has laid from `before`, the state this one
    /// replaces: as far as the two agree. Where they part — this member built
    /// while cut off, and its authority built otherwise — what it laid of its
    /// own cannot be unlaid, and the authority's builds from there are laid
    /// over it, each passed over where the map already holds it.
    pub fn carry_laid(&mut self, before: &Built) {
        let agreed = before
            .edits
            .iter()
            .zip(&self.edits)
            .take(before.laid)
            .take_while(|(mine, theirs)| mine == theirs)
            .count();
        if agreed < before.laid {
            tracing::warn!(
                laid = before.laid,
                agreed,
                "this member's builds part from the world's; laying the world's from there"
            );
        }
        self.laid = agreed;
    }
}

#[cfg(test)]
mod tests {
    use npc_map::schema::Where;

    use super::*;

    fn describe(text: &str) -> MapEdit {
        MapEdit::Describe {
            at: Where::new("vault-casting", "green-room"),
            character: text.into(),
        }
    }

    /// **What a member lays is what it has not laid yet**: its own builds are
    /// laid as they are made; a state taken in brings only what came since.
    #[test]
    fn a_member_lays_only_what_it_has_not_laid() {
        let mut here = Built::default();
        here.laid_now(describe("Quiet."));
        assert_eq!(here.unlaid(), [], "laid as it was made");

        let made = || Built {
            edits: vec![describe("Quiet."), describe("Loud.")],
            ..Built::default()
        };
        let mut from_authority = made();
        from_authority.carry_laid(&here);
        assert_eq!(from_authority.unlaid(), [describe("Loud.")]);
        assert_eq!(from_authority.unlaid(), []);
        assert_eq!(from_authority, made(), "however far each laid it");

        let mut shorter = Built::default();
        shorter.carry_laid(&from_authority);
        assert_eq!(shorter.unlaid(), []);
    }

    /// **Where a member's builds part from its authority's**, the
    /// authority's are laid from where they part.
    #[test]
    fn where_the_builds_part_the_worlds_are_laid_from_there() {
        let mut cut_off = Built::default();
        cut_off.laid_now(describe("Quiet."));
        cut_off.laid_now(describe("Mine."));
        let mut theirs = Built {
            edits: vec![describe("Quiet."), describe("Theirs."), describe("Loud.")],
            ..Built::default()
        };
        theirs.carry_laid(&cut_off);
        assert_eq!(theirs.unlaid(), [describe("Theirs."), describe("Loud.")]);
    }
}
