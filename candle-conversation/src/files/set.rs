//! A conversation's working set of the knowledge base's files
//! (`docs/provenance_system_collection.md` §3, §4): each file's momentum, the
//! kind it was scored in, and the order the placed files keep.
//!
//! Provenance proposes: every scan's fresh score for a file feeds its
//! momentum, `m ← (1 − β)·m + fresh`, and a file whose momentum falls below
//! the floor leaves the set. The fill decides what of the set is placed; the
//! set remembers where each placed file sits, so a reprojection never moves a
//! file it keeps — a newcomer follows the kept ones, and a file let go leaves
//! the others in their order.

use std::collections::BTreeMap;

use super::key::FileKey;
use crate::projection::GroupId;

/// One conversation's files and their momentum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilesSet {
    /// Each file's momentum and the kind it was last scored in.
    pub members: BTreeMap<FileKey, (f32, GroupId)>,
    /// The files placed by the last projection, in the order they sit.
    pub placed_order: Vec<FileKey>,
}

impl FilesSet {
    /// Fold one scan's fresh scores into the momentum: every member decays by
    /// `beta`, a scored file gains its fresh score, and a file left under
    /// `min_momentum` leaves the set.
    pub fn observe(
        &mut self,
        fresh: &[(GroupId, Vec<(FileKey, f32)>)],
        beta: f32,
        min_momentum: f32,
    ) {
        for (m, _) in self.members.values_mut() {
            *m *= 1.0 - beta;
        }
        for (kind, scores) in fresh {
            for &(file, score) in scores {
                let entry = self.members.entry(file).or_insert((0.0, *kind));
                entry.0 += score.max(0.0);
                entry.1 = *kind;
            }
        }
        self.members.retain(|_, (m, _)| *m >= min_momentum);
    }

    /// The set's files of `kind`, strongest first (ties by key).
    pub fn ranked(&self, kind: GroupId) -> Vec<(FileKey, f32)> {
        let mut out: Vec<(FileKey, f32)> = self
            .members
            .iter()
            .filter(|(_, (_, k))| *k == kind)
            .map(|(&f, &(m, _))| (f, m))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// The order `chosen` — the files a projection places, strongest first —
    /// sits in: at the opening (nothing placed before), the order given;
    /// afterwards the files already placed keep their order and every
    /// newcomer follows them, in the order given.
    pub fn order(&self, chosen: &[FileKey]) -> Vec<FileKey> {
        let mut order: Vec<FileKey> = self
            .placed_order
            .iter()
            .copied()
            .filter(|f| chosen.contains(f))
            .collect();
        for &f in chosen {
            if !order.contains(&f) {
                order.push(f);
            }
        }
        order
    }

    /// Remember `order` as where the placed files sit.
    pub fn set_placed(&mut self, order: Vec<FileKey>) {
        self.placed_order = order;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(raw: u32) -> GroupId {
        GroupId::from_raw(raw).unwrap()
    }

    /// **Momentum decays by β and gains each fresh score; a file under the
    /// floor leaves.**
    #[test]
    fn momentum_decays_gains_and_drops_under_the_floor() {
        let mut set = FilesSet::default();
        let (a, b) = (FileKey(1), FileKey(2));
        set.observe(&[(kind(1), vec![(a, 500.0), (b, 120.0)])], 0.2, 100.0);
        assert_eq!(set.members[&a], (500.0, kind(1)));
        assert_eq!(set.members[&b], (120.0, kind(1)));
        set.observe(&[(kind(1), vec![(a, 100.0)])], 0.2, 100.0);
        assert_eq!(set.members[&a].0, 500.0 * 0.8 + 100.0);
        assert!(
            !set.members.contains_key(&b),
            "120 × 0.8 = 96 is under the floor"
        );
        assert_eq!(set.ranked(kind(1)), vec![(a, 500.0)]);
        assert!(set.ranked(kind(2)).is_empty());
    }

    /// **The opening orders files strongest first; a kept file keeps its place
    /// and a newcomer follows the kept ones.**
    #[test]
    fn a_kept_file_keeps_its_place_and_a_newcomer_follows_the_kept_ones() {
        let (a, b, c, d) = (FileKey(1), FileKey(2), FileKey(3), FileKey(4));
        let mut set = FilesSet::default();
        let first = set.order(&[b, a, c]);
        assert_eq!(first, vec![b, a, c]);
        set.set_placed(first);
        // c now ranks first, a is let go, d arrives: b and c keep their order.
        assert_eq!(set.order(&[c, d, b]), vec![b, c, d]);
    }
}
