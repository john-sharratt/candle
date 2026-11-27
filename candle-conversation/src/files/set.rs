//! A conversation's working set of the knowledge base's sections
//! (`docs/provenance_system_collection.md` §3, §4;
//! `docs/placed_section_layout.md` §3.5c): each section's momentum, the kind
//! it was scored in and its file, and the order the placed sections keep.
//!
//! Provenance proposes: every scan's fresh score for a section feeds its
//! momentum, `m ← (1 − β)·m + fresh`, and a section whose momentum falls below
//! the floor leaves the set. The fill decides what of the set is placed; the
//! set remembers where each placed section sits, so a reprojection never moves
//! a section it keeps — a newcomer follows the kept ones, and a section let go
//! leaves the others in their order.

use std::collections::{BTreeMap, BTreeSet};

use super::key::{FileKey, SectionKey};
use crate::projection::GroupId;

/// What a conversation's knowledge base holds, when it is not provenance's
/// to choose (§12 D11).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum FilesPin {
    /// Provenance chooses: the fill places the set's strongest sections.
    #[default]
    Free,
    /// Nothing: `files_open` with no file under it — a calibration's empty
    /// frame.
    Empty,
    /// These sections alone, in this order — a calibration's draft holds one,
    /// a probe the whole files it names.
    Only(Vec<SectionKey>),
}

/// Where one section stands in the set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionStanding {
    pub momentum: f32,
    /// The kind it was last scored in.
    pub kind: GroupId,
    /// The file it is a section of.
    pub file: FileKey,
}

/// One section's fresh score from a scan, with its file.
pub type FreshSection = (SectionKey, FileKey, f32);

/// One conversation's sections and their momentum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilesSet {
    pub members: BTreeMap<SectionKey, SectionStanding>,
    /// The sections placed by the last projection, in the order they sit.
    pub placed_order: Vec<SectionKey>,
    /// Files the turn's own reads hold verbatim in the dialogue: none of
    /// their sections placed until the next question
    /// ([`Self::clear_withheld`]), unless already placed, which keeps its
    /// place.
    pub withheld: BTreeSet<FileKey>,
    /// What the knowledge base holds when it is pinned; [`FilesPin::Free`]
    /// for every dialogue.
    pub pin: FilesPin,
}

impl FilesSet {
    /// Fold one scan's fresh scores into the momentum: every member decays by
    /// `beta`, a scored section gains its fresh score, and a section left
    /// under `min_momentum` leaves the set. A pinned set's knowledge base is
    /// not provenance's, and folds nothing.
    pub fn observe(
        &mut self,
        fresh: &[(GroupId, Vec<FreshSection>)],
        beta: f32,
        min_momentum: f32,
    ) {
        if self.pin != FilesPin::Free {
            return;
        }
        for s in self.members.values_mut() {
            s.momentum *= 1.0 - beta;
        }
        for (kind, scores) in fresh {
            for &(section, file, score) in scores {
                let entry = self.members.entry(section).or_insert(SectionStanding {
                    momentum: 0.0,
                    kind: *kind,
                    file,
                });
                entry.momentum += score.max(0.0);
                entry.kind = *kind;
                entry.file = file;
            }
        }
        self.members.retain(|_, s| s.momentum >= min_momentum);
    }

    /// Keep `files` from being placed until [`Self::clear_withheld`]: a file
    /// the turn read stands verbatim in the dialogue, and placed again it
    /// would be carried twice. A section already placed keeps its place.
    pub fn withhold(&mut self, files: impl IntoIterator<Item = FileKey>) {
        self.withheld.extend(files);
    }

    /// Let every withheld file be placed again: a new question starts.
    pub fn clear_withheld(&mut self) {
        self.withheld.clear();
    }

    /// The set's sections of `kind` a projection may place, strongest first
    /// (ties by key): one of a withheld file only while it is placed.
    pub fn ranked(&self, kind: GroupId) -> Vec<(SectionKey, f32)> {
        let mut out: Vec<(SectionKey, f32)> = self
            .members
            .iter()
            .filter(|(_, s)| s.kind == kind)
            .filter(|(k, s)| !self.withheld.contains(&s.file) || self.placed_order.contains(k))
            .map(|(&k, s)| (k, s.momentum))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// The order `chosen` — the sections a projection places, strongest
    /// first — sits in: at the opening (nothing placed before), the order
    /// given; afterwards the sections already placed keep their order and
    /// every newcomer follows them, in the order given.
    pub fn order(&self, chosen: &[SectionKey]) -> Vec<SectionKey> {
        let mut order: Vec<SectionKey> = self
            .placed_order
            .iter()
            .copied()
            .filter(|s| chosen.contains(s))
            .collect();
        for &s in chosen {
            if !order.contains(&s) {
                order.push(s);
            }
        }
        order
    }

    /// Remember `order` as where the placed sections sit.
    pub fn set_placed(&mut self, order: Vec<SectionKey>) {
        self.placed_order = order;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(raw: u32) -> GroupId {
        GroupId::from_raw(raw).unwrap()
    }

    fn standing(momentum: f32, file: u64) -> SectionStanding {
        SectionStanding {
            momentum,
            kind: kind(1),
            file: FileKey(file),
        }
    }

    /// **Momentum decays by β and gains each fresh score; a section under the
    /// floor leaves.**
    #[test]
    fn momentum_decays_gains_and_drops_under_the_floor() {
        let mut set = FilesSet::default();
        let (a, b) = (SectionKey(1), SectionKey(2));
        let f = FileKey(10);
        set.observe(&[(kind(1), vec![(a, f, 500.0), (b, f, 120.0)])], 0.2, 100.0);
        assert_eq!(set.members[&a], standing(500.0, 10));
        assert_eq!(set.members[&b], standing(120.0, 10));
        set.observe(&[(kind(1), vec![(a, f, 100.0)])], 0.2, 100.0);
        assert_eq!(set.members[&a].momentum, 500.0 * 0.8 + 100.0);
        assert!(
            !set.members.contains_key(&b),
            "120 × 0.8 = 96 is under the floor"
        );
        assert_eq!(set.ranked(kind(1)), vec![(a, 500.0)]);
        assert!(set.ranked(kind(2)).is_empty());
    }

    /// **The opening orders sections strongest first; a kept section keeps
    /// its place and a newcomer follows the kept ones.**
    #[test]
    fn a_kept_section_keeps_its_place_and_a_newcomer_follows_the_kept_ones() {
        let (a, b, c, d) = (SectionKey(1), SectionKey(2), SectionKey(3), SectionKey(4));
        let mut set = FilesSet::default();
        let first = set.order(&[b, a, c]);
        assert_eq!(first, vec![b, a, c]);
        set.set_placed(first);
        // c now ranks first, a is let go, d arrives: b and c keep their order.
        assert_eq!(set.order(&[c, d, b]), vec![b, c, d]);
    }

    /// **A pinned set folds no scan.**
    #[test]
    fn a_pinned_set_folds_no_scan() {
        let mut set = FilesSet {
            pin: FilesPin::Only(vec![SectionKey(9)]),
            ..FilesSet::default()
        };
        set.observe(
            &[(kind(1), vec![(SectionKey(1), FileKey(1), 500.0)])],
            0.2,
            100.0,
        );
        assert!(set.members.is_empty());
    }

    /// **A file the turn read waits until the next question — every section
    /// of it — and a placed section keeps its place.**
    #[test]
    fn a_withheld_files_sections_wait_and_a_placed_one_stays() {
        let (a, b, c) = (SectionKey(1), SectionKey(2), SectionKey(3));
        let mut set = FilesSet::default();
        set.observe(
            &[(
                kind(1),
                vec![
                    (a, FileKey(7), 500.0),
                    (b, FileKey(7), 400.0),
                    (c, FileKey(8), 300.0),
                ],
            )],
            0.2,
            100.0,
        );
        set.set_placed(vec![a]);
        set.withhold([FileKey(7)]);
        assert_eq!(set.ranked(kind(1)), vec![(a, 500.0), (c, 300.0)]);
        set.clear_withheld();
        assert_eq!(
            set.ranked(kind(1)),
            vec![(a, 500.0), (b, 400.0), (c, 300.0)]
        );
    }
}
