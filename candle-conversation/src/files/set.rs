//! A conversation's working set of the knowledge base's sections
//! (`docs/provenance_system_collection.md` §3, §4;
//! `docs/placed_section_layout.md` §3.5c): each section's momentum, the kind
//! it was scored in and its file, and the order the placed sections keep.
//!
//! Provenance proposes: every scan's fresh score for a section feeds its
//! momentum, `m ← (1 − β)·m + fresh`, and a section whose momentum falls below
//! the floor leaves the set. The fill decides what of the set is placed; the
//! set keeps the knowledge base's range as it was placed ([`RangeScope`]), so
//! a reprojection never moves a section it keeps — a newcomer enters in front
//! of the placed ones, and a section that leaves leaves a hole.

use std::collections::{BTreeMap, BTreeSet};

use super::key::{FileKey, SectionKey};
use super::scope::{Bars, Candidate, RangeScope, StepChange};
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
    /// The knowledge base's range as the last projection placed it; `None`
    /// before the opening.
    pub scope: Option<RangeScope>,
    /// Sections whose hit rate evicted them during this request: none comes
    /// back by momentum before the next question.
    pub evicted: BTreeSet<SectionKey>,
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

    /// A new question starts: every withheld file may be placed again, and
    /// the range is compacted — the holes the sections that left behind them
    /// close, between requests, where no decoded token is attending
    /// (`docs/placed_section_layout.md` §3.2).
    pub fn clear_withheld(&mut self) {
        self.withheld.clear();
        self.evicted.clear();
        if let Some(scope) = self.scope.as_mut() {
            scope.compact();
        }
    }

    /// `sections` leave the range, each leaving a hole, and stay out until
    /// the next question; whether any left.
    pub fn evict(&mut self, sections: &[SectionKey]) -> bool {
        let Some(scope) = self.scope.as_mut() else {
            return false;
        };
        let mut any = false;
        for &s in sections {
            if scope.leave(s) {
                self.evicted.insert(s);
                any = true;
            }
        }
        any
    }

    /// A step's entries: every one of `candidates` at or over `bars`' entry
    /// bar by `rate` enters the range in front, strongest first
    /// ([`RangeScope::admit`]); what it changed. One an eviction sent out this
    /// request stays out, and so does one of a file withheld since the
    /// candidates were chosen. One a full range slid out past its top has
    /// left as an evicted one has, and stays out until the next question too,
    /// so two sections over the bar never trade places step after step.
    pub fn admit(
        &mut self,
        rate: impl Fn(SectionKey) -> f32,
        candidates: &[Candidate],
        bars: &Bars,
    ) -> StepChange {
        let Some(scope) = self.scope.as_mut() else {
            return StepChange::default();
        };
        let open: Vec<Candidate> = candidates
            .iter()
            .filter(|c| !self.evicted.contains(&c.key))
            .filter(|c| {
                !self
                    .members
                    .get(&c.key)
                    .is_some_and(|s| self.withheld.contains(&s.file))
            })
            .copied()
            .collect();
        let change = scope.admit(rate, &open, bars);
        self.evicted.extend(change.slid.iter().copied());
        change
    }

    /// Whether `section` is placed.
    pub fn places(&self, section: SectionKey) -> bool {
        self.scope.as_ref().is_some_and(|s| s.holds(section))
    }

    /// The set's sections of `kind` a projection may place, strongest first
    /// (ties by key): one of a withheld file only while it is placed.
    pub fn ranked(&self, kind: GroupId) -> Vec<(SectionKey, f32)> {
        let mut out: Vec<(SectionKey, f32)> = self
            .members
            .iter()
            .filter(|(_, s)| s.kind == kind)
            .filter(|(&k, s)| !self.withheld.contains(&s.file) || self.places(k))
            .map(|(&k, s)| (k, s.momentum))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// Every section of the set that may enter the range during this request,
    /// strongest first (ties by key), whatever its kind: none an eviction
    /// sent out, none of a withheld file. A pinned set's range is its pin's,
    /// and none may enter it.
    pub fn may_enter(&self) -> Vec<(SectionKey, f32)> {
        if self.pin != FilesPin::Free {
            return Vec::new();
        }
        let mut out: Vec<(SectionKey, f32)> = self
            .members
            .iter()
            .filter(|(k, s)| !self.evicted.contains(k) && !self.withheld.contains(&s.file))
            .map(|(&k, s)| (k, s.momentum))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// Remember `scope` as the range a projection placed.
    pub fn set_scope(&mut self, scope: RangeScope) {
        self.scope = Some(scope);
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

    /// **A new question compacts the range**: the hole a section left
    /// closes.
    #[test]
    fn a_new_question_compacts_the_range() {
        let mut scope = RangeScope::new(100);
        for (k, t) in [(1, 30), (2, 20), (3, 10)] {
            scope.enter(SectionKey(k), t, false);
        }
        scope.leave(SectionKey(2));
        let mut set = FilesSet::default();
        set.set_scope(scope);
        assert_eq!(set.scope.as_ref().unwrap().used(), 60);
        set.clear_withheld();
        assert_eq!(set.scope.as_ref().unwrap().used(), 40);
        assert!(set.places(SectionKey(3)) && !set.places(SectionKey(2)));
    }

    /// **An evicted section leaves a hole and stays out until the next
    /// question**, which closes the hole.
    #[test]
    fn an_evicted_section_stays_out_until_the_next_question() {
        let mut scope = RangeScope::new(100);
        scope.enter(SectionKey(1), 30, false);
        scope.enter(SectionKey(2), 20, false);
        let mut set = FilesSet::default();
        set.set_scope(scope);
        assert!(set.evict(&[SectionKey(1)]));
        assert!(!set.evict(&[SectionKey(9)]));
        assert_eq!(set.scope.as_ref().unwrap().used(), 50, "the hole stays");
        assert!(set.evicted.contains(&SectionKey(1)));
        set.clear_withheld();
        assert!(set.evicted.is_empty());
        assert_eq!(set.scope.as_ref().unwrap().used(), 20);
    }

    /// **What may enter during a request is every kind's, strongest first,
    /// with neither an evicted section nor a withheld file's**; a pinned set
    /// admits nothing.
    #[test]
    fn what_may_enter_is_every_kinds_but_the_evicted_and_withheld() {
        let (a, b, c, d) = (SectionKey(1), SectionKey(2), SectionKey(3), SectionKey(4));
        let mut set = FilesSet::default();
        set.observe(
            &[
                (
                    kind(1),
                    vec![(a, FileKey(7), 500.0), (b, FileKey(8), 400.0)],
                ),
                (
                    kind(2),
                    vec![(c, FileKey(9), 450.0), (d, FileKey(9), 300.0)],
                ),
            ],
            0.2,
            100.0,
        );
        set.evicted.insert(b);
        set.withhold([FileKey(9)]);
        assert_eq!(set.may_enter(), vec![(a, 500.0)]);
        set.clear_withheld();
        assert_eq!(
            set.may_enter(),
            vec![(a, 500.0), (c, 450.0), (b, 400.0), (d, 300.0)]
        );
        set.pin = FilesPin::Empty;
        assert!(set.may_enter().is_empty());
    }

    /// **A step admits the candidates over the entry bar in front, but not one
    /// an eviction sent out this request, nor one of a withheld file.**
    #[test]
    fn a_step_admits_over_the_bar_but_not_the_evicted() {
        let mut scope = RangeScope::new(100);
        scope.enter(SectionKey(1), 30, false);
        let mut set = FilesSet::default();
        set.set_scope(scope);
        set.members.insert(SectionKey(4), standing(500.0, 9));
        set.evicted.insert(SectionKey(3));
        set.withhold([FileKey(9)]);
        let candidates = [
            Candidate {
                key: SectionKey(2),
                tokens: 20,
            },
            Candidate {
                key: SectionKey(3),
                tokens: 20,
            },
            Candidate {
                key: SectionKey(4),
                tokens: 20,
            },
        ];
        let change = set.admit(|_| 900.0, &candidates, &Bars::KNOWLEDGE_BASE);
        assert_eq!(change.entered, vec![SectionKey(2)]);
        assert!(set.places(SectionKey(2)) && !set.places(SectionKey(3)));
        assert!(!set.places(SectionKey(4)));
    }

    /// **A section a full range slid out stays out until the next question**:
    /// in a 50-range holding a (30), b (40) enters and slides a out; a, over
    /// the bar at the next step, does not come back, and a new question lets
    /// it.
    #[test]
    fn a_section_slid_out_stays_out_until_the_next_question() {
        let (a, b) = (SectionKey(1), SectionKey(2));
        let mut scope = RangeScope::new(50);
        scope.enter(a, 30, false);
        let mut set = FilesSet::default();
        set.set_scope(scope);
        let c = |key, tokens| Candidate { key, tokens };
        let change = set.admit(|_| 900.0, &[c(b, 40)], &Bars::KNOWLEDGE_BASE);
        assert_eq!(change.slid, vec![a]);
        let again = set.admit(|_| 900.0, &[c(a, 30)], &Bars::KNOWLEDGE_BASE);
        assert!(again.entered.is_empty(), "a stays out this request");
        set.clear_withheld();
        assert!(!set.evicted.contains(&a));
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
        let mut scope = RangeScope::new(1_000);
        scope.enter(a, 10, false);
        set.set_scope(scope);
        set.withhold([FileKey(7)]);
        assert_eq!(set.ranked(kind(1)), vec![(a, 500.0), (c, 300.0)]);
        set.clear_withheld();
        assert_eq!(
            set.ranked(kind(1)),
            vec![(a, 500.0), (b, 400.0), (c, 300.0)]
        );
    }
}
