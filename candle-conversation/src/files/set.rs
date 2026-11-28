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
//!
//! The model's own index orders it: each section's hit momentum this question
//! ([`Momentum`]) — first folded from the question's first reading, then from
//! every decode step — sets the range's order, strongest beside the question;
//! a section the index has not read is ordered by its scan momentum
//! ([`FilesSet::order_score`], `docs/placed_section_layout.md` §3.2).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::key::{FileKey, SectionKey};
use super::scope::{Bars, Candidate, Momentum, RangeScope, StepChange};
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
    /// Each section's hit momentum this question: the question's first
    /// reading's rates taken whole, then every decode step's folded in. Emptied
    /// by a new question ([`Self::clear_withheld`]).
    pub hits: Momentum,
}

/// `section`'s place in the order: its hit momentum, or — not yet read by the
/// index this question — its scan momentum; `None` for one neither ranks.
fn order_score(
    hits: &Momentum,
    members: &BTreeMap<SectionKey, SectionStanding>,
    section: SectionKey,
) -> Option<f32> {
    hits.score(section)
        .or_else(|| members.get(&section).map(|s| s.momentum))
}

impl FilesSet {
    /// Fold one scan's fresh scores into the momentum: every member decays by
    /// `beta`, a scored section gains its fresh score, and a section left
    /// under `min_momentum` leaves the set. A pinned set folds the scores of
    /// its pinned sections alone: the pin says what is placed, and the scan
    /// still ranks it, so the range opens in relevance order
    /// (`docs/placed_section_layout.md` §3.2). An empty pin folds nothing.
    pub fn observe(
        &mut self,
        fresh: &[(GroupId, Vec<FreshSection>)],
        beta: f32,
        min_momentum: f32,
    ) {
        let pinned: Option<BTreeSet<SectionKey>> = match &self.pin {
            FilesPin::Free => None,
            FilesPin::Empty => return,
            FilesPin::Only(sections) => Some(sections.iter().copied().collect()),
        };
        for s in self.members.values_mut() {
            s.momentum *= 1.0 - beta;
        }
        for (kind, scores) in fresh {
            for &(section, file, score) in scores {
                if pinned.as_ref().is_some_and(|p| !p.contains(&section)) {
                    continue;
                }
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
        self.hits = Momentum::default();
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

    /// Fold one reading's hit `rates` into the hit momentum
    /// ([`Momentum::observe`]): a section's first rate this question is taken
    /// whole.
    pub fn observe_hits(&mut self, rates: &HashMap<SectionKey, f32>) {
        self.hits.observe(rates.iter().map(|(&k, &r)| (k, r)));
    }

    /// `section`'s place in the range's order: its hit momentum this question,
    /// or its scan momentum when the index has not read it; `None` for one
    /// neither ranks.
    pub fn order_score(&self, section: SectionKey) -> Option<f32> {
        order_score(&self.hits, &self.members, section)
    }

    /// A step's entries: every one of `candidates` at or over `bars`' entry
    /// bar by `rate` enters the range at its rank by [`Self::order_score`],
    /// strongest first ([`RangeScope::admit`]); what it changed. One an
    /// eviction sent out this request stays out, and so does one of a file
    /// withheld since the candidates were chosen. One an entry pushed out of a
    /// full range has left as an evicted one has, and stays out until the next
    /// question too, so two sections over the bar never trade places step
    /// after step.
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
        let (hits, members) = (&self.hits, &self.members);
        let change = scope.admit(rate, |k| order_score(hits, members, k), &open, bars);
        self.evicted.extend(change.slid.iter().copied());
        change
    }

    /// Order the range by [`Self::order_score`], strongest at the top beside
    /// the question, a section passing the one above it only by more than
    /// `margin` ([`RangeScope::reorder`]); whether anything moved.
    pub fn reorder(&mut self, margin: f32) -> bool {
        let (hits, members) = (&self.hits, &self.members);
        self.scope
            .as_mut()
            .is_some_and(|scope| scope.reorder(|k| order_score(hits, members, k), margin))
    }

    /// `section`'s scan momentum in the set — what the fill chooses by — or
    /// `None` for a section the set does not hold.
    pub fn momentum_of(&self, section: SectionKey) -> Option<f32> {
        self.members.get(&section).map(|s| s.momentum)
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

    /// **A section a full range pushed out stays out until the next
    /// question**: in a 50-range holding a (30, no momentum yet), b (40) enters
    /// and pushes a out; a, over the bar at the next step, does not come back,
    /// and a new question lets it.
    #[test]
    fn a_section_pushed_out_stays_out_until_the_next_question() {
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

    /// **A pinned set folds the scan for its pinned sections alone, and an
    /// empty pin folds nothing**: pinned to 9, the scan's 9 (500) is ranked and
    /// its 1 (500) is not.
    #[test]
    fn a_pinned_set_folds_the_scan_for_its_own_sections() {
        let scan = [(
            kind(1),
            vec![
                (SectionKey(1), FileKey(1), 500.0),
                (SectionKey(9), FileKey(2), 500.0),
            ],
        )];
        let mut set = FilesSet {
            pin: FilesPin::Only(vec![SectionKey(9)]),
            ..FilesSet::default()
        };
        set.observe(&scan, 0.2, 100.0);
        assert_eq!(set.momentum_of(SectionKey(9)), Some(500.0));
        assert_eq!(set.momentum_of(SectionKey(1)), None);
        let mut empty = FilesSet {
            pin: FilesPin::Empty,
            ..FilesSet::default()
        };
        empty.observe(&scan, 0.2, 100.0);
        assert!(empty.members.is_empty());
    }

    /// **The index orders the range, the scan only what the index has not
    /// read, and a new question starts the index afresh**: a scanned 900
    /// over b's 300, the first reading rates a at 100 and b at 700 — taken
    /// whole — so b sits beside the question; c, unread, keeps its scan 500.
    /// A step's 1000 for a moves it a quarter of the way (325); a new
    /// question forgets every rate.
    #[test]
    fn the_index_orders_the_range_and_the_scan_what_it_has_not_read() {
        let (a, b, c) = (SectionKey(1), SectionKey(2), SectionKey(3));
        let mut set = FilesSet::default();
        set.observe(
            &[(
                kind(1),
                vec![
                    (a, FileKey(7), 900.0),
                    (b, FileKey(8), 300.0),
                    (c, FileKey(9), 500.0),
                ],
            )],
            0.2,
            100.0,
        );
        let mut scope = RangeScope::new(1_000);
        for k in [a, b, c] {
            scope.enter(k, 10, false);
        }
        set.set_scope(scope);
        set.observe_hits(&HashMap::from([(a, 100.0), (b, 700.0)]));
        assert_eq!(set.order_score(a), Some(100.0));
        assert_eq!(set.order_score(b), Some(700.0));
        assert_eq!(set.order_score(c), Some(500.0), "unread: its scan momentum");
        set.reorder(0.0);
        let top_first: Vec<SectionKey> = set
            .scope
            .as_ref()
            .unwrap()
            .sections()
            .map(|e| e.key)
            .collect();
        assert_eq!(top_first, vec![b, c, a], "b beside the question");
        set.observe_hits(&HashMap::from([(a, 1_000.0)]));
        assert_eq!(set.order_score(a), Some(325.0));
        set.clear_withheld();
        assert_eq!(
            set.order_score(a),
            Some(900.0),
            "a new question: the scan's"
        );
        assert_eq!(set.hits, Momentum::default());
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
