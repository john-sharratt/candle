//! One dialogue's working set: the conversations it carries ahead of its own
//! turns, in the order they entered (`docs/zend_working_set.md` §4.3).

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use super::type_shares::TypeCap;
use crate::projection::{HitMomentum, HitsConfig, TimelineId, WorkingSetShare};

/// How much stronger a newcomer must be than the provenance it dislodges.
/// Without the margin two files near the floor trade places every
/// reprojection, and each trade rebuilds everything to the right of it.
const DISLODGE_MARGIN: f32 = 1.5;

/// The size limits an admission is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Tokens the members — seeds and provenance — may hold together.
    pub budget_tokens: usize,
    /// Tokens the folder members may hold between them; the files take what
    /// is left of `budget_tokens`.
    pub folder_tokens: usize,
    /// Tokens the guidance members may hold between them — a pool of their
    /// own, outside `budget_tokens`.
    pub guidance_tokens: usize,
    /// Tokens of one member a slot places. Where members are emitted whole, a
    /// longer one is never seeded or admitted; where the fill carves the slot,
    /// a longer reading is admitted at this cost ([`Candidate::carved_to`]).
    pub max_file_tokens: usize,
}

/// What admitting a conversation needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Its recorded token total; `None` when none was recorded.
    pub tokens: Option<usize>,
    /// The share of the budget it draws on; `None` when it belongs to no
    /// working-set group.
    pub share: Option<WorkingSetShare>,
    /// The type of document it is and that type's cap
    /// ([`super::TypeShares::classify`]); `None` for one of no declared type,
    /// which is unbounded.
    pub kind: Option<TypeCap>,
}

impl Candidate {
    /// What this candidate costs a slot the fill carves
    /// (`docs/hit_driven_projection.md` §7.2, §8.2): the fill places at most
    /// `max_file_tokens` of one reading — its first and last turns and its
    /// best-scored pages between — so a longer reading is charged that much
    /// and competes for the budget at the size it can take up.
    pub fn carved_to(self, max_file_tokens: usize) -> Self {
        Self {
            tokens: self.tokens.map(|t| t.min(max_file_tokens)),
            ..self
        }
    }
}

/// One member of the sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Member {
    timeline: TimelineId,
    tokens: usize,
    share: WorkingSetShare,
    kind: Option<TypeCap>,
}

/// What a dialogue carries ahead of its own turns: one sequence, in the order
/// its members entered.
///
/// **Position is insertion order, never age or rank.** Every member enters
/// after the ones already there, keeps its place until it leaves, and when it
/// leaves the gap closes at once.
///
/// That is what keeps the projection stable under RoPE. A change at one
/// position leaves everything to its right at the same distance from the
/// question and everything to its left at the same distance from the system
/// prompt. Every change here is an insertion at the end or a removal, so
/// long-lived members on the left keep their place beside the system prompt,
/// and the churn — newcomers arriving, the weakest leaving — lands nearer the
/// dialogue.
///
/// Members decay by momentum and leave below the floor. When a newcomer needs
/// room, the weakest member is dislodged only when the newcomer is
/// [`DISLODGE_MARGIN`] stronger. The **seeds** are the first members a
/// conversation admits.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkingSet {
    members: Vec<Member>,
    /// Momentum per conversation — members and candidates alike.
    momentum: HashMap<TimelineId, f32>,
    /// Conversations that may not enter for now: the readings of files the
    /// current turn's loop read for real, already verbatim in its dialogue
    /// ([`Self::withhold`]). They keep their momentum, and a member among them
    /// keeps its place — evicting it would rebuild the slot past it for one
    /// file's tokens.
    withheld: HashSet<TimelineId>,
    /// Each conversation's fresh score in the last scan folded in
    /// ([`Self::observe`]) — whether the reasoning was on it then, apart from
    /// the momentum it has carried since.
    fresh: HashMap<TimelineId, f32>,
    /// Hit momentum per conversation: how much of it the model's own indexer
    /// keeps reading (`docs/hit_driven_projection.md` §7.1), folded in by
    /// [`Self::observe_hits`]. It lifts a conversation's strength wherever
    /// momentum is judged, by [`Self::score_of`].
    hits: HitMomentum<TimelineId>,
    /// The `hits:` block the momentum was last folded under — its gain is how
    /// far the momentum lifts.
    hits_cfg: HitsConfig,
}

impl WorkingSet {
    /// Keep `timelines` from entering until [`Self::clear_withheld`]. A member
    /// among them stays where it is.
    pub fn withhold(&mut self, timelines: impl IntoIterator<Item = TimelineId>) {
        self.withheld.extend(timelines);
    }

    /// Let every withheld conversation enter again: the turn's loop has ended.
    pub fn clear_withheld(&mut self) {
        self.withheld.clear();
    }
    /// Every member in emission order.
    pub fn members(&self) -> Vec<TimelineId> {
        self.members.iter().map(|m| m.timeline).collect()
    }

    /// Whether `timeline` is a member.
    pub fn is_member(&self, timeline: TimelineId) -> bool {
        self.position(timeline).is_some()
    }

    /// The members with their momentum, in emission order.
    pub fn provenance(&self) -> Vec<(TimelineId, f32)> {
        self.members
            .iter()
            .map(|m| (m.timeline, self.momentum_of(m.timeline).unwrap_or(0.0)))
            .collect()
    }

    /// `timeline`'s momentum, when it has any.
    pub fn momentum_of(&self, timeline: TimelineId) -> Option<f32> {
        self.momentum.get(&timeline).copied()
    }

    /// `timeline`'s hit momentum — `0` for one the indexer has not read.
    pub fn hits_of(&self, timeline: TimelineId) -> f32 {
        self.hits.get(timeline)
    }

    /// What `timeline`'s momentum is lifted by: `1 + gain · h`.
    fn lift(&self, timeline: TimelineId) -> f32 {
        self.hits.lift(timeline, &self.hits_cfg)
    }

    /// `timeline`'s strength: its momentum lifted by its hit momentum,
    /// `momentum · (1 + gain · h)` (`docs/hit_driven_projection.md` §7.1) —
    /// what every admission, eviction and retention here judges by. `None`
    /// for one with no momentum.
    pub fn score_of(&self, timeline: TimelineId) -> Option<f32> {
        self.momentum_of(timeline).map(|m| m * self.lift(timeline))
    }

    /// Fold one stride's hit densities in ([`HitMomentum::fold`]) under
    /// `cfg`, whose gain then lifts every judgment.
    pub fn observe_hits(&mut self, density: &HashMap<TimelineId, f32>, cfg: &HitsConfig) {
        self.hits.fold(density.iter().map(|(&t, &d)| (t, d)), cfg);
        self.hits_cfg = cfg.clone();
    }

    /// `timeline`'s fresh score in the last scan folded in: `0` for one that
    /// scan did not score.
    pub fn fresh_of(&self, timeline: TimelineId) -> f32 {
        self.fresh.get(&timeline).copied().unwrap_or(0.0)
    }

    /// Tokens every member together holds.
    pub fn tokens(&self) -> usize {
        self.members.iter().map(|m| m.tokens).sum()
    }

    fn position(&self, timeline: TimelineId) -> Option<usize> {
        self.members.iter().position(|m| m.timeline == timeline)
    }

    /// Seed provenance with `candidates`, in list order: each gains `momentum`
    /// (or keeps what it has, if that is higher) and enters while the budget
    /// has room — seeds dislodge nothing. Returns the candidates that did not
    /// enter, in list order; one that missed only for room stays a candidate
    /// and can enter later on its momentum.
    pub fn seed(
        &mut self,
        candidates: &[(TimelineId, Candidate)],
        limits: Limits,
        momentum: f32,
    ) -> Vec<TimelineId> {
        let mut refused = Vec::new();
        for &(timeline, candidate) in candidates {
            if self.is_member(timeline) {
                continue;
            }
            let (Some(tokens), Some(share)) = (candidate.tokens, candidate.share) else {
                refused.push(timeline);
                continue;
            };
            if tokens > limits.max_file_tokens {
                refused.push(timeline);
                continue;
            }
            let entry = self.momentum.entry(timeline).or_insert(0.0);
            *entry = entry.max(momentum);
            if self
                .room_for(tokens, share, candidate.kind, limits, &|_| false)
                .is_none()
            {
                refused.push(timeline);
                continue;
            }
            self.admit(timeline, tokens, share, candidate.kind);
        }
        refused
    }

    /// Fold one reprojection in.
    ///
    /// 1. Momentum: `m ← (1 − β)·m + fresh` for every conversation, where one
    ///    the scan did not score counts as a fresh zero.
    /// 2. Below `min_momentum` a conversation leaves the map, and a member
    ///    leaves the sequence — the gap closes. So does a member `candidate`
    ///    no longer allows (out of scope, gone). Then, while the members are
    ///    over budget, the weakest of them leaves.
    /// 3. The strongest candidates that are not members try to enter, each
    ///    dislodging only provenance it beats by [`DISLODGE_MARGIN`] — at most
    ///    `max_admits` of them. One reprojection that admits a dozen files
    ///    rebuilds the slot past every one of them and lifts each into VRAM;
    ///    the ones turned away keep their momentum and enter on the next
    ///    reprojections, so the set takes in new content without churning.
    pub fn observe(
        &mut self,
        fresh: &HashMap<TimelineId, f32>,
        beta: f32,
        min_momentum: f32,
        limits: Limits,
        max_admits: NonZeroUsize,
        candidate: &dyn Fn(TimelineId) -> Option<Candidate>,
    ) {
        let max_admits = max_admits.get();
        let keep = 1.0 - beta;
        for m in self.momentum.values_mut() {
            *m *= keep;
        }
        for (&timeline, &score) in fresh {
            *self.momentum.entry(timeline).or_insert(0.0) += score;
        }
        self.fresh.clone_from(fresh);
        // Retention judges strength, so a file the indexer keeps reading
        // outlasts its fading provenance score.
        let (hits, hits_cfg) = (&self.hits, &self.hits_cfg);
        self.momentum
            .retain(|timeline, m| *m * hits.lift(*timeline, hits_cfg) >= min_momentum);

        let leaving: Vec<usize> = self
            .members
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                !self.momentum.contains_key(&m.timeline) || candidate(m.timeline).is_none()
            })
            .map(|(i, _)| i)
            .collect();
        self.evict(&leaving);
        for share in [
            WorkingSetShare::Remainder,
            WorkingSetShare::Folders,
            WorkingSetShare::Guidance,
        ] {
            if let Some(over) = self.room_for(0, share, None, limits, &|_| true) {
                self.evict(&over);
            }
        }
        self.trim_types(limits);

        let mut newcomers: Vec<(TimelineId, f32)> = self
            .momentum
            .iter()
            .filter(|(timeline, _)| {
                !self.is_member(**timeline) && !self.withheld.contains(timeline)
            })
            .map(|(timeline, m)| (*timeline, *m * self.lift(*timeline)))
            .collect();
        newcomers.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
        let mut admitted = 0usize;
        for (timeline, strength) in newcomers {
            if admitted == max_admits {
                break;
            }
            let Some(Candidate {
                tokens: Some(tokens),
                share: Some(share),
                kind,
            }) = candidate(timeline)
            else {
                continue;
            };
            if tokens > limits.max_file_tokens {
                continue;
            }
            let beaten = |weaker: f32| weaker * DISLODGE_MARGIN < strength;
            if let Some(victims) = self.room_for(tokens, share, kind, limits, &beaten) {
                self.evict(&victims);
                self.admit(timeline, tokens, share, kind);
                admitted += 1;
            }
        }
    }

    /// Take `timeline` out of the set — its conversation is gone.
    pub fn remove(&mut self, timeline: TimelineId) {
        if let Some(at) = self.position(timeline) {
            self.evict(&[at]);
        }
        self.momentum.remove(&timeline);
    }

    /// Enter `timeline` after every member already there.
    fn admit(
        &mut self,
        timeline: TimelineId,
        tokens: usize,
        share: WorkingSetShare,
        kind: Option<TypeCap>,
    ) {
        self.members.push(Member {
            timeline,
            tokens,
            share,
            kind,
        });
    }

    /// Tokens the budget's members of type `id` hold. Guidance is a pool
    /// apart and holds no type's share.
    fn type_tokens(&self, id: u8) -> usize {
        self.members
            .iter()
            .filter(|m| m.share != WorkingSetShare::Guidance && m.kind.is_some_and(|k| k.id == id))
            .map(|m| m.tokens)
            .sum()
    }

    /// Bring every type back under its share: while one holds more than its
    /// cap, its weakest member leaves.
    fn trim_types(&mut self, limits: Limits) {
        let mut kinds: Vec<TypeCap> = self
            .members
            .iter()
            .filter(|m| m.share != WorkingSetShare::Guidance)
            .filter_map(|m| m.kind)
            .collect();
        kinds.sort_by_key(|k| k.id);
        kinds.dedup_by_key(|k| k.id);
        for kind in kinds {
            let cap = kind.tokens(limits.budget_tokens);
            while self.type_tokens(kind.id) > cap {
                let weakest = self
                    .members
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| {
                        m.share != WorkingSetShare::Guidance
                            && m.kind.is_some_and(|k| k.id == kind.id)
                    })
                    .map(|(i, m)| (i, self.score_of(m.timeline).unwrap_or(0.0)))
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
                match weakest {
                    Some((i, _)) => self.evict(&[i]),
                    None => break,
                }
            }
        }
    }

    /// The members that must leave for `tokens` more tokens of `share`, of
    /// type `kind`, to fit, weakest first among those `may_dislodge` allows
    /// (judged on their strength, [`Self::score_of`]) — or `None` when no such
    /// set makes it fit. On a momentum tie the newest goes first: it moves the
    /// fewest members on its left. Guidance is a pool apart: it is bounded by `guidance_tokens`
    /// alone and dislodges only guidance, and nothing else dislodges it. A
    /// typed newcomer its type has no room for dislodges only its own type,
    /// as a folder past `folder_tokens` dislodges only folders — so a burst of
    /// strong documents cannot take the room the code would have.
    fn room_for(
        &self,
        tokens: usize,
        share: WorkingSetShare,
        kind: Option<TypeCap>,
        limits: Limits,
        may_dislodge: &dyn Fn(f32) -> bool,
    ) -> Option<Vec<usize>> {
        let guidance = share == WorkingSetShare::Guidance;
        let kind = kind.filter(|_| !guidance);
        let of_kind = |m: &Member| kind.is_some_and(|k| m.kind.is_some_and(|mk| mk.id == k.id));
        let mut typed: usize = kind.map_or(0, |k| self.type_tokens(k.id));
        let type_bound =
            |typed: usize| kind.is_some_and(|k| typed + tokens > k.tokens(limits.budget_tokens));
        let in_pool = |m: &Member| (m.share == WorkingSetShare::Guidance) == guidance;
        let mut total: usize = self
            .members
            .iter()
            .filter(|m| in_pool(m))
            .map(|m| m.tokens)
            .sum();
        let ceiling = match guidance {
            true => limits.guidance_tokens,
            false => limits.budget_tokens,
        };
        let mut folders: usize = self
            .members
            .iter()
            .filter(|m| m.share == WorkingSetShare::Folders)
            .map(|m| m.tokens)
            .sum();
        let folder_bound = |folders: usize| {
            share == WorkingSetShare::Folders && folders + tokens > limits.folder_tokens
        };
        let mut pool: Vec<(usize, f32)> = self
            .members
            .iter()
            .enumerate()
            .filter(|(_, m)| in_pool(m))
            .map(|(i, m)| (i, self.score_of(m.timeline).unwrap_or(0.0)))
            .filter(|&(_, m)| may_dislodge(m))
            .collect();
        pool.sort_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        let mut victims = Vec::new();
        loop {
            let over_total = total + tokens > ceiling;
            let over_folders = folder_bound(folders);
            let over_type = type_bound(typed);
            if !over_total && !over_folders && !over_type {
                return Some(victims);
            }
            let next = pool.iter().position(|&(i, _)| {
                let m = &self.members[i];
                (!over_folders || m.share == WorkingSetShare::Folders) && (!over_type || of_kind(m))
            })?;
            let (i, _) = pool.remove(next);
            let m = &self.members[i];
            total -= m.tokens;
            if m.share == WorkingSetShare::Folders {
                folders -= m.tokens;
            }
            if of_kind(m) {
                typed -= m.tokens;
            }
            victims.push(i);
        }
    }

    /// Remove the members at `indices`, closing each gap.
    fn evict(&mut self, indices: &[usize]) {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted {
            self.members.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILES: WorkingSetShare = WorkingSetShare::Remainder;
    const FOLDERS: WorkingSetShare = WorkingSetShare::Folders;
    const GUIDES: WorkingSetShare = WorkingSetShare::Guidance;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn limits(budget_tokens: usize) -> Limits {
        Limits {
            budget_tokens,
            folder_tokens: budget_tokens,
            guidance_tokens: budget_tokens,
            max_file_tokens: 1_000,
        }
    }

    /// Documents may hold a fifth of the budget.
    const DOCS: TypeCap = TypeCap {
        id: 0,
        per_mille: 200,
    };

    /// A 100-token file, except 9 (no total) and 8 (2,000 tokens); 50–59 are
    /// 100-token folders, 60–69 100-token guides, 70–79 100-token documents.
    fn info(timeline: TimelineId) -> Option<Candidate> {
        let raw = timeline.raw();
        Some(Candidate {
            tokens: match raw {
                9 => None,
                8 => Some(2_000),
                _ => Some(100),
            },
            share: Some(match raw {
                50..60 => FOLDERS,
                60..70 => GUIDES,
                _ => FILES,
            }),
            kind: (70..80).contains(&raw).then_some(DOCS),
        })
    }

    fn cand(raw: u64) -> Candidate {
        info(tl(raw)).unwrap()
    }

    fn order(ws: &WorkingSet) -> Vec<u64> {
        ws.members().iter().map(|t| t.raw()).collect()
    }

    fn observe(ws: &mut WorkingSet, fresh: &[(u64, f32)], budget: usize) {
        let fresh: HashMap<TimelineId, f32> = fresh.iter().map(|&(r, s)| (tl(r), s)).collect();
        ws.observe(&fresh, 0.2, 100.0, limits(budget), NonZeroUsize::MAX, &info);
    }

    fn read(raws: &[u64]) -> HashMap<TimelineId, f32> {
        raws.iter().map(|&r| (tl(r), 1.0)).collect()
    }

    /// **A carved reading costs what the slot can place of it**: the 2,000-token
    /// file is refused whole and admitted carved, at the 1,000-token cap, beside
    /// a 100-token file in a budget of 1,100; a smaller one keeps its own size.
    #[test]
    fn a_carved_reading_is_charged_the_cap() {
        let mut whole = WorkingSet::default();
        let refused = whole.seed(&[(tl(8), cand(8))], limits(1_100), 1000.0);
        assert_eq!(refused, vec![tl(8)], "2,000 whole is past the cap");

        let carved = |raw| cand(raw).carved_to(1_000);
        assert_eq!(carved(8).tokens, Some(1_000));
        assert_eq!(carved(1).tokens, Some(100));
        assert_eq!(
            cand(9).carved_to(1_000).tokens,
            None,
            "no total stays unknown"
        );
        let mut ws = WorkingSet::default();
        let refused = ws.seed(
            &[(tl(8), carved(8)), (tl(1), carved(1))],
            limits(1_100),
            1000.0,
        );
        assert!(refused.is_empty(), "{refused:?}");
        assert_eq!(order(&ws), vec![8, 1]);
        assert_eq!(ws.tokens(), 1_100);
    }

    /// **A file the indexer keeps reading outlasts its fading score**: two
    /// seeds at one momentum, only one read each stride. Two reprojections
    /// later the unread one's momentum is under the floor and it leaves; the
    /// read one's strength is its momentum lifted by `1 + h`.
    #[test]
    fn a_read_file_outlasts_an_unread_one() {
        let mut ws = WorkingSet::default();
        ws.seed(&[(tl(1), cand(1)), (tl(2), cand(2))], limits(1000), 150.0);
        let cfg = HitsConfig::default();
        ws.observe_hits(&read(&[1]), &cfg);
        observe(&mut ws, &[], 1000);
        assert_eq!(order(&ws), vec![1, 2], "120 each, both above the floor");
        ws.observe_hits(&read(&[1]), &cfg);
        observe(&mut ws, &[], 1000);
        assert_eq!(order(&ws), vec![1], "96 unread leaves; 96 · 2.8 stays");
        assert!((ws.hits_of(tl(1)) - 1.8).abs() < 1e-6);
        assert!((ws.score_of(tl(1)).unwrap() - 96.0 * 2.8).abs() < 1e-3);
        assert_eq!(ws.hits_of(tl(2)), 0.0);
    }

    /// **The weakest by strength is the one dislodged**: two members at one
    /// momentum, the newcomer strong enough to dislodge either; the one the
    /// indexer reads stays, whichever it is.
    #[test]
    fn hits_decide_which_member_a_newcomer_dislodges() {
        for (kept_by_hits, dislodged) in [(1u64, 2u64), (2, 1)] {
            let mut ws = WorkingSet::default();
            ws.seed(&[(tl(1), cand(1)), (tl(2), cand(2))], limits(200), 1000.0);
            ws.observe_hits(&read(&[kept_by_hits]), &HitsConfig::default());
            observe(&mut ws, &[(3, 2500.0)], 200);
            let members = order(&ws);
            assert!(members.contains(&kept_by_hits), "{members:?}");
            assert!(members.contains(&3), "{members:?}");
            assert!(!members.contains(&dislodged), "{members:?}");
        }
    }

    /// Hit momentum leaks like momentum and leaves the map once it lifts
    /// nothing; with no hits folded in, strength is momentum.
    #[test]
    fn hit_momentum_leaks_and_leaves() {
        let mut ws = WorkingSet::default();
        ws.seed(&[(tl(1), cand(1))], limits(1000), 1000.0);
        assert_eq!(ws.score_of(tl(1)), ws.momentum_of(tl(1)));
        let cfg = HitsConfig::default();
        ws.observe_hits(&read(&[1]), &cfg);
        for _ in 0..60 {
            ws.observe_hits(&HashMap::new(), &cfg);
        }
        assert_eq!(ws.hits_of(tl(1)), 0.0, "0.8^60 is under the floor");
    }

    /// **A fresh score is the last scan's alone**, apart from momentum: a
    /// member the next scan does not score reads `0` though its momentum
    /// keeps it placed.
    #[test]
    fn a_fresh_score_is_the_last_scans_alone() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 300.0)], 1_000);
        assert_eq!((ws.fresh_of(tl(1)), ws.fresh_of(tl(2))), (900.0, 300.0));
        observe(&mut ws, &[(2, 150.0)], 1_000);
        assert_eq!((ws.fresh_of(tl(1)), ws.fresh_of(tl(2))), (0.0, 150.0));
        assert_eq!(ws.momentum_of(tl(1)), Some(720.0));
        assert!(ws.is_member(tl(1)));
    }

    /// **A withheld conversation does not enter, and a withheld member stays**:
    /// the turn's loop read that file for real, so placing it again would be a
    /// duplicate, and evicting it would rebuild the slot. Cleared, it enters on
    /// the momentum it kept.
    #[test]
    fn a_withheld_conversation_waits_and_a_withheld_member_stays() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0)], 1_000);
        ws.withhold([tl(1), tl(2)]);
        observe(&mut ws, &[(2, 900.0)], 1_000);
        assert_eq!(
            order(&ws),
            vec![1],
            "a member keeps its place; a newcomer waits"
        );
        ws.clear_withheld();
        observe(&mut ws, &[], 1_000);
        assert_eq!(order(&ws), vec![1, 2], "it enters on the momentum it kept");
    }

    /// **A newcomer enters after the members already there**, whatever its
    /// strength: the oldest members keep their places.
    #[test]
    fn a_newcomer_enters_after_the_members_already_there() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 300.0)], 1_000);
        observe(&mut ws, &[(2, 5_000.0)], 1_000);
        assert_eq!(order(&ws), vec![1, 2]);
    }

    /// A conversation with no total, past the cap, or in no group never enters.
    #[test]
    fn a_conversation_with_no_total_past_the_cap_or_in_no_group_never_enters() {
        let nowhere = Candidate {
            tokens: Some(100),
            share: None,
            kind: None,
        };
        let mut ws = WorkingSet::default();
        let refused = ws.seed(
            &[(tl(9), cand(9)), (tl(8), cand(8)), (tl(7), nowhere)],
            limits(10_000),
            1_000.0,
        );
        assert_eq!(refused, vec![tl(9), tl(8), tl(7)]);
        assert!(ws.members().is_empty());
    }

    /// **New provenance dislodges only what it beats by the margin.** At 400
    /// against a weakest of 300 it waits; at 500 it takes the place.
    #[test]
    fn new_provenance_dislodges_only_provenance_it_clearly_beats() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 375.0)], 200);
        // 2 decays to 300 and 3 arrives at 400: 300 × 1.5 = 450 > 400.
        observe(&mut ws, &[(1, 180.0), (3, 400.0)], 200);
        assert_eq!(order(&ws), vec![1, 2]);
        // 2 → 240, 3 → 320 + 300 = 620: 240 × 1.5 = 360 < 620.
        observe(&mut ws, &[(1, 180.0), (3, 300.0)], 200);
        assert_eq!(order(&ws), vec![1, 3]);
    }

    /// **A member that fades leaves, and the gap closes** — the members on
    /// either side keep their order.
    #[test]
    fn a_faded_member_leaves_and_the_gap_closes() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 5_000.0), (2, 130.0), (3, 5_000.0)], 1_000);
        observe(&mut ws, &[(4, 5_000.0)], 1_000);
        assert_eq!(order(&ws), vec![1, 3, 2, 4], "2 is at 104");
        observe(&mut ws, &[], 1_000);
        assert_eq!(order(&ws), vec![1, 3, 4], "2 fell to 83, under 100");
    }

    /// **Seeds enter in list order at the seeding momentum, while there is
    /// room, dislodging nothing**; one that misses for room stays a candidate.
    #[test]
    fn seeds_enter_in_list_order_while_they_fit() {
        let mut ws = WorkingSet::default();
        let seeds: Vec<(TimelineId, Candidate)> =
            [1, 8, 2, 9, 3].iter().map(|&r| (tl(r), cand(r))).collect();
        let refused = ws.seed(&seeds, limits(200), 1_000.0);
        assert_eq!(order(&ws), vec![1, 2]);
        assert_eq!(refused, vec![tl(8), tl(9), tl(3)]);
        assert_eq!(
            ws.momentum_of(tl(3)),
            Some(1_000.0),
            "3 waits as a candidate"
        );
    }

    /// An unattended seed fades like any other file: 1,000 × 0.8¹⁰ ≈ 107 is
    /// still in after ten reprojections, and the eleventh takes it out.
    #[test]
    fn an_unattended_seed_fades_out() {
        let mut ws = WorkingSet::default();
        ws.seed(&[(tl(1), cand(1))], limits(1_000), 1_000.0);
        for _ in 0..10 {
            observe(&mut ws, &[], 1_000);
        }
        assert_eq!(order(&ws), vec![1]);
        observe(&mut ws, &[], 1_000);
        assert!(ws.members().is_empty());
    }

    /// **Folders keep to their share**: a folder newcomer dislodges a folder,
    /// never a file, even when a file is weaker.
    #[test]
    fn a_folder_makes_room_among_the_folders() {
        let mut ws = WorkingSet::default();
        let lim = Limits {
            budget_tokens: 300,
            folder_tokens: 100,
            guidance_tokens: 0,
            max_file_tokens: 1_000,
        };
        let fresh: HashMap<TimelineId, f32> = HashMap::from([(tl(50), 900.0), (tl(1), 200.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(order(&ws), vec![50, 1]);
        // 50 decays to 720, and 720 × 1.5 < 5,000.
        let fresh: HashMap<TimelineId, f32> = HashMap::from([(tl(51), 5_000.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(
            order(&ws),
            vec![1, 51],
            "50 left for 51; the weaker file 1 stayed"
        );
    }

    /// Room for one 100-token guide beside two 100-token files.
    fn guide_limits() -> Limits {
        Limits {
            budget_tokens: 200,
            folder_tokens: 0,
            guidance_tokens: 100,
            max_file_tokens: 1_000,
        }
    }

    /// **Guidance is a pool of its own**: a guide enters beside a full budget
    /// of files without dislodging one, and a stronger guide dislodges the
    /// guide, never the weaker file.
    #[test]
    fn a_guide_makes_room_among_the_guides_alone() {
        let mut ws = WorkingSet::default();
        let lim = guide_limits();
        let fresh = HashMap::from([(tl(1), 200.0), (tl(2), 200.0), (tl(60), 900.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(
            order(&ws),
            vec![60, 1, 2],
            "the files' budget is full and the guide still entered"
        );
        // 60 decays to 720, and 720 × 1.5 < 5,000; the files sit at 160.
        let fresh = HashMap::from([(tl(61), 5_000.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(order(&ws), vec![1, 2, 61], "60 left for 61; no file moved");
    }

    /// **No burst of files crowds out a guide**: a file far stronger than the
    /// guide dislodges a file, never the guide.
    #[test]
    fn a_file_never_dislodges_a_guide() {
        let mut ws = WorkingSet::default();
        let lim = guide_limits();
        let fresh = HashMap::from([(tl(1), 900.0), (tl(2), 900.0), (tl(60), 200.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(order(&ws), vec![1, 2, 60]);
        let fresh = HashMap::from([(tl(3), 50_000.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        let members = order(&ws);
        assert!(members.contains(&60), "the guide stayed: {members:?}");
        assert!(members.contains(&3), "the file entered: {members:?}");
        assert_eq!(members.len(), 3, "one file left for it: {members:?}");
    }

    /// With no guidance pool declared, a guide never enters.
    #[test]
    fn with_no_guidance_pool_no_guide_enters() {
        let mut ws = WorkingSet::default();
        let lim = Limits {
            guidance_tokens: 0,
            ..guide_limits()
        };
        let fresh = HashMap::from([(tl(60), 5_000.0), (tl(1), 100.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, NonZeroUsize::MAX, &info);
        assert_eq!(order(&ws), vec![1]);
    }

    /// **A member the scope no longer offers leaves.**
    #[test]
    fn a_member_out_of_scope_leaves() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 900.0)], 1_000);
        let gone = |t: TimelineId| (t.raw() == 2).then(|| cand(t.raw()));
        ws.observe(
            &HashMap::new(),
            0.2,
            100.0,
            limits(1_000),
            NonZeroUsize::MAX,
            &gone,
        );
        assert_eq!(order(&ws), vec![2]);
    }

    /// **One reprojection admits at most `max_admits` newcomers**, strongest
    /// first; the rest keep their momentum and enter on the next ones.
    #[test]
    fn a_reprojection_admits_at_most_max_admits_newcomers() {
        let mut ws = WorkingSet::default();
        let fresh: HashMap<TimelineId, f32> = HashMap::from([
            (tl(1), 900.0),
            (tl(2), 800.0),
            (tl(3), 700.0),
            (tl(4), 600.0),
        ]);
        let two = NonZeroUsize::new(2).unwrap();
        ws.observe(&fresh, 0.2, 100.0, limits(1_000), two, &info);
        assert_eq!(order(&ws), vec![1, 2]);
        assert_eq!(
            ws.momentum_of(tl(4)),
            Some(600.0),
            "a waiting file keeps its momentum"
        );
        ws.observe(&HashMap::new(), 0.2, 100.0, limits(1_000), two, &info);
        assert_eq!(order(&ws), vec![1, 2, 3, 4]);
    }

    /// **Documents enter up to their share, then the code does**: three strong
    /// documents and two weaker files against a 1,000-token budget whose
    /// documents may hold 200. Two documents fill the share; the third, no
    /// stronger than either, waits; both files enter on the room it left.
    #[test]
    fn documents_enter_up_to_their_share_then_code() {
        let mut ws = WorkingSet::default();
        observe(
            &mut ws,
            &[
                (70, 900.0),
                (71, 900.0),
                (72, 900.0),
                (1, 300.0),
                (2, 300.0),
            ],
            1_000,
        );
        assert_eq!(order(&ws), vec![70, 71, 1, 2]);
        assert_eq!(
            ws.momentum_of(tl(72)),
            Some(900.0),
            "72 waits as a candidate"
        );
        assert_eq!(ws.type_tokens(DOCS.id), 200);
    }

    /// **A typed newcomer dislodges only its own type**: a far stronger
    /// document takes the weakest document's place, never the weaker file's.
    #[test]
    fn a_strong_document_dislodges_a_document_never_code() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(70, 300.0), (71, 400.0), (1, 150.0)], 1_000);
        assert_eq!(order(&ws), vec![71, 70, 1]);
        observe(&mut ws, &[(72, 5_000.0)], 1_000);
        assert_eq!(
            order(&ws),
            vec![71, 1, 72],
            "70, the weakest document, left"
        );
    }

    /// **Seeds keep to the share too**: of three document seeds two enter,
    /// and the code seed after them.
    #[test]
    fn document_seeds_keep_to_the_share() {
        let mut ws = WorkingSet::default();
        let seeds: Vec<(TimelineId, Candidate)> =
            [70, 71, 72, 1].iter().map(|&r| (tl(r), cand(r))).collect();
        let refused = ws.seed(&seeds, limits(1_000), 1_000.0);
        assert_eq!(order(&ws), vec![70, 71, 1]);
        assert_eq!(refused, vec![tl(72)]);
    }

    /// **A document guide holds no share of the budget**: guidance is its own
    /// pool, so a typed guide neither counts toward nor is refused by it.
    #[test]
    fn a_typed_guide_is_outside_the_share() {
        let guide_doc = Candidate {
            tokens: Some(100),
            share: Some(GUIDES),
            kind: Some(DOCS),
        };
        let mut ws = WorkingSet::default();
        let seeds: Vec<(TimelineId, Candidate)> =
            [70, 71].iter().map(|&r| (tl(r), cand(r))).collect();
        ws.seed(&seeds, limits(1_000), 1_000.0);
        let refused = ws.seed(&[(tl(60), guide_doc)], limits(1_000), 1_000.0);
        assert!(refused.is_empty());
        assert_eq!(ws.type_tokens(DOCS.id), 200, "the guide is not counted");
    }

    #[test]
    fn a_removed_conversation_leaves_and_the_gap_closes() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 800.0), (3, 700.0)], 1_000);
        ws.remove(tl(1));
        assert_eq!(order(&ws), vec![2, 3]);
        assert_eq!(ws.momentum_of(tl(1)), None);
    }
}
