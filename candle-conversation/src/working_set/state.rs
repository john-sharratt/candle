//! One dialogue's working set: the pool of whole conversations the slot fill
//! draws its working-set rounds from, in the order they entered
//! (`docs/zend_working_set.md` §4.3, `docs/hit_driven_projection.md` §7.2
//! step 1).

use std::collections::{HashMap, HashSet};

use super::pool_size::PoolSize;
use super::type_shares::TypeCap;
use crate::projection::{HitMomentum, HitsConfig, TimelineId, TurnKey, WorkingSetShare};

/// What pooling a conversation needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Its recorded token total; `None` when none was recorded.
    pub tokens: Option<usize>,
    /// The kind of piece it is, by its group's share; `None` when it belongs
    /// to no working-set group.
    pub share: Option<WorkingSetShare>,
    /// The type of document it is and that type's cap
    /// ([`super::TypeShares::classify`]); `None` for one of no declared type,
    /// which is unbounded.
    pub kind: Option<TypeCap>,
}

/// One member of the sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Member {
    timeline: TimelineId,
    tokens: usize,
    share: WorkingSetShare,
    kind: Option<TypeCap>,
}

/// Every kind of piece the pool holds a minimum of.
const KINDS: [WorkingSetShare; 4] = [
    WorkingSetShare::Readings,
    WorkingSetShare::Folders,
    WorkingSetShare::Guides,
    WorkingSetShare::Journals,
];

/// What a dialogue's pool holds: one sequence of whole conversations, in the
/// order its members entered.
///
/// **Membership is chosen again at every observation** ([`Self::observe`]):
/// each kind's strongest pieces up to [`PoolSize::per_kind`], then the
/// strongest of every kind while [`PoolSize::tokens`] has room, a typed
/// reading only while its type is under its share. Nothing throttles it: a
/// piece that ranks in enters at once, and one that ranks out leaves.
///
/// **Position is insertion order, never rank.** A member chosen again keeps
/// its place, a newcomer enters after every member already there, and when
/// one leaves the gap closes — so the slot seats placed pieces in an order
/// that only ever changes where a piece enters or leaves.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkingSet {
    members: Vec<Member>,
    /// Momentum per conversation — members and candidates alike.
    momentum: HashMap<TimelineId, f32>,
    /// Conversations that may not enter for now: the readings of files the
    /// current turn's loop read for real, already verbatim in its dialogue
    /// ([`Self::withhold`]). They keep their momentum, and a member among them
    /// keeps its place.
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
    /// The turns the dialogue's slot held at the last boundary
    /// ([`Self::boundary`]) — what the fill keeps seated
    /// (`docs/hit_driven_projection.md` §7.2 step 5).
    held: HashSet<TurnKey>,
    /// Set when a new question starts ([`Self::clear_withheld`]) and taken by
    /// the next boundary, which then chooses whole.
    opening_pending: bool,
    /// Whether the last boundary was the first after a new question.
    opening: bool,
}

impl WorkingSet {
    /// Keep `timelines` from entering until [`Self::clear_withheld`]. A member
    /// among them stays where it is.
    pub fn withhold(&mut self, timelines: impl IntoIterator<Item = TimelineId>) {
        self.withheld.extend(timelines);
    }

    /// Let every withheld conversation enter again: a new question starts a
    /// new loop. Its first boundary chooses the slot whole ([`Self::opening`]).
    pub fn clear_withheld(&mut self) {
        self.withheld.clear();
        self.opening_pending = true;
    }

    /// A boundary: the slot holds `placed` now, and the first boundary after
    /// a new question takes the opening ([`Self::clear_withheld`]).
    pub fn boundary(&mut self, placed: impl IntoIterator<Item = TurnKey>) {
        self.held = placed.into_iter().collect();
        self.opening = std::mem::take(&mut self.opening_pending);
    }

    /// The turns the slot held at the last boundary.
    pub fn held(&self) -> &HashSet<TurnKey> {
        &self.held
    }

    /// Whether the last boundary was the first after a new question.
    pub fn opening(&self) -> bool {
        self.opening
    }

    /// Every member in insertion order.
    pub fn members(&self) -> Vec<TimelineId> {
        self.members.iter().map(|m| m.timeline).collect()
    }

    /// Every member in insertion order, with its tokens and its kind.
    pub fn pooled(&self) -> Vec<(TimelineId, usize, WorkingSetShare)> {
        self.members
            .iter()
            .map(|m| (m.timeline, m.tokens, m.share))
            .collect()
    }

    /// Whether `timeline` is a member.
    pub fn is_member(&self, timeline: TimelineId) -> bool {
        self.position(timeline).is_some()
    }

    /// The members with their momentum, in insertion order.
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
    /// what the pool ranks by. `None` for one with no momentum.
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

    /// Every conversation the last scan folded in scored, with its fresh
    /// score — pooled or not.
    pub fn scanned(&self) -> impl Iterator<Item = (TimelineId, f32)> + '_ {
        self.fresh.iter().map(|(&t, &s)| (t, s))
    }

    /// Tokens every member together holds.
    pub fn tokens(&self) -> usize {
        self.members.iter().map(|m| m.tokens).sum()
    }

    fn position(&self, timeline: TimelineId) -> Option<usize> {
        self.members.iter().position(|m| m.timeline == timeline)
    }

    /// Fold one reprojection in.
    ///
    /// 1. Momentum: `m ← (1 − β)·m + fresh` for every conversation, where one
    ///    the scan did not score counts as a fresh zero. Below `min_momentum`
    ///    — judged on strength, so a piece the indexer keeps reading outlasts
    ///    its fading score — a conversation leaves the map.
    /// 2. The pool is chosen again from what is left ([`Self::choose`]).
    pub fn observe(
        &mut self,
        fresh: &HashMap<TimelineId, f32>,
        beta: f32,
        min_momentum: f32,
        pool: PoolSize,
        candidate: &dyn Fn(TimelineId) -> Option<Candidate>,
    ) {
        let keep = 1.0 - beta;
        for m in self.momentum.values_mut() {
            *m *= keep;
        }
        for (&timeline, &score) in fresh {
            *self.momentum.entry(timeline).or_insert(0.0) += score;
        }
        self.fresh.clone_from(fresh);
        let (hits, hits_cfg) = (&self.hits, &self.hits_cfg);
        self.momentum
            .retain(|timeline, m| *m * hits.lift(*timeline, hits_cfg) >= min_momentum);
        self.choose(pool, candidate);
    }

    /// Choose the pool from every conversation with momentum that `candidate`
    /// offers whole: each kind's strongest up to `pool.per_kind`, whatever
    /// they cost, then the strongest of every kind while `pool.tokens` has
    /// room. A typed piece enters only while its type holds under its share
    /// of `pool.tokens`. A withheld conversation that is not a member waits.
    /// Members chosen again keep their order; newcomers enter after them,
    /// strongest first.
    fn choose(&mut self, pool: PoolSize, candidate: &dyn Fn(TimelineId) -> Option<Candidate>) {
        let mut ranked: Vec<(Member, f32)> = self
            .momentum
            .keys()
            .filter(|t| self.is_member(**t) || !self.withheld.contains(t))
            .filter_map(|&timeline| {
                let c = candidate(timeline)?;
                let member = Member {
                    timeline,
                    tokens: c.tokens?,
                    share: c.share?,
                    kind: c.kind,
                };
                Some((member, self.score_of(timeline).unwrap_or(0.0)))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then(a.0.timeline.raw().cmp(&b.0.timeline.raw()))
        });

        let mut chosen: HashSet<TimelineId> = HashSet::new();
        let mut used = 0usize;
        let mut typed: HashMap<u8, usize> = HashMap::new();
        let mut take = |m: &Member, chosen: &mut HashSet<TimelineId>, used: &mut usize| -> bool {
            if let Some(k) = m.kind {
                let held = typed.entry(k.id).or_default();
                if *held + m.tokens > k.tokens(pool.tokens) {
                    return false;
                }
                *held += m.tokens;
            }
            chosen.insert(m.timeline);
            *used += m.tokens;
            true
        };
        for kind in KINDS {
            let mut count = 0usize;
            // The scan's own leaders first: the pieces this boundary's scan
            // scores highest hold their places whatever the hits lift the
            // others to, so the hits reorder what the question ranks and
            // never push it out (`docs/hit_driven_projection.md` §7.2 step 1).
            let mut leaders: Vec<&Member> = ranked
                .iter()
                .map(|(m, _)| m)
                .filter(|m| m.share == kind && self.fresh_of(m.timeline) > 0.0)
                .collect();
            leaders.sort_by(|a, b| {
                self.fresh_of(b.timeline)
                    .total_cmp(&self.fresh_of(a.timeline))
                    .then(a.timeline.raw().cmp(&b.timeline.raw()))
            });
            for m in leaders {
                if count == pool.per_kind {
                    break;
                }
                if take(m, &mut chosen, &mut used) {
                    count += 1;
                }
            }
            for (m, _) in ranked.iter().filter(|(m, _)| m.share == kind) {
                if count == pool.per_kind {
                    break;
                }
                if chosen.contains(&m.timeline) {
                    continue;
                }
                if take(m, &mut chosen, &mut used) {
                    count += 1;
                }
            }
        }
        for (m, _) in &ranked {
            if chosen.contains(&m.timeline) || used + m.tokens > pool.tokens {
                continue;
            }
            take(m, &mut chosen, &mut used);
        }

        let by_timeline: HashMap<TimelineId, Member> =
            ranked.iter().map(|(m, _)| (m.timeline, *m)).collect();
        let mut members: Vec<Member> = self
            .members
            .iter()
            .filter(|m| chosen.contains(&m.timeline))
            .map(|m| by_timeline[&m.timeline])
            .collect();
        for (m, _) in &ranked {
            if chosen.contains(&m.timeline) && !members.iter().any(|x| x.timeline == m.timeline) {
                members.push(*m);
            }
        }
        self.members = members;
    }

    /// Take `timeline` out of the set — its conversation is gone.
    pub fn remove(&mut self, timeline: TimelineId) {
        if let Some(at) = self.position(timeline) {
            self.members.remove(at);
        }
        self.momentum.remove(&timeline);
    }

    /// Tokens the members of type `id` hold.
    #[cfg(test)]
    fn type_tokens(&self, id: u8) -> usize {
        self.members
            .iter()
            .filter(|m| m.kind.is_some_and(|k| k.id == id))
            .map(|m| m.tokens)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILES: WorkingSetShare = WorkingSetShare::Readings;
    const FOLDERS: WorkingSetShare = WorkingSetShare::Folders;
    const GUIDES: WorkingSetShare = WorkingSetShare::Guides;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A pool of `tokens`, no kind held to a minimum.
    fn pool(tokens: usize) -> PoolSize {
        PoolSize {
            tokens,
            per_kind: 0,
        }
    }

    /// Documents may hold a fifth of the pool.
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

    fn observe_in(ws: &mut WorkingSet, fresh: &[(u64, f32)], size: PoolSize) {
        let fresh: HashMap<TimelineId, f32> = fresh.iter().map(|&(r, s)| (tl(r), s)).collect();
        ws.observe(&fresh, 0.2, 100.0, size, &info);
    }

    fn observe(ws: &mut WorkingSet, fresh: &[(u64, f32)], tokens: usize) {
        observe_in(ws, fresh, pool(tokens));
    }

    fn read(raws: &[u64]) -> HashMap<TimelineId, f32> {
        raws.iter().map(|&r| (tl(r), 1.0)).collect()
    }

    /// **A whole piece of any size is pooled**: no per-piece cap; only the
    /// pool's tokens bound it.
    #[test]
    fn a_large_reading_is_pooled_whole() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(8, 900.0), (1, 300.0)], 2_100);
        assert_eq!(order(&ws), vec![8, 1]);
        assert_eq!(ws.tokens(), 2_100);
    }

    /// **The pool is chosen by strength, unthrottled**: four candidates enter
    /// at one observation when they fit, and past the pool's tokens the
    /// weakest are left out.
    #[test]
    fn the_strongest_enter_at_once_while_the_pool_has_room() {
        let mut ws = WorkingSet::default();
        observe(
            &mut ws,
            &[(1, 900.0), (2, 800.0), (3, 700.0), (4, 600.0)],
            1_000,
        );
        assert_eq!(order(&ws), vec![1, 2, 3, 4]);
        let mut small = WorkingSet::default();
        observe(
            &mut small,
            &[(1, 900.0), (2, 800.0), (3, 700.0), (4, 600.0)],
            200,
        );
        assert_eq!(order(&small), vec![1, 2]);
    }

    /// **Each kind holds its minimum whatever the tokens**: a pool of 200
    /// tokens with one of each kind held takes the strongest file and the
    /// strongest folder first, though two stronger files would fill it.
    #[test]
    fn each_kind_holds_its_minimum() {
        let mut ws = WorkingSet::default();
        let size = PoolSize {
            tokens: 200,
            per_kind: 1,
        };
        observe_in(
            &mut ws,
            &[(1, 900.0), (2, 800.0), (50, 150.0), (60, 120.0)],
            size,
        );
        let members = order(&ws);
        assert_eq!(members.len(), 3, "{members:?}");
        for raw in [1, 50, 60] {
            assert!(members.contains(&raw), "{raw} in {members:?}");
        }
    }

    /// **A member chosen again keeps its place; a newcomer enters after the
    /// members**, whatever its strength.
    #[test]
    fn a_newcomer_enters_after_the_members_already_there() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 300.0)], 1_000);
        observe(&mut ws, &[(2, 5_000.0)], 1_000);
        assert_eq!(order(&ws), vec![1, 2]);
    }

    /// **A member that ranks out leaves and the gap closes**: when a far
    /// stronger piece needs the room, the weakest member goes, and the rest
    /// keep their order.
    #[test]
    fn a_member_ranked_out_leaves_and_the_gap_closes() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 300.0), (3, 800.0)], 300);
        assert_eq!(order(&ws), vec![1, 3, 2]);
        observe(&mut ws, &[(4, 5_000.0)], 300);
        assert_eq!(order(&ws), vec![1, 3, 4], "2 was the weakest");
    }

    /// **A file the indexer keeps reading outlasts its fading score**: two
    /// files at one momentum, only one read each stride. Two reprojections
    /// later the unread one's momentum is under the floor and it leaves; the
    /// read one's strength is its momentum lifted by `1 + h`.
    #[test]
    fn a_read_file_outlasts_an_unread_one() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 150.0), (2, 150.0)], 1000);
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

    /// **Hits decide which member stays** when the pool is full: of two
    /// members at one momentum, the one the indexer reads keeps its place
    /// against a newcomer between them in strength.
    #[test]
    fn hits_decide_which_member_stays() {
        for (kept_by_hits, dropped) in [(1u64, 2u64), (2, 1)] {
            let mut ws = WorkingSet::default();
            observe(&mut ws, &[(1, 1000.0), (2, 1000.0)], 200);
            ws.observe_hits(&read(&[kept_by_hits]), &HitsConfig::default());
            observe(&mut ws, &[(3, 900.0)], 200);
            let members = order(&ws);
            assert!(members.contains(&kept_by_hits), "{members:?}");
            assert!(members.contains(&3), "{members:?}");
            assert!(!members.contains(&dropped), "{members:?}");
        }
    }

    /// **The scan's own leaders hold their places against hit-lifted
    /// strength**: a member the indexer read stands at 800 · (1 + 1) = 1,600;
    /// a newcomer the scan scores 900 is weaker but leads the scan, so it
    /// takes the kind's one place and the read member no longer fits.
    #[test]
    fn the_scans_leaders_hold_their_places_against_hits() {
        let size = PoolSize {
            tokens: 100,
            per_kind: 1,
        };
        let mut ws = WorkingSet::default();
        observe_in(&mut ws, &[(1, 1_000.0)], size);
        ws.observe_hits(&read(&[1]), &HitsConfig::default());
        observe_in(&mut ws, &[(2, 900.0)], size);
        assert!((ws.score_of(tl(1)).unwrap() - 1_600.0).abs() < 1e-3);
        assert_eq!(order(&ws), vec![2]);
    }

    /// **A scan leader counts toward its kind's minimum**: with two places
    /// per kind, the scan's leader and the strongest of the rest are both
    /// pooled, and the leader is not counted twice against the tokens.
    #[test]
    fn a_scan_leader_counts_toward_its_kinds_minimum() {
        let size = PoolSize {
            tokens: 200,
            per_kind: 2,
        };
        let mut ws = WorkingSet::default();
        observe_in(&mut ws, &[(1, 1_000.0), (3, 1_000.0)], size);
        ws.observe_hits(&read(&[1, 3]), &HitsConfig::default());
        observe_in(&mut ws, &[(2, 900.0)], size);
        let members = order(&ws);
        assert_eq!(members.len(), 2, "{members:?}");
        assert!(members.contains(&2), "{members:?}");
        assert_eq!(ws.tokens(), 200);
    }

    /// Hit momentum leaks like momentum and leaves the map once it lifts
    /// nothing; with no hits folded in, strength is momentum.
    #[test]
    fn hit_momentum_leaks_and_leaves() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 1000.0)], 1000);
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
    /// keeps it pooled.
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
    /// duplicate. Cleared, it enters on the momentum it kept.
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

    /// A conversation with no total, or in no group, never enters.
    #[test]
    fn a_conversation_with_no_total_or_in_no_group_never_enters() {
        let nowhere = Candidate {
            tokens: Some(100),
            share: None,
            kind: None,
        };
        let mut ws = WorkingSet::default();
        let fresh: HashMap<TimelineId, f32> = HashMap::from([(tl(9), 1_000.0), (tl(7), 1_000.0)]);
        let candidate = |t: TimelineId| match t.raw() {
            7 => Some(nowhere),
            _ => info(t),
        };
        ws.observe(&fresh, 0.2, 100.0, pool(10_000), &candidate);
        assert!(ws.members().is_empty());
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

    /// **A conversation opens on an empty pool**: nothing is a member until a
    /// scan scores it.
    #[test]
    fn a_pool_opens_empty() {
        let ws = WorkingSet::default();
        assert!(ws.members().is_empty());
        assert_eq!(ws.tokens(), 0);
    }

    /// An unattended file fades: 1,000 × 0.8¹⁰ ≈ 107 is still in after ten
    /// reprojections that do not score it, and the eleventh takes it out.
    #[test]
    fn an_unattended_file_fades_out() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 1_000.0)], 1_000);
        for _ in 0..10 {
            observe(&mut ws, &[], 1_000);
        }
        assert_eq!(order(&ws), vec![1]);
        observe(&mut ws, &[], 1_000);
        assert!(ws.members().is_empty());
    }

    /// **A member the scope no longer offers leaves.**
    #[test]
    fn a_member_out_of_scope_leaves() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 900.0)], 1_000);
        let gone = |t: TimelineId| (t.raw() == 2).then(|| cand(t.raw()));
        ws.observe(&HashMap::new(), 0.2, 100.0, pool(1_000), &gone);
        assert_eq!(order(&ws), vec![2]);
    }

    /// **Documents hold up to their share of the pool, then the code does**:
    /// three strong documents and two weaker files against a 1,000-token pool
    /// whose documents may hold 200. Two documents fill the share; the third
    /// waits; both files enter.
    #[test]
    fn documents_hold_up_to_their_share_then_code() {
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
        assert_eq!(ws.momentum_of(tl(72)), Some(900.0), "72 keeps its momentum");
        assert_eq!(ws.type_tokens(DOCS.id), 200);
    }

    /// The pooled pieces carry their tokens and kinds, in insertion order.
    #[test]
    fn the_pool_lists_each_piece_with_its_kind() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (50, 800.0), (60, 700.0)], 1_000);
        assert_eq!(
            ws.pooled(),
            vec![
                (tl(1), 100, FILES),
                (tl(50), 100, FOLDERS),
                (tl(60), 100, GUIDES)
            ]
        );
    }

    /// **A new question opens the next boundary, and only that one**: the
    /// boundary after `clear_withheld` is the opening, the one after it is
    /// not; each records what the slot holds.
    #[test]
    fn a_new_question_opens_the_next_boundary_only() {
        use crate::projection::TurnIndex;
        let key = |i| TurnKey::new(tl(1), TurnIndex(i));
        let mut ws = WorkingSet::default();
        ws.boundary([key(0)]);
        assert!(!ws.opening());
        ws.clear_withheld();
        ws.boundary([key(1)]);
        assert!(ws.opening());
        assert_eq!(ws.held(), &HashSet::from([key(1)]));
        ws.boundary([key(2)]);
        assert!(!ws.opening());
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
