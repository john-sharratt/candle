//! One dialogue's working set: the conversations it carries ahead of its own
//! turns, in the order they entered (`docs/zend_working_set.md` §4.3).

use std::collections::HashMap;

use crate::projection::{TimelineId, WorkingSetShare};

/// How much stronger a newcomer must be than the provenance it dislodges.
/// Without the margin two files near the floor trade places every
/// reprojection, and each trade rebuilds everything to the right of it.
const DISLODGE_MARGIN: f32 = 1.5;

/// The size limits an admission is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Tokens every member together may hold.
    pub budget_tokens: usize,
    /// Tokens the folder members may hold between them; the files take what is
    /// left of `budget_tokens`.
    pub folder_tokens: usize,
    /// Tokens one member may hold.
    pub max_file_tokens: usize,
}

/// Why a lock was refused. Every refusal sends the call to a real read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The conversation cannot deliver its content: it is tombstoned, archived,
    /// holds no sealed turn, is the dialogue itself, or belongs to no
    /// working-set group.
    Gone,
    /// The conversation has no recorded token total, so what it would cost is
    /// unknown.
    NoCost,
    /// The conversation alone is past `max_file_tokens`.
    TooLarge,
    /// The locks already held leave no room for it.
    Full,
}

/// What admitting a conversation needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Its recorded token total; `None` when none was recorded.
    pub tokens: Option<usize>,
    /// The share of the budget it draws on; `None` when it belongs to no
    /// working-set group.
    pub share: Option<WorkingSetShare>,
}

/// One member of the sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Member {
    timeline: TimelineId,
    tokens: usize,
    share: WorkingSetShare,
    locked: bool,
}

/// What a dialogue carries ahead of its own turns: one sequence, in the order
/// its members entered, and one insertion point.
///
/// **Position is insertion order, never age or rank.** Everything enters at
/// the insertion point: a lock just before it, so the oldest lock stays
/// nearest the dialogue; provenance just after the provenance already there,
/// so the newest provenance sits beside the locks. A member keeps its place
/// until it leaves, and when it leaves the gap closes at once.
///
/// That is what keeps the projection stable under RoPE. A change at one
/// position leaves everything to its right at the same distance from the
/// question and everything to its left at the same distance from the system
/// prompt. Every change here is an insertion at the point or a removal, so:
/// a lock never moves relative to the question while it is held; long-lived
/// provenance on the left keeps its place beside the system prompt; and the
/// churn — newcomers arriving, the weakest leaving — lands in the middle,
/// where attention is weakest anyway.
///
/// **Locks** are pinned: they never decay and are never dislodged. A lock is
/// released into provenance in place when the turn ends ([`Self::release`]),
/// and the insertion point moves past it, so the next task's members enter
/// beside the dialogue. **Provenance** decays by momentum and leaves below the
/// floor. When a newcomer needs room, the weakest provenance is dislodged —
/// by a lock always, by new provenance only when the newcomer is
/// [`DISLODGE_MARGIN`] stronger. The **seeds** are the first provenance a
/// conversation admits.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkingSet {
    members: Vec<Member>,
    /// Where the next member enters: members before it entered as provenance,
    /// members from it on as this task's locks.
    insert_at: usize,
    /// Momentum per conversation — members and candidates alike. A locked
    /// member's entry is frozen until its release.
    momentum: HashMap<TimelineId, f32>,
}

impl WorkingSet {
    /// Every member in emission order, each with whether it is locked.
    pub fn members(&self) -> Vec<(TimelineId, bool)> {
        self.members
            .iter()
            .map(|m| (m.timeline, m.locked))
            .collect()
    }

    /// The locked members, in emission order — the pinned tier.
    pub fn pinned(&self) -> Vec<TimelineId> {
        self.members
            .iter()
            .filter(|m| m.locked)
            .map(|m| m.timeline)
            .collect()
    }

    /// Whether `timeline` is locked.
    pub fn is_pinned(&self, timeline: TimelineId) -> bool {
        self.members
            .iter()
            .any(|m| m.timeline == timeline && m.locked)
    }

    /// Whether `timeline` is a member, locked or not.
    pub fn is_member(&self, timeline: TimelineId) -> bool {
        self.position(timeline).is_some()
    }

    /// The unlocked members with their momentum, in emission order.
    pub fn provenance(&self) -> Vec<(TimelineId, f32)> {
        self.members
            .iter()
            .filter(|m| !m.locked)
            .map(|m| (m.timeline, self.momentum_of(m.timeline).unwrap_or(0.0)))
            .collect()
    }

    /// `timeline`'s momentum, when it has any.
    pub fn momentum_of(&self, timeline: TimelineId) -> Option<f32> {
        self.momentum.get(&timeline).copied()
    }

    /// Tokens every member together holds.
    pub fn tokens(&self) -> usize {
        self.members.iter().map(|m| m.tokens).sum()
    }

    /// Where the next member enters.
    pub fn insertion_point(&self) -> usize {
        self.insert_at
    }

    fn position(&self, timeline: TimelineId) -> Option<usize> {
        self.members.iter().position(|m| m.timeline == timeline)
    }

    /// Lock `timeline` for the rest of the task.
    ///
    /// A member is locked where it stands — moving it would undo the stability
    /// this order exists for, and its content is in context wherever it sits.
    /// A newcomer dislodges the weakest provenance until it fits, then enters
    /// just before the insertion point. Locks are never dislodged, so a lock
    /// that cannot fit beside the ones already held is refused and the call
    /// runs for real.
    pub fn lock(
        &mut self,
        timeline: TimelineId,
        candidate: Candidate,
        limits: Limits,
    ) -> Result<(), Refusal> {
        if let Some(at) = self.position(timeline) {
            self.members[at].locked = true;
            return Ok(());
        }
        let share = candidate.share.ok_or(Refusal::Gone)?;
        let tokens = candidate.tokens.ok_or(Refusal::NoCost)?;
        if tokens > limits.max_file_tokens {
            return Err(Refusal::TooLarge);
        }
        let victims = self
            .room_for(tokens, share, limits, &|_| true)
            .ok_or(Refusal::Full)?;
        self.evict(&victims);
        self.members.insert(
            self.insert_at,
            Member {
                timeline,
                tokens,
                share,
                locked: true,
            },
        );
        Ok(())
    }

    /// Put back a lock the conversation's history says was served, whatever it
    /// costs now: the model was told it has this content, and a restart must
    /// not quietly take it away.
    pub fn restore_lock(&mut self, timeline: TimelineId, tokens: usize, share: WorkingSetShare) {
        if let Some(at) = self.position(timeline) {
            self.members[at].locked = true;
            return;
        }
        self.members.insert(
            self.insert_at,
            Member {
                timeline,
                tokens,
                share,
                locked: true,
            },
        );
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
            if self.room_for(tokens, share, limits, &|_| false).is_none() {
                refused.push(timeline);
                continue;
            }
            self.admit(timeline, tokens, share);
        }
        refused
    }

    /// The end of a task: every lock becomes provenance where it stands, at
    /// `released` momentum (or what it already had, if that is higher), and the
    /// insertion point moves past them all — so a file the model kept working
    /// in stays, one it read once decays out, and the next task's members enter
    /// beside the dialogue. Nothing moves.
    pub fn release(&mut self, released: f32) {
        for m in &mut self.members {
            if m.locked {
                m.locked = false;
                let entry = self.momentum.entry(m.timeline).or_insert(0.0);
                *entry = entry.max(released);
            }
        }
        self.insert_at = self.members.len();
    }

    /// Fold one reprojection in.
    ///
    /// 1. Momentum: `m ← (1 − β)·m + fresh` for every conversation that is not
    ///    locked, where one the scan did not score counts as a fresh zero.
    /// 2. Below `min_momentum` a conversation leaves the map, and a member
    ///    leaves the sequence — the gap closes. So does an unlocked member
    ///    `admissible` no longer allows (out of scope, gone).
    /// 3. The strongest candidates that are not members try to enter, each
    ///    dislodging only provenance it beats by [`DISLODGE_MARGIN`].
    pub fn observe(
        &mut self,
        fresh: &HashMap<TimelineId, f32>,
        beta: f32,
        min_momentum: f32,
        limits: Limits,
        candidate: &dyn Fn(TimelineId) -> Option<Candidate>,
    ) {
        let locked: Vec<TimelineId> = self.pinned();
        let keep = 1.0 - beta;
        for (timeline, m) in self.momentum.iter_mut() {
            if !locked.contains(timeline) {
                *m *= keep;
            }
        }
        for (&timeline, &score) in fresh {
            if !locked.contains(&timeline) {
                *self.momentum.entry(timeline).or_insert(0.0) += score;
            }
        }
        self.momentum
            .retain(|timeline, m| *m >= min_momentum || locked.contains(timeline));

        let leaving: Vec<usize> = self
            .members
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                !m.locked
                    && (!self.momentum.contains_key(&m.timeline) || candidate(m.timeline).is_none())
            })
            .map(|(i, _)| i)
            .collect();
        self.evict(&leaving);

        let mut newcomers: Vec<(TimelineId, f32)> = self
            .momentum
            .iter()
            .filter(|(timeline, _)| !self.is_member(**timeline))
            .map(|(timeline, m)| (*timeline, *m))
            .collect();
        newcomers.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
        for (timeline, strength) in newcomers {
            let Some(Candidate {
                tokens: Some(tokens),
                share: Some(share),
            }) = candidate(timeline)
            else {
                continue;
            };
            if tokens > limits.max_file_tokens {
                continue;
            }
            let beaten = |weaker: f32| weaker * DISLODGE_MARGIN < strength;
            if let Some(victims) = self.room_for(tokens, share, limits, &beaten) {
                self.evict(&victims);
                self.admit(timeline, tokens, share);
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

    /// Enter `timeline` as provenance at the insertion point, which moves past
    /// it.
    fn admit(&mut self, timeline: TimelineId, tokens: usize, share: WorkingSetShare) {
        self.members.insert(
            self.insert_at,
            Member {
                timeline,
                tokens,
                share,
                locked: false,
            },
        );
        self.insert_at += 1;
    }

    /// The members that must leave for `tokens` of `share` to fit, weakest
    /// first among the unlocked ones `may_dislodge` allows (judged on their
    /// momentum) — or `None` when no such set makes it fit. On a momentum tie
    /// the one nearest the insertion point goes first: it moves the fewest
    /// members on its left.
    fn room_for(
        &self,
        tokens: usize,
        share: WorkingSetShare,
        limits: Limits,
        may_dislodge: &dyn Fn(f32) -> bool,
    ) -> Option<Vec<usize>> {
        let mut total = self.tokens();
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
            .filter(|(_, m)| !m.locked)
            .map(|(i, m)| (i, self.momentum_of(m.timeline).unwrap_or(0.0)))
            .filter(|&(_, m)| may_dislodge(m))
            .collect();
        pool.sort_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        let mut victims = Vec::new();
        loop {
            let over_total = total + tokens > limits.budget_tokens;
            let over_folders = folder_bound(folders);
            if !over_total && !over_folders {
                return Some(victims);
            }
            let next = pool.iter().position(|&(i, _)| {
                !over_folders || self.members[i].share == WorkingSetShare::Folders
            })?;
            let (i, _) = pool.remove(next);
            let m = &self.members[i];
            total -= m.tokens;
            if m.share == WorkingSetShare::Folders {
                folders -= m.tokens;
            }
            victims.push(i);
        }
    }

    /// Remove the members at `indices`, closing each gap and keeping the
    /// insertion point between the same two members.
    fn evict(&mut self, indices: &[usize]) {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted {
            self.members.remove(i);
            if i < self.insert_at {
                self.insert_at -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILES: WorkingSetShare = WorkingSetShare::Remainder;
    const FOLDERS: WorkingSetShare = WorkingSetShare::Folders;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn limits(budget_tokens: usize) -> Limits {
        Limits {
            budget_tokens,
            folder_tokens: budget_tokens,
            max_file_tokens: 1_000,
        }
    }

    /// A 100-token file, except 9 (no total) and 8 (2,000 tokens); 50–59 are
    /// 100-token folders.
    fn info(timeline: TimelineId) -> Option<Candidate> {
        let raw = timeline.raw();
        Some(Candidate {
            tokens: match raw {
                9 => None,
                8 => Some(2_000),
                _ => Some(100),
            },
            share: Some(if (50..60).contains(&raw) {
                FOLDERS
            } else {
                FILES
            }),
        })
    }

    fn cand(raw: u64) -> Candidate {
        info(tl(raw)).unwrap()
    }

    fn order(ws: &WorkingSet) -> Vec<u64> {
        ws.members().iter().map(|(t, _)| t.raw()).collect()
    }

    fn observe(ws: &mut WorkingSet, fresh: &[(u64, f32)], budget: usize) {
        let fresh: HashMap<TimelineId, f32> = fresh.iter().map(|&(r, s)| (tl(r), s)).collect();
        ws.observe(&fresh, 0.2, 100.0, limits(budget), &info);
    }

    /// **Locks enter just before the insertion point**: the oldest lock stays
    /// nearest the dialogue, the youngest beside the provenance.
    #[test]
    fn a_new_lock_enters_left_of_the_older_locks() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0)], 1_000);
        for raw in [2, 3, 4] {
            ws.lock(tl(raw), cand(raw), limits(1_000)).unwrap();
        }
        assert_eq!(order(&ws), vec![1, 4, 3, 2]);
        assert_eq!(ws.pinned(), vec![tl(4), tl(3), tl(2)]);
    }

    /// **Provenance enters after the provenance already there**, beside the
    /// locks: the newest provenance sits next to the youngest lock.
    #[test]
    fn new_provenance_enters_between_the_provenance_and_the_locks() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0)], 1_000);
        ws.lock(tl(2), cand(2), limits(1_000)).unwrap();
        observe(&mut ws, &[(3, 900.0)], 1_000);
        assert_eq!(order(&ws), vec![1, 3, 2]);
    }

    /// **A member is locked where it stands** — nothing moves.
    #[test]
    fn locking_a_member_leaves_it_in_place() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 800.0)], 1_000);
        ws.lock(tl(3), cand(3), limits(1_000)).unwrap();
        let before = order(&ws);
        ws.lock(tl(1), cand(1), limits(1_000)).unwrap();
        ws.lock(tl(3), cand(3), limits(1_000)).unwrap();
        assert_eq!(order(&ws), before);
        assert!(ws.is_pinned(tl(1)) && ws.is_pinned(tl(3)));
    }

    /// **Release turns every lock into provenance in place** and moves the
    /// insertion point past them, so the next task enters beside the dialogue.
    #[test]
    fn release_keeps_every_position_and_moves_the_insertion_point_to_the_end() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0)], 1_000);
        ws.lock(tl(2), cand(2), limits(1_000)).unwrap();
        ws.lock(tl(3), cand(3), limits(1_000)).unwrap();
        ws.release(5_000.0);
        assert_eq!(order(&ws), vec![1, 3, 2]);
        assert!(ws.pinned().is_empty());
        assert_eq!(ws.momentum_of(tl(2)), Some(5_000.0));
        assert_eq!(ws.insertion_point(), 3);
        ws.lock(tl(4), cand(4), limits(1_000)).unwrap();
        assert_eq!(order(&ws), vec![1, 3, 2, 4]);
    }

    /// **A lock dislodges the weakest provenance**, and the gap closes.
    #[test]
    fn a_lock_dislodges_the_weakest_provenance() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 300.0), (3, 600.0)], 300);
        assert_eq!(order(&ws), vec![1, 3, 2]);
        ws.lock(tl(4), cand(4), limits(300)).unwrap();
        assert_eq!(order(&ws), vec![1, 3, 4], "2, the weakest, left");
    }

    /// **Locks are never dislodged**: a lock that cannot fit beside the
    /// others is refused, and nothing is evicted for it.
    #[test]
    fn a_lock_that_only_locks_could_make_room_for_is_refused() {
        let mut ws = WorkingSet::default();
        ws.lock(tl(1), cand(1), limits(200)).unwrap();
        ws.lock(tl(2), cand(2), limits(200)).unwrap();
        assert_eq!(ws.lock(tl(3), cand(3), limits(200)), Err(Refusal::Full));
        assert_eq!(order(&ws), vec![2, 1]);
    }

    #[test]
    fn a_conversation_with_no_total_past_the_cap_or_in_no_group_is_refused() {
        let mut ws = WorkingSet::default();
        assert_eq!(
            ws.lock(tl(9), cand(9), limits(10_000)),
            Err(Refusal::NoCost)
        );
        assert_eq!(
            ws.lock(tl(8), cand(8), limits(10_000)),
            Err(Refusal::TooLarge)
        );
        let nowhere = Candidate {
            tokens: Some(100),
            share: None,
        };
        assert_eq!(ws.lock(tl(7), nowhere, limits(10_000)), Err(Refusal::Gone));
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
    /// either side keep their order and the insertion point stays between the
    /// same two.
    #[test]
    fn a_faded_member_leaves_and_the_gap_closes() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 5_000.0), (2, 120.0), (3, 5_000.0)], 1_000);
        ws.lock(tl(4), cand(4), limits(1_000)).unwrap();
        assert_eq!(order(&ws), vec![1, 3, 2, 4]);
        observe(&mut ws, &[], 1_000);
        assert_eq!(order(&ws), vec![1, 3, 4], "2 fell under 100");
        assert_eq!(ws.insertion_point(), 2);
    }

    /// **A locked member neither decays nor leaves** while it is held.
    #[test]
    fn a_lock_does_not_decay() {
        let mut ws = WorkingSet::default();
        ws.lock(tl(1), cand(1), limits(1_000)).unwrap();
        for _ in 0..20 {
            observe(&mut ws, &[], 1_000);
        }
        assert_eq!(order(&ws), vec![1]);
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
        assert!(ws.pinned().is_empty());
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
            max_file_tokens: 1_000,
        };
        let fresh: HashMap<TimelineId, f32> = HashMap::from([(tl(50), 900.0), (tl(1), 200.0)]);
        ws.observe(&fresh, 0.2, 100.0, lim, &info);
        assert_eq!(order(&ws), vec![50, 1]);
        ws.lock(tl(51), cand(51), lim).unwrap();
        assert_eq!(
            order(&ws),
            vec![1, 51],
            "50 left for 51; the weaker file 1 stayed"
        );
    }

    /// **An unlocked member the scope no longer offers leaves**; a lock stays —
    /// it is a promise.
    #[test]
    fn a_member_out_of_scope_leaves_unless_locked() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0)], 1_000);
        ws.lock(tl(2), cand(2), limits(1_000)).unwrap();
        let gone = |t: TimelineId| (t.raw() > 2).then(|| cand(t.raw()));
        ws.observe(&HashMap::new(), 0.2, 100.0, limits(1_000), &gone);
        assert_eq!(order(&ws), vec![2]);
    }

    /// A restored lock is put back whatever it costs — the model was told.
    #[test]
    fn a_restored_lock_ignores_the_budget() {
        let mut ws = WorkingSet::default();
        ws.restore_lock(tl(8), 2_000, FILES);
        ws.restore_lock(tl(1), 100, FILES);
        assert_eq!(
            order(&ws),
            vec![1, 8],
            "restored in order, each before the last"
        );
        assert_eq!(ws.tokens(), 2_100);
    }

    #[test]
    fn a_removed_conversation_leaves_and_the_gap_closes() {
        let mut ws = WorkingSet::default();
        observe(&mut ws, &[(1, 900.0), (2, 800.0)], 1_000);
        ws.lock(tl(3), cand(3), limits(1_000)).unwrap();
        ws.remove(tl(1));
        assert_eq!(order(&ws), vec![2, 3]);
        assert_eq!(ws.insertion_point(), 1);
        assert_eq!(ws.momentum_of(tl(1)), None);
    }
}
