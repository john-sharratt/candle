//! How much of a belief-driven turn group's selection one mid-decode
//! reprojection may replace.
//!
//! A reprojection that replaces a group's members rebuilds the slot from the
//! first piece that changed: every member placed after it is re-injected and
//! every one newly selected is elevated into VRAM. A group whose fresh scores
//! shuffle a handful of near-equal files on every cadence reprojection paid that
//! on every one of them, and the decode it interrupted read a context that never
//! settled. Belief hysteresis does not prevent it — it delays an eviction, it
//! does not bound how many newcomers arrive at once.
//!
//! So a reprojection admits at most `cap` new conversations into a group — a
//! file for `code_reading`, a folder for `repo_map` — and keeps one of the
//! members it would have dropped for every newcomer it turns away. A strong
//! newcomer still arrives on this reprojection; the rest keep their belief and
//! arrive on the next ones, so new content comes in while the selection does
//! not churn. The unit is the conversation, not the turn: a pick brings its
//! exchanges whole, and two turns of one file are one thing arriving.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::ids::{TimelineId, TurnKey};

/// Newcomers a group admits per reprojection when its schema declares no
/// `max_swaps`. Two keeps a cadence reprojection's rebuild small while a topic
/// that moved still lands within a couple of reprojections — a few seconds of
/// decode.
pub const DEFAULT_MAX_SWAPS: usize = 2;

/// What [`cap_new_members`] changed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SwapCap {
    /// Conversations that would have entered and were held back.
    pub held_back: usize,
    /// Conversations that would have left and were kept instead.
    pub kept: usize,
}

/// Bound the conversations entering a group's selection to `cap`.
///
/// `keys`, `selected` (this reprojection's picks, edited in place), `prior`
/// (the previous reprojection's) and `score` (each candidate's belief) are
/// aligned. Newcomers are admitted best score first; for each one held back,
/// the leaving conversation with the best current belief keeps the turns it
/// had. A group with no prior selection is left alone — there is nothing to
/// churn.
pub fn cap_new_members(
    keys: &[TurnKey],
    selected: &mut [bool],
    prior: &[bool],
    score: &[f32],
    cap: usize,
) -> SwapCap {
    let mut was: BTreeSet<TimelineId> = BTreeSet::new();
    let mut now: BTreeMap<TimelineId, f32> = BTreeMap::new();
    let mut best: BTreeMap<TimelineId, f32> = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        let b = best.entry(key.timeline).or_insert(f32::MIN);
        *b = b.max(score[i]);
        if prior[i] {
            was.insert(key.timeline);
        }
        if selected[i] {
            let n = now.entry(key.timeline).or_insert(f32::MIN);
            *n = n.max(score[i]);
        }
    }
    if was.is_empty() {
        return SwapCap::default();
    }
    let mut entering: Vec<TimelineId> = now.keys().filter(|t| !was.contains(t)).copied().collect();
    if entering.len() <= cap {
        return SwapCap::default();
    }
    entering.sort_by(|a, b| best_first(&now, a, b));
    let held_back: BTreeSet<TimelineId> = entering[cap..].iter().copied().collect();
    let mut leaving: Vec<TimelineId> = was
        .iter()
        .filter(|t| !now.contains_key(t))
        .copied()
        .collect();
    leaving.sort_by(|a, b| best_first(&best, a, b));
    let kept: BTreeSet<TimelineId> = leaving.into_iter().take(held_back.len()).collect();
    for (i, key) in keys.iter().enumerate() {
        if held_back.contains(&key.timeline) {
            selected[i] = false;
        } else if kept.contains(&key.timeline) && prior[i] {
            selected[i] = true;
        }
    }
    SwapCap {
        held_back: held_back.len(),
        kept: kept.len(),
    }
}

/// Higher score first, then lower timeline id — a total order, so the same
/// scores always admit the same conversations.
fn best_first(score: &BTreeMap<TimelineId, f32>, a: &TimelineId, b: &TimelineId) -> Ordering {
    score[b]
        .partial_cmp(&score[a])
        .unwrap_or(Ordering::Equal)
        .then(a.cmp(b))
}

#[cfg(test)]
mod tests {
    use super::{cap_new_members, SwapCap};
    use crate::projection::{TimelineId, TurnIndex, TurnKey};

    fn key(tl: u64, idx: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(tl).unwrap(), TurnIndex(idx))
    }

    /// Files 1 and 2 were selected; the fresh pick drops both and brings in
    /// files 3, 4 and 5, each two turns. At a cap of one the best newcomer (5)
    /// arrives and 3 and 4 wait — two held back, so both leavers keep their
    /// place.
    #[test]
    fn newcomers_beyond_the_cap_wait_and_leavers_stay_in_their_place() {
        let keys = [
            key(1, 0),
            key(2, 0),
            key(3, 0),
            key(3, 1),
            key(4, 0),
            key(4, 1),
            key(5, 0),
            key(5, 1),
        ];
        let prior = [true, true, false, false, false, false, false, false];
        let score = [300.0, 400.0, 500.0, 500.0, 450.0, 450.0, 900.0, 900.0];
        let mut selected = [false, false, true, true, true, true, true, true];
        let cap = cap_new_members(&keys, &mut selected, &prior, &score, 1);
        assert_eq!(
            cap,
            SwapCap {
                held_back: 2,
                kept: 2
            }
        );
        assert_eq!(
            selected,
            [true, true, false, false, false, false, true, true]
        );
    }

    /// A leaver keeps only the turns it had; more newcomers held back than
    /// leavers keeps every leaver and no more.
    #[test]
    fn a_kept_leaver_keeps_only_its_own_turns() {
        let keys = [key(1, 0), key(1, 1), key(3, 0), key(4, 0), key(5, 0)];
        let prior = [true, false, false, false, false];
        let score = [100.0, 100.0, 700.0, 600.0, 500.0];
        let mut selected = [false, false, true, true, true];
        let cap = cap_new_members(&keys, &mut selected, &prior, &score, 1);
        assert_eq!(
            cap,
            SwapCap {
                held_back: 2,
                kept: 1
            }
        );
        assert_eq!(selected, [true, false, true, false, false]);
    }

    /// Within the cap, with no prior selection, or with no change, nothing
    /// moves.
    #[test]
    fn a_selection_within_the_cap_is_left_alone() {
        let keys = [key(1, 0), key(2, 0), key(3, 0)];
        let score = [1.0, 2.0, 3.0];

        let mut within = [false, true, true];
        let none = cap_new_members(&keys, &mut within, &[true, false, false], &score, 2);
        assert_eq!((none, within), (SwapCap::default(), [false, true, true]));

        let mut opening = [true, true, true];
        let none = cap_new_members(&keys, &mut opening, &[false; 3], &score, 0);
        assert_eq!((none, opening), (SwapCap::default(), [true, true, true]));

        let mut same = [true, true, false];
        let none = cap_new_members(&keys, &mut same, &[true, true, false], &score, 0);
        assert_eq!((none, same), (SwapCap::default(), [true, true, false]));
    }
}
