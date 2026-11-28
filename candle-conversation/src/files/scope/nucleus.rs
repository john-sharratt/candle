//! Which placed sections the index is reading, as shares of the range's
//! attention (`docs/placed_section_layout.md` §3.2, §3.4).
//!
//! Each placed section's hit momentum — its per-block hit rate, averaged step
//! to step — is taken as a share of the range's: the momenta normalized to sum
//! to 1. The **nucleus** at `p` is the smallest set of the strongest sections
//! that together hold `p` of it. A share is a proportion of what the range is
//! read with, whatever the prompt's length, the selection's budget or the
//! model, so `p` holds on every one; and the nucleus adapts by itself — one
//! section when the decode reads one hard, several when it reads several.
//!
//! A section **locks** when it enters the nucleus at [`P_LOCK`], at the share
//! it holds then. It stays locked while it is within the wider nucleus at
//! [`P_UNLOCK`] — so one on the edge does not flip in and out step after
//! step — and holds at least [`KEEP_OF_LOCKED`] of the share it locked at, so
//! one the decode has moved off does not keep its seat because the attention
//! is spread wide enough for the wider nucleus to reach it. While the
//! selection reads the range no more than it reads the rest of the slot,
//! nothing is locked: a share of almost nothing would otherwise lock the
//! largest of it.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::files::key::SectionKey;

/// The share of the range's attention whose sections lock.
pub const P_LOCK: f32 = 0.5;

/// The share of the range's attention within which a locked section stays
/// locked.
pub const P_UNLOCK: f32 = 0.7;

/// The part of the share a section locked at that it must keep to stay
/// locked. Measured on BlackWorld (2026-11-28): with only the wider nucleus
/// to hold it, a section that locked at the top beside the question held its
/// seat at 5% of the range's attention while one at 19% sat below it.
pub const KEEP_OF_LOCKED: f32 = 0.5;

/// How far an unlocked section's share must pass the one above it before it
/// takes its place: a twentieth of the range's attention. The mildly read
/// sections' shares, 3–6% each, swing by two hundredths a step on BlackWorld
/// (2026-11-28), and at that margin the range was laid out again on nearly
/// every step.
pub const REORDER_SHARE: f32 = 0.05;

/// The share of an even split under which a placed section counts a step
/// toward leaving: a quarter of `1 / n`.
pub const EVICT_OF_EVEN: f32 = 0.25;

/// `momenta` taken as shares of their sum; empty when they sum to nothing.
pub fn shares(momenta: &[(SectionKey, f32)]) -> HashMap<SectionKey, f32> {
    let total: f32 = momenta.iter().map(|&(_, m)| m.max(0.0)).sum();
    if total <= 0.0 {
        return HashMap::new();
    }
    momenta
        .iter()
        .map(|&(k, m)| (k, m.max(0.0) / total))
        .collect()
}

/// The smallest set of `momenta`'s strongest sections (ties by key) whose
/// shares reach `p`, the one that crosses it included; empty when the momenta
/// sum to nothing.
pub fn nucleus(momenta: &[(SectionKey, f32)], p: f32) -> BTreeSet<SectionKey> {
    let share = shares(momenta);
    let mut ranked: Vec<(SectionKey, f32)> = share.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out = BTreeSet::new();
    let mut held = 0.0;
    for (k, s) in ranked {
        if held >= p {
            break;
        }
        out.insert(k);
        held += s;
    }
    out
}

/// The sections locked this step, each with the share it locked at: none
/// while the range is not `engaged` — read no more than the rest of the slot.
/// Otherwise those of `was` still within the nucleus at [`P_UNLOCK`] and
/// holding [`KEEP_OF_LOCKED`] of the share they locked at keep it, and the
/// rest of the nucleus at [`P_LOCK`] lock at the share they hold now.
pub fn relock(
    was: &BTreeMap<SectionKey, f32>,
    momenta: &[(SectionKey, f32)],
    engaged: bool,
) -> BTreeMap<SectionKey, f32> {
    if !engaged {
        return BTreeMap::new();
    }
    let share = shares(momenta);
    let window = nucleus(momenta, P_UNLOCK);
    let mut out: BTreeMap<SectionKey, f32> = was
        .iter()
        .filter(|(k, &at)| {
            window.contains(k) && share.get(k).is_some_and(|&s| s >= KEEP_OF_LOCKED * at)
        })
        .map(|(&k, &at)| (k, at))
        .collect();
    for k in nucleus(momenta, P_LOCK) {
        out.entry(k).or_insert(share[&k]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(n: u64) -> SectionKey {
        SectionKey(n)
    }

    /// **Shares sum to 1, and nothing is no shares.**
    #[test]
    fn shares_sum_to_one() {
        let s = shares(&[(k(1), 300.0), (k(2), 100.0), (k(3), 100.0)]);
        assert_eq!((s[&k(1)], s[&k(2)], s[&k(3)]), (0.6, 0.2, 0.2));
        assert!(shares(&[(k(1), 0.0)]).is_empty());
    }

    /// **The nucleus is the fewest strongest sections reaching `p`**: one
    /// section read hard is alone at 0.5; read evenly, three of five reach it.
    #[test]
    fn the_nucleus_is_the_fewest_strongest_reaching_p() {
        let hard = [(k(1), 600.0), (k(2), 100.0), (k(3), 100.0), (k(4), 200.0)];
        assert_eq!(nucleus(&hard, 0.5), BTreeSet::from([k(1)]));
        let even: Vec<(SectionKey, f32)> = (1..=5).map(|n| (k(n), 100.0)).collect();
        assert_eq!(nucleus(&even, 0.5), BTreeSet::from([k(1), k(2), k(3)]));
        assert!(nucleus(&[(k(1), 0.0)], 0.5).is_empty());
    }

    /// **A locked section stays locked inside the wider nucleus while it keeps
    /// half the share it locked at, and nothing locks while the range is not
    /// engaged**: shares 0.4, 0.25, 0.2, 0.15 — the nucleus at 0.5 is {1, 2},
    /// at 0.7 {1, 2, 3}. 3, locked at 0.3, keeps 0.2 and stays; 4 is outside
    /// 0.7 and unlocks; 1 and 2 lock at the shares they hold. Had 3 locked at
    /// 0.5, its 0.2 is under half of it and it unlocks.
    #[test]
    fn a_locked_section_stays_while_it_keeps_half_its_share() {
        let m = [(k(1), 40.0), (k(2), 25.0), (k(3), 20.0), (k(4), 15.0)];
        let was = BTreeMap::from([(k(3), 0.3), (k(4), 0.2)]);
        assert_eq!(
            relock(&was, &m, true),
            BTreeMap::from([(k(1), 0.4), (k(2), 0.25), (k(3), 0.3)])
        );
        let faded = BTreeMap::from([(k(3), 0.5)]);
        assert_eq!(
            relock(&faded, &m, true),
            BTreeMap::from([(k(1), 0.4), (k(2), 0.25)])
        );
        assert!(relock(&was, &m, false).is_empty());
    }
}
